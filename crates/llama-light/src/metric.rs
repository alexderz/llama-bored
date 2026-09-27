//! The numbers a light can show, read from a validated snapshot.

use std::collections::VecDeque;
use std::time::Duration;

use llama_core::rate::{counter_delta, delta_per_s};
use llama_core::wire::SnapshotV1;

/// One snapshot quantity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Metric {
    /// `activity_pct`, falling back to `load_pct` from an older watcher. 0–125.
    Activity,
    /// `gpu_pct`.
    Gpu,
    /// `cpu_pct`.
    Cpu,
    /// `cpu_topk_pct`: the hottest cores.
    CpuTopk,
    /// `load_pct`.
    Load,
    /// Decoded tokens per second, from `decoded_total` deltas.
    TokensRate,
    /// `coolant_c`, °C.
    Coolant,
    /// `gpu_c`, °C.
    GpuTemp,
    /// `cpu_c`, °C.
    CpuTemp,
    /// `mem_pct`.
    Mem,
}

impl Metric {
    /// Every metric with its config name.
    pub const ALL: [(&'static str, Metric); 10] = [
        ("activity", Metric::Activity),
        ("gpu", Metric::Gpu),
        ("cpu", Metric::Cpu),
        ("cpu_topk", Metric::CpuTopk),
        ("load", Metric::Load),
        ("tokens_rate", Metric::TokensRate),
        ("coolant", Metric::Coolant),
        ("gpu_temp", Metric::GpuTemp),
        ("cpu_temp", Metric::CpuTemp),
        ("mem", Metric::Mem),
    ];

    /// Other accepted names: the snapshot's own field names.
    pub const ALIASES: [(&'static str, Metric); 10] = [
        ("activity_pct", Metric::Activity),
        ("gpu_pct", Metric::Gpu),
        ("cpu_pct", Metric::Cpu),
        ("cpu_topk_pct", Metric::CpuTopk),
        ("load_pct", Metric::Load),
        ("mem_pct", Metric::Mem),
        ("tokens_per_s", Metric::TokensRate),
        ("coolant_c", Metric::Coolant),
        ("gpu_c", Metric::GpuTemp),
        ("cpu_c", Metric::CpuTemp),
    ];

    /// Parse a config name or alias.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .chain(Self::ALIASES.iter())
            .find(|(key, _)| *key == name)
            .map(|(_, metric)| *metric)
    }

    /// Whether the metric is a rate read from a counter (so it takes
    /// `rate_window_s`).
    #[must_use]
    pub fn is_counter_rate(self) -> bool {
        self == Metric::TokensRate
    }

    /// The config name.
    #[must_use]
    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(_, metric)| *metric == self)
            .map_or("?", |(key, _)| key)
    }

    /// Range used when an entry gives none, in the metric's units.
    #[must_use]
    pub fn default_range(self) -> (f32, f32) {
        match self {
            // 100 is nominal sustained load; up to 125 reads hottest on
            // the act palette, as on the LCD.
            Metric::Activity
            | Metric::Gpu
            | Metric::Cpu
            | Metric::CpuTopk
            | Metric::Load
            | Metric::Mem => (0.0, 100.0),
            Metric::TokensRate => (0.0, 100.0),
            Metric::Coolant => (25.0, 45.0),
            Metric::GpuTemp => (30.0, 85.0),
            Metric::CpuTemp => (30.0, 90.0),
        }
    }

    /// The value in `snapshot`, or `None` when the watcher had none.
    /// [`Metric::TokensRate`] comes from [`TokenRate`], not from here.
    #[must_use]
    pub fn read(self, snapshot: &SnapshotV1, tokens_rate: Option<f32>) -> Option<f32> {
        let host = &snapshot.host;
        match self {
            Metric::Activity => host.activity_pct.or(host.load_pct),
            Metric::Gpu => host.gpu_pct,
            Metric::Cpu => host.cpu_pct,
            Metric::CpuTopk => host.cpu_topk_pct,
            Metric::Load => host.load_pct,
            Metric::TokensRate => tokens_rate,
            Metric::Coolant => host.coolant_c,
            Metric::GpuTemp => host.gpu_c,
            Metric::CpuTemp => host.cpu_c,
            Metric::Mem => host.mem_pct,
        }
    }
}

/// Most counter samples kept for a rate window.
pub const MAX_HISTORY: usize = 4096;

/// Decoded tokens per second from the `decoded_total` counter.
///
/// [`TokenRate::update`] gives the rate between consecutive distinct
/// snapshots (the raw, per-sample rate). [`TokenRate::rate_over`] gives the
/// rate across a sliding window: Δcounter over the newest sample and the
/// newest sample at least `window_s` older (or the oldest kept, early on).
/// Tokens land in bursts, so the raw rate reads 0, 20, 0, 40; the windowed
/// rate at the same real speed stays steady.
///
/// A new `run_id` (watcher restart), a counter that goes backwards or a
/// missing counter resets the baseline. A snapshot with the same `seq` as
/// the last one changes nothing.
#[derive(Debug, Default)]
pub struct TokenRate {
    last: Option<(u64, u64, u64, u64)>,
    rate: Option<f32>,
    /// `(t_mono_ns, total)` of this run, oldest first.
    history: VecDeque<(u64, u64)>,
    /// Longest window any reader asks for, nanoseconds.
    keep_ns: u64,
}

impl TokenRate {
    /// A rate that keeps samples for windows up to `window_s` seconds.
    #[must_use]
    pub fn with_window(window_s: f32) -> Self {
        Self {
            keep_ns: secs_to_ns(window_s),
            ..Self::default()
        }
    }

    /// Feed one validated snapshot. Returns the raw (per-sample) rate.
    pub fn update(&mut self, snapshot: &SnapshotV1) -> Option<f32> {
        let Some(total) = snapshot.tokens.decoded_total else {
            self.last = None;
            self.rate = None;
            self.history.clear();
            return None;
        };
        let now = (snapshot.run_id, snapshot.seq, snapshot.t_mono_ns, total);
        match self.last {
            Some((run, seq, _, _)) if run == now.0 && seq == now.1 => {}
            Some((run, _, t, last_total)) if run == now.0 && now.2 > t && total >= last_total => {
                let delta = counter_delta(Some(last_total), total);
                let window = Duration::from_nanos(now.2 - t);
                self.rate = Some(delta_per_s(delta, window) as f32);
                self.last = Some(now);
                self.push(now.2, total);
            }
            _ => {
                self.last = Some(now);
                self.rate = Some(0.0);
                self.history.clear();
                self.push(now.2, total);
            }
        }
        self.rate
    }

    fn push(&mut self, t: u64, total: u64) {
        self.history.push_back((t, total));
        let cutoff = t.saturating_sub(self.keep_ns);
        // Keep one sample at or before the cutoff as the window's base.
        while self.history.len() > 2 && self.history[1].0 <= cutoff {
            self.history.pop_front();
        }
        while self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }
    }

    /// Tokens per second across the last `window_s` seconds. A window of 0
    /// is the raw per-sample rate, as [`TokenRate::update`] returns.
    #[must_use]
    pub fn rate_over(&self, window_s: f32) -> Option<f32> {
        if window_s <= 0.0 {
            return self.rate;
        }
        self.rate?;
        let &(t, total) = self.history.back()?;
        let cutoff = t.saturating_sub(secs_to_ns(window_s));
        let base = self
            .history
            .iter()
            .rev()
            .find(|(at, _)| *at <= cutoff)
            .or_else(|| self.history.front())
            .copied()?;
        if base.0 >= t {
            return Some(0.0);
        }
        let delta = counter_delta(Some(base.1), total);
        Some(delta_per_s(delta, Duration::from_nanos(t - base.0)) as f32)
    }
}

fn secs_to_ns(s: f32) -> u64 {
    if s.is_finite() && s > 0.0 {
        (f64::from(s) * 1e9) as u64
    } else {
        0
    }
}
