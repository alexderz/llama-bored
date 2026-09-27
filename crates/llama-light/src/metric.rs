//! The numbers a light can show, read from a validated snapshot.

use llama_core::rate::{counter_delta, delta_per_s};
use llama_core::wire::SnapshotV1;
use std::time::Duration;

/// One snapshot quantity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Metric {
    /// `activity_pct`, falling back to `load_pct` from an older watcher. 0–125.
    Activity,
    /// `gpu_pct`.
    Gpu,
    /// `cpu_pct`.
    Cpu,
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
    pub const ALL: [(&'static str, Metric); 9] = [
        ("activity", Metric::Activity),
        ("gpu", Metric::Gpu),
        ("cpu", Metric::Cpu),
        ("load", Metric::Load),
        ("tokens_rate", Metric::TokensRate),
        ("coolant", Metric::Coolant),
        ("gpu_temp", Metric::GpuTemp),
        ("cpu_temp", Metric::CpuTemp),
        ("mem", Metric::Mem),
    ];

    /// Parse a config name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, metric)| *metric)
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
            // 100 is nominal sustained load; up to 125 reads white-hot on
            // the act palette, as on the LCD.
            Metric::Activity | Metric::Gpu | Metric::Cpu | Metric::Load | Metric::Mem => {
                (0.0, 100.0)
            }
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
            Metric::Load => host.load_pct,
            Metric::TokensRate => tokens_rate,
            Metric::Coolant => host.coolant_c,
            Metric::GpuTemp => host.gpu_c,
            Metric::CpuTemp => host.cpu_c,
            Metric::Mem => host.mem_pct,
        }
    }
}

/// Decoded tokens per second between consecutive distinct snapshots.
///
/// A new `run_id` (watcher restart) or a missing counter resets the
/// baseline. A snapshot with the same `seq` as the last one changes nothing.
#[derive(Debug, Default)]
pub struct TokenRate {
    last: Option<(u64, u64, u64, u64)>,
    rate: Option<f32>,
}

impl TokenRate {
    /// Feed one validated snapshot. Returns the current rate.
    pub fn update(&mut self, snapshot: &SnapshotV1) -> Option<f32> {
        let Some(total) = snapshot.tokens.decoded_total else {
            self.last = None;
            self.rate = None;
            return None;
        };
        let now = (snapshot.run_id, snapshot.seq, snapshot.t_mono_ns, total);
        match self.last {
            Some((run, seq, _, _)) if run == now.0 && seq == now.1 => {}
            Some((run, _, t, last_total)) if run == now.0 && now.2 > t => {
                let delta = counter_delta(Some(last_total), total);
                let window = Duration::from_nanos(now.2 - t);
                self.rate = Some(delta_per_s(delta, window) as f32);
                self.last = Some(now);
            }
            _ => {
                self.last = Some(now);
                self.rate = Some(0.0);
            }
        }
        self.rate
    }
}
