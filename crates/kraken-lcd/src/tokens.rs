//! Tokens · 24 h: a 40-point integer cascade over the decoded-token counter.
//!
//! Tiers are 10 × 30 s, 11 × 5 min, 10 × 30 min and 9 × 2 h, exactly 24 h.
//! The live bucket collects token deltas and their milliseconds; at 30 s it
//! is pushed onto the first tier. A tier that overflows pops its oldest
//! bucket into a carry, and the carry moves on once it is as long as one
//! bucket of the next tier. Only integer adds happen while collecting; tok/s
//! is a division at draw time. Buckets older than the last tier fall off.
//!
//! [`TokenFeed`] turns watcher readings into deltas. A counter that goes
//! down (a restart or a model swap), a missing counter, a `run_id` change or
//! a stall longer than `dial.max_gap_s` is a gap: that interval adds neither
//! tokens nor time. It never becomes a negative.

use std::collections::VecDeque;

use llama_core::sample::TokenReading;

/// `(bucket milliseconds, buckets kept)` per tier, fastest first.
pub const TIERS: [(u32, usize); 4] = [(30_000, 10), (300_000, 11), (1_800_000, 10), (7_200_000, 9)];

/// Points on the chart.
pub const SLOTS: usize = 40;

/// The live bucket is plotted once it holds this many milliseconds.
pub const LIVE_MIN_MS: u32 = 3_000;

/// Tokens and the milliseconds they were counted over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bucket {
    /// Tokens decoded.
    pub tok: u64,
    /// Measured time, ms.
    pub ms: u32,
}

impl Bucket {
    fn absorb(&mut self, other: Bucket) {
        self.tok = self.tok.saturating_add(other.tok);
        self.ms = self.ms.saturating_add(other.ms);
    }

    /// tok/s. Zero for an empty bucket.
    #[must_use]
    pub fn rate(&self) -> f32 {
        if self.ms == 0 {
            0.0
        } else {
            (self.tok as f64 * 1000.0 / f64::from(self.ms)) as f32
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tier {
    dur_ms: u32,
    cap: usize,
    /// Newest first.
    items: VecDeque<Bucket>,
    carry: Bucket,
}

/// The cascade. State is 40 buckets, 4 carries and the live one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenChart {
    tiers: [Tier; 4],
    live: Bucket,
    dropped: Bucket,
}

impl Default for TokenChart {
    fn default() -> Self {
        Self {
            tiers: TIERS.map(|(dur_ms, cap)| Tier {
                dur_ms,
                cap,
                items: VecDeque::with_capacity(cap + 1),
                carry: Bucket::default(),
            }),
            live: Bucket::default(),
            dropped: Bucket::default(),
        }
    }
}

impl TokenChart {
    /// Add `tok` tokens counted over `ms` milliseconds.
    pub fn add(&mut self, tok: u64, ms: u32) {
        if tok == 0 && ms == 0 {
            return;
        }
        self.live.absorb(Bucket { tok, ms });
        if self.live.ms >= self.tiers[0].dur_ms {
            let done = std::mem::take(&mut self.live);
            self.push(0, done);
        }
    }

    fn push(&mut self, first: usize, bucket: Bucket) {
        let mut k = first;
        let mut bucket = bucket;
        loop {
            let tier = &mut self.tiers[k];
            tier.items.push_front(bucket);
            if tier.items.len() <= tier.cap {
                return;
            }
            let Some(old) = tier.items.pop_back() else {
                return;
            };
            if k + 1 >= self.tiers.len() {
                // Older than 24 h.
                self.dropped.absorb(old);
                return;
            }
            let tier = &mut self.tiers[k];
            tier.carry.absorb(old);
            if tier.carry.ms < self.tiers[k + 1].dur_ms {
                return;
            }
            bucket = std::mem::take(&mut self.tiers[k].carry);
            k += 1;
        }
    }

    /// tok/s per point, newest first, at most [`SLOTS`]. The live bucket is
    /// the first point once it holds [`LIVE_MIN_MS`].
    #[must_use]
    pub fn points(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(SLOTS + 1);
        if self.live.ms >= LIVE_MIN_MS {
            out.push(self.live.rate());
        }
        for tier in &self.tiers {
            out.extend(tier.items.iter().map(Bucket::rate));
        }
        out.truncate(SLOTS);
        out
    }

    /// Every token still held: live, tiers and carries.
    #[must_use]
    pub fn held(&self) -> Bucket {
        let mut total = self.live;
        for tier in &self.tiers {
            total.absorb(tier.carry);
            for item in &tier.items {
                total.absorb(*item);
            }
        }
        total
    }

    /// Tokens that fell off the 24 h end.
    #[must_use]
    pub fn dropped(&self) -> Bucket {
        self.dropped
    }

    /// Buckets stored, carries and the live one included.
    #[must_use]
    pub fn stored(&self) -> usize {
        1 + self
            .tiers
            .iter()
            .map(|tier| tier.items.len() + 1)
            .sum::<usize>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Seen {
    run_id: u64,
    seq: u64,
    t_mono_ns: u64,
    decoded_total: Option<u64>,
}

/// What one reading meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// A new sample (not a re-read of the last one).
    pub fresh: bool,
    /// Tokens and milliseconds to add, when the interval is not a gap.
    pub interval: Option<(u64, u32)>,
}

/// Counter deltas between consecutive readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenFeed {
    prev: Option<Seen>,
    max_gap_ns: u64,
}

impl TokenFeed {
    /// `max_gap_s` is the longest interval that still counts.
    #[must_use]
    pub fn new(max_gap_s: f64) -> Self {
        let ns = (max_gap_s * 1e9).round();
        let max_gap_ns = if ns.is_finite() && ns > 0.0 {
            ns as u64
        } else {
            0
        };
        Self {
            prev: None,
            max_gap_ns,
        }
    }

    /// Fold in one reading. A re-read (same run, `seq` not greater) is not
    /// fresh and leaves the baseline where it was.
    pub fn accept(&mut self, reading: TokenReading) -> Step {
        let cur = Seen {
            run_id: reading.run_id,
            seq: reading.seq,
            t_mono_ns: reading.t_mono_ns,
            decoded_total: reading.decoded_total,
        };
        let Some(prev) = self.prev else {
            self.prev = Some(cur);
            return Step {
                fresh: true,
                interval: None,
            };
        };
        if cur.run_id == prev.run_id && cur.seq <= prev.seq {
            return Step {
                fresh: false,
                interval: None,
            };
        }
        self.prev = Some(cur);
        if cur.run_id != prev.run_id || cur.t_mono_ns <= prev.t_mono_ns {
            return Step {
                fresh: true,
                interval: None,
            };
        }
        let dt_ns = cur.t_mono_ns - prev.t_mono_ns;
        if dt_ns > self.max_gap_ns {
            return Step {
                fresh: true,
                interval: None,
            };
        }
        let interval = match (prev.decoded_total, cur.decoded_total) {
            (Some(before), Some(after)) if after >= before => {
                let ms = u32::try_from(dt_ns / 1_000_000).unwrap_or(u32::MAX);
                Some((after - before, ms))
            }
            _ => None,
        };
        Step {
            fresh: true,
            interval,
        }
    }
}

/// Time constant of the current generation rate, seconds.
pub const GEN_TAU_S: f64 = 5.0;

/// The current generation rate (#55): an exponential moving average of
/// the decoded-token counter's intervals with a [`GEN_TAU_S`] time
/// constant while tokens flow. Once the counter has not moved over
/// intervals adding up to [`GEN_IDLE_MS`] (prompt processing, a pause, the
/// end of a reply) it reads zero rather than a decaying tail. A gap (no
/// interval: the first reading, a `run_id` change, a stall or a counter that
/// went down or away) also resets it to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenRate {
    ema: f64,
    /// Milliseconds since the counter last moved, over counted intervals.
    idle_ms: u64,
    seen: bool,
}

/// The counter standing still this long snaps the current rate to zero.
pub const GEN_IDLE_MS: u64 = 3_000;

impl GenRate {
    /// Fold in one [`TokenFeed::accept`] step. `counter` says whether the
    /// reading carried a decoded-token total.
    pub fn step(&mut self, step: Step, counter: bool) {
        if !step.fresh {
            return;
        }
        self.seen |= counter;
        match step.interval {
            Some((tok, ms)) if ms > 0 => {
                if tok == 0 {
                    self.idle_ms = self.idle_ms.saturating_add(u64::from(ms));
                    if self.idle_ms >= GEN_IDLE_MS {
                        self.ema = 0.0;
                        return;
                    }
                } else {
                    self.idle_ms = 0;
                }
                let rate = tok as f64 * 1000.0 / f64::from(ms);
                let dt = f64::from(ms) / 1000.0;
                self.ema += (rate - self.ema) * (1.0 - (-dt / GEN_TAU_S).exp());
            }
            Some(_) => {}
            None => self.gap(),
        }
    }

    /// A snapshot without a counter reading: a gap.
    pub fn gap(&mut self) {
        self.ema = 0.0;
        self.idle_ms = 0;
    }

    /// tok/s, or `None` until a reading has carried a counter.
    #[must_use]
    pub fn rate(&self) -> Option<f32> {
        self.seen.then_some(self.ema as f32)
    }
}

/// Display step of the current rate, in tenths of a tok/s: 0.1 below 10,
/// 1 below 1000, 100 below 10 k, then 1000. Each matches what
/// [`gen_rate_text`] prints, so a new step is new text.
#[must_use]
pub fn gen_step_tenths(tenths: f64) -> u32 {
    if tenths < 99.5 {
        1
    } else if tenths < 9_950.0 {
        10
    } else if tenths < 99_500.0 {
        1_000
    } else {
        10_000
    }
}

/// Dead zone around the shown rate, as a fraction of its step.
pub const GEN_MARGIN: f64 = 0.3;

/// The rate to show, in tenths. With `shown`, it is held while `raw` is
/// within half a step plus [`GEN_MARGIN`] of a step; otherwise `raw` is
/// rounded to its own step. Non-finite or negative is zero.
#[must_use]
pub fn hold_gen_tenths(raw: f32, shown: Option<u32>) -> u32 {
    let raw = if raw.is_finite() && raw > 0.0 {
        f64::from(raw) * 10.0
    } else {
        0.0
    };
    if let Some(shown) = shown {
        let shown_f = f64::from(shown);
        let slack = f64::from(gen_step_tenths(shown_f)) * (0.5 + GEN_MARGIN);
        if (raw - shown_f).abs() <= slack {
            return shown;
        }
    }
    let step = f64::from(gen_step_tenths(raw));
    ((raw / step).round() * step).min(f64::from(u32::MAX)) as u32
}

/// The current rate as drawn, from tenths of a tok/s: "7.5" below 10,
/// "112" below 1000, "1.2k" below 10 k, then "12k". Zero is "0" and `None`
/// (no counter, or no data) is "—".
#[must_use]
pub fn gen_rate_text(tenths: Option<u32>) -> String {
    let Some(tenths) = tenths else {
        return "\u{2014}".to_owned();
    };
    if tenths < 100 {
        return if tenths == 0 {
            "0".to_owned()
        } else {
            format!("{}.{}", tenths / 10, tenths % 10)
        };
    }
    let whole = (u64::from(tenths) + 5) / 10;
    if whole < 1000 {
        return format!("{whole}");
    }
    let hundreds = (whole + 50) / 100;
    if hundreds < 100 {
        format!("{}.{}k", hundreds / 10, hundreds % 10)
    } else {
        format!("{}k", (whole + 500) / 1000)
    }
}

/// The ceiling for a peak: the smallest `1 · 2 · 5 × 10ⁿ` at or above it,
/// never below 10 tok/s.
#[must_use]
pub fn nice_ceiling(peak: f32) -> f32 {
    if !peak.is_finite() || peak <= 10.0 {
        return 10.0;
    }
    let exp = peak.log10().floor();
    let base = 10_f32.powf(exp);
    for step in [1.0, 2.0, 5.0, 10.0] {
        let value = step * base;
        if value >= peak * (1.0 - 1e-6) {
            return value;
        }
    }
    10.0 * base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_ceiling_rounds_up_to_one_two_five() {
        assert_eq!(nice_ceiling(0.0), 10.0);
        assert_eq!(nice_ceiling(7.0), 10.0);
        assert_eq!(nice_ceiling(10.0), 10.0);
        assert_eq!(nice_ceiling(11.0), 20.0);
        assert_eq!(nice_ceiling(20.0), 20.0);
        assert_eq!(nice_ceiling(21.0), 50.0);
        assert_eq!(nice_ceiling(51.0), 100.0);
        assert_eq!(nice_ceiling(99.0), 100.0);
        assert_eq!(nice_ceiling(101.0), 200.0);
        assert_eq!(nice_ceiling(1500.0), 2000.0);
        assert_eq!(nice_ceiling(f32::NAN), 10.0);
    }
}
