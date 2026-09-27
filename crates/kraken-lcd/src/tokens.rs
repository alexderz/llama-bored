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
