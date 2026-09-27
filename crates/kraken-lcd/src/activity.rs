//! Activity dial: the 24 bars are per-window mean activity.
//!
//! The writer keeps its own history. Each fresh snapshot's activity lands in
//! a 0.5 s slot (`idx = t_mono_ns / SLOT_NS`). A bar is the mean of the
//! samples in its window. Nothing crosses the wire; a writer restart starts
//! the history again.
//!
//! Windows follow the tiers in `[dial] tiers` (10×0.5 s, 2×5 s, 3×15 s,
//! 4×1 min, 5×5 min by default) and are aligned to the clock per tier. Tier
//! `k` starts `acc_k` seconds back, where `acc_k` is the span of the faster
//! tiers. Its newest bar is the partial window `(⌊t/w⌋·w − acc_k, t − acc_k]`
//! that "collects" until it is `w` long; then the tier shifts by one bar.
//! Older bars are whole windows `w` wide on the same grid. The fill of that
//! partial window is `(t mod w) / w`.
//!
//! A bar with fewer covered slots than `dial.xff` of its window is no data.

use std::collections::VecDeque;

use crate::config::{Dial as DialCfg, Tier};

/// One slot is 0.5 s of `CLOCK_MONOTONIC`.
pub const SLOT_NS: u64 = 500_000_000;

/// Top of the activity scale. 100 is nominal; spikes read up to this.
pub const ACTIVITY_MAX: f32 = 125.0;

/// Bars on the dial.
pub const BARS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot {
    idx: u64,
    sum: f32,
    count: u32,
}

/// One tier in whole slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSlots {
    /// Window width in slots.
    pub width: u64,
    /// Bars in this tier.
    pub bars: usize,
}

/// Per-slot sums for the dial span. Bounded by the span plus one widest window.
#[derive(Clone, Debug)]
pub struct ActivityDial {
    slots: VecDeque<Slot>,
    tiers: Vec<TierSlots>,
    keep: u64,
    xff: f64,
}

impl Default for ActivityDial {
    fn default() -> Self {
        Self::new(&DialCfg::default())
    }
}

impl ActivityDial {
    /// Empty history on the configured tiers and coverage threshold.
    #[must_use]
    pub fn new(settings: &DialCfg) -> Self {
        let tiers = tier_slots(&settings.tiers);
        let span: u64 = tiers.iter().map(|tier| tier.width * tier.bars as u64).sum();
        let widest = tiers.iter().map(|tier| tier.width).max().unwrap_or(1);
        Self {
            slots: VecDeque::new(),
            tiers,
            keep: span.saturating_add(widest).saturating_add(1),
            xff: settings.xff,
        }
    }

    /// Tiers in slots, fastest first.
    #[must_use]
    pub fn tiers(&self) -> &[TierSlots] {
        &self.tiers
    }

    /// Slots currently held. Never more than the span plus one widest window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// `true` before the first sample.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Add one activity sample taken at `t_ns`. Non-finite values are dropped;
    /// the rest are clamped to `0..=125`.
    pub fn add(&mut self, t_ns: u64, value: f32) {
        if !value.is_finite() {
            return;
        }
        let value = value.clamp(0.0, ACTIVITY_MAX);
        let idx = t_ns / SLOT_NS;
        if let Some(back) = self.slots.back_mut()
            && back.idx == idx
        {
            back.sum += value;
            back.count = back.count.saturating_add(1);
            return;
        }
        match self.slots.binary_search_by_key(&idx, |slot| slot.idx) {
            Ok(pos) => {
                if let Some(slot) = self.slots.get_mut(pos) {
                    slot.sum += value;
                    slot.count = slot.count.saturating_add(1);
                }
            }
            Err(pos) => self.slots.insert(
                pos,
                Slot {
                    idx,
                    sum: value,
                    count: 1,
                },
            ),
        }
        self.retain_through(t_ns);
    }

    /// Drop slots older than any window at `now_ns` can reach.
    pub fn retain_through(&mut self, now_ns: u64) {
        let newest = now_ns / SLOT_NS;
        let keep_from = newest.saturating_sub(self.keep);
        while self.slots.front().is_some_and(|slot| slot.idx < keep_from) {
            self.slots.pop_front();
        }
    }

    /// Half-open slot ranges `[start, end)` of the 24 bars at `now_ns`, newest
    /// first. A missing tier bar is `(0, 0)`.
    #[must_use]
    pub fn windows(&self, now_ns: u64) -> [(u64, u64); BARS] {
        let mut out = [(0, 0); BARS];
        let current = now_ns / SLOT_NS;
        let mut acc = 0_u64;
        let mut bar = 0;
        for tier in &self.tiers {
            let width = tier.width.max(1);
            let aligned = current - current % width;
            for j in 0..tier.bars {
                if bar >= BARS {
                    return out;
                }
                let j = j as u64;
                let window = if j == 0 {
                    (
                        aligned.checked_sub(acc),
                        current.checked_sub(acc).map(|end| end + 1),
                    )
                } else {
                    (
                        aligned.checked_sub(acc + j * width),
                        aligned.checked_sub(acc + (j - 1) * width),
                    )
                };
                if let (Some(start), Some(end)) = window {
                    out[bar] = (start, end);
                }
                bar += 1;
            }
            acc += width * tier.bars as u64;
        }
        out
    }

    /// Mean activity of each bar at `now_ns`, newest first. `None` when the
    /// window has fewer covered slots than `xff` of its length.
    #[must_use]
    pub fn means(&self, now_ns: u64) -> [Option<f32>; BARS] {
        let mut out = [None; BARS];
        for (bar, (start, end)) in self.windows(now_ns).into_iter().enumerate() {
            out[bar] = self.mean(start, end);
        }
        out
    }

    /// How far each tier's newest (partial) window has filled, `0..1`.
    #[must_use]
    pub fn fill(&self, now_ns: u64) -> Vec<f32> {
        self.tiers
            .iter()
            .map(|tier| {
                let width_ns = tier.width.max(1).saturating_mul(SLOT_NS);
                (now_ns % width_ns) as f32 / width_ns as f32
            })
            .collect()
    }

    fn mean(&self, start: u64, end: u64) -> Option<f32> {
        if end <= start {
            return None;
        }
        let from = self.slots.partition_point(|slot| slot.idx < start);
        let mut sum = 0.0_f64;
        let mut count = 0_u64;
        let mut covered = 0_u64;
        for slot in self.slots.iter().skip(from) {
            if slot.idx >= end {
                break;
            }
            sum += f64::from(slot.sum);
            count += u64::from(slot.count);
            covered += 1;
        }
        if count == 0 {
            return None;
        }
        let length = (end - start) as f64;
        if (covered as f64) < self.xff * length {
            return None;
        }
        let mean = (sum / count as f64) as f32;
        mean.is_finite().then_some(mean.clamp(0.0, ACTIVITY_MAX))
    }
}

/// `[dial] tiers` in whole 0.5 s slots. The config validates that each width
/// is a positive multiple of 0.5 s and the bars sum to 24.
fn tier_slots(tiers: &[Tier]) -> Vec<TierSlots> {
    tiers
        .iter()
        .map(|tier| {
            let slots = (tier.width_s / 0.5).round();
            let width = if slots.is_finite() && slots >= 1.0 {
                slots as u64
            } else {
                1
            };
            TierSlots {
                width,
                bars: usize::try_from(tier.bars).unwrap_or(BARS),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tiers_are_the_design_tiers() {
        let dial = ActivityDial::default();
        let widths: Vec<(u64, usize)> = dial
            .tiers()
            .iter()
            .map(|tier| (tier.width, tier.bars))
            .collect();
        assert_eq!(widths, [(1, 10), (10, 2), (30, 3), (120, 4), (600, 5)]);
    }
}
