//! Per-slot context history for the SLOTS sparkline (T53). tty only: the
//! snapshot and the wire never see it.
//!
//! Each slot keeps a cascade of integer buckets, newest first. A bucket holds
//! the MAX used context seen in it, its length in milliseconds, and a reset
//! flag. The finest tier's live bucket collects samples; when it reaches
//! 10 s it is pushed into tier 0. A tier that overflows pops its oldest
//! bucket into a carry, and the carry moves up one tier once it spans that
//! tier's bucket length. Merging is `max` and `or`, never a re-bin, so a
//! peak and a reset marker survive every rollover.
//!
//! A reset marker is placed when this book sees the drop, without a reason.
//! The poller decides the reason later (#9, [`crate::resets`]); when a slot's
//! count for a reason goes up, the newest marker still without one takes it.
//!
//! Tiers: 30 × 10 s, 25 × 1 min, 18 × 5 min (2 h), then 30 min buckets out to
//! `tty.ctx_history_h`. At most [`MAX_POINTS`] buckets per slot and
//! [`MAX_SLOTS`] slots.
//!
//! A history belongs to a (model display name, slot) and outlives a model
//! swap (#45): it is dropped only when its slot has been away for
//! [`KEEP_UNSEEN_MS`] (or the span, if shorter), or to make room at
//! [`MAX_SLOTS`]. A slot that comes back after the model set changed gets
//! the swap marker.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::resets::{ResetCounts, ResetReason};
use crate::slots::SlotView;

/// Bucket length per tier, in milliseconds. Each is a whole multiple of the
/// one before, so a full carry is exactly one bucket of the next tier.
pub const TIER_MS: [u64; 4] = [10_000, 60_000, 300_000, 1_800_000];

/// Buckets kept in the three fine tiers: 5 min + 25 min + 90 min = 2 h.
const FINE_CAPS: [usize; 3] = [30, 25, 18];
const FINE_SPAN_MS: u64 = 7_200_000;

/// Longest `tty.ctx_history_h`.
pub const MAX_SPAN_H: u32 = 24;

/// Most buckets one slot can hold: every tier full, three carries and the
/// live bucket, at the 24 h span.
pub const MAX_POINTS: usize = 30 + 25 + 18 + 44 + 3 + 1;

/// Most slots tracked at once. A new slot past this has no history.
pub const MAX_SLOTS: usize = 64;

/// A drop counts as a reset when it is more than this percent of the
/// previous sample...
pub const DROP_PCT: u64 = 30;
/// ...and more than this many tokens.
pub const DROP_MIN_TOKENS: u64 = 2_000;

/// One bucket, newest first in [`CtxHistory::points`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtxPoint {
    /// How long this bucket covers.
    pub ms: u64,
    /// Largest used context seen in it. `None` when nothing was sampled.
    pub max: Option<u64>,
    /// A drop or a model swap happened in this bucket.
    pub reset: bool,
    /// Why, as [`ResetReason::bit`]s (#9). Empty for a swap or a drop whose
    /// reason is not decided yet.
    pub reasons: u8,
}

impl CtxPoint {
    fn merge(&mut self, other: CtxPoint) {
        self.ms = self.ms.saturating_add(other.ms);
        self.max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.reset |= other.reset;
        self.reasons |= other.reasons;
    }

    /// The reason this bucket's marker shows, when it has one.
    #[must_use]
    pub fn reason(&self) -> Option<ResetReason> {
        ResetReason::strongest(self.reasons)
    }
}

#[derive(Clone, Debug)]
struct Tier {
    cap: usize,
    /// Newest first.
    items: VecDeque<CtxPoint>,
    /// Buckets popped off the old end, waiting to fill one bucket of the
    /// next tier. Older than every item here, newer than the next tier.
    carry: CtxPoint,
}

/// One slot's history.
#[derive(Clone, Debug)]
pub struct CtxHistory {
    tiers: [Tier; 4],
    live: CtxPoint,
    /// The last value recorded, so an idle slot keeps its cached context.
    held: Option<u64>,
}

impl CtxHistory {
    /// Empty history spanning `span_h` hours, clamped to 1..=24.
    #[must_use]
    pub fn new(span_h: u32) -> Self {
        let span_ms = u64::from(span_h.clamp(1, MAX_SPAN_H)) * 3_600_000;
        let coarse = span_ms
            .saturating_sub(FINE_SPAN_MS)
            .div_ceil(TIER_MS[3])
            .max(1);
        let caps = [
            FINE_CAPS[0],
            FINE_CAPS[1],
            FINE_CAPS[2],
            usize::try_from(coarse).unwrap_or(44),
        ];
        Self {
            tiers: caps.map(|cap| Tier {
                cap,
                items: VecDeque::with_capacity(cap + 1),
                carry: CtxPoint::default(),
            }),
            live: CtxPoint::default(),
            held: None,
        }
    }

    /// Let `ms` pass with no sample. A gap longer than the whole history
    /// empties it; the held value stays.
    pub fn advance(&mut self, mut ms: u64) {
        if ms >= self.capacity_ms() {
            for tier in &mut self.tiers {
                tier.items.clear();
                tier.carry = CtxPoint::default();
            }
            self.live = CtxPoint {
                ms: ms % TIER_MS[0],
                ..CtxPoint::default()
            };
            return;
        }
        while ms > 0 {
            let room = TIER_MS[0].saturating_sub(self.live.ms);
            if ms < room {
                self.live.ms += ms;
                return;
            }
            self.live.ms += room;
            ms -= room;
            let done = std::mem::take(&mut self.live);
            self.push(0, done);
        }
    }

    /// Record one reading. `used` is prompt plus decoded tokens, `None` when
    /// the body lacked them. Returns `true` when this marked a reset.
    ///
    /// Only a busy slot can mark a drop. An idle slot that reads lower than
    /// before (llama-server reports an idle slot as 0) keeps the held value:
    /// its context is still cached until the next task says otherwise.
    pub fn sample(&mut self, used: Option<u64>, busy: bool) -> bool {
        let prev = self.held;
        let value = match (used, prev) {
            (Some(v), Some(p)) if !busy && v < p => Some(p),
            (Some(v), _) => Some(v),
            (None, p) => p,
        };
        let reset = busy && matches!((used, prev), (Some(v), Some(p)) if is_drop(p, v));
        self.live.merge(CtxPoint {
            ms: 0,
            max: value,
            reset,
            reasons: 0,
        });
        self.held = value;
        reset
    }

    /// Put a reset marker in the live bucket (a model swap).
    pub fn mark_reset(&mut self) {
        self.live.reset = true;
    }

    /// The slot is back after a model swap (#45): mark it, and forget the
    /// held context, which the reloaded model no longer caches.
    pub fn resume_after_swap(&mut self) {
        self.mark_reset();
        self.held = None;
    }

    /// Give `reason` to the newest marker that has none yet (#9). With no
    /// such marker the live bucket gets a marker with it.
    pub fn label(&mut self, reason: ResetReason) {
        let bit = reason.bit();
        let live = std::iter::once(&mut self.live);
        let older = self.tiers.iter_mut().flat_map(|tier| {
            tier.items
                .iter_mut()
                .chain(std::iter::once(&mut tier.carry))
        });
        match live
            .chain(older)
            .find(|point| point.reset && point.reasons == 0)
        {
            Some(point) => point.reasons |= bit,
            None => {
                self.live.reset = true;
                self.live.reasons |= bit;
            }
        }
    }

    /// The reason of the newest marker, when it has one (#9).
    #[must_use]
    pub fn last_reason(&self) -> Option<ResetReason> {
        last_reason(&self.points())
    }

    /// Every bucket, newest first, carries in their place by age.
    #[must_use]
    pub fn points(&self) -> Vec<CtxPoint> {
        let mut out = Vec::with_capacity(self.len());
        out.push(self.live);
        for tier in &self.tiers {
            out.extend(tier.items.iter().copied());
            if tier.carry.ms > 0 {
                out.push(tier.carry);
            }
        }
        out
    }

    /// Buckets held, counting the live one and any non-empty carry.
    #[must_use]
    pub fn len(&self) -> usize {
        1 + self
            .tiers
            .iter()
            .map(|tier| tier.items.len() + usize::from(tier.carry.ms > 0))
            .sum::<usize>()
    }

    /// Never true: the live bucket is always there.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    fn capacity_ms(&self) -> u64 {
        self.tiers
            .iter()
            .zip(TIER_MS)
            .map(|(tier, ms)| ms * tier.cap as u64)
            .sum::<u64>()
            + TIER_MS[3]
    }

    fn push(&mut self, k: usize, bucket: CtxPoint) {
        let tier = &mut self.tiers[k];
        tier.items.push_front(bucket);
        if tier.items.len() <= tier.cap {
            return;
        }
        let Some(old) = tier.items.pop_back() else {
            return;
        };
        if k + 1 >= TIER_MS.len() {
            return;
        }
        tier.carry.merge(old);
        if tier.carry.ms >= TIER_MS[k + 1] {
            let full = std::mem::take(&mut tier.carry);
            self.push(k + 1, full);
        }
    }
}

/// More than [`DROP_PCT`] percent and more than [`DROP_MIN_TOKENS`] down.
#[must_use]
pub fn is_drop(prev: u64, cur: u64) -> bool {
    let fall = prev.saturating_sub(cur);
    fall > DROP_MIN_TOKENS && u128::from(fall) * 100 > u128::from(prev) * u128::from(DROP_PCT)
}

/// Used context of one slot: prompt tokens in context plus decoded.
#[must_use]
pub fn used_ctx(slot: &SlotView) -> Option<u64> {
    slot.ctx_prompt
        .map(|prompt| prompt.saturating_add(slot.n_decoded))
}

/// The reason of the newest marker in newest-first `points`, when it has
/// one (#9). A newest marker still without a reason gives `None`.
#[must_use]
pub fn last_reason(points: &[CtxPoint]) -> Option<ResetReason> {
    points.iter().find(|point| point.reset)?.reason()
}

/// How long a slot's history is kept after its slot was last seen, at
/// most (#45): a model swapped out and back within this keeps its line.
pub const KEEP_UNSEEN_MS: u64 = 1_800_000;

/// One slot's history and when its slot was last on `/slots`.
#[derive(Debug)]
struct Kept {
    history: CtxHistory,
    last_seen: Instant,
    /// On the last tick's `slots`.
    present: bool,
    /// [`CtxBook::swaps`] when it was last present.
    swaps: u64,
}

/// Every slot's history, keyed by model display name and slot id.
///
/// A history is kept while its slot is away (#45): a model swap (A→B→A, or
/// A→A+B) does not clear the others. Every history advances with time, so
/// an absent slot's line shows the gap; a slot that comes back after the
/// set of `/running` models changed gets the swap marker, and its held
/// context is dropped (the model was reloaded; its cache is gone). A
/// history whose slot has not been seen for [`Self::keep_ms`] is dropped.
/// Only the slots on `/slots` are drawn; the book is asked for those.
#[derive(Debug)]
pub struct CtxBook {
    span_h: u32,
    slots: HashMap<(String, i64), Kept>,
    /// Each slot's reset counts at the last tick (#9), kept across swaps so
    /// a new history does not relabel old drops. At most [`MAX_SLOTS`].
    counted: HashMap<(String, i64), ResetCounts>,
    last_at: Option<Instant>,
    /// Last non-empty `/running` list, sorted.
    running: Vec<String>,
    /// Changes of that list seen so far.
    swaps: u64,
    /// The list changed; the next new histories carry a marker.
    swap_pending: bool,
}

impl CtxBook {
    #[must_use]
    pub fn new(span_h: u32) -> Self {
        Self {
            span_h: span_h.clamp(1, MAX_SPAN_H),
            slots: HashMap::new(),
            last_at: None,
            running: Vec::new(),
            swaps: 0,
            swap_pending: false,
            counted: HashMap::new(),
        }
    }

    /// Hours the sparkline spans.
    #[must_use]
    pub fn span_h(&self) -> u32 {
        self.span_h
    }

    /// How long a history outlives its slot: the sparkline's span or
    /// [`KEEP_UNSEEN_MS`], whichever is less.
    #[must_use]
    pub fn keep_ms(&self) -> u64 {
        (u64::from(self.span_h) * 3_600_000).min(KEEP_UNSEEN_MS)
    }

    /// One tick: advance every history to `now`, then record each slot.
    ///
    /// `running` is the model names on `/running`. An empty list (llama down,
    /// or nothing loaded) changes nothing. A non-empty list that differs from
    /// the last one is a swap: new histories start with a reset marker, and
    /// so does a kept history whose slot comes back after it. A history
    /// whose slot was not seen for [`Self::keep_ms`] is dropped.
    pub fn record(&mut self, now: Instant, running: &[String], slots: &[SlotView]) {
        let mut names = running.to_vec();
        names.sort();
        names.dedup();
        if !names.is_empty() {
            if !self.running.is_empty() && self.running != names {
                self.swaps += 1;
                self.swap_pending = true;
            }
            self.running = names;
        }
        let ms = self.last_at.map_or(0, |then| {
            u64::try_from(now.saturating_duration_since(then).as_millis()).unwrap_or(u64::MAX)
        });
        self.last_at = Some(now);
        for kept in self.slots.values_mut() {
            kept.history.advance(ms);
        }
        let keep = Duration::from_millis(self.keep_ms());
        self.slots
            .retain(|_, kept| now.saturating_duration_since(kept.last_seen) <= keep);
        let mut created = false;
        let mut seen: HashSet<(String, i64)> = HashSet::new();
        for slot in slots {
            if !self.running.is_empty() && !self.running.contains(&slot.model) {
                continue;
            }
            let key = (slot.model.clone(), slot.id);
            if !self.slots.contains_key(&key) {
                if self.slots.len() >= MAX_SLOTS && !self.evict_one(&seen) {
                    continue;
                }
                let mut fresh = CtxHistory::new(self.span_h);
                if self.swap_pending {
                    fresh.mark_reset();
                }
                self.slots.insert(
                    key.clone(),
                    Kept {
                        history: fresh,
                        last_seen: now,
                        present: true,
                        swaps: self.swaps,
                    },
                );
                created = true;
            }
            let Some(kept) = self.slots.get_mut(&key) else {
                continue;
            };
            if !kept.present && kept.swaps != self.swaps {
                kept.history.resume_after_swap();
            }
            kept.present = true;
            kept.last_seen = now;
            kept.swaps = self.swaps;
            let history = &mut kept.history;
            history.sample(used_ctx(slot), slot.is_processing);
            if self.counted.len() < MAX_SLOTS || self.counted.contains_key(&key) {
                let before = self.counted.insert(key.clone(), slot.resets);
                for reason in ResetReason::ALL {
                    let new = before.map_or(0, |old| {
                        slot.resets.get(reason).saturating_sub(old.get(reason))
                    });
                    // A burst larger than this is one marker per reason
                    // too many to see anyway.
                    for _ in 0..new.min(4) {
                        history.label(reason);
                    }
                }
            }
            seen.insert(key);
        }
        for (key, kept) in &mut self.slots {
            kept.present = seen.contains(key);
        }
        if created {
            self.swap_pending = false;
        }
    }

    /// Make room for a new slot at [`MAX_SLOTS`]: drop the history whose
    /// slot has been away longest. False when every one is on this tick.
    fn evict_one(&mut self, seen: &HashSet<(String, i64)>) -> bool {
        let Some(key) = self
            .slots
            .iter()
            .filter(|(key, _)| !seen.contains(*key))
            .min_by_key(|(_, kept)| kept.last_seen)
            .map(|(key, _)| key.clone())
        else {
            return false;
        };
        self.slots.remove(&key);
        true
    }

    /// Newest-first buckets for one slot. Empty when it has no history.
    #[must_use]
    pub fn points(&self, model: &str, id: i64) -> Vec<CtxPoint> {
        self.slots
            .get(&(model.to_owned(), id))
            .map(|kept| kept.history.points())
            .unwrap_or_default()
    }

    /// Slots tracked, present or not.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Resample newest-first buckets onto `width` columns spanning `span_ms`,
/// column 0 newest. Each column takes the max of every bucket that overlaps
/// its age range and any reset among them, so no peak or marker falls
/// between two columns. A bucket wider than a column repeats its max; its
/// reset marks only the newest of those columns.
#[must_use]
pub fn columns(points: &[CtxPoint], width: usize, span_ms: u64) -> Vec<CtxPoint> {
    let mut out = vec![CtxPoint::default(); width];
    if width == 0 || span_ms == 0 {
        return out;
    }
    let w = width as u64;
    let mut age = 0u64;
    for point in points {
        let lo = age;
        let hi = age.saturating_add(point.ms.max(1));
        age = age.saturating_add(point.ms);
        if lo >= span_ms {
            break;
        }
        if point.max.is_none() && !point.reset {
            continue;
        }
        // Columns i with [i*S/W, (i+1)*S/W) overlapping [lo, hi).
        let first = (u128::from(lo) * u128::from(w) / u128::from(span_ms)) as u64;
        let last = ((u128::from(hi - 1) * u128::from(w)) / u128::from(span_ms)) as u64;
        for i in first..=last.min(w - 1) {
            let col = &mut out[i as usize];
            col.max = match (col.max, point.max) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // A bucket wider than a column marks only its newest column, so
            // one reset is one marker.
            if point.reset && i == first {
                col.reset = true;
                col.reasons |= point.reasons;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn slot(model: &str, id: i64, busy: bool, prompt: u64, decoded: u64) -> SlotView {
        SlotView {
            model: model.to_owned(),
            id,
            id_task: 1,
            is_processing: busy,
            n_prompt_tokens: prompt,
            n_prompt_tokens_processed: prompt,
            n_decoded: decoded,
            n_ctx: Some(262_144),
            ctx_prompt: Some(prompt),
            ctx_used: None,
            resets: Default::default(),
            last_reset: None,
            input: Vec::new(),
            output: Vec::new(),
        }
    }

    fn maxes(points: &[CtxPoint]) -> Vec<Option<u64>> {
        points.iter().map(|point| point.max).collect()
    }

    fn resets(points: &[CtxPoint]) -> usize {
        points.iter().filter(|point| point.reset).count()
    }

    #[test]
    fn growth_accumulates_newest_first() {
        let mut history = CtxHistory::new(6);
        for step in 0..60u64 {
            history.advance(1_000);
            history.sample(Some(10_000 + step * 1_000), true);
        }
        let points = history.points();
        let got = maxes(&points);
        // 60 s is six 10 s buckets; the live one holds the last sample.
        assert_eq!(got[0], Some(69_000), "{got:?}");
        assert_eq!(got[1], Some(68_000), "{got:?}");
        assert!(got.windows(2).all(|pair| pair[0] >= pair[1]), "{got:?}");
        assert_eq!(resets(&points), 0);
        let total: u64 = points.iter().map(|point| point.ms).sum();
        assert_eq!(total, 60_000);
    }

    #[test]
    fn a_drop_over_thirty_percent_and_two_k_is_a_reset_and_a_small_one_is_not() {
        let mut history = CtxHistory::new(6);
        history.sample(Some(90_000), true);
        history.advance(1_000);
        // 20 % down: a pruned reasoning turn, not a reset.
        assert!(!history.sample(Some(72_000), true));
        history.advance(1_000);
        // 50 % down: a compaction.
        assert!(history.sample(Some(36_000), true));
        assert_eq!(resets(&history.points()), 1);
        // 60 % of a small context, but under 2k tokens: not a reset.
        let mut small = CtxHistory::new(6);
        small.sample(Some(3_000), true);
        assert!(!small.sample(Some(1_200), true));
    }

    #[test]
    fn going_idle_with_the_cache_is_not_a_drop() {
        let mut history = CtxHistory::new(6);
        history.sample(Some(70_000), true);
        history.advance(1_000);
        // llama-server reports an idle slot as 0 prompt tokens.
        assert!(!history.sample(Some(0), false));
        history.advance(1_000);
        assert!(!history.sample(None, false));
        for _ in 0..20 {
            history.advance(1_000);
            assert!(!history.sample(Some(0), false));
        }
        let points = history.points();
        assert_eq!(resets(&points), 0);
        assert!(
            points.iter().all(|point| point.max == Some(70_000)),
            "{points:?}"
        );
        // The next task continues from the cache: still no drop.
        history.advance(1_000);
        assert!(!history.sample(Some(71_000), true));
        // A fresh session after idling does mark one.
        history.advance(1_000);
        assert!(history.sample(Some(1_600), true));
    }

    #[test]
    fn the_cascade_keeps_the_max_across_a_rollover() {
        // One 100k spike in an otherwise 10k stream, then enough time to push
        // it through tier 0 into the carry and on to a 1 min bucket.
        let mut history = CtxHistory::new(6);
        history.sample(Some(10_000), true);
        history.advance(1_000);
        history.sample(Some(100_000), true);
        history.advance(1_000);
        for _ in 0..400 {
            history.sample(Some(10_000), true);
            history.advance(1_000);
        }
        let points = history.points();
        let peak = points
            .iter()
            .filter(|point| point.max == Some(100_000))
            .collect::<Vec<_>>();
        assert_eq!(peak.len(), 1, "{points:?}");
        // 400 s later the spike is in a 1 min bucket (tier 1), not a 10 s one.
        assert_eq!(peak[0].ms, 60_000, "{points:?}");
        assert!(peak[0].reset, "the drop from the spike rides along");
        let total: u64 = points.iter().map(|point| point.ms).sum();
        assert_eq!(total, 402_000);
    }

    #[test]
    fn memory_is_bounded() {
        for span in [1u32, 6, 24] {
            let mut history = CtxHistory::new(span);
            for step in 0..(30 * 3_600u64) {
                history.advance(1_000);
                history.sample(Some(step % 90_000), step % 7 != 0);
            }
            assert!(history.len() <= MAX_POINTS, "{span}h: {}", history.len());
            let covered: u64 = history.points().iter().map(|point| point.ms).sum();
            assert!(
                covered >= u64::from(span) * 3_600_000,
                "{span}h covers {covered} ms"
            );
            assert!(covered <= u64::from(span.max(2)) * 3_600_000 + 2 * TIER_MS[3]);
        }
        // A long gap empties it in one step.
        let mut history = CtxHistory::new(6);
        history.sample(Some(5_000), true);
        history.advance(u64::MAX);
        assert_eq!(history.len(), 1);

        let mut book = CtxBook::new(6);
        let now = Instant::now();
        let slots: Vec<SlotView> = (0..200).map(|id| slot("m", id, true, 10, 0)).collect();
        book.record(now, &["m".to_owned()], &slots);
        assert_eq!(book.len(), MAX_SLOTS);
    }

    #[test]
    fn a_model_swap_keeps_old_histories_and_marks_new_ones() {
        let mut book = CtxBook::new(6);
        let t0 = Instant::now();
        let a = vec!["Qwen".to_owned()];
        book.record(t0, &a, &[slot("Qwen", 0, true, 50_000, 10)]);
        book.record(
            t0 + Duration::from_secs(1),
            &a,
            &[slot("Qwen", 0, true, 60_000, 10)],
        );
        assert_eq!(book.points("Qwen", 0)[0].max, Some(60_010));
        assert_eq!(resets(&book.points("Qwen", 0)), 0);

        // llama down for a tick: nothing resets.
        book.record(t0 + Duration::from_secs(2), &[], &[]);
        assert_eq!(book.points("Qwen", 0)[0].max, Some(60_010));

        let b = vec!["Bonsai".to_owned()];
        book.record(
            t0 + Duration::from_secs(3),
            &b,
            &[slot("Bonsai", 0, true, 70_000, 0)],
        );
        // Qwen's history is kept (not drawn: its slot is not on /slots).
        assert_eq!(resets(&book.points("Qwen", 0)), 0);
        assert_eq!(book.len(), 2);
        let points = book.points("Bonsai", 0);
        assert_eq!(points.len(), 1, "{points:?}");
        assert!(points[0].reset);
        assert_eq!(points[0].max, Some(70_000));
        // The marker is placed once.
        book.record(
            t0 + Duration::from_secs(4),
            &b,
            &[
                slot("Bonsai", 0, true, 71_000, 0),
                slot("Bonsai", 1, true, 5, 0),
            ],
        );
        assert_eq!(resets(&book.points("Bonsai", 0)), 1);
        assert_eq!(resets(&book.points("Bonsai", 1)), 0);
    }

    /// #45: A→A+B keeps A's line as it was; B's new lines start marked.
    #[test]
    fn a_second_model_loading_leaves_the_first_alone() {
        let mut book = CtxBook::new(6);
        let t0 = Instant::now();
        let a = vec!["A".to_owned()];
        for step in 0..30u64 {
            book.record(
                t0 + Duration::from_secs(step),
                &a,
                &[slot("A", 0, true, 40_000 + step * 100, 0)],
            );
        }
        let both = vec!["A".to_owned(), "B".to_owned()];
        book.record(
            t0 + Duration::from_secs(30),
            &both,
            &[slot("A", 0, true, 43_000, 0), slot("B", 0, true, 1_000, 0)],
        );
        let after = book.points("A", 0);
        assert_eq!(resets(&after), 0, "{after:?}");
        let covered: u64 = after.iter().map(|point| point.ms).sum();
        assert_eq!(covered, 30_000, "the same line, not a new one");
        assert_eq!(after[0].max, Some(43_000));
        assert!(after[1..].iter().any(|point| point.max == Some(42_900)));
        assert_eq!(resets(&book.points("B", 0)), 1);
    }

    /// #45: A→B→A within the keep time: A's line resumes with one marker,
    /// the gap in between, and no held context from before the swap.
    #[test]
    fn a_model_swapped_out_and_back_resumes_its_history() {
        let mut book = CtxBook::new(6);
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let a = vec!["A".to_owned()];
        let b = vec!["B".to_owned()];
        book.record(at(0), &a, &[slot("A", 0, true, 80_000, 0)]);
        book.record(at(1), &a, &[slot("A", 0, false, 0, 0)]);
        assert_eq!(book.points("A", 0)[0].max, Some(80_000), "held while idle");
        book.record(at(2), &b, &[slot("B", 0, true, 5_000, 0)]);
        for s in 3..300 {
            book.record(at(s), &b, &[slot("B", 0, true, 5_000, 0)]);
        }
        // A is back, idle: a fresh process holds nothing yet.
        book.record(at(300), &a, &[slot("A", 0, false, 0, 0)]);
        let points = book.points("A", 0);
        assert_eq!(resets(&points), 1, "{points:?}");
        assert!(points[0].reset, "the marker is where A came back");
        assert_eq!(points[0].max, Some(0), "no held 80k after the reload");
        assert!(
            points.iter().any(|point| point.max == Some(80_000)),
            "the line from before the swap is still there: {points:?}"
        );
        let gap: u64 = points
            .iter()
            .filter(|point| point.max.is_none())
            .map(|point| point.ms)
            .sum();
        assert!(gap >= 290_000, "the time away is a gap: {gap} ms");
        // Its next sample is just a sample: one marker still.
        book.record(at(301), &a, &[slot("A", 0, true, 10_000, 0)]);
        assert_eq!(resets(&book.points("A", 0)), 1);
        // B, away now, keeps its line too.
        assert!(!book.points("B", 0).is_empty());
    }

    /// #45: a slot away longer than the keep time loses its history, and
    /// a full book makes room by dropping the one away longest.
    #[test]
    fn absent_histories_age_out_and_give_way_at_the_cap() {
        let mut book = CtxBook::new(6);
        assert_eq!(book.keep_ms(), KEEP_UNSEEN_MS);
        let t0 = Instant::now();
        let a = vec!["A".to_owned()];
        let b = vec!["B".to_owned()];
        book.record(t0, &a, &[slot("A", 0, true, 9_000, 0)]);
        book.record(t0, &b, &[slot("B", 0, true, 9_000, 0)]);
        let keep = Duration::from_millis(KEEP_UNSEEN_MS);
        book.record(t0 + keep, &b, &[slot("B", 0, true, 9_000, 0)]);
        assert!(!book.points("A", 0).is_empty(), "exactly the keep time");
        book.record(
            t0 + keep + Duration::from_secs(1),
            &b,
            &[slot("B", 0, true, 9_000, 0)],
        );
        assert!(book.points("A", 0).is_empty(), "aged out");
        assert_eq!(book.len(), 1);

        // Cap: 64 slots of model C, then 64 of model D replace them.
        let mut book = CtxBook::new(6);
        let c: Vec<SlotView> = (0..MAX_SLOTS as i64)
            .map(|id| slot("C", id, true, 10, 0))
            .collect();
        book.record(t0, &["C".to_owned()], &c);
        assert_eq!(book.len(), MAX_SLOTS);
        let d: Vec<SlotView> = (0..MAX_SLOTS as i64)
            .map(|id| slot("D", id, true, 10, 0))
            .collect();
        book.record(t0 + Duration::from_secs(1), &["D".to_owned()], &d);
        assert_eq!(book.len(), MAX_SLOTS);
        assert!(book.points("C", 0).is_empty());
        assert!(!book.points("D", 63).is_empty());
        // A present slot is never dropped for a newcomer.
        let more: Vec<SlotView> = (0..MAX_SLOTS as i64 + 1)
            .map(|id| slot("D", id, true, 10, 0))
            .collect();
        book.record(t0 + Duration::from_secs(2), &["D".to_owned()], &more);
        assert_eq!(book.len(), MAX_SLOTS);
        assert!(book.points("D", MAX_SLOTS as i64).is_empty());
    }

    /// #9: a reason decided later labels the drop's own marker, even after
    /// it rolled into an older bucket, and a newer marker is not relabelled.
    #[test]
    fn a_late_reason_labels_the_newest_unlabelled_marker() {
        let mut history = CtxHistory::new(6);
        history.sample(Some(90_000), true);
        history.advance(1_000);
        assert!(history.sample(Some(20_000), true));
        assert_eq!(history.last_reason(), None, "undecided");
        // Two minutes pass before the request's activity row arrives.
        for _ in 0..120 {
            history.advance(1_000);
            history.sample(Some(21_000), true);
        }
        history.label(ResetReason::Compacted);
        let points = history.points();
        assert_eq!(resets(&points), 1, "{points:?}");
        let marked = points.iter().find(|point| point.reset).expect("marker");
        assert_eq!(marked.reason(), Some(ResetReason::Compacted));
        assert_eq!(history.last_reason(), Some(ResetReason::Compacted));
        // A reason with no marker waiting makes one in the live bucket.
        history.label(ResetReason::Evicted);
        let points = history.points();
        assert_eq!(resets(&points), 2);
        assert_eq!(points[0].reason(), Some(ResetReason::Evicted));
        assert_eq!(history.last_reason(), Some(ResetReason::Evicted));
    }

    #[test]
    fn the_book_labels_markers_from_the_slot_counts() {
        let mut book = CtxBook::new(6);
        let t0 = Instant::now();
        let m = vec!["M".to_owned()];
        let mut s = slot("M", 0, true, 80_000, 0);
        // A first sighting with counts already set is a baseline, no label.
        s.resets.new = 5;
        book.record(t0, &m, std::slice::from_ref(&s));
        s.ctx_prompt = Some(10_000);
        book.record(t0 + Duration::from_secs(1), &m, std::slice::from_ref(&s));
        assert_eq!(resets(&book.points("M", 0)), 1);
        assert_eq!(last_reason(&book.points("M", 0)), None);
        s.resets.new = 6;
        book.record(t0 + Duration::from_secs(2), &m, std::slice::from_ref(&s));
        let points = book.points("M", 0);
        assert_eq!(resets(&points), 1, "{points:?}");
        assert_eq!(last_reason(&points), Some(ResetReason::New));
        // The same counts again change nothing.
        book.record(t0 + Duration::from_secs(3), &m, std::slice::from_ref(&s));
        assert_eq!(resets(&book.points("M", 0)), 1);
        // Columns keep the reason on the marker's column.
        let cols = columns(&book.points("M", 0), 12, 120_000);
        let marked: Vec<&CtxPoint> = cols.iter().filter(|col| col.reset).collect();
        assert_eq!(marked.len(), 1);
        assert_eq!(marked[0].reason(), Some(ResetReason::New));
    }

    #[test]
    fn columns_keep_the_max_and_the_marker_of_every_bucket() {
        let points = [
            CtxPoint {
                ms: 5_000,
                max: Some(10),
                reset: false,
                reasons: 0,
            },
            CtxPoint {
                ms: 10_000,
                max: Some(40),
                reset: true,
                reasons: 0,
            },
            CtxPoint {
                ms: 10_000,
                max: Some(30),
                reset: false,
                reasons: 0,
            },
            CtxPoint {
                ms: 60_000,
                max: Some(20),
                reset: false,
                reasons: 0,
            },
        ];
        // 4 columns of 30 s.
        let cols = columns(&points, 4, 120_000);
        assert_eq!(maxes(&cols), vec![Some(40), Some(20), Some(20), None]);
        assert!(cols[0].reset);
        assert!(!cols[1].reset);
        // 12 columns of 10 s: the 60 s bucket repeats, a wide reset does not.
        let wide = [CtxPoint {
            ms: 60_000,
            max: Some(7),
            reset: true,
            reasons: 0,
        }];
        let cols = columns(&wide, 12, 120_000);
        assert_eq!(maxes(&cols[..6]), vec![Some(7); 6]);
        assert_eq!(resets(&cols), 1);
        assert!(cols[0].reset);
    }
}
