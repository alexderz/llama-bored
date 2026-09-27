use std::collections::VecDeque;
use std::time::{Duration, Instant};

const RAW_WINDOW: Duration = Duration::from_secs(60);
const MIN_RING_SAMPLES: u32 = 5;
const MINUTE_CAP: usize = 1440;

/// Half-open minute windows `[now - start_ago, now - end_ago)`.
const BLOCKS: [(u64, u64); 3] = [(15, 1), (120, 15), (1440, 120)];

/// Mean load and coverage for one history block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// Unweighted mean of the per-minute means in the window.
    /// `None` when no minute in the window has samples.
    pub mean: Option<f32>,
    /// Minutes present divided by the number of minute slots in the window.
    pub coverage: f32,
}

/// In-memory load history. Callers pass every timestamp; this type never reads a clock.
#[derive(Debug)]
pub struct History {
    origin: Instant,
    /// Added to monotonic elapsed time before the minute index is taken.
    /// `mark_gap` grows this so a suspend skips minute slots.
    gap: Duration,
    raw: VecDeque<(Instant, f32)>,
    minutes: VecDeque<(u64, f32, u16)>,
}

impl History {
    /// `origin` is minute index 0.
    pub fn new(origin: Instant) -> Self {
        Self {
            origin,
            gap: Duration::ZERO,
            raw: VecDeque::new(),
            minutes: VecDeque::new(),
        }
    }

    /// Advance the minute index by `duration` without recording samples.
    ///
    /// The skipped minutes stay absent, so later windows report them as missing
    /// coverage rather than as quiet load. Sub-minute remainders accumulate.
    /// Raw samples are untouched: the ring is still the last 60 s of monotonic time.
    pub fn mark_gap(&mut self, duration: Duration) {
        self.gap = self.gap.saturating_add(duration);
    }

    /// Record one load sample at `t`.
    ///
    /// `None` and non-finite loads are ignored and do not change history.
    /// Timestamps must be at or after `origin` and non-decreasing across calls.
    /// `mark_gap` moves the minute index forward between calls without inserting samples.
    pub fn add(&mut self, t: Instant, load: Option<f32>) {
        let Some(load) = load.filter(|sample| sample.is_finite()) else {
            return;
        };
        self.prune_raw(t);
        self.raw.push_back((t, load));
        self.record_minute(self.minute_index(t), load);
    }

    fn minute_index(&self, t: Instant) -> u64 {
        let elapsed = t.saturating_duration_since(self.origin);
        elapsed.saturating_add(self.gap).as_secs() / 60
    }

    fn record_minute(&mut self, index: u64, load: f32) {
        if let Some((last_index, mean, n)) = self.minutes.back_mut()
            && *last_index == index
        {
            let new_n = n.saturating_add(1);
            // A stuck count must not fold another sample; that biases the mean.
            if new_n == *n {
                return;
            }
            let updated = ((f64::from(*mean) * f64::from(*n)) + f64::from(load)) / f64::from(new_n);
            *mean = updated as f32;
            *n = new_n;
            return;
        }
        self.minutes.push_back((index, load, 1));
        if self.minutes.len() > MINUTE_CAP {
            self.minutes.pop_front();
        }
    }

    /// Mean of raw samples in the last 60 s ending at `now`.
    ///
    /// `None` when fewer than 5 samples fall in that window. A sample exactly
    /// 60 s old is included.
    pub fn ring_mean(&self, now: Instant) -> Option<f32> {
        let mut sum = 0.0f64;
        let mut n = 0u32;
        for &(t, load) in &self.raw {
            if t <= now && now.saturating_duration_since(t) <= RAW_WINDOW {
                sum += f64::from(load);
                n += 1;
            }
        }
        if n < MIN_RING_SAMPLES {
            None
        } else {
            Some((sum / f64::from(n)) as f32)
        }
    }

    /// The three blocks at `now`: `b1` [now−15, now−1), `b2` [now−120, now−15),
    /// `b3` [now−1440, now−120), in minute indices.
    ///
    /// Coverage divides by the full window length, so minutes before the
    /// origin count as missing. The mean is the unweighted mean of the
    /// per-minute means that are present.
    pub fn windows(&self, now: Instant) -> [Window; 3] {
        let now_index = self.minute_index(now);
        BLOCKS.map(|(start_ago, end_ago)| self.window(now_index, start_ago, end_ago))
    }

    fn window(&self, now_index: u64, start_ago: u64, end_ago: u64) -> Window {
        let len = start_ago - end_ago;
        let start = i128::from(now_index) - i128::from(start_ago);
        let end = i128::from(now_index) - i128::from(end_ago);
        let mut sum = 0.0f64;
        let mut present = 0u32;
        for &(index, mean, _) in &self.minutes {
            let index = i128::from(index);
            if index >= start && index < end {
                sum += f64::from(mean);
                present += 1;
            }
        }
        Window {
            mean: if present == 0 {
                None
            } else {
                Some((sum / f64::from(present)) as f32)
            },
            coverage: present as f32 / len as f32,
        }
    }

    fn prune_raw(&mut self, now: Instant) {
        while self
            .raw
            .front()
            .is_some_and(|&(t, _)| now.saturating_duration_since(t) > RAW_WINDOW)
        {
            self.raw.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn at(origin: Instant, secs: u64) -> Instant {
        origin + Duration::from_secs(secs)
    }

    #[test]
    fn ring_mean_is_none_with_fewer_than_five_samples() {
        let origin = Instant::now();
        let cases: &[&[f32]] = &[&[], &[10.0], &[10.0, 20.0, 30.0, 40.0]];
        for loads in cases {
            let mut history = History::new(origin);
            for (i, load) in loads.iter().enumerate() {
                history.add(at(origin, i as u64), Some(*load));
            }
            assert_eq!(
                history.ring_mean(at(origin, loads.len() as u64)),
                None,
                "loads {loads:?}"
            );
        }
    }

    #[test]
    fn ring_mean_is_the_mean_of_five_or_more_samples() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        let loads = [10.0, 20.0, 30.0, 40.0, 50.0];
        for (i, load) in loads.iter().enumerate() {
            history.add(at(origin, i as u64), Some(*load));
        }
        assert_eq!(history.ring_mean(at(origin, 4)), Some(30.0));

        history.add(at(origin, 5), Some(60.0));
        assert_eq!(history.ring_mean(at(origin, 5)), Some(35.0));
    }

    #[test]
    fn none_samples_are_not_stored() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(at(origin, 0), None);
        history.add(at(origin, 1), Some(10.0));
        history.add(at(origin, 2), None);
        history.add(at(origin, 3), Some(30.0));
        history.add(at(origin, 4), None);
        assert_eq!(
            history
                .raw
                .iter()
                .map(|(_, load)| *load)
                .collect::<Vec<_>>(),
            vec![10.0, 30.0]
        );

        let mut history = History::new(origin);
        for i in 0..5 {
            history.add(at(origin, i), None);
            history.add(at(origin, i), Some(50.0));
        }
        assert_eq!(history.ring_mean(at(origin, 4)), Some(50.0));
        assert_eq!(history.raw.len(), 5);
    }

    #[test]
    fn non_finite_samples_do_not_change_ring_or_minute_mean() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        for secs in 0..5 {
            history.add(at(origin, secs), Some(10.0));
        }
        let now = at(origin, 4);
        assert_eq!(history.ring_mean(now), Some(10.0));
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 10.0, 5)]
        );

        for load in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            history.add(at(origin, 4), Some(load));
        }

        assert_eq!(history.ring_mean(now), Some(10.0));
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 10.0, 5)]
        );
        assert_eq!(history.raw.len(), 5);
    }

    #[test]
    fn minute_sample_count_saturates_at_u16_max() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        for _ in 0..u32::from(u16::MAX) + 1 {
            history.add(origin, Some(10.0));
        }
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 10.0, u16::MAX)]
        );
    }

    #[test]
    fn raw_deque_keeps_only_the_last_60s() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(at(origin, 0), Some(100.0));
        for secs in 61..66 {
            history.add(at(origin, secs), Some(10.0));
        }
        let now = at(origin, 65);
        assert_eq!(history.ring_mean(now), Some(10.0));
        assert_eq!(history.raw.len(), 5);
        assert!(history.raw.iter().all(|(_, load)| *load == 10.0));

        // A sample exactly 60s old is still inside the window.
        let mut history = History::new(origin);
        history.add(at(origin, 0), Some(100.0));
        for secs in [15, 30, 45, 60] {
            history.add(at(origin, secs), Some(0.0));
        }
        assert_eq!(history.ring_mean(at(origin, 60)), Some(20.0));
        assert_eq!(history.raw.len(), 5);

        // One nanosecond past 60s is outside the window.
        let mut history = History::new(origin);
        let later = origin + Duration::from_secs(60) + Duration::from_nanos(1);
        history.add(origin, Some(100.0));
        for secs in 0..5 {
            history.add(later + Duration::from_secs(secs), Some(10.0));
        }
        assert_eq!(
            history.ring_mean(later + Duration::from_secs(4)),
            Some(10.0)
        );
        assert_eq!(history.raw.len(), 5);

        // `now` is the argument, not the clock: these samples are fresh on the
        // monotonic clock but stale relative to the supplied instant.
        let mut history = History::new(origin);
        for secs in 0..5 {
            history.add(at(origin, secs), Some(12.0));
        }
        assert_eq!(history.ring_mean(at(origin, 70)), None);
    }

    #[test]
    fn minute_mean_rolls_over_at_the_minute_boundary() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(at(origin, 0), Some(10.0));
        history.add(at(origin, 30), Some(30.0));
        history.add(at(origin, 59), Some(50.0));
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 30.0, 3)]
        );

        history.add(at(origin, 60), Some(12.0));
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 30.0, 3), (1, 12.0, 1)]
        );
    }

    #[test]
    fn minutes_without_samples_are_absent() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(at(origin, 0), Some(8.0));
        history.add(at(origin, 5 * 60), Some(18.0));
        assert_eq!(
            history
                .minutes
                .iter()
                .map(|(index, _, _)| *index)
                .collect::<Vec<_>>(),
            vec![0, 5]
        );
    }

    fn at_minute(origin: Instant, minute: u64) -> Instant {
        origin + Duration::from_secs(minute * 60)
    }

    #[test]
    fn window_edges_assign_a_sample_on_each_boundary() {
        let origin = Instant::now();
        let now_minute = 2_000u64;
        // Mid-minute, so a duration window and a minute-index window disagree
        // at the inclusive start.
        let now = at_minute(origin, now_minute) + Duration::from_secs(30);
        let cases = [
            ("b1 start now-15", 15u64, true, false, false),
            ("last minute inside b1 now-2", 2, true, false, false),
            ("b1 end now-1", 1, false, false, false),
            ("current minute", 0, false, false, false),
            ("b2 last minute now-16", 16, false, true, false),
            ("b2 start / b3 end now-120", 120, false, true, false),
            ("b3 last minute now-121", 121, false, false, true),
            ("b3 start now-1440", 1440, false, false, true),
            ("before b3 now-1441", 1441, false, false, false),
        ];
        for (name, ago, in_b1, in_b2, in_b3) in cases {
            let mut history = History::new(origin);
            history.add(at_minute(origin, now_minute - ago), Some(42.0));
            let [b1, b2, b3] = history.windows(now);
            assert_eq!(b1.mean.is_some(), in_b1, "{name} b1");
            assert_eq!(b2.mean.is_some(), in_b2, "{name} b2");
            assert_eq!(b3.mean.is_some(), in_b3, "{name} b3");
            let hit = [in_b1, in_b2, in_b3]
                .into_iter()
                .filter(|in_window| *in_window)
                .count();
            assert_eq!(hit, usize::from(in_b1 || in_b2 || in_b3), "{name} overlaps");
            if in_b1 {
                assert_eq!(b1.mean, Some(42.0), "{name}");
            }
            if in_b2 {
                assert_eq!(b2.mean, Some(42.0), "{name}");
            }
            if in_b3 {
                assert_eq!(b3.mean, Some(42.0), "{name}");
            }
        }
    }

    #[test]
    fn coverage_is_minutes_present_over_the_whole_window() {
        let origin = Instant::now();
        let now_minute = 2_000u64;
        let now = at_minute(origin, now_minute);
        let [b1, b2, b3] = History::new(origin).windows(now);
        assert_eq!(b1.mean, None);
        assert_eq!(b2.mean, None);
        assert_eq!(b3.mean, None);
        assert_eq!(b1.coverage, 0.0);
        assert_eq!(b2.coverage, 0.0);
        assert_eq!(b3.coverage, 0.0);

        let mut history = History::new(origin);
        for ago in 9..=15 {
            history.add(at_minute(origin, now_minute - ago), Some(20.0));
        }
        let [b1, b2, b3] = history.windows(now);
        assert_eq!(b1.mean, Some(20.0));
        assert_eq!(b1.coverage, 7.0 / 14.0);
        assert_eq!(b2.mean, None);
        assert_eq!(b2.coverage, 0.0);
        assert_eq!(b3.mean, None);
        assert_eq!(b3.coverage, 0.0);

        let mut history = History::new(origin);
        for ago in 2..=15 {
            history.add(at_minute(origin, now_minute - ago), Some(5.0));
        }
        let [b1, _, _] = history.windows(now);
        assert_eq!(b1.coverage, 1.0);
        assert_eq!(b1.mean, Some(5.0));

        // Minutes count equally. A one-sample minute does not outweigh a full one.
        let mut history = History::new(origin);
        history.add(at_minute(origin, now_minute - 20), Some(100.0));
        for extra in 0..10 {
            history.add(
                at_minute(origin, now_minute - 21) + Duration::from_secs(extra),
                Some(0.0),
            );
        }
        let [_, b2, _] = history.windows(now);
        assert_eq!(b2.mean, Some(50.0));
        assert_eq!(b2.coverage, 2.0 / 105.0);
    }

    #[test]
    fn mark_gap_is_missing_coverage_not_quiet_time() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(origin, Some(80.0));
        history.mark_gap(Duration::from_secs(30 * 60));
        // CLOCK_MONOTONIC does not advance across suspend. The next sample
        // must land `duration` later on the minute index.
        history.add(origin, Some(0.0));
        assert_eq!(
            history.minutes.iter().copied().collect::<Vec<_>>(),
            vec![(0, 80.0, 1), (30, 0.0, 1)]
        );

        let [b1, b2, b3] = history.windows(origin);
        assert_eq!(b1.mean, None);
        assert_eq!(b1.coverage, 0.0);
        assert_eq!(b2.mean, Some(80.0));
        assert_eq!(b2.coverage, 1.0 / 105.0);
        assert_eq!(b3.mean, None);
        assert_eq!(b3.coverage, 0.0);

        let mut history = History::new(origin);
        for secs in 0..5 {
            history.add(at(origin, secs), Some(10.0));
        }
        history.mark_gap(Duration::from_secs(3_600));
        assert_eq!(history.ring_mean(at(origin, 4)), Some(10.0));

        // Remainders add up. Each call is shorter than a second.
        let mut history = History::new(origin);
        let t = origin + Duration::from_millis(59_000);
        history.add(t, Some(4.0));
        history.mark_gap(Duration::from_millis(600));
        history.mark_gap(Duration::from_millis(600));
        history.add(t, Some(6.0));
        assert_eq!(
            history
                .minutes
                .iter()
                .map(|(index, _, _)| *index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn a_stored_zero_counts_as_quiet_time() {
        let origin = Instant::now();
        let now_minute = 100;
        let mut history = History::new(origin);
        history.add(at_minute(origin, now_minute - 20), Some(80.0));
        history.add(at_minute(origin, now_minute - 21), Some(0.0));
        let [_, b2, _] = history.windows(at_minute(origin, now_minute));
        assert_eq!(b2.mean, Some(40.0));
        assert_eq!(b2.coverage, 2.0 / 105.0);
    }

    #[test]
    fn early_window_coverage_uses_the_full_denominator() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(origin, Some(40.0));
        let [b1, b2, b3] = history.windows(at_minute(origin, 10));
        assert_eq!(b1.mean, Some(40.0));
        assert_eq!(b1.coverage, 1.0 / 14.0);
        assert_eq!(b2.mean, None);
        assert_eq!(b2.coverage, 0.0);
        assert_eq!(b3.coverage, 0.0);
    }

    #[test]
    fn minute_means_outlive_pruned_raw_samples() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        history.add(origin, Some(70.0));
        history.add(at_minute(origin, 3), Some(10.0));
        assert_eq!(history.raw.len(), 1);
        let [_, b2, _] = history.windows(at_minute(origin, 20));
        assert_eq!(b2.mean, Some(40.0));
        assert_eq!(
            history
                .minutes
                .iter()
                .map(|(index, _, _)| *index)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
    }

    #[test]
    fn minutes_are_capped_at_1440_entries() {
        let origin = Instant::now();
        let mut history = History::new(origin);
        for minute in 0..1440 {
            history.add(at_minute(origin, minute), Some(1.0));
        }
        assert_eq!(history.minutes.len(), 1440);
        assert_eq!(history.minutes.front().map(|(index, _, _)| *index), Some(0));

        let mut history = History::new(origin);
        for minute in 0..=1440 {
            let load = if minute == 0 { 100.0 } else { 0.0 };
            history.add(at_minute(origin, minute), Some(load));
        }
        assert_eq!(history.minutes.len(), 1440);
        assert_eq!(history.minutes.front().map(|(index, _, _)| *index), Some(1));
        assert_eq!(
            history.minutes.back().map(|(index, _, _)| *index),
            Some(1440)
        );

        let [_, _, b3] = history.windows(at_minute(origin, 1440));
        assert_eq!(b3.mean, Some(0.0));
        assert_eq!(b3.coverage, 1319.0f32 / 1320.0f32);
    }
}
