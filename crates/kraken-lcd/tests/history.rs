//! Public contract for the in-memory history. Time is supplied by the test.

use std::time::{Duration, Instant};

use kraken_lcd::history::{History, Window};

fn at(origin: Instant, secs: u64) -> Instant {
    origin + Duration::from_secs(secs)
}

fn at_minute(origin: Instant, minute: u64) -> Instant {
    at(origin, minute * 60)
}

fn blocks(history: &History, now: Instant) -> [Window; 3] {
    history.windows(now)
}

#[test]
fn fewer_than_five_raw_samples_have_no_ring_mean() {
    let origin = Instant::now();
    let cases: &[&[f32]] = &[&[], &[10.0], &[10.0, 20.0, 30.0, 40.0]];
    for loads in cases {
        let mut history = History::new(origin);
        for (i, load) in loads.iter().enumerate() {
            history.add(at(origin, i as u64), Some(*load));
        }
        assert_eq!(history.ring_mean(at(origin, 10)), None, "{loads:?}");
    }

    let mut history = History::new(origin);
    for secs in 0..5 {
        history.add(at(origin, secs), None);
        history.add(at(origin, secs), Some(10.0));
    }
    assert_eq!(history.ring_mean(at(origin, 4)), Some(10.0));
}

#[test]
fn ring_window_includes_a_sample_exactly_60s_old() {
    let origin = Instant::now();
    let mut history = History::new(origin);
    history.add(origin, Some(100.0));
    for secs in [15, 30, 45, 60] {
        history.add(at(origin, secs), Some(0.0));
    }
    assert_eq!(history.ring_mean(at(origin, 60)), Some(20.0));

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

    let mut history = History::new(origin);
    for secs in 0..5 {
        history.add(at(origin, secs), Some(12.0));
    }
    assert_eq!(history.ring_mean(at(origin, 70)), None);
}

#[test]
fn minute_rollover_splits_the_block_mean() {
    let origin = Instant::now();
    let mut history = History::new(origin);
    history.add(at(origin, 59), Some(10.0));
    history.add(at(origin, 60), Some(30.0));
    let [_, b2, _] = blocks(&history, at_minute(origin, 20));
    assert_eq!(b2.mean, Some(20.0));
    assert_eq!(b2.coverage, 2.0 / 105.0);
}

#[test]
fn window_edges_assign_each_boundary_sample() {
    let origin = Instant::now();
    let now_minute = 2_000u64;
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
        let [b1, b2, b3] = blocks(&history, now);
        assert_eq!(b1.mean.is_some(), in_b1, "{name} b1");
        assert_eq!(b2.mean.is_some(), in_b2, "{name} b2");
        assert_eq!(b3.mean.is_some(), in_b3, "{name} b3");
    }
}

#[test]
fn coverage_is_present_minutes_over_the_full_window() {
    let origin = Instant::now();
    let now_minute = 2_000u64;
    let now = at_minute(origin, now_minute);
    let [b1, b2, b3] = blocks(&History::new(origin), now);
    assert_eq!([b1.mean, b2.mean, b3.mean], [None, None, None]);
    assert_eq!([b1.coverage, b2.coverage, b3.coverage], [0.0, 0.0, 0.0]);

    let mut history = History::new(origin);
    for ago in 9..=15 {
        history.add(at_minute(origin, now_minute - ago), Some(20.0));
    }
    let [b1, _, _] = blocks(&history, now);
    assert_eq!(b1.mean, Some(20.0));
    assert_eq!(b1.coverage, 7.0 / 14.0);

    let mut history = History::new(origin);
    for ago in 2..=15 {
        history.add(at_minute(origin, now_minute - ago), Some(5.0));
    }
    let [b1, _, _] = blocks(&history, now);
    assert_eq!(b1.coverage, 1.0);

    let mut history = History::new(origin);
    history.add(origin, Some(40.0));
    let [b1, _, _] = blocks(&history, at_minute(origin, 10));
    assert_eq!(b1.mean, Some(40.0));
    assert_eq!(b1.coverage, 1.0 / 14.0);
}

#[test]
fn mark_gap_leaves_a_hole_instead_of_quiet_minutes() {
    let origin = Instant::now();
    let mut history = History::new(origin);
    history.add(origin, Some(80.0));
    history.mark_gap(Duration::from_secs(30 * 60));
    history.add(origin, Some(0.0));

    let [b1, b2, b3] = blocks(&history, origin);
    assert_eq!(b1.mean, None);
    assert_eq!(b1.coverage, 0.0);
    assert_eq!(b2.mean, Some(80.0));
    assert_eq!(b2.coverage, 1.0 / 105.0);
    assert_eq!(b3.mean, None);

    let mut quiet = History::new(origin);
    quiet.add(at_minute(origin, 80), Some(80.0));
    quiet.add(at_minute(origin, 79), Some(0.0));
    let [_, quiet_b2, _] = blocks(&quiet, at_minute(origin, 100));
    assert_eq!(quiet_b2.mean, Some(40.0));
    assert_eq!(quiet_b2.coverage, 2.0 / 105.0);

    let mut ring = History::new(origin);
    for secs in 0..5 {
        ring.add(at(origin, secs), Some(10.0));
    }
    ring.mark_gap(Duration::from_secs(3_600));
    assert_eq!(ring.ring_mean(at(origin, 4)), Some(10.0));
}

#[test]
fn minutes_drop_the_oldest_past_1440() {
    let origin = Instant::now();
    let mut history = History::new(origin);
    for minute in 0..=1440 {
        let load = if minute == 0 { 100.0 } else { 0.0 };
        history.add(at_minute(origin, minute), Some(load));
    }
    let [_, _, b3] = blocks(&history, at_minute(origin, 1440));
    assert_eq!(b3.mean, Some(0.0));
    assert_eq!(b3.coverage, 1319.0f32 / 1320.0f32);
}
