//! Reset-aware counter deltas and a per-second rate.

use std::time::Duration;

/// Increment to add for one sample of a monotonic counter that may reset.
///
/// `last` is `None` on the first sample: that sample is the baseline and the
/// delta is 0. The caller stores `value` as the next `last`.
/// `value >= last` yields `value - last`. `value < last` (the counter
/// restarted) yields `value`.
#[must_use]
pub fn counter_delta(last: Option<u64>, value: u64) -> u64 {
    match last {
        None => 0,
        Some(last) if value >= last => value - last,
        Some(_) => value,
    }
}

/// `delta` events per second across `window`.
///
/// A zero window yields 0. Otherwise the result is `delta / seconds`.
#[must_use]
pub fn delta_per_s(delta: u64, window: Duration) -> f64 {
    let seconds = window.as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        delta as f64 / seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_read_is_a_baseline_and_adds_nothing() {
        assert_eq!(counter_delta(None, 0), 0);
        assert_eq!(counter_delta(None, 100), 0);
    }

    #[test]
    fn increase_is_the_difference() {
        assert_eq!(counter_delta(Some(100), 100), 0);
        assert_eq!(counter_delta(Some(100), 140), 40);
        assert_eq!(counter_delta(Some(0), u64::MAX), u64::MAX);
    }

    #[test]
    fn reset_adds_the_new_value() {
        assert_eq!(counter_delta(Some(140), 3), 3);
        assert_eq!(counter_delta(Some(1), 0), 0);
    }

    #[test]
    fn delta_per_s_divides_by_the_window() {
        assert_eq!(delta_per_s(10, Duration::from_secs(2)), 5.0);
        assert_eq!(delta_per_s(1, Duration::from_millis(500)), 2.0);
        assert_eq!(delta_per_s(0, Duration::from_secs(1)), 0.0);
        assert_eq!(delta_per_s(5, Duration::ZERO), 0.0);
    }
}
