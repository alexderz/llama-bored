//! The big PROMPT number (#188).
//!
//! llama.cpp's live `/slots` rate is kept. A burst engine does not use the
//! one-second wall-clock spike of `prompt_tokens_total`: vLLM holds the
//! histogram prefill speed until the next prefill finishes, and SGLang holds
//! the last rate from one finished request that was alone. Idle is 0. A
//! missing rate stays missing.

/// Headline prompt tok/s.
///
/// `slots` is llama.cpp's prefill rate while it is in flight: the live
/// rate, or 0 while it is decoding. `None` when no llama.cpp slot is in
/// flight, so an idle stored 0 is not passed in. `wall` is tokens over the
/// last second of wall clock. `held` is the burst engine's last honest
/// prefill speed. `burst_engine` is set when every model busy right now is
/// vLLM or SGLang, so `wall` must not be shown.
#[must_use]
pub fn headline(
    running: bool,
    busy: bool,
    slots: Option<f64>,
    wall: Option<f64>,
    burst_engine: bool,
    held: Option<f64>,
) -> Option<f64> {
    if !running {
        return None;
    }
    if !busy {
        return Some(0.0);
    }
    if let Some(rate) = finite(slots) {
        return Some(rate);
    }
    if burst_engine {
        return finite(held);
    }
    finite(wall)
}

/// Mean of the newest five valid completed per-request engine rates.
/// A missing measurement is not a zero, and models must be filtered first.
pub fn last_five(rates: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let rates: Vec<f64> = rates
        .flatten()
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .take(5)
        .collect();
    (!rates.is_empty()).then(|| rates.iter().sum::<f64>() / rates.len() as f64)
}

fn finite(rate: Option<f64>) -> Option<f64> {
    rate.filter(|rate| rate.is_finite() && *rate >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_rates_ignore_poll_spikes_and_other_models() {
        assert_eq!(
            last_five(
                [
                    Some(300.0),
                    Some(1000.0),
                    None,
                    Some(500.0),
                    Some(600.0),
                    Some(700.0),
                    Some(34000.0)
                ]
                .into_iter()
            ),
            Some(620.0)
        );
        assert_eq!(last_five([None, Some(0.0)].into_iter()), None);
    }

    #[test]
    fn missing_and_invalid_readings_do_not_count_toward_five() {
        assert_eq!(
            last_five([None, None, None, None, None, Some(600.0)].into_iter()),
            Some(600.0)
        );
        assert_eq!(
            last_five(
                [
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    Some(0.0),
                    Some(-1.0),
                    Some(100.0),
                    None,
                    Some(200.0),
                    Some(300.0),
                    Some(400.0),
                    Some(500.0),
                    Some(9000.0)
                ]
                .into_iter()
            ),
            Some(300.0)
        );
        assert_eq!(last_five([Some(f64::NAN), Some(-1.0)].into_iter()), None);
    }

    /// 20,000 tokens in 10 s is 2,000. The wall-clock spike is 20,000 on
    /// the sample that lands them and 0 a second later. Neither replaces
    /// the held prefill speed.
    #[test]
    fn a_burst_prefill_holds_through_the_wall_clock_spike() {
        let held = Some(2_000.0);
        assert_eq!(
            headline(true, true, None, Some(20_000.0), true, held),
            Some(2_000.0),
            "the landing sample is not 20,000"
        );
        assert_eq!(
            headline(true, true, None, Some(0.0), true, held),
            Some(2_000.0),
            "a second later is not 0"
        );
    }

    #[test]
    fn a_live_slot_rate_is_not_replaced() {
        assert_eq!(
            headline(
                true,
                true,
                Some(1_193.6),
                Some(20_000.0),
                false,
                Some(2_000.0)
            ),
            Some(1_193.6)
        );
        assert_eq!(
            headline(true, true, Some(0.0), Some(20_000.0), false, Some(2_000.0)),
            Some(0.0),
            "llama.cpp decoding really is 0 prefill"
        );
    }

    #[test]
    fn idle_is_zero_and_a_missing_burst_rate_stays_missing() {
        assert_eq!(
            headline(true, false, None, Some(20_000.0), true, Some(2_000.0)),
            Some(0.0)
        );
        assert_eq!(headline(true, true, None, Some(20_000.0), true, None), None);
        assert_eq!(
            headline(false, true, Some(1.0), Some(1.0), false, Some(1.0)),
            None
        );
    }

    #[test]
    fn a_streaming_engine_still_uses_the_wall_clock() {
        assert_eq!(
            headline(true, true, None, Some(80.0), false, None),
            Some(80.0)
        );
    }
}
