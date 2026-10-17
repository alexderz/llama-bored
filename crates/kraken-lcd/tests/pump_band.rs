//! The pump band follows a one-step envelope around the compared `pwm1` duties.

use kraken_lcd::device::{PUMP_DUTY_STEP_MAX, pump_band_rpm, pump_in_band};

fn explained(before: u32, from: u32, duty: u32) -> u32 {
    let n = (u64::from(before) * u64::from(duty) + u64::from(from) / 2) / u64::from(from);
    u32::try_from(n).unwrap()
}

#[test]
fn unknown_or_zero_held_duty_keeps_the_fixed_band() {
    let band = pump_band_rpm(1127);
    assert_eq!(band, 169);
    for duty in [None, Some(0)] {
        assert!(pump_in_band(1127, 1127 + band, duty, duty), "{duty:?}");
        assert!(!pump_in_band(1127, 1127 + band + 1, duty, duty), "{duty:?}");
        assert!(pump_in_band(1127, 1127 - band, duty, duty), "{duty:?}");
        assert!(!pump_in_band(1127, 1127 - band - 1, duty, duty), "{duty:?}");
    }
}

#[test]
fn the_logged_halt_and_the_hunt_are_in_band() {
    assert!(pump_in_band(1127, 1298, Some(77), Some(77)));
    assert!(pump_in_band(1127, 1298, Some(77), Some(89)));
    assert!(pump_in_band(1298, 1127, Some(89), Some(77)));
}

#[test]
fn envelope_runs_from_one_step_below_to_one_step_above() {
    // 77 -> 89: duties [64, 102]. 1127 * 64 / 77 = 937; 1127 * 102 / 77 = 1493.
    let low = explained(1127, 77, 64);
    let high = explained(1127, 77, 102);
    assert_eq!((low, high), (937, 1493));
    let low_edge = low - pump_band_rpm(low);
    let high_edge = high + pump_band_rpm(high);
    assert!(pump_in_band(1127, high_edge, Some(77), Some(89)));
    assert!(!pump_in_band(1127, high_edge + 1, Some(77), Some(89)));
    assert!(pump_in_band(1127, low_edge, Some(77), Some(89)));
    assert!(!pump_in_band(1127, low_edge - 1, Some(77), Some(89)));
    assert!(!pump_in_band(1127, 0, Some(77), Some(89)));
}

#[test]
fn unknown_or_zero_duty_on_either_side_keeps_the_fixed_band() {
    for (from, to) in [
        (None, Some(89)),
        (Some(77), None),
        (Some(0), Some(89)),
        (Some(77), Some(0)),
    ] {
        assert!(!pump_in_band(1127, 1298, from, to), "{from:?} -> {to:?}");
        assert!(pump_in_band(1127, 1296, from, to), "{from:?} -> {to:?}");
    }
}

#[test]
fn extreme_values_do_not_overflow() {
    // 242 -> 255 is inside the step limit, so the multiply runs and the scaled
    // speed is above u32::MAX: it must saturate, not wrap to a small number.
    assert!(pump_in_band(u32::MAX, u32::MAX, Some(242), Some(255)));
    assert!(!pump_in_band(u32::MAX, 0, Some(242), Some(255)));
    // A wrapped scaled speed (about 2.3e8) would accept this.
    assert!(!pump_in_band(u32::MAX, 1_000_000_000, Some(242), Some(255)));
    assert!(pump_in_band(0, 150, Some(64), Some(51)));
    assert!(!pump_in_band(0, 151, Some(64), Some(51)));
}

#[test]
fn a_duty_drop_widens_the_envelope_downward() {
    // 89 -> 77: duties [64, 102]. 1304 * 64 / 89 = 938; low edge 788.
    // The fixed band alone would stop at 1304 - 195 = 1109.
    let low = explained(1304, 89, 64);
    assert_eq!(low, 938);
    let low_edge = low - pump_band_rpm(low);
    assert!(pump_in_band(1304, 1000, Some(89), Some(77)));
    assert!(pump_in_band(1304, low_edge, Some(89), Some(77)));
    assert!(!pump_in_band(1304, low_edge - 1, Some(89), Some(77)));
    // Held 89 is the same hunt: 1000 is inside duties [76, 102].
    assert!(pump_in_band(1304, 1000, Some(89), Some(89)));
}

#[test]
fn duty_jumps_beyond_the_step_limit_keep_the_fixed_band() {
    assert_eq!(PUMP_DUTY_STEP_MAX, 13);
    // Down to a few percent with the pump stopped.
    assert!(!pump_in_band(1127, 0, Some(77), Some(10)));
    // A drop of a third with the pump slowing to match.
    assert!(!pump_in_band(1127, 600, Some(77), Some(51)));
    // A jump to full duty with the pump following it, from low or from 84 %.
    assert!(!pump_in_band(1127, 2800, Some(77), Some(255)));
    assert!(!pump_in_band(2400, 2860, Some(214), Some(255)));
    // 13 counts follow; 14 do not (77 -> 90 / 91, 77 -> 64 / 63).
    assert!(pump_in_band(1127, 1450, Some(77), Some(90)));
    assert!(!pump_in_band(1127, 1450, Some(77), Some(91)));
    // 1127 * 64 / 77 = 937, low edge 787; the fixed band stops at 958.
    assert!(pump_in_band(1127, 790, Some(77), Some(64)));
    assert!(!pump_in_band(1127, 790, Some(77), Some(63)));
}

#[test]
fn every_five_point_firmware_step_follows() {
    // DIV_ROUND_CLOSEST(percent * 255, 100) for 20, 25, ... 100 %.
    let pwm: Vec<u32> = (4..=20).map(|step| (step * 5 * 255 + 50) / 100).collect();
    for pair in pwm.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        assert!(to - from <= PUMP_DUTY_STEP_MAX, "{from} -> {to}");
        let up = (1127 * to + from / 2) / from;
        assert!(
            pump_in_band(1127, up, Some(from), Some(to)),
            "{from} -> {to}"
        );
        let down = (1127 * from + to / 2) / to;
        assert!(
            pump_in_band(1127, down, Some(to), Some(from)),
            "{to} -> {from}"
        );
    }
    // The step one quantum below #147: 25 -> 30 %, pump about 937 -> 1127.
    // Held 64 is the same hunt as 21:54.
    assert!(pump_in_band(937, 1127, Some(64), Some(64)));
    assert!(pump_in_band(937, 1127, Some(64), Some(77)));
}
