//! #147: the pump band follows the duty the firmware reports in `pwm1`.

use kraken_lcd::device::{PUMP_DUTY_STEP_MAX, pump_band_rpm, pump_in_band};

#[test]
fn held_duty_keeps_the_fixed_band() {
    let band = pump_band_rpm(1127);
    assert_eq!(band, 169);
    for duty in [Some(77), None, Some(0)] {
        assert!(pump_in_band(1127, 1127 + band, duty, duty), "{duty:?}");
        assert!(!pump_in_band(1127, 1127 + band + 1, duty, duty), "{duty:?}");
        assert!(pump_in_band(1127, 1127 - band, duty, duty), "{duty:?}");
        assert!(!pump_in_band(1127, 1127 - band - 1, duty, duty), "{duty:?}");
    }
}

#[test]
fn the_logged_halt_is_inside_the_band_now() {
    assert!(!pump_in_band(1127, 1298, Some(77), Some(77)));
    assert!(pump_in_band(1127, 1298, Some(77), Some(89)));
    assert!(pump_in_band(1298, 1127, Some(89), Some(77)));
}

#[test]
fn range_runs_from_the_old_speed_to_the_scaled_one() {
    // 1127 * 89 / 77 = 1302.6 -> 1303; band at that end is 195.
    let high = 1303 + pump_band_rpm(1303);
    assert!(pump_in_band(1127, high, Some(77), Some(89)));
    assert!(!pump_in_band(1127, high + 1, Some(77), Some(89)));
    let low = 1127 - pump_band_rpm(1127);
    assert!(pump_in_band(1127, low, Some(77), Some(89)));
    assert!(!pump_in_band(1127, low - 1, Some(77), Some(89)));
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
fn a_duty_drop_widens_the_band_downward() {
    // 1304 * 77 / 89 = 1128; low edge 1128 - 169 = 959. The fixed band alone
    // would stop at 1304 - 195 = 1109.
    assert!(pump_in_band(1304, 1000, Some(89), Some(77)));
    assert!(pump_in_band(1304, 959, Some(89), Some(77)));
    assert!(!pump_in_band(1304, 958, Some(89), Some(77)));
    assert!(!pump_in_band(1304, 1000, Some(89), Some(89)));
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
    assert!(!pump_in_band(937, 1127, Some(64), Some(64)));
    assert!(pump_in_band(937, 1127, Some(64), Some(77)));
}
