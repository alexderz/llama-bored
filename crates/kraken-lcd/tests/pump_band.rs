//! #147: the pump band follows the duty the firmware reports in `pwm1`.

use kraken_lcd::device::{PUMP_DUTY_STEP_PERCENT, pump_band_rpm, pump_in_band};

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
    assert!(pump_in_band(u32::MAX, u32::MAX, Some(1), Some(u32::MAX)));
    assert!(!pump_in_band(u32::MAX, 0, Some(1), Some(u32::MAX)));
    assert!(pump_in_band(0, 150, Some(255), Some(1)));
    assert!(!pump_in_band(0, 151, Some(255), Some(1)));
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
    assert_eq!(PUMP_DUTY_STEP_PERCENT, 20);
    // Down to a few percent with the pump stopped: the review's case.
    assert!(!pump_in_band(1127, 0, Some(77), Some(10)));
    // A drop of a third with the pump slowing to match.
    assert!(!pump_in_band(1127, 600, Some(77), Some(51)));
    // A jump to full duty with the pump following it.
    assert!(!pump_in_band(1127, 2800, Some(77), Some(255)));
    // Exactly 20 % follows; one count more does not (100 -> 120 / 121).
    assert!(pump_in_band(1000, 1350, Some(100), Some(120)));
    assert!(!pump_in_band(1000, 1350, Some(100), Some(121)));
    assert!(pump_in_band(1000, 650, Some(100), Some(80)));
    assert!(!pump_in_band(1000, 650, Some(100), Some(79)));
}
