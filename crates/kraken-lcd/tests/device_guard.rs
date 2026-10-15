//! S7 device-level cooling guard. Fake sysfs only; no real device is opened.

#[path = "device_fixture.rs"]
mod fixture;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use kraken_lcd::device::proto::{self, Cmd};
use kraken_lcd::device::{LcdSink, SinkError};
use kraken_lcd::render::Frame;

use fixture::{Tree, ack, install_kraken, script_show_slot};

fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn show_ready(label: &str) -> fixture::Opened {
    let opened = fixture::opened(label);
    script_show_slot(&opened.hid, 0);
    opened
}

fn assert_halted(opened: &mut fixture::Opened, reason: &str) {
    let latch = std::fs::read_to_string(opened.tree.latch()).expect("latch");
    assert!(
        latch.contains("CRITICAL") && latch.contains("baseline") && latch.contains("current"),
        "{reason}: {latch}"
    );
    assert!(opened.lcd.is_halted(), "{reason}");
    let after = opened.hid.sent().len();
    let bulk_at_halt = opened.bulk.chunks().len();
    opened.lcd.restore_stock();
    assert_eq!(
        opened.lcd.show(&Frame::new()),
        Err(SinkError::Halted),
        "{reason}"
    );
    assert_eq!(
        opened.hid.sent().len(),
        after,
        "{reason}: no further reports"
    );
    assert_eq!(
        opened.bulk.chunks().len(),
        bulk_at_halt,
        "{reason}: halt must not write more bulk bytes"
    );
}

fn mutate_after_baseline(opened: &fixture::Opened, mutate: impl Fn(&Tree) + Send + Sync + 'static) {
    let sys_tree = tree_clone(&opened.tree);
    opened.hid.on_send(move |report| {
        if report[0] == 0x36 && report[1] == 0x03 {
            mutate(&sys_tree);
        }
    });
}

fn tree_clone(tree: &Tree) -> Tree {
    Tree {
        root: tree.root.clone(),
        sys: tree.sys.clone(),
        state: tree.state.clone(),
    }
}

#[test]
fn stream_pace_guard_sees_a_drift_after_the_first_check() {
    let mut opened = show_ready("stream-rearm");
    opened.lcd.show(&Frame::new()).expect("show");
    opened
        .lcd
        .pace_guard()
        .expect("first stream check consumes the baseline and arms the next one");
    std::fs::write(opened.tree.hwmon().join("pwm2_enable"), "2\n").expect("enable");
    assert_eq!(
        opened.lcd.pace_guard(),
        Err(SinkError::Halted),
        "a 1 Hz stream check must still see a cooling change"
    );
    assert_halted(&mut opened, "stream pace guard");
}

#[test]
fn z53_removed_after_the_op_and_at_the_next_tick() {
    let mut opened = show_ready("z53-after");
    mutate_after_baseline(&opened, |tree| {
        std::fs::remove_dir_all(tree.hwmon()).expect("remove z53");
    });
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "z53 after op");

    let mut opened = show_ready("z53-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    std::fs::remove_dir_all(opened.tree.hwmon()).expect("remove z53");
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "z53 tick");
}

#[test]
fn pwm2_enable_change_halts_after_the_op_and_at_the_next_tick() {
    let mut opened = show_ready("pwm-after");
    mutate_after_baseline(&opened, |tree| {
        std::fs::write(tree.hwmon().join("pwm2_enable"), "1\n").expect("enable");
    });
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    let latch = std::fs::read_to_string(opened.tree.latch()).expect("latch");
    assert!(latch.contains("pwm2_enable=0"), "{latch}");
    assert!(latch.contains("pwm2_enable=1"), "{latch}");
    assert_halted(&mut opened, "pwm2 after op");

    let mut opened = show_ready("pwm-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    std::fs::write(opened.tree.hwmon().join("pwm2_enable"), "2\n").expect("enable");
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "pwm2 tick");
}

#[test]
fn pump_drop_of_twenty_percent_halts_after_the_op_and_at_the_next_tick() {
    let dropped = fixture::PUMP_RPM * 80 / 100;
    let mut opened = show_ready("pump-after");
    mutate_after_baseline(&opened, move |tree| {
        std::fs::write(tree.hwmon().join("fan1_input"), format!("{dropped}\n")).expect("rpm");
    });
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "pump after op");

    let mut opened = show_ready("pump-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    std::fs::write(
        opened.tree.hwmon().join("fan1_input"),
        format!("{dropped}\n"),
    )
    .expect("rpm");
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "pump tick");
}

#[test]
fn devnum_change_halts_after_the_op_and_at_the_next_tick() {
    let mut opened = show_ready("dev-after");
    mutate_after_baseline(&opened, |tree| {
        std::fs::write(tree.dev_dir().join("devnum"), "9\n").expect("devnum");
    });
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "devnum after op");

    let mut opened = show_ready("dev-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    std::fs::write(opened.tree.dev_dir().join("devnum"), "9\n").expect("devnum");
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "devnum tick");
}

#[test]
fn bootloader_appearance_halts_after_the_op_and_at_the_next_tick() {
    let mut opened = show_ready("bl-after");
    mutate_after_baseline(&opened, fixture::add_bootloader);
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "bootloader after op");

    let mut opened = show_ready("bl-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    fixture::add_bootloader(&opened.tree);
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "bootloader tick");
}

#[test]
fn pwm_moving_with_enable_unchanged_does_not_halt() {
    let mut opened = show_ready("pwm-curve");
    mutate_after_baseline(&opened, |tree| {
        std::fs::write(tree.hwmon().join("pwm1"), "77\n").expect("pwm1");
    });
    opened.lcd.show(&Frame::new()).expect("firmware curve");
    assert!(!opened.tree.latch().exists());
    assert!(!opened.lcd.is_halted());
    opened.lcd.tick().expect("tick");
    assert!(!opened.tree.latch().exists());
}

#[test]
fn pump_change_inside_the_band_does_not_halt() {
    let mut opened = show_ready("band");
    let nudged = fixture::PUMP_RPM + 150;
    mutate_after_baseline(&opened, move |tree| {
        std::fs::write(tree.hwmon().join("fan1_input"), format!("{nudged}\n")).expect("rpm");
    });
    opened.lcd.show(&Frame::new()).expect("inside band");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn transfer_drop_with_a_deviation_sends_nothing_and_latches() {
    let mut opened = show_ready("drop");
    opened.bulk.fail_at(0);
    let sys = tree_clone(&opened.tree);
    opened.hid.on_send(move |report| {
        if report[0] == 0x36 && report[1] == 0x01 {
            std::fs::write(sys.hwmon().join("pwm2_enable"), "1\n").expect("enable");
        }
    });
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    let sent = opened.hid.sent();
    assert_eq!(
        fixture::count_prefix(&sent, &[0x38, 0x01, 0x02, 0x00]),
        0,
        "ShowLiquid must not be sent"
    );
    assert!(opened.bulk.chunks().is_empty());
    assert_eq!(opened.hid.alive.load(Ordering::Relaxed), 0);
    assert_eq!(opened.bulk.alive.load(Ordering::Relaxed), 0);
    let latch = std::fs::read_to_string(opened.tree.latch()).expect("latch");
    assert!(
        latch.contains("baseline") && latch.contains("current"),
        "{latch}"
    );
    let after = sent.len();
    opened.lcd.restore_stock();
    assert_eq!(opened.hid.sent().len(), after);
}

#[test]
fn guard_reads_do_not_require_the_real_sys_tree() {
    let tree = Tree::new("fixture-hwmon");
    install_kraken(&tree);
    let text = std::fs::read_to_string(tree.hwmon().join("fan1_input")).expect("rpm");
    assert_eq!(text.trim(), "1304");
    let committed = std::fs::read_to_string(fixtures().join("sys/class/hwmon/hwmon5/fan1_input"))
        .expect("committed z53 pump");
    assert_eq!(committed.trim(), "1304");
    let _ = ack(proto::expected_prefix(&Cmd::ShowLiquid));
    let _ = Arc::new(0);
}

/// Writes the pump rpm and both duties, keeping both modes at firmware control.
fn set_pump(tree: &Tree, rpm: u32, duty: u32) {
    fixture::write_hwmon(tree, rpm, duty, duty, 0, 0);
}

#[test]
fn firmware_duty_step_with_the_pump_following_does_not_halt() {
    // The values a real Kraken Z53 logged when it latched under #147.
    let mut opened = show_ready("duty-up");
    set_pump(&opened.tree, 1127, 77);
    mutate_after_baseline(&opened, |tree| set_pump(tree, 1298, 89));
    opened
        .lcd
        .show(&Frame::new())
        .expect("duty step after the op");
    opened.lcd.tick().expect("duty step at the follow-up");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());

    let mut opened = show_ready("duty-up-tick");
    set_pump(&opened.tree, 1127, 77);
    opened.lcd.show(&Frame::new()).expect("show");
    set_pump(&opened.tree, 1298, 89);
    opened.lcd.tick().expect("duty step at the next tick");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn pump_still_ramping_after_a_duty_step_does_not_halt() {
    let mut opened = show_ready("ramping");
    mutate_after_baseline(&opened, |tree| set_pump(tree, fixture::PUMP_RPM, 77));
    opened
        .lcd
        .show(&Frame::new())
        .expect("pump not caught up yet");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn stopped_pump_after_a_duty_step_halts() {
    let mut opened = show_ready("duty-up-stall");
    set_pump(&opened.tree, 1127, 77);
    mutate_after_baseline(&opened, |tree| set_pump(tree, 0, 89));
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "stall after a duty rise");

    let mut opened = show_ready("duty-down-stall-tick");
    opened.lcd.show(&Frame::new()).expect("show");
    set_pump(&opened.tree, 600, 77);
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "pump far under a lowered duty");
}

#[test]
fn pump_change_the_duty_does_not_explain_halts() {
    let mut opened = show_ready("duty-up-overshoot");
    set_pump(&opened.tree, 1127, 77);
    mutate_after_baseline(&opened, |tree| set_pump(tree, 1700, 89));
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "pump far over a raised duty");

    // Same rpm jump with the duty held: the band does not move.
    let mut opened = show_ready("held-duty-jump");
    set_pump(&opened.tree, 1127, 77);
    mutate_after_baseline(&opened, |tree| set_pump(tree, 1298, 77));
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "jump with the duty held");
}

#[test]
fn firmware_duty_step_down_with_the_pump_following_does_not_halt() {
    // Outside the fixed band (1304 - 195 = 1109), inside the followed one.
    let mut opened = show_ready("duty-down");
    mutate_after_baseline(&opened, |tree| set_pump(tree, 1000, 77));
    opened
        .lcd
        .show(&Frame::new())
        .expect("duty step down after the op");
    opened.lcd.tick().expect("duty step down at the follow-up");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn pump_still_ramping_to_a_duty_set_before_the_baseline_does_not_halt() {
    // #154: firmware stepped 64 -> 77 just before the baseline; the pump was
    // still at 953 rpm, then 1121 at the follow-up, with pwm1 held at 77.
    let mut opened = show_ready("prior-up");
    set_pump(&opened.tree, 1127, 77);
    opened
        .lcd
        .pace_guard()
        .expect("one firmware step down from the open baseline");
    set_pump(&opened.tree, 953, 64);
    opened
        .lcd
        .pace_guard()
        .expect("one firmware step down to 25 %");
    set_pump(&opened.tree, 953, 77);
    opened
        .lcd
        .show(&Frame::new())
        .expect("baseline after the rise, pump still at 25 %");
    set_pump(&opened.tree, 1121, 77);
    opened
        .lcd
        .tick()
        .expect("pump arrived at 30 % by the follow-up");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn pump_still_fast_after_a_duty_drop_before_the_baseline_does_not_halt() {
    // Open remembered pwm1=89 at 1304 rpm. Duty already 77, pump still fast.
    let mut opened = show_ready("prior-down");
    set_pump(&opened.tree, 1304, 77);
    opened
        .lcd
        .show(&Frame::new())
        .expect("baseline after the drop, pump still at 35 %");
    set_pump(&opened.tree, 1000, 77);
    opened
        .lcd
        .tick()
        .expect("pump arrived at 30 % by the follow-up");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn duty_jump_beyond_the_step_limit_halts() {
    let mut opened = show_ready("duty-collapse");
    set_pump(&opened.tree, 1127, 77);
    mutate_after_baseline(&opened, |tree| set_pump(tree, 0, 10));
    assert_eq!(opened.lcd.show(&Frame::new()), Err(SinkError::Halted));
    assert_halted(&mut opened, "duty down to 4 % with the pump stopped");

    let mut opened = show_ready("duty-full-tick");
    set_pump(&opened.tree, 1127, 77);
    opened.lcd.show(&Frame::new()).expect("show");
    set_pump(&opened.tree, 2800, 255);
    assert_eq!(opened.lcd.tick(), Err(SinkError::Halted));
    assert_halted(&mut opened, "duty jump to full speed");
}
