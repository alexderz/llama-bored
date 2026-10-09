//! The loop with a fake keyboard: software mode once per open, frames on
//! change, absent logged once, fans unaffected, restart request, hand-back
//! on disable and on restore.

mod common;

use common::{FakeConfig, FakeKbOpener, FakeOpener, Feed, Lines, ManualClock, NoNotify, SEC, snap};
use llama_core::color::Rgb;
use llama_light::aura::AuraBackend;
use llama_light::config::parse;
use llama_light::keyboard::KeyboardBackend;
use llama_light::keyboard::device::OpenError;
use llama_light::keyboard::keymap::{KEYS, key_index};
use llama_light::palette::Palette;
use llama_light::service::{self, Light, Parts, RunEnd};

type TestLight = Light<ManualClock, Feed, FakeConfig, NoNotify, Lines>;

const SOFTWARE: [u8; 6] = [0x00, 0x07, 0x05, 0x02, 0x00, 0x03];
const HARDWARE: [u8; 6] = [0x00, 0x07, 0x05, 0x01, 0x00, 0x03];

const KB: &str = r##"
[aura]
brightness_max = 100
[keyboard]
enabled = true
[[light]]
target = "keyboard.all"
color = "#101010"
[[light]]
target = 'keyboard.keys["F1".."F12"]'
metric = "activity"
style = "bar"
"##;

struct Rig {
    light: TestLight,
    clock: ManualClock,
    feed: Feed,
    aura: FakeOpener,
    kb: FakeKbOpener,
    lines: Lines,
    config: FakeConfig,
}

fn rig(text: &str, kb: FakeKbOpener) -> Rig {
    let clock = ManualClock::at(100 * SEC);
    let lines = Lines::default();
    let config = FakeConfig::new(text);
    let aura = FakeOpener::present();
    let feed = Feed::new(snap(1, 100 * SEC));
    let light = Light::new(Parts {
        clock: clock.clone(),
        snapshots: feed.clone(),
        config_source: config.clone(),
        notify: NoNotify,
        sink: lines.clone(),
        config: parse(text).expect("config"),
        aura: Some(Box::new(AuraBackend::new(aura.clone()))),
        keyboard: Some(Box::new(KeyboardBackend::new(kb.clone()))),
    });
    Rig {
        light,
        clock,
        feed,
        aura,
        kb,
        lines,
        config,
    }
}

fn publish(r: &Rig, seq: u64, activity: f32) {
    let mut s = snap(seq, r.clock.now());
    s.host.activity_pct = Some(activity);
    r.feed.set(Ok(s));
}

/// The last frame the fake keyboard got, as slot colours.
fn last_slots(kb: &FakeKbOpener) -> [Rgb; 144] {
    let reports = kb.reports();
    let n = reports.len();
    assert!(n >= 12, "no whole frame: {n} reports");
    let frame = &reports[n - 12..];
    let mut out = [Rgb { r: 0, g: 0, b: 0 }; 144];
    for (c, chunk) in frame.chunks(4).enumerate() {
        let mut values = Vec::new();
        for stream in &chunk[..3] {
            assert_eq!(stream[1], 0x7F);
            let len = usize::from(stream[3]);
            values.extend_from_slice(&stream[5..5 + len]);
        }
        assert_eq!(values.len(), 144);
        for (slot, v) in out.iter_mut().zip(values) {
            match c {
                0 => slot.r = v,
                1 => slot.g = v,
                _ => slot.b = v,
            }
        }
    }
    out
}

fn slot_of(name: &str) -> usize {
    usize::from(KEYS[key_index(name).expect(name)].led)
}

#[test]
fn the_first_open_enters_software_mode_then_sends_one_frame() {
    let mut r = rig(KB, FakeKbOpener::present());
    publish(&r, 1, 50.0);
    r.light.run(Some(1));
    let reports = r.kb.reports();
    assert_eq!(reports.len(), 1 + 12);
    assert_eq!(&reports[0][..6], &SOFTWARE);
    assert!(reports[0][6..].iter().all(|b| *b == 0));
    assert_eq!(r.kb.frames(), 1);
    // Activity 50 %: F1..F6 lit by the bar, F7..F12 keep the base colour.
    let slots = last_slots(&r.kb);
    let base = Rgb {
        r: 0x10,
        g: 0x10,
        b: 0x10,
    };
    let lit = Palette::Act.color(50.0);
    for name in ["F1", "F6"] {
        assert_eq!(slots[slot_of(name)], lit, "{name}");
    }
    for name in ["F7", "F12", "Escape", "W", "Space"] {
        assert_eq!(slots[slot_of(name)], base, "{name}");
    }
    assert_eq!(slots[10], Rgb { r: 0, g: 0, b: 0 }, "an unused slot");
    // The fans got their own frame too (the default aura entry).
    assert_eq!(r.aura.direct_reports(), 1);
}

#[test]
fn an_unchanged_frame_is_not_sent_again() {
    let mut r = rig(KB, FakeKbOpener::present());
    for seq in 1..=20 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.kb.frames(), 1);
    publish(&r, 21, 90.0);
    r.light.run(Some(1));
    assert_eq!(r.kb.frames(), 2);
    assert_eq!(r.light.keyboard_frames_sent(), 2);
    assert_eq!(r.kb.opens(), 1);
}

#[test]
fn an_absent_keyboard_is_logged_once_and_the_fans_keep_working() {
    let kb = FakeKbOpener::absent();
    let mut r = rig(KB, kb.clone());
    for seq in 0..600 {
        publish(&r, seq, (seq % 100) as f32);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("keyboard absent"), 1, "{:?}", r.lines.all());
    assert!(r.aura.direct_reports() > 100, "fans kept updating");
    assert!(kb.reports().is_empty());
    // It returns: one line, software mode, frames.
    kb.set_present(true);
    for seq in 600..720 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("keyboard present"), 1);
    assert_eq!(kb.opens(), 1);
    assert_eq!(&kb.reports()[0][..6], &SOFTWARE);
    // Unplugged mid-run: one line, then silence; the fans carry on.
    kb.set_present(false);
    let fans_before = r.aura.direct_reports();
    for seq in 720..1300 {
        publish(&r, seq, (seq % 100) as f32);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("keyboard absent"), 2, "{:?}", r.lines.all());
    assert_eq!(r.lines.count("keyboard present"), 1);
    assert!(r.aura.direct_reports() > fans_before + 100);
    // Plugged back in: one line, software mode again.
    kb.set_present(true);
    for seq in 1300..1420 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("keyboard present"), 2);
    assert_eq!(kb.opens(), 2);
    let lines = r.lines.all();
    assert!(lines.len() <= 6, "no spam: {lines:?}");
}

#[test]
fn a_keyboard_that_needs_a_unit_restart_ends_the_run_once_with_one_line() {
    let kb = FakeKbOpener::absent();
    kb.refuse(Some(OpenError::NeedsRestart));
    let mut r = rig(KB, kb.clone());
    publish(&r, 1, 50.0);
    assert_eq!(r.light.run(Some(100)), RunEnd::Restart);
    assert_eq!(r.lines.count("systemd restarts llama-light"), 1);
    // The fans were served on that tick.
    assert_eq!(r.aura.direct_reports(), 1);
}

#[test]
fn a_refusal_for_a_known_node_is_just_absent() {
    let kb = FakeKbOpener::absent();
    kb.refuse(Some(OpenError::Denied));
    let mut r = rig(KB, kb.clone());
    publish(&r, 1, 50.0);
    assert_eq!(r.light.run(Some(300)), RunEnd::Ticks);
    assert_eq!(r.lines.count("keyboard absent"), 1);
    assert_eq!(r.lines.count("restart"), 0);
}

#[test]
fn a_keyboard_left_off_is_never_opened() {
    let kb = FakeKbOpener::present();
    let mut r = rig("[aura]\nbrightness_max = 100\n", kb.clone());
    publish(&r, 1, 50.0);
    r.light.run(Some(300));
    assert_eq!(kb.opens(), 0);
    assert!(kb.reports().is_empty());
}

#[test]
fn disabling_the_keyboard_by_reload_hands_it_back_once() {
    let kb = FakeKbOpener::present();
    let mut r = rig(KB, kb.clone());
    publish(&r, 1, 50.0);
    r.light.run(Some(1));
    r.config.edit("[aura]\nbrightness_max = 100\n");
    for seq in 2..30 {
        publish(&r, seq, 70.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("config reloaded"), 1);
    let reports = kb.reports();
    assert_eq!(&reports.last().expect("report")[..6], &HARDWARE);
    assert_eq!(reports.iter().filter(|b| b[..6] == HARDWARE).count(), 1);
    let after = reports.len();
    publish(&r, 40, 10.0);
    r.light.run(Some(50));
    assert_eq!(kb.reports().len(), after, "nothing after the hand-back");
}

#[test]
fn a_stale_snapshot_fades_the_keyboard_to_the_dim_neutral() {
    let mut r = rig(KB, FakeKbOpener::present());
    publish(&r, 1, 100.0);
    r.light.run(Some(1));
    r.light.run(Some(400));
    let slots = last_slots(&r.kb);
    let n = llama_light::mapping::NEUTRAL;
    assert_eq!(slots[slot_of("F1")], n);
    assert_eq!(slots[slot_of("W")], n);
}

#[test]
fn restore_hands_the_keyboard_back_to_hardware_lighting() {
    let config = parse(KB).expect("config");
    let kb = FakeKbOpener::present();
    let mut keyboard = KeyboardBackend::new(kb.clone());
    let mut aura = AuraBackend::new(FakeOpener::present());
    let mut lines = Lines::default();
    assert_eq!(
        service::restore(&config, &mut aura, Some(&mut keyboard), &mut lines),
        0
    );
    let reports = kb.reports();
    assert_eq!(
        reports.len(),
        1,
        "hardware mode only, no software mode first"
    );
    assert_eq!(&reports[0][..6], &HARDWARE);
    assert!(reports[0][6..].iter().all(|b| *b == 0));
    assert_eq!(lines.count("keyboard handed back"), 1);
}

#[test]
fn restore_with_the_keyboard_absent_is_one_line_and_success() {
    let config = parse(KB).expect("config");
    let mut keyboard = KeyboardBackend::new(FakeKbOpener::absent());
    let mut aura = AuraBackend::new(FakeOpener::absent());
    let mut lines = Lines::default();
    assert_eq!(
        service::restore(&config, &mut aura, Some(&mut keyboard), &mut lines),
        0
    );
    assert_eq!(lines.count("keyboard absent"), 1);
    assert_eq!(lines.all().len(), 2);
}

#[test]
fn restore_reports_a_failed_keyboard_write() {
    let config = parse(KB).expect("config");
    let kb = FakeKbOpener::present();
    kb.0.borrow_mut().fail_writes = true;
    let mut keyboard = KeyboardBackend::new(kb);
    let mut aura = AuraBackend::new(FakeOpener::absent());
    let mut lines = Lines::default();
    assert_eq!(
        service::restore(&config, &mut aura, Some(&mut keyboard), &mut lines),
        1
    );
    assert_eq!(lines.count("keyboard write failed"), 1);
}
