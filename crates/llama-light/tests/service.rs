//! The loop with fakes: change-only sending, staleness and fade, absent
//! devices logged once, config reload, restore. Keyboard: tests/keyboard_service.rs.

mod common;

use common::{FakeConfig, FakeOpener, Feed, Lines, ManualClock, NoNotify, SEC, snap};
use llama_core::color::act_color;
use llama_light::aura::AuraBackend;
use llama_light::backend::Backend;
use llama_light::config::{LightConfig, parse};
use llama_light::mapping::{NEUTRAL, cap, neutral_frame};
use llama_light::service::{self, Light, Parts, Phase, phase};
use std::time::Duration;

type TestLight = Light<ManualClock, Feed, FakeConfig, NoNotify, Lines>;

const CONFIG: &str = "[aura]\nbrightness_max = 100\nfps = 10\n";

struct Rig {
    light: TestLight,
    clock: ManualClock,
    feed: Feed,
    aura: FakeOpener,
    lines: Lines,
    config: FakeConfig,
}

fn rig_with(text: &str, aura: FakeOpener, feed: Feed, keyboard: Option<Box<dyn Backend>>) -> Rig {
    let clock = ManualClock::at(100 * SEC);
    let lines = Lines::default();
    let config = FakeConfig::new(text);
    let light = Light::new(Parts {
        clock: clock.clone(),
        snapshots: feed.clone(),
        config_source: config.clone(),
        notify: NoNotify,
        sink: lines.clone(),
        config: parse(text).expect("config"),
        aura: Some(Box::new(AuraBackend::new(aura.clone()))),
        keyboard,
    });
    Rig {
        light,
        clock,
        feed,
        aura,
        lines,
        config,
    }
}

fn rig() -> Rig {
    rig_with(
        CONFIG,
        FakeOpener::present(),
        Feed::new(snap(1, 100 * SEC)),
        None,
    )
}

/// Advance the watcher: a new snapshot stamped "now".
fn publish(r: &Rig, seq: u64, activity: f32) {
    let mut s = snap(seq, r.clock.now());
    s.host.activity_pct = Some(activity);
    r.feed.set(Ok(s));
}

#[test]
fn the_first_open_enters_direct_mode_then_sends_one_frame() {
    let mut r = rig();
    r.light.run(Some(1));
    let reports = r.aura.reports();
    assert_eq!(reports.len(), 2);
    assert_eq!(&reports[0][..6], &[0xEC, 0x35, 0x01, 0x00, 0x00, 0xFF]);
    assert_eq!(&reports[1][..5], &[0xEC, 0x40, 0x80, 0x00, 0x06]);
    let color = act_color(50.0);
    assert_eq!(&reports[1][5..8], &[color.r, color.g, color.b]);
}

#[test]
fn an_unchanged_frame_is_not_sent_again() {
    let mut r = rig();
    for seq in 1..=20 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.aura.direct_reports(), 1, "twenty ticks, one frame");
    publish(&r, 21, 60.0);
    r.light.run(Some(1));
    assert_eq!(r.aura.direct_reports(), 2, "a changed value sends once");
    publish(&r, 22, 60.0);
    r.light.run(Some(1));
    assert_eq!(r.aura.direct_reports(), 2);
    assert_eq!(r.light.frames_sent(), 2);
}

#[test]
fn stale_policy_holds_then_fades() {
    assert_eq!(phase(Duration::from_secs(0)), Phase::Live);
    assert_eq!(phase(Duration::from_secs(5)), Phase::Live);
    assert_eq!(phase(Duration::from_millis(5_001)), Phase::Hold);
    assert_eq!(phase(Duration::from_secs(30)), Phase::Hold);
    assert_eq!(phase(Duration::from_millis(32_500)), Phase::Fade(0.5));
    assert_eq!(phase(Duration::from_secs(35)), Phase::Fade(1.0));
    assert_eq!(phase(Duration::from_secs(3600)), Phase::Fade(1.0));
}

#[test]
fn a_stale_snapshot_holds_the_frame_then_fades_to_dim_neutral() {
    let mut r = rig();
    publish(&r, 1, 100.0);
    r.light.run(Some(1));
    assert_eq!(r.aura.direct_reports(), 1);
    // The watcher stops. The same snapshot keeps being read.
    // 0.1 s per tick: 30 s is 300 ticks; nothing new may be sent.
    r.light.run(Some(299));
    assert_eq!(r.aura.direct_reports(), 1, "held for 30 s: nothing sent");
    assert!(r.lines.count("snapshot stale") == 1, "{:?}", r.lines.all());
    // Fade over the next 5 s: frames move toward neutral, then stop.
    r.light.run(Some(60));
    let sent = r.aura.direct_reports();
    assert!(sent > 10, "the fade sends frames: {sent}");
    let last = r.aura.reports().last().copied().expect("report");
    let neutral = NEUTRAL;
    assert_eq!(
        &last[5..8],
        &[neutral.r, neutral.g, neutral.b],
        "ends on the neutral"
    );
    r.light.run(Some(100));
    assert_eq!(r.aura.direct_reports(), sent, "nothing more once faded");
    // The watcher returns: live again.
    publish(&r, 2, 100.0);
    r.light.run(Some(1));
    assert_eq!(r.aura.direct_reports(), sent + 1);
    assert_eq!(r.lines.count("snapshot fresh"), 2);
}

#[test]
fn with_no_snapshot_ever_nothing_is_sent_until_the_neutral_at_thirty_seconds() {
    let mut r = rig_with(CONFIG, FakeOpener::present(), Feed::missing(), None);
    r.light.run(Some(290));
    assert_eq!(r.aura.direct_reports(), 0);
    assert_eq!(r.lines.count("snapshot missing"), 1);
    r.light.run(Some(20));
    assert!(r.aura.direct_reports() >= 1);
    let last = r.aura.reports().last().copied().expect("report");
    assert_eq!(&last[5..8], &[NEUTRAL.r, NEUTRAL.g, NEUTRAL.b]);
}

#[test]
fn a_snapshot_from_the_future_is_not_accepted() {
    let r = rig();
    let mut s = snap(1, r.clock.now() + 10 * SEC);
    s.host.activity_pct = Some(10.0);
    r.feed.set(Ok(s));
    let mut r = r;
    r.light.run(Some(3));
    assert_eq!(r.aura.direct_reports(), 0);
    assert_eq!(r.lines.count("snapshot from the future"), 1);
}

#[test]
fn an_absent_aura_is_logged_once_and_rescanned_every_ten_seconds() {
    let aura = FakeOpener::absent();
    let mut r = rig_with(CONFIG, aura.clone(), Feed::new(snap(1, 100 * SEC)), None);
    // 60 s of ticks at 10 fps.
    for seq in 0..600 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("aura absent"), 1, "{:?}", r.lines.all());
    assert_eq!(r.lines.count("aura present"), 0);
    assert!(r.aura.reports().is_empty());
    // It returns: one line, Direct mode, one frame.
    aura.set_present(true);
    for seq in 600..720 {
        publish(&r, seq, 50.0);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("aura present"), 1);
    assert_eq!(aura.0.borrow().opens, 1, "rescans are 10 s apart; one open");
    assert_eq!(r.aura.direct_reports(), 1);
    // Unplugged mid-run: one line, then silence.
    aura.set_present(false);
    for seq in 720..1000 {
        publish(&r, seq, (seq % 100) as f32);
        r.light.run(Some(1));
    }
    assert_eq!(r.lines.count("aura absent"), 2, "{:?}", r.lines.all());
    assert_eq!(r.lines.count("aura present"), 1);
    // Every line is either a presence or a snapshot transition: no spam.
    assert!(r.lines.all().len() <= 5, "{:?}", r.lines.all());
}

#[test]
fn rescans_are_ten_seconds_apart() {
    let aura = FakeOpener::absent();
    let mut r = rig_with(CONFIG, aura.clone(), Feed::new(snap(1, 100 * SEC)), None);
    r.light.run(Some(1));
    aura.set_present(true);
    r.light.run(Some(99)); // 9.9 s later
    assert_eq!(aura.0.borrow().opens, 0);
    r.light.run(Some(1)); // 10 s
    assert_eq!(aura.0.borrow().opens, 1);
}

#[test]
fn a_valid_config_edit_is_picked_up_within_two_seconds() {
    let mut r = rig();
    r.light.run(Some(1));
    let before = r.aura.direct_reports();
    r.config
        .edit("[aura]\nbrightness_max = 100\nstyle = \"ring\"\n");
    r.light.run(Some(25));
    assert_eq!(r.lines.count("config reloaded"), 1);
    assert_eq!(
        r.light.config().layers[0].style,
        llama_light::config::Style::Gauge
    );
    assert_eq!(r.aura.direct_reports(), before + 1, "new style, new frame");
}

#[test]
fn an_invalid_config_edit_is_logged_once_and_ignored() {
    let mut r = rig();
    r.light.run(Some(1));
    let running: LightConfig = r.light.config().clone();
    r.config.edit("[aura]\nbrightness_max = 500\n");
    r.light.run(Some(100));
    assert_eq!(
        r.lines.count("config reload rejected"),
        1,
        "{:?}",
        r.lines.all()
    );
    assert!(
        r.lines
            .all()
            .iter()
            .any(|l| l.contains("aura.brightness_max = 500"))
    );
    assert_eq!(r.light.config(), &running);
    // Fixing it is picked up.
    r.config.edit("[aura]\nbrightness_max = 40\n");
    r.light.run(Some(25));
    assert_eq!(r.lines.count("config reloaded"), 1);
    assert_eq!(r.light.config().aura.brightness_max, 40);
}

#[test]
fn a_deleted_config_file_is_logged_once_and_the_running_config_kept() {
    let mut r = rig();
    r.light.run(Some(1));
    let running = r.light.config().clone();
    r.config.remove();
    r.light.run(Some(300));
    assert_eq!(
        r.lines.count("config reload rejected"),
        1,
        "{:?}",
        r.lines.all()
    );
    assert_eq!(r.light.config(), &running);
    r.config.edit("[aura]\nbrightness_max = 30\n");
    r.light.run(Some(25));
    assert_eq!(r.light.config().aura.brightness_max, 30);
}

#[test]
fn disabling_aura_by_reload_leaves_the_neutral_and_stops_sending() {
    let mut r = rig();
    r.light.run(Some(1));
    r.config
        .edit("[aura]\nenabled = false\nbrightness_max = 100\n");
    r.light.run(Some(25));
    let last = r.aura.reports().last().copied().expect("report");
    assert_eq!(&last[5..8], &[NEUTRAL.r, NEUTRAL.g, NEUTRAL.b]);
    let sent = r.aura.direct_reports();
    publish(&r, 9, 99.0);
    r.light.run(Some(10));
    assert_eq!(r.aura.direct_reports(), sent);
}

#[test]
fn restore_sends_the_neutral_static_frame_in_direct_mode() {
    let config =
        parse("[aura]\nbrightness_max = 50\nfans = \"chain\"\nchain_len = 5\n").expect("config");
    let opener = FakeOpener::present();
    let mut aura = AuraBackend::new(opener.clone());
    let mut lines = Lines::default();
    assert_eq!(service::restore(&config, &mut aura, None, &mut lines), 0);
    let reports = opener.reports();
    assert_eq!(&reports[0][..6], &[0xEC, 0x35, 0x01, 0x00, 0x00, 0xFF]);
    assert_eq!(reports.len(), 1 + 2, "30 LEDs are two direct reports");
    let n = cap(&[NEUTRAL], 50)[0];
    assert_eq!(&reports[1][5..8], &[n.r, n.g, n.b]);
    assert_eq!(neutral_frame(&config.aura).len(), 30);
    assert!(reports.iter().all(|r| r[1] == 0x35 || r[1] == 0x40));
}

#[test]
fn restore_with_the_aura_absent_is_one_line_and_success() {
    let config = parse("").expect("config");
    let mut aura = AuraBackend::new(FakeOpener::absent());
    let mut lines = Lines::default();
    assert_eq!(service::restore(&config, &mut aura, None, &mut lines), 0);
    assert_eq!(lines.all().len(), 1);
    assert!(lines.all()[0].contains("aura absent"));
}
