//! Change-only upload policy: first frame, idle, flap, force, and the interval bound.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use kraken_lcd::config::{Config, InvalidConfig};
use kraken_lcd::policy::{Decision, Policy};
use kraken_lcd::present::{Ai, Band, View};

fn origin() -> Instant {
    Instant::now()
}

fn view_with(ai: Ai, ring_pct: Option<u8>) -> View {
    View {
        ring_pct,
        ring_band: None,
        blocks: [Band::Quiet; 3],
        coolant_c: Some(36),
        cpu_c: Some(42),
        gpu_c: Some(51),
        cpu_pct: Some(3),
        mem_pct: Some(40),
        ai,
        models: Vec::new(),
        model_count: 0,
        ..View::default()
    }
}

fn idle() -> View {
    view_with(Ai::Idle, Some(0))
}

fn loaded(ring_pct: u8) -> View {
    view_with(Ai::Loaded, Some(ring_pct))
}

/// Nothing has been sent since the device was opened, so the first frame uploads at once.
#[test]
fn first_frame_after_open_uploads_immediately() {
    let policy = Policy::new(Duration::from_secs(60));
    assert_eq!(policy.decide(&idle(), origin()), Decision::Upload);
}

/// `decide` does not record success. The same view stays `Upload` until `mark_uploaded`.
#[test]
fn upload_counts_only_after_mark_uploaded() {
    let mut policy = Policy::new(Duration::from_secs(60));
    let t0 = origin();
    let view = idle();
    assert_eq!(policy.decide(&view, t0), Decision::Upload);
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Upload,
        "deciding Upload does not record success"
    );
    policy.mark_uploaded(&view, t0);
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Nothing
    );
}

/// A different view waits until `min_interval`, then the view passed at that tick is uploaded.
#[test]
fn changed_view_waits_until_the_interval_then_uploads_the_latest() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let first = idle();
    let middle = loaded(40);
    let latest = loaded(80);
    policy.mark_uploaded(&first, t0);

    assert_eq!(
        policy.decide(&middle, t0 + interval - Duration::from_nanos(1)),
        Decision::Wait
    );
    assert_eq!(
        policy.decide(&first, t0 + Duration::from_secs(10)),
        Decision::Nothing,
        "returning to the uploaded view drops the intermediate frame"
    );
    assert_eq!(
        policy.decide(&latest, t0 + Duration::from_secs(59)),
        Decision::Wait
    );
    assert_eq!(policy.decide(&latest, t0 + interval), Decision::Upload);
    policy.mark_uploaded(&latest, t0 + interval);
    assert_eq!(
        policy.decide(&latest, t0 + interval + Duration::from_secs(1)),
        Decision::Nothing
    );
}

/// `force` still wants a send, but not before `min_interval` from the last success.
#[test]
fn force_waits_out_the_interval_then_uploads_the_same_view() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let view = idle();
    policy.mark_uploaded(&view, t0);
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Nothing
    );
    policy.force();
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Wait
    );
    assert_eq!(
        policy.decide(&view, t0 + interval - Duration::from_nanos(1)),
        Decision::Wait
    );
    assert_eq!(policy.decide(&view, t0 + interval), Decision::Upload);
    policy.mark_uploaded(&view, t0 + interval);
    assert_eq!(
        policy.decide(&view, t0 + interval + Duration::from_secs(1)),
        Decision::Nothing
    );
}

/// A forced upload that is actually sent clears the flag and restarts the gap.
#[test]
fn mark_uploaded_consumes_force_and_restarts_the_interval() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let view = idle();
    policy.mark_uploaded(&view, t0);
    policy.force();
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(5)),
        Decision::Wait
    );
    let sent = t0 + interval;
    assert_eq!(policy.decide(&view, sent), Decision::Upload);
    policy.mark_uploaded(&view, sent);
    assert_eq!(
        policy.decide(&view, sent + Duration::from_secs(1)),
        Decision::Nothing
    );
    let changed = loaded(80);
    assert_eq!(
        policy.decide(&changed, sent + interval - Duration::from_nanos(1)),
        Decision::Wait
    );
    assert_eq!(policy.decide(&changed, sent + interval), Decision::Upload);
}

/// Reopen forgets the view and keeps the success time, so the next frame waits out the gap.
#[test]
fn reset_on_open_keeps_the_success_time() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let view = idle();
    policy.mark_uploaded(&view, t0);
    let t1 = t0 + Duration::from_secs(5);
    assert_eq!(policy.decide(&view, t1), Decision::Nothing);
    policy.reset_on_open();
    assert_eq!(policy.decide(&view, t1), Decision::Wait);
    assert_eq!(
        policy.decide(&view, t0 + interval - Duration::from_nanos(1)),
        Decision::Wait
    );
    let sent = t0 + interval;
    assert_eq!(policy.decide(&view, sent), Decision::Upload);
    policy.mark_uploaded(&view, sent);
    assert_eq!(
        policy.decide(&loaded(20), sent + Duration::from_secs(10)),
        Decision::Wait
    );
}

/// A failed attempt still occupies the min_interval slot (Open 0f).
#[test]
fn mark_attempted_gates_the_next_upload_on_the_interval() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let view = loaded(40);
    assert_eq!(policy.decide(&view, t0), Decision::Upload);
    policy.mark_attempted(t0);
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Wait,
        "a failed attempt is not retried until min_interval"
    );
    assert_eq!(
        policy.decide(&view, t0 + interval - Duration::from_nanos(1)),
        Decision::Wait
    );
    assert_eq!(policy.decide(&view, t0 + interval), Decision::Upload);
}

/// The gate is max(last_success, last_attempt). A later attempt delays the next send.
#[test]
fn decide_uses_the_later_of_success_and_attempt() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let first = idle();
    let later = loaded(20);
    policy.mark_uploaded(&first, t0);
    let t_attempt = t0 + Duration::from_secs(10);
    policy.reset_on_open();
    assert_eq!(policy.decide(&later, t_attempt), Decision::Wait);
    let due_from_success = t0 + interval;
    assert_eq!(policy.decide(&later, due_from_success), Decision::Upload);
    policy.mark_attempted(due_from_success);
    assert_eq!(
        policy.decide(&later, due_from_success + Duration::from_secs(1)),
        Decision::Wait
    );
    assert_eq!(
        policy.decide(&later, due_from_success + interval),
        Decision::Upload
    );
}

/// A decided upload that the sink did not accept is still due, and the old view stays put.
#[test]
fn an_unmarked_upload_does_not_stick_and_stays_due() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    let first = idle();
    let changed = loaded(5);
    policy.mark_uploaded(&first, t0);
    let due = t0 + interval;
    assert_eq!(policy.decide(&changed, due), Decision::Upload);
    assert_eq!(policy.decide(&first, due), Decision::Nothing);
    assert_eq!(
        policy.decide(&changed, due + Duration::from_secs(2)),
        Decision::Upload
    );
    policy.mark_uploaded(&changed, due + Duration::from_secs(2));
    assert_eq!(
        policy.decide(&changed, due + Duration::from_secs(3)),
        Decision::Nothing
    );
}

/// `force` does not pull a waiting view forward of the last success.
#[test]
fn force_does_not_skip_the_interval_for_a_waiting_view() {
    let interval = Duration::from_secs(60);
    let mut policy = Policy::new(interval);
    let t0 = origin();
    policy.mark_uploaded(&idle(), t0);
    let waiting = loaded(15);
    let t1 = t0 + Duration::from_secs(4);
    assert_eq!(policy.decide(&waiting, t1), Decision::Wait);
    policy.force();
    assert_eq!(policy.decide(&waiting, t1), Decision::Wait);
    assert_eq!(policy.decide(&waiting, t0 + interval), Decision::Upload);
    policy.mark_uploaded(&waiting, t0 + interval);
    assert_eq!(policy.decide(&idle(), t0 + interval), Decision::Wait);
}

/// Equality is the whole [`View`]. One field moving is enough to leave `Nothing`.
#[test]
fn one_changed_field_counts_as_a_different_view() {
    let mut policy = Policy::new(Duration::from_secs(60));
    let t0 = origin();
    let mut view = idle();
    policy.mark_uploaded(&view, t0);
    view.cpu_c = Some(43);
    assert_eq!(
        policy.decide(&view, t0 + Duration::from_secs(1)),
        Decision::Wait
    );
}

/// `mark_uploaded` stores a snapshot. Later edits to the caller's view do not rewrite it.
#[test]
fn mark_uploaded_keeps_its_own_copy_of_the_view() {
    let mut policy = Policy::new(Duration::from_secs(60));
    let t0 = origin();
    let mut view = idle();
    policy.mark_uploaded(&view, t0);
    view.ring_pct = Some(80);
    view.ai = Ai::Loaded;
    let t1 = t0 + Duration::from_secs(1);
    assert_eq!(policy.decide(&idle(), t1), Decision::Nothing);
    assert_eq!(policy.decide(&view, t1), Decision::Wait);
}

/// A caller-supplied timestamp that moves backwards does not panic and does not upload.
#[test]
fn an_earlier_timestamp_does_not_skip_the_interval() {
    let mut policy = Policy::new(Duration::from_secs(60));
    let t0 = origin();
    policy.mark_uploaded(&idle(), t0);
    assert_eq!(
        policy.decide(&loaded(9), t0 - Duration::from_secs(1)),
        Decision::Wait
    );
}

/// After the opening frame, an unchanged view uploads nothing across an hour of 2 s ticks.
#[test]
fn idle_view_uploads_nothing_over_one_hour_of_ticks() {
    let tick = Duration::from_secs(2);
    let mut policy = Policy::new(Duration::from_secs(60));
    let start = origin();
    let view = idle();
    assert_eq!(policy.decide(&view, start), Decision::Upload);
    policy.mark_uploaded(&view, start);

    let end = start + Duration::from_secs(3600);
    let mut t = start + tick;
    let mut ticks = 0u32;
    while t <= end {
        assert_eq!(policy.decide(&view, t), Decision::Nothing);
        ticks += 1;
        t += tick;
    }
    assert_eq!(ticks, 1800);
}

/// A view that changes every tick uploads at most once per interval, and that upload is the latest view.
#[test]
fn flapping_view_uploads_at_most_once_per_interval() {
    let tick = Duration::from_secs(2);
    let start = origin();

    let two_state = record_uploads(
        Duration::from_secs(60),
        tick,
        start,
        start + Duration::from_secs(3600),
        |i, _| {
            if i % 2 == 0 { idle() } else { loaded(90) }
        },
    );
    assert_gaps_at_least(&two_state, Duration::from_secs(60));

    let interval = Duration::from_secs(60);
    let end = start + Duration::from_secs(3600);
    let changing = record_uploads(interval, tick, start, end, |i, _| distinct(i));
    assert_eq!(
        changing
            .iter()
            .map(|(at, _)| at.saturating_duration_since(start))
            .collect::<Vec<_>>(),
        (0..=60)
            .map(|step| Duration::from_secs(step * 60))
            .collect::<Vec<_>>()
    );
    for (at, view) in &changing {
        let index = at.saturating_duration_since(start).as_secs() / tick.as_secs();
        assert_eq!(view.models.as_slice(), &[index.to_string()]);
    }
    let hour = Duration::from_secs(3600);
    let in_first_hour = changing
        .iter()
        .filter(|(at, _)| at.saturating_duration_since(start) < hour)
        .count();
    assert!(in_first_hour <= 60, "{in_first_hour}");

    let floor = Duration::from_secs(10);
    let short_end = start + Duration::from_secs(120);
    let at_floor = record_uploads(floor, tick, start, short_end, |i, _| distinct(i));
    assert_eq!(
        at_floor
            .iter()
            .map(|(at, _)| at.saturating_duration_since(start))
            .collect::<Vec<_>>(),
        (0..=12)
            .map(|step| Duration::from_secs(step * 10))
            .collect::<Vec<_>>()
    );
}

/// A change that arrives between ticks is on screen by `min_interval + tick`.
#[test]
fn a_state_change_shows_within_min_interval_plus_one_tick() {
    let cases = [
        ("just after an upload", 2, 60, Duration::from_millis(1)),
        (
            "just before the interval, between ticks",
            2,
            60,
            Duration::from_millis(59_000),
        ),
        (
            "just after the interval, between ticks",
            2,
            60,
            Duration::from_millis(61_000),
        ),
        (
            "tick does not divide the interval",
            3,
            10,
            Duration::from_millis(1),
        ),
        (
            "change falls just after a tick and before the boundary",
            3,
            10,
            Duration::from_millis(9_001),
        ),
    ];
    for (name, tick_s, interval_s, change_after) in cases {
        show_within(name, tick_s, interval_s, change_after);
    }
}

/// The 10 s floor stays in [`Config::validate`]. Policy does not grow a second copy.
#[test]
fn config_validate_rejects_min_interval_below_the_floor() {
    let mut cfg = Config::default();
    assert_eq!(cfg.validate(), Ok(()));
    cfg.upload.min_interval_s = 9;
    assert_eq!(
        cfg.validate(),
        Err(InvalidConfig::MinInterval { min_interval_s: 9 })
    );
    cfg.upload.min_interval_s = 10;
    assert_eq!(cfg.validate(), Ok(()));
}

/// `force` on every 2 s tick for an hour still uploads at most once a minute.
#[test]
fn force_every_tick_for_one_hour_stays_within_the_upload_cap() {
    let uploads = pressured_hour(|policy| policy.force());
    assert_hour_cap("force", &uploads);
}

/// Reopen on every 2 s tick for an hour still uploads at most once a minute.
#[test]
fn reset_every_tick_for_one_hour_stays_within_the_upload_cap() {
    let uploads = pressured_hour(|policy| policy.reset_on_open());
    assert_hour_cap("reset", &uploads);
}

fn pressured_hour(mut each_tick: impl FnMut(&mut Policy)) -> Vec<Duration> {
    let interval = Duration::from_secs(60);
    let tick = Duration::from_secs(2);
    let hour = Duration::from_secs(3600);
    let mut policy = Policy::new(interval);
    let start = origin();
    let view = idle();
    let mut uploads = Vec::new();
    let mut t = start;
    loop {
        each_tick(&mut policy);
        if policy.decide(&view, t) == Decision::Upload {
            policy.mark_uploaded(&view, t);
            uploads.push(t.saturating_duration_since(start));
        }
        if t >= start + hour {
            break;
        }
        t += tick;
    }
    uploads
}

fn assert_hour_cap(label: &str, uploads: &[Duration]) {
    let interval = Duration::from_secs(60);
    println!("{label} uploads={}", uploads.len());
    assert!(
        uploads.len() <= 61,
        "{label}: {} uploads exceeds 61 (first + 60)",
        uploads.len()
    );
    assert_eq!(
        uploads.first().copied(),
        Some(Duration::ZERO),
        "{label} first frame"
    );
    for pair in uploads.windows(2) {
        let gap = pair[1].saturating_sub(pair[0]);
        assert!(gap >= interval, "{label} gap {gap:?} is under {interval:?}");
    }
    assert_eq!(uploads.len(), 61, "{label} first + one per minute");
}

/// Production `policy.rs` takes `now` from the caller. It does not read a clock.
#[test]
fn policy_rs_does_not_read_a_clock() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/policy.rs");
    let text = fs::read_to_string(&path).expect("read src/policy.rs");
    let production = text
        .find("#[cfg(test)]")
        .map_or(text.as_str(), |at| &text[..at]);
    assert!(production.contains("fn decide"));
    for needle in ["Instant::now", "SystemTime::now", ".elapsed("] {
        assert!(
            !production.contains(needle),
            "src/policy.rs production code contains {needle}"
        );
    }
}

fn distinct(index: u64) -> View {
    let mut view = loaded(u8::try_from(index % 101).expect("ring fits in u8"));
    view.model_count = 1;
    view.models = vec![index.to_string()];
    view
}

fn record_uploads(
    interval: Duration,
    tick: Duration,
    start: Instant,
    end: Instant,
    mut view_at: impl FnMut(u64, Instant) -> View,
) -> Vec<(Instant, View)> {
    let mut policy = Policy::new(interval);
    let mut uploads = Vec::new();
    let mut t = start;
    let mut index = 0u64;
    loop {
        let view = view_at(index, t);
        if policy.decide(&view, t) == Decision::Upload {
            policy.mark_uploaded(&view, t);
            uploads.push((t, view));
        }
        if t == end {
            break;
        }
        let next = t + tick;
        if next > end {
            break;
        }
        t = next;
        index += 1;
    }
    uploads
}

fn assert_gaps_at_least(uploads: &[(Instant, View)], interval: Duration) {
    assert!(!uploads.is_empty());
    for pair in uploads.windows(2) {
        let gap = pair[1].0.saturating_duration_since(pair[0].0);
        assert!(
            gap >= interval,
            "uploads {gap:?} apart, interval {interval:?}"
        );
    }
    for (at, _) in uploads {
        let count = uploads
            .iter()
            .filter(|(when, _)| *when >= *at && when.saturating_duration_since(*at) < interval)
            .count();
        assert_eq!(count, 1, "half-open window at {at:?}");
    }
}

fn show_within(name: &str, tick_s: u64, interval_s: u64, change_after: Duration) {
    let tick = Duration::from_secs(tick_s);
    let interval = Duration::from_secs(interval_s);
    let mut policy = Policy::new(interval);
    let start = origin();
    let before = idle();
    let after = loaded(70);
    assert_eq!(policy.decide(&before, start), Decision::Upload, "{name}");
    policy.mark_uploaded(&before, start);
    let changed_at = start + change_after;
    let bound = interval + tick;
    let mut t = start;
    for _ in 0..10_000 {
        t += tick;
        if t < changed_at {
            assert_eq!(policy.decide(&before, t), Decision::Nothing, "{name}");
            continue;
        }
        let decision = policy.decide(&after, t);
        let delay = t.saturating_duration_since(changed_at);
        assert!(
            delay <= bound,
            "{name}: {decision:?} after {delay:?}, bound {bound:?}"
        );
        if decision == Decision::Upload {
            assert!(
                t.saturating_duration_since(start) >= interval,
                "{name}: uploaded before the interval"
            );
            policy.mark_uploaded(&after, t);
            assert_eq!(policy.decide(&after, t), Decision::Nothing, "{name}");
            assert_eq!(policy.decide(&before, t), Decision::Wait, "{name}");
            return;
        }
    }
    panic!("{name}: never uploaded");
}
