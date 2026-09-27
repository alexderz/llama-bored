//! `bench-upload`: paced ping-pong of the `show-image` one-frame path.
//!
//! Fake sysfs, fake ports, and a fake clock only. The binary is never spawned.

#[path = "device_fixture.rs"]
mod fixture;

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kraken_lcd::device::proto::{Cmd, SlotId};
use kraken_lcd::device::{BenchClock, BenchSpec, KrakenLcd, SinkError, bench_slack_ms};
use kraken_lcd::render::Frame;

fn slot(id: u8) -> SlotId {
    SlotId::try_new(id).expect("rotation slot")
}

fn claim_usbfs_after_pre_open(sys: &Path) {
    let iface = sys.join("bus/usb/devices/3-5/3-5:1.0");
    std::fs::create_dir_all(&iface).expect("iface");
    let link = iface.join("driver");
    if link.symlink_metadata().is_err() {
        std::os::unix::fs::symlink("../../../../bus/usb/drivers/usbfs", link).expect("usbfs");
    }
}

struct FakeClock {
    ns: Arc<Mutex<u64>>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            ns: Arc::new(Mutex::new(0)),
        }
    }

    fn share(&self) -> Arc<Mutex<u64>> {
        Arc::clone(&self.ns)
    }
}

impl BenchClock for FakeClock {
    fn now_ns(&self) -> u64 {
        *self.ns.lock().expect("clock")
    }

    fn sleep(&self, d: Duration) {
        let add = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        let mut ns = self.ns.lock().expect("clock");
        *ns = ns.saturating_add(add);
    }
}

fn spec(count: u32, fps: f64) -> BenchSpec {
    BenchSpec::try_new(count, fps, 2).expect("bench spec")
}

fn bench(
    tree: &fixture::Tree,
    hid: Arc<fixture::SharedHid>,
    bulk: Arc<fixture::SharedBulk>,
    frame: &Frame,
    spec: BenchSpec,
    clock: &FakeClock,
    out: &mut impl Write,
) -> Result<kraken_lcd::device::BenchSummary, SinkError> {
    let sys = tree.sys.clone();
    KrakenLcd::bench_upload(
        &tree.request(0, false),
        frame,
        spec,
        move |bus, dev| {
            assert_eq!(bus, fixture::BUSNUM, "bulk open bus");
            assert_eq!(u32::from(dev), fixture::DEVNUM, "bulk open devnum");
            claim_usbfs_after_pre_open(&sys);
            Ok(fixture::FakeBulk::attach(bulk))
        },
        move |path| {
            assert_eq!(path, Path::new("/dev/hidraw0"));
            Ok(fixture::FakeHid::attach(hid))
        },
        clock,
        out,
    )
}

fn script_open_and_shows(hid: &fixture::SharedHid, count: u32) {
    fixture::script_empty_open(hid, 0);
    for seq in 0..count {
        let id = u8::try_from(seq % 2).expect("slot");
        fixture::script_show_slot(hid, id);
    }
}

fn show_slot_prefix(id: u8) -> [u8; 4] {
    fixture::encoded(&Cmd::ShowSlot(slot(id)))[0..4]
        .try_into()
        .expect("prefix")
}

fn write_start_prefix(id: u8) -> [u8; 3] {
    fixture::encoded(&Cmd::WriteStart(slot(id)))[0..3]
        .try_into()
        .expect("prefix")
}

#[test]
fn pacing_slack_is_period_minus_total() {
    assert_eq!(bench_slack_ms(100_000_000, 40_000_000), 60.0);
    assert_eq!(bench_slack_ms(100_000_000, 100_000_000), 0.0);
}

#[test]
fn pacing_overrun_is_negative_slack() {
    assert_eq!(bench_slack_ms(100_000_000, 150_000_000), -50.0);
}

#[test]
fn bench_spec_rejects_out_of_range_count_fps_and_slots() {
    assert!(BenchSpec::try_new(0, 10.0, 2).is_err());
    assert!(BenchSpec::try_new(36_001, 10.0, 2).is_err());
    assert!(BenchSpec::try_new(1, 0.0, 2).is_err());
    assert!(BenchSpec::try_new(1, -1.0, 2).is_err());
    assert!(BenchSpec::try_new(1, 30.1, 2).is_err());
    assert!(BenchSpec::try_new(1, f64::NAN, 2).is_err());
    assert!(BenchSpec::try_new(1, 10.0, 1).is_err());
    assert!(BenchSpec::try_new(1, 10.0, 8).is_err());
    assert!(BenchSpec::try_new(36_000, 30.0, 2).is_ok());
    assert!(BenchSpec::try_new(1, 0.2, 2).is_ok());
}

#[test]
fn bench_upload_ping_pongs_slots_zero_and_one() {
    let tree = fixture::Tree::new("bench-pong");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 4);
    let clock = FakeClock::new();
    let mut out = Vec::new();
    bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(4, 10.0),
        &clock,
        &mut out,
    )
    .expect("four paced uploads");
    let sent = hid.sent();
    assert_eq!(
        fixture::count_prefix(&sent, &write_start_prefix(0)),
        2,
        "slot 0 twice"
    );
    assert_eq!(
        fixture::count_prefix(&sent, &write_start_prefix(1)),
        2,
        "slot 1 twice"
    );
    assert_eq!(fixture::count_prefix(&sent, &show_slot_prefix(0)), 2);
    assert_eq!(fixture::count_prefix(&sent, &show_slot_prefix(1)), 2);
    let starts: Vec<u8> = sent
        .iter()
        .filter(|report| report[0..2] == fixture::encoded(&Cmd::WriteStart(slot(0)))[0..2])
        .map(|report| report[2])
        .collect();
    assert_eq!(starts, [0, 1, 0, 1], "ping-pong order: {starts:?}");
}

#[test]
fn bench_upload_never_sends_show_liquid_on_the_happy_path() {
    let tree = fixture::Tree::new("bench-no-liquid");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 3);
    let clock = FakeClock::new();
    let mut out = Vec::new();
    bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(3, 10.0),
        &clock,
        &mut out,
    )
    .expect("happy path");
    let sent = hid.sent();
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::ShowLiquid)[..4]),
        0,
        "bench-upload must not send ShowLiquid"
    );
    assert!(
        sent.iter()
            .all(|report| report[0..3] != fixture::encoded(&Cmd::ShowLiquid)[0..3]),
        "ShowLiquid was sent: {sent:?}"
    );
}

#[test]
fn bench_upload_prints_per_frame_lines_and_a_summary() {
    let tree = fixture::Tree::new("bench-print");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 2);
    let clock = FakeClock::new();
    let ns = clock.share();
    hid.on_send(move |report| {
        let add = if report[0..2] == [0x38, 0x01] {
            2_000_000
        } else {
            5_000_000
        };
        let mut now = ns.lock().expect("clock");
        *now = now.saturating_add(add);
    });
    let mut out = Vec::new();
    let summary = bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(2, 10.0),
        &clock,
        &mut out,
    )
    .expect("printed run");
    let text = String::from_utf8(out).expect("utf8");
    assert!(
        text.contains("seq=0 ") && text.contains("seq=1 "),
        "one line per upload: {text}"
    );
    assert!(text.contains("upload_ms="), "{text}");
    assert!(text.contains("switch_ms="), "{text}");
    assert!(text.contains("total_ms="), "{text}");
    assert!(text.contains("slack_ms="), "{text}");
    assert!(text.contains("summary upload_ms min="), "{text}");
    assert!(text.contains("median="), "{text}");
    assert!(text.contains("p95="), "{text}");
    assert!(text.contains("max="), "{text}");
    assert!(text.contains("summary switch_ms"), "{text}");
    assert!(text.contains("summary total_ms"), "{text}");
    assert!(text.contains("summary slack_ms"), "{text}");
    assert!(text.contains("achieved_fps="), "{text}");
    assert_eq!(summary.samples.len(), 2);
    assert!(
        summary.samples[0].switch_ms > 0.0,
        "switch is timed separately: {:?}",
        summary.samples[0]
    );
    assert!(
        summary.samples[0].upload_ms > summary.samples[0].switch_ms,
        "upload includes prepare+write: {:?}",
        summary.samples[0]
    );
}

#[test]
fn bench_upload_sleeps_slack_and_skips_sleep_on_overrun() {
    let tree = fixture::Tree::new("bench-slack");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 3);
    let clock = FakeClock::new();
    let mut out = Vec::new();
    let summary = bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(3, 10.0),
        &clock,
        &mut out,
    )
    .expect("zero-cost frames");
    for sample in &summary.samples {
        assert_eq!(sample.total_ms, 0.0, "{sample:?}");
        assert_eq!(sample.slack_ms, 100.0, "{sample:?}");
    }
    // Two inter-frame waits of 100 ms plus the 2 s follow-up.
    assert_eq!(clock.now_ns(), 2_200_000_000);

    let tree = fixture::Tree::new("bench-overrun");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 2);
    let clock = FakeClock::new();
    let ns = clock.share();
    hid.on_send(move |_| {
        let mut now = ns.lock().expect("clock");
        *now = now.saturating_add(20_000_000);
    });
    let mut out = Vec::new();
    let summary = bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(2, 10.0),
        &clock,
        &mut out,
    )
    .expect("overrun frames");
    assert!(
        summary.samples.iter().all(|sample| sample.slack_ms < 0.0),
        "overrun slack is negative: {:?}",
        summary.samples
    );
    let after_frames = summary.samples.iter().map(|s| s.total_ms).sum::<f64>();
    assert!(
        after_frames > 100.0,
        "each 10 fps frame overran 100 ms: {after_frames}"
    );
}

#[test]
fn bench_upload_guard_trip_mid_run_latches_and_stops() {
    let tree = fixture::Tree::new("bench-halt");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 4);
    let shown = Arc::new(AtomicBool::new(false));
    let sys = tree.sys.clone();
    let shown_hook = Arc::clone(&shown);
    hid.on_send(move |report| {
        if report[0] == 0x38
            && report[1] == 0x01
            && report[2] == 0x04
            && shown_hook.swap(true, Ordering::Relaxed)
        {
            std::fs::write(sys.join("class/hwmon/hwmon4/pwm1_enable"), "1").expect("flip enable");
        }
    });
    let clock = FakeClock::new();
    let mut out = Vec::new();
    let err = bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(4, 10.0),
        &clock,
        &mut out,
    )
    .expect_err("cooling change");
    assert!(matches!(err, SinkError::Halted), "{err}");
    let sent = hid.sent();
    let shows = fixture::count_prefix(&sent, &[0x38, 0x01, 0x04]);
    assert_eq!(shows, 2, "stop after the second show trips: {sent:?}");
    let body = std::fs::read_to_string(tree.latch()).expect("latch written");
    assert!(body.contains("reason=pwm1_enable"), "{body}");
}

#[test]
fn bench_upload_ticks_during_the_wait_and_latches_on_pwm_change() {
    let tree = fixture::Tree::new("bench-tick");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 2);
    let clock = FakeClock::new();
    let sys = tree.sys.clone();
    let flipped = Arc::new(AtomicBool::new(false));
    // Flip pwm once fake time has passed one second, which happens in the
    // inter-frame slack wait at 1 fps (period 1 s).
    struct FlipClock {
        inner: FakeClock,
        sys: std::path::PathBuf,
        flipped: Arc<AtomicBool>,
    }
    impl BenchClock for FlipClock {
        fn now_ns(&self) -> u64 {
            self.inner.now_ns()
        }
        fn sleep(&self, d: Duration) {
            self.inner.sleep(d);
            if self.inner.now_ns() >= 1_000_000_000 && !self.flipped.swap(true, Ordering::Relaxed) {
                std::fs::write(self.sys.join("class/hwmon/hwmon4/pwm1_enable"), "1")
                    .expect("flip enable");
            }
        }
    }
    let clock = FlipClock {
        inner: clock,
        sys,
        flipped,
    };
    let sys = tree.sys.clone();
    let err = KrakenLcd::bench_upload(
        &tree.request(0, false),
        &Frame::new(),
        spec(2, 1.0),
        {
            let bulk = Arc::clone(&bulk);
            move |bus, dev| {
                assert_eq!(bus, fixture::BUSNUM);
                assert_eq!(u32::from(dev), fixture::DEVNUM);
                claim_usbfs_after_pre_open(&sys);
                Ok(fixture::FakeBulk::attach(bulk))
            }
        },
        {
            let hid = Arc::clone(&hid);
            move |path| {
                assert_eq!(path, Path::new("/dev/hidraw0"));
                Ok(fixture::FakeHid::attach(hid))
            }
        },
        &clock,
        &mut Vec::new(),
    )
    .expect_err("tick during slack must see the pwm change");
    assert!(matches!(err, SinkError::Halted), "{err}");
    let body = std::fs::read_to_string(tree.latch()).expect("latch written");
    assert!(body.contains("reason=pwm1_enable"), "{body}");
    let shows = fixture::count_prefix(&hid.sent(), &[0x38, 0x01]);
    assert_eq!(
        shows,
        1,
        "the second upload must not run after the tick latches: {:?}",
        hid.sent()
    );
}

#[test]
fn bench_upload_failure_exits_without_latching() {
    let tree = fixture::Tree::new("bench-fail");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 4);
    bulk.fail_at(1);
    let clock = FakeClock::new();
    let mut out = Vec::new();
    let err = bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(4, 10.0),
        &clock,
        &mut out,
    )
    .expect_err("upload failure");
    assert!(
        matches!(err, SinkError::TransferAborted | SinkError::UploadFailed(_)),
        "{err}"
    );
    assert!(
        !tree.latch().exists(),
        "an upload failure must not write the halt latch"
    );
    let sent = hid.sent();
    let starts = fixture::count_prefix(&sent, &fixture::encoded(&Cmd::WriteStart(slot(0)))[..2]);
    assert_eq!(
        starts, 1,
        "abort on the first WriteStart failure; no later frames: {sent:?}"
    );
    assert_eq!(
        fixture::count_prefix(&sent, &write_start_prefix(1)),
        0,
        "slot 1 must not be reached after the first failure"
    );
}

#[test]
fn bench_upload_skips_the_open_when_the_latch_is_present() {
    let tree = fixture::Tree::new("bench-latched");
    fixture::install_kraken(&tree);
    std::fs::write(tree.latch(), b"already halted\n").expect("latch");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 2);
    let clock = FakeClock::new();
    let err = KrakenLcd::bench_upload(
        &tree.request(0, false),
        &Frame::new(),
        spec(2, 10.0),
        {
            let bulk = Arc::clone(&bulk);
            move |_, _| Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
        &clock,
        &mut Vec::new(),
    )
    .expect_err("latch blocks device I/O");
    assert!(matches!(err, SinkError::Halted), "{err}");
    assert_eq!(hid.alive.load(Ordering::Relaxed), 0, "must not open hidraw");
    assert_eq!(bulk.alive.load(Ordering::Relaxed), 0, "must not open bulk");
    assert!(hid.sent().is_empty());
    assert_eq!(
        std::fs::read(tree.latch()).expect("latch"),
        b"already halted\n"
    );
}

#[test]
fn bench_upload_follow_up_runs_after_the_last_frame() {
    let tree = fixture::Tree::new("bench-follow");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_and_shows(&hid, 1);
    let clock = FakeClock::new();
    let mut out = Vec::new();
    bench(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        &Frame::new(),
        spec(1, 10.0),
        &clock,
        &mut out,
    )
    .expect("single frame");
    assert_eq!(
        clock.now_ns(),
        2_000_000_000,
        "exactly one 2 s follow-up after the last upload"
    );
    assert!(!tree.latch().exists());
}

#[test]
fn cli_source_dispatches_bench_upload_without_show_liquid() {
    let src = include_str!("../src/main.rs");
    assert!(
        src.contains("\"bench-upload\""),
        "the CLI command is bench-upload"
    );
    assert!(
        src.contains("bench_upload_resolved"),
        "main dispatches to the paced device entry"
    );
    assert!(
        src.contains("kraken_lcd::render::render"),
        "bench-upload uses the same render() as show-image"
    );
}
