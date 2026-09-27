//! `show-image`: one-frame upload on the `run` path, then a QueryBucket table.
//!
//! Fake sysfs and fake ports only. The binary is never spawned.

#[path = "device_fixture.rs"]
mod fixture;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kraken_lcd::config::DisplayCfg;
use kraken_lcd::device::proto::{self, Cmd, SlotId};
use kraken_lcd::device::{BucketRow, KrakenLcd, SinkError, format_bucket_table};
use kraken_lcd::present::View;
use kraken_lcd::render::{self, Assets, Frame, NO_DATA, TEXT};

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

fn show_image(
    tree: &fixture::Tree,
    hid: Arc<fixture::SharedHid>,
    bulk: Arc<fixture::SharedBulk>,
    bucket: u8,
    frame: &Frame,
) -> Result<Vec<BucketRow>, SinkError> {
    show_image_sleeping(tree, hid, bulk, bucket, frame, |_| {})
}

fn show_image_sleeping(
    tree: &fixture::Tree,
    hid: Arc<fixture::SharedHid>,
    bulk: Arc<fixture::SharedBulk>,
    bucket: u8,
    frame: &Frame,
    sleep: impl FnMut(Duration),
) -> Result<Vec<BucketRow>, SinkError> {
    let sys = tree.sys.clone();
    KrakenLcd::show_image(
        &tree.request(0, false),
        slot(bucket),
        frame,
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
        sleep,
    )
}

fn script_open_show_and_table(hid: &fixture::SharedHid, bucket: u8) {
    fixture::script_empty_open(hid, 0);
    fixture::script_show_slot(hid, bucket);
    for id in 0..proto::BUCKET_COUNT {
        if id == bucket {
            hid.push(fixture::occupied(
                u16::from(id) * proto::SLOT_UNITS,
                proto::SLOT_UNITS,
            ));
        } else {
            hid.push(fixture::empty_bucket());
        }
    }
}

fn fixture_view() -> std::path::PathBuf {
    fixture_named("working-hard.json")
}

fn fixture_named(name: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("fixtures/views");
    path.push(name);
    path
}

#[test]
fn show_image_uploads_once_shows_once_and_prints_the_table() {
    let tree = fixture::Tree::new("show-happy");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let rows = show_image(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, &Frame::new())
        .expect("one-frame upload");
    let sent = hid.sent();
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::WriteStart(slot(0)))[..3]),
        1,
        "exactly one upload"
    );
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::ShowSlot(slot(0)))[..4]),
        1,
        "exactly one show"
    );
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::ShowLiquid)[..4]),
        0,
        "show-image must not send ShowLiquid"
    );
    let chunks = bulk.chunks();
    assert_eq!(chunks.len(), 801, "header plus 800 pixel transfers");
    assert_eq!(chunks[0], proto::BULK_HEADER);
    assert_eq!(rows.len(), 16);
    assert!(!rows[0].table.empty);
    assert_eq!(rows[0].table.start_kib, 0);
    assert_eq!(rows[0].table.size_kib, proto::SLOT_UNITS);
    let text = format_bucket_table(&rows);
    assert!(
        text.contains("bucket=0 empty=no start_kib=0 size_kib=401\n"),
        "{text}"
    );
    assert!(!tree.latch().exists(), "a clean upload must not latch");
}

#[test]
fn show_image_never_sends_show_liquid_on_the_happy_path() {
    let tree = fixture::Tree::new("show-no-liquid");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 3);
    show_image(&tree, Arc::clone(&hid), Arc::clone(&bulk), 3, &Frame::new())
        .expect("upload to slot 3");
    let sent = hid.sent();
    assert!(
        sent.iter()
            .all(|report| report[0..3] != fixture::encoded(&Cmd::ShowLiquid)[0..3]),
        "ShowLiquid was sent: {sent:?}"
    );
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::ShowSlot(slot(3)))[..4]),
        1
    );
    assert_eq!(
        fixture::count_prefix(&sent, &fixture::encoded(&Cmd::WriteStart(slot(3)))[..3]),
        1
    );
}

#[test]
fn show_image_skips_the_open_when_the_latch_is_present() {
    let tree = fixture::Tree::new("show-latched");
    fixture::install_kraken(&tree);
    std::fs::write(tree.latch(), b"already halted\n").expect("latch");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let err = KrakenLcd::show_image(
        &tree.request(0, false),
        slot(0),
        &Frame::new(),
        {
            let bulk = Arc::clone(&bulk);
            move |_, _| Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
        |_| {},
    )
    .expect_err("latch blocks device I/O");
    assert!(matches!(err, SinkError::Halted), "{err}");
    assert_eq!(
        hid.alive.load(Ordering::Relaxed),
        0,
        "a latched state dir must not open hidraw"
    );
    assert_eq!(
        bulk.alive.load(Ordering::Relaxed),
        0,
        "a latched state dir must not open bulk"
    );
    assert!(hid.sent().is_empty(), "hidraw was not written");
    assert!(bulk.chunks().is_empty(), "bulk was not written");
    assert_eq!(
        std::fs::read(tree.latch()).expect("latch"),
        b"already halted\n",
        "an existing latch is left in place"
    );
}

#[test]
fn show_image_halts_mid_sequence_and_stops() {
    let tree = fixture::Tree::new("show-halt");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    fixture::script_empty_open(&hid, 0);
    fixture::script_show_slot(&hid, 0);
    hid.push(fixture::empty_bucket());
    let shown = Arc::new(AtomicBool::new(false));
    let sys = tree.sys.clone();
    let shown_hook = Arc::clone(&shown);
    hid.on_send(move |report| {
        if report[0] == 0x38 && report[1] == 0x01 && report[2] == 0x04 {
            shown_hook.store(true, Ordering::Relaxed);
        }
        if shown_hook.load(Ordering::Relaxed) && report[0] == 0x30 && report[1] == 0x04 {
            std::fs::write(sys.join("class/hwmon/hwmon4/pwm1_enable"), "1").expect("flip enable");
        }
    });
    let err = show_image(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, &Frame::new())
        .expect_err("cooling change");
    assert!(matches!(err, SinkError::Halted), "{err}");
    let sent = hid.sent();
    let show_at = sent
        .iter()
        .position(|report| report[0..3] == [0x38, 0x01, 0x04])
        .expect("ShowSlot was sent before the trip");
    let table_queries = sent[show_at + 1..]
        .iter()
        .filter(|report| report[0..2] == [0x30, 0x04])
        .count();
    assert_eq!(
        table_queries, 1,
        "no further QueryBucket after the guard trips: {sent:?}"
    );
    let body = std::fs::read_to_string(tree.latch()).expect("latch written");
    assert!(body.contains("reason=pwm1_enable"), "{body}");
}

#[test]
fn show_image_refuses_when_interface0_is_usbfs() {
    let tree = fixture::Tree::new("show-usbfs");
    fixture::install_kraken(&tree);
    fixture::claim_usbfs(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let err = KrakenLcd::show_image(
        &tree.request(0, false),
        slot(0),
        &Frame::new(),
        {
            let bulk = Arc::clone(&bulk);
            move |_, _| Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
        |_| {},
    )
    .expect_err("usbfs means the service is active");
    assert!(matches!(err, SinkError::DeviceUnavailable), "{err}");
    assert_eq!(
        hid.alive.load(Ordering::Relaxed),
        0,
        "a bound interface 0 is not opened"
    );
    assert!(hid.sent().is_empty());
}

#[test]
fn show_image_refuses_the_bootloader_before_opening() {
    let tree = fixture::Tree::new("show-bootloader");
    fixture::install_kraken(&tree);
    fixture::add_bootloader(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let err = show_image(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, &Frame::new())
        .expect_err("bootloader");
    assert!(matches!(err, SinkError::DeviceInBootloader), "{err}");
    assert!(hid.sent().is_empty());
}

#[test]
fn show_image_uploads_a_rendered_fixture_view() {
    let json = std::fs::read_to_string(fixture_view()).expect("view fixture");
    let view: View = serde_json::from_str(&json).expect("view json");
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view, &DisplayCfg::default(), &mut assets);
    assert_eq!((frame.0.width(), frame.0.height()), (320, 320));
    let tree = fixture::Tree::new("show-render");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    show_image(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, &frame).expect("rendered upload");
    assert_eq!(
        fixture::count_prefix(&hid.sent(), &fixture::encoded(&Cmd::ShowSlot(slot(0)))[..4]),
        1
    );
    assert_eq!(bulk.chunks().len(), 801);
}

#[test]
fn cli_source_dispatches_show_image_without_show_liquid() {
    let src = include_str!("../src/main.rs");
    assert!(
        src.contains("\"show-image\""),
        "the CLI command is show-image"
    );
    assert!(
        src.contains("show_image_resolved"),
        "main dispatches to the one-frame device entry"
    );
    assert!(
        src.contains("kraken_lcd::render::render"),
        "show-image uses the same render() as the golden tests"
    );
    assert!(
        src.contains("fixtures/views/test-card.json"),
        "usage recommends the power-cycle test card"
    );
    assert!(
        src.contains("rotation is fixed at 0"),
        "usage states that rotation is fixed at 0"
    );
}

#[test]
fn show_image_follow_up_runs_once_on_the_happy_path() {
    let tree = fixture::Tree::new("show-follow-ok");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let sleeps = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&sleeps);
    show_image_sleeping(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        0,
        &Frame::new(),
        move |wait| recorded.lock().expect("sleeps").push(wait),
    )
    .expect("follow-up on a stable cooler");
    let sleeps = sleeps.lock().expect("sleeps").clone();
    assert_eq!(
        sleeps,
        [Duration::from_secs(2)],
        "exactly one 2 s follow-up wait"
    );
    assert!(!tree.latch().exists(), "a clean follow-up must not latch");
}

#[test]
fn show_image_follow_up_deviation_latches() {
    let tree = fixture::Tree::new("show-follow-halt");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let sys = tree.sys.clone();
    let err = show_image_sleeping(
        &tree,
        Arc::clone(&hid),
        Arc::clone(&bulk),
        0,
        &Frame::new(),
        move |_| {
            std::fs::write(sys.join("class/hwmon/hwmon4/pwm1_enable"), "1").expect("flip enable");
        },
    )
    .expect_err("follow-up cooling change");
    assert!(matches!(err, SinkError::Halted), "{err}");
    let body = std::fs::read_to_string(tree.latch()).expect("latch written");
    assert!(body.contains("reason=pwm1_enable"), "{body}");
}

#[test]
fn show_image_refuses_when_z53_is_missing() {
    let tree = fixture::Tree::new("show-no-z53");
    fixture::install_kraken(&tree);
    std::fs::remove_dir_all(tree.hwmon()).expect("remove z53");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let err = KrakenLcd::show_image(
        &tree.request(0, false),
        slot(0),
        &Frame::new(),
        {
            let bulk = Arc::clone(&bulk);
            move |_, _| Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
        |_| {},
    )
    .expect_err("no z53");
    assert!(matches!(err, SinkError::DeviceUnavailable), "{err}");
    assert_eq!(hid.alive.load(Ordering::Relaxed), 0, "must not open hidraw");
    assert_eq!(bulk.alive.load(Ordering::Relaxed), 0, "must not open bulk");
    assert!(
        !tree.latch().exists(),
        "a missing z53 must not write the halt latch"
    );
}

#[test]
fn show_image_refuses_when_state_dir_is_not_writable() {
    let tree = fixture::Tree::new("show-ro-state");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_open_show_and_table(&hid, 0);
    let mut perms = std::fs::metadata(&tree.state)
        .expect("state meta")
        .permissions();
    perms.set_mode(0o555);
    std::fs::set_permissions(&tree.state, perms).expect("ro state");
    let result = KrakenLcd::show_image(
        &tree.request(0, false),
        slot(0),
        &Frame::new(),
        {
            let bulk = Arc::clone(&bulk);
            move |_, _| Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
        |_| {},
    );
    let mut restore = std::fs::metadata(&tree.state)
        .expect("state meta")
        .permissions();
    restore.set_mode(0o755);
    std::fs::set_permissions(&tree.state, restore).expect("restore state");
    let err = result.expect_err("read-only state dir");
    assert!(matches!(err, SinkError::DeviceUnavailable), "{err}");
    assert_eq!(hid.alive.load(Ordering::Relaxed), 0, "must not open hidraw");
    assert_eq!(bulk.alive.load(Ordering::Relaxed), 0, "must not open bulk");
    assert!(
        !tree.latch().exists(),
        "an unwritable state dir must not write the halt latch"
    );
}

#[test]
fn show_image_test_card_temperature_zone_is_the_no_data_dash() {
    let json = std::fs::read_to_string(fixture_named("test-card.json")).expect("test card");
    let view: View = serde_json::from_str(&json).expect("view json");
    assert_eq!(view.models, ["TEST IMAGE"]);
    assert_eq!(view.coolant_c, None);
    assert_eq!(view.cpu_c, None);
    assert_eq!(view.gpu_c, None);
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view, &DisplayCfg::default(), &mut assets);
    let no_data = (NO_DATA.r, NO_DATA.g, NO_DATA.b);
    let text = (TEXT.r, TEXT.g, TEXT.b);
    for (name, x0, y0, x1, y1) in [
        ("coolant", 60, 118, 120, 156),
        ("cpu temp", 130, 118, 190, 156),
        ("gpu temp", 200, 118, 260, 156),
    ] {
        let dashes = count_near(&frame, x0, y0, x1, y1, no_data, 16);
        assert!(
            dashes >= 30,
            "{name} should draw \"—\", saw {dashes} pixels"
        );
        let digits = count_near(&frame, x0, y0, x1, y1, text, 16);
        assert_eq!(
            digits, 0,
            "{name} should not draw digits, saw {digits} pixels"
        );
    }
}

fn count_near(
    frame: &Frame,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    expect: (u8, u8, u8),
    tol: u8,
) -> usize {
    let mut count = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let pixel = frame.0.pixel(x, y).expect("pixel on the 320 frame");
            let got = (pixel.red(), pixel.green(), pixel.blue());
            if got.0.abs_diff(expect.0) <= tol
                && got.1.abs_diff(expect.1) <= tol
                && got.2.abs_diff(expect.2) <= tol
            {
                count += 1;
            }
        }
    }
    count
}

/// T29 nit: the three per-command root tests were one test. Instead, every
/// device entry point must start with the `reject_root` guard, before any
/// sysfs or device step.
#[test]
fn every_device_entry_point_rejects_root_first() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/device/mod.rs");
    let src = std::fs::read_to_string(&path).expect("device/mod.rs");
    for name in ["query_buckets", "show_image", "bench_upload"] {
        let head = format!("pub fn {name}(");
        let at = src
            .find(&head)
            .unwrap_or_else(|| panic!("{name} not found"));
        let body = &src[at..];
        let open = body
            .find(") -> ")
            .and_then(|sig| body[sig..].find('{').map(|b| sig + b));
        let open = open.unwrap_or_else(|| panic!("{name} has no body"));
        let first = body[open + 1..].trim_start();
        assert!(
            first.starts_with("if reject_root(rustix::process::geteuid().is_root()) {"),
            "{name} must reject root before anything else, starts with: {}",
            first.lines().next().unwrap_or("")
        );
    }
}
