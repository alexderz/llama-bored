//! `--query-buckets`: pre-open checks, hidraw only, `QueryBucket(0..=15)` inside the guard.
//!
//! No bulk claim, no delete, no `ShowLiquid`. A cooling change latches and
//! stops the remaining queries.

#[path = "device_fixture.rs"]
mod fixture;

use std::path::Path;
use std::sync::Arc;

use kraken_lcd::device::proto::{self, Cmd};
use kraken_lcd::device::{KrakenLcd, SinkError};

fn bucket(id: u8) -> proto::BucketId {
    proto::BucketId::try_new(id).expect("bucket")
}

fn push_tables(hid: &fixture::SharedHid) {
    for id in 0..proto::BUCKET_COUNT {
        if id == 3 {
            hid.push(fixture::occupied(u16::from(id) * 401, 401));
        } else {
            hid.push(fixture::empty_bucket());
        }
    }
}

fn query(
    tree: &fixture::Tree,
    hid: Arc<fixture::SharedHid>,
) -> Result<Vec<kraken_lcd::device::BucketRow>, SinkError> {
    KrakenLcd::query_buckets(&tree.request(0, false), move |path| {
        assert_eq!(path, Path::new("/dev/hidraw0"));
        Ok(fixture::FakeHid::attach(hid))
    })
}

#[test]
fn query_buckets_sends_only_query_bucket_and_prints_the_table() {
    let tree = fixture::Tree::new("query-only");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    push_tables(&hid);
    let rows = query(&tree, Arc::clone(&hid)).expect("sixteen bucket queries");
    let sent = hid.sent();
    assert_eq!(sent.len(), 16, "one report per bucket and nothing else");
    for (id, report) in sent.iter().enumerate() {
        let id = u8::try_from(id).expect("bucket id");
        assert_eq!(
            report.as_slice(),
            fixture::encoded(&Cmd::QueryBucket(bucket(id)))
        );
    }
    assert!(
        sent.iter()
            .all(|report| report[0] != 0x32 && report[0] != 0x36 && report[0] != 0x38),
        "query-buckets must not delete, set up, or show"
    );
    assert_eq!(rows.len(), 16);
    assert!(rows[0].table.empty);
    assert!(!rows[3].table.empty);
    assert_eq!(rows[3].table.start_kib, 1203);
    assert_eq!(rows[3].table.size_kib, 401);
    let text = kraken_lcd::device::format_bucket_table(&rows);
    assert_eq!(
        text.lines().count(),
        16,
        "one printed row per bucket: {text}"
    );
    assert!(
        text.contains("bucket=0 empty=yes start_kib=0 size_kib=0\n"),
        "{text}"
    );
    assert!(
        text.contains("bucket=3 empty=no start_kib=1203 size_kib=401\n"),
        "{text}"
    );
    assert!(
        !tree.latch().exists(),
        "a clean query must not write the halt latch"
    );
}

#[test]
fn query_buckets_skips_the_open_when_the_latch_is_present() {
    let tree = fixture::Tree::new("query-latched");
    fixture::install_kraken(&tree);
    std::fs::write(tree.latch(), b"already halted\n").expect("latch");
    let hid = fixture::SharedHid::new();
    push_tables(&hid);
    let err = query(&tree, Arc::clone(&hid)).expect_err("latch blocks device I/O");
    assert!(matches!(err, SinkError::Halted), "{err}");
    assert!(hid.sent().is_empty(), "hidraw was not written");
    assert_eq!(
        std::fs::read(tree.latch()).expect("latch"),
        b"already halted\n",
        "an existing latch is left in place"
    );
}

#[test]
fn query_buckets_refuses_when_interface0_is_usbfs() {
    let tree = fixture::Tree::new("query-usbfs");
    fixture::install_kraken(&tree);
    fixture::claim_usbfs(&tree);
    assert!(
        kraken_lcd::device::interface0_usbfs(&tree.sys).expect("sysfs"),
        "the service's usbfs claim is visible before any open"
    );
    let hid = fixture::SharedHid::new();
    push_tables(&hid);
    let err = query(&tree, Arc::clone(&hid)).expect_err("usbfs means the service is active");
    assert!(matches!(err, SinkError::DeviceUnavailable), "{err}");
    assert!(hid.sent().is_empty(), "a bound interface 0 is not opened");
}

#[test]
fn query_buckets_does_not_open_when_the_device_is_absent() {
    let tree = fixture::Tree::new("query-absent");
    let hid = fixture::SharedHid::new();
    let err = query(&tree, Arc::clone(&hid)).expect_err("no kraken in sysfs");
    assert!(matches!(err, SinkError::DeviceUnavailable), "{err}");
    assert!(hid.sent().is_empty());
    assert!(!tree.latch().exists());
}

#[test]
fn query_buckets_refuses_the_bootloader_before_opening() {
    let tree = fixture::Tree::new("query-bootloader");
    fixture::install_kraken(&tree);
    fixture::add_bootloader(&tree);
    let hid = fixture::SharedHid::new();
    push_tables(&hid);
    let err = query(&tree, Arc::clone(&hid)).expect_err("bootloader");
    assert!(matches!(err, SinkError::DeviceInBootloader), "{err}");
    assert!(hid.sent().is_empty());
}

#[test]
fn query_buckets_halts_when_pwm_enable_changes_and_stops() {
    let tree = fixture::Tree::new("query-halt");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    hid.push(fixture::empty_bucket());
    let sys = tree.sys.clone();
    hid.on_send(move |_| {
        std::fs::write(sys.join("class/hwmon/hwmon4/pwm1_enable"), "1").expect("flip enable");
    });
    let err = query(&tree, Arc::clone(&hid)).expect_err("cooling change");
    assert!(matches!(err, SinkError::Halted), "{err}");
    let sent = hid.sent();
    assert_eq!(sent.len(), 1, "no further report after the guard trips");
    assert_eq!(sent[0][0..3], [0x30, 0x04, 0]);
    let body = std::fs::read_to_string(tree.latch()).expect("latch written");
    assert!(body.contains("reason=pwm1_enable"), "{body}");
}

#[test]
fn query_buckets_stops_on_a_missing_reply_without_another_command() {
    let tree = fixture::Tree::new("query-noreply");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    hid.push(fixture::junk());
    let err = query(&tree, Arc::clone(&hid)).expect_err("no matching reply");
    assert!(matches!(err, SinkError::UploadFailed(_)), "{err}");
    assert_eq!(hid.sent().len(), 1);
    assert!(
        !tree.latch().exists(),
        "a protocol miss with cooling unchanged does not latch"
    );
}

#[test]
fn cli_source_dispatches_query_buckets_without_bulk_or_show_liquid() {
    let src = include_str!("../src/main.rs");
    assert!(
        src.contains("\"--query-buckets\""),
        "the CLI command is --query-buckets"
    );
    assert!(
        src.contains("query_buckets"),
        "main dispatches to the hidraw-only query"
    );
    assert!(
        !src.contains("BulkLink"),
        "the query command must not name the bulk port"
    );
}
