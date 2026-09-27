//! Upload and restore byte sequences against fake ports. No real device is opened.

#[path = "device_fixture.rs"]
mod fixture;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use kraken_lcd::device::proto::{self, Cmd, SlotId};
use kraken_lcd::device::{FakeLcd, LcdSink, Record, SinkError, Step, UploadFailed};
use kraken_lcd::render::Frame;
use tiny_skia::{Color, PremultipliedColorU8};

use fixture::{
    Tree, ack, count_prefix, encoded, install_kraken, junk, open_lcd, script_empty_open,
    script_show_slot,
};

fn slot(id: u8) -> SlotId {
    SlotId::try_new(id).expect("slot")
}

fn show_cmds(id: u8) -> [Cmd; 6] {
    let slot = slot(id);
    let bucket = proto::BucketId::try_new(id).expect("bucket");
    [
        Cmd::PreTransfer,
        Cmd::DeleteBucket(bucket),
        Cmd::SetupBucket { slot },
        Cmd::WriteStart(slot),
        Cmd::WriteEnd,
        Cmd::ShowSlot(slot),
    ]
}

#[test]
fn upload_and_restore_match_the_poc_capture() {
    let mut opened = fixture::opened("poc");
    assert!(opened.lcd.reupload_pending());
    assert!(
        opened
            .hid
            .timeouts()
            .iter()
            .any(|timeout| *timeout >= Duration::from_millis(400)
                && *timeout <= Duration::from_millis(500)),
        "reply budget is 500 ms, saw {:?}",
        opened.hid.timeouts()
    );
    script_show_slot(&opened.hid, 0);
    opened.lcd.show(&Frame::new()).expect("upload");

    let sent = opened.hid.sent();
    let queries: Vec<Cmd> = (0..16)
        .map(|id| Cmd::QueryBucket(proto::BucketId::try_new(id).expect("bucket")))
        .chain(std::iter::once(Cmd::LcdInfo))
        .chain(show_cmds(0))
        .collect();
    assert_eq!(sent.len(), queries.len(), "hid report count");
    for (report, cmd) in sent.iter().zip(&queries) {
        assert_eq!(report, &encoded(cmd), "{cmd:?}");
    }

    let chunks = opened.bulk.chunks();
    assert_eq!(chunks.len(), 801, "header plus 800 pixels");
    assert_eq!(chunks[0], proto::BULK_HEADER);
    assert!(chunks[1..].iter().all(|chunk| chunk.len() == 512));
    let pixels: Vec<u8> = chunks[1..]
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    assert_eq!(pixels.len(), proto::FRAME_BYTES);
    assert!(pixels.iter().all(|byte| *byte == 0));

    let restore_tree = Tree::new("restore");
    install_kraken(&restore_tree);
    let restore_hid = fixture::SharedHid::new();
    restore_hid.push(ack(proto::expected_prefix(&Cmd::ShowLiquid)));
    let mut restore = KrakenRestore::open(&restore_tree, Arc::clone(&restore_hid));
    let sent = restore_hid.sent();
    assert_eq!(sent, vec![encoded(&Cmd::ShowLiquid)]);
    assert_eq!(count_prefix(&sent, &[0x30, 0x01]), 0);
    assert_eq!(count_prefix(&sent, &[0x30, 0x04]), 0);
    let _ = &mut restore;
}

struct KrakenRestore;

impl KrakenRestore {
    fn open(
        tree: &Tree,
        hid: Arc<fixture::SharedHid>,
    ) -> kraken_lcd::device::KrakenLcd<fixture::FakeHid, kraken_lcd::device::NoBulk> {
        kraken_lcd::device::KrakenLcd::open_restore(&tree.request(0, false), move |path| {
            assert_eq!(path, Path::new("/dev/hidraw0"));
            Ok(fixture::FakeHid::attach(hid))
        })
    }
}

#[test]
fn refused_setup_sends_no_bulk_and_keeps_the_link() {
    let mut opened = fixture::opened("refused-setup");
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::DeleteBucket(
            proto::BucketId::try_new(0).expect("bucket"),
        ))));
    let mut refused = ack([0x33, 0x01]);
    refused[14] = 0x00;
    opened.hid.push(refused);
    let err = opened.lcd.show(&Frame::new()).expect_err("refused");
    assert_eq!(
        err,
        SinkError::UploadFailed(UploadFailed::Refused(Step::SetupBucket))
    );
    assert!(opened.bulk.chunks().is_empty());
    assert_eq!(
        opened.hid.alive.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(
        opened.bulk.alive.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn interleaved_status_reports_are_skipped() {
    let mut opened = fixture::opened("status");
    opened.hid.push(junk());
    opened.hid.push(junk());
    for cmd in show_cmds(0) {
        opened.hid.push(ack(proto::expected_prefix(&cmd)));
    }
    opened.lcd.show(&Frame::new()).expect("skipped status");
    let sent = opened.hid.sent();
    let show: Vec<[u8; 64]> = show_cmds(0).iter().map(encoded).collect();
    assert_eq!(&sent[sent.len() - show.len()..], &show);
}

#[test]
fn no_reply_is_upload_failed_before_write_start() {
    let mut opened = fixture::opened("no-reply");
    opened
        .hid
        .push_stale(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    let err = opened.lcd.show(&Frame::new()).expect_err("no reply");
    assert_eq!(
        err,
        SinkError::UploadFailed(UploadFailed::NoReply(Step::PreTransfer))
    );
    assert!(opened.bulk.chunks().is_empty());
    assert_eq!(
        opened.hid.alive.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn sixteen_unmatched_reports_are_no_reply_and_the_sixteenth_can_match() {
    let mut opened = fixture::opened("sixteen-miss");
    for _ in 0..16 {
        opened.hid.push(junk());
    }
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    let err = opened.lcd.show(&Frame::new()).expect_err("limit");
    assert_eq!(
        err,
        SinkError::UploadFailed(UploadFailed::NoReply(Step::PreTransfer))
    );

    let tree = Tree::new("sixteen-hit");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_empty_open(&hid, 0);
    for _ in 0..15 {
        hid.push(junk());
    }
    for cmd in show_cmds(0) {
        hid.push(ack(proto::expected_prefix(&cmd)));
    }
    let mut lcd = open_lcd(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, false).expect("open");
    lcd.show(&Frame::new()).expect("16th report matches");
}

#[test]
fn slots_rotate_without_overwriting_the_active_slot() {
    let mut opened = fixture::opened("rotate");
    for id in 0..8 {
        script_show_slot(&opened.hid, id);
        opened.lcd.show(&Frame::new()).expect("slot");
        assert_eq!(opened.lcd.active_slot(), Some(slot(id)));
    }
    script_show_slot(&opened.hid, 0);
    opened.lcd.show(&Frame::new()).expect("wrap");
    assert_eq!(opened.lcd.active_slot(), Some(slot(0)));

    let slots: Vec<u8> = opened
        .hid
        .sent()
        .into_iter()
        .filter(|report| report[0] == 0x36 && report[1] == 0x01)
        .map(|report| report[2])
        .collect();
    assert_eq!(slots, vec![0, 1, 2, 3, 4, 5, 6, 7, 0]);
    for pair in slots.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }
}

#[test]
fn orientation_and_rotate_deg_select_the_packed_pixels() {
    let tree = Tree::new("pack");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_empty_open(&hid, 1);
    script_show_slot(&hid, 0);
    let mut lcd = open_lcd(&tree, Arc::clone(&hid), Arc::clone(&bulk), 90, false).expect("open");
    assert_eq!(lcd.orientation(), Some(1));
    let mut frame = Frame::new();
    frame.0.fill(Color::from_rgba8(0, 0, 0, 255));
    frame.0.pixels_mut()[0] = PremultipliedColorU8::from_rgba(255, 0, 0, 255).expect("red");
    lcd.show(&frame).expect("show");
    let chunks = bulk.chunks();
    let pixels: Vec<u8> = chunks[1..]
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    let expected = proto::pack(&frame, 1, 90).expect("pack");
    assert_eq!(pixels, expected);
}

#[test]
fn bootloader_blocks_open_before_either_port() {
    let tree = Tree::new("bootloader");
    install_kraken(&tree);
    fixture::add_bootloader(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let err =
        open_lcd(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, false).expect_err("bootloader");
    assert_eq!(err, SinkError::DeviceInBootloader);
    assert_eq!(hid.alive.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(bulk.alive.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(hid.sent().is_empty());
}

#[test]
fn latch_blocks_open_and_restore() {
    let tree = Tree::new("latch-open");
    install_kraken(&tree);
    std::fs::write(tree.latch(), "already\n").expect("latch");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let err = open_lcd(&tree, hid, bulk, 0, false).expect_err("latched");
    assert_eq!(err, SinkError::Halted);

    let restore_hid = fixture::SharedHid::new();
    let restore = KrakenRestore::open(&tree, Arc::clone(&restore_hid));
    assert!(restore.is_halted());
    assert!(restore_hid.sent().is_empty());
    assert_eq!(
        restore_hid.alive.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn second_tick_does_not_halt_on_later_drift() {
    let mut opened = fixture::opened("tick-once");
    fixture::script_show_slot(&opened.hid, 0);
    opened.lcd.show(&Frame::new()).expect("show");
    opened.lcd.tick().expect("quiet tick");
    std::fs::write(opened.tree.dev_dir().join("devnum"), "9\n").expect("devnum");
    std::fs::write(
        opened.tree.hwmon().join("fan1_input"),
        format!("{}\n", fixture::PUMP_RPM / 2),
    )
    .expect("rpm");
    opened.lcd.tick().expect("idle drift is not a halt");
    assert!(!opened.lcd.is_halted());
    assert!(!opened.tree.latch().exists());
}

#[test]
fn post_open_without_usbfs_is_fatal() {
    let tree = Tree::new("no-usbfs");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let err = kraken_lcd::device::KrakenLcd::open(
        &tree.request(0, false),
        |_, _| Ok(fixture::FakeBulk::attach(std::sync::Arc::clone(&bulk))),
        |_| Ok(fixture::FakeHid::attach(std::sync::Arc::clone(&hid))),
    )
    .expect_err("post-open");
    assert!(
        matches!(err, SinkError::Fatal(_)),
        "post-open must be fatal, got {err:?}"
    );
    assert!(hid.sent().is_empty(), "no hygiene after a fatal post-open");
    assert_eq!(hid.alive.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(bulk.alive.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn cooling_change_during_the_claim_halts_before_hygiene() {
    let tree = Tree::new("claim-guard");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_empty_open(&hid, 0);
    let sys = tree.sys.clone();
    let err = kraken_lcd::device::KrakenLcd::open(
        &tree.request(0, false),
        move |_, _| {
            let iface = sys.join("bus/usb/devices/3-5/3-5:1.0");
            std::fs::create_dir_all(&iface).expect("iface");
            let link = iface.join("driver");
            if link.symlink_metadata().is_err() {
                std::os::unix::fs::symlink("../../../../bus/usb/drivers/usbfs", &link)
                    .expect("usbfs");
            }
            std::fs::write(sys.join("class/hwmon/hwmon4/pwm2_enable"), "1\n").expect("enable");
            Ok(fixture::FakeBulk::attach(bulk))
        },
        {
            let hid = std::sync::Arc::clone(&hid);
            move |_| Ok(fixture::FakeHid::attach(hid))
        },
    )
    .expect_err("claim changed cooling");
    assert_eq!(err, SinkError::Halted);
    assert!(tree.latch().is_file(), "latch");
    assert!(hid.sent().is_empty(), "hygiene must not run");
}

#[test]
fn abort_show_liquid_is_checked_by_the_guard() {
    let mut opened = fixture::opened("liquid-then-guard");
    fixture::script_show_slot(&opened.hid, 0);
    opened.bulk.fail_at(0);
    let hwmon = opened.tree.hwmon();
    opened.hid.on_send(move |report| {
        if report.starts_with(&[0x38, 0x01, 0x02]) {
            std::fs::write(hwmon.join("pwm2_enable"), "1\n").expect("enable");
        }
    });
    let err = opened.lcd.show(&Frame::new()).expect_err("aborted");
    assert_eq!(err, SinkError::Halted);
    assert!(opened.tree.latch().is_file());
    assert_eq!(
        count_prefix(&opened.hid.sent(), &[0x38, 0x01, 0x02, 0x00]),
        1
    );
}

#[test]
fn connect_is_bound_to_the_host_roots() {
    assert_eq!(kraken_lcd::device::SYS_ROOT, "/sys");
    assert_eq!(kraken_lcd::device::STATE_DIR, "/var/lib/kraken-lcd");
}

#[test]
fn fake_lcd_records_the_slot_packed_bytes_and_abort_show_liquid() {
    let mut fake = FakeLcd::new();
    fake.show(&Frame::new()).expect("slot 0");
    fake.show(&Frame::new()).expect("slot 1");
    fake.script(Err(SinkError::TransferAborted));
    assert_eq!(fake.show(&Frame::new()), Err(SinkError::TransferAborted));
    let cmds: Vec<&Cmd> = fake
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Cmd(cmd) => Some(cmd),
            Record::Bulk(_) => None,
        })
        .collect();
    assert_eq!(
        cmds,
        vec![
            &Cmd::ShowSlot(slot(0)),
            &Cmd::ShowSlot(slot(1)),
            &Cmd::ShowLiquid,
        ]
    );
    let bulk: Vec<&[u8]> = fake
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Bulk(bytes) => Some(bytes.as_slice()),
            Record::Cmd(_) => None,
        })
        .collect();
    assert_eq!(bulk.len(), 2);
    assert!(
        bulk.iter()
            .all(|bytes| bytes.len() == kraken_lcd::device::proto::FRAME_BYTES)
    );
    fake.script_tick(Err(SinkError::Halted));
    assert_eq!(fake.tick(), Err(SinkError::Halted));
    assert!(fake.is_halted());
}

#[test]
fn latch_that_is_not_a_regular_file_still_blocks_open() {
    let tree = Tree::new("latch-dir");
    install_kraken(&tree);
    std::fs::create_dir(tree.latch()).expect("dir latch");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let err = open_lcd(&tree, hid, bulk, 0, false).expect_err("dir latch");
    assert_eq!(err, SinkError::Halted);

    let tree = Tree::new("latch-link");
    install_kraken(&tree);
    std::os::unix::fs::symlink("nowhere", tree.latch()).expect("symlink latch");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let err = open_lcd(&tree, hid, bulk, 0, false).expect_err("symlink latch");
    assert_eq!(err, SinkError::Halted);

    let tree = Tree::new("latch-perm");
    install_kraken(&tree);
    let mut blocked = std::fs::metadata(&tree.state).expect("state").permissions();
    blocked.set_mode(0o0);
    std::fs::set_permissions(&tree.state, blocked).expect("hide state dir");
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    let opened = open_lcd(&tree, hid, bulk, 0, false);
    let mut restored = std::fs::metadata(&tree.state).expect("state").permissions();
    restored.set_mode(0o755);
    std::fs::set_permissions(&tree.state, restored).expect("restore state dir");
    assert_eq!(opened.expect_err("unreadable state dir"), SinkError::Halted);
}

#[test]
fn fake_lcd_records_cmds_and_scripts_the_sink_reply() {
    let mut fake = FakeLcd::new();
    fake.needs_reupload();
    assert!(fake.reupload_pending());
    fake.show(&Frame::new()).expect("show");
    fake.script(Err(SinkError::Halted));
    assert_eq!(fake.show(&Frame::new()), Err(SinkError::Halted));
    fake.restore_stock();
    let cmds: Vec<&Cmd> = fake
        .records()
        .iter()
        .filter_map(|record| match record {
            Record::Cmd(cmd) => Some(cmd),
            Record::Bulk(_) => None,
        })
        .collect();
    assert_eq!(cmds, vec![&Cmd::ShowSlot(slot(0))]);
    assert!(
        fake.records()
            .iter()
            .any(|record| matches!(record, Record::Bulk(bytes) if !bytes.is_empty()))
    );
    assert!(fake.is_halted());
}

#[test]
fn delete_is_sent_twice_when_the_first_reply_is_refused() {
    let mut opened = fixture::opened("delete-retry");
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    let mut refused = ack([0x33, 0x02]);
    refused[14] = 0x00;
    opened.hid.push(refused);
    opened.hid.push(ack([0x33, 0x02]));
    opened.hid.push(ack([0x33, 0x01]));
    opened.hid.push(ack([0x37, 0x01]));
    opened.hid.push(ack([0x37, 0x02]));
    opened.hid.push(ack([0x39, 0x01]));
    opened.lcd.show(&Frame::new()).expect("retried delete");
    let deletes = count_prefix(&opened.hid.sent(), &[0x32, 0x02, 0x00]);
    assert_eq!(deletes, 2);
    assert_eq!(opened.bulk.chunks().len(), 801);
}

#[test]
fn stale_matching_reply_is_drained_and_does_not_count() {
    let mut opened = fixture::opened("drain");
    opened
        .hid
        .push_stale(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    let err = opened.lcd.show(&Frame::new()).expect_err("drained");
    assert_eq!(
        err,
        SinkError::UploadFailed(UploadFailed::NoReply(Step::PreTransfer))
    );
}

#[test]
fn trace_flag_still_completes_one_upload() {
    let tree = Tree::new("trace");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_empty_open(&hid, 0);
    script_show_slot(&hid, 0);
    let mut lcd = open_lcd(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, true).expect("open");
    lcd.show(&Frame::new()).expect("traced upload");
}
