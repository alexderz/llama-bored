//! S8: a failure after WriteStart sends one ShowLiquid, drops both handles,
//! and the next open starts with slot hygiene.

#[path = "device_fixture.rs"]
mod fixture;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use kraken_lcd::device::proto::{self, Cmd};
use kraken_lcd::device::{LcdSink, SinkError};
use kraken_lcd::render::Frame;

use fixture::{Tree, ack, install_kraken, open_lcd, script_empty_open};

fn bucket(id: u8) -> proto::BucketId {
    proto::BucketId::try_new(id).expect("bucket")
}

fn slot(id: u8) -> proto::SlotId {
    proto::SlotId::try_new(id).expect("slot")
}

fn prepare(hid: &fixture::SharedHid) {
    hid.push(ack(proto::expected_prefix(&Cmd::PreTransfer)));
    hid.push(ack(proto::expected_prefix(&Cmd::DeleteBucket(bucket(0)))));
    hid.push(ack(proto::expected_prefix(&Cmd::SetupBucket {
        slot: slot(0),
    })));
}

fn assert_one_liquid(opened: &fixture::Opened) {
    let sent = opened.hid.sent();
    let liquids: Vec<usize> = sent
        .iter()
        .enumerate()
        .filter(|(_, report)| report.starts_with(&[0x38, 0x01, 0x02, 0x00]))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(liquids.len(), 1, "one ShowLiquid");
    assert_eq!(liquids[0], sent.len() - 1, "nothing after ShowLiquid");
    assert_eq!(opened.hid.alive.load(Ordering::Relaxed), 0);
    assert_eq!(opened.bulk.alive.load(Ordering::Relaxed), 0);
}

fn reopen_starts_with_hygiene(label: &str) {
    let tree = Tree::new(label);
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    script_empty_open(&hid, 0);
    let lcd = open_lcd(&tree, Arc::clone(&hid), bulk, 0, false).expect("reopen");
    drop(lcd);
    let sent = hid.sent();
    assert!(sent.len() >= 16, "hygiene queries");
    for id in 0..16u8 {
        let index = usize::from(id);
        assert_eq!(sent[index][0..3], [0x30, 0x04, id], "query {id}");
    }
}

#[test]
fn write_start_without_a_reply_sends_one_show_liquid() {
    let mut opened = fixture::opened("step4");
    prepare(&opened.hid);
    let err = opened.lcd.show(&Frame::new()).expect_err("step 4");
    assert_eq!(err, SinkError::TransferAborted);
    assert!(opened.bulk.chunks().is_empty());
    assert_one_liquid(&opened);
    drop(opened);
    reopen_starts_with_hygiene("step4-next");
}

#[test]
fn bulk_failure_sends_one_show_liquid() {
    let mut opened = fixture::opened("step5");
    prepare(&opened.hid);
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::WriteStart(slot(0)))));
    opened.bulk.fail_at(0);
    let err = opened.lcd.show(&Frame::new()).expect_err("step 5");
    assert_eq!(err, SinkError::TransferAborted);
    assert!(opened.bulk.chunks().is_empty());
    assert_one_liquid(&opened);
    drop(opened);
    reopen_starts_with_hygiene("step5-next");
}

#[test]
fn write_end_without_a_reply_sends_one_show_liquid() {
    let mut opened = fixture::opened("step6");
    prepare(&opened.hid);
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::WriteStart(slot(0)))));
    let err = opened.lcd.show(&Frame::new()).expect_err("step 6");
    assert_eq!(err, SinkError::TransferAborted);
    assert_eq!(opened.bulk.chunks().len(), 801);
    assert_one_liquid(&opened);
    drop(opened);
    reopen_starts_with_hygiene("step6-next");
}

#[test]
fn refused_show_slot_sends_one_show_liquid() {
    let mut opened = fixture::opened("step7");
    prepare(&opened.hid);
    opened
        .hid
        .push(ack(proto::expected_prefix(&Cmd::WriteStart(slot(0)))));
    opened.hid.push(ack(proto::expected_prefix(&Cmd::WriteEnd)));
    let mut refused = ack([0x39, 0x01]);
    refused[14] = 0x00;
    opened.hid.push(refused);
    let err = opened.lcd.show(&Frame::new()).expect_err("step 7");
    assert_eq!(err, SinkError::TransferAborted);
    assert_one_liquid(&opened);
    let sent = opened.hid.sent();
    assert!(
        sent.iter()
            .any(|report| report.starts_with(&[0x38, 0x01, 0x04, 0x00])),
        "ShowSlot was attempted"
    );
    drop(opened);
    reopen_starts_with_hygiene("step7-next");
}
