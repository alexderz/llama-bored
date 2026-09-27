//! S9: foreign buckets are deleted at open; restore sends only ShowLiquid.

#[path = "device_fixture.rs"]
mod fixture;

use std::path::Path;
use std::sync::Arc;

use kraken_lcd::device::LcdSink;
use kraken_lcd::device::proto::{self, Cmd};

use fixture::{Tree, ack, install_kraken, occupied, open_lcd};

fn bucket_reply(id: u8) -> [u8; 64] {
    match id {
        0 | 1 | 3 | 4 => occupied(u16::from(id) * 401, 401),
        2 => occupied(100, 50),
        5..=7 => fixture::empty_bucket(),
        8 => occupied(4000, 100),
        9 => occupied(3000, 400),
        10 => occupied(3208, 10),
        11 => occupied(3200, 20),
        _ => fixture::empty_bucket(),
    }
}

#[test]
fn open_deletes_only_off_layout_and_overlapping_buckets() {
    let tree = Tree::new("hygiene");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    let bulk = fixture::SharedBulk::new();
    for id in 0..16 {
        hid.push(bucket_reply(id));
        let delete = matches!(id, 2 | 9 | 11);
        if delete {
            hid.push(ack(proto::expected_prefix(&Cmd::DeleteBucket(
                proto::BucketId::try_new(id).expect("bucket"),
            ))));
        }
    }
    hid.push(fixture::lcd_info(0));
    let lcd = open_lcd(&tree, Arc::clone(&hid), bulk, 0, false).expect("open");
    let sent = hid.sent();
    let deletes: Vec<u8> = sent
        .iter()
        .filter(|report| report[0] == 0x32 && report[1] == 0x02)
        .map(|report| report[2])
        .collect();
    assert_eq!(deletes, vec![2, 9, 11]);
    assert!(sent.iter().any(|report| report.starts_with(&[0x30, 0x01])));
    assert_eq!(
        sent.iter()
            .filter(|report| report.starts_with(&[0x30, 0x04]))
            .count(),
        16
    );
    drop(lcd);
}

#[test]
fn restore_stock_sends_only_show_liquid() {
    let tree = Tree::new("restore-only");
    install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    hid.push(ack(proto::expected_prefix(&Cmd::ShowLiquid)));
    let mut lcd = kraken_lcd::device::KrakenLcd::open_restore(&tree.request(0, false), {
        let hid = Arc::clone(&hid);
        move |path| {
            assert_eq!(path, Path::new("/dev/hidraw0"));
            Ok(fixture::FakeHid::attach(hid))
        }
    });
    let sent = hid.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0][0..4], [0x38, 0x01, 0x02, 0x00]);
    assert!(sent.iter().all(|report| report[0] != 0x30));
    let before = sent.len();
    lcd.restore_stock();
    let sent = hid.sent();
    assert_eq!(sent.len(), before + 1);
    assert_eq!(sent.last().expect("second")[0..4], [0x38, 0x01, 0x02, 0x00]);
    assert!(sent.iter().all(|report| report[0] != 0x30));
}
