//! S9 CLI: restore-stock sends exactly ShowLiquid, and skips the device when latched.

#[path = "device_fixture.rs"]
mod fixture;

use std::sync::Arc;

use kraken_lcd::device::proto::{self, Cmd};
use kraken_lcd::service;

#[test]
fn restore_stock_sends_exactly_show_liquid() {
    let tree = fixture::Tree::new("cli-restore");
    fixture::install_kraken(&tree);
    let hid = fixture::SharedHid::new();
    hid.push(fixture::ack(proto::expected_prefix(&Cmd::ShowLiquid)));
    let code = service::restore_stock_on(&tree.request(0, false), {
        let hid = Arc::clone(&hid);
        move |path| {
            assert_eq!(path, std::path::Path::new("/dev/hidraw0"));
            Ok(fixture::FakeHid::attach(hid))
        }
    });
    assert_eq!(code, 0);
    let sent = hid.sent();
    assert_eq!(sent.len(), 1, "only ShowLiquid");
    assert_eq!(sent[0][0..4], [0x38, 0x01, 0x02, 0x00]);
    assert!(sent.iter().all(|report| report[0] != 0x30));
}

#[test]
fn restore_stock_absent_device_exits_0() {
    let tree = fixture::Tree::new("cli-absent");
    let hid = fixture::SharedHid::new();
    let code = service::restore_stock_on(&tree.request(0, false), {
        let hid = Arc::clone(&hid);
        move |_path| Ok(fixture::FakeHid::attach(hid))
    });
    assert_eq!(code, 0);
    assert!(hid.sent().is_empty());
}
