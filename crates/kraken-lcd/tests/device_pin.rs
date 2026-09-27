//! T44 / RR7: the udev pin `/dev/kraken-lcd/hid` must name the node sysfs gives.
//!
//! The pin is read under `<sys_root>/../dev`, so production (`/sys`) reads
//! `/dev/kraken-lcd/hid` and these tests read a scratch `dev/` next to the
//! fake `sys/`. No real device node is opened, and none is stat'ed.

#[path = "device_fixture.rs"]
mod fixture;

use std::path::{Path, PathBuf};

use kraken_lcd::device::{KrakenLcd, SinkError, pre_open};

use fixture::{FakeBulk, FakeHid, Tree, install_kraken};

fn dev(tree: &Tree) -> PathBuf {
    tree.root.join("dev")
}

/// Fake `dev/hidrawN` files plus `dev/kraken-lcd/hid -> ../<target>`.
fn plant_pin(tree: &Tree, target: &str) {
    let dev = dev(tree);
    std::fs::create_dir_all(dev.join("kraken-lcd")).expect("pin dir");
    for name in ["hidraw0", "hidraw3"] {
        std::fs::write(dev.join(name), b"").expect("fake node");
    }
    std::os::unix::fs::symlink(format!("../{target}"), dev.join("kraken-lcd/hid"))
        .expect("pin link");
}

fn kraken(label: &str) -> Tree {
    let tree = Tree::new(label);
    install_kraken(&tree);
    tree
}

#[test]
fn absent_pin_keeps_the_sysfs_node() {
    let tree = kraken("pin-absent");
    let node = pre_open(&tree.sys).expect("pre_open without a pin");
    assert_eq!(node.hidraw, Path::new("/dev/hidraw0"));
}

#[test]
fn pin_on_the_sysfs_node_is_accepted() {
    let tree = kraken("pin-match");
    plant_pin(&tree, "hidraw0");
    let node = pre_open(&tree.sys).expect("matching pin");
    assert_eq!(node.hidraw, Path::new("/dev/hidraw0"));
}

#[test]
fn pin_on_another_hidraw_node_is_refused() {
    let tree = kraken("pin-mismatch");
    plant_pin(&tree, "hidraw3");
    assert_eq!(pre_open(&tree.sys), Err(SinkError::DeviceUnavailable));
}

#[test]
fn dangling_pin_is_refused() {
    let tree = kraken("pin-dangling");
    plant_pin(&tree, "hidraw9");
    assert_eq!(pre_open(&tree.sys), Err(SinkError::DeviceUnavailable));
}

#[test]
fn pin_that_is_not_a_symlink_is_refused() {
    let tree = kraken("pin-regular");
    let pin_dir = dev(&tree).join("kraken-lcd");
    std::fs::create_dir_all(&pin_dir).expect("pin dir");
    std::fs::write(pin_dir.join("hid"), b"").expect("regular pin");
    assert_eq!(pre_open(&tree.sys), Err(SinkError::DeviceUnavailable));
}

#[test]
fn pin_mismatch_opens_no_port() {
    let tree = kraken("pin-no-open");
    plant_pin(&tree, "hidraw3");
    let result = KrakenLcd::<FakeHid, FakeBulk>::open(
        &tree.request(0, false),
        |_, _| -> Result<FakeBulk, _> { panic!("bulk opened past a pin mismatch") },
        |_| -> Result<FakeHid, _> { panic!("hid opened past a pin mismatch") },
    );
    assert!(matches!(result, Err(SinkError::DeviceUnavailable)));
}
