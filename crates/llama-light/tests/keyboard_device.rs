//! Keyboard node resolution against fake /dev and /sys trees. No real
//! device is opened: the pins point at scratch regular files, and the one
//! open attempted is refused before any write.

mod common;

use std::os::unix::fs::symlink;
use std::path::Path;

use common::scratch;
use llama_light::hidraw::open_checked;
use llama_light::keyboard::device::{
    KEYBOARD_HID_ID, KeyboardOpener, KeyboardOpenerHidraw, OpenError, classify_denied, resolve,
};

/// The lighting interface's descriptor opens with Usage Page 0xFFC2.
const LIGHTING_DESC: &[u8] = &[0x06, 0xC2, 0xFF, 0x09, 0x04, 0xA1, 0x01];
/// The boot keyboard interface: Usage Page (Generic Desktop), Usage (Keyboard).
const KEYBOARD_DESC: &[u8] = &[0x05, 0x01, 0x09, 0x06, 0xA1, 0x01];

fn seed(root: &Path, hidraw: &str, hid_id: &str, descriptor: &[u8], numbers: &str) {
    let dev = root.join("dev");
    std::fs::create_dir_all(dev.join("llama-light")).expect("dev");
    std::fs::write(dev.join(hidraw), b"").expect("node stand-in");
    let class = root.join("sys/class/hidraw").join(hidraw);
    std::fs::create_dir_all(class.join("device")).expect("class");
    std::fs::write(
        class.join("device/uevent"),
        format!("DRIVER=hid-generic\n{hid_id}\nHID_NAME=Corsair Corsair STRAFE RGB MK.2\n"),
    )
    .expect("uevent");
    std::fs::write(class.join("device/report_descriptor"), descriptor).expect("descriptor");
    std::fs::write(class.join("dev"), numbers).expect("dev numbers");
}

fn pin(root: &Path, hidraw: &str) {
    symlink(
        format!("../{hidraw}"),
        root.join("dev/llama-light/keyboard"),
    )
    .expect("pin");
}

#[test]
fn the_pin_on_the_lighting_interface_resolves() {
    let root = scratch("kbd-ok");
    seed(&root, "hidraw5", KEYBOARD_HID_ID, LIGHTING_DESC, "240:5\n");
    pin(&root, "hidraw5");
    let node = resolve(&root.join("dev"), &root.join("sys")).expect("resolve");
    assert_eq!(node.dev, (240, 5));
    assert_eq!(
        node.path,
        root.join("dev/hidraw5").canonicalize().expect("canon")
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_typing_interface_behind_the_pin_is_refused() {
    let root = scratch("kbd-typing");
    seed(&root, "hidraw4", KEYBOARD_HID_ID, KEYBOARD_DESC, "240:4\n");
    pin(&root, "hidraw4");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::WrongDevice)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_missing_descriptor_is_refused() {
    let root = scratch("kbd-nodesc");
    seed(&root, "hidraw5", KEYBOARD_HID_ID, LIGHTING_DESC, "240:5\n");
    std::fs::remove_file(root.join("sys/class/hidraw/hidraw5/device/report_descriptor"))
        .expect("rm");
    pin(&root, "hidraw5");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::WrongDevice)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn another_vendor_behind_the_pin_is_refused() {
    // The Kraken's id, with a vendor descriptor: still not the keyboard.
    let root = scratch("kbd-kraken");
    seed(
        &root,
        "hidraw1",
        "HID_ID=0003:00001E71:00003008",
        LIGHTING_DESC,
        "240:1\n",
    );
    pin(&root, "hidraw1");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::WrongDevice)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn no_pin_is_absent() {
    let root = scratch("kbd-nopin");
    seed(&root, "hidraw5", KEYBOARD_HID_ID, LIGHTING_DESC, "240:5\n");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::NoPin)
    );
    let mut opener = KeyboardOpenerHidraw::new(root.join("dev"), root.join("sys"));
    assert_eq!(opener.open().err(), Some(OpenError::NoPin));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_regular_file_behind_a_good_pin_is_not_opened_for_writing() {
    let root = scratch("kbd-swap");
    seed(&root, "hidraw5", KEYBOARD_HID_ID, LIGHTING_DESC, "240:5\n");
    pin(&root, "hidraw5");
    let mut opener = KeyboardOpenerHidraw::new(root.join("dev"), root.join("sys"));
    assert_eq!(opener.open().err(), Some(OpenError::Swapped));
    let node = resolve(&root.join("dev"), &root.join("sys")).expect("resolve");
    assert_eq!(open_checked(&node).err(), Some(OpenError::Swapped));
    assert_eq!(
        std::fs::read(root.join("dev/hidraw5")).expect("read"),
        b"",
        "nothing was written"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_refused_open_asks_for_a_restart_only_for_a_node_new_since_start() {
    assert_eq!(classify_denied((240, 5), Some((240, 5))), OpenError::Denied);
    assert_eq!(classify_denied((240, 5), None), OpenError::NeedsRestart);
    assert_eq!(
        classify_denied((240, 6), Some((240, 5))),
        OpenError::NeedsRestart
    );
}
