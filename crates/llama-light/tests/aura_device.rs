//! Node resolution against fake /dev and /sys trees. No real device is
//! opened: the one open here is a scratch regular file, which is refused.

mod common;

use std::os::unix::fs::symlink;
use std::path::Path;

use common::scratch;
use llama_light::aura::device::{AURA_HID_ID, AuraNode, OpenError, open_checked, resolve};

fn seed(root: &Path, hidraw: &str, hid_id: &str) {
    let dev = root.join("dev");
    std::fs::create_dir_all(dev.join("llama-light")).expect("dev");
    std::fs::write(dev.join(hidraw), b"").expect("node stand-in");
    let class = root.join("sys/class/hidraw").join(hidraw);
    std::fs::create_dir_all(class.join("device")).expect("class");
    std::fs::write(
        class.join("device/uevent"),
        format!(
            "DRIVER=hid-generic\n{hid_id}\nHID_NAME=AsusTek Computer Inc. AURA LED Controller\n"
        ),
    )
    .expect("uevent");
    std::fs::write(class.join("dev"), "240:2\n").expect("dev numbers");
}

#[test]
fn the_pin_resolves_to_a_checked_node() {
    let root = scratch("dev-ok");
    seed(&root, "hidraw2", AURA_HID_ID);
    symlink("../hidraw2", root.join("dev/llama-light/aura")).expect("pin");
    let node = resolve(&root.join("dev"), &root.join("sys")).expect("resolve");
    assert_eq!(
        node.path,
        root.join("dev/hidraw2").canonicalize().expect("canon")
    );
    assert_eq!(node.dev, (240, 2));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_missing_pin_is_absent() {
    let root = scratch("dev-nopin");
    seed(&root, "hidraw2", AURA_HID_ID);
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::NoPin)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn another_hid_behind_the_pin_is_refused() {
    let root = scratch("dev-wrong");
    seed(&root, "hidraw0", "HID_ID=0003:00001B1C:00001B48");
    symlink("../hidraw0", root.join("dev/llama-light/aura")).expect("pin");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::WrongDevice)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_pin_that_is_a_file_or_points_outside_hidraw_is_refused() {
    let root = scratch("dev-bad");
    seed(&root, "hidraw2", AURA_HID_ID);
    std::fs::write(root.join("dev/llama-light/aura"), b"").expect("plain file");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::BadPin)
    );
    std::fs::remove_file(root.join("dev/llama-light/aura")).expect("rm");
    std::fs::write(root.join("dev/sda"), b"").expect("other node");
    symlink("../sda", root.join("dev/llama-light/aura")).expect("pin");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::BadPin)
    );
    std::fs::remove_file(root.join("dev/llama-light/aura")).expect("rm");
    let outside = scratch("dev-outside");
    std::fs::write(outside.join("hidraw9"), b"").expect("outside");
    symlink(outside.join("hidraw9"), root.join("dev/llama-light/aura")).expect("pin");
    assert_eq!(
        resolve(&root.join("dev"), &root.join("sys")),
        Err(OpenError::BadPin)
    );
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&outside);
}

#[test]
fn a_regular_file_is_not_the_character_device_sysfs_named() {
    let root = scratch("dev-swap");
    let path = root.join("hidraw2");
    std::fs::write(&path, b"").expect("file");
    let node = AuraNode {
        path: path.clone(),
        dev: (240, 2),
    };
    assert_eq!(open_checked(&node).err(), Some(OpenError::Swapped));
    assert_eq!(
        std::fs::read(&path).expect("read"),
        b"",
        "nothing was written"
    );
    let _ = std::fs::remove_dir_all(&root);
}
