//! Finding and opening the keyboard's lighting interface.
//!
//! The keyboard exposes several HID interfaces. Typing goes through the
//! boot keyboard interface and the kernel's input layer (evdev), never
//! through hidraw, so nothing here can affect it. The lighting interface
//! is the vendor one: usage page `0xFFC2`, USB interface 1. It is chosen
//! twice:
//!
//! - udev (`94-llama-light-hidraw.rules`) puts the `llama-light/keyboard`
//!   symlink only on the hidraw node whose USB interface has
//!   `bInterfaceNumber == 01`;
//! - here, before the open, sysfs must name HID `0003:1B1C:1B48` behind the
//!   pin, and the node's `report_descriptor` must open with
//!   `06 C2 FF` (Usage Page 0xFFC2). A pin on any other interface is
//!   refused as the wrong device and never opened.
//!
//! The open itself goes through [`crate::hidraw`]: write-only, then `fstat`
//! must match sysfs.

use std::path::{Path, PathBuf};

use super::proto::EncodedReport;
use crate::hidraw::{self, HidFile, Pin};

pub use crate::hidraw::{Node as KeyboardNode, OpenError, PortError};

/// The keyboard's HID id line in the hidraw parent's `uevent`.
pub const KEYBOARD_HID_ID: &str = "HID_ID=0003:00001B1C:00001B48";
/// The udev symlink, relative to the dev root.
pub const KEYBOARD_PIN: &str = "llama-light/keyboard";
/// First item of the lighting interface's report descriptor:
/// Usage Page (0xFFC2), a 2-byte global item.
pub const LIGHTING_USAGE_PAGE: [u8; 3] = [0x06, 0xC2, 0xFF];

/// Whether the node's report descriptor is the lighting interface's.
#[must_use]
pub fn is_lighting_interface(class: &Path) -> bool {
    hidraw::read_sysfs_bytes(&class.join("device/report_descriptor"))
        .is_some_and(|bytes| bytes.starts_with(&LIGHTING_USAGE_PAGE))
}

const PIN: Pin = Pin {
    link: KEYBOARD_PIN,
    hid_id: KEYBOARD_HID_ID,
    check: is_lighting_interface,
};

/// Byte sink for encoded reports. Tests supply a fake; production is [`KeyboardLink`].
pub trait KeyboardPort {
    /// Write one whole report.
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError>;
}

/// Opens a [`KeyboardPort`]. Tests supply a fake; production is [`KeyboardOpenerHidraw`].
pub trait KeyboardOpener {
    /// The port type.
    type Port: KeyboardPort;
    /// Resolve, check and open the lighting interface.
    fn open(&mut self) -> Result<Self::Port, OpenError>;
}

/// Resolve the pin under `dev_root` and check it against `sys_root`.
/// Read-only: `readlink`, `realpath` and sysfs reads.
pub fn resolve(dev_root: &Path, sys_root: &Path) -> Result<KeyboardNode, OpenError> {
    hidraw::resolve(&PIN, dev_root, sys_root)
}

/// The real hidraw file.
pub struct KeyboardLink {
    file: HidFile,
}

impl KeyboardPort for KeyboardLink {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        self.file.send(report.as_bytes())
    }
}

/// What a refused open means, given the node the pin named when the
/// process started. The unit's `DeviceAllow=` list is fixed when the unit
/// starts: a keyboard plugged in later (or given a new node number) is
/// refused with `EPERM` until the unit restarts.
#[must_use]
pub fn classify_denied(now: (u32, u32), at_start: Option<(u32, u32)>) -> OpenError {
    if at_start == Some(now) {
        OpenError::Denied
    } else {
        OpenError::NeedsRestart
    }
}

/// Production opener: [`resolve`] under `/dev` and `/sys`, then the checked open.
pub struct KeyboardOpenerHidraw {
    dev_root: PathBuf,
    sys_root: PathBuf,
    at_start: Option<(u32, u32)>,
}

impl KeyboardOpenerHidraw {
    /// `dev_root` is `/dev` and `sys_root` is `/sys` in production. Call it
    /// at process start: it records which node (if any) the pin names now.
    #[must_use]
    pub fn new(dev_root: impl Into<PathBuf>, sys_root: impl Into<PathBuf>) -> Self {
        let dev_root = dev_root.into();
        let sys_root = sys_root.into();
        let at_start = resolve(&dev_root, &sys_root).ok().map(|node| node.dev);
        Self {
            dev_root,
            sys_root,
            at_start,
        }
    }
}

impl KeyboardOpener for KeyboardOpenerHidraw {
    type Port = KeyboardLink;

    fn open(&mut self) -> Result<KeyboardLink, OpenError> {
        let node = resolve(&self.dev_root, &self.sys_root)?;
        match hidraw::open_checked(&node) {
            Ok(file) => Ok(KeyboardLink { file }),
            Err(OpenError::Denied) => Err(classify_denied(node.dev, self.at_start)),
            Err(err) => Err(err),
        }
    }
}
