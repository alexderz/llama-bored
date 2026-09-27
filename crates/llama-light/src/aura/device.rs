//! Finding and opening the Aura controller's hidraw node.
//!
//! The only node this module opens is the target of the udev symlink
//! `<dev>/llama-light/aura` (from `94-llama-light-hidraw.rules`), through
//! [`crate::hidraw`]: sysfs must say that node is HID `0003:0B05:18F3`
//! before the open, and `fstat` must match sysfs after it.

use std::path::{Path, PathBuf};

use super::proto::EncodedReport;
use crate::hidraw::{self, HidFile, Pin};

pub use crate::hidraw::{Node as AuraNode, OpenError, PortError};

/// The Aura controller's HID id line in the hidraw parent's `uevent`.
pub const AURA_HID_ID: &str = "HID_ID=0003:00000B05:000018F3";
/// The udev symlink, relative to the dev root.
pub const AURA_PIN: &str = "llama-light/aura";

const PIN: Pin = Pin {
    link: AURA_PIN,
    hid_id: AURA_HID_ID,
    check: hidraw::no_check,
};

/// Byte sink for encoded reports. Tests supply a fake; production is [`AuraLink`].
pub trait AuraPort {
    /// Write one whole report.
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError>;
}

/// Opens an [`AuraPort`]. Tests supply a fake; production is [`HidrawOpener`].
pub trait AuraOpener {
    /// The port type.
    type Port: AuraPort;
    /// Resolve, check and open the controller.
    fn open(&mut self) -> Result<Self::Port, OpenError>;
}

/// Resolve the pin under `dev_root` and check it against `sys_root`.
/// Read-only: `readlink`, `realpath` and sysfs reads.
pub fn resolve(dev_root: &Path, sys_root: &Path) -> Result<AuraNode, OpenError> {
    hidraw::resolve(&PIN, dev_root, sys_root)
}

/// Open `node` write-only and check it is the character device sysfs named.
pub fn open_checked(node: &AuraNode) -> Result<AuraLink, OpenError> {
    hidraw::open_checked(node).map(|file| AuraLink { file })
}

/// The real hidraw file.
pub struct AuraLink {
    file: HidFile,
}

impl AuraPort for AuraLink {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        self.file.send(report.as_bytes())
    }
}

/// Production opener: [`resolve`] under `/dev` and `/sys`, then [`open_checked`].
pub struct HidrawOpener {
    dev_root: PathBuf,
    sys_root: PathBuf,
}

impl HidrawOpener {
    /// `dev_root` is `/dev` and `sys_root` is `/sys` in production.
    #[must_use]
    pub fn new(dev_root: impl Into<PathBuf>, sys_root: impl Into<PathBuf>) -> Self {
        Self {
            dev_root: dev_root.into(),
            sys_root: sys_root.into(),
        }
    }
}

impl AuraOpener for HidrawOpener {
    type Port = AuraLink;

    fn open(&mut self) -> Result<AuraLink, OpenError> {
        let node = resolve(&self.dev_root, &self.sys_root)?;
        open_checked(&node)
    }
}
