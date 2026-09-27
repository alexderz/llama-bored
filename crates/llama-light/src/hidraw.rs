//! Finding and opening one pinned hidraw node. Shared by the Aura and the
//! keyboard backends.
//!
//! The only node opened is the target of a udev symlink under
//! `<dev>/llama-light/` (from `94-llama-light-hidraw.rules`). Before the
//! open, sysfs must name the expected HID id behind that node (and a
//! [`Pin::check`] may look at more of sysfs). After the open, `fstat` must
//! show a character device whose number is the one sysfs gave, so a node
//! swapped between the check and the open is refused. Nodes are opened
//! write-only. Nothing under sysfs is written.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// Bytes per write: the report id (or `0x00` for an unnumbered report),
/// then 64 bytes. Both devices use this size.
pub const WRITE_LEN: usize = 65;

/// Largest sysfs file read.
const MAX_SYSFS: usize = 4096;

/// One pinned device.
#[derive(Clone, Copy, Debug)]
pub struct Pin {
    /// The udev symlink, relative to the dev root.
    pub link: &'static str,
    /// The line the hidraw parent's `uevent` must hold.
    pub hid_id: &'static str,
    /// Extra check on `<sys>/class/hidraw/hidrawN` (read-only). `true` passes.
    pub check: fn(&Path) -> bool,
}

/// No extra check.
#[must_use]
pub fn no_check(_class: &Path) -> bool {
    true
}

/// Why a device is not usable right now. Every variant is "absent" to the
/// service; the text is for the one log line on a transition.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum OpenError {
    /// The udev symlink does not exist.
    #[error("no udev symlink under /dev/llama-light")]
    NoPin,
    /// The symlink is not a symlink, or does not name a `hidrawN` in /dev.
    #[error("the udev symlink does not point at a hidraw node")]
    BadPin,
    /// sysfs does not describe the node as the expected device.
    #[error("sysfs does not show the expected device behind the pinned node")]
    WrongDevice,
    /// The kernel refused the open with `EPERM`: the unit's device list
    /// (fixed at unit start) does not cover this node.
    #[error("open refused by the unit's device allow list")]
    Denied,
    /// `Denied`, for a node that did not exist when the process started:
    /// a restart of the unit would let it in.
    #[error(
        "the device appeared after the unit started; its node is not in the unit's device allow list"
    )]
    NeedsRestart,
    /// The open failed.
    #[error("open failed: {0}")]
    Open(String),
    /// The opened file is not the character device sysfs named.
    #[error("opened node is not the character device sysfs names")]
    Swapped,
}

/// A write failed. The service treats the device as gone.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("write failed: {0}")]
pub struct PortError(pub String);

/// The node sysfs and the pin agree on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    /// `<dev>/hidrawN`.
    pub path: PathBuf,
    /// `major:minor` from sysfs.
    pub dev: (u32, u32),
}

/// Resolve `pin` under `dev_root` and check it against `sys_root`.
/// Read-only: `readlink`, `realpath` and sysfs reads.
pub fn resolve(pin: &Pin, dev_root: &Path, sys_root: &Path) -> Result<Node, OpenError> {
    let link = dev_root.join(pin.link);
    match link.symlink_metadata() {
        Err(err) if err.kind() == ErrorKind::NotFound => return Err(OpenError::NoPin),
        Err(_) => return Err(OpenError::BadPin),
        Ok(meta) if !meta.file_type().is_symlink() => return Err(OpenError::BadPin),
        Ok(_) => {}
    }
    let dev = dev_root.canonicalize().map_err(|_| OpenError::BadPin)?;
    let target = link.canonicalize().map_err(|_| OpenError::NoPin)?;
    if target.parent() != Some(dev.as_path()) {
        return Err(OpenError::BadPin);
    }
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(OpenError::BadPin)?
        .to_owned();
    let digits = name.strip_prefix("hidraw").ok_or(OpenError::BadPin)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(OpenError::BadPin);
    }
    let class = sys_root.join("class/hidraw").join(&name);
    let uevent = read_small(&class.join("device/uevent")).ok_or(OpenError::WrongDevice)?;
    if !uevent.lines().any(|line| line.trim() == pin.hid_id) {
        return Err(OpenError::WrongDevice);
    }
    if !(pin.check)(&class) {
        return Err(OpenError::WrongDevice);
    }
    let numbers = read_small(&class.join("dev")).ok_or(OpenError::WrongDevice)?;
    let dev_numbers = parse_dev(numbers.trim()).ok_or(OpenError::WrongDevice)?;
    Ok(Node {
        path: target,
        dev: dev_numbers,
    })
}

fn parse_dev(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.split_once(':')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn read_small(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    if text.len() > MAX_SYSFS {
        return None;
    }
    Some(text)
}

/// The first `MAX_SYSFS` bytes at most of a binary sysfs file (such as a
/// HID `report_descriptor`). Read-only.
#[must_use]
pub fn read_sysfs_bytes(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() > MAX_SYSFS {
        return None;
    }
    Some(bytes)
}

/// Open `node` write-only and check it is the character device sysfs named.
pub fn open_checked(node: &Node) -> Result<HidFile, OpenError> {
    let mut options = OpenOptions::new();
    options.write(true);
    options.custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32);
    let file = options.open(&node.path).map_err(|err| {
        if err.raw_os_error() == Some(rustix::io::Errno::PERM.raw_os_error()) {
            OpenError::Denied
        } else {
            OpenError::Open(err.kind().to_string())
        }
    })?;
    let stat = rustix::fs::fstat(&file).map_err(|_| OpenError::Swapped)?;
    let kind = rustix::fs::FileType::from_raw_mode(stat.st_mode);
    if kind != rustix::fs::FileType::CharacterDevice {
        return Err(OpenError::Swapped);
    }
    let rdev = stat.st_rdev;
    if (rustix::fs::major(rdev), rustix::fs::minor(rdev)) != node.dev {
        return Err(OpenError::Swapped);
    }
    Ok(HidFile { file })
}

/// An open, checked hidraw node.
pub struct HidFile {
    file: File,
}

impl HidFile {
    /// Write one whole report.
    pub fn send(&mut self, bytes: &[u8; WRITE_LEN]) -> Result<(), PortError> {
        for _ in 0..8 {
            match self.file.write(bytes) {
                Ok(n) if n == bytes.len() => return Ok(()),
                // A short write is not completed: the tail would be read as
                // a new report.
                Ok(_) => return Err(PortError("short write".to_owned())),
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(PortError(err.kind().to_string())),
            }
        }
        Err(PortError("interrupted".to_owned()))
    }
}
