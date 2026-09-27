//! Finding and opening the Aura controller's hidraw node.
//!
//! The only node this module opens is the target of the udev symlink
//! `<dev>/llama-light/aura` (from `94-llama-light-hidraw.rules`). Before
//! the open, sysfs must say that node is HID `0003:0B05:18F3`. After the
//! open, `fstat` must show a character device whose number is the one sysfs
//! gave, so a node swapped between the check and the open is refused.
//! The node is opened write-only. Nothing under sysfs is written.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use super::proto::{EncodedReport, REPORT_LEN};

/// The Aura controller's HID id line in the hidraw parent's `uevent`.
pub const AURA_HID_ID: &str = "HID_ID=0003:00000B05:000018F3";
/// The udev symlink, relative to the dev root.
pub const AURA_PIN: &str = "llama-light/aura";

/// Why the controller is not usable right now. Every variant is "absent"
/// to the service; the text is for the one log line on a transition.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum OpenError {
    /// The udev symlink does not exist.
    #[error("no llama-light/aura udev symlink")]
    NoPin,
    /// The symlink is not a symlink, or does not name a `hidrawN` in /dev.
    #[error("the llama-light/aura udev symlink does not point at a hidraw node")]
    BadPin,
    /// sysfs does not describe the node as `0b05:18f3`.
    #[error("sysfs does not show 0b05:18f3 behind the pinned node")]
    WrongDevice,
    /// The open failed.
    #[error("open failed: {0}")]
    Open(String),
    /// The opened file is not the character device sysfs named.
    #[error("opened node is not the character device sysfs names")]
    Swapped,
}

/// A write failed. The service treats the device as gone.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("aura write failed: {0}")]
pub struct PortError(pub String);

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

/// The node sysfs and the pin agree on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuraNode {
    /// `<dev>/hidrawN`.
    pub path: PathBuf,
    /// `major:minor` from sysfs.
    pub dev: (u32, u32),
}

/// Resolve the pin under `dev_root` and check it against `sys_root`.
/// Read-only: `readlink`, `realpath` and sysfs reads.
pub fn resolve(dev_root: &Path, sys_root: &Path) -> Result<AuraNode, OpenError> {
    let pin = dev_root.join(AURA_PIN);
    match pin.symlink_metadata() {
        Err(err) if err.kind() == ErrorKind::NotFound => return Err(OpenError::NoPin),
        Err(_) => return Err(OpenError::BadPin),
        Ok(meta) if !meta.file_type().is_symlink() => return Err(OpenError::BadPin),
        Ok(_) => {}
    }
    let dev = dev_root.canonicalize().map_err(|_| OpenError::BadPin)?;
    let target = pin.canonicalize().map_err(|_| OpenError::NoPin)?;
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
    if !uevent.lines().any(|line| line.trim() == AURA_HID_ID) {
        return Err(OpenError::WrongDevice);
    }
    let numbers = read_small(&class.join("dev")).ok_or(OpenError::WrongDevice)?;
    let dev_numbers = parse_dev(numbers.trim()).ok_or(OpenError::WrongDevice)?;
    Ok(AuraNode {
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
    if text.len() > 4096 {
        return None;
    }
    Some(text)
}

/// Open `node` write-only and check it is the character device sysfs named.
pub fn open_checked(node: &AuraNode) -> Result<AuraLink, OpenError> {
    let mut options = OpenOptions::new();
    options.write(true);
    options.custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32);
    let file = options
        .open(&node.path)
        .map_err(|err| OpenError::Open(err.kind().to_string()))?;
    let stat = rustix::fs::fstat(&file).map_err(|_| OpenError::Swapped)?;
    let kind = rustix::fs::FileType::from_raw_mode(stat.st_mode);
    if kind != rustix::fs::FileType::CharacterDevice {
        return Err(OpenError::Swapped);
    }
    let rdev = stat.st_rdev;
    if (rustix::fs::major(rdev), rustix::fs::minor(rdev)) != node.dev {
        return Err(OpenError::Swapped);
    }
    Ok(AuraLink { file })
}

/// The real hidraw file.
pub struct AuraLink {
    file: File,
}

impl AuraPort for AuraLink {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        let bytes: &[u8; REPORT_LEN] = report.as_bytes();
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
