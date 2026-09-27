//! Read-only cooling baseline. A deviation latches and stops device I/O.
//!
//! `pwm1` and `pwm2` are copied into the snapshot for the log. They are not
//! compared: the firmware curve moves them while the enable files stay put.
//! The pump band is fixed here and is not configuration.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long the service waits before the follow-up cooling check.
///
/// The check that used to run on the next service tick now runs on the first
/// tick at least this long after the operation. At a 0.5 s tick that is the
/// fourth tick. The immediate post-operation check is unchanged.
pub const FOLLOW_UP_AFTER: Duration = Duration::from_secs(2);

/// `true` when `now` is at least [`FOLLOW_UP_AFTER`] after the operation.
#[must_use]
pub fn follow_up_due(operated_at: Instant, now: Instant) -> bool {
    now.saturating_duration_since(operated_at) >= FOLLOW_UP_AFTER
}

use super::log_at;
use super::sysfs;
use crate::log::Priority;

/// Percent half of `max(15% of baseline, 150 rpm)`.
pub const PUMP_BAND_PERCENT: u32 = 15;

/// Absolute floor of the pump band, in rpm.
pub const PUMP_BAND_FLOOR_RPM: u32 = 150;

/// Why a later snapshot does not match the baseline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Deviation {
    Z53Missing,
    Pwm1Enable,
    Pwm2Enable,
    Pump,
    Devnum,
    Bootloader,
    Unreadable,
}

impl Deviation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Z53Missing => "z53_missing",
            Self::Pwm1Enable => "pwm1_enable",
            Self::Pwm2Enable => "pwm2_enable",
            Self::Pump => "fan1_input",
            Self::Devnum => "devnum",
            Self::Bootloader => "bootloader",
            Self::Unreadable => "unreadable",
        }
    }
}

/// One read of the cooling inputs. Missing files stay `None` so the log can
/// still show the other side of the comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoolingSnapshot {
    pub z53: bool,
    pub pwm1_enable: Option<u32>,
    pub pwm2_enable: Option<u32>,
    pub pwm1: Option<u32>,
    pub pwm2: Option<u32>,
    pub fan1_rpm: Option<u32>,
    pub devnum: Option<u32>,
    pub bootloader: bool,
}

impl CoolingSnapshot {
    fn line(&self, label: &str) -> String {
        format!(
            "{label} z53={} pwm1_enable={} pwm2_enable={} pwm1={} pwm2={} fan1_rpm={} devnum={} bootloader={}",
            flag(self.z53),
            num(self.pwm1_enable),
            num(self.pwm2_enable),
            num(self.pwm1),
            num(self.pwm2),
            num(self.fan1_rpm),
            num(self.devnum),
            flag(self.bootloader),
        )
    }
}

fn flag(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn num(value: Option<u32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "missing".to_owned(),
    }
}

/// `max(baseline * 15 / 100, 150)`.
#[must_use]
pub fn pump_band_rpm(baseline: u32) -> u32 {
    let percent = baseline.saturating_mul(PUMP_BAND_PERCENT) / 100;
    percent.max(PUMP_BAND_FLOOR_RPM)
}

/// Sysfs root, the Kraken device directory, and the directory that holds `halted`.
#[derive(Clone, Debug)]
pub struct CoolingGuard {
    sys_root: PathBuf,
    device_dir: PathBuf,
    state_dir: PathBuf,
}

impl CoolingGuard {
    pub(crate) fn new(sys_root: &Path, device_dir: &Path, state_dir: &Path) -> Self {
        Self {
            sys_root: sys_root.to_path_buf(),
            device_dir: device_dir.to_path_buf(),
            state_dir: state_dir.to_path_buf(),
        }
    }

    /// Fail closed. Only a confirmed absence (`NotFound`) means the latch is clear.
    pub(crate) fn latch_present(&self) -> bool {
        match self.state_dir.join("halted").symlink_metadata() {
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
            Err(_) => true,
            Ok(_) => true,
        }
    }

    /// Baseline taken immediately before a device operation.
    pub(crate) fn capture(&self) -> Result<CoolingSnapshot, Deviation> {
        let snap = self.read();
        if let Some(dev) = required(&snap) {
            return Err(dev);
        }
        Ok(snap)
    }

    /// Compare `current` with `baseline`. `Ok` means cooling still matches.
    pub(crate) fn check(&self, baseline: &CoolingSnapshot) -> Result<(), Deviation> {
        let current = self.read();
        if let Some(dev) = compare(baseline, &current) {
            Err(dev)
        } else {
            Ok(())
        }
    }

    pub(crate) fn read(&self) -> CoolingSnapshot {
        let root = self.sys_root.canonicalize().ok();
        let z53 = root.as_deref().and_then(find_z53);
        let bootloader = root
            .as_deref()
            .and_then(|root| sysfs::bootloader_present_at(root).ok())
            .unwrap_or(true);
        CoolingSnapshot {
            z53: z53.is_some(),
            pwm1_enable: z53
                .as_deref()
                .and_then(|dir| read_u32(&dir.join("pwm1_enable"))),
            pwm2_enable: z53
                .as_deref()
                .and_then(|dir| read_u32(&dir.join("pwm2_enable"))),
            pwm1: z53.as_deref().and_then(|dir| read_u32(&dir.join("pwm1"))),
            pwm2: z53.as_deref().and_then(|dir| read_u32(&dir.join("pwm2"))),
            fan1_rpm: z53
                .as_deref()
                .and_then(|dir| read_u32(&dir.join("fan1_input"))),
            devnum: read_u32(&self.device_dir.join("devnum")),
            bootloader,
        }
    }

    /// Write `<state_dir>/halted` and fsync it. The body is the critical log line.
    pub(crate) fn write_latch(&self, body: &str) -> std::io::Result<()> {
        let path = self.state_dir.join("halted");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(body.as_bytes())?;
        rustix::fs::fsync(&file)?;
        // The directory entry itself has to reach disk, not only the file bytes.
        let dir = std::fs::File::open(&self.state_dir)?;
        rustix::fs::fsync(&dir)?;
        Ok(())
    }
}

pub(crate) fn format_halt(
    baseline: Option<&CoolingSnapshot>,
    current: &CoolingSnapshot,
    dev: Deviation,
) -> String {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let baseline = match baseline {
        Some(snap) => snap.line("baseline"),
        None => "baseline missing".to_owned(),
    };
    format!(
        "CRITICAL time={time} reason={}\n{baseline}\n{}",
        dev.as_str(),
        current.line("current")
    )
}

pub(crate) fn log_halt(body: &str) {
    log_at(Priority::Crit, body);
}

fn required(snap: &CoolingSnapshot) -> Option<Deviation> {
    if snap.bootloader {
        return Some(Deviation::Bootloader);
    }
    if !snap.z53 {
        return Some(Deviation::Z53Missing);
    }
    if snap.pwm1_enable.is_none() || snap.pwm2_enable.is_none() || snap.fan1_rpm.is_none() {
        return Some(Deviation::Unreadable);
    }
    if snap.devnum.is_none() {
        return Some(Deviation::Devnum);
    }
    None
}

fn compare(baseline: &CoolingSnapshot, current: &CoolingSnapshot) -> Option<Deviation> {
    if current.bootloader {
        return Some(Deviation::Bootloader);
    }
    if !current.z53 {
        return Some(Deviation::Z53Missing);
    }
    if current.pwm1_enable != baseline.pwm1_enable {
        return Some(Deviation::Pwm1Enable);
    }
    if current.pwm2_enable != baseline.pwm2_enable {
        return Some(Deviation::Pwm2Enable);
    }
    let (Some(before), Some(now)) = (baseline.fan1_rpm, current.fan1_rpm) else {
        return Some(Deviation::Pump);
    };
    if now.abs_diff(before) > pump_band_rpm(before) {
        return Some(Deviation::Pump);
    }
    if current.devnum != baseline.devnum {
        return Some(Deviation::Devnum);
    }
    None
}

/// `true` when `state_dir` is writable (`W_OK`). show-image pre-flight uses this.
pub(crate) fn state_dir_writable(state_dir: &Path) -> bool {
    rustix::fs::access(state_dir, rustix::fs::Access::WRITE_OK).is_ok()
}

pub(crate) fn find_z53(root: &Path) -> Option<PathBuf> {
    let class = root.join("class/hwmon");
    let entries = std::fs::read_dir(class).ok()?;
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(index) = name.strip_prefix("hwmon") else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(dir) = entry.path().canonicalize() else {
            continue;
        };
        if !dir.starts_with(root) {
            continue;
        }
        let Ok(sensor) = std::fs::read_to_string(dir.join("name")) else {
            continue;
        };
        if sensor.trim() == "z53" {
            found.push(dir);
        }
    }
    if found.len() == 1 { found.pop() } else { None }
}

fn read_u32(path: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    text.trim().parse().ok()
}
