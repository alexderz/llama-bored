//! Read-only cooling baseline. A deviation latches and stops device I/O.
//!
//! A pump latch (`reason=fan1_input`) can clear itself after
//! [`PUMP_LATCH_CLEAR_AFTER`] of normal pump reads on the monotonic clock,
//! unless [`PUMP_LATCH_CLEAR_LIMIT`] automatic clears already fall in the
//! [`PUMP_LATCH_CLEAR_WINDOW`] before and including the latch's `time=`.
//!
//! `pwm1` and `pwm2` are copied into the snapshot for the log. Neither is
//! compared on its own: the firmware curve moves them while the enable files
//! stay put. `pwm1` (the pump duty) only steers the pump band: a speed
//! explained by any duty within one firmware step of the compared duties
//! is in band (see [`pump_in_band`]). The band and the step limit are
//! fixed here and are not configuration.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
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

/// Largest pump duty step, in `pwm1` counts, that the band follows.
///
/// The firmware reports the pump duty as a whole percent, which the driver
/// scales to 0..=255, and its curve has moved it one 5-point step at a time
/// as the coolant warms or cools (30 -> 35 %, pwm 77 -> 89, under #147). A
/// 5-point step is at most 13 counts anywhere on the scale. [`pump_in_band`]
/// accepts a speed explained by any duty within this many counts of the
/// duties on the two reads, so a hunt that steps up and back between them
/// (#156: pwm1 held at 64, pump 953 -> 1115) and a step that landed before
/// the baseline (#154: pwm1 already 77, pump still 953, then 1121) stay in
/// band. A bigger jump between the two reads is not that curve, so the
/// fixed band applies and a pump that does not keep its speed halts.
pub const PUMP_DUTY_STEP_MAX: u32 = 13;

/// Continuous normal pump reads that clear a pump latch (`reason=fan1_input`).
///
/// The window uses the monotonic clock. A wall-clock jump does not move it.
/// A service restart starts it over.
pub const PUMP_LATCH_CLEAR_AFTER: Duration = Duration::from_secs(600);

/// A longer monotonic gap between halted cooling reads restarts the clear window.
///
/// A service tick is at most 2 s, so a tick on time stays inside the window.
/// A stall of the monotonic clock restarts it. A wall-clock jump alone does not.
pub const PUMP_LATCH_MAX_GAP: Duration = Duration::from_secs(5);

/// Automatic clears allowed in the 24 h before and including a latch's `time=`.
pub const PUMP_LATCH_CLEAR_LIMIT: usize = 3;

/// How far before a latch's `time=` an automatic clear still counts.
///
/// The bound is the latch time, not the clock at the clear. A latch that is
/// already at [`PUMP_LATCH_CLEAR_LIMIT`] stays until `clear-halt`.
pub const PUMP_LATCH_CLEAR_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

/// State file of recent automatic-clear unix times, beside the latch.
pub const PUMP_AUTO_CLEARED_FILE: &str = "auto-cleared";

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

/// `true` when the pump speed `now` is explained by `before` and the duties.
///
/// A stopped pump (`now == 0` while `before > 0`) is always out of band,
/// before any envelope math: at a low duty the clamp at 1 plus the 150 rpm
/// floor can make the low edge 0. When both duties are readable and
/// non-zero and they differ by at most [`PUMP_DUTY_STEP_MAX`] counts
/// (including equal), `now` may match any speed explained by a duty in
/// `[min(D) - PUMP_DUTY_STEP_MAX, max(D) + PUMP_DUTY_STEP_MAX]`, clamped at
/// 1 count, with [`pump_band_rpm`] at each end. D is the two compared
/// duties. "Explained" is `before` scaled by that duty / the baseline duty,
/// rounding half up. A duty of 0 or unknown, or a jump of more than
/// [`PUMP_DUTY_STEP_MAX`], keeps the fixed band around `before`. A speed
/// no duty in the envelope explains is still out of band.
#[must_use]
pub fn pump_in_band(
    before: u32,
    now: u32,
    duty_before: Option<u32>,
    duty_now: Option<u32>,
) -> bool {
    if now == 0 && before > 0 {
        return false;
    }
    let (low, high) = match (duty_before, duty_now) {
        (Some(from), Some(to))
            if from != 0 && to != 0 && from.abs_diff(to) <= PUMP_DUTY_STEP_MAX =>
        {
            let duty_lo = from.min(to).saturating_sub(PUMP_DUTY_STEP_MAX).max(1);
            let duty_hi = from.max(to).saturating_add(PUMP_DUTY_STEP_MAX);
            let a = explained_rpm(before, from, duty_lo);
            let b = explained_rpm(before, from, duty_hi);
            (a.min(b), a.max(b))
        }
        _ => (before, before),
    };
    now >= low.saturating_sub(pump_band_rpm(low)) && now <= high.saturating_add(pump_band_rpm(high))
}

/// `before` scaled by `duty / duty_from`, rounding half up.
fn explained_rpm(before: u32, duty_from: u32, duty: u32) -> u32 {
    let scaled =
        (u64::from(before) * u64::from(duty) + u64::from(duty_from) / 2) / u64::from(duty_from);
    u32::try_from(scaled).unwrap_or(u32::MAX)
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
        cooling_snapshot(root.as_deref(), Some(&self.device_dir))
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

/// Parsed pump latch. Equality is the race check in the auto-clear.
///
/// `time` is the `time=` token on the first line. `None` means it was missing
/// or unparsable, and the latch is human-only. The duty and rpm fields are
/// stored so a human edit of the baseline line compares unequal. The clear
/// window still judges the pump with [`compare`] against the previous read,
/// and the identity check uses `z53`, the enable files, and `devnum`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PumpBaseline {
    time: Option<u64>,
    z53: bool,
    pwm1_enable: Option<u32>,
    pwm2_enable: Option<u32>,
    pwm1: Option<u32>,
    pwm2: Option<u32>,
    fan1_rpm: Option<u32>,
    devnum: Option<u32>,
    bootloader: bool,
}

impl PumpBaseline {
    /// `time=` from the latch's first line. `None` is missing or unparsable.
    #[must_use]
    pub(crate) fn time(&self) -> Option<u64> {
        self.time
    }
}

/// Recent automatic-clear times, or a file that cannot be trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AutoClearLog {
    /// Unix seconds, oldest first. Missing and empty files are an empty vec.
    Times(Vec<u64>),
    /// Unreadable or corrupt. The caller must not clear.
    Bad,
}

/// One halted-phase cooling read. Discovers `1e71:3008` read-only.
///
/// `idVendor`, `idProduct`, and `devnum` are the only USB files opened, plus
/// the same hwmon files as [`CoolingGuard::read`]. No HID, USB, or bulk node
/// is opened. Zero or several `3008` devices leave `devnum` missing.
#[must_use]
pub(crate) fn read_halt_cooling(sys_root: &Path) -> CoolingSnapshot {
    let root = sys_root.canonicalize().ok();
    let device = root.as_deref().and_then(find_kraken_device_dir);
    cooling_snapshot(root.as_deref(), device.as_deref())
}

/// `Some` only for a pump latch whose baseline line parses in full.
///
/// The first line must contain exactly one `reason=` token and its value
/// must be `fan1_input`. Any other reason, a missing baseline line, or a
/// baseline line that is not the [`format_halt`] field list returns `None`.
/// A missing or unparsable `time=` on that first line still returns `Some`,
/// with the time left empty, so the caller can keep the latch for a human.
#[must_use]
pub(crate) fn parse_pump_latch(text: &str) -> Option<PumpBaseline> {
    let mut lines = text.lines();
    let header = lines.next()?;
    if !header_is_pump(header) {
        return None;
    }
    let mut baseline = parse_baseline_line(lines.next()?)?;
    baseline.time = header_time(header);
    Some(baseline)
}

/// `true` when `current` can count toward the pump-latch clear window.
///
/// `previous` is the read from the tick before. The first read of a streak
/// passes `None` and skips [`compare`].
#[must_use]
pub(crate) fn halt_read_is_normal(
    baseline: &PumpBaseline,
    previous: Option<&CoolingSnapshot>,
    current: &CoolingSnapshot,
) -> bool {
    if required(current).is_some() {
        return false;
    }
    if previous.is_some_and(|prev| compare(prev, current).is_some()) {
        return false;
    }
    match current.fan1_rpm {
        Some(rpm) if rpm != 0 => {}
        _ => return false,
    }
    current.z53 == baseline.z53
        && current.pwm1_enable == baseline.pwm1_enable
        && current.pwm2_enable == baseline.pwm2_enable
        && current.devnum == baseline.devnum
        && !current.bootloader
}

/// Read [`PUMP_AUTO_CLEARED_FILE`]. Missing is no clears yet.
///
/// Anything that is not a regular file, including a symlink, is [`AutoClearLog::Bad`].
#[must_use]
pub(crate) fn read_auto_clears(state_dir: &Path) -> AutoClearLog {
    let path = state_dir.join(PUMP_AUTO_CLEARED_FILE);
    let text = match std::fs::symlink_metadata(&path) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return AutoClearLog::Times(Vec::new());
        }
        Err(_) => return AutoClearLog::Bad,
        Ok(meta) if meta.file_type().is_file() => match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => return AutoClearLog::Bad,
        },
        Ok(_) => return AutoClearLog::Bad,
    };
    parse_auto_clears(&text).map_or(AutoClearLog::Bad, AutoClearLog::Times)
}

/// Replace [`PUMP_AUTO_CLEARED_FILE`] via a temp file, fsync, rename, and a
/// directory fsync. The caller records a clear before it removes the latch.
pub(crate) fn store_auto_clears(state_dir: &Path, times: &[u64]) -> std::io::Result<()> {
    let mut body = String::new();
    for time in times {
        body.push_str(&time.to_string());
        body.push('\n');
    }
    write_durable(state_dir, PUMP_AUTO_CLEARED_FILE, body.as_bytes())
}

/// Result of removing `<state_dir>/halted`.
///
/// [`LatchRemoval::DirSyncFailed`] means the file is already gone.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LatchRemoval {
    /// The latch file is gone and the directory was synced.
    Removed,
    /// The latch file is still there.
    UnlinkFailed,
    /// The latch file is gone. Syncing its directory failed.
    DirSyncFailed,
}

/// Remove `<state_dir>/halted` and fsync the directory.
///
/// This is the only removal besides `clear-halt`. A latch that is already
/// gone still gets a directory sync, and that sync's failure is
/// [`LatchRemoval::DirSyncFailed`] rather than a failed removal.
pub(crate) fn remove_latch(state_dir: &Path) -> LatchRemoval {
    if let Err(err) = std::fs::remove_file(state_dir.join("halted"))
        && err.kind() != std::io::ErrorKind::NotFound
    {
        return LatchRemoval::UnlinkFailed;
    }
    match std::fs::File::open(state_dir) {
        Ok(dir) => match rustix::fs::fsync(&dir) {
            Ok(()) => LatchRemoval::Removed,
            Err(_) => LatchRemoval::DirSyncFailed,
        },
        Err(_) => LatchRemoval::DirSyncFailed,
    }
}

/// Body of `<state_dir>/halted`.
///
/// Only a regular file is readable. A symlink or any other node is an error,
/// so the caller treats it as unparsable. [`std::io::ErrorKind::NotFound`]
/// still means the latch is absent.
pub(crate) fn read_latch(state_dir: &Path) -> std::io::Result<String> {
    let path = state_dir.join("halted");
    let meta = std::fs::symlink_metadata(&path)?;
    if !meta.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "halted is not a regular file",
        ));
    }
    std::fs::read_to_string(path)
}

fn cooling_snapshot(root: Option<&Path>, device_dir: Option<&Path>) -> CoolingSnapshot {
    let z53 = root.and_then(find_z53);
    let bootloader = root
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
        devnum: device_dir.and_then(|dir| read_u32(&dir.join("devnum"))),
        bootloader,
    }
}

fn find_kraken_device_dir(root: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root.join("bus/usb/devices")).ok()?;
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.contains(':') || name.starts_with("usb") {
            continue;
        }
        let Ok(dir) = entry.path().canonicalize() else {
            continue;
        };
        if !dir.starts_with(root) {
            continue;
        }
        if is_kraken_lcd(&dir) {
            found.push(dir);
        }
    }
    if found.len() == 1 { found.pop() } else { None }
}

fn is_kraken_lcd(dir: &Path) -> bool {
    let Ok(vendor) = std::fs::read_to_string(dir.join("idVendor")) else {
        return false;
    };
    let Ok(product) = std::fs::read_to_string(dir.join("idProduct")) else {
        return false;
    };
    vendor.trim().eq_ignore_ascii_case("1e71") && product.trim().eq_ignore_ascii_case("3008")
}

fn header_is_pump(header: &str) -> bool {
    let mut reasons = 0u32;
    let mut pump = false;
    for token in header.split_whitespace() {
        let Some(value) = token.strip_prefix("reason=") else {
            continue;
        };
        reasons += 1;
        pump = value == "fan1_input";
    }
    reasons == 1 && pump
}

/// The single `time=` token on the latch's first line, or `None` when it is
/// missing, repeated, or not a decimal unix second.
fn header_time(header: &str) -> Option<u64> {
    let mut found: Option<Option<u64>> = None;
    for token in header.split_whitespace() {
        let Some(value) = token.strip_prefix("time=") else {
            continue;
        };
        if found.is_some() {
            return None;
        }
        found = Some(parse_unix_secs(value));
    }
    found.flatten()
}

fn parse_unix_secs(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn parse_baseline_line(line: &str) -> Option<PumpBaseline> {
    let mut tokens = line.split_whitespace();
    if tokens.next()? != "baseline" {
        return None;
    }
    let z53 = parse_flag(keyed(&mut tokens, "z53")?)?;
    let pwm1_enable = parse_opt_u32(keyed(&mut tokens, "pwm1_enable")?)?;
    let pwm2_enable = parse_opt_u32(keyed(&mut tokens, "pwm2_enable")?)?;
    let pwm1 = parse_opt_u32(keyed(&mut tokens, "pwm1")?)?;
    let pwm2 = parse_opt_u32(keyed(&mut tokens, "pwm2")?)?;
    let fan1_rpm = parse_opt_u32(keyed(&mut tokens, "fan1_rpm")?)?;
    let devnum = parse_opt_u32(keyed(&mut tokens, "devnum")?)?;
    let bootloader = parse_flag(keyed(&mut tokens, "bootloader")?)?;
    if tokens.next().is_some() {
        return None;
    }
    Some(PumpBaseline {
        time: None,
        z53,
        pwm1_enable,
        pwm2_enable,
        pwm1,
        pwm2,
        fan1_rpm,
        devnum,
        bootloader,
    })
}

fn keyed<'a>(tokens: &mut impl Iterator<Item = &'a str>, key: &str) -> Option<&'a str> {
    let token = tokens.next()?;
    let (got, value) = token.split_once('=')?;
    if got != key || value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn parse_flag(value: &str) -> Option<bool> {
    match value {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

fn parse_opt_u32(value: &str) -> Option<Option<u32>> {
    if value == "missing" {
        return Some(None);
    }
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(Some(value.parse().ok()?))
}

fn parse_auto_clears(text: &str) -> Option<Vec<u64>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() || !line.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        out.push(line.parse().ok()?);
    }
    Some(out)
}

fn write_durable(state_dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = state_dir.join(format!(".{name}.tmp"));
    // A leftover temp must be a regular file or absent. A directory, a
    // symlink, or anything else fails closed and is left in place.
    match std::fs::symlink_metadata(&tmp) {
        Ok(meta) if meta.file_type().is_file() => rustix::fs::unlink(&tmp)?,
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "auto-clear temp is not a regular file",
            ));
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    let result = publish_temp(state_dir, &tmp, name, bytes);
    if result.is_err()
        && let Ok(meta) = std::fs::symlink_metadata(&tmp)
        && meta.file_type().is_file()
    {
        let _ = rustix::fs::unlink(&tmp);
    }
    result
}

fn publish_temp(state_dir: &Path, tmp: &Path, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32)
        .open(tmp)?;
    file.write_all(bytes)?;
    rustix::fs::fsync(&file)?;
    drop(file);
    std::fs::rename(tmp, state_dir.join(name))?;
    let dir = std::fs::File::open(state_dir)?;
    rustix::fs::fsync(&dir)?;
    Ok(())
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
    if !pump_in_band(before, now, baseline.pwm1, current.pwm1) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(pwm1: u32, rpm: u32) -> CoolingSnapshot {
        CoolingSnapshot {
            z53: true,
            pwm1_enable: Some(0),
            pwm2_enable: Some(0),
            pwm1: Some(pwm1),
            pwm2: Some(51),
            fan1_rpm: Some(rpm),
            devnum: Some(2),
            bootloader: false,
        }
    }

    fn envelope_edges(before: u32, from: u32, to: u32) -> (u32, u32) {
        let duty_lo = from.min(to).saturating_sub(PUMP_DUTY_STEP_MAX).max(1);
        let duty_hi = from.max(to).saturating_add(PUMP_DUTY_STEP_MAX);
        let a = explained_rpm(before, from, duty_lo);
        let b = explained_rpm(before, from, duty_hi);
        let low = a.min(b);
        let high = a.max(b);
        (
            low.saturating_sub(pump_band_rpm(low)),
            high.saturating_add(pump_band_rpm(high)),
        )
    }

    #[test]
    fn hunt_at_held_duty_is_in_band() {
        // 21:54: pwm1 held at 64, pump 953 -> 1115 (the unseen 77 step).
        let baseline = snap(64, 953);
        let current = snap(64, 1115);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(953, 1115, Some(64), Some(64)));
    }

    #[test]
    fn hunt_down_at_held_duty_is_in_band() {
        // Mirror: pwm1 held at 77, pump 1120 -> 950 (the unseen 64 step).
        let baseline = snap(77, 1120);
        let current = snap(77, 950);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(1120, 950, Some(77), Some(77)));
    }

    #[test]
    fn firmware_duty_step_between_the_reads_is_in_band() {
        // #147: 77/1127 -> 89/1298.
        let baseline = snap(77, 1127);
        let current = snap(89, 1298);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(1127, 1298, Some(77), Some(89)));
        assert!(pump_in_band(1127, 1127, Some(77), Some(89)));
    }

    #[test]
    fn step_that_landed_before_the_baseline_is_in_band() {
        // #154: duty already 77, pump still at the 25 % speed, then arrived.
        let baseline = snap(77, 953);
        let current = snap(77, 1121);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(953, 1121, Some(77), Some(77)));
        // Same envelope covers the step-down that landed before the baseline.
        assert_eq!(compare(&snap(77, 1304), &snap(77, 1000)), None);
    }

    #[test]
    fn stopped_pump_latches() {
        assert_eq!(compare(&snap(64, 953), &snap(64, 0)), Some(Deviation::Pump));
        assert!(!pump_in_band(953, 0, Some(64), Some(64)));
        assert_eq!(compare(&snap(77, 953), &snap(77, 0)), Some(Deviation::Pump));
        assert!(!pump_in_band(953, 0, Some(77), Some(89)));
    }

    #[test]
    fn stopped_pump_at_a_low_duty_latches() {
        // Duty 20, envelope [7, 33]: 300 * 7 / 20 = 105, band 150, low edge 0.
        // Without the stopped-pump check, 20/300 -> 20/0 would pass.
        assert_eq!(compare(&snap(20, 300), &snap(20, 0)), Some(Deviation::Pump));
        assert!(!pump_in_band(300, 0, Some(20), Some(20)));
    }

    #[test]
    fn current_duty_widens_the_envelope() {
        // Held 64: duties [51, 77], 953 * 77 / 64 = 1147, high edge 1319.
        // With current 77: duties [51, 90], 953 * 90 / 64 = 1340, high edge 1541.
        // 1400 is past the held-64 edge and inside the widened one. Scaling
        // from the current duty (953 * 90 / 77 = 1114, high 1281) would latch.
        assert_eq!(compare(&snap(64, 953), &snap(77, 1400)), None);
        assert!(pump_in_band(953, 1400, Some(64), Some(77)));
        assert!(!pump_in_band(953, 1400, Some(64), Some(64)));
        assert_eq!(
            compare(&snap(64, 953), &snap(77, 1542)),
            Some(Deviation::Pump)
        );
    }

    #[test]
    fn full_speed_at_held_duty_latches() {
        assert_eq!(
            compare(&snap(64, 953), &snap(64, 2800)),
            Some(Deviation::Pump)
        );
        assert!(!pump_in_band(953, 2800, Some(64), Some(64)));
    }

    #[test]
    fn duty_jump_beyond_the_step_limit_keeps_the_fixed_band() {
        assert_eq!(PUMP_DUTY_STEP_MAX, 13);
        // 14 counts between the compared reads: 77 -> 91, pump followed.
        let jumped = snap(91, 1121);
        assert_eq!(compare(&snap(77, 953), &jumped), Some(Deviation::Pump));
        assert!(!pump_in_band(953, 1121, Some(77), Some(91)));
        // 77 -> 255 with the pump at full speed.
        assert_eq!(
            compare(&snap(64, 953), &snap(255, 2800)),
            Some(Deviation::Pump)
        );
        // A two-step duty pair (51 -> 77) is also a jump; the envelope is
        // not used. 1121 is outside the fixed band around 953 (1103).
        assert!(!pump_in_band(953, 1121, Some(51), Some(77)));
        assert_eq!(
            compare(&snap(51, 953), &snap(77, 1121)),
            Some(Deviation::Pump)
        );
    }

    #[test]
    fn envelope_edges_pass_and_one_past_them_latches() {
        let (low, high) = envelope_edges(953, 64, 64);
        // duties [51, 77]: 953 * 51 / 64 = 759, 953 * 77 / 64 = 1147;
        // band 150 / 172 -> 609 .. 1319.
        assert_eq!((low, high), (609, 1319));
        assert_eq!(compare(&snap(64, 953), &snap(64, low)), None);
        assert_eq!(
            compare(&snap(64, 953), &snap(64, low - 1)),
            Some(Deviation::Pump)
        );
        assert_eq!(compare(&snap(64, 953), &snap(64, high)), None);
        assert_eq!(
            compare(&snap(64, 953), &snap(64, high + 1)),
            Some(Deviation::Pump)
        );

        let (low, high) = envelope_edges(1120, 77, 77);
        // duties [64, 90]: 1120 * 64 / 77 = 931, 1120 * 90 / 77 = 1309;
        // band 150 / 196 -> 781 .. 1505.
        assert_eq!((low, high), (781, 1505));
        assert_eq!(compare(&snap(77, 1120), &snap(77, low)), None);
        assert_eq!(
            compare(&snap(77, 1120), &snap(77, low - 1)),
            Some(Deviation::Pump)
        );
        assert_eq!(compare(&snap(77, 1120), &snap(77, high)), None);
        assert_eq!(
            compare(&snap(77, 1120), &snap(77, high + 1)),
            Some(Deviation::Pump)
        );
    }

    #[test]
    fn hunt_at_the_new_duty_after_two_steps_is_in_band() {
        // 64 -> 77 -> 89, then 1127 -> 1298 at pwm1 89. Envelope around 89
        // includes one step down; 1298 is the 35 % speed.
        assert_eq!(compare(&snap(89, 1127), &snap(89, 1298)), None);
    }

    #[test]
    fn halt_line_format_is_unchanged() {
        let body = format_halt(Some(&snap(64, 953)), &snap(64, 1115), Deviation::Pump);
        assert!(body.contains("pwm1=64"), "{body}");
        assert!(body.contains("fan1_rpm=953"), "{body}");
        assert!(body.contains("fan1_rpm=1115"), "{body}");
        assert!(body.contains("reason=fan1_input"), "{body}");
        assert!(!body.contains("pwm1_prior"), "{body}");
        assert!(!body.contains(" prior="), "{body}");
    }

    #[test]
    fn pump_latch_text_parses_and_other_reasons_do_not() {
        let pump = format_halt(Some(&snap(64, 1000)), &snap(64, 0), Deviation::Pump);
        let baseline = parse_pump_latch(&pump).expect("pump latch");
        assert!(baseline.time.is_some(), "{pump}");
        assert!(halt_read_is_normal(&baseline, None, &snap(64, 1000)));
        assert!(!halt_read_is_normal(&baseline, None, &snap(64, 0)));
        let steady = snap(64, 1000);
        assert!(halt_read_is_normal(
            &baseline,
            Some(&steady),
            &snap(64, 1000)
        ));
        assert!(!halt_read_is_normal(
            &baseline,
            Some(&steady),
            &snap(64, 2800)
        ));
        let mut enable = snap(64, 1000);
        enable.pwm1_enable = Some(2);
        assert!(!halt_read_is_normal(&baseline, None, &enable));
        let mut devnum = snap(64, 1000);
        devnum.devnum = Some(9);
        assert!(!halt_read_is_normal(&baseline, None, &devnum));

        for dev in [
            Deviation::Z53Missing,
            Deviation::Pwm1Enable,
            Deviation::Pwm2Enable,
            Deviation::Devnum,
            Deviation::Bootloader,
            Deviation::Unreadable,
        ] {
            let body = format_halt(Some(&snap(64, 1000)), &snap(64, 1000), dev);
            assert!(parse_pump_latch(&body).is_none(), "{body}");
        }
        assert!(parse_pump_latch("not a latch\n").is_none());
        let missing = format_halt(None, &snap(64, 0), Deviation::Pump);
        assert!(parse_pump_latch(&missing).is_none(), "{missing}");
        let doubled = pump.replacen(
            "reason=fan1_input",
            "reason=fan1_input reason=fan1_input",
            1,
        );
        assert!(parse_pump_latch(&doubled).is_none());
    }

    #[test]
    fn baseline_line_with_a_trailing_token_is_rejected() {
        let pump = format_halt(Some(&snap(64, 1000)), &snap(64, 0), Deviation::Pump);
        let trailing = pump.replacen("bootloader=no\n", "bootloader=no extra\n", 1);
        assert_ne!(trailing, pump);
        assert!(parse_pump_latch(&trailing).is_none(), "{trailing}");
    }

    #[test]
    fn a_pump_latch_with_a_bad_time_still_parses() {
        let pump = format_halt(Some(&snap(64, 1000)), &snap(64, 0), Deviation::Pump);
        let baseline_line = pump.lines().nth(1).expect("baseline");
        for header in [
            "CRITICAL reason=fan1_input",
            "CRITICAL time= reason=fan1_input",
            "CRITICAL time=nope reason=fan1_input",
            "CRITICAL time=1 time=2 reason=fan1_input",
        ] {
            let text = format!("{header}\n{baseline_line}\n");
            let parsed = parse_pump_latch(&text).expect(header);
            assert_eq!(parsed.time, None, "{header}");
        }
    }

    #[test]
    fn auto_clear_file_rejects_corrupt_lines() {
        assert_eq!(parse_auto_clears(""), Some(vec![]));
        assert_eq!(parse_auto_clears("10\n20\n"), Some(vec![10, 20]));
        assert_eq!(parse_auto_clears("10\nnope\n"), None);
        assert_eq!(parse_auto_clears("\n"), None);
        assert_eq!(parse_auto_clears("10 \n"), None);
        assert_eq!(parse_auto_clears("10\n\n"), None);
    }

    #[test]
    fn store_auto_clears_replaces_the_file_and_drops_the_temp() {
        let dir = std::env::temp_dir().join(format!(
            "pump-auto-clear-store-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        assert!(matches!(read_auto_clears(&dir), AutoClearLog::Times(times) if times.is_empty()));
        store_auto_clears(&dir, &[10, 20]).expect("store");
        assert!(!dir.join(".auto-cleared.tmp").exists());
        assert_eq!(
            std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("read"),
            b"10\n20\n"
        );
        assert!(matches!(read_auto_clears(&dir), AutoClearLog::Times(times) if times == [10, 20]));
        std::fs::write(dir.join(PUMP_AUTO_CLEARED_FILE), b"nope\n").expect("corrupt");
        assert!(matches!(read_auto_clears(&dir), AutoClearLog::Bad));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_temp_file_is_replaced_and_a_non_file_temp_fails_closed() {
        let dir = std::env::temp_dir().join(format!(
            "pump-auto-clear-temp-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let tmp = dir.join(".auto-cleared.tmp");
        std::fs::write(&tmp, b"stale").expect("stale");
        store_auto_clears(&dir, &[1]).expect("replace");
        assert!(!tmp.exists());
        assert_eq!(
            std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("read"),
            b"1\n"
        );

        std::fs::create_dir(&tmp).expect("dir temp");
        assert!(store_auto_clears(&dir, &[2]).is_err());
        assert!(tmp.is_dir());
        assert_eq!(
            std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("untouched"),
            b"1\n"
        );
        std::fs::remove_dir(&tmp).expect("rmdir");

        std::os::unix::fs::symlink("nope", &tmp).expect("symlink");
        assert!(store_auto_clears(&dir, &[3]).is_err());
        assert!(
            tmp.symlink_metadata()
                .expect("meta")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("untouched"),
            b"1\n"
        );
        rustix::fs::unlink(&tmp).expect("unlink symlink");

        std::fs::remove_file(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("clear dest");
        std::fs::create_dir(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("dest dir");
        assert!(store_auto_clears(&dir, &[4]).is_err());
        assert!(
            !tmp.exists(),
            "a temp created before a failed rename is removed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_symlink_latch_and_a_symlink_auto_cleared_are_unreadable() {
        let dir = std::env::temp_dir().join(format!(
            "pump-auto-clear-symlink-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(
            dir.join("real-latch"),
            b"CRITICAL time=1 reason=fan1_input\n",
        )
        .expect("real");
        std::os::unix::fs::symlink("real-latch", dir.join("halted")).expect("link");
        let err = read_latch(&dir).expect_err("symlink latch");
        assert_ne!(err.kind(), std::io::ErrorKind::NotFound);
        std::fs::create_dir(dir.join("halted-dir")).expect("dir node");
        std::fs::rename(dir.join("halted"), dir.join("halted-link")).expect("move link");
        std::fs::rename(dir.join("halted-dir"), dir.join("halted")).expect("dir named halted");
        assert!(read_latch(&dir).is_err());

        std::fs::write(dir.join("real-log"), b"10\n").expect("log");
        std::os::unix::fs::symlink("real-log", dir.join(PUMP_AUTO_CLEARED_FILE)).expect("link");
        assert!(matches!(read_auto_clears(&dir), AutoClearLog::Bad));
        assert_eq!(
            std::fs::read(dir.join("real-log")).expect("target"),
            b"10\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_named_auto_cleared_is_unreadable() {
        let dir = std::env::temp_dir().join(format!(
            "pump-auto-clear-dir-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("dir");
        assert!(matches!(read_auto_clears(&dir), AutoClearLog::Bad));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_latch_removes_halted_only() {
        let dir = std::env::temp_dir().join(format!(
            "pump-remove-latch-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("halted"), b"latch\n").expect("latch");
        std::fs::write(dir.join(PUMP_AUTO_CLEARED_FILE), b"10\n").expect("log");
        assert_eq!(remove_latch(&dir), LatchRemoval::Removed);
        assert!(!dir.join("halted").exists());
        assert_eq!(
            std::fs::read(dir.join(PUMP_AUTO_CLEARED_FILE)).expect("kept"),
            b"10\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn halt_cooling_read_uses_the_one_kraken_and_skips_interfaces() {
        let dir = std::env::temp_dir().join(format!(
            "pump-halt-sysfs-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let hwmon = dir.join("class/hwmon/hwmon0");
        let device = dir.join("bus/usb/devices/3-5");
        std::fs::create_dir_all(&hwmon).expect("hwmon");
        std::fs::create_dir_all(&device).expect("device");
        std::fs::create_dir_all(dir.join("bus/usb/devices/3-5:1.0")).expect("iface");
        std::fs::create_dir_all(dir.join("bus/usb/devices/usb3")).expect("hub");
        std::fs::write(hwmon.join("name"), "z53\n").expect("name");
        for (name, value) in [
            ("pwm1_enable", "0"),
            ("pwm2_enable", "0"),
            ("pwm1", "64"),
            ("pwm2", "51"),
            ("fan1_input", "1000"),
        ] {
            std::fs::write(hwmon.join(name), format!("{value}\n")).expect(name);
        }
        std::fs::write(device.join("idVendor"), "1E71\n").expect("vendor");
        std::fs::write(device.join("idProduct"), "3008\n").expect("product");
        std::fs::write(device.join("devnum"), "2\n").expect("devnum");
        let snap = read_halt_cooling(&dir);
        assert!(snap.z53);
        assert_eq!(snap.pwm1_enable, Some(0));
        assert_eq!(snap.fan1_rpm, Some(1000));
        assert_eq!(snap.devnum, Some(2));
        assert!(!snap.bootloader);
        assert!(required(&snap).is_none());

        let other = dir.join("bus/usb/devices/3-6");
        std::fs::create_dir_all(&other).expect("second");
        std::fs::write(other.join("idVendor"), "1e71\n").expect("vendor");
        std::fs::write(other.join("idProduct"), "3008\n").expect("product");
        std::fs::write(other.join("devnum"), "4\n").expect("devnum");
        assert_eq!(read_halt_cooling(&dir).devnum, None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
