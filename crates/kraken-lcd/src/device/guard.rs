//! Read-only cooling baseline. A deviation latches and stops device I/O.
//!
//! `pwm1` and `pwm2` are copied into the snapshot for the log. Neither is
//! compared on its own: the firmware curve moves them while the enable files
//! stay put. `pwm1` (the pump duty) only steers the pump band: a small duty
//! step lets the expected pump speed follow it (see [`pump_in_band`]). The
//! band, the step limit, and how long a previous duty is remembered are
//! fixed here and are not configuration.

use std::cell::Cell;
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

/// Largest pump duty step, in `pwm1` counts, that the band follows.
///
/// The firmware reports the pump duty as a whole percent, which the driver
/// scales to 0..=255, and its curve has moved it one 5-point step at a time
/// as the coolant warms or cools (30 -> 35 %, pwm 77 -> 89, under #147). A
/// 5-point step is at most 13 counts anywhere on the scale. A bigger step
/// between two reads about 2 s apart is not that curve, so the fixed band
/// applies and a pump that does not keep its speed halts. The same limit
/// applies to a step that landed just before the baseline (see
/// [`PUMP_DUTY_PRIOR_MAX`]).
pub const PUMP_DUTY_STEP_MAX: u32 = 13;

/// How long a previous `pwm1` read may steer the next baseline's pump band.
///
/// #147 follows a small duty step between the baseline and the check. The
/// firmware can also step *before* the baseline, while the pump is still at
/// the old speed (#154: pwm1 already 77, fan1 still 953, then 1121 two
/// seconds later). Capture keeps the previous read's duty on that baseline
/// when the previous read is at most this old, and the check uses it as
/// the "from" duty when the two compared duties are equal. Stream checks
/// run every 1 s and the follow-up at 2 s, so 3 s covers those cadences. The
/// prior lives only on that baseline (the immediate check and the follow-up),
/// and an older read is forgotten, so a pump that slowly fails at a constant
/// duty cannot keep borrowing an old step.
pub const PUMP_DUTY_PRIOR_MAX: Duration = Duration::from_secs(3);

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
    /// Duty from the previous read, if that read was within
    /// [`PUMP_DUTY_PRIOR_MAX`]. Not logged; it only steers [`pump_in_band`].
    pwm1_prior: Option<u32>,
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

/// `true` when the pump speed `now` is explained by `before` and the duty.
///
/// With the duty unchanged, unknown or zero on either side, or moved by more
/// than [`PUMP_DUTY_STEP_MAX`] counts, `now` must be within
/// [`pump_band_rpm`] of `before`. When the firmware's curve made a small duty
/// (`pwm1`) step between the two reads, the pump is expected anywhere between
/// `before` (still ramping) and `before` scaled by the duty ratio (arrived),
/// and the band is applied at each end of that range. A caller that saw the
/// same duty on both reads may pass an earlier recent duty as `duty_before`
/// so a step that landed just before the first read is followed the same
/// way. A stopped pump, a speed the duty does not explain, or a duty jump is
/// still out of band.
#[must_use]
pub fn pump_in_band(
    before: u32,
    now: u32,
    duty_before: Option<u32>,
    duty_now: Option<u32>,
) -> bool {
    let scaled = match (duty_before, duty_now) {
        (Some(from), Some(to))
            if from != 0 && to != 0 && from != to && from.abs_diff(to) <= PUMP_DUTY_STEP_MAX =>
        {
            let scaled =
                (u64::from(before) * u64::from(to) + u64::from(from) / 2) / u64::from(from);
            u32::try_from(scaled).unwrap_or(u32::MAX)
        }
        _ => before,
    };
    let low = before.min(scaled);
    let high = before.max(scaled);
    now >= low.saturating_sub(pump_band_rpm(low)) && now <= high.saturating_add(pump_band_rpm(high))
}

/// Sysfs root, the Kraken device directory, and the directory that holds `halted`.
#[derive(Clone, Debug)]
pub struct CoolingGuard {
    sys_root: PathBuf,
    device_dir: PathBuf,
    state_dir: PathBuf,
    /// Last readable `pwm1` and when it was read. Capture may copy it onto
    /// the baseline as `pwm1_prior` when it is still fresh.
    last_pwm1: Cell<Option<(u32, Instant)>>,
}

impl CoolingGuard {
    pub(crate) fn new(sys_root: &Path, device_dir: &Path, state_dir: &Path) -> Self {
        Self {
            sys_root: sys_root.to_path_buf(),
            device_dir: device_dir.to_path_buf(),
            state_dir: state_dir.to_path_buf(),
            last_pwm1: Cell::new(None),
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
    ///
    /// If a previous read still has a fresh `pwm1`, it is stored on the
    /// snapshot as `pwm1_prior` and then replaced by this read's duty.
    pub(crate) fn capture(&self) -> Result<CoolingSnapshot, Deviation> {
        let snap = self.read_and_remember(true);
        if let Some(dev) = required(&snap) {
            return Err(dev);
        }
        Ok(snap)
    }

    /// Compare `current` with `baseline`. `Ok` means cooling still matches.
    pub(crate) fn check(&self, baseline: &CoolingSnapshot) -> Result<(), Deviation> {
        let current = self.read_and_remember(false);
        if let Some(dev) = compare(baseline, &current) {
            Err(dev)
        } else {
            Ok(())
        }
    }

    pub(crate) fn read(&self) -> CoolingSnapshot {
        self.read_and_remember(false)
    }

    fn read_and_remember(&self, keep_prior: bool) -> CoolingSnapshot {
        let mut snap = self.read_sysfs();
        let now = Instant::now();
        if keep_prior {
            snap.pwm1_prior = remembered_duty(self.last_pwm1.get(), now);
        }
        if let Some(duty) = snap.pwm1 {
            self.last_pwm1.set(Some((duty, now)));
        }
        snap
    }

    fn read_sysfs(&self) -> CoolingSnapshot {
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
            pwm1_prior: None,
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
    if !pump_in_band(
        before,
        now,
        pump_duty_from(baseline.pwm1, current.pwm1, baseline.pwm1_prior),
        current.pwm1,
    ) {
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

/// When both reads show the same duty, a recent previous duty is the "from"
/// side of a step that landed before the baseline.
fn pump_duty_from(
    duty_before: Option<u32>,
    duty_now: Option<u32>,
    duty_prior: Option<u32>,
) -> Option<u32> {
    match (duty_before, duty_now, duty_prior) {
        (Some(before), Some(now), Some(prior)) if before == now => Some(prior),
        _ => duty_before,
    }
}

/// `Some(duty)` when `last` was read at most [`PUMP_DUTY_PRIOR_MAX`] ago.
fn remembered_duty(last: Option<(u32, Instant)>, now: Instant) -> Option<u32> {
    let (duty, at) = last?;
    (now.saturating_duration_since(at) <= PUMP_DUTY_PRIOR_MAX).then_some(duty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(pwm1: u32, rpm: u32, prior: Option<u32>) -> CoolingSnapshot {
        CoolingSnapshot {
            z53: true,
            pwm1_enable: Some(0),
            pwm2_enable: Some(0),
            pwm1: Some(pwm1),
            pwm1_prior: prior,
            pwm2: Some(51),
            fan1_rpm: Some(rpm),
            devnum: Some(2),
            bootloader: false,
        }
    }

    #[test]
    fn prior_duty_step_up_before_baseline_is_in_band() {
        // The #154 latch: duty already 77, pump still at the 25 % speed, then
        // arrived by the follow-up. 64 -> 77 is one firmware step.
        let baseline = snap(77, 953, Some(64));
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(953, 1121, Some(64), Some(77)));
        assert!(!pump_in_band(953, 1121, Some(77), Some(77)));
    }

    #[test]
    fn prior_duty_step_down_before_baseline_is_in_band() {
        // Mirror: duty already 77, pump still at the 35 % speed, then slowed.
        // 1304 * 77 / 89 = 1128; low edge 959. The fixed band stops at 1109.
        let baseline = snap(77, 1304, Some(89));
        let current = snap(77, 1000, None);
        assert_eq!(compare(&baseline, &current), None);
        assert!(pump_in_band(1304, 1000, Some(89), Some(77)));
        assert!(!pump_in_band(1304, 1000, Some(77), Some(77)));
    }

    #[test]
    fn stopped_pump_after_a_prior_step_latches() {
        let baseline = snap(77, 953, Some(64));
        let current = snap(77, 0, None);
        assert_eq!(compare(&baseline, &current), Some(Deviation::Pump));
        assert!(!pump_in_band(953, 0, Some(64), Some(77)));
    }

    #[test]
    fn held_duty_without_a_prior_step_latches() {
        let baseline = snap(77, 953, None);
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), Some(Deviation::Pump));
        let same = snap(77, 953, Some(77));
        assert_eq!(compare(&same, &current), Some(Deviation::Pump));
    }

    #[test]
    fn prior_step_beyond_the_limit_latches() {
        assert_eq!(PUMP_DUTY_STEP_MAX, 13);
        // 51 -> 77 is 26 counts, two firmware steps.
        let baseline = snap(77, 953, Some(51));
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), Some(Deviation::Pump));
        assert!(!pump_in_band(953, 1121, Some(51), Some(77)));
        // 14 counts between the compared reads still use the fixed band.
        let jumped = snap(91, 1121, Some(77));
        assert_eq!(
            compare(&snap(77, 953, None), &jumped),
            Some(Deviation::Pump)
        );
    }

    #[test]
    fn stale_prior_duty_is_forgotten() {
        let now = Instant::now();
        assert_eq!(remembered_duty(Some((64, now)), now), Some(64));
        let fresh_at = now
            .checked_sub(PUMP_DUTY_PRIOR_MAX)
            .expect("clock is past the prior window");
        assert_eq!(remembered_duty(Some((64, fresh_at)), now), Some(64));
        let stale_at = now
            .checked_sub(PUMP_DUTY_PRIOR_MAX + Duration::from_nanos(1))
            .expect("clock is past the prior window");
        assert_eq!(remembered_duty(Some((64, stale_at)), now), None);
        // A forgotten 64 -> 77 step does not explain 953 -> 1121.
        let baseline = snap(77, 953, None);
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), Some(Deviation::Pump));
    }

    #[test]
    fn a_step_between_the_reads_still_follows() {
        let baseline = snap(77, 1127, Some(64));
        let current = snap(89, 1298, None);
        assert_eq!(compare(&baseline, &current), None);
    }

    #[test]
    fn halt_line_does_not_name_the_prior_duty() {
        let body = format_halt(
            Some(&snap(77, 953, Some(64))),
            &snap(77, 1121, None),
            Deviation::Pump,
        );
        assert!(body.contains("pwm1=77"), "{body}");
        assert!(body.contains("fan1_rpm=953"), "{body}");
        assert!(body.contains("fan1_rpm=1121"), "{body}");
        assert!(!body.contains("pwm1_prior"), "{body}");
        assert!(!body.contains(" prior="), "{body}");
    }
}
