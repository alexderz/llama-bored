//! Read-only cooling baseline. A deviation latches and stops device I/O.
//!
//! `pwm1` and `pwm2` are copied into the snapshot for the log. Neither is
//! compared on its own: the firmware curve moves them while the enable files
//! stay put. `pwm1` (the pump duty) only steers the pump band: a speed
//! explained by any duty within one firmware step of the compared duties
//! is in band (see [`pump_in_band`]). The band and the step limit are
//! fixed here and are not configuration.

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
/// 5-point step is at most 13 counts anywhere on the scale. [`pump_in_band`]
/// accepts a speed explained by any duty within this many counts of the
/// duties on the two reads, so a hunt that steps up and back between them
/// (#156: pwm1 held at 64, pump 953 -> 1115) and a step that landed before
/// the baseline (#154: pwm1 already 77, pump still 953, then 1121) stay in
/// band. A bigger jump between the two reads is not that curve, so the
/// fixed band applies and a pump that does not keep its speed halts.
pub const PUMP_DUTY_STEP_MAX: u32 = 13;

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
/// When both duties are readable and non-zero and they differ by at most
/// [`PUMP_DUTY_STEP_MAX`] counts (including equal), `now` may match any
/// speed explained by a duty in
/// `[min(D) - PUMP_DUTY_STEP_MAX, max(D) + PUMP_DUTY_STEP_MAX]`, clamped at
/// 1 count, with [`pump_band_rpm`] at each end. D is the two compared
/// duties. "Explained" is `before` scaled by that duty / the baseline duty,
/// the same rounding as a one-step follow. A duty of 0 or unknown, or a
/// jump of more than [`PUMP_DUTY_STEP_MAX`], keeps the fixed band around
/// `before`. A stopped pump, or a speed no duty in the envelope explains,
/// is still out of band.
#[must_use]
pub fn pump_in_band(
    before: u32,
    now: u32,
    duty_before: Option<u32>,
    duty_now: Option<u32>,
) -> bool {
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
}
