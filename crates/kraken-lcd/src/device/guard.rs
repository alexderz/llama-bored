//! Read-only cooling baseline. A deviation latches and stops device I/O.
//!
//! `pwm1` and `pwm2` are copied into the snapshot for the log. Neither is
//! compared on its own: the firmware curve moves them while the enable files
//! stay put. `pwm1` (the pump duty) only steers the pump band: a small duty
//! step lets the expected pump speed follow it (see [`pump_in_band`]). The
//! band, the step limit, and how long a last duty change is remembered are
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
/// applies to a step that landed before the baseline (see
/// [`PUMP_DUTY_CHANGE_MAX`]).
pub const PUMP_DUTY_STEP_MAX: u32 = 13;

/// How long a last observed `pwm1` *change* may steer the next baseline's
/// pump band.
///
/// #147 follows a small duty step between the baseline and the check. The
/// firmware can also step before the baseline, while the pump is still at
/// the old speed (#154: pwm1 already 77, fan1 still 953, then 1121 two
/// seconds later). The pump takes a few seconds to change speed, so the
/// read just before the baseline often already shows the new duty. Capture
/// therefore keeps the last observed change (`from` → `to`, `seen_at`), not
/// the last read, and stamps `from` onto the baseline when `pwm1 == to` and
/// `seen_at` is at most this old. The 01:24 latch on titan reached the new
/// speed within about 2 s of the baseline; 5 s covers that ramp.
///
/// Cadence, so the window can be checked against how often the guard reads:
///
/// - Stream (titan: `upload.mode = "stream"`, 10 fps). Idle (pixels
///   unchanged) calls `pace_guard` every 1 s (`STREAM_GUARD`): a check then
///   a new baseline. Active (a new frame) also captures on every
///   `show_slot`, about 100 ms apart. A duty change is seen within one
///   second idle, or one frame when uploading. The 01:24 sequence is this
///   mode: if the change was already visible on the previous 1 s read,
///   `seen_at` is at most 1 s before capture and the 5 s window still
///   covers the follow-up ramp.
/// - Change / rotation. Cooling is read on open, on each upload (capture
///   and the post-op check), and once more [`FOLLOW_UP_AFTER`] (2 s) later.
///   The loop ticks every `writer.tick_s` (default 0.5 s) but does not read
///   cooling while idle. After the follow-up there is no read until the
///   next upload (`min_interval_s` default 60 s, and longer when the view
///   is still). A duty change in that gap is unseen; the next capture
///   records `seen_at = now`, so the 5 s window runs from that capture,
///   not from the real firmware step. If the pump had already arrived, the
///   same-speed check still passes. A drop in that follow-up that matches
///   the old duty's speed can be explained. This mode does not see ramps
///   that finish between two uploads.
///
/// The prior lives only on that baseline (the immediate check and the
/// follow-up). A change older than this, or a newer change, replaces it, so
/// a pump that slowly fails at a constant duty cannot keep borrowing an
/// old step.
pub const PUMP_DUTY_CHANGE_MAX: Duration = Duration::from_secs(5);

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
    /// `from` of the last observed `pwm1` change, if that change still
    /// matches this snapshot's duty and is within [`PUMP_DUTY_CHANGE_MAX`].
    /// Not logged; it only steers [`pump_in_band`].
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
/// way (the last observed change, not merely the previous read). A stopped
/// pump, a speed the duty does not explain, or a duty jump is still out of
/// band.
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
    /// Last readable `pwm1` and the last observed change of it. Capture may
    /// copy the change's `from` onto the baseline as `pwm1_prior`.
    pwm1_mem: Cell<Pwm1Memory>,
}

#[derive(Clone, Copy, Debug)]
struct Pwm1Memory {
    last: Option<u32>,
    change: Option<DutyChange>,
}

#[derive(Clone, Copy, Debug)]
struct DutyChange {
    from: u32,
    to: u32,
    seen_at: Instant,
}

impl CoolingGuard {
    pub(crate) fn new(sys_root: &Path, device_dir: &Path, state_dir: &Path) -> Self {
        Self {
            sys_root: sys_root.to_path_buf(),
            device_dir: device_dir.to_path_buf(),
            state_dir: state_dir.to_path_buf(),
            pwm1_mem: Cell::new(Pwm1Memory {
                last: None,
                change: None,
            }),
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
    /// If a last observed `pwm1` change is still fresh and matches this
    /// read's duty, its `from` is stored on the snapshot as `pwm1_prior`.
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
        let mem = note_pwm1(self.pwm1_mem.get(), snap.pwm1, now);
        self.pwm1_mem.set(mem);
        if keep_prior {
            snap.pwm1_prior = prior_from_change(mem.change, snap.pwm1, now);
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

fn note_pwm1(mem: Pwm1Memory, duty: Option<u32>, now: Instant) -> Pwm1Memory {
    let Some(duty) = duty else {
        return mem;
    };
    let change = match mem.last {
        Some(prev) if prev != duty => Some(DutyChange {
            from: prev,
            to: duty,
            seen_at: now,
        }),
        _ => mem.change,
    };
    Pwm1Memory {
        last: Some(duty),
        change,
    }
}

/// `Some(from)` when `duty` is this change's `to` and `seen_at` is at most
/// [`PUMP_DUTY_CHANGE_MAX`] ago. [`pump_in_band`] still applies
/// [`PUMP_DUTY_STEP_MAX`].
fn prior_from_change(change: Option<DutyChange>, duty: Option<u32>, now: Instant) -> Option<u32> {
    let change = change?;
    let duty = duty?;
    // Defence in depth: note_pwm1 always makes `to` the latest readable pwm1.
    if duty != change.to {
        return None;
    }
    (now.saturating_duration_since(change.seen_at) <= PUMP_DUTY_CHANGE_MAX).then_some(change.from)
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

    fn change(from: u32, to: u32, seen_at: Instant) -> DutyChange {
        DutyChange { from, to, seen_at }
    }

    #[test]
    fn change_seen_two_reads_before_baseline_is_in_band() {
        // Firmware stepped 64 -> 77; the next read already shows 77; a later
        // baseline is still 77 / 953. The last *change* is 64, not the last
        // read.
        let now = Instant::now();
        let mut mem = Pwm1Memory {
            last: None,
            change: None,
        };
        mem = note_pwm1(mem, Some(64), now);
        mem = note_pwm1(mem, Some(77), now);
        mem = note_pwm1(mem, Some(77), now);
        assert_eq!(prior_from_change(mem.change, Some(77), now), Some(64));
        let baseline = snap(77, 953, Some(64));
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), None);
    }

    #[test]
    fn stale_prior_duty_is_forgotten() {
        let now = Instant::now();
        let fresh = change(64, 77, now);
        assert_eq!(prior_from_change(Some(fresh), Some(77), now), Some(64));
        let fresh_at = now
            .checked_sub(PUMP_DUTY_CHANGE_MAX)
            .expect("clock is past the change window");
        assert_eq!(
            prior_from_change(Some(change(64, 77, fresh_at)), Some(77), now),
            Some(64)
        );
        let stale_at = now
            .checked_sub(PUMP_DUTY_CHANGE_MAX + Duration::from_nanos(1))
            .expect("clock is past the change window");
        assert_eq!(
            prior_from_change(Some(change(64, 77, stale_at)), Some(77), now),
            None
        );
        // A forgotten 64 -> 77 step does not explain 953 -> 1121.
        let baseline = snap(77, 953, None);
        let current = snap(77, 1121, None);
        assert_eq!(compare(&baseline, &current), Some(Deviation::Pump));
    }

    #[test]
    fn same_duty_reads_do_not_refresh_the_change_window() {
        let t0 = Instant::now();
        let mut mem = Pwm1Memory {
            last: None,
            change: None,
        };
        mem = note_pwm1(mem, Some(64), t0);
        mem = note_pwm1(mem, Some(77), t0);
        mem = note_pwm1(mem, Some(77), t0 + Duration::from_secs(2));
        mem = note_pwm1(mem, Some(77), t0 + Duration::from_secs(4));
        mem = note_pwm1(mem, Some(77), t0 + Duration::from_secs(6));
        assert_eq!(
            prior_from_change(mem.change, Some(77), t0 + Duration::from_secs(6)),
            None,
            "the window counts from the first sight of 64 -> 77, not later 77 reads"
        );
    }

    #[test]
    fn two_changes_in_a_row_use_only_the_latest() {
        let now = Instant::now();
        let mut mem = Pwm1Memory {
            last: None,
            change: None,
        };
        mem = note_pwm1(mem, Some(64), now);
        mem = note_pwm1(mem, Some(77), now);
        mem = note_pwm1(mem, Some(89), now);
        assert_eq!(prior_from_change(mem.change, Some(89), now), Some(77));
        assert_eq!(prior_from_change(mem.change, Some(77), now), None);
        // 1127 -> 1298 at pwm1 89 is the #147 step from 77, not from 64
        // (64 -> 89 is 25 counts and would keep the fixed band, which
        // 1298 exceeds: 1127 + 169 = 1296).
        let baseline = snap(89, 1127, Some(77));
        let current = snap(89, 1298, None);
        assert_eq!(compare(&baseline, &current), None);
        let from_64 = snap(89, 1127, Some(64));
        assert_eq!(compare(&from_64, &current), Some(Deviation::Pump));
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
