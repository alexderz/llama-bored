//! Read-only fan speeds from one Super-I/O hwmon, for the tty FANS panel.
//!
//! The device is the `hwmonN` directory whose `name` is `[fans] hwmon`
//! (e.g. `nct6798`). It is found by that name, never by `N`, which
//! changes across reboots. No match, or more than one, leaves the source
//! absent: one log line, a rescan every [`RESCAN`], and the rest of the
//! dashboard keeps working.
//!
//! Per channel `N` the source reads `fanN_input` (rpm), `pwmN` (0..=255)
//! and `pwmN_enable` (the fan-control mode). Those files are writable on
//! sysfs and drive the fans. This module only ever reads them, with
//! [`std::fs::read_to_string`]. `tests/s14_fans_readonly.rs` fences it.
//!
//! Fans stay in the watcher: the snapshot and wire schema do not carry them.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};

use crate::config::Fans;
use crate::sources::hwmon::{Located, find_optional, locate, read_sensor_name, sys_root};
use crate::sources::{Roots, SourceId};

/// How long an absent device waits before the next `class/hwmon` scan.
pub const RESCAN: Duration = Duration::from_secs(10);

/// One configured fan on one tick. A field is `None` when its file could
/// not be read or parsed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FanReading {
    /// Channel `N` from `[fans] channels`.
    pub channel: u32,
    /// Display label, printable ASCII.
    pub label: String,
    /// `fanN_input`, rpm.
    pub rpm: Option<u32>,
    /// `pwmN`, 0..=255.
    pub pwm: Option<u8>,
    /// `pwmN_enable`: 0 full speed, 1 manual, 2 and up an automatic mode.
    pub mode: Option<u32>,
}

/// What the FANS panel draws on one tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FanPanel {
    /// The configured hwmon name, for the panel header.
    pub chip: String,
    /// `false` when no single hwmon has that name. `fans` is then empty.
    pub present: bool,
    /// One reading per configured channel, in config order.
    pub fans: Vec<FanReading>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Seen {
    Unknown,
    Present,
    Absent,
}

/// Resolves the named hwmon and reads the configured channels.
pub struct FanSource {
    chip: String,
    channels: Vec<(u32, String)>,
    /// Canonical `hwmonN` directory from the last good scan.
    dir: Option<PathBuf>,
    next_scan: Option<Instant>,
    seen: Seen,
}

impl FanSource {
    /// `None` when `[fans] enabled = false`.
    #[must_use]
    pub fn new(config: &Fans) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let channels = config
            .channels
            .iter()
            .copied()
            .zip(config.resolved_labels())
            .collect();
        Some(Self {
            chip: config.hwmon.clone(),
            channels,
            dir: None,
            next_scan: None,
            seen: Seen::Unknown,
        })
    }

    /// Read every channel. Logs once when the device goes absent or returns.
    pub fn read(&mut self, roots: &Roots, mono: Instant, log: &mut impl Sink) -> FanPanel {
        let id = SourceId::HwmonCpu;
        let Ok(root) = sys_root(roots, id) else {
            self.mark_absent(mono, log, "sys root unreadable".to_owned());
            return self.absent();
        };
        // A cached directory is kept only while its `name` still matches.
        if let Some(dir) = &self.dir
            && read_sensor_name(id, &root, dir).as_deref() != Some(self.chip.as_str())
        {
            self.dir = None;
            self.next_scan = None;
        }
        if self.dir.is_none() && self.next_scan.is_none_or(|at| mono >= at) {
            match find_optional(roots, &root, id, &self.chip) {
                Ok(Some(dir)) => {
                    if self.seen != Seen::Present {
                        log::emit(
                            log,
                            Priority::Info,
                            &format!("fans: hwmon {} found", self.chip),
                        );
                    }
                    self.seen = Seen::Present;
                    self.dir = Some(dir);
                    self.next_scan = None;
                }
                Ok(None) => {
                    let why = format!("no hwmon named {}", self.chip);
                    self.mark_absent(mono, log, why);
                }
                Err(err) => self.mark_absent(mono, log, err.message),
            }
        }
        let Some(dir) = self.dir.clone() else {
            return self.absent();
        };
        let fans = self
            .channels
            .iter()
            .map(|(channel, label)| FanReading {
                channel: *channel,
                label: label.clone(),
                rpm: read_number(&root, &dir, &format!("fan{channel}_input"))
                    .and_then(|v| u32::try_from(v).ok()),
                pwm: read_number(&root, &dir, &format!("pwm{channel}"))
                    .map(|v| u8::try_from(v.min(255)).unwrap_or(u8::MAX)),
                mode: read_number(&root, &dir, &format!("pwm{channel}_enable"))
                    .and_then(|v| u32::try_from(v).ok()),
            })
            .collect();
        FanPanel {
            chip: self.chip.clone(),
            present: true,
            fans,
        }
    }

    fn mark_absent(&mut self, mono: Instant, log: &mut impl Sink, why: String) {
        self.dir = None;
        self.next_scan = Some(mono + RESCAN);
        if self.seen != Seen::Absent {
            log::emit(
                log,
                Priority::Warning,
                &format!("fans: absent ({why}); the FANS panel stays empty"),
            );
        }
        self.seen = Seen::Absent;
    }

    fn absent(&self) -> FanPanel {
        FanPanel {
            chip: self.chip.clone(),
            present: false,
            fans: Vec::new(),
        }
    }
}

/// A non-negative integer file under `dir`, or `None`.
fn read_number(root: &Path, dir: &Path, file: &str) -> Option<u64> {
    let Ok(Located::Inside(path)) = locate(SourceId::HwmonCpu, root, &dir.join(file)) else {
        return None;
    };
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The mode word for `pwmN_enable`: `full`, `manual` or `auto`.
#[must_use]
pub fn mode_word(mode: Option<u32>) -> &'static str {
    match mode {
        None => "--",
        Some(0) => "full",
        Some(1) => "manual",
        Some(_) => "auto",
    }
}
