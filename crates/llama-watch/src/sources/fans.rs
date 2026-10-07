//! Read-only fan speeds, for the tty FANS panel and llama-metrics.
//!
//! Two ways to say which fans (`[fans]`):
//!
//! - **Discovered** (#74, `hwmon` unset): every `fanN_input` on every
//!   hwmon chip ([`crate::sources::chips`]), so the AIO's pump and fan
//!   (`z53`) show beside the board's headers. The chips are found again
//!   every [`REDISCOVER`] and read at most every [`READ_EVERY`]. `allow`
//!   and `block` filter like `[temps]`: `block` wins, a non-empty `allow`
//!   limits to its matches. A fan no `allow` named and that has never read
//!   above 0 rpm this run is an empty header and stays hidden (`defaults =
//!   true`); once it spins it stays, so a later stall shows.
//! - **Named** (`hwmon` + `channels`, the older keys, shorthand for
//!   `allow = ["<hwmon>:fanN", ...]`): the `hwmonN` directory whose `name`
//!   is `hwmon`, found by that name, never by `N`. No match, or more than
//!   one, leaves the source absent: one log line, a rescan every
//!   [`RESCAN`], and the rest of the dashboard keeps working. Every
//!   configured channel is a row, in config order, read on every tick.
//!
//! Per fan the source reads `fanN_input` (rpm), `pwmN` (0..=255) and
//! `pwmN_enable` (the fan-control mode), and on discovery `fanN_label`.
//! Those control files are writable on sysfs and drive the fans. This
//! module only ever reads them, with [`std::fs::read_to_string`].
//! `tests/s14_fans_readonly.rs` fences it.
//!
//! The snapshot carries each fan's rpm and pwm as read-only numbers for
//! llama-metrics (#11); nothing reads them back to drive a fan.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};

use crate::config::{Fans, MAX_FAN_LABEL};
use crate::sources::chips::{self, Chip, Filter, Verdict, chip_name, input_index, printable};
use crate::sources::hwmon::{Located, find_optional, locate, read_sensor_name, sys_root};
use crate::sources::{Roots, SourceId};

/// How long an absent named device waits before the next `class/hwmon` scan.
pub const RESCAN: Duration = Duration::from_secs(10);
/// How often discovered chips and their fans are found again.
pub const REDISCOVER: Duration = Duration::from_secs(30);
/// Discovered fans are read at most this often.
pub const READ_EVERY: Duration = Duration::from_secs(1);
/// Most fans one discovery keeps.
pub const MAX_DISCOVERED: usize = 32;

/// One fan on one tick. A field is `None` when its file could not be read
/// or parsed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FanReading {
    /// Stable chip name ([`crate::sources::chips`]).
    pub chip: String,
    /// Channel `N` of `fanN_input`.
    pub channel: u32,
    /// Display label, printable ASCII, at most 10 characters.
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
    /// The panel header: the named hwmon, or the chips the rows come from.
    pub chip: String,
    /// `false` when the named hwmon is absent, or nothing was discovered.
    /// `fans` is then empty.
    pub present: bool,
    /// One reading per fan, in config or discovery order.
    pub fans: Vec<FanReading>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Seen {
    Unknown,
    Present,
    Absent,
}

/// Reads the configured or discovered fans.
pub struct FanSource {
    filter: Filter,
    rename: BTreeMap<String, String>,
    mode: Mode,
}

enum Mode {
    Named(Named),
    Discover(Discover),
}

struct Named {
    chip: String,
    channels: Vec<(u32, String)>,
    /// Canonical `hwmonN` directory from the last good scan.
    dir: Option<PathBuf>,
    next_scan: Option<Instant>,
    seen: Seen,
}

struct Discover {
    defaults: bool,
    chips: Vec<Chip>,
    inputs: Vec<FanInput>,
    next_discover: Option<Instant>,
    next_read: Option<Instant>,
    error: Option<String>,
    shown: Vec<FanReading>,
}

struct FanInput {
    chip: usize,
    index: u32,
    label: String,
    spun: bool,
}

impl FanSource {
    /// `None` when `[fans] enabled = false`.
    #[must_use]
    pub fn new(config: &Fans) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let mode = if config.hwmon.is_empty() {
            Mode::Discover(Discover {
                defaults: config.defaults,
                chips: Vec::new(),
                inputs: Vec::new(),
                next_discover: None,
                next_read: None,
                error: None,
                shown: Vec::new(),
            })
        } else {
            Mode::Named(Named {
                chip: config.hwmon.clone(),
                channels: config
                    .channels
                    .iter()
                    .copied()
                    .zip(config.resolved_labels())
                    .collect(),
                dir: None,
                next_scan: None,
                seen: Seen::Unknown,
            })
        };
        Some(Self {
            filter: Filter::new(&config.allow, &config.block),
            rename: config
                .rename
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), printable(v, MAX_FAN_LABEL)))
                .collect(),
            mode,
        })
    }

    /// Read the fans. Logs once when a device or input goes absent or returns.
    pub fn read(&mut self, roots: &Roots, mono: Instant, log: &mut impl Sink) -> FanPanel {
        match &mut self.mode {
            Mode::Named(named) => {
                let mut panel = named.read(roots, mono, log);
                panel.fans.retain(|fan| {
                    let input = format!("fan{}", fan.channel);
                    self.filter.verdict(&fan.chip, &[&input]) != Verdict::Blocked
                });
                for fan in &mut panel.fans {
                    if let Some(name) =
                        renamed(&self.rename, &fan.chip, &[&format!("fan{}", fan.channel)])
                    {
                        fan.label = name;
                    }
                }
                panel
            }
            Mode::Discover(found) => found.read(roots, mono, &self.filter, &self.rename, log),
        }
    }
}

/// A `rename` entry for `chip:<name>` or, for a whole chip, `chip`.
fn renamed(rename: &BTreeMap<String, String>, chip: &str, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| rename.get(&format!("{chip}:{name}").to_ascii_lowercase()))
        .cloned()
}

impl Named {
    fn read(&mut self, roots: &Roots, mono: Instant, log: &mut impl Sink) -> FanPanel {
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
        let chip = chip_name(&self.chip);
        let fans = self
            .channels
            .iter()
            .map(|(channel, label)| read_fan(&root, &dir, &chip, *channel, label.clone()))
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

impl Discover {
    fn read(
        &mut self,
        roots: &Roots,
        mono: Instant,
        filter: &Filter,
        rename: &BTreeMap<String, String>,
        log: &mut impl Sink,
    ) -> FanPanel {
        let root = sys_root(roots, SourceId::HwmonCpu).ok();
        if self.next_discover.is_none_or(|at| mono >= at) {
            self.next_discover = Some(mono + REDISCOVER);
            self.next_read = None;
            if let Some(root) = &root {
                self.discover(roots, root, log);
            }
        }
        if self.next_read.is_none_or(|at| mono >= at) {
            self.next_read = Some(mono + READ_EVERY);
            if let Some(root) = &root {
                self.read_values(root, filter, rename);
            }
        }
        let mut chips: Vec<&str> = Vec::new();
        for fan in &self.shown {
            if !chips.contains(&fan.chip.as_str()) {
                chips.push(&fan.chip);
            }
        }
        FanPanel {
            chip: chips.join(" \u{00B7} "),
            present: !self.shown.is_empty(),
            fans: self.shown.clone(),
        }
    }

    fn discover(&mut self, roots: &Roots, root: &Path, log: &mut impl Sink) {
        let chips = match chips::discover(roots) {
            Ok(chips) => {
                if self.error.take().is_some() {
                    log::emit(log, Priority::Info, "fans: hwmon readable again");
                }
                chips
            }
            Err(err) => {
                if self.error.is_none() {
                    log::emit(
                        log,
                        Priority::Warning,
                        &format!("fans: cannot list hwmon ({err}); the FANS panel stays empty"),
                    );
                }
                self.error = Some(err);
                self.chips.clear();
                self.inputs.clear();
                self.shown.clear();
                return;
            }
        };
        let mut inputs: Vec<FanInput> = Vec::new();
        for (chip_i, chip) in chips.iter().enumerate() {
            let Ok(entries) = std::fs::read_dir(&chip.dir) else {
                continue;
            };
            let mut indexes: Vec<u32> = entries
                .flatten()
                .filter_map(|e| input_index(e.file_name().to_str()?, "fan"))
                .collect();
            indexes.sort_unstable();
            indexes.dedup();
            for index in indexes.into_iter().filter(|n| *n <= 16) {
                if inputs.len() >= MAX_DISCOVERED {
                    break;
                }
                let label = chips::read_text(root, &chip.dir, &format!("fan{index}_label"))
                    .map(|text| printable(&text, 32))
                    .unwrap_or_default();
                let spun = self.inputs.iter().any(|old| {
                    old.spun
                        && old.index == index
                        && self
                            .chips
                            .get(old.chip)
                            .is_some_and(|c| c.chip == chip.chip)
                });
                inputs.push(FanInput {
                    chip: chip_i,
                    index,
                    label,
                    spun,
                });
            }
        }
        if inputs.len() != self.inputs.len() {
            log::emit(
                log,
                Priority::Info,
                &format!("fans: {} fan inputs discovered", inputs.len()),
            );
        }
        self.chips = chips;
        self.inputs = inputs;
    }

    fn read_values(&mut self, root: &Path, filter: &Filter, rename: &BTreeMap<String, String>) {
        let mut shown: Vec<FanReading> = Vec::new();
        for input in &mut self.inputs {
            let chip = &self.chips[input.chip];
            let channel = format!("fan{}", input.index);
            let mut names: Vec<&str> = vec![&channel];
            if !input.label.is_empty() {
                names.push(&input.label);
            }
            let fan = read_fan(root, &chip.dir, &chip.chip, input.index, String::new());
            if fan.rpm.is_some_and(|rpm| rpm > 0) {
                input.spun = true;
            }
            let show = match filter.verdict(&chip.chip, &names) {
                Verdict::Blocked | Verdict::NotAllowed => false,
                Verdict::Allowed => true,
                Verdict::Open => !self.defaults || input.spun,
            };
            if !show {
                continue;
            }
            let label = renamed(rename, &chip.chip, &names)
                .unwrap_or_else(|| default_label(&input.label, input.index));
            shown.push(FanReading { label, ..fan });
        }
        // Two chips with a `fan1` each: the later one says whose.
        for i in 1..shown.len() {
            if shown[..i].iter().any(|f| f.label == shown[i].label) {
                let label = format!("{} {}", shown[i].chip, shown[i].label);
                shown[i].label = label.chars().take(MAX_FAN_LABEL).collect();
            }
        }
        self.shown = shown;
    }
}

/// `fanN_label` without a trailing ` speed` (`Pump speed` is `Pump`), at
/// most 10 characters; `fanN` without one.
fn default_label(label: &str, index: u32) -> String {
    let trimmed = label.trim();
    let short = trimmed
        .strip_suffix(" speed")
        .or_else(|| trimmed.strip_suffix(" Speed"))
        .unwrap_or(trimmed)
        .trim();
    if short.is_empty() {
        format!("fan{index}")
    } else {
        short.chars().take(MAX_FAN_LABEL).collect()
    }
}

fn read_fan(root: &Path, dir: &Path, chip: &str, channel: u32, label: String) -> FanReading {
    FanReading {
        chip: chip.to_owned(),
        channel,
        label,
        rpm: read_number(root, dir, &format!("fan{channel}_input"))
            .and_then(|v| u32::try_from(v).ok()),
        pwm: read_number(root, dir, &format!("pwm{channel}"))
            .map(|v| u8::try_from(v.min(255)).unwrap_or(u8::MAX)),
        mode: read_number(root, dir, &format!("pwm{channel}_enable"))
            .and_then(|v| u32::try_from(v).ok()),
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

#[cfg(test)]
mod tests {
    use super::default_label;

    #[test]
    fn default_labels_drop_speed_and_fall_back_to_fan_n() {
        assert_eq!(default_label("Pump speed", 1), "Pump");
        assert_eq!(default_label("Fan speed", 2), "Fan");
        assert_eq!(default_label("", 3), "fan3");
        assert_eq!(default_label("CPU_OPT header", 7), "CPU_OPT he");
    }
}
