//! Every hwmon temperature, discovered, for the TEMPS panel and
//! `llamabored_temperature_celsius` (#74). Read only.
//!
//! [`TempSource`] scans `/sys/class/hwmon/hwmon*` ([`crate::sources::chips`])
//! for `tempN_input` with its `tempN_label`, `tempN_max` and `tempN_crit`
//! every [`REDISCOVER`], so a chip that appears or goes away (hotplug, a
//! module loaded late) is followed, and reads the values at most every
//! [`READ_EVERY`]. Files are read with [`std::fs::read_to_string`] only,
//! inside the sys root; `tests/s14_fans_readonly.rs` fences this file.
//!
//! An input that cannot be read (`EIO`, `ENODATA`: some Super-I/O and NVMe
//! inputs fail while the device sleeps) is skipped for that read. One log
//! line when it starts failing and one when it reads again.
//!
//! # Which inputs show
//!
//! In this order:
//!
//! 1. **Plausible range, always.** Below 5 °C or at 127 °C and above is an
//!    unconnected or bogus input (`0` on an unwired PCH input, `-62` or
//!    `127` on a floating Super-I/O AUXTIN), never a temperature. Not even
//!    `allow` shows it.
//! 2. **`block`** hides; it wins over everything below.
//! 3. **`allow`**, when not empty, shows only its matches, and shows them
//!    past the built-in rules in 4.
//! 4. **Built-in rules** (`[temps] defaults = true`, the default), for
//!    inputs no `allow` named:
//!    - [`DEFAULT_BLOCK`]: on Super-I/O chips (`nct*`, `w83*`), the `PCH_*`
//!      inputs (unwired on most boards, read 0) and `TSI*`, `SMBUSMASTER*`
//!      and `PECI*`, which repeat the CPU's own sensor through the board;
//!    - **unchanging**: an input whose value has not moved for
//!      [`STUCK_AFTER`] while another input on the same chip has, such as a
//!      floating AUXTIN that reads a constant. It shows again as soon as it
//!      moves. Nothing is hidden this way in the first ten minutes.
//!
//! The GPU (NVML, chip `gpu`, sensor `gpu`) joins the list under the same
//! rules.
//!
//! # Display
//!
//! [`panel`] groups the shown inputs by chip into TEMPS rows with short
//! labels (`CPU  Tctl 68 · CCD1 64 · CCD2 62`, `NVMe0  50 · s1 50 · s2 67`)
//! and colours each value by its thresholds: `[temps] crit` / `warn`, else
//! the chip's `tempN_crit` / `tempN_max`, else 90 / 80 °C (55 / 45 for a
//! coolant).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};

use crate::config::Temps;
use crate::sources::Roots;
use crate::sources::chips::{self, Chip, Filter, Kind, Pattern, Verdict, input_index, printable};
use crate::sources::hwmon::sys_root;

/// How often the chips and their inputs are discovered again.
pub const REDISCOVER: Duration = Duration::from_secs(30);
/// Values are read at most this often.
pub const READ_EVERY: Duration = Duration::from_secs(1);
/// An input unchanged this long while its chip's others move is hidden.
pub const STUCK_AFTER: Duration = Duration::from_secs(600);
/// Lowest plausible reading, millidegrees: 5 °C.
pub const MIN_MILLI: i64 = 5_000;
/// First implausible reading, millidegrees: 127 °C.
pub const MAX_MILLI: i64 = 127_000;
/// Most inputs one discovery keeps.
pub const MAX_INPUTS: usize = 128;
/// Longest sensor name, as the wire allows.
pub const MAX_SENSOR: usize = llama_core::wire::MAX_TEMP_SENSOR_CHARS;
/// The built-in Super-I/O exclusions (module docs), `[temps] defaults`.
pub const DEFAULT_BLOCK: [&str; 8] = [
    "nct*:PCH_*",
    "nct*:TSI*",
    "nct*:SMBUSMASTER*",
    "nct*:PECI*",
    "w83*:PCH_*",
    "w83*:TSI*",
    "w83*:SMBUSMASTER*",
    "w83*:PECI*",
];
/// The GPU's chip and sensor name.
pub const GPU_CHIP: &str = "gpu";

/// One shown temperature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TempReading {
    /// Stable chip name ([`crate::sources::chips`]).
    pub chip: String,
    /// `tempN_label`, or `tempN` without one: printable ASCII.
    pub sensor: String,
    /// `tempN`.
    pub input: String,
    /// `true` when the chip gave a label.
    pub labelled: bool,
    /// What the chip measures.
    pub kind: Kind,
    /// Tenths of a degree Celsius.
    pub tenths: i32,
    /// `tempN_max`, tenths, when the chip gives a plausible one.
    pub max_tenths: Option<i32>,
    /// `tempN_crit`, tenths, when the chip gives a plausible one.
    pub crit_tenths: Option<i32>,
}

struct Input {
    chip: usize,
    index: u32,
    sensor: String,
    labelled: bool,
    max_tenths: Option<i32>,
    crit_tenths: Option<i32>,
    last_raw: Option<i64>,
    changed_at: Option<Instant>,
    failing: bool,
}

/// Discovers and reads every hwmon temperature.
pub struct TempSource {
    filter: Filter,
    defaults: bool,
    default_block: Filter,
    chips: Vec<Chip>,
    inputs: Vec<Input>,
    next_discover: Option<Instant>,
    next_read: Option<Instant>,
    discover_error: Option<String>,
    shown: Vec<TempReading>,
}

impl TempSource {
    /// `None` when `[temps] enabled = false`.
    #[must_use]
    pub fn new(config: &Temps) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let default_block: Vec<String> = DEFAULT_BLOCK.iter().map(|p| (*p).to_owned()).collect();
        Some(Self {
            filter: Filter::new(&config.allow, &config.block),
            defaults: config.defaults,
            default_block: Filter::new(&[], &default_block),
            chips: Vec::new(),
            inputs: Vec::new(),
            next_discover: None,
            next_read: None,
            discover_error: None,
            shown: Vec::new(),
        })
    }

    /// The shown temperatures, GPU included, most important first.
    pub fn read(
        &mut self,
        roots: &Roots,
        mono: Instant,
        gpu_c: Option<f32>,
        log: &mut impl Sink,
    ) -> Vec<TempReading> {
        let root = sys_root(roots, crate::sources::SourceId::HwmonCpu).ok();
        if self.next_discover.is_none_or(|at| mono >= at) {
            self.next_discover = Some(mono + REDISCOVER);
            if let Some(root) = &root {
                self.discover(roots, root, log);
            }
            self.next_read = None;
        }
        if self.next_read.is_none_or(|at| mono >= at) {
            self.next_read = Some(mono + READ_EVERY);
            if let Some(root) = &root {
                self.read_values(root, mono, log);
            }
        }
        let mut out = self.shown.clone();
        if let Some(gpu) = self.gpu_reading(gpu_c) {
            let at = out
                .iter()
                .position(|r| r.kind > Kind::Gpu)
                .unwrap_or(out.len());
            out.insert(at, gpu);
        }
        out
    }

    fn gpu_reading(&self, gpu_c: Option<f32>) -> Option<TempReading> {
        let tenths = tenths_of_c(gpu_c?)?;
        if !plausible(tenths)
            || matches!(
                self.filter.verdict(GPU_CHIP, &[GPU_CHIP]),
                Verdict::Blocked | Verdict::NotAllowed
            )
        {
            return None;
        }
        Some(TempReading {
            chip: GPU_CHIP.to_owned(),
            sensor: GPU_CHIP.to_owned(),
            input: GPU_CHIP.to_owned(),
            labelled: false,
            kind: Kind::Gpu,
            tenths,
            max_tenths: None,
            crit_tenths: None,
        })
    }

    fn discover(&mut self, roots: &Roots, root: &Path, log: &mut impl Sink) {
        let chips = match chips::discover(roots) {
            Ok(chips) => {
                if self.discover_error.take().is_some() {
                    log::emit(log, Priority::Info, "temps: hwmon readable again");
                }
                chips
            }
            Err(err) => {
                if self.discover_error.is_none() {
                    log::emit(
                        log,
                        Priority::Warning,
                        &format!("temps: cannot list hwmon ({err}); TEMPS stays empty"),
                    );
                }
                self.discover_error = Some(err);
                self.chips.clear();
                self.inputs.clear();
                self.shown.clear();
                return;
            }
        };
        let mut inputs: Vec<Input> = Vec::new();
        for (chip_i, chip) in chips.iter().enumerate() {
            let Ok(entries) = std::fs::read_dir(&chip.dir) else {
                continue;
            };
            let mut indexes: Vec<u32> = entries
                .flatten()
                .filter_map(|e| input_index(e.file_name().to_str()?, "temp"))
                .collect();
            indexes.sort_unstable();
            indexes.dedup();
            let mut seen: Vec<String> = Vec::new();
            for index in indexes {
                if inputs.len() >= MAX_INPUTS {
                    break;
                }
                let input = format!("temp{index}");
                let label = chips::read_text(root, &chip.dir, &format!("temp{index}_label"))
                    .map(|text| printable(&text, MAX_SENSOR))
                    .unwrap_or_default();
                let labelled = !label.trim().is_empty() && !seen.contains(&label);
                let sensor = if labelled { label } else { input.clone() };
                seen.push(sensor.clone());
                let limit = |suffix: &str| {
                    chips::read_text(root, &chip.dir, &format!("temp{index}_{suffix}"))
                        .ok()
                        .and_then(|text| text.parse::<i64>().ok())
                        .and_then(tenths_of_milli)
                        .filter(|t| (200..=1500).contains(t))
                };
                let old = self.inputs.iter().find(|i| {
                    i.index == index && self.chips.get(i.chip).is_some_and(|c| c.chip == chip.chip)
                });
                inputs.push(Input {
                    chip: chip_i,
                    index,
                    sensor,
                    labelled,
                    max_tenths: limit("max"),
                    crit_tenths: limit("crit"),
                    last_raw: old.and_then(|o| o.last_raw),
                    changed_at: old.and_then(|o| o.changed_at),
                    failing: old.is_some_and(|o| o.failing),
                });
            }
        }
        if inputs.len() != self.inputs.len() || chips.len() != self.chips.len() {
            log::emit(
                log,
                Priority::Info,
                &format!(
                    "temps: {} inputs on {} hwmon chips",
                    inputs.len(),
                    chips.len()
                ),
            );
        }
        self.chips = chips;
        self.inputs = inputs;
    }

    fn read_values(&mut self, root: &Path, mono: Instant, log: &mut impl Sink) {
        let mut values: Vec<Option<i32>> = Vec::with_capacity(self.inputs.len());
        for input in &mut self.inputs {
            let chip = &self.chips[input.chip];
            let file = format!("temp{}_input", input.index);
            let raw = chips::read_text(root, &chip.dir, &file).and_then(|text| {
                text.parse::<i64>()
                    .map_err(|_| format!("not a number: {text:.16}"))
            });
            match raw {
                Ok(raw) => {
                    if input.failing {
                        input.failing = false;
                        log::emit(
                            log,
                            Priority::Info,
                            &format!("temps: {} temp{} readable again", chip.chip, input.index),
                        );
                    }
                    if input.last_raw != Some(raw) {
                        input.last_raw = Some(raw);
                        input.changed_at = Some(mono);
                    }
                    // The range is checked on the file's own number, before
                    // rounding: 4999 is out, 126999 in.
                    values.push(
                        tenths_of_milli(raw).filter(|_| (MIN_MILLI..MAX_MILLI).contains(&raw)),
                    );
                }
                Err(why) => {
                    if !input.failing {
                        input.failing = true;
                        log::emit(
                            log,
                            Priority::Warning,
                            &format!(
                                "temps: {} temp{} unreadable ({why}); skipped",
                                chip.chip, input.index
                            ),
                        );
                    }
                    values.push(None);
                }
            }
        }
        let mut shown = Vec::new();
        for (i, (input, value)) in self.inputs.iter().zip(&values).enumerate() {
            let Some(tenths) = *value else {
                continue;
            };
            let chip = &self.chips[input.chip];
            let input_name = format!("temp{}", input.index);
            let names = [input.sensor.as_str(), input_name.as_str()];
            let show = match self.filter.verdict(&chip.chip, &names) {
                Verdict::Blocked | Verdict::NotAllowed => false,
                Verdict::Allowed => true,
                Verdict::Open => {
                    !self.defaults
                        || (self.default_block.verdict(&chip.chip, &names) != Verdict::Blocked
                            && !self.stuck(i, mono))
                }
            };
            if show {
                shown.push(TempReading {
                    chip: chip.chip.clone(),
                    sensor: input.sensor.clone(),
                    input: input_name,
                    labelled: input.labelled,
                    kind: chip.kind,
                    tenths,
                    max_tenths: input.max_tenths,
                    crit_tenths: input.crit_tenths,
                });
            }
        }
        self.shown = shown;
    }

    /// Unchanged for [`STUCK_AFTER`] while another input of the same chip
    /// changed within it.
    fn stuck(&self, i: usize, mono: Instant) -> bool {
        let input = &self.inputs[i];
        let still = |at: Option<Instant>| {
            at.is_some_and(|at| mono.saturating_duration_since(at) >= STUCK_AFTER)
        };
        still(input.changed_at)
            && self.inputs.iter().enumerate().any(|(j, other)| {
                j != i
                    && other.chip == input.chip
                    && !other.failing
                    && other.changed_at.is_some()
                    && !still(other.changed_at)
            })
    }
}

/// The GPU's reading, in tenths, against the same range.
fn plausible(tenths: i32) -> bool {
    (MIN_MILLI / 100..MAX_MILLI / 100).contains(&i64::from(tenths))
}

fn tenths_of_milli(milli: i64) -> Option<i32> {
    // Clamped first: no file's number overflows the conversion.
    let tenths = (milli.clamp(-1_000_000, 1_000_000) as f64 / 100.0).round();
    i32::try_from(tenths as i64).ok()
}

fn tenths_of_c(c: f32) -> Option<i32> {
    c.is_finite()
        .then(|| (f64::from(c) * 10.0).round().clamp(-10_000.0, 10_000.0) as i32)
}

/// Value colour step of one TEMPS item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Level {
    /// Below the warning threshold.
    Normal,
    /// At or above the warning threshold.
    Warn,
    /// At or above the critical threshold.
    Crit,
}

/// One value on a TEMPS row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TempItem {
    /// Short label, possibly empty (the row name says it).
    pub label: String,
    /// Tenths of a degree Celsius.
    pub tenths: i32,
    /// Colour step.
    pub level: Level,
}

/// One TEMPS row: a chip's shown inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TempGroup {
    /// `CPU`, `NVMe0`, `board`, or a `rename`.
    pub name: String,
    /// In input order.
    pub items: Vec<TempItem>,
}

impl TempGroup {
    /// The hottest item, for the one-line summary.
    #[must_use]
    pub fn hottest(&self) -> Option<&TempItem> {
        self.items.iter().max_by_key(|item| item.tenths)
    }
}

/// What the TEMPS panel draws.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TempPanel {
    /// Rows, most important first. Empty draws a note.
    pub groups: Vec<TempGroup>,
}

/// The display half of `[temps]`: renames and thresholds.
#[derive(Clone, Debug, Default)]
pub struct TempLook {
    rename: BTreeMap<String, String>,
    warn: Vec<(Pattern, i32)>,
    crit: Vec<(Pattern, i32)>,
}

impl TempLook {
    /// From a validated `[temps]`.
    #[must_use]
    pub fn new(config: &Temps) -> Self {
        let thresholds = |map: &BTreeMap<String, f64>| {
            let mut out: Vec<(Pattern, i32)> = map
                .iter()
                .filter_map(|(key, c)| Some((Pattern::parse(key)?, (c * 10.0).round() as i32)))
                .collect();
            // The most specific (chip:sensor) pattern is tried first.
            out.sort_by_key(|(p, _)| !p.has_sensor());
            out
        };
        Self {
            rename: config
                .rename
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
                .collect(),
            warn: thresholds(&config.warn),
            crit: thresholds(&config.crit),
        }
    }

    fn renamed(&self, key: &str) -> Option<&str> {
        self.rename
            .get(&key.to_ascii_lowercase())
            .map(String::as_str)
    }

    fn level(&self, reading: &TempReading) -> Level {
        let names = [reading.sensor.as_str(), reading.input.as_str()];
        let find = |list: &[(Pattern, i32)]| {
            list.iter()
                .find(|(p, _)| p.matches(&reading.chip, &names))
                .map(|(_, t)| *t)
        };
        let coolant = reading.kind == Kind::Coolant;
        let crit = find(&self.crit)
            .or(reading.crit_tenths)
            .unwrap_or(if coolant { 550 } else { 900 });
        let warn = find(&self.warn)
            .or(reading.max_tenths.filter(|max| *max < crit))
            .unwrap_or(if coolant { 450 } else { 800 });
        if reading.tenths >= crit {
            Level::Crit
        } else if reading.tenths >= warn {
            Level::Warn
        } else {
            Level::Normal
        }
    }
}

/// Group shown readings into TEMPS rows (module docs).
#[must_use]
pub fn panel(readings: &[TempReading], look: &TempLook) -> TempPanel {
    let mut chips: Vec<(&str, Kind)> = Vec::new();
    for r in readings {
        if !chips.iter().any(|(c, _)| *c == r.chip) {
            chips.push((&r.chip, r.kind));
        }
    }
    let groups = chips
        .iter()
        .map(|(chip, kind)| {
            let same_kind: Vec<&str> = chips
                .iter()
                .filter(|(_, k)| k == kind)
                .map(|(c, _)| *c)
                .collect();
            let index = same_kind.iter().position(|c| c == chip).unwrap_or(0);
            let name = match (look.renamed(chip), kind.word()) {
                (Some(name), _) => name.to_owned(),
                (None, Some(word)) if kind.always_numbered() || same_kind.len() > 1 => {
                    format!("{word}{index}")
                }
                (None, Some(word)) => word.to_owned(),
                (None, None) => chip.chars().take(8).collect(),
            };
            let rows: Vec<&TempReading> = readings.iter().filter(|r| r.chip == *chip).collect();
            let single = rows.len() == 1;
            let items = rows
                .iter()
                .map(|r| {
                    let label = look
                        .renamed(&format!("{}:{}", r.chip, r.sensor))
                        .or_else(|| look.renamed(&format!("{}:{}", r.chip, r.input)))
                        .map_or_else(|| short_label(r, single), str::to_owned);
                    TempItem {
                        label: if label.eq_ignore_ascii_case(&name) {
                            String::new()
                        } else {
                            label
                        },
                        tenths: r.tenths,
                        level: look.level(r),
                    }
                })
                .collect();
            TempGroup { name, items }
        })
        .collect();
    TempPanel { groups }
}

/// `Tccd1` is `CCD1`, `Sensor 2` is `s2`, `AUXTIN3` is `aux3`, `Composite`
/// and a chip's only input are bare; other labels at most 12 characters.
fn short_label(r: &TempReading, single: bool) -> String {
    let label = r.sensor.as_str();
    if single && (r.kind != Kind::Cpu || !r.labelled) {
        return String::new();
    }
    if !r.labelled {
        return format!("t{}", r.input.trim_start_matches("temp"));
    }
    let num = |prefix: &str| {
        label
            .strip_prefix(prefix)
            .map(str::trim)
            .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    match r.kind {
        Kind::Cpu => {
            if let Some(n) = num("Tccd") {
                return format!("CCD{n}");
            }
            if let Some(n) = num("Core ") {
                return format!("c{n}");
            }
            if label.starts_with("Package id") {
                return "pkg".to_owned();
            }
        }
        Kind::Nvme => {
            if label == "Composite" {
                return String::new();
            }
            if let Some(n) = num("Sensor ") {
                return format!("s{n}");
            }
        }
        Kind::Board => match label {
            "SYSTIN" => return "sys".to_owned(),
            "CPUTIN" => return "cpu".to_owned(),
            _ => {
                if let Some(n) = num("AUXTIN") {
                    return format!("aux{n}");
                }
            }
        },
        _ => {}
    }
    label.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millidegrees_round_to_tenths() {
        assert_eq!(tenths_of_milli(50_850), Some(509));
        assert_eq!(tenths_of_milli(-62_000), Some(-620));
        assert_eq!(tenths_of_milli(0), Some(0));
        assert_eq!(tenths_of_c(71.04), Some(710));
        assert_eq!(tenths_of_c(f32::NAN), None);
    }

    #[test]
    fn plausible_range_is_5_to_below_127() {
        assert!(!plausible(0));
        assert!(!plausible(49));
        assert!(plausible(50));
        assert!(plausible(1269));
        assert!(!plausible(1270));
        assert!(!plausible(-620));
    }
}
