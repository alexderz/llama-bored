//! `light.toml`: parse, then validate into a [`LightConfig`].
//!
//! Every mapping lives here, so a colour tweak never needs a rebuild. Every
//! error names the key and says what is wrong. The file path is the CLI
//! argument; the file names no paths.

use std::path::{Path, PathBuf};

use llama_core::color::Rgb;
use serde::Deserialize;

use crate::keyboard::key_index;
use crate::metric::Metric;
use crate::palette::{MAX_PCT, Palette, Scale, parse_hex};

/// Largest config file read.
pub const MAX_CONFIG_BYTES: u64 = 64 * 1024;
/// Most fans on a daisy chain.
pub const MAX_CHAIN: usize = 8;
/// Most LEDs per fan.
pub const MAX_LEDS_PER_FAN: usize = 20;
/// Most `[[light]]` entries.
pub const MAX_LIGHTS: usize = 32;

/// A config that cannot be used. The text is the whole explanation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

fn err<T>(text: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(text.into()))
}

/// How the fans hang on addressable header 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// A splitter: every fan shows the same `leds_per_fan` LEDs.
    Mirrored,
    /// A daisy chain of this many fans; fan k is LEDs `k*n .. (k+1)*n`.
    Chain(usize),
}

/// `[aura]`.
#[derive(Clone, Debug, PartialEq)]
pub struct AuraCfg {
    pub enabled: bool,
    pub leds_per_fan: usize,
    pub layout: Layout,
    /// 0..=100. A current cap applied after every entry's own brightness.
    pub brightness_max: u8,
    /// 1..=20 frames per second.
    pub fps: u8,
}

impl AuraCfg {
    /// LEDs in one frame.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        match self.layout {
            Layout::Mirrored => self.leds_per_fan,
            Layout::Chain(fans) => fans * self.leds_per_fan,
        }
    }
}

/// Which LEDs an entry drives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Target {
    /// Every fan. Each fan is its own span.
    AuraFans,
    /// Chain fans `start..end` (end exclusive) as one span.
    AuraChain { start: usize, end: usize },
    /// Keyboard keys `start..=end` in [`crate::keyboard::KEY_ORDER`]. Parsed and
    /// checked; not drawn until the keyboard protocol lands.
    KeyboardKeys { start: usize, end: usize },
}

/// How the colour is laid on the target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Style {
    /// Every LED the value's colour.
    Solid,
    /// `k` of `N` LEDs lit, from the first LED of the span.
    Gauge,
    /// Every LED, breathing; the breath quickens with the value.
    Pulse,
}

/// One validated `[[light]]` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub target: Target,
    pub metric: Metric,
    pub range: (f32, f32),
    pub scale: Scale,
    pub style: Style,
    pub palette: Palette,
    /// 0..=100.
    pub brightness: u8,
    /// EMA time constant, seconds. 0 is off.
    pub smooth_s: f32,
    /// Shown below `range.min` and when the metric is missing.
    pub idle_color: Option<Rgb>,
}

/// The validated file.
#[derive(Clone, Debug, PartialEq)]
pub struct LightConfig {
    pub aura: AuraCfg,
    pub keyboard_enabled: bool,
    /// Entries in file order; later entries draw over earlier ones.
    pub layers: Vec<Layer>,
}

impl Default for LightConfig {
    fn default() -> Self {
        parse("").unwrap_or_else(|_| unreachable_default())
    }
}

fn unreachable_default() -> LightConfig {
    LightConfig {
        aura: AuraCfg {
            enabled: true,
            leds_per_fan: 6,
            layout: Layout::Mirrored,
            brightness_max: 80,
            fps: 10,
        },
        keyboard_enabled: false,
        layers: Vec::new(),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    #[serde(default)]
    aura: RawAura,
    #[serde(default)]
    keyboard: RawKeyboard,
    #[serde(default)]
    light: Vec<RawLight>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAura {
    enabled: Option<bool>,
    leds_per_fan: Option<i64>,
    fans: Option<String>,
    chain_len: Option<i64>,
    chain: Option<Vec<RawLight>>,
    metric: Option<String>,
    style: Option<String>,
    brightness_max: Option<i64>,
    fps: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawKeyboard {
    enabled: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLight {
    target: Option<String>,
    metric: Option<String>,
    range: Option<Vec<f64>>,
    scale: Option<String>,
    style: Option<String>,
    palette: Option<String>,
    stops: Option<Vec<(StopPos, String)>>,
    brightness: Option<i64>,
    smooth_s: Option<f64>,
    idle_color: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopPos {
    Value(f64),
    Text(String),
}

/// Parse and validate TOML text.
pub fn parse(text: &str) -> Result<LightConfig, ConfigError> {
    let raw: RawFile = toml::from_str(text).map_err(|e| ConfigError(format!("light.toml: {e}")))?;
    validate(raw)
}

fn int_in(
    key: &str,
    value: Option<i64>,
    default: i64,
    low: i64,
    high: i64,
) -> Result<i64, ConfigError> {
    let value = value.unwrap_or(default);
    if !(low..=high).contains(&value) {
        return err(format!(
            "{key} = {value} is out of range; allowed {low}..={high}"
        ));
    }
    Ok(value)
}

fn validate(raw: RawFile) -> Result<LightConfig, ConfigError> {
    let a = raw.aura;
    let leds_per_fan = int_in(
        "aura.leds_per_fan",
        a.leds_per_fan,
        6,
        1,
        MAX_LEDS_PER_FAN as i64,
    )? as usize;
    let brightness_max = int_in("aura.brightness_max", a.brightness_max, 80, 0, 100)? as u8;
    let fps = int_in("aura.fps", a.fps, 10, 1, 20)? as u8;
    let chain_items = a.chain.unwrap_or_default();
    let layout = match (a.fans.as_deref(), chain_items.is_empty()) {
        (None | Some("mirrored"), true) => {
            if a.chain_len.is_some() {
                return err("aura.chain_len is set but aura.fans is not \"chain\"");
            }
            Layout::Mirrored
        }
        (Some("mirrored"), false) => {
            return err(
                "aura.chain is set but aura.fans = \"mirrored\"; a splitter cannot show per-fan values (use fans = \"chain\" or remove aura.chain)",
            );
        }
        (None | Some("chain"), false) => {
            if let Some(len) = a.chain_len
                && len != chain_items.len() as i64
            {
                return err(format!(
                    "aura.chain_len = {len} but aura.chain lists {} fans",
                    chain_items.len()
                ));
            }
            if chain_items.len() > MAX_CHAIN {
                return err(format!(
                    "aura.chain lists {} fans; at most {MAX_CHAIN}",
                    chain_items.len()
                ));
            }
            Layout::Chain(chain_items.len())
        }
        (Some("chain"), true) => {
            let Some(len) = a.chain_len else {
                return err(
                    "aura.fans = \"chain\" needs aura.chain_len (fans on the chain) or an aura.chain list",
                );
            };
            Layout::Chain(int_in("aura.chain_len", Some(len), 1, 1, MAX_CHAIN as i64)? as usize)
        }
        (Some(other), _) => {
            return err(format!(
                "aura.fans = \"{other}\" is not \"mirrored\" or \"chain\""
            ));
        }
    };
    let aura = AuraCfg {
        enabled: a.enabled.unwrap_or(true),
        leds_per_fan,
        layout,
        brightness_max,
        fps,
    };
    let keyboard_enabled = raw.keyboard.enabled.unwrap_or(false);

    let default_metric = match a.metric.as_deref() {
        None => Metric::Activity,
        Some(name) => parse_metric("aura.metric", name)?,
    };
    let default_style = match a.style.as_deref() {
        None => Style::Solid,
        Some(name) => parse_style("aura.style", name)?,
    };

    let mut layers = Vec::new();
    for (index, item) in chain_items.into_iter().enumerate() {
        let key = format!("aura.chain[{index}]");
        if item.target.is_some() {
            return err(format!(
                "{key}.target: a chain entry is fan {index}; remove target"
            ));
        }
        let target = Target::AuraChain {
            start: index,
            end: index + 1,
        };
        layers.push(validate_light(
            &key,
            item,
            Some(target),
            &aura,
            keyboard_enabled,
            default_style,
        )?);
    }
    if raw.light.len() > MAX_LIGHTS {
        return err(format!(
            "{} [[light]] entries; at most {MAX_LIGHTS}",
            raw.light.len()
        ));
    }
    for (index, item) in raw.light.into_iter().enumerate() {
        let key = format!("light[{index}]");
        layers.push(validate_light(
            &key,
            item,
            None,
            &aura,
            keyboard_enabled,
            Style::Solid,
        )?);
    }
    if layers.is_empty() {
        layers.push(Layer {
            target: Target::AuraFans,
            metric: default_metric,
            range: default_metric.default_range(),
            scale: Scale::Linear,
            style: default_style,
            palette: Palette::Act,
            brightness: 100,
            smooth_s: 0.0,
            idle_color: None,
        });
    }
    Ok(LightConfig {
        aura,
        keyboard_enabled,
        layers,
    })
}

fn parse_metric(key: &str, name: &str) -> Result<Metric, ConfigError> {
    if let Some(metric) = Metric::from_name(name) {
        return Ok(metric);
    }
    if name == "ctx" || name.starts_with("ctx") || name.starts_with("slot") {
        return err(format!(
            "{key} = \"{name}\": slot context fill is not in snapshot v1, so llama-light cannot read it yet"
        ));
    }
    let names: Vec<&str> = Metric::ALL.iter().map(|(n, _)| *n).collect();
    err(format!(
        "{key} = \"{name}\" is not a metric; use one of {}",
        names.join(", ")
    ))
}

fn parse_style(key: &str, name: &str) -> Result<Style, ConfigError> {
    match name {
        "solid" => Ok(Style::Solid),
        "ring" | "bar" => Ok(Style::Gauge),
        "pulse" => Ok(Style::Pulse),
        other => err(format!(
            "{key} = \"{other}\" is not a style; use solid, ring, bar or pulse"
        )),
    }
}

fn finite_f32(key: &str, value: f64) -> Result<f32, ConfigError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() {
        return err(format!("{key}: {value} is not a finite number"));
    }
    Ok(narrowed)
}

fn validate_light(
    key: &str,
    raw: RawLight,
    fixed_target: Option<Target>,
    aura: &AuraCfg,
    keyboard_enabled: bool,
    default_style: Style,
) -> Result<Layer, ConfigError> {
    let target = match fixed_target {
        Some(target) => target,
        None => parse_target(
            key,
            raw.target.as_deref().unwrap_or("aura.fans"),
            aura,
            keyboard_enabled,
        )?,
    };
    let metric = match raw.metric.as_deref() {
        None => Metric::Activity,
        Some(name) => parse_metric(&format!("{key}.metric"), name)?,
    };
    let scale = match raw.scale.as_deref() {
        None | Some("linear") => Scale::Linear,
        Some("log") => Scale::Log,
        Some(other) => {
            return err(format!(
                "{key}.scale = \"{other}\" is not \"linear\" or \"log\""
            ));
        }
    };
    let range = match raw.range {
        None => metric.default_range(),
        Some(values) => {
            if values.len() != 2 {
                return err(format!(
                    "{key}.range must be [min, max], got {} numbers",
                    values.len()
                ));
            }
            let min = finite_f32(&format!("{key}.range min"), values[0])?;
            let max = finite_f32(&format!("{key}.range max"), values[1])?;
            if min >= max {
                return err(format!("{key}.range: min {min} must be below max {max}"));
            }
            (min, max)
        }
    };
    if scale == Scale::Log && range.0 <= 0.0 {
        return err(format!(
            "{key}: scale = \"log\" needs range min above 0 (got {}); try range = [1, {}]",
            range.0, range.1
        ));
    }
    let style = match raw.style.as_deref() {
        None => default_style,
        Some(name) => parse_style(&format!("{key}.style"), name)?,
    };
    let palette = match (raw.palette.as_deref(), raw.stops) {
        (Some(_), Some(_)) => {
            return err(format!("{key}: set palette or stops, not both"));
        }
        (None, None) => Palette::Act,
        (Some(name), None) => Palette::named(name).ok_or_else(|| {
            ConfigError(format!(
                "{key}.palette = \"{name}\" is not a palette; use act, thermal or mono, or give stops"
            ))
        })?,
        (None, Some(stops)) => parse_stops(key, stops, range, scale)?,
    };
    let brightness = int_in(&format!("{key}.brightness"), raw.brightness, 100, 0, 100)? as u8;
    let smooth_s = match raw.smooth_s {
        None => 0.0,
        Some(value) => {
            let value = finite_f32(&format!("{key}.smooth_s"), value)?;
            if !(0.0..=60.0).contains(&value) {
                return err(format!(
                    "{key}.smooth_s = {value} is out of range; allowed 0..=60"
                ));
            }
            value
        }
    };
    let idle_color = match raw.idle_color.as_deref() {
        None => None,
        Some(text) => Some(parse_hex(text).ok_or_else(|| {
            ConfigError(format!(
                "{key}.idle_color = \"{text}\" is not a #RRGGBB colour"
            ))
        })?),
    };
    Ok(Layer {
        target,
        metric,
        range,
        scale,
        style,
        palette,
        brightness,
        smooth_s,
        idle_color,
    })
}

fn parse_stops(
    key: &str,
    stops: Vec<(StopPos, String)>,
    range: (f32, f32),
    scale: Scale,
) -> Result<Palette, ConfigError> {
    if stops.len() < 2 {
        return err(format!(
            "{key}.stops needs at least 2 stops, got {}",
            stops.len()
        ));
    }
    let mut out: Vec<(f32, Rgb)> = Vec::with_capacity(stops.len());
    for (index, (pos, color)) in stops.into_iter().enumerate() {
        let at = format!("{key}.stops[{index}]");
        let pct = match pos {
            StopPos::Value(value) => {
                let value = finite_f32(&at, value)?;
                if scale == Scale::Log && value <= 0.0 {
                    return err(format!(
                        "{at}: position {value} must be above 0 on a log scale"
                    ));
                }
                raw_percent(value, range, scale)
            }
            StopPos::Text(text) => {
                let Some(number) = text.trim().strip_suffix('%') else {
                    return err(format!(
                        "{at}: position \"{text}\" must be a number (metric units) or \"N%\" (percent of range)"
                    ));
                };
                let pct: f32 = number
                    .trim()
                    .parse()
                    .map_err(|_| ConfigError(format!("{at}: \"{text}\" is not a percent")))?;
                if !pct.is_finite() {
                    return err(format!("{at}: \"{text}\" is not a finite percent"));
                }
                pct
            }
        };
        if !(0.0..=MAX_PCT).contains(&pct) {
            return err(format!(
                "{at}: position is {pct:.1}% of the range; stops must sit between 0% and {MAX_PCT}% (range min to 1.25 × range)"
            ));
        }
        let rgb = parse_hex(&color)
            .ok_or_else(|| ConfigError(format!("{at}: \"{color}\" is not a #RRGGBB colour")))?;
        if let Some((last, _)) = out.last()
            && pct <= *last
        {
            return err(format!(
                "{at}: stops must ascend; this one is at {pct:.1}% but the one before is at {last:.1}%"
            ));
        }
        out.push((pct, rgb));
    }
    Ok(Palette::Stops(out))
}

/// Percent of range without the clamp, so an out-of-range stop is an error.
fn raw_percent(value: f32, range: (f32, f32), scale: Scale) -> f32 {
    let (min, max) = range;
    let fraction = match scale {
        Scale::Linear => (value - min) / (max - min),
        Scale::Log => (value / min).ln() / (max / min).ln(),
    };
    fraction * 100.0
}

fn parse_target(
    key: &str,
    text: &str,
    aura: &AuraCfg,
    keyboard_enabled: bool,
) -> Result<Target, ConfigError> {
    let at = format!("{key}.target = \"{text}\"");
    if text == "aura.fans" {
        return Ok(Target::AuraFans);
    }
    if let Some(inner) = text
        .strip_prefix("aura.chain[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        let Layout::Chain(fans) = aura.layout else {
            return err(format!(
                "{at}: needs aura.fans = \"chain\"; mirrored fans all show aura.fans"
            ));
        };
        let parse_index = |part: &str| -> Result<usize, ConfigError> {
            part.trim()
                .parse::<usize>()
                .map_err(|_| ConfigError(format!("{at}: \"{part}\" is not a fan index")))
        };
        let (start, end) = if let Some((a, b)) = inner.split_once("..=") {
            (parse_index(a)?, parse_index(b)? + 1)
        } else if let Some((a, b)) = inner.split_once("..") {
            (parse_index(a)?, parse_index(b)?)
        } else {
            let index = parse_index(inner)?;
            (index, index + 1)
        };
        if start >= end {
            return err(format!("{at}: the range is empty"));
        }
        if end > fans {
            return err(format!(
                "{at}: the chain has {fans} fans (indexes 0..{}), so fan {} does not exist",
                fans - 1,
                end - 1
            ));
        }
        return Ok(Target::AuraChain { start, end });
    }
    if let Some(inner) = text
        .strip_prefix("keyboard.keys[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        if !keyboard_enabled {
            return err(format!(
                "{at}: [keyboard] enabled = true is required for keyboard targets"
            ));
        }
        let key_of = |part: &str| -> Result<usize, ConfigError> {
            let name = part
                .trim()
                .strip_prefix('"')
                .and_then(|rest| rest.strip_suffix('"'))
                .ok_or_else(|| {
                    ConfigError(format!("{at}: key names are quoted, like keys[\"F1\"]"))
                })?;
            key_index(name)
                .ok_or_else(|| ConfigError(format!("{at}: \"{name}\" is not a known key")))
        };
        let (start, end) = match inner.split_once("..") {
            Some((a, b)) => (key_of(a)?, key_of(b.trim_start_matches('='))?),
            None => {
                let index = key_of(inner)?;
                (index, index)
            }
        };
        if start > end {
            return err(format!("{at}: the first key comes after the last"));
        }
        return Ok(Target::KeyboardKeys { start, end });
    }
    err(format!(
        "{at} is not a target; use aura.fans, aura.chain[N], aura.chain[A..B] or keyboard.keys[\"F1\"..\"F12\"]"
    ))
}

/// Identity of the file on disk, for the reload check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stamp {
    mtime_ns: i128,
    size: u64,
    ino: u64,
}

impl Stamp {
    /// A stamp from its parts (tests fake a file with these).
    #[must_use]
    pub fn from_parts(mtime_ns: i128, size: u64, ino: u64) -> Self {
        Self {
            mtime_ns,
            size,
            ino,
        }
    }
}

/// Where the service gets its config. Tests supply a fake.
pub trait ConfigSource {
    /// The file's identity now, or `None` when it cannot be stat'ed.
    fn stamp(&self) -> Option<Stamp>;
    /// Read and validate the file.
    fn load(&self) -> Result<LightConfig, ConfigError>;
}

/// The config file named on the command line.
pub struct ConfigFile {
    path: PathBuf,
}

impl ConfigFile {
    /// `path` is the `--config` argument.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl ConfigSource for ConfigFile {
    fn stamp(&self) -> Option<Stamp> {
        stamp_of(&self.path)
    }

    fn load(&self) -> Result<LightConfig, ConfigError> {
        load(&self.path)
    }
}

/// Stat `path`.
#[must_use]
pub fn stamp_of(path: &Path) -> Option<Stamp> {
    let stat = rustix::fs::stat(path).ok()?;
    Some(Stamp {
        mtime_ns: i128::from(stat.st_mtime) * 1_000_000_000 + i128::from(stat.st_mtime_nsec),
        size: u64::try_from(stat.st_size).unwrap_or(0),
        ino: stat.st_ino,
    })
}

/// Read and validate `path`.
pub fn load(path: &Path) -> Result<LightConfig, ConfigError> {
    let meta = std::fs::metadata(path)
        .map_err(|e| ConfigError(format!("{}: {}", path.display(), e.kind())))?;
    if !meta.is_file() {
        return err(format!("{}: not a regular file", path.display()));
    }
    if meta.len() > MAX_CONFIG_BYTES {
        return err(format!(
            "{}: larger than {MAX_CONFIG_BYTES} bytes",
            path.display()
        ));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError(format!("{}: {}", path.display(), e.kind())))?;
    parse(&text)
}
