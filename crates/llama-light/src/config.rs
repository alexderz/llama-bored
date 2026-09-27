//! `light.toml`: parse, then validate into a [`LightConfig`].
//!
//! Every mapping lives here, so a colour tweak never needs a rebuild. Every
//! error names the key and says what is wrong. The file path is the CLI
//! argument; the file names no paths.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use llama_core::color::Rgb;
use serde::Deserialize;

use crate::keyboard::keymap::{KEYS, key_index, suggest};
use crate::metric::Metric;
use crate::palette::{MAX_PCT, Palette, Scale, dim, parse_hex};

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

/// `[keyboard]`.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyboardCfg {
    pub enabled: bool,
    /// 0..=100. Applied after every entry's own brightness.
    pub brightness_max: u8,
}

/// `[engine]`: the frame pipeline.
///
/// Every tick samples the snapshot and advances smoothing. Every
/// `target_period_s` the LED targets are recomputed; the frame shown then
/// moves from the last frame shown to the new target in a straight line
/// over `tween_s`. Without an `[engine]` section every tick is a target and
/// there is no tween (the frame is the target).
#[derive(Clone, Debug, PartialEq)]
pub struct EngineCfg {
    /// Ticks per second: the loop rate. 1..=20; default `aura.fps`.
    pub tick_hz: u8,
    /// Seconds between target recomputes. 0: every tick.
    pub target_period_s: f32,
    /// Seconds from the last frame shown to a new target. 0: jump.
    pub tween_s: f32,
    /// Most keyboard frames written per second. 1..=20, at most `tick_hz`.
    /// The fans keep `aura.fps`.
    pub tween_fps: u8,
}

/// Which LEDs an entry drives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Target {
    /// Every fan. Each fan is its own span.
    AuraFans,
    /// Chain fans `start..end` (end exclusive) as one span.
    AuraChain { start: usize, end: usize },
    /// Keyboard keys as one span, in the order given: indexes into
    /// [`crate::keyboard::keymap::KEYS`]. A gauge lights them in this order.
    KeyboardKeys(Vec<usize>),
    /// Every named keyboard key, as one span in visual order.
    KeyboardAll,
}

impl Target {
    /// Whether this target is on the keyboard.
    #[must_use]
    pub fn is_keyboard(&self) -> bool {
        matches!(self, Self::KeyboardKeys(_) | Self::KeyboardAll)
    }
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
    /// One rung per LED, in the order given, each with its own threshold.
    Ladder,
    /// On while the raw value was above `threshold` within `hold_s`.
    Gate,
    /// The highest value of the last `peak_s`, then a linear decay.
    Peak,
}

impl Style {
    /// The config name (`bar` for a gauge).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Style::Solid => "solid",
            Style::Gauge => "bar",
            Style::Pulse => "pulse",
            Style::Ladder => "ladder",
            Style::Gate => "gate",
            Style::Peak => "peak",
        }
    }
}

/// Where a bar or ladder ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Edge {
    /// Whole LEDs: rounded, any value above the start lights one.
    Round,
    /// The last LED is lit by the fraction filled, blended from what is
    /// under it to the lit colour.
    Fractional,
}

/// How a bar or ladder picks the colour of its lit LEDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gradient {
    /// Every lit LED the colour of the current value.
    Value,
    /// LED `i` of `n` the colour of its own place: `(i + 0.5) / n` of the
    /// bar (a bar), or its own threshold (a ladder).
    Position,
}

/// `shimmer = { depth, hz }`: brightness × (1 ± depth) at `hz`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shimmer {
    /// 0..=1.
    pub depth: f32,
    /// 0..=10.
    pub hz: f32,
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
    /// 0..=100 percent. With `brightness_to`, the brightness at 0 % of range.
    pub brightness: f32,
    /// `brightness = [lo, hi]`: the brightness at 100 % of range (solid).
    pub brightness_to: Option<f32>,
    /// EMA time constant while the value falls (release), seconds. 0 is off.
    pub smooth_s: f32,
    /// EMA time constant while the value rises (attack), seconds. Defaults
    /// to `smooth_s`.
    pub attack_s: f32,
    /// Counter metrics: seconds of counter history per rate. 0 is the raw
    /// rate between consecutive snapshots.
    pub rate_window_s: f32,
    /// Bar and ladder end.
    pub edge: Edge,
    /// Bar and ladder colours.
    pub gradient: Gradient,
    /// Bar: the percent of range the full bar stands for (100..=125).
    pub fill_to: f32,
    /// Ladder: the value at which each rung is fully lit, one per LED,
    /// ascending, in metric units.
    pub thresholds: Vec<f32>,
    /// Solid, pulse and peak: below this (metric units) the entry draws
    /// nothing. Gate: the raw value must be above this to open.
    pub threshold: Option<f32>,
    /// Gate: seconds it stays open after the value was last above threshold.
    pub hold_s: f32,
    /// Gate: brightness shimmer while open.
    pub shimmer: Option<Shimmer>,
    /// Peak: seconds the peak is held before it decays, and the decay time
    /// across the whole range.
    pub peak_s: f32,
    /// Shown below `range.min` and when the metric is missing.
    pub idle_color: Option<Rgb>,
    /// `color`: a fixed colour on every LED of the target; the metric is
    /// not read.
    pub fixed: Option<Rgb>,
    /// Gate: the colour it shows while open.
    pub gate_color: Option<Rgb>,
}

/// The validated file.
#[derive(Clone, Debug, PartialEq)]
pub struct LightConfig {
    pub aura: AuraCfg,
    pub keyboard: KeyboardCfg,
    pub engine: EngineCfg,
    /// `[base]`: under every entry, so every LED no entry lights shows it
    /// (brightness applied). `None`: black.
    pub base: Option<Rgb>,
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
        keyboard: KeyboardCfg {
            enabled: false,
            brightness_max: 100,
        },
        engine: EngineCfg {
            tick_hz: 10,
            target_period_s: 0.0,
            tween_s: 0.0,
            tween_fps: 10,
        },
        base: None,
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
    engine: Option<RawEngine>,
    base: Option<RawBase>,
    #[serde(default)]
    palette: BTreeMap<String, RawPalette>,
    #[serde(default)]
    light: Vec<RawLight>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEngine {
    tick_hz: Option<i64>,
    target_hz: Option<f64>,
    tween_fps: Option<i64>,
    tween_s: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBase {
    color: Option<String>,
    brightness: Option<Num>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPalette {
    stops: Vec<(StopPos, String)>,
}

/// A TOML number: an integer is a whole percent, a float a fraction.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
enum Num {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawBrightness {
    One(Num),
    Span(Vec<Num>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawTarget {
    One(String),
    List(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawShimmer {
    depth: f64,
    hz: f64,
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
    brightness_max: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLight {
    target: Option<RawTarget>,
    metric: Option<String>,
    range: Option<Vec<f64>>,
    scale: Option<String>,
    style: Option<String>,
    palette: Option<String>,
    stops: Option<Vec<(StopPos, String)>>,
    brightness: Option<RawBrightness>,
    smooth_s: Option<f64>,
    attack_s: Option<f64>,
    rate_window_s: Option<f64>,
    edge: Option<String>,
    gradient: Option<String>,
    fill_to: Option<f64>,
    thresholds: Option<Vec<f64>>,
    threshold: Option<f64>,
    hold_s: Option<f64>,
    shimmer: Option<RawShimmer>,
    peak_s: Option<f64>,
    idle_color: Option<String>,
    color: Option<String>,
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
    let keyboard = KeyboardCfg {
        enabled: raw.keyboard.enabled.unwrap_or(false),
        brightness_max: int_in(
            "keyboard.brightness_max",
            raw.keyboard.brightness_max,
            100,
            0,
            100,
        )? as u8,
    };
    let keyboard_enabled = keyboard.enabled;
    let engine = validate_engine(raw.engine, aura.fps)?;
    let base = match raw.base {
        None => None,
        Some(base) => Some(validate_base(base)?),
    };
    let mut palettes = BTreeMap::new();
    for (name, palette) in raw.palette {
        let key = format!("palette.{name}");
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return err(format!(
                "[{key}]: a palette name is letters, digits, _ and - only"
            ));
        }
        palettes.insert(name, parse_percent_stops(&key, palette.stops)?);
    }
    let ctx = Ctx {
        aura: &aura,
        keyboard_enabled,
        palettes: &palettes,
    };

    let default_metric = match a.metric.as_deref() {
        None => Metric::Activity,
        Some(name) => parse_metric("aura.metric", name)?,
    };
    let default_style = match a.style.as_deref() {
        None => Style::Solid,
        Some(name) => match parse_style("aura.style", name)? {
            style @ (Style::Solid | Style::Gauge | Style::Pulse) => style,
            other => {
                return err(format!(
                    "aura.style = \"{}\": the default entry is solid, ring, bar or pulse; write a [[light]] entry for style = \"{}\"",
                    other.name(),
                    other.name()
                ));
            }
        },
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
            &ctx,
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
        layers.push(validate_light(&key, item, None, &ctx, Style::Solid)?);
    }
    // The fans keep their default when every entry is on the keyboard.
    if !layers.iter().any(|layer| !layer.target.is_keyboard()) {
        let mut fans = Layer::plain(Target::AuraFans, default_metric, default_style);
        if let Some(act) = palettes.get("act") {
            fans.palette = act.clone();
        }
        layers.push(fans);
    }
    Ok(LightConfig {
        aura,
        keyboard,
        engine,
        base,
        layers,
    })
}

impl Layer {
    /// An entry with every option at its default.
    #[must_use]
    pub fn plain(target: Target, metric: Metric, style: Style) -> Self {
        Self {
            target,
            metric,
            range: metric.default_range(),
            scale: Scale::Linear,
            style,
            palette: Palette::Act,
            brightness: 100.0,
            brightness_to: None,
            smooth_s: 0.0,
            attack_s: 0.0,
            rate_window_s: 0.0,
            edge: Edge::Round,
            gradient: Gradient::Value,
            fill_to: 100.0,
            thresholds: Vec::new(),
            threshold: None,
            hold_s: DEFAULT_HOLD_S,
            shimmer: None,
            peak_s: DEFAULT_PEAK_S,
            idle_color: None,
            fixed: None,
            gate_color: None,
        }
    }
}

/// Gate: default `hold_s`.
pub const DEFAULT_HOLD_S: f32 = 1.0;
/// Peak: default `peak_s`.
pub const DEFAULT_PEAK_S: f32 = 30.0;
/// Most seconds for any time constant, hold, window or peak.
pub const MAX_SECONDS: f32 = 60.0;

struct Ctx<'a> {
    aura: &'a AuraCfg,
    keyboard_enabled: bool,
    palettes: &'a BTreeMap<String, Palette>,
}

fn validate_engine(raw: Option<RawEngine>, aura_fps: u8) -> Result<EngineCfg, ConfigError> {
    let Some(raw) = raw else {
        return Ok(EngineCfg {
            tick_hz: aura_fps,
            target_period_s: 0.0,
            tween_s: 0.0,
            tween_fps: aura_fps,
        });
    };
    let tween_fps = match raw.tween_fps {
        None => None,
        Some(value) => Some(int_in("engine.tween_fps", Some(value), 10, 1, 20)? as u8),
    };
    let tick_hz = match raw.tick_hz {
        None => aura_fps.max(tween_fps.unwrap_or(0)),
        Some(value) => int_in("engine.tick_hz", Some(value), 10, 1, 20)? as u8,
    };
    let tween_fps = tween_fps.unwrap_or(tick_hz);
    if tween_fps > tick_hz {
        return err(format!(
            "engine.tween_fps = {tween_fps} is above engine.tick_hz = {tick_hz}; a frame is made once per tick, so raise tick_hz or lower tween_fps"
        ));
    }
    let tween_s = match raw.tween_s {
        None => None,
        Some(value) => Some(seconds("engine.tween_s", value, 10.0)?),
    };
    let target_period_s = match raw.target_hz {
        None => tween_s.unwrap_or(0.0),
        Some(hz) => {
            let hz = finite_f32("engine.target_hz", hz)?;
            if !(hz > 0.0 && hz <= f32::from(tick_hz)) {
                return err(format!(
                    "engine.target_hz = {hz} is out of range; allowed above 0 up to engine.tick_hz ({tick_hz})"
                ));
            }
            1.0 / hz
        }
    };
    let tween_s = tween_s.unwrap_or(target_period_s);
    if tween_s > target_period_s + 1e-4 {
        return err(format!(
            "engine.tween_s = {tween_s} is longer than one target period ({target_period_s} s = 1 / target_hz); a tween must end by the next target"
        ));
    }
    Ok(EngineCfg {
        tick_hz,
        target_period_s,
        tween_s,
        tween_fps,
    })
}

fn validate_base(raw: RawBase) -> Result<Rgb, ConfigError> {
    let Some(text) = raw.color else {
        return err("[base] needs color = \"#RRGGBB\"");
    };
    let color = parse_hex(&text)
        .ok_or_else(|| ConfigError(format!("base.color = \"{text}\" is not a #RRGGBB colour")))?;
    let brightness = match raw.brightness {
        None => 100.0,
        Some(value) => brightness_pct("base.brightness", value)?,
    };
    Ok(dim(color, brightness))
}

/// A brightness as a percent: an integer is 0..=100 %, a float 0.0..=1.0.
fn brightness_pct(key: &str, value: Num) -> Result<f32, ConfigError> {
    match value {
        Num::Int(value) => Ok(int_in(key, Some(value), 100, 0, 100)? as f32),
        Num::Float(value) => {
            let value = finite_f32(key, value)?;
            if !(0.0..=1.0).contains(&value) {
                return err(format!(
                    "{key} = {value} is out of range; a fraction is 0.0..=1.0 (or a whole percent 0..=100)"
                ));
            }
            Ok(value * 100.0)
        }
    }
}

/// Seconds, finite, `0..=max`.
fn seconds(key: &str, value: f64, max: f32) -> Result<f32, ConfigError> {
    let value = finite_f32(key, value)?;
    if !(0.0..=max).contains(&value) {
        return err(format!(
            "{key} = {value} is out of range; allowed 0..={max}"
        ));
    }
    Ok(value)
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
        "ladder" => Ok(Style::Ladder),
        "gate" => Ok(Style::Gate),
        "peak" => Ok(Style::Peak),
        other => err(format!(
            "{key} = \"{other}\" is not a style; use solid, ring, bar, pulse, ladder, gate or peak"
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
    ctx: &Ctx<'_>,
    default_style: Style,
) -> Result<Layer, ConfigError> {
    let target = match (fixed_target, raw.target) {
        (Some(target), _) => target,
        (None, None) => Target::AuraFans,
        (None, Some(RawTarget::One(text))) => parse_target(key, &text, ctx)?,
        (None, Some(RawTarget::List(names))) => parse_target_list(key, &names, ctx)?,
    };
    let style = match raw.style.as_deref() {
        None => default_style,
        Some(name) => parse_style(&format!("{key}.style"), name)?,
    };
    // Keys that belong to one style (or a few) only.
    let only = |name: &str, set: bool, styles: &[Style]| -> Result<(), ConfigError> {
        if set && !styles.contains(&style) {
            let wanted: Vec<String> = styles.iter().map(|s| format!("\"{}\"", s.name())).collect();
            return err(format!(
                "{key}.{name} is for style = {} (this entry is \"{}\")",
                wanted.join(" or "),
                style.name()
            ));
        }
        Ok(())
    };
    only("edge", raw.edge.is_some(), &[Style::Gauge, Style::Ladder])?;
    only(
        "gradient",
        raw.gradient.is_some(),
        &[Style::Gauge, Style::Ladder],
    )?;
    only("fill_to", raw.fill_to.is_some(), &[Style::Gauge])?;
    only("thresholds", raw.thresholds.is_some(), &[Style::Ladder])?;
    only("hold_s", raw.hold_s.is_some(), &[Style::Gate])?;
    only("shimmer", raw.shimmer.is_some(), &[Style::Gate])?;
    only("peak_s", raw.peak_s.is_some(), &[Style::Peak])?;
    only(
        "threshold",
        raw.threshold.is_some(),
        &[Style::Solid, Style::Pulse, Style::Gate, Style::Peak],
    )?;
    only(
        "brightness = [lo, hi]",
        matches!(raw.brightness, Some(RawBrightness::Span(_))),
        &[Style::Solid],
    )?;

    let mut fixed = None;
    let mut gate_color = None;
    if let Some(text) = raw.color.as_deref() {
        let color = parse_hex(text).ok_or_else(|| {
            ConfigError(format!("{key}.color = \"{text}\" is not a #RRGGBB colour"))
        })?;
        if style == Style::Gate {
            let set: Vec<&str> = [
                ("palette", raw.palette.is_some()),
                ("stops", raw.stops.is_some()),
                ("idle_color", raw.idle_color.is_some()),
            ]
            .into_iter()
            .filter_map(|(name, on)| on.then_some(name))
            .collect();
            if !set.is_empty() {
                return err(format!(
                    "{key}: a gate shows color; remove {}",
                    set.join(", ")
                ));
            }
            gate_color = Some(color);
        } else {
            let set: Vec<&str> = [
                ("metric", raw.metric.is_some()),
                ("range", raw.range.is_some()),
                ("scale", raw.scale.is_some()),
                ("palette", raw.palette.is_some()),
                ("stops", raw.stops.is_some()),
                ("smooth_s", raw.smooth_s.is_some()),
                ("attack_s", raw.attack_s.is_some()),
                ("rate_window_s", raw.rate_window_s.is_some()),
                ("threshold", raw.threshold.is_some()),
                ("idle_color", raw.idle_color.is_some()),
                ("style", raw.style.is_some() && style != Style::Solid),
                (
                    "brightness = [lo, hi]",
                    matches!(raw.brightness, Some(RawBrightness::Span(_))),
                ),
            ]
            .into_iter()
            .filter_map(|(name, on)| on.then_some(name))
            .collect();
            if !set.is_empty() {
                return err(format!(
                    "{key}: color is a fixed colour; remove {} (or remove color)",
                    set.join(", ")
                ));
            }
            fixed = Some(color);
        }
    } else if style == Style::Gate {
        return err(format!(
            "{key}: style = \"gate\" needs color = \"#RRGGBB\" (the colour while open)"
        ));
    }

    // `decoded_total` is the raw counter: a gate opens on any increase.
    let raw_counter = raw.metric.as_deref() == Some("decoded_total");
    let metric = match raw.metric.as_deref() {
        None => Metric::Activity,
        Some("decoded_total") => {
            if style != Style::Gate {
                return err(format!(
                    "{key}.metric = \"decoded_total\" is the raw token counter; use it with style = \"gate\", or use tokens_rate (with rate_window_s) for tokens/s"
                ));
            }
            Metric::TokensRate
        }
        Some(name) => parse_metric(&format!("{key}.metric"), name)?,
    };
    let rate_window_s = match raw.rate_window_s {
        None => 0.0,
        Some(value) => {
            if raw_counter {
                return err(format!(
                    "{key}.rate_window_s: decoded_total is read raw (any increase opens the gate); remove rate_window_s"
                ));
            }
            if !metric.is_counter_rate() {
                return err(format!(
                    "{key}.rate_window_s: only a counter rate (tokens_rate) has a rate window; {} is read as it is",
                    metric.name()
                ));
            }
            seconds(&format!("{key}.rate_window_s"), value, MAX_SECONDS)?
        }
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
    let palette = match (raw.palette.as_deref(), raw.stops) {
        (Some(_), Some(_)) => {
            return err(format!("{key}: set palette or stops, not both"));
        }
        (None, None) => ctx.palettes.get("act").cloned().unwrap_or(Palette::Act),
        (Some(name), None) => match ctx.palettes.get(name) {
            Some(palette) => palette.clone(),
            None => Palette::named(name).ok_or_else(|| {
                let own: Vec<&str> = ctx.palettes.keys().map(String::as_str).collect();
                let own = if own.is_empty() {
                    String::new()
                } else {
                    format!(", {}", own.join(", "))
                };
                ConfigError(format!(
                    "{key}.palette = \"{name}\" is not a palette; use act, thermal, mono{own}, or give stops"
                ))
            })?,
        },
        (None, Some(stops)) => parse_stops(key, stops, range, scale)?,
    };
    let (brightness, brightness_to) = match raw.brightness {
        None => (100.0, None),
        Some(RawBrightness::One(value)) => {
            (brightness_pct(&format!("{key}.brightness"), value)?, None)
        }
        Some(RawBrightness::Span(values)) => {
            if values.len() != 2 {
                return err(format!(
                    "{key}.brightness must be one number or [lo, hi], got {} numbers",
                    values.len()
                ));
            }
            let lo = brightness_pct(&format!("{key}.brightness lo"), values[0])?;
            let hi = brightness_pct(&format!("{key}.brightness hi"), values[1])?;
            (lo, Some(hi))
        }
    };
    let smooth_s = match raw.smooth_s {
        None => 0.0,
        Some(value) => seconds(&format!("{key}.smooth_s"), value, MAX_SECONDS)?,
    };
    let attack_s = match raw.attack_s {
        None => smooth_s,
        Some(value) => seconds(&format!("{key}.attack_s"), value, MAX_SECONDS)?,
    };
    let edge = match raw.edge.as_deref() {
        None | Some("round") => Edge::Round,
        Some("fractional") => Edge::Fractional,
        Some(other) => {
            return err(format!(
                "{key}.edge = \"{other}\" is not \"round\" or \"fractional\""
            ));
        }
    };
    let gradient = match raw.gradient.as_deref() {
        None | Some("value") => Gradient::Value,
        Some("position") => Gradient::Position,
        Some(other) => {
            return err(format!(
                "{key}.gradient = \"{other}\" is not \"value\" or \"position\""
            ));
        }
    };
    let fill_to = match raw.fill_to {
        None => 100.0,
        Some(value) => {
            let value = finite_f32(&format!("{key}.fill_to"), value)?;
            if !(100.0..=MAX_PCT).contains(&value) {
                return err(format!(
                    "{key}.fill_to = {value} is out of range; allowed 100..={MAX_PCT} (percent of range the full bar stands for)"
                ));
            }
            value
        }
    };
    let span = span_len(&target, ctx.aura);
    let thresholds = if style == Style::Ladder {
        match raw.thresholds {
            None => (0..span)
                .map(|k| range.0 + (range.1 - range.0) * (k + 1) as f32 / span as f32)
                .collect(),
            Some(values) => {
                if values.len() != span {
                    return err(format!(
                        "{key}.thresholds has {} values but the target has {span} LEDs; give one threshold per rung",
                        values.len()
                    ));
                }
                let mut out: Vec<f32> = Vec::with_capacity(values.len());
                for (index, value) in values.into_iter().enumerate() {
                    let value = finite_f32(&format!("{key}.thresholds[{index}]"), value)?;
                    if value <= range.0 {
                        return err(format!(
                            "{key}.thresholds[{index}] = {value} must be above range min {}",
                            range.0
                        ));
                    }
                    if let Some(last) = out.last()
                        && value <= *last
                    {
                        return err(format!(
                            "{key}.thresholds must ascend; [{index}] = {value} is not above [{}] = {last}",
                            index - 1
                        ));
                    }
                    out.push(value);
                }
                out
            }
        }
    } else {
        Vec::new()
    };
    let threshold = match raw.threshold {
        None => None,
        Some(value) => Some(finite_f32(&format!("{key}.threshold"), value)?),
    };
    let hold_s = match raw.hold_s {
        None => DEFAULT_HOLD_S,
        Some(value) => seconds(&format!("{key}.hold_s"), value, MAX_SECONDS)?,
    };
    let shimmer = match raw.shimmer {
        None => None,
        Some(raw) => {
            let depth = finite_f32(&format!("{key}.shimmer.depth"), raw.depth)?;
            if !(0.0..=1.0).contains(&depth) {
                return err(format!(
                    "{key}.shimmer.depth = {depth} is out of range; allowed 0..=1 (a fraction of the brightness)"
                ));
            }
            let hz = finite_f32(&format!("{key}.shimmer.hz"), raw.hz)?;
            if !(0.0..=10.0).contains(&hz) {
                return err(format!(
                    "{key}.shimmer.hz = {hz} is out of range; allowed 0..=10"
                ));
            }
            Some(Shimmer { depth, hz })
        }
    };
    let peak_s = match raw.peak_s {
        None => DEFAULT_PEAK_S,
        Some(value) => {
            let value = seconds(&format!("{key}.peak_s"), value, MAX_SECONDS)?;
            if value <= 0.0 {
                return err(format!("{key}.peak_s must be above 0"));
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
        brightness_to,
        smooth_s,
        attack_s,
        rate_window_s,
        edge,
        gradient,
        fill_to,
        thresholds,
        threshold,
        hold_s,
        shimmer,
        peak_s,
        idle_color,
        fixed,
        gate_color,
    })
}

/// LEDs in one span of `target` (every span of a target is this long).
fn span_len(target: &Target, aura: &AuraCfg) -> usize {
    match target {
        Target::AuraFans => aura.leds_per_fan,
        Target::AuraChain { start, end } => (end - start) * aura.leds_per_fan,
        Target::KeyboardKeys(keys) => keys.len(),
        Target::KeyboardAll => KEYS.len(),
    }
}

/// `[palette.NAME] stops`: positions are percent of range (a number or
/// `"N%"`), 0..=125, ascending.
fn parse_percent_stops(key: &str, stops: Vec<(StopPos, String)>) -> Result<Palette, ConfigError> {
    let stops = stops
        .into_iter()
        .map(|(pos, color)| {
            let pos = match pos {
                StopPos::Value(value) => StopPos::Text(format!("{value}%")),
                text => text,
            };
            (pos, color)
        })
        .collect();
    parse_stops(key, stops, (0.0, 100.0), Scale::Linear)
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

fn parse_target(key: &str, text: &str, ctx: &Ctx<'_>) -> Result<Target, ConfigError> {
    let aura = ctx.aura;
    let keyboard_enabled = ctx.keyboard_enabled;
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
    if text == "keyboard.all" || text.starts_with("keyboard.") {
        if !keyboard_enabled {
            return err(format!(
                "{at}: [keyboard] enabled = true is required for keyboard targets"
            ));
        }
        if text == "keyboard.all" {
            return Ok(Target::KeyboardAll);
        }
        if let Some(inner) = text
            .strip_prefix("keyboard.keys[")
            .and_then(|rest| rest.strip_suffix(']'))
        {
            return parse_keys(&at, inner).map(Target::KeyboardKeys);
        }
    }
    if text.starts_with("led:") || key_index(text).is_some() {
        if !keyboard_enabled {
            return err(format!(
                "{at}: [keyboard] enabled = true is required for keyboard targets"
            ));
        }
        return one_key(&at, text).map(|index| Target::KeyboardKeys(vec![index]));
    }
    err(format!(
        "{at} is not a target; use aura.fans, aura.chain[N], aura.chain[A..B], keyboard.all, keyboard.keys[\"F1\"..\"F12\"], a key name such as \"Number Pad +\", or led:N"
    ))
}

/// `target = ["Insert", "Delete", "led:110"]`: keyboard keys as one span,
/// in the order given.
fn parse_target_list(key: &str, names: &[String], ctx: &Ctx<'_>) -> Result<Target, ConfigError> {
    let at = format!("{key}.target");
    if !ctx.keyboard_enabled {
        return err(format!(
            "{at}: a list of keys needs [keyboard] enabled = true"
        ));
    }
    if names.is_empty() {
        return err(format!("{at} = []: list at least one key"));
    }
    let mut out: Vec<usize> = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let item = one_key(&format!("{at}[{index}] = \"{name}\""), name)?;
        if out.contains(&item) {
            return err(format!(
                "{at}[{index}]: \"{}\" is listed twice",
                KEYS[item].name
            ));
        }
        out.push(item);
    }
    Ok(Target::KeyboardKeys(out))
}

/// A key name or `led:N` → its index in [`KEYS`].
///
/// `led:N` is the N-th known LED in [`KEYS`] order (row by row, left to
/// right), for keys a driver lists without a name: `led:97` is
/// "Number Pad Enter" and `led:110` is "Number Pad .".
fn one_key(at: &str, text: &str) -> Result<usize, ConfigError> {
    if let Some(number) = text.strip_prefix("led:") {
        let last = KEYS.len() - 1;
        let index: usize = number.trim().parse().map_err(|_| {
            ConfigError(format!(
                "{at}: \"{number}\" is not an LED index; write led:N with N in 0..={last}"
            ))
        })?;
        if index > last {
            return err(format!(
                "{at}: LED index {index} is out of range; the keyboard has {} known LEDs, led:0..=led:{last} in key order (led:{} is \"{}\")",
                KEYS.len(),
                last,
                KEYS[last].name
            ));
        }
        return Ok(index);
    }
    key_index(text).ok_or_else(|| {
        let hint = match suggest(text) {
            Some(known) => format!("; did you mean \"{known}\"?"),
            None => "; names are OpenRGB's without \"Key: \", such as \"Escape\", \"F1\", \"W\", \"Space\", \"Number Pad 7\", or led:N".to_owned(),
        };
        ConfigError(format!("{at}: \"{text}\" is not a known key{hint}"))
    })
}

/// `"F1".."F12", "W"` → key indexes, in the order written. A range is every
/// key from the first to the last in visual order (`..` and `..=` both
/// include the last key).
fn parse_keys(at: &str, inner: &str) -> Result<Vec<usize>, ConfigError> {
    let mut rest = inner.trim_start();
    let mut out: Vec<usize> = Vec::new();
    loop {
        let (first, after) = quoted_key(at, rest)?;
        rest = after.trim_start();
        let range = if let Some(after) = rest.strip_prefix("..=") {
            Some(after)
        } else {
            rest.strip_prefix("..")
        };
        let mut picked = vec![first];
        if let Some(after) = range {
            let (last, after) = quoted_key(at, after.trim_start())?;
            rest = after.trim_start();
            if first > last {
                return err(format!(
                    "{at}: \"{}\" comes after \"{}\"; the first key comes after the last (keys run row by row, left to right)",
                    KEYS[first].name, KEYS[last].name
                ));
            }
            picked = (first..=last).collect();
        }
        for index in picked {
            if out.contains(&index) {
                return err(format!("{at}: \"{}\" is listed twice", KEYS[index].name));
            }
            out.push(index);
        }
        if rest.is_empty() {
            return Ok(out);
        }
        rest = rest
            .strip_prefix(',')
            .ok_or_else(|| {
                ConfigError(format!(
                    "{at}: expected , or .. between key names, like keys[\"W\",\"A\",\"S\",\"D\"] or keys[\"F1\"..\"F12\"]"
                ))
            })?
            .trim_start();
    }
}

/// A leading `"Name"` → its key index and the text after it.
fn quoted_key<'a>(at: &str, text: &'a str) -> Result<(usize, &'a str), ConfigError> {
    let quoted_hint = || ConfigError(format!("{at}: key names are quoted, like keys[\"F1\"]"));
    let body = text.strip_prefix('"').ok_or_else(quoted_hint)?;
    let end = body.find('"').ok_or_else(quoted_hint)?;
    let name = &body[..end];
    let index = key_index(name).ok_or_else(|| {
        let hint = match suggest(name) {
            Some(known) => format!("; did you mean \"{known}\"?"),
            None => "; names are OpenRGB's without \"Key: \", such as \"Escape\", \"F1\", \"W\", \"Space\", \"Number Pad 7\"".to_owned(),
        };
        ConfigError(format!("{at}: \"{name}\" is not a known key{hint}"))
    })?;
    Ok((index, &body[end + 1..]))
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
