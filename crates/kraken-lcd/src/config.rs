//! Writer TOML (`config.toml`).
//!
//! Service code takes [`ValidConfig`] from [`Config::load_validated`]. That
//! newtype's field is private, so an unchecked [`Config`] cannot be passed off
//! as validated. This module has no path, URL, or address key. It does not
//! read the host, the network, or the device.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

/// Failure reading or parsing a writer config file.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The path could not be read.
    #[error("failed to read config {path}: {source}")]
    Read {
        /// Path passed to [`Config::load`].
        path: PathBuf,
        /// Filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// TOML at `path` did not match the writer schema.
    ///
    /// [`ParseSummary`] is the error kind and line number. The parser's own
    /// message quotes the file, so that text is not stored and cannot reach
    /// the journal via [`Display`].
    #[error("failed to parse config {path}: {summary}")]
    Parse {
        /// Path passed to [`Config::load`].
        path: PathBuf,
        /// Kind and line number. No file text.
        summary: ParseSummary,
    },
    /// A file that parsed but failed [`Config::validate`].
    #[error("invalid config {path}: {source}")]
    Invalid {
        /// Path passed to [`Config::load_validated`].
        path: PathBuf,
        /// The limit that failed.
        #[source]
        source: InvalidConfig,
    },
}

/// Where a writer config failed to parse, without any file text.
#[derive(Debug)]
pub struct ParseSummary {
    line: Option<u32>,
}

impl ParseSummary {
    fn at(line: u32) -> Self {
        Self { line: Some(line) }
    }

    fn unknown() -> Self {
        Self { line: None }
    }
}

impl std::fmt::Display for ParseSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(line) => write!(formatter, "parse error at line {line}"),
            None => formatter.write_str("parse error"),
        }
    }
}

fn parse_summary(text: &str, source: &toml::de::Error) -> ParseSummary {
    match source.span() {
        Some(span) => ParseSummary::at(line_number(text, span.start)),
        None => ParseSummary::unknown(),
    }
}

fn line_number(text: &str, byte: usize) -> u32 {
    let end = floor_char_boundary(text, byte.min(text.len()));
    let lines = text[..end].bytes().filter(|byte| *byte == b'\n').count();
    u32::try_from(lines.saturating_add(1)).unwrap_or(u32::MAX)
}

fn floor_char_boundary(text: &str, byte: usize) -> usize {
    let mut index = byte.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// A parsed value that breaks a limit enforced by [`Config::validate`].
#[derive(Debug, Error, PartialEq)]
pub enum InvalidConfig {
    /// `writer.tick_s` is outside 0.1..=2.0.
    #[error("writer.tick_s {tick_s} is outside 0.1..=2.0")]
    TickS {
        /// Offending value.
        tick_s: f64,
    },
    /// `snapshot.stale_after_s` is outside 0.3..=10.0.
    #[error("snapshot.stale_after_s {stale_after_s} is outside 0.3..=10.0")]
    StaleAfter {
        /// Offending age.
        stale_after_s: f64,
    },
    /// `snapshot.watch_down_stock_after_s` is outside 5..=600.
    #[error("snapshot.watch_down_stock_after_s {watch_down_stock_after_s} is outside 5..=600")]
    WatchDownStockAfter {
        /// Offending delay.
        watch_down_stock_after_s: u64,
    },
    /// `snapshot.watch_down_restore_min_s` is outside 60..=3600.
    #[error("snapshot.watch_down_restore_min_s {watch_down_restore_min_s} is outside 60..=3600")]
    WatchDownRestoreMin {
        /// Offending gap.
        watch_down_restore_min_s: u64,
    },
    /// `dial.tiers` is empty or its first width is not 0.5.
    #[error("dial.tiers must start with width 0.5 (found {width_s:?})")]
    TierZero {
        /// First width, or `None` when the list is empty.
        width_s: Option<f64>,
    },
    /// A tier width is not strictly above the previous width.
    #[error("dial.tiers width {width_s} at index {index} is not strictly above {previous_s}")]
    TierOrder {
        /// Index of the offending tier. T0 is 0.
        index: usize,
        /// Offending width.
        width_s: f64,
        /// Previous width.
        previous_s: f64,
    },
    /// A tier width is not a whole multiple of the previous width.
    #[error("dial.tiers width {width_s} at index {index} is not a whole multiple of {previous_s}")]
    TierMultiple {
        /// Index of the offending tier.
        index: usize,
        /// Offending width.
        width_s: f64,
        /// Previous width.
        previous_s: f64,
    },
    /// Tier bar counts do not sum to 24.
    #[error("dial.tiers bar counts sum to {bars}, not 24")]
    TierBars {
        /// Sum of the bar counts.
        bars: u64,
    },
    /// A tier contributes no bars.
    #[error("dial.tiers bars at index {index} is {bars}; each tier needs at least 1")]
    TierBarCount {
        /// Index of the offending tier.
        index: usize,
        /// Offending count.
        bars: u32,
    },
    /// A tier width is not finite or is above 300 seconds.
    #[error("dial.tiers width {width_s} at index {index} must be finite and <= 300")]
    TierWidth {
        /// Index of the offending tier.
        index: usize,
        /// Offending width.
        width_s: f64,
    },
    /// The displayed tier window is above 30 minutes.
    #[error("dial.tiers displayed window {window_s}s is above 1800")]
    TierWindow {
        /// Sum of width times bars, in seconds.
        window_s: f64,
    },
    /// `dial.ceiling_tps` is outside 10..=10000.
    #[error("dial.ceiling_tps {ceiling_tps} is outside 10..=10000")]
    CeilingTps {
        /// Offending ceiling.
        ceiling_tps: f64,
    },
    /// `dial.xff` is outside 0.0..=1.0.
    #[error("dial.xff {xff} is outside 0.0..=1.0")]
    Xff {
        /// Offending fraction.
        xff: f64,
    },
    /// `dial.max_gap_s` is outside 0.5..=10.0.
    #[error("dial.max_gap_s {max_gap_s} is outside 0.5..=10.0")]
    MaxGap {
        /// Offending gap.
        max_gap_s: f64,
    },
    /// `upload.min_interval_s` is below the floor of 10.
    #[error("upload.min_interval_s {min_interval_s} is below the floor of 10")]
    MinInterval {
        /// Offending interval.
        min_interval_s: u64,
    },
    /// `upload.fail_limit` is outside 1..=10.
    #[error("upload.fail_limit {fail_limit} is outside 1..=10")]
    FailLimit {
        /// Offending limit.
        fail_limit: u32,
    },
    /// `upload.stream_fps` is outside 1..=12.
    #[error("upload.stream_fps {stream_fps} is outside 1..=12")]
    StreamFps {
        /// Offending rate.
        stream_fps: u8,
    },
    /// `bands.enter` is not strictly ascending within 1..=100.
    #[error("bands.enter {enter:?} must be strictly ascending and each value in 1..=100")]
    BandsEnter {
        /// Offending thresholds.
        enter: [u8; 3],
    },
    /// `bands.margin` is not strictly smaller than the smallest gap between enters.
    #[error(
        "bands.margin {margin} must be smaller than the smallest gap between enters ({min_gap})"
    )]
    BandsMargin {
        /// Offending margin.
        margin: u8,
        /// Smallest gap between consecutive enter values.
        min_gap: u8,
    },
    /// `bands.fill_min_coverage` is outside 0.0..=1.0.
    #[error("bands.fill_min_coverage {fill_min_coverage} is outside 0.0..=1.0")]
    FillMinCoverage {
        /// Offending fraction.
        fill_min_coverage: f64,
    },
    /// A hysteresis step is below 1.
    #[error("hysteresis.{field}.step {step} is < 1")]
    HysteresisStep {
        /// `ring`, `percent`, or `temp`.
        field: &'static str,
        /// Offending step.
        step: u32,
    },
    /// `display.rotate_deg` is not 0, 90, 180, or 270.
    #[error("display.rotate_deg {rotate_deg} is not 0, 90, 180, or 270")]
    RotateDeg {
        /// Offending angle.
        rotate_deg: u16,
    },
}

/// `[writer]`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Writer {
    /// Frame tick in seconds.
    #[serde(default = "defaults::tick_s")]
    pub tick_s: f64,
}

/// `[snapshot]`. The snapshot path is a compile-time constant, not a key.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Age at which a snapshot stops being fresh.
    #[serde(default = "defaults::stale_after_s")]
    pub stale_after_s: f64,
    /// Seconds of not-fresh before the stock readout.
    #[serde(default = "defaults::watch_down_stock_after_s")]
    pub watch_down_stock_after_s: u64,
    /// Minimum seconds between watch-down stock restores.
    #[serde(default = "defaults::watch_down_restore_min_s")]
    pub watch_down_restore_min_s: u64,
}

/// One dial tier: `[bar seconds, bars]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tier {
    /// Width of each bar in this tier, in seconds.
    pub width_s: f64,
    /// How many bars this tier contributes.
    pub bars: u32,
}

/// `[dial]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dial {
    /// Tiers from newest to oldest. T0's width is 0.5 and the bars sum to 24.
    #[serde(default = "defaults::tiers")]
    pub tiers: Vec<Tier>,
    /// Unused since T54: the dial shows activity and the tokens chart scales
    /// itself. Still validated so existing config files keep loading.
    #[serde(default = "defaults::ceiling_tps")]
    pub ceiling_tps: f64,
    /// Minimum covered share of a bar's 0.5 s slots before the bar is data.
    #[serde(default = "defaults::xff")]
    pub xff: f64,
    /// Largest token-counter gap that still counts toward the tokens chart.
    #[serde(default = "defaults::max_gap_s")]
    pub max_gap_s: f64,
}

/// How the writer decides when to send a frame.
///
/// `change` is the on-change upload gated by `min_interval_s`. `stream` renders
/// at `stream_fps` and does not use that gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UploadMode {
    /// Upload when the view changes, at most once per `min_interval_s`.
    #[default]
    Change,
    /// Render one frame per tick at `stream_fps`.
    Stream,
}

/// `[upload]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upload {
    /// Minimum seconds between device writes in change mode. Floor is 10.
    /// Stream mode does not use this gap; `stream_fps` bounds that rate.
    #[serde(default = "defaults::min_interval_s")]
    pub min_interval_s: u64,
    /// Consecutive upload failures before restore-stock.
    #[serde(default = "defaults::fail_limit")]
    pub fail_limit: u32,
    /// `change` (default) or `stream`.
    #[serde(default)]
    pub mode: UploadMode,
    /// Frames per second in stream mode. 1..=12, default 10.
    #[serde(default = "defaults::stream_fps")]
    pub stream_fps: u8,
}

/// `[bands]`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bands {
    /// Enter thresholds for Light, Busy, and FlatOut.
    #[serde(default = "defaults::enter", deserialize_with = "deserialize_enter")]
    pub enter: [u8; 3],
    /// How far below an enter value a band is left. Must be smaller than every gap.
    #[serde(default = "defaults::band_margin")]
    pub margin: u8,
    /// Coverage below this fraction is the Filling band.
    #[serde(default = "defaults::fill_min_coverage")]
    pub fill_min_coverage: f64,
}

/// One hysteresis `[step, margin]` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepMargin {
    /// Quantisation step. Must be at least 1.
    pub step: u32,
    /// Extra dead-zone around the shown value.
    pub margin: u32,
}

/// `[hysteresis]`. `percent` covers CPU and memory percent; `temp` covers the three temperatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hysteresis {
    /// Ring percent.
    #[serde(default = "defaults::ring")]
    pub ring: StepMargin,
    /// `cpu_pct` and `mem_pct`.
    #[serde(default = "defaults::percent")]
    pub percent: StepMargin,
    /// Coolant, CPU, and GPU temperatures.
    #[serde(default = "defaults::temp")]
    pub temp: StepMargin,
}

/// `display.variant`. Unknown names fail at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    /// Halo layout.
    #[default]
    A1,
    /// Dial-first layout.
    A3,
}

/// `[display]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplayCfg {
    /// Layout variant. `a1` or `a3`.
    #[serde(default)]
    pub variant: Variant,
    /// Extra rotation applied on top of the device orientation.
    #[serde(default = "defaults::rotate_deg")]
    pub rotate_deg: u16,
}

/// Parsed writer configuration, not yet checked by [`Config::validate`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Writer tick.
    #[serde(default)]
    pub writer: Writer,
    /// Snapshot freshness and watch-down timing.
    #[serde(default)]
    pub snapshot: Snapshot,
    /// Token dial tiers and scaling.
    #[serde(default)]
    pub dial: Dial,
    /// Upload pacing.
    #[serde(default)]
    pub upload: Upload,
    /// Load bands.
    #[serde(default)]
    pub bands: Bands,
    /// Per-field hysteresis.
    #[serde(default)]
    pub hysteresis: Hysteresis,
    /// Variant and rotation.
    #[serde(default)]
    pub display: DisplayCfg,
}

impl<'de> Deserialize<'de> for Tier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visit;
        impl<'de> serde::de::Visitor<'de> for Visit {
            type Value = Tier;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a [width_s, bars] pair")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let width_s = match seq.next_element::<FlexF64>()? {
                    Some(FlexF64(width_s)) => width_s,
                    None => return Err(serde::de::Error::invalid_length(0, &self)),
                };
                let bars = match seq.next_element::<u32>()? {
                    Some(bars) => bars,
                    None => return Err(serde::de::Error::invalid_length(1, &self)),
                };
                if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(serde::de::Error::invalid_length(3, &self));
                }
                Ok(Tier { width_s, bars })
            }
        }
        deserializer.deserialize_seq(Visit)
    }
}

/// TOML tier widths are sometimes bare integers (`5`, not `5.0`).
struct FlexF64(f64);

impl<'de> Deserialize<'de> for FlexF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visit;
        impl serde::de::Visitor<'_> for Visit {
            type Value = FlexF64;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a tier width")
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(FlexF64(value))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                exact_f64_from_i64(value)
                    .map(FlexF64)
                    .ok_or_else(|| E::custom("tier width is not an exact f64"))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                exact_f64_from_u64(value)
                    .map(FlexF64)
                    .ok_or_else(|| E::custom("tier width is not an exact f64"))
            }
        }
        deserializer.deserialize_any(Visit)
    }
}

/// Integers with magnitude above 2^53 are not exact in f64.
fn exact_f64_from_i64(value: i64) -> Option<f64> {
    let magnitude = exact_f64_from_u64(value.unsigned_abs())?;
    Some(if value < 0 { -magnitude } else { magnitude })
}

fn exact_f64_from_u64(value: u64) -> Option<f64> {
    const EXACT_INT_LIMIT: u64 = 1 << 53;
    if value > EXACT_INT_LIMIT {
        None
    } else {
        Some(value as f64)
    }
}

impl<'de> Deserialize<'de> for StepMargin {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let [step, margin] = deserialize_fixed::<2, u32, D>(deserializer, "an array of length 2")?;
        Ok(Self { step, margin })
    }
}

/// toml 1.1 stops a fixed array after N elements and drops the rest.
fn deserialize_fixed<'de, const N: usize, T, D>(
    deserializer: D,
    expected: &'static str,
) -> Result<[T; N], D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let values = Vec::<T>::deserialize(deserializer)?;
    let len = values.len();
    <[T; N]>::try_from(values).map_err(|_| serde::de::Error::invalid_length(len, &expected))
}

fn deserialize_enter<'de, D>(deserializer: D) -> Result<[u8; 3], D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_fixed::<3, u8, D>(deserializer, "an array of length 3")
}

mod defaults {
    use super::{StepMargin, Tier};

    pub(super) fn tick_s() -> f64 {
        0.5
    }
    pub(super) fn stale_after_s() -> f64 {
        1.0
    }
    pub(super) fn watch_down_stock_after_s() -> u64 {
        30
    }
    pub(super) fn watch_down_restore_min_s() -> u64 {
        600
    }
    pub(super) fn tiers() -> Vec<Tier> {
        vec![
            Tier {
                width_s: 0.5,
                bars: 10,
            },
            Tier {
                width_s: 5.0,
                bars: 2,
            },
            Tier {
                width_s: 15.0,
                bars: 3,
            },
            Tier {
                width_s: 60.0,
                bars: 4,
            },
            Tier {
                width_s: 300.0,
                bars: 5,
            },
        ]
    }
    pub(super) fn ceiling_tps() -> f64 {
        150.0
    }
    pub(super) fn xff() -> f64 {
        0.5
    }
    pub(super) fn max_gap_s() -> f64 {
        2.0
    }
    pub(super) fn min_interval_s() -> u64 {
        60
    }
    pub(super) fn fail_limit() -> u32 {
        3
    }
    pub(super) fn stream_fps() -> u8 {
        10
    }
    pub(super) fn enter() -> [u8; 3] {
        [15, 40, 70]
    }
    pub(super) fn band_margin() -> u8 {
        5
    }
    pub(super) fn fill_min_coverage() -> f64 {
        0.5
    }
    pub(super) fn ring() -> StepMargin {
        StepMargin { step: 5, margin: 2 }
    }
    pub(super) fn percent() -> StepMargin {
        StepMargin { step: 1, margin: 1 }
    }
    pub(super) fn temp() -> StepMargin {
        StepMargin { step: 1, margin: 1 }
    }
    pub(super) fn rotate_deg() -> u16 {
        0
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self {
            tick_s: defaults::tick_s(),
        }
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            stale_after_s: defaults::stale_after_s(),
            watch_down_stock_after_s: defaults::watch_down_stock_after_s(),
            watch_down_restore_min_s: defaults::watch_down_restore_min_s(),
        }
    }
}

impl Default for Dial {
    fn default() -> Self {
        Self {
            tiers: defaults::tiers(),
            ceiling_tps: defaults::ceiling_tps(),
            xff: defaults::xff(),
            max_gap_s: defaults::max_gap_s(),
        }
    }
}

impl Default for Upload {
    fn default() -> Self {
        Self {
            min_interval_s: defaults::min_interval_s(),
            fail_limit: defaults::fail_limit(),
            mode: UploadMode::Change,
            stream_fps: defaults::stream_fps(),
        }
    }
}

impl Default for Bands {
    fn default() -> Self {
        Self {
            enter: defaults::enter(),
            margin: defaults::band_margin(),
            fill_min_coverage: defaults::fill_min_coverage(),
        }
    }
}

impl Default for Hysteresis {
    fn default() -> Self {
        Self {
            ring: defaults::ring(),
            percent: defaults::percent(),
            temp: defaults::temp(),
        }
    }
}

impl Default for DisplayCfg {
    fn default() -> Self {
        Self {
            variant: Variant::A1,
            rotate_deg: defaults::rotate_deg(),
        }
    }
}

impl Config {
    /// Parse TOML text. Does not validate and has no path to report.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Read the root-owned writer config. The path is the caller's argument,
    /// not a key in the file. Private so only [`Self::load_validated`] opens it.
    fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            summary: parse_summary(&text, &source),
        })
    }

    /// Read `path` and validate it. This is the only constructor of [`ValidConfig`].
    pub fn load_validated(path: impl AsRef<Path>) -> Result<ValidConfig, ConfigError> {
        let path = path.as_ref();
        let config = Self::load(path)?;
        config.validate().map_err(|source| ConfigError::Invalid {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(ValidConfig { config })
    }

    /// Enforce every writer limit. Does not build a [`ValidConfig`].
    ///
    /// Does not read the host, the network, or the device.
    pub fn validate(&self) -> Result<(), InvalidConfig> {
        let tick_s = self.writer.tick_s;
        if !(0.1..=2.0).contains(&tick_s) {
            return Err(InvalidConfig::TickS { tick_s });
        }
        let stale_after_s = self.snapshot.stale_after_s;
        if !(0.3..=10.0).contains(&stale_after_s) {
            return Err(InvalidConfig::StaleAfter { stale_after_s });
        }
        let watch_down_stock_after_s = self.snapshot.watch_down_stock_after_s;
        if !(5..=600).contains(&watch_down_stock_after_s) {
            return Err(InvalidConfig::WatchDownStockAfter {
                watch_down_stock_after_s,
            });
        }
        let watch_down_restore_min_s = self.snapshot.watch_down_restore_min_s;
        if !(60..=3600).contains(&watch_down_restore_min_s) {
            return Err(InvalidConfig::WatchDownRestoreMin {
                watch_down_restore_min_s,
            });
        }
        validate_tiers(&self.dial.tiers)?;
        let ceiling_tps = self.dial.ceiling_tps;
        if !(10.0..=10_000.0).contains(&ceiling_tps) {
            return Err(InvalidConfig::CeilingTps { ceiling_tps });
        }
        let xff = self.dial.xff;
        if !(0.0..=1.0).contains(&xff) {
            return Err(InvalidConfig::Xff { xff });
        }
        let max_gap_s = self.dial.max_gap_s;
        if !(0.5..=10.0).contains(&max_gap_s) {
            return Err(InvalidConfig::MaxGap { max_gap_s });
        }
        let min_interval_s = self.upload.min_interval_s;
        if min_interval_s < 10 {
            return Err(InvalidConfig::MinInterval { min_interval_s });
        }
        let fail_limit = self.upload.fail_limit;
        if !(1..=10).contains(&fail_limit) {
            return Err(InvalidConfig::FailLimit { fail_limit });
        }
        let stream_fps = self.upload.stream_fps;
        if !(1..=12).contains(&stream_fps) {
            return Err(InvalidConfig::StreamFps { stream_fps });
        }
        let enter = self.bands.enter;
        if !enter_ok(enter) {
            return Err(InvalidConfig::BandsEnter { enter });
        }
        let min_gap = (enter[1] - enter[0]).min(enter[2] - enter[1]);
        let margin = self.bands.margin;
        if margin >= min_gap {
            return Err(InvalidConfig::BandsMargin { margin, min_gap });
        }
        let fill_min_coverage = self.bands.fill_min_coverage;
        if !(0.0..=1.0).contains(&fill_min_coverage) {
            return Err(InvalidConfig::FillMinCoverage { fill_min_coverage });
        }
        for (field, pair) in [
            ("ring", self.hysteresis.ring),
            ("percent", self.hysteresis.percent),
            ("temp", self.hysteresis.temp),
        ] {
            if pair.step < 1 {
                return Err(InvalidConfig::HysteresisStep {
                    field,
                    step: pair.step,
                });
            }
        }
        let rotate_deg = self.display.rotate_deg;
        if !matches!(rotate_deg, 0 | 90 | 180 | 270) {
            return Err(InvalidConfig::RotateDeg { rotate_deg });
        }
        Ok(())
    }
}

fn validate_tiers(tiers: &[Tier]) -> Result<(), InvalidConfig> {
    let Some(first) = tiers.first() else {
        return Err(InvalidConfig::TierZero { width_s: None });
    };
    if first.width_s.to_bits() != 0.5_f64.to_bits() {
        return Err(InvalidConfig::TierZero {
            width_s: Some(first.width_s),
        });
    }
    let mut bars: u64 = 0;
    let mut window_s = 0.0;
    for (index, tier) in tiers.iter().enumerate() {
        if tier.bars < 1 {
            return Err(InvalidConfig::TierBarCount {
                index,
                bars: tier.bars,
            });
        }
        if !tier.width_s.is_finite() || tier.width_s > MAX_TIER_WIDTH_S {
            return Err(InvalidConfig::TierWidth {
                index,
                width_s: tier.width_s,
            });
        }
        if index > 0 {
            let previous_s = tiers[index - 1].width_s;
            if tier.width_s <= previous_s {
                return Err(InvalidConfig::TierOrder {
                    index,
                    width_s: tier.width_s,
                    previous_s,
                });
            }
            if !is_whole_multiple(tier.width_s, previous_s) {
                return Err(InvalidConfig::TierMultiple {
                    index,
                    width_s: tier.width_s,
                    previous_s,
                });
            }
        }
        bars += u64::from(tier.bars);
        window_s += tier.width_s * f64::from(tier.bars);
    }
    if bars != 24 {
        return Err(InvalidConfig::TierBars { bars });
    }
    if !window_s.is_finite() || window_s > MAX_WINDOW_S {
        return Err(InvalidConfig::TierWindow { window_s });
    }
    Ok(())
}

/// Widest bar in the dial design, 5 minutes.
const MAX_TIER_WIDTH_S: f64 = 300.0;
/// Displayed span of the dial design, 30 minutes.
const MAX_WINDOW_S: f64 = 1800.0;

/// Whole multiples of a 0.5-second T0 stay dyadic, so `previous * k` is bit-exact.
fn is_whole_multiple(width: f64, previous: f64) -> bool {
    if !width.is_finite() || !previous.is_finite() || previous <= 0.0 {
        return false;
    }
    let ratio = width / previous;
    let factor = ratio.round();
    if !factor.is_finite() || factor < 2.0 {
        return false;
    }
    let expected = previous * factor;
    width.to_bits() == expected.to_bits()
}

fn enter_ok(enter: [u8; 3]) -> bool {
    enter.iter().all(|value| (1..=100).contains(value))
        && enter[0] < enter[1]
        && enter[1] < enter[2]
}

/// A [`Config`] that has passed [`Config::validate`].
///
/// The inner field is private. [`Config::load_validated`] is the only constructor.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidConfig {
    config: Config,
}

impl std::ops::Deref for ValidConfig {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
