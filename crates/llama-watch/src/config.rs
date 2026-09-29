//! Watcher TOML (`watch.toml`).
//!
//! Service code takes [`ValidWatchConfig`] from [`Config::load_validated`].
//! That newtype's field is private, so an unchecked [`Config`] cannot be
//! passed off as validated. `nproc` is supplied by the caller; this module
//! does not read the host.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use llama_core::backend::Backend;
use serde::Deserialize;
use thiserror::Error;

/// Failure reading or parsing a watcher config file.
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
    /// TOML at `path` did not match the watcher schema.
    ///
    /// [`ParseSummary`] is the error kind and line number. The parser's own
    /// message quotes the file, so that text is not stored.
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
        source: InvalidWatchConfig,
    },
}

/// Where a watcher config failed to parse, without any file text.
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
pub enum InvalidWatchConfig {
    /// `collector.tick_s` is outside 0.05..=1.0.
    #[error("collector.tick_s {tick_s} is outside 0.05..=1.0")]
    TickS {
        /// Offending value.
        tick_s: f64,
    },
    /// `collector.cpu_window_s` is outside `tick_s`..=5.0.
    #[error("collector.cpu_window_s {cpu_window_s} is outside tick_s ({tick_s})..=5.0")]
    CpuWindow {
        /// Offending window.
        cpu_window_s: f64,
        /// Tick it was compared with.
        tick_s: f64,
    },
    /// `collector.cpu_top_k` is outside 1..=nproc.
    #[error("collector.cpu_top_k {cpu_top_k} is outside 1..={nproc}")]
    CpuTopK {
        /// Offending count.
        cpu_top_k: u32,
        /// Caller-supplied logical CPU count.
        nproc: u32,
    },
    /// `llama.url` is not an `http://` loopback IP literal, or it has a path.
    #[error("llama.url {url:?}: {reason}")]
    LlamaUrl {
        /// Offending URL.
        url: String,
        /// Why it was rejected. A path, query, or fragment uses `no path allowed`.
        reason: &'static str,
    },
    /// A `*_timeout_s` is not strictly positive and strictly less than its interval.
    #[error(
        "llama.{field}_timeout_s {timeout_s} must be > 0 and < {field}_interval_s {interval_s}"
    )]
    Timeout {
        /// `running`, `metrics`, `slots`, or `activity`.
        field: &'static str,
        /// Offending timeout.
        timeout_s: f64,
        /// Interval it was compared with.
        interval_s: f64,
    },
    /// `llama.slots_interval_s` is outside 0.5..=10.0.
    #[error("llama.slots_interval_s {slots_interval_s} is outside 0.5..=10.0")]
    SlotsInterval {
        /// Offending interval.
        slots_interval_s: f64,
    },
    /// `llama.slots_max_bytes` is above 16 MiB.
    #[error("llama.slots_max_bytes {slots_max_bytes} is above 16 MiB")]
    SlotsMaxBytes {
        /// Offending cap.
        slots_max_bytes: u64,
    },
    /// A llama poll interval is outside 0.1..=60.
    #[error("llama.{field}_interval_s {interval_s} is outside 0.1..=60")]
    PollInterval {
        /// `running`, `metrics`, or `activity`.
        field: &'static str,
        /// Offending interval.
        interval_s: f64,
    },
    /// `llama.input_tail_chars` is outside 256..=32768.
    #[error("llama.input_tail_chars {input_tail_chars} is outside 256..=32768")]
    InputTail {
        /// Offending length.
        input_tail_chars: u32,
    },
    /// `llama.output_tail_chars` is outside 256..=32768.
    #[error("llama.output_tail_chars {output_tail_chars} is outside 256..=32768")]
    OutputTail {
        /// Offending length.
        output_tail_chars: u32,
    },
    /// `tty.prompt_ceiling_tps` is not finite or is outside (0, 100000].
    #[error("tty.prompt_ceiling_tps {prompt_ceiling_tps} must be finite and in (0, 100000]")]
    PromptCeiling {
        /// Offending ceiling.
        prompt_ceiling_tps: f64,
    },
    /// `tty.gen_ceiling_tps` is not finite or is outside (0, 100000].
    #[error("tty.gen_ceiling_tps {gen_ceiling_tps} must be finite and in (0, 100000]")]
    GenCeiling {
        /// Offending ceiling.
        gen_ceiling_tps: f64,
    },
    /// `tty.chart_bucket_s` is outside 1..=60.
    #[error("tty.chart_bucket_s {chart_bucket_s} is outside 1..=60")]
    ChartBucket {
        /// Offending bucket width.
        chart_bucket_s: u64,
    },
    /// `tty.ctx_history_h` is outside 1..=24.
    #[error("tty.ctx_history_h {ctx_history_h} is outside 1..=24")]
    CtxHistory {
        /// Offending span in hours.
        ctx_history_h: u32,
    },
    /// `[models.aliases]` has more than 32 entries.
    #[error("models.aliases has {count} entries, above the cap of 32")]
    AliasCount {
        /// Offending count.
        count: usize,
    },
    /// `[llama.backends]` has more than 32 entries.
    #[error("llama.backends has {count} entries, above the cap of 32")]
    BackendCount {
        /// Offending count.
        count: usize,
    },
    /// An alias display name is longer than 64 characters.
    #[error("models.aliases value is {chars} characters, above the cap of 64")]
    AliasValue {
        /// Offending character count.
        chars: usize,
    },
    /// `models.max_name_chars` is outside 2..=12.
    #[error("models.max_name_chars {max_name_chars} is outside 2..=12")]
    MaxNameChars {
        /// Offending length.
        max_name_chars: u32,
    },
    /// `tty.fps` is outside 1..=20.
    #[error("tty.fps {fps} is outside 1..=20")]
    Fps {
        /// Offending rate.
        fps: u32,
    },
    /// `tty.full_redraw_s` is below 1.
    #[error("tty.full_redraw_s {full_redraw_s} is < 1")]
    FullRedraw {
        /// Offending period.
        full_redraw_s: u64,
    },
    /// `tty.blank_min` is outside 0..=60.
    #[error("tty.blank_min {blank_min} is outside 0..=60")]
    BlankMin {
        /// Offending interval in minutes.
        blank_min: u32,
    },
    /// `tty.sleep_min` is outside 0..=60, or is non-zero while
    /// `tty.blank_min` is 0 or not below it. The kernel only powers down a
    /// blanked console, so sleep needs a blank first.
    #[error(
        "tty.sleep_min {sleep_min} must be 0, or <= 60 and above a non-zero blank_min (got {blank_min})"
    )]
    SleepMin {
        /// Offending interval in minutes.
        sleep_min: u32,
        /// `tty.blank_min` it was checked against.
        blank_min: u32,
    },
    /// `load.smooth_s` is non-finite or outside 0..=5.
    #[error("load.smooth_s {smooth_s} is outside 0..=5")]
    SmoothS {
        /// Offending time constant.
        smooth_s: f64,
    },
    /// `load.nominal_frac` is non-finite or outside 0.5..=1.0.
    #[error("load.nominal_frac {nominal_frac} is outside 0.5..=1.0")]
    NominalFrac {
        /// Offending fraction.
        nominal_frac: f64,
    },
    /// `load.cpu_limit_w` is non-finite or not strictly positive.
    #[error("load.cpu_limit_w {cpu_limit_w} must be finite and > 0")]
    CpuLimit {
        /// Offending wattage.
        cpu_limit_w: f64,
    },
    /// `load.gpu_idle_w` or `load.cpu_idle_w` is non-finite or negative, or
    /// `cpu_idle_w` is not below `nominal_frac × cpu_limit_w`.
    #[error("load.{field} {watts} must be finite, >= 0, and below the device limit")]
    IdleWatts {
        /// `gpu_idle_w` or `cpu_idle_w`.
        field: &'static str,
        /// Offending wattage.
        watts: f64,
    },
    /// A `[fans]` value is out of range. `reason` names the limit.
    #[error("fans: {reason}")]
    Fans {
        /// Which limit failed.
        reason: &'static str,
    },
}

/// `[collector]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collector {
    /// Sample period in seconds.
    #[serde(default = "defaults::tick_s")]
    pub tick_s: f64,
    /// CPU percent window in seconds. Inclusive lower bound is `tick_s`.
    #[serde(default = "defaults::cpu_window_s")]
    pub cpu_window_s: f64,
    /// How many busiest CPUs feed `cpu_topk_pct`.
    #[serde(default = "defaults::cpu_top_k")]
    pub cpu_top_k: u32,
}

/// `[llama]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Llama {
    /// `false` turns llama-swap polling off. The watcher then never dials
    /// `url`, publishes `ai = idle` with no models, and the tty shows NO LLAMA.
    #[serde(default = "defaults::enabled")]
    pub enabled: bool,
    /// llama-swap base URL. Loopback IP literal only. Point it at llama-swap
    /// itself: a proxy in front of it may hide `/slots` and `/api/metrics`.
    #[serde(default = "defaults::url")]
    pub url: String,
    /// Seconds between `/running` polls.
    #[serde(default = "defaults::running_interval_s")]
    pub running_interval_s: f64,
    /// Deadline for one `/running` poll.
    #[serde(default = "defaults::running_timeout_s")]
    pub running_timeout_s: f64,
    /// Seconds between `/metrics` polls.
    #[serde(default = "defaults::metrics_interval_s")]
    pub metrics_interval_s: f64,
    /// Deadline for one `/metrics` poll.
    #[serde(default = "defaults::metrics_timeout_s")]
    pub metrics_timeout_s: f64,
    /// Seconds between `/slots` polls.
    #[serde(default = "defaults::slots_interval_s")]
    pub slots_interval_s: f64,
    /// Deadline for one `/slots` poll.
    #[serde(default = "defaults::slots_timeout_s")]
    pub slots_timeout_s: f64,
    /// Maximum accepted `/slots` body.
    #[serde(default = "defaults::slots_max_bytes")]
    pub slots_max_bytes: u64,
    /// Seconds between activity polls.
    #[serde(default = "defaults::activity_interval_s")]
    pub activity_interval_s: f64,
    /// Deadline for one activity poll.
    #[serde(default = "defaults::activity_timeout_s")]
    pub activity_timeout_s: f64,
    /// Characters kept from the input side of slot text.
    #[serde(default = "defaults::input_tail_chars")]
    pub input_tail_chars: u32,
    /// Characters kept from the output side of slot text.
    #[serde(default = "defaults::output_tail_chars")]
    pub output_tail_chars: u32,
    /// `[llama.backends]`: model id to `llamacpp`, `sglang`, `vllm` or
    /// `openai`, over what the launch command says (T72). Unknown words fail
    /// the parse; ids that are not loaded are fine.
    #[serde(default = "defaults::backends")]
    pub backends: BTreeMap<String, Backend>,
}

/// `[models]`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Models {
    /// Truncate sanitised model names to this many characters, before the ellipsis.
    #[serde(default = "defaults::max_name_chars")]
    pub max_name_chars: u32,
    /// llama-swap model id to display name.
    #[serde(default = "defaults::aliases")]
    pub aliases: BTreeMap<String, String>,
}

/// How `[load]` picks each device's idle floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IdleMode {
    /// The lowest 10 s mean draw seen since the watcher started, ignoring the
    /// first 30 s after start or after the source reappears. The configured
    /// watts hold until the first mean; after that the floor may sit above
    /// them. It never rises, so sustained load cannot become the new idle,
    /// and it stays at or below `(nominal_frac - 0.1) × limit`. A box loaded
    /// from boot should use `Fixed`.
    Auto,
    /// Use `gpu_idle_w` and `cpu_idle_w` as they are.
    Fixed,
}

/// `[load]`. Bottleneck activity: each device's draw over its own headroom.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Load {
    /// Socket power that counts as the CPU fully busy, in watts.
    #[serde(default = "defaults::cpu_limit_w")]
    pub cpu_limit_w: f64,
    /// `"auto"` (lowest seen, capped at the watts below) or `"fixed"`.
    #[serde(default = "defaults::idle")]
    pub idle: IdleMode,
    /// GPU idle floor in watts. With `auto` the floor starts here until learned.
    #[serde(default = "defaults::gpu_idle_w")]
    pub gpu_idle_w: f64,
    /// CPU socket idle floor in watts. With `auto` the floor starts here until learned.
    #[serde(default = "defaults::cpu_idle_w")]
    pub cpu_idle_w: f64,
    /// EMA time constant in seconds. `0` disables smoothing.
    #[serde(default = "defaults::smooth_s")]
    pub smooth_s: f64,
    /// Share of each device's limit that reads 100 %. A device's fraction is
    /// `(w - idle) / (nominal_frac × limit - idle)`, so sustained heavy load
    /// reads about 100 and a spike up to the 125 % peg.
    #[serde(default = "defaults::nominal_frac")]
    pub nominal_frac: f64,
}

/// `[tty]`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tty {
    /// Console redraws per second.
    #[serde(default = "defaults::fps")]
    pub fps: u32,
    /// Seconds between full console redraws.
    #[serde(default = "defaults::full_redraw_s")]
    pub full_redraw_s: u64,
    /// Generation (decode) tok/s at the top of the tty's log-scaled rate
    /// scales. Separate from the writer's `[dial] ceiling_tps`.
    #[serde(default = "defaults::gen_ceiling_tps")]
    pub gen_ceiling_tps: f64,
    /// Prompt (prefill) tok/s at the top of the tty's log-scaled rate scales.
    #[serde(default = "defaults::prompt_ceiling_tps")]
    pub prompt_ceiling_tps: f64,
    /// Seconds of token-rate history in one tty chart column.
    #[serde(default = "defaults::chart_bucket_s")]
    pub chart_bucket_s: u64,
    /// Chart resolution. See [`ChartGlyphs`].
    #[serde(default)]
    pub chart_glyphs: ChartGlyphs,
    /// `false` hides the IN/OUT llama text panels and gives their rows to the
    /// chart and RECENT. The poller then keeps no prompt or generated text
    /// (RR-LV1). `llama-watch run --no-text` forces this off.
    #[serde(default = "defaults::show_text")]
    pub show_text: bool,
    /// How the IN panel shows the prompt tail. See [`PromptView`].
    #[serde(default)]
    pub prompt_view: PromptView,
    /// Hours of per-slot context history in the SLOTS sparklines, 1..=24.
    #[serde(default = "defaults::ctx_history_h")]
    pub ctx_history_h: u32,
    /// Minutes without a keypress before the kernel blanks the console,
    /// 0..=60; 0 (default) sends nothing. Sent once at start-up as
    /// `ESC [ 9 ; n ]`. On KMS the blank is also DPMS standby.
    #[serde(default)]
    pub blank_min: u32,
    /// Minutes without a keypress before the monitor powers down, 0..=60;
    /// 0 (default) sends nothing. Non-zero must be above a non-zero
    /// `blank_min`. The kernel counts powerdown from the blank, so the
    /// watcher sends `ESC [ 14 ; sleep_min - blank_min ]` once at start-up.
    #[serde(default)]
    pub sleep_min: u32,
}

/// `tty.prompt_view`: whether chat-template control tokens stay in IN.
///
/// Either way the S12 sanitiser runs last on the text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptView {
    /// Strip `<|im_start|>`, `[INST]`, `<think>` and the like, and mark each
    /// role turn with a `-- user --` label line.
    #[default]
    Clean,
    /// The prompt tail as llama-server holds it.
    Raw,
}

/// `tty.chart_glyphs`: which block glyphs the token chart draws with.
///
/// Whether `setfont` loaded llama-hack-12x24 cannot be seen from the
/// watcher, so this is a setting. `halves` is the code default and uses only
/// glyphs eurlatgr has; `packaging/watch.example.toml` sets `eighths`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChartGlyphs {
    /// `▄ ▀ █`: 2 levels per row. Renders with eurlatgr.
    #[default]
    Halves,
    /// Lower eighths `▁`–`▇` and `█`: 8 levels per row. The falling half
    /// inverts the lower eighths. Needs llama-hack-12x24.
    Eighths,
}

/// `[fans]`: read-only fan speeds from one Super-I/O hwmon, for the tty.
///
/// Off by default. The watcher only reads `fanN_input`, `pwmN` and
/// `pwmN_enable`; it never writes them. The snapshot does not carry fans.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fans {
    /// `true` draws the FANS panel on tty11.
    #[serde(default)]
    pub enabled: bool,
    /// The hwmon `name` to match, e.g. `nct6798`. Never the `hwmonN` number,
    /// which changes across reboots.
    #[serde(default)]
    pub hwmon: String,
    /// Fan indices `N` for `fanN_input`, 1..=16, at most 8.
    #[serde(default)]
    pub channels: Vec<u32>,
    /// One label per channel, at most 10 characters. Default `fanN`.
    #[serde(default)]
    pub labels: Option<Vec<String>>,
}

/// Most fans the panel shows.
pub const MAX_FANS: usize = 8;
/// Longest fan label, in characters.
pub const MAX_FAN_LABEL: usize = 10;
const MAX_FAN_HWMON: usize = 32;

impl Fans {
    /// Labels for [`Self::channels`], in order: the configured label reduced
    /// to printable ASCII, or `fanN` when there is none or it is blank.
    #[must_use]
    pub fn resolved_labels(&self) -> Vec<String> {
        self.channels
            .iter()
            .enumerate()
            .map(|(i, channel)| {
                let given = self
                    .labels
                    .as_ref()
                    .and_then(|labels| labels.get(i))
                    .map(|label| printable_ascii(label, MAX_FAN_LABEL))
                    .unwrap_or_default();
                if given.trim().is_empty() {
                    format!("fan{channel}")
                } else {
                    given.trim().to_owned()
                }
            })
            .collect()
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.channels.len() > MAX_FANS {
            return Err("at most 8 channels");
        }
        if self.channels.iter().any(|n| !(1..=16).contains(n)) {
            return Err("channels must be 1..=16");
        }
        let mut seen = self.channels.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != self.channels.len() {
            return Err("channels must not repeat");
        }
        if let Some(labels) = &self.labels {
            if labels.len() != self.channels.len() {
                return Err("labels must match channels in length");
            }
            if labels.iter().any(|l| l.chars().count() > MAX_FAN_LABEL) {
                return Err("labels must be at most 10 characters");
            }
        }
        if self.hwmon.chars().count() > MAX_FAN_HWMON
            || self
                .hwmon
                .chars()
                .any(|ch| !ch.is_ascii_graphic() || ch == '/')
        {
            return Err("hwmon must be a plain name of printable ASCII, at most 32");
        }
        if self.enabled {
            if self.hwmon.is_empty() {
                return Err("hwmon is required when enabled");
            }
            if self.channels.is_empty() {
                return Err("channels is required when enabled");
            }
        }
        Ok(())
    }
}

/// Printable ASCII only: other characters become `?`. At most `cap` chars.
fn printable_ascii(text: &str, cap: usize) -> String {
    text.chars()
        .take(cap)
        .map(|ch| {
            if ch == ' ' || ch.is_ascii_graphic() {
                ch
            } else {
                '?'
            }
        })
        .collect()
}

/// Parsed watcher configuration, not yet checked by [`Config::validate`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Collector timing and CPU top-k.
    #[serde(default)]
    pub collector: Collector,
    /// Llama poll targets and caps.
    #[serde(default)]
    pub llama: Llama,
    /// Model name length and aliases.
    #[serde(default)]
    pub models: Models,
    /// tty11 redraw.
    #[serde(default)]
    pub tty: Tty,
    /// Power-weighted activity.
    #[serde(default)]
    pub load: Load,
    /// Read-only fan panel on tty11. Off by default.
    #[serde(default)]
    pub fans: Fans,
}

mod defaults {
    use std::collections::BTreeMap;

    pub(super) fn tick_s() -> f64 {
        0.1
    }
    pub(super) fn cpu_window_s() -> f64 {
        1.0
    }
    pub(super) fn cpu_top_k() -> u32 {
        8
    }
    pub(super) fn enabled() -> bool {
        true
    }
    /// llama-swap's own default listen address.
    pub(super) fn url() -> String {
        "http://127.0.0.1:8080".to_owned()
    }
    pub(super) fn running_interval_s() -> f64 {
        0.5
    }
    pub(super) fn running_timeout_s() -> f64 {
        0.25
    }
    pub(super) fn metrics_interval_s() -> f64 {
        0.25
    }
    pub(super) fn metrics_timeout_s() -> f64 {
        0.2
    }
    pub(super) fn slots_interval_s() -> f64 {
        1.0
    }
    pub(super) fn slots_timeout_s() -> f64 {
        0.5
    }
    pub(super) fn slots_max_bytes() -> u64 {
        4_194_304
    }
    pub(super) fn activity_interval_s() -> f64 {
        2.0
    }
    pub(super) fn activity_timeout_s() -> f64 {
        0.25
    }
    pub(super) fn input_tail_chars() -> u32 {
        8192
    }
    pub(super) fn output_tail_chars() -> u32 {
        24_576
    }
    pub(super) fn max_name_chars() -> u32 {
        12
    }
    pub(super) fn aliases() -> BTreeMap<String, String> {
        BTreeMap::new()
    }
    pub(super) fn backends() -> BTreeMap<String, super::Backend> {
        BTreeMap::new()
    }
    pub(super) fn fps() -> u32 {
        10
    }
    pub(super) fn full_redraw_s() -> u64 {
        5
    }
    pub(super) fn gen_ceiling_tps() -> f64 {
        250.0
    }
    pub(super) fn prompt_ceiling_tps() -> f64 {
        1500.0
    }
    pub(super) fn chart_bucket_s() -> u64 {
        2
    }
    pub(super) fn show_text() -> bool {
        true
    }
    pub(super) fn ctx_history_h() -> u32 {
        6
    }
    pub(super) fn cpu_limit_w() -> f64 {
        230.0
    }
    pub(super) fn idle() -> super::IdleMode {
        super::IdleMode::Auto
    }
    pub(super) fn gpu_idle_w() -> f64 {
        30.0
    }
    pub(super) fn cpu_idle_w() -> f64 {
        25.0
    }
    pub(super) fn smooth_s() -> f64 {
        0.3
    }
    pub(super) fn nominal_frac() -> f64 {
        0.8
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self {
            tick_s: defaults::tick_s(),
            cpu_window_s: defaults::cpu_window_s(),
            cpu_top_k: defaults::cpu_top_k(),
        }
    }
}

impl Default for Llama {
    fn default() -> Self {
        Self {
            enabled: defaults::enabled(),
            url: defaults::url(),
            running_interval_s: defaults::running_interval_s(),
            running_timeout_s: defaults::running_timeout_s(),
            metrics_interval_s: defaults::metrics_interval_s(),
            metrics_timeout_s: defaults::metrics_timeout_s(),
            slots_interval_s: defaults::slots_interval_s(),
            slots_timeout_s: defaults::slots_timeout_s(),
            slots_max_bytes: defaults::slots_max_bytes(),
            activity_interval_s: defaults::activity_interval_s(),
            activity_timeout_s: defaults::activity_timeout_s(),
            input_tail_chars: defaults::input_tail_chars(),
            output_tail_chars: defaults::output_tail_chars(),
            backends: defaults::backends(),
        }
    }
}

impl Default for Models {
    fn default() -> Self {
        Self {
            max_name_chars: defaults::max_name_chars(),
            aliases: defaults::aliases(),
        }
    }
}

impl Default for Load {
    fn default() -> Self {
        Self {
            cpu_limit_w: defaults::cpu_limit_w(),
            idle: defaults::idle(),
            gpu_idle_w: defaults::gpu_idle_w(),
            cpu_idle_w: defaults::cpu_idle_w(),
            smooth_s: defaults::smooth_s(),
            nominal_frac: defaults::nominal_frac(),
        }
    }
}

impl Default for Tty {
    fn default() -> Self {
        Self {
            fps: defaults::fps(),
            full_redraw_s: defaults::full_redraw_s(),
            gen_ceiling_tps: defaults::gen_ceiling_tps(),
            prompt_ceiling_tps: defaults::prompt_ceiling_tps(),
            chart_bucket_s: defaults::chart_bucket_s(),
            chart_glyphs: ChartGlyphs::default(),
            show_text: defaults::show_text(),
            prompt_view: PromptView::default(),
            ctx_history_h: defaults::ctx_history_h(),
            blank_min: 0,
            sleep_min: 0,
        }
    }
}

impl Config {
    /// Read TOML at `path`. Does not validate and does not return a [`ValidWatchConfig`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            summary: parse_summary(&text, &source),
        })
    }

    /// Parse TOML text. Does not validate and has no path to report.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Read `path` and validate it. This is the only constructor of [`ValidWatchConfig`].
    ///
    /// `nproc` is the logical CPU count from the caller.
    pub fn load_validated(
        path: impl AsRef<Path>,
        nproc: u32,
    ) -> Result<ValidWatchConfig, ConfigError> {
        let path = path.as_ref();
        let config = Self::load(path)?;
        config
            .validate(nproc)
            .map_err(|source| ConfigError::Invalid {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(ValidWatchConfig { config })
    }

    /// Enforce every watcher limit. Does not build a [`ValidWatchConfig`].
    ///
    /// `nproc` is the logical CPU count from the caller. This function does not
    /// read the host.
    pub fn validate(&self, nproc: u32) -> Result<(), InvalidWatchConfig> {
        let tick_s = self.collector.tick_s;
        if !(0.05..=1.0).contains(&tick_s) {
            return Err(InvalidWatchConfig::TickS { tick_s });
        }
        let cpu_window_s = self.collector.cpu_window_s;
        if !(tick_s..=5.0).contains(&cpu_window_s) {
            return Err(InvalidWatchConfig::CpuWindow {
                cpu_window_s,
                tick_s,
            });
        }
        let cpu_top_k = self.collector.cpu_top_k;
        if !(1..=nproc).contains(&cpu_top_k) {
            return Err(InvalidWatchConfig::CpuTopK { cpu_top_k, nproc });
        }
        if let Err(reason) = llama_url_reason(&self.llama.url) {
            return Err(InvalidWatchConfig::LlamaUrl {
                url: self.llama.url.clone(),
                reason,
            });
        }
        let slots_interval_s = self.llama.slots_interval_s;
        if !(0.5..=10.0).contains(&slots_interval_s) {
            return Err(InvalidWatchConfig::SlotsInterval { slots_interval_s });
        }
        // 0.1..=60 keeps every poll period inside what Duration::from_secs_f64 accepts.
        // slots_interval_s is the tighter 0.5..=10 subset of that range.
        for (field, interval_s) in [
            ("running", self.llama.running_interval_s),
            ("metrics", self.llama.metrics_interval_s),
            ("activity", self.llama.activity_interval_s),
        ] {
            if !(0.1..=60.0).contains(&interval_s) {
                return Err(InvalidWatchConfig::PollInterval { field, interval_s });
            }
        }
        for (field, timeout_s, interval_s) in [
            (
                "running",
                self.llama.running_timeout_s,
                self.llama.running_interval_s,
            ),
            (
                "metrics",
                self.llama.metrics_timeout_s,
                self.llama.metrics_interval_s,
            ),
            (
                "slots",
                self.llama.slots_timeout_s,
                self.llama.slots_interval_s,
            ),
            (
                "activity",
                self.llama.activity_timeout_s,
                self.llama.activity_interval_s,
            ),
        ] {
            if !(timeout_s.is_finite()
                && interval_s.is_finite()
                && timeout_s > 0.0
                && timeout_s < interval_s)
            {
                return Err(InvalidWatchConfig::Timeout {
                    field,
                    timeout_s,
                    interval_s,
                });
            }
        }
        let slots_max_bytes = self.llama.slots_max_bytes;
        if slots_max_bytes > MAX_SLOTS_BYTES {
            return Err(InvalidWatchConfig::SlotsMaxBytes { slots_max_bytes });
        }
        let input_tail_chars = self.llama.input_tail_chars;
        if !(256..=32768).contains(&input_tail_chars) {
            return Err(InvalidWatchConfig::InputTail { input_tail_chars });
        }
        let output_tail_chars = self.llama.output_tail_chars;
        if !(256..=32768).contains(&output_tail_chars) {
            return Err(InvalidWatchConfig::OutputTail { output_tail_chars });
        }
        if self.llama.backends.len() > MAX_ALIASES {
            return Err(InvalidWatchConfig::BackendCount {
                count: self.llama.backends.len(),
            });
        }
        if self.models.aliases.len() > MAX_ALIASES {
            return Err(InvalidWatchConfig::AliasCount {
                count: self.models.aliases.len(),
            });
        }
        for value in self.models.aliases.values() {
            let chars = value.chars().count();
            if chars > MAX_ALIAS_VALUE_CHARS {
                return Err(InvalidWatchConfig::AliasValue { chars });
            }
        }
        let max_name_chars = self.models.max_name_chars;
        if !(2..=12).contains(&max_name_chars) {
            return Err(InvalidWatchConfig::MaxNameChars { max_name_chars });
        }
        let fps = self.tty.fps;
        if !(1..=20).contains(&fps) {
            return Err(InvalidWatchConfig::Fps { fps });
        }
        let full_redraw_s = self.tty.full_redraw_s;
        if full_redraw_s < 1 {
            return Err(InvalidWatchConfig::FullRedraw { full_redraw_s });
        }
        let gen_ceiling_tps = self.tty.gen_ceiling_tps;
        if !gen_ceiling_tps.is_finite()
            || gen_ceiling_tps <= 0.0
            || gen_ceiling_tps > MAX_RATE_CEILING_TPS
        {
            return Err(InvalidWatchConfig::GenCeiling { gen_ceiling_tps });
        }
        let prompt_ceiling_tps = self.tty.prompt_ceiling_tps;
        if !prompt_ceiling_tps.is_finite()
            || prompt_ceiling_tps <= 0.0
            || prompt_ceiling_tps > MAX_RATE_CEILING_TPS
        {
            return Err(InvalidWatchConfig::PromptCeiling { prompt_ceiling_tps });
        }
        let chart_bucket_s = self.tty.chart_bucket_s;
        if !(1..=60).contains(&chart_bucket_s) {
            return Err(InvalidWatchConfig::ChartBucket { chart_bucket_s });
        }
        let ctx_history_h = self.tty.ctx_history_h;
        if !(1..=24).contains(&ctx_history_h) {
            return Err(InvalidWatchConfig::CtxHistory { ctx_history_h });
        }
        let blank_min = self.tty.blank_min;
        if blank_min > 60 {
            return Err(InvalidWatchConfig::BlankMin { blank_min });
        }
        let sleep_min = self.tty.sleep_min;
        if sleep_min > 60 || (sleep_min != 0 && (blank_min == 0 || sleep_min <= blank_min)) {
            return Err(InvalidWatchConfig::SleepMin {
                sleep_min,
                blank_min,
            });
        }
        let smooth_s = self.load.smooth_s;
        if !smooth_s.is_finite() || !(0.0..=5.0).contains(&smooth_s) {
            return Err(InvalidWatchConfig::SmoothS { smooth_s });
        }
        let nominal_frac = self.load.nominal_frac;
        if !nominal_frac.is_finite() || !(0.5..=1.0).contains(&nominal_frac) {
            return Err(InvalidWatchConfig::NominalFrac { nominal_frac });
        }
        let cpu_limit_w = self.load.cpu_limit_w;
        if !cpu_limit_w.is_finite() || cpu_limit_w <= 0.0 {
            return Err(InvalidWatchConfig::CpuLimit { cpu_limit_w });
        }
        for (field, watts) in [
            ("gpu_idle_w", self.load.gpu_idle_w),
            ("cpu_idle_w", self.load.cpu_idle_w),
        ] {
            if !watts.is_finite() || watts < 0.0 {
                return Err(InvalidWatchConfig::IdleWatts { field, watts });
            }
        }
        if self.load.cpu_idle_w >= nominal_frac * cpu_limit_w {
            return Err(InvalidWatchConfig::IdleWatts {
                field: "cpu_idle_w",
                watts: self.load.cpu_idle_w,
            });
        }
        self.fans
            .validate()
            .map_err(|reason| InvalidWatchConfig::Fans { reason })?;
        Ok(())
    }
}

const MAX_RATE_CEILING_TPS: f64 = 100_000.0;
const MAX_ALIASES: usize = 32;
const MAX_ALIAS_VALUE_CHARS: usize = 64;

const MAX_SLOTS_BYTES: u64 = 16 * 1024 * 1024;

const URL_ORIGIN: &str = "must be http:// with a loopback IP literal host";
const URL_PATH: &str = "no path allowed";

/// `Ok(())` for `http://` plus a loopback IP literal and an optional port.
/// A path, query, or fragment is [`URL_PATH`]. Anything else is [`URL_ORIGIN`].
fn llama_url_reason(url: &str) -> Result<(), &'static str> {
    let Some(rest) = url.strip_prefix("http://") else {
        return Err(URL_ORIGIN);
    };
    if rest.is_empty()
        || !rest.is_ascii()
        || rest
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte < 0x20)
    {
        return Err(URL_ORIGIN);
    }
    let Some((host, rest)) = split_host(rest) else {
        return Err(URL_ORIGIN);
    };
    let Ok(ip) = host.parse::<IpAddr>() else {
        return Err(URL_ORIGIN);
    };
    if !ip.is_loopback() {
        return Err(URL_ORIGIN);
    }
    let Some(after_port) = split_port(rest) else {
        return Err(URL_ORIGIN);
    };
    if after_port.is_empty() {
        Ok(())
    } else {
        Err(URL_PATH)
    }
}

fn split_host(rest: &str) -> Option<(&str, &str)> {
    if let Some(inner) = rest.strip_prefix('[') {
        let end = inner.find(']')?;
        let host = &inner[..end];
        if host.is_empty() {
            return None;
        }
        return Some((host, &inner[end + 1..]));
    }
    let end = rest.find([':', '/', '?', '#', '@']).unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    Some((&rest[..end], &rest[end..]))
}

/// Returns the suffix after a valid port. No colon means the whole `rest` is the suffix.
fn split_port(rest: &str) -> Option<&str> {
    let Some(after_colon) = rest.strip_prefix(':') else {
        return Some(rest);
    };
    let end = after_colon
        .find(['/', '?', '#'])
        .unwrap_or(after_colon.len());
    let port = &after_colon[..end];
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value: u32 = port.parse().ok()?;
    if (1..=65535).contains(&value) {
        Some(&after_colon[end..])
    } else {
        None
    }
}

/// A [`Config`] that has passed [`Config::validate`].
///
/// The inner field is private. [`Config::load_validated`] is the only constructor.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidWatchConfig {
    config: Config,
}

impl ValidWatchConfig {
    /// The same config with `tty.show_text = false` (`run --no-text`).
    ///
    /// Turning text off cannot break a limit, so the result stays validated.
    #[must_use]
    pub fn with_text_off(mut self) -> Self {
        self.config.tty.show_text = false;
        self
    }
}

impl std::ops::Deref for ValidWatchConfig {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
