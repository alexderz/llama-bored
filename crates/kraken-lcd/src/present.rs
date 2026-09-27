//! Snapshot + history → [`View`], with bands, hysteresis, and the dial.
//!
//! Pure: no clock and no I/O. Callers pass the previous [`View`] when they
//! have one. The same inputs always produce the same frame key. Writer-local
//! memory (activity history, the tokens cascade, the frame clock and the
//! peg's particles) rides along in [`View::dial_state`] so the next call can
//! ingest; it is not part of that key. Dial bars are recomputed every call,
//! with no hysteresis.
//!
//! `present` reads the writer config: `[dial]` sets the bar tiers and
//! coverage, `upload.mode` picks stream smoothing, and `display.variant` is
//! copied onto the frame. Fixtures can still say `a3`.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use llama_core::detail::ModelDetail;

use crate::activity::{ACTIVITY_MAX, ActivityDial, BARS};
use crate::anim::Anim;
use crate::collector::{AiState, Snapshot};
use crate::config::{Bands, Config, StepMargin, UploadMode};
use crate::history::History;
use crate::tokens::{TokenChart, TokenFeed};

/// Load colour for the ring and the three history blocks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Band {
    /// Coverage is below `bands.fill_min_coverage`. The ring never uses this.
    Filling,
    /// Lowest load band.
    Quiet,
    /// Entered at `bands.enter[0]`.
    Light,
    /// Entered at `bands.enter[1]`.
    Busy,
    /// Entered at `bands.enter[2]`.
    FlatOut,
}

/// llama-swap reachability drawn on the frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Ai {
    /// llama-swap could not be read.
    Down,
    /// Reachable, nothing loaded.
    Idle,
    /// One or more models loaded.
    Loaded,
    /// Watcher snapshot missing, stale, or invalid.
    NoData,
}

/// Layout selected by `display.variant`. Render reads this off the frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    /// Halo layout.
    #[default]
    A1,
    /// Dial-first layout.
    A3,
}

fn no_data_dial() -> [Option<u8>; BARS] {
    [None; BARS]
}

fn default_scale() -> Vec<ScaleTier> {
    scale_of(&ActivityDial::default(), None)
}

/// One dial tier on the time scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScaleTier {
    /// Bars in the tier.
    pub bars: u8,
    /// Window of one bar, tenths of a second.
    pub width_ds: u32,
    /// How full the tier's newest bar is, per mille. Set for the 1 min and
    /// 5 min tiers in stream mode, where the fill sweep is drawn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<u16>,
}

/// Tiers whose newest bar carries the fill sweep: 1 min and slower.
const SWEEP_MIN_DS: u32 = 600;

fn scale_of(dial: &ActivityDial, fill: Option<&[f32]>) -> Vec<ScaleTier> {
    dial.tiers()
        .iter()
        .enumerate()
        .map(|(index, tier)| {
            let width_ds = u32::try_from(tier.width.saturating_mul(5)).unwrap_or(u32::MAX);
            let fill = fill
                .filter(|_| width_ds >= SWEEP_MIN_DS)
                .and_then(|fill| fill.get(index))
                .map(|f| (f.clamp(0.0, 1.0) * 1000.0).round() as u16);
            ScaleTier {
                bars: u8::try_from(tier.bars).unwrap_or(u8::MAX),
                width_ds,
                fill,
            }
        })
        .collect()
}

/// Writer-local memory carried to the next [`present`].
#[derive(Clone, Debug)]
pub struct Memory {
    /// Activity samples for the dial bars.
    pub activity: ActivityDial,
    /// Tokens · 24 h cascade.
    pub chart: TokenChart,
    /// Counter deltas for the cascade.
    pub feed: TokenFeed,
    /// Writer `Instant` paired with a sample's `t_mono_ns`.
    anchor: Option<(Instant, u64)>,
    /// Frame clock, stream smoothing and the peg's particles.
    pub anim: Anim,
}

impl Memory {
    /// Empty memory on the `[dial]` settings.
    #[must_use]
    pub fn new(settings: &crate::config::Dial) -> Self {
        Self {
            activity: ActivityDial::new(settings),
            chart: TokenChart::default(),
            feed: TokenFeed::new(settings.max_gap_s),
            anchor: None,
            anim: Anim::default(),
        }
    }

    /// `t_mono_ns` corresponding to `instant`, once a reading anchored it.
    #[must_use]
    pub fn project_ns(&self, instant: Instant) -> Option<u64> {
        let (origin, t_mono_ns) = self.anchor?;
        let delta = instant.saturating_duration_since(origin);
        let delta_ns = u64::try_from(delta.as_nanos()).unwrap_or(u64::MAX);
        Some(t_mono_ns.saturating_add(delta_ns))
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new(&crate::config::Dial::default())
    }
}

/// [`Memory`] as a [`View`] field.
///
/// Equality is always true and hashing writes nothing, so the memory is not
/// part of the upload key. [`View`] derives [`Eq`] and [`Hash`], and every
/// other field participates.
#[derive(Clone, Debug, Default)]
pub struct DialMemory(pub Memory);

impl PartialEq for DialMemory {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for DialMemory {}

impl std::hash::Hash for DialMemory {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

impl From<Memory> for DialMemory {
    fn from(memory: Memory) -> Self {
        Self(memory)
    }
}

impl std::ops::Deref for DialMemory {
    type Target = Memory;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DialMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Frame key. Compared with `==` to decide whether a new frame is worth uploading.
///
/// [`Self::dial_state`] is writer memory for the next [`present`]. It is
/// skipped by serde and ignored by [`DialMemory`]'s equality, so moving
/// buckets or an animation frame do not upload on their own. Any other
/// field, including [`Self::dial`], [`Self::tokens`] and [`Self::variant`],
/// is part of the key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// Activity on the ring, 0..=125 (100 is nominal). The snapshot's
    /// activity, or the last-minute mean load from an older watcher.
    pub ring_pct: Option<u8>,
    /// Band of the raw ring value. Never [`Band::Filling`]. Not drawn since
    /// the ring is coloured by position.
    pub ring_band: Option<Band>,
    /// 15 m, 2 h, and 24 h load bands. Not drawn since the tokens chart took
    /// their slot.
    pub blocks: [Band; 3],
    /// Quantised coolant temperature, °C.
    pub coolant_c: Option<i16>,
    /// Quantised CPU temperature, °C.
    pub cpu_c: Option<i16>,
    /// Quantised GPU temperature, °C.
    pub gpu_c: Option<i16>,
    /// Quantised plain CPU percent.
    pub cpu_pct: Option<u8>,
    /// Quantised memory percent.
    pub mem_pct: Option<u8>,
    /// Mapped from [`AiState`].
    pub ai: Ai,
    /// Display-safe model names, in snapshot order.
    pub models: Vec<String>,
    /// `models.len()`, saturated at [`u8::MAX`].
    pub model_count: u8,
    /// Tuning detail of the first model. `None` from an older watcher.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ModelDetail>,
    /// 24 dial bars, newest at 12 o'clock: mean activity (0..=125) over each
    /// bar's window. `None` is no data. No hysteresis.
    #[serde(default = "no_data_dial")]
    pub dial: [Option<u8>; BARS],
    /// The dial's tiers, for the time scale and the fill sweeps.
    #[serde(default = "default_scale")]
    pub scale: Vec<ScaleTier>,
    /// Tokens · 24 h in centi-tok/s, newest first, at most 40. Empty is no data.
    #[serde(default)]
    pub tokens: Vec<u32>,
    /// Layout variant from `display.variant`.
    #[serde(default)]
    pub variant: Variant,
    /// Writer memory. Not a frame-key field.
    #[serde(skip)]
    pub dial_state: DialMemory,
}

impl Default for View {
    fn default() -> Self {
        Self {
            ring_pct: None,
            ring_band: None,
            blocks: [Band::Filling; 3],
            coolant_c: None,
            cpu_c: None,
            gpu_c: None,
            cpu_pct: None,
            mem_pct: None,
            ai: Ai::Down,
            models: Vec::new(),
            model_count: 0,
            detail: None,
            dial: no_data_dial(),
            scale: default_scale(),
            tokens: Vec::new(),
            variant: Variant::A1,
            dial_state: DialMemory::default(),
        }
    }
}

/// Stream-mode ring smoothing per frame, as the design's 10 fps demo.
const RING_SMOOTH: f32 = 0.35;

/// Warm a view for a still picture (`render-once`, goldens, previews): the
/// peg's particles after four seconds at the ring value, on the variant's ring.
pub fn warm_still(view: &mut View) {
    let value = view.ring_pct.map_or(0.0, f32::from);
    let radius = crate::render::ring_radius(view.variant);
    view.dial_state.anim = if view.ai == Ai::NoData || view.ring_pct.is_none() {
        Anim {
            frame: 40,
            ..Anim::default()
        }
    } else {
        Anim::still(value, radius, 40, 7)
    };
}

/// Build the frame key for one tick.
///
/// A number is re-quantised to `round(raw / step) * step` when there is no
/// shown value or `raw` is outside `[shown - step/2 - margin, shown + step/2 + margin]`.
/// `None` replaces the shown value immediately. A block whose coverage is below
/// `bands.fill_min_coverage` is [`Band::Filling`] and does not consult the
/// previous band. The ring band follows the raw ring value and is never
/// [`Band::Filling`]. In stream mode the ring is smoothed per frame instead
/// of held. With no previous view the memory comes from [`Memory::new`] on
/// `config.dial`; later ticks keep it. `display.variant` is copied onto the
/// frame every call.
#[must_use]
pub fn present(
    snapshot: &Snapshot,
    history: &History,
    previous: Option<&View>,
    config: &Config,
) -> View {
    let streaming = config.upload.mode == UploadMode::Stream;
    // The ring tracks power-weighted activity (0..=125). An older snapshot
    // has no activity, and the ring stays the 60 s mean of `load`.
    let ring_raw = match finite(snapshot.activity) {
        Some(activity) => Some(activity),
        None => finite(history.ring_mean(snapshot.t_mono)),
    }
    .map(|raw| raw.clamp(0.0, ACTIVITY_MAX));
    let windows = history.windows(snapshot.t_mono);
    let blocks = std::array::from_fn(|index| {
        block_band(
            windows[index].mean,
            windows[index].coverage,
            previous.map(|view| view.blocks[index]),
            &config.bands,
        )
    });
    // The untruncated name when the watcher sent one. Render fits it.
    let models: Vec<String> = snapshot
        .models
        .iter()
        .map(|model| {
            model
                .full_name
                .clone()
                .unwrap_or_else(|| model.name.clone())
        })
        .collect();
    let detail = snapshot
        .models
        .first()
        .and_then(|model| model.detail.clone());
    let mut memory = match previous {
        Some(view) => view.dial_state.clone(),
        None => DialMemory(Memory::new(&config.dial)),
    };
    // Bars take the snapshot's own activity, or its load from an older watcher.
    let bar_sample = finite(snapshot.activity).or_else(|| finite(snapshot.load));
    if let Some(reading) = snapshot.tokens {
        memory.anchor = Some((snapshot.t_mono, reading.t_mono_ns));
        let step = memory.feed.accept(reading);
        if step.fresh
            && let Some(value) = bar_sample
        {
            memory.activity.add(reading.t_mono_ns, value);
        }
        if let Some((tok, ms)) = step.interval {
            memory.chart.add(tok, ms);
        }
    }
    let now_ns = snapshot
        .tokens
        .map(|reading| reading.t_mono_ns)
        .or_else(|| memory.project_ns(snapshot.t_mono))
        .unwrap_or(0);
    memory.activity.retain_through(now_ns);
    let step = if streaming {
        1
    } else {
        config.hysteresis.ring.step.max(1)
    };
    let dial = memory
        .activity
        .means(now_ns)
        .map(|mean| mean.map(|mean| clamp_u8(quantise(mean, step))));
    // The sweeps need a clock: a reading now, or an anchor from an earlier one.
    let clocked = snapshot.tokens.is_some() || memory.anchor.is_some();
    let fill = memory.activity.fill(now_ns);
    let scale = scale_of(
        &memory.activity,
        (streaming && clocked).then_some(fill.as_slice()),
    );
    let tokens = memory
        .chart
        .points()
        .into_iter()
        .map(|rate| {
            let centi = (f64::from(rate) * 100.0).round();
            if centi.is_finite() && centi > 0.0 {
                centi.min(f64::from(u32::MAX)) as u32
            } else {
                0
            }
        })
        .collect();

    let ring_pct = if streaming {
        let shown = ring_raw.map(|raw| {
            let from = memory.anim.shown.unwrap_or(raw);
            let next = from + (raw - from) * RING_SMOOTH;
            if (raw - next).abs() < 0.05 { raw } else { next }
        });
        memory.anim.shown = shown;
        shown.map(|value| clamp_u8(quantise(value, 1)))
    } else {
        memory.anim.shown = None;
        hold_u8(
            ring_raw,
            previous.and_then(|view| view.ring_pct),
            config.hysteresis.ring,
        )
    };
    let ai = map_ai(snapshot.ai);
    let variant = map_variant(config.display.variant);
    memory.anim.frame = memory.anim.frame.wrapping_add(1);
    let peg = memory
        .anim
        .shown
        .or_else(|| ring_pct.map(f32::from))
        .filter(|_| ai != Ai::NoData);
    match peg {
        Some(value) => {
            let radius = crate::render::ring_radius(variant);
            memory.anim.particles.step(value, radius);
        }
        // Off when the view is stale or no data.
        None => memory.anim.particles.clear(),
    }
    View {
        ring_pct,
        ring_band: ring_raw
            .map(|raw| move_band(raw, previous.and_then(|view| view.ring_band), &config.bands)),
        blocks,
        coolant_c: hold_i16(
            finite(snapshot.coolant_c),
            previous.and_then(|view| view.coolant_c),
            config.hysteresis.temp,
        ),
        cpu_c: hold_i16(
            finite(snapshot.cpu_c),
            previous.and_then(|view| view.cpu_c),
            config.hysteresis.temp,
        ),
        gpu_c: hold_i16(
            finite(snapshot.gpu_c),
            previous.and_then(|view| view.gpu_c),
            config.hysteresis.temp,
        ),
        cpu_pct: hold_u8(
            finite(snapshot.cpu_pct),
            previous.and_then(|view| view.cpu_pct),
            config.hysteresis.percent,
        ),
        mem_pct: hold_u8(
            finite(snapshot.mem_pct),
            previous.and_then(|view| view.mem_pct),
            config.hysteresis.percent,
        ),
        ai,
        model_count: u8::try_from(models.len()).unwrap_or(u8::MAX),
        models,
        detail,
        dial,
        scale,
        tokens,
        variant,
        dial_state: memory,
    }
}

fn finite(value: Option<f32>) -> Option<f32> {
    value.filter(|sample| sample.is_finite())
}

fn map_ai(ai: AiState) -> Ai {
    match ai {
        AiState::Down => Ai::Down,
        AiState::Idle => Ai::Idle,
        AiState::Loaded => Ai::Loaded,
        AiState::NoData => Ai::NoData,
    }
}

fn map_variant(variant: crate::config::Variant) -> Variant {
    match variant {
        crate::config::Variant::A1 => Variant::A1,
        crate::config::Variant::A3 => Variant::A3,
    }
}

fn hold_u8(raw: Option<f32>, shown: Option<u8>, pair: StepMargin) -> Option<u8> {
    let raw = raw?;
    let quantised = match shown {
        Some(shown) if inside(raw, i32::from(shown), pair) => i32::from(shown),
        _ => quantise(raw, pair.step),
    };
    Some(clamp_u8(quantised))
}

fn hold_i16(raw: Option<f32>, shown: Option<i16>, pair: StepMargin) -> Option<i16> {
    let raw = raw?;
    let quantised = match shown {
        Some(shown) if inside(raw, i32::from(shown), pair) => i32::from(shown),
        _ => quantise(raw, pair.step),
    };
    Some(clamp_i16(quantised))
}

fn inside(raw: f32, shown: i32, pair: StepMargin) -> bool {
    let shown = f64::from(shown);
    let slack = f64::from(pair.step) / 2.0 + f64::from(pair.margin);
    let raw = f64::from(raw);
    raw >= shown - slack && raw <= shown + slack
}

fn quantise(raw: f32, step: u32) -> i32 {
    debug_assert!(step > 0);
    let step = f64::from(step);
    let rounded = ((f64::from(raw) / step).round() * step).round();
    if !rounded.is_finite() || rounded >= f64::from(i32::MAX) {
        return if rounded.is_sign_negative() {
            i32::MIN
        } else {
            i32::MAX
        };
    }
    if rounded <= f64::from(i32::MIN) {
        return i32::MIN;
    }
    // Finite integer inside the i32 range: the float is exact up to 2^53.
    rounded as i32
}

fn clamp_u8(value: i32) -> u8 {
    match u8::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => 0,
        Err(_) => u8::MAX,
    }
}

fn clamp_i16(value: i32) -> i16 {
    match i16::try_from(value) {
        Ok(value) => value,
        Err(_) if value < 0 => i16::MIN,
        Err(_) => i16::MAX,
    }
}

/// `Filling` when coverage is below the config threshold or there is no mean.
fn block_band(mean: Option<f32>, coverage: f32, previous: Option<Band>, bands: &Bands) -> Band {
    if !coverage.is_finite() || f64::from(coverage) < bands.fill_min_coverage {
        return Band::Filling;
    }
    let Some(mean) = mean.filter(|value| value.is_finite()) else {
        return Band::Filling;
    };
    move_band(mean, previous, bands)
}

/// Move up at the next enter value, then down only below the current leave value.
fn move_band(value: f32, previous: Option<Band>, bands: &Bands) -> Band {
    let mut index = match previous {
        Some(Band::Quiet) => 0,
        Some(Band::Light) => 1,
        Some(Band::Busy) => 2,
        Some(Band::FlatOut) => 3,
        Some(Band::Filling) | None => return classify(value, bands.enter),
    };
    let enter = [
        0.0,
        f64::from(bands.enter[0]),
        f64::from(bands.enter[1]),
        f64::from(bands.enter[2]),
    ];
    let value = f64::from(value);
    while index < 3 && value >= enter[index + 1] {
        index += 1;
    }
    let margin = i32::from(bands.margin);
    let leave = [
        i32::from(bands.enter[0]) - margin,
        i32::from(bands.enter[1]) - margin,
        i32::from(bands.enter[2]) - margin,
    ];
    while index > 0 && value < f64::from(leave[index - 1]) {
        index -= 1;
    }
    [Band::Quiet, Band::Light, Band::Busy, Band::FlatOut][index]
}

fn classify(value: f32, enter: [u8; 3]) -> Band {
    let value = f64::from(value);
    if value >= f64::from(enter[2]) {
        Band::FlatOut
    } else if value >= f64::from(enter[1]) {
        Band::Busy
    } else if value >= f64::from(enter[0]) {
        Band::Light
    } else {
        Band::Quiet
    }
}

#[cfg(test)]
mod tests {
    use super::{Band, block_band};
    use crate::config::Config;

    #[test]
    fn filling_at_coverage_0_49_and_quiet_at_0_5() {
        let bands = Config::default().bands;
        assert_eq!(
            block_band(Some(0.0), 0.49, Some(Band::Quiet), &bands),
            Band::Filling,
            "0.49 is below fill_min_coverage"
        );
        assert_eq!(
            block_band(Some(95.0), 0.49, Some(Band::FlatOut), &bands),
            Band::Filling,
            "Filling ignores the previous band and the mean"
        );
        assert_eq!(
            block_band(Some(0.0), 0.5, Some(Band::Filling), &bands),
            Band::Quiet,
            "0.5 is not below the threshold, so a quiet mean is Quiet"
        );
        assert_eq!(
            block_band(Some(f32::NAN), 1.0, Some(Band::Busy), &bands),
            Band::Filling,
            "a non-finite mean has no band"
        );
        assert_eq!(
            block_band(Some(10.0), f32::NAN, Some(Band::Quiet), &bands),
            Band::Filling,
            "non-finite coverage is not enough data"
        );
    }

    #[test]
    fn no_data_state_maps_to_no_data_ai() {
        use std::collections::BTreeSet;
        use std::time::{Instant, SystemTime};

        use llama_core::sample::{AiState, Snapshot};

        use super::{Ai, present};
        use crate::history::History;

        let now = Instant::now();
        let history = History::new(now);
        let snapshot = Snapshot {
            t_mono: now,
            t_wall: SystemTime::UNIX_EPOCH,
            load: None,
            activity: None,
            cpu_pct: None,
            cpu_topk_pct: None,
            gpu_pct: None,
            mem_pct: None,
            coolant_c: None,
            cpu_c: None,
            gpu_c: None,
            ai: AiState::NoData,
            models: Vec::new(),
            tokens: None,
            errors: BTreeSet::new(),
        };
        let view = present(&snapshot, &history, None, &Config::default());
        assert_eq!(view.ai, Ai::NoData);
    }
}
