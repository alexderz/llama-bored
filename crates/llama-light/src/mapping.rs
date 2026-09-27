//! Config layers + a snapshot → one frame of LED colours.

use std::ops::Range;

use llama_core::color::{BLACK, Rgb, hex, mix};
use llama_core::wire::SnapshotV1;

use crate::config::{AuraCfg, Layer, Layout, LightConfig, Style, Target};
use crate::metric::TokenRate;
use crate::palette::dim;

/// Dim neutral grey: a missing metric with no `idle_color`, the stale fade
/// target, and the colour `restore` leaves on the header. Scaled by
/// `brightness_max` like every other colour.
pub const NEUTRAL: Rgb = hex(0x282828);

/// Pulse: slowest and fastest breath, in hertz.
pub const PULSE_HZ: (f32, f32) = (0.2, 2.0);
/// Pulse: dimmest point of a breath, as a fraction of full.
pub const PULSE_FLOOR: f32 = 0.35;

#[derive(Clone, Debug, Default)]
struct LayerState {
    smoothed: Option<f32>,
    phase: f32,
}

/// Renders frames for one config. Build a new one when the config changes.
pub struct Renderer {
    config: LightConfig,
    states: Vec<LayerState>,
    tokens: TokenRate,
}

impl Renderer {
    /// A renderer with fresh smoothing and pulse state.
    #[must_use]
    pub fn new(config: LightConfig) -> Self {
        let states = vec![LayerState::default(); config.layers.len()];
        Self {
            config,
            states,
            tokens: TokenRate::default(),
        }
    }

    /// The config this renderer draws.
    #[must_use]
    pub fn config(&self) -> &LightConfig {
        &self.config
    }

    /// The Aura frame for `snapshot`, `dt_s` seconds after the last call.
    pub fn aura_frame(&mut self, snapshot: &SnapshotV1, dt_s: f32) -> Vec<Rgb> {
        let tokens_rate = self.tokens.update(snapshot);
        let aura = self.config.aura.clone();
        let mut frame = vec![BLACK; aura.frame_len()];
        for (layer, state) in self.config.layers.iter().zip(self.states.iter_mut()) {
            let spans = spans(&layer.target, &aura);
            if spans.is_empty() {
                continue;
            }
            let raw = layer.metric.read(snapshot, tokens_rate);
            let value = smooth(state, raw, layer.smooth_s, dt_s);
            draw(layer, state, value, dt_s, &spans, &mut frame);
        }
        cap(&frame, aura.brightness_max)
    }
}

/// `frame` with every LED scaled by `brightness_max` percent.
#[must_use]
pub fn cap(frame: &[Rgb], brightness_max: u8) -> Vec<Rgb> {
    frame
        .iter()
        .map(|led| dim(*led, f32::from(brightness_max)))
        .collect()
}

/// A frame of [`NEUTRAL`] for `aura`, after the cap.
#[must_use]
pub fn neutral_frame(aura: &AuraCfg) -> Vec<Rgb> {
    cap(&vec![NEUTRAL; aura.frame_len()], aura.brightness_max)
}

/// `from` faded toward `to` by `amount` (0..=1), LED by LED.
#[must_use]
pub fn fade(from: &[Rgb], to: &[Rgb], amount: f32) -> Vec<Rgb> {
    to.iter()
        .enumerate()
        .map(|(index, target)| {
            let start = from.get(index).copied().unwrap_or(*target);
            mix(start, *target, amount.clamp(0.0, 1.0))
        })
        .collect()
}

/// LED index spans of `target` in the Aura frame. Keyboard targets have none.
#[must_use]
// A one-span list is the point: the caller iterates spans.
#[allow(clippy::single_range_in_vec_init)]
pub fn spans(target: &Target, aura: &AuraCfg) -> Vec<Range<usize>> {
    let n = aura.leds_per_fan;
    match (target, aura.layout) {
        (Target::AuraFans, Layout::Mirrored) => vec![0..n],
        (Target::AuraFans, Layout::Chain(fans)) => (0..fans).map(|k| k * n..(k + 1) * n).collect(),
        (Target::AuraChain { start, end }, Layout::Chain(fans)) if *end <= fans => {
            vec![start * n..end * n]
        }
        _ => Vec::new(),
    }
}

fn smooth(state: &mut LayerState, raw: Option<f32>, smooth_s: f32, dt_s: f32) -> Option<f32> {
    let Some(value) = raw else {
        state.smoothed = None;
        return None;
    };
    let next = match state.smoothed {
        Some(prev) if smooth_s > 0.0 => {
            let alpha = 1.0 - (-dt_s.max(0.0) / smooth_s).exp();
            prev + (value - prev) * alpha
        }
        _ => value,
    };
    state.smoothed = Some(next);
    Some(next)
}

fn draw(
    layer: &Layer,
    state: &mut LayerState,
    value: Option<f32>,
    dt_s: f32,
    spans: &[Range<usize>],
    frame: &mut [Rgb],
) {
    let idle = match value {
        None => Some(layer.idle_color.unwrap_or(NEUTRAL)),
        Some(v) if v < layer.range.0 => layer.idle_color,
        Some(_) => None,
    };
    if let Some(color) = idle {
        let color = dim(color, f32::from(layer.brightness));
        for span in spans {
            paint(frame, span.clone(), color);
        }
        return;
    }
    let Some(value) = value else { return };
    let pct = layer.scale.percent(value, layer.range);
    let color = dim(layer.palette.color(pct), f32::from(layer.brightness));
    match layer.style {
        Style::Solid => {
            for span in spans {
                paint(frame, span.clone(), color);
            }
        }
        Style::Gauge => {
            for span in spans {
                let lit = gauge_count(pct, span.len());
                paint(frame, span.start..span.start + lit, color);
            }
        }
        Style::Pulse => {
            let fraction = (pct / 100.0).clamp(0.0, 1.0);
            let hz = PULSE_HZ.0 + (PULSE_HZ.1 - PULSE_HZ.0) * fraction;
            state.phase = (state.phase + hz * dt_s.max(0.0)).fract();
            let level = pulse_level(state.phase);
            let color = dim(color, level * 100.0);
            for span in spans {
                paint(frame, span.clone(), color);
            }
        }
    }
}

/// Brightness fraction at `phase` (0..1) of a breath: full at 0, the floor at 0.5.
#[must_use]
pub fn pulse_level(phase: f32) -> f32 {
    let wave = 0.5 + 0.5 * (std::f32::consts::TAU * phase).cos();
    PULSE_FLOOR + (1.0 - PULSE_FLOOR) * wave
}

/// LEDs lit for `pct` of a span of `len`. Rounded; any value above 0 %
/// lights at least one LED, and 100 % or more lights all of them.
#[must_use]
pub fn gauge_count(pct: f32, len: usize) -> usize {
    let fraction = (pct / 100.0).clamp(0.0, 1.0);
    let lit = (fraction * len as f32).round() as usize;
    if pct > 0.0 { lit.max(1).min(len) } else { 0 }
}

fn paint(frame: &mut [Rgb], range: Range<usize>, color: Rgb) {
    let end = range.end.min(frame.len());
    for led in frame.iter_mut().take(end).skip(range.start) {
        *led = color;
    }
}
