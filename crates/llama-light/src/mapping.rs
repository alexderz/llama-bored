//! Config layers + a snapshot → one frame of LED colours.

use llama_core::color::{BLACK, Rgb, hex, mix};
use llama_core::wire::SnapshotV1;

use crate::config::{
    AuraCfg, Edge, Gradient, KeyboardCfg, Layer, Layout, LightConfig, Shimmer, Style, Target,
};
use crate::keyboard::keymap::KEYS;
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
    /// The smoothed value (every style but gate).
    smoothed: Option<f32>,
    /// Pulse: where in the breath, 0..1.
    phase: f32,
    /// Threshold and gate: how open, 0..=1, smoothed.
    level: f32,
    /// Gate: seconds since the raw value was last above threshold.
    since_on: Option<f32>,
    /// Peak: the held value and how long it has been held.
    peak: Option<f32>,
    peak_age: f32,
    /// Seconds this layer has run (the shimmer's clock).
    clock: f32,
}

/// Renders frames for one config. Build a new one when the config changes.
///
/// Every call samples the snapshot and advances smoothing, gates and peaks.
/// LED targets are recomputed every `engine.target_period_s` (every call
/// when 0), and the frame returned moves in a straight line from the last
/// frame returned to the newest target over `engine.tween_s`.
pub struct Renderer {
    config: LightConfig,
    states: Vec<LayerState>,
    tokens: TokenRate,
    /// Uncapped frames: the target being tweened to, where the tween
    /// started, and the last frame returned.
    target: Option<Frames>,
    from: Option<Frames>,
    shown: Option<Frames>,
    since_target: f32,
}

impl Renderer {
    /// A renderer with fresh smoothing and pulse state.
    #[must_use]
    pub fn new(config: LightConfig) -> Self {
        let states = vec![LayerState::default(); config.layers.len()];
        let window = config
            .layers
            .iter()
            .map(|layer| layer.rate_window_s)
            .fold(0.0_f32, f32::max);
        Self {
            config,
            states,
            tokens: TokenRate::with_window(window),
            target: None,
            from: None,
            shown: None,
            since_target: 0.0,
        }
    }

    /// The config this renderer draws.
    #[must_use]
    pub fn config(&self) -> &LightConfig {
        &self.config
    }

    /// Each layer's value after smoothing (a gate: its raw-gated opening,
    /// 0..=1; a peak: the held peak), in layer order.
    #[must_use]
    pub fn values(&self) -> Vec<Option<f32>> {
        self.config
            .layers
            .iter()
            .zip(&self.states)
            .map(|(layer, state)| match layer.style {
                Style::Gate => Some(state.level),
                Style::Peak => state.peak,
                _ => state.smoothed,
            })
            .collect()
    }

    /// The Aura frame for `snapshot`, `dt_s` seconds after the last call.
    pub fn aura_frame(&mut self, snapshot: &SnapshotV1, dt_s: f32) -> Vec<Rgb> {
        self.frames(snapshot, dt_s).aura
    }

    /// Both frames for `snapshot`, `dt_s` seconds after the last call. Every
    /// layer advances once per call, whichever device it is on.
    pub fn frames(&mut self, snapshot: &SnapshotV1, dt_s: f32) -> Frames {
        let dt_s = if dt_s.is_finite() { dt_s.max(0.0) } else { 0.0 };
        self.tokens.update(snapshot);
        for (layer, state) in self.config.layers.iter().zip(self.states.iter_mut()) {
            advance(layer, state, snapshot, &self.tokens, dt_s);
        }
        let engine = &self.config.engine;
        let period = engine.target_period_s;
        if self.target.is_some() && period > 0.0 {
            self.since_target += dt_s;
        }
        let due = self.target.is_none() || period <= 0.0 || self.since_target >= period - 1e-3;
        if due {
            let target = self.targets();
            self.from = Some(self.shown.clone().unwrap_or_else(|| target.clone()));
            self.target = Some(target);
            self.since_target = 0.0;
        }
        let tick_s = 1.0 / f32::from(engine.tick_hz.max(1));
        let amount = if engine.tween_s <= 0.0 {
            1.0
        } else {
            ((self.since_target + tick_s) / engine.tween_s).clamp(0.0, 1.0)
        };
        let target = self.target.clone().unwrap_or_else(|| self.targets());
        let shown = match (&self.from, amount < 1.0) {
            (Some(from), true) => tween(from, &target, amount),
            _ => target,
        };
        self.shown = Some(shown.clone());
        Frames {
            aura: cap(&shown.aura, self.config.aura.brightness_max),
            keyboard: cap(&shown.keyboard, self.config.keyboard.brightness_max),
        }
    }

    /// Every layer drawn from its current state, uncapped.
    fn targets(&self) -> Frames {
        let aura = &self.config.aura;
        let under = self.config.base.unwrap_or(BLACK);
        let mut aura_frame = vec![under; aura.frame_len()];
        let mut keyboard_frame = vec![under; KEYS.len()];
        for (layer, state) in self.config.layers.iter().zip(&self.states) {
            let spans = spans(&layer.target, aura);
            if spans.is_empty() {
                continue;
            }
            let frame = if layer.target.is_keyboard() {
                &mut keyboard_frame
            } else {
                &mut aura_frame
            };
            draw(layer, state, &spans, frame);
        }
        Frames {
            aura: aura_frame,
            keyboard: keyboard_frame,
        }
    }
}

/// `from` moved `amount` (0..=1) of the way to `to`, LED by LED.
#[must_use]
pub fn tween(from: &Frames, to: &Frames, amount: f32) -> Frames {
    Frames {
        aura: fade(&from.aura, &to.aura, amount),
        keyboard: fade(&from.keyboard, &to.keyboard, amount),
    }
}

/// One tick's frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Frames {
    /// The Aura header, [`AuraCfg::frame_len`] LEDs.
    pub aura: Vec<Rgb>,
    /// The keyboard, one colour per key in [`KEYS`] order.
    pub keyboard: Vec<Rgb>,
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

/// A keyboard frame of [`NEUTRAL`], after the keyboard's cap.
#[must_use]
pub fn keyboard_neutral_frame(keyboard: &KeyboardCfg) -> Vec<Rgb> {
    cap(&vec![NEUTRAL; KEYS.len()], keyboard.brightness_max)
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

/// LED index spans of `target`, each in gauge order: in the Aura frame for
/// Aura targets, in the keyboard frame ([`KEYS`] order) for keyboard ones.
#[must_use]
pub fn spans(target: &Target, aura: &AuraCfg) -> Vec<Vec<usize>> {
    let n = aura.leds_per_fan;
    match (target, aura.layout) {
        (Target::AuraFans, Layout::Mirrored) => vec![(0..n).collect()],
        (Target::AuraFans, Layout::Chain(fans)) => {
            (0..fans).map(|k| (k * n..(k + 1) * n).collect()).collect()
        }
        (Target::AuraChain { start, end }, Layout::Chain(fans)) if *end <= fans => {
            vec![(start * n..end * n).collect()]
        }
        (Target::KeyboardKeys(keys), _) => vec![keys.clone()],
        (Target::KeyboardAll, _) => vec![(0..KEYS.len()).collect()],
        _ => Vec::new(),
    }
}

/// One step of the asymmetric EMA: `attack_s` while `value` is above the
/// state, `release_s` otherwise. A time constant of 0 jumps.
#[must_use]
pub fn ema(prev: f32, value: f32, attack_s: f32, release_s: f32, dt_s: f32) -> f32 {
    let tau = if value > prev { attack_s } else { release_s };
    if tau <= 0.0 {
        return value;
    }
    let alpha = 1.0 - (-dt_s.max(0.0) / tau).exp();
    prev + (value - prev) * alpha
}

/// Sample and advance one layer by `dt_s`.
fn advance(
    layer: &Layer,
    state: &mut LayerState,
    snapshot: &SnapshotV1,
    tokens: &TokenRate,
    dt_s: f32,
) {
    state.clock += dt_s;
    if layer.fixed.is_some() {
        return;
    }
    let raw = if layer.metric.is_counter_rate() {
        tokens.rate_over(layer.rate_window_s)
    } else {
        layer.metric.read(snapshot, None)
    };
    if layer.style == Style::Gate {
        let above = raw.is_some_and(|v| v > layer.threshold.unwrap_or(0.0));
        state.since_on = if above {
            Some(0.0)
        } else {
            state.since_on.map(|s| s + dt_s)
        };
        let open = state.since_on.is_some_and(|s| s <= layer.hold_s);
        let goal = if open { 1.0 } else { 0.0 };
        state.level = ema(state.level, goal, layer.attack_s, layer.smooth_s, dt_s);
        return;
    }
    let value = match raw {
        None => {
            state.smoothed = None;
            state.peak = None;
            state.level = 0.0;
            return;
        }
        Some(value) => match state.smoothed {
            Some(prev) => ema(prev, value, layer.attack_s, layer.smooth_s, dt_s),
            None => value,
        },
    };
    state.smoothed = Some(value);
    if let Some(threshold) = layer.threshold
        && layer.style != Style::Peak
    {
        let goal = if value > threshold { 1.0 } else { 0.0 };
        state.level = ema(state.level, goal, layer.attack_s, layer.smooth_s, dt_s);
    }
    match layer.style {
        Style::Pulse if !(value < layer.range.0 && layer.idle_color.is_some()) => {
            let pct = layer.scale.percent(value, layer.range);
            let fraction = (pct / 100.0).clamp(0.0, 1.0);
            let hz = PULSE_HZ.0 + (PULSE_HZ.1 - PULSE_HZ.0) * fraction;
            state.phase = (state.phase + hz * dt_s).fract();
        }
        Style::Peak => match state.peak {
            Some(peak) if value < peak => {
                state.peak_age += dt_s;
                if state.peak_age > layer.peak_s {
                    let per_s = (layer.range.1 - layer.range.0) / layer.peak_s;
                    state.peak = Some((peak - per_s * dt_s).max(value));
                }
            }
            _ => {
                state.peak = Some(value);
                state.peak_age = 0.0;
            }
        },
        _ => {}
    }
}

/// Brightness percent for `pct` of range: `brightness`, or with
/// `brightness_to` the straight line from one to the other over 0..=100 %.
fn brightness_at(layer: &Layer, pct: f32) -> f32 {
    match layer.brightness_to {
        None => layer.brightness,
        Some(hi) => {
            let fraction = (pct / 100.0).clamp(0.0, 1.0);
            layer.brightness + (hi - layer.brightness) * fraction
        }
    }
}

/// Shimmer factor at `clock_s`: `1 + depth · sin(2π · hz · t)`.
#[must_use]
pub fn shimmer_factor(shimmer: Option<Shimmer>, clock_s: f32) -> f32 {
    match shimmer {
        None => 1.0,
        Some(Shimmer { depth, hz }) => 1.0 + depth * (std::f32::consts::TAU * hz * clock_s).sin(),
    }
}

fn draw(layer: &Layer, state: &LayerState, spans: &[Vec<usize>], frame: &mut [Rgb]) {
    if let Some(color) = layer.fixed {
        let color = dim(color, layer.brightness);
        for span in spans {
            paint(frame, span, color);
        }
        return;
    }
    if layer.style == Style::Gate {
        let Some(color) = layer.gate_color else {
            return;
        };
        if state.level <= 0.0 {
            return;
        }
        let percent = layer.brightness * shimmer_factor(layer.shimmer, state.clock);
        let color = dim(color, percent);
        for span in spans {
            blend(frame, span, color, state.level);
        }
        return;
    }
    let value = state.smoothed;
    let idle = match value {
        None => Some(layer.idle_color.unwrap_or(NEUTRAL)),
        Some(v) if v < layer.range.0 => layer.idle_color,
        Some(_) => None,
    };
    if let Some(color) = idle {
        let color = dim(color, layer.brightness);
        for span in spans {
            paint(frame, span, color);
        }
        return;
    }
    let Some(value) = value else { return };
    let pct = layer.scale.percent(value, layer.range);
    let brightness = brightness_at(layer, pct);
    let at = |pct: f32| dim(layer.palette.color(pct), brightness);
    let color = at(pct);
    // Threshold: how far the entry is drawn over what is under it.
    let level = if layer.threshold.is_some() {
        state.level
    } else {
        1.0
    };
    match layer.style {
        Style::Solid | Style::Pulse => {
            if level <= 0.0 {
                return;
            }
            let color = if layer.style == Style::Pulse {
                dim(color, pulse_level(state.phase) * 100.0)
            } else {
                color
            };
            for span in spans {
                blend(frame, span, color, level);
            }
        }
        Style::Gauge => {
            for span in spans {
                let n = span.len();
                let fill_pct = pct * 100.0 / layer.fill_to;
                let fill = (fill_pct / 100.0).clamp(0.0, 1.0) * n as f32;
                let lit = gauge_count(fill_pct, n);
                for (i, led) in span.iter().enumerate() {
                    let amount = match layer.edge {
                        Edge::Round => {
                            if i < lit {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        Edge::Fractional => (fill - i as f32).clamp(0.0, 1.0),
                    };
                    if amount <= 0.0 {
                        continue;
                    }
                    let lit_color = match layer.gradient {
                        Gradient::Value => color,
                        Gradient::Position => at((i as f32 + 0.5) / n as f32 * layer.fill_to),
                    };
                    blend(frame, &[*led], lit_color, amount);
                }
            }
        }
        Style::Ladder => {
            for span in spans {
                for (i, led) in span.iter().enumerate() {
                    let Some(&top) = layer.thresholds.get(i) else {
                        continue;
                    };
                    let start = if i == 0 {
                        layer.range.0
                    } else {
                        layer.thresholds[i - 1]
                    };
                    let amount = match layer.edge {
                        Edge::Round => {
                            if value >= top {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        Edge::Fractional => ((value - start) / (top - start)).clamp(0.0, 1.0),
                    };
                    if amount <= 0.0 {
                        continue;
                    }
                    let rung_color = match layer.gradient {
                        Gradient::Value => color,
                        Gradient::Position => at(layer.scale.percent(top, layer.range)),
                    };
                    blend(frame, &[*led], rung_color, amount);
                }
            }
        }
        Style::Peak => {
            let Some(peak) = state.peak else { return };
            if peak <= layer.threshold.unwrap_or(layer.range.0) {
                return;
            }
            let color = at(layer.scale.percent(peak, layer.range));
            for span in spans {
                paint(frame, span, color);
            }
        }
        Style::Gate => {}
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

/// `color` over what `frame` has at `leds`, by `amount` (0..=1).
fn blend(frame: &mut [Rgb], leds: &[usize], color: Rgb, amount: f32) {
    if amount >= 1.0 {
        paint(frame, leds, color);
        return;
    }
    for index in leds {
        if let Some(led) = frame.get_mut(*index) {
            *led = mix(*led, color, amount);
        }
    }
}

fn paint(frame: &mut [Rgb], leds: &[usize], color: Rgb) {
    for index in leds {
        if let Some(led) = frame.get_mut(*index) {
            *led = color;
        }
    }
}
