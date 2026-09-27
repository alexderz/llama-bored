//! V3b Plasma Blackbody ring: a 270° speedometer, gap at 6 o'clock.
//!
//! Linear 0–125 from 225° (7:30) to 135° (4:30), 2.16° per point. The arc is
//! coloured by position with [`act_color`], so a colour always sits at the
//! same place. The dim scale is the same ramp 80 % toward black, then the
//! redline track. The value arc ends in a round cap whose outer edge is the
//! value angle; its last 30° blend into a light comet head, with a glow
//! trailing 45°. From 120 the head goes white-hot. Break lines cut the whole
//! stroke every 10 % (and at 105 and 115); the 100 gate is a light tick.
//!
//! The renderer notes allow a sweep gradient in place of 1.5° pieces; tiny-skia
//! has one, so each coloured run is one stroke. Glow is stacked wide strokes
//! at low alpha with `BlendMode::Plus`, no blur.

use tiny_skia::{
    BlendMode, Color, GradientStop, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, Point,
    Shader, SpreadMode, Stroke, SweepGradient, Transform,
};

use super::Rgb;
use super::color::{BLACK, HOT_GOLD, WHITE, act_color, cold_tip_lift, hex, mix};
use super::geometry::LayoutGeometry;
use super::text::{GlyphCache, Pen, TextStyle, Weight};
use crate::anim::{RING_MAX, RING_START_DEG, RING_SWEEP_DEG, pulse, ring_angle};

const CENTRE: f32 = 160.0;
/// Degrees per activity point.
const PER: f32 = RING_SWEEP_DEG / RING_MAX;
/// Redline track, 100–125.
const BB_TRACK: Rgb = hex(0x3A1216);
/// The 100 gate.
const GATE: Rgb = hex(0xF4F4F2);
/// Hot-core halo: hot pale gold, not white.
const HALO: Rgb = HOT_GOLD;

/// Paint the track, the value arc, the glow and the break lines.
///
/// `value` is activity 0–125; `None` leaves the bare track (no data). 0 is a
/// single dot at the start. `t` is the frame clock in seconds.
pub(super) fn draw(pixmap: &mut Pixmap, value: Option<f32>, geom: &LayoutGeometry, t: f32) {
    let r = geom.ring_radius;
    let w = geom.ring_stroke;
    draw_track(pixmap, r, w);
    if let Some(v) = value.filter(|v| v.is_finite()) {
        draw_value(pixmap, v.clamp(0.0, RING_MAX), r, w, t);
    }
    draw_breaks(pixmap, r, w);
}

/// The value in the gap: `12 px / 800`, centred at x 160.
pub(super) fn draw_readout(
    pixmap: &mut Pixmap,
    cache: &mut GlyphCache,
    value: Option<f32>,
    geom: &LayoutGeometry,
) {
    let Some(v) = value.filter(|v| v.is_finite()) else {
        return;
    };
    let v = v.clamp(0.0, RING_MAX);
    let color = readout_color(v);
    let text = format!("{}%", v.round() as i32);
    let px = geom.readout_px;
    let width = cache.measure(&text, px, Weight::ExtraBold, 0.0).advance;
    cache.draw(
        pixmap,
        &text,
        Pen {
            x: CENTRE - width * 0.5,
            baseline: geom.readout_baseline,
        },
        TextStyle {
            px,
            weight: Weight::ExtraBold,
            color,
            tracking_em: 0.0,
        },
    );
}

/// `mix(act_color(v), white, 0.35)` up to 100, `act_color(max(112, v))` above.
#[must_use]
pub fn readout_color(v: f32) -> Rgb {
    if v > 100.0 {
        act_color(v.max(112.0))
    } else {
        mix(act_color(v), WHITE, 0.35)
    }
}

/// Head colour: `mix(act_color(v), white, 0.5)`, capped at 15 % under 50 so
/// the cold comet head stays saturated (T62). Above 100 it mixes toward
/// [`HOT_GOLD`] instead of white.
#[must_use]
pub fn head_color(v: f32) -> Rgb {
    let toward = if v > 100.0 { HOT_GOLD } else { WHITE };
    mix(act_color(v), toward, cold_tip_lift(v, 0.5))
}

/// Degrees the leading cap's centre sits short of the value angle.
#[must_use]
pub fn cap_deg(radius: f32, stroke: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    (stroke * 0.5 / radius).to_degrees()
}

fn draw_track(pixmap: &mut Pixmap, r: f32, w: f32) {
    let stops: Vec<(f32, Rgb)> = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        .into_iter()
        .map(|p| (p, mix(act_color(p), BLACK, 0.8)))
        .collect();
    if let (Some(path), Some(shader)) = (
        arc_path(r, ring_angle(0.0), ring_angle(100.0)),
        sweep(&stops),
    ) {
        stroke_shader(
            pixmap,
            &path,
            shader,
            w,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
    if let Some(path) = arc_path(r, ring_angle(100.0), ring_angle(RING_MAX)) {
        stroke_solid(
            pixmap,
            &path,
            BB_TRACK,
            1.0,
            w,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
}

fn draw_value(pixmap: &mut Pixmap, v: f32, r: f32, w: f32, t: f32) {
    let start = ring_angle(0.0);
    if v <= 0.0 {
        dot(
            pixmap,
            r,
            start,
            w * 0.5,
            act_color(0.0),
            1.0,
            BlendMode::SourceOver,
        );
        return;
    }
    let end = ring_angle(v) - cap_deg(r, w);
    let p_end = (end - RING_START_DEG) / PER;
    let col = act_color(v);
    let tip = head_color(v);
    let pulse = pulse(t);

    // Trailing glow over the last 45°: 0.38 below 100, 0.55 × pulse above.
    let glow_from = start.max(end - 45.0);
    let glow_alpha = if v > 100.0 { 0.55 * pulse } else { 0.38 };
    glow(pixmap, r, glow_from, end, col, w + 10.0, glow_alpha);
    if v > 100.0 {
        let hot = act_color(100.0 + (v - 100.0) * 0.6);
        glow(
            pixmap,
            r,
            ring_angle(100.0) - 1.0,
            end,
            hot,
            w + 12.0,
            0.4 * pulse,
        );
    }
    dot(
        pixmap,
        r,
        start,
        w * 0.5,
        act_color(0.0),
        1.0,
        BlendMode::SourceOver,
    );
    if end <= start {
        return;
    }
    // Value arc coloured by position, with the comet head blended in over
    // the last 30°: mix(act(p), tip, ((p - h0) / (p_end - h0))²).
    let h0 = (p_end - 30.0 / PER).max(0.0);
    let mut stops: Vec<(f32, Rgb)> = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 107.0, 114.0, 120.0]
        .into_iter()
        .filter(|p| *p < h0)
        .map(|p| (p, act_color(p)))
        .collect();
    const HEAD_STEPS: usize = 12;
    let span = (p_end - h0).max(0.01);
    for step in 0..=HEAD_STEPS {
        let p = h0 + span * step as f32 / HEAD_STEPS as f32;
        let k = ((p - h0) / span).powi(2);
        stops.push((p, mix(act_color(p), tip, k)));
    }
    if let (Some(path), Some(shader)) = (arc_path(r, start, end), sweep(&stops)) {
        stroke_shader(
            pixmap,
            &path,
            shader,
            w,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
    dot(pixmap, r, end, w * 0.5, tip, 1.0, BlendMode::SourceOver);

    // Hot core at the peg, in hot pale gold.
    if v >= 120.0 {
        let k = ((v - 120.0) / 5.0).clamp(0.0, 1.0);
        let halo = ((0.30 + 0.20 * (std::f32::consts::TAU * 1.1 * t).sin()) * k).max(0.0);
        if let Some(path) = arc_path(r, end - 10.0, end) {
            for (width, share) in [(w + 8.0, 0.45), (w + 4.0, 0.55)] {
                stroke_solid(
                    pixmap,
                    &path,
                    HALO,
                    halo * share,
                    width,
                    LineCap::Round,
                    BlendMode::Plus,
                );
            }
        }
        if let Some(path) = arc_path(r, end - 8.0, end) {
            stroke_solid(
                pixmap,
                &path,
                HOT_GOLD,
                0.85 * k,
                w,
                LineCap::Round,
                BlendMode::SourceOver,
            );
        }
        dot(pixmap, r, end, w * 0.5, HOT_GOLD, k, BlendMode::SourceOver);
    }
}

/// Three concentric strokes (`width`, −4, −8 px) at 0.08 / 0.12 / 0.16 of a
/// 0.38 glow, scaled by `alpha / 0.38`, in Plus.
fn glow(pixmap: &mut Pixmap, r: f32, a0: f32, a1: f32, color: Rgb, width: f32, alpha: f32) {
    let Some(path) = arc_path(r, a0, a1) else {
        return;
    };
    let scale = alpha / 0.38;
    for (grow, share) in [(0.0, 0.08), (-4.0, 0.12), (-8.0, 0.16)] {
        stroke_solid(
            pixmap,
            &path,
            color,
            share * scale,
            width + grow,
            LineCap::Round,
            BlendMode::Plus,
        );
    }
}

fn draw_breaks(pixmap: &mut Pixmap, r: f32, w: f32) {
    let r0 = r - w * 0.5 - 1.0;
    let r1 = r + w * 0.5 + 1.0;
    for p in (10..=120).step_by(10) {
        if p == 100 {
            continue;
        }
        tick(pixmap, ring_angle(p as f32), r0, r1, BLACK, 1.0, 1.6);
    }
    for p in [105.0, 115.0] {
        tick(pixmap, ring_angle(p), r0, r1, BLACK, 1.0, 1.5);
    }
    tick(pixmap, ring_angle(100.0), r0, r1, GATE, 0.9, 2.2);
}

fn tick(pixmap: &mut Pixmap, deg: f32, r0: f32, r1: f32, color: Rgb, alpha: f32, width: f32) {
    let (x0, y0) = polar(r0, deg);
    let (x1, y1) = polar(r1, deg);
    let mut path = PathBuilder::new();
    path.move_to(x0, y0);
    path.line_to(x1, y1);
    if let Some(path) = path.finish() {
        stroke_solid(
            pixmap,
            &path,
            color,
            alpha,
            width,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
}

fn dot(
    pixmap: &mut Pixmap,
    r: f32,
    deg: f32,
    radius: f32,
    color: Rgb,
    alpha: f32,
    blend: BlendMode,
) {
    if radius <= 0.0 || alpha <= 0.0 {
        return;
    }
    let (x, y) = polar(r, deg);
    let mut path = PathBuilder::new();
    path.push_circle(x, y, radius);
    if let Some(path) = path.finish() {
        pixmap.fill_path(
            &path,
            &paint(color, alpha, blend),
            tiny_skia::FillRule::Winding,
            Transform::identity(),
            None,
        );
    }
}

/// A sweep gradient whose stops are ring positions (activity points).
fn sweep(stops: &[(f32, Rgb)]) -> Option<Shader<'static>> {
    let points: Vec<GradientStop> = stops
        .iter()
        .map(|(p, rgb)| {
            let turn = (p.clamp(0.0, RING_MAX) * PER / 360.0).clamp(0.0, 1.0);
            GradientStop::new(turn, Color::from_rgba8(rgb.r, rgb.g, rgb.b, 255))
        })
        .collect();
    // Skia measures sweep angles clockwise from +x. The ring starts at 225°
    // clockwise from 12, which is 135° from +x, so rotate t = 0 there.
    SweepGradient::new(
        Point::from_xy(CENTRE, CENTRE),
        0.0,
        360.0,
        points,
        SpreadMode::Pad,
        Transform::from_rotate_at(RING_START_DEG - 90.0, CENTRE, CENTRE),
    )
}

fn paint(color: Rgb, alpha: f32, blend: BlendMode) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(
        color.r,
        color.g,
        color.b,
        alpha_u8(alpha),
    ));
    paint.anti_alias = true;
    paint.blend_mode = blend;
    paint
}

fn alpha_u8(alpha: f32) -> u8 {
    if alpha.is_finite() {
        (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        0
    }
}

fn stroke_solid(
    pixmap: &mut Pixmap,
    path: &Path,
    color: Rgb,
    alpha: f32,
    width: f32,
    cap: LineCap,
    blend: BlendMode,
) {
    if alpha <= 0.0 || width <= 0.0 {
        return;
    }
    pixmap.stroke_path(
        path,
        &paint(color, alpha, blend),
        &stroke_of(width, cap),
        Transform::identity(),
        None,
    );
}

fn stroke_shader(
    pixmap: &mut Pixmap,
    path: &Path,
    shader: Shader<'static>,
    width: f32,
    cap: LineCap,
    blend: BlendMode,
) {
    let paint = Paint {
        shader,
        anti_alias: true,
        blend_mode: blend,
        ..Paint::default()
    };
    pixmap.stroke_path(
        path,
        &paint,
        &stroke_of(width, cap),
        Transform::identity(),
        None,
    );
}

fn stroke_of(width: f32, cap: LineCap) -> Stroke {
    Stroke {
        width,
        miter_limit: 4.0,
        line_cap: cap,
        line_join: LineJoin::Round,
        dash: None,
    }
}

/// Point at `radius`, `deg` clockwise from 12 o'clock.
#[must_use]
pub(super) fn polar(radius: f32, deg: f32) -> (f32, f32) {
    let theta = deg.to_radians();
    (CENTRE + radius * theta.sin(), CENTRE - radius * theta.cos())
}

/// Clockwise arc from `a0` to `a1` degrees (clockwise from 12). `None` when
/// the sweep is under 0.01°.
#[must_use]
pub(super) fn arc_path(radius: f32, a0: f32, a1: f32) -> Option<Path> {
    let sweep = (a1 - a0).to_radians();
    if radius <= 0.0 || !sweep.is_finite() || sweep < 0.01_f32.to_radians() {
        return None;
    }
    let mut angle = a0.to_radians();
    let (mut x, mut y) = polar_rad(radius, angle);
    let mut path = PathBuilder::new();
    path.move_to(x, y);
    let mut remaining = sweep;
    let max_seg = std::f32::consts::FRAC_PI_2;
    while remaining > 0.0001 {
        let seg = remaining.min(max_seg);
        let next = angle + seg;
        let (x1, y1) = polar_rad(radius, next);
        let kappa = (4.0 / 3.0) * (seg * 0.25).tan() * radius;
        path.cubic_to(
            x + angle.cos() * kappa,
            y + angle.sin() * kappa,
            x1 - next.cos() * kappa,
            y1 - next.sin() * kappa,
            x1,
            y1,
        );
        x = x1;
        y = y1;
        angle = next;
        remaining -= seg;
    }
    path.finish()
}

fn polar_rad(radius: f32, theta: f32) -> (f32, f32) {
    (CENTRE + radius * theta.sin(), CENTRE - radius * theta.cos())
}
