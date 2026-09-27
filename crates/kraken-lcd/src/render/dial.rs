//! Activity dial: 24 rounded annular sectors, zero dots, the now dot, and
//! the time scale.
//!
//! Placement, direction and radii come from [`super::geometry::LayoutGeometry`].
//! A bar is the mean activity of its window. `None` draws nothing. A mean of
//! 1 or less is a `#3A3A3A` dot at the base. Otherwise the length is
//! `L = length_min + length_span · min(mean, 100) / 100` and the colour is
//! [`act_color`]`(mean)` in the dial's radial gradient (base 55 % to black,
//! the colour at 72 %, tip 30 % to white).
//!
//! Above 100 a bar is incandescent: full length, a blackbody gradient along
//! it (ember → red → orange → the tip colour), a Plus bloom under it, a seeded
//! per-bar flicker, a sub-pixel tip shimmer, and above h 0.6 a filament.
//!
//! The time scale is design option A plus the fill sweeps: tier-boundary
//! ticks inside the bases, labels (A1 only), and a thin arc inside the newest
//! bar of each 1 min and 5 min tier as its window fills.

use tiny_skia::{
    BlendMode, Color, FillRule, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, Stroke,
    Transform,
};

use super::color::{BLACK, HOT_GOLD, WHITE, act_color, cold_tip_lift, hex, mix};
use super::geometry::LayoutGeometry;
use super::ring::arc_path;
use super::text::{GlyphCache, Pen, TextStyle, Weight};
use super::{LABEL, OUTLINE, Rgb};
use crate::anim::flicker;
use crate::present::{Ai, ScaleTier, View};

const CENTRE: f32 = 160.0;
/// Coal base of an incandescent bar.
const HOT_BASE: Rgb = hex(0x6E1000);
/// L6, the red at 100.
const RED: Rgb = hex(0xFF2A14);
/// Tier-boundary ticks.
const SCALE_TICK: Rgb = hex(0x55555C);
/// Time-scale labels.
const SCALE_LABEL: Rgb = hex(0x7A7A82);
/// Fill sweep.
const SWEEP: Rgb = hex(0xC8C8CC);

/// Draw the dial under the text. The caller has already painted the ring.
/// `t` is the frame clock for the flicker.
pub(super) fn draw(
    pixmap: &mut Pixmap,
    view: &View,
    geom: &LayoutGeometry,
    t: f32,
    cache: &mut GlyphCache,
) {
    let bars: Vec<Bar> = view
        .dial
        .iter()
        .enumerate()
        .filter_map(|(index, mean)| mean.map(|mean| Bar::new(geom, index, f32::from(mean), t)))
        .collect();
    // Blooms first, so a neighbour's halo never covers a bar.
    for bar in bars.iter().filter(|bar| bar.heat > 0.0) {
        draw_bloom(pixmap, geom, bar);
    }
    for bar in &bars {
        if bar.mean <= 1.0 {
            draw_zero(pixmap, geom, bar.index);
        } else {
            draw_bar(pixmap, geom, bar);
        }
    }
    for bar in bars.iter().filter(|bar| bar.heat > 0.6) {
        draw_filament(pixmap, geom, bar);
    }
    draw_scale(pixmap, view, geom, cache);
    draw_dot(pixmap, geom.now_radius, 0.0, geom.now_dot_radius, LABEL);
}

/// One drawn bar and its heat.
struct Bar {
    index: usize,
    mean: f32,
    /// `(mean − 100) / 25`, 0..=1.
    heat: f32,
    /// Flicker brightness factor, `1 + (0.05 + 0.10·h)·n`.
    bright: f32,
    /// Flicker noise, `-1..=1`. Zero for a bar at or under 100.
    noise: f32,
    /// Bar length before the shimmer.
    length: f32,
    /// Tip radius, shimmer included.
    tip: f32,
}

impl Bar {
    fn new(geom: &LayoutGeometry, index: usize, mean: f32, t: f32) -> Self {
        let heat = ((mean - 100.0) / 25.0).clamp(0.0, 1.0);
        let noise = if heat > 0.0 { flicker(index, t) } else { 0.0 };
        let bright = 1.0 + (0.05 + 0.10 * heat) * noise;
        let length = geom.length_min + geom.length_span * (mean / 100.0).clamp(0.0, 1.0);
        let shimmer = if heat > 0.0 { 0.4 * heat * noise } else { 0.0 };
        Self {
            index,
            mean,
            heat,
            bright,
            noise,
            length,
            tip: geom.base_radius + length + shimmer,
        }
    }

    /// Tip colour, pushed toward hot gold or black by the flicker when hot.
    fn tip_color(&self) -> Rgb {
        let tip = act_color(self.mean);
        if self.heat <= 0.0 {
            tip
        } else if self.noise > 0.0 {
            mix(tip, HOT_GOLD, 0.15 * self.heat * self.noise)
        } else {
            mix(tip, BLACK, 0.10 * self.heat * -self.noise)
        }
    }
}

fn draw_bar(pixmap: &mut Pixmap, geom: &LayoutGeometry, bar: &Bar) {
    let (a0, a1) = bar_angles(geom, bar.index);
    let base = geom.base_radius;
    let Some(path) = sector_path(geom, base, bar.tip, a0, a1, bar.tip - base) else {
        return;
    };
    let stops = if bar.heat <= 0.0 {
        let color = act_color(bar.mean);
        vec![
            (base, mix(color, BLACK, 0.55)),
            (base + 0.72 * bar.length, color),
            (bar.tip, mix(color, WHITE, cold_tip_lift(bar.mean, 0.30))),
        ]
    } else {
        let at = |x: f32| base + x * bar.length;
        let ember = mix(HOT_BASE, RED, 0.25 * bar.heat);
        let mid = act_color(100.0 + (bar.mean - 100.0) * 0.55);
        vec![
            (at(0.0), ember),
            (at(0.35), RED),
            (at(0.7), mid),
            (bar.tip, bar.tip_color()),
        ]
    };
    let Some(shader) = radial_shader(&stops, bar.tip) else {
        return;
    };
    let paint = Paint {
        anti_alias: true,
        shader,
        ..Paint::default()
    };
    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}

/// Three strokes of the bar outline, 12 / 8 / 4 px at 0.06 / 0.12 / 0.22 of
/// `h · f`, in `act_color(100 + 0.8·(mean − 100))`, Plus.
fn draw_bloom(pixmap: &mut Pixmap, geom: &LayoutGeometry, bar: &Bar) {
    let (a0, a1) = bar_angles(geom, bar.index);
    let base = geom.base_radius;
    let Some(path) = sector_path(geom, base, bar.tip, a0, a1, bar.tip - base) else {
        return;
    };
    let color = act_color(100.0 + 0.8 * (bar.mean - 100.0));
    let hf = bar.heat * bar.bright;
    for (width, share) in [(12.0, 0.06), (8.0, 0.12), (4.0, 0.22)] {
        stroke(
            pixmap,
            &path,
            color,
            share * hf,
            width,
            LineCap::Butt,
            BlendMode::Plus,
        );
    }
}

/// A 1.2 px line along the mid-angle from 45 % of the length to 0.8 px short
/// of the tip, `mix(tip, HOT_GOLD, 0.6)` at `0.8·(h − 0.6)/0.4·f`, Plus.
fn draw_filament(pixmap: &mut Pixmap, geom: &LayoutGeometry, bar: &Bar) {
    let (a0, a1) = bar_angles(geom, bar.index);
    let mid = (a0 + a1) * 0.5;
    let (x0, y0) = polar(geom.base_radius + 0.45 * bar.length, mid);
    let (x1, y1) = polar(bar.tip - 0.8, mid);
    let mut path = PathBuilder::new();
    path.move_to(x0, y0);
    path.line_to(x1, y1);
    let Some(path) = path.finish() else {
        return;
    };
    let alpha = 0.8 * (bar.heat - 0.6) / 0.4 * bar.bright;
    stroke(
        pixmap,
        &path,
        mix(bar.tip_color(), HOT_GOLD, 0.6),
        alpha,
        1.2,
        LineCap::Round,
        BlendMode::Plus,
    );
}

/// Tier-boundary ticks, labels and the fill sweeps.
fn draw_scale(pixmap: &mut Pixmap, view: &View, geom: &LayoutGeometry, cache: &mut GlyphCache) {
    let base = geom.base_radius;
    let mut bar = 0_usize;
    let mut seconds_ds = 0_u64;
    for tier in &view.scale {
        let first = bar;
        bar += usize::from(tier.bars);
        seconds_ds += u64::from(tier.width_ds) * u64::from(tier.bars);
        if bar > geom.bar_deg.len() || tier.bars == 0 {
            break;
        }
        let edge = boundary_deg(geom, bar);
        tick(pixmap, edge, base - 3.8, base - 0.8, SCALE_TICK, 1.2);
        if geom.scale_labels {
            let at = if edge > 354.0 { 348.0 } else { edge };
            label(pixmap, cache, &duration_label(seconds_ds), base - 9.0, at);
        }
        draw_sweep(pixmap, geom, view, tier, first);
    }
}

/// A 1.2 px butt arc at `base − 2.4` across `fill` of the newest bar's span.
fn draw_sweep(
    pixmap: &mut Pixmap,
    geom: &LayoutGeometry,
    view: &View,
    tier: &ScaleTier,
    first: usize,
) {
    let Some(fill) = tier.fill else {
        return;
    };
    if view.ai == Ai::NoData {
        return;
    }
    let (a0, a1) = bar_angles(geom, first);
    let (a0, a1) = (a0.to_degrees(), a1.to_degrees());
    let fraction = (f32::from(fill) / 1000.0).clamp(0.02, 1.0);
    if let Some(path) = arc_path(geom.base_radius - 2.4, a0, a0 + (a1 - a0) * fraction) {
        stroke(
            pixmap,
            &path,
            SWEEP,
            0.85,
            1.2,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
}

/// "5s", "15s", "1m", "5m", "30m" from tenths of a second.
fn duration_label(ds: u64) -> String {
    if ds.is_multiple_of(600) && ds >= 600 {
        format!("{}m", ds / 600)
    } else if ds.is_multiple_of(10) {
        format!("{}s", ds / 10)
    } else {
        format!("{}.{}s", ds / 10, ds % 10)
    }
}

/// Angle, degrees clockwise from 12, where bar `end` starts (the far edge of
/// bar `end − 1`).
fn boundary_deg(geom: &LayoutGeometry, end: usize) -> f32 {
    geom.bar_deg.iter().take(end).sum()
}

fn label(pixmap: &mut Pixmap, cache: &mut GlyphCache, text: &str, radius: f32, deg: f32) {
    let theta = deg.to_radians();
    let x = CENTRE + radius * theta.sin();
    let y = CENTRE - radius * theta.cos();
    let px = 9.0;
    let width = cache.measure(text, px, Weight::SemiBold, 0.0).advance;
    cache.draw(
        pixmap,
        text,
        Pen {
            x: x - width * 0.5,
            baseline: y + 3.0,
        },
        TextStyle {
            px,
            weight: Weight::SemiBold,
            color: SCALE_LABEL,
            tracking_em: 0.0,
        },
    );
}

fn tick(pixmap: &mut Pixmap, deg: f32, r0: f32, r1: f32, color: Rgb, width: f32) {
    let theta = deg.to_radians();
    let (x0, y0) = polar(r0, theta);
    let (x1, y1) = polar(r1, theta);
    let mut path = PathBuilder::new();
    path.move_to(x0, y0);
    path.line_to(x1, y1);
    if let Some(path) = path.finish() {
        stroke(
            pixmap,
            &path,
            color,
            1.0,
            width,
            LineCap::Butt,
            BlendMode::SourceOver,
        );
    }
}

fn stroke(
    pixmap: &mut Pixmap,
    path: &Path,
    color: Rgb,
    alpha: f32,
    width: f32,
    cap: LineCap,
    blend: BlendMode,
) {
    if alpha.is_nan() || alpha <= 0.0 {
        return;
    }
    let mut paint = Paint::default();
    let a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    paint.set_color(Color::from_rgba8(color.r, color.g, color.b, a));
    paint.anti_alias = true;
    paint.blend_mode = blend;
    let stroke = Stroke {
        width,
        miter_limit: 4.0,
        line_cap: cap,
        line_join: LineJoin::Round,
        dash: None,
    };
    pixmap.stroke_path(path, &paint, &stroke, Transform::identity(), None);
}

fn draw_zero(pixmap: &mut Pixmap, geom: &LayoutGeometry, index: usize) {
    let (a0, a1) = bar_angles(geom, index);
    let mid = (a0 + a1) * 0.5;
    let radius = geom.base_radius + geom.zero_dot_radius;
    draw_dot(pixmap, radius, mid, geom.zero_dot_radius, OUTLINE);
}

fn draw_dot(pixmap: &mut Pixmap, radius: f32, theta: f32, dot_radius: f32, color: Rgb) {
    if dot_radius <= 0.0 {
        return;
    }
    let (x, y) = polar(radius, theta);
    let mut path = PathBuilder::new();
    path.push_circle(x, y, dot_radius);
    let Some(path) = path.finish() else {
        return;
    };
    pixmap.fill_path(
        &path,
        &solid(color),
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}

/// `(start, end)` of bar `index` in radians, clockwise from 12 o'clock.
/// Half the base gap is taken off each side.
fn bar_angles(geom: &LayoutGeometry, index: usize) -> (f32, f32) {
    let mut start = 0.0_f32;
    for (i, deg) in geom.bar_deg.iter().enumerate() {
        let span = deg.to_radians();
        if i == index {
            let gap = if geom.base_radius > 0.0 {
                geom.gap_px / geom.base_radius
            } else {
                0.0
            };
            return (start + gap * 0.5, start + span - gap * 0.5);
        }
        start += span;
    }
    (0.0, 0.0)
}

fn sector_path(
    geom: &LayoutGeometry,
    base: f32,
    tip: f32,
    a0: f32,
    a1: f32,
    length: f32,
) -> Option<Path> {
    if base <= 0.0 || tip <= 0.0 || length <= 0.0 {
        return None;
    }
    let angle = (a1 - a0).abs();
    let width = base * angle;
    let corner = geom.corner_px.min(width * 0.5).min(length * 0.5).max(0.0);
    let dir = (a1 - a0).signum();
    let da_tip = corner / tip;
    let da_base = corner / base;
    let tip_arc_from = a0 + dir * da_tip;
    let tip_arc_to = a1 - dir * da_tip;
    let base_arc_from = a1 - dir * da_base;
    let base_arc_to = a0 + dir * da_base;
    let base_inset = base + corner;
    let tip_inset = tip - corner;

    let start = polar(base_inset, a0);
    let mut path = PathBuilder::new();
    path.move_to(start.0, start.1);
    let along_start = polar(tip_inset, a0);
    line_to_distinct(&mut path, start, along_start);
    let tip_corner_start = polar(tip, a0);
    let tip_arc_start = polar(tip, tip_arc_from);
    path.quad_to(
        tip_corner_start.0,
        tip_corner_start.1,
        tip_arc_start.0,
        tip_arc_start.1,
    );
    push_arc(&mut path, tip, tip_arc_from, tip_arc_to);
    let tip_corner_end = polar(tip, a1);
    let tip_side_end = polar(tip_inset, a1);
    path.quad_to(
        tip_corner_end.0,
        tip_corner_end.1,
        tip_side_end.0,
        tip_side_end.1,
    );
    let base_side_end = polar(base_inset, a1);
    line_to_distinct(&mut path, tip_side_end, base_side_end);
    let base_corner_end = polar(base, a1);
    let base_arc_start = polar(base, base_arc_from);
    path.quad_to(
        base_corner_end.0,
        base_corner_end.1,
        base_arc_start.0,
        base_arc_start.1,
    );
    push_arc(&mut path, base, base_arc_from, base_arc_to);
    let base_corner_start = polar(base, a0);
    path.quad_to(base_corner_start.0, base_corner_start.1, start.0, start.1);
    path.close();
    path.finish()
}

fn line_to_distinct(path: &mut PathBuilder, from: (f32, f32), to: (f32, f32)) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    if dx * dx + dy * dy > 0.01 {
        path.line_to(to.0, to.1);
    }
}

/// Cubic approximation of the arc from `from` to `to` at `radius`.
/// The current point is already `from`.
fn push_arc(path: &mut PathBuilder, radius: f32, from: f32, to: f32) {
    let sweep = to - from;
    if sweep.abs() <= 0.0001 {
        return;
    }
    let dir = sweep.signum();
    let mut remaining = sweep.abs();
    let mut angle = from;
    let (mut x, mut y) = polar(radius, angle);
    let max_seg = std::f32::consts::FRAC_PI_2;
    while remaining > 0.0001 {
        let seg = remaining.min(max_seg);
        let next = angle + dir * seg;
        let (x1, y1) = polar(radius, next);
        let (tx0, ty0) = tangent(angle, dir);
        let (tx1, ty1) = tangent(next, dir);
        let kappa = (4.0 / 3.0) * (seg * 0.25).tan() * radius;
        path.cubic_to(
            x + tx0 * kappa,
            y + ty0 * kappa,
            x1 - tx1 * kappa,
            y1 - ty1 * kappa,
            x1,
            y1,
        );
        x = x1;
        y = y1;
        angle = next;
        remaining -= seg;
    }
}

/// Radial gradient from the frame centre; stops are at radii.
fn radial_shader(stops: &[(f32, Rgb)], tip: f32) -> Option<tiny_skia::Shader<'static>> {
    if tip.is_nan() || tip <= 0.0 {
        return None;
    }
    let mut stops: Vec<(f32, Rgb)> = stops
        .iter()
        .map(|(radius, rgb)| ((radius / tip).clamp(0.0, 1.0), *rgb))
        .collect();
    stops.sort_by(|left, right| left.0.total_cmp(&right.0));
    let points = stops
        .into_iter()
        .map(|(offset, rgb)| {
            tiny_skia::GradientStop::new(offset, Color::from_rgba8(rgb.r, rgb.g, rgb.b, 255))
        })
        .collect();
    tiny_skia::RadialGradient::new(
        tiny_skia::Point::from_xy(CENTRE, CENTRE),
        0.0,
        tiny_skia::Point::from_xy(CENTRE, CENTRE),
        tip,
        points,
        tiny_skia::SpreadMode::Pad,
        Transform::identity(),
    )
}

fn polar(radius: f32, theta: f32) -> (f32, f32) {
    (CENTRE + radius * theta.sin(), CENTRE - radius * theta.cos())
}

fn tangent(theta: f32, dir: f32) -> (f32, f32) {
    (dir * theta.cos(), dir * theta.sin())
}

fn solid(color: Rgb) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.r, color.g, color.b, 255);
    paint.anti_alias = true;
    paint
}
