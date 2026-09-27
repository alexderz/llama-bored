//! Tokens · 24 h, in the slot the three load blocks had (A1 only).
//!
//! Plot area 150×32 at (85, 174). Forty points at equal spacing, newest on
//! the left like the dial: 10 × 30 s, 11 × 5 min, 10 × 30 min, 9 × 2 h, so
//! "1h · 6h · 24h" sit on the tier boundaries. The ceiling is the visible
//! peak rounded up to 1 · 2 · 5 × 10ⁿ tok/s (at least 10), printed top-left.
//! The fill is a vertical [`act_color`] gradient at 0.6 α; the line is 1.5 px
//! `#F4F4F2`. With no data the frame, ticks and labels stay, and the ceiling
//! reads "—/s".

use tiny_skia::{
    Color, FillRule, GradientStop, LineCap, LineJoin, LinearGradient, Paint, PathBuilder, Pixmap,
    Point, SpreadMode, Stroke, Transform,
};

use super::color::{act_color, hex};
use super::geometry::LayoutGeometry;
use super::text::{GlyphCache, Pen, TextStyle, Weight};
use super::{OUTLINE, Rgb, TEXT};
use crate::present::{Ai, View};
use crate::tokens::{SLOTS, nice_ceiling};

const TICK: Rgb = hex(0x55555C);
const LABEL_GREY: Rgb = hex(0x7A7A82);
const PX: f32 = 9.0;
/// `(slot, label, anchor)`: 0 start, 1 middle, 2 end.
const TICKS: [(usize, &str, u8); 4] = [(0, "now", 0), (21, "1h", 1), (31, "6h", 1), (39, "24h", 2)];

/// Draw the chart. The layout calls this in place of the blocks.
pub(super) fn draw(
    pixmap: &mut Pixmap,
    view: &View,
    geom: &LayoutGeometry,
    cache: &mut GlyphCache,
) {
    if !geom.show_chart {
        return;
    }
    let x0 = geom.chart_x;
    let top = geom.chart_y;
    let w = geom.chart_w;
    let h = geom.chart_h;
    let base = top + h;
    let dx = w / (SLOTS - 1) as f32;

    let rates: Vec<f32> = if view.ai == Ai::NoData {
        Vec::new()
    } else {
        view.tokens
            .iter()
            .take(SLOTS)
            .map(|centi| *centi as f32 / 100.0)
            .collect()
    };

    line(
        pixmap,
        (x0, base + 0.5),
        (x0 + w, base + 0.5),
        OUTLINE,
        1.0,
        1.0,
    );
    for (slot, text, anchor) in TICKS {
        let x = x0 + slot as f32 * dx;
        line(pixmap, (x, base + 1.0), (x, base + 3.5), TICK, 1.0, 1.0);
        write(cache, pixmap, text, x, geom.chart_tick_baseline, anchor);
    }

    let ceiling_text = if rates.is_empty() {
        "\u{2014}/s".to_owned()
    } else {
        let peak = rates.iter().copied().fold(0.0_f32, f32::max);
        let ceiling = nice_ceiling(peak);
        let points: Vec<(f32, f32)> = rates
            .iter()
            .enumerate()
            .map(|(i, rate)| {
                let y = base - (rate / ceiling).clamp(0.0, 1.0) * h;
                (x0 + i as f32 * dx, y)
            })
            .collect();
        plot(pixmap, &points, top, base);
        ceiling_label(ceiling)
    };
    write(
        cache,
        pixmap,
        &ceiling_text,
        x0,
        geom.chart_title_baseline,
        0,
    );
    write(
        cache,
        pixmap,
        "tokens \u{00B7} 24h",
        x0 + w,
        geom.chart_title_baseline,
        2,
    );
}

/// "100/s", "2k/s".
#[must_use]
pub fn ceiling_label(ceiling: f32) -> String {
    let whole = ceiling.round() as u64;
    if whole >= 1000 && whole.is_multiple_of(1000) {
        format!("{}k/s", whole / 1000)
    } else if whole >= 1000 {
        format!("{:.1}k/s", ceiling / 1000.0)
    } else {
        format!("{whole}/s")
    }
}

fn plot(pixmap: &mut Pixmap, points: &[(f32, f32)], top: f32, base: f32) {
    match points {
        [] => {}
        [(x, y)] => {
            let mut path = PathBuilder::new();
            path.push_circle(*x, *y, 1.5);
            if let Some(path) = path.finish() {
                pixmap.fill_path(
                    &path,
                    &solid(TEXT, 1.0),
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }
        [first, .., last] => {
            let mut area = PathBuilder::new();
            area.move_to(first.0, base);
            for (x, y) in points {
                area.line_to(*x, *y);
            }
            area.line_to(last.0, base);
            area.close();
            if let (Some(path), Some(shader)) = (area.finish(), fill_shader(top, base)) {
                let paint = Paint {
                    shader,
                    anti_alias: true,
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
            let mut stroke_path = PathBuilder::new();
            stroke_path.move_to(first.0, first.1);
            for (x, y) in &points[1..] {
                stroke_path.line_to(*x, *y);
            }
            if let Some(path) = stroke_path.finish() {
                pixmap.stroke_path(
                    &path,
                    &solid(TEXT, 1.0),
                    &Stroke {
                        width: 1.5,
                        miter_limit: 4.0,
                        line_cap: LineCap::Round,
                        line_join: LineJoin::Round,
                        dash: None,
                    },
                    Transform::identity(),
                    None,
                );
            }
        }
    }
}

/// Vertical `act_color` gradient: 0 at the baseline to 100 at the top, stops
/// every 20, 0.6 α.
fn fill_shader(top: f32, base: f32) -> Option<tiny_skia::Shader<'static>> {
    let stops = (0..=5)
        .map(|k| {
            let rgb = act_color(k as f32 * 20.0);
            GradientStop::new(k as f32 / 5.0, Color::from_rgba8(rgb.r, rgb.g, rgb.b, 153))
        })
        .collect();
    LinearGradient::new(
        Point::from_xy(0.0, base),
        Point::from_xy(0.0, top),
        stops,
        SpreadMode::Pad,
        Transform::identity(),
    )
}

fn write(
    cache: &mut GlyphCache,
    pixmap: &mut Pixmap,
    text: &str,
    x: f32,
    baseline: f32,
    anchor: u8,
) {
    let width = cache.measure(text, PX, Weight::SemiBold, 0.0).advance;
    let left = match anchor {
        0 => x,
        1 => x - width * 0.5,
        _ => x - width,
    };
    cache.draw(
        pixmap,
        text,
        Pen { x: left, baseline },
        TextStyle {
            px: PX,
            weight: Weight::SemiBold,
            color: LABEL_GREY,
            tracking_em: 0.0,
        },
    );
}

fn line(pixmap: &mut Pixmap, from: (f32, f32), to: (f32, f32), color: Rgb, alpha: f32, width: f32) {
    let mut path = PathBuilder::new();
    path.move_to(from.0, from.1);
    path.line_to(to.0, to.1);
    if let Some(path) = path.finish() {
        pixmap.stroke_path(
            &path,
            &solid(color, alpha),
            &Stroke {
                width,
                miter_limit: 4.0,
                line_cap: LineCap::Butt,
                line_join: LineJoin::Miter,
                dash: None,
            },
            Transform::identity(),
            None,
        );
    }
}

fn solid(color: Rgb, alpha: f32) -> Paint<'static> {
    let mut paint = Paint::default();
    let a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    paint.set_color(Color::from_rgba8(color.r, color.g, color.b, a));
    paint.anti_alias = true;
    paint
}

#[cfg(test)]
mod tests {
    use super::ceiling_label;

    #[test]
    fn ceiling_labels_use_k_from_a_thousand() {
        assert_eq!(ceiling_label(10.0), "10/s");
        assert_eq!(ceiling_label(100.0), "100/s");
        assert_eq!(ceiling_label(500.0), "500/s");
        assert_eq!(ceiling_label(1000.0), "1k/s");
        assert_eq!(ceiling_label(2000.0), "2k/s");
    }
}
