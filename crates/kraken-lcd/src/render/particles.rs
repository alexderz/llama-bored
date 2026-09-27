//! Smoke wisps and embers off the 125 % peg, drawn over everything.
//!
//! Wisps: two discs (the outer at 0.55 α, an inner 55 % disc at 0.7 α),
//! source-over, colour `#FFE0B8 → #A8A8B4`. Embers: a solid disc plus a
//! 2.2× halo at ¼ α, Plus, colour `#FFE6A0 → #FF6A1E`.

use tiny_skia::{BlendMode, Color, FillRule, Paint, PathBuilder, Pixmap, Transform};

use super::Rgb;
use super::color::{HOT_GOLD, hex, mix};
use crate::anim::Particles;

const WISP_A: Rgb = hex(0xFFE0B8);
const WISP_B: Rgb = hex(0xA8A8B4);
const EMBER_A: Rgb = HOT_GOLD;
const EMBER_B: Rgb = hex(0xFF6A1E);

/// Paint the alive particles. At most 20 × 2 discs.
pub(super) fn draw(pixmap: &mut Pixmap, particles: &Particles) {
    for wisp in &particles.wisps {
        let t = (wisp.age / wisp.life).clamp(0.0, 1.0);
        let radius = 3.0 + 7.0 * (1.0 - (1.0 - t).powi(2));
        let alpha = 0.6 * (t / 0.1).min(1.0) * (1.0 - t);
        let curl = 4.0 * (std::f32::consts::TAU * 0.7 * wisp.age + wisp.phase).sin();
        let color = mix(WISP_A, WISP_B, (t * 1.6).min(1.0));
        let x = wisp.x + curl * wisp.side;
        disc(
            pixmap,
            x,
            wisp.y,
            radius,
            color,
            alpha * 0.55,
            BlendMode::SourceOver,
        );
        disc(
            pixmap,
            x,
            wisp.y,
            radius * 0.55,
            color,
            alpha * 0.7,
            BlendMode::SourceOver,
        );
    }
    for ember in &particles.embers {
        let t = (ember.age / ember.life).clamp(0.0, 1.0);
        let radius = 2.0 + (0.8 - 2.0) * t;
        let alpha = 0.95 * (1.0 - t);
        let color = mix(EMBER_A, EMBER_B, t);
        disc(
            pixmap,
            ember.x,
            ember.y,
            radius,
            color,
            alpha,
            BlendMode::Plus,
        );
        disc(
            pixmap,
            ember.x,
            ember.y,
            radius * 2.2,
            color,
            alpha * 0.25,
            BlendMode::Plus,
        );
    }
}

fn disc(
    pixmap: &mut Pixmap,
    x: f32,
    y: f32,
    radius: f32,
    color: Rgb,
    alpha: f32,
    blend: BlendMode,
) {
    if !(radius > 0.0 && alpha > 0.0 && x.is_finite() && y.is_finite()) {
        return;
    }
    let mut path = PathBuilder::new();
    path.push_circle(x, y, radius);
    let Some(path) = path.finish() else {
        return;
    };
    let mut paint = Paint::default();
    let a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    paint.set_color(Color::from_rgba8(color.r, color.g, color.b, a));
    paint.anti_alias = true;
    paint.blend_mode = blend;
    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}
