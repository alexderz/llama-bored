//! Scaffold checks: the disc mask and the ring layer.

use kraken_lcd::config::DisplayCfg;
use kraken_lcd::present::{Ai, Band, View};
use kraken_lcd::render::{self, Assets};

fn view(ring_pct: Option<u8>, ring_band: Option<Band>) -> View {
    View {
        ring_pct,
        ring_band,
        blocks: [Band::Quiet, Band::Quiet, Band::Quiet],
        coolant_c: None,
        cpu_c: None,
        gpu_c: None,
        cpu_pct: None,
        mem_pct: None,
        ai: Ai::Idle,
        models: Vec::new(),
        model_count: 0,
        ..View::default()
    }
}

fn outside(x: u32, y: u32) -> bool {
    let dx = f64::from(x) + 0.5 - 160.0;
    let dy = f64::from(y) + 0.5 - 160.0;
    dx * dx + dy * dy > 160.0 * 160.0
}

fn sample_ring(frame: &render::Frame, turns: f32) -> (u8, u8, u8) {
    let theta = turns * std::f32::consts::TAU;
    let x = 160.0 + 148.0 * theta.sin();
    let y = 160.0 - 148.0 * theta.cos();
    let ix = (x - 0.5).round() as u32;
    let iy = (y - 0.5).round() as u32;
    let pixel = frame.0.pixel(ix, iy).expect("on the frame");
    (pixel.red(), pixel.green(), pixel.blue())
}

fn near(got: (u8, u8, u8), expect: (u8, u8, u8), tol: u8) -> bool {
    got.0.abs_diff(expect.0) <= tol
        && got.1.abs_diff(expect.1) <= tol
        && got.2.abs_diff(expect.2) <= tol
}

#[test]
fn nothing_is_drawn_outside_the_disc() {
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(
        &view(Some(100), Some(Band::FlatOut)),
        &DisplayCfg::default(),
        &mut assets,
    );
    let centre = frame.0.pixel(160, 160).expect("centre");
    assert_eq!(
        (centre.red(), centre.green(), centre.blue(), centre.alpha()),
        (0, 0, 0, 255),
        "the disc background is opaque black"
    );
    for y in 0..frame.0.height() {
        for x in 0..frame.0.width() {
            if outside(x, y) {
                let pixel = frame.0.pixel(x, y).expect("pixel");
                assert_eq!(
                    (pixel.red(), pixel.green(), pixel.blue()),
                    (0, 0, 0),
                    "pixel ({x}, {y}) is outside the disc"
                );
            }
        }
    }
}

/// No value is the bare 270° track: the ramp 80 % toward black up to 100,
/// the dim redline after it, the light 100 gate, and nothing in the gap.
#[test]
fn a_none_ring_draws_only_the_bare_track() {
    let mut assets = Assets::load().expect("assets");
    let frame = render::render(&view(None, None), &DisplayCfg::default(), &mut assets);
    // 12 o'clock is 62.5: act_color #C81CBA at 20 %.
    assert!(
        near(sample_ring(&frame, 0.0), (43, 14, 32), 10),
        "12 o'clock got {:?}",
        sample_ring(&frame, 0.0)
    );
    assert!(
        near(sample_ring(&frame, 0.5), (0, 0, 0), 4),
        "6 o'clock is the gap, got {:?}",
        sample_ring(&frame, 0.5)
    );
    for y in 0..frame.0.height() {
        for x in 0..frame.0.width() {
            let dx = f64::from(x) + 0.5 - 160.0;
            let dy = f64::from(y) + 0.5 - 160.0;
            let radius = (dx * dx + dy * dy).sqrt();
            if !(147.0..=155.0).contains(&radius) {
                continue;
            }
            // Clockwise from 12. The gate tick at 81° is light on purpose.
            let deg = dx.atan2(-dy).to_degrees().rem_euclid(360.0);
            if (deg - 81.0).abs() < 3.0 {
                continue;
            }
            let pixel = frame.0.pixel(x, y).expect("pixel");
            let peak = pixel.red().max(pixel.green()).max(pixel.blue());
            assert!(
                peak <= 72,
                "a None ring drew {:02x}{:02x}{:02x} at ({x}, {y}), {deg:.1}°",
                pixel.red(),
                pixel.green(),
                pixel.blue()
            );
        }
    }
}

#[test]
fn rotation_does_not_change_the_drawn_frame() {
    let mut assets = Assets::load().expect("assets");
    let view = view(Some(70), Some(Band::Busy));
    let upright = render::render(
        &view,
        &DisplayCfg {
            rotate_deg: 0,
            ..DisplayCfg::default()
        },
        &mut assets,
    );
    let turned = render::render(
        &view,
        &DisplayCfg {
            rotate_deg: 90,
            ..DisplayCfg::default()
        },
        &mut assets,
    );
    assert_eq!(upright.0.data(), turned.0.data());
    assert_ne!(
        sample_ring(&upright, 0.0).0,
        0,
        "the ring was drawn, so the comparison is not two blank frames"
    );
}
