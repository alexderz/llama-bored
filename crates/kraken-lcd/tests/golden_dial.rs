//! Activity dial layer and the watch-down status.
//!
//! Geometry is the token-dial mockup: 24 bars, 12 o'clock newest, clockwise,
//! A1 outward from r 118, A3 outward from r 104. A bar is mean activity
//! (0..=125). `None` leaves a gap. A mean of 1 or less is a grey dot.
//! Watch-down is a circle-with-bar, not the dead brain.

use kraken_lcd::config::DisplayCfg;
use kraken_lcd::present::{Ai, Band, Variant, View};
use kraken_lcd::render::{self, Assets, Frame, LABEL, OUTLINE, TEXT};

fn paint(view: &View, assets: &mut Assets) -> Frame {
    render::render(view, &DisplayCfg::default(), assets)
}

fn frame_of(dial: [Option<u8>; 24], variant: Variant, ai: Ai) -> View {
    View {
        ring_pct: Some(100),
        ring_band: Some(Band::FlatOut),
        blocks: [Band::Quiet, Band::Light, Band::Busy],
        coolant_c: Some(40),
        cpu_c: Some(79),
        gpu_c: Some(80),
        cpu_pct: Some(41),
        mem_pct: Some(11),
        ai,
        models: vec!["Qwen3.6 35B-A3B".to_owned()],
        model_count: 1,
        dial,
        variant,
        ..View::default()
    }
}

fn rgb(frame: &Frame, x: u32, y: u32) -> (u8, u8, u8) {
    let pixel = frame.0.pixel(x, y).expect("pixel on the 320 frame");
    (pixel.red(), pixel.green(), pixel.blue())
}

fn near(got: (u8, u8, u8), expect: (u8, u8, u8), tol: u8) -> bool {
    got.0.abs_diff(expect.0) <= tol
        && got.1.abs_diff(expect.1) <= tol
        && got.2.abs_diff(expect.2) <= tol
}

/// Pixel nearest the point at `radius` px, `degrees` clockwise from 12 o'clock.
fn at(radius: f32, degrees: f32) -> (u32, u32) {
    let theta = degrees.to_radians();
    let x = 160.0 + radius * theta.sin();
    let y = 160.0 - radius * theta.cos();
    let ix = (x - 0.5).round().clamp(0.0, 319.0) as u32;
    let iy = (y - 0.5).round().clamp(0.0, 319.0) as u32;
    (ix, iy)
}

fn sample_at(frame: &Frame, radius: f32, degrees: f32) -> (u8, u8, u8) {
    let (x, y) = at(radius, degrees);
    rgb(frame, x, y)
}

fn any_near(
    frame: &Frame,
    radius: f32,
    degrees: f32,
    expect: (u8, u8, u8),
    tol: u8,
) -> Option<(u32, u32, (u8, u8, u8))> {
    let (x, y) = at(radius, degrees);
    for dy in -2..=2 {
        for dx in -2..=2 {
            let px = x.saturating_add_signed(dx);
            let py = y.saturating_add_signed(dy);
            if px >= 320 || py >= 320 {
                continue;
            }
            let got = rgb(frame, px, py);
            if near(got, expect, tol) {
                return Some((px, py, got));
            }
        }
    }
    None
}

fn lit(color: (u8, u8, u8)) -> bool {
    let peak = color.0.max(color.1).max(color.2);
    peak > 40 && !near(color, (0, 0, 0), 12)
}

fn outside_disc(x: u32, y: u32) -> bool {
    let dx = f64::from(x) + 0.5 - 160.0;
    let dy = f64::from(y) + 0.5 - 160.0;
    dx * dx + dy * dy > 160.0 * 160.0
}

/// Centre angle of bar `index`, in degrees clockwise from 12. Gap is symmetric,
/// so the centre is the slot centre.
fn slot_centre_deg(index: usize) -> f32 {
    let angles = [
        6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 15.0, 15.0, 20.0, 20.0, 20.0, 22.5, 22.5,
        22.5, 22.5, 24.0, 24.0, 24.0, 24.0, 24.0,
    ];
    let mut start = 0.0;
    for (i, angle) in angles.iter().enumerate() {
        if i == index {
            return start + angle * 0.5;
        }
        start += angle;
    }
    unreachable!("24 bars");
}

const L6: (u8, u8, u8) = (0xFF, 0x3A, 0x22);
const L1: (u8, u8, u8) = (0x4A, 0x55, 0xC8);
/// Redline track: 3 o'clock is 104 on the 0–125 scale.
const REDLINE_TRACK: (u8, u8, u8) = (0x3A, 0x12, 0x16);

#[test]
fn newest_bar_starts_at_twelve_and_runs_clockwise() {
    let mut assets = Assets::load().expect("assets");
    let mut dial = [None; 24];
    dial[0] = Some(100);
    let frame = paint(&frame_of(dial, Variant::A1, Ai::Loaded), &mut assets);
    let newest = sample_at(&frame, 130.0, slot_centre_deg(0));
    assert!(
        lit(newest) && newest.0 > newest.1 && newest.0 > newest.2,
        "the newest bar is the clockwise side of 12 o'clock, got {newest:?} at {:?}",
        at(130.0, slot_centre_deg(0))
    );
    let before = sample_at(&frame, 130.0, 357.0);
    assert!(
        !lit(before),
        "nothing is drawn just counter-clockwise of 12 when only the newest bar is set, got {before:?}"
    );
}

#[test]
fn oldest_bar_ends_just_left_of_twelve() {
    let mut assets = Assets::load().expect("assets");
    let mut dial = [None; 24];
    dial[23] = Some(100);
    let frame = paint(&frame_of(dial, Variant::A1, Ai::Loaded), &mut assets);
    let oldest = sample_at(&frame, 130.0, slot_centre_deg(23));
    assert!(
        lit(oldest),
        "the oldest bar sits just left of 12, got {oldest:?} at {}°",
        slot_centre_deg(23)
    );
    let newest = sample_at(&frame, 130.0, slot_centre_deg(0));
    assert!(
        !lit(newest),
        "the oldest bar does not paint the newest slot, got {newest:?}"
    );
}

#[test]
fn nodata_leaves_the_sector_empty() {
    let mut assets = Assets::load().expect("assets");
    let mut dial = [Some(70); 24];
    dial[4] = None;
    dial[5] = None;
    let frame = paint(&frame_of(dial, Variant::A1, Ai::Loaded), &mut assets);
    for index in [4, 5] {
        let got = sample_at(&frame, 128.0, slot_centre_deg(index));
        assert!(
            !lit(got),
            "NoData bar {index} should draw nothing, got {got:?}"
        );
    }
    let neighbor = sample_at(&frame, 128.0, slot_centre_deg(3));
    assert!(
        lit(neighbor),
        "a levelled neighbour is drawn, so the empty sector is the gap, got {neighbor:?}"
    );
}

#[test]
fn zero_is_a_grey_dot_at_the_base_and_not_a_bar() {
    let mut assets = Assets::load().expect("assets");
    let mut dial = [None; 24];
    dial[0] = Some(0);
    let frame = paint(&frame_of(dial, Variant::A1, Ai::Idle), &mut assets);
    // Dot centre sits one radius outward of the A1 base (r 118 + 1.5).
    let dot = any_near(
        &frame,
        119.5,
        slot_centre_deg(0),
        (OUTLINE.r, OUTLINE.g, OUTLINE.b),
        28,
    );
    assert!(
        dot.is_some(),
        "Zero draws a #3A3A3A dot at the base, sampled {:?}",
        sample_at(&frame, 119.5, slot_centre_deg(0))
    );
    let body = sample_at(&frame, 132.0, slot_centre_deg(0));
    assert!(!lit(body), "Zero does not grow a bar, got {body:?}");
}

#[test]
fn now_dot_marks_twelve_inside_the_base() {
    let mut assets = Assets::load().expect("assets");
    let dial = [None; 24];
    let a1 = paint(&frame_of(dial, Variant::A1, Ai::Idle), &mut assets);
    let a1_dot = rgb(&a1, 160, 48);
    assert!(
        near(a1_dot, (LABEL.r, LABEL.g, LABEL.b), 36),
        "A1 now dot is #9A9A9A at (r 112, 0°), got {a1_dot:?}"
    );
    let a3 = paint(&frame_of(dial, Variant::A3, Ai::Idle), &mut assets);
    let a3_dot = rgb(&a3, 160, 62);
    assert!(
        near(a3_dot, (LABEL.r, LABEL.g, LABEL.b), 36),
        "A3 now dot is #9A9A9A just inside the base, got {a3_dot:?}"
    );
}

#[test]
fn variant_changes_ring_and_dial_radius() {
    let mut assets = Assets::load().expect("assets");
    let dial = [None; 24];
    let a1 = paint(&frame_of(dial, Variant::A1, Ai::Idle), &mut assets);
    let a3 = paint(&frame_of(dial, Variant::A3, Ai::Idle), &mut assets);
    // A1 ring is r 145–157. r 142 at 3 o'clock is inside that, and clear of an empty dial.
    let a1_gap = sample_at(&a1, 142.0, 90.0);
    assert!(
        !lit(a1_gap) && near(a1_gap, (0, 0, 0), 8),
        "A1 leaves r 142 dark when the dial is empty, got {a1_gap:?}"
    );
    let a3_ring = sample_at(&a3, 142.0, 90.0);
    assert!(
        near(a3_ring, REDLINE_TRACK, 20) || lit(a3_ring),
        "A3 keeps the 16 px ring, so r 142 is on the stroke, got {a3_ring:?}"
    );

    let mut full = [Some(100); 24];
    full[0] = Some(100);
    let a1_bar = paint(&frame_of(full, Variant::A1, Ai::Idle), &mut assets);
    let a3_bar = paint(&frame_of(full, Variant::A3, Ai::Idle), &mut assets);
    // r 110 is inside A1's base (118) and on an A3 bar (base 104, L6 tip 136).
    let a1_hole = sample_at(&a1_bar, 110.0, slot_centre_deg(0));
    let a3_on = sample_at(&a3_bar, 110.0, slot_centre_deg(0));
    assert!(
        !lit(a1_hole),
        "A1 bars start at r 118, so r 110 stays empty, got {a1_hole:?}"
    );
    assert!(
        lit(a3_on),
        "A3 bars grow outward from r 104, so r 110 is on the bar, got {a3_on:?}"
    );
}

#[test]
fn step_colour_and_shading_follow_the_level() {
    let mut assets = Assets::load().expect("assets");
    let angle = slot_centre_deg(23);
    let mut dial = [None; 24];
    dial[23] = Some(100);
    let l6 = paint(&frame_of(dial, Variant::A1, Ai::Idle), &mut assets);
    // 72% of L6 (L = 24) is r 135.28. The wide 24° bar keeps the centre off the edge.
    let mid = sample_at(&l6, 135.0, angle);
    assert!(
        near(mid, L6, 48),
        "L6 at 72% of the bar is #FF3A22, got {mid:?}"
    );
    let base = sample_at(&l6, 122.0, angle);
    let mid_sum = u16::from(mid.0) + u16::from(mid.1) + u16::from(mid.2);
    let base_sum = u16::from(base.0) + u16::from(base.1) + u16::from(base.2);
    assert!(
        lit(base) && base_sum + 30 < mid_sum,
        "the base is the step colour mixed toward black, base {base:?} mid {mid:?}"
    );

    dial[23] = Some(10);
    let l1 = paint(&frame_of(dial, Variant::A1, Ai::Idle), &mut assets);
    // L1 length is 5.1. 72% of the way is about r 121.7.
    let indigo = sample_at(&l1, 121.5, angle);
    assert!(
        near(indigo, L1, 56) || (indigo.2 > indigo.0 && indigo.2 > 60),
        "L1 is the indigo step, not red, got {indigo:?}"
    );
}

#[test]
fn a1_and_a3_draw_nothing_outside_the_disc() {
    let mut assets = Assets::load().expect("assets");
    let dial = [Some(100); 24];
    for variant in [Variant::A1, Variant::A3] {
        let frame = paint(&frame_of(dial, variant, Ai::Loaded), &mut assets);
        for y in 0..frame.0.height() {
            for x in 0..frame.0.width() {
                if outside_disc(x, y) {
                    assert_eq!(
                        rgb(&frame, x, y),
                        (0, 0, 0),
                        "{variant:?} pixel ({x}, {y}) is outside the disc"
                    );
                }
            }
        }
    }
}

#[test]
fn watch_down_is_not_the_dead_brain() {
    let mut assets = Assets::load().expect("assets");
    let dial = {
        let mut bars = [Some(70); 24];
        for bar in &mut bars[..12] {
            *bar = None;
        }
        bars
    };
    let mut down = frame_of(dial, Variant::A1, Ai::Down);
    down.models.clear();
    down.model_count = 0;
    down.tokens = vec![4_500; 40];
    let mut watch = down.clone();
    watch.ai = Ai::NoData;
    let down_frame = paint(&down, &mut assets);
    let watch_frame = paint(&watch, &mut assets);

    let mut diff = 0;
    for y in 64..108 {
        for x in 70..230 {
            let a = rgb(&down_frame, x, y);
            let b = rgb(&watch_frame, x, y);
            if a.0.abs_diff(b.0) > 8 || a.1.abs_diff(b.1) > 8 || a.2.abs_diff(b.2) > 8 {
                diff += 1;
            }
        }
    }
    assert!(
        diff > 40,
        "watch-down and AI down differ in the status zone, saw {diff} pixels"
    );

    // Left side of the A1 mark circle (T51 V1 slot: centre 94, 78, radius ~8.3).
    let stroke = rgb(&watch_frame, 86, 78);
    assert!(
        near(stroke, (LABEL.r, LABEL.g, LABEL.b), 40),
        "watch-down draws the #9A9A9A circle, got {stroke:?}"
    );
    let hole = rgb(&watch_frame, 94, 73);
    assert!(
        near(hole, (0, 0, 0), 16),
        "the mark is a stroked circle, not a filled brain, got {hole:?}"
    );
    let mut text = 0;
    for y in 78..92 {
        for x in 108..210 {
            if near(rgb(&watch_frame, x, y), (TEXT.r, TEXT.g, TEXT.b), 24) {
                text += 1;
            }
        }
    }
    assert!(
        text > 12,
        "\"no data\" is drawn in the text colour, saw {text} pixels"
    );

    let ring = sample_at(&watch_frame, 151.0, 90.0);
    assert!(
        near(ring, REDLINE_TRACK, 16),
        "watch-down draws the ring track and no value arc, got {ring:?}"
    );
    // 30° is 76 on the scale, inside a 100 % arc.
    let live = sample_at(&down_frame, 151.0, 30.0);
    assert!(
        live.0 > 180 && live.0 > live.1 + 80,
        "AI down keeps the activity arc, got {live:?}"
    );
    let off = sample_at(&watch_frame, 151.0, 30.0);
    assert!(off.0 < 120, "watch-down has no arc at 30°, got {off:?}");

    // The chart keeps its frame but drops the line; the ceiling reads "—/s".
    let mut line_live = 0;
    let mut line_watch = 0;
    for y in 174..206 {
        for x in 88..232 {
            if near(rgb(&down_frame, x, y), (TEXT.r, TEXT.g, TEXT.b), 30) {
                line_live += 1;
            }
            if near(rgb(&watch_frame, x, y), (TEXT.r, TEXT.g, TEXT.b), 30) {
                line_watch += 1;
            }
        }
    }
    assert!(
        line_live > 100,
        "AI down still plots tokens, saw {line_live}"
    );
    assert_eq!(line_watch, 0, "watch-down plots no line");
    let baseline = rgb(&watch_frame, 150, 206);
    assert!(
        near(baseline, (0x3A, 0x3A, 0x3A), 12),
        "the chart baseline stays, got {baseline:?}"
    );
}

const DIAL_STATES: &[&str] = &[
    "a1-dial-spike",
    "a1-dial-sustained",
    "a1-dial-gaps",
    "a1-watch-down",
    "a3-working-hard",
    "a3-dial-spike",
    "a3-watch-down",
    "a1-v3b-pinned",
    "a1-v3b-spike",
    "a1-v3b-cool-down",
    "a3-v3b-pinned",
];

#[test]
fn golden_dial_frames_match_reviewed_pngs() {
    let mut assets = Assets::load().expect("assets");
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("../../fixtures");
    let total = 320_usize * 320;
    let mut failed = Vec::new();
    for name in DIAL_STATES {
        let json = std::fs::read(fixtures.join(format!("views/{name}.json")))
            .unwrap_or_else(|err| panic!("read {name} view: {err}"));
        let mut view = render::layout_a::view_from_json(&json)
            .unwrap_or_else(|err| panic!("{name} view: {err}"));
        // As `render-once`: a still with the peg's smoke four seconds in.
        kraken_lcd::present::warm_still(&mut view);
        let frame = paint(&view, &mut assets);
        let png = std::fs::read(root.join(format!("tests/golden/layout_a/{name}.png")))
            .unwrap_or_else(|err| panic!("read {name} golden: {err}"));
        let diff = render::layout_a::png_channel_disagreements(&frame, &png, 8)
            .unwrap_or_else(|err| panic!("{name} golden: {err}"));
        if diff * 1000 <= total * 5 {
            continue;
        }
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-actual");
        std::fs::create_dir_all(&dir).unwrap_or_else(|err| panic!("mkdir golden-actual: {err}"));
        let path = dir.join(format!("{name}.png"));
        let bytes = render::layout_a::frame_png(&frame)
            .unwrap_or_else(|err| panic!("encode {name}: {err}"));
        std::fs::write(&path, bytes)
            .unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
        failed.push(format!(
            "{name}: {diff} of {total} pixels differ by more than 8 levels ({:.3}%), wrote {}",
            (diff as f64) / (total as f64) * 100.0,
            path.display()
        ));
    }
    assert!(
        failed.is_empty(),
        "golden mismatch in {} state(s):\n{}",
        failed.len(),
        failed.join("\n")
    );
}

#[test]
fn a3_has_no_tokens_chart() {
    let mut assets = Assets::load().expect("assets");
    let dial = [Some(0); 24];
    let mut a1_view = frame_of(dial, Variant::A1, Ai::Loaded);
    a1_view.tokens = vec![1_000; 40];
    let mut a3_view = a1_view.clone();
    a3_view.variant = Variant::A3;
    let a1 = paint(&a1_view, &mut assets);
    let a3 = paint(&a3_view, &mut assets);
    // The chart's baseline row at y 206, between the A3 temperatures and footer.
    let mut a1_ink = 0;
    let mut a3_ink = 0;
    for y in 205..208 {
        for x in 110..130 {
            if rgb(&a1, x, y) != (0, 0, 0) {
                a1_ink += 1;
            }
            if rgb(&a3, x, y) != (0, 0, 0) {
                a3_ink += 1;
            }
        }
    }
    assert!(a1_ink > 8, "A1 draws the chart, saw {a1_ink} pixels");
    assert_eq!(a3_ink, 0, "A3 has no chart slot, saw {a3_ink} pixels");
}
