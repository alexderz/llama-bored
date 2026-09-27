//! Layout A: stacked centre, locked to the accepted mockup.
//!
//! Pixel checks build [`View`] values directly. The reviewed PNG compare loads
//! `fixtures/views/*.json` and `tests/golden/layout_a/*.png`.

use kraken_lcd::config::DisplayCfg;

fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}
use kraken_lcd::present::{Ai, Band, Variant, View};
use kraken_lcd::render::{self, Assets, Frame, LABEL, NO_DATA, QUIET, TEXT};
use llama_core::detail::ModelDetail;

fn loaded(models: &[&str]) -> View {
    View {
        ring_pct: Some(70),
        ring_band: Some(Band::FlatOut),
        blocks: [Band::FlatOut, Band::Busy, Band::Light],
        coolant_c: Some(40),
        cpu_c: Some(79),
        gpu_c: Some(80),
        cpu_pct: Some(41),
        mem_pct: Some(11),
        ai: Ai::Loaded,
        models: models.iter().map(|name| (*name).to_owned()).collect(),
        model_count: u8::try_from(models.len()).unwrap_or(u8::MAX),
        ..View::default()
    }
}

fn paint(view: &View, assets: &mut Assets) -> Frame {
    render::render(view, &DisplayCfg::default(), assets)
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

fn count_near(frame: &Frame, rect: Rect, expect: (u8, u8, u8), tol: u8) -> usize {
    let mut count = 0;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            if near(rgb(frame, x, y), expect, tol) {
                count += 1;
            }
        }
    }
    count
}

fn zone_diff(left: &Frame, right: &Frame, rect: Rect) -> usize {
    let mut count = 0;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let a = rgb(left, x, y);
            let b = rgb(right, x, y);
            if a.0.abs_diff(b.0) > 8 || a.1.abs_diff(b.1) > 8 || a.2.abs_diff(b.2) > 8 {
                count += 1;
            }
        }
    }
    count
}

#[derive(Clone, Copy)]
struct Rect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

/// Brain sprite plus the model line. Temp labels start at y = 112.
const STATUS: Rect = Rect {
    x0: 40,
    y0: 48,
    x1: 280,
    y1: 100,
};

/// Tokens · 24 h plot area (A1), 150×32 at (85, 174), plus its baseline.
const CHART: Rect = Rect {
    x0: 85,
    y0: 174,
    x1: 236,
    y1: 207,
};

const COOL_NUMERAL: Rect = Rect {
    x0: 60,
    y0: 118,
    x1: 120,
    y1: 156,
};
const CPU_NUMERAL: Rect = Rect {
    x0: 130,
    y0: 118,
    x1: 190,
    y1: 156,
};
const GPU_NUMERAL: Rect = Rect {
    x0: 200,
    y0: 118,
    x1: 260,
    y1: 156,
};
/// Footer line. The token-dial mockup paints the whole run in label grey.
const FOOTER: Rect = Rect {
    x0: 70,
    y0: 226,
    x1: 250,
    y1: 248,
};

const NO_DATA_RGB: (u8, u8, u8) = (NO_DATA.r, NO_DATA.g, NO_DATA.b);
const TEXT_RGB: (u8, u8, u8) = (TEXT.r, TEXT.g, TEXT.b);
const QUIET_RGB: (u8, u8, u8) = (QUIET.r, QUIET.g, QUIET.b);
const LABEL_RGB: (u8, u8, u8) = (LABEL.r, LABEL.g, LABEL.b);

fn outside_disc(x: u32, y: u32) -> bool {
    let dx = f64::from(x) + 0.5 - 160.0;
    let dy = f64::from(y) + 0.5 - 160.0;
    dx * dx + dy * dy > 160.0 * 160.0
}

#[test]
fn down_and_idle_differ_in_the_status_zone() {
    let mut assets = Assets::load().expect("assets");
    let mut down = loaded(&[]);
    down.ai = Ai::Down;
    down.ring_pct = Some(4);
    down.ring_band = Some(Band::Quiet);
    let mut idle = down.clone();
    idle.ai = Ai::Idle;
    let down_frame = paint(&down, &mut assets);
    let idle_frame = paint(&idle, &mut assets);
    let diff = zone_diff(&down_frame, &idle_frame, STATUS);
    assert!(
        diff > 40,
        "dead brain plus \"AI down\" should differ from the sleeping brain plus \"no model\" in the status zone, saw {diff} pixels"
    );
}

#[test]
fn the_tokens_chart_takes_the_block_slot() {
    let mut assets = Assets::load().expect("assets");
    let mut busy = loaded(&["Qwen3.6 35B-A3B"]);
    // 42 tok/s now, falling to 2 tok/s at 24 h. The ceiling is 50/s.
    busy.tokens = (0..40).map(|i| 4_200 - i * 100).collect();
    let mut empty = busy.clone();
    empty.tokens.clear();
    let busy_frame = paint(&busy, &mut assets);
    let empty_frame = paint(&empty, &mut assets);
    let line = count_near(&busy_frame, CHART, TEXT_RGB, 30);
    assert!(line > 100, "the 1.5 px line is #F4F4F2, saw {line} pixels");
    assert_eq!(
        count_near(&empty_frame, CHART, TEXT_RGB, 30),
        0,
        "no data draws no line"
    );
    // Slot 0 at 42 of 50 tok/s sits at y = 206 − 32 × 0.84 ≈ 179.
    let head = Rect {
        x0: 85,
        y0: 177,
        x1: 88,
        y1: 182,
    };
    assert!(
        count_near(&busy_frame, head, TEXT_RGB, 60) > 0,
        "the newest point is at the left, scaled to the 1·2·5 ceiling"
    );
    // The fill is act_color from blue at the baseline.
    let low = rgb(&busy_frame, 200, 204);
    assert!(
        low.2 > low.1 && low.2 > 40,
        "the fill near the baseline is L1 blue, got {low:?}"
    );
    let diff = zone_diff(&busy_frame, &empty_frame, CHART);
    assert!(diff > 400, "the chart fills its slot, saw {diff} pixels");
}

#[test]
fn none_fields_draw_the_grey_dash() {
    let mut assets = Assets::load().expect("assets");
    let mut missing = loaded(&["Qwen3.6 35B-A3B"]);
    missing.ring_pct = None;
    missing.ring_band = None;
    missing.coolant_c = None;
    missing.cpu_c = None;
    missing.gpu_c = None;
    missing.cpu_pct = None;
    missing.mem_pct = None;
    let frame = paint(&missing, &mut assets);
    // Temperature dashes and a missing footer percent are #666666.
    // The footer words stay label grey.
    let slots = [
        ("coolant", COOL_NUMERAL, 30, NO_DATA_RGB),
        ("cpu temp", CPU_NUMERAL, 30, NO_DATA_RGB),
        ("gpu temp", GPU_NUMERAL, 30, NO_DATA_RGB),
        ("footer percents", FOOTER, 8, NO_DATA_RGB),
    ];
    for (name, rect, min, expect) in slots {
        let dashes = count_near(&frame, rect, expect, 16);
        assert!(
            dashes >= min,
            "{name} should draw \"—\", saw {dashes} pixels"
        );
        let bright = count_near(&frame, rect, TEXT_RGB, 16);
        assert_eq!(
            bright, 0,
            "{name} should not draw a numeral in the text colour when the field is None, saw {bright}"
        );
    }
    let words = count_near(&frame, FOOTER, LABEL_RGB, 16);
    assert!(
        words > 20,
        "footer words stay label grey around the dashes, saw {words} pixels"
    );
}

#[test]
fn numerals_use_text_colour_not_a_band() {
    let mut assets = Assets::load().expect("assets");
    let frame = paint(&loaded(&["Qwen3.6 35B-A3B"]), &mut assets);
    for (name, rect) in [
        ("coolant", COOL_NUMERAL),
        ("cpu", CPU_NUMERAL),
        ("gpu", GPU_NUMERAL),
    ] {
        let ink = count_near(&frame, rect, TEXT_RGB, 16);
        assert!(
            ink > 30,
            "{name} numeral should be #F4F4F2, saw {ink} pixels"
        );
        for (band, rgb) in [
            ("quiet", QUIET_RGB),
            ("light", (0x2E, 0xB8, 0x8A)),
            ("busy", (0xFF, 0xB0, 0x30)),
            ("flat-out", (0xFF, 0xF3, 0xE0)),
        ] {
            let stained = count_near(&frame, rect, rgb, 12);
            assert_eq!(
                stained, 0,
                "{name} numeral picked up the {band} colour, {stained} pixels"
            );
        }
    }
    let percents = count_near(&frame, FOOTER, TEXT_RGB, 16);
    let words = count_near(&frame, FOOTER, LABEL_RGB, 16);
    assert!(
        percents > 8,
        "footer percents are #F4F4F2, saw {percents} pixels"
    );
    assert!(
        words > 20,
        "footer labels are label grey, saw {words} pixels"
    );
}

#[test]
fn layout_draws_inside_the_disc_and_nothing_outside() {
    let mut assets = Assets::load().expect("assets");
    let frame = paint(&loaded(&["Qwen3.6 35B-A3B"]), &mut assets);
    let mut interior = 0;
    for y in 0..frame.0.height() {
        for x in 0..frame.0.width() {
            let (r, g, b) = rgb(&frame, x, y);
            if outside_disc(x, y) {
                assert_eq!((r, g, b), (0, 0, 0), "pixel ({x}, {y}) is outside the disc");
            } else {
                let dx = f64::from(x) + 0.5 - 160.0;
                let dy = f64::from(y) + 0.5 - 160.0;
                if dx * dx + dy * dy < 100.0 * 100.0 && (r > 8 || g > 8 || b > 8) {
                    interior += 1;
                }
            }
        }
    }
    assert!(
        interior > 50,
        "temps and history should put ink inside r = 100, saw {interior} pixels"
    );
}

fn outside_radius(x: u32, y: u32, radius: f64) -> bool {
    let dx = f64::from(x) + 0.5 - 160.0;
    let dy = f64::from(y) + 0.5 - 160.0;
    dx * dx + dy * dy > radius * radius
}

#[test]
fn wide_twelve_char_name_stays_inside_r_130() {
    let mut assets = Assets::load().expect("assets");
    let wide = paint(&loaded(&["WWWWWWWWWWWW"]), &mut assets);
    let narrow = paint(&loaded(&["W"]), &mut assets);
    let mut overflow = 0;
    for y in 0..wide.0.height() {
        for x in 0..wide.0.width() {
            if !outside_radius(x, y, 130.0) {
                continue;
            }
            let left = rgb(&wide, x, y);
            let right = rgb(&narrow, x, y);
            if left.0.abs_diff(right.0) > 8
                || left.1.abs_diff(right.1) > 8
                || left.2.abs_diff(right.2) > 8
            {
                overflow += 1;
            }
        }
    }
    let name_band = Rect {
        x0: 108,
        y0: 82,
        x1: 250,
        y1: 102,
    };
    let ink = count_near(&wide, name_band, TEXT_RGB, 24);
    assert!(
        ink > 20,
        "the wide name should still be drawn, saw {ink} text pixels"
    );
    assert_eq!(
        overflow, 0,
        "a 12-wide-glyph name painted {overflow} pixels outside r = 130"
    );
}

#[test]
fn two_model_names_change_the_status_line() {
    let mut assets = Assets::load().expect("assets");
    let one = paint(&loaded(&["Qwen3.6 35B-A3B"]), &mut assets);
    let two = paint(&loaded(&["Qwen3.6 27B", "Qwen3-VL 8B"]), &mut assets);
    let diff = zone_diff(&one, &two, STATUS);
    assert!(
        diff > 20,
        "two stacked names should differ from one name in the status zone, saw {diff}"
    );
}

const STATES: &[&str] = &[
    "working-hard",
    "quiet-history",
    "no-model",
    "ai-down",
    "fresh-start",
    "failed-source",
    "two-models",
    "bonsai-detail",
    "qwen-detail",
    "nemotron-detail",
    "a3-bonsai-detail",
];

#[test]
fn golden_frames_match_reviewed_pngs() {
    let mut assets = Assets::load().expect("assets");
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let total = 320_usize * 320;
    let mut failed = Vec::new();
    for name in STATES {
        let json = std::fs::read(fixtures().join(format!("views/{name}.json")))
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
fn three_models_leave_the_upper_stack_line() {
    let mut assets = Assets::load().expect("assets");
    let two = paint(&loaded(&["Qwen3.6 27B", "Qwen3-VL 8B"]), &mut assets);
    let three = paint(
        &loaded(&["Qwen3.6 27B", "Qwen3-VL 8B", "another"]),
        &mut assets,
    );
    // First stacked baseline is 88. The single-line count sits lower, at 96.
    let upper = Rect {
        x0: 108,
        y0: 74,
        x1: 240,
        y1: 84,
    };
    let diff = zone_diff(&two, &three, upper);
    assert!(
        diff > 10,
        "three models should show a count on the single line, not the upper stacked name, saw {diff}"
    );
}

/// Everything V3b can draw at once: pinned at 125, all 24 bars at 125 with
/// bloom and filament, 40 chart points, both sweeps, and the peg's particles.
fn worst_case(mut view: View) -> View {
    view.ring_pct = Some(125);
    view.dial = [Some(125); 24];
    view.tokens = (0..40).map(|i| 4_000 + (i * 997) % 9_000).collect();
    for tier in view.scale.iter_mut().skip(3) {
        tier.fill = Some(640);
    }
    kraken_lcd::present::warm_still(&mut view);
    assert!(view.dial_state.anim.particles.alive() > 0, "the peg smokes");
    view
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn render_cost() {
    let mut assets = Assets::load().expect("assets");
    let view = worst_case(loaded(&["Qwen3.6 35B-A3B"]));
    // Faces are decoded in `Assets::load`. This times one `render` call,
    // including the first frame's glyph rasterisation, every bar
    // incandescent, the chart full and the peg smoking.
    let started = std::time::Instant::now();
    let frame = paint(&view, &mut assets);
    let elapsed = started.elapsed();
    println!("render_cost: first V3b worst-case frame took {elapsed:?}");
    assert_eq!(frame.0.width(), 320);
    assert!(
        elapsed.as_secs_f64() < 0.020,
        "a frame took {elapsed:?}, budget is 20 ms"
    );
    // Stream mode draws at 10 fps. Rendering has to stay under 20 ms so the
    // upload still fits in the frame.
    assert!(
        elapsed < std::time::Duration::from_millis(20),
        "10 fps stream render budget is 20 ms, took {elapsed:?}"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn render_cost_long_name() {
    let mut assets = Assets::load().expect("assets");
    let name = "W".repeat(10_000);
    let view = worst_case(loaded(&[&name]));
    let started = std::time::Instant::now();
    let frame = paint(&view, &mut assets);
    let elapsed = started.elapsed();
    assert_eq!(frame.0.width(), 320);
    assert!(
        elapsed.as_secs_f64() < 0.020,
        "a 10_000-character name took {elapsed:?}, budget is 20 ms"
    );
}

fn with_detail(name: &str, detail: ModelDetail, variant: Variant) -> View {
    let mut view = loaded(&[name]);
    view.detail = Some(detail);
    view.variant = variant;
    view
}

fn q8(ctx: u32, quant: &str, ncmoe: Option<u16>) -> ModelDetail {
    ModelDetail {
        ctx: Some(ctx),
        ncmoe,
        kv_k: Some("q8_0".to_owned()),
        kv_v: Some("q8_0".to_owned()),
        quant: Some(quant.to_owned()),
        fa: Some(true),
    }
}

const STRESS: &str = "Hermes 5 Nemotron 3 Super 120B-A12B Instruct Uncensored";

#[test]
fn bonsai_wraps_to_two_lines_over_the_detail_line() {
    let mut assets = Assets::load().expect("assets");
    for variant in [Variant::A1, Variant::A3] {
        let view = with_detail("Ternary Bonsai 2 27B", q8(262_144, "PTQ1_0", None), variant);
        let lines = render::layout_a::status_text(&view, &mut assets);
        assert_eq!(
            lines.last().map(String::as_str),
            Some("256k · kv q8 · PTQ1_0")
        );
        let name = lines[..lines.len() - 1].join(" ");
        assert_eq!(name, "Ternary Bonsai 2 27B", "{variant:?}: {lines:?}");
    }
    let view = with_detail(
        "Ternary Bonsai 2 27B",
        q8(262_144, "PTQ1_0", None),
        Variant::A1,
    );
    assert_eq!(
        render::layout_a::status_text(&view, &mut assets),
        ["Ternary", "Bonsai 2 27B", "256k · kv q8 · PTQ1_0"],
        "V1: balanced split at a space"
    );
}

#[test]
fn a_name_too_long_for_two_lines_keeps_its_start_and_its_end() {
    let mut assets = Assets::load().expect("assets");
    let cases = [
        (STRESS, "Hermes", "Uncensored"),
        ("Nemotron 3 Super 120B-A12B", "Nemotron", "120B-A12B"),
        (
            "WwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwEND",
            "Www",
            "END",
        ),
    ];
    for variant in [Variant::A1, Variant::A3] {
        for (name, start, end) in cases {
            for detail in [None, Some(q8(131_072, "UD-Q4_K_XL", Some(24)))] {
                let mut view = loaded(&[name]);
                view.variant = variant;
                view.detail = detail.clone();
                let lines = render::layout_a::status_text(&view, &mut assets);
                let names = if detail.is_some() {
                    &lines[..lines.len() - 1]
                } else {
                    &lines[..]
                };
                assert!(names.len() <= 2, "{variant:?} {name}: {lines:?}");
                assert!(
                    names[0].starts_with(start),
                    "{variant:?} {name}: start lost: {lines:?}"
                );
                assert!(
                    names.last().expect("a name line").ends_with(end),
                    "{variant:?} {name}: end cut: {lines:?}"
                );
            }
        }
    }
    let mut view = loaded(&[STRESS]);
    view.detail = Some(q8(131_072, "UD-Q4_K_XL", Some(24)));
    let lines = render::layout_a::status_text(&view, &mut assets);
    assert!(lines[0].ends_with('\u{2026}'), "middle dropped: {lines:?}");
}

#[test]
fn an_old_snapshot_draws_the_name_alone() {
    let mut assets = Assets::load().expect("assets");
    let view = loaded(&["Ternary Bon\u{2026}"]);
    assert_eq!(view.detail, None);
    assert_eq!(
        render::layout_a::status_text(&view, &mut assets),
        ["Ternary Bon\u{2026}"]
    );
}

#[test]
fn two_models_keep_the_stacked_pair_and_no_detail() {
    let mut assets = Assets::load().expect("assets");
    let mut view = loaded(&["Qwen3.6 27B", "Qwen3-VL 8B"]);
    view.detail = Some(q8(262_144, "PTQ1_0", None));
    assert_eq!(
        render::layout_a::status_text(&view, &mut assets),
        ["Qwen3.6 27B", "Qwen3-VL 8B"]
    );
}

#[test]
fn name_and_detail_stay_inside_r_114_and_off_the_temps() {
    let mut assets = Assets::load().expect("assets");
    for variant in [Variant::A1, Variant::A3] {
        let mut short = loaded(&["Q"]);
        short.variant = variant;
        let reference = paint(&short, &mut assets);
        // First row of the temperature labels, and nothing below it, is shared.
        let temps_top: u32 = if variant == Variant::A1 { 113 } else { 139 };
        for (name, detail) in [
            ("Ternary Bonsai 2 27B", q8(262_144, "PTQ1_0", None)),
            (
                "Nemotron 3 Super 120B-A12B",
                q8(262_144, "UD-Q4_K_M", Some(88)),
            ),
            (STRESS, q8(131_072, "UD-Q4_K_XL", Some(24))),
            (
                "WWWWWWWWWWWWWWWWWWWWWWWW",
                q8(1_048_576, "UD-Q4_K_XL", Some(88)),
            ),
        ] {
            let frame = paint(&with_detail(name, detail, variant), &mut assets);
            let mut outside = 0;
            let mut temps = 0;
            for y in 0..320 {
                for x in 0..320 {
                    let a = rgb(&frame, x, y);
                    let b = rgb(&reference, x, y);
                    let differs =
                        a.0.abs_diff(b.0) > 8 || a.1.abs_diff(b.1) > 8 || a.2.abs_diff(b.2) > 8;
                    if !differs {
                        continue;
                    }
                    if outside_radius(x, y, 114.0) {
                        outside += 1;
                    }
                    if y >= temps_top {
                        temps += 1;
                    }
                }
            }
            assert_eq!(outside, 0, "{variant:?} {name}: drew outside r 114");
            assert_eq!(temps, 0, "{variant:?} {name}: overlapped the temperatures");
        }
    }
}
