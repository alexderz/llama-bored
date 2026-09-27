//! T65: the engine features behind keyboard Variant B — rate window,
//! ladder, gate, peak, asymmetric smoothing, gradients, tween, led:N,
//! [base], value-scaled brightness, thresholds, the keyboard frame cap,
//! the validation errors, and the shipped keyboard-c example.

mod common;

use common::{FakeConfig, FakeKbOpener, FakeOpener, Feed, Lines, ManualClock, NoNotify, SEC, snap};
use llama_core::color::{BLACK, Rgb, act_color, hex, mix};
use llama_core::wire::SnapshotV1;
use llama_light::aura::AuraBackend;
use llama_light::config::{Style, Target, parse};
use llama_light::keyboard::KeyboardBackend;
use llama_light::keyboard::keymap::{KEYS, key_index};
use llama_light::mapping::{Renderer, ema, fade};
use llama_light::metric::TokenRate;
use llama_light::palette::dim;
use llama_light::service::{Light, Parts};

const MS: u64 = 1_000_000;

fn key(name: &str) -> usize {
    key_index(name).unwrap_or_else(|| panic!("{name}"))
}

fn kb(body: &str) -> String {
    format!("[aura]\nbrightness_max = 100\n[keyboard]\nenabled = true\n{body}")
}

fn renderer(body: &str) -> Renderer {
    Renderer::new(parse(&kb(body)).unwrap_or_else(|e| panic!("{e}\n{body}")))
}

fn gpu_snap(seq: u64, t: u64, gpu: f32) -> SnapshotV1 {
    let mut s = snap(seq, t);
    s.host.gpu_pct = Some(gpu);
    s
}

fn error(text: &str) -> String {
    match parse(text) {
        Ok(config) => panic!("accepted:\n{text}\n{config:?}"),
        Err(err) => err.0,
    }
}

// ---------------------------------------------------------------- rate window

const STEADY: &str = r#"
[[light]]
target = 'keyboard.keys["1".."0"]'
metric = "tokens_rate"
rate_window_s = 2.0
range = [0, 100]
style = "bar"
"#;

/// 60 tok/s for real, but the counter moves in bursts: 18 tokens every
/// third 10 Hz snapshot and nothing in between. The windowed rate stays
/// near 60 and the bar never drops between bursts.
#[test]
fn the_rate_window_turns_bursty_counter_samples_into_a_steady_rate() {
    let mut r = renderer(STEADY);
    let mut total = 1000_u64;
    for i in 0..80_u64 {
        if i % 3 == 0 && i > 0 {
            total += 18;
        }
        let mut s = snap(i + 1, 10 * SEC + i * 100 * MS);
        s.tokens.decoded_total = Some(total);
        let frame = r.frames(&s, 0.1).keyboard;
        if i < 25 {
            continue;
        }
        let rate = r.values()[0].expect("rate");
        assert!(
            (45.0..=75.0).contains(&rate),
            "tick {i}: {rate} tok/s from a steady 60"
        );
        for name in ["1", "2", "3", "4", "5"] {
            assert_ne!(frame[key(name)], BLACK, "tick {i}: key {name} dropped out");
        }
    }
}

#[test]
fn rate_over_uses_the_newest_sample_at_least_a_window_old() {
    let mut rate = TokenRate::with_window(2.0);
    let feed = |rate: &mut TokenRate, seq: u64, t_ms: u64, total: u64| {
        let mut s = snap(seq, t_ms * MS);
        s.tokens.decoded_total = Some(total);
        rate.update(&s)
    };
    assert_eq!(feed(&mut rate, 1, 10_000, 100), Some(0.0));
    assert_eq!(rate.rate_over(2.0), Some(0.0), "one sample is a baseline");
    feed(&mut rate, 2, 10_500, 100);
    assert_eq!(
        feed(&mut rate, 3, 11_000, 140),
        Some(80.0),
        "raw: 40 in 0.5 s"
    );
    // Early on the window is what there is: 40 tokens in 1 s.
    assert_eq!(rate.rate_over(2.0), Some(40.0));
    feed(&mut rate, 4, 11_500, 140);
    feed(&mut rate, 5, 12_000, 180);
    feed(&mut rate, 6, 12_500, 180);
    // Newest at 12.5 s; the newest sample at or before 10.5 s is 10.5 s (100).
    assert_eq!(rate.rate_over(2.0), Some(40.0));
    assert_eq!(
        rate.rate_over(0.0),
        Some(0.0),
        "raw rate between the last two"
    );
    // A missing counter clears the window.
    let mut none = snap(7, 13 * SEC);
    none.tokens.decoded_total = None;
    assert_eq!(rate.update(&none), None);
    assert_eq!(rate.rate_over(2.0), None);
}

// ---------------------------------------------------------------- ladder

const LADDER: &str = r#"
[[light]]
target = ["Number Pad 0","Number Pad 1","Number Pad 2","Number Pad 3"]
metric = "gpu"
range = [0, 100]
thresholds = [1, 10, 20, 30]
style = "ladder"
edge = "fractional"
gradient = "position"
"#;

#[test]
fn ladder_rungs_light_at_their_thresholds_with_a_fractional_last_rung() {
    let rungs = [
        "Number Pad 0",
        "Number Pad 1",
        "Number Pad 2",
        "Number Pad 3",
    ];
    let mut r = renderer(LADDER);
    let frame = r.frames(&gpu_snap(1, SEC, 15.0), 0.1).keyboard;
    assert_eq!(frame[key(rungs[0])], act_color(1.0), "full at 1");
    assert_eq!(frame[key(rungs[1])], act_color(10.0), "full at 10");
    assert_eq!(
        frame[key(rungs[2])],
        mix(BLACK, act_color(20.0), 0.5),
        "15 is half way from 10 to 20"
    );
    assert_eq!(frame[key(rungs[3])], BLACK, "30 not reached");
    let mut r = renderer(LADDER);
    let frame = r.frames(&gpu_snap(1, SEC, 0.5), 0.1).keyboard;
    assert_eq!(frame[key(rungs[0])], mix(BLACK, act_color(1.0), 0.5));
    assert_eq!(frame[key(rungs[1])], BLACK);
    // Over the top: every rung lit, colours clipped at 125 %.
    let mut r = renderer(&LADDER.replace("[1, 10, 20, 30]", "[1, 10, 20, 130]"));
    let frame = r.frames(&gpu_snap(1, SEC, 200.0), 0.1).keyboard;
    assert_eq!(frame[key(rungs[3])], act_color(125.0));
}

#[test]
fn a_round_edge_ladder_lights_whole_rungs_and_value_gradient_uses_one_colour() {
    let body = LADDER
        .replace("edge = \"fractional\"", "edge = \"round\"")
        .replace("gradient = \"position\"", "gradient = \"value\"");
    let mut r = renderer(&body);
    let frame = r.frames(&gpu_snap(1, SEC, 15.0), 0.1).keyboard;
    assert_eq!(frame[key("Number Pad 0")], act_color(15.0));
    assert_eq!(frame[key("Number Pad 1")], act_color(15.0));
    assert_eq!(frame[key("Number Pad 2")], BLACK);
}

#[test]
fn ladder_thresholds_default_to_even_steps_over_the_range() {
    let config = parse(&kb(&LADDER.replace("thresholds = [1, 10, 20, 30]\n", ""))).expect("ok");
    assert_eq!(config.layers[0].thresholds, vec![25.0, 50.0, 75.0, 100.0]);
}

// ---------------------------------------------------------------- gate

const GATE: &str = r##"
[[light]]
target = "led:110"
metric = "decoded_total"
style = "gate"
hold_s = 1.5
color = "#1428D8"
brightness = 0.45
"##;

fn tok_snap(seq: u64, t: u64, total: u64) -> SnapshotV1 {
    let mut s = snap(seq, t);
    s.tokens.decoded_total = Some(total);
    s
}

#[test]
fn the_gate_opens_on_any_token_and_holds_for_hold_s() {
    let dot = key("Number Pad .");
    let mut r = renderer(GATE);
    let lit = dim(hex(0x1428D8), 45.0);
    assert_eq!(
        r.frames(&tok_snap(1, 10 * SEC, 500), 0.1).keyboard[dot],
        BLACK
    );
    assert_eq!(
        r.frames(&tok_snap(2, 10 * SEC + 100 * MS, 500), 0.1)
            .keyboard[dot],
        BLACK
    );
    // One token: open at once.
    assert_eq!(
        r.frames(&tok_snap(3, 10 * SEC + 200 * MS, 501), 0.1)
            .keyboard[dot],
        lit
    );
    // No more tokens: open for 1.5 s, then shut.
    let mut seq = 4;
    for step in 1..=14_u64 {
        let f = r.frames(&tok_snap(seq, 10 * SEC + (200 + step * 100) * MS, 501), 0.1);
        assert_eq!(f.keyboard[dot], lit, "{step} ticks after the token");
        seq += 1;
    }
    r.frames(&tok_snap(seq, 11 * SEC + 800 * MS, 501), 0.1);
    let f = r.frames(&tok_snap(seq + 1, 11 * SEC + 900 * MS, 501), 0.1);
    assert_eq!(f.keyboard[dot], BLACK, "shut after hold_s");
}

#[test]
fn the_gate_shimmer_stays_within_depth_of_its_brightness() {
    let body = format!("{GATE}shimmer = {{ depth = 0.2, hz = 0.7 }}\n");
    let dot = key("Number Pad .");
    let mut r = renderer(&body);
    let (mut lo, mut hi) = (u8::MAX, 0_u8);
    for i in 0..50_u64 {
        let f = r.frames(&tok_snap(i + 1, 10 * SEC + i * 100 * MS, 500 + i * 3), 0.1);
        if i < 1 {
            continue;
        }
        let b = f.keyboard[dot].b;
        lo = lo.min(b);
        hi = hi.max(b);
    }
    // 216 × 0.45 = 97.2; ± 20 % is 77.8 ..= 116.6.
    assert!(lo >= 77 && hi <= 117, "shimmer out of bounds: {lo}..{hi}");
    assert!(lo <= 82 && hi >= 112, "shimmer too shallow: {lo}..{hi}");
}

#[test]
fn a_smoothed_gate_fades_in() {
    let body = format!("{GATE}smooth_s = 0.3\n");
    let dot = key("Number Pad .");
    let mut r = renderer(&body);
    r.frames(&tok_snap(1, 10 * SEC, 500), 0.1);
    let first = r
        .frames(&tok_snap(2, 10 * SEC + 100 * MS, 501), 0.1)
        .keyboard[dot];
    let full = dim(hex(0x1428D8), 45.0);
    assert!(first.b > 0 && first.b < full.b, "{first:?}");
    let level = r.values()[0].expect("level");
    assert!(
        (level - (1.0 - (-0.1_f32 / 0.3).exp())).abs() < 1e-4,
        "{level}"
    );
}

// ---------------------------------------------------------------- peak

#[test]
fn the_peak_holds_for_peak_s_then_decays_linearly() {
    let body = r#"
[[light]]
target = "Number Pad +"
metric = "gpu"
range = [0, 100]
style = "peak"
peak_s = 2
brightness = 0.6
"#;
    let plus = key("Number Pad +");
    let mut r = renderer(body);
    let f = r.frames(&gpu_snap(1, SEC, 80.0), 0.1);
    assert_eq!(f.keyboard[plus], dim(act_color(80.0), 60.0));
    let mut t = SEC;
    let mut seq = 2;
    let mut step = |r: &mut Renderer| {
        t += 100 * MS;
        seq += 1;
        r.frames(&gpu_snap(seq, t, 20.0), 0.1)
    };
    // Held for 2 s.
    for _ in 0..19 {
        let f = step(&mut r);
        assert_eq!(r.values()[0], Some(80.0));
        assert_eq!(f.keyboard[plus], dim(act_color(80.0), 60.0));
    }
    // Then 50 per second (the range in peak_s): 0.5 s later about 55.
    for _ in 0..5 {
        step(&mut r);
    }
    let peak = r.values()[0].expect("peak");
    assert!((50.0..=60.0).contains(&peak), "{peak}");
    // It stops at the live value.
    for _ in 0..20 {
        step(&mut r);
    }
    assert_eq!(r.values()[0], Some(20.0));
}

#[test]
fn a_peak_below_its_threshold_draws_nothing() {
    let body = r#"
[[light]]
target = "Number Pad +"
metric = "gpu"
style = "peak"
threshold = 1.0
"#;
    let mut r = renderer(body);
    let f = r.frames(&gpu_snap(1, SEC, 0.5), 0.1);
    assert_eq!(f.keyboard[key("Number Pad +")], BLACK);
}

// ---------------------------------------------------------------- smoothing

#[test]
fn attack_is_the_rise_time_and_smooth_s_the_fall_time() {
    assert!((ema(0.0, 100.0, 1.0, 5.0, 1.0) - 63.212).abs() < 0.01);
    assert!((ema(100.0, 0.0, 1.0, 5.0, 1.0) - 81.873).abs() < 0.01);
    assert_eq!(ema(3.0, 9.0, 0.0, 5.0, 0.1), 9.0, "zero attack jumps");
    let body = r#"
[[light]]
target = "F1"
metric = "gpu"
attack_s = 0.1
smooth_s = 10
"#;
    let mut r = renderer(body);
    r.frames(&gpu_snap(1, SEC, 0.0), 0.1);
    let mut seq = 1;
    for _ in 0..10 {
        seq += 1;
        r.frames(&gpu_snap(seq, seq * SEC, 100.0), 0.1);
    }
    let up = r.values()[0].expect("up");
    assert!(up > 99.0, "fast attack: {up}");
    for _ in 0..10 {
        seq += 1;
        r.frames(&gpu_snap(seq, seq * SEC, 0.0), 0.1);
    }
    let down = r.values()[0].expect("down");
    assert!(down > 89.0 && down < 92.0, "slow release: {down}");
}

#[test]
fn without_attack_s_smoothing_is_symmetric_as_before() {
    let config = parse(&kb("[[light]]\ntarget = \"F1\"\nsmooth_s = 2.0\n")).expect("ok");
    assert_eq!(config.layers[0].attack_s, 2.0);
}

// ---------------------------------------------------------------- bars

#[test]
fn a_position_gradient_gives_each_key_the_colour_of_its_own_place() {
    let body = r#"
[[light]]
target = 'keyboard.keys["F1".."F10"]'
metric = "gpu"
range = [0, 100]
style = "bar"
edge = "fractional"
gradient = "position"
"#;
    let f1 = key("F1");
    let mut r = renderer(body);
    let frame = r.frames(&gpu_snap(1, SEC, 100.0), 0.1).keyboard;
    for i in 0..10 {
        let at = (i as f32 + 0.5) * 10.0;
        assert_eq!(frame[f1 + i], act_color(at), "F{}", i + 1);
    }
    // 45 %: four whole keys, the fifth half lit, the rest unlit.
    let mut r = renderer(body);
    let frame = r.frames(&gpu_snap(1, SEC, 45.0), 0.1).keyboard;
    assert_eq!(frame[f1 + 3], act_color(35.0));
    assert_eq!(frame[f1 + 4], mix(BLACK, act_color(45.0), 0.5));
    assert_eq!(frame[f1 + 5], BLACK);
}

#[test]
fn fill_to_125_gives_the_last_keys_the_over_range() {
    let body = r#"
[[light]]
target = 'keyboard.keys["F1".."F12"]'
metric = "activity"
range = [0, 100]
fill_to = 125
style = "bar"
edge = "fractional"
gradient = "position"
"#;
    let f1 = key("F1");
    let mut r = renderer(body);
    let mut s = snap(1, SEC);
    s.host.activity_pct = Some(125.0);
    let frame = r.frames(&s, 0.1).keyboard;
    assert_eq!(
        frame[f1 + 11],
        act_color(11.5 / 12.0 * 125.0),
        "F12 is blackbody"
    );
    let mut r = renderer(body);
    s.host.activity_pct = Some(62.5);
    let frame = r.frames(&s, 0.1).keyboard;
    assert_eq!(
        frame[f1 + 5],
        act_color(5.5 / 12.0 * 125.0),
        "F6 full at half"
    );
    assert_eq!(frame[f1 + 6], BLACK, "F7 unlit at half");
}

#[test]
fn value_scaled_brightness_runs_from_lo_to_hi_over_the_range() {
    let body = r#"
[[light]]
target = ["Insert", "Delete"]
metric = "coolant_c"
range = [25, 60]
style = "solid"
brightness = [0.35, 1.0]
"#;
    let mut r = renderer(body);
    let mut s = snap(1, SEC);
    s.host.coolant_c = Some(25.0);
    assert_eq!(
        r.frames(&s, 0.1).keyboard[key("Insert")],
        dim(act_color(0.0), 35.0)
    );
    let mut r = renderer(body);
    s.host.coolant_c = Some(60.0);
    let frame = r.frames(&s, 0.1).keyboard;
    assert_eq!(frame[key("Delete")], act_color(100.0));
}

#[test]
fn a_thresholded_lamp_is_off_below_threshold_and_fades_on_above() {
    let body = r#"
[[light]]
target = "led:97"
metric = "gpu"
range = [0, 100]
threshold = 1.0
brightness = 0.6
attack_s = 0.5
smooth_s = 2.0
"#;
    let enter = key("Number Pad Enter");
    let mut r = renderer(body);
    assert_eq!(r.frames(&gpu_snap(1, SEC, 0.5), 0.1).keyboard[enter], BLACK);
    let mut r = renderer(&body.replace("attack_s = 0.5\nsmooth_s = 2.0\n", ""));
    assert_eq!(
        r.frames(&gpu_snap(1, SEC, 60.0), 0.1).keyboard[enter],
        dim(act_color(60.0), 60.0)
    );
}

// ---------------------------------------------------------------- tween

#[test]
fn targets_come_at_target_hz_and_frames_tween_to_them_in_a_line() {
    let body = r#"
[engine]
tick_hz = 10
target_hz = 2
tween_fps = 10
tween_s = 0.5
[[light]]
target = "F1"
metric = "gpu"
range = [0, 100]
"#;
    let f1 = key("F1");
    let mut r = renderer(body);
    let low = act_color(0.0);
    let high = act_color(100.0);
    let mut shown = Vec::new();
    for tick in 0..12_u64 {
        let gpu = if tick < 3 { 0.0 } else { 100.0 };
        let f = r.frames(&gpu_snap(tick + 1, SEC + tick * 100 * MS, gpu), 0.1);
        shown.push(f.keyboard[f1]);
    }
    // Ticks 0..=4 show the first target (the value moved at tick 3, but the
    // next target is at tick 5); 5..=9 walk to the new one in fifths.
    for (tick, color) in shown.iter().enumerate().take(5) {
        assert_eq!(*color, low, "tick {tick}");
    }
    for (step, tick) in (5..10).enumerate() {
        let amount = (step + 1) as f32 / 5.0;
        assert_eq!(shown[tick], mix(low, high, amount), "tick {tick}");
    }
    assert_eq!(shown[10], high);
    assert_eq!(shown[11], high);
    // No LED ever moves more than a fifth of the way per frame.
    for pair in shown.windows(2) {
        let jump = (i32::from(pair[1].r) - i32::from(pair[0].r)).abs();
        assert!(jump <= (i32::from(high.r) - i32::from(low.r)).abs() / 5 + 1);
    }
}

#[test]
fn without_an_engine_section_every_tick_is_its_own_target() {
    let body = "[[light]]\ntarget = \"F1\"\nmetric = \"gpu\"\n";
    let mut r = renderer(body);
    r.frames(&gpu_snap(1, SEC, 0.0), 0.1);
    let f = r.frames(&gpu_snap(2, SEC + 100 * MS, 100.0), 0.1);
    assert_eq!(f.keyboard[key("F1")], act_color(100.0));
    let config = parse(&kb(body)).expect("ok");
    assert_eq!(config.engine.tick_hz, config.aura.fps);
    assert_eq!(config.engine.target_period_s, 0.0);
    assert_eq!(config.engine.tween_s, 0.0);
    assert_eq!(fade(&[low()], &[low()], 0.3), vec![low()]);
}

fn low() -> Rgb {
    act_color(0.0)
}

#[test]
fn tween_fps_caps_keyboard_writes_below_the_tick_rate() {
    let text = kb(r#"
[engine]
tick_hz = 10
tween_fps = 5
[[light]]
target = "F1"
metric = "gpu"
"#);
    let clock = ManualClock::at(100 * SEC);
    let feed = Feed::new(snap(1, 100 * SEC));
    let kbd = FakeKbOpener::present();
    let aura = FakeOpener::present();
    let mut light = Light::new(Parts {
        clock: clock.clone(),
        snapshots: feed.clone(),
        config_source: FakeConfig::new(&text),
        notify: NoNotify,
        sink: Lines::default(),
        config: parse(&text).expect("config"),
        aura: Some(Box::new(AuraBackend::new(aura.clone()))),
        keyboard: Some(Box::new(KeyboardBackend::new(kbd.clone()))),
    });
    let mut clock_ref = clock.clone();
    for i in 0..20_u64 {
        let mut s = gpu_snap(i + 1, clock.now(), (i * 5) as f32);
        s.host.activity_pct = Some((i * 5) as f32);
        feed.set(Ok(s));
        light.tick();
        llama_light::service::Clock::sleep(&mut clock_ref, std::time::Duration::from_millis(100));
    }
    assert_eq!(kbd.frames(), 10, "5 fps over 2 s");
    // The fans keep aura.fps (10): every changed frame goes out.
    assert_eq!(light.frames_sent(), 20);
}

// ---------------------------------------------------------------- targets

#[test]
fn led_n_targets_the_nth_known_key() {
    let config = parse(&kb(r##"
[[light]]
target = "led:97"
color = "#FFFFFF"
[[light]]
target = ["led:110", "Number Pad +"]
color = "#FFFFFF"
"##))
    .expect("ok");
    assert_eq!(config.layers[0].target, Target::KeyboardKeys(vec![97]));
    assert_eq!(KEYS[97].name, "Number Pad Enter");
    assert_eq!(
        config.layers[1].target,
        Target::KeyboardKeys(vec![110, key("Number Pad +")])
    );
    assert_eq!(KEYS[110].name, "Number Pad .");
    let err = error(&kb("[[light]]\ntarget = \"led:111\"\n"));
    assert!(err.contains("light[0].target = \"led:111\""), "{err}");
    assert!(err.contains("led:0..=led:110"), "{err}");
    let err = error(&kb("[[light]]\ntarget = \"led:x\"\n"));
    assert!(err.contains("\"x\" is not an LED index"), "{err}");
    let err = error("[[light]]\ntarget = \"led:3\"\n");
    assert!(err.contains("[keyboard] enabled = true"), "{err}");
}

// ---------------------------------------------------------------- base

#[test]
fn base_fills_every_led_no_entry_lights() {
    let body = r##"
[base]
color = "#1428D8"
brightness = 0.10
[[light]]
target = 'keyboard.keys["F1".."F10"]'
metric = "gpu"
style = "bar"
"##;
    let base = dim(hex(0x1428D8), 10.0);
    assert_eq!(base, Rgb { r: 2, g: 4, b: 22 });
    let mut r = renderer(body);
    let frame = r.frames(&gpu_snap(1, SEC, 30.0), 0.1).keyboard;
    assert_eq!(frame[key("F1")], act_color(30.0));
    assert_eq!(frame[key("F4")], base, "an unlit bar key shows the base");
    for name in ["Escape", "Space", "Up Arrow", "Number Pad ."] {
        assert_eq!(frame[key(name)], base, "{name}");
    }
    // An integer is a whole percent: 10 is the same as 0.10.
    let same = parse(&kb(&body.replace("0.10", "10"))).expect("ok");
    assert_eq!(same.base, Some(base));
    // Without [base] unclaimed keys stay black.
    let mut r = renderer(&body.replace("[base]\ncolor = \"#1428D8\"\nbrightness = 0.10\n", ""));
    assert_eq!(
        r.frames(&gpu_snap(1, SEC, 30.0), 0.1).keyboard[key("Space")],
        BLACK
    );
}

// ---------------------------------------------------------------- validation

#[test]
fn validation_errors_name_the_key_and_the_problem() {
    let cases: &[(&str, &str)] = &[
        (
            "[engine]\ntick_hz = 21\n",
            "engine.tick_hz = 21 is out of range",
        ),
        (
            "[engine]\ntween_fps = 0\n",
            "engine.tween_fps = 0 is out of range",
        ),
        (
            "[engine]\ntick_hz = 5\ntween_fps = 10\n",
            "engine.tween_fps = 10 is above engine.tick_hz = 5",
        ),
        (
            "[engine]\ntarget_hz = 0\n",
            "engine.target_hz = 0 is out of range",
        ),
        (
            "[engine]\ntarget_hz = 2\ntween_s = 1.0\n",
            "engine.tween_s = 1 is longer than one target period",
        ),
        ("[engine]\nfps = 2\n", "unknown field `fps`"),
        ("[base]\nbrightness = 0.1\n", "[base] needs color"),
        (
            "[base]\ncolor = \"blue\"\n",
            "base.color = \"blue\" is not a #RRGGBB colour",
        ),
        (
            "[base]\ncolor = \"#000000\"\nbrightness = 1.5\n",
            "base.brightness = 1.5 is out of range; a fraction is 0.0..=1.0",
        ),
        (
            "[palette.act]\nstops = [[0, \"#000000\"]]\n",
            "palette.act.stops needs at least 2 stops",
        ),
        (
            "[palette.act]\nstops = [[0, \"#000000\"], [130, \"#FFFFFF\"]]\n",
            "palette.act.stops[1]: position is 130.0%",
        ),
        (
            "[[light]]\npalette = \"hot\"\n",
            "light[0].palette = \"hot\" is not a palette",
        ),
        (
            "[[light]]\nbrightness = 1.2\n",
            "light[0].brightness = 1.2 is out of range",
        ),
        (
            "[[light]]\nbrightness = [0.1]\n",
            "light[0].brightness must be one number or [lo, hi], got 1 numbers",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nbrightness = [0.1, 1.0]\n",
            "light[0].brightness = [lo, hi] is for style = \"solid\" (this entry is \"bar\")",
        ),
        (
            "[[light]]\nattack_s = 61\n",
            "light[0].attack_s = 61 is out of range",
        ),
        (
            "[[light]]\nmetric = \"gpu\"\nrate_window_s = 2\n",
            "light[0].rate_window_s: only a counter rate (tokens_rate) has a rate window; gpu",
        ),
        (
            "[[light]]\nmetric = \"tokens_rate\"\nrate_window_s = 61\n",
            "light[0].rate_window_s = 61 is out of range",
        ),
        (
            "[[light]]\nmetric = \"decoded_total\"\n",
            "light[0].metric = \"decoded_total\" is the raw token counter; use it with style = \"gate\"",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nedge = \"soft\"\n",
            "light[0].edge = \"soft\" is not \"round\" or \"fractional\"",
        ),
        (
            "[[light]]\nstyle = \"bar\"\ngradient = \"rainbow\"\n",
            "light[0].gradient = \"rainbow\" is not \"value\" or \"position\"",
        ),
        (
            "[[light]]\nedge = \"fractional\"\n",
            "light[0].edge is for style = \"bar\" or \"ladder\" (this entry is \"solid\")",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nfill_to = 130\n",
            "light[0].fill_to = 130 is out of range; allowed 100..=125",
        ),
        (
            "[[light]]\nhold_s = 1\n",
            "light[0].hold_s is for style = \"gate\"",
        ),
        (
            "[[light]]\npeak_s = 1\n",
            "light[0].peak_s is for style = \"peak\"",
        ),
        (
            "[[light]]\nthresholds = [1]\n",
            "light[0].thresholds is for style = \"ladder\"",
        ),
        (
            "[[light]]\nstyle = \"bar\"\nthreshold = 1\n",
            "light[0].threshold is for style = \"solid\" or \"pulse\" or \"gate\" or \"peak\"",
        ),
        (
            "[[light]]\nstyle = \"ladder\"\nthresholds = [10, 20]\n",
            "light[0].thresholds has 2 values but the target has 6 LEDs",
        ),
        (
            "[[light]]\nstyle = \"ladder\"\nthresholds = [10, 20, 20, 30, 40, 50]\n",
            "light[0].thresholds must ascend; [2] = 20 is not above [1] = 20",
        ),
        (
            "[[light]]\nstyle = \"ladder\"\nthresholds = [0, 20, 30, 40, 50, 60]\n",
            "light[0].thresholds[0] = 0 must be above range min 0",
        ),
        (
            "[[light]]\nstyle = \"gate\"\n",
            "light[0]: style = \"gate\" needs color",
        ),
        (
            "[[light]]\nstyle = \"gate\"\ncolor = \"#FFFFFF\"\npalette = \"act\"\n",
            "light[0]: a gate shows color; remove palette",
        ),
        (
            "[[light]]\nstyle = \"gate\"\ncolor = \"#FFFFFF\"\nshimmer = { depth = 2.0, hz = 1 }\n",
            "light[0].shimmer.depth = 2 is out of range",
        ),
        (
            "[[light]]\nstyle = \"gate\"\ncolor = \"#FFFFFF\"\nshimmer = { depth = 0.2, hz = 11 }\n",
            "light[0].shimmer.hz = 11 is out of range",
        ),
        (
            "[[light]]\nstyle = \"peak\"\npeak_s = 0\n",
            "light[0].peak_s must be above 0",
        ),
        (
            "[[light]]\ncolor = \"#FFFFFF\"\nattack_s = 1\n",
            "light[0]: color is a fixed colour; remove attack_s",
        ),
        (
            "[[light]]\nstyle = \"sparkle\"\n",
            "use solid, ring, bar, pulse, ladder, gate or peak",
        ),
        (
            "[aura]\nstyle = \"ladder\"\n",
            "aura.style = \"ladder\": the default entry is solid, ring, bar or pulse",
        ),
        (
            "[[light]]\ntarget = [\"F1\"]\n",
            "light[0].target: a list of keys needs [keyboard] enabled = true",
        ),
    ];
    for (text, needle) in cases {
        let err = error(text);
        assert!(
            err.contains(needle),
            "{text}\nwanted: {needle}\ngot:    {err}"
        );
    }
    let kb_cases: &[(&str, &str)] = &[
        (
            "[[light]]\ntarget = []\n",
            "light[0].target = []: list at least one key",
        ),
        (
            "[[light]]\ntarget = [\"F1\", \"F1\"]\n",
            "light[0].target[1]: \"F1\" is listed twice",
        ),
        (
            "[[light]]\ntarget = [\"f1\"]\n",
            "light[0].target[0] = \"f1\": \"f1\" is not a known key; did you mean \"F1\"?",
        ),
        (
            "[[light]]\ntarget = [\"F1\", \"F2\"]\nstyle = \"ladder\"\nthresholds = [5]\n",
            "light[0].thresholds has 1 values but the target has 2 LEDs",
        ),
    ];
    for (body, needle) in kb_cases {
        let err = error(&kb(body));
        assert!(
            err.contains(needle),
            "{body}\nwanted: {needle}\ngot:    {err}"
        );
    }
}

#[test]
fn old_keys_keep_their_meaning() {
    let config = parse(&kb(r#"
[[light]]
target = 'keyboard.keys["W","A","S","D"]'
metric = "gpu_temp"
style = "pulse"
palette = "thermal"
brightness = 30
smooth_s = 1.5
"#))
    .expect("ok");
    let layer = &config.layers[0];
    assert_eq!(layer.style, Style::Pulse);
    assert_eq!(layer.brightness, 30.0);
    assert_eq!(layer.attack_s, 1.5);
    assert_eq!(layer.rate_window_s, 0.0);
    assert_eq!(config.base, None);
}

// ---------------------------------------------------------------- example

fn example_block(name: &str) -> String {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../packaging/light.example.toml"
    ))
    .expect("example");
    let mut body = String::new();
    let mut inside = false;
    for line in text.lines() {
        if line == format!("# --- example: {name}") {
            inside = true;
        } else if inside && line.starts_with("# --- end") {
            return body;
        } else if inside {
            let line = line
                .strip_prefix("# ")
                .or_else(|| line.strip_prefix('#'))
                .unwrap_or(line);
            body.push_str(line);
            body.push('\n');
        }
    }
    panic!("no example {name}");
}

#[test]
fn the_keyboard_c_example_is_variant_c1() {
    let config = parse(&example_block("keyboard-c")).expect("keyboard-c");
    assert!(config.keyboard.enabled);
    assert_eq!(config.engine.tick_hz, 10);
    assert_eq!(config.engine.tween_fps, 10);
    assert!((config.engine.target_period_s - 0.5).abs() < 1e-6);
    assert!((config.engine.tween_s - 0.5).abs() < 1e-6);
    let styles: Vec<Style> = config.layers.iter().map(|l| l.style).collect();
    assert_eq!(
        styles,
        [
            Style::Gauge,
            Style::Gauge,
            Style::Gauge,
            Style::Solid,
            Style::Gauge,
            Style::Gauge,
            Style::Pulse,
            // The fans keep their default.
            Style::Solid,
        ]
    );
    assert_eq!(config.layers[7].target, Target::AuraFans);
    // The two-row bars are interleaved by x: each alternates rows.
    let Target::KeyboardKeys(gpu) = &config.layers[1].target else {
        panic!("gpu target");
    };
    assert_eq!(gpu.len(), 28);
    let names: Vec<&str> = gpu.iter().take(4).map(|k| KEYS[*k].name).collect();
    assert_eq!(names, ["`", "Tab", "1", "Q"]);
    let Target::KeyboardKeys(topk) = &config.layers[2].target else {
        panic!("top-k target");
    };
    assert_eq!(topk.len(), 25);
    // The dial: an 8-key ring round "5", the outer lap, the centre.
    let Target::KeyboardKeys(ring) = &config.layers[4].target else {
        panic!("ring target");
    };
    assert_eq!(ring.len(), 8);
    assert_eq!(KEYS[ring[0]].name, "Number Pad 8");
    assert_eq!(
        config.layers[6].target,
        Target::KeyboardKeys(vec![key("Number Pad 5")])
    );
    for layer in &config.layers[4..=6] {
        assert_eq!(layer.rate_window_s, 2.0, "tokens/s is windowed");
    }
    // Every LED is claimed by at most one entry, so no entry hides another.
    let mut seen = vec![false; KEYS.len()];
    for layer in &config.layers {
        if let Target::KeyboardKeys(keys) = &layer.target {
            for k in keys {
                assert!(!seen[*k], "{} claimed twice", KEYS[*k].name);
                seen[*k] = true;
            }
        }
    }
    // An idle machine: the bars and the dial show the base.
    let base = config.base.expect("base");
    let mut r = Renderer::new(config);
    let mut s = snap(1, SEC);
    s.host.activity_pct = Some(0.0);
    s.host.gpu_pct = Some(0.0);
    let frame = r.frames(&s, 0.1).keyboard;
    assert_eq!(frame[key("Escape")], base);
    assert_eq!(frame[key("F1")], base);
    assert_eq!(frame[key("Number Pad 8")], base);
}
