//! Mapping and colour: the act ramp, stops, scales, every style, smoothing,
//! the brightness cap and the tokens rate.

mod common;

use common::{SEC, snap};
use llama_core::color::{Rgb, hex, mix};
use llama_light::config::parse;
use llama_light::mapping::{NEUTRAL, Renderer, cap, fade, gauge_count, pulse_level};
use llama_light::metric::{Metric, TokenRate};
use llama_light::palette::{Palette, Scale, dim};

fn frame(config: &str, activity: f32) -> Vec<Rgb> {
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut s = snap(1, SEC);
    s.host.activity_pct = Some(activity);
    renderer.aura_frame(&s, 0.1)
}

const FULL: &str = "[aura]\nbrightness_max = 100\n";

#[test]
fn the_default_is_the_act_palette_color_of_activity_on_all_six_leds() {
    for activity in [0.0, 25.0, 50.0, 100.0, 112.0, 125.0] {
        let got = frame(FULL, activity);
        assert_eq!(got.len(), 6);
        assert!(
            got.iter().all(|led| *led == Palette::Act.color(activity)),
            "{activity}: {got:?}"
        );
    }
}

#[test]
fn act_is_blue_to_red_held_at_red_past_100() {
    // llama-light's own LED ramp (#93): no blackbody tail, unlike the LCD's
    // act_color (crates/llama-light/src/palette.rs has the shape tests).
    assert_eq!(Palette::Act.color(0.0), hex(0x0010FF));
    assert_eq!(Palette::Act.color(100.0), hex(0xFF0000));
    assert_eq!(Palette::Act.color(125.0), hex(0xFF0000));
    assert_eq!(Palette::Act.color(60.0), hex(0xC000E0));
}

#[test]
fn brightness_max_caps_every_led() {
    let got = frame("[aura]\nbrightness_max = 50\n", 100.0);
    assert!(
        got.iter()
            .all(|led| *led == dim(Palette::Act.color(100.0), 50.0)),
        "{got:?}"
    );
    assert_eq!(
        dim(hex(0xFF8000), 50.0),
        Rgb {
            r: 128,
            g: 64,
            b: 0
        }
    );
    let dark = frame("[aura]\nbrightness_max = 0\n", 100.0);
    assert!(dark.iter().all(|led| *led == Rgb::default()));
    assert_eq!(
        cap(&[hex(0xFFFFFF)], 80),
        vec![Rgb {
            r: 204,
            g: 204,
            b: 204
        }]
    );
}

#[test]
fn activity_falls_back_to_load() {
    let mut s = snap(1, SEC);
    s.host.activity_pct = None;
    s.host.load_pct = Some(42.0);
    assert_eq!(Metric::Activity.read(&s, None), Some(42.0));
    assert_eq!(Metric::Gpu.read(&s, None), Some(40.0));
    assert_eq!(Metric::Coolant.read(&s, None), Some(33.0));
    assert_eq!(Metric::GpuTemp.read(&s, None), Some(60.0));
    assert_eq!(Metric::CpuTemp.read(&s, None), Some(55.0));
    assert_eq!(Metric::Mem.read(&s, None), Some(60.0));
    assert_eq!(Metric::Cpu.read(&s, None), Some(20.0));
    assert_eq!(Metric::Load.read(&s, None), Some(42.0));
}

#[test]
fn stops_interpolate_linearly_between_positions() {
    let stops = Palette::Stops(vec![
        (0.0, hex(0x000000)),
        (50.0, hex(0xFF0000)),
        (100.0, hex(0xFFFFFF)),
    ]);
    assert_eq!(stops.color(0.0), hex(0x000000));
    assert_eq!(stops.color(25.0), Rgb { r: 128, g: 0, b: 0 });
    assert_eq!(stops.color(50.0), hex(0xFF0000));
    assert_eq!(
        stops.color(75.0),
        Rgb {
            r: 255,
            g: 128,
            b: 128
        }
    );
    assert_eq!(
        stops.color(125.0),
        hex(0xFFFFFF),
        "past the last stop holds"
    );
    // Through the config: coolant 30 °C on [20, 40] is 50 %.
    let config = "[aura]\nbrightness_max = 100\n[[light]]\nmetric = \"coolant\"\nrange = [20, 40]\n\
                  stops = [[20, \"#000000\"], [40, \"#FF0000\"]]\n";
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut s = snap(1, SEC);
    s.host.coolant_c = Some(30.0);
    assert!(
        renderer
            .aura_frame(&s, 0.1)
            .iter()
            .all(|l| *l == Rgb { r: 128, g: 0, b: 0 })
    );
}

#[test]
fn linear_and_log_scales() {
    assert_eq!(Scale::Linear.percent(50.0, (0.0, 100.0)), 50.0);
    assert_eq!(Scale::Linear.percent(30.0, (20.0, 40.0)), 50.0);
    assert_eq!(Scale::Linear.percent(-5.0, (0.0, 100.0)), 0.0);
    assert_eq!(Scale::Linear.percent(1000.0, (0.0, 100.0)), 125.0);
    let log = |v| Scale::Log.percent(v, (1.0, 100.0));
    assert!((log(10.0) - 50.0).abs() < 1e-3, "{}", log(10.0));
    assert!((log(1.0)).abs() < 1e-3);
    assert!((log(100.0) - 100.0).abs() < 1e-3);
    assert_eq!(log(0.0), 0.0);
    assert_eq!(log(0.5), 0.0, "below min clamps to 0");
}

#[test]
fn solid_fills_every_fan_in_a_chain() {
    let config = "[aura]\nbrightness_max = 100\nfans = \"chain\"\nchain_len = 3\n";
    let got = frame(config, 50.0);
    assert_eq!(got.len(), 18);
    assert!(got.iter().all(|l| *l == Palette::Act.color(50.0)));
}

#[test]
fn per_fan_chain_entries_light_only_their_fan() {
    let config =
        "[aura]\nbrightness_max = 100\nchain = [ { metric = \"gpu\" }, { metric = \"cpu\" } ]\n";
    let got = frame(config, 50.0);
    assert_eq!(got.len(), 12);
    assert!(
        got[..6].iter().all(|l| *l == Palette::Act.color(40.0)),
        "fan 0 is gpu 40"
    );
    assert!(
        got[6..].iter().all(|l| *l == Palette::Act.color(20.0)),
        "fan 1 is cpu 20"
    );
}

#[test]
fn ring_lights_k_of_six_from_led_zero() {
    assert_eq!(gauge_count(0.0, 6), 0);
    assert_eq!(gauge_count(1.0, 6), 1, "anything above zero shows one LED");
    assert_eq!(gauge_count(50.0, 6), 3);
    assert_eq!(gauge_count(66.0, 6), 4);
    assert_eq!(gauge_count(100.0, 6), 6);
    assert_eq!(gauge_count(125.0, 6), 6);
    let got = frame("[aura]\nbrightness_max = 100\nstyle = \"ring\"\n", 50.0);
    let color = Palette::Act.color(50.0);
    assert_eq!(&got[..3], &[color; 3]);
    assert_eq!(&got[3..], &[Rgb::default(); 3], "unlit LEDs are off");
}

#[test]
fn a_ring_over_a_solid_leaves_the_solid_on_unlit_leds() {
    let config = "[aura]\nbrightness_max = 100\n\
                  [[light]]\nmetric = \"mem\"\npalette = \"mono\"\n\
                  [[light]]\nmetric = \"gpu\"\nstyle = \"ring\"\n";
    let got = frame(config, 0.0);
    let base = Palette::named("mono").expect("mono").color(60.0);
    assert_eq!(
        &got[..2],
        &[Palette::Act.color(40.0); 2],
        "gpu 40 % is 2 of 6"
    );
    assert_eq!(&got[2..], &[base; 4]);
}

#[test]
fn a_bar_over_a_chain_range_is_one_gauge_across_the_fans() {
    let config = "[aura]\nbrightness_max = 100\nfans = \"chain\"\nchain_len = 4\n\
                  [[light]]\ntarget = \"aura.chain[1..3]\"\nstyle = \"bar\"\n";
    let got = frame(config, 50.0);
    let lit: Vec<usize> = (0..24).filter(|i| got[*i] != Rgb::default()).collect();
    assert_eq!(
        lit,
        (6..12).collect::<Vec<_>>(),
        "half of 12 LEDs starting at fan 1"
    );
}

#[test]
fn pulse_breathes_faster_with_the_value() {
    assert!((pulse_level(0.0) - 1.0).abs() < 1e-6);
    assert!((pulse_level(0.5) - 0.35).abs() < 1e-6);
    let config = |_: ()| parse("[aura]\nbrightness_max = 100\nstyle = \"pulse\"\n").expect("pulse");
    // Low value: 0.2 Hz. After 1.25 s the phase is a quarter breath.
    let mut slow = Renderer::new(config(()));
    let mut fast = Renderer::new(config(()));
    let mut low = snap(1, SEC);
    low.host.activity_pct = Some(0.0);
    let mut high = snap(1, SEC);
    high.host.activity_pct = Some(100.0);
    let slow_frame = slow.aura_frame(&low, 1.25);
    // High value: 2 Hz. After 0.25 s the phase is half a breath: the floor.
    let fast_frame = fast.aura_frame(&high, 0.25);
    assert_eq!(
        slow_frame[0],
        dim(Palette::Act.color(0.0), pulse_level(0.25) * 100.0)
    );
    assert_eq!(fast_frame[0], dim(Palette::Act.color(100.0), 35.0));
    // The frame moves every tick while pulsing.
    let next = fast.aura_frame(&high, 0.1);
    assert_ne!(next, fast_frame);
}

#[test]
fn idle_color_below_min_and_neutral_when_missing() {
    let config = "[aura]\nbrightness_max = 100\n[[light]]\nmetric = \"gpu\"\nrange = [10, 100]\nidle_color = \"#000040\"\n";
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut s = snap(1, SEC);
    s.host.gpu_pct = Some(5.0);
    assert!(
        renderer
            .aura_frame(&s, 0.1)
            .iter()
            .all(|l| *l == hex(0x000040))
    );
    s.host.gpu_pct = None;
    assert!(
        renderer
            .aura_frame(&s, 0.1)
            .iter()
            .all(|l| *l == hex(0x000040))
    );
    let plain = "[aura]\nbrightness_max = 100\n[[light]]\nmetric = \"gpu\"\n";
    let mut renderer = Renderer::new(parse(plain).expect("config"));
    assert!(renderer.aura_frame(&s, 0.1).iter().all(|l| *l == NEUTRAL));
}

#[test]
fn smoothing_is_an_ema_with_the_configured_time_constant() {
    let config = "[aura]\nbrightness_max = 100\n[[light]]\nmetric = \"gpu\"\nsmooth_s = 1.0\npalette = \"mono\"\n";
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut s = snap(1, SEC);
    s.host.gpu_pct = Some(0.0);
    renderer.aura_frame(&s, 0.1);
    s.host.gpu_pct = Some(100.0);
    let got = renderer.aura_frame(&s, 1.0)[0];
    let expected = 100.0 * (1.0 - (-1.0f32).exp());
    assert_eq!(got, Palette::named("mono").expect("mono").color(expected));
}

#[test]
fn per_entry_brightness_scales_under_the_cap() {
    let got = frame(
        "[aura]\nbrightness_max = 50\n[[light]]\nbrightness = 50\n",
        100.0,
    );
    assert_eq!(got[0], dim(dim(Palette::Act.color(100.0), 50.0), 50.0));
}

#[test]
fn tokens_rate_is_the_counter_delta_over_time() {
    let mut rate = TokenRate::default();
    let mut s = snap(1, 10 * SEC);
    s.tokens.decoded_total = Some(1000);
    assert_eq!(rate.update(&s), Some(0.0), "first sample is a baseline");
    assert_eq!(rate.update(&s), Some(0.0), "same seq changes nothing");
    let mut s2 = snap(2, 12 * SEC);
    s2.tokens.decoded_total = Some(1100);
    assert_eq!(rate.update(&s2), Some(50.0));
    let mut restarted = snap(1, 13 * SEC);
    restarted.run_id = 8;
    restarted.tokens.decoded_total = Some(5);
    assert_eq!(
        rate.update(&restarted),
        Some(0.0),
        "a new run is a new baseline"
    );
    let mut none = snap(2, 14 * SEC);
    none.run_id = 8;
    none.tokens.decoded_total = None;
    assert_eq!(rate.update(&none), None);
}

#[test]
fn fade_mixes_each_led_toward_the_target() {
    let from = vec![hex(0xFF0000), hex(0x00FF00)];
    let to = vec![hex(0x000000); 2];
    assert_eq!(fade(&from, &to, 0.0), from);
    assert_eq!(fade(&from, &to, 1.0), to);
    assert_eq!(
        fade(&from, &to, 0.5)[0],
        mix(hex(0xFF0000), hex(0x000000), 0.5)
    );
}
