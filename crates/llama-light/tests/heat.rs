//! The heat bar against the prototype's demo traces, and the renderer wiring.
//!
//! `ref-demo-transit60.csv` and `ref-demo-transit10.csv` are the flow model
//! stepped at `dt = 0.05` through the prototype's `DEMO` load. The load is
//! sampled at the start of each step. Rows are the prototype's once-a-second
//! dump (`t_s` is the time after the step). Values are percent of a
//! `[0, 125]` range: the prototype's 0..1.25 multiplied by 100.

mod common;

use common::{SEC, snap};
use llama_core::color::{BLACK, Rgb, hex};
use llama_light::config::parse;
use llama_light::heat::{HeatBar, HeatParams};
use llama_light::mapping::{NEUTRAL, Renderer};
use llama_light::palette::{Palette, dim};

const DEMO: [(f64, f64); 12] = [
    (5.0, 0.0),
    (2.0, 1.25),
    (18.0, 0.0),
    (2.0, 1.25),
    (2.0, 0.0),
    (2.0, 1.25),
    (2.0, 0.0),
    (2.0, 1.25),
    (19.0, 0.0),
    (90.0, 1.0),
    (90.0, 1.25),
    (90.0, 0.0),
];

/// The prototype's `demo_load`: seconds into the looping sequence, fraction
/// of range (1.0 is 100 %).
fn demo_load(mut t: f64) -> f64 {
    let len: f64 = DEMO.iter().map(|(seconds, _)| *seconds).sum();
    t %= len;
    for (seconds, value) in DEMO {
        if t < seconds {
            return value;
        }
        t -= seconds;
    }
    0.0
}

struct Row {
    t_s: f64,
    load_pct: f64,
    heater_pct: f64,
    leds: Vec<f64>,
}

fn fixture(name: &str) -> Vec<Row> {
    let text = std::fs::read_to_string(format!(
        "{}/tests/fixtures/heat/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|err| panic!("{name}: {err}"));
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 {
            continue;
        }
        let fields: Vec<f64> = line
            .split(',')
            .map(|field| {
                field
                    .parse()
                    .unwrap_or_else(|_| panic!("{name}:{index}: {field}"))
            })
            .collect();
        assert_eq!(fields.len(), 31, "{name}:{index}");
        rows.push(Row {
            t_s: fields[0],
            load_pct: fields[1],
            heater_pct: fields[2],
            leds: fields[3..].to_vec(),
        });
    }
    rows
}

fn replay(transit_s: f64, conduction_s: f64) -> Vec<(f64, f64, f64, Vec<f64>)> {
    let params = HeatParams {
        transit_s,
        foot_leds: 5,
        conduction_s,
        attack_s: 0.7,
        smooth_s: 0.7,
    };
    let mut bar = HeatBar::new(28, params).expect("28 LEDs");
    let dt = 0.05;
    let mut sim_t = 0.0;
    let mut next_print = 0.0;
    let mut samples = Vec::new();
    while sim_t < 324.0 + 1e-9 {
        let load = demo_load(sim_t) * 100.0;
        bar.step(load, dt);
        sim_t += dt;
        if sim_t >= next_print {
            samples.push((sim_t, load, bar.heater(), bar.leds()));
            next_print += 1.0;
        }
    }
    samples
}

fn matches(name: &str, transit_s: f64, conduction_s: f64) {
    let rows = fixture(name);
    let samples = replay(transit_s, conduction_s);
    assert_eq!(
        samples.len(),
        rows.len(),
        "{name}: replay wrote {} rows, the fixture has {}",
        samples.len(),
        rows.len()
    );
    for (index, (sample, row)) in samples.iter().zip(&rows).enumerate() {
        let (t_s, load, heater, leds) = sample;
        assert!(
            (t_s - row.t_s).abs() < 1e-6,
            "{name} row {index}: replay t {t_s} vs fixture {}",
            row.t_s
        );
        assert!(
            (load - row.load_pct).abs() < 1e-6,
            "{name} row {index}: load {load} vs {}",
            row.load_pct
        );
        assert!(
            (heater - row.heater_pct).abs() <= 0.05,
            "{name} t {}: heater {heater} vs {} (tolerance 0.05)",
            row.t_s,
            row.heater_pct
        );
        for (led, (got, want)) in leds.iter().zip(&row.leds).enumerate() {
            assert!(
                (got - want).abs() <= 0.5,
                "{name} t {} led{led}: {got} vs {want} (tolerance 0.5)",
                row.t_s
            );
        }
    }
}

#[test]
fn transit_60_matches_the_prototype_demo() {
    matches("ref-demo-transit60.csv", 60.0, 20_000.0);
}

#[test]
fn transit_10_matches_the_prototype_demo() {
    matches("ref-demo-transit10.csv", 10.0, 5_000.0);
}

fn chain(body: &str) -> String {
    format!("[aura]\nbrightness_max = 100\nfans = \"chain\"\n{body}")
}

fn activity(value: Option<f32>, seq: u64) -> llama_core::wire::SnapshotV1 {
    let mut snapshot = snap(seq, SEC);
    snapshot.host.activity_pct = value;
    snapshot
}

fn paint(config: &str, value: Option<f32>, dt_s: f32, steps: u32) -> Vec<Rgb> {
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut frame = Vec::new();
    for step in 0..steps {
        frame = renderer.aura_frame(&activity(value, u64::from(step) + 1), dt_s);
    }
    frame
}

#[test]
fn follow_paints_every_led_in_the_top_leds_colour() {
    let config = chain(
        "chain_len = 2\nleds_per_fan = 14\n\
         [[light]]\n\
         target = \"aura.chain[0]\"\n\
         style = \"heat\"\n\
         brightness = 50\n\
         attack_s = 0\n\
         smooth_s = 0\n\
         follow = \"aura.chain[1]\"\n",
    );
    let frame = paint(&config, Some(125.0), 0.5, 40);
    let hot = dim(Palette::Act.color(125.0), 50.0);
    assert_eq!(hot, Rgb { r: 128, g: 0, b: 0 });
    assert_eq!(frame.len(), 28);
    assert_eq!(frame[13], hot, "heater LED");
    assert!(
        frame[14..].iter().all(|led| *led == hot),
        "follow span {frame:?}"
    );
    assert_ne!(frame[0], hot, "the sink stays cooler than the heater");
}

#[test]
fn a_reversed_span_heats_the_last_listed_led() {
    let config = chain(
        "chain_len = 1\nleds_per_fan = 14\n\
         [[light]]\n\
         target = \"aura.chain[1..0]\"\n\
         style = \"heat\"\n\
         foot_leds = 3\n\
         attack_s = 0\n\
         smooth_s = 0\n",
    );
    let frame = paint(&config, Some(125.0), 0.5, 60);
    let hot = hex(0xFF0000);
    assert_eq!(frame[0], hot, "last listed LED is the heater");
    assert_ne!(frame[13], hot, "first listed LED is the sink");
}

#[test]
fn an_unknown_value_keeps_drawing_and_cools_the_bar() {
    let config = chain(
        "chain_len = 2\nleds_per_fan = 14\n\
         [[light]]\n\
         target = \"aura.chain[0]\"\n\
         style = \"heat\"\n\
         attack_s = 0\n\
         smooth_s = 0\n\
         follow = \"aura.chain[1]\"\n",
    );
    let mut renderer = Renderer::new(parse(&config).expect("config"));
    let cold = renderer.aura_frame(&activity(None, 1), 0.0);
    let blue = Palette::Act.color(0.0);
    assert_eq!(cold[13], blue, "a cold bar draws the zero colour");
    assert_ne!(cold[13], NEUTRAL);
    assert!(cold[14..].iter().all(|led| *led == blue));

    let mut hot = cold;
    for step in 0..40_u64 {
        hot = renderer.aura_frame(&activity(Some(125.0), step + 2), 0.5);
    }
    let red = hex(0xFF0000);
    assert_eq!(hot[13], red);
    assert!(hot[14..].iter().all(|led| *led == red));

    let held = renderer.aura_frame(&activity(None, 50), 0.1);
    assert_eq!(held[13], red, "one cold cell does not leave the top tent");
    assert_ne!(held[13], NEUTRAL);
    assert!(held[14..].iter().all(|led| *led == held[13]));

    let mut cooled = held;
    for step in 0..60_u64 {
        cooled = renderer.aura_frame(&activity(None, 51 + step), 0.5);
    }
    assert_ne!(cooled[13], red, "the cold traveled down to the heater LED");
    assert_ne!(cooled[13], NEUTRAL);
    assert!(cooled[14..].iter().all(|led| *led == cooled[13]));
}

#[test]
fn below_range_idle_colour_covers_both_spans_while_the_bar_keeps_cooling() {
    let config = chain(
        "chain_len = 2\nleds_per_fan = 14\n\
         [[light]]\n\
         target = \"aura.chain[0]\"\n\
         style = \"heat\"\n\
         attack_s = 0\n\
         smooth_s = 0\n\
         idle_color = \"#00FF00\"\n\
         follow = \"aura.chain[1]\"\n",
    );
    let mut cooling = Renderer::new(parse(&config).expect("config"));
    let mut staying = Renderer::new(parse(&config).expect("config"));
    for step in 0..40_u64 {
        let snapshot = activity(Some(125.0), step + 1);
        cooling.aura_frame(&snapshot, 0.5);
        staying.aura_frame(&snapshot, 0.5);
    }
    let idle = hex(0x00FF00);
    for step in 0..60_u64 {
        let shown = cooling.aura_frame(&activity(Some(-1.0), 100 + step), 0.5);
        assert!(
            shown.iter().all(|led| *led == idle),
            "idle covers the bar and the follow span"
        );
        staying.aura_frame(&activity(Some(125.0), 100 + step), 0.5);
    }
    let cooled = cooling.aura_frame(&activity(Some(125.0), 200), 0.0);
    let hot = staying.aura_frame(&activity(Some(125.0), 200), 0.0);
    assert_ne!(
        cooled[13], hot[13],
        "time below range still stepped the bar"
    );
    assert_eq!(hot[13], hex(0xFF0000));
    assert!(cooled[14..].iter().all(|led| *led == cooled[13]));
}

#[test]
fn a_fresh_renderer_is_cold_and_an_impossible_foot_does_not_panic() {
    let config = chain(
        "chain_len = 1\nleds_per_fan = 14\n\
         [[light]]\nstyle = \"heat\"\nattack_s = 0\nfoot_leds = 3\n",
    );
    let heated = paint(&config, Some(125.0), 0.5, 40);
    assert_eq!(heated[13], hex(0xFF0000));
    let mut fresh = Renderer::new(parse(&config).expect("config"));
    let cold = fresh.aura_frame(&activity(Some(125.0), 1), 0.0);
    assert_eq!(
        cold[13],
        Palette::Act.color(0.0),
        "the first tick with dt 0 has not integrated"
    );

    let mut broken = parse(&config).expect("config");
    broken.layers[0].foot_leds = 100;
    let mut renderer = Renderer::new(broken);
    let frame = renderer.aura_frame(&activity(Some(125.0), 1), 1.0);
    assert_eq!(frame.len(), 14);
    assert!(
        frame.iter().all(|led| *led == BLACK),
        "a foot the span cannot hold is skipped"
    );
}

#[test]
fn two_leds_with_a_foot_of_one_draw() {
    let config = "[aura]\nbrightness_max = 100\nleds_per_fan = 2\n\
                  [[light]]\nstyle = \"heat\"\nfoot_leds = 1\nattack_s = 0\n";
    let frame = paint(config, Some(125.0), 0.5, 4);
    assert_eq!(frame.len(), 2);
    assert!(frame.iter().all(|led| *led != BLACK));
}

#[test]
fn the_bar_steps_on_ticks_that_only_tween_and_a_long_gap_stays_finite() {
    let config = "\
[engine]\n\
tick_hz = 10\n\
target_hz = 2\n\
tween_s = 0\n\
[aura]\n\
brightness_max = 100\n\
fans = \"chain\"\n\
chain_len = 1\n\
leds_per_fan = 14\n\
[[light]]\n\
style = \"heat\"\n\
foot_leds = 3\n\
transit_s = 30\n\
conduction_s = 20000\n\
attack_s = 0\n\
smooth_s = 0\n";
    let mut renderer = Renderer::new(parse(config).expect("config"));
    let mut bar = HeatBar::new(
        14,
        HeatParams {
            transit_s: 30.0,
            foot_leds: 3,
            conduction_s: 20_000.0,
            attack_s: 0.0,
            smooth_s: 0.0,
        },
    )
    .expect("bar");
    let dt = 0.1_f32;
    let mut shown = Vec::new();
    for step in 0..6_u64 {
        bar.step(125.0, f64::from(dt));
        let frame = renderer.aura_frame(&activity(Some(125.0), step + 1), dt);
        let colour = Palette::Act.color(bar.leds()[13] as f32);
        shown.push(frame[13]);
        if step == 0 || step == 5 {
            assert_eq!(
                frame[13], colour,
                "step {step}: the drawn top LED is the bar's"
            );
        }
    }
    assert_eq!(
        shown[1], shown[0],
        "ticks between targets keep the last frame"
    );
    assert_ne!(
        shown[0], shown[5],
        "the in-between ticks still moved the bar"
    );

    let mut gap = Renderer::new(parse(config).expect("config"));
    let frame = gap.aura_frame(&activity(Some(125.0), 1), 90.0);
    assert_eq!(frame.len(), 14);
    assert_eq!(
        frame[13],
        hex(0xFF0000),
        "a 90 s gap is capped and simulated, not skipped"
    );
}
