//! Public contract for `present`: hysteresis, bands, and the round-3 churn replay.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::collector::{AiState, ModelInfo, Snapshot};
use kraken_lcd::config::{Config, StepMargin, UploadMode, Variant as ConfigVariant};
use kraken_lcd::history::History;
use kraken_lcd::policy::{Decision, Policy};
use kraken_lcd::present::{Ai, Band, Variant, View, present};
use llama_core::sample::TokenReading;

fn snapshot(t: Instant) -> Snapshot {
    Snapshot {
        t_mono: t,
        t_wall: SystemTime::UNIX_EPOCH,
        load: None,
        activity: None,
        cpu_pct: None,
        cpu_topk_pct: None,
        gpu_pct: None,
        mem_pct: None,
        coolant_c: None,
        cpu_c: None,
        gpu_c: None,
        ai: AiState::Idle,
        models: Vec::new(),
        tokens: None,
        errors: BTreeSet::new(),
    }
}

fn base_view() -> View {
    View::default()
}

fn ring_history(load: f32) -> (History, Instant) {
    let origin = Instant::now();
    let mut history = History::new(origin);
    for secs in 0..5 {
        history.add(origin + Duration::from_secs(secs), Some(load));
    }
    (history, origin + Duration::from_secs(4))
}

#[test]
fn ring_uses_the_load_history_when_activity_is_absent() {
    let (history, now) = ring_history(40.0);
    let mut snap = snapshot(now);
    snap.load = Some(10.0);
    snap.activity = None;
    let view = present(&snap, &history, None, &Config::default());
    assert_eq!(view.ring_pct, Some(40));
}

#[test]
fn ring_uses_activity_when_the_snapshot_has_it() {
    let (history, now) = ring_history(10.0);
    let mut snap = snapshot(now);
    snap.load = Some(10.0);
    snap.activity = Some(80.0);
    let view = present(&snap, &history, None, &Config::default());
    assert_eq!(view.ring_pct, Some(80));
}

fn presented_ring(raw: Option<f32>, shown: Option<u8>, cfg: &Config) -> Option<u8> {
    let origin = Instant::now();
    let mut history = History::new(origin);
    let now = origin + Duration::from_secs(4);
    if let Some(raw) = raw {
        for secs in 0..5 {
            history.add(origin + Duration::from_secs(secs), Some(raw));
        }
    }
    let mut previous = base_view();
    previous.ring_pct = shown;
    present(&snapshot(now), &history, Some(&previous), cfg).ring_pct
}

fn presented_ring_band(value: f32, previous: Option<Band>, cfg: &Config) -> Option<Band> {
    let (history, now) = ring_history(value);
    let prev = previous.map(|band| {
        let mut view = base_view();
        view.ring_band = Some(band);
        view
    });
    present(&snapshot(now), &history, prev.as_ref(), cfg).ring_band
}

/// Seven minutes of `load` land in b1 at coverage 0.5. Fewer minutes stay under it.
fn presented_block0(load: f32, minutes: u64, previous: Option<Band>, cfg: &Config) -> Band {
    let origin = Instant::now();
    let mut history = History::new(origin);
    for minute in 5..5 + minutes {
        history.add(origin + Duration::from_secs(minute * 60), Some(load));
    }
    let now = origin + Duration::from_secs(20 * 60);
    let prev = previous.map(|band| {
        let mut view = base_view();
        view.blocks[0] = band;
        view
    });
    present(&snapshot(now), &history, prev.as_ref(), cfg).blocks[0]
}

#[test]
fn hysteresis_quantises_from_config_and_holds_inside_the_window() {
    let cfg = Config::default();
    let ring = [
        ("72.4 quantises to 70", Some(72.4), None, Some(70)),
        ("72.5 quantises to 75", Some(72.5), None, Some(75)),
        (
            "74.5 stays on the high edge",
            Some(74.5),
            Some(70),
            Some(70),
        ),
        ("65.5 stays on the low edge", Some(65.5), Some(70), Some(70)),
        ("74.6 re-quantises up", Some(74.6), Some(70), Some(75)),
        ("65.4 re-quantises down", Some(65.4), Some(70), Some(65)),
        ("none clears the ring", None, Some(70), None),
    ];
    for (name, raw, shown, expect) in ring {
        assert_eq!(presented_ring(raw, shown, &cfg), expect, "ring {name}");
    }

    let percent = [
        ("41.6 quantises to 42", Some(41.6), None, Some(42)),
        ("41.6 stays against 41", Some(41.6), Some(41), Some(41)),
        (
            "42.5 stays on the high edge",
            Some(42.5),
            Some(41),
            Some(41),
        ),
        ("39.5 stays on the low edge", Some(39.5), Some(41), Some(41)),
        ("42.6 re-quantises up", Some(42.6), Some(41), Some(43)),
        ("39.4 re-quantises down", Some(39.4), Some(41), Some(39)),
        ("none clears", None, Some(41), None),
    ];
    for (name, raw, shown, expect) in percent {
        for field in [Field::Cpu, Field::Mem] {
            assert_eq!(
                presented_field(field, raw, shown, &cfg),
                expect,
                "{field:?} {name}"
            );
        }
    }

    let temp = [
        ("40.6 stays against 40", Some(40.6), Some(40), Some(40)),
        ("41.6 re-quantises", Some(41.6), Some(40), Some(42)),
        ("38.5 stays on the low edge", Some(38.5), Some(40), Some(40)),
        ("38.4 re-quantises down", Some(38.4), Some(40), Some(38)),
        ("-2.5 stays against -1", Some(-2.5), Some(-1), Some(-1)),
        ("-2.6 re-quantises", Some(-2.6), Some(-1), Some(-3)),
        ("none clears", None, Some(40), None),
    ];
    for (name, raw, shown, expect) in temp {
        for field in [Field::Coolant, Field::CpuTemp, Field::GpuTemp] {
            assert_eq!(
                presented_field(field, raw, shown, &cfg),
                expect,
                "{field:?} {name}"
            );
        }
    }
}

#[test]
fn each_field_reads_its_own_step_and_margin() {
    let mut cfg = Config::default();
    cfg.hysteresis.ring = StepMargin {
        step: 10,
        margin: 0,
    };
    assert_eq!(
        presented_ring(Some(14.0), None, &cfg),
        Some(10),
        "ring step 10 quantises 14 to 10, not the default step of 5"
    );

    cfg.hysteresis.ring = StepMargin { step: 5, margin: 0 };
    assert_eq!(
        presented_ring(Some(74.0), Some(70), &cfg),
        Some(75),
        "margin 0 releases 74"
    );
    assert_eq!(
        presented_ring(Some(74.0), Some(70), &Config::default()),
        Some(70),
        "default margin 2 holds 74"
    );

    let mut cfg = Config::default();
    cfg.hysteresis.percent = StepMargin {
        step: 10,
        margin: 0,
    };
    cfg.hysteresis.temp = StepMargin { step: 2, margin: 0 };
    assert_eq!(
        presented_field(Field::Cpu, Some(14.0), None, &cfg),
        Some(10)
    );
    assert_eq!(
        presented_field(Field::Mem, Some(14.0), None, &cfg),
        Some(10)
    );
    assert_eq!(
        presented_field(Field::Coolant, Some(14.0), None, &cfg),
        Some(14)
    );
    assert_eq!(
        presented_field(Field::CpuTemp, Some(14.0), None, &cfg),
        Some(14)
    );
    assert_eq!(
        presented_field(Field::GpuTemp, Some(14.0), None, &cfg),
        Some(14)
    );
}

#[test]
fn non_finite_percent_clears_and_negative_clamps_at_zero() {
    let cfg = Config::default();
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            presented_field(Field::Cpu, Some(raw), Some(40), &cfg),
            None,
            "{raw}"
        );
    }
    assert_eq!(presented_field(Field::Cpu, Some(-3.0), None, &cfg), Some(0));
    assert_eq!(
        presented_field(Field::Mem, Some(140.0), None, &cfg),
        Some(140)
    );
    assert_eq!(presented_ring(Some(103.0), None, &cfg), Some(105));
}

#[derive(Clone, Copy, Debug)]
enum Field {
    Cpu,
    Mem,
    Coolant,
    CpuTemp,
    GpuTemp,
}

fn presented_field(
    field: Field,
    raw: Option<f32>,
    shown: Option<i32>,
    cfg: &Config,
) -> Option<i32> {
    let now = Instant::now();
    let history = History::new(now);
    let mut current = snapshot(now);
    let mut previous = base_view();
    match field {
        Field::Cpu => {
            current.cpu_pct = raw;
            previous.cpu_pct = shown.map(|value| value as u8);
        }
        Field::Mem => {
            current.mem_pct = raw;
            previous.mem_pct = shown.map(|value| value as u8);
        }
        Field::Coolant => {
            current.coolant_c = raw;
            previous.coolant_c = shown.map(|value| value as i16);
        }
        Field::CpuTemp => {
            current.cpu_c = raw;
            previous.cpu_c = shown.map(|value| value as i16);
        }
        Field::GpuTemp => {
            current.gpu_c = raw;
            previous.gpu_c = shown.map(|value| value as i16);
        }
    }
    let view = present(&current, &history, Some(&previous), cfg);
    match field {
        Field::Cpu => view.cpu_pct.map(i32::from),
        Field::Mem => view.mem_pct.map(i32::from),
        Field::Coolant => view.coolant_c.map(i32::from),
        Field::CpuTemp => view.cpu_c.map(i32::from),
        Field::GpuTemp => view.gpu_c.map(i32::from),
    }
}

struct Edge {
    name: &'static str,
    previous: Option<Band>,
    value: f32,
    expect: Band,
}

#[test]
fn band_edges_move_up_at_enter_and_down_below_leave() {
    let cfg = Config::default();
    let edges = [
        Edge {
            name: "14 stays quiet",
            previous: Some(Band::Quiet),
            value: 14.0,
            expect: Band::Quiet,
        },
        Edge {
            name: "15 enters light",
            previous: Some(Band::Quiet),
            value: 15.0,
            expect: Band::Light,
        },
        Edge {
            name: "39 stays light",
            previous: Some(Band::Light),
            value: 39.0,
            expect: Band::Light,
        },
        Edge {
            name: "40 enters busy",
            previous: Some(Band::Light),
            value: 40.0,
            expect: Band::Busy,
        },
        Edge {
            name: "69 stays busy",
            previous: Some(Band::Busy),
            value: 69.0,
            expect: Band::Busy,
        },
        Edge {
            name: "70 enters flat out",
            previous: Some(Band::Busy),
            value: 70.0,
            expect: Band::FlatOut,
        },
        Edge {
            name: "10 stays light",
            previous: Some(Band::Light),
            value: 10.0,
            expect: Band::Light,
        },
        Edge {
            name: "9.9 leaves light",
            previous: Some(Band::Light),
            value: 9.9,
            expect: Band::Quiet,
        },
        Edge {
            name: "35 stays busy",
            previous: Some(Band::Busy),
            value: 35.0,
            expect: Band::Busy,
        },
        Edge {
            name: "34.9 leaves busy",
            previous: Some(Band::Busy),
            value: 34.9,
            expect: Band::Light,
        },
        Edge {
            name: "65 stays flat out",
            previous: Some(Band::FlatOut),
            value: 65.0,
            expect: Band::FlatOut,
        },
        Edge {
            name: "64.9 leaves flat out",
            previous: Some(Band::FlatOut),
            value: 64.9,
            expect: Band::Busy,
        },
        Edge {
            name: "no previous 14 is quiet",
            previous: None,
            value: 14.0,
            expect: Band::Quiet,
        },
        Edge {
            name: "no previous 70 is flat out",
            previous: None,
            value: 70.0,
            expect: Band::FlatOut,
        },
    ];
    for edge in edges {
        assert_eq!(
            presented_ring_band(edge.value, edge.previous, &cfg),
            Some(edge.expect),
            "ring {}",
            edge.name
        );
        assert_eq!(
            presented_block0(edge.value, 7, edge.previous, &cfg),
            edge.expect,
            "block {}",
            edge.name
        );
    }
}

#[test]
fn band_crosses_several_thresholds_in_one_step() {
    let cfg = Config::default();
    assert_eq!(
        presented_ring_band(80.0, Some(Band::Quiet), &cfg),
        Some(Band::FlatOut)
    );
    assert_eq!(
        presented_ring_band(9.9, Some(Band::FlatOut), &cfg),
        Some(Band::Quiet)
    );
    assert_eq!(
        presented_block0(9.9, 7, Some(Band::Busy), &cfg),
        Band::Quiet
    );
}

#[test]
fn band_enter_and_margin_come_from_config() {
    let mut cfg = Config::default();
    cfg.bands.enter = [20, 40, 70];
    assert_eq!(
        presented_ring_band(15.0, None, &cfg),
        Some(Band::Quiet),
        "15 does not enter Light when Light's enter is 20"
    );

    let mut cfg = Config::default();
    cfg.bands.margin = 4;
    assert_eq!(
        presented_ring_band(10.5, Some(Band::Light), &cfg),
        Some(Band::Quiet),
        "leave is enter 15 minus margin 4, so 10.5 drops out of Light"
    );
    assert_eq!(
        presented_ring_band(10.5, Some(Band::Light), &Config::default()),
        Some(Band::Light),
        "default leave is 10, so 10.5 stays Light"
    );
}

#[test]
fn filling_under_half_coverage_and_quiet_at_half() {
    let cfg = Config::default();
    assert_eq!(
        presented_block0(90.0, 6, Some(Band::FlatOut), &cfg),
        Band::Filling,
        "6/14 is under 0.5 even when the mean is hot and the previous band is FlatOut"
    );
    assert_eq!(
        presented_block0(0.0, 7, Some(Band::Filling), &cfg),
        Band::Quiet,
        "7/14 is exactly 0.5, so a quiet mean leaves Filling"
    );
}

#[test]
fn recovered_coverage_classifies_fresh_while_a_held_band_keeps_hysteresis() {
    let cfg = Config::default();
    assert_eq!(
        presented_block0(69.0, 7, Some(Band::Filling), &cfg),
        Band::Busy
    );
    assert_eq!(
        presented_block0(69.0, 7, Some(Band::FlatOut), &cfg),
        Band::FlatOut
    );
}

#[test]
fn fill_min_coverage_comes_from_config() {
    let mut cfg = Config::default();
    cfg.bands.fill_min_coverage = 0.25;
    assert_eq!(
        presented_block0(0.0, 6, Some(Band::Filling), &cfg),
        Band::Quiet,
        "6/14 is above a threshold of 0.25"
    );
}

#[test]
fn blocks_are_b1_b2_b3() {
    let mut cfg = Config::default();
    cfg.bands.fill_min_coverage = 0.0;
    let origin = Instant::now();
    let mut history = History::new(origin);
    history.add(origin, Some(90.0));
    history.add(origin + Duration::from_secs(1320 * 60), Some(50.0));
    history.add(origin + Duration::from_secs(1425 * 60), Some(0.0));
    let now = origin + Duration::from_secs(1440 * 60);
    let view = present(&snapshot(now), &history, None, &cfg);
    assert_eq!(view.blocks, [Band::Quiet, Band::Busy, Band::FlatOut]);
}

#[test]
fn ring_uses_the_raw_mean_and_is_never_filling() {
    let cfg = Config::default();
    let (history, now) = ring_history(69.0);
    let view = present(&snapshot(now), &history, None, &cfg);
    assert_eq!(view.ring_pct, Some(70));
    assert_eq!(view.ring_band, Some(Band::Busy));

    let mut previous = base_view();
    previous.ring_pct = Some(70);
    previous.ring_band = Some(Band::Filling);
    let (history, now) = ring_history(70.0);
    let view = present(&snapshot(now), &history, Some(&previous), &cfg);
    assert_eq!(view.ring_band, Some(Band::FlatOut));

    let origin = Instant::now();
    let mut history = History::new(origin);
    for secs in 0..4 {
        history.add(origin + Duration::from_secs(secs), Some(80.0));
    }
    let now = origin + Duration::from_secs(3);
    let mut previous = base_view();
    previous.ring_pct = Some(70);
    previous.ring_band = Some(Band::FlatOut);
    let view = present(&snapshot(now), &history, Some(&previous), &cfg);
    assert_eq!(view.ring_pct, None);
    assert_eq!(view.ring_band, None);
}

#[test]
fn models_are_the_snapshot_names_and_ai_maps_from_state() {
    let cfg = Config::default();
    let now = Instant::now();
    let history = History::new(now);
    for (state, ai) in [
        (AiState::Down, Ai::Down),
        (AiState::Idle, Ai::Idle),
        (AiState::Loaded, Ai::Loaded),
    ] {
        let mut current = snapshot(now);
        current.ai = state;
        let view = present(&current, &history, None, &cfg);
        assert_eq!(view.ai, ai, "{state:?}");
        assert_eq!(view.model_count, 0);
        assert!(view.models.is_empty());
    }

    let mut current = snapshot(now);
    current.ai = AiState::Loaded;
    current.models = vec![
        ModelInfo {
            backend: None,
            name: "Qwen 35B".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        },
        ModelInfo {
            backend: None,
            name: "Other…".to_owned(),
            state: "starting".to_owned(),
            full_name: None,
            detail: None,
        },
    ];
    let view = present(&current, &history, None, &cfg);
    assert_eq!(view.ai, Ai::Loaded);
    assert_eq!(view.models, ["Qwen 35B", "Other…"]);
    assert_eq!(view.model_count, 2);
}

#[test]
fn model_count_saturates_at_u8_max() {
    let cfg = Config::default();
    let now = Instant::now();
    let history = History::new(now);
    let mut current = snapshot(now);
    current.ai = AiState::Loaded;
    current.models = (0..256)
        .map(|index| ModelInfo {
            backend: None,
            name: format!("m{index}"),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        })
        .collect();
    let view = present(&current, &history, None, &cfg);
    assert_eq!(view.models.len(), 256);
    assert_eq!(view.model_count, u8::MAX);
}

#[test]
fn view_roundtrips_the_render_once_fixture_shape() {
    let json = r#"{
        "ring_pct": 70,
        "ring_band": "FlatOut",
        "blocks": ["Quiet", "Light", "Busy"],
        "coolant_c": 40,
        "cpu_c": null,
        "gpu_c": 81,
        "cpu_pct": 12,
        "mem_pct": null,
        "ai": "Loaded",
        "models": ["Qwen 35B", "Other"],
        "model_count": 2
    }"#;
    let view: View = serde_json::from_str(json).expect("fixture json");
    assert_eq!(view.ring_pct, Some(70));
    assert_eq!(view.ring_band, Some(Band::FlatOut));
    assert_eq!(view.blocks, [Band::Quiet, Band::Light, Band::Busy]);
    assert_eq!(view.coolant_c, Some(40));
    assert_eq!(view.cpu_c, None);
    assert_eq!(view.gpu_c, Some(81));
    assert_eq!(view.cpu_pct, Some(12));
    assert_eq!(view.mem_pct, None);
    assert_eq!(view.ai, Ai::Loaded);
    assert_eq!(view.models, ["Qwen 35B", "Other"]);
    assert_eq!(view.model_count, 2);
    assert_eq!(view.dial, [None; 24]);
    assert!(view.tokens.is_empty());
    assert_eq!(
        view.scale.len(),
        5,
        "an old fixture gets the default time scale"
    );
    assert_eq!(view.variant, Variant::A1);
    let again: View =
        serde_json::from_str(&serde_json::to_string(&view).expect("encode")).expect("decode");
    assert_eq!(again, view);

    let extra = r#"{
        "ring_pct": null,
        "ring_band": null,
        "blocks": ["Filling", "Filling", "Filling"],
        "coolant_c": null,
        "cpu_c": null,
        "gpu_c": null,
        "cpu_pct": null,
        "mem_pct": null,
        "ai": "Down",
        "models": [],
        "model_count": 0,
        "extra": 1
    }"#;
    assert!(
        serde_json::from_str::<View>(extra).is_err(),
        "unknown fixture fields are rejected"
    );
}

#[test]
fn equal_views_hash_together() {
    let view = base_view();
    let mut set = HashSet::new();
    set.insert(view.clone());
    assert!(set.contains(&view));
    let mut other = view.clone();
    other.ring_pct = Some(5);
    assert_ne!(view, other);
    assert!(!set.contains(&other));
}

#[test]
fn view_fixture_keeps_dial_means_tokens_and_variant() {
    let mut bars = ["null"; 24];
    bars[0] = "118";
    bars[23] = "0";
    let json = format!(
        r#"{{
        "ring_pct": 125,
        "ring_band": null,
        "blocks": ["Filling", "Filling", "Filling"],
        "coolant_c": null,
        "cpu_c": null,
        "gpu_c": null,
        "cpu_pct": null,
        "mem_pct": null,
        "ai": "NoData",
        "models": [],
        "model_count": 0,
        "dial": [{bars}],
        "tokens": [4200, 0, 12],
        "variant": "a3"
    }}"#,
        bars = bars.join(", ")
    );
    let view: View = serde_json::from_str(&json).expect("fixture json");
    assert_eq!(view.variant, Variant::A3);
    assert_eq!(view.ring_pct, Some(125));
    assert_eq!(view.dial[0], Some(118));
    assert_eq!(view.dial[23], Some(0));
    assert!(view.dial[1..23].iter().all(Option::is_none));
    assert_eq!(view.tokens, [4200, 0, 12]);
    let again: View =
        serde_json::from_str(&serde_json::to_string(&view).expect("encode")).expect("decode");
    assert_eq!(again, view);
    assert_eq!(again.variant, Variant::A3);
    let old = json.replace("118", "\"L6\"");
    assert!(
        serde_json::from_str::<View>(&old).is_err(),
        "token-dial levels are not activity means"
    );
}

fn token(seq: u64, t_mono_ns: u64, decoded_total: Option<u64>) -> TokenReading {
    TokenReading {
        run_id: 7,
        seq,
        t_mono_ns,
        decoded_total,
    }
}

fn stream() -> Config {
    let mut cfg = Config::default();
    cfg.upload.mode = UploadMode::Stream;
    cfg
}

/// Present `frames` fresh snapshots 100 ms apart. Activity comes from
/// `activity(frame)`; the counter grows by `tok_per_frame`.
struct Drive {
    origin: Instant,
    start_ns: u64,
    frame: u64,
    total: u64,
    view: Option<View>,
}

impl Drive {
    fn new(start_ns: u64) -> Self {
        Self {
            origin: Instant::now(),
            start_ns,
            frame: 0,
            total: 1_000,
            view: None,
        }
    }

    fn run(
        &mut self,
        cfg: &Config,
        frames: u64,
        tok_per_frame: u64,
        activity: impl Fn(u64) -> Option<f32>,
    ) -> &View {
        let history = History::new(self.origin);
        for _ in 0..frames {
            self.frame += 1;
            self.total += tok_per_frame;
            let mut snap = snapshot(self.origin + Duration::from_millis(100 * self.frame));
            snap.ai = AiState::Loaded;
            snap.activity = activity(self.frame);
            snap.tokens = Some(token(
                self.frame,
                self.start_ns + self.frame * 100_000_000,
                Some(self.total),
            ));
            let next = present(&snap, &history, self.view.as_ref(), cfg);
            self.view = Some(next);
        }
        self.view.as_ref().expect("at least one frame")
    }
}

#[test]
fn activity_over_100_reaches_the_ring_and_the_bars() {
    let mut drive = Drive::new(50_000_000_000_000);
    let view = drive.run(&stream(), 40, 0, |_| Some(118.0));
    assert_eq!(
        view.ring_pct,
        Some(118),
        "stream mode settles on the raw value"
    );
    assert_eq!(
        view.dial[0],
        Some(118),
        "the newest bar keeps the overshoot"
    );
    let pinned = drive.run(&stream(), 40, 0, |_| Some(400.0)).clone();
    assert_eq!(pinned.ring_pct, Some(125), "values above 125 pin at 125");
    assert_eq!(pinned.dial[0], Some(125));

    let mut change = Drive::new(51_000_000_000_000);
    let view = change.run(&Config::default(), 5, 0, |_| Some(118.0));
    assert_eq!(
        view.ring_pct,
        Some(120),
        "change mode keeps the 5-point hold"
    );
}

#[test]
fn stream_mode_slides_the_ring_toward_the_value() {
    let mut drive = Drive::new(52_000_000_000_000);
    drive.run(&stream(), 30, 0, |_| Some(20.0));
    let first = drive.run(&stream(), 1, 0, |_| Some(100.0)).ring_pct;
    // One frame: 20 + (100 − 20) × 0.35 = 48.
    assert_eq!(first, Some(48));
    let later = drive.run(&stream(), 3, 0, |_| Some(100.0)).ring_pct;
    assert!(later.is_some_and(|pct| pct > 48 && pct < 100), "{later:?}");
    let settled = drive.run(&stream(), 40, 0, |_| Some(100.0)).ring_pct;
    assert_eq!(settled, Some(100));
}

#[test]
fn tokens_feed_the_24h_chart() {
    let mut drive = Drive::new(53_000_000_000_000);
    // 9 tokens per 100 ms is 90 tok/s.
    let view = drive.run(&stream(), 600, 9, |_| Some(70.0));
    assert!(!view.tokens.is_empty());
    for (i, centi) in view.tokens.iter().enumerate() {
        assert!(
            centi.abs_diff(9_000) <= 1,
            "point {i} is {centi} centi-tok/s"
        );
    }
}

#[test]
fn current_generation_rate_reaches_the_view() {
    // No counter yet: no rate, so the face draws "—".
    let (history, now) = ring_history(10.0);
    let fresh = present(&snapshot(now), &history, None, &Config::default());
    assert_eq!(fresh.gen_tps_tenths, None);

    // 9 tokens per 100 ms is 90 tok/s. Change mode climbs in held steps
    // and changes the frame key only when the text moves.
    let mut drive = Drive::new(59_000_000_000_000);
    let first = drive.run(&Config::default(), 1, 9, |_| Some(70.0)).clone();
    assert_eq!(first.gen_tps_tenths, Some(0), "the first reading is a gap");
    let mut changes = 0;
    let mut last = first.gen_tps_tenths;
    for _ in 0..600 {
        let view = drive.run(&Config::default(), 1, 9, |_| Some(70.0));
        if view.gen_tps_tenths != last {
            changes += 1;
            last = view.gen_tps_tenths;
        }
    }
    assert_eq!(last, Some(900), "settles on 90 tok/s");
    assert!(changes < 120, "held steps, not one per frame: {changes}");
    // A steady 90 tok/s holds.
    let held = drive.run(&Config::default(), 100, 9, |_| Some(70.0));
    assert_eq!(held.gen_tps_tenths, Some(900));

    // Stream mode shows the average, rounded, every frame.
    let mut stream_drive = Drive::new(60_000_000_000_000);
    stream_drive.run(&stream(), 1, 9, |_| Some(70.0));
    let early = stream_drive.run(&stream(), 10, 9, |_| Some(70.0)).clone();
    // 1 s at 90 tok/s with τ = 5 s: 90 × (1 − e^−0.2) ≈ 16.3.
    assert_eq!(early.gen_tps_tenths, Some(160));
    let next = stream_drive.run(&stream(), 1, 9, |_| Some(70.0));
    assert_eq!(next.gen_tps_tenths, Some(180), "no hold in stream mode");

    // Prefill or idle: the counter stops and the rate decays toward zero:
    // 90 × e^−2 after 10 s, and "0" within a minute.
    let prefill = drive.run(&Config::default(), 100, 0, |_| Some(70.0));
    assert_eq!(prefill.gen_tps_tenths, Some(120));
    let quiet = drive.run(&Config::default(), 500, 0, |_| Some(70.0));
    assert_eq!(quiet.gen_tps_tenths, Some(0));

    // No data clears it.
    let previous = drive.view.clone();
    let mut snap = snapshot(drive.origin + Duration::from_secs(3_600));
    snap.ai = AiState::NoData;
    let gone = present(&snap, &history, previous.as_ref(), &Config::default());
    assert_eq!(gone.gen_tps_tenths, None);
}

#[test]
fn stream_mode_carries_the_fill_sweeps_on_the_slow_tiers_only() {
    let mut drive = Drive::new(54_000_000_000_000 + 90_000_000_000);
    let view = drive.run(&stream(), 3, 0, |_| Some(50.0)).clone();
    let fills: Vec<Option<u16>> = view.scale.iter().map(|tier| tier.fill).collect();
    assert_eq!(
        fills[..3],
        [None, None, None],
        "0.5 s, 5 s and 15 s tiers have no sweep"
    );
    assert!(fills[3].is_some() && fills[4].is_some(), "{fills:?}");
    let mut change = Drive::new(54_000_000_000_000 + 90_000_000_000);
    let view = change.run(&Config::default(), 3, 0, |_| Some(50.0));
    assert!(
        view.scale.iter().all(|tier| tier.fill.is_none()),
        "change mode draws no sweep, so the frame key does not move with it"
    );
}

#[test]
fn idle_bars_stay_zero_across_ticks_so_nothing_uploads() {
    let cfg = Config::default();
    let mut drive = Drive::new(55_000_000_000_000);
    // 31 min idle at ~0.8 % activity and an unchanged counter.
    let first = drive
        .run(&cfg, 31 * 60 * 10 + 5, 0, |f| {
            Some(0.5 + (f % 3) as f32 * 0.2)
        })
        .clone();
    assert!(
        first.dial.iter().all(|bar| *bar == Some(0)),
        "idle dial {:?}",
        first.dial
    );
    let second = drive.run(&cfg, 1, 0, |_| Some(0.9)).clone();
    assert_eq!(first, second, "an idle dial does not change the frame");

    let origin = Instant::now();
    let mut policy = Policy::new(Duration::from_secs(60));
    assert_eq!(policy.decide(&first, origin), Decision::Upload);
    policy.mark_uploaded(&first, origin);
    assert_eq!(
        policy.decide(&second, origin + Duration::from_millis(100)),
        Decision::Nothing,
        "an idle box uploads nothing while the dial stays at zero"
    );
}

#[test]
fn not_fresh_keeps_older_bars_and_puts_out_the_smoke() {
    let cfg = stream();
    let mut drive = Drive::new(56_000_000_000_000);
    // Two minutes pinned, then a stale snapshot 10 s later.
    let hot = drive.run(&cfg, 1_200, 0, |_| Some(125.0)).clone();
    assert!(hot.dial_state.anim.particles.alive() > 0, "the peg smokes");
    let history = History::new(drive.origin);
    let mut stale = snapshot(drive.origin + Duration::from_secs(130));
    stale.ai = AiState::NoData;
    let view = present(&stale, &history, Some(&hot), &cfg);
    assert_eq!(view.ai, Ai::NoData);
    assert_eq!(view.ring_pct, None, "no activity, no ring");
    assert_eq!(view.dial_state.anim.particles.alive(), 0, "smoke is off");
    assert_eq!(view.dial[0], None, "the newest bar is the stall");
    assert_eq!(view.dial[15], Some(125), "the 1 min bar of 60..70 s stays");
}

#[test]
fn writer_variant_reaches_the_view() {
    let mut cfg = Config::default();
    cfg.display.variant = ConfigVariant::A3;
    let mut drive = Drive::new(57_000_000_000_000);
    assert_eq!(drive.run(&cfg, 2, 0, |_| Some(10.0)).variant, Variant::A3);
}

#[test]
fn writer_memory_is_not_the_upload_key_but_a_bar_is() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut same_frame = View::default();
    let mut drive = Drive::new(58_000_000_000_000);
    let other_memory = View {
        dial_state: drive
            .run(&stream(), 50, 3, |_| Some(125.0))
            .dial_state
            .clone(),
        ..View::default()
    };
    assert!(other_memory.dial_state.anim.frame > 0);
    assert_eq!(same_frame, other_memory);
    let hash = |view: &View| {
        let mut hasher = DefaultHasher::new();
        view.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(hash(&same_frame), hash(&other_memory));

    same_frame.dial[0] = Some(90);
    assert_ne!(same_frame, other_memory);
    assert_ne!(hash(&same_frame), hash(&other_memory));
}

#[test]
fn same_inputs_return_an_equal_view() {
    let cfg = Config::default();
    let (history, now) = ring_history(50.0);
    let current = snapshot(now);
    assert_eq!(
        present(&current, &history, None, &cfg),
        present(&current, &history, None, &cfg)
    );
    let (other_history, other_now) = ring_history(10.0);
    assert_ne!(
        present(&current, &history, None, &cfg).ring_pct,
        present(&snapshot(other_now), &other_history, None, &cfg).ring_pct
    );
}

#[test]
#[should_panic(expected = "step > 0")]
fn quantise_rejects_a_zero_step() {
    let mut cfg = Config::default();
    cfg.hysteresis.temp = StepMargin { step: 0, margin: 1 };
    let now = Instant::now();
    let history = History::new(now);
    let mut current = snapshot(now);
    current.cpu_c = Some(40.0);
    let _ = present(&current, &history, None, &cfg);
}

struct Sample {
    t_s: u64,
    gpu_pct: f32,
    cpus: Vec<f32>,
    coolant_c: f32,
    cpu_c: f32,
    gpu_c: f32,
    mem_pct: f32,
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn read_series(name: &str) -> Vec<Sample> {
    let path = fixtures().join("churn").join(name);
    let text =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut lines = text.lines().filter(|line| {
        let line = line.trim();
        !line.is_empty() && !line.starts_with('#')
    });
    let header: Vec<&str> = lines.next().expect("header").split(',').collect();
    assert_eq!(header.first().copied(), Some("t_s"));
    assert_eq!(header.get(1).copied(), Some("gpu_pct"));
    let tail = &header[header.len() - 4..];
    assert_eq!(tail, ["coolant_c", "cpu_c", "gpu_c", "mem_pct"]);
    let cpu_columns = header.len() - 6;
    assert_eq!(cpu_columns, 32, "32 logical CPUs");
    lines
        .map(|line| {
            let cells: Vec<&str> = line.split(',').collect();
            assert_eq!(cells.len(), header.len(), "{line}");
            let num = |index: usize| {
                cells[index]
                    .parse::<f32>()
                    .unwrap_or_else(|err| panic!("column {index} of {line}: {err}"))
            };
            let cpus: Vec<f32> = (2..2 + cpu_columns).map(num).collect();
            Sample {
                t_s: cells[0].parse().expect("t_s"),
                gpu_pct: num(1),
                cpus,
                coolant_c: num(header.len() - 4),
                cpu_c: num(header.len() - 3),
                gpu_c: num(header.len() - 2),
                mem_pct: num(header.len() - 1),
            }
        })
        .collect()
}

fn instantaneous_band(load: f32) -> Band {
    if load >= 70.0 {
        Band::FlatOut
    } else if load >= 40.0 {
        Band::Busy
    } else if load >= 15.0 {
        Band::Light
    } else {
        Band::Quiet
    }
}

fn changed_fields(before: &View, after: &View) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if before.ring_pct != after.ring_pct {
        fields.push("ring_pct");
    }
    if before.ring_band != after.ring_band {
        fields.push("ring_band");
    }
    if before.blocks != after.blocks {
        fields.push("blocks");
    }
    if before.coolant_c != after.coolant_c {
        fields.push("coolant_c");
    }
    if before.cpu_c != after.cpu_c {
        fields.push("cpu_c");
    }
    if before.gpu_c != after.gpu_c {
        fields.push("gpu_c");
    }
    if before.cpu_pct != after.cpu_pct {
        fields.push("cpu_pct");
    }
    if before.mem_pct != after.mem_pct {
        fields.push("mem_pct");
    }
    if before.ai != after.ai {
        fields.push("ai");
    }
    if before.models != after.models {
        fields.push("models");
    }
    if before.model_count != after.model_count {
        fields.push("model_count");
    }
    if before.dial != after.dial {
        fields.push("dial");
    }
    if before.variant != after.variant {
        fields.push("variant");
    }
    fields
}

fn core_parts(row: &Sample, k: usize) -> (f32, f32, f32) {
    let mut cpus = row.cpus.clone();
    cpus.sort_by(|left, right| right.total_cmp(left));
    let top = cpus.iter().take(k).sum::<f32>() / k as f32;
    let plain = row.cpus.iter().sum::<f32>() / row.cpus.len() as f32;
    let load = top.max(row.gpu_pct);
    (top, plain, load)
}

fn replay(rows: &[Sample]) -> Vec<(u64, View)> {
    let cfg = Config::default();
    let k = 8;
    let origin = Instant::now();
    let mut history = History::new(origin);
    let mut previous = None;
    let mut views = Vec::new();
    for row in rows {
        let (top, plain, load) = core_parts(row, k);
        let t = origin + Duration::from_secs(row.t_s);
        history.add(t, Some(load));
        let mut current = snapshot(t);
        current.load = Some(load);
        current.cpu_topk_pct = Some(top);
        current.cpu_pct = Some(plain);
        current.gpu_pct = Some(row.gpu_pct);
        current.mem_pct = Some(row.mem_pct);
        current.coolant_c = Some(row.coolant_c);
        current.cpu_c = Some(row.cpu_c);
        current.gpu_c = Some(row.gpu_c);
        let view = present(&current, &history, previous.as_ref(), &cfg);
        views.push((row.t_s, view.clone()));
        previous = Some(view);
    }
    views
}

fn outside_window(raw: f32, shown: i32, pair: StepMargin) -> bool {
    let slack = f64::from(pair.step) / 2.0 + f64::from(pair.margin);
    let raw = f64::from(raw);
    let shown = f64::from(shown);
    raw < shown - slack || raw > shown + slack
}

fn step_round(raw: f32, step: u32) -> i32 {
    let step = f64::from(step);
    let rounded = ((f64::from(raw) / step).round() * step).round();
    rounded as i32
}

fn shown_number(view: &View, field: &str) -> i32 {
    match field {
        "coolant_c" => i32::from(view.coolant_c.expect("coolant")),
        "cpu_c" => i32::from(view.cpu_c.expect("cpu temp")),
        "gpu_c" => i32::from(view.gpu_c.expect("gpu temp")),
        "cpu_pct" => i32::from(view.cpu_pct.expect("cpu percent")),
        "mem_pct" => i32::from(view.mem_pct.expect("memory percent")),
        other => panic!("not a numeric field: {other}"),
    }
}

fn raw_number(row: &Sample, field: &str, plain: f32) -> f32 {
    match field {
        "coolant_c" => row.coolant_c,
        "cpu_c" => row.cpu_c,
        "gpu_c" => row.gpu_c,
        "cpu_pct" => plain,
        "mem_pct" => row.mem_pct,
        other => panic!("not a numeric field: {other}"),
    }
}

fn pair_for(field: &str, cfg: &Config) -> StepMargin {
    match field {
        "cpu_pct" | "mem_pct" => cfg.hysteresis.percent,
        "coolant_c" | "cpu_c" | "gpu_c" => cfg.hysteresis.temp,
        other => panic!("not a 1/1 field: {other}"),
    }
}

#[test]
fn churn_ring_changes_at_most_once_after_it_becomes_some() {
    let rows = read_series("round3.csv");
    assert_eq!(rows.len(), 30);
    assert_eq!(rows[0].t_s, 2);
    assert_eq!(rows[29].t_s, 60);
    for pair in rows.windows(2) {
        assert_eq!(pair[1].t_s, pair[0].t_s + 2);
        assert!(pair[0].cpus.iter().all(|cpu| *cpu == pair[0].cpus[0]));
    }
    let swing = rows.iter().position(|row| row.t_s == 56).expect("t=56");
    assert_eq!(rows[swing].cpus[0], 41.0);
    assert_eq!(rows[swing + 2].t_s, 60);
    assert_eq!(rows[swing + 2].cpus[0], 3.0);

    let k = 8;
    let mut naive_changes = 0u32;
    let mut previous_band = None;
    for row in &rows {
        let mut cpus = row.cpus.clone();
        cpus.sort_by(|left, right| right.total_cmp(left));
        let top = cpus.iter().take(k).sum::<f32>() / k as f32;
        let load = top.max(row.gpu_pct);
        let band = instantaneous_band(load);
        if previous_band.is_some_and(|previous| previous != band) {
            naive_changes += 1;
        }
        previous_band = Some(band);
    }
    assert_eq!(
        naive_changes, 13,
        "instantaneous load still crosses the 70 edge; the fixture must keep that noise"
    );

    let views = replay(&rows);

    let start = views
        .iter()
        .position(|(_, view)| view.ring_pct.is_some())
        .expect("ring_pct becomes Some once five samples are in the window");
    assert_eq!(views[start].0, 10);
    // Blocks stay Filling for the whole 60 s, so this replay does not exercise
    // block hysteresis. The band-edge tests do.
    assert!(
        views
            .iter()
            .all(|(_, view)| view.blocks == [Band::Filling; 3])
    );
    for (t, view) in &views[start..] {
        assert_eq!(view.ring_pct, Some(70), "t={t}");
        assert_eq!(view.ring_band, Some(Band::FlatOut), "t={t}");
        assert_eq!(view.mem_pct, Some(11), "t={t}");
        assert_eq!(view.coolant_c, Some(40), "t={t}");
    }
    let at = |t: u64| -> &View {
        &views
            .iter()
            .find(|(stamp, _)| *stamp == t)
            .unwrap_or_else(|| panic!("missing t={t}"))
            .1
    };
    assert_eq!(at(8).gpu_c, Some(80));
    assert_eq!(at(10).cpu_pct, Some(41));
    assert_eq!(at(12).cpu_pct, Some(28));
    assert_eq!(at(14).cpu_pct, Some(4));
    assert_eq!(at(14).cpu_c, Some(77));
    assert_eq!(at(14).gpu_c, Some(75));
    assert_eq!(at(18).cpu_pct, Some(41));
    assert_eq!(at(60).cpu_pct, Some(3));
    assert_eq!(at(60).cpu_c, Some(77));
    assert_eq!(at(60).gpu_c, Some(75));

    // Ring percent, ring band, and the blocks are the round-3 churn. They stay
    // put after the ring appears (0, which is ≤ 1). CPU% and the CPU/GPU
    // temperatures leave the 1/1 window on this capture, so the whole View
    // still updates; holding those fields would contradict the hysteresis rule.
    let changes: Vec<(u64, Vec<&'static str>)> = views[start..]
        .windows(2)
        .filter(|pair| pair[0].1 != pair[1].1)
        .map(|pair| (pair[1].0, changed_fields(&pair[0].1, &pair[1].1)))
        .collect();
    let ring_changes = changes
        .iter()
        .filter(|(_, fields)| {
            fields
                .iter()
                .any(|field| matches!(*field, "ring_pct" | "ring_band" | "blocks"))
        })
        .count();
    assert!(
        ring_changes <= 1,
        "ring, ring band, or blocks changed {ring_changes} times after ring_pct became Some: {changes:?}"
    );
    assert_eq!(
        changes,
        vec![
            (12, vec!["gpu_c", "cpu_pct"]),
            (14, vec!["cpu_c", "gpu_c", "cpu_pct"]),
            (16, vec!["gpu_c", "cpu_pct"]),
            (18, vec!["cpu_pct"]),
            (24, vec!["cpu_c"]),
            (30, vec!["gpu_c"]),
            (58, vec!["gpu_c", "cpu_pct"]),
            (60, vec!["cpu_c", "gpu_c", "cpu_pct"]),
        ]
    );
    println!(
        "busy replay: ring/band/block changes={ring_changes}, whole-view changes={}",
        changes.len()
    );
}

#[test]
fn churn_idle_ring_and_blocks_do_not_change() {
    let rows = read_series("idle.csv");
    assert_eq!(rows.len(), 60);
    assert_eq!(rows[0].t_s, 2);
    assert_eq!(rows[59].t_s, 120);
    for pair in rows.windows(2) {
        assert_eq!(pair[1].t_s, pair[0].t_s + 2);
    }
    let cfg = Config::default();
    let k = 8;
    for row in &rows {
        let mut cpus = row.cpus.clone();
        cpus.sort_by(|left, right| right.total_cmp(left));
        assert!(
            cpus[..k].iter().all(|cpu| *cpu == cpus[0]),
            "t={} top-8 cores are one value",
            row.t_s
        );
        assert!(
            cpus[k..].iter().all(|cpu| *cpu == cpus[k]),
            "t={} the other cores are the remainder",
            row.t_s
        );
        assert!(cpus[k] <= cpus[0], "t={} remainder is not busier", row.t_s);
    }
    let (top, plain, _) = core_parts(&rows[0], k);
    assert_eq!(rows[0].gpu_pct, 0.0);
    assert_eq!(rows[0].cpu_c, 42.0);
    assert_eq!(rows[0].gpu_c, 56.0);
    assert_eq!(rows[0].coolant_c, 37.6);
    assert_eq!(rows[0].mem_pct, 8.7);
    assert!((f64::from(top) - 0.3).abs() < 1e-5, "top-8 mean {top}");
    assert!((f64::from(plain) - 0.1).abs() < 1e-5, "plain mean {plain}");
    let (top, plain, _) = core_parts(&rows[59], k);
    assert_eq!(rows[59].cpu_c, 38.4);
    assert_eq!(rows[59].gpu_c, 52.0);
    assert_eq!(rows[59].coolant_c, 35.5);
    assert_eq!(rows[59].mem_pct, 8.6);
    assert!((f64::from(top) - 0.7).abs() < 1e-5, "top-8 mean {top}");
    assert!((f64::from(plain) - 0.2).abs() < 1e-5, "plain mean {plain}");

    let views = replay(&rows);
    let start = views
        .iter()
        .position(|(_, view)| view.ring_pct.is_some())
        .expect("ring_pct becomes Some");
    assert_eq!(views[start].0, 10);
    for (t, view) in &views[start..] {
        assert_eq!(view.ring_pct, Some(0), "t={t}");
        assert_eq!(view.ring_band, Some(Band::Quiet), "t={t}");
        assert_eq!(view.blocks, [Band::Filling; 3], "t={t}");
    }

    let changes: Vec<(u64, Vec<&'static str>)> = views[start..]
        .windows(2)
        .filter(|pair| pair[0].1 != pair[1].1)
        .map(|pair| (pair[1].0, changed_fields(&pair[0].1, &pair[1].1)))
        .collect();
    let ring_changes = changes
        .iter()
        .filter(|(_, fields)| {
            fields
                .iter()
                .any(|field| matches!(*field, "ring_pct" | "ring_band" | "blocks"))
        })
        .count();
    assert_eq!(
        ring_changes, 0,
        "ring, ring band, or blocks changed: {changes:?}"
    );

    let mut numeric = Vec::new();
    for pair in views.windows(2) {
        let mut fields = Vec::new();
        for field in changed_fields(&pair[0].1, &pair[1].1) {
            if !matches!(
                field,
                "coolant_c" | "cpu_c" | "gpu_c" | "cpu_pct" | "mem_pct"
            ) {
                continue;
            }
            let row = rows
                .iter()
                .find(|row| row.t_s == pair[1].0)
                .unwrap_or_else(|| panic!("missing t={}", pair[1].0));
            let (_, plain, _) = core_parts(row, k);
            let raw = raw_number(row, field, plain);
            let previous = shown_number(&pair[0].1, field);
            let step = pair_for(field, &cfg);
            assert!(
                outside_window(raw, previous, step),
                "t={} {field} raw={raw} shown={previous} is still inside the 1/1 window",
                pair[1].0
            );
            assert_eq!(
                shown_number(&pair[1].1, field),
                step_round(raw, step.step),
                "t={} {field} was not re-quantised from the raw value",
                pair[1].0
            );
            fields.push(field);
        }
        if !fields.is_empty() {
            numeric.push((pair[1].0, fields));
        }
    }
    assert_eq!(
        numeric,
        vec![
            (14, vec!["cpu_c"]),
            (20, vec!["gpu_c"]),
            (58, vec!["coolant_c"]),
            (68, vec!["gpu_c"]),
            (118, vec!["cpu_c"]),
        ]
    );
    println!(
        "idle replay: ring/band/block changes={ring_changes}, numeric changes={}",
        numeric.len()
    );
}

#[test]
fn full_name_and_first_model_detail_reach_the_view() {
    let cfg = Config::default();
    let now = Instant::now();
    let history = History::new(now);
    let detail = llama_core::detail::ModelDetail {
        ctx: Some(262_144),
        ..llama_core::detail::ModelDetail::default()
    };
    let mut current = snapshot(now);
    current.ai = AiState::Loaded;
    current.models = vec![
        ModelInfo {
            backend: None,
            name: "Ternary Bon…".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Ternary Bonsai 2 27B".to_owned()),
            detail: Some(detail.clone()),
        },
        ModelInfo {
            backend: None,
            name: "Qwen 35B".to_owned(),
            state: "ready".to_owned(),
            full_name: None,
            detail: None,
        },
    ];
    let view = present(&current, &history, None, &cfg);
    assert_eq!(view.models, ["Ternary Bonsai 2 27B", "Qwen 35B"]);
    assert_eq!(view.detail, Some(detail));

    current.models.remove(0);
    let view = present(&current, &history, None, &cfg);
    assert_eq!(view.models, ["Qwen 35B"]);
    assert_eq!(view.detail, None, "an older watcher sends no detail");
}

/// #31: the first model's speculative acceptance reaches the view in whole
/// percents; a second model's does not.
#[test]
fn first_model_spec_acceptance_reaches_the_view() {
    let cfg = Config::default();
    let now = Instant::now();
    let history = History::new(now);
    let spec = |permille: u16| {
        Some(llama_core::backend::BackendInfo {
            kind: llama_core::backend::Backend::Vllm,
            engine: llama_core::backend::EngineStats {
                spec_permille: Some(permille),
                ..llama_core::backend::EngineStats::default()
            },
            ..llama_core::backend::BackendInfo::default()
        })
    };
    let model = |name: &str, backend| ModelInfo {
        backend,
        name: name.to_owned(),
        state: "ready".to_owned(),
        full_name: None,
        detail: None,
    };
    let mut current = snapshot(now);
    current.ai = AiState::Loaded;
    current.models = vec![model("qwen3.8-27b", spec(784))];
    let view = present(&current, &history, None, &cfg);
    assert_eq!(view.spec_permille, Some(780));
    current.models = vec![model("qwen3.8-27b", spec(785))];
    assert_eq!(
        present(&current, &history, None, &cfg).spec_permille,
        Some(790)
    );
    current.models = vec![model("a", None), model("b", spec(500))];
    assert_eq!(present(&current, &history, None, &cfg).spec_permille, None);
}

/// #33: the first model's engine reaches the view; a model without a
/// backend is llama.cpp; no model is no engine.
#[test]
fn first_model_engine_reaches_the_view() {
    use llama_core::backend::{Backend, BackendInfo};
    let cfg = Config::default();
    let now = Instant::now();
    let history = History::new(now);
    let model = |name: &str, kind: Option<Backend>| ModelInfo {
        backend: kind.map(|kind| BackendInfo {
            kind,
            ..BackendInfo::default()
        }),
        name: name.to_owned(),
        state: "ready".to_owned(),
        full_name: None,
        detail: None,
    };
    let mut current = snapshot(now);
    current.ai = AiState::Loaded;
    for kind in Backend::ALL {
        current.models = vec![model("m", Some(kind)), model("n", Some(Backend::Vllm))];
        assert_eq!(present(&current, &history, None, &cfg).engine, Some(kind));
    }
    current.models = vec![model("m", None)];
    assert_eq!(
        present(&current, &history, None, &cfg).engine,
        Some(Backend::LlamaCpp)
    );
    current.models.clear();
    assert_eq!(present(&current, &history, None, &cfg).engine, None);
}
