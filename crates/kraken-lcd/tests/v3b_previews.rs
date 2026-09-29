//! V3b preview views: the design's six values and four stories, aggregated
//! through the real `present` path (activity tiers, token cascade, stream
//! smoothing). Writes `target/preview-v3b-*.json`; `render-once` turns each
//! into a PNG. Ignored by default because it replays hours of 10 Hz samples:
//!
//! `cargo test --release -p kraken-lcd --test v3b_previews -- --ignored`

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use kraken_lcd::config::{Config, UploadMode, Variant as ConfigVariant};
use kraken_lcd::history::History;
use kraken_lcd::present::{Memory, View, present};
use llama_core::detail::ModelDetail;
use llama_core::sample::{AiState, ModelInfo, Snapshot, TokenReading};

/// tok/s per activity point: 100 % is about 90 tok/s on this box.
const TOK_PER_ACT: f64 = 0.9;

fn detail() -> ModelDetail {
    serde_json::from_str(
        r#"{"ctx": 262144, "ncmoe": 88, "quant": "UD-Q4_K_M", "kv_k": "q8_0", "kv_v": "q8_0"}"#,
    )
    .expect("detail")
}

/// A plausible day before the bar history, hours ago → activity.
fn day_activity(h: f64) -> f64 {
    if h > 20.0 {
        4.0 + 2.0 * (h * 9.0).sin()
    } else if h > 16.0 {
        55.0 + 25.0 * (h * 2.1).sin()
    } else if h > 13.0 {
        3.0
    } else if h > 9.0 {
        94.0 + 4.0 * (h * 7.0).sin()
    } else if h > 8.0 {
        100.0 + 18.0 * (h * 23.0).sin().max(0.0)
    } else if h > 3.0 {
        38.0 + 10.0 * (h * 3.3).sin()
    } else if h > 1.0 {
        88.0 + 6.0 * (h * 11.0).sin()
    } else {
        8.0
    }
}

struct Replay {
    cfg: Config,
    origin: Instant,
    start_ns: u64,
    frame: u64,
    total: f64,
    view: View,
}

impl Replay {
    /// `recent_s` is how long the bar history will run; the day before it
    /// seeds the token chart at 1 Hz.
    fn new(variant: ConfigVariant, recent_s: f64) -> Self {
        let mut cfg = Config::default();
        cfg.upload.mode = UploadMode::Stream;
        cfg.display.variant = variant;
        let mut memory = Memory::new(&cfg.dial);
        let day_s = (24.0 * 3600.0 - recent_s).max(0.0) as u64;
        for s in (1..=day_s).rev() {
            let h = (s as f64 + recent_s) / 3600.0;
            memory
                .chart
                .add((TOK_PER_ACT * day_activity(h)).round() as u64, 1_000);
        }
        let view = View {
            dial_state: memory.into(),
            ..View::default()
        };
        Self {
            cfg,
            origin: Instant::now(),
            start_ns: 7_000_000_000_000,
            frame: 0,
            total: 5_000_000.0,
            view,
        }
    }

    /// `seconds` at 10 Hz of `value(s)`, `s` from 0 within this segment.
    fn seed(&mut self, seconds: f64, value: impl Fn(f64) -> f64) {
        let frames = (seconds * 10.0).round() as u64;
        let history = History::new(self.origin);
        for i in 0..frames {
            let v = value(i as f64 / 10.0).clamp(0.0, 125.0);
            self.frame += 1;
            self.total += TOK_PER_ACT * v * 0.1;
            let snap = snapshot(
                self.origin + Duration::from_millis(100 * self.frame),
                v as f32,
                TokenReading {
                    run_id: 42,
                    seq: self.frame,
                    t_mono_ns: self.start_ns + self.frame * 100_000_000,
                    decoded_total: Some(self.total as u64),
                },
            );
            self.view = present(&snap, &history, Some(&self.view), &self.cfg);
        }
    }

    fn write(&self, name: &str) {
        self.write_as("v3b", name);
    }

    fn write_as(&self, set: &str, name: &str) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        std::fs::create_dir_all(&dir).expect("target dir");
        let path = dir.join(format!("preview-{set}-{name}.json"));
        let json = serde_json::to_vec_pretty(&self.view).expect("encode");
        assert!(
            json.len() < 16 * 1024,
            "{name} view is {} bytes",
            json.len()
        );
        std::fs::write(&path, json).expect("write view");
        println!("{}", path.display());
    }
}

fn snapshot(t: Instant, v: f32, tokens: TokenReading) -> Snapshot {
    Snapshot {
        t_mono: t,
        t_wall: SystemTime::UNIX_EPOCH,
        load: Some(v.min(100.0)),
        activity: Some(v),
        cpu_pct: Some(8.0 + v * 0.35),
        cpu_topk_pct: None,
        gpu_pct: Some(v.min(100.0)),
        mem_pct: Some(38.0),
        coolant_c: Some(34.0 + v * 0.06),
        cpu_c: Some(46.0 + v * 0.26),
        gpu_c: Some(41.0 + v * 0.34),
        ai: AiState::Loaded,
        models: vec![ModelInfo {
            backend: None,
            name: "Nemotron 3 S…".to_owned(),
            state: "ready".to_owned(),
            full_name: Some("Nemotron 3 Super 120B-A12B".to_owned()),
            detail: Some(detail()),
        }],
        tokens: Some(tokens),
        errors: BTreeSet::new(),
    }
}

/// The six-value row: ~30 min near 88 %, 25 s settling toward the value,
/// then 5 s at it.
fn six_value(variant: ConfigVariant, v: f64) -> Replay {
    let recent = 1_800.0 + 137.0 + 25.0 + 5.0;
    let mut replay = Replay::new(variant, recent);
    replay.seed(1_800.0 + 137.0, |s| {
        88.0 + 4.0 * (s / 37.0).sin() + 2.0 * (s / 7.3).sin()
    });
    replay.seed(25.0, |s| {
        if v > 0.0 {
            (v - (v - 60.0) * (-s / 4.0).exp()).max(0.0)
        } else {
            0.0
        }
    });
    replay.seed(5.0, |_| v);
    replay
}

#[test]
#[ignore = "writes preview views; run on demand in release"]
fn write_v3b_preview_views() {
    for v in [0.0, 35.0, 70.0, 100.0, 112.0, 125.0] {
        six_value(ConfigVariant::A1, v).write(&format!("{}", v as u32));
    }
    six_value(ConfigVariant::A3, 112.0).write("a3-112");
    six_value(ConfigVariant::A3, 125.0).write("a3-125");

    let mut spike = Replay::new(ConfigVariant::A1, 3_601.5);
    spike.seed(3_600.0, |s| 90.0 + 3.0 * (s / 41.0).sin());
    spike.seed(1.5, |_| 118.0);
    spike.write("spike-on-sustained-hour");

    let mut pinned = Replay::new(ConfigVariant::A1, 1_680.0);
    pinned.seed(1_500.0, |s| 70.0 + 5.0 * (s / 23.0).sin());
    pinned.seed(180.0, |_| 125.0);
    pinned.write("pinned-3-min");

    let mut cool = Replay::new(ConfigVariant::A1, 3_640.0);
    cool.seed(3_600.0, |s| 96.0 + 3.0 * (s / 31.0).sin());
    cool.seed(40.0, |s| 8.0 + (60.0 - s * 2.0).max(0.0));
    cool.write("cool-down");

    let mut warm = Replay::new(ConfigVariant::A1, 1_812.0);
    warm.seed(1_800.0, |s| 4.0 + 2.0 * (s / 9.0).sin());
    warm.seed(12.0, |s| (s * 12.0).min(96.0));
    warm.write("warm-up");
}

/// T62: the deeper-blues ramp across the cold end and into the blackbody.
#[test]
#[ignore = "writes preview views; run on demand in release"]
fn write_blues_preview_views() {
    for v in [0.0, 20.0, 45.0, 70.0, 100.0, 118.0, 125.0] {
        six_value(ConfigVariant::A1, v).write_as("blues", &format!("{}", v as u32));
    }
}
