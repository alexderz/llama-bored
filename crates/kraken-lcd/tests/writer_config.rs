//! Writer config limits and `packaging/config.example.toml`.

use std::path::Path;

use kraken_lcd::config::{
    Config, ConfigError, InvalidConfig, StepMargin, Tier, UploadMode, ValidConfig, Variant,
};

fn packaging(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packaging")
        .join(name)
}

fn tiers(pairs: &[(f64, u32)]) -> Vec<Tier> {
    pairs
        .iter()
        .copied()
        .map(|(width_s, bars)| Tier { width_s, bars })
        .collect()
}

#[test]
fn packaged_example_parses_and_validates() {
    let path = packaging("config.example.toml");
    let text = std::fs::read_to_string(&path).expect("config.example.toml");
    assert!(text.contains("[writer]"), "{text}");
    assert!(text.contains("[snapshot]"), "{text}");
    assert!(text.contains("[dial]"), "{text}");
    assert!(text.contains("variant = \"a1\""), "{text}");
    assert!(!text.contains("[collector]"), "{text}");
    assert!(!text.contains("[models]"), "{text}");
    assert!(!text.contains("llama_swap_url"), "{text}");
    assert!(!text.contains("http://"), "{text}");

    let cfg: ValidConfig = Config::load_validated(&path).expect("example validates");
    assert_eq!(cfg.writer.tick_s, 0.5);
    assert_eq!(cfg.snapshot.stale_after_s, 1.0);
    assert_eq!(cfg.snapshot.watch_down_stock_after_s, 30);
    assert_eq!(cfg.snapshot.watch_down_restore_min_s, 600);
    assert_eq!(
        cfg.dial.tiers,
        tiers(&[(0.5, 10), (5.0, 2), (15.0, 3), (60.0, 4), (300.0, 5)])
    );
    assert_eq!(cfg.dial.ceiling_tps, 150.0);
    assert_eq!(cfg.dial.xff, 0.5);
    assert_eq!(cfg.dial.max_gap_s, 2.0);
    assert_eq!(cfg.display.variant, Variant::A1);
    assert_eq!(cfg.display.rotate_deg, 0);
    assert_eq!(cfg.upload.min_interval_s, 60);
    assert_eq!(cfg.upload.fail_limit, 3);
    assert_eq!(cfg.upload.mode, UploadMode::Change);
    assert_eq!(cfg.upload.stream_fps, 10);
    assert!(text.contains("mode = \"change\""), "{text}");
    assert!(text.contains("stream_fps = 10"), "{text}");
    assert_eq!(cfg.bands.enter, [15, 40, 70]);
    assert_eq!(cfg.bands.margin, 5);
    assert_eq!(cfg.bands.fill_min_coverage, 0.5);
    assert_eq!(cfg.hysteresis.ring, StepMargin { step: 5, margin: 2 });
    assert_eq!(cfg.hysteresis.percent, StepMargin { step: 1, margin: 1 });
    assert_eq!(cfg.hysteresis.temp, StepMargin { step: 1, margin: 1 });
    assert_eq!(&*cfg, &Config::default());
}

#[test]
fn omitted_keys_take_the_example_defaults() {
    let cfg = Config::from_toml("").expect("empty toml");
    assert_eq!(cfg, Config::default());
    assert_eq!(cfg.display.variant, Variant::A1);
    assert!(cfg.validate().is_ok());
}

#[test]
fn upload_mode_is_change_or_stream_and_defaults_to_change() {
    let cfg = Config::from_toml("").expect("empty");
    assert_eq!(cfg.upload.mode, UploadMode::Change);
    assert_eq!(cfg.upload.stream_fps, 10);

    let cfg = Config::from_toml("[upload]\nmode = \"stream\"\n").expect("stream");
    assert_eq!(cfg.upload.mode, UploadMode::Stream);
    assert_eq!(cfg.upload.stream_fps, 10);
    assert!(cfg.validate().is_ok());

    let cfg = Config::from_toml("[upload]\nmode = \"change\"\nstream_fps = 12\n").expect("change");
    assert_eq!(cfg.upload.mode, UploadMode::Change);
    assert_eq!(cfg.upload.stream_fps, 12);
    assert!(cfg.validate().is_ok());

    let err = Config::from_toml("[upload]\nmode = \"burst\"\n");
    assert!(err.is_err(), "burst is not a mode: {err:?}");
}

#[test]
fn partial_override_keeps_other_defaults() {
    let cfg = Config::from_toml("[writer]\ntick_s = 1.0\n").expect("partial toml");
    assert_eq!(cfg.writer.tick_s, 1.0);
    assert_eq!(cfg.display.variant, Variant::A1);
    assert_eq!(cfg.display.rotate_deg, 0);
    assert_eq!(cfg.upload.min_interval_s, 60);
    assert_eq!(cfg.snapshot.stale_after_s, 1.0);
}

#[test]
fn collector_models_and_address_keys_fail_to_parse() {
    let cases = [
        ("collector section", "[collector]\ntick_s = 1.0\n"),
        ("models section", "[models]\nmax_name_chars = 12\n"),
        ("models aliases", "[models.aliases]\n\"a\" = \"b\"\n"),
        ("path", "path = \"/run/llama-watch/snapshot.json\"\n"),
        ("url", "url = \"http://127.0.0.1:8080\"\n"),
        ("address", "address = \"127.0.0.1:8080\"\n"),
        ("host", "host = \"127.0.0.1\"\n"),
        (
            "llama_swap_url",
            "llama_swap_url = \"http://127.0.0.1:8080\"\n",
        ),
        ("snapshot_dir", "snapshot_dir = \"/run/llama-watch\"\n"),
        ("snapshot_path", "snapshot_path = \"/tmp/snapshot.json\"\n"),
        (
            "snapshot path key",
            "[snapshot]\npath = \"/run/llama-watch/snapshot.json\"\n",
        ),
        ("llama table", "[llama]\nurl = \"http://127.0.0.1:8080\"\n"),
    ];
    for (name, toml) in cases {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn tier_pairs_accept_integer_widths_and_reject_a_bad_shape() {
    let cfg = Config::from_toml("[dial]\ntiers = [[0.5, 24]]\n").expect("one tier");
    assert_eq!(cfg.dial.tiers, tiers(&[(0.5, 24)]));

    let cfg = Config::from_toml("[dial]\ntiers = [[1, 24]]\n").expect("integer width");
    assert_eq!(cfg.dial.tiers[0].width_s, 1.0);
    assert_eq!(cfg.dial.tiers[0].bars, 24);

    for (name, toml) in [
        ("three elements", "[dial]\ntiers = [[0.5, 10, 1]]\n"),
        ("one element", "[dial]\ntiers = [[0.5]]\n"),
        ("float bar count", "[dial]\ntiers = [[0.5, 10.5]]\n"),
        (
            "inexact integer width",
            "[dial]\ntiers = [[9007199254740993, 24]]\n",
        ),
    ] {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn hysteresis_pair_must_be_a_two_element_array() {
    let cfg = Config::from_toml("[hysteresis]\nring = [9, 4]\n").expect("array form");
    assert_eq!(cfg.hysteresis.ring, StepMargin { step: 9, margin: 4 });
    assert_eq!(cfg.hysteresis.percent, StepMargin { step: 1, margin: 1 });
    for (name, toml) in [
        ("map", "[hysteresis.ring]\nstep = 5\nmargin = 2\n"),
        ("short", "[hysteresis]\nring = [5]\n"),
        ("long", "[hysteresis]\nring = [5, 2, 1]\n"),
    ] {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn bands_enter_must_be_three_values() {
    let cfg = Config::from_toml("[bands]\nenter = [10, 20, 30]\n").expect("three enters");
    assert_eq!(cfg.bands.enter, [10, 20, 30]);
    for (name, toml) in [
        ("two", "[bands]\nenter = [15, 40]\n"),
        ("four", "[bands]\nenter = [1, 2, 3, 4]\n"),
    ] {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn variant_is_rejected_at_parse_unless_it_is_a1_or_a3() {
    let a1 = Config::from_toml("[display]\nvariant = \"a1\"\n").expect("a1");
    assert_eq!(a1.display.variant, Variant::A1);
    let a3 = Config::from_toml("[display]\nvariant = \"a3\"\n").expect("a3");
    assert_eq!(a3.display.variant, Variant::A3);
    for (name, toml) in [
        ("a2", "[display]\nvariant = \"a2\"\n"),
        ("A1", "[display]\nvariant = \"A1\"\n"),
        ("empty", "[display]\nvariant = \"\"\n"),
    ] {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn negative_watch_down_delays_fail_to_parse() {
    assert!(Config::from_toml("[snapshot]\nwatch_down_stock_after_s = -1\n").is_err());
    assert!(Config::from_toml("[snapshot]\nwatch_down_restore_min_s = -1\n").is_err());
}

#[test]
fn load_missing_file_is_a_read_error() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer-missing.toml");
    let err = Config::load_validated(&path).expect_err("missing file");
    assert!(matches!(err, ConfigError::Read { .. }), "{err}");
}

#[test]
fn parse_error_names_the_line_and_not_the_file_text() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer-parse-leak.toml");
    let text = "secret_marker = \"SUPERSECRETVALUE\"\nnot toml at all\n";
    std::fs::write(&path, text).expect("write fixture");
    let err = Config::load_validated(&path).expect_err("syntax error");
    let shown = err.to_string();
    assert!(shown.contains("parse error at line 2"), "{shown}");
    assert!(!shown.contains("SUPERSECRETVALUE"), "{shown}");
    assert!(!shown.contains("secret_marker"), "{shown}");
    assert!(!shown.contains("not toml"), "{shown}");
}

#[test]
fn load_validated_rejects_an_invalid_file() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("writer-below-tick.toml");
    std::fs::write(&path, "[writer]\ntick_s = 0.01\n").expect("write fixture");
    let err = Config::load_validated(&path).expect_err("tick_s 0.01");
    assert!(
        matches!(
            err,
            ConfigError::Invalid {
                source: InvalidConfig::TickS { .. },
                ..
            }
        ),
        "{err}"
    );
    assert!(
        err.to_string().contains(&path.display().to_string()),
        "{err}"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Ok,
    TickS,
    StaleAfter,
    WatchDownStockAfter,
    WatchDownRestoreMin,
    TierZero,
    TierOrder,
    TierMultiple,
    TierBars,
    TierBarCount,
    TierWidth,
    TierWindow,
    CeilingTps,
    Xff,
    MaxGap,
    MinInterval,
    FailLimit,
    StreamFps,
    BandsEnter,
    BandsMargin,
    FillMinCoverage,
    HysteresisStep,
    RotateDeg,
}

fn rule_of(result: Result<(), InvalidConfig>) -> Rule {
    match result {
        Ok(()) => Rule::Ok,
        Err(InvalidConfig::TickS { .. }) => Rule::TickS,
        Err(InvalidConfig::StaleAfter { .. }) => Rule::StaleAfter,
        Err(InvalidConfig::WatchDownStockAfter { .. }) => Rule::WatchDownStockAfter,
        Err(InvalidConfig::WatchDownRestoreMin { .. }) => Rule::WatchDownRestoreMin,
        Err(InvalidConfig::TierZero { .. }) => Rule::TierZero,
        Err(InvalidConfig::TierOrder { .. }) => Rule::TierOrder,
        Err(InvalidConfig::TierMultiple { .. }) => Rule::TierMultiple,
        Err(InvalidConfig::TierBars { .. }) => Rule::TierBars,
        Err(InvalidConfig::TierBarCount { .. }) => Rule::TierBarCount,
        Err(InvalidConfig::TierWidth { .. }) => Rule::TierWidth,
        Err(InvalidConfig::TierWindow { .. }) => Rule::TierWindow,
        Err(InvalidConfig::CeilingTps { .. }) => Rule::CeilingTps,
        Err(InvalidConfig::Xff { .. }) => Rule::Xff,
        Err(InvalidConfig::MaxGap { .. }) => Rule::MaxGap,
        Err(InvalidConfig::MinInterval { .. }) => Rule::MinInterval,
        Err(InvalidConfig::FailLimit { .. }) => Rule::FailLimit,
        Err(InvalidConfig::StreamFps { .. }) => Rule::StreamFps,
        Err(InvalidConfig::BandsEnter { .. }) => Rule::BandsEnter,
        Err(InvalidConfig::BandsMargin { .. }) => Rule::BandsMargin,
        Err(InvalidConfig::FillMinCoverage { .. }) => Rule::FillMinCoverage,
        Err(InvalidConfig::HysteresisStep { .. }) => Rule::HysteresisStep,
        Err(InvalidConfig::RotateDeg { .. }) => Rule::RotateDeg,
    }
}

struct Case {
    name: &'static str,
    mutate: fn(&mut Config),
    expect: Rule,
}

fn cases() -> &'static [Case] {
    &[
        Case {
            name: "tick_s lower bound",
            mutate: |cfg| cfg.writer.tick_s = 0.1,
            expect: Rule::Ok,
        },
        Case {
            name: "tick_s upper bound",
            mutate: |cfg| cfg.writer.tick_s = 2.0,
            expect: Rule::Ok,
        },
        Case {
            name: "tick_s below range",
            mutate: |cfg| cfg.writer.tick_s = (0.1_f64).next_down(),
            expect: Rule::TickS,
        },
        Case {
            name: "tick_s above range",
            mutate: |cfg| cfg.writer.tick_s = (2.0_f64).next_up(),
            expect: Rule::TickS,
        },
        Case {
            name: "tick_s nan",
            mutate: |cfg| cfg.writer.tick_s = f64::NAN,
            expect: Rule::TickS,
        },
        Case {
            name: "stale_after_s lower bound",
            mutate: |cfg| cfg.snapshot.stale_after_s = 0.3,
            expect: Rule::Ok,
        },
        Case {
            name: "stale_after_s upper bound",
            mutate: |cfg| cfg.snapshot.stale_after_s = 10.0,
            expect: Rule::Ok,
        },
        Case {
            name: "stale_after_s below range",
            mutate: |cfg| cfg.snapshot.stale_after_s = (0.3_f64).next_down(),
            expect: Rule::StaleAfter,
        },
        Case {
            name: "stale_after_s above range",
            mutate: |cfg| cfg.snapshot.stale_after_s = (10.0_f64).next_up(),
            expect: Rule::StaleAfter,
        },
        Case {
            name: "stale_after_s nan",
            mutate: |cfg| cfg.snapshot.stale_after_s = f64::NAN,
            expect: Rule::StaleAfter,
        },
        Case {
            name: "watch_down_stock_after_s lower bound",
            mutate: |cfg| cfg.snapshot.watch_down_stock_after_s = 5,
            expect: Rule::Ok,
        },
        Case {
            name: "watch_down_stock_after_s upper bound",
            mutate: |cfg| cfg.snapshot.watch_down_stock_after_s = 600,
            expect: Rule::Ok,
        },
        Case {
            name: "watch_down_stock_after_s below range",
            mutate: |cfg| cfg.snapshot.watch_down_stock_after_s = 4,
            expect: Rule::WatchDownStockAfter,
        },
        Case {
            name: "watch_down_stock_after_s above range",
            mutate: |cfg| cfg.snapshot.watch_down_stock_after_s = 601,
            expect: Rule::WatchDownStockAfter,
        },
        Case {
            name: "watch_down_restore_min_s lower bound",
            mutate: |cfg| cfg.snapshot.watch_down_restore_min_s = 60,
            expect: Rule::Ok,
        },
        Case {
            name: "watch_down_restore_min_s upper bound",
            mutate: |cfg| cfg.snapshot.watch_down_restore_min_s = 3600,
            expect: Rule::Ok,
        },
        Case {
            name: "watch_down_restore_min_s below range",
            mutate: |cfg| cfg.snapshot.watch_down_restore_min_s = 59,
            expect: Rule::WatchDownRestoreMin,
        },
        Case {
            name: "watch_down_restore_min_s above range",
            mutate: |cfg| cfg.snapshot.watch_down_restore_min_s = 3601,
            expect: Rule::WatchDownRestoreMin,
        },
        Case {
            name: "single tier at 0.5 with 24 bars",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 24)]),
            expect: Rule::Ok,
        },
        Case {
            name: "width exactly twice the previous",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 12), (1.0, 12)]),
            expect: Rule::Ok,
        },
        Case {
            name: "non-integer width that is a whole multiple",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 12), (1.5, 12)]),
            expect: Rule::Ok,
        },
        Case {
            name: "empty tiers",
            mutate: |cfg| cfg.dial.tiers.clear(),
            expect: Rule::TierZero,
        },
        Case {
            name: "T0 is not 0.5",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(1.0, 24)]),
            expect: Rule::TierZero,
        },
        Case {
            name: "widths not strictly ascending",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 12), (0.5, 12)]),
            expect: Rule::TierOrder,
        },
        Case {
            name: "width not a whole multiple of the previous",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 12), (0.75, 12)]),
            expect: Rule::TierMultiple,
        },
        Case {
            name: "bar total 23",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 23)]),
            expect: Rule::TierBars,
        },
        Case {
            name: "bar total 25",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 25)]),
            expect: Rule::TierBars,
        },
        Case {
            name: "a tier with zero bars",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 24), (1.0, 0)]),
            expect: Rule::TierBarCount,
        },
        Case {
            name: "width at the 5 min cap",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 23), (300.0, 1)]),
            expect: Rule::Ok,
        },
        Case {
            name: "width above the 5 min cap",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 23), ((300.0_f64).next_up(), 1)]),
            expect: Rule::TierWidth,
        },
        Case {
            name: "non-finite width",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 23), (f64::INFINITY, 1)]),
            expect: Rule::TierWidth,
        },
        Case {
            name: "nan width",
            mutate: |cfg| cfg.dial.tiers = tiers(&[(0.5, 23), (f64::NAN, 1)]),
            expect: Rule::TierWidth,
        },
        Case {
            name: "displayed window of 30 min",
            mutate: |cfg| {
                cfg.dial.tiers = tiers(&[(0.5, 10), (5.0, 2), (15.0, 3), (60.0, 4), (300.0, 5)]);
            },
            expect: Rule::Ok,
        },
        Case {
            name: "displayed window above 30 min",
            mutate: |cfg| {
                cfg.dial.tiers = tiers(&[(0.5, 9), (5.0, 3), (15.0, 3), (60.0, 4), (300.0, 5)]);
            },
            expect: Rule::TierWindow,
        },
        Case {
            name: "ceiling_tps lower bound",
            mutate: |cfg| cfg.dial.ceiling_tps = 10.0,
            expect: Rule::Ok,
        },
        Case {
            name: "ceiling_tps upper bound",
            mutate: |cfg| cfg.dial.ceiling_tps = 10_000.0,
            expect: Rule::Ok,
        },
        Case {
            name: "ceiling_tps below range",
            mutate: |cfg| cfg.dial.ceiling_tps = (10.0_f64).next_down(),
            expect: Rule::CeilingTps,
        },
        Case {
            name: "ceiling_tps above range",
            mutate: |cfg| cfg.dial.ceiling_tps = (10_000.0_f64).next_up(),
            expect: Rule::CeilingTps,
        },
        Case {
            name: "ceiling_tps nan",
            mutate: |cfg| cfg.dial.ceiling_tps = f64::NAN,
            expect: Rule::CeilingTps,
        },
        Case {
            name: "xff lower bound",
            mutate: |cfg| cfg.dial.xff = 0.0,
            expect: Rule::Ok,
        },
        Case {
            name: "xff upper bound",
            mutate: |cfg| cfg.dial.xff = 1.0,
            expect: Rule::Ok,
        },
        Case {
            name: "xff below range",
            mutate: |cfg| cfg.dial.xff = (0.0_f64).next_down(),
            expect: Rule::Xff,
        },
        Case {
            name: "xff above range",
            mutate: |cfg| cfg.dial.xff = (1.0_f64).next_up(),
            expect: Rule::Xff,
        },
        Case {
            name: "xff nan",
            mutate: |cfg| cfg.dial.xff = f64::NAN,
            expect: Rule::Xff,
        },
        Case {
            name: "max_gap_s lower bound",
            mutate: |cfg| cfg.dial.max_gap_s = 0.5,
            expect: Rule::Ok,
        },
        Case {
            name: "max_gap_s upper bound",
            mutate: |cfg| cfg.dial.max_gap_s = 10.0,
            expect: Rule::Ok,
        },
        Case {
            name: "max_gap_s below range",
            mutate: |cfg| cfg.dial.max_gap_s = (0.5_f64).next_down(),
            expect: Rule::MaxGap,
        },
        Case {
            name: "max_gap_s above range",
            mutate: |cfg| cfg.dial.max_gap_s = (10.0_f64).next_up(),
            expect: Rule::MaxGap,
        },
        Case {
            name: "min_interval_s = 10 accepted",
            mutate: |cfg| cfg.upload.min_interval_s = 10,
            expect: Rule::Ok,
        },
        Case {
            name: "min_interval_s = 9 rejected",
            mutate: |cfg| cfg.upload.min_interval_s = 9,
            expect: Rule::MinInterval,
        },
        Case {
            name: "fail_limit lower bound",
            mutate: |cfg| cfg.upload.fail_limit = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "fail_limit upper bound",
            mutate: |cfg| cfg.upload.fail_limit = 10,
            expect: Rule::Ok,
        },
        Case {
            name: "fail_limit zero",
            mutate: |cfg| cfg.upload.fail_limit = 0,
            expect: Rule::FailLimit,
        },
        Case {
            name: "fail_limit above 10",
            mutate: |cfg| cfg.upload.fail_limit = 11,
            expect: Rule::FailLimit,
        },
        Case {
            name: "stream_fps lower bound",
            mutate: |cfg| cfg.upload.stream_fps = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "stream_fps upper bound",
            mutate: |cfg| cfg.upload.stream_fps = 12,
            expect: Rule::Ok,
        },
        Case {
            name: "stream_fps zero",
            mutate: |cfg| cfg.upload.stream_fps = 0,
            expect: Rule::StreamFps,
        },
        Case {
            name: "stream_fps above 12",
            mutate: |cfg| cfg.upload.stream_fps = 13,
            expect: Rule::StreamFps,
        },
        Case {
            name: "enter bounds",
            mutate: |cfg| cfg.bands.enter = [1, 50, 100],
            expect: Rule::Ok,
        },
        Case {
            name: "enter below 1",
            mutate: |cfg| cfg.bands.enter = [0, 40, 70],
            expect: Rule::BandsEnter,
        },
        Case {
            name: "enter above 100",
            mutate: |cfg| cfg.bands.enter = [15, 40, 101],
            expect: Rule::BandsEnter,
        },
        Case {
            name: "enter not strictly ascending",
            mutate: |cfg| cfg.bands.enter = [15, 70, 40],
            expect: Rule::BandsEnter,
        },
        Case {
            name: "enter equal neighbours",
            mutate: |cfg| cfg.bands.enter = [15, 15, 70],
            expect: Rule::BandsEnter,
        },
        Case {
            name: "margin just under the smallest gap",
            mutate: |cfg| cfg.bands.margin = 24,
            expect: Rule::Ok,
        },
        Case {
            name: "margin equal to the smallest gap",
            mutate: |cfg| cfg.bands.margin = 25,
            expect: Rule::BandsMargin,
        },
        Case {
            name: "fill_min_coverage lower bound",
            mutate: |cfg| cfg.bands.fill_min_coverage = 0.0,
            expect: Rule::Ok,
        },
        Case {
            name: "fill_min_coverage upper bound",
            mutate: |cfg| cfg.bands.fill_min_coverage = 1.0,
            expect: Rule::Ok,
        },
        Case {
            name: "fill_min_coverage below 0",
            mutate: |cfg| cfg.bands.fill_min_coverage = -0.1,
            expect: Rule::FillMinCoverage,
        },
        Case {
            name: "fill_min_coverage above 1",
            mutate: |cfg| cfg.bands.fill_min_coverage = 1.1,
            expect: Rule::FillMinCoverage,
        },
        Case {
            name: "fill_min_coverage nan",
            mutate: |cfg| cfg.bands.fill_min_coverage = f64::NAN,
            expect: Rule::FillMinCoverage,
        },
        Case {
            name: "hysteresis ring step 1",
            mutate: |cfg| cfg.hysteresis.ring.step = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "hysteresis ring step 0",
            mutate: |cfg| cfg.hysteresis.ring.step = 0,
            expect: Rule::HysteresisStep,
        },
        Case {
            name: "hysteresis percent step 0",
            mutate: |cfg| cfg.hysteresis.percent.step = 0,
            expect: Rule::HysteresisStep,
        },
        Case {
            name: "hysteresis temp step 0",
            mutate: |cfg| cfg.hysteresis.temp.step = 0,
            expect: Rule::HysteresisStep,
        },
        Case {
            name: "rotate 0",
            mutate: |cfg| cfg.display.rotate_deg = 0,
            expect: Rule::Ok,
        },
        Case {
            name: "rotate 90",
            mutate: |cfg| cfg.display.rotate_deg = 90,
            expect: Rule::Ok,
        },
        Case {
            name: "rotate 180",
            mutate: |cfg| cfg.display.rotate_deg = 180,
            expect: Rule::Ok,
        },
        Case {
            name: "rotate 270",
            mutate: |cfg| cfg.display.rotate_deg = 270,
            expect: Rule::Ok,
        },
        Case {
            name: "rotate 45",
            mutate: |cfg| cfg.display.rotate_deg = 45,
            expect: Rule::RotateDeg,
        },
        Case {
            name: "rotate 360",
            mutate: |cfg| cfg.display.rotate_deg = 360,
            expect: Rule::RotateDeg,
        },
    ]
}

#[test]
fn validate_accepts_one_and_rejects_one_per_rule() {
    let mut missed = Vec::new();
    for case in cases() {
        let mut cfg = Config::default();
        (case.mutate)(&mut cfg);
        let got = rule_of(cfg.validate());
        if got != case.expect {
            missed.push(format!(
                "{}: expected {:?}, got {got:?}",
                case.name, case.expect
            ));
        }
    }
    assert!(missed.is_empty(), "\n{}", missed.join("\n"));
}
