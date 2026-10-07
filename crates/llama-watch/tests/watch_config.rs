//! Watcher config limits and `packaging/watch.example.toml`.

use std::path::Path;

use llama_watch::config::{
    ChartGlyphs, Config, ConfigError, InvalidWatchConfig, PromptView, TtyFont, TtySize,
    ValidWatchConfig,
};

fn packaging(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packaging")
        .join(name)
}

#[test]
fn packaged_example_parses_and_validates() {
    let path = packaging("watch.example.toml");
    let text = std::fs::read_to_string(&path).expect("watch.example.toml");
    assert!(text.contains("[collector]"), "{text}");
    assert!(text.contains("[llama]"), "{text}");
    assert!(text.contains("[models.aliases]"), "{text}");
    assert!(text.contains("[tty]"), "{text}");
    assert!(text.contains("chart_bucket_s"), "{text}");
    assert!(text.contains("ctx_history_h = 6"), "{text}");
    assert!(text.contains("chart_glyphs = \"eighths\""), "{text}");
    assert!(text.contains("url = \"http://127.0.0.1:8080\""), "{text}");
    assert!(text.contains("enabled = true"), "{text}");
    // A proxy in front of llama-swap hides /slots. The example says so.
    assert!(text.contains("proxy"), "{text}");
    // Host-specific model ids are not shipped. The alias is a commented example.
    assert!(text.contains("# \"my-model-id\" = "), "{text}");
    assert!(!text.contains("[snapshot]"), "{text}");
    assert!(!text.contains("snapshot_dir"), "{text}");
    assert!(!text.contains("snapshot_path"), "{text}");
    // T52: fans ship as a commented example; the code default stays off.
    assert!(text.contains("#[fans]"), "{text}");
    assert!(text.contains("#hwmon = \"nct6798\""), "{text}");
    // T72: backend overrides ship as a commented example.
    assert!(text.contains("#[llama.backends]"), "{text}");
    assert!(text.contains("#\"my-sglang-model\" = \"sglang\""), "{text}");
    assert!(text.contains("#channels = [2, 3, 5, 6]"), "{text}");

    let cfg: ValidWatchConfig = Config::load_validated(&path, 8).expect("example validates");
    assert_eq!(cfg.collector.tick_s, 0.1);
    assert_eq!(cfg.collector.cpu_window_s, 1.0);
    assert_eq!(cfg.collector.cpu_top_k, 8);
    assert_eq!(cfg.llama.url, "http://127.0.0.1:8080");
    assert!(cfg.llama.enabled);
    assert_eq!(cfg.llama.running_interval_s, 0.5);
    assert_eq!(cfg.llama.running_timeout_s, 0.25);
    assert_eq!(cfg.llama.metrics_interval_s, 0.25);
    assert_eq!(cfg.llama.metrics_timeout_s, 0.2);
    assert_eq!(cfg.llama.slots_interval_s, 1.0);
    assert_eq!(cfg.llama.slots_timeout_s, 0.5);
    assert_eq!(cfg.llama.slots_max_bytes, 4_194_304);
    assert_eq!(cfg.llama.activity_interval_s, 2.0);
    assert_eq!(cfg.llama.activity_timeout_s, 0.25);
    assert_eq!(cfg.llama.input_tail_chars, 8192);
    assert_eq!(cfg.llama.output_tail_chars, 24576);
    assert_eq!(cfg.models.max_name_chars, 12);
    assert!(cfg.models.aliases.is_empty(), "{:?}", cfg.models.aliases);
    assert!(cfg.llama.backends.is_empty(), "{:?}", cfg.llama.backends);
    assert_eq!(cfg.tty.fps, 10);
    assert_eq!(cfg.tty.full_redraw_s, 5);
    assert_eq!(cfg.tty.gen_ceiling_tps, 250.0);
    assert_eq!(cfg.tty.prompt_ceiling_tps, 1500.0);
    assert_eq!(cfg.tty.chart_bucket_s, 2);
    assert_eq!(cfg.tty.ctx_history_h, 6);
    assert_eq!(cfg.tty.prompt_view, PromptView::Clean);
    // T61: console blank ships as a commented example; the code default is off.
    assert!(text.contains("#blank_min = 10"), "{text}");
    assert!(text.contains("#sleep_min = 15"), "{text}");
    // #7: the console font ships explicit; the size is a commented example.
    assert!(text.contains("font = \"12x24\""), "{text}");
    assert!(text.contains("#size = \"160x49\""), "{text}");
    assert_eq!(cfg.tty.font, TtyFont::Hack12x24);
    assert_eq!(cfg.tty.size, None);
    assert_eq!(cfg.tty.blank_min, 0);
    assert_eq!(cfg.tty.sleep_min, 0);
    // #26: the llama palette ships explicit, same as the code default.
    assert!(text.contains("palette = \"llama\""), "{text}");
    assert_eq!(cfg.tty.palette, llama_watch::config::Palette::Llama);
    assert_eq!(cfg.load.cpu_limit_w, 230.0);
    assert_eq!(cfg.load.idle, llama_watch::config::IdleMode::Auto);
    assert_eq!(cfg.load.gpu_idle_w, 30.0);
    assert_eq!(cfg.load.cpu_idle_w, 25.0);
    assert_eq!(cfg.load.smooth_s, 0.3);
    assert_eq!(cfg.load.nominal_frac, 0.8);
    assert!(!cfg.fans.enabled);
    // The example opts in to the llama-hack-12x24 eighths; the code default
    // stays on glyphs eurlatgr also has.
    assert_eq!(cfg.tty.chart_glyphs, ChartGlyphs::Eighths);
    let mut same = (*cfg).clone();
    same.tty.chart_glyphs = ChartGlyphs::Halves;
    assert_eq!(same, Config::default());

    let err = Config::load_validated(&path, 7).expect_err("cpu_top_k 8 is above nproc 7");
    assert!(
        matches!(
            err,
            ConfigError::Invalid {
                source: InvalidWatchConfig::CpuTopK {
                    cpu_top_k: 8,
                    nproc: 7,
                },
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn omitted_keys_take_the_example_defaults() {
    let cfg = Config::from_toml("").expect("empty toml");
    assert_eq!(cfg, Config::default());
    assert!(cfg.validate(32).is_ok());
}

#[test]
fn chart_glyphs_default_to_halves_and_parse_both_modes() {
    assert_eq!(Config::default().tty.chart_glyphs, ChartGlyphs::Halves);
    let cfg = Config::from_toml("[tty]\nchart_glyphs = \"eighths\"\n").expect("eighths");
    assert_eq!(cfg.tty.chart_glyphs, ChartGlyphs::Eighths);
    let cfg = Config::from_toml("[tty]\nchart_glyphs = \"halves\"\n").expect("halves");
    assert_eq!(cfg.tty.chart_glyphs, ChartGlyphs::Halves);
    for bad in ["\"quarters\"", "\"Eighths\"", "8", "true"] {
        let text = format!("[tty]\nchart_glyphs = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "accepted {bad}");
    }
}

/// #26: `[tty] palette`.
#[test]
fn palette_defaults_to_llama_and_parses_both() {
    use llama_watch::config::Palette;
    assert_eq!(Config::default().tty.palette, Palette::Llama);
    let cfg = Config::from_toml("[tty]\npalette = \"vga\"\n").expect("vga");
    assert_eq!(cfg.tty.palette, Palette::Vga);
    let cfg = Config::from_toml("[tty]\npalette = \"llama\"\n").expect("llama");
    assert_eq!(cfg.tty.palette, Palette::Llama);
    for bad in ["\"xterm\"", "\"Llama\"", "1", "true"] {
        let text = format!("[tty]\npalette = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "accepted {bad}");
    }
}

#[test]
fn prompt_view_defaults_to_clean_and_parses_both_modes() {
    assert_eq!(Config::default().tty.prompt_view, PromptView::Clean);
    let cfg = Config::from_toml("[tty]\nprompt_view = \"raw\"\n").expect("raw");
    assert_eq!(cfg.tty.prompt_view, PromptView::Raw);
    let cfg = Config::from_toml("[tty]\nprompt_view = \"clean\"\n").expect("clean");
    assert_eq!(cfg.tty.prompt_view, PromptView::Clean);
    for bad in ["\"Clean\"", "\"pretty\"", "1", "false"] {
        let text = format!("[tty]\nprompt_view = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "accepted {bad}");
    }
}

#[test]
fn partial_override_keeps_other_defaults() {
    let cfg = Config::from_toml("[collector]\ntick_s = 0.5\n").expect("partial toml");
    assert_eq!(cfg.collector.tick_s, 0.5);
    assert_eq!(cfg.collector.cpu_window_s, 1.0);
    assert_eq!(cfg.collector.cpu_top_k, 8);
    assert_eq!(cfg.llama.url, "http://127.0.0.1:8080");
    assert!(cfg.llama.enabled);
    assert_eq!(cfg.tty.fps, 10);
}

#[test]
fn llama_can_be_disabled_and_still_validates() {
    let cfg = Config::from_toml("[llama]\nenabled = false\n").expect("disabled llama");
    assert!(!cfg.llama.enabled);
    assert_eq!(cfg.llama.url, "http://127.0.0.1:8080");
    assert!(cfg.validate(8).is_ok());
    // Disabling llama does not relax the loopback rule on the URL.
    let remote = Config::from_toml("[llama]\nenabled = false\nurl = \"http://192.0.2.1:8080\"\n")
        .expect("parses");
    assert!(matches!(
        remote.validate(8),
        Err(InvalidWatchConfig::LlamaUrl { .. })
    ));
}

#[test]
fn default_aliases_are_empty() {
    assert!(Config::default().models.aliases.is_empty());
}

#[test]
fn explicit_empty_aliases_stay_empty() {
    let cfg = Config::from_toml("[models.aliases]\n").expect("empty alias table");
    assert!(cfg.models.aliases.is_empty());
    assert_eq!(cfg.models.max_name_chars, 12);
}

#[test]
fn snapshot_dir_and_path_keys_fail_to_parse() {
    let cases = [
        ("snapshot table", "[snapshot]\nstale_after_s = 1.0\n"),
        ("snapshot_dir", "snapshot_dir = \"/run/llama-watch\"\n"),
        (
            "snapshot_path",
            "snapshot_path = \"/run/llama-watch/snapshot.json\"\n",
        ),
        ("snapshot_file", "snapshot_file = \"snapshot.json\"\n"),
        ("path", "path = \"/tmp/snapshot.json\"\n"),
        ("collector path", "[collector]\npath = \"/tmp\"\n"),
    ];
    for (name, toml) in cases {
        let err = Config::from_toml(toml);
        assert!(err.is_err(), "{name} should fail to parse, got {err:?}");
    }
}

#[test]
fn load_missing_file_is_a_read_error() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("watch-missing.toml");
    let err = Config::load(&path).expect_err("missing file");
    assert!(matches!(err, ConfigError::Read { .. }), "{err}");
}

#[test]
fn parse_error_names_the_line_and_not_the_file_text() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("watch-parse-leak.toml");
    let text = "secret_marker = \"SUPERSECRETVALUE\"\nnot toml at all\n";
    std::fs::write(&path, text).expect("write fixture");
    let err = Config::load(&path).expect_err("syntax error");
    let shown = err.to_string();
    assert!(shown.contains("parse error at line 2"), "{shown}");
    assert!(!shown.contains("SUPERSECRETVALUE"), "{shown}");
    assert!(!shown.contains("secret_marker"), "{shown}");
    assert!(!shown.contains("not toml"), "{shown}");
}

#[test]
fn load_parse_error_includes_the_file_path() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("watch-unknown.toml");
    std::fs::write(&path, "brightness = 50\n").expect("write fixture");
    let err = Config::load(&path).expect_err("unknown key");
    let message = err.to_string();
    assert!(
        message.contains(&path.display().to_string()),
        "parse error should name the file, got {message}"
    );
}

#[test]
fn load_validated_rejects_an_invalid_file() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("watch-below-tick.toml");
    std::fs::write(&path, "[collector]\ntick_s = 0.01\n").expect("write fixture");
    let err = Config::load_validated(&path, 32).expect_err("tick_s 0.01");
    assert!(
        matches!(
            err,
            ConfigError::Invalid {
                source: InvalidWatchConfig::TickS { .. },
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

#[test]
fn llama_url_with_a_trailing_slash_says_no_path_allowed() {
    let mut cfg = Config::default();
    cfg.llama.url = "http://127.0.0.1:8080/".to_owned();
    let err = cfg.validate(8).expect_err("trailing slash");
    assert!(err.to_string().contains("no path allowed"), "{err}");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Ok,
    TickS,
    CpuWindow,
    CpuTopK,
    LlamaUrl,
    Timeout,
    SlotsInterval,
    SlotsMaxBytes,
    PollInterval,
    InputTail,
    OutputTail,
    GenCeiling,
    PromptCeiling,
    ChartBucket,
    CtxHistory,
    AliasCount,
    AliasValue,
    BackendCount,
    MaxNameChars,
    Fps,
    FullRedraw,
    BlankMin,
    SleepMin,
    SmoothS,
    NominalFrac,
    CpuLimit,
    IdleWatts,
    Fans,
    Temps,
    Setup,
    TtySize,
}

fn rule_of(result: Result<(), InvalidWatchConfig>) -> Rule {
    match result {
        Ok(()) => Rule::Ok,
        Err(InvalidWatchConfig::TickS { .. }) => Rule::TickS,
        Err(InvalidWatchConfig::CpuWindow { .. }) => Rule::CpuWindow,
        Err(InvalidWatchConfig::CpuTopK { .. }) => Rule::CpuTopK,
        Err(InvalidWatchConfig::LlamaUrl { .. }) => Rule::LlamaUrl,
        Err(InvalidWatchConfig::Timeout { .. }) => Rule::Timeout,
        Err(InvalidWatchConfig::SlotsInterval { .. }) => Rule::SlotsInterval,
        Err(InvalidWatchConfig::SlotsMaxBytes { .. }) => Rule::SlotsMaxBytes,
        Err(InvalidWatchConfig::PollInterval { .. }) => Rule::PollInterval,
        Err(InvalidWatchConfig::InputTail { .. }) => Rule::InputTail,
        Err(InvalidWatchConfig::OutputTail { .. }) => Rule::OutputTail,
        Err(InvalidWatchConfig::GenCeiling { .. }) => Rule::GenCeiling,
        Err(InvalidWatchConfig::PromptCeiling { .. }) => Rule::PromptCeiling,
        Err(InvalidWatchConfig::ChartBucket { .. }) => Rule::ChartBucket,
        Err(InvalidWatchConfig::CtxHistory { .. }) => Rule::CtxHistory,
        Err(InvalidWatchConfig::AliasCount { .. }) => Rule::AliasCount,
        Err(InvalidWatchConfig::AliasValue { .. }) => Rule::AliasValue,
        Err(InvalidWatchConfig::BackendCount { .. }) => Rule::BackendCount,
        Err(InvalidWatchConfig::MaxNameChars { .. }) => Rule::MaxNameChars,
        Err(InvalidWatchConfig::Fps { .. }) => Rule::Fps,
        Err(InvalidWatchConfig::FullRedraw { .. }) => Rule::FullRedraw,
        Err(InvalidWatchConfig::BlankMin { .. }) => Rule::BlankMin,
        Err(InvalidWatchConfig::SleepMin { .. }) => Rule::SleepMin,
        Err(InvalidWatchConfig::SmoothS { .. }) => Rule::SmoothS,
        Err(InvalidWatchConfig::NominalFrac { .. }) => Rule::NominalFrac,
        Err(InvalidWatchConfig::CpuLimit { .. }) => Rule::CpuLimit,
        Err(InvalidWatchConfig::IdleWatts { .. }) => Rule::IdleWatts,
        Err(InvalidWatchConfig::Fans { .. }) => Rule::Fans,
        Err(InvalidWatchConfig::Temps { .. }) => Rule::Temps,
        Err(InvalidWatchConfig::Setup { .. }) => Rule::Setup,
        Err(InvalidWatchConfig::TtySize { .. }) => Rule::TtySize,
    }
}

struct Case {
    name: &'static str,
    nproc: u32,
    mutate: fn(&mut Config),
    expect: Rule,
}

fn cases() -> &'static [Case] {
    &[
        Case {
            name: "tick_s lower bound",
            nproc: 32,
            mutate: |cfg| cfg.collector.tick_s = 0.05,
            expect: Rule::Ok,
        },
        Case {
            name: "tick_s upper bound",
            nproc: 32,
            mutate: |cfg| cfg.collector.tick_s = 1.0,
            expect: Rule::Ok,
        },
        Case {
            name: "tick_s below range",
            nproc: 32,
            mutate: |cfg| cfg.collector.tick_s = (0.05_f64).next_down(),
            expect: Rule::TickS,
        },
        Case {
            name: "tick_s above range",
            nproc: 32,
            mutate: |cfg| {
                cfg.collector.tick_s = (1.0_f64).next_up();
                cfg.collector.cpu_window_s = 5.0;
            },
            expect: Rule::TickS,
        },
        Case {
            name: "tick_s nan",
            nproc: 32,
            mutate: |cfg| cfg.collector.tick_s = f64::NAN,
            expect: Rule::TickS,
        },
        Case {
            name: "tick_s infinity",
            nproc: 32,
            mutate: |cfg| cfg.collector.tick_s = f64::INFINITY,
            expect: Rule::TickS,
        },
        Case {
            name: "cpu_window_s equal to tick_s",
            nproc: 32,
            mutate: |cfg| {
                cfg.collector.tick_s = 0.05;
                cfg.collector.cpu_window_s = 0.05;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_window_s upper bound",
            nproc: 32,
            mutate: |cfg| cfg.collector.cpu_window_s = 5.0,
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_window_s below tick_s",
            nproc: 32,
            mutate: |cfg| {
                cfg.collector.tick_s = 1.0;
                cfg.collector.cpu_window_s = (1.0_f64).next_down();
            },
            expect: Rule::CpuWindow,
        },
        Case {
            name: "cpu_window_s above 5",
            nproc: 32,
            mutate: |cfg| cfg.collector.cpu_window_s = (5.0_f64).next_up(),
            expect: Rule::CpuWindow,
        },
        Case {
            name: "cpu_window_s nan",
            nproc: 32,
            mutate: |cfg| cfg.collector.cpu_window_s = f64::NAN,
            expect: Rule::CpuWindow,
        },
        Case {
            name: "cpu_top_k lower bound",
            nproc: 8,
            mutate: |cfg| cfg.collector.cpu_top_k = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_top_k equal to nproc",
            nproc: 8,
            mutate: |cfg| cfg.collector.cpu_top_k = 8,
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_top_k zero",
            nproc: 8,
            mutate: |cfg| cfg.collector.cpu_top_k = 0,
            expect: Rule::CpuTopK,
        },
        Case {
            name: "cpu_top_k above nproc",
            nproc: 8,
            mutate: |cfg| cfg.collector.cpu_top_k = 9,
            expect: Rule::CpuTopK,
        },
        Case {
            name: "cpu_top_k uses the passed nproc",
            nproc: 1,
            mutate: |cfg| cfg.collector.cpu_top_k = 2,
            expect: Rule::CpuTopK,
        },
        Case {
            name: "loopback ipv4 without port",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://127.0.0.1".to_owned(),
            expect: Rule::Ok,
        },
        Case {
            name: "loopback 127.0.0.0/8",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://127.255.255.255:8080".to_owned(),
            expect: Rule::Ok,
        },
        Case {
            name: "loopback ipv6",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://[::1]:8080".to_owned(),
            expect: Rule::Ok,
        },
        Case {
            name: "expanded ipv6 loopback",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://[0:0:0:0:0:0:0:1]:8080".to_owned(),
            expect: Rule::Ok,
        },
        Case {
            name: "https rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "https://127.0.0.1:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "localhost name rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://localhost:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "non-loopback ip rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://8.8.8.8:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "unspecified ip rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://0.0.0.0:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "ipv4-mapped loopback rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://[::ffff:127.0.0.1]:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "uppercase scheme rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "HTTP://127.0.0.1:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "path rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://127.0.0.1:8080/running".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "trailing slash rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://127.0.0.1:8080/".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "userinfo rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://user@127.0.0.1:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "userinfo with password rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://user:pass@127.0.0.1:8080".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "empty url rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url.clear(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "port 0 rejected",
            nproc: 32,
            mutate: |cfg| cfg.llama.url = "http://127.0.0.1:0".to_owned(),
            expect: Rule::LlamaUrl,
        },
        Case {
            name: "running timeout just under its interval",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.running_interval_s = 0.5;
                cfg.llama.running_timeout_s = (0.5_f64).next_down();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "running timeout equal to its interval",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.running_interval_s = 0.5;
                cfg.llama.running_timeout_s = 0.5;
            },
            expect: Rule::Timeout,
        },
        Case {
            name: "running timeout zero",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_timeout_s = 0.0,
            expect: Rule::Timeout,
        },
        Case {
            name: "running timeout negative",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_timeout_s = -1.0,
            expect: Rule::Timeout,
        },
        Case {
            name: "running timeout negative zero",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_timeout_s = -0.0,
            expect: Rule::Timeout,
        },
        Case {
            name: "running timeout nan",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_timeout_s = f64::NAN,
            expect: Rule::Timeout,
        },
        Case {
            name: "metrics timeout equal to its interval",
            nproc: 32,
            mutate: |cfg| cfg.llama.metrics_timeout_s = cfg.llama.metrics_interval_s,
            expect: Rule::Timeout,
        },
        Case {
            name: "slots timeout equal to its interval",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_timeout_s = cfg.llama.slots_interval_s,
            expect: Rule::Timeout,
        },
        Case {
            name: "activity timeout equal to its interval",
            nproc: 32,
            mutate: |cfg| cfg.llama.activity_timeout_s = cfg.llama.activity_interval_s,
            expect: Rule::Timeout,
        },
        Case {
            name: "running interval lower bound",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.running_interval_s = 0.1;
                cfg.llama.running_timeout_s = 0.05;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "running interval upper bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_interval_s = 60.0,
            expect: Rule::Ok,
        },
        Case {
            name: "running interval below range",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.running_interval_s = (0.1_f64).next_down();
                cfg.llama.running_timeout_s = 0.01;
            },
            expect: Rule::PollInterval,
        },
        Case {
            name: "running interval above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.running_interval_s = (60.0_f64).next_up(),
            expect: Rule::PollInterval,
        },
        Case {
            name: "metrics interval below range",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.metrics_interval_s = (0.1_f64).next_down();
                cfg.llama.metrics_timeout_s = 0.01;
            },
            expect: Rule::PollInterval,
        },
        Case {
            name: "metrics interval above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.metrics_interval_s = (60.0_f64).next_up(),
            expect: Rule::PollInterval,
        },
        Case {
            name: "activity interval below range",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.activity_interval_s = (0.1_f64).next_down();
                cfg.llama.activity_timeout_s = 0.01;
            },
            expect: Rule::PollInterval,
        },
        Case {
            name: "activity interval above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.activity_interval_s = (60.0_f64).next_up(),
            expect: Rule::PollInterval,
        },
        Case {
            name: "activity interval nan",
            nproc: 32,
            mutate: |cfg| cfg.llama.activity_interval_s = f64::NAN,
            expect: Rule::PollInterval,
        },
        Case {
            name: "slots_interval_s lower bound",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.slots_interval_s = 0.5;
                cfg.llama.slots_timeout_s = 0.25;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "slots_interval_s upper bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_interval_s = 10.0,
            expect: Rule::Ok,
        },
        Case {
            name: "slots_interval_s below range",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.slots_interval_s = (0.5_f64).next_down();
                cfg.llama.slots_timeout_s = 0.1;
            },
            expect: Rule::SlotsInterval,
        },
        Case {
            name: "slots_interval_s above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_interval_s = (10.0_f64).next_up(),
            expect: Rule::SlotsInterval,
        },
        Case {
            name: "slots_interval_s nan",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_interval_s = f64::NAN,
            expect: Rule::SlotsInterval,
        },
        Case {
            name: "slots_max_bytes zero",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_max_bytes = 0,
            expect: Rule::Ok,
        },
        Case {
            name: "slots_max_bytes 16 MiB",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_max_bytes = 16 * 1024 * 1024,
            expect: Rule::Ok,
        },
        Case {
            name: "slots_max_bytes above 16 MiB",
            nproc: 32,
            mutate: |cfg| cfg.llama.slots_max_bytes = 16 * 1024 * 1024 + 1,
            expect: Rule::SlotsMaxBytes,
        },
        Case {
            name: "input_tail_chars lower bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.input_tail_chars = 256,
            expect: Rule::Ok,
        },
        Case {
            name: "input_tail_chars upper bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.input_tail_chars = 32768,
            expect: Rule::Ok,
        },
        Case {
            name: "input_tail_chars below range",
            nproc: 32,
            mutate: |cfg| cfg.llama.input_tail_chars = 255,
            expect: Rule::InputTail,
        },
        Case {
            name: "input_tail_chars above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.input_tail_chars = 32769,
            expect: Rule::InputTail,
        },
        Case {
            name: "output_tail_chars lower bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.output_tail_chars = 256,
            expect: Rule::Ok,
        },
        Case {
            name: "output_tail_chars upper bound",
            nproc: 32,
            mutate: |cfg| cfg.llama.output_tail_chars = 32768,
            expect: Rule::Ok,
        },
        Case {
            name: "output_tail_chars below range",
            nproc: 32,
            mutate: |cfg| cfg.llama.output_tail_chars = 255,
            expect: Rule::OutputTail,
        },
        Case {
            name: "output_tail_chars above range",
            nproc: 32,
            mutate: |cfg| cfg.llama.output_tail_chars = 32769,
            expect: Rule::OutputTail,
        },
        Case {
            name: "32 aliases accepted",
            nproc: 32,
            mutate: |cfg| {
                cfg.models.aliases = (0..32)
                    .map(|i| (format!("m{i}"), "ok".to_owned()))
                    .collect();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "33 aliases rejected",
            nproc: 32,
            mutate: |cfg| {
                cfg.models.aliases = (0..33)
                    .map(|i| (format!("m{i}"), "ok".to_owned()))
                    .collect();
            },
            expect: Rule::AliasCount,
        },
        Case {
            name: "32 backend overrides accepted",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.backends = (0..32)
                    .map(|i| (format!("m{i}"), llama_core::backend::Backend::SgLang))
                    .collect();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "33 backend overrides rejected",
            nproc: 32,
            mutate: |cfg| {
                cfg.llama.backends = (0..33)
                    .map(|i| (format!("m{i}"), llama_core::backend::Backend::Vllm))
                    .collect();
            },
            expect: Rule::BackendCount,
        },
        Case {
            name: "alias value of 64 chars accepted",
            nproc: 32,
            mutate: |cfg| {
                cfg.models.aliases.insert("id".to_owned(), "a".repeat(64));
            },
            expect: Rule::Ok,
        },
        Case {
            name: "alias value of 65 chars rejected",
            nproc: 32,
            mutate: |cfg| {
                cfg.models.aliases.insert("id".to_owned(), "a".repeat(65));
            },
            expect: Rule::AliasValue,
        },
        Case {
            name: "max_name_chars lower bound",
            nproc: 32,
            mutate: |cfg| cfg.models.max_name_chars = 2,
            expect: Rule::Ok,
        },
        Case {
            name: "max_name_chars upper bound",
            nproc: 32,
            mutate: |cfg| cfg.models.max_name_chars = 12,
            expect: Rule::Ok,
        },
        Case {
            name: "max_name_chars below 2",
            nproc: 32,
            mutate: |cfg| cfg.models.max_name_chars = 1,
            expect: Rule::MaxNameChars,
        },
        Case {
            name: "max_name_chars above 12",
            nproc: 32,
            mutate: |cfg| cfg.models.max_name_chars = 13,
            expect: Rule::MaxNameChars,
        },
        Case {
            name: "fps lower bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.fps = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "fps upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.fps = 20,
            expect: Rule::Ok,
        },
        Case {
            name: "fps zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.fps = 0,
            expect: Rule::Fps,
        },
        Case {
            name: "fps above 20",
            nproc: 32,
            mutate: |cfg| cfg.tty.fps = 21,
            expect: Rule::Fps,
        },
        Case {
            name: "full_redraw_s lower bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.full_redraw_s = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "full_redraw_s zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.full_redraw_s = 0,
            expect: Rule::FullRedraw,
        },
        Case {
            name: "gen ceiling just above zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = (0.0_f64).next_up(),
            expect: Rule::Ok,
        },
        Case {
            name: "gen ceiling upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = 100_000.0,
            expect: Rule::Ok,
        },
        Case {
            name: "gen ceiling zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = 0.0,
            expect: Rule::GenCeiling,
        },
        Case {
            name: "gen ceiling negative",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = -1.0,
            expect: Rule::GenCeiling,
        },
        Case {
            name: "gen ceiling nan",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = f64::NAN,
            expect: Rule::GenCeiling,
        },
        Case {
            name: "gen ceiling infinity",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = f64::INFINITY,
            expect: Rule::GenCeiling,
        },
        Case {
            name: "gen ceiling above the cap",
            nproc: 32,
            mutate: |cfg| cfg.tty.gen_ceiling_tps = (100_000.0_f64).next_up(),
            expect: Rule::GenCeiling,
        },
        Case {
            name: "prompt ceiling just above zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = (0.0_f64).next_up(),
            expect: Rule::Ok,
        },
        Case {
            name: "prompt ceiling upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = 100_000.0,
            expect: Rule::Ok,
        },
        Case {
            name: "prompt ceiling zero",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = 0.0,
            expect: Rule::PromptCeiling,
        },
        Case {
            name: "prompt ceiling negative",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = -1.0,
            expect: Rule::PromptCeiling,
        },
        Case {
            name: "prompt ceiling nan",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = f64::NAN,
            expect: Rule::PromptCeiling,
        },
        Case {
            name: "prompt ceiling infinity",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = f64::INFINITY,
            expect: Rule::PromptCeiling,
        },
        Case {
            name: "prompt ceiling above the cap",
            nproc: 32,
            mutate: |cfg| cfg.tty.prompt_ceiling_tps = (100_000.0_f64).next_up(),
            expect: Rule::PromptCeiling,
        },
        Case {
            name: "chart_bucket_s lower bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.chart_bucket_s = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "chart_bucket_s upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.chart_bucket_s = 60,
            expect: Rule::Ok,
        },
        Case {
            name: "chart_bucket_s below range",
            nproc: 32,
            mutate: |cfg| cfg.tty.chart_bucket_s = 0,
            expect: Rule::ChartBucket,
        },
        Case {
            name: "chart_bucket_s above range",
            nproc: 32,
            mutate: |cfg| cfg.tty.chart_bucket_s = 61,
            expect: Rule::ChartBucket,
        },
        Case {
            name: "ctx_history_h lower bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.ctx_history_h = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "ctx_history_h upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.ctx_history_h = 24,
            expect: Rule::Ok,
        },
        Case {
            name: "blank_min off (default)",
            nproc: 32,
            mutate: |cfg| cfg.tty.blank_min = 0,
            expect: Rule::Ok,
        },
        Case {
            name: "blank_min lower on bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.blank_min = 1,
            expect: Rule::Ok,
        },
        Case {
            name: "blank_min upper bound",
            nproc: 32,
            mutate: |cfg| cfg.tty.blank_min = 60,
            expect: Rule::Ok,
        },
        Case {
            name: "blank_min above 60",
            nproc: 32,
            mutate: |cfg| cfg.tty.blank_min = 61,
            expect: Rule::BlankMin,
        },
        Case {
            name: "blank_min far above 60",
            nproc: 32,
            mutate: |cfg| cfg.tty.blank_min = u32::MAX,
            expect: Rule::BlankMin,
        },
        Case {
            name: "sleep_min off with blank on",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 0;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "sleep_min above blank_min (10/15 example)",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 15;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "sleep_min one above blank_min",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 11;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "sleep_min upper bound",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 59;
                cfg.tty.sleep_min = 60;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "sleep_min above 60",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 61;
            },
            expect: Rule::SleepMin,
        },
        Case {
            name: "sleep_min equal to blank_min",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 10;
            },
            expect: Rule::SleepMin,
        },
        Case {
            name: "sleep_min below blank_min",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 10;
                cfg.tty.sleep_min = 5;
            },
            expect: Rule::SleepMin,
        },
        Case {
            name: "sleep_min without blank_min",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 0;
                cfg.tty.sleep_min = 15;
            },
            expect: Rule::SleepMin,
        },
        Case {
            name: "tty size unset (default)",
            nproc: 32,
            mutate: |cfg| cfg.tty.size = None,
            expect: Rule::Ok,
        },
        Case {
            name: "tty size at the layout floor",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 160,
                    rows: 26,
                })
            },
            expect: Rule::Ok,
        },
        Case {
            name: "tty size at the upper bound",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 1024,
                    rows: 512,
                })
            },
            expect: Rule::Ok,
        },
        Case {
            name: "tty size one column short of the floor",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 159,
                    rows: 49,
                })
            },
            expect: Rule::TtySize,
        },
        Case {
            name: "tty size one row short of the floor",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 160,
                    rows: 25,
                })
            },
            expect: Rule::TtySize,
        },
        Case {
            name: "tty size too wide",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 1025,
                    rows: 49,
                })
            },
            expect: Rule::TtySize,
        },
        Case {
            name: "tty size too tall",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.size = Some(TtySize {
                    cols: 160,
                    rows: 513,
                })
            },
            expect: Rule::TtySize,
        },
        Case {
            name: "blank_min 60 leaves no room for sleep",
            nproc: 32,
            mutate: |cfg| {
                cfg.tty.blank_min = 60;
                cfg.tty.sleep_min = 60;
            },
            expect: Rule::SleepMin,
        },
        Case {
            name: "ctx_history_h below range",
            nproc: 32,
            mutate: |cfg| cfg.tty.ctx_history_h = 0,
            expect: Rule::CtxHistory,
        },
        Case {
            name: "ctx_history_h above range",
            nproc: 32,
            mutate: |cfg| cfg.tty.ctx_history_h = 25,
            expect: Rule::CtxHistory,
        },
        Case {
            name: "smooth_s zero",
            nproc: 32,
            mutate: |cfg| cfg.load.smooth_s = 0.0,
            expect: Rule::Ok,
        },
        Case {
            name: "smooth_s upper bound",
            nproc: 32,
            mutate: |cfg| cfg.load.smooth_s = 5.0,
            expect: Rule::Ok,
        },
        Case {
            name: "smooth_s below zero",
            nproc: 32,
            mutate: |cfg| cfg.load.smooth_s = -0.000_001,
            expect: Rule::SmoothS,
        },
        Case {
            name: "smooth_s above 5",
            nproc: 32,
            mutate: |cfg| cfg.load.smooth_s = (5.0_f64).next_up(),
            expect: Rule::SmoothS,
        },
        Case {
            name: "smooth_s nan",
            nproc: 32,
            mutate: |cfg| cfg.load.smooth_s = f64::NAN,
            expect: Rule::SmoothS,
        },
        Case {
            name: "nominal_frac lower bound",
            nproc: 32,
            mutate: |cfg| cfg.load.nominal_frac = 0.5,
            expect: Rule::Ok,
        },
        Case {
            name: "nominal_frac upper bound",
            nproc: 32,
            mutate: |cfg| cfg.load.nominal_frac = 1.0,
            expect: Rule::Ok,
        },
        Case {
            name: "nominal_frac below 0.5",
            nproc: 32,
            mutate: |cfg| cfg.load.nominal_frac = (0.5_f64).next_down(),
            expect: Rule::NominalFrac,
        },
        Case {
            name: "nominal_frac above 1",
            nproc: 32,
            mutate: |cfg| cfg.load.nominal_frac = (1.0_f64).next_up(),
            expect: Rule::NominalFrac,
        },
        Case {
            name: "nominal_frac nan",
            nproc: 32,
            mutate: |cfg| cfg.load.nominal_frac = f64::NAN,
            expect: Rule::NominalFrac,
        },
        Case {
            name: "cpu_idle_w at the nominal ceiling",
            nproc: 32,
            mutate: |cfg| {
                cfg.load.cpu_limit_w = 100.0;
                cfg.load.nominal_frac = 0.8;
                cfg.load.cpu_idle_w = 80.0;
            },
            expect: Rule::IdleWatts,
        },
        Case {
            name: "cpu_limit_w just above zero",
            nproc: 32,
            mutate: |cfg| {
                cfg.load.cpu_limit_w = (0.0_f64).next_up();
                cfg.load.cpu_idle_w = 0.0;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_limit_w zero",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_limit_w = 0.0,
            expect: Rule::CpuLimit,
        },
        Case {
            name: "cpu_limit_w negative",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_limit_w = -1.0,
            expect: Rule::CpuLimit,
        },
        Case {
            name: "cpu_limit_w nan",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_limit_w = f64::NAN,
            expect: Rule::CpuLimit,
        },
        Case {
            name: "idle floors zero and fixed",
            nproc: 32,
            mutate: |cfg| {
                cfg.load.idle = llama_watch::config::IdleMode::Fixed;
                cfg.load.gpu_idle_w = 0.0;
                cfg.load.cpu_idle_w = 0.0;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "gpu_idle_w negative",
            nproc: 32,
            mutate: |cfg| cfg.load.gpu_idle_w = -0.1,
            expect: Rule::IdleWatts,
        },
        Case {
            name: "gpu_idle_w nan",
            nproc: 32,
            mutate: |cfg| cfg.load.gpu_idle_w = f64::NAN,
            expect: Rule::IdleWatts,
        },
        Case {
            name: "cpu_idle_w negative",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_idle_w = -0.1,
            expect: Rule::IdleWatts,
        },
        Case {
            name: "cpu_idle_w infinite",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_idle_w = f64::INFINITY,
            expect: Rule::IdleWatts,
        },
        Case {
            name: "cpu_idle_w just below the nominal cpu ceiling",
            nproc: 32,
            mutate: |cfg| {
                cfg.load.cpu_idle_w = (cfg.load.nominal_frac * cfg.load.cpu_limit_w).next_down();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "cpu_idle_w at cpu_limit_w",
            nproc: 32,
            mutate: |cfg| cfg.load.cpu_idle_w = cfg.load.cpu_limit_w,
            expect: Rule::IdleWatts,
        },
        Case {
            name: "fans reference block",
            nproc: 32,
            mutate: |cfg| cfg.fans = ref_fans(),
            expect: Rule::Ok,
        },
        Case {
            name: "fans disabled with no hwmon",
            nproc: 32,
            mutate: |cfg| cfg.fans.channels = vec![2],
            expect: Rule::Ok,
        },
        Case {
            name: "fans channel 16 and eight fans",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels = vec![1, 2, 3, 4, 5, 6, 7, 16];
                cfg.fans.labels = None;
            },
            expect: Rule::Ok,
        },
        Case {
            name: "fans channel 0",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels[0] = 0;
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans channel 17",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels[0] = 17;
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans nine channels",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels = (1..=9).collect();
                cfg.fans.labels = None;
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans repeated channel",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels = vec![2, 2, 5, 6];
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans label count differs",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.labels = Some(vec!["a".to_owned()]);
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans label of 10 chars",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.labels.as_mut().unwrap()[0] = "abcdefghij".to_owned();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "fans label of 11 chars",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.labels.as_mut().unwrap()[0] = "abcdefghijk".to_owned();
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans enabled without hwmon",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.hwmon.clear();
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans hwmon is a path",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.hwmon = "../hwmon8".to_owned();
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans enabled without channels",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.channels.clear();
                cfg.fans.labels = None;
            },
            expect: Rule::Fans,
        },
        // #74: discovery, allow/block and the shorthand conflict.
        Case {
            name: "fans enabled with nothing else discovers every fan",
            nproc: 32,
            mutate: |cfg| cfg.fans.enabled = true,
            expect: Rule::Ok,
        },
        Case {
            name: "fans allow and block",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans.enabled = true;
                cfg.fans.allow = vec!["nct6798:fan?".into(), "z53".into()];
                cfg.fans.block = vec!["nct6798:fan7".into()];
                cfg.fans.rename = [("z53:fan1".to_owned(), "pump".to_owned())].into();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "fans shorthand and allow together",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.allow = vec!["z53".into()];
            },
            expect: Rule::Fans,
        },
        Case {
            name: "fans shorthand with block and rename",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans = ref_fans();
                cfg.fans.block = vec!["nct6798:fan6".into()];
                cfg.fans.rename = [("nct6798:fan2".to_owned(), "rad top".to_owned())].into();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "fans invalid glob",
            nproc: 32,
            mutate: |cfg| cfg.fans.block = vec!["nct[0-9]".into()],
            expect: Rule::Fans,
        },
        Case {
            name: "fans rename of 11 chars",
            nproc: 32,
            mutate: |cfg| {
                cfg.fans.rename = [("z53:fan1".to_owned(), "abcdefghijk".to_owned())].into();
            },
            expect: Rule::Fans,
        },
        Case {
            name: "temps allow, block, rename, warn and crit",
            nproc: 32,
            mutate: |cfg| {
                cfg.temps.allow = vec!["k10temp".into(), "nvme*".into(), "nct6798:SYSTIN".into()];
                cfg.temps.block = vec!["nvme*:Sensor 2".into()];
                cfg.temps.rename = [("nct6798:SYSTIN".to_owned(), "board".to_owned())].into();
                cfg.temps.warn = [("z53".to_owned(), 45.0)].into();
                cfg.temps.crit = [("z53".to_owned(), 55.0)].into();
            },
            expect: Rule::Ok,
        },
        Case {
            name: "temps glob with two colons",
            nproc: 32,
            mutate: |cfg| cfg.temps.allow = vec!["a:b:c".into()],
            expect: Rule::Temps,
        },
        Case {
            name: "temps empty sensor side",
            nproc: 32,
            mutate: |cfg| cfg.temps.block = vec!["nct6798:".into()],
            expect: Rule::Temps,
        },
        Case {
            name: "temps 33 patterns",
            nproc: 32,
            mutate: |cfg| cfg.temps.block = (0..33).map(|n| format!("chip{n}")).collect(),
            expect: Rule::Temps,
        },
        Case {
            name: "temps rename key with a wildcard",
            nproc: 32,
            mutate: |cfg| {
                cfg.temps.rename = [("nvme*".to_owned(), "disk".to_owned())].into();
            },
            expect: Rule::Temps,
        },
        Case {
            name: "temps rename of 13 chars",
            nproc: 32,
            mutate: |cfg| {
                cfg.temps.rename = [("z53".to_owned(), "abcdefghijklm".to_owned())].into();
            },
            expect: Rule::Temps,
        },
        Case {
            name: "temps warn not below crit",
            nproc: 32,
            mutate: |cfg| {
                cfg.temps.warn = [("z53".to_owned(), 55.0)].into();
                cfg.temps.crit = [("z53".to_owned(), 55.0)].into();
            },
            expect: Rule::Temps,
        },
        Case {
            name: "temps crit above 150",
            nproc: 32,
            mutate: |cfg| cfg.temps.crit = [("gpu".to_owned(), 151.0)].into(),
            expect: Rule::Temps,
        },
        Case {
            name: "temps warn not finite",
            nproc: 32,
            mutate: |cfg| cfg.temps.warn = [("gpu".to_owned(), f64::NAN)].into(),
            expect: Rule::Temps,
        },
    ]
}

fn ref_fans() -> llama_watch::config::Fans {
    llama_watch::config::Fans {
        enabled: true,
        hwmon: "nct6798".to_owned(),
        channels: vec![2, 3, 5, 6],
        labels: Some(
            ["front1", "front2", "rear", "top"]
                .map(str::to_owned)
                .to_vec(),
        ),
        ..llama_watch::config::Fans::default()
    }
}

#[test]
fn fans_default_off_and_parse_the_reference_block() {
    let cfg = Config::from_toml("").expect("empty");
    assert!(!cfg.fans.enabled);
    assert!(cfg.fans.channels.is_empty());
    let cfg = Config::from_toml(
        "[fans]\nenabled = true\nhwmon = \"nct6798\"\nchannels = [2, 3, 5, 6]\n\
         labels = [\"front1\", \"front2\", \"rear\", \"top\"]\n",
    )
    .expect("fans block");
    assert_eq!(cfg.fans, ref_fans());
    assert!(cfg.validate(32).is_ok());
    let unlabelled =
        Config::from_toml("[fans]\nenabled = true\nhwmon = \"nct6798\"\nchannels = [2, 3]\n")
            .expect("no labels");
    assert_eq!(unlabelled.fans.resolved_labels(), ["fan2", "fan3"]);
    assert!(Config::from_toml("[fans]\nhwmon_number = 8\n").is_err());
}

#[test]
fn validate_accepts_one_and_rejects_one_per_rule() {
    let mut missed = Vec::new();
    for case in cases() {
        let mut cfg = Config::default();
        (case.mutate)(&mut cfg);
        let got = rule_of(cfg.validate(case.nproc));
        if got != case.expect {
            missed.push(format!(
                "{}: expected {:?}, got {got:?}",
                case.name, case.expect
            ));
        }
    }
    assert!(missed.is_empty(), "\n{}", missed.join("\n"));
}

#[test]
fn show_text_defaults_on_and_can_be_turned_off() {
    assert!(Config::default().tty.show_text);
    let cfg = Config::from_toml("[tty]\nshow_text = false\n").expect("show_text false");
    assert!(!cfg.tty.show_text);
    assert!(cfg.validate(8).is_ok());
    let cfg = Config::from_toml("[tty]\nshow_text = true\n").expect("show_text true");
    assert!(cfg.tty.show_text);
    for bad in ["\"no\"", "0", "\"false\""] {
        let text = format!("[tty]\nshow_text = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "accepted {bad}");
    }

    let text = std::fs::read_to_string(packaging("watch.example.toml")).expect("example");
    assert!(text.contains("show_text = true"), "{text}");
    assert!(text.contains("--no-text"), "{text}");
}

#[test]
fn text_off_override_clears_show_text_and_keeps_the_rest() {
    let path = packaging("watch.example.toml");
    let cfg = Config::load_validated(&path, 8).expect("example validates");
    assert!(cfg.tty.show_text);
    let off = cfg.clone().with_text_off();
    assert!(!off.tty.show_text);
    let mut back = (*off).clone();
    back.tty.show_text = true;
    assert_eq!(back, *cfg);
}

/// #7: `[tty] size = "COLSxROWS"` and `font = "12x24" | "12x22" | "10x18"` (#73).
#[test]
fn tty_size_and_font_parse_strictly() {
    let cfg = Config::default();
    assert_eq!(cfg.tty.size, None);
    assert_eq!(cfg.tty.font, TtyFont::Hack12x24);
    let cfg = Config::from_toml("[tty]\nsize = \"160x49\"\nfont = \"12x22\"\n").expect("config");
    assert_eq!(
        cfg.tty.size,
        Some(TtySize {
            cols: 160,
            rows: 49
        })
    );
    assert_eq!(cfg.tty.font, TtyFont::Hack12x22);
    assert!(cfg.validate(8).is_ok());
    let cfg = Config::from_toml("[tty]\nfont = \"12x24\"\n").expect("12x24");
    assert_eq!(cfg.tty.font, TtyFont::Hack12x24);
    assert_eq!(TtyFont::Hack12x24.file_name(), "llama-hack-12x24.psfu");
    assert_eq!(TtyFont::Hack12x22.file_name(), "llama-hack-12x22.psfu");
    let cfg = Config::from_toml("[tty]\nsize = \"192x60\"\nfont = \"10x18\"\n").expect("10x18");
    assert_eq!(cfg.tty.font, TtyFont::Hack10x18);
    assert!(cfg.validate(8).is_ok());
    assert_eq!(TtyFont::Hack10x18.file_name(), "llama-hack-10x18.psfu");
    for bad in [
        "\"160X49\"",
        "\"160x\"",
        "\"x49\"",
        "\"160 x 49\"",
        "\" 160x49\"",
        "\"160x49 \"",
        "\"+160x49\"",
        "\"-160x49\"",
        "\"0160x49\"",
        "\"160x49x2\"",
        "\"160x49; rm -rf /\"",
        "\"65536x49\"",
        "\"１６０x49\"",
        "\"\"",
        "160",
        "[160, 49]",
    ] {
        let text = format!("[tty]\nsize = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "size accepted {bad}");
    }
    for bad in [
        "\"8x16\"",
        "\"12X22\"",
        "\"hack\"",
        "\"/tmp/evil.psfu\"",
        "22",
        "\"\"",
    ] {
        let text = format!("[tty]\nfont = {bad}\n");
        assert!(Config::from_toml(&text).is_err(), "font accepted {bad}");
    }
    let err = Config::from_toml("[tty]\nsize = \"120x30\"\n")
        .expect("parses")
        .validate(8)
        .expect_err("below the layout floor");
    assert!(err.to_string().contains("120x30"), "{err}");
}

#[test]
fn console_blank_defaults_off_and_parses() {
    let cfg = Config::default();
    assert_eq!((cfg.tty.blank_min, cfg.tty.sleep_min), (0, 0));
    let cfg = Config::from_toml("[tty]\nblank_min = 10\nsleep_min = 15\n").expect("10/15 example");
    assert_eq!((cfg.tty.blank_min, cfg.tty.sleep_min), (10, 15));
    assert!(cfg.validate(32).is_ok());
    assert!(Config::from_toml("[tty]\nblank_min = -1\n").is_err());
    assert!(Config::from_toml("[tty]\npowerdown_min = 15\n").is_err());
}

#[test]
fn backend_overrides_parse_known_words_only() {
    use llama_core::backend::Backend;
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("t72-backends");
    std::fs::create_dir_all(&dir).expect("dir");
    let good = dir.join("good.toml");
    std::fs::write(
        &good,
        "[llama.backends]\n\"flash\" = \"sglang\"\n\"big\" = \"vllm\"\n\"tabby\" = \"openai\"\n\"q\" = \"llamacpp\"\n\"not-loaded\" = \"sglang\"\n\"next\" = \"strata\"\n",
    )
    .expect("write");
    let cfg = Config::load_validated(&good, 8).expect("valid overrides");
    assert_eq!(cfg.llama.backends.get("flash"), Some(&Backend::SgLang));
    assert_eq!(cfg.llama.backends.get("big"), Some(&Backend::Vllm));
    assert_eq!(cfg.llama.backends.get("tabby"), Some(&Backend::OpenAi));
    assert_eq!(cfg.llama.backends.get("q"), Some(&Backend::LlamaCpp));
    assert_eq!(cfg.llama.backends.get("next"), Some(&Backend::Strata));
    for bad in ["\"tabbyapi\"", "\"SGLang\"", "\"Strata\"", "1", "\"\""] {
        let path = dir.join("bad.toml");
        std::fs::write(&path, format!("[llama.backends]\n\"m\" = {bad}\n")).expect("write");
        assert!(
            matches!(
                Config::load_validated(&path, 8),
                Err(ConfigError::Parse { .. })
            ),
            "{bad} must not parse"
        );
    }
}
