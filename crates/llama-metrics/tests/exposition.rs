//! Exposition format: goldens, stale handling, label sanitisation, and the
//! no-prompt-text assertion.

mod common;

use std::collections::BTreeSet;
use std::time::Duration;

use llama_core::wire::{self, WireSnapshot};
use llama_metrics::expo::{self, Rejected, Scrape, escape_label_value};
use llama_metrics::http::Body;
use llama_metrics::service::SnapshotMetrics;
use llama_metrics::snapshot::{ReadError, SnapshotFile};

const T0: u64 = 1_000_000_000_000;
const STALE: Duration = Duration::from_secs(5);

fn load(name: &str) -> WireSnapshot {
    SnapshotFile::at(common::fixture(name))
        .read()
        .expect("fixture parses and validates")
}

fn render(read: &Result<WireSnapshot, ReadError>, now_ns: u64) -> String {
    expo::render(&Scrape {
        read,
        now_ns,
        stale_after: STALE,
        rejected: Rejected { denied: 3, busy: 1 },
    })
}

fn golden(name: &str, got: &str) {
    let path = common::fixture(name);
    if std::env::var_os("LLAMA_METRICS_BLESS").is_some() {
        std::fs::write(&path, got).expect("bless golden");
    }
    let want = std::fs::read_to_string(&path).expect("golden");
    assert_eq!(
        got, want,
        "golden {name} differs; LLAMA_METRICS_BLESS=1 rewrites it"
    );
}

/// Minimal 0.0.4 parser: every line is a comment (`# HELP` / `# TYPE`) or
/// `name{labels} value`. Returns `(name, labels, value)`.
type Sample = (String, Vec<(String, String)>, String);

fn samples(text: &str) -> Vec<Sample> {
    let mut out = Vec::new();
    let mut typed: BTreeSet<String> = BTreeSet::new();
    assert!(text.ends_with('\n'), "exposition must end with a newline");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').expect("TYPE line");
            assert!(
                ["gauge", "counter"].contains(&kind),
                "unexpected type {kind}"
            );
            assert!(typed.insert(name.to_owned()), "{name} typed twice");
            continue;
        }
        if line.starts_with("# HELP ") {
            continue;
        }
        assert!(!line.starts_with('#'), "stray comment {line:?}");
        let (series, value) = line.rsplit_once(' ').expect("sample line");
        value
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("value {value:?} is not a float"));
        let (name, labels) = match series.split_once('{') {
            Some((name, rest)) => (name, parse_labels(rest.strip_suffix('}').expect("}"))),
            None => (series, Vec::new()),
        };
        assert!(name.starts_with("llamabored_"), "{name} lacks the prefix");
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "{name} is not a metric name"
        );
        assert!(typed.contains(name), "{name} sampled before its TYPE");
        out.push((name.to_owned(), labels, value.to_owned()));
    }
    out
}

fn parse_labels(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while chars.peek().is_some() {
        let key: String = chars.by_ref().take_while(|c| *c != '=').collect();
        assert_eq!(chars.next(), Some('"'), "label {key} value is not quoted");
        let mut value = String::new();
        loop {
            match chars.next().expect("unterminated label value") {
                '\\' => match chars.next().expect("escape") {
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    'n' => value.push('\n'),
                    other => panic!("bad escape \\{other}"),
                },
                '"' => break,
                '\n' => panic!("raw newline in a label value"),
                c => value.push(c),
            }
        }
        out.push((key, value));
        if chars.peek() == Some(&',') {
            chars.next();
        }
    }
    out
}

#[test]
fn loaded_snapshot_matches_the_golden() {
    let read = Ok(load("snapshot-loaded.json"));
    let text = render(&read, T0 + 250_000_000);
    golden("loaded.prom", &text);
    let all = samples(&text);
    // Every series is unique, or Prometheus rejects the whole scrape.
    let keys: BTreeSet<String> = all.iter().map(|(n, l, _)| format!("{n}{l:?}")).collect();
    assert_eq!(keys.len(), all.len(), "duplicate series");
}

#[test]
fn down_snapshot_matches_the_golden() {
    let read = Ok(load("snapshot-down.json"));
    let text = render(&read, T0 + 1_000_000_000);
    golden("down.prom", &text);
    samples(&text);
    assert!(
        !text.contains("llamabored_gpu_pct"),
        "absent host field was exported"
    );
    assert!(!text.contains("llamabored_tokens_decoded_total"));
    assert!(!text.contains("llamabored_model_loaded"));
    assert!(text.contains("llamabored_ai_state{state=\"down\"} 1\n"));
}

#[test]
fn stale_snapshot_keeps_seq_and_age_and_drops_values() {
    let read = Ok(load("snapshot-loaded.json"));
    // Exactly the limit is fresh; one nanosecond more is stale.
    let edge = render(&read, T0 + 5_000_000_000);
    assert!(edge.contains("llamabored_snapshot_stale 0\n"), "{edge}");
    assert!(edge.contains("llamabored_gpu_pct 97\n"));

    let text = render(&read, T0 + 5_000_000_001);
    golden("stale.prom", &text);
    samples(&text);
    assert!(text.contains("llamabored_snapshot_up 1\n"));
    assert!(text.contains("llamabored_snapshot_stale 1\n"));
    assert!(text.contains("llamabored_snapshot_seq 4242\n"));
    assert!(text.contains("llamabored_snapshot_age_seconds 5.000\n"));
    for gone in [
        "llamabored_gpu_pct",
        "llamabored_activity_pct",
        "llamabored_coolant_celsius",
        "llamabored_tokens_decoded_total",
        "llamabored_ai_state",
        "llamabored_model_loaded",
        "llamabored_model_ctx_size_tokens",
    ] {
        assert!(!text.contains(gone), "stale scrape still exports {gone}");
    }
}

#[test]
fn a_snapshot_from_the_future_is_stale() {
    let read = Ok(load("snapshot-loaded.json"));
    let text = render(&read, T0 - 1_000_000_000);
    assert!(text.contains("llamabored_snapshot_stale 1\n"), "{text}");
    assert!(text.contains("llamabored_snapshot_age_seconds 0.000\n"));
    // Within the 50 ms slack it still counts.
    let slack = render(&read, T0 - 40_000_000);
    assert!(slack.contains("llamabored_snapshot_stale 0\n"), "{slack}");
}

#[test]
fn missing_snapshot_is_down_and_stale() {
    for err in [
        ReadError::Missing,
        ReadError::Invalid,
        ReadError::Rejected(wire::WireError::Schema),
        ReadError::TooLarge,
        ReadError::NotRegular,
    ] {
        let text = render(&Err(err), T0);
        samples(&text);
        assert!(text.contains("llamabored_snapshot_up 0\n"));
        assert!(text.contains("llamabored_snapshot_stale 1\n"));
        assert!(!text.contains("llamabored_snapshot_seq"));
        assert!(text.contains("llamabored_exporter_build_info{version=\""));
        assert!(
            text.contains("llamabored_exporter_rejected_connections_total{reason=\"denied\"} 3\n")
        );
    }
}

#[test]
fn file_reader_refuses_symlinks_oversize_and_bad_json() {
    let dir = common::scratch("reader");
    let good = dir.join("good.json");
    std::fs::copy(common::fixture("snapshot-loaded.json"), &good).unwrap();
    assert!(SnapshotFile::at(&good).read().is_ok());

    let link = dir.join("link.json");
    std::os::unix::fs::symlink(&good, &link).unwrap();
    assert_eq!(SnapshotFile::at(&link).read(), Err(ReadError::NotRegular));

    assert_eq!(
        SnapshotFile::at(dir.join("absent.json")).read(),
        Err(ReadError::Missing)
    );
    assert_eq!(SnapshotFile::at(&dir).read(), Err(ReadError::NotRegular));

    let big = dir.join("big.json");
    std::fs::write(&big, vec![b' '; wire::MAX_BYTES + 1]).unwrap();
    assert_eq!(SnapshotFile::at(&big).read(), Err(ReadError::TooLarge));

    let bad = dir.join("bad.json");
    std::fs::write(&bad, b"{\"schema\":1}").unwrap();
    assert_eq!(
        SnapshotFile::at(&bad).read(),
        Err(ReadError::Rejected(wire::WireError::Parse))
    );

    // A validated-parser rule, not only JSON: out-of-range percent.
    let text = std::fs::read_to_string(&good)
        .unwrap()
        .replace("\"gpu_pct\": 97", "\"gpu_pct\": 170");
    std::fs::write(&bad, text).unwrap();
    assert_eq!(
        SnapshotFile::at(&bad).read(),
        Err(ReadError::Rejected(wire::WireError::OutOfRange {
            field: "gpu_pct"
        }))
    );
}

#[test]
fn body_reads_the_file_on_every_scrape() {
    let dir = common::scratch("body");
    let path = dir.join("snapshot.json");
    let metrics = SnapshotMetrics::new(
        SnapshotFile::at(&path),
        STALE,
        Box::new(|| T0 + 1_000_000_000),
    );
    assert!(
        metrics
            .metrics(Rejected::default())
            .contains("llamabored_snapshot_up 0\n")
    );
    std::fs::copy(common::fixture("snapshot-loaded.json"), &path).unwrap();
    let text = metrics.metrics(Rejected::default());
    assert!(text.contains("llamabored_snapshot_up 1\n"));
    assert!(text.contains("llamabored_snapshot_stale 0\n"));
}

/// Log lines captured from the exporter.
#[derive(Clone, Default)]
struct Lines(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl llama_core::log::Sink for Lines {
    fn write_line(&mut self, line: &str) {
        self.0.lock().unwrap().push(line.to_owned());
    }
}

impl Lines {
    fn all(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

#[test]
fn a_rejected_snapshot_logs_once_per_reason_until_a_good_read() {
    let dir = common::scratch("reject-log");
    let path = dir.join("snapshot.json");
    let lines = Lines::default();
    let metrics = SnapshotMetrics::new(
        SnapshotFile::at(&path),
        STALE,
        Box::new(|| T0 + 1_000_000_000),
    )
    .with_log(Box::new(lines.clone()));
    let good = std::fs::read_to_string(common::fixture("snapshot-loaded.json")).unwrap();
    let scrape = |times: usize| {
        for _ in 0..times {
            let _ = metrics.metrics(Rejected::default());
        }
    };

    std::fs::write(&path, good.replace("\"schema\": 1", "\"schema\": 2")).unwrap();
    scrape(3);
    assert_eq!(
        lines.all(),
        ["<4>snapshot rejected: snapshot schema is not 1"]
    );

    // A different reason logs again, once.
    std::fs::write(&path, good.replace("\"gpu_pct\": 97", "\"gpu_pct\": 170")).unwrap();
    scrape(3);
    assert_eq!(lines.all().len(), 2, "{:?}", lines.all());
    assert_eq!(
        lines.all()[1],
        "<4>snapshot rejected: snapshot field gpu_pct is out of range"
    );

    // A good read resets it: the same failure later is logged again.
    std::fs::write(&path, &good).unwrap();
    scrape(2);
    std::fs::write(&path, good.replace("\"gpu_pct\": 97", "\"gpu_pct\": 170")).unwrap();
    scrape(2);
    let all = lines.all();
    assert_eq!(all.len(), 4, "{all:?}");
    assert_eq!(all[2], "<6>snapshot accepted");
    assert_eq!(all[3], all[1]);

    std::fs::remove_file(&path).unwrap();
    scrape(2);
    assert_eq!(lines.all()[4], "<4>snapshot rejected: missing");
    assert_eq!(lines.all().len(), 5);
}

#[test]
fn label_values_are_escaped_and_bounded() {
    assert_eq!(escape_label_value("plain"), "plain");
    assert_eq!(escape_label_value("a\"b"), "a\\\"b");
    assert_eq!(escape_label_value("a\\b"), "a\\\\b");
    assert_eq!(escape_label_value("a\nb"), "a\\nb");
    assert_eq!(
        escape_label_value("a\rb\tc\u{7}d\u{7f}e\u{85}f"),
        "a?b?c?d?e?f"
    );
    assert_eq!(escape_label_value("Qwen3\u{2026}"), "Qwen3\u{2026}");
    let long = "x".repeat(500);
    assert_eq!(
        escape_label_value(&long).chars().count(),
        expo::MAX_LABEL_CHARS
    );
    // Escaping never lets a value end the quoted string early.
    let evil = "\"} llamabored_evil 1\n# TYPE x gauge";
    let escaped = escape_label_value(evil);
    assert!(!escaped.contains('\n'));
    assert!(escaped.starts_with("\\\""));
}

#[test]
fn model_labels_round_trip_through_the_parser() {
    let read = Ok(load("snapshot-loaded.json"));
    let text = render(&read, T0);
    let models: Vec<Vec<(String, String)>> = samples(&text)
        .into_iter()
        .filter(|(n, _, _)| n == "llamabored_model_loaded")
        .map(|(_, l, _)| l)
        .collect();
    assert_eq!(models.len(), 4);
    let qwen = models
        .iter()
        .find(|l| l[0].1 == "Qwen3-Coder\u{2026}")
        .expect("qwen");
    let keys: Vec<&str> = qwen.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        ["name", "full_name", "quant", "kv", "ctx", "moe", "backend"]
    );
    assert_eq!(qwen[6].1, "llamacpp", "no backend on the wire is llama.cpp");
    assert_eq!(qwen[1].1, "Qwen3-Coder-30B \"fast\" \\ build");
    assert_eq!(qwen[2].1, "UD-Q4_K_M");
    assert_eq!(qwen[3].1, "q8_0");
    assert_eq!(qwen[4].1, "262144");
    assert_eq!(qwen[5].1, "16");
    let bonsai = models
        .iter()
        .find(|l| l[0].1 == "Bonsai 8B")
        .expect("bonsai");
    assert_eq!(
        bonsai[1].1, "Bonsai 8B",
        "absent full_name repeats the name"
    );
    assert_eq!(bonsai[3].1, "q8_0/q4_0");
    assert_eq!(bonsai[5].1, "all");
    assert_eq!(bonsai[6].1, "sglang");
    let tiny = models.iter().find(|l| l[0].1 == "tiny").expect("tiny");
    assert!(tiny[2..6].iter().all(|(_, v)| v.is_empty()), "{tiny:?}");
    // #31: a vLLM model's KV dtype from cache_config_info is its kv label.
    let vllm = models
        .iter()
        .find(|l| l[0].1 == "qwen3.8-27b")
        .expect("vllm");
    assert_eq!(vllm[3].1, "fp8_e4m3");
    assert_eq!(vllm[6].1, "vllm");
}

#[test]
fn strata_is_one_more_backend_label_value() {
    let mut snap = load("snapshot-loaded.json");
    snap.ai.models[2].backend = Some(wire::Backend::Strata);
    snap.ai.models[2].running = Some(1);
    snap.ai.models[2].queued = Some(0);
    let text = render(&Ok(snap), T0);
    let all = samples(&text);
    let tiny = all
        .iter()
        .find(|(n, l, _)| n == "llamabored_model_loaded" && l[0].1 == "tiny")
        .expect("tiny");
    assert_eq!(tiny.1[6], ("backend".to_owned(), "strata".to_owned()));
    let running = all
        .iter()
        .find(|(n, l, _)| n == "llamabored_model_requests_running" && l[0].1 == "tiny")
        .expect("running");
    assert_eq!(running.2, "1");
    assert!(
        !all.iter()
            .any(|(n, l, _)| n == "llamabored_model_kv_cache_usage_ratio" && l[0].1 == "tiny"),
        "Strata has no KV fill"
    );
}

#[test]
fn duplicate_models_export_one_series() {
    let mut snap = load("snapshot-loaded.json");
    let first = snap.ai.models[0].clone();
    snap.ai.models.push(first);
    let text = render(&Ok(snap), T0);
    let loaded = text
        .lines()
        .filter(|l| l.starts_with("llamabored_model_loaded{"))
        .count();
    assert_eq!(loaded, 4);
    let ctx = text
        .lines()
        .filter(|l| l.starts_with("llamabored_model_ctx_size_tokens{"))
        .count();
    assert_eq!(ctx, 2);
}

/// The wire carries no prompt or output text, and the exporter exports none.
///
/// Two halves. (1) The snapshot's JSON keys are pinned: a new wire field
/// (such as a prompt tail) fails here and forces a review of this exporter.
/// (2) Every exported string is a model name or an allowlisted token from
/// the snapshot, or one of the exporter's fixed label values.
#[test]
fn no_prompt_or_output_text_is_exported() {
    let snap = load("snapshot-loaded.json");
    let json: serde_json::Value =
        serde_json::from_slice(&wire::to_json(&snap).expect("encode")).expect("json");
    let mut keys = BTreeSet::new();
    collect_keys(&json, "", &mut keys);
    let pinned: BTreeSet<String> = [
        "schema",
        "run_id",
        "seq",
        "t_mono_ns",
        "t_wall_ms",
        "host",
        "host.load_pct",
        "host.activity_pct",
        "host.cpu_pct",
        "host.cpu_topk_pct",
        "host.gpu_pct",
        "host.mem_pct",
        "host.coolant_c",
        "host.cpu_c",
        "host.gpu_c",
        "host.gpu_w",
        "host.gpu_limit_w",
        "host.cpu_w",
        "host.vram_used_bytes",
        "host.vram_total_bytes",
        "host.mem_used_bytes",
        "host.mem_total_bytes",
        "ai",
        "ai.state",
        "ai.models",
        "ai.models.name",
        "ai.models.state",
        "ai.models.full_name",
        "ai.models.detail",
        "ai.models.detail.ctx",
        "ai.models.detail.ncmoe",
        "ai.models.detail.kv_k",
        "ai.models.detail.kv_v",
        "ai.models.detail.quant",
        "ai.models.detail.fa",
        "ai.models.detail.kv_block",
        "ai.models.detail.prefix_cache",
        "ai.models.backend",
        "ai.models.running",
        "ai.models.queued",
        "ai.models.kv_fill",
        "ai.models.cache_hit",
        "ai.models.slots_busy",
        "ai.models.slots_total",
        "ai.models.prompt_tokens",
        "ai.models.prompt_cached_tokens",
        "ai.models.slot_ctx",
        "ai.models.slot_ctx.slot",
        "ai.models.slot_ctx.used",
        "ai.models.slot_ctx.resets",
        "ai.models.slot_ctx.resets.compacted",
        "ai.models.slot_ctx.resets.new",
        "ai.models.slot_ctx.resets.evicted",
        "ai.models.slot_ctx.resets.unknown",
        "ai.models.engine",
        "ai.models.engine.spec_accept",
        "ai.models.engine.spec_len",
        "ai.models.engine.spec_drafts",
        "ai.models.engine.spec_draft_tokens",
        "ai.models.engine.spec_accepted_tokens",
        "ai.models.engine.preemptions",
        "ai.models.engine.sleeping",
        "ai.models.engine.ttft_s",
        "ai.models.engine.itl_s",
        "ai.models.engine.e2e_s",
        "tokens",
        "tokens.decoded_total",
        "tokens.prompt_total",
        "fans",
        "fans.channel",
        "fans.label",
        "fans.rpm",
        "fans.pwm",
        "sources",
        "sources.llama-swap",
        "sources.llama-swap.up",
        "sources.llama-swap.latency_s",
        "sources.running",
        "sources.running.up",
        "sources.running.latency_s",
        "sources.metrics",
        "sources.metrics.up",
        "sources.metrics.latency_s",
        "sources.activity",
        "sources.activity.up",
        "sources.gpu",
        "sources.gpu.up",
        "sources.hwmon",
        "sources.hwmon.up",
        "sources.proc",
        "sources.proc.up",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(
        keys, pinned,
        "the snapshot wire changed; review llama-metrics for text before updating this pin"
    );
    // A token count is not text: `prompt_total` (#11) and the per-model
    // prompt counters (#10) are u64 counters, typed on the wire, so they are
    // the keys allowed to say "prompt".
    const COUNTS: [&str; 6] = [
        "tokens.prompt_total",
        "llamabored_tokens_prompt_total",
        "ai.models.prompt_tokens",
        "ai.models.prompt_cached_tokens",
        "llamabored_model_prompt_tokens_total",
        "llamabored_model_prompt_cached_tokens_total",
    ];
    assert!(json["tokens"]["prompt_total"].is_u64());
    assert!(json["ai"]["models"][0]["prompt_tokens"].is_u64());
    assert!(json["ai"]["models"][0]["prompt_cached_tokens"].is_u64());
    for key in keys.iter().filter(|key| !COUNTS.contains(&key.as_str())) {
        for word in ["prompt", "output", "input", "text", "content", "message"] {
            assert!(!key.contains(word), "wire key {key} looks like text");
        }
    }

    // The only strings allowed out are snapshot names and tokens, and the
    // exporter's own fixed label values.
    let text = render(&Ok(snap.clone()), T0);
    let mut allowed: BTreeSet<String> =
        ["", "down", "idle", "loaded", "busy", "denied", "all", "f16"]
            .into_iter()
            .chain(["llamacpp", "sglang", "vllm", "strata", "openai"])
            .chain(["compacted", "new", "evicted", "unknown"])
            .map(str::to_owned)
            .collect();
    allowed.insert(env!("CARGO_PKG_VERSION").to_owned());
    for state in ["ready", "starting", "stopping", "other"] {
        allowed.insert(state.to_owned());
    }
    for fan in &snap.fans {
        allowed.insert(fan.channel.to_string());
        allowed.insert(fan.label.clone());
    }
    for (source, _) in wire::Sources::default().entries() {
        allowed.insert(source.to_owned());
    }
    allowed.insert(wire::SCHEMA.to_string());
    for model in &snap.ai.models {
        for row in &model.slot_ctx {
            allowed.insert(row.slot.to_string());
        }
        allowed.insert(model.name.clone());
        if let Some(full) = &model.full_name {
            allowed.insert(full.clone());
        }
        if let Some(d) = &model.detail {
            for token in [&d.quant, &d.kv_k, &d.kv_v].into_iter().flatten() {
                allowed.insert(token.clone());
            }
            if let (Some(k), Some(v)) = (&d.kv_k, &d.kv_v) {
                allowed.insert(format!("{k}/{v}"));
            }
            if let Some(ctx) = d.ctx {
                allowed.insert(ctx.to_string());
            }
            if let Some(n) = d.ncmoe {
                allowed.insert(n.to_string());
            }
        }
    }
    let label_keys: BTreeSet<&str> = [
        "version",
        "wire_schema",
        "reason",
        "state",
        "name",
        "full_name",
        "quant",
        "kv",
        "ctx",
        "moe",
        "backend",
        "channel",
        "label",
        "source",
        "slot",
    ]
    .into_iter()
    .collect();
    for (name, labels, _) in samples(&text) {
        for word in ["prompt", "output", "input", "text", "content", "message"] {
            assert!(
                COUNTS.contains(&name.as_str()) || !name.contains(word),
                "{name} looks like a text metric"
            );
        }
        for (key, value) in labels {
            assert!(label_keys.contains(key.as_str()), "{name} has label {key}");
            assert!(
                allowed.contains(&value),
                "{name}{{{key}}} exports {value:?}"
            );
        }
    }
    for line in text.lines().filter(|l| l.starts_with("# HELP ")) {
        assert!(line.len() < 200, "HELP text is long: {line}");
    }
}

/// Snapshot leaves that are exported, and the series that carries each one
/// (a value or a label value).
const EXPORTED: &[(&str, &str)] = &[
    ("seq", "llamabored_snapshot_seq"),
    ("t_mono_ns", "llamabored_snapshot_age_seconds"),
    ("host.load_pct", "llamabored_load_pct"),
    ("host.activity_pct", "llamabored_activity_pct"),
    ("host.cpu_pct", "llamabored_cpu_pct"),
    ("host.cpu_topk_pct", "llamabored_cpu_topk_pct"),
    ("host.gpu_pct", "llamabored_gpu_pct"),
    ("host.mem_pct", "llamabored_mem_pct"),
    ("host.coolant_c", "llamabored_coolant_celsius"),
    ("host.cpu_c", "llamabored_cpu_celsius"),
    ("host.gpu_c", "llamabored_gpu_celsius"),
    ("host.gpu_w", "llamabored_gpu_power_watts"),
    ("host.gpu_limit_w", "llamabored_gpu_power_limit_watts"),
    ("host.cpu_w", "llamabored_cpu_power_watts"),
    ("host.vram_used_bytes", "llamabored_gpu_memory_used_bytes"),
    ("host.vram_total_bytes", "llamabored_gpu_memory_total_bytes"),
    ("host.mem_used_bytes", "llamabored_memory_used_bytes"),
    ("host.mem_total_bytes", "llamabored_memory_total_bytes"),
    ("ai.state", "llamabored_ai_state"),
    ("ai.models.name", "llamabored_model_loaded"),
    ("ai.models.full_name", "llamabored_model_loaded"),
    ("ai.models.state", "llamabored_model_state"),
    ("ai.models.backend", "llamabored_model_loaded"),
    ("ai.models.detail.ctx", "llamabored_model_ctx_size_tokens"),
    ("ai.models.detail.ncmoe", "llamabored_model_loaded"),
    ("ai.models.detail.kv_k", "llamabored_model_loaded"),
    ("ai.models.detail.kv_v", "llamabored_model_loaded"),
    ("ai.models.detail.quant", "llamabored_model_loaded"),
    ("ai.models.running", "llamabored_model_requests_running"),
    ("ai.models.queued", "llamabored_model_requests_queued"),
    ("ai.models.kv_fill", "llamabored_model_kv_cache_usage_ratio"),
    ("ai.models.cache_hit", "llamabored_model_cache_hit_ratio"),
    ("ai.models.slots_busy", "llamabored_slots_busy"),
    ("ai.models.slots_total", "llamabored_slots_total"),
    (
        "ai.models.prompt_tokens",
        "llamabored_model_prompt_tokens_total",
    ),
    (
        "ai.models.prompt_cached_tokens",
        "llamabored_model_prompt_cached_tokens_total",
    ),
    (
        "ai.models.detail.kv_block",
        "llamabored_model_kv_block_size_tokens",
    ),
    (
        "ai.models.detail.prefix_cache",
        "llamabored_model_prefix_caching",
    ),
    (
        "ai.models.engine.spec_accept",
        "llamabored_model_spec_acceptance_ratio",
    ),
    (
        "ai.models.engine.spec_len",
        "llamabored_model_spec_accepted_length",
    ),
    (
        "ai.models.engine.spec_drafts",
        "llamabored_model_spec_drafts_total",
    ),
    (
        "ai.models.engine.spec_draft_tokens",
        "llamabored_model_spec_draft_tokens_total",
    ),
    (
        "ai.models.engine.spec_accepted_tokens",
        "llamabored_model_spec_accepted_tokens_total",
    ),
    (
        "ai.models.engine.preemptions",
        "llamabored_model_preemptions_total",
    ),
    ("ai.models.engine.sleeping", "llamabored_model_sleeping"),
    ("ai.models.engine.ttft_s", "llamabored_model_ttft_seconds"),
    ("ai.models.engine.itl_s", "llamabored_model_itl_seconds"),
    (
        "ai.models.engine.e2e_s",
        "llamabored_model_e2e_latency_seconds",
    ),
    ("ai.models.slot_ctx.slot", "llamabored_slot_ctx_used_tokens"),
    ("ai.models.slot_ctx.used", "llamabored_slot_ctx_used_tokens"),
    (
        "ai.models.slot_ctx.resets.compacted",
        "llamabored_slot_ctx_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.new",
        "llamabored_slot_ctx_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.evicted",
        "llamabored_slot_ctx_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.unknown",
        "llamabored_slot_ctx_resets_total",
    ),
    ("tokens.decoded_total", "llamabored_tokens_decoded_total"),
    ("tokens.prompt_total", "llamabored_tokens_prompt_total"),
    ("fans.channel", "llamabored_fan_rpm"),
    ("fans.label", "llamabored_fan_rpm"),
    ("fans.rpm", "llamabored_fan_rpm"),
    ("fans.pwm", "llamabored_fan_pwm_ratio"),
    ("sources.*.up", "llamabored_source_up"),
    ("sources.*.latency_s", "llamabored_source_latency_seconds"),
];

/// Snapshot leaves that are deliberately not exported, with the reason.
const NOT_EXPORTED: &[(&str, &str)] = &[
    (
        "schema",
        "validated equal to wire::SCHEMA; build_info's wire_schema carries it",
    ),
    (
        "run_id",
        "random per watcher start; a label would be unbounded, a value meaningless",
    ),
    ("t_wall_ms", "wall clock for logs; age comes from t_mono_ns"),
    (
        "ai.models.detail.fa",
        "flash attention flag; carried on the wire but drawn by no dashboard",
    ),
];

fn collect_leaves(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                // Source names are a fixed set; one pattern covers them.
                let key = if prefix == "sources" {
                    "*"
                } else {
                    key.as_str()
                };
                let path = if prefix.is_empty() {
                    key.to_owned()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_leaves(child, &path, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_leaves(item, prefix, out);
            }
        }
        _ => {
            out.insert(prefix.to_owned());
        }
    }
}

/// #11: every snapshot field reaches Prometheus, unless NOT_EXPORTED says
/// why not. A new wire field fails here until it is exported or listed.
#[test]
fn every_snapshot_field_is_exported_or_listed() {
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(common::fixture("snapshot-loaded.json")).unwrap())
            .unwrap();
    let snap = load("snapshot-loaded.json");
    let json: serde_json::Value =
        serde_json::from_slice(&wire::to_json(&snap).expect("encode")).expect("json");
    // The fixture holds only fields the wire knows (none silently ignored),
    // and it sets every one of them.
    let mut raw_leaves = BTreeSet::new();
    collect_leaves(&raw, "", &mut raw_leaves);
    let mut leaves = BTreeSet::new();
    collect_leaves(&json, "", &mut leaves);
    assert_eq!(raw_leaves, leaves, "fixture has keys the wire ignores");

    let exported: BTreeSet<&str> = EXPORTED.iter().map(|(path, _)| *path).collect();
    let skipped: BTreeSet<&str> = NOT_EXPORTED.iter().map(|(path, _)| *path).collect();
    assert!(exported.is_disjoint(&skipped));
    for leaf in &leaves {
        assert!(
            exported.contains(leaf.as_str()) || skipped.contains(leaf.as_str()),
            "snapshot field {leaf} has no exported series; export it or add it to NOT_EXPORTED with a reason"
        );
    }
    for path in exported.iter().chain(&skipped) {
        assert!(
            leaves.contains(*path),
            "{path} is listed but not in snapshot-loaded.json; set it there"
        );
    }
    let text = render(&Ok(snap), T0);
    let names: BTreeSet<String> = samples(&text).into_iter().map(|(n, _, _)| n).collect();
    for (path, metric) in EXPORTED {
        assert!(names.contains(*metric), "{path}: {metric} is not rendered");
    }
    for (_, reason) in NOT_EXPORTED {
        assert!(!reason.is_empty());
    }
}

fn collect_keys(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                out.insert(path.clone());
                collect_keys(child, &path, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_keys(item, prefix, out);
            }
        }
        _ => {}
    }
}

fn value_of(text: &str, metric: &str, slot: Option<&str>) -> Option<String> {
    samples(text)
        .into_iter()
        .find(|(name, labels, _)| {
            name == metric
                && labels
                    .iter()
                    .any(|(k, v)| k == "name" && v == "Qwen3-Coder…")
                && slot.is_none_or(|slot| labels.iter().any(|(k, v)| k == "slot" && v == slot))
                && labels.iter().all(|(k, v)| k != "reason" || v == "new")
        })
        .map(|(_, _, value)| value)
}

/// #10: the counters are the watcher's, passed through. A watcher restart
/// (new run_id) starts them at 0 again, which Prometheus reads as a counter
/// reset; the exporter neither holds nor stitches them.
#[test]
fn slot_and_prompt_counters_pass_through_and_restart_with_the_watcher() {
    let first = load("snapshot-loaded.json");
    let text = render(&Ok(first.clone()), T0);
    let counters = [
        ("llamabored_model_prompt_tokens_total", None, "1622000"),
        (
            "llamabored_model_prompt_cached_tokens_total",
            None,
            "1500000",
        ),
        ("llamabored_slot_ctx_resets_total", Some("1"), "2"),
        ("llamabored_slot_ctx_used_tokens", Some("1"), "91500"),
    ];
    for (metric, slot, want) in counters {
        assert_eq!(
            value_of(&text, metric, slot).as_deref(),
            Some(want),
            "{metric}"
        );
    }
    let mut restarted = first;
    restarted.run_id += 1;
    restarted.seq = 1;
    let qwen = &mut restarted.ai.models[0];
    qwen.prompt_tokens = Some(0);
    qwen.prompt_cached_tokens = Some(0);
    for row in &mut qwen.slot_ctx {
        row.resets = wire::SlotResetsWire::default();
    }
    let text = render(&Ok(restarted), T0);
    for (metric, slot, _) in &counters[..3] {
        assert_eq!(
            value_of(&text, metric, *slot).as_deref(),
            Some("0"),
            "{metric}"
        );
    }
    assert!(text.contains("# TYPE llamabored_slot_ctx_resets_total counter"));
    assert!(text.contains("# TYPE llamabored_model_prompt_cached_tokens_total counter"));
    assert!(text.contains("# TYPE llamabored_slot_ctx_used_tokens gauge"));
}
