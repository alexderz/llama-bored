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
    assert_eq!(SnapshotFile::at(&bad).read(), Err(ReadError::Invalid));

    // A validated-parser rule, not only JSON: out-of-range percent.
    let text = std::fs::read_to_string(&good)
        .unwrap()
        .replace("\"gpu_pct\": 97", "\"gpu_pct\": 170");
    std::fs::write(&bad, text).unwrap();
    assert_eq!(SnapshotFile::at(&bad).read(), Err(ReadError::Invalid));
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
    assert_eq!(models.len(), 3);
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
    assert_eq!(loaded, 3);
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
        "ai.models.backend",
        "ai.models.running",
        "ai.models.queued",
        "ai.models.kv_fill",
        "tokens",
        "tokens.decoded_total",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(
        keys, pinned,
        "the snapshot wire changed; review llama-metrics for text before updating this pin"
    );
    for key in &keys {
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
            .chain(["llamacpp", "sglang", "vllm", "openai"])
            .map(str::to_owned)
            .collect();
    allowed.insert(env!("CARGO_PKG_VERSION").to_owned());
    allowed.insert(wire::SCHEMA.to_string());
    for model in &snap.ai.models {
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
    ]
    .into_iter()
    .collect();
    for (name, labels, _) in samples(&text) {
        for word in ["prompt", "output", "input", "text", "content", "message"] {
            assert!(!name.contains(word), "{name} looks like a text metric");
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
