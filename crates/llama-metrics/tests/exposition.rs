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

/// Strict 0.0.4 parser. Every line is `# HELP`, `# TYPE` or
/// `name{labels} value`. It checks that names match
/// `[a-zA-Z_:][a-zA-Z0-9_:]*` (and are lowercase `llamabored_*`), that
/// each family has its HELP then its TYPE before any sample and is typed
/// once, that only counters end in `_total` and every counter does, that
/// a summary has only `_sum` and `_count` samples, and that no series
/// repeats. Returns `(name, labels, value)`.
type Sample = (String, Vec<(String, String)>, String);

fn is_metric_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_' || b == b':')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b':')
}

fn samples(text: &str) -> Vec<Sample> {
    let mut out = Vec::new();
    let mut typed: std::collections::BTreeMap<String, String> = Default::default();
    let mut helped: BTreeSet<String> = BTreeSet::new();
    let mut series: BTreeSet<String> = BTreeSet::new();
    assert!(text.ends_with('\n'), "exposition must end with a newline");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest.split_once(' ').expect("HELP line");
            assert!(is_metric_name(name), "{name} is not a metric name");
            assert!(!help.is_empty(), "{name} has no help");
            assert!(helped.insert(name.to_owned()), "{name} has two HELP lines");
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').expect("TYPE line");
            assert!(
                ["gauge", "counter", "summary"].contains(&kind),
                "unexpected type {kind}"
            );
            assert!(helped.contains(name), "{name} typed before its HELP");
            assert_eq!(
                name.ends_with("_total"),
                kind == "counter",
                "{name}: only counters end in _total, and every counter does"
            );
            assert!(
                typed.insert(name.to_owned(), kind.to_owned()).is_none(),
                "{name} typed twice"
            );
            continue;
        }
        assert!(!line.starts_with('#'), "stray comment {line:?}");
        let (key, value) = line.rsplit_once(' ').expect("sample line");
        value
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("value {value:?} is not a float"));
        let (name, labels) = match key.split_once('{') {
            Some((name, rest)) => (name, parse_labels(rest.strip_suffix('}').expect("}"))),
            None => (key, Vec::new()),
        };
        assert!(is_metric_name(name), "{name} is not a metric name");
        assert!(name.starts_with("llamabored_"), "{name} lacks the prefix");
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "{name} is not a lowercase metric name"
        );
        let family = match typed.get(name).map(String::as_str) {
            Some("summary") => panic!("{name}: a summary has only _sum and _count samples"),
            Some(_) => name,
            None => {
                let base = name
                    .strip_suffix("_sum")
                    .or_else(|| name.strip_suffix("_count"))
                    .unwrap_or_else(|| panic!("{name} sampled before its TYPE"));
                assert_eq!(
                    typed.get(base).map(String::as_str),
                    Some("summary"),
                    "{name} sampled before its TYPE"
                );
                base
            }
        };
        let _ = family;
        let mut keys: Vec<&str> = labels.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "{line}: a label repeats");
        assert!(
            series.insert(format!("{name}{labels:?}")),
            "duplicate series {line}"
        );
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
        !text.contains("llamabored_gpu_utilization_ratio"),
        "absent host field was exported"
    );
    assert!(text.contains("llamabored_cpu_utilization_ratio 0.05\n"));
    assert!(!text.contains("llamabored_model_"));
    assert!(text.contains("llamabored_ai_state{state=\"down\"} 1\n"));
}

#[test]
fn stale_snapshot_keeps_seq_and_age_and_drops_values() {
    let read = Ok(load("snapshot-loaded.json"));
    // Exactly the limit is fresh; one nanosecond more is stale.
    let edge = render(&read, T0 + 5_000_000_000);
    assert!(edge.contains("llamabored_snapshot_stale 0\n"), "{edge}");
    assert!(edge.contains("llamabored_gpu_utilization_ratio 0.97\n"));

    let text = render(&read, T0 + 5_000_000_001);
    golden("stale.prom", &text);
    samples(&text);
    assert!(text.contains("llamabored_snapshot_up 1\n"));
    assert!(text.contains("llamabored_snapshot_stale 1\n"));
    assert!(text.contains("llamabored_snapshot_seq 4242\n"));
    assert!(text.contains("llamabored_snapshot_age_seconds 5.000\n"));
    for gone in [
        "llamabored_gpu_utilization_ratio",
        "llamabored_activity_ratio",
        "llamabored_coolant_celsius",
        "llamabored_ai_state",
        "llamabored_model_info",
        "llamabored_model_context_size_tokens",
        "llamabored_model_generation_tokens_total",
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
    assert!(text.contains("\"gpu_pct\": 170"));
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

/// The labels of `name`'s sample for `model`, when there is one.
fn find<'a>(all: &'a [Sample], name: &str, model: &str) -> Option<&'a Sample> {
    all.iter()
        .find(|(n, l, _)| n == name && l.first().is_some_and(|(k, v)| k == "model" && v == model))
}

fn value(all: &[Sample], name: &str, model: &str) -> Option<String> {
    find(all, name, model).map(|(_, _, v)| v.clone())
}

/// #71: one info series per model with its descriptive strings; every
/// other per-model series has `model` and `engine` only, first.
#[test]
fn model_info_carries_the_strings_and_every_series_is_keyed() {
    let read = Ok(load("snapshot-loaded.json"));
    let text = render(&read, T0);
    let all = samples(&text);
    let infos: Vec<&Vec<(String, String)>> = all
        .iter()
        .filter(|(n, _, _)| n == "llamabored_model_info")
        .map(|(_, l, _)| l)
        .collect();
    assert_eq!(infos.len(), 6);
    let info = |model: &str| -> Vec<(&str, &str)> {
        infos
            .iter()
            .find(|l| l[0].1 == model)
            .unwrap_or_else(|| panic!("{model}"))
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    };
    assert_eq!(
        info("qwen3-coder-30b"),
        [
            ("model", "qwen3-coder-30b"),
            ("engine", "llamacpp"),
            ("display_name", "Qwen3-Coder-30B \"fast\" \\ build"),
            ("quant", "UD-Q4_K_M"),
            ("kv_type", "q8_0"),
            ("version", ""),
        ],
        "no backend on the wire is llama.cpp"
    );
    assert_eq!(
        info("bonsai-8b")[2..5],
        [
            ("display_name", "Bonsai 8B"),
            ("quant", "PTQ1_0"),
            ("kv_type", "q8_0/q4_0"),
        ]
    );
    assert_eq!(info("bonsai-8b")[1], ("engine", "sglang"));
    // An older watcher's model has no id: its display name keys it, and
    // unknown strings are empty.
    assert_eq!(
        info("tiny"),
        [
            ("model", "tiny"),
            ("engine", "llamacpp"),
            ("display_name", "tiny"),
            ("quant", ""),
            ("kv_type", ""),
            ("version", ""),
        ]
    );
    assert_eq!(info("qwen3.8-27b-vllm")[4], ("kv_type", "fp8_e4m3"));
    assert_eq!(info("qwen3.8-27b-vllm")[2], ("display_name", "qwen3.8-27b"));
    assert_eq!(info("flash-next")[1], ("engine", "strata"));
    assert_eq!(info("flash-next")[5], ("version", "0.1.41"));
    assert_eq!(info("tabby-exl3")[1], ("engine", "openai"));

    for (name, labels, _) in all.iter().filter(|(n, _, _)| {
        (n.starts_with("llamabored_model_") || n.starts_with("llamabored_slot_"))
            && n != "llamabored_model_info"
    }) {
        let keys: Vec<&str> = labels.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys[..2], ["model", "engine"], "{name}");
        assert!(
            keys[2..]
                .iter()
                .all(|k| ["state", "status", "slot", "reason"].contains(k)),
            "{name} {keys:?}"
        );
    }
}

/// #71: the same series names for every engine that has the quantity;
/// what an engine cannot report is absent, not zero.
#[test]
fn every_engine_gets_the_same_series_names() {
    let text = render(&Ok(load("snapshot-loaded.json")), T0);
    let all = samples(&text);
    let engines = [
        ("qwen3-coder-30b", "llamacpp"),
        ("bonsai-8b", "sglang"),
        ("qwen3.8-27b-vllm", "vllm"),
        ("flash-next", "strata"),
        ("tabby-exl3", "openai"),
    ];
    let has = |name: &str, model: &str| find(&all, name, model).is_some();
    // name: the engines whose source has it.
    let table: [(&str, &[&str]); 16] = [
        (
            "llamabored_model_info",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_state",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_generation_tokens_total",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_prompt_tokens_total",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_requests_total",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_request_duration_seconds_count",
            &["llamacpp", "sglang", "vllm", "strata", "openai"],
        ),
        (
            "llamabored_model_prompt_cached_tokens_total",
            &["llamacpp", "sglang", "vllm", "strata"],
        ),
        (
            "llamabored_model_requests_running",
            &["llamacpp", "sglang", "vllm", "strata"],
        ),
        (
            "llamabored_model_requests_waiting",
            &["llamacpp", "sglang", "vllm", "strata"],
        ),
        (
            "llamabored_model_slots",
            &["llamacpp", "sglang", "vllm", "strata"],
        ),
        (
            "llamabored_model_prefill_seconds_total",
            &["llamacpp", "vllm", "strata"],
        ),
        (
            "llamabored_model_decode_seconds_total",
            &["llamacpp", "vllm", "strata"],
        ),
        (
            "llamabored_model_spec_draft_tokens_total",
            &["llamacpp", "vllm", "strata"],
        ),
        (
            "llamabored_model_time_to_first_token_seconds_sum",
            &["sglang", "vllm"],
        ),
        (
            "llamabored_model_inter_token_latency_seconds_count",
            &["sglang", "vllm"],
        ),
        ("llamabored_model_kv_cache_usage_ratio", &["sglang", "vllm"]),
    ];
    for (name, want) in table {
        for (model, engine) in engines {
            assert_eq!(
                has(name, model),
                want.contains(&engine),
                "{name} for {engine}"
            );
        }
    }
    // One name for one quantity: requests by status, latencies as summaries.
    assert_eq!(
        value(
            &all,
            "llamabored_model_generation_tokens_total",
            "qwen3-coder-30b"
        )
        .as_deref(),
        Some("52000")
    );
    assert_eq!(
        value(
            &all,
            "llamabored_model_prefill_seconds_total",
            "qwen3.8-27b-vllm"
        )
        .as_deref(),
        Some("50.000")
    );
    assert_eq!(
        value(&all, "llamabored_model_decode_seconds_total", "flash-next").as_deref(),
        Some("81.000")
    );
    let status = |model: &str, status: &str| {
        all.iter()
            .find(|(n, l, _)| {
                n == "llamabored_model_requests_total"
                    && l[0].1 == model
                    && l.iter().any(|(k, v)| k == "status" && v == status)
            })
            .map(|(_, _, v)| v.clone())
    };
    assert_eq!(status("tabby-exl3", "ok").as_deref(), Some("2"));
    assert_eq!(status("tabby-exl3", "error").as_deref(), Some("1"));
    assert_eq!(
        value(
            &all,
            "llamabored_model_time_to_first_token_seconds_sum",
            "qwen3.8-27b-vllm"
        )
        .as_deref(),
        Some("31.500")
    );
    assert_eq!(
        value(
            &all,
            "llamabored_model_time_to_first_token_seconds_count",
            "qwen3.8-27b-vllm"
        )
        .as_deref(),
        Some("42")
    );
    for summary in [
        "llamabored_model_time_to_first_token_seconds",
        "llamabored_model_inter_token_latency_seconds",
        "llamabored_model_request_duration_seconds",
    ] {
        assert!(
            text.contains(&format!("# TYPE {summary} summary\n")),
            "{summary}"
        );
    }
    // Strata counts no draft rounds; vLLM does.
    assert!(has(
        "llamabored_model_spec_drafts_total",
        "qwen3.8-27b-vllm"
    ));
    assert!(!has("llamabored_model_spec_drafts_total", "flash-next"));
    assert!(!has(
        "llamabored_model_spec_drafts_total",
        "qwen3-coder-30b"
    ));
    assert_eq!(
        value(
            &all,
            "llamabored_model_expert_cache_hit_ratio",
            "flash-next"
        )
        .as_deref(),
        Some("0.875")
    );
    assert_eq!(
        value(&all, "llamabored_model_pcie_share_ratio", "flash-next").as_deref(),
        Some("0.09375")
    );
    assert!(!has(
        "llamabored_model_expert_cache_hit_ratio",
        "qwen3.8-27b-vllm"
    ));
    assert_eq!(
        value(&all, "llamabored_model_slots", "qwen3-coder-30b").as_deref(),
        Some("4"),
        "llama.cpp slots"
    );
    assert_eq!(
        value(&all, "llamabored_model_slots", "qwen3.8-27b-vllm").as_deref(),
        Some("8"),
        "vLLM --max-num-seqs"
    );
    // A starting model still has its series; `tiny` (no counters) has none.
    assert!(has("llamabored_model_generation_tokens_total", "bonsai-8b"));
    assert!(!has("llamabored_model_generation_tokens_total", "tiny"));
}

/// #71: host percents are 0..1 ratios; old names, `name` and `full_name`
/// labels, the memory percent and the box token totals are gone.
#[test]
fn host_ratios_replace_percents_and_old_names_are_gone() {
    let text = render(&Ok(load("snapshot-loaded.json")), T0);
    for line in [
        "llamabored_activity_ratio 1.1225\n",
        "llamabored_load_ratio 0.645\n",
        "llamabored_cpu_utilization_ratio 0.235\n",
        "llamabored_cpu_topk_utilization_ratio 0.645\n",
        "llamabored_gpu_utilization_ratio 0.97\n",
    ] {
        assert!(text.contains(line), "{line}");
    }
    for gone in [
        "_pct",
        "llamabored_tokens_",
        "llamabored_model_loaded",
        "llamabored_slots_",
        "llamabored_slot_ctx_",
        "llamabored_model_ctx_size_tokens",
        "llamabored_model_requests_queued",
        "llamabored_model_cache_hit_ratio",
        "llamabored_model_spec_acceptance_ratio",
        "llamabored_model_spec_accepted_length",
        "llamabored_model_ttft_seconds",
        "llamabored_model_itl_seconds",
        "llamabored_model_e2e_latency_seconds",
        "_tokens_per_second",
    ] {
        assert!(!text.contains(gone), "{gone} is still exported");
    }
    for (name, labels, _) in samples(&text) {
        for (key, _) in labels {
            assert!(
                !["name", "full_name", "backend", "moe", "kv", "ctx"].contains(&key.as_str()),
                "{name} still has label {key}"
            );
        }
    }
}

#[test]
fn duplicate_models_export_one_series() {
    let mut snap = load("snapshot-loaded.json");
    let first = snap.ai.models[0].clone();
    snap.ai.models.push(first);
    let text = render(&Ok(snap), T0);
    samples(&text);
    let count = |prefix: &str| text.lines().filter(|l| l.starts_with(prefix)).count();
    assert_eq!(count("llamabored_model_info{"), 6);
    assert_eq!(count("llamabored_model_context_size_tokens{"), 3);
    assert_eq!(count("llamabored_slot_context_used_tokens{"), 2);
}

/// The wire carries no prompt or output text, and the exporter exports none.
///
/// Two halves. (1) The snapshot's JSON keys are pinned: a new wire field
/// (such as a prompt tail) fails here and forces a review of this exporter.
/// (2) Every exported string is a model id, name or an allowlisted token
/// from the snapshot, or one of the exporter's fixed label values.
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
        "ai.models.id",
        "ai.models.version",
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
        "ai.models.slots_total",
        "ai.models.max_running",
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
        "ai.models.engine.spec_drafts",
        "ai.models.engine.spec_draft_tokens",
        "ai.models.engine.spec_accepted_tokens",
        "ai.models.engine.preemptions",
        "ai.models.engine.sleeping",
        "ai.models.engine.expert_hit",
        "ai.models.engine.pcie_share",
        // #71: numbers only.
        "ai.models.counters",
        "ai.models.counters.gen_tokens",
        "ai.models.counters.prefill_ms",
        "ai.models.counters.decode_ms",
        "ai.models.counters.req_ok",
        "ai.models.counters.req_err",
        "ai.models.counters.ttft",
        "ai.models.counters.ttft.ms",
        "ai.models.counters.ttft.n",
        "ai.models.counters.itl",
        "ai.models.counters.itl.ms",
        "ai.models.counters.itl.n",
        "ai.models.counters.e2e",
        "ai.models.counters.e2e.ms",
        "ai.models.counters.e2e.n",
        "tokens",
        "tokens.decoded_total",
        "tokens.prompt_total",
        "fans",
        "fans.channel",
        "fans.label",
        "fans.rpm",
        "fans.pwm",
        // #74: hwmon chip and sensor names, numbers.
        "fan_rows",
        "fan_rows.c",
        "fan_rows.n",
        "fan_rows.l",
        "fan_rows.r",
        "fan_rows.p",
        "temps",
        "temps.c",
        "temps.s",
        "temps.t",
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
        // #70: a sanitised llama-swap model id and a count, no text.
        "suspected_loads",
        "suspected_loads.model",
        "suspected_loads.count",
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
    const COUNTS: [&str; 5] = [
        "tokens.prompt_total",
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

    // The only strings allowed out are snapshot ids, names and tokens, and
    // the exporter's own fixed label values.
    let text = render(&Ok(snap.clone()), T0);
    let mut allowed: BTreeSet<String> = ["", "down", "idle", "loaded", "busy", "denied", "f16"]
        .into_iter()
        .chain(["llamacpp", "sglang", "vllm", "strata", "openai"])
        .chain(["compacted", "new", "evicted", "unknown"])
        .chain(["ok", "error"])
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
    for fan in &snap.fan_rows {
        allowed.insert(fan.chip.clone());
        allowed.insert(fan.channel.to_string());
        allowed.insert(fan.label.clone());
    }
    for t in &snap.temps {
        allowed.insert(t.chip.clone());
        allowed.insert(t.sensor.clone());
    }
    for (source, _) in wire::Sources::default().entries() {
        allowed.insert(source.to_owned());
    }
    allowed.insert(wire::SCHEMA.to_string());
    // #70: suspected loads carry the sanitised llama-swap model id.
    for row in &snap.suspected_loads {
        allowed.insert(row.model.clone());
    }
    for model in &snap.ai.models {
        for row in &model.slot_ctx {
            allowed.insert(row.slot.to_string());
        }
        allowed.insert(model.name.clone());
        allowed.extend(model.full_name.clone());
        allowed.extend(model.id.clone());
        allowed.extend(model.version.clone());
        if let Some(d) = &model.detail {
            for token in [&d.quant, &d.kv_k, &d.kv_v].into_iter().flatten() {
                allowed.insert(token.clone());
            }
            if let (Some(k), Some(v)) = (&d.kv_k, &d.kv_v) {
                allowed.insert(format!("{k}/{v}"));
            }
        }
    }
    let label_keys: BTreeSet<&str> = [
        "version",
        "wire_schema",
        "reason",
        "state",
        "model",
        "engine",
        "display_name",
        "quant",
        "kv_type",
        "status",
        "channel",
        "label",
        "chip",
        "sensor",
        "source",
        "slot",
    ]
    .into_iter()
    .collect();
    for (name, labels, _) in samples(&text) {
        // A context size is a token count (#71): "context" is not "text".
        let plain = name.replace("context", "");
        for word in ["prompt", "output", "input", "text", "content", "message"] {
            assert!(
                COUNTS.contains(&name.as_str()) || !plain.contains(word),
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
    ("host.load_pct", "llamabored_load_ratio"),
    ("host.activity_pct", "llamabored_activity_ratio"),
    ("host.cpu_pct", "llamabored_cpu_utilization_ratio"),
    ("host.cpu_topk_pct", "llamabored_cpu_topk_utilization_ratio"),
    ("host.gpu_pct", "llamabored_gpu_utilization_ratio"),
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
    ("ai.models.id", "llamabored_model_info"),
    ("ai.models.name", "llamabored_model_info"),
    ("ai.models.full_name", "llamabored_model_info"),
    ("ai.models.version", "llamabored_model_info"),
    ("ai.models.backend", "llamabored_model_info"),
    ("ai.models.detail.kv_k", "llamabored_model_info"),
    ("ai.models.detail.kv_v", "llamabored_model_info"),
    ("ai.models.detail.quant", "llamabored_model_info"),
    ("ai.models.state", "llamabored_model_state"),
    (
        "ai.models.detail.ctx",
        "llamabored_model_context_size_tokens",
    ),
    ("ai.models.running", "llamabored_model_requests_running"),
    ("ai.models.queued", "llamabored_model_requests_waiting"),
    ("ai.models.slots_total", "llamabored_model_slots"),
    ("ai.models.max_running", "llamabored_model_slots"),
    ("ai.models.kv_fill", "llamabored_model_kv_cache_usage_ratio"),
    (
        "ai.models.prompt_tokens",
        "llamabored_model_prompt_tokens_total",
    ),
    (
        "ai.models.prompt_cached_tokens",
        "llamabored_model_prompt_cached_tokens_total",
    ),
    (
        "ai.models.counters.gen_tokens",
        "llamabored_model_generation_tokens_total",
    ),
    (
        "ai.models.counters.prefill_ms",
        "llamabored_model_prefill_seconds_total",
    ),
    (
        "ai.models.counters.decode_ms",
        "llamabored_model_decode_seconds_total",
    ),
    (
        "ai.models.counters.req_ok",
        "llamabored_model_requests_total",
    ),
    (
        "ai.models.counters.req_err",
        "llamabored_model_requests_total",
    ),
    (
        "ai.models.counters.ttft.ms",
        "llamabored_model_time_to_first_token_seconds_sum",
    ),
    (
        "ai.models.counters.ttft.n",
        "llamabored_model_time_to_first_token_seconds_count",
    ),
    (
        "ai.models.counters.itl.ms",
        "llamabored_model_inter_token_latency_seconds_sum",
    ),
    (
        "ai.models.counters.itl.n",
        "llamabored_model_inter_token_latency_seconds_count",
    ),
    (
        "ai.models.counters.e2e.ms",
        "llamabored_model_request_duration_seconds_sum",
    ),
    (
        "ai.models.counters.e2e.n",
        "llamabored_model_request_duration_seconds_count",
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
    (
        "ai.models.detail.kv_block",
        "llamabored_model_kv_block_size_tokens",
    ),
    (
        "ai.models.detail.prefix_cache",
        "llamabored_model_prefix_caching",
    ),
    (
        "ai.models.engine.expert_hit",
        "llamabored_model_expert_cache_hit_ratio",
    ),
    (
        "ai.models.engine.pcie_share",
        "llamabored_model_pcie_share_ratio",
    ),
    (
        "ai.models.slot_ctx.slot",
        "llamabored_slot_context_used_tokens",
    ),
    (
        "ai.models.slot_ctx.used",
        "llamabored_slot_context_used_tokens",
    ),
    (
        "ai.models.slot_ctx.resets.compacted",
        "llamabored_slot_context_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.new",
        "llamabored_slot_context_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.evicted",
        "llamabored_slot_context_resets_total",
    ),
    (
        "ai.models.slot_ctx.resets.unknown",
        "llamabored_slot_context_resets_total",
    ),
    (
        "suspected_loads.model",
        "llamabored_collector_suspected_loads_total",
    ),
    (
        "suspected_loads.count",
        "llamabored_collector_suspected_loads_total",
    ),
    ("fans.channel", "llamabored_fan_rpm"),
    ("fans.label", "llamabored_fan_rpm"),
    ("fans.rpm", "llamabored_fan_rpm"),
    ("fans.pwm", "llamabored_fan_pwm_ratio"),
    ("fan_rows.c", "llamabored_fan_rpm"),
    ("fan_rows.n", "llamabored_fan_rpm"),
    ("fan_rows.l", "llamabored_fan_rpm"),
    ("fan_rows.r", "llamabored_fan_rpm"),
    ("fan_rows.p", "llamabored_fan_pwm_ratio"),
    ("temps.c", "llamabored_temperature_celsius"),
    ("temps.s", "llamabored_temperature_celsius"),
    ("temps.t", "llamabored_temperature_celsius"),
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
        "host.mem_pct",
        "#71: derived; memory_used_bytes / memory_total_bytes",
    ),
    (
        "tokens.decoded_total",
        "#71: the LCD's box counter; sum of llamabored_model_generation_tokens_total",
    ),
    (
        "tokens.prompt_total",
        "#71: box total; sum of llamabored_model_prompt_tokens_total",
    ),
    (
        "ai.models.engine.spec_accept",
        "#71: the LCD's window acceptance; rate(spec_accepted) / rate(spec_draft)",
    ),
    (
        "ai.models.detail.ncmoe",
        "#71: launch tuning shown on the LCD and tty; not a model_info string",
    ),
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
                    .any(|(k, v)| k == "model" && v == "qwen3-coder-30b")
                && slot.is_none_or(|slot| labels.iter().any(|(k, v)| k == "slot" && v == slot))
                && labels.iter().all(|(k, v)| k != "reason" || v == "new")
                && labels.iter().all(|(k, v)| k != "status" || v == "ok")
        })
        .map(|(_, _, value)| value)
}

/// #10, #71: the counters are the watcher's, passed through. A watcher
/// restart (new run_id) starts them at 0 again, which Prometheus reads as a
/// counter reset; the exporter neither holds nor stitches them.
#[test]
fn counters_pass_through_and_restart_with_the_watcher() {
    let first = load("snapshot-loaded.json");
    let text = render(&Ok(first.clone()), T0);
    let counters = [
        ("llamabored_model_prompt_tokens_total", None, "1622000"),
        (
            "llamabored_model_prompt_cached_tokens_total",
            None,
            "1500000",
        ),
        ("llamabored_model_generation_tokens_total", None, "52000"),
        ("llamabored_model_requests_total", None, "120"),
        ("llamabored_model_decode_seconds_total", None, "1040.250"),
        ("llamabored_slot_context_resets_total", Some("1"), "2"),
        ("llamabored_slot_context_used_tokens", Some("1"), "91500"),
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
    qwen.counters = Some(wire::CountersWire {
        gen_tokens: Some(0),
        req_ok: Some(0),
        decode_ms: Some(0),
        ..wire::CountersWire::default()
    });
    for row in &mut qwen.slot_ctx {
        row.resets = wire::SlotResetsWire::default();
    }
    let text = render(&Ok(restarted), T0);
    for (metric, slot, _) in &counters[..6] {
        let want = if metric.contains("seconds") {
            "0.000"
        } else {
            "0"
        };
        assert_eq!(
            value_of(&text, metric, *slot).as_deref(),
            Some(want),
            "{metric}"
        );
    }
    assert!(text.contains("# TYPE llamabored_slot_context_resets_total counter"));
    assert!(text.contains("# TYPE llamabored_model_prompt_cached_tokens_total counter"));
    assert!(text.contains("# TYPE llamabored_slot_context_used_tokens gauge"));
}

/// #70: llama-metrics renders only what the snapshot holds. With no model
/// running (llama-swap idle, or every model unloaded), no model series is
/// exported, the exposition still parses, `llamabored_ai_state` is there,
/// and the scrape is a 200.
#[test]
fn no_model_running_is_a_valid_scrape_without_model_series() {
    let mut snap = load("snapshot-loaded.json");
    snap.ai.state = wire::AiWire::Idle;
    snap.ai.models.clear();
    snap.suspected_loads.clear();
    snap.tokens.decoded_total = Some(0);
    wire::validate(&snap).expect("an idle snapshot validates");
    let dir = common::scratch("no-models");
    let path = dir.join("snapshot.json");
    std::fs::write(&path, wire::to_json(&snap).expect("encode")).expect("write");
    let body = SnapshotMetrics::new(SnapshotFile::at(&path), STALE, Box::new(|| T0));
    let harness = common::spawn(
        &["127.0.0.1/32"],
        4,
        llama_metrics::http::Limits::default(),
        std::sync::Arc::new(body),
    );
    let resp = common::exchange(harness.addr, b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(common::status_line(&resp), "HTTP/1.1 200 OK", "{resp}");
    let text = resp.split_once("\r\n\r\n").expect("body").1;
    let all = samples(text);
    assert!(text.contains("llamabored_snapshot_stale 0\n"), "{text}");
    assert!(
        text.contains("llamabored_ai_state{state=\"idle\"} 1\n"),
        "{text}"
    );
    for (name, labels, _) in &all {
        assert!(
            !name.starts_with("llamabored_model_") && !name.starts_with("llamabored_slot_"),
            "{name} with no model running"
        );
        assert!(
            labels
                .iter()
                .all(|(key, _)| key != "model" && key != "engine"),
            "{name} has a model label: {labels:?}"
        );
    }
    let keys: BTreeSet<String> = all.iter().map(|(n, l, _)| format!("{n}{l:?}")).collect();
    assert_eq!(keys.len(), all.len(), "duplicate series");
}

/// #74: every temperature by chip and sensor, fans with their chip, and the
/// older coolant / CPU / GPU series kept for the LCD's dashboards.
#[test]
fn temperatures_and_chip_fans_are_exported() {
    let text = render(&Ok(load("snapshot-loaded.json")), T0);
    let all = samples(&text);
    let temp = |chip: &str, sensor: &str| {
        all.iter()
            .find(|(name, labels, _)| {
                name == "llamabored_temperature_celsius"
                    && labels.contains(&("chip".to_owned(), chip.to_owned()))
                    && labels.contains(&("sensor".to_owned(), sensor.to_owned()))
            })
            .map(|(_, _, value)| value.clone())
    };
    assert_eq!(temp("k10temp", "Tctl").as_deref(), Some("68.3"));
    assert_eq!(temp("gpu", "gpu").as_deref(), Some("71.2"));
    assert_eq!(temp("nvme-317k", "Sensor 2").as_deref(), Some("66.9"));
    assert_eq!(temp("nct6798", "AUXTIN2").as_deref(), Some("15.5"));
    assert!(text.contains("# TYPE llamabored_temperature_celsius gauge"));
    for kept in [
        "llamabored_coolant_celsius",
        "llamabored_cpu_celsius",
        "llamabored_gpu_celsius",
    ] {
        assert!(all.iter().any(|(name, _, _)| name == kept), "{kept}");
    }
    assert!(
        text.contains("llamabored_fan_rpm{chip=\"z53\",channel=\"1\",label=\"Pump\"} 2810"),
        "{text}"
    );
    assert!(
        text.contains("llamabored_fan_pwm_ratio{chip=\"z53\",channel=\"1\",label=\"Pump\"} 0.6"),
        "{text}"
    );
    // An older watcher's fans keep their label set.
    assert!(
        text.contains("llamabored_fan_rpm{channel=\"3\",label=\"rad \\\"top\\\"\"} 1450"),
        "{text}"
    );
    // None at all: no family.
    let mut snap = load("snapshot-loaded.json");
    snap.temps.clear();
    let text = render(&Ok(snap), T0);
    assert!(!text.contains("llamabored_temperature_celsius"));
}
