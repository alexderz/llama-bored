//! Prometheus text exposition format 0.0.4, from one snapshot read.
//!
//! Every metric is named `llamabored_*`. The only strings exported are
//! llama-swap model ids, model display names and their allowlisted tuning
//! and version tokens, the five engine words, fan labels (short printable
//! ASCII from `[fans]`), hwmon chip and sensor names (#74: the chip's
//! `name` and `tempN_label`, short printable ASCII) and fixed source names,
//! as label values. The
//! snapshot carries no prompt or output text, and nothing here could
//! export it: every value is a number from a typed field.
//!
//! A stale or unreadable snapshot exports only the exporter's own series,
//! `llamabored_snapshot_up`, `llamabored_snapshot_stale`, and (when it could be
//! read) its seq and age. The value series disappear rather than freeze.
//!
//! # Normalized series (#71)
//!
//! Every per-model series has the labels `model` (the llama-swap id, the
//! join key) and `engine` (`llamacpp`, `vllm`, `sglang`, `strata`,
//! `openai`), and the same name for every engine that has the quantity;
//! an engine that cannot report one leaves its series absent. Descriptive
//! strings live on `llamabored_model_info` only. Names carry their unit,
//! ratios are 0 to 1, and only counters end in `_total`. A model that is
//! not loaded has no series; host series stay.
//!
//! Latencies are exported as `summary` families with no quantiles: a
//! `<name>_sum` and a `<name>_count` sample under one `# TYPE <name>
//! summary`. That is valid 0.0.4 exposition, and the honest shape of what
//! the engines report (a histogram's sum and count); Prometheus and its
//! tools read the pair as one family, and both parts are counters to
//! `rate()`. Two separate counter families would each need a `_total`
//! name and would lose the pairing.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::Duration;

use llama_core::detail::{self, KV_DEFAULT, ModelDetail};
use llama_core::wire::{
    self, AiWire, EngineWire, FanRowWire, FanWire, KvWire, ModelState, ModelWire, SlotCtxWire,
    SumCountWire, TempWire, WireSnapshot,
};

use crate::snapshot::ReadError;

/// `Content-Type` of the body.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";
/// Longest label value, in characters, after escaping input is cut.
pub const MAX_LABEL_CHARS: usize = 64;
/// `t_mono_ns` may be this far ahead of the exporter's clock and still count.
pub const FUTURE_SLACK_NS: u64 = 50_000_000;

/// Connections refused before a request was read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rejected {
    /// Peer outside the allowlist.
    pub denied: u64,
    /// All `max_conns` workers busy.
    pub busy: u64,
}

/// Everything one scrape renders.
#[derive(Clone, Debug)]
pub struct Scrape<'a> {
    pub read: &'a Result<WireSnapshot, ReadError>,
    /// `CLOCK_MONOTONIC` now, nanoseconds.
    pub now_ns: u64,
    pub stale_after: Duration,
    pub rejected: Rejected,
}

/// True when the snapshot is missing, unreadable, from the future, or older
/// than `stale_after`.
#[must_use]
pub fn is_stale(
    read: &Result<WireSnapshot, ReadError>,
    now_ns: u64,
    stale_after: Duration,
) -> bool {
    let Ok(snap) = read else {
        return true;
    };
    if snap.t_mono_ns > now_ns.saturating_add(FUTURE_SLACK_NS) {
        return true;
    }
    let age = now_ns.saturating_sub(snap.t_mono_ns);
    let limit = u64::try_from(stale_after.as_nanos()).unwrap_or(u64::MAX);
    age > limit
}

/// Escape a label value per the exposition format (`\\`, `\"`, `\n`).
///
/// Other control characters become `?`, and the input is cut at
/// [`MAX_LABEL_CHARS`] characters, so a value is always one short line.
#[must_use]
pub fn escape_label_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars().take(MAX_LABEL_CHARS) {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c if c.is_control() => out.push('?'),
            c => out.push(c),
        }
    }
    out
}

struct Out {
    text: String,
}

/// One sample: its labels and its value text.
type Row<'a> = (Vec<(&'a str, &'a str)>, String);

impl Out {
    fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.text, "# HELP {name} {help}");
        let _ = writeln!(self.text, "# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: &str) {
        self.text.push_str(name);
        if !labels.is_empty() {
            self.text.push('{');
            for (i, (key, val)) in labels.iter().enumerate() {
                if i > 0 {
                    self.text.push(',');
                }
                let _ = write!(self.text, "{key}=\"{}\"", escape_label_value(val));
            }
            self.text.push('}');
        }
        let _ = writeln!(self.text, " {value}");
    }

    /// One unlabelled gauge, when the value is known and finite.
    fn gauge_opt(&mut self, name: &str, help: &str, value: Option<f32>) {
        if let Some(value) = value.filter(|v| v.is_finite()) {
            self.family(name, "gauge", help);
            self.sample(name, &[], &format!("{value}"));
        }
    }

    fn bytes_opt(&mut self, name: &str, help: &str, value: Option<u64>) {
        if let Some(value) = value {
            self.family(name, "gauge", help);
            self.sample(name, &[], &value.to_string());
        }
    }

    /// One family of rows; nothing when `rows` is empty.
    fn rows(&mut self, name: &str, kind: &str, help: &str, rows: &[Row<'_>]) {
        if rows.is_empty() {
            return;
        }
        self.family(name, kind, help);
        for (labels, value) in rows {
            self.sample(name, labels, value);
        }
    }

    /// A summary with no quantiles: `<name>_sum` and `<name>_count` per
    /// label set, under one `# TYPE <name> summary`.
    fn summary(&mut self, name: &str, help: &str, rows: &[(Vec<(&str, &str)>, SumCountWire)]) {
        if rows.is_empty() {
            return;
        }
        self.family(name, "summary", help);
        let sum = format!("{name}_sum");
        let count = format!("{name}_count");
        for (labels, pair) in rows {
            self.sample(&sum, labels, &seconds(pair.ms));
            self.sample(&count, labels, &pair.n.to_string());
        }
    }
}

/// An `f32` sample value, as Rust prints it.
fn num(value: f32) -> String {
    format!("{value}")
}

/// Whole milliseconds as seconds, exactly: `1234` is `1.234`.
fn seconds(ms: u64) -> String {
    format!("{}.{:03}", ms / 1000, ms % 1000)
}

/// A 0..=100 percent as a 0..=1 ratio (0..=1.25 for activity), printed
/// as the shortest decimal that reads back as the same `f32`.
fn ratio_of_pct(pct: Option<f32>) -> Option<f32> {
    pct.filter(|v| v.is_finite()).map(|v| v / 100.0)
}

/// `1` or `0`.
fn flag(on: bool) -> String {
    if on { "1" } else { "0" }.to_owned()
}

/// `q8_0` when K and V match, else `q8_0/q4_0`. Absent is `f16`, llama.cpp's
/// default; no detail at all is empty.
fn kv_type(detail: Option<&ModelDetail>) -> String {
    let Some(detail) = detail else {
        return String::new();
    };
    let token = |value: &Option<String>| -> String {
        value
            .as_deref()
            .filter(|t| detail::is_token(t))
            .unwrap_or(KV_DEFAULT)
            .to_owned()
    };
    let k = token(&detail.kv_k);
    let v = token(&detail.kv_v);
    if k == v { k } else { format!("{k}/{v}") }
}

fn model_state_label(state: ModelState) -> &'static str {
    match state {
        ModelState::Ready => "ready",
        ModelState::Starting => "starting",
        ModelState::Stopping => "stopping",
        ModelState::Other => "other",
    }
}

fn ai_label(state: AiWire) -> &'static str {
    match state {
        AiWire::Down => "down",
        AiWire::Idle => "idle",
        AiWire::Loaded => "loaded",
    }
}

/// One exported model: the wire entry and its two key labels.
struct Keyed<'a> {
    wire: &'a ModelWire,
    /// The llama-swap id. An older watcher sends none; its full display
    /// name stands in.
    model: String,
    /// One of five fixed words; an older watcher's model is `llamacpp`.
    engine: &'static str,
}

impl Keyed<'_> {
    fn labels(&self) -> Vec<(&str, &str)> {
        vec![("model", self.model.as_str()), ("engine", self.engine)]
    }
}

/// The models to export, sorted by `model`; the first with a given id
/// wins, so a repeated id never repeats a label set.
fn keyed(models: &[ModelWire]) -> Vec<Keyed<'_>> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<Keyed<'_>> = Vec::new();
    for wire in models {
        let model = wire
            .id
            .clone()
            .or_else(|| wire.full_name.clone())
            .unwrap_or_else(|| wire.name.clone());
        if seen.insert(model.clone()) {
            out.push(Keyed {
                wire,
                model,
                engine: wire.backend.unwrap_or_default().as_str(),
            });
        }
    }
    out.sort_by(|a, b| a.model.cmp(&b.model));
    out
}

/// Picks one model's value for a per-model family.
type ModelValue = fn(&ModelWire) -> Option<String>;

fn engine(model: &ModelWire) -> Option<&EngineWire> {
    model.engine.as_ref()
}

fn counters(model: &ModelWire) -> Option<&wire::CountersWire> {
    model.counters.as_ref()
}

fn kv(model: &ModelWire) -> Option<&KvWire> {
    model.kv.as_ref()
}

/// Per-model families with one sample per model, in export order: the
/// first ten, then the requests and the latency summaries, then the rest.
const MODEL_FAMILIES: [(&str, &str, &str, ModelValue); 25] = [
    (
        "llamabored_model_context_size_tokens",
        "gauge",
        "Configured context size, tokens.",
        |m| {
            m.detail
                .as_ref()
                .and_then(|d| d.ctx)
                .filter(|c| *c > 0)
                .map(|c| c.to_string())
        },
    ),
    (
        "llamabored_model_requests_running",
        "gauge",
        "Requests the engine is running now.",
        |m| m.running.map(|n| n.to_string()),
    ),
    (
        "llamabored_model_requests_waiting",
        "gauge",
        "Requests waiting for the engine (llama.cpp: deferred).",
        |m| m.queued.map(|n| n.to_string()),
    ),
    (
        "llamabored_model_slots",
        "gauge",
        "Requests the engine can run at once: llama.cpp slots, else its configured cap.",
        |m| m.slots_total.or(m.max_running).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_kv_cache_usage_ratio",
        "gauge",
        "KV cache fill across all sessions, 0 to 1: used / capacity tokens, else the engine's own ratio.",
        |m| {
            kv(m)
                .and_then(KvWire::ratio)
                .map(|ratio| ratio as f32)
                .or(m.kv_fill)
                .filter(|v| v.is_finite())
                .map(num)
        },
    ),
    (
        "llamabored_model_prompt_tokens_total",
        "counter",
        "Prompt tokens of finished requests, cached ones included.",
        |m| m.prompt_tokens.map(|n| n.to_string()),
    ),
    (
        "llamabored_model_prompt_cached_tokens_total",
        "counter",
        "Prompt tokens served from the prompt or prefix cache.",
        |m| m.prompt_cached_tokens.map(|n| n.to_string()),
    ),
    (
        "llamabored_model_generation_tokens_total",
        "counter",
        "Tokens generated.",
        |m| {
            counters(m)
                .and_then(|c| c.gen_tokens)
                .map(|n| n.to_string())
        },
    ),
    (
        "llamabored_model_prefill_seconds_total",
        "counter",
        "Seconds spent on prompt processing (prefill).",
        |m| counters(m).and_then(|c| c.prefill_ms).map(seconds),
    ),
    (
        "llamabored_model_decode_seconds_total",
        "counter",
        "Seconds spent generating (decode).",
        |m| counters(m).and_then(|c| c.decode_ms).map(seconds),
    ),
    (
        "llamabored_model_spec_draft_tokens_total",
        "counter",
        "Speculative decoding: tokens drafted.",
        |m| {
            engine(m)
                .and_then(|e| e.spec_draft_tokens)
                .map(|n| n.to_string())
        },
    ),
    (
        "llamabored_model_spec_accepted_tokens_total",
        "counter",
        "Speculative decoding: drafted tokens accepted.",
        |m| {
            engine(m)
                .and_then(|e| e.spec_accepted_tokens)
                .map(|n| n.to_string())
        },
    ),
    (
        "llamabored_model_spec_drafts_total",
        "counter",
        "Speculative decoding: draft rounds, where the engine counts them.",
        |m| engine(m).and_then(|e| e.spec_drafts).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_preemptions_total",
        "counter",
        "Requests the engine preempted; a rising count means KV cache pressure.",
        |m| engine(m).and_then(|e| e.preemptions).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_sleeping",
        "gauge",
        "1 when the engine is asleep, 0 when awake.",
        |m| engine(m).and_then(|e| e.sleeping).map(flag),
    ),
    (
        "llamabored_model_kv_block_size_tokens",
        "gauge",
        "KV cache block size, tokens.",
        |m| {
            m.detail
                .as_ref()
                .and_then(|d| d.kv_block)
                .map(|n| n.to_string())
        },
    ),
    (
        "llamabored_model_prefix_caching",
        "gauge",
        "1 when prefix caching is on, 0 when off.",
        |m| m.detail.as_ref().and_then(|d| d.prefix_cache).map(flag),
    ),
    (
        "llamabored_model_expert_cache_hit_ratio",
        "gauge",
        "Expert cache hit rate of the newest finished request, 0 to 1.",
        |m| {
            engine(m)
                .and_then(|e| e.expert_hit)
                .filter(|v| v.is_finite())
                .map(num)
        },
    ),
    (
        "llamabored_model_pcie_share_ratio",
        "gauge",
        "Share of the newest finished request's expert reads served over PCIe, 0 to 1.",
        |m| {
            engine(m)
                .and_then(|e| e.pcie_share)
                .filter(|v| v.is_finite())
                .map(num)
        },
    ),
    // #79: the KV cache across all sessions.
    (
        "llamabored_model_kv_used_tokens",
        "gauge",
        "KV cache tokens held for live sessions, idle llama.cpp and Strata slots included (vLLM: its usage ratio times the capacity, block-rounded).",
        |m| kv(m).and_then(|kv| kv.used).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_kv_capacity_tokens",
        "gauge",
        "KV cache capacity, tokens (llama.cpp with a unified cache: the one pool every slot shares).",
        |m| kv(m).and_then(|kv| kv.capacity).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_kv_cached_tokens",
        "gauge",
        "KV cache tokens kept only as reusable prefix cache, not held by a session (SGLang).",
        |m| kv(m).and_then(|kv| kv.cached).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_kv_sessions",
        "gauge",
        "Sessions holding KV cache: slots with tokens (llama.cpp, Strata), else running requests.",
        |m| kv(m).and_then(|kv| kv.sessions).map(|n| n.to_string()),
    ),
    (
        "llamabored_model_kv_unified",
        "gauge",
        "llama.cpp: 1 when every slot shares one KV pool, 0 when each slot has its own.",
        |m| kv(m).and_then(|kv| kv.unified).map(flag),
    ),
    (
        "llamabored_model_inflight_requests",
        "gauge",
        "Requests llama-swap has in flight for the model, from its event stream while a vLLM or SGLang model is loaded.",
        |m| m.inflight.map(|n| n.to_string()),
    ),
];

/// Latency summaries: name, help, and the pair a model carries.
type ModelPair = fn(&ModelWire) -> Option<SumCountWire>;
const SUMMARIES: [(&str, &str, ModelPair); 3] = [
    (
        "llamabored_model_time_to_first_token_seconds",
        "Time to first token: seconds summed, and requests.",
        |m| counters(m).and_then(|c| c.ttft),
    ),
    (
        "llamabored_model_inter_token_latency_seconds",
        "Inter-token latency: seconds summed, and tokens.",
        |m| counters(m).and_then(|c| c.itl),
    ),
    (
        "llamabored_model_request_duration_seconds",
        "Request duration: seconds summed, and requests.",
        |m| counters(m).and_then(|c| c.e2e),
    ),
];

/// Render one scrape.
#[must_use]
pub fn render(scrape: &Scrape<'_>) -> String {
    let mut out = Out {
        text: String::with_capacity(8192),
    };
    out.family(
        "llamabored_exporter_build_info",
        "gauge",
        "llama-metrics build; always 1.",
    );
    let schema = wire::SCHEMA.to_string();
    out.sample(
        "llamabored_exporter_build_info",
        &[
            ("version", env!("CARGO_PKG_VERSION")),
            ("wire_schema", &schema),
        ],
        "1",
    );
    out.family(
        "llamabored_exporter_rejected_connections_total",
        "counter",
        "Connections closed before a request was read, by reason.",
    );
    out.sample(
        "llamabored_exporter_rejected_connections_total",
        &[("reason", "busy")],
        &scrape.rejected.busy.to_string(),
    );
    out.sample(
        "llamabored_exporter_rejected_connections_total",
        &[("reason", "denied")],
        &scrape.rejected.denied.to_string(),
    );

    let stale = is_stale(scrape.read, scrape.now_ns, scrape.stale_after);
    out.family(
        "llamabored_snapshot_up",
        "gauge",
        "1 when the snapshot file was read and validated.",
    );
    out.sample(
        "llamabored_snapshot_up",
        &[],
        if scrape.read.is_ok() { "1" } else { "0" },
    );
    out.family(
        "llamabored_snapshot_stale",
        "gauge",
        "1 when the snapshot is missing, invalid, or older than stale_after_s; value series are then omitted.",
    );
    out.sample(
        "llamabored_snapshot_stale",
        &[],
        if stale { "1" } else { "0" },
    );

    let Ok(snap) = scrape.read else {
        return out.text;
    };
    out.family(
        "llamabored_snapshot_seq",
        "gauge",
        "Snapshot sequence number within the watcher run.",
    );
    out.sample("llamabored_snapshot_seq", &[], &snap.seq.to_string());
    let age_ns = scrape.now_ns.saturating_sub(snap.t_mono_ns);
    out.family(
        "llamabored_snapshot_age_seconds",
        "gauge",
        "Seconds since the watcher sampled the snapshot.",
    );
    out.sample(
        "llamabored_snapshot_age_seconds",
        &[],
        &format!("{:.3}", age_ns as f64 / 1e9),
    );
    if stale {
        return out.text;
    }

    host(&mut out, snap);
    // #70: loads llama-watch suspects its own reads caused. The watcher
    // sends each id once, so every row is its own series.
    let suspects: Vec<Row<'_>> = snap
        .suspected_loads
        .iter()
        .map(|row| (vec![("model", row.model.as_str())], row.count.to_string()))
        .collect();
    out.rows(
        "llamabored_collector_suspected_loads_total",
        "counter",
        "Model loads llama-watch suspects its own llama-swap reads caused, by llama-swap model id.",
        &suspects,
    );

    out.family(
        "llamabored_ai_state",
        "gauge",
        "llama-swap state; the current state is 1.",
    );
    for state in [AiWire::Down, AiWire::Idle, AiWire::Loaded] {
        let value = if snap.ai.state == state { "1" } else { "0" };
        out.sample("llamabored_ai_state", &[("state", ai_label(state))], value);
    }

    let models = keyed(&snap.ai.models);
    per_model(&mut out, &models);
    per_slot(&mut out, &models);
    fans(&mut out, &snap.fans, &snap.fan_rows);
    temperatures(&mut out, &snap.temps);

    if let Some(sources) = &snap.sources {
        let entries = sources.entries();
        let up: Vec<Row<'_>> = entries
            .iter()
            .filter_map(|(name, source)| {
                let source = (*source)?;
                Some((vec![("source", *name)], flag(source.up)))
            })
            .collect();
        out.rows(
            "llamabored_source_up",
            "gauge",
            "1 when a watcher source answered on its last poll, 0 when it failed; absent when not polled.",
            &up,
        );
        let latency: Vec<Row<'_>> = entries
            .iter()
            .filter_map(|(name, source)| {
                let seconds = (*source)?.latency_s.filter(|v| v.is_finite())?;
                Some((vec![("source", *name)], num(seconds)))
            })
            .collect();
        out.rows(
            "llamabored_source_latency_seconds",
            "gauge",
            "Duration of a watcher source's last poll, seconds.",
            &latency,
        );
    }
    out.text
}

/// Host series: ratios 0 to 1, temperatures, power and bytes.
fn host(out: &mut Out, snap: &WireSnapshot) {
    let host = &snap.host;
    out.gauge_opt(
        "llamabored_activity_ratio",
        "Power-weighted activity, 0 to 1.25; 1 is nominal sustained load.",
        ratio_of_pct(host.activity_pct),
    );
    out.gauge_opt(
        "llamabored_load_ratio",
        "Composite load, max(GPU, CPU top-k utilization), 0 to 1.",
        ratio_of_pct(host.load_pct),
    );
    out.gauge_opt(
        "llamabored_cpu_utilization_ratio",
        "Mean CPU utilization, 0 to 1.",
        ratio_of_pct(host.cpu_pct),
    );
    out.gauge_opt(
        "llamabored_cpu_topk_utilization_ratio",
        "Mean utilization of the busiest CPUs, 0 to 1.",
        ratio_of_pct(host.cpu_topk_pct),
    );
    out.gauge_opt(
        "llamabored_gpu_utilization_ratio",
        "GPU utilization, 0 to 1.",
        ratio_of_pct(host.gpu_pct),
    );
    out.gauge_opt(
        "llamabored_coolant_celsius",
        "Coolant temperature, degrees Celsius.",
        host.coolant_c,
    );
    out.gauge_opt(
        "llamabored_cpu_celsius",
        "CPU temperature, degrees Celsius.",
        host.cpu_c,
    );
    out.gauge_opt(
        "llamabored_gpu_celsius",
        "GPU temperature, degrees Celsius.",
        host.gpu_c,
    );
    out.gauge_opt(
        "llamabored_gpu_power_watts",
        "GPU power draw, watts.",
        host.gpu_w,
    );
    out.gauge_opt(
        "llamabored_gpu_power_limit_watts",
        "GPU enforced power limit, watts.",
        host.gpu_limit_w,
    );
    out.gauge_opt(
        "llamabored_cpu_power_watts",
        "CPU socket power, watts.",
        host.cpu_w,
    );
    out.bytes_opt(
        "llamabored_gpu_memory_used_bytes",
        "GPU memory in use, bytes.",
        host.vram_used_bytes,
    );
    out.bytes_opt(
        "llamabored_gpu_memory_total_bytes",
        "GPU memory total, bytes.",
        host.vram_total_bytes,
    );
    out.bytes_opt(
        "llamabored_memory_used_bytes",
        "System memory in use (MemTotal - MemAvailable), bytes.",
        host.mem_used_bytes,
    );
    out.bytes_opt(
        "llamabored_memory_total_bytes",
        "System memory total, bytes.",
        host.mem_total_bytes,
    );
}

/// Every per-model family (#71), labelled `model` and `engine`.
fn per_model(out: &mut Out, models: &[Keyed<'_>]) {
    if models.is_empty() {
        return;
    }
    // Descriptive strings, once per model.
    let infos: Vec<[String; 4]> = models
        .iter()
        .map(|m| {
            let detail = m.wire.detail.as_ref();
            [
                m.wire
                    .full_name
                    .clone()
                    .unwrap_or_else(|| m.wire.name.clone()),
                detail
                    .and_then(|d| d.quant.clone())
                    .filter(|q| detail::is_token(q))
                    .unwrap_or_default(),
                kv_type(detail),
                m.wire
                    .version
                    .clone()
                    .filter(|v| detail::is_token(v))
                    .unwrap_or_default(),
            ]
        })
        .collect();
    let info_rows: Vec<Row<'_>> = models
        .iter()
        .zip(&infos)
        .map(|(m, [display, quant, kv, version])| {
            let mut labels = m.labels();
            labels.extend([
                ("display_name", display.as_str()),
                ("quant", quant.as_str()),
                ("kv_type", kv.as_str()),
                ("version", version.as_str()),
            ]);
            (labels, "1".to_owned())
        })
        .collect();
    out.rows(
        "llamabored_model_info",
        "gauge",
        "A model llama-swap lists, with its descriptive strings; always 1.",
        &info_rows,
    );
    let state_rows: Vec<Row<'_>> = models
        .iter()
        .flat_map(|m| {
            let current = model_state_label(m.wire.state);
            ["ready", "starting", "stopping", "other"].map(|state| {
                let mut labels = m.labels();
                labels.push(("state", state));
                (labels, flag(state == current))
            })
        })
        .collect();
    out.rows(
        "llamabored_model_state",
        "gauge",
        "Lifecycle of a listed model; the current state is 1.",
        &state_rows,
    );
    for (name, kind, help, value) in MODEL_FAMILIES.iter().take(10) {
        model_family(out, models, name, kind, help, *value);
    }
    // Requests by outcome, from llama-swap's activity rows.
    let status_rows: Vec<Row<'_>> = models
        .iter()
        .flat_map(|m| {
            let c = counters(m.wire);
            [
                ("ok", c.and_then(|c| c.req_ok)),
                ("error", c.and_then(|c| c.req_err)),
            ]
            .into_iter()
            .filter_map(|(status, count)| {
                let mut labels = m.labels();
                labels.push(("status", status));
                Some((labels, count?.to_string()))
            })
            .collect::<Vec<_>>()
        })
        .collect();
    out.rows(
        "llamabored_model_requests_total",
        "counter",
        "Finished requests by outcome: ok (2xx) or error.",
        &status_rows,
    );
    for (name, help, pair) in SUMMARIES {
        let rows: Vec<(Vec<(&str, &str)>, SumCountWire)> = models
            .iter()
            .filter_map(|m| Some((m.labels(), pair(m.wire)?)))
            .collect();
        out.summary(name, help, &rows);
    }
    for (name, kind, help, value) in MODEL_FAMILIES.iter().skip(10) {
        model_family(out, models, name, kind, help, *value);
    }
}

fn model_family(
    out: &mut Out,
    models: &[Keyed<'_>],
    name: &str,
    kind: &str,
    help: &str,
    value: ModelValue,
) {
    let rows: Vec<Row<'_>> = models
        .iter()
        .filter_map(|m| Some((m.labels(), value(m.wire)?)))
        .collect();
    out.rows(name, kind, help, &rows);
}

/// Per-slot context (#10), labelled `model`, `engine` and `slot`; the wire
/// keeps slot ids unique within a model.
fn per_slot(out: &mut Out, models: &[Keyed<'_>]) {
    let slot_rows: Vec<(&Keyed<'_>, &SlotCtxWire, String)> = models
        .iter()
        .flat_map(|m| {
            let mut rows: Vec<&SlotCtxWire> = m.wire.slot_ctx.iter().collect();
            rows.sort_by_key(|row| row.slot);
            rows.into_iter()
                .map(move |row| (m, row, row.slot.to_string()))
        })
        .collect();
    let used: Vec<Row<'_>> = slot_rows
        .iter()
        .map(|(m, row, slot)| {
            let mut labels = m.labels();
            labels.push(("slot", slot.as_str()));
            (labels, row.used.to_string())
        })
        .collect();
    out.rows(
        "llamabored_slot_context_used_tokens",
        "gauge",
        "Context tokens a llama.cpp slot holds (prompt plus generated; an idle slot keeps its last value).",
        &used,
    );
    // #9: every reason on every slot, so a rate starts from 0.
    let resets: Vec<Row<'_>> = slot_rows
        .iter()
        .flat_map(|(m, row, slot)| {
            row.resets.entries().map(|(reason, count)| {
                let mut labels = m.labels();
                labels.extend([("slot", slot.as_str()), ("reason", reason)]);
                (labels, count.to_string())
            })
        })
        .collect();
    out.rows(
        "llamabored_slot_context_resets_total",
        "counter",
        "Context drops of a llama.cpp slot, by best-guess reason.",
        &resets,
    );
}

/// Fans: `fan_rows` (#74) carry their chip; `fans` from an older watcher do
/// not, and keep their old label set. The wire refuses a repeated channel
/// (per chip), so each label set is unique.
fn fans(out: &mut Out, fans: &[FanWire], fan_rows: &[FanRowWire]) {
    struct Fan<'a> {
        chip: Option<&'a str>,
        channel: String,
        label: &'a str,
        rpm: Option<u32>,
        pwm: Option<f32>,
    }
    let mut all: Vec<Fan<'_>> = fans
        .iter()
        .map(|fan| Fan {
            chip: None,
            channel: fan.channel.to_string(),
            label: &fan.label,
            rpm: fan.rpm,
            pwm: fan.pwm.filter(|v| v.is_finite()),
        })
        .chain(fan_rows.iter().map(|fan| Fan {
            chip: Some(&fan.chip),
            channel: fan.channel.to_string(),
            label: &fan.label,
            rpm: fan.rpm,
            pwm: fan.pwm.map(|p| f32::from(p) / 255.0),
        }))
        .collect();
    all.sort_by(|a, b| {
        (a.chip, a.channel.len(), &a.channel).cmp(&(b.chip, b.channel.len(), &b.channel))
    });
    let fan_rows = |value: &dyn Fn(&Fan<'_>) -> Option<String>| -> Vec<Row<'_>> {
        all.iter()
            .filter_map(|fan| {
                let mut labels = Vec::with_capacity(3);
                if let Some(chip) = fan.chip {
                    labels.push(("chip", chip));
                }
                labels.extend([("channel", fan.channel.as_str()), ("label", fan.label)]);
                Some((labels, value(fan)?))
            })
            .collect()
    };
    out.rows(
        "llamabored_fan_rpm",
        "gauge",
        "Fan speed, rpm, per fan (read only).",
        &fan_rows(&|fan| fan.rpm.map(|rpm| rpm.to_string())),
    );
    out.rows(
        "llamabored_fan_pwm_ratio",
        "gauge",
        "Fan PWM duty, 0 to 1, per fan (read only).",
        &fan_rows(&|fan| fan.pwm.map(num)),
    );
}

/// Every temperature the watcher shows (#74), by chip and sensor; the wire
/// refuses a repeated pair. The coolant, CPU and GPU series stay as well.
fn temperatures(out: &mut Out, temps: &[TempWire]) {
    let values: Vec<String> = temps.iter().map(|t| tenths(t.tenths)).collect();
    let rows: Vec<Row<'_>> = temps
        .iter()
        .zip(&values)
        .map(|(t, value)| {
            (
                vec![("chip", t.chip.as_str()), ("sensor", t.sensor.as_str())],
                value.clone(),
            )
        })
        .collect();
    out.rows(
        "llamabored_temperature_celsius",
        "gauge",
        "Temperature, degrees Celsius, per hwmon chip and sensor (and the GPU).",
        &rows,
    );
}

/// Tenths of a degree as an exact decimal: `-25` is `-2.5`.
fn tenths(t: i16) -> String {
    let sign = if t < 0 { "-" } else { "" };
    let abs = t.unsigned_abs();
    format!("{sign}{}.{}", abs / 10, abs % 10)
}
