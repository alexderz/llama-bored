//! Prometheus text exposition format 0.0.4, from one snapshot read.
//!
//! Every metric is named `llamabored_*`. The only strings exported are model
//! display names and their allowlisted tuning tokens, fan labels (short
//! printable ASCII from `[fans]`) and fixed source names, as label values.
//! The snapshot carries no prompt or output text, and nothing here could
//! export it: every value is a number from a typed field.
//!
//! A stale or unreadable snapshot exports only the exporter's own series,
//! `llamabored_snapshot_up`, `llamabored_snapshot_stale`, and (when it could be
//! read) its seq and age. The value series disappear rather than freeze.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::Duration;

use llama_core::detail::{self, KV_DEFAULT, ModelDetail, NCMOE_ALL};
use llama_core::wire::{self, AiWire, FanWire, ModelState, ModelWire, SlotCtxWire, WireSnapshot};

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

    /// One family of `(labels, value)` rows; nothing when `rows` is empty.
    fn rows(&mut self, name: &str, kind: &str, help: &str, rows: &[(Vec<(&str, &str)>, String)]) {
        if rows.is_empty() {
            return;
        }
        self.family(name, kind, help);
        for (labels, value) in rows {
            self.sample(name, labels, value);
        }
    }
}

/// An `f32` sample value, printed like [`Out::gauge_opt`] prints one.
fn num(value: f32) -> String {
    format!("{value}")
}

/// Picks one model's value for a per-model family.
type ModelValue = fn(&ModelWire) -> Option<String>;

/// One value per model, keyed by `(name, full_name)` like
/// `llamabored_model_ctx_size_tokens`: sorted, and the first model with a
/// given key wins, so a repeated name never repeats a label set.
fn per_model(
    models: &[ModelWire],
    value: impl Fn(&ModelWire) -> Option<String>,
) -> Vec<(String, String, String)> {
    let mut rows: BTreeSet<(String, String, String)> = BTreeSet::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for model in models {
        let Some(value) = value(model) else {
            continue;
        };
        let labels = ModelLabels::of(model);
        if seen.insert((labels.name.clone(), labels.full_name.clone())) {
            rows.insert((labels.name, labels.full_name, value));
        }
    }
    rows.into_iter().collect()
}

/// Model labels, in export order.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ModelLabels {
    pub name: String,
    pub full_name: String,
    pub quant: String,
    pub kv: String,
    pub ctx: String,
    pub moe: String,
    /// One of four fixed words (T72); an older watcher's model is `llamacpp`.
    pub backend: &'static str,
}

impl ModelLabels {
    /// Labels for one wire model. Absent detail is an empty value.
    #[must_use]
    pub fn of(model: &ModelWire) -> Self {
        let detail = model.detail.as_ref();
        Self {
            name: model.name.clone(),
            full_name: model
                .full_name
                .clone()
                .unwrap_or_else(|| model.name.clone()),
            quant: detail
                .and_then(|d| d.quant.as_deref())
                .filter(|q| detail::is_token(q))
                .unwrap_or("")
                .to_owned(),
            kv: detail.map(kv_label).unwrap_or_default(),
            ctx: detail
                .and_then(|d| d.ctx)
                .filter(|c| *c > 0)
                .map(|c| c.to_string())
                .unwrap_or_default(),
            moe: match detail.and_then(|d| d.ncmoe) {
                None => String::new(),
                Some(NCMOE_ALL) => "all".to_owned(),
                Some(n) => n.to_string(),
            },
            backend: model.backend.unwrap_or_default().as_str(),
        }
    }

    fn pairs(&self) -> [(&str, &str); 7] {
        [
            ("name", &self.name),
            ("full_name", &self.full_name),
            ("quant", &self.quant),
            ("kv", &self.kv),
            ("ctx", &self.ctx),
            ("moe", &self.moe),
            ("backend", self.backend),
        ]
    }
}

/// `q8_0` when K and V match, else `q8_0/q4_0`. Absent is `f16`.
fn kv_label(detail: &ModelDetail) -> String {
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

/// Render one scrape.
#[must_use]
pub fn render(scrape: &Scrape<'_>) -> String {
    let mut out = Out {
        text: String::with_capacity(4096),
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

    let host = &snap.host;
    out.gauge_opt(
        "llamabored_activity_pct",
        "Power-weighted activity percent, 0..125; 100 is nominal sustained load.",
        host.activity_pct,
    );
    out.gauge_opt(
        "llamabored_load_pct",
        "Composite load percent, max(gpu, cpu top-k).",
        host.load_pct,
    );
    out.gauge_opt("llamabored_cpu_pct", "Mean CPU percent.", host.cpu_pct);
    out.gauge_opt(
        "llamabored_cpu_topk_pct",
        "Mean of the busiest CPUs, percent.",
        host.cpu_topk_pct,
    );
    out.gauge_opt(
        "llamabored_gpu_pct",
        "GPU utilisation percent.",
        host.gpu_pct,
    );
    out.gauge_opt("llamabored_mem_pct", "Memory percent.", host.mem_pct);
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

    if let Some(total) = snap.tokens.decoded_total {
        out.family(
            "llamabored_tokens_decoded_total",
            "counter",
            "Tokens decoded since the watcher started.",
        );
        out.sample("llamabored_tokens_decoded_total", &[], &total.to_string());
    }
    if let Some(total) = snap.tokens.prompt_total {
        out.family(
            "llamabored_tokens_prompt_total",
            "counter",
            "Prompt tokens processed since the watcher started.",
        );
        out.sample("llamabored_tokens_prompt_total", &[], &total.to_string());
    }

    out.family(
        "llamabored_ai_state",
        "gauge",
        "llama-swap state; the current state is 1.",
    );
    for state in [AiWire::Down, AiWire::Idle, AiWire::Loaded] {
        let value = if snap.ai.state == state { "1" } else { "0" };
        out.sample("llamabored_ai_state", &[("state", ai_label(state))], value);
    }

    // Sorted and de-duplicated: a repeated label set would fail the scrape.
    let models: BTreeSet<ModelLabels> = snap.ai.models.iter().map(ModelLabels::of).collect();
    if !models.is_empty() {
        out.family(
            "llamabored_model_loaded",
            "gauge",
            "A model llama-swap has loaded, with its tuning detail.",
        );
        for labels in &models {
            out.sample("llamabored_model_loaded", &labels.pairs(), "1");
        }
    }
    let models = &snap.ai.models;
    let states = per_model(models, |m| Some(model_state_label(m.state).to_owned()));
    if !states.is_empty() {
        out.family(
            "llamabored_model_state",
            "gauge",
            "Lifecycle of a loaded model; the current state is 1.",
        );
        for (name, full_name, current) in &states {
            for state in ["ready", "starting", "stopping", "other"] {
                out.sample(
                    "llamabored_model_state",
                    &[("name", name), ("full_name", full_name), ("state", state)],
                    if state == current { "1" } else { "0" },
                );
            }
        }
    }
    let model_families: [(&str, &str, &str, ModelValue); 9] = [
        (
            "llamabored_model_ctx_size_tokens",
            "gauge",
            "Configured context size of a loaded model, tokens.",
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
            "Requests a loaded model is running now (SGLang, vLLM and other backends without slots).",
            |m| m.running.map(|n| n.to_string()),
        ),
        (
            "llamabored_model_requests_queued",
            "gauge",
            "Requests waiting for a loaded model (SGLang, vLLM and other backends without slots).",
            |m| m.queued.map(|n| n.to_string()),
        ),
        (
            "llamabored_model_kv_cache_usage_ratio",
            "gauge",
            "KV cache fill of a loaded model, 0 to 1 (SGLang, vLLM and other backends without slots).",
            |m| m.kv_fill.filter(|v| v.is_finite()).map(num),
        ),
        (
            "llamabored_model_cache_hit_ratio",
            "gauge",
            "Prefix cache hit ratio of a loaded model, 0 to 1 (SGLang, vLLM).",
            |m| m.cache_hit.filter(|v| v.is_finite()).map(num),
        ),
        (
            "llamabored_slots_busy",
            "gauge",
            "llama.cpp slots of a loaded model that are processing.",
            |m| m.slots_busy.map(|n| n.to_string()),
        ),
        (
            "llamabored_slots_total",
            "gauge",
            "llama.cpp slots of a loaded model.",
            |m| m.slots_total.map(|n| n.to_string()),
        ),
        (
            "llamabored_model_prompt_tokens_total",
            "counter",
            "Prompt tokens of a model's finished requests since the watcher started, cached ones included.",
            |m| m.prompt_tokens.map(|n| n.to_string()),
        ),
        (
            "llamabored_model_prompt_cached_tokens_total",
            "counter",
            "Prompt tokens served from the prompt cache since the watcher started; hit ratio = rate of this / rate of prompt_tokens_total.",
            |m| m.prompt_cached_tokens.map(|n| n.to_string()),
        ),
    ];
    for (name, kind, help, value) in model_families {
        let keyed = per_model(models, value);
        let rows: Vec<(Vec<(&str, &str)>, String)> = keyed
            .iter()
            .map(|(model, full, value)| {
                (
                    vec![("name", model.as_str()), ("full_name", full.as_str())],
                    value.clone(),
                )
            })
            .collect();
        out.rows(name, kind, help, &rows);
    }

    // Per-slot context (#10): the first model with a given (name,
    // full_name) wins, as in `per_model`; the wire keeps slot ids unique.
    let mut slot_rows: Vec<(ModelLabels, &SlotCtxWire)> = Vec::new();
    let mut slot_models: BTreeSet<(String, String)> = BTreeSet::new();
    for model in models.iter().filter(|m| !m.slot_ctx.is_empty()) {
        let labels = ModelLabels::of(model);
        if !slot_models.insert((labels.name.clone(), labels.full_name.clone())) {
            continue;
        }
        slot_rows.extend(model.slot_ctx.iter().map(|row| (labels.clone(), row)));
    }
    slot_rows.sort_by(|a, b| {
        (&a.0.name, &a.0.full_name, a.1.slot).cmp(&(&b.0.name, &b.0.full_name, b.1.slot))
    });
    let slot_ids: Vec<String> = slot_rows
        .iter()
        .map(|(_, row)| row.slot.to_string())
        .collect();
    let used: Vec<(Vec<(&str, &str)>, String)> = slot_rows
        .iter()
        .zip(&slot_ids)
        .map(|((labels, row), slot)| {
            (
                vec![
                    ("name", labels.name.as_str()),
                    ("full_name", labels.full_name.as_str()),
                    ("slot", slot.as_str()),
                ],
                row.used.to_string(),
            )
        })
        .collect();
    out.rows(
        "llamabored_slot_ctx_used_tokens",
        "gauge",
        "Context tokens a llama.cpp slot holds (prompt plus decoded; an idle slot keeps its last value).",
        &used,
    );
    // #9: every reason on every slot, so a rate starts from 0.
    let resets: Vec<(Vec<(&str, &str)>, String)> = slot_rows
        .iter()
        .zip(&slot_ids)
        .flat_map(|((labels, row), slot)| {
            row.resets.entries().map(|(reason, count)| {
                (
                    vec![
                        ("name", labels.name.as_str()),
                        ("full_name", labels.full_name.as_str()),
                        ("slot", slot.as_str()),
                        ("reason", reason),
                    ],
                    count.to_string(),
                )
            })
        })
        .collect();
    out.rows(
        "llamabored_slot_ctx_resets_total",
        "counter",
        "Context drops of a llama.cpp slot since the watcher started, by best-guess reason.",
        &resets,
    );

    // Fans: the wire already refuses a repeated channel, so each label set
    // is unique; the channel is the key and the label rides along.
    let mut fans: Vec<&FanWire> = snap.fans.iter().collect();
    fans.sort_by_key(|fan| fan.channel);
    let channels: Vec<String> = fans.iter().map(|fan| fan.channel.to_string()).collect();
    let fan_rows = |value: fn(&FanWire) -> Option<String>| -> Vec<(Vec<(&str, &str)>, String)> {
        fans.iter()
            .zip(&channels)
            .filter_map(|(fan, channel)| {
                Some((
                    vec![("channel", channel.as_str()), ("label", fan.label.as_str())],
                    value(fan)?,
                ))
            })
            .collect()
    };
    out.rows(
        "llamabored_fan_rpm",
        "gauge",
        "Fan speed, rpm, per configured [fans] channel.",
        &fan_rows(|fan| fan.rpm.map(|rpm| rpm.to_string())),
    );
    out.rows(
        "llamabored_fan_pwm_ratio",
        "gauge",
        "Fan PWM duty, 0 to 1, per configured [fans] channel (read only).",
        &fan_rows(|fan| fan.pwm.filter(|v| v.is_finite()).map(num)),
    );

    if let Some(sources) = &snap.sources {
        let entries = sources.entries();
        let up: Vec<(Vec<(&str, &str)>, String)> = entries
            .iter()
            .filter_map(|(name, source)| {
                let source = (*source)?;
                Some((
                    vec![("source", *name)],
                    if source.up { "1" } else { "0" }.to_owned(),
                ))
            })
            .collect();
        out.rows(
            "llamabored_source_up",
            "gauge",
            "1 when a watcher source answered on its last poll, 0 when it failed; absent when not polled.",
            &up,
        );
        let latency: Vec<(Vec<(&str, &str)>, String)> = entries
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
