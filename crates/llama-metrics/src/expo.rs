//! Prometheus text exposition format 0.0.4, from one snapshot read.
//!
//! Every metric is named `llamabored_*`. The only strings exported are model
//! display names and their allowlisted tuning tokens, as label values. The
//! snapshot carries no prompt or output text, and nothing here could export
//! it: every value is a number from a typed field.
//!
//! A stale or unreadable snapshot exports only the exporter's own series,
//! `llamabored_snapshot_up`, `llamabored_snapshot_stale`, and (when it could be
//! read) its seq and age. The value series disappear rather than freeze.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::Duration;

use llama_core::detail::{self, KV_DEFAULT, ModelDetail, NCMOE_ALL};
use llama_core::wire::{self, AiWire, ModelWire, WireSnapshot};

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

    if let Some(total) = snap.tokens.decoded_total {
        out.family(
            "llamabored_tokens_decoded_total",
            "counter",
            "Tokens decoded since the watcher started.",
        );
        out.sample("llamabored_tokens_decoded_total", &[], &total.to_string());
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
    let mut ctx: BTreeSet<(String, String, u32)> = BTreeSet::new();
    for model in &snap.ai.models {
        if let Some(size) = model.detail.as_ref().and_then(|d| d.ctx).filter(|c| *c > 0) {
            let labels = ModelLabels::of(model);
            ctx.insert((labels.name, labels.full_name, size));
        }
    }
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    let mut first = true;
    for (name, full_name, size) in ctx {
        if !seen.insert((name.clone(), full_name.clone())) {
            continue;
        }
        if first {
            out.family(
                "llamabored_model_ctx_size_tokens",
                "gauge",
                "Configured context size of a loaded model, tokens.",
            );
            first = false;
        }
        out.sample(
            "llamabored_model_ctx_size_tokens",
            &[("name", &name), ("full_name", &full_name)],
            &size.to_string(),
        );
    }
    out.text
}
