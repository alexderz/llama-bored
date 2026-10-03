//! `/upstream/<model>/metrics` line parse and the box decoded counter.
//!
//! Each backend has its own Prometheus names (T72). Names are matched
//! exactly; SGLang and vLLM label their series (`model_name`, `engine`), so
//! their values are summed across label sets, and a ratio takes the largest.
//! Strata answers JSON instead ([`parse_strata`]).
//!
//! #31 adds vLLM's engine numbers (names checked against vLLM 0.30's
//! `v1/metrics/loggers.py` and `v1/spec_decode/metrics.py`): speculative
//! decoding counters, preemptions, the sleep state, TTFT / ITL / request
//! latency histogram sums and counts, and the `cache_config_info` info
//! gauge, whose labels are read for three known keys only
//! ([`CACHE_KEYS`]), each value capped at [`MAX_LABEL_VALUE`] bytes.
//! [`detect_backend`] tells a server the launch command did not name by
//! its metric prefix.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use llama_core::backend::{self, Backend, EngineStats, MAX_SPEC_LEN, SpecCounts};
use llama_core::detail::{MAX_KV_BLOCK, is_token};
use llama_core::rate::{counter_delta, delta_per_s};
use serde::{Deserialize, Deserializer};

const FRESH: Duration = Duration::from_secs(1);

/// The series one backend is read for. Empty slices are never matched.
struct Names {
    decode: &'static [&'static str],
    running: &'static [&'static str],
    prompt: &'static [&'static str],
    queued: &'static [&'static str],
    /// KV fill 0..=1, newest name first; the first name that appears wins.
    kv: &'static [&'static str],
    /// Cache hit ratio 0..=1.
    hit: &'static [&'static str],
    /// Prefix cache hit and query counters, for a ratio.
    prefix_hits: &'static [&'static str],
    prefix_queries: &'static [&'static str],
    /// Cached prompt-token counter (#10), newest name first; the first name
    /// that appears wins. vLLM's is `prompt_tokens_cached` (local and
    /// external), else its prefix hit counter.
    cached: &'static [&'static str],
    /// Speculative-decoding counters (#31): drafts, draft tokens, accepted.
    spec_drafts: &'static [&'static str],
    spec_draft_tokens: &'static [&'static str],
    spec_accepted: &'static [&'static str],
    /// Speculative acceptance rate and mean accepted length gauges (SGLang).
    spec_rate: &'static [&'static str],
    spec_len: &'static [&'static str],
    /// Preemption counter.
    preemptions: &'static [&'static str],
    /// Sleep-state gauge, labelled `sleep_state`.
    sleep: &'static [&'static str],
    /// Histogram base names (without `_sum` / `_count`), newest first.
    ttft: &'static [&'static str],
    itl: &'static [&'static str],
    e2e: &'static [&'static str],
    /// Info gauge whose labels carry the KV cache config.
    cache_info: &'static [&'static str],
    /// Sum across label sets. llama.cpp keeps its last value, as before T72.
    sum: bool,
}

const NONE: &[&str] = &[];

const LLAMACPP: Names = Names {
    decode: &["llamacpp:n_decode_total"],
    running: &["llamacpp:requests_processing"],
    prompt: &["llamacpp:prompt_tokens_total"],
    queued: &[],
    kv: &[],
    hit: &[],
    prefix_hits: &[],
    prefix_queries: &[],
    cached: &[],
    spec_drafts: NONE,
    spec_draft_tokens: NONE,
    spec_accepted: NONE,
    spec_rate: NONE,
    spec_len: NONE,
    preemptions: NONE,
    sleep: NONE,
    ttft: NONE,
    itl: NONE,
    e2e: NONE,
    cache_info: NONE,
    sum: false,
};

const SGLANG: Names = Names {
    decode: &["sglang:generation_tokens_total"],
    running: &["sglang:num_running_reqs"],
    prompt: &["sglang:prompt_tokens_total"],
    queued: &["sglang:num_queue_reqs"],
    kv: &["sglang:token_usage"],
    hit: &["sglang:cache_hit_rate"],
    prefix_hits: &[],
    prefix_queries: &[],
    cached: &["sglang:cached_tokens_total"],
    spec_drafts: NONE,
    spec_draft_tokens: NONE,
    spec_accepted: NONE,
    spec_rate: &["sglang:spec_accept_rate"],
    spec_len: &["sglang:spec_accept_length"],
    preemptions: NONE,
    sleep: NONE,
    ttft: &["sglang:time_to_first_token_seconds"],
    itl: &["sglang:inter_token_latency_seconds"],
    e2e: &["sglang:e2e_request_latency_seconds"],
    cache_info: NONE,
    sum: true,
};

const VLLM: Names = Names {
    decode: &["vllm:generation_tokens_total"],
    running: &["vllm:num_requests_running"],
    prompt: &["vllm:prompt_tokens_total"],
    queued: &["vllm:num_requests_waiting"],
    kv: &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    hit: &[],
    prefix_hits: &["vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"],
    prefix_queries: &[
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
    ],
    cached: &[
        "vllm:prompt_tokens_cached_total",
        "vllm:prompt_tokens_cached",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    ],
    spec_drafts: &["vllm:spec_decode_num_drafts_total"],
    spec_draft_tokens: &["vllm:spec_decode_num_draft_tokens_total"],
    spec_accepted: &["vllm:spec_decode_num_accepted_tokens_total"],
    spec_rate: NONE,
    spec_len: NONE,
    preemptions: &["vllm:num_preemptions_total"],
    sleep: &["vllm:engine_sleep_state"],
    ttft: &["vllm:time_to_first_token_seconds"],
    itl: &[
        "vllm:inter_token_latency_seconds",
        "vllm:time_per_output_token_seconds",
    ],
    e2e: &["vllm:e2e_request_latency_seconds"],
    cache_info: &["vllm:cache_config_info"],
    sum: true,
};

/// `cache_config_info` label keys read; every other label is skipped.
pub const CACHE_KEYS: [&str; 3] = ["cache_dtype", "block_size", "enable_prefix_caching"];
/// Longest label value kept, bytes. A longer value is dropped.
pub const MAX_LABEL_VALUE: usize = 64;
/// Most labels scanned on one series line.
const MAX_LABELS: usize = 128;

/// Server kind from the metric names in a `/metrics` body (#31): the first
/// sample whose name starts with `vllm:`, `sglang:` or `llamacpp:`. Comments
/// are skipped. `None` when no sample has one of those prefixes.
#[must_use]
pub fn detect_backend(body: &str) -> Option<Backend> {
    body.lines().find_map(|line| {
        let line = line.trim_start();
        if line.starts_with('#') {
            return None;
        }
        let name = &line[..line.find([' ', '\t', '{']).unwrap_or(line.len())];
        [
            ("vllm:", Backend::Vllm),
            ("sglang:", Backend::SgLang),
            ("llamacpp:", Backend::LlamaCpp),
        ]
        .into_iter()
        .find(|(prefix, _)| name.len() > prefix.len() && name.starts_with(prefix))
        .map(|(_, kind)| kind)
    })
}

/// A histogram's `_sum` and `_count`, summed over label sets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hist {
    /// `<name>_sum`.
    pub sum: f64,
    /// `<name>_count`.
    pub count: f64,
}

/// One parsed metrics document. Missing or rejected values stay `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricsSample {
    /// Decoded-token counter (`llamacpp:n_decode_total`,
    /// `*:generation_tokens_total`) when finite and ≥ 0.
    pub n_decode_total: Option<u64>,
    /// Requests in flight (`llamacpp:requests_processing`, running requests).
    pub requests_processing: Option<f64>,
    /// Prompt-token counter (`llamacpp:prompt_tokens_total`,
    /// `*:prompt_tokens_total`).
    pub prompt_total: Option<u64>,
    /// Requests waiting. SGLang and vLLM only.
    pub queued: Option<f64>,
    /// KV cache fill, 0..=1.
    pub kv_fill: Option<f64>,
    /// Prefix cache hit ratio, 0..=1.
    pub cache_hit: Option<f64>,
    /// Cached prompt-token counter (`sglang:cached_tokens_total`,
    /// `vllm:prompt_tokens_cached_total`), part of [`Self::prompt_total`] (#10).
    pub cached_total: Option<u64>,
    /// vLLM speculative counters (#31), when all three are present.
    pub spec: Option<SpecCounts>,
    /// SGLang's speculative acceptance rate gauge, 0..=1 (#31).
    pub spec_rate: Option<f64>,
    /// SGLang's mean accepted length gauge; below 1 means spec is off.
    pub spec_len: Option<f64>,
    /// vLLM preemption counter.
    pub preemptions: Option<u64>,
    /// vLLM sleep state: `true` when any engine's `awake` series is 0.
    pub sleeping: Option<bool>,
    /// Time to first token histogram.
    pub ttft: Option<Hist>,
    /// Inter-token latency histogram.
    pub itl: Option<Hist>,
    /// End-to-end request latency histogram.
    pub e2e: Option<Hist>,
}

/// llama.cpp parse, the pre-T72 behaviour. See [`parse_metrics_for`].
#[must_use]
pub fn parse_metrics(body: &str) -> MetricsSample {
    parse_metrics_for(Backend::LlamaCpp, body)
}

/// Exact-name parse for `backend`. Unknown lines, comments, and non-finite
/// or negative values are ignored. A `{labels}` suffix does not change the
/// name. [`Backend::OpenAi`] reads nothing.
#[must_use]
pub fn parse_metrics_for(backend: Backend, body: &str) -> MetricsSample {
    parse_metrics_full(backend, body).0
}

/// Parse for `backend`, plus the engine facts the server reports about
/// itself: Strata's `engine` object, or vLLM's `cache_config_info` labels.
#[must_use]
pub fn parse_metrics_full(backend: Backend, body: &str) -> (MetricsSample, Option<EngineFacts>) {
    let names = match backend {
        Backend::LlamaCpp => &LLAMACPP,
        Backend::SgLang => &SGLANG,
        Backend::Vllm => &VLLM,
        Backend::Strata => {
            let (sample, facts) = parse_strata(body);
            return (sample, Some(facts));
        }
        Backend::OpenAi => return (MetricsSample::default(), None),
    };
    let mut decode = Acc::default();
    let mut running = Acc::default();
    let mut prompt = Acc::default();
    let mut queued = Acc::default();
    let mut kv: [Option<f64>; 2] = [None, None];
    let mut hit = None;
    let mut hits = Acc::default();
    let mut queries = Acc::default();
    let mut cached: [Acc; 4] = Default::default();
    let mut spec: [Acc; 3] = Default::default();
    let mut spec_rate = None;
    let mut spec_len: Option<f64> = None;
    let mut preemptions = Acc::default();
    let mut sleeping: Option<bool> = None;
    let mut hists: [[HistAcc; 2]; 3] = Default::default();
    let mut facts: Option<EngineFacts> = None;
    for line in body.lines() {
        let Some((name, value)) = metric_line(line) else {
            continue;
        };
        if !value.is_finite() || value < 0.0 {
            continue;
        }
        if names.decode.contains(&name) {
            decode.add(value, names.sum);
        } else if names.running.contains(&name) {
            running.add(value, names.sum);
        } else if names.prompt.contains(&name) {
            prompt.add(value, names.sum);
        } else if names.queued.contains(&name) {
            queued.add(value, names.sum);
        } else if let Some(index) = names.kv.iter().position(|kv_name| *kv_name == name) {
            if let Some(slot) = kv.get_mut(index) {
                *slot = max_ratio(*slot, value);
            }
        } else if names.hit.contains(&name) {
            hit = max_ratio(hit, value);
        } else if names.prefix_hits.contains(&name) {
            hits.add(value, true);
        } else if names.prefix_queries.contains(&name) {
            queries.add(value, true);
        } else if names.spec_rate.contains(&name) {
            spec_rate = max_ratio(spec_rate, value);
        } else if names.spec_len.contains(&name) {
            spec_len = Some(spec_len.map_or(value, |old| old.max(value)));
        } else if names.preemptions.contains(&name) {
            preemptions.add(value, true);
        } else if names.sleep.contains(&name) {
            if label_value(line, "sleep_state").as_deref() == Some("awake") {
                sleeping = Some(sleeping.unwrap_or(false) || value == 0.0);
            }
        } else if names.cache_info.contains(&name) {
            if facts.is_none() {
                facts = Some(cache_facts(line));
            }
        } else if let Some(index) = [
            names.spec_drafts,
            names.spec_draft_tokens,
            names.spec_accepted,
        ]
        .iter()
        .position(|group| group.contains(&name))
        {
            spec[index].add(value, true);
        } else {
            for (which, group) in [names.ttft, names.itl, names.e2e].iter().enumerate() {
                if let Some((rank, part)) = hist_part(group, name) {
                    hists[which][rank].add(part, value);
                }
            }
        }
        if let Some(rank) = names
            .cached
            .iter()
            .position(|cached_name| *cached_name == name)
            && let Some(slot) = cached.get_mut(rank)
        {
            slot.add(value, names.sum);
        }
    }
    let prefix = match (hits.0, queries.0) {
        (Some(hits), Some(queries)) if queries > 0.0 && hits <= queries => Some(hits / queries),
        _ => None,
    };
    let spec = match (spec[0].0, spec[1].0, spec[2].0) {
        (Some(drafts), Some(tokens), Some(accepted)) => {
            match (finite_u64(drafts), finite_u64(tokens), finite_u64(accepted)) {
                (Some(drafts), Some(draft_tokens), Some(accepted)) => Some(SpecCounts {
                    drafts,
                    draft_tokens,
                    accepted,
                }),
                _ => None,
            }
        }
        _ => None,
    };
    let hist = |ranks: &[HistAcc; 2]| ranks.iter().find_map(HistAcc::get);
    let sample = MetricsSample {
        n_decode_total: decode.0.and_then(finite_u64),
        requests_processing: running.0,
        prompt_total: prompt.0.and_then(finite_u64),
        queued: queued.0,
        kv_fill: kv[0].or(kv[1]),
        cache_hit: hit.or(prefix),
        cached_total: cached.iter().find_map(|acc| acc.0).and_then(finite_u64),
        spec,
        spec_rate,
        spec_len,
        preemptions: preemptions.0.and_then(finite_u64),
        sleeping,
        ttft: hist(&hists[0]),
        itl: hist(&hists[1]),
        e2e: hist(&hists[2]),
    };
    (sample, facts)
}

/// `(rank, is_count)` when `name` is `<base>_sum` or `<base>_count` of a
/// base in `group`; rank is the base's place in it.
fn hist_part(group: &[&str], name: &str) -> Option<(usize, bool)> {
    let (base, count) = if let Some(base) = name.strip_suffix("_sum") {
        (base, false)
    } else {
        (name.strip_suffix("_count")?, true)
    };
    let rank = group.iter().position(|candidate| *candidate == base)?;
    (rank < 2).then_some((rank, count))
}

#[derive(Clone, Copy, Default)]
struct HistAcc {
    sum: Option<f64>,
    count: Option<f64>,
}

impl HistAcc {
    fn add(&mut self, is_count: bool, value: f64) {
        let slot = if is_count {
            &mut self.count
        } else {
            &mut self.sum
        };
        *slot = Some(slot.unwrap_or(0.0) + value);
    }

    fn get(&self) -> Option<Hist> {
        Some(Hist {
            sum: self.sum?,
            count: self.count?,
        })
    }
}

/// The three known keys of a `cache_config_info` line ([`CACHE_KEYS`]).
/// `cache_dtype` must be a detail token; `block_size` a whole number in
/// 1..=[`MAX_KV_BLOCK`]; `enable_prefix_caching` `True` or `False`.
fn cache_facts(line: &str) -> EngineFacts {
    let labels = labels_of(line);
    let get = |key: &str| {
        labels
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.as_str())
    };
    EngineFacts {
        ctx: None,
        kv: get(CACHE_KEYS[0])
            .map(str::to_ascii_lowercase)
            .filter(|kv| is_token(kv)),
        kv_block: get(CACHE_KEYS[1])
            .and_then(|block| block.parse::<u32>().ok())
            .filter(|block| (1..=MAX_KV_BLOCK).contains(block)),
        prefix_cache: match get(CACHE_KEYS[2]) {
            Some("True" | "true") => Some(true),
            Some("False" | "false") => Some(false),
            _ => None,
        },
    }
}

/// The value of label `key` on a series line, under [`labels_of`]'s rules.
fn label_value(line: &str, key: &str) -> Option<String> {
    labels_of(line)
        .into_iter()
        .find(|(name, _)| *name == key)
        .map(|(_, value)| value)
}

/// `key="value"` pairs of a series line's `{...}`. Escapes (`\\`, `\"`,
/// `\n`) are decoded. Scanning stops at the first malformed byte or after
/// [`MAX_LABELS`] pairs; a value longer than [`MAX_LABEL_VALUE`] bytes is
/// left out. Only the keys in [`CACHE_KEYS`] and `sleep_state` are kept,
/// so the result is small whatever the line holds.
fn labels_of(line: &str) -> Vec<(&str, String)> {
    let mut out = Vec::new();
    let Some(open) = line.find('{') else {
        return out;
    };
    let bytes = line.as_bytes();
    let mut at = open + 1;
    for _ in 0..MAX_LABELS {
        while bytes.get(at) == Some(&b' ') {
            at += 1;
        }
        let key_start = at;
        while bytes
            .get(at)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            at += 1;
        }
        let key = &line[key_start..at];
        if key.is_empty() || bytes.get(at) != Some(&b'=') || bytes.get(at + 1) != Some(&b'"') {
            break;
        }
        at += 2;
        let mut value = String::new();
        let mut too_long = false;
        loop {
            let Some(&byte) = bytes.get(at) else {
                return out;
            };
            at += 1;
            let ch = match byte {
                b'"' => break,
                b'\\' => {
                    let escaped = bytes.get(at).copied();
                    at += 1;
                    match escaped {
                        Some(b'n') => '\n',
                        Some(b'\\') => '\\',
                        Some(b'"') => '"',
                        _ => return out,
                    }
                }
                _ => char::from(byte),
            };
            if value.len() >= MAX_LABEL_VALUE {
                too_long = true;
            } else {
                value.push(ch);
            }
        }
        if !too_long && value.is_ascii() && (key == "sleep_state" || CACHE_KEYS.contains(&key)) {
            out.push((key, value));
        }
        match bytes.get(at) {
            Some(b',') => at += 1,
            _ => break,
        }
    }
    out
}

/// Tuning facts a server reports about itself (Strata's `engine` object,
/// vLLM's `cache_config_info`), for a detail line its launch command
/// cannot give.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EngineFacts {
    /// `engine.max_context`, when a positive `u32`.
    pub ctx: Option<u32>,
    /// `engine.kv` or `cache_dtype` lowercased, when it is a short token
    /// (`q8`, `fp8_e4m3`).
    pub kv: Option<String>,
    /// vLLM `block_size`, 1..=[`MAX_KV_BLOCK`].
    pub kv_block: Option<u32>,
    /// vLLM `enable_prefix_caching`.
    pub prefix_cache: Option<bool>,
}

/// The parts of Strata's `GET /metrics` JSON this reads. Everything else
/// (the last requests, hardware and its history) is skipped unread.
#[derive(Default, Deserialize)]
#[serde(default)]
struct StrataDoc {
    engine: StrataEngine,
    live: StrataLive,
    totals: StrataTotals,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct StrataEngine {
    #[serde(deserialize_with = "number")]
    max_context: Option<f64>,
    #[serde(deserialize_with = "word")]
    kv: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct StrataLive {
    #[serde(deserialize_with = "word")]
    state: Option<String>,
    #[serde(deserialize_with = "number")]
    queued: Option<f64>,
    #[serde(deserialize_with = "number")]
    generated: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct StrataTotals {
    #[serde(deserialize_with = "number")]
    prompt_tokens: Option<f64>,
    #[serde(deserialize_with = "number")]
    reused: Option<f64>,
    #[serde(deserialize_with = "number")]
    output_tokens: Option<f64>,
}

/// A finite, non-negative number; any other JSON value is `None`.
fn number<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(value.as_f64().filter(|n| n.is_finite() && *n >= 0.0))
}

/// A JSON string; any other value is `None`.
fn word<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::String(text) => Ok(Some(text)),
        _ => Ok(None),
    }
}

/// Strata's `/metrics` JSON (not Prometheus text).
///
/// - decode: `totals.output_tokens` (finished requests) plus
///   `live.generated` while a request runs, so tok/s moves live. Strata
///   copies both under one lock and moves a request's tokens from `live` to
///   `totals` in the same step, so the sum never goes down while it runs; a
///   restart zeroes it, which [`counter_delta`] already treats as a restart.
/// - prompt and cached: `totals.prompt_tokens` and `totals.reused` (the
///   reused prefix is part of the prompt, as SGLang's cached counter is).
/// - running: 1 while `live.state` is `reading` or `generating`, 0 when
///   `idle`; queued: `live.queued`.
/// - no KV fill (Strata has none) and no hit ratio: `totals` holds only
///   lifetime sums, not the recent-window rate other backends report.
///
/// A body that is not such a JSON object reads as all `None`.
#[must_use]
pub fn parse_strata(body: &str) -> (MetricsSample, EngineFacts) {
    let Ok(doc) = serde_json::from_str::<StrataDoc>(body) else {
        return (MetricsSample::default(), EngineFacts::default());
    };
    let busy = match doc.live.state.as_deref() {
        Some("reading" | "generating") => Some(true),
        Some("idle") => Some(false),
        _ => None,
    };
    let live = match busy {
        Some(true) => doc.live.generated.unwrap_or(0.0),
        _ => 0.0,
    };
    let sample = MetricsSample {
        n_decode_total: doc
            .totals
            .output_tokens
            .and_then(|done| finite_u64(done + live)),
        requests_processing: busy.map(|busy| if busy { 1.0 } else { 0.0 }),
        prompt_total: doc.totals.prompt_tokens.and_then(finite_u64),
        queued: doc.live.queued,
        kv_fill: None,
        cache_hit: None,
        cached_total: doc.totals.reused.and_then(finite_u64),
        ..MetricsSample::default()
    };
    let facts = EngineFacts {
        ctx: doc
            .engine
            .max_context
            .and_then(finite_u64)
            .and_then(|ctx| u32::try_from(ctx).ok())
            .filter(|ctx| *ctx > 0),
        kv: doc
            .engine
            .kv
            .map(|kv| kv.to_ascii_lowercase())
            .filter(|kv| is_token(kv)),
        ..EngineFacts::default()
    };
    (sample, facts)
}

/// One series value: the sum over label sets, or the last one.
#[derive(Default)]
struct Acc(Option<f64>);

impl Acc {
    fn add(&mut self, value: f64, sum: bool) {
        self.0 = Some(match self.0 {
            Some(old) if sum => old + value,
            _ => value,
        });
    }
}

/// The larger of two 0..=1 ratios. A value above 1 is dropped.
fn max_ratio(old: Option<f64>, value: f64) -> Option<f64> {
    if value > 1.0 {
        return old;
    }
    Some(old.map_or(value, |old| old.max(value)))
}

/// Per-model baseline for `n_decode_total` since the watcher started.
#[derive(Debug, Default)]
pub struct DecodedCounter {
    last: HashMap<String, u64>,
    fresh_at: HashMap<String, Instant>,
    /// How long a read stays fresh, when not [`FRESH`] (activity fallback).
    window: HashMap<String, Duration>,
    processing: HashMap<String, f64>,
    total: u64,
}

impl DecodedCounter {
    /// Apply one accepted decode sample.
    ///
    /// The first sample for `model` sets the baseline and adds nothing.
    /// `value >= last` adds the difference. `value < last` (the process
    /// restarted) adds `value`.
    pub fn observe(&mut self, model: &str, value: u64, processing: Option<f64>, at: Instant) {
        let delta = counter_delta(self.last.get(model).copied(), value);
        self.last.insert(model.to_owned(), value);
        self.total = self.total.saturating_add(delta);
        self.fresh_at.insert(model.to_owned(), at);
        self.window.remove(model);
        if let Some(processing) = processing {
            self.processing.insert(model.to_owned(), processing);
        }
    }

    /// Add `tokens` counted outside `/metrics` (llama-swap activity rows)
    /// and mark `model` read at `at`, fresh for `window`.
    ///
    /// The model's `/metrics` baseline is dropped, so a counter that comes
    /// back later starts a new baseline instead of adding what the
    /// activity rows already counted.
    pub fn add(&mut self, model: &str, tokens: u64, at: Instant, window: Duration) {
        self.total = self.total.saturating_add(tokens);
        self.last.remove(model);
        self.processing.remove(model);
        self.fresh_at.insert(model.to_owned(), at);
        self.window.insert(model.to_owned(), window.max(FRESH));
    }

    /// Running sum when `/running` is up and every ready model was read
    /// within the last second. Zero ready models is a measured total.
    #[must_use]
    pub fn total_if_fresh(&self, running_up: bool, ready: &[&str], now: Instant) -> Option<u64> {
        if !running_up {
            return None;
        }
        for id in ready {
            let window = self.window.get(*id).copied().unwrap_or(FRESH);
            match self.fresh_at.get(*id) {
                Some(at) if now.saturating_duration_since(*at) <= window => {}
                _ => return None,
            }
        }
        Some(self.total)
    }

    /// Gauge from a metrics read that is still ≤ 1 s old.
    #[must_use]
    pub fn requests_processing(&self, model: &str, now: Instant) -> Option<f64> {
        let at = self.fresh_at.get(model)?;
        if now.saturating_duration_since(*at) > FRESH {
            return None;
        }
        self.processing.get(model).copied()
    }

    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }
}

/// Most models [`PromptCache`] keeps counters for in one watcher run.
pub const PROMPT_CACHE_MODELS: usize = 64;

/// Per-model prompt and cached-prompt token counters since the watcher
/// started (#10), keyed by llama-swap model id.
///
/// Two sources, never both for one model:
/// - SGLang and vLLM `/metrics` with both a prompt and a cached counter
///   ([`Self::observe_metrics`]): each read adds its delta, and a counter
///   that went down (the server restarted) adds its new value.
/// - Otherwise llama-swap activity rows, once each ([`Self::add_row`]). A
///   row with a known `cache_tokens` came from llama.cpp timings, where
///   `input_tokens` counts only the tokens processed, so the prompt is
///   `input + cache`. A row without it (OpenAI `usage`) counts `input` as the
///   whole prompt and adds nothing cached.
///
/// A model keeps its counters for the whole run, across unloads, so a
/// counter only goes back to zero when the watcher restarts. At most
/// [`PROMPT_CACHE_MODELS`] models; later ones are not counted.
#[derive(Debug, Default)]
pub struct PromptCache {
    models: HashMap<String, PromptTotals>,
}

#[derive(Debug, Default)]
struct PromptTotals {
    prompt: u64,
    cached: Option<u64>,
    last_prompt: Option<u64>,
    last_cached: Option<u64>,
    /// The last `/metrics` read carried both counters.
    from_metrics: bool,
}

impl PromptCache {
    fn entry(&mut self, model: &str) -> Option<&mut PromptTotals> {
        if !self.models.contains_key(model) && self.models.len() >= PROMPT_CACHE_MODELS {
            return None;
        }
        Some(self.models.entry(model.to_owned()).or_default())
    }

    /// One `/metrics` read. With both counters the model is counted from
    /// `/metrics` from now on (the first read is the baseline); with either
    /// missing it goes back to activity rows.
    pub fn observe_metrics(&mut self, model: &str, prompt: Option<u64>, cached: Option<u64>) {
        let Some(totals) = self.entry(model) else {
            return;
        };
        let (Some(prompt), Some(cached)) = (prompt, cached) else {
            totals.from_metrics = false;
            totals.last_prompt = None;
            totals.last_cached = None;
            return;
        };
        let add_prompt = counter_delta(totals.last_prompt, prompt);
        let add_cached = counter_delta(totals.last_cached, cached);
        totals.prompt = totals.prompt.saturating_add(add_prompt);
        totals.cached = Some(totals.cached.unwrap_or(0).saturating_add(add_cached));
        totals.last_prompt = Some(prompt);
        totals.last_cached = Some(cached);
        totals.from_metrics = true;
    }

    /// Start `model`'s counters at zero once llama-swap's activity log has
    /// been read, so a model with no finished request yet reads 0, not
    /// unknown. `cached_known` starts the cached counter too (llama.cpp,
    /// whose timings always carry `cache_n`).
    pub fn touch(&mut self, model: &str, cached_known: bool) {
        if let Some(totals) = self.entry(model)
            && cached_known
            && totals.cached.is_none()
        {
            totals.cached = Some(0);
        }
    }

    /// One new activity row of `model`. Ignored while `/metrics` counts it.
    pub fn add_row(&mut self, model: &str, input: Option<u64>, cache: Option<u64>) {
        let Some(totals) = self.entry(model) else {
            return;
        };
        let Some(input) = input else {
            return;
        };
        if totals.from_metrics {
            return;
        }
        totals.prompt = totals
            .prompt
            .saturating_add(input.saturating_add(cache.unwrap_or(0)));
        if let Some(cache) = cache {
            totals.cached = Some(totals.cached.unwrap_or(0).saturating_add(cache));
        }
    }

    /// `(prompt, cached)` for `model`. `cached` is `None` until a source
    /// reported one, and never above `prompt`.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<(u64, Option<u64>)> {
        let totals = self.models.get(model)?;
        Some((
            totals.prompt,
            totals.cached.map(|cached| cached.min(totals.prompt)),
        ))
    }
}

/// Most models [`EngineBook`] keeps counters for in one watcher run.
pub const ENGINE_MODELS: usize = 64;

/// Per-model engine numbers from `/metrics` (#31), keyed by llama-swap id.
///
/// - Window values (speculative acceptance and step length, TTFT, ITL and
///   request latency means) are the change between two reads: accepted /
///   draft tokens, 1 + accepted / drafts, and Δsum / Δcount of each
///   histogram. A window with no new drafts (or requests) keeps the last
///   value, so an idle server still shows its last numbers. The first read
///   after a load has no window yet and uses the server's totals since it
///   started. A counter that went down (the server restarted) makes its
///   new value the window.
/// - Counters (speculative drafts, draft tokens, accepted tokens,
///   preemptions) count since the watcher started, like [`PromptCache`]:
///   the first read is the baseline, a restart adds its new value, and a
///   model keeps them across unloads.
/// - SGLang reports acceptance and step length as gauges, used as they are.
///
/// At most [`ENGINE_MODELS`] models; later ones get no numbers.
#[derive(Debug, Default)]
pub struct EngineBook {
    models: HashMap<String, EngineTrack>,
}

#[derive(Debug, Default)]
struct EngineTrack {
    spec_total: Option<SpecCounts>,
    preempt_total: Option<u64>,
    last_spec: Option<SpecCounts>,
    last_preempt: Option<u64>,
    last_hist: [Option<Hist>; 3],
    spec_rate: Option<f64>,
    spec_len: Option<f64>,
    means: [Option<f64>; 3],
}

impl EngineBook {
    /// Fold one `/metrics` read of `model` in and return its numbers.
    pub fn observe(&mut self, model: &str, sample: &MetricsSample) -> EngineStats {
        if !self.models.contains_key(model) && self.models.len() >= ENGINE_MODELS {
            return EngineStats::default();
        }
        let track = self.models.entry(model.to_owned()).or_default();
        track.observe(sample)
    }

    /// `model` is no longer loaded: drop its window values and baselines,
    /// keep its counters. The next read starts afresh.
    pub fn forget(&mut self, model: &str) {
        if let Some(track) = self.models.get_mut(model) {
            track.last_spec = None;
            track.last_preempt = None;
            track.last_hist = [None; 3];
            track.spec_rate = None;
            track.spec_len = None;
            track.means = [None; 3];
        }
    }
}

impl EngineTrack {
    fn observe(&mut self, sample: &MetricsSample) -> EngineStats {
        self.observe_spec(sample);
        match sample.preemptions {
            Some(now) => {
                let add = counter_delta(self.last_preempt, now);
                self.preempt_total = Some(self.preempt_total.unwrap_or(0).saturating_add(add));
                self.last_preempt = Some(now);
            }
            None => self.last_preempt = None,
        }
        for (index, hist) in [sample.ttft, sample.itl, sample.e2e]
            .into_iter()
            .enumerate()
        {
            let Some(now) = hist else {
                self.last_hist[index] = None;
                self.means[index] = None;
                continue;
            };
            let window = match self.last_hist[index] {
                Some(last) if now.count >= last.count && now.sum >= last.sum => Hist {
                    sum: now.sum - last.sum,
                    count: now.count - last.count,
                },
                _ => now,
            };
            if window.count > 0.0 {
                self.means[index] = Some(window.sum / window.count);
            }
            self.last_hist[index] = Some(now);
        }
        EngineStats {
            spec_permille: self.spec_rate.and_then(backend::permille),
            spec_len_centi: self.spec_len.and_then(backend::centi_len),
            spec_counts: self.spec_total.filter(|_| self.last_spec.is_some()),
            preemptions: self.preempt_total.filter(|_| self.last_preempt.is_some()),
            sleeping: sample.sleeping,
            ttft_us: self.means[0].and_then(backend::micros),
            itl_us: self.means[1].and_then(backend::micros),
            e2e_us: self.means[2].and_then(backend::micros),
        }
    }

    fn observe_spec(&mut self, sample: &MetricsSample) {
        if let Some(now) = sample.spec {
            let window = match self.last_spec {
                Some(last)
                    if now.drafts >= last.drafts
                        && now.draft_tokens >= last.draft_tokens
                        && now.accepted >= last.accepted =>
                {
                    SpecCounts {
                        drafts: now.drafts - last.drafts,
                        draft_tokens: now.draft_tokens - last.draft_tokens,
                        accepted: now.accepted - last.accepted,
                    }
                }
                _ => now,
            };
            let accepted = window.accepted.min(window.draft_tokens);
            if self.last_spec.is_some() {
                let total = self.spec_total.get_or_insert_with(SpecCounts::default);
                total.drafts = total.drafts.saturating_add(window.drafts);
                total.draft_tokens = total.draft_tokens.saturating_add(window.draft_tokens);
                total.accepted = total
                    .accepted
                    .saturating_add(accepted)
                    .min(total.draft_tokens);
            } else if self.spec_total.is_none() {
                self.spec_total = Some(SpecCounts::default());
            }
            if window.draft_tokens > 0 {
                self.spec_rate = Some(accepted as f64 / window.draft_tokens as f64);
            }
            if window.drafts > 0 {
                self.spec_len =
                    Some((1.0 + accepted as f64 / window.drafts as f64).min(MAX_SPEC_LEN));
            }
            self.last_spec = Some(now);
        } else {
            self.last_spec = None;
            // SGLang: gauges, as they are. A step length below 1 is spec off.
            match (sample.spec_rate, sample.spec_len) {
                (Some(rate), Some(len)) if len >= 1.0 => {
                    self.spec_rate = Some(rate);
                    self.spec_len = Some(len.min(MAX_SPEC_LEN));
                }
                _ => {
                    self.spec_rate = None;
                    self.spec_len = None;
                }
            }
        }
    }
}

/// Generation tok/s from `decoded_total` over the last second.
#[derive(Debug, Default)]
pub struct GenRate {
    samples: Vec<(Instant, u64)>,
}

impl GenRate {
    /// Record `total` and return tok/s once a sample is at least one second old.
    #[must_use]
    pub fn observe(&mut self, now: Instant, total: u64) -> Option<f64> {
        self.samples.push((now, total));
        let cutoff = now.checked_sub(FRESH)?;
        while self.samples.len() > 2 && self.samples[1].0 <= cutoff {
            self.samples.remove(0);
        }
        let base = self.samples.iter().rev().find(|(at, _)| *at <= cutoff)?;
        let window = now.saturating_duration_since(base.0);
        if window < FRESH {
            return None;
        }
        let delta = total.saturating_sub(base.1);
        Some(delta_per_s(delta, window))
    }
}

fn metric_line(line: &str) -> Option<(&str, f64)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let name_end = line.find([' ', '\t', '{']).unwrap_or(line.len());
    if name_end == 0 {
        return None;
    }
    let name = &line[..name_end];
    let rest = if line.as_bytes().get(name_end) == Some(&b'{') {
        let close = labels_end(&line[name_end..])?;
        line[name_end + close + 1..].trim_start()
    } else {
        line[name_end..].trim_start()
    };
    let token = rest.split_whitespace().next()?;
    let value: f64 = token.parse().ok()?;
    Some((name, value))
}

/// Index of the `}` closing a `{...}` label block, skipping quoted values
/// (a label value may hold `}`, `,` or an escaped quote).
fn labels_end(labels: &str) -> Option<usize> {
    let mut quoted = false;
    let mut escaped = false;
    for (at, byte) in labels.bytes().enumerate() {
        match byte {
            _ if escaped => escaped = false,
            b'\\' if quoted => escaped = true,
            b'"' => quoted = !quoted,
            b'}' if !quoted => return Some(at),
            _ => {}
        }
    }
    None
}

fn finite_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
        None
    } else {
        Some(value as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRATA: &str = include_str!("../../../fixtures/llama/strata-metrics.json");

    /// The fixture with `live` replaced.
    fn strata_live(live: &str) -> String {
        let mut doc: serde_json::Value = serde_json::from_str(STRATA).expect("fixture");
        doc["live"] = serde_json::from_str(live).expect("live");
        doc.to_string()
    }

    #[test]
    fn strata_generating_counts_the_running_request() {
        let (sample, facts) = parse_strata(STRATA);
        assert_eq!(
            sample,
            MetricsSample {
                n_decode_total: Some(2540),
                requests_processing: Some(1.0),
                prompt_total: Some(12_000),
                queued: Some(2.0),
                kv_fill: None,
                cache_hit: None,
                cached_total: Some(9000),
                ..MetricsSample::default()
            }
        );
        assert_eq!(
            facts,
            EngineFacts {
                ctx: Some(262_144),
                kv: Some("q8".to_owned()),
                ..EngineFacts::default()
            }
        );
        assert_eq!(parse_metrics_for(Backend::Strata, STRATA), sample);
        assert_eq!(
            parse_metrics_full(Backend::Strata, STRATA),
            (sample, Some(facts))
        );
    }

    #[test]
    fn strata_reading_and_idle() {
        let reading = strata_live(
            r#"{"state":"reading","queued":0,"phase":"reading the prompt","prompt_tokens":1800,
                "prompt_read":900,"prompt_total":1800,"generated":0,"max_tokens":4096,"elapsed_s":1.0,
                "tok_s":null,"tok_s_mean":null,"tok_s_window_s":null}"#,
        );
        let sample = parse_metrics_for(Backend::Strata, &reading);
        assert_eq!(sample.requests_processing, Some(1.0));
        assert_eq!(sample.queued, Some(0.0));
        assert_eq!(sample.n_decode_total, Some(2500));

        let idle = strata_live(
            r#"{"state":"idle","queued":0,"phase":null,"prompt_tokens":null,"prompt_read":null,
                "prompt_total":null,"generated":null,"max_tokens":null,"elapsed_s":null,
                "tok_s":null,"tok_s_mean":null,"tok_s_window_s":null}"#,
        );
        let sample = parse_metrics_for(Backend::Strata, &idle);
        assert_eq!(sample.requests_processing, Some(0.0));
        assert_eq!(sample.n_decode_total, Some(2500));
        assert_eq!(sample.cached_total, Some(9000));

        // An unknown state is not a guess.
        let odd = strata_live(r#"{"state":"sleeping","queued":1,"generated":5}"#);
        let sample = parse_metrics_for(Backend::Strata, &odd);
        assert_eq!(sample.requests_processing, None);
        assert_eq!(sample.n_decode_total, Some(2500));
    }

    #[test]
    fn strata_rejects_what_is_not_its_json() {
        let empty = (MetricsSample::default(), EngineFacts::default());
        assert_eq!(parse_strata(""), empty);
        assert_eq!(parse_strata("llamacpp:n_decode_total 5\n"), empty);
        assert_eq!(parse_strata("[1,2]"), empty);
        assert_eq!(parse_strata(r#"{"totals":{"output_tokens":5}"#), empty);
        // Wrong types and hostile values drop one field, not the rest.
        let (sample, facts) = parse_strata(
            r#"{"engine":{"max_context":-5,"kv":"../../etc/passwd"},
                "live":{"state":7,"queued":"many","generated":-3},
                "totals":{"output_tokens":10,"prompt_tokens":"x","reused":null}}"#,
        );
        assert_eq!(sample.n_decode_total, Some(10));
        assert_eq!(sample.requests_processing, None);
        assert_eq!(sample.queued, None);
        assert_eq!(sample.prompt_total, None);
        assert_eq!(sample.cached_total, None);
        assert_eq!(facts, EngineFacts::default());
        let (_, facts) = parse_strata(r#"{"engine":{"max_context":1e12,"kv":"Q8_0"}}"#);
        assert_eq!(facts.ctx, None);
        assert_eq!(facts.kv.as_deref(), Some("q8_0"));
        // Strata's Prometheus-less body read as SGLang gives nothing.
        assert_eq!(
            parse_metrics_for(Backend::SgLang, STRATA),
            MetricsSample::default()
        );
    }

    #[test]
    fn strata_restart_resets_the_counter_without_a_jump() {
        let mut counter = DecodedCounter::default();
        let at = Instant::now();
        let first = parse_metrics_for(Backend::Strata, STRATA);
        counter.observe("flash", first.n_decode_total.expect("decode"), None, at);
        // The request finishes: its 40 tokens move from live to totals in one
        // step, so the counter stays put instead of looking like a restart.
        let mut doc: serde_json::Value = serde_json::from_str(STRATA).expect("fixture");
        doc["totals"]["output_tokens"] = serde_json::json!(2540);
        doc["live"] = serde_json::json!({"state": "idle", "queued": 0});
        let finished = parse_metrics_for(Backend::Strata, &doc.to_string());
        assert_eq!(finished.n_decode_total, Some(2540));
        counter.observe("flash", 2540, None, at);
        assert_eq!(counter.total(), 0);
        // Strata restarted: totals start again from zero.
        doc["totals"]["output_tokens"] = serde_json::json!(12);
        let restarted = parse_metrics_for(Backend::Strata, &doc.to_string());
        counter.observe("flash", restarted.n_decode_total.expect("decode"), None, at);
        assert_eq!(counter.total(), 12);

        let mut cache = PromptCache::default();
        cache.observe_metrics("flash", Some(12_000), Some(9000));
        cache.observe_metrics("flash", Some(12_500), Some(9400));
        cache.observe_metrics("flash", Some(300), Some(100));
        let (prompt, cached) = cache.get("flash").expect("counted");
        assert_eq!((prompt, cached), (800, Some(500)));
    }

    #[test]
    fn metrics_parse_ignores_extra_lines_missing_names_and_nan() {
        let body = "\
# HELP llamacpp:n_decode_total decoded
# TYPE llamacpp:n_decode_total counter
llamacpp:prompt_tokens_total 999
llamacpp:n_decode_total 42
llamacpp:n_decode_total_extra 7
not_llamacpp:n_decode_total 8
llamacpp:requests_processing NaN
llamacpp:requests_processing +Inf
llamacpp:requests_processing -1
other_line
llamacpp:requests_processing{lane=\"a b\"} 2
";
        let sample = parse_metrics(body);
        assert_eq!(sample.n_decode_total, Some(42));
        assert_eq!(sample.requests_processing, Some(2.0));

        assert_eq!(sample.prompt_total, Some(999));

        let missing = parse_metrics("llamacpp:prompt_tokens_total 3\n");
        assert_eq!(missing.n_decode_total, None);
        assert_eq!(missing.requests_processing, None);
        assert_eq!(missing.prompt_total, Some(3));

        let fractional = parse_metrics("llamacpp:n_decode_total 10.9\n");
        assert_eq!(fractional.n_decode_total, Some(10));
    }

    #[test]
    fn metrics_counter_reset_and_new_model_baseline() {
        let mut counter = DecodedCounter::default();
        let t0 = Instant::now();
        assert_eq!(counter.total_if_fresh(true, &[], t0), Some(0));
        assert_eq!(counter.total_if_fresh(false, &[], t0), None);

        counter.observe("a", 100, Some(1.0), t0);
        assert_eq!(counter.total_if_fresh(true, &["a"], t0), Some(0));
        assert_eq!(counter.requests_processing("a", t0), Some(1.0));

        counter.observe("a", 140, Some(1.0), t0);
        assert_eq!(counter.total_if_fresh(true, &["a"], t0), Some(40));

        counter.observe("a", 10, Some(0.0), t0);
        assert_eq!(counter.total_if_fresh(true, &["a"], t0), Some(50));
        assert_eq!(counter.requests_processing("a", t0), Some(0.0));

        counter.observe("b", 1000, None, t0);
        assert_eq!(counter.total_if_fresh(true, &["a", "b"], t0), Some(50));
        assert_eq!(counter.requests_processing("b", t0), None);

        counter.observe("b", 1005, Some(2.0), t0);
        assert_eq!(counter.total_if_fresh(true, &["a", "b"], t0), Some(55));
        assert_eq!(counter.total_if_fresh(false, &["a", "b"], t0), None);
        assert_eq!(counter.total_if_fresh(true, &[], t0), Some(55));

        assert_eq!(
            counter.total_if_fresh(true, &["a"], t0 + Duration::from_secs(1)),
            Some(55)
        );
        assert_eq!(
            counter.total_if_fresh(true, &["a"], t0 + Duration::from_millis(1001)),
            None
        );
        assert_eq!(
            counter.total_if_fresh(true, &["missing"], t0),
            None,
            "a ready model with no metrics read is stale"
        );
    }

    #[test]
    fn sglang_names_sum_over_labels() {
        let body = "\
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name=\"flash\"} 1200.0
sglang:generation_tokens_total{model_name=\"flash\",dp=\"1\"} 34.0
sglang:prompt_tokens_total{model_name=\"flash\"} 5000.0
sglang:num_running_reqs{model_name=\"flash\"} 1.0
sglang:num_queue_reqs{model_name=\"flash\"} 2.0
sglang:token_usage{model_name=\"flash\"} 0.37
sglang:cache_hit_rate{model_name=\"flash\"} 0.8
sglang:gen_throughput{model_name=\"flash\"} 44.0
sglang:num_running_reqs_extra 9
llamacpp:n_decode_total 99
sglang:token_usage{model_name=\"x\"} NaN
sglang:num_queue_reqs{model_name=\"x\"} -3
";
        let sample = parse_metrics_for(Backend::SgLang, body);
        assert_eq!(sample.n_decode_total, Some(1234));
        assert_eq!(sample.prompt_total, Some(5000));
        assert_eq!(sample.requests_processing, Some(1.0));
        assert_eq!(sample.queued, Some(2.0));
        assert_eq!(sample.kv_fill, Some(0.37));
        assert_eq!(sample.cache_hit, Some(0.8));
        assert_eq!(sample.cached_total, None);
        let cached = format!(
            "{body}sglang:cached_tokens_total{{cache_source=\"device\",model_name=\"flash\"}} 4000.0\n\
             sglang:cached_tokens_total{{cache_source=\"host\",model_name=\"flash\"}} 500.0\n"
        );
        let sample = parse_metrics_for(Backend::SgLang, &cached);
        assert_eq!(sample.cached_total, Some(4500), "summed over cache sources");
        assert_eq!(parse_metrics(&cached).cached_total, None);
        // The llama.cpp name is not read for SGLang, nor SGLang's for llama.cpp.
        assert_eq!(parse_metrics(body).n_decode_total, Some(99));
        assert_eq!(parse_metrics(body).requests_processing, None);
        assert_eq!(
            parse_metrics_for(Backend::OpenAi, body),
            MetricsSample::default()
        );
    }

    #[test]
    fn vllm_names_kv_fallback_and_prefix_ratio() {
        let body = "\
vllm:generation_tokens_total{model_name=\"m\"} 77.0
vllm:prompt_tokens_total{model_name=\"m\"} 300.0
vllm:num_requests_running{model_name=\"m\"} 3.0
vllm:num_requests_waiting{model_name=\"m\"} 0.0
vllm:gpu_cache_usage_perc{model_name=\"m\"} 0.25
vllm:prefix_cache_hits_total{model_name=\"m\"} 30.0
vllm:prefix_cache_queries_total{model_name=\"m\"} 120.0
";
        let sample = parse_metrics_for(Backend::Vllm, body);
        assert_eq!(sample.n_decode_total, Some(77));
        assert_eq!(sample.prompt_total, Some(300));
        assert_eq!(sample.requests_processing, Some(3.0));
        assert_eq!(sample.queued, Some(0.0));
        assert_eq!(sample.kv_fill, Some(0.25));
        assert_eq!(sample.cache_hit, Some(0.25));
        assert_eq!(
            sample.cached_total,
            Some(30),
            "prefix hits are cached tokens"
        );
        let newer = format!("{body}vllm:kv_cache_usage_perc{{model_name=\"m\"}} 0.5\n");
        assert_eq!(parse_metrics_for(Backend::Vllm, &newer).kv_fill, Some(0.5));
        let over = "vllm:kv_cache_usage_perc 1.5\nvllm:prefix_cache_hits 5\n";
        let sample = parse_metrics_for(Backend::Vllm, over);
        assert_eq!(sample.kv_fill, None);
        assert_eq!(sample.cache_hit, None, "no queries, no ratio");
        assert_eq!(parse_metrics_for(Backend::Vllm, "").n_decode_total, None);
    }

    const VLLM_SAMPLE: &str = include_str!("../../../fixtures/llama/vllm-metrics.txt");

    /// #31: the vLLM 0.30 fixture, names as its source defines them.
    #[test]
    fn vllm_030_fixture_reads_every_engine_number() {
        let (sample, facts) = parse_metrics_full(Backend::Vllm, VLLM_SAMPLE);
        assert_eq!(sample.n_decode_total, Some(42_000));
        assert_eq!(sample.prompt_total, Some(500_000));
        assert_eq!(sample.requests_processing, Some(1.0));
        assert_eq!(sample.queued, Some(2.0), "not the by-reason split");
        assert_eq!(sample.kv_fill, Some(0.4125));
        assert_eq!(sample.cache_hit, Some(0.75));
        assert_eq!(
            sample.cached_total,
            Some(375_000),
            "prompt_tokens_cached wins over prefix_cache_hits"
        );
        assert_eq!(
            sample.spec,
            Some(SpecCounts {
                drafts: 20_000,
                draft_tokens: 60_000,
                accepted: 38_000,
            }),
            "the per-position counter is not the accepted total"
        );
        assert_eq!(sample.preemptions, Some(3));
        assert_eq!(sample.sleeping, Some(false));
        assert_eq!(
            sample.ttft,
            Some(Hist {
                sum: 31.5,
                count: 42.0
            })
        );
        assert_eq!(
            sample.itl,
            Some(Hist {
                sum: 1006.992,
                count: 41958.0
            })
        );
        assert_eq!(
            sample.e2e,
            Some(Hist {
                sum: 1050.0,
                count: 42.0
            })
        );
        assert_eq!(
            facts,
            Some(EngineFacts {
                ctx: None,
                kv: Some("fp8_e4m3".to_owned()),
                kv_block: Some(16),
                prefix_cache: Some(true),
            })
        );
        assert_eq!(detect_backend(VLLM_SAMPLE), Some(Backend::Vllm));
        // Read as another backend, nothing of it counts.
        assert_eq!(
            parse_metrics_for(Backend::SgLang, VLLM_SAMPLE),
            MetricsSample::default()
        );
    }

    #[test]
    fn vllm_older_names_and_sleep() {
        let body = "\
vllm:prefix_cache_hits_total{model_name=\"m\"} 30.0
vllm:prefix_cache_queries_total{model_name=\"m\"} 120.0
vllm:time_per_output_token_seconds_sum{model_name=\"m\"} 2.0
vllm:time_per_output_token_seconds_count{model_name=\"m\"} 100.0
vllm:engine_sleep_state{engine=\"0\",model_name=\"m\",sleep_state=\"awake\"} 0.0
vllm:engine_sleep_state{engine=\"0\",model_name=\"m\",sleep_state=\"weights_offloaded\"} 1.0
vllm:spec_decode_num_drafts_total{model_name=\"m\"} 5.0
";
        let (sample, facts) = parse_metrics_full(Backend::Vllm, body);
        assert_eq!(sample.cached_total, Some(30), "prefix hits as the fallback");
        assert_eq!(
            sample.itl,
            Some(Hist {
                sum: 2.0,
                count: 100.0
            }),
            "time_per_output_token on older builds"
        );
        assert_eq!(sample.sleeping, Some(true));
        assert_eq!(sample.spec, None, "all three spec counters or none");
        assert_eq!(sample.ttft, None);
        assert_eq!(facts, None);
        // Two engines: asleep when any is.
        let two = "vllm:engine_sleep_state{engine=\"0\",sleep_state=\"awake\"} 1\n\
                   vllm:engine_sleep_state{engine=\"1\",sleep_state=\"awake\"} 0\n";
        assert_eq!(parse_metrics_for(Backend::Vllm, two).sleeping, Some(true));
    }

    #[test]
    fn cache_config_labels_are_bounded_and_known_keys_only() {
        let facts = |labels: &str| {
            parse_metrics_full(
                Backend::Vllm,
                &format!("vllm:cache_config_info{{{labels}}} 1.0\n"),
            )
            .1
            .expect("facts")
        };
        let hostile = facts(&format!(
            "cache_dtype=\"../../etc\",block_size=\"0\",enable_prefix_caching=\"yes\",x=\"{}\"",
            "y".repeat(10_000)
        ));
        assert_eq!(hostile, EngineFacts::default());
        let long = facts(&format!(
            "cache_dtype=\"{}\",block_size=\"32\"",
            "a".repeat(65)
        ));
        assert_eq!(long.kv, None);
        assert_eq!(long.kv_block, Some(32), "a long value drops itself only");
        let odd = facts(
            r#"note="a,b=\"c\"}",cache_dtype="Kvarn_K4V2_G128",block_size="2000000",enable_prefix_caching="False""#,
        );
        assert_eq!(odd.kv.as_deref(), Some("kvarn_k4v2_g128"));
        assert_eq!(odd.kv_block, None, "above MAX_KV_BLOCK");
        assert_eq!(odd.prefix_cache, Some(false));
        // An unterminated value: the line is not a sample at all.
        let broken = "vllm:cache_config_info{block_size=\"16\",cache_dtype=\"fp8} 1.0\n";
        assert_eq!(parse_metrics_full(Backend::Vllm, broken).1, None);
        // A bad escape ends the scan; keys before it are kept.
        let escape = facts(r#"block_size="16",cache_dtype="fp\q8""#);
        assert_eq!(escape.kv_block, Some(16));
        assert_eq!(escape.kv, None);
        // A 17-character dtype is not a detail token.
        assert_eq!(facts(r#"cache_dtype="turboquant_k3v4_nc""#).kv, None);
    }

    #[test]
    fn detection_is_by_exact_sample_prefix() {
        assert_eq!(detect_backend(VLLM_SAMPLE), Some(Backend::Vllm));
        assert_eq!(
            detect_backend("# HELP vllm:x\nsglang:num_running_reqs{a=\"b\"} 1\n"),
            Some(Backend::SgLang),
            "comments are not samples"
        );
        assert_eq!(
            detect_backend("llamacpp:n_decode_total 5\n"),
            Some(Backend::LlamaCpp)
        );
        for none in [
            "",
            "\n",
            "# HELP vllm:num_requests_running x\n",
            "python_gc_objects_collected_total{generation=\"0\"} 1\nvllmx:a 1\n",
            "vllm: 1\n",
            "{\"engine\":{}}",
            "VLLM:num_requests_running 1\n",
        ] {
            assert_eq!(detect_backend(none), None, "{none:?}");
        }
    }

    fn spec(drafts: u64, draft_tokens: u64, accepted: u64) -> MetricsSample {
        MetricsSample {
            spec: Some(SpecCounts {
                drafts,
                draft_tokens,
                accepted,
            }),
            ..MetricsSample::default()
        }
    }

    #[test]
    fn spec_window_rates_and_counters_across_a_reset() {
        let mut book = EngineBook::default();
        // First read: the server's totals since start, counters at zero.
        let first = book.observe("m", &spec(100, 300, 150));
        assert_eq!(first.spec_permille, Some(500));
        assert_eq!(first.spec_len_centi, Some(250));
        assert_eq!(first.spec_counts, Some(SpecCounts::default()));
        // A window: 10 drafts of 3, 24 accepted: 80 %, 3.4 per step.
        let next = book.observe("m", &spec(110, 330, 174));
        assert_eq!(next.spec_permille, Some(800));
        assert_eq!(next.spec_len_centi, Some(340));
        assert_eq!(
            next.spec_counts,
            Some(SpecCounts {
                drafts: 10,
                draft_tokens: 30,
                accepted: 24,
            })
        );
        // Idle: no new drafts keeps the last window.
        let idle = book.observe("m", &spec(110, 330, 174));
        assert_eq!(idle.spec_permille, Some(800));
        assert_eq!(idle.spec_len_centi, Some(340));
        // The server restarted: its new totals are the window and add on.
        let reset = book.observe("m", &spec(4, 12, 3));
        assert_eq!(reset.spec_permille, Some(250));
        assert_eq!(reset.spec_len_centi, Some(175));
        assert_eq!(
            reset.spec_counts,
            Some(SpecCounts {
                drafts: 14,
                draft_tokens: 42,
                accepted: 27,
            })
        );
        // One counter alone going back is a restart too.
        let odd = book.observe("m", &spec(5, 15, 2));
        assert_eq!(odd.spec_permille, Some(133), "2 of 15");
        // Unloaded: window gone, counters kept for the next load.
        book.forget("m");
        let again = book.observe("m", &spec(1, 3, 3));
        assert_eq!(again.spec_permille, Some(1000));
        assert_eq!(again.spec_counts.map(|c| c.drafts), Some(19));
        // A server that stops reporting spec shows none.
        let gone = book.observe("m", &MetricsSample::default());
        assert_eq!(gone.spec_permille, None);
        assert_eq!(gone.spec_counts, None);
    }

    #[test]
    fn spec_accepted_above_drafted_is_clamped() {
        let mut book = EngineBook::default();
        book.observe("m", &spec(0, 0, 0));
        let odd = book.observe("m", &spec(1, 3, 9));
        assert_eq!(odd.spec_permille, Some(1000));
        let counts = odd.spec_counts.expect("counts");
        assert!(counts.accepted <= counts.draft_tokens);
        assert_eq!(odd.spec_len_centi, Some(400));
    }

    #[test]
    fn preemptions_and_latency_means_over_the_window() {
        let mut book = EngineBook::default();
        let hist = |sum: f64, count: f64| Some(Hist { sum, count });
        let read = |preempt: u64, ttft: Option<Hist>, e2e: Option<Hist>| MetricsSample {
            preemptions: Some(preempt),
            ttft,
            itl: hist(10.0, 1000.0),
            e2e,
            ..MetricsSample::default()
        };
        let first = book.observe("m", &read(7, hist(30.0, 40.0), hist(400.0, 40.0)));
        assert_eq!(first.preemptions, Some(0));
        assert_eq!(
            first.ttft_us,
            Some(750_000),
            "lifetime mean on the first read"
        );
        assert_eq!(first.itl_us, Some(10_000));
        assert_eq!(first.e2e_us, Some(10_000_000));
        let next = book.observe("m", &read(9, hist(31.0, 42.0), hist(440.0, 42.0)));
        assert_eq!(next.preemptions, Some(2));
        assert_eq!(next.ttft_us, Some(500_000), "1 s over 2 requests");
        assert_eq!(next.e2e_us, Some(20_000_000));
        assert_eq!(next.itl_us, Some(10_000), "no new tokens keeps the mean");
        // Restart: counters go down; the new value adds, the new totals are the window.
        let reset = book.observe("m", &read(1, hist(0.2, 1.0), hist(3.0, 1.0)));
        assert_eq!(reset.preemptions, Some(3));
        assert_eq!(reset.ttft_us, Some(200_000));
        assert_eq!(reset.e2e_us, Some(3_000_000));
        // A mean past the cap is not shown.
        let slow = book.observe("m", &read(1, hist(0.2, 1.0), hist(10_003.0, 2.0)));
        assert_eq!(slow.e2e_us, None);
    }

    #[test]
    fn sglang_spec_gauges_and_histograms() {
        let body = "\
sglang:spec_accept_rate{model_name=\"flash\"} 0.72
sglang:spec_accept_length{model_name=\"flash\"} 3.1
sglang:time_to_first_token_seconds_sum{model_name=\"flash\"} 4.0
sglang:time_to_first_token_seconds_count{model_name=\"flash\"} 8.0
sglang:e2e_request_latency_seconds_sum{model_name=\"flash\"} 80.0
sglang:e2e_request_latency_seconds_count{model_name=\"flash\"} 8.0
";
        let sample = parse_metrics_for(Backend::SgLang, body);
        assert_eq!(sample.spec_rate, Some(0.72));
        assert_eq!(sample.spec_len, Some(3.1));
        assert_eq!(sample.spec, None);
        let mut book = EngineBook::default();
        let stats = book.observe("flash", &sample);
        assert_eq!(stats.spec_permille, Some(720));
        assert_eq!(stats.spec_len_centi, Some(310));
        assert_eq!(stats.spec_counts, None, "SGLang has no spec counters");
        assert_eq!(stats.ttft_us, Some(500_000));
        assert_eq!(stats.e2e_us, Some(10_000_000));
        assert_eq!(stats.itl_us, None);
        // Spec off: SGLang keeps both gauges at 0.
        let off = parse_metrics_for(
            Backend::SgLang,
            "sglang:spec_accept_rate 0\nsglang:spec_accept_length 0\n",
        );
        let stats = book.observe("flash", &off);
        assert_eq!(stats.spec_permille, None);
        assert_eq!(stats.spec_len_centi, None);
    }

    #[test]
    fn engine_book_is_capped() {
        let mut book = EngineBook::default();
        for i in 0..ENGINE_MODELS + 3 {
            book.observe(&format!("m{i}"), &spec(1, 1, 1));
        }
        assert_eq!(book.models.len(), ENGINE_MODELS);
        assert!(
            book.observe("late", &spec(1, 1, 1)).is_empty(),
            "a model past the cap gets nothing"
        );
    }

    #[test]
    fn activity_tokens_count_once_and_rebase_the_metrics_counter() {
        let mut counter = DecodedCounter::default();
        let t0 = Instant::now();
        let window = Duration::from_secs(3);
        counter.observe("m", 100, Some(1.0), t0);
        counter.add("m", 25, t0, window);
        assert_eq!(counter.total(), 25);
        assert_eq!(counter.requests_processing("m", t0), None);
        assert_eq!(
            counter.total_if_fresh(true, &["m"], t0 + Duration::from_millis(2900)),
            Some(25),
            "an activity read stays fresh for its window"
        );
        assert_eq!(
            counter.total_if_fresh(true, &["m"], t0 + Duration::from_millis(3100)),
            None
        );
        // /metrics comes back: a new baseline, no jump.
        counter.observe("m", 500, Some(0.0), t0);
        assert_eq!(counter.total(), 25);
        counter.observe("m", 510, Some(0.0), t0);
        assert_eq!(counter.total(), 35);
        assert_eq!(
            counter.total_if_fresh(true, &["m"], t0 + Duration::from_millis(1100)),
            None,
            "a /metrics read is fresh for one second again"
        );
    }

    #[test]
    fn prompt_cache_counts_rows_once_and_prefers_metrics_counters() {
        let mut cache = PromptCache::default();
        assert_eq!(cache.get("llama"), None);
        // llama.cpp timings: input is what was processed, cache the reuse.
        cache.add_row("llama", Some(69), Some(553));
        cache.add_row("llama", Some(1_000), Some(0));
        assert_eq!(cache.get("llama"), Some((1_622, Some(553))));
        // An OpenAI usage row: the prompt only, nothing known cached.
        cache.add_row("tabby", Some(300), None);
        cache.add_row("tabby", None, Some(5));
        assert_eq!(cache.get("tabby"), Some((300, None)));

        // SGLang: the first read is the baseline, then deltas, then a
        // server restart adds its new value. Rows are ignored meanwhile.
        cache.observe_metrics("flash", Some(10_000), Some(9_000));
        assert_eq!(cache.get("flash"), Some((0, Some(0))));
        cache.add_row("flash", Some(123), None);
        cache.observe_metrics("flash", Some(10_500), Some(9_400));
        assert_eq!(cache.get("flash"), Some((500, Some(400))));
        cache.observe_metrics("flash", Some(50), Some(10));
        assert_eq!(cache.get("flash"), Some((550, Some(410))));
        // Metrics without the cached counter: rows count again, from here.
        cache.observe_metrics("flash", Some(60), None);
        cache.add_row("flash", Some(7), None);
        assert_eq!(cache.get("flash"), Some((557, Some(410))));
        // Counters never go down within a run.
        cache.observe_metrics("flash", Some(100), Some(90));
        cache.observe_metrics("flash", Some(100), Some(95));
        assert_eq!(cache.get("flash"), Some((557, Some(415))));
    }

    #[test]
    fn prompt_cache_cached_never_exceeds_prompt_and_models_are_capped() {
        let mut cache = PromptCache::default();
        cache.observe_metrics("m", Some(0), Some(0));
        cache.observe_metrics("m", Some(10), Some(50));
        assert_eq!(cache.get("m"), Some((10, Some(10))));
        for i in 0..PROMPT_CACHE_MODELS + 5 {
            cache.add_row(&format!("x{i}"), Some(1), Some(0));
        }
        assert_eq!(cache.models.len(), PROMPT_CACHE_MODELS);
        assert_eq!(cache.get(&format!("x{}", PROMPT_CACHE_MODELS + 1)), None);
    }

    #[test]
    fn metrics_gen_rate_over_one_second() {
        let mut rate = GenRate::default();
        let t0 = Instant::now();
        assert_eq!(rate.observe(t0, 0), None);
        assert_eq!(rate.observe(t0 + Duration::from_millis(500), 10), None);
        let got = rate
            .observe(t0 + Duration::from_millis(1000), 40)
            .expect("one second of samples");
        assert!((got - 40.0).abs() < 1e-6, "{got}");
    }
}
