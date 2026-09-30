//! `/upstream/<model>/metrics` line parse and the box decoded counter.
//!
//! Each backend has its own Prometheus names (T72). Names are matched
//! exactly; SGLang and vLLM label their series (`model_name`, `engine`), so
//! their values are summed across label sets, and a ratio takes the largest.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use llama_core::backend::Backend;
use llama_core::rate::{counter_delta, delta_per_s};

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
    /// Cached prompt-token counter (#10). vLLM's is its prefix hit counter.
    cached: &'static [&'static str],
    /// Sum across label sets. llama.cpp keeps its last value, as before T72.
    sum: bool,
}

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
    cached: &["vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"],
    sum: true,
};

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
    /// `vllm:prefix_cache_hits_total`), part of [`Self::prompt_total`] (#10).
    pub cached_total: Option<u64>,
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
    let names = match backend {
        Backend::LlamaCpp => &LLAMACPP,
        Backend::SgLang => &SGLANG,
        Backend::Vllm => &VLLM,
        Backend::OpenAi => return MetricsSample::default(),
    };
    let mut decode = Acc::default();
    let mut running = Acc::default();
    let mut prompt = Acc::default();
    let mut queued = Acc::default();
    let mut kv: [Option<f64>; 2] = [None, None];
    let mut hit = None;
    let mut hits = Acc::default();
    let mut queries = Acc::default();
    let mut cached = Acc::default();
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
        }
        if names.cached.contains(&name) {
            cached.add(value, names.sum);
        }
    }
    let prefix = match (hits.0, queries.0) {
        (Some(hits), Some(queries)) if queries > 0.0 && hits <= queries => Some(hits / queries),
        _ => None,
    };
    MetricsSample {
        n_decode_total: decode.0.and_then(finite_u64),
        requests_processing: running.0,
        prompt_total: prompt.0.and_then(finite_u64),
        queued: queued.0,
        kv_fill: kv[0].or(kv[1]),
        cache_hit: hit.or(prefix),
        cached_total: cached.0.and_then(finite_u64),
    }
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
        let close = line[name_end..].find('}')?;
        line[name_end + close + 1..].trim_start()
    } else {
        line[name_end..].trim_start()
    };
    let token = rest.split_whitespace().next()?;
    let value: f64 = token.parse().ok()?;
    Some((name, value))
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
