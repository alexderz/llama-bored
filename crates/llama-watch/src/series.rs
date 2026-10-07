//! Per-model cumulative numbers for llama-metrics' normalized series (#71).
//!
//! Every engine's quantities that mean the same thing are folded into one
//! set of counters per llama-swap model id, so llama-metrics exports the
//! same series names for llama.cpp, vLLM, SGLang, Strata and any other
//! OpenAI-compatible server, told apart only by the `engine` label.
//!
//! # Sources
//!
//! | Number | llama.cpp | vLLM | SGLang | Strata | OpenAI-compatible |
//! |---|---|---|---|---|---|
//! | generated tokens | `tokens_predicted_total` | `generation_tokens_total` | `generation_tokens_total` | `totals.output_tokens` + `live.generated` | activity `output_tokens` |
//! | prefill / decode seconds | `prompt_seconds_total` / `tokens_predicted_seconds_total` | per-request prefill / decode time sums | none | `totals.prompt_ms` / `decode_ms` | none |
//! | TTFT, ITL | none | histogram sum and count | histogram sum and count | none | none |
//! | request duration | activity `duration_ms` | e2e histogram | e2e histogram | activity | activity |
//! | requests by status | activity | activity | activity | activity | activity |
//! | spec drafts (tokens, accepted) | activity draft fields | engine counters ([`crate::metrics::EngineBook`]) | none | engine counters | none |
//! | running, waiting, KV fill | `requests_processing`, `requests_deferred`, `kv_cache_usage_ratio` (older builds) | on [`llama_core::backend::BackendInfo`] | same | same | none |
//!
//! A model whose server has no usable `/metrics` (and is counted from
//! llama-swap's activity rows, like [`crate::metrics::DecodedCounter`])
//! gets its generated tokens from those rows instead.
//!
//! # Monotonic within a run
//!
//! Each engine counter is folded in the way [`crate::metrics::DecodedCounter`]
//! does it: the first read of a model is its baseline and adds nothing; a
//! read at or above the last adds the difference; a read below it means the
//! server restarted and adds the new value. When the poller knows the
//! process is gone (the model left `ready`, or llama-swap went away), the
//! baseline drops to zero, so the next process is counted from its start.
//! A histogram's sum and count are one reading: if either went down, both
//! new values are added. Activity rows are counted once each, by
//! llama-watch's own row numbering (#44).
//!
//! A model keeps its totals for the whole run, across unloads and reloads,
//! so llama-metrics' counters only go back to zero when llama-watch
//! restarts. Two cases undercount rather than go backwards: a server that
//! restarts unseen and reports more than it had before by the next read,
//! and the requests a new process finishes before its first read when the
//! poller did not see the old one go. At most [`SERIES_MODELS`] ids are
//! tracked per run; later ones get no numbers.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use llama_core::backend::{self, Backend};

use crate::activity::ActivityRow;
use crate::metrics::{Hist, MetricsSample};

/// Most models [`SeriesBook`] keeps numbers for in one watcher run.
pub const SERIES_MODELS: usize = 64;

/// One model's numbers, as the publisher puts them on the wire.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSeries {
    /// Display name, as [`llama_core::sample::ModelInfo::name`] has it.
    pub model: String,
    /// The llama-swap model id, unsanitised; the publisher sanitises it.
    pub id: String,
    /// The engine's version, when it reports one (Strata, #54).
    pub version: Option<String>,
    /// Requests running now, from llama.cpp's `/metrics` (#71). Other
    /// engines carry theirs on [`llama_core::backend::BackendInfo`].
    pub running: Option<u16>,
    /// Requests waiting for a slot, llama.cpp only.
    pub waiting: Option<u16>,
    /// KV cache fill 0..=1000, llama.cpp builds that report it.
    pub kv_permille: Option<u16>,
    /// Tokens generated since the watcher started.
    pub generation_tokens: Option<u64>,
    /// Seconds of prompt processing since the watcher started.
    pub prefill_seconds: Option<f64>,
    /// Seconds of generation since the watcher started.
    pub decode_seconds: Option<f64>,
    /// Finished requests with a 2xx answer, from llama-swap's activity rows.
    pub requests_ok: Option<u64>,
    /// Finished requests with any other answer.
    pub requests_error: Option<u64>,
    /// Time to first token: seconds summed, and requests.
    pub ttft: Option<Hist>,
    /// Inter-token latency: seconds summed, and tokens.
    pub itl: Option<Hist>,
    /// Request duration: seconds summed, and requests.
    pub e2e: Option<Hist>,
    /// Draft tokens from llama.cpp's activity timings (#71).
    pub spec_draft_tokens: Option<u64>,
    /// Of those, accepted.
    pub spec_accepted_tokens: Option<u64>,
}

/// One source counter folded into a total since the watcher started.
#[derive(Clone, Copy, Debug, Default)]
struct Count {
    last: Option<f64>,
    total: Option<f64>,
}

impl Count {
    fn observe(&mut self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        let add = match self.last {
            None => 0.0,
            Some(last) if value >= last => value - last,
            Some(_) => value,
        };
        self.total = Some(self.total.unwrap_or(0.0) + add);
        self.last = Some(value);
    }

    /// Counted outside `/metrics`: a counter that comes back later starts a
    /// new baseline instead of adding what was counted here.
    fn add(&mut self, value: f64) {
        self.total = Some(self.total.unwrap_or(0.0) + value);
        self.last = None;
    }

    fn restart(&mut self) {
        if self.last.is_some() {
            self.last = Some(0.0);
        }
    }

    fn get(&self) -> Option<f64> {
        self.total
    }
}

/// A histogram's `_sum` and `_count`, folded in together.
#[derive(Clone, Copy, Debug, Default)]
struct SumCount {
    last: Option<Hist>,
    total: Option<Hist>,
}

impl SumCount {
    fn observe(&mut self, now: Hist) {
        if !(now.sum.is_finite() && now.count.is_finite() && now.sum >= 0.0 && now.count >= 0.0) {
            return;
        }
        let add = match self.last {
            None => Hist::default(),
            Some(last) if now.count >= last.count && now.sum >= last.sum => Hist {
                sum: now.sum - last.sum,
                count: now.count - last.count,
            },
            Some(_) => now,
        };
        self.add_values(add);
        self.last = Some(now);
    }

    fn add_values(&mut self, add: Hist) {
        let total = self.total.get_or_insert_with(Hist::default);
        total.sum += add.sum;
        total.count += add.count;
    }

    fn restart(&mut self) {
        if self.last.is_some() {
            self.last = Some(Hist::default());
        }
    }
}

/// llama.cpp's running, waiting and KV fill, and when they were read.
#[derive(Clone, Copy, Debug)]
struct LiveGauges {
    running: Option<u16>,
    waiting: Option<u16>,
    kv_permille: Option<u16>,
    at: Instant,
}

#[derive(Debug, Default)]
struct Track {
    generation: Count,
    prefill: Count,
    decode: Count,
    ttft: SumCount,
    itl: SumCount,
    e2e: SumCount,
    /// The engine reported an e2e histogram this run: activity durations
    /// no longer count.
    e2e_engine: bool,
    requests_ok: Option<u64>,
    requests_error: Option<u64>,
    spec: Option<(u64, u64)>,
    gauges: Option<LiveGauges>,
}

/// Per-model cumulative numbers, keyed by llama-swap id.
#[derive(Debug, Default)]
pub struct SeriesBook {
    models: HashMap<String, Track>,
}

impl SeriesBook {
    fn entry(&mut self, id: &str) -> Option<&mut Track> {
        if !self.models.contains_key(id) && self.models.len() >= SERIES_MODELS {
            return None;
        }
        Some(self.models.entry(id.to_owned()).or_default())
    }

    /// Fold in one `/metrics` read of `id`, served by `backend`.
    pub fn observe_metrics(
        &mut self,
        id: &str,
        backend: Backend,
        sample: &MetricsSample,
        at: Instant,
    ) {
        let Some(track) = self.entry(id) else {
            return;
        };
        let lcpp = backend == Backend::LlamaCpp;
        let generated = if lcpp {
            sample.predicted_total
        } else {
            sample.n_decode_total
        };
        if let Some(generated) = generated {
            track.generation.observe(generated as f64);
        }
        let speeds = sample.speeds.unwrap_or_default();
        let prefill = if lcpp {
            sample.prompt_seconds
        } else {
            speeds.prefill.map(|phase| phase.seconds.sum)
        };
        let decode = if lcpp {
            sample.predicted_seconds
        } else {
            speeds.decode.map(|phase| phase.seconds.sum)
        };
        if let Some(seconds) = prefill {
            track.prefill.observe(seconds);
        }
        if let Some(seconds) = decode {
            track.decode.observe(seconds);
        }
        if let Some(hist) = sample.ttft {
            track.ttft.observe(hist);
        }
        if let Some(hist) = sample.itl {
            track.itl.observe(hist);
        }
        if let Some(hist) = sample.e2e {
            track.e2e_engine = true;
            track.e2e.observe(hist);
        }
        if lcpp {
            track.gauges = Some(LiveGauges {
                running: sample.requests_processing.and_then(backend::reqs),
                waiting: sample.queued.and_then(backend::reqs),
                kv_permille: sample.kv_fill.and_then(backend::permille),
                at,
            });
        }
    }

    /// `tokens` generated by `id`, counted from activity rows because its
    /// server has no usable `/metrics`.
    pub fn add_generation(&mut self, id: &str, tokens: u64) {
        if let Some(track) = self.entry(id) {
            track.generation.add(tokens as f64);
        }
    }

    /// llama-swap's activity log was read with `id` ready: its request
    /// counters exist from now on, at 0 until a request finishes, and so
    /// does its request duration unless the engine reports one.
    pub fn touch(&mut self, id: &str) {
        let Some(track) = self.entry(id) else {
            return;
        };
        track.requests_ok.get_or_insert(0);
        track.requests_error.get_or_insert(0);
        if !track.e2e_engine && track.e2e.total.is_none() {
            track.e2e.total = Some(Hist::default());
        }
    }

    /// One new activity row of `id`. `drafts` reads its speculative
    /// draft fields (llama.cpp, whose timings carry them).
    pub fn add_row(&mut self, id: &str, row: &ActivityRow, drafts: bool) {
        let Some(track) = self.entry(id) else {
            return;
        };
        let ok = row.status.is_some_and(|code| (200..300).contains(&code));
        let counter = if ok {
            &mut track.requests_ok
        } else {
            &mut track.requests_error
        };
        *counter = Some(counter.unwrap_or(0).saturating_add(1));
        if !track.e2e_engine
            && let Some(ms) = row.duration_ms
        {
            track.e2e.add_values(Hist {
                sum: ms as f64 / 1000.0,
                count: 1.0,
            });
        }
        if drafts && let Some(draft) = row.draft_tokens.filter(|n| *n > 0) {
            let accepted = row.draft_accepted.unwrap_or(0).min(draft);
            let (tokens, kept) = track.spec.get_or_insert((0, 0));
            *tokens = tokens.saturating_add(draft);
            *kept = kept.saturating_add(accepted);
        }
    }

    /// `id`'s process is gone (#46): the next one is counted from 0.
    pub fn forget_process(&mut self, id: &str) {
        if let Some(track) = self.models.get_mut(id) {
            track.generation.restart();
            track.prefill.restart();
            track.decode.restart();
            track.ttft.restart();
            track.itl.restart();
            track.e2e.restart();
            track.gauges = None;
        }
    }

    /// `id`'s live gauges are no longer known (unloaded, not available).
    pub fn forget_gauges(&mut self, id: &str) {
        if let Some(track) = self.models.get_mut(id) {
            track.gauges = None;
        }
    }

    /// `id`'s numbers, with its gauges only when read within `fresh`.
    #[must_use]
    pub fn get(&self, id: &str, now: Instant, fresh: Duration) -> ModelSeries {
        let Some(track) = self.models.get(id) else {
            return ModelSeries {
                id: id.to_owned(),
                ..ModelSeries::default()
            };
        };
        let (running, waiting, kv_permille) = track
            .gauges
            .filter(|gauges| now.saturating_duration_since(gauges.at) <= fresh)
            .map_or((None, None, None), |gauges| {
                (gauges.running, gauges.waiting, gauges.kv_permille)
            });
        ModelSeries {
            model: String::new(),
            id: id.to_owned(),
            version: None,
            running,
            waiting,
            kv_permille,
            generation_tokens: track.generation.get().map(|n| n as u64),
            prefill_seconds: track.prefill.get(),
            decode_seconds: track.decode.get(),
            requests_ok: track.requests_ok,
            requests_error: track.requests_error,
            ttft: track.ttft.total,
            itl: track.itl.total,
            e2e: track.e2e.total,
            spec_draft_tokens: track.spec.map(|(tokens, _)| tokens),
            spec_accepted_tokens: track.spec.map(|(_, accepted)| accepted),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speeds::{Phase, SpeedTotals};

    const FRESH: Duration = Duration::from_secs(5);

    fn lcpp(predicted: u64, prompt_s: f64, predicted_s: f64) -> MetricsSample {
        MetricsSample {
            n_decode_total: Some(predicted * 3),
            predicted_total: Some(predicted),
            prompt_seconds: Some(prompt_s),
            predicted_seconds: Some(predicted_s),
            requests_processing: Some(1.0),
            queued: Some(2.0),
            ..MetricsSample::default()
        }
    }

    fn vllm(generated: u64, ttft: (f64, f64), e2e: (f64, f64), prefill_s: f64) -> MetricsSample {
        let hist = |(sum, count): (f64, f64)| Some(Hist { sum, count });
        MetricsSample {
            n_decode_total: Some(generated),
            ttft: hist(ttft),
            itl: hist((ttft.0 / 10.0, ttft.1 * 10.0)),
            e2e: hist(e2e),
            speeds: Some(SpeedTotals {
                prefill: Some(Phase {
                    seconds: Hist {
                        sum: prefill_s,
                        count: e2e.1,
                    },
                    tokens: Hist::default(),
                }),
                decode: None,
            }),
            ..MetricsSample::default()
        }
    }

    fn row(status: u16, ms: u64, drafts: Option<(u64, u64)>) -> ActivityRow {
        ActivityRow {
            id: 1,
            seq: 1,
            time: String::new(),
            source: String::new(),
            model: "m".to_owned(),
            input_tokens: Some(1),
            cached_tokens: None,
            output_tokens: Some(1),
            prompt_tps: None,
            gen_tps: None,
            engine_prompt_tps: None,
            engine_gen_tps: None,
            duration_ms: Some(ms),
            status: Some(status),
            draft_tokens: drafts.map(|(tokens, _)| tokens),
            draft_accepted: drafts.map(|(_, accepted)| accepted),
            captured: false,
        }
    }

    /// llama.cpp's request-end counters, not its decode calls, are the
    /// generated tokens; its seconds are its own counters.
    #[test]
    fn llamacpp_counters_are_monotonic_across_a_restart() {
        let mut book = SeriesBook::default();
        let t0 = Instant::now();
        book.observe_metrics("m", Backend::LlamaCpp, &lcpp(100, 2.0, 10.0), t0);
        let first = book.get("m", t0, FRESH);
        assert_eq!(first.generation_tokens, Some(0), "first read is a baseline");
        assert_eq!(first.prefill_seconds, Some(0.0));
        assert_eq!(first.running, Some(1));
        assert_eq!(first.waiting, Some(2));
        book.observe_metrics("m", Backend::LlamaCpp, &lcpp(150, 3.0, 15.0), t0);
        let second = book.get("m", t0, FRESH);
        assert_eq!(second.generation_tokens, Some(50));
        assert_eq!(second.prefill_seconds, Some(1.0));
        assert_eq!(second.decode_seconds, Some(5.0));
        // The server restarted unseen: its new value is added.
        book.observe_metrics("m", Backend::LlamaCpp, &lcpp(20, 0.5, 1.0), t0);
        let third = book.get("m", t0, FRESH);
        assert_eq!(third.generation_tokens, Some(70));
        assert_eq!(third.decode_seconds, Some(6.0));
        // The poller saw the process go: the next one counts from 0.
        book.forget_process("m");
        assert_eq!(book.get("m", t0, FRESH).running, None);
        book.observe_metrics("m", Backend::LlamaCpp, &lcpp(30, 0.5, 2.0), t0);
        let fourth = book.get("m", t0, FRESH);
        assert_eq!(fourth.generation_tokens, Some(100));
        assert_eq!(fourth.decode_seconds, Some(8.0));
        assert_eq!(fourth.ttft, None, "llama.cpp reports no TTFT");
        // Gauges go stale; counters stay.
        let late = book.get("m", t0 + FRESH * 2, FRESH);
        assert_eq!(late.running, None);
        assert_eq!(late.generation_tokens, Some(100));
    }

    /// vLLM's histogram sums and counts are folded together; a reset of
    /// either adds both new values. Its e2e replaces activity durations.
    #[test]
    fn engine_histograms_are_cumulative_and_restart_safe() {
        let mut book = SeriesBook::default();
        let t0 = Instant::now();
        book.touch("v");
        book.add_row("v", &row(200, 1000, None), false);
        assert_eq!(
            book.get("v", t0, FRESH).e2e,
            Some(Hist {
                sum: 1.0,
                count: 1.0
            }),
            "activity duration before the engine's"
        );
        book.observe_metrics(
            "v",
            Backend::Vllm,
            &vllm(1000, (4.0, 8.0), (40.0, 8.0), 2.0),
            t0,
        );
        book.observe_metrics(
            "v",
            Backend::Vllm,
            &vllm(1500, (6.0, 10.0), (50.0, 10.0), 3.0),
            t0,
        );
        let got = book.get("v", t0, FRESH);
        assert_eq!(got.generation_tokens, Some(500));
        assert_eq!(
            got.ttft,
            Some(Hist {
                sum: 2.0,
                count: 2.0
            })
        );
        assert_eq!(
            got.e2e,
            Some(Hist {
                sum: 11.0,
                count: 3.0
            })
        );
        assert_eq!(got.prefill_seconds, Some(1.0));
        assert_eq!(got.running, None, "vLLM gauges ride on BackendInfo");
        // A row now adds a request but no duration.
        book.add_row("v", &row(500, 1000, None), false);
        let got = book.get("v", t0, FRESH);
        assert_eq!(got.e2e.map(|h| h.count), Some(3.0));
        assert_eq!((got.requests_ok, got.requests_error), (Some(1), Some(1)));
        // Restart: count went down, both new values are added.
        book.observe_metrics(
            "v",
            Backend::Vllm,
            &vllm(10, (1.0, 1.0), (5.0, 1.0), 0.5),
            t0,
        );
        let got = book.get("v", t0, FRESH);
        assert_eq!(
            got.ttft,
            Some(Hist {
                sum: 3.0,
                count: 3.0
            })
        );
        assert_eq!(got.generation_tokens, Some(510));
    }

    #[test]
    fn activity_counts_requests_drafts_and_fallback_tokens() {
        let mut book = SeriesBook::default();
        let t0 = Instant::now();
        assert_eq!(book.get("o", t0, FRESH).requests_ok, None);
        book.touch("o");
        let got = book.get("o", t0, FRESH);
        assert_eq!((got.requests_ok, got.requests_error), (Some(0), Some(0)));
        assert_eq!(got.e2e, Some(Hist::default()));
        book.add_row("o", &row(200, 250, Some((10, 7))), true);
        book.add_row("o", &row(200, 250, Some((0, 0))), true);
        book.add_row("o", &row(503, 500, None), true);
        book.add_generation("o", 42);
        let got = book.get("o", t0, FRESH);
        assert_eq!((got.requests_ok, got.requests_error), (Some(2), Some(1)));
        assert_eq!(
            got.e2e,
            Some(Hist {
                sum: 1.0,
                count: 3.0
            })
        );
        assert_eq!(got.spec_draft_tokens, Some(10));
        assert_eq!(got.spec_accepted_tokens, Some(7));
        assert_eq!(got.generation_tokens, Some(42));
        // Without draft reading (not llama.cpp), no spec counters.
        book.touch("x");
        book.add_row("x", &row(200, 1, Some((10, 7))), false);
        assert_eq!(book.get("x", t0, FRESH).spec_draft_tokens, None);
    }

    #[test]
    fn the_book_is_bounded() {
        let mut book = SeriesBook::default();
        for n in 0..SERIES_MODELS + 3 {
            book.touch(&format!("m{n}"));
        }
        assert_eq!(book.models.len(), SERIES_MODELS);
        let t0 = Instant::now();
        assert_eq!(
            book.get(&format!("m{}", SERIES_MODELS + 1), t0, FRESH)
                .requests_ok,
            None
        );
    }
}
