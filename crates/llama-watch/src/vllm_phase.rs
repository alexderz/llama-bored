//! Whether a vLLM model's running requests are prefilling or decoding, for
//! the READ chart (DER-451).
//!
//! vLLM reports no per-request phase. Its time-to-first-token histogram
//! (`vllm:time_to_first_token_seconds`) counts a request when its first
//! token comes out, at the end of its prefill, and its end-to-end histogram
//! (`vllm:e2e_request_latency_seconds`) counts it when it finishes. Both
//! come from the same scrape as `vllm:num_requests_running`, so within one
//! read
//!
//! - decoding = TTFT count − e2e count − offset, and
//! - prefilling = running − decoding.
//!
//! A request queued in `num_requests_waiting` is in neither: it is not
//! prefilling. The offset absorbs requests that never balance the two
//! counts (an abort can record one histogram and not the other). It starts
//! at 0, because both counters start with the process, and it is set to the
//! difference at every read with nothing running or waiting, when the true
//! answer is 0. A counter that goes down (a restart or reload) puts it back
//! to 0. A difference outside `[0, running]` is drift this read cannot
//! explain, and the phase is unknown until the next idle read corrects it.
//! So is "all decoding" when `vllm:generation_tokens_total` has not moved
//! for [`DECODE_STALL`]: a decoding request produces tokens every step, so
//! a stall means the difference counts a request that is not running.
//!
//! A preempted request keeps its first-token count while it waits and
//! recomputes, so it reads as decoding when it is not. While
//! `vllm:num_preemptions_total` is above its value at the last idle read,
//! the phase is unknown until the next one. And "prefill, nothing
//! decoding" is unknown when `generation_tokens_total` moved by more than
//! the running count since the read before: something produced tokens, so
//! the zero is drift (an abort that recorded an e2e without a first token).
//!
//! The phase is per model: one request prefilling while another decodes is
//! a prefill. It is known only from the latest read, and only while that
//! read is fresh. Nothing here is a rate.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::metrics::MetricsSample;
use crate::slots::ChartPhase;

/// Longest time all-decoding is believed with no generated token.
pub const DECODE_STALL: Duration = Duration::from_secs(1);

/// Most models tracked in one watcher run.
const MAX_MODELS: usize = 64;

#[derive(Debug, Default)]
struct Track {
    /// Time of the latest read.
    at: Option<Instant>,
    /// TTFT and e2e counts of the latest read.
    ttft: f64,
    e2e: f64,
    /// TTFT − e2e at the latest idle read; 0 for a fresh process.
    offset: f64,
    /// `generation_tokens_total` of the latest read, and when it last moved.
    generated: Option<u64>,
    generated_at: Option<Instant>,
    /// `num_preemptions_total` at the latest idle read; 0 for a fresh
    /// process, so an earlier preemption is unknown until the first one.
    preempt_base: u64,
    /// The phase the latest read gave, while something was running.
    phase: Option<ChartPhase>,
}

/// Per-model phase of the running vLLM requests, keyed by llama-swap id.
#[derive(Debug, Default)]
pub struct VllmPhaseBook {
    models: HashMap<String, Track>,
}

impl VllmPhaseBook {
    /// Fold one `/metrics` read of `model` in.
    pub fn observe(&mut self, model: &str, sample: &MetricsSample, at: Instant) {
        if !self.models.contains_key(model) && self.models.len() >= MAX_MODELS {
            return;
        }
        let track = self.models.entry(model.to_owned()).or_default();
        track.observe(sample, at);
    }

    /// The phase of `model`'s running requests from a read no older than
    /// `fresh`. `None` when nothing is running, the read is stale or the
    /// counts do not add up. Unknown is never coerced to decode.
    #[must_use]
    pub fn phase(&self, model: &str, now: Instant, fresh: Duration) -> Option<ChartPhase> {
        let track = self.models.get(model)?;
        if now.saturating_duration_since(track.at?) > fresh {
            return None;
        }
        track.phase
    }

    /// `model`'s process is gone: the next read starts a new baseline.
    pub fn forget(&mut self, model: &str) {
        self.models.remove(model);
    }
}

impl Track {
    fn observe(&mut self, sample: &MetricsSample, at: Instant) {
        let (Some(ttft), Some(e2e), Some(running)) =
            (sample.ttft, sample.e2e, sample.requests_processing)
        else {
            *self = Self::default();
            return;
        };
        let (ttft, e2e) = (ttft.count, e2e.count);
        if !ttft.is_finite() || !e2e.is_finite() || !running.is_finite() {
            *self = Self::default();
            return;
        }
        // A counter that went down is a new process: its counts start at 0.
        let restarted = self.at.is_some()
            && (ttft < self.ttft
                || e2e < self.e2e
                || sample.preemptions.is_some_and(|n| n < self.preempt_base));
        if restarted {
            self.offset = 0.0;
            self.generated = None;
            self.generated_at = None;
            self.preempt_base = 0;
        }
        let advanced = match (self.generated, sample.n_decode_total) {
            (Some(last), Some(now)) => now.checked_sub(last),
            _ => None,
        };
        match (self.generated, sample.n_decode_total) {
            (Some(last), Some(now)) if now == last => {}
            (_, Some(_)) => self.generated_at = Some(at),
            (_, None) => self.generated_at = None,
        }
        self.generated = sample.n_decode_total;
        self.at = Some(at);
        self.ttft = ttft;
        self.e2e = e2e;
        let queued = sample.queued.unwrap_or(0.0);
        if running <= 0.0 {
            // Nothing past its first token is in flight now, unless it was
            // preempted back to the queue.
            if queued <= 0.0 {
                self.offset = ttft - e2e;
                self.preempt_base = sample.preemptions.unwrap_or(0);
            }
            self.phase = None;
            return;
        }
        let decoding = (ttft - e2e - self.offset).round();
        let running = running.round();
        // A preempted request leaves the TTFT/e2e difference unreliable
        // until the next idle read. With nothing counted as decoding, more
        // tokens out than requests running means an aborted request moved
        // the difference: unknown, not prefill (DER-451).
        let preempted = sample.preemptions.is_some_and(|n| n > self.preempt_base);
        let drifted = decoding == 0.0 && advanced.is_some_and(|n| n as f64 > running);
        self.phase = if preempted || drifted || decoding < 0.0 || decoding > running {
            None
        } else if decoding < running {
            Some(ChartPhase::Prefill)
        } else if self
            .generated_at
            .is_some_and(|moved| at.saturating_duration_since(moved) <= DECODE_STALL)
        {
            Some(ChartPhase::Decode)
        } else {
            None
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Hist;

    const FRESH: Duration = Duration::from_millis(500);

    fn preempted(mut sample: MetricsSample, preemptions: u64) -> MetricsSample {
        sample.preemptions = Some(preemptions);
        sample
    }

    fn sample(running: f64, queued: f64, ttft: f64, e2e: f64, generated: u64) -> MetricsSample {
        MetricsSample {
            requests_processing: Some(running),
            queued: Some(queued),
            ttft: Some(Hist {
                sum: ttft,
                count: ttft,
            }),
            e2e: Some(Hist {
                sum: e2e,
                count: e2e,
            }),
            n_decode_total: Some(generated),
            ..MetricsSample::default()
        }
    }

    /// Reads `steps` (running, queued, ttft, e2e, generated), 250 ms apart,
    /// and the phase after each.
    fn run(
        book: &mut VllmPhaseBook,
        start: Instant,
        steps: &[(f64, f64, f64, f64, u64)],
    ) -> Vec<Option<ChartPhase>> {
        steps
            .iter()
            .enumerate()
            .map(|(index, (running, queued, ttft, e2e, generated))| {
                let at = start + Duration::from_millis(250 * index as u64);
                book.observe("m", &sample(*running, *queued, *ttft, *e2e, *generated), at);
                book.phase("m", at, FRESH)
            })
            .collect()
    }

    use ChartPhase::{Decode, Prefill};

    #[test]
    fn one_request_prefills_then_decodes_then_completes() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                (0.0, 0.0, 4.0, 4.0, 100),
                (1.0, 0.0, 4.0, 4.0, 100),
                (1.0, 0.0, 4.0, 4.0, 100),
                (1.0, 0.0, 5.0, 4.0, 101),
                (1.0, 0.0, 5.0, 4.0, 120),
                (0.0, 0.0, 5.0, 5.0, 130),
            ],
        );
        assert_eq!(
            phases,
            [
                None,
                Some(Prefill),
                Some(Prefill),
                Some(Decode),
                Some(Decode),
                None
            ]
        );
    }

    #[test]
    fn a_prefill_beside_a_decode_is_a_prefill() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                (0.0, 0.0, 0.0, 0.0, 0),
                (1.0, 0.0, 1.0, 0.0, 10),
                (2.0, 0.0, 1.0, 0.0, 20),
                (2.0, 0.0, 2.0, 0.0, 30),
                (1.0, 0.0, 2.0, 1.0, 40),
            ],
        );
        assert_eq!(
            phases,
            [
                None,
                Some(Decode),
                Some(Prefill),
                Some(Decode),
                Some(Decode)
            ]
        );
    }

    #[test]
    fn a_queued_request_is_not_prefilling() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 1.0, 1.0, 0.0, 20),
                (1.0, 2.0, 1.0, 0.0, 30),
            ],
        );
        assert_eq!(phases, [Some(Decode); 3]);
        // Only queued, nothing running: the model is not busy, no phase.
        let mut book = VllmPhaseBook::default();
        assert_eq!(
            run(&mut book, Instant::now(), &[(0.0, 3.0, 0.0, 0.0, 0)]),
            [None]
        );
    }

    /// An idle read with a request waiting does not correct drift: the
    /// waiting request may be a preempted one past its first token. The
    /// correction waits for a read with nothing running or waiting.
    #[test]
    fn a_waiting_request_defers_the_idle_correction() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                // An abort recorded an e2e without a first token: −1.
                (0.0, 1.0, 3.0, 4.0, 50),
                (1.0, 0.0, 3.0, 4.0, 50),
                (0.0, 0.0, 3.0, 4.0, 50),
                (1.0, 0.0, 3.0, 4.0, 50),
            ],
        );
        assert_eq!(
            phases,
            [None, None, None, Some(Prefill)],
            "not corrected while one waited, corrected once nothing did"
        );
    }

    /// A preempted request keeps its first-token count while it waits and
    /// recomputes. Beside a real prefill the counts would say all-decoding.
    #[test]
    fn a_preemption_is_unknown_until_the_next_idle_read() {
        let mut book = VllmPhaseBook::default();
        let start = Instant::now();
        let reads = [
            (sample(0.0, 0.0, 0.0, 0.0, 0), 0),
            (sample(1.0, 0.0, 1.0, 0.0, 10), 0),
            // Preempted: back to the queue, then running again beside a new
            // request that is prefilling. TTFT − e2e still says 1 decoding.
            (sample(1.0, 1.0, 2.0, 0.0, 20), 1),
            (sample(2.0, 0.0, 2.0, 0.0, 21), 1),
            (sample(1.0, 0.0, 2.0, 1.0, 40), 1),
            (sample(0.0, 0.0, 2.0, 2.0, 50), 1),
            (sample(1.0, 0.0, 2.0, 2.0, 50), 1),
        ];
        let phases: Vec<_> = reads
            .into_iter()
            .enumerate()
            .map(|(index, (read, preemptions))| {
                let at = start + Duration::from_millis(250 * index as u64);
                book.observe("m", &preempted(read, preemptions), at);
                book.phase("m", at, FRESH)
            })
            .collect();
        assert_eq!(phases[1], Some(Decode), "no preemption yet");
        assert_eq!(phases[2..5], [None, None, None], "preempted since idle");
        assert_eq!(phases[5..], [None, Some(Prefill)], "idle read: new base");
        // A process that already preempted before this run: unknown until
        // its first idle read.
        let mut book = VllmPhaseBook::default();
        book.observe("m", &preempted(sample(1.0, 0.0, 1.0, 0.0, 10), 4), start);
        assert_eq!(book.phase("m", start, FRESH), None);
    }

    /// −1 drift under continuous load: one request decodes, the counts say
    /// nothing does. Tokens came out, so it is not a prefill.
    #[test]
    fn a_prefill_with_tokens_coming_out_is_unknown() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                (0.0, 0.0, 5.0, 5.0, 100),
                (1.0, 0.0, 6.0, 5.0, 101),
                // An abort's e2e with no TTFT, while busy: decoding reads 0.
                (1.0, 0.0, 6.0, 6.0, 115),
                (1.0, 0.0, 6.0, 6.0, 130),
                // Two running, the counts say none decode, one token came
                // out: not more than running, so still a prefill.
                (2.0, 0.0, 6.0, 6.0, 131),
            ],
        );
        assert_eq!(phases, [None, Some(Decode), None, None, Some(Prefill)]);
    }

    #[test]
    fn a_counter_reset_starts_from_zero() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                // Drift: an abort left one TTFT without an e2e.
                (0.0, 0.0, 9.0, 8.0, 500),
                (1.0, 0.0, 9.0, 8.0, 500),
                // Reload: both counts start again.
                (1.0, 0.0, 0.0, 0.0, 0),
                (1.0, 0.0, 1.0, 0.0, 1),
            ],
        );
        assert_eq!(phases, [None, Some(Prefill), Some(Prefill), Some(Decode)]);
    }

    #[test]
    fn drift_is_unknown_until_an_idle_read_corrects_it() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                // An earlier abort recorded an e2e with no TTFT: −1.
                (1.0, 0.0, 3.0, 4.0, 50),
                (1.0, 0.0, 3.0, 4.0, 50),
                (0.0, 0.0, 3.0, 4.0, 50),
                (1.0, 0.0, 3.0, 4.0, 50),
                (1.0, 0.0, 4.0, 4.0, 51),
            ],
        );
        assert_eq!(phases, [None, None, None, Some(Prefill), Some(Decode)]);
    }

    #[test]
    fn all_decoding_without_new_tokens_is_unknown() {
        let mut book = VllmPhaseBook::default();
        let phases = run(
            &mut book,
            Instant::now(),
            &[
                // An abort left a TTFT with no e2e while busy; the new
                // request is prefilling but the counts call it decoding.
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 0.0, 1.0, 0.0, 10),
                (1.0, 0.0, 1.0, 0.0, 10),
            ],
        );
        assert_eq!(phases[..5], [Some(Decode); 5]);
        assert_eq!(phases[5], None, "no token for 1.25 s");
    }

    #[test]
    fn a_stale_or_missing_read_is_unknown() {
        let mut book = VllmPhaseBook::default();
        let start = Instant::now();
        run(
            &mut book,
            start,
            &[(0.0, 0.0, 0.0, 0.0, 0), (1.0, 0.0, 0.0, 0.0, 0)],
        );
        let at = start + Duration::from_millis(250);
        assert_eq!(book.phase("m", at, FRESH), Some(Prefill));
        assert_eq!(book.phase("m", at + Duration::from_secs(1), FRESH), None);
        assert_eq!(book.phase("other", at, FRESH), None);
        let mut missing = sample(1.0, 0.0, 0.0, 0.0, 0);
        missing.e2e = None;
        book.observe("m", &missing, at);
        assert_eq!(book.phase("m", at, FRESH), None);
        book.forget("m");
        assert_eq!(book.phase("m", at, FRESH), None);
    }
}
