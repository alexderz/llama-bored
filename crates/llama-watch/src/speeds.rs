//! Engine-measured prefill and decode speeds (#35).
//!
//! llama-swap's activity rows carry prompt and generation tok/s only from
//! llama.cpp's response timings; for vLLM they are `-1`. vLLM 0.30 observes
//! four histograms for every finished request, in the same step
//! (`v1/metrics/loggers.py`, `record_finished_request`):
//!
//! - `vllm:request_prefill_time_seconds`: scheduled to first token;
//! - `vllm:request_prefill_kv_computed_tokens`: prompt minus cached tokens;
//! - `vllm:request_decode_time_seconds`: first token to last token;
//! - `vllm:request_generation_tokens`: tokens generated.
//!
//! Between two reads, the requests that finished give
//! prefill tok/s = Δ(prompt − cached) ÷ Δprefill time and
//! decode tok/s = (Δgenerated − Δrequests) ÷ Δdecode time: the first token
//! of each request is the prefill's, the rest are decoded in its decode
//! time, speculative decoding included. The running counters
//! (`prompt_tokens_total`, `generation_tokens_total`) move while a request
//! is in flight and the times only when it ends, so their quotient over a
//! short window is not a speed; the per-request sums are used instead.
//! SGLang has no per-request prefill or decode time, so it gets none.
//!
//! [`window`] is that maths. [`SpeedBook`] gives a new activity row the
//! speeds of the requests that finished between the metrics read before
//! the activity read that last lacked the row and the first metrics read
//! after the one that showed it; with several requests there, it is their
//! average. No speed is ever split out of a row's duration.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};

use llama_core::backend::MAX_ENGINE_TPS;

use crate::metrics::Hist;

/// One phase's per-request histograms: its time and its tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Phase {
    /// Seconds spent in the phase, summed over finished requests.
    pub seconds: Hist,
    /// Tokens of the phase, summed over the same requests.
    pub tokens: Hist,
}

/// The totals one `/metrics` read gives for both phases. A phase whose
/// histograms are missing is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedTotals {
    /// `request_prefill_time_seconds` and `request_prefill_kv_computed_tokens`.
    pub prefill: Option<Phase>,
    /// `request_decode_time_seconds` and `request_generation_tokens`.
    pub decode: Option<Phase>,
}

impl SpeedTotals {
    /// `None` when neither phase is reported.
    #[must_use]
    pub fn reported(self) -> Option<Self> {
        (self.prefill.is_some() || self.decode.is_some()).then_some(self)
    }
}

/// Tokens per second of each phase. `None` when the window had no finished
/// request, no time or no tokens, or the value is out of range.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speeds {
    /// Prefill tokens per second.
    pub prefill: Option<f64>,
    /// Decode tokens per second.
    pub decode: Option<f64>,
}

impl Speeds {
    /// True when neither speed is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prefill.is_none() && self.decode.is_none()
    }
}

/// Speeds of the requests that finished between `before` and `now`.
///
/// `before = None` (no earlier read) uses `now`'s totals, every request
/// since the server started. Any count or sum that went down means the
/// server restarted: `now`'s totals are the window, as in
/// [`crate::metrics::EngineBook`]. A phase needs at least one finished
/// request, a positive time and positive tokens.
#[must_use]
pub fn window(before: Option<&SpeedTotals>, now: &SpeedTotals) -> Speeds {
    Speeds {
        prefill: now
            .prefill
            .and_then(|phase| rate(before.and_then(|b| b.prefill), phase, false)),
        decode: now
            .decode
            .and_then(|phase| rate(before.and_then(|b| b.decode), phase, true)),
    }
}

fn rate(before: Option<Phase>, now: Phase, skip_first: bool) -> Option<f64> {
    let base = before
        .filter(|b| {
            now.seconds.count >= b.seconds.count
                && now.seconds.sum >= b.seconds.sum
                && now.tokens.count >= b.tokens.count
                && now.tokens.sum >= b.tokens.sum
        })
        .unwrap_or_default();
    let requests = now.seconds.count - base.seconds.count;
    let seconds = now.seconds.sum - base.seconds.sum;
    let mut tokens = now.tokens.sum - base.tokens.sum;
    if skip_first {
        tokens -= now.tokens.count - base.tokens.count;
    }
    if !(requests >= 1.0 && seconds > 0.0 && tokens > 0.0) {
        return None;
    }
    let tps = tokens / seconds;
    (tps.is_finite() && tps <= MAX_ENGINE_TPS).then_some(tps)
}

/// Most rows waiting for a metrics read.
pub const MAX_PENDING: usize = 64;
/// Most rows whose speeds are kept. RECENT shows at most 32.
pub const MAX_ROWS: usize = 256;
/// Most models with reads.
pub const MAX_MODELS: usize = 64;
/// A row with no metrics read after it within this long gets no speeds.
pub const PENDING_FOR: Duration = Duration::from_secs(10);

#[derive(Debug)]
struct Pending {
    /// The row's [`crate::activity::ActivityRow::seq`].
    row: u64,
    model: String,
    /// The activity read before the one that showed the row.
    after: Instant,
    /// The activity read that showed the row.
    seen: Instant,
}

/// Engine speeds attributed to activity rows, keyed by llama-watch's row
/// number ([`crate::activity::ActivityRow::seq`], #44), never llama-swap's
/// id: a restarted llama-swap reuses ids, and an old row keeps its speeds.
///
/// Feed it every metrics read of a model without `/slots`
/// ([`Self::observe`]) and every activity read with the rows it showed
/// for the first time ([`Self::activity`]). A row resolves at the first
/// metrics read at or after the activity read that showed it: its speeds
/// are [`window`] between the last metrics read at or before the previous
/// activity read and that one. No earlier read, no finished request in
/// that span, or no read within [`PENDING_FOR`] leaves the row without.
#[derive(Debug, Default)]
pub struct SpeedBook {
    reads: HashMap<String, VecDeque<(Instant, SpeedTotals)>>,
    activity_at: Option<Instant>,
    pending: Vec<Pending>,
    rows: BTreeMap<u64, Speeds>,
}

impl SpeedBook {
    /// One metrics read of `model` at `at`. `None` (the server reports no
    /// speed histograms) drops its reads.
    ///
    /// Only the reads a row can still need are kept: the newest one at or
    /// before each activity read a row will be measured from, and the
    /// newest one, so at most [`MAX_PENDING`] + 2.
    pub fn observe(&mut self, model: &str, at: Instant, totals: Option<SpeedTotals>) {
        let Some(totals) = totals else {
            self.reads.remove(model);
            return;
        };
        if !self.reads.contains_key(model) && self.reads.len() >= MAX_MODELS {
            return;
        }
        self.reads
            .entry(model.to_owned())
            .or_default()
            .push_back((at, totals));
        self.resolve(at);
        let anchors: Vec<Instant> = self
            .pending
            .iter()
            .filter(|p| p.model == model)
            .map(|p| p.after)
            .chain(self.activity_at)
            .collect();
        if let Some(reads) = self.reads.get_mut(model) {
            let times: Vec<Instant> = reads.iter().map(|(t, _)| *t).collect();
            let mut index = 0;
            reads.retain(|_| {
                let i = index;
                index += 1;
                i + 1 == times.len()
                    || anchors
                        .iter()
                        .any(|a| times[i] <= *a && times.get(i + 1).is_none_or(|next| next > a))
            });
        }
    }

    /// An activity read at `at`, with the `(row number, model id)` of each
    /// row it showed for the first time. The first call only sets the
    /// baseline.
    pub fn activity(&mut self, at: Instant, new_rows: &[(u64, String)]) {
        if let Some(after) = self.activity_at {
            for (row, model) in new_rows {
                if self.pending.len() >= MAX_PENDING {
                    break;
                }
                if self.reads.contains_key(model) {
                    self.pending.push(Pending {
                        row: *row,
                        model: model.clone(),
                        after,
                        seen: at,
                    });
                }
            }
        }
        self.activity_at = Some(at);
    }

    /// Resolve every row that has a metrics read at or after its activity
    /// read; drop the ones waiting longer than [`PENDING_FOR`].
    pub fn resolve(&mut self, now: Instant) {
        let pending = std::mem::take(&mut self.pending);
        for wait in pending {
            let Some(reads) = self.reads.get(&wait.model) else {
                continue;
            };
            let Some((_, end)) = reads.iter().find(|(t, _)| *t >= wait.seen) else {
                if now.saturating_duration_since(wait.seen) <= PENDING_FOR {
                    self.pending.push(wait);
                }
                continue;
            };
            let Some((_, base)) = reads.iter().rev().find(|(t, _)| *t <= wait.after) else {
                continue;
            };
            let speeds = window(Some(base), end);
            if !speeds.is_empty() {
                self.rows.insert(wait.row, speeds);
                while self.rows.len() > MAX_ROWS {
                    self.rows.pop_first();
                }
            }
        }
    }

    /// The speeds attributed to the activity row numbered `row`.
    #[must_use]
    pub fn speeds(&self, row: u64) -> Option<Speeds> {
        self.rows.get(&row).copied()
    }

    /// `model` was unloaded: its reads and waiting rows go; rows already
    /// resolved keep their speeds.
    pub fn forget(&mut self, model: &str) {
        self.reads.remove(model);
        self.pending.retain(|p| p.model != model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn totals(prefill: (f64, f64, f64), decode: (f64, f64, f64)) -> SpeedTotals {
        // (requests, seconds, tokens) per phase.
        let phase = |(count, seconds, tokens): (f64, f64, f64)| Phase {
            seconds: Hist {
                sum: seconds,
                count,
            },
            tokens: Hist { sum: tokens, count },
        };
        SpeedTotals {
            prefill: Some(phase(prefill)),
            decode: Some(phase(decode)),
        }
    }

    fn close(got: Option<f64>, want: f64) {
        let got = got.expect("a speed");
        assert!((got - want).abs() < 1e-9, "{got} != {want}");
    }

    #[test]
    fn window_maths_over_one_and_several_requests() {
        let before = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        // One request: 8,000 uncached prompt tokens in 2 s; 401 tokens, the
        // first from prefill, 400 decoded in 8 s.
        let one = totals((11.0, 7.0, 18_000.0), (11.0, 108.0, 4_411.0));
        let speeds = window(Some(&before), &one);
        close(speeds.prefill, 4_000.0);
        close(speeds.decode, 50.0);
        // Two more: their average, not the mean of their speeds.
        let two = totals((13.0, 8.0, 19_000.0), (13.0, 118.0, 4_613.0));
        let speeds = window(Some(&one), &two);
        close(speeds.prefill, 1_000.0);
        close(speeds.decode, 20.0);
        // No earlier read: every request since the server started.
        let speeds = window(None, &before);
        close(speeds.prefill, 2_000.0);
        close(speeds.decode, 40.0);
    }

    #[test]
    fn window_guards_zero_and_negative_deltas() {
        let read = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        assert!(window(Some(&read), &read).is_empty(), "nothing finished");
        // A request finished with no prefill time, or no tokens past the first.
        let zero = totals((11.0, 5.0, 10_500.0), (11.0, 101.0, 4_011.0));
        assert!(window(Some(&read), &zero).is_empty());
        // Time moved but no request finished (cannot happen; still no speed).
        let odd = totals((10.0, 6.0, 10_100.0), (10.0, 101.0, 4_020.0));
        assert!(window(Some(&read), &odd).is_empty());
        // Out of range is no speed.
        let fast = totals((11.0, 5.000_001, 20_000.0), (10.0, 100.0, 4_010.0));
        assert_eq!(window(Some(&read), &fast).prefill, None);
        // A missing phase stays missing.
        let half = SpeedTotals {
            prefill: None,
            ..totals((0.0, 0.0, 0.0), (11.0, 108.0, 4_411.0))
        };
        let speeds = window(Some(&read), &half);
        assert_eq!(speeds.prefill, None);
        close(speeds.decode, 50.0);
        assert_eq!(SpeedTotals::default().reported(), None);
        assert!(half.reported().is_some());
    }

    #[test]
    fn window_after_a_restart_is_the_new_totals() {
        let before = totals((100.0, 50.0, 100_000.0), (100.0, 1000.0, 40_100.0));
        // The server restarted and served one request since.
        let after = totals((1.0, 0.5, 1_000.0), (1.0, 2.0, 61.0));
        let speeds = window(Some(&before), &after);
        close(speeds.prefill, 2_000.0);
        close(speeds.decode, 30.0);
        // One series alone going back is a restart too.
        let odd = totals((101.0, 49.0, 101_000.0), (100.0, 1000.0, 40_100.0));
        let speeds = window(Some(&before), &odd);
        close(speeds.prefill, 101_000.0 / 49.0);
        assert_eq!(speeds.decode, None, "decode did not move");
    }

    struct Clock(Instant);
    impl Clock {
        fn at(&self, ms: u64) -> Instant {
            self.0 + Duration::from_millis(ms)
        }
    }

    #[test]
    fn a_row_gets_the_window_its_request_finished_in() {
        let clock = Clock(Instant::now());
        let mut book = SpeedBook::default();
        let idle = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        let done = totals((11.0, 7.0, 18_000.0), (11.0, 108.0, 4_411.0));
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[]);
        book.observe("m", clock.at(250), Some(idle));
        // The request finishes; the next metrics read sees it ...
        book.observe("m", clock.at(500), Some(done));
        // ... and the activity read after shows its row.
        book.activity(clock.at(2_100), &[(7, "m".to_owned())]);
        assert_eq!(book.speeds(7), None, "no read after the row yet");
        book.observe("m", clock.at(2_250), Some(done));
        let speeds = book.speeds(7).expect("attributed");
        close(speeds.prefill, 4_000.0);
        close(speeds.decode, 50.0);
        assert_eq!(book.speeds(8), None);
    }

    #[test]
    fn a_metrics_read_after_the_row_resolves_it() {
        let clock = Clock(Instant::now());
        let mut book = SpeedBook::default();
        let idle = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        let done = totals((11.0, 7.0, 18_000.0), (11.0, 108.0, 4_411.0));
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[]);
        // The row shows up before any metrics read has seen the request.
        book.activity(clock.at(2_100), &[(7, "m".to_owned())]);
        book.observe("m", clock.at(2_250), Some(done));
        close(book.speeds(7).expect("attributed").decode, 50.0);
    }

    #[test]
    fn several_rows_in_one_span_share_its_average() {
        let clock = Clock(Instant::now());
        let mut book = SpeedBook::default();
        let idle = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        // Two requests: 3,000 tokens in 1 s and 1,000 in 1 s; 100 and 300
        // tokens decoded in 5 s each.
        let both = totals((12.0, 7.0, 14_000.0), (12.0, 110.0, 4_412.0));
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[]);
        book.observe("m", clock.at(1_000), Some(both));
        book.activity(clock.at(2_100), &[(8, "m".to_owned()), (9, "m".to_owned())]);
        book.observe("m", clock.at(2_250), Some(both));
        for row in [8, 9] {
            let speeds = book.speeds(row).expect("attributed");
            close(speeds.prefill, 2_000.0);
            close(speeds.decode, 40.0);
        }
    }

    #[test]
    fn no_match_leaves_the_row_without_speeds() {
        let clock = Clock(Instant::now());
        let idle = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        // Nothing finished in the span (the request was not this engine's).
        let mut book = SpeedBook::default();
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[]);
        book.activity(clock.at(2_100), &[(1, "m".to_owned())]);
        book.observe("m", clock.at(2_250), Some(idle));
        assert_eq!(book.speeds(1), None);
        // No metrics read before the span: no base.
        let mut book = SpeedBook::default();
        book.activity(clock.at(0), &[]);
        book.observe("m", clock.at(1_000), Some(idle));
        book.activity(clock.at(2_000), &[(2, "m".to_owned())]);
        book.observe("m", clock.at(2_250), Some(idle));
        assert_eq!(book.speeds(2), None);
        // A model with no speed histograms gets nothing, and no wait.
        let mut book = SpeedBook::default();
        book.observe("sg", clock.at(0), None);
        book.activity(clock.at(100), &[]);
        book.activity(clock.at(2_100), &[(3, "sg".to_owned())]);
        assert!(book.pending.is_empty());
        // The first activity read is a baseline: its rows are history.
        let mut book = SpeedBook::default();
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[(4, "m".to_owned())]);
        assert!(book.pending.is_empty());
        // No metrics read within PENDING_FOR: given up.
        let mut book = SpeedBook::default();
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(100), &[]);
        book.activity(clock.at(2_100), &[(5, "m".to_owned())]);
        book.resolve(clock.at(2_100) + PENDING_FOR + Duration::from_millis(1));
        assert!(book.pending.is_empty());
    }

    #[test]
    fn reads_rows_and_models_are_bounded() {
        let clock = Clock(Instant::now());
        let idle = totals((10.0, 5.0, 10_000.0), (10.0, 100.0, 4_010.0));
        let mut book = SpeedBook::default();
        book.activity(clock.at(0), &[]);
        for ms in 1..1_000 {
            book.observe("m", clock.at(ms), Some(idle));
        }
        assert_eq!(
            book.reads["m"].len(),
            1,
            "no read is a base yet: the newest only"
        );
        book.activity(clock.at(1_500), &[]);
        for ms in 2_000..3_000 {
            book.observe("m", clock.at(ms), Some(idle));
        }
        assert_eq!(
            book.reads["m"].len(),
            2,
            "the base for the next rows and the newest"
        );
        for i in 0..MAX_MODELS + 3 {
            book.observe(&format!("m{i}"), clock.at(0), Some(idle));
        }
        assert_eq!(book.reads.len(), MAX_MODELS);
        // Rows: the newest MAX_ROWS ids stay.
        let mut book = SpeedBook::default();
        let mut cum = idle;
        book.observe("m", clock.at(0), Some(cum));
        book.activity(clock.at(1), &[]);
        let mut t = 1;
        for row in 0..(MAX_ROWS as u64 + 10) {
            let p = cum.prefill.as_mut().unwrap();
            p.seconds.count += 1.0;
            p.seconds.sum += 1.0;
            p.tokens.count += 1.0;
            p.tokens.sum += 100.0;
            book.observe("m", clock.at(t + 1), Some(cum));
            book.activity(clock.at(t + 2), &[(row, "m".to_owned())]);
            book.observe("m", clock.at(t + 3), Some(cum));
            t += 3;
        }
        assert_eq!(book.rows.len(), MAX_ROWS);
        assert_eq!(book.speeds(0), None);
        close(book.speeds(MAX_ROWS as u64 + 9).unwrap().prefill, 100.0);
        // Pending is capped.
        let mut book = SpeedBook::default();
        book.observe("m", clock.at(0), Some(idle));
        book.activity(clock.at(1), &[]);
        let many: Vec<(u64, String)> = (0..100).map(|i| (i, "m".to_owned())).collect();
        book.activity(clock.at(2), &many);
        assert_eq!(book.pending.len(), MAX_PENDING);
        // Unload drops reads and waits.
        book.forget("m");
        assert!(book.pending.is_empty() && !book.reads.contains_key("m"));
    }
}
