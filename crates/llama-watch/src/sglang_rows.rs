//! One SGLang request's cached tokens and speeds, from the engine's counters (#133).
//!
//! SGLang has no per-request prefill or decode histogram. Its counters step
//! when a request finishes (`num_requests_total`, `prompt_tokens_total`,
//! `generation_tokens_total`, `cached_tokens_total`, the end-to-end latency
//! histogram), and time-to-first-token steps when the first token is
//! produced, which can be several scrapes earlier. `num_running_reqs` is
//! how many requests the engine is running.
//!
//! A fill needs one streaming request alone in the engine. The window from
//! one finish to the next is clean when exactly one request finished,
//! exactly one first token was observed, the first-token count equals the
//! finished count at both ends (nothing is already past its first token),
//! nothing else ran, no counter went backwards, and no scrape was missed
//! for longer than [`crate::speeds::PENDING_FOR`]. The row's prompt, output
//! and duration have to match that window (no more than 0.1 s shorter than
//! the engine's end-to-end time, and no more than the larger of 0.5 s and
//! two percent longer), and its timestamp has to fall in the finish's
//! scrape interval. Those checks only ever refuse a fill.
//!
//! A row is filled only when its request was the only one the engine
//! finished within a few seconds, and it is filled a few seconds after it
//! ends. The scrapes have to cover that span: a gap, a clock step, a reset,
//! a restart or the watcher's own start inside it leaves the row empty,
//! because whatever happened there was not seen. llama-swap and llama-watch
//! run on one host, so the two clocks agree. A row for which the engine has
//! no request, matching another request alone in that window in lengths and
//! duration, cannot be told apart.
//!
//! A non-streaming finish (`is_streaming="false"` on `num_requests_total`)
//! fills the cached count only: its time to first token is the whole
//! request, so neither rate is a measurement. Without an `is_streaming`
//! label on the first-token series, only the cached count is filled. The
//! prompt rate also stays unset unless that request's own queue time is
//! known — the k-th finish after an idle scrape owns the k-th
//! queue-histogram step after it, and the step was a single +1 — and that
//! wait is at most five percent of its time to the first token. The prompt
//! rate is shown only while the engine's queue count and request count
//! agree at an idle point, so on an engine whose queue count runs ahead of
//! its request count it stays unset. The cached count and the generation
//! rate stay.
//! A value llama-swap already reported is never replaced. A missing cached
//! count is still filled.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use crate::localtime::parse_rfc3339;

use llama_core::backend::MAX_ENGINE_TPS;

use crate::metrics::MetricsSample;
use crate::speeds::{MAX_MODELS, MAX_PENDING, MAX_ROWS, PENDING_FOR};

/// Scrapes kept per model. Past this the open window is dropped.
const HISTORY_CAP: usize = 512;
/// Untaken finishes older than this cannot meet a row. Finish scrapes
/// that can block a fill are kept for the same time.
const FINISH_FOR: Duration = Duration::from_secs(30);
/// Finishes, and the finish scrapes that can block one, kept per model.
const KEPT_FINISHES: usize = 64;
/// Queue steps remembered since the idle anchor. Past this the order is
/// no longer trusted.
const QUEUE_CAP: usize = 64;
/// How far before a row's second a finish still shares its isolation
/// window. Scrape latency and llama-swap's delay in writing the row both
/// stay under this.
const ISOLATION_BEFORE_S: f64 = 5.0;
/// How far after a row's second the isolation window extends. The row is
/// left empty until a scrape is stamped later than this.
const ISOLATION_AFTER_S: f64 = 6.0;
/// A hole in coverage. Longer than this between scrapes, and whatever
/// finished in the gap was not seen.
const COVERAGE_GAP: Duration = Duration::from_secs(3);
/// Wall time and monotonic time disagree by more than this. The host clock
/// stepped; the stretch is a hole and the queue chain is dropped.
const CLOCK_STEP: Duration = Duration::from_millis(500);

/// The counters one SGLang `/metrics` scrape contributes. `None` from
/// [`Totals::from_sample`] when any of them is missing: that scrape is a
/// hole and the open window is not clean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Totals {
    /// `sglang:num_requests_total`, both streaming series summed.
    pub requests: u64,
    /// `sglang:num_requests_total` with `is_streaming="false"` only.
    /// Absent on the sample counts as zero.
    pub batch: u64,
    /// `sglang:prompt_tokens_total`.
    pub prompt: u64,
    /// `sglang:cached_tokens_total`, every `cache_source` summed.
    pub cached: u64,
    /// `sglang:generation_tokens_total`.
    pub generated: u64,
    /// `sglang:e2e_request_latency_seconds_sum`.
    pub e2e_sum: f64,
    /// `sglang:e2e_request_latency_seconds_count`.
    pub e2e_count: f64,
    /// `sglang:time_to_first_token_seconds_sum`.
    pub ttft_sum: f64,
    /// `sglang:time_to_first_token_seconds_count`.
    pub ttft_count: f64,
    /// `sglang:queue_time_seconds_sum`. Absent on the sample counts as zero.
    pub queue_sum: f64,
    /// `sglang:queue_time_seconds_count`.
    pub queue_count: f64,
    /// `sglang:num_running_reqs`.
    pub running: f64,
}

impl Totals {
    /// The counters a scrape must have before it can attribute a row.
    ///
    /// An engine without a `cached_tokens_total` series fills nothing:
    /// this returns `None`, and the scrape is a hole.
    #[must_use]
    pub fn from_sample(sample: &MetricsSample) -> Option<Self> {
        let e2e = sample.e2e?;
        let ttft = sample.ttft?;
        let running = sample.requests_processing?;
        let (queue_sum, queue_count) = match sample.queue_time {
            Some(hist) => (hist.sum, hist.count),
            None => (0.0, 0.0),
        };
        if ![
            e2e.sum,
            e2e.count,
            ttft.sum,
            ttft.count,
            running,
            queue_sum,
            queue_count,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return None;
        }
        Some(Self {
            requests: sample.requests_total?,
            batch: sample.requests_batch.unwrap_or(0),
            prompt: sample.prompt_total?,
            cached: sample.cached_total?,
            generated: sample.n_decode_total?,
            e2e_sum: e2e.sum,
            e2e_count: e2e.count,
            ttft_sum: ttft.sum,
            ttft_count: ttft.count,
            queue_sum,
            queue_count,
            running,
        })
    }
}

/// What a clean window measured for one request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    /// Cached prompt tokens. A measured 0 is `0`, not unknown.
    pub cached: u64,
    /// `(prompt − cached) / TTFT`, when both parts are positive.
    pub prompt_tps: Option<f64>,
    /// `(generated − 1) / (e2e − TTFT)`, when at least two tokens were
    /// generated and decode took time.
    pub gen_tps: Option<f64>,
}

/// A new activity row that might belong to a window.
#[derive(Clone, Debug)]
pub struct Claim {
    /// [`crate::activity::ActivityRow::seq`].
    pub seq: u64,
    /// llama-swap model id, the same key [`Self::observe`] uses.
    pub model: String,
    /// The row's `input_tokens`. SGLang's count includes the cached part.
    pub input: u64,
    /// The row's `output_tokens`. `None` cannot match a finish.
    pub output: Option<u64>,
    /// The row's `duration_ms`. `None` cannot match a finish.
    pub duration_ms: Option<u64>,
    /// llama-swap reported a prompt rate. A reported rate is never replaced.
    pub prompt_known: bool,
    /// llama-swap reported a generation rate.
    pub gen_known: bool,
    /// The row's `timestamp`: the end time, to the second. Unparseable
    /// matches nothing.
    pub timestamp: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct Snap {
    at: Instant,
    /// Wall clock before the metrics request was sent. The engine read its
    /// counters somewhere between this and [`Self::wall`].
    before: SystemTime,
    /// Wall clock after the read and the parse.
    wall: SystemTime,
    requests: u64,
    batch: u64,
    prompt: u64,
    cached: u64,
    generated: u64,
    e2e_sum: f64,
    e2e_count: f64,
    ttft_sum: f64,
    ttft_count: f64,
    queue_sum: f64,
    queue_count: f64,
    running: f64,
}

impl Snap {
    fn new(at: Instant, before: SystemTime, wall: SystemTime, totals: Totals) -> Self {
        Self {
            at,
            before,
            wall,
            requests: totals.requests,
            batch: totals.batch,
            prompt: totals.prompt,
            cached: totals.cached,
            generated: totals.generated,
            e2e_sum: totals.e2e_sum,
            e2e_count: totals.e2e_count,
            ttft_sum: totals.ttft_sum,
            ttft_count: totals.ttft_count,
            queue_sum: totals.queue_sum,
            queue_count: totals.queue_count,
            running: totals.running,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Finish {
    at: Instant,
    /// Previous scrape's before-time, and this scrape's stamp. The finish
    /// happened in `(wall_lo, wall_hi]`.
    wall_lo: SystemTime,
    wall_hi: SystemTime,
    prompt: u64,
    /// Generation-token increase over the window.
    generated: u64,
    cached: u64,
    /// End-to-end seconds over the window.
    e2e_s: f64,
    prompt_tps: Option<f64>,
    gen_tps: Option<f64>,
    taken: bool,
}

/// A scrape where the finished count stepped. `n` is `None` after a gap,
/// a reset or a restart: the increase is not known, so the scrape is not
/// one finish.
#[derive(Clone, Copy, Debug)]
struct Sight {
    at: Instant,
    wall_lo: SystemTime,
    wall_hi: SystemTime,
    /// How many requests the finished count stepped by.
    n: Option<u64>,
}

#[derive(Debug, Default)]
struct Model {
    base: Option<Snap>,
    /// Scrapes since [`Self::base`] while a window is open.
    since: Vec<Snap>,
    /// A hole, a gap, an overlap or a counter that moved out of turn.
    dirty: bool,
    history: VecDeque<Snap>,
    finishes: VecDeque<Finish>,
    /// Every scrape where the finished count stepped, clean or not.
    /// Kept for as long as [`Self::finishes`], and capped the same way.
    sights: VecDeque<Sight>,
    /// Queue-histogram steps since [`Self::queue_anchor`], in order.
    /// `None` is a step that was not a single +1, so its seconds are unknown.
    queue_obs: VecDeque<Option<f64>>,
    /// How many of [`Self::queue_obs`] have been given to finishes.
    queue_used: usize,
    /// An idle scrape (nothing running) whose later steps can be ordered.
    queue_anchor: bool,
    /// The histogram has been non-zero since the last restart. While it
    /// has not, a missing step is not a reason to hide the prompt rate.
    queue_live: bool,
    /// One entry per scrape, including a failed read. `gap` means the
    /// stretch since the previous entry was not seen.
    probes: VecDeque<Probe>,
    /// The time-to-first-token series had no `is_streaming` label, so a
    /// streaming finish cannot be told from a batch finish. Rates stay off.
    ttft_unlabeled: bool,
}

/// A scrape's place on the two clocks. Coverage of an isolation window is
/// the run of these with no [`Probe::gap`] inside it.
#[derive(Debug)]
struct Probe {
    at: Instant,
    stamp: SystemTime,
    gap: bool,
}

/// What the queue chain can say about the finish on this scrape.
#[derive(Clone, Copy, Debug)]
enum QueueTie {
    /// No queue series has been seen. Do not hide the prompt rate.
    Absent,
    /// This finish's own queue time, in seconds.
    Known(f64),
    /// The series is live and this finish cannot be tied to one step.
    Unknown,
}

#[derive(Clone, Debug)]
struct Pending {
    seq: u64,
    model: String,
    input: u64,
    output: Option<u64>,
    duration_ms: Option<u64>,
    prompt_known: bool,
    gen_known: bool,
    timestamp: Option<String>,
    after: Instant,
    seen: Instant,
}

/// Fills for activity rows, keyed by [`crate::activity::ActivityRow::seq`].
#[derive(Debug, Default)]
pub struct SglangBook {
    models: HashMap<String, Model>,
    activity_at: Option<Instant>,
    pending: Vec<Pending>,
    rows: BTreeMap<u64, Fill>,
}

impl SglangBook {
    /// One metrics scrape of `model`, stamped once. `before` and the stamp
    /// are the same instant: the caller did not time the read. Prefer
    /// [`Self::observe_scrape`] when both clocks were taken.
    pub fn observe(&mut self, model: &str, at: Instant, wall: SystemTime, totals: Option<Totals>) {
        self.observe_scrape(model, at, wall, wall, totals, true);
    }

    /// One metrics scrape of `model`. `None` is a hole: the open window
    /// cannot be attributed. The first complete scrape is only a baseline.
    ///
    /// `before` is the wall clock before the request is sent. `wall` is the
    /// stamp after the parse. The engine read its counters somewhere between
    /// the two. Scrapes are ordered by `at`, not by either wall clock.
    ///
    /// `ttft_labeled` is false when the time-to-first-token series carries
    /// no `is_streaming` label. The cached count can still be filled; the
    /// rates cannot, because a streaming finish and a batch finish look the
    /// same.
    pub fn observe_scrape(
        &mut self,
        model: &str,
        at: Instant,
        before: SystemTime,
        wall: SystemTime,
        totals: Option<Totals>,
        ttft_labeled: bool,
    ) {
        if !self.models.contains_key(model) && self.models.len() >= MAX_MODELS {
            self.resolve(at);
            return;
        }
        let Some(totals) = totals else {
            if let Some(state) = self.models.get_mut(model) {
                state.dirty = true;
                // The next delta would cover the hole. Drop the chain; the
                // series stays live so a later finish does not pretend the
                // missed step was zero.
                state.queue_drop();
                state.ttft_unlabeled |= !ttft_labeled;
                state.push_probe(at, wall, true);
            }
            self.resolve(at);
            return;
        };
        let state = self.models.entry(model.to_owned()).or_default();
        state.ttft_unlabeled |= !ttft_labeled;
        state.note(Snap::new(at, before, wall, totals));
        self.resolve(at);
    }

    /// An activity read at `at`. The first call only sets the baseline,
    /// as [`crate::speeds::SpeedBook::activity`] does.
    pub fn activity(&mut self, at: Instant, rows: &[Claim]) {
        if let Some(after) = self.activity_at {
            for row in rows {
                if self.pending.len() >= MAX_PENDING {
                    break;
                }
                if self.models.contains_key(&row.model) {
                    self.pending.push(Pending {
                        seq: row.seq,
                        model: row.model.clone(),
                        input: row.input,
                        output: row.output,
                        duration_ms: row.duration_ms,
                        prompt_known: row.prompt_known,
                        gen_known: row.gen_known,
                        timestamp: row.timestamp.clone(),
                        after,
                        seen: at,
                    });
                }
            }
        }
        self.activity_at = Some(at);
        self.resolve(at);
    }

    /// The fill stored for activity row `seq`.
    #[must_use]
    pub fn fill(&self, seq: u64) -> Option<Fill> {
        self.rows.get(&seq).copied()
    }

    /// `model` was unloaded. Rows already filled keep their numbers.
    pub fn forget(&mut self, model: &str) {
        self.models.remove(model);
        self.pending.retain(|row| row.model != model);
    }

    /// Give every row that has a metrics scrape at or after it the finishes
    /// in its span, or drop it once [`PENDING_FOR`] has passed with none.
    fn resolve(&mut self, now: Instant) {
        let pending = std::mem::take(&mut self.pending);
        let mut waiting = Vec::new();
        let mut ready: Vec<Pending> = Vec::new();
        let mut span: Vec<(Instant, Instant)> = Vec::new();
        for wait in pending {
            let Some(model) = self.models.get(&wait.model) else {
                continue;
            };
            let end = model.history.iter().find(|snap| snap.at >= wait.seen);
            let base = model
                .history
                .iter()
                .rev()
                .find(|snap| snap.at <= wait.after);
            match (base, end) {
                (Some(base), Some(end)) => {
                    span.push((base.at, end.at));
                    ready.push(wait);
                }
                (None, _) => {}
                (Some(_), None) => {
                    if now.saturating_duration_since(wait.seen) <= PENDING_FOR {
                        waiting.push(wait);
                    }
                }
            }
        }
        let mut listed: Vec<Vec<usize>> = Vec::with_capacity(ready.len());
        let mut claims: BTreeMap<(String, usize), usize> = BTreeMap::new();
        for (wait, (base_at, end_at)) in ready.iter().zip(&span) {
            let Some(model) = self.models.get(&wait.model) else {
                listed.push(Vec::new());
                continue;
            };
            let hits: Vec<usize> = model
                .finishes
                .iter()
                .enumerate()
                .filter(|(_, finish)| {
                    // A finish already matched to some row still counts.
                    // Dropping it would make a different finish look like
                    // this row's only match.
                    finish.at > *base_at
                        && finish.at <= *end_at
                        && finish.prompt == wait.input
                        && wait.output == Some(finish.generated)
                        && wait
                            .duration_ms
                            .is_some_and(|ms| duration_agrees(ms, finish.e2e_s))
                        && stamp_fits(wait.timestamp.as_deref(), finish.wall_lo, finish.wall_hi)
                })
                .map(|(index, _)| index)
                .collect();
            for index in &hits {
                *claims.entry((wait.model.clone(), *index)).or_default() += 1;
            }
            listed.push(hits);
        }
        let mut stored: Vec<(u64, Fill)> = Vec::new();
        for ((wait, hits), _) in ready.iter().zip(&listed).zip(&span) {
            // A row is filled late: not until a scrape is stamped after
            // T + 6 s. Until then it stays empty and the finish stays free.
            if !hits.is_empty()
                && self
                    .models
                    .get(&wait.model)
                    .is_some_and(|model| !isolation_decided(model, wait.timestamp.as_deref()))
            {
                if now.saturating_duration_since(wait.seen) <= FINISH_FOR {
                    waiting.push(wait.clone());
                }
                continue;
            }
            let unique =
                hits.len() == 1 && claims.get(&(wait.model.clone(), hits[0])).copied() == Some(1);
            if unique
                && let Some(fill) = self.models.get(&wait.model).and_then(|model| {
                    let finish = model.finishes.get(hits[0]).filter(|finish| !finish.taken)?;
                    let finish_at = finish.at;
                    if !isolation_allows(model, finish_at, wait.timestamp.as_deref()) {
                        return None;
                    }
                    // A reported value is never stored over the row. A missing
                    // cached count is; so is a rate the row did not report.
                    // Another finish that could be this row already failed
                    // isolation: its scrape interval overlaps the row's
                    // window, so the length checks never get a fill to refuse.
                    Some(Fill {
                        cached: finish.cached,
                        prompt_tps: if wait.prompt_known {
                            None
                        } else {
                            finish.prompt_tps
                        },
                        gen_tps: if wait.gen_known { None } else { finish.gen_tps },
                    })
                })
            {
                stored.push((wait.seq, fill));
            }
            // Every finish this row fit is done. None of them is filled
            // unless it was the one stored above, and a later row must
            // still see them so it does not take a neighbour instead.
            if let Some(model) = self.models.get_mut(&wait.model) {
                for index in hits {
                    if let Some(finish) = model.finishes.get_mut(*index) {
                        finish.taken = true;
                    }
                }
            }
        }
        for (seq, fill) in stored {
            self.rows.insert(seq, fill);
            while self.rows.len() > MAX_ROWS {
                self.rows.pop_first();
            }
        }
        self.pending = waiting;
        for model in self.models.values_mut() {
            // A taken finish stays until it ages out. The next row's span
            // can still contain it, and it has to count as a fit there.
            model
                .finishes
                .retain(|finish| now.saturating_duration_since(finish.at) <= FINISH_FOR);
            model
                .sights
                .retain(|sight| now.saturating_duration_since(sight.at) <= FINISH_FOR);
            model
                .probes
                .retain(|probe| now.saturating_duration_since(probe.at) <= FINISH_FOR);
        }
    }
}

impl Model {
    fn note(&mut self, snap: Snap) {
        let Some(base) = self.base else {
            self.dirty = snap.running > 1.0;
            self.base = Some(snap);
            self.remember(snap);
            self.queue_baseline(&snap);
            // Nothing before the first scrape was seen. A row whose window
            // reaches back past this stamp is not filled.
            self.push_probe(snap.at, snap.wall, true);
            return;
        };
        let prev = self.since.last().copied().unwrap_or(base);
        if snap.at <= prev.at {
            return;
        }
        if went_back(&prev, &snap) {
            self.push_probe(snap.at, snap.wall, true);
            self.restart(&prev, snap);
            return;
        }
        let stepped = clock_stepped(&prev, &snap);
        let wide = coverage_gap(&prev, &snap);
        if stepped {
            // The wall clock moved relative to the monotonic clock. The
            // queue steps on either side are not one series.
            self.queue_drop();
        }
        self.push_probe(snap.at, snap.wall, stepped || wide);
        // Steps on this scrape are recorded before the finish consumes one,
        // so a request that left the queue as the previous one ended is the
        // next observation, not this finish's.
        let queue = self.queue_advance(&prev, &snap);
        let gap = snap.at.saturating_duration_since(prev.at) > PENDING_FOR;
        if self.idle(base, &snap) {
            // Nothing finished and nothing overlapped. A gap here hid no
            // request, so the next window starts clean.
            self.rebase(snap);
            return;
        }
        if gap || snap.running > 1.0 {
            self.dirty = true;
        }
        // Every step of the finished count, clean or not. A row is filled
        // from a clean window only when no other one of these could be it.
        self.record_sight(&prev, &snap, gap);
        if one_finish(&prev, &snap) {
            if self.window_clean(&base, &snap) {
                self.record(&base, &prev, &snap, queue);
            }
            self.close(snap);
            return;
        }
        if finish_event(&prev, &snap) {
            // Two finishes, or the request counter and the latency
            // histogram disagree. Close with nothing attributed.
            self.close(snap);
            return;
        }
        self.open(prev, snap);
    }

    fn idle(&self, base: Snap, snap: &Snap) -> bool {
        if !same_counters(&base, snap) || snap.running > 1.0 || base.running > 1.0 {
            return false;
        }
        self.since
            .iter()
            .all(|old| old.running <= 1.0 && same_counters(&base, old))
    }

    fn window_clean(&self, base: &Snap, snap: &Snap) -> bool {
        // `running > 1` is already `dirty`: `note` sets it on the scrape
        // that showed the overlap, and the baseline keeps it until a close
        // at `running <= 1`. An idle rebase will not clear it while the
        // baseline or any scrape since still overlapped.
        if self.dirty {
            return false;
        }
        if snap.requests != base.requests + 1 {
            return false;
        }
        if whole_delta(base.e2e_count, snap.e2e_count) != Some(1) {
            return false;
        }
        // The first token of this finish, not one left over from the
        // previous window and not one that arrived for a request still running.
        if whole_delta(base.ttft_count, snap.ttft_count) != Some(1) {
            return false;
        }
        if !settled(base) || !settled(snap) {
            return false;
        }
        let batch = snap.batch.saturating_sub(base.batch);
        let streaming = streaming_of(snap).saturating_sub(streaming_of(base));
        // Both series stepped: the token counters are two requests.
        if batch > 0 && streaming > 0 {
            return false;
        }
        let prompt = snap.prompt.saturating_sub(base.prompt);
        let cached = snap.cached.saturating_sub(base.cached);
        if cached > prompt || snap.generated < base.generated {
            return false;
        }
        snap.e2e_sum + 1e-6 >= base.e2e_sum && snap.ttft_sum + 1e-6 >= base.ttft_sum
    }

    fn record(&mut self, base: &Snap, prev: &Snap, snap: &Snap, queue: QueueTie) {
        let prompt = snap.prompt - base.prompt;
        let cached = snap.cached - base.cached;
        let generated = snap.generated - base.generated;
        let ttft_s = snap.ttft_sum - base.ttft_sum;
        let e2e_s = snap.e2e_sum - base.e2e_sum;
        let (mut prompt_tps, mut gen_tps) = rates(prompt, cached, generated, ttft_s, e2e_s);
        let batch = snap.batch.saturating_sub(base.batch);
        if batch > 0 || self.ttft_unlabeled {
            // Non-streaming, or a time-to-first-token series with no
            // `is_streaming` label: the rate is not a measurement. The
            // cached count still is.
            prompt_tps = None;
            gen_tps = None;
        } else if hide_prompt(queue, ttft_s) {
            prompt_tps = None;
        }
        self.finishes.push_back(Finish {
            at: snap.at,
            wall_lo: prev.before,
            wall_hi: snap.wall,
            prompt,
            generated,
            cached,
            e2e_s,
            prompt_tps,
            gen_tps,
            taken: false,
        });
        while self.finishes.len() > KEPT_FINISHES {
            self.finishes.pop_front();
        }
    }

    /// Remember a step of the finished count. Across a gap the increases
    /// are not known: the delta can hide a reset.
    fn record_sight(&mut self, prev: &Snap, snap: &Snap, gap: bool) {
        let n = snap.requests.saturating_sub(prev.requests);
        if n == 0 {
            return;
        }
        self.push_sight(Sight {
            at: snap.at,
            wall_lo: prev.before,
            wall_hi: snap.wall,
            n: (!gap).then_some(n),
        });
    }

    fn push_sight(&mut self, sight: Sight) {
        self.sights.push_back(sight);
        while self.sights.len() > KEPT_FINISHES {
            self.sights.pop_front();
        }
    }

    fn open(&mut self, prev: Snap, snap: Snap) {
        let step = whole_delta(prev.ttft_count, snap.ttft_count).unwrap_or(-1);
        // Tokens or a finished-request counter moved with no finish. The
        // first-token count from the baseline is not a separate flag: the
        // window already requires that count to grow by exactly one.
        let moved = snap.requests != prev.requests
            || snap.batch != prev.batch
            || snap.prompt != prev.prompt
            || snap.cached != prev.cached
            || snap.generated != prev.generated
            || !near(snap.e2e_sum, prev.e2e_sum)
            || !near(snap.e2e_count, prev.e2e_count);
        if moved || (step != 0 && step != 1) || snap.ttft_sum + 1e-6 < prev.ttft_sum {
            self.dirty = true;
        }
        self.since.push(snap);
        self.remember(snap);
        if self.since.len() > HISTORY_CAP {
            self.since.clear();
            self.base = Some(snap);
            self.dirty = snap.running > 1.0;
        }
    }

    fn close(&mut self, snap: Snap) {
        self.since.clear();
        self.dirty = snap.running > 1.0;
        self.base = Some(snap);
        self.remember(snap);
    }

    fn rebase(&mut self, snap: Snap) {
        self.since.clear();
        self.dirty = false;
        self.base = Some(snap);
        self.remember(snap);
    }

    fn restart(&mut self, prev: &Snap, snap: Snap) {
        self.finishes.clear();
        self.sights.clear();
        self.since.clear();
        self.dirty = snap.running > 1.0;
        self.base = Some(snap);
        self.remember(snap);
        self.queue_baseline(&snap);
        // The new process may already have finished requests. Their token
        // increases are not a delta of the process this scrape replaced.
        if snap.requests > 0 {
            self.push_sight(Sight {
                at: snap.at,
                wall_lo: prev.before,
                wall_hi: snap.wall,
                n: None,
            });
        }
    }

    /// The first scrape of a process. Steps before it were not seen.
    /// Anchor only when nothing is between leaving the queue and finishing:
    /// `running` is zero and the queue count equals the finished count and
    /// the first-token count. A gauge that still reads zero while a request
    /// has already started would otherwise lend the next finish the wrong
    /// queue step.
    fn queue_baseline(&mut self, snap: &Snap) {
        self.queue_drop();
        self.queue_live = queue_nonzero(snap);
        self.queue_anchor = idle_counts(snap);
    }

    /// Forget the order. The series stays live: an untied finish must not
    /// be treated as a queue time of zero.
    fn queue_drop(&mut self) {
        self.queue_obs.clear();
        self.queue_used = 0;
        self.queue_anchor = false;
    }

    /// Fold the histogram delta since `prev` into the chain, then give this
    /// scrape's finishes the oldest unused steps.
    fn queue_advance(&mut self, prev: &Snap, snap: &Snap) -> QueueTie {
        if queue_nonzero(snap) {
            self.queue_live = true;
        }
        let gap = snap.at.saturating_duration_since(prev.at) > PENDING_FOR;
        let overlapped = prev.running > 1.0 || snap.running > 1.0;
        if gap || overlapped {
            // The delta crosses a hole in the order. Do not record it.
            self.queue_drop();
        } else if self.queue_anchor {
            self.queue_push(prev, snap);
        }
        let finishes = snap.requests.saturating_sub(prev.requests);
        let mut tie = self.queue_take(finishes);
        let outstanding = self.queue_obs.len().saturating_sub(self.queue_used);
        // More departures than one running request can explain. The order
        // has drifted, so this finish's step is not known either.
        if outstanding > 1 && snap.running <= 1.0 {
            self.queue_drop();
            if matches!(tie, QueueTie::Known(_)) {
                tie = QueueTie::Unknown;
            }
        }
        if snap.running <= 0.0 && self.queue_obs.len() == self.queue_used {
            self.queue_obs.clear();
            self.queue_used = 0;
            self.queue_anchor = idle_counts(snap);
        }
        tie
    }

    fn queue_push(&mut self, prev: &Snap, snap: &Snap) {
        let Some(count) = whole_delta(prev.queue_count, snap.queue_count) else {
            self.queue_drop();
            return;
        };
        let sum = snap.queue_sum - prev.queue_sum;
        if sum < -1e-6 {
            self.queue_drop();
            return;
        }
        let sum = sum.max(0.0);
        match count {
            0 => {
                if sum > 1e-6 {
                    self.queue_drop();
                }
            }
            1 => self.queue_obs.push_back(Some(sum)),
            n if (2..=QUEUE_CAP as i64).contains(&n) => {
                for _ in 0..n {
                    self.queue_obs.push_back(None);
                }
            }
            _ => self.queue_drop(),
        }
        if self.queue_obs.len() > QUEUE_CAP {
            self.queue_drop();
        }
    }

    fn queue_take(&mut self, finishes: u64) -> QueueTie {
        if finishes == 0 || !self.queue_live {
            return QueueTie::Absent;
        }
        if finishes != 1 {
            // A jump bigger than the chain can hold is not a burst of
            // finishes. Walking it would stall the poller.
            if finishes > QUEUE_CAP as u64 {
                self.queue_drop();
                return QueueTie::Unknown;
            }
            for _ in 0..finishes {
                let _ = self.queue_pop();
            }
            return QueueTie::Unknown;
        }
        match self.queue_pop() {
            Some(Some(seconds)) => QueueTie::Known(seconds),
            _ => QueueTie::Unknown,
        }
    }

    fn queue_pop(&mut self) -> Option<Option<f64>> {
        if self.queue_used >= self.queue_obs.len() {
            return None;
        }
        let step = self.queue_obs[self.queue_used];
        self.queue_used += 1;
        Some(step)
    }

    fn remember(&mut self, snap: Snap) {
        self.history.push_back(snap);
        while self.history.len() > HISTORY_CAP {
            self.history.pop_front();
        }
    }

    fn push_probe(&mut self, at: Instant, stamp: SystemTime, gap: bool) {
        self.probes.push_back(Probe { at, stamp, gap });
        while self.probes.len() > HISTORY_CAP {
            self.probes.pop_front();
        }
    }
}

fn same_counters(left: &Snap, right: &Snap) -> bool {
    left.requests == right.requests
        && left.batch == right.batch
        && left.prompt == right.prompt
        && left.cached == right.cached
        && left.generated == right.generated
        && near(left.e2e_sum, right.e2e_sum)
        && near(left.e2e_count, right.e2e_count)
        && near(left.ttft_sum, right.ttft_sum)
        && near(left.ttft_count, right.ttft_count)
        && near(left.queue_sum, right.queue_sum)
        && near(left.queue_count, right.queue_count)
}

fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6
}

fn went_back(prev: &Snap, snap: &Snap) -> bool {
    snap.requests < prev.requests
        || snap.batch < prev.batch
        || snap.prompt < prev.prompt
        || snap.cached < prev.cached
        || snap.generated < prev.generated
        || snap.e2e_sum + 1e-6 < prev.e2e_sum
        || snap.e2e_count + 1e-6 < prev.e2e_count
        || snap.ttft_sum + 1e-6 < prev.ttft_sum
        || snap.ttft_count + 1e-6 < prev.ttft_count
        || snap.queue_sum + 1e-6 < prev.queue_sum
        || snap.queue_count + 1e-6 < prev.queue_count
}

/// The first-token count equals the finished count: nothing is past its
/// first token and still running.
fn settled(snap: &Snap) -> bool {
    whole_delta(snap.requests as f64, snap.ttft_count) == Some(0)
}

fn streaming_of(snap: &Snap) -> u64 {
    snap.requests.saturating_sub(snap.batch)
}

/// Queue time at most five percent of the time to the first token.
fn queue_within(queue_s: f64, ttft_s: f64) -> bool {
    queue_s <= ttft_s * 0.05
}

fn queue_nonzero(snap: &Snap) -> bool {
    snap.queue_count > 1e-6 || snap.queue_sum > 1e-6
}

/// Hide the prompt rate when this finish's own queue time is over five
/// percent of its time to first token, or when that time is not known.
/// An engine with no queue series ([`QueueTie::Absent`]) does not hide it.
fn hide_prompt(queue: QueueTie, ttft_s: f64) -> bool {
    match queue {
        QueueTie::Absent => false,
        QueueTie::Known(queue_s) => !queue_within(queue_s, ttft_s),
        QueueTie::Unknown => true,
    }
}

/// The row's timestamp, truncated to the second, fits the finish.
///
/// `T` is the end time with the fraction dropped, so the row ended in
/// `[T, T+1)`. The finish is in `(wall_lo, wall_hi]`, and llama-swap's
/// end is at most 0.5 s after the engine's, so the row fits when
/// `wall_lo - 1 s < T <= wall_hi + 0.5 s`.
fn stamp_fits(timestamp: Option<&str>, wall_lo: SystemTime, wall_hi: SystemTime) -> bool {
    let Some(text) = timestamp else {
        return false;
    };
    let Some(row) = parse_rfc3339(text) else {
        return false;
    };
    let (Some(lo), Some(hi)) = (wall_secs(wall_lo), wall_secs(wall_hi)) else {
        return false;
    };
    let row = row as f64;
    row > lo - 1.0 && row <= hi + 0.5
}

fn wall_secs(wall: SystemTime) -> Option<f64> {
    let secs = wall
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    secs.is_finite().then_some(secs)
}

/// Nothing is left between the queue and the finish: the gauge is idle and
/// the queue count, the finished count and the first-token count agree.
fn idle_counts(snap: &Snap) -> bool {
    snap.running <= 0.0
        && whole_delta(snap.requests as f64, snap.queue_count) == Some(0)
        && whole_delta(snap.requests as f64, snap.ttft_count) == Some(0)
}

fn clock_stepped(prev: &Snap, snap: &Snap) -> bool {
    let mono = snap.at.saturating_duration_since(prev.at).as_secs_f64();
    let Ok(wall) = snap.wall.duration_since(prev.wall) else {
        return true;
    };
    (wall.as_secs_f64() - mono).abs() > CLOCK_STEP.as_secs_f64()
}

fn coverage_gap(prev: &Snap, snap: &Snap) -> bool {
    if snap.at.saturating_duration_since(prev.at) > COVERAGE_GAP {
        return true;
    }
    match snap.wall.duration_since(prev.wall) {
        Ok(delta) => delta > COVERAGE_GAP,
        Err(_) => true,
    }
}

/// A scrape stamped after `T + 6 s` has been seen, so the isolation window
/// is closed. A row with no timestamp never gets here: [`stamp_fits`]
/// refuses it before a finish can be a hit, so a missing window is not
/// treated as a decision.
fn isolation_decided(model: &Model, timestamp: Option<&str>) -> bool {
    let Some(win_hi) = isolation_hi(timestamp) else {
        return false;
    };
    model
        .probes
        .iter()
        .any(|probe| wall_secs(probe.stamp).is_some_and(|stamp| stamp > win_hi))
}

/// `finish_at` is the only finish whose scrape interval overlaps
/// `[T - 5 s, T + 6 s]`, and the scrapes cover that span.
fn isolation_allows(model: &Model, finish_at: Instant, timestamp: Option<&str>) -> bool {
    let (Some(win_lo), Some(win_hi)) = (isolation_lo(timestamp), isolation_hi(timestamp)) else {
        return false;
    };
    if !finish_alone(model, finish_at, win_lo, win_hi) {
        return false;
    }
    span_covered(model, win_lo, win_hi)
}

fn isolation_lo(timestamp: Option<&str>) -> Option<f64> {
    let row = parse_rfc3339(timestamp?)? as f64;
    Some(row - ISOLATION_BEFORE_S)
}

fn isolation_hi(timestamp: Option<&str>) -> Option<f64> {
    let row = parse_rfc3339(timestamp?)? as f64;
    Some(row + ISOLATION_AFTER_S)
}

fn finish_alone(model: &Model, finish_at: Instant, win_lo: f64, win_hi: f64) -> bool {
    let mut total = 0u64;
    let mut own = false;
    for sight in &model.sights {
        let (Some(lo), Some(hi)) = (wall_secs(sight.wall_lo), wall_secs(sight.wall_hi)) else {
            return false;
        };
        // `(lo, hi]` overlaps `[win_lo, win_hi]`.
        if hi < win_lo || lo >= win_hi {
            continue;
        }
        let Some(n) = sight.n else {
            return false;
        };
        total = total.saturating_add(n);
        if sight.at == finish_at {
            own = true;
        }
    }
    own && total == 1
}

/// The probes tile `[win_lo, win_hi]` with no gap, clock step, reset,
/// restart or watcher start inside it.
fn span_covered(model: &Model, win_lo: f64, win_hi: f64) -> bool {
    let mut covered_from: Option<f64> = None;
    for probe in &model.probes {
        let Some(stamp) = wall_secs(probe.stamp) else {
            return false;
        };
        if probe.gap {
            if stamp <= win_lo {
                // The break ended before the window. Coverage may start here.
                covered_from = Some(stamp);
            } else {
                // The unseen stretch reaches into the window.
                covered_from = None;
            }
        } else if covered_from.is_none() && stamp <= win_lo {
            // The watcher's gap probe ages out after [`FINISH_FOR`]. A
            // later scrape at or before the window is not a gap, and it
            // is what coverage starts from.
            covered_from = Some(stamp);
        }
        if covered_from.is_some_and(|from| from <= win_lo) && stamp > win_hi {
            return true;
        }
    }
    false
}

/// `duration_ms` agrees with the window's end-to-end seconds.
/// llama-swap's duration covers the engine's time plus its own overhead,
/// so it is never meaningfully shorter: at least 0.1 s under, and at most
/// the larger of 0.5 s and two percent over (36_468 ms against 36.47 s).
fn duration_agrees(duration_ms: u64, e2e_s: f64) -> bool {
    if !e2e_s.is_finite() || e2e_s < 0.0 {
        return false;
    }
    // Whole milliseconds, so 100 ms under is in and 101 ms under is out
    // with no float edge.
    let e2e_ms = (e2e_s * 1000.0).round();
    if !e2e_ms.is_finite() || !(0.0..=i64::MAX as f64).contains(&e2e_ms) {
        return false;
    }
    // `duration_ms` is a `u64` from llama-swap. Casting it to `i64` wraps
    // a value past `i64::MAX`, and the subtraction then overflows.
    let Ok(duration_ms) = i64::try_from(duration_ms) else {
        return false;
    };
    let delta_ms = duration_ms - e2e_ms as i64;
    let over_ms = ((e2e_s * 0.02).max(0.5) * 1000.0).round() as i64;
    delta_ms >= -100 && delta_ms <= over_ms
}

fn whole_delta(before: f64, now: f64) -> Option<i64> {
    let delta = now - before;
    if !delta.is_finite() {
        return None;
    }
    let rounded = delta.round();
    if (delta - rounded).abs() > 1e-6 || !(i64::MIN as f64..=i64::MAX as f64).contains(&rounded) {
        return None;
    }
    Some(rounded as i64)
}

fn one_finish(prev: &Snap, snap: &Snap) -> bool {
    snap.requests == prev.requests + 1 && whole_delta(prev.e2e_count, snap.e2e_count) == Some(1)
}

fn finish_event(prev: &Snap, snap: &Snap) -> bool {
    snap.requests > prev.requests || snap.e2e_count > prev.e2e_count + 1e-6
}

fn rates(
    prompt: u64,
    cached: u64,
    generated: u64,
    ttft_s: f64,
    e2e_s: f64,
) -> (Option<f64>, Option<f64>) {
    let prompt_tps = (ttft_s > 0.0 && prompt > cached).then(|| (prompt - cached) as f64 / ttft_s);
    let gen_tps =
        (generated >= 2 && e2e_s > ttft_s).then(|| (generated - 1) as f64 / (e2e_s - ttft_s));
    (kept(prompt_tps), kept(gen_tps))
}

fn kept(tps: Option<f64>) -> Option<f64> {
    tps.filter(|tps| tps.is_finite() && *tps <= MAX_ENGINE_TPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Clock(Instant);

    impl Clock {
        fn at(&self, ms: u64) -> Instant {
            self.0 + Duration::from_millis(ms)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn totals(
        requests: u64,
        prompt: u64,
        cached: u64,
        generated: u64,
        e2e: (f64, f64),
        ttft: (f64, f64),
        running: f64,
    ) -> Totals {
        Totals {
            requests,
            batch: 0,
            prompt,
            cached,
            generated,
            e2e_sum: e2e.0,
            e2e_count: e2e.1,
            ttft_sum: ttft.0,
            ttft_count: ttft.1,
            queue_sum: 0.0,
            queue_count: 0.0,
            running,
        }
    }

    /// A streaming row whose output and duration match the usual window:
    /// 401 generated tokens, 10 s end to end.
    fn claim(seq: u64, input: u64) -> Claim {
        measured(seq, input, 401, 10_000)
    }

    fn measured(seq: u64, input: u64, output: u64, duration_ms: u64) -> Claim {
        Claim {
            seq,
            model: "flash".to_owned(),
            input,
            output: Some(output),
            duration_ms: Some(duration_ms),
            prompt_known: false,
            gen_known: false,
            // Callers that expect a fill overwrite this. The epoch stamp
            // matches no finish whose wall clock has moved with the
            // monotonic clock, so a forgotten overwrite fills nothing.
            timestamp: Some("1970-01-01T00:00:00Z".to_owned()),
        }
    }

    fn zeros() -> Totals {
        totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)
    }

    /// `sample` every 2 s from `from` through `to`, walls matching the
    /// monotonic clock. Two seconds is inside the coverage gap.
    fn grid(book: &mut SglangBook, clock: &Clock, from: u64, to: u64, sample: Totals) {
        let mut ms = from;
        loop {
            see(book, clock, ms, sample);
            if ms >= to {
                break;
            }
            let next = ms.saturating_add(2_000);
            ms = if next > to { to } else { next };
        }
    }

    /// Idle copies of `done` until a scrape is stamped strictly after
    /// `row_sec + 6`, which is when isolation decides the row.
    fn settle(book: &mut SglangBook, clock: &Clock, fin_ms: u64, row_sec: i64, done: Totals) {
        let horizon = (row_sec as u64 + 6) * 1_000;
        let mut ms = fin_ms.saturating_add(2_000);
        loop {
            see(book, clock, ms, done);
            if ms > horizon {
                break;
            }
            ms = ms.saturating_add(2_000);
        }
    }

    fn stamp_claim(mut row: Claim, sec: i64) -> Claim {
        row.timestamp = Some(stamp_secs(sec));
        row
    }

    /// One streaming finish at second 12, decided by second 20.
    /// Prompt 1,000, cached 600, TTFT 2 s, 401 generated, e2e 10 s.
    /// The first token lands at second 10, two seconds before the finish.
    /// The next scrape a caller adds must be at 22_000 or later.
    fn one(book: &mut SglangBook, clock: &Clock, seq: u64) -> Fill {
        let after = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        grid(book, clock, 0, 8_000, zeros());
        book.activity(clock.at(100), &[]);
        see(
            book,
            clock,
            10_000,
            totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0),
        );
        see(book, clock, 12_000, after);
        book.activity(clock.at(12_500), &[stamp_claim(claim(seq, 1_000), 12)]);
        settle(book, clock, 12_000, 12, after);
        book.fill(seq).expect("filled")
    }

    /// One clean finish at second 12, from `before` to `after`. The row is
    /// stamped at that second and decided by second 20. `before` stays idle
    /// across the lead, so the watcher starts outside `[7, 18]`.
    fn finish_at(
        book: &mut SglangBook,
        clock: &Clock,
        before: Totals,
        after: Totals,
        row: Claim,
    ) -> Fill {
        let seq = row.seq;
        grid(book, clock, 0, 10_000, before);
        book.activity(clock.at(100), &[]);
        see(book, clock, 12_000, after);
        book.activity(clock.at(12_500), &[stamp_claim(row, 12)]);
        settle(book, clock, 12_000, 12, after);
        book.fill(seq).expect("filled")
    }

    #[test]
    fn one_request_at_a_time_fills_each_row_from_the_formula() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let first = one(&mut book, &clock, 1);
        assert_eq!(first.cached, 600);
        assert_eq!(first.prompt_tps, Some(200.0), "(1000 - 600) / 2");
        assert_eq!(first.gen_tps, Some(50.0), "(401 - 1) / (10 - 2)");
        assert_eq!(first.cached + (1_000 - first.cached), 1_000);

        // Second request, still alone in its own window: 800 prompt, 200
        // cached, TTFT 1 s, 101 generated, e2e 3 s. It ends at second 26,
        // so the first finish is outside [21, 32].
        let held = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let second_done = totals(2, 1_800, 800, 502, (13.0, 2.0), (3.0, 2.0), 0.0);
        see(&mut book, &clock, 22_000, held);
        see(
            &mut book,
            &clock,
            24_000,
            totals(1, 1_000, 600, 401, (10.0, 1.0), (3.0, 2.0), 1.0),
        );
        see(&mut book, &clock, 26_000, second_done);
        book.activity(
            clock.at(26_500),
            &[stamp_claim(measured(2, 800, 101, 3_000), 26)],
        );
        settle(&mut book, &clock, 26_000, 26, second_done);
        let second = book.fill(2).expect("second");
        assert_eq!(second.cached, 200);
        assert_eq!(second.prompt_tps, Some(600.0));
        assert_eq!(second.gen_tps, Some(50.0), "(101 - 1) / (3 - 1)");
        assert_eq!(book.fill(1).expect("kept").cached, 600);
    }

    #[test]
    fn ttft_on_the_finish_scrape_still_counts() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 500, 500, 11, (4.0, 1.0), (2.0, 1.0), 0.0);
        let fill = finish_at(
            &mut book,
            &clock,
            zeros(),
            done,
            measured(1, 500, 11, 4_000),
        );
        assert_eq!(fill.cached, 500);
        assert_eq!(fill.prompt_tps, None, "nothing new to prefill");
        assert_eq!(fill.gen_tps, Some(5.0), "(11 - 1) / (4 - 2)");
    }

    #[test]
    fn two_finishes_in_one_scrape_fill_nothing_and_the_next_request_does() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let burst = totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, burst);
        book.activity(
            clock.at(12_500),
            &[
                stamp_claim(claim(1, 1_000), 12),
                stamp_claim(claim(2, 1_000), 12),
            ],
        );
        see(&mut book, &clock, 14_000, burst);
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
        // The burst interval is (10, 12]. Second 24's window [19, 30] is
        // clear of it, so the single request after the burst can fill.
        grid(&mut book, &clock, 16_000, 22_000, burst);
        let later = totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0);
        see(&mut book, &clock, 24_000, later);
        book.activity(clock.at(24_500), &[stamp_claim(claim(3, 1_000), 24)]);
        settle(&mut book, &clock, 24_000, 24, later);
        let fill = book.fill(3).expect("the later request");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, Some(200.0));
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    #[test]
    fn an_overlapping_request_fills_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 2.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_counter_reset_drops_the_straddling_request_and_the_next_one_fills() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let old = totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 0.0);
        grid(&mut book, &clock, 0, 2_000, old);
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            4_000,
            totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 1.0),
        );
        // Restart. Its probe is a hole, so the next row's second has to
        // sit at least 5 s later or the hole falls inside the window.
        see(&mut book, &clock, 6_000, zeros());
        book.activity(clock.at(6_500), &[stamp_claim(claim(1, 1_000), 6)]);
        see(&mut book, &clock, 8_000, zeros());
        assert_eq!(
            book.fill(1),
            None,
            "the reset is not a measurement of that row"
        );
        grid(&mut book, &clock, 10_000, 14_000, zeros());
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        see(&mut book, &clock, 16_000, done);
        book.activity(clock.at(16_500), &[stamp_claim(claim(2, 1_000), 16)]);
        settle(&mut book, &clock, 16_000, 16, done);
        let fill = book.fill(2).expect("after the restart");
        assert_eq!(
            (fill.cached, fill.prompt_tps, fill.gen_tps),
            (600, Some(200.0), Some(50.0))
        );
    }

    #[test]
    fn a_gap_longer_than_ten_seconds_fills_nothing_and_an_idle_gap_does_not_stick() {
        let clock = Clock(Instant::now());
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        // Ten seconds still satisfies the clean-window bound, and it is
        // wider than the 3 s coverage gap, so isolation leaves the row
        // empty. The row is decided: a looser coverage gap would fill it.
        let mut book = SglangBook::default();
        see(&mut book, &clock, 0, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 10_000, done);
        book.activity(clock.at(10_500), &[stamp_claim(claim(7, 1_000), 10)]);
        settle(&mut book, &clock, 10_000, 10, done);
        assert_eq!(book.fill(7), None, "ten seconds is a coverage hole");

        // Exactly 3 s between scrapes is still covered.
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 8_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 11_000, done);
        book.activity(clock.at(11_500), &[stamp_claim(claim(3, 1_000), 11)]);
        settle(&mut book, &clock, 11_000, 11, done);
        assert_eq!(book.fill(3).expect("three seconds").prompt_tps, Some(200.0));

        // One millisecond over 3 s is a hole.
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 8_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 11_001, done);
        book.activity(clock.at(11_500), &[stamp_claim(claim(4, 1_000), 11)]);
        settle(&mut book, &clock, 11_001, 11, done);
        assert_eq!(book.fill(4), None, "3.001 s is a coverage hole");

        // An idle gap rebases. The request after it, once the hole is
        // behind the isolation window, is filled. Cached zero is a number.
        let mut book = SglangBook::default();
        see(&mut book, &clock, 0, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 11_000, zeros());
        grid(&mut book, &clock, 13_000, 22_000, zeros());
        let idle_done = totals(1, 1_000, 0, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        see(&mut book, &clock, 24_000, idle_done);
        book.activity(clock.at(24_500), &[stamp_claim(claim(8, 1_000), 24)]);
        settle(&mut book, &clock, 24_000, 24, idle_done);
        let fill = book.fill(8).expect("after the idle gap");
        assert_eq!(fill.cached, 0, "a measured zero is a number");
        assert_eq!(fill.prompt_tps, Some(500.0));
    }

    #[test]
    fn no_row_and_two_rows_with_the_same_input_fill_nothing() {
        let clock = Clock(Instant::now());
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, done);
        // The unmatched row and the row it did match share this read, so
        // the miss must not take the finish away from the match.
        book.activity(
            clock.at(12_500),
            &[
                stamp_claim(claim(1, 999), 12),
                stamp_claim(claim(2, 1_000), 12),
            ],
        );
        settle(&mut book, &clock, 12_000, 12, done);
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2).expect("the one match").cached, 600);

        // Two rows and one finish: neither is filled.
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, done);
        book.activity(
            clock.at(12_500),
            &[
                stamp_claim(claim(3, 1_000), 12),
                stamp_claim(claim(4, 1_000), 12),
            ],
        );
        settle(&mut book, &clock, 12_000, 12, done);
        assert_eq!(book.fill(3), None);
        assert_eq!(book.fill(4), None);
    }

    #[test]
    fn two_rows_in_one_span_each_get_their_own_finish() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        // Eight seconds apart, so each finish is alone in the other's
        // isolation window. Both rows are still one activity read.
        let first_done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let second_done = totals(2, 1_500, 1_100, 402, (12.0, 2.0), (3.0, 2.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, first_done);
        grid(&mut book, &clock, 14_000, 18_000, first_done);
        see(&mut book, &clock, 20_000, second_done);
        book.activity(
            clock.at(20_500),
            &[
                stamp_claim(claim(1, 1_000), 12),
                stamp_claim(measured(2, 500, 1, 2_000), 20),
            ],
        );
        settle(&mut book, &clock, 20_000, 20, second_done);
        let first = book.fill(1).expect("first");
        let second = book.fill(2).expect("second");
        assert_eq!(first.cached, 600);
        assert_eq!(first.prompt_tps, Some(200.0));
        assert_eq!(second.cached, 500, "fully cached");
        assert_eq!(second.prompt_tps, None);
        assert_eq!(second.gen_tps, None, "one generated token");
    }

    #[test]
    fn a_reported_rate_keeps_the_rate_and_still_fills_a_missing_cached_count() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let mut row = claim(1, 1_000);
        row.prompt_known = true;
        let fill = finish_at(&mut book, &clock, zeros(), done, row);
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None, "a reported rate is not replaced");
        assert_eq!(fill.gen_tps, Some(50.0), "a rate the row did not report");
        // Same lengths and the same second. The finish is already taken.
        book.activity(clock.at(21_000), &[stamp_claim(claim(2, 1_000), 12)]);
        see(&mut book, &clock, 22_000, done);
        assert_eq!(book.fill(2), None, "the row already took the finish");
    }

    #[test]
    fn one_generated_token_and_a_fully_cached_prompt_omit_the_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let fill = finish_at(
            &mut book,
            &clock,
            zeros(),
            totals(1, 1_000, 400, 1, (5.0, 1.0), (2.0, 1.0), 0.0),
            measured(1, 1_000, 1, 5_000),
        );
        assert_eq!(fill.cached, 400);
        assert_eq!(fill.prompt_tps, Some(300.0));
        assert_eq!(fill.gen_tps, None);
    }

    #[test]
    fn cached_above_the_prompt_is_not_a_clean_window() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 100, 150, 10, (4.0, 1.0), (1.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 100)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 100, 150, 10, (4.0, 1.0), (1.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_request_and_the_latency_histogram_must_agree() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 0.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 0.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn two_first_tokens_before_the_finish_fill_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (4.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (4.0, 2.0), 0.0)),
        );
        book.activity(clock.at(3_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (4.0, 2.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_missing_counter_and_a_rate_past_the_engine_ceiling_do_not_fill_a_number() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe("flash", clock.at(500), SystemTime::UNIX_EPOCH, None);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);

        let mut book = SglangBook::default();
        // 1_000 new tokens in 0.0005 s is 2_000_000 tok/s, past 1_000_000.
        let fill = finish_at(
            &mut book,
            &clock,
            zeros(),
            totals(1, 1_000, 0, 401, (10.0, 1.0), (0.0005, 1.0), 0.0),
            claim(4, 1_000),
        );
        assert_eq!(fill.cached, 0);
        assert_eq!(fill.prompt_tps, None);
        assert_eq!(fill.gen_tps, Some(400.0 / 9.9995));
    }

    #[test]
    fn a_row_with_no_metrics_within_eleven_seconds_is_not_filled_later() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.activity(clock.at(1_500) + Duration::from_secs(11), &[]);
        assert_eq!(book.fill(1), None);
        book.observe(
            "flash",
            clock.at(1_500) + Duration::from_secs(11) + Duration::from_millis(50),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    /// B's first token fell in a window that was rejected. Between the next
    /// two scrapes C starts, produces its first token and B finishes, and
    /// `num_running_reqs` never shows 2. The window sees one first token
    /// and one finish; B must not take C's time to first token.
    #[test]
    fn a_first_token_from_an_earlier_window_is_not_this_finish() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 2.0)),
        );
        // B's first token, while the overlap is still visible.
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (1.5, 1.0), 2.0)),
        );
        // The other request finishes (its first token too). B is still in
        // flight, so the first-token count is ahead of the finished count.
        // This window is rejected; its snap is the next window's start.
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 400, 0, 50, (5.0, 1.0), (3.0, 2.0), 1.0)),
        );
        // C's first token and B's finish both land between these scrapes.
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_400, 600, 451, (15.0, 2.0), (3.4, 3.0), 1.0)),
        );
        book.activity(clock.at(4_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_400, 600, 451, (15.0, 2.0), (3.4, 3.0), 1.0)),
        );
        assert_eq!(
            book.fill(1),
            None,
            "B's row must not show C's time to first token"
        );
    }

    /// The first token was already counted before this window, so the
    /// finish scrape adds none. Nothing is filled.
    #[test]
    fn a_first_token_increase_of_zero_fills_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (1.5, 1.0), 1.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (1.5, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (1.5, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_non_streaming_request_fills_the_cached_count_and_no_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        // The false series' time to first token is the whole request,
        // 5e-6 s short of the end-to-end sum.
        let done = Totals {
            batch: 1,
            ..totals(1, 1_000, 600, 401, (10.0, 1.0), (10.0 - 5e-6, 1.0), 0.0)
        };
        let fill = finish_at(&mut book, &clock, zeros(), done, claim(1, 1_000));
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None);
        assert_eq!(fill.gen_tps, None);
    }

    #[test]
    fn a_non_streaming_request_after_a_streaming_one_fills_cached_only() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let streaming = one(&mut book, &clock, 1);
        assert_eq!(streaming.prompt_tps, Some(200.0));
        assert_eq!(streaming.gen_tps, Some(50.0));
        let held = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let batch = Totals {
            batch: 1,
            ..totals(
                2,
                1_800,
                800,
                502,
                (13.0, 2.0),
                (2.0 + 3.0 - 5e-6, 2.0),
                0.0,
            )
        };
        see(&mut book, &clock, 22_000, held);
        see(&mut book, &clock, 24_000, batch);
        book.activity(
            clock.at(24_500),
            &[stamp_claim(measured(2, 800, 101, 3_000), 24)],
        );
        settle(&mut book, &clock, 24_000, 24, batch);
        let fill = book.fill(2).expect("cached count");
        assert_eq!(fill.cached, 200);
        assert_eq!(fill.prompt_tps, None);
        assert_eq!(fill.gen_tps, None);
        assert_eq!(book.fill(1).expect("kept").prompt_tps, Some(200.0));
    }

    #[test]
    fn a_mixed_window_fills_nothing_and_the_next_streaming_request_does() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        // One streaming finish and one non-streaming finish, one scrape.
        let burst = Totals {
            batch: 1,
            ..totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)
        };
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, burst);
        book.activity(
            clock.at(12_500),
            &[
                stamp_claim(claim(1, 1_000), 12),
                stamp_claim(claim(2, 1_000), 12),
            ],
        );
        see(&mut book, &clock, 14_000, burst);
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
        // The mixed scrape stays outside second 24's window. The new
        // request is streaming: the batch counter does not step.
        grid(&mut book, &clock, 16_000, 22_000, burst);
        let later = Totals {
            batch: 1,
            ..totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)
        };
        see(&mut book, &clock, 24_000, later);
        book.activity(clock.at(24_500), &[stamp_claim(claim(3, 1_000), 24)]);
        settle(&mut book, &clock, 24_000, 24, later);
        let fill = book.fill(3).expect("the later streaming request");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, Some(200.0));
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    #[test]
    fn one_row_does_not_take_the_first_of_two_matching_finishes() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 2_000, 1_200, 802, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 2_000, 1_200, 802, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        assert_eq!(book.fill(1), None, "two finishes match this one row");
    }

    #[test]
    fn a_later_row_cannot_fill_from_a_finish_already_taken() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        book.activity(clock.at(10_500), &[stamp_claim(claim(1, 1_000), 12)]);
        see(&mut book, &clock, 12_000, done);
        settle(&mut book, &clock, 12_000, 12, done);
        assert_eq!(book.fill(1).expect("the first row").cached, 600);
        book.activity(clock.at(21_000), &[stamp_claim(claim(2, 1_000), 12)]);
        see(&mut book, &clock, 22_000, done);
        assert_eq!(book.fill(2), None, "one finish fills one row");
    }

    #[test]
    fn output_and_duration_must_match_the_window() {
        let clock = Clock(Instant::now());
        let run = |output: u64, duration_ms: u64, e2e: f64, early: bool| {
            let mut book = SglangBook::default();
            let done = totals(1, 1_000, 600, 401, (e2e, 1.0), (2.0, 1.0), 0.0);
            grid(&mut book, &clock, 0, 10_000, zeros());
            book.activity(clock.at(100), &[]);
            let row = stamp_claim(measured(1, 1_000, output, duration_ms), 12);
            if early {
                book.activity(clock.at(10_500), &[row]);
                see(&mut book, &clock, 12_000, done);
            } else {
                see(&mut book, &clock, 12_000, done);
                book.activity(clock.at(12_500), &[row]);
            }
            settle(&mut book, &clock, 12_000, 12, done);
            book.fill(1)
        };
        assert_eq!(
            run(400, 10_000, 10.0, false),
            None,
            "equal input, different output"
        );
        assert_eq!(
            run(401, 20_000, 10.0, false),
            None,
            "duration 20 s against 10 s"
        );
        // One second over does not agree. Two percent of 10 s is 0.2 s,
        // so the upper bound is 0.5 s.
        assert_eq!(run(401, 11_000, 10.0, false), None, "one second over 10 s");

        // 36_468 ms against 36.47 s agrees; two seconds past it does not.
        let before = run(401, 36_468, 36.47, true).expect("read before the finish");
        assert_eq!(before.cached, 600);
        assert_eq!(before.prompt_tps, Some(200.0));
        assert_eq!(
            run(401, 38_470, 36.47, true),
            None,
            "36.47 s plus 2 s is past two percent"
        );
        let after = run(401, 36_468, 36.47, false).expect("read after the finish");
        assert_eq!(after.cached, 600);
    }

    /// llama-swap's duration is the engine's end-to-end time plus its own
    /// overhead: no more than 0.1 s shorter, and no more than the larger
    /// of 0.5 s and two percent longer.
    #[test]
    fn duration_slack_is_one_sided() {
        let clock = Clock(Instant::now());
        let run = |e2e: f64, duration_ms: u64| {
            let mut book = SglangBook::default();
            let done = totals(1, 1_000, 600, 401, (e2e, 1.0), (1.0, 1.0), 0.0);
            grid(&mut book, &clock, 0, 10_000, zeros());
            book.activity(clock.at(100), &[]);
            book.activity(
                clock.at(10_500),
                &[stamp_claim(measured(1, 1_000, 401, duration_ms), 12)],
            );
            see(&mut book, &clock, 12_000, done);
            settle(&mut book, &clock, 12_000, 12, done);
            book.fill(1).map(|fill| fill.cached)
        };
        // 10 s engine: two percent is 0.2 s, so the upper bound is 0.5 s.
        assert_eq!(run(10.0, 9_900), Some(600), "0.1 s shorter still agrees");
        assert_eq!(run(10.0, 9_899), None, "0.101 s shorter does not");
        assert_eq!(run(10.0, 10_500), Some(600), "0.5 s longer still agrees");
        assert_eq!(run(10.0, 10_501), None, "0.501 s longer does not");
        // 100 s engine: two percent is 2 s, which is the upper bound.
        assert_eq!(
            run(100.0, 102_000),
            Some(600),
            "two percent longer still agrees"
        );
        assert_eq!(
            run(100.0, 102_001),
            None,
            "one millisecond past two percent does not"
        );
    }

    #[test]
    fn queue_time_over_five_percent_of_ttft_drops_only_the_prompt_rate() {
        let clock = Clock(Instant::now());
        let run = |queue_sum: f64| {
            let mut book = SglangBook::default();
            let finish = Totals {
                queue_sum,
                queue_count: 1.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
            };
            finish_at(&mut book, &clock, zeros(), finish, claim(1, 1_000))
        };
        // 0.10 s is five percent of a 2 s time to first token.
        let at = run(0.10);
        assert_eq!(at.cached, 600);
        assert_eq!(at.prompt_tps, Some(200.0));
        assert_eq!(at.gen_tps, Some(50.0));
        let over = run(0.11);
        assert_eq!(over.cached, 600);
        assert_eq!(over.prompt_tps, None);
        assert_eq!(over.gen_tps, Some(50.0));
    }

    #[test]
    fn a_prompt_that_moves_before_the_finish_fills_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 100, 0, 0, (0.0, 0.0), (0.0, 0.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn an_overlap_that_changes_no_counter_is_not_rebased_away() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 2.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(3_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_finish_beside_another_request_dirties_the_next_window() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 400, 0, 11, (4.0, 1.0), (1.0, 1.0), 2.0)),
        );
        book.activity(clock.at(1_500), &[measured(1, 400, 11, 4_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 400, 0, 11, (4.0, 1.0), (1.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None, "the finish itself overlapped");
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_400, 600, 412, (14.0, 2.0), (3.0, 2.0), 0.0)),
        );
        book.activity(clock.at(3_500), &[claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_400, 600, 412, (14.0, 2.0), (3.0, 2.0), 0.0)),
        );
        assert_eq!(
            book.fill(2),
            None,
            "the next window still started overlapped"
        );
    }

    #[test]
    fn the_first_scrape_already_overlapping_fills_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 2.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_scrape_without_cached_tokens_total_fills_nothing() {
        use llama_core::backend::Backend;

        use crate::metrics::parse_metrics_for;
        let body = "\
sglang:num_requests_total{model_name=\"flash\",is_streaming=\"true\"} 3.0
sglang:num_requests_total{model_name=\"flash\",is_streaming=\"false\"} 4.0
sglang:prompt_tokens_total{model_name=\"flash\"} 1000.0
sglang:generation_tokens_total{model_name=\"flash\"} 401.0
sglang:e2e_request_latency_seconds_sum{model_name=\"flash\"} 10.0
sglang:e2e_request_latency_seconds_count{model_name=\"flash\"} 1.0
sglang:time_to_first_token_seconds_sum{model_name=\"flash\"} 2.0
sglang:time_to_first_token_seconds_count{model_name=\"flash\"} 1.0
sglang:num_running_reqs{model_name=\"flash\"} 0.0
";
        let sample = parse_metrics_for(Backend::SgLang, body);
        assert_eq!(Totals::from_sample(&sample), None);
        let with_cached =
            format!("{body}sglang:cached_tokens_total{{model_name=\"flash\"}} 600.0\n");
        let totals = Totals::from_sample(&parse_metrics_for(Backend::SgLang, &with_cached))
            .expect("the series is present");
        assert_eq!(totals.requests, 7);
        assert_eq!(totals.batch, 4);
        assert_eq!(totals.cached, 600);
    }

    #[test]
    fn forgetting_a_model_keeps_a_row_already_filled() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let fill = one(&mut book, &clock, 1);
        assert_eq!(fill.cached, 600);
        book.forget("flash");
        assert_eq!(book.fill(1).expect("kept").gen_tps, Some(50.0));
        see(&mut book, &clock, 22_000, zeros());
        book.activity(clock.at(22_100), &[stamp_claim(claim(2, 0), 24)]);
        see(&mut book, &clock, 24_000, zeros());
        assert_eq!(
            book.fill(2),
            None,
            "the new process has not finished a request"
        );
    }

    /// Queue time is observed when a request leaves the queue. Here that
    /// step is on the scrape that finishes the previous request, so the
    /// later window's own queue increase is zero.
    #[test]
    fn a_queue_wait_already_on_the_books_drops_only_the_prompt_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 8_000, zeros());
        book.activity(clock.at(100), &[]);
        // The running request left the queue after 0.02 s.
        let queued = Totals {
            queue_sum: 0.02,
            queue_count: 1.0,
            running: 1.0,
            ..totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)
        };
        see(&mut book, &clock, 10_000, queued);
        // It finishes, and the request that was waiting has already left
        // the queue. 3 s of that wait is not this finish's.
        let first_done = Totals {
            queue_sum: 3.02,
            queue_count: 2.0,
            ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 1.0)
        };
        see(&mut book, &clock, 12_000, first_done);
        book.activity(clock.at(12_500), &[stamp_claim(claim(1, 1_000), 12)]);
        // Decide this row before the waiter finishes, or that finish
        // shares the window.
        settle(&mut book, &clock, 12_000, 12, first_done);
        let first = book.fill(1).expect("the request that barely waited");
        assert_eq!(first.cached, 600);
        assert_eq!(
            first.prompt_tps,
            Some(200.0),
            "its own queue time is 0.02 s"
        );
        assert_eq!(first.gen_tps, Some(50.0));

        // The waiter finishes at second 32, clear of [7, 18]. Nothing new
        // was added to the histogram, so a window delta of zero is not its
        // queue time. TTFT is 4 s and 3 s of it was waiting.
        grid(&mut book, &clock, 22_000, 30_000, first_done);
        let second_done = Totals {
            queue_sum: 3.02,
            queue_count: 2.0,
            ..totals(2, 1_500, 700, 602, (18.0, 2.0), (6.0, 2.0), 0.0)
        };
        see(&mut book, &clock, 32_000, second_done);
        book.activity(
            clock.at(32_500),
            &[stamp_claim(measured(2, 500, 201, 8_000), 32)],
        );
        settle(&mut book, &clock, 32_000, 32, second_done);
        let second = book.fill(2).expect("cached and generation still fill");
        assert_eq!(second.cached, 100);
        assert_eq!(second.prompt_tps, None, "the 3 s wait is this request's");
        assert_eq!(second.gen_tps, Some(50.0), "(201 - 1) / (8 - 4)");
    }

    /// Two requests leave the queue between the same two scrapes. The sum
    /// cannot be split, even when it is under five percent of either TTFT.
    #[test]
    fn a_queue_count_that_jumps_by_more_than_one_hides_the_prompt_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let finish = Totals {
            queue_sum: 0.01,
            queue_count: 2.0,
            ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
        };
        let fill = finish_at(&mut book, &clock, zeros(), finish, claim(1, 1_000));
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None, "two departures, one sum");
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    fn wall_at(ms: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(ms)
    }

    /// Unix second `secs` as llama-swap writes a timestamp.
    fn stamp_secs(secs: i64) -> String {
        format!(
            "{}Z",
            crate::localtime::format_local(secs, crate::localtime::Zone::utc())
        )
    }

    fn see(book: &mut SglangBook, clock: &Clock, ms: u64, totals: Totals) {
        book.observe("flash", clock.at(ms), wall_at(ms), Some(totals));
    }

    /// The timestamp is the end time truncated to the second. The finish
    /// lies in `(t_prev, t_fin]` and llama-swap's end is at most 0.5 s
    /// later, so the row fits when `t_prev - 1 s < T <= t_fin + 0.5 s`.
    #[test]
    fn the_end_time_fits_only_inside_the_scrape_bounds() {
        let clock = Clock(Instant::now());
        let run = |prev_ms: u64, fin_ms: u64, row_secs: i64| {
            let mut book = SglangBook::default();
            let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
            grid(&mut book, &clock, 0, prev_ms - 2_000, zeros());
            see(&mut book, &clock, prev_ms, zeros());
            book.activity(clock.at(prev_ms + 1), &[]);
            let mut row = measured(1, 1_000, 401, 10_000);
            row.timestamp = Some(stamp_secs(row_secs));
            book.activity(clock.at(prev_ms + 2), &[row]);
            see(&mut book, &clock, fin_ms, done);
            settle(&mut book, &clock, fin_ms, row_secs, done);
            book.fill(1).map(|fill| fill.cached)
        };
        assert_eq!(
            run(11_000, 12_000, 10),
            None,
            "T = t_prev - 1 s does not fit"
        );
        assert_eq!(
            run(10_999, 12_000, 10),
            Some(600),
            "just after t_prev - 1 s fits"
        );
        assert_eq!(run(10_000, 10_500, 11), Some(600), "T = t_fin + 0.5 s fits");
        assert_eq!(
            run(10_000, 10_499, 11),
            None,
            "just after t_fin + 0.5 s does not"
        );
    }

    /// Two finishes with the same prompt and output. The earlier row's
    /// duration accepts both end-to-end times; its end time picks one.
    #[test]
    fn equal_prompt_and_output_rows_match_on_their_end_time() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let first_done = totals(1, 4_691, 2_467, 823, (2.35, 1.0), (1.0, 1.0), 0.0);
        let second_done = totals(2, 9_382, 2_763, 1_646, (4.32, 2.0), (2.0, 2.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, first_done);
        // Idle, so the next finish's previous scrape is this one, and the
        // two finishes sit outside each other's isolation windows.
        grid(&mut book, &clock, 14_000, 22_000, first_done);
        see(&mut book, &clock, 24_000, second_done);
        let first = stamp_claim(measured(1, 4_691, 823, 2_462), 12);
        let second = stamp_claim(measured(2, 4_691, 823, 2_062), 24);
        book.activity(clock.at(24_500), &[first, second]);
        settle(&mut book, &clock, 24_000, 24, second_done);
        let earlier = book.fill(1).expect("the row that ended at second 12");
        let later = book.fill(2).expect("the row that ended at second 24");
        assert_eq!(earlier.cached, 2_467);
        assert_eq!(earlier.prompt_tps, Some(2_224.0), "(4691 - 2467) / 1");
        assert_eq!(later.cached, 296);
        assert_eq!(later.prompt_tps, Some(4_395.0), "(4691 - 296) / 1");
    }

    /// The neighbour's finish is the only one recorded, and the row's
    /// duration would accept it. The row's second is not after one second
    /// before that finish's previous scrape, so it is not that request.
    #[test]
    fn a_row_timestamp_outside_the_finish_scrapes_fills_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        // The first token is already counted, so the finish on the next
        // scrape is not a clean window and is not recorded.
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.5, 1.0), 1.0),
        );
        book.activity(clock.at(50), &[]);
        see(
            &mut book,
            &clock,
            3_000,
            totals(1, 27_990, 9_509, 1_720, (0.967, 1.0), (0.5, 1.0), 0.0),
        );
        see(
            &mut book,
            &clock,
            9_000,
            totals(2, 55_980, 26_485, 3_440, (1.941, 2.0), (1.036, 2.0), 0.0),
        );
        let mut row = measured(1, 27_990, 1_720, 1_047);
        row.timestamp = Some(stamp_secs(0));
        book.activity(clock.at(9_500), &[row]);
        see(
            &mut book,
            &clock,
            10_000,
            totals(2, 55_980, 26_485, 3_440, (1.941, 2.0), (1.036, 2.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "second 0 is outside the later finish");
    }

    #[test]
    fn an_unparseable_timestamp_matches_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let mut row = claim(1, 1_000);
        row.timestamp = Some("not a timestamp".to_owned());
        book.activity(clock.at(1_500), &[row]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
    }

    /// Both rows' end times sit in both finishes. Neither is filled.
    #[test]
    fn two_rows_inside_both_finish_intervals_fill_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            2_000,
            totals(1, 4_691, 2_467, 823, (2.35, 1.0), (1.0, 1.0), 0.0),
        );
        see(
            &mut book,
            &clock,
            4_000,
            totals(2, 9_382, 2_763, 1_646, (4.32, 2.0), (2.0, 2.0), 0.0),
        );
        // Second 2 lies in both `(0 - 1, 2 + 0.5]` and `(2 - 1, 4 + 0.5]`.
        // Second 3 lies only in the second. The first row claims both
        // finishes, so the second row is not the only claim on its finish.
        let mut first = measured(1, 4_691, 823, 2_462);
        first.timestamp = Some(stamp_secs(2));
        let mut second = measured(2, 4_691, 823, 2_062);
        second.timestamp = Some(stamp_secs(3));
        book.activity(clock.at(4_500), &[first, second]);
        see(
            &mut book,
            &clock,
            5_000,
            totals(2, 9_382, 2_763, 1_646, (4.32, 2.0), (2.0, 2.0), 0.0),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
    }

    /// The first row fits two finishes, so neither finish is that row.
    /// The next row also fits the second of those and the finish after it.
    /// Burning the pair would make the later finish look unique and put
    /// its cached count on the next row.
    #[test]
    fn an_ambiguous_row_does_not_hand_the_next_row_a_neighbour() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        // A different prompt, so the rows below do not fit this finish.
        // Its scrape is the one before the first equal-length finish.
        see(
            &mut book,
            &clock,
            2_000,
            totals(1, 9_000, 100, 10, (1.0, 1.0), (0.4, 1.0), 0.0),
        );
        book.activity(clock.at(2_500), &[]);
        see(
            &mut book,
            &clock,
            3_300,
            totals(2, 13_691, 2_567, 833, (3.35, 2.0), (1.4, 2.0), 0.0),
        );
        let mut first = measured(1, 4_691, 823, 2_300);
        first.timestamp = Some(stamp_secs(3));
        book.activity(clock.at(4_000), &[first]);
        // The confirming scrape is also the next finish. Second 3 sits in
        // both intervals, and 2.3 s agrees with 2.35 s and 1.97 s.
        see(
            &mut book,
            &clock,
            4_900,
            totals(3, 18_382, 2_863, 1_656, (5.32, 3.0), (2.4, 3.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "second 3 fits two finishes");
        let mut second = measured(2, 4_691, 823, 1_880);
        second.timestamp = Some(stamp_secs(5));
        book.activity(clock.at(5_500), &[second]);
        see(
            &mut book,
            &clock,
            6_400,
            totals(4, 23_073, 3_453, 2_479, (6.72, 4.0), (3.2, 4.0), 0.0),
        );
        assert_eq!(
            book.fill(2),
            None,
            "1.88 s also fits the finish that cached 590"
        );
    }

    /// Request 0 finishes cleanly. Request 1's own finish shares its scrape
    /// with the next request's first token, so that window is not clean.
    /// The row is request 1: the same prompt and output, a duration the
    /// neighbour accepts, and an end second that fits both scrapes. Nothing
    /// is filled.
    #[test]
    fn a_row_whose_own_unclean_finish_is_not_filled_from_its_neighbour() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            2_000,
            totals(1, 48_306, 42_066, 548, (0.704, 1.0), (0.2, 1.0), 0.0),
        );
        let mut row = measured(1, 48_306, 548, 1_144);
        row.timestamp = Some(stamp_secs(2));
        book.activity(clock.at(2_500), &[row]);
        // One request finished, and two first tokens landed here.
        see(
            &mut book,
            &clock,
            3_000,
            totals(2, 96_612, 42_066, 1_096, (1.848, 2.0), (1.4, 3.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "the row's own finish was seen");
    }

    /// Two finishes land a second apart. The batch scrape's lengths are too
    /// short to be this row, which used to leave the row's own window free.
    /// Isolation does not look at lengths: both finishes sit in the row's
    /// window, so nothing is filled.
    #[test]
    fn a_shorter_batch_finish_does_not_block_the_rows_own_window() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            1_000,
            totals(2, 100, 0, 10, (1.0, 2.0), (0.4, 2.0), 0.0),
        );
        let mut row = measured(1, 10_000, 500, 10_000);
        row.timestamp = Some(stamp_secs(1));
        book.activity(clock.at(1_500), &[row]);
        see(
            &mut book,
            &clock,
            2_000,
            totals(3, 10_100, 1_000, 510, (11.0, 3.0), (2.4, 3.0), 0.0),
        );
        assert_eq!(
            book.fill(1),
            None,
            "both finishes sit inside the row's isolation window"
        );
    }

    /// Two finishes in one scrape, and the row's prompt and output are at
    /// most that scrape's increases. The row's second also fits it, so the
    /// neighbour's clean window is not filled.
    #[test]
    fn a_batch_finish_long_enough_to_hold_the_row_blocks_the_neighbour() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            1_000,
            totals(2, 20_000, 0, 1_000, (1.0, 2.0), (0.4, 2.0), 0.0),
        );
        let mut row = measured(1, 10_000, 500, 10_000);
        row.timestamp = Some(stamp_secs(1));
        book.activity(clock.at(1_500), &[row]);
        see(
            &mut book,
            &clock,
            2_000,
            totals(3, 30_000, 4_000, 1_500, (11.0, 3.0), (2.4, 3.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "the batch scrape could be this row");
    }

    /// A scrape after a gap has no known token increases, so it can be any
    /// row whose timestamp fits, even when the counters moved by less than
    /// the row.
    #[test]
    fn a_finish_after_a_gap_blocks_any_row_in_its_interval() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            11_000,
            totals(1, 10, 0, 1, (0.2, 1.0), (0.1, 1.0), 0.0),
        );
        let mut row = measured(1, 1_000, 401, 10_000);
        row.timestamp = Some(stamp_secs(11));
        book.activity(clock.at(11_500), &[row]);
        see(
            &mut book,
            &clock,
            12_000,
            totals(2, 1_010, 600, 402, (10.2, 2.0), (1.1, 2.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "the gap scrape could be this row");
    }

    /// A restart's counters are not a delta of the process before it. A row
    /// whose second falls in that interval is not filled from a later window.
    #[test]
    fn a_restart_with_finishes_already_counted_blocks_a_row_in_that_interval() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(
            &mut book,
            &clock,
            0,
            totals(4, 9_000, 1_000, 800, (40.0, 4.0), (8.0, 4.0), 0.0),
        );
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            2_000,
            totals(1, 50, 0, 2, (0.3, 1.0), (0.1, 1.0), 0.0),
        );
        let mut row = measured(1, 1_000, 401, 10_000);
        row.timestamp = Some(stamp_secs(2));
        book.activity(clock.at(2_500), &[row]);
        see(
            &mut book,
            &clock,
            3_000,
            totals(2, 1_050, 600, 403, (10.3, 2.0), (1.1, 2.0), 0.0),
        );
        assert_eq!(book.fill(1), None, "the restart could be this row");
    }

    fn obs(
        book: &mut SglangBook,
        clock: &Clock,
        ms: u64,
        before_ms: u64,
        stamp_ms: u64,
        sample: Option<Totals>,
        labeled: bool,
    ) {
        book.observe_scrape(
            "flash",
            clock.at(ms),
            wall_at(before_ms),
            wall_at(stamp_ms),
            sample,
            labeled,
        );
    }

    /// The engine read its counters between the previous scrape's before-time
    /// and this scrape's stamp. A row at second 9 fits `(8, 12]` and does
    /// not fit `(10, 12]`.
    #[test]
    fn the_finish_interval_starts_at_the_previous_before_time() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let mut ms = 0;
        while ms <= 8_000 {
            obs(&mut book, &clock, ms, ms, ms, Some(zeros()), true);
            ms += 2_000;
        }
        book.activity(clock.at(100), &[]);
        obs(
            &mut book,
            &clock,
            10_000,
            8_000,
            10_000,
            Some(zeros()),
            true,
        );
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        obs(&mut book, &clock, 12_000, 10_000, 12_000, Some(done), true);
        book.activity(clock.at(12_500), &[stamp_claim(claim(1, 1_000), 9)]);
        ms = 14_000;
        while ms <= 18_000 {
            obs(&mut book, &clock, ms, ms, ms, Some(done), true);
            ms += 2_000;
        }
        let fill = book.fill(1).expect("the before-time still covers second 9");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, Some(200.0));
    }

    #[test]
    fn a_wall_step_is_a_hole_and_drops_the_queue_chain() {
        let clock = Clock(Instant::now());
        let show = |extra_ms: u64| {
            let mut book = SglangBook::default();
            grid(&mut book, &clock, 0, 8_000, zeros());
            book.activity(clock.at(100), &[]);
            let step = Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                running: 1.0,
                ..zeros()
            };
            see(&mut book, &clock, 10_000, step);
            // Monotonic delta is 2 s. `extra_ms` is how far the wall delta
            // disagrees. Half a second is still the same clock.
            obs(
                &mut book,
                &clock,
                12_000,
                12_000 + extra_ms,
                12_000 + extra_ms,
                Some(step),
                true,
            );
            let shift = extra_ms;
            let mut at = 14_000;
            let fin = 32_000;
            while at < fin {
                obs(
                    &mut book,
                    &clock,
                    at,
                    at + shift,
                    at + shift,
                    Some(step),
                    true,
                );
                at += 2_000;
            }
            let done = Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
            };
            obs(
                &mut book,
                &clock,
                fin,
                fin + shift,
                fin + shift,
                Some(done),
                true,
            );
            let row_sec = ((fin + shift) / 1_000) as i64;
            book.activity(
                clock.at(fin + 500),
                &[stamp_claim(claim(1, 1_000), row_sec)],
            );
            let mut settle_at = fin + 2_000;
            let horizon = (row_sec as u64 + 6) * 1_000;
            loop {
                obs(
                    &mut book,
                    &clock,
                    settle_at,
                    settle_at + shift,
                    settle_at + shift,
                    Some(done),
                    true,
                );
                if settle_at + shift > horizon {
                    break;
                }
                settle_at += 2_000;
            }
            book.fill(1).expect("filled")
        };
        let kept = show(500);
        assert_eq!(kept.prompt_tps, Some(200.0), "0.5 s is not a step");
        let stepped = show(501);
        assert_eq!(stepped.cached, 600);
        assert_eq!(stepped.prompt_tps, None, "the step dropped the chain");
        assert_eq!(stepped.gen_tps, Some(50.0));
    }

    #[test]
    fn a_running_gauge_at_zero_does_not_anchor_when_the_queue_is_ahead() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let ahead = Totals {
            queue_sum: 1.0,
            queue_count: 1.0,
            ..zeros()
        };
        grid(&mut book, &clock, 0, 10_000, ahead);
        assert!(
            !book.models["flash"].queue_anchor,
            "running == 0 is not enough"
        );
        book.activity(clock.at(100), &[]);
        let done = Totals {
            queue_sum: 1.01,
            queue_count: 2.0,
            ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
        };
        see(&mut book, &clock, 12_000, done);
        book.activity(clock.at(12_500), &[stamp_claim(claim(1, 1_000), 12)]);
        settle(&mut book, &clock, 12_000, 12, done);
        let fill = book.fill(1).expect("cached count");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None, "the chain was never anchored");
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    #[test]
    fn reanchor_waits_until_the_counts_agree() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        // Queue count stays behind the finished count, so running == 0
        // does not put the anchor back.
        let first = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        see(&mut book, &clock, 12_000, first);
        book.activity(clock.at(12_500), &[stamp_claim(claim(1, 1_000), 12)]);
        settle(&mut book, &clock, 12_000, 12, first);
        assert!(!book.models["flash"].queue_anchor);
        grid(&mut book, &clock, 22_000, 30_000, first);
        let second = Totals {
            queue_sum: 0.01,
            queue_count: 1.0,
            ..totals(2, 1_800, 800, 502, (13.0, 2.0), (3.0, 2.0), 0.0)
        };
        see(&mut book, &clock, 32_000, second);
        book.activity(
            clock.at(32_500),
            &[stamp_claim(measured(2, 800, 101, 3_000), 32)],
        );
        settle(&mut book, &clock, 32_000, 32, second);
        let fill = book.fill(2).expect("cached count");
        assert_eq!(fill.cached, 200);
        assert_eq!(fill.prompt_tps, None, "no anchor, so 0.01 s is not a step");
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    #[test]
    fn two_queue_steps_while_one_request_runs_drop_the_chain() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        grid(&mut book, &clock, 0, 8_000, zeros());
        book.activity(clock.at(100), &[]);
        see(
            &mut book,
            &clock,
            10_000,
            Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                running: 1.0,
                ..zeros()
            },
        );
        let parked = Totals {
            queue_sum: 0.02,
            queue_count: 2.0,
            running: 1.0,
            ..zeros()
        };
        see(&mut book, &clock, 12_000, parked);
        assert!(!book.models["flash"].queue_anchor, "two steps, one running");
        grid(&mut book, &clock, 14_000, 30_000, parked);
        let done = Totals {
            queue_sum: 0.02,
            queue_count: 2.0,
            ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
        };
        see(&mut book, &clock, 32_000, done);
        book.activity(clock.at(32_500), &[stamp_claim(claim(1, 1_000), 32)]);
        settle(&mut book, &clock, 32_000, 32, done);
        let fill = book.fill(1).expect("cached count");
        assert_eq!(fill.cached, 600);
        assert_eq!(
            fill.prompt_tps, None,
            "the drifted step is not this request"
        );
        assert_eq!(fill.gen_tps, Some(50.0));
    }

    #[test]
    fn a_ten_second_gap_keeps_the_queue_chain_and_a_longer_one_drops_it() {
        let clock = Clock(Instant::now());
        // A pending gap dirties the attribution window on the same scrape,
        // so the drop is not visible as a later fill. `running` stays 1,
        // which keeps the re-anchor from putting the chain back.
        let run = |gap_ms: u64| {
            let mut model = Model {
                queue_anchor: true,
                queue_live: true,
                ..Model::default()
            };
            model.queue_obs.push_back(Some(0.01));
            let prev = Snap::new(
                clock.at(0),
                wall_at(0),
                wall_at(0),
                Totals {
                    queue_sum: 0.01,
                    queue_count: 1.0,
                    running: 1.0,
                    ..zeros()
                },
            );
            let next = Snap::new(
                clock.at(gap_ms),
                wall_at(gap_ms),
                wall_at(gap_ms),
                Totals {
                    queue_sum: 0.01,
                    queue_count: 1.0,
                    running: 1.0,
                    ..zeros()
                },
            );
            let _ = model.queue_advance(&prev, &next);
            (model.queue_anchor, model.queue_take(1))
        };
        let (kept, tie) = run(10_000);
        assert!(kept, "exactly ten seconds still orders the chain");
        assert!(matches!(tie, QueueTie::Known(step) if (step - 0.01).abs() < 1e-9));
        let (dropped, tie) = run(10_001);
        assert!(!dropped, "10.001 s drops the chain");
        assert!(matches!(tie, QueueTie::Unknown));
    }

    #[test]
    fn an_overlap_and_a_metrics_hole_drop_the_queue_chain() {
        let clock = Clock(Instant::now());
        let mut model = Model {
            queue_anchor: true,
            queue_live: true,
            ..Model::default()
        };
        let prev = Snap::new(
            clock.at(0),
            wall_at(0),
            wall_at(0),
            Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                running: 1.0,
                ..zeros()
            },
        );
        model.queue_obs.push_back(Some(0.01));
        let overlapped = Snap::new(
            clock.at(1_000),
            wall_at(1_000),
            wall_at(1_000),
            Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                running: 2.0,
                ..zeros()
            },
        );
        let _ = model.queue_advance(&prev, &overlapped);
        assert!(!model.queue_anchor, "running > 1 drops the chain");

        let mut book = SglangBook::default();
        see(&mut book, &clock, 0, zeros());
        see(
            &mut book,
            &clock,
            2_000,
            Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                ..zeros()
            },
        );
        assert!(book.models["flash"].queue_anchor);
        book.observe("flash", clock.at(4_000), wall_at(4_000), None);
        assert!(
            !book.models["flash"].queue_anchor,
            "a missed scrape drops the chain"
        );
    }

    #[test]
    fn a_zero_queue_count_with_seconds_drops_the_chain() {
        let clock = Clock(Instant::now());
        let mut model = Model {
            queue_anchor: true,
            queue_live: true,
            ..Model::default()
        };
        let base = Snap::new(clock.at(0), wall_at(0), wall_at(0), zeros());
        let step = Snap::new(
            clock.at(1_000),
            wall_at(1_000),
            wall_at(1_000),
            Totals {
                queue_sum: 0.01,
                queue_count: 1.0,
                ..zeros()
            },
        );
        model.queue_push(&base, &step);
        let bogus = Snap::new(
            clock.at(2_000),
            wall_at(2_000),
            wall_at(2_000),
            Totals {
                queue_sum: 0.51,
                queue_count: 1.0,
                ..zeros()
            },
        );
        model.queue_push(&step, &bogus);
        assert!(!model.queue_anchor, "count stayed put while seconds moved");
        assert!(matches!(model.queue_take(1), QueueTie::Unknown));
    }

    #[test]
    fn a_finish_jump_past_the_queue_cap_returns_at_once() {
        let clock = Clock(Instant::now());
        let mut model = Model {
            queue_anchor: true,
            queue_live: true,
            ..Model::default()
        };
        let base = Snap::new(clock.at(0), wall_at(0), wall_at(0), zeros());
        let mut prev = base;
        for i in 1..=2 {
            let snap = Snap::new(
                clock.at(i * 1_000),
                wall_at(i * 1_000),
                wall_at(i * 1_000),
                Totals {
                    queue_sum: i as f64 * 0.01,
                    queue_count: i as f64,
                    ..zeros()
                },
            );
            model.queue_push(&prev, &snap);
            prev = snap;
        }
        assert_eq!(model.queue_obs.len(), 2);
        let tie = model.queue_take(2);
        assert!(matches!(tie, QueueTie::Unknown));
        assert_eq!(model.queue_used, 2, "both steps are consumed");
        assert!(matches!(model.queue_take(1), QueueTie::Unknown));

        model.queue_anchor = true;
        model.queue_obs.push_back(Some(0.01));
        // Without the cap guard this walk is `u64::MAX / 2` steps and the
        // test hangs. A deadline makes that failure return.
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let jumped = model.queue_take(u64::MAX / 2);
            let _ = tx.send((
                matches!(jumped, QueueTie::Unknown),
                !model.queue_anchor,
                model.queue_obs.is_empty(),
            ));
        });
        let (unknown, dropped, empty) = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("a jump past the cap returns at once");
        handle.join().expect("queue_take");
        assert!(unknown, "a jump is not a burst of finishes");
        assert!(dropped, "a jump empties the chain");
        assert!(empty);
    }

    #[test]
    fn the_queue_chain_holds_sixty_four_steps_and_not_sixty_five() {
        let clock = Clock(Instant::now());
        let mut model = Model {
            queue_anchor: true,
            ..Model::default()
        };
        let base = Snap::new(clock.at(0), wall_at(0), wall_at(0), zeros());
        let mut prev = base;
        for i in 1..=64 {
            let snap = Snap::new(
                clock.at(i * 100),
                wall_at(i * 100),
                wall_at(i * 100),
                Totals {
                    queue_sum: i as f64 * 0.01,
                    queue_count: i as f64,
                    ..zeros()
                },
            );
            model.queue_push(&prev, &snap);
            prev = snap;
        }
        assert!(model.queue_anchor);
        assert_eq!(model.queue_obs.len(), 64);
        let extra = Snap::new(
            clock.at(6_500),
            wall_at(6_500),
            wall_at(6_500),
            Totals {
                queue_sum: 0.65,
                queue_count: 65.0,
                ..zeros()
            },
        );
        model.queue_push(&prev, &extra);
        assert!(!model.queue_anchor, "65 steps is more than the chain holds");
        assert!(model.queue_obs.is_empty());
    }

    #[test]
    fn an_unlabeled_time_to_first_token_fills_cached_and_no_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        let mut ms = 0;
        while ms <= 10_000 {
            let labeled = ms != 4_000;
            obs(&mut book, &clock, ms, ms, ms, Some(zeros()), labeled);
            ms += 2_000;
        }
        book.activity(clock.at(100), &[]);
        obs(&mut book, &clock, 12_000, 12_000, 12_000, Some(done), true);
        book.activity(clock.at(12_500), &[stamp_claim(claim(1, 1_000), 12)]);
        ms = 14_000;
        while ms <= 20_000 {
            obs(&mut book, &clock, ms, ms, ms, Some(done), true);
            ms += 2_000;
        }
        let fill = book.fill(1).expect("cached count");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None);
        assert_eq!(fill.gen_tps, None);
        let labeled = "\
sglang:time_to_first_token_seconds_sum{model_name=\"flash\",is_streaming=\"true\"} 2\n\
sglang:time_to_first_token_seconds_count{model_name=\"flash\",is_streaming=\"true\"} 1\n";
        let bare = "\
sglang:time_to_first_token_seconds_sum{model_name=\"flash\"} 2\n\
sglang:time_to_first_token_seconds_count{model_name=\"flash\",is_streaming=\"true\"} 1\n";
        assert!(crate::metrics::sglang_ttft_labeled(labeled));
        assert!(!crate::metrics::sglang_ttft_labeled(bare));
        assert!(!crate::metrics::sglang_ttft_labeled(""));
    }

    #[test]
    fn a_row_without_a_timestamp_or_a_duration_fills_nothing() {
        let clock = Clock(Instant::now());
        let run = |mut bare: Claim| {
            let mut book = SglangBook::default();
            let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
            grid(&mut book, &clock, 0, 10_000, zeros());
            book.activity(clock.at(100), &[]);
            see(&mut book, &clock, 12_000, done);
            bare.seq = 1;
            book.activity(clock.at(12_500), &[bare, stamp_claim(claim(2, 1_000), 12)]);
            settle(&mut book, &clock, 12_000, 12, done);
            assert_eq!(book.fill(1), None);
            book.fill(2).expect("the row that has both").cached
        };
        let mut no_stamp = claim(1, 1_000);
        no_stamp.timestamp = None;
        assert_eq!(run(no_stamp), 600);
        let mut no_duration = claim(1, 1_000);
        no_duration.duration_ms = None;
        no_duration.timestamp = Some(stamp_secs(12));
        assert_eq!(run(no_duration), 600);
    }

    #[test]
    fn a_finish_at_the_activity_baseline_is_not_inside_the_span() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        book.activity(clock.at(100), &[]);
        see(&mut book, &clock, 12_000, done);
        settle(&mut book, &clock, 12_000, 12, done);
        // The next row's span starts at this scrape. The finish is that
        // scrape, and the bound is strict, so it is not inside the span.
        book.activity(clock.at(12_000), &[]);
        book.activity(clock.at(22_000), &[stamp_claim(claim(1, 1_000), 12)]);
        see(&mut book, &clock, 24_000, done);
        assert_eq!(book.fill(1), None, "finish.at == base_at does not match");
    }

    #[test]
    fn a_finish_is_kept_for_thirty_seconds_and_not_thirty_one() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        let done = totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0);
        grid(&mut book, &clock, 0, 10_000, zeros());
        see(&mut book, &clock, 12_000, done);
        assert_eq!(book.models["flash"].finishes.len(), 1);
        see(&mut book, &clock, 42_000, done);
        assert_eq!(
            book.models["flash"].finishes.len(),
            1,
            "thirty seconds is still inside"
        );
        see(&mut book, &clock, 43_001, done);
        assert_eq!(
            book.models["flash"].finishes.len(),
            0,
            "thirty-one seconds has aged out"
        );
    }

    #[test]
    fn sixty_five_finishes_keep_the_newest_sixty_four() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        see(&mut book, &clock, 0, zeros());
        for i in 1..=65 {
            see(
                &mut book,
                &clock,
                i * 100,
                totals(
                    i,
                    i * 1_000,
                    i * 100,
                    i * 10,
                    (i as f64, i as f64),
                    (i as f64 * 0.2, i as f64),
                    0.0,
                ),
            );
        }
        assert_eq!(book.models["flash"].finishes.len(), 64);
        assert_eq!(book.models["flash"].sights.len(), 64);
    }

    fn streaming_done() -> Totals {
        totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
    }

    /// A clean streaming finish at second `row_sec`, covered through the
    /// scrape before it and not yet decided. The window is
    /// `[row_sec - 5, row_sec + 6]`.
    fn open_at(book: &mut SglangBook, clock: &Clock, row_sec: i64) {
        let fin = (row_sec as u64) * 1_000;
        grid(book, clock, 0, fin - 2_000, zeros());
        book.activity(clock.at(100), &[]);
        see(book, clock, fin, streaming_done());
        book.activity(
            clock.at(fin + 100),
            &[stamp_claim(claim(1, 1_000), row_sec)],
        );
    }

    fn decide(book: &mut SglangBook, clock: &Clock, row_sec: i64) {
        settle(
            book,
            clock,
            (row_sec as u64) * 1_000,
            row_sec,
            streaming_done(),
        );
    }

    /// One extra finish sight on the open row at second 40. `n` is the
    /// finished-count step that sight recorded.
    fn with_neighbour(lo_ms: u64, hi_ms: u64, n: Option<u64>) -> Option<Fill> {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        book.models
            .get_mut("flash")
            .unwrap()
            .sights
            .push_back(Sight {
                at: clock.at(hi_ms.saturating_add(3)),
                wall_lo: wall_at(lo_ms),
                wall_hi: wall_at(hi_ms),
                n,
            });
        decide(&mut book, &clock, 40);
        book.fill(1)
    }

    /// The row is second 40, so the window is `[35, 46]`. A second finish
    /// whose interval touches that window blocks the row. One that ends a
    /// millisecond before it, or starts on its end, does not. `n` is 1 for
    /// two separate finishes, 2 for one scrape that stepped twice, and 5
    /// for a longer step.
    #[test]
    fn a_second_finish_just_inside_the_window_blocks_and_just_outside_does_not() {
        for n in [Some(1u64), Some(2), Some(5)] {
            assert_eq!(
                with_neighbour(34_000, 35_000, n).map(|fill| fill.cached),
                None,
                "n = {n:?} ending at T - 5 still overlaps"
            );
            assert_eq!(
                with_neighbour(34_000, 34_999, n).map(|fill| fill.cached),
                Some(600),
                "n = {n:?} ending just before T - 5 does not overlap"
            );
            assert_eq!(
                with_neighbour(45_999, 47_000, n).map(|fill| fill.cached),
                None,
                "n = {n:?} starting just before T + 6 still overlaps"
            );
            assert_eq!(
                with_neighbour(46_000, 47_000, n).map(|fill| fill.cached),
                Some(600),
                "n = {n:?} starting at T + 6 does not overlap"
            );
        }
    }

    /// The neighbour is a real second request, shorter than the row, so the
    /// length checks would still accept the row's own finish.
    #[test]
    fn a_shorter_second_request_inside_the_window_blocks_and_one_past_the_end_does_not() {
        let run = |prev_ms: u64, fin_ms: u64| {
            let clock = Clock(Instant::now());
            let mut book = SglangBook::default();
            let own = streaming_done();
            let short = totals(2, 1_010, 600, 403, (11.0, 2.0), (2.2, 2.0), 0.0);
            grid(&mut book, &clock, 0, 38_000, zeros());
            book.activity(clock.at(100), &[]);
            see(&mut book, &clock, 40_000, own);
            book.activity(clock.at(40_100), &[stamp_claim(claim(1, 1_000), 40)]);
            let mut ms = 42_000u64;
            while ms < prev_ms {
                see(&mut book, &clock, ms, own);
                ms += 2_000;
            }
            see(&mut book, &clock, prev_ms, own);
            see(&mut book, &clock, fin_ms, short);
            if fin_ms <= 46_000 {
                settle(&mut book, &clock, fin_ms, 40, short);
            }
            book.fill(1).map(|fill| fill.cached)
        };
        assert_eq!(
            run(45_999, 47_000),
            None,
            "the shorter finish starts just before T + 6"
        );
        assert_eq!(
            run(46_000, 48_000),
            Some(600),
            "the shorter finish starts at T + 6"
        );
    }

    /// `hi == T - 5` overlaps, and this sight has no count. Coverage is
    /// intact, so skipping the sight would fill the row.
    #[test]
    fn a_sight_with_no_count_blocks_even_when_the_span_is_covered() {
        assert_eq!(
            with_neighbour(34_000, 35_000, None).map(|fill| fill.cached),
            None
        );
    }

    /// The finish's own sight is gone. Another sight in the window has
    /// `n == 1` but a different instant, which is what the cap does when
    /// later unclean steps push the own sight out. The count is one and
    /// it is not this finish.
    #[test]
    fn a_finish_whose_own_sight_is_missing_stays_empty() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        let model = book.models.get_mut("flash").unwrap();
        let finish_at = model.finishes[0].at;
        model.sights.clear();
        model.sights.push_back(Sight {
            at: clock.at(39_000),
            wall_lo: wall_at(38_000),
            wall_hi: wall_at(40_000),
            n: Some(1),
        });
        assert_ne!(model.sights[0].at, finish_at);
        decide(&mut book, &clock, 40);
        assert_eq!(book.fill(1), None);
    }

    #[test]
    fn a_sight_with_no_wall_time_blocks_the_row() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        let before = SystemTime::UNIX_EPOCH
            .checked_sub(Duration::from_secs(1))
            .expect("before the epoch");
        book.models
            .get_mut("flash")
            .unwrap()
            .sights
            .push_back(Sight {
                at: clock.at(39_000),
                wall_lo: before,
                wall_hi: before,
                n: Some(1),
            });
        decide(&mut book, &clock, 40);
        assert_eq!(book.fill(1), None, "a stamp with no wall time blocks");
    }

    #[test]
    fn a_probe_with_no_wall_time_blocks_the_row() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        let before = SystemTime::UNIX_EPOCH
            .checked_sub(Duration::from_secs(1))
            .expect("before the epoch");
        book.models
            .get_mut("flash")
            .unwrap()
            .probes
            .push_back(Probe {
                at: clock.at(40_000),
                stamp: before,
                gap: false,
            });
        decide(&mut book, &clock, 40);
        assert_eq!(book.fill(1), None, "a probe with no wall time blocks");
    }

    /// Isolation decides on a scrape strictly after `T + 6`, and the span
    /// is covered only by a scrape strictly after that too. A scrape sitting
    /// on `T + 6` does neither.
    #[test]
    fn a_scrape_stamped_at_t_plus_6_does_not_decide_the_row() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        let done = streaming_done();
        see(&mut book, &clock, 42_000, done);
        see(&mut book, &clock, 44_000, done);
        see(&mut book, &clock, 46_000, done);
        assert_eq!(book.fill(1), None, "T + 6 is not past the window");
        see(&mut book, &clock, 48_000, done);
        assert_eq!(book.fill(1).map(|fill| fill.cached), Some(600));
    }

    /// The scrape on `T + 6` would cover the span if the bound were `>=`.
    /// The next probe is a gap past the window, so it decides the row and
    /// then drops the coverage it would have closed.
    #[test]
    fn a_scrape_on_the_window_end_does_not_cover_when_the_next_probe_is_a_gap() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        let done = streaming_done();
        see(&mut book, &clock, 42_000, done);
        see(&mut book, &clock, 44_000, done);
        see(&mut book, &clock, 46_000, done);
        see(&mut book, &clock, 49_001, done);
        assert_eq!(book.fill(1), None);
    }

    /// The only probe at or before `T - 5` is the watcher, stamped on that
    /// edge or one millisecond inside it.
    #[test]
    fn a_break_that_ends_on_the_window_start_can_cover_and_one_just_inside_cannot() {
        let run = |start_ms: u64| {
            let clock = Clock(Instant::now());
            let mut book = SglangBook::default();
            let done = streaming_done();
            let mut ms = start_ms;
            obs(&mut book, &clock, ms, ms, ms, Some(zeros()), true);
            book.activity(clock.at(ms + 1), &[]);
            ms += 1_000;
            while ms < 40_000 {
                obs(&mut book, &clock, ms, ms, ms, Some(zeros()), true);
                ms += 1_000;
            }
            obs(&mut book, &clock, 40_000, 40_000, 40_000, Some(done), true);
            book.activity(clock.at(40_050), &[stamp_claim(claim(1, 1_000), 40)]);
            settle(&mut book, &clock, 40_000, 40, done);
            book.fill(1).map(|fill| fill.cached)
        };
        assert_eq!(run(35_000), Some(600), "a break ending at T - 5 can cover");
        assert_eq!(run(35_001), None, "a break ending just inside cannot");
    }

    /// The watcher's gap probe is older than [`FINISH_FOR`] by the time a
    /// row at second 40 is decided. Coverage then starts from a later
    /// scrape that is not a gap. A non-gap stamped exactly on `T - 5`
    /// starts it, and one a millisecond inside does not.
    #[test]
    fn a_later_scrape_starts_coverage_once_the_watcher_probe_has_aged_out() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        open_at(&mut book, &clock, 40);
        decide(&mut book, &clock, 40);
        let probes = &book.models["flash"].probes;
        assert!(
            probes.iter().all(|probe| probe.at >= clock.at(18_000)),
            "the watcher probe has aged out"
        );
        assert_eq!(book.fill(1).map(|fill| fill.cached), Some(600));

        let covered = |stamp_ms: u64| {
            let mut model = Model::default();
            model.probes.push_back(Probe {
                at: clock.at(stamp_ms),
                stamp: wall_at(stamp_ms),
                gap: false,
            });
            model.probes.push_back(Probe {
                at: clock.at(50_000),
                stamp: wall_at(50_000),
                gap: false,
            });
            span_covered(&model, 35.0, 46.0)
        };
        assert!(covered(35_000), "a non-gap scrape on T - 5 starts coverage");
        assert!(
            !covered(35_001),
            "a non-gap scrape just inside T - 5 does not"
        );
    }

    /// Three seconds is not a hole. A millisecond over is, inside the
    /// window and on each edge, on the monotonic clock and the wall clock.
    /// The row is second 40, so the window is `[35, 46]`. The scrape before
    /// the pair sits at or before 35 s, and the row's own finish is later.
    #[test]
    fn a_coverage_gap_of_three_seconds_is_covered_and_a_millisecond_over_is_not() {
        let gap_between = |earlier: u64, later: u64| {
            let clock = Clock(Instant::now());
            let mut book = SglangBook::default();
            let done = streaming_done();
            obs(&mut book, &clock, 0, 0, 0, Some(zeros()), true);
            obs(
                &mut book,
                &clock,
                earlier,
                earlier,
                earlier,
                Some(zeros()),
                true,
            );
            book.activity(clock.at(earlier + 1), &[]);
            obs(&mut book, &clock, later, later, later, Some(zeros()), true);
            let mut ms = later + 2_000;
            while ms < 40_000 {
                obs(&mut book, &clock, ms, ms, ms, Some(zeros()), true);
                ms += 2_000;
            }
            obs(&mut book, &clock, 40_000, 40_000, 40_000, Some(done), true);
            book.activity(clock.at(40_050), &[stamp_claim(claim(1, 1_000), 40)]);
            settle(&mut book, &clock, 40_000, 40, done);
            book.fill(1).map(|fill| fill.cached)
        };
        // Just inside the low end: the later scrape is 1 ms after T - 5.
        assert_eq!(
            gap_between(31_999, 35_000),
            Some(600),
            "3.001 s ending on T - 5 still starts coverage"
        );
        assert_eq!(
            gap_between(32_001, 35_001),
            Some(600),
            "exactly 3 s just inside the window is covered"
        );
        assert_eq!(
            gap_between(32_000, 35_001),
            None,
            "3.001 s just inside the window is a hole"
        );
        assert_eq!(
            gap_between(34_000, 37_000),
            Some(600),
            "exactly 3 s inside the window is covered"
        );
        assert_eq!(
            gap_between(33_999, 37_000),
            None,
            "3.001 s inside the window is a hole"
        );

        // On the high end the finish is already recorded and the spacing
        // is two later idle scrapes, so the row's own match stays put.
        let edge = |gap_ms: u64| {
            let clock = Clock(Instant::now());
            let mut book = SglangBook::default();
            open_at(&mut book, &clock, 40);
            let done = streaming_done();
            let end = 46_000u64;
            see(&mut book, &clock, 42_000, done);
            see(&mut book, &clock, end - gap_ms, done);
            see(&mut book, &clock, end, done);
            see(&mut book, &clock, end + 2_000, done);
            book.fill(1).map(|fill| fill.cached)
        };
        assert_eq!(
            edge(3_000),
            Some(600),
            "exactly 3 s ending at T + 6 is covered"
        );
        assert_eq!(edge(3_001), None, "3.001 s ending at T + 6 is a hole");

        // Wall delta alone. Monotonic is 2.6 s, inside both the coverage
        // gap and the clock-step bound. Wall is exactly 3 s, or 1 ms over.
        let wall = |wall_ms: u64| {
            let clock = Clock(Instant::now());
            let mut book = SglangBook::default();
            open_at(&mut book, &clock, 40);
            let done = streaming_done();
            obs(&mut book, &clock, 42_600, 42_600, wall_ms, Some(done), true);
            see(&mut book, &clock, 44_600, done);
            see(&mut book, &clock, 46_600, done);
            see(&mut book, &clock, 48_600, done);
            book.fill(1).map(|fill| fill.cached)
        };
        assert_eq!(
            wall(43_000),
            Some(600),
            "a wall delta of exactly 3 s is covered"
        );
        assert_eq!(wall(43_001), None, "a wall delta of 3.001 s is a hole");
    }

    #[test]
    fn a_duration_past_i64_does_not_overflow() {
        assert!(duration_agrees(10_000, 10.0));
        assert!(!duration_agrees(i64::MAX as u64, 10.0));
        assert!(!duration_agrees((i64::MAX as u64).saturating_add(1), 10.0));
        assert!(!duration_agrees(u64::MAX, 10.0));
    }
}
