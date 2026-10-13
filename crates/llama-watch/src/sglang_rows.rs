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
//! for longer than [`crate::speeds::PENDING_FOR`]. The llama-swap row is
//! that request only when its prompt, output and duration match the
//! window (no more than 0.1 s shorter than the engine's end-to-end time,
//! and no more than the larger of 0.5 s and two percent longer) and its
//! timestamp is the end time truncated to the second, so the row ended in
//! `[T, T+1)`. It fits a finish seen in `(t_prev, t_fin]` when
//! `t_prev - 1 s < T <= t_fin + 0.5 s`: llama-swap's end is the engine's
//! end or at most half a second later. The match has to be unique in both
//! directions. No match, or more than one, fills nothing.
//!
//! A non-streaming finish (`is_streaming="false"` on `num_requests_total`)
//! fills the cached count only: its time to first token is the whole
//! request, so neither rate is a measurement. The prompt rate also stays
//! unset unless that request's own queue time is known — the k-th finish
//! after an idle scrape owns the k-th queue-histogram step after it, and
//! the step was a single +1 — and that wait is at most five percent of its
//! time to the first token. The cached count and the generation rate stay.
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
/// Untaken finishes older than this cannot meet a row.
const FINISH_FOR: Duration = Duration::from_secs(30);
/// Queue steps remembered since the idle anchor. Past this the order is
/// no longer trusted.
const QUEUE_CAP: usize = 64;

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
    /// Wall clock of this scrape. Compared with a row's timestamp only.
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
    fn new(at: Instant, wall: SystemTime, totals: Totals) -> Self {
        Self {
            at,
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
    /// Wall clock of the scrape before this finish, and of the finish scrape.
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

#[derive(Debug, Default)]
struct Model {
    base: Option<Snap>,
    /// Scrapes since [`Self::base`] while a window is open.
    since: Vec<Snap>,
    /// A hole, a gap, an overlap or a counter that moved out of turn.
    dirty: bool,
    history: VecDeque<Snap>,
    finishes: VecDeque<Finish>,
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

#[derive(Debug)]
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
    /// One metrics scrape of `model`. `None` is a hole: the open window
    /// cannot be attributed. The first complete scrape is only a baseline.
    ///
    /// `wall` is the host clock when the scrape was taken. It is compared
    /// with a row's timestamp and is not used to order scrapes.
    pub fn observe(&mut self, model: &str, at: Instant, wall: SystemTime, totals: Option<Totals>) {
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
            }
            self.resolve(at);
            return;
        };
        self.models
            .entry(model.to_owned())
            .or_default()
            .note(Snap::new(at, wall, totals));
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
            let unique =
                hits.len() == 1 && claims.get(&(wait.model.clone(), hits[0])).copied() == Some(1);
            if unique
                && let Some(finish) = self
                    .models
                    .get(&wait.model)
                    .and_then(|model| model.finishes.get(hits[0]).filter(|finish| !finish.taken))
            {
                // A reported value is never stored over the row. A missing
                // cached count is; so is a rate the row did not report.
                stored.push((
                    wait.seq,
                    Fill {
                        cached: finish.cached,
                        prompt_tps: if wait.prompt_known {
                            None
                        } else {
                            finish.prompt_tps
                        },
                        gen_tps: if wait.gen_known { None } else { finish.gen_tps },
                    },
                ));
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
            return;
        };
        let prev = self.since.last().copied().unwrap_or(base);
        if snap.at <= prev.at {
            return;
        }
        if went_back(&prev, &snap) {
            self.restart(snap);
            return;
        }
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
        if batch > 0 {
            // Non-streaming: time to first token is the whole request.
            prompt_tps = None;
            gen_tps = None;
        } else if hide_prompt(queue, ttft_s) {
            prompt_tps = None;
        }
        self.finishes.push_back(Finish {
            at: snap.at,
            wall_lo: prev.wall,
            wall_hi: snap.wall,
            prompt,
            generated,
            cached,
            e2e_s,
            prompt_tps,
            gen_tps,
            taken: false,
        });
        while self.finishes.len() > 64 {
            self.finishes.pop_front();
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

    fn restart(&mut self, snap: Snap) {
        self.finishes.clear();
        self.since.clear();
        self.dirty = snap.running > 1.0;
        self.base = Some(snap);
        self.remember(snap);
        self.queue_baseline(&snap);
    }

    /// The first scrape of a process. Steps before it were not seen.
    fn queue_baseline(&mut self, snap: &Snap) {
        self.queue_drop();
        self.queue_live = queue_nonzero(snap);
        self.queue_anchor = snap.running <= 0.0;
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
            self.queue_anchor = true;
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

/// `duration_ms` agrees with the window's end-to-end seconds.
/// llama-swap's duration covers the engine's time plus its own overhead,
/// so it is never meaningfully shorter: at least 0.1 s under, and at most
/// the larger of 0.5 s and two percent over (36_468 ms against 36.47 s).
fn duration_agrees(duration_ms: u64, e2e_s: f64) -> bool {
    if !e2e_s.is_finite() || e2e_s < 0.0 {
        return false;
    }
    let row_s = duration_ms as f64 / 1000.0;
    let delta = row_s - e2e_s;
    let over = (e2e_s * 0.02).max(0.5);
    delta >= -0.1 && delta <= over
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
            // Existing windows are stamped at the Unix epoch, and so is
            // this row: the end-time check then accepts them. A test of
            // that check sets its own timestamp.
            timestamp: Some("1970-01-01T00:00:00Z".to_owned()),
        }
    }

    /// Baseline, then one request: TTFT two scrapes before the finish.
    /// Prompt 1,000, cached 600, TTFT 2 s, 401 generated, e2e 10 s.
    fn one(book: &mut SglangBook, clock: &Clock, seq: u64) -> Fill {
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(4_500), &[claim(seq, 1_000)]);
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
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

        // Second request, still alone: 800 prompt, 200 cached, TTFT 1 s,
        // 101 generated, e2e 3 s.
        book.observe(
            "flash",
            clock.at(6_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(7_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (3.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(8_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_800, 800, 502, (13.0, 2.0), (3.0, 2.0), 0.0)),
        );
        book.activity(clock.at(8_500), &[measured(2, 800, 101, 3_000)]);
        book.observe(
            "flash",
            clock.at(9_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_800, 800, 502, (13.0, 2.0), (3.0, 2.0), 0.0)),
        );
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
            Some(totals(1, 500, 500, 11, (4.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[measured(1, 500, 11, 4_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 500, 500, 11, (4.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let fill = book.fill(1).expect("filled");
        assert_eq!(fill.cached, 500);
        assert_eq!(fill.prompt_tps, None, "nothing new to prefill");
        assert_eq!(fill.gen_tps, Some(5.0), "(11 - 1) / (4 - 2)");
    }

    #[test]
    fn two_finishes_in_one_scrape_fill_nothing_and_the_next_request_does() {
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
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000), claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)),
        );
        book.activity(clock.at(4_500), &[claim(3, 1_000)]);
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)),
        );
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        assert_eq!(
            book.fill(1),
            None,
            "the reset is not a measurement of that row"
        );
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(5_500), &[claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(6_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let fill = book.fill(2).expect("after the restart");
        assert_eq!(
            (fill.cached, fill.prompt_tps, fill.gen_tps),
            (600, Some(200.0), Some(50.0))
        );
    }

    #[test]
    fn a_gap_longer_than_ten_seconds_fills_nothing_and_an_idle_gap_does_not_stick() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        let late = clock.at(0) + Duration::from_secs(10) + Duration::from_millis(1);
        book.observe(
            "flash",
            late,
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(late + Duration::from_millis(100), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            late + Duration::from_millis(200),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);

        // Exactly ten seconds is still inside the bound.
        let mut book = SglangBook::default();
        let start = clock.at(0);
        book.observe(
            "flash",
            start,
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(start + Duration::from_millis(50), &[]);
        let edge = start + Duration::from_secs(10);
        book.observe(
            "flash",
            edge,
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(edge + Duration::from_millis(50), &[claim(7, 1_000)]);
        book.observe(
            "flash",
            edge + Duration::from_millis(100),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(7).expect("ten seconds").prompt_tps, Some(200.0));

        // An idle gap rebases. The request after it is filled.
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        let idle = clock.at(0) + Duration::from_secs(11);
        book.observe(
            "flash",
            idle,
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.observe(
            "flash",
            idle + Duration::from_millis(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(idle + Duration::from_millis(1_100), &[claim(8, 1_000)]);
        book.observe(
            "flash",
            idle + Duration::from_millis(1_200),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let fill = book.fill(8).expect("after the idle gap");
        assert_eq!(fill.cached, 0, "a measured zero is a number");
        assert_eq!(fill.prompt_tps, Some(500.0));
    }

    #[test]
    fn no_row_and_two_rows_with_the_same_input_fill_nothing() {
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
        // The unmatched row and the row it did match share this read, so
        // the miss must not take the finish away from the match.
        book.activity(clock.at(1_500), &[claim(1, 999), claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2).expect("the one match").cached, 600);

        // Two rows and one finish: neither is filled.
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
        book.activity(clock.at(1_500), &[claim(3, 1_000), claim(4, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(3), None);
        assert_eq!(book.fill(4), None);
    }

    #[test]
    fn two_rows_in_one_span_each_get_their_own_finish() {
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
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_500, 1_100, 402, (12.0, 2.0), (3.0, 2.0), 0.0)),
        );
        book.activity(
            clock.at(2_500),
            &[claim(1, 1_000), measured(2, 500, 1, 2_000)],
        );
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(2, 1_500, 1_100, 402, (12.0, 2.0), (3.0, 2.0), 0.0)),
        );
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
        let mut row = claim(1, 1_000);
        row.prompt_known = true;
        book.activity(clock.at(1_500), &[row]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let fill = book.fill(1).expect("the missing cached count is filled");
        assert_eq!(fill.cached, 600);
        assert_eq!(fill.prompt_tps, None, "a reported rate is not replaced");
        assert_eq!(fill.gen_tps, Some(50.0), "a rate the row did not report");
        book.activity(clock.at(2_500), &[claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(2), None, "the row already took the finish");
    }

    #[test]
    fn one_generated_token_and_a_fully_cached_prompt_omit_the_rate() {
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
            Some(totals(1, 1_000, 400, 1, (5.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[measured(1, 1_000, 1, 5_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 400, 1, (5.0, 1.0), (2.0, 1.0), 0.0)),
        );
        let fill = book.fill(1).expect("filled");
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        // 1_000 new tokens in 0.0005 s is 2_000_000 tok/s, past 1_000_000.
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (0.0005, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(4, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (0.0005, 1.0), 0.0)),
        );
        let fill = book.fill(4).expect("the count is still a measurement");
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        // The false series' time to first token is the whole request,
        // 5e-6 s short of the end-to-end sum.
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (10.0 - 5e-6, 1.0), 0.0)
            }),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (10.0 - 5e-6, 1.0), 0.0)
            }),
        );
        let fill = book.fill(1).expect("cached count");
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
        book.observe(
            "flash",
            clock.at(6_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
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
            }),
        );
        book.activity(clock.at(6_500), &[measured(2, 800, 101, 3_000)]);
        book.observe(
            "flash",
            clock.at(7_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
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
            }),
        );
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        // One streaming finish and one non-streaming finish, one scrape.
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)
            }),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000), claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)
            }),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 1.0)
            }),
        );
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)
            }),
        );
        book.activity(clock.at(4_500), &[claim(3, 1_000)]);
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                batch: 1,
                ..totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)
            }),
        );
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        book.activity(clock.at(80), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1).expect("the first row").cached, 600);
        book.activity(clock.at(1_100), &[claim(2, 1_000)]);
        book.observe(
            "flash",
            clock.at(1_200),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(2), None, "one finish fills one row");
    }

    #[test]
    fn output_and_duration_must_match_the_window() {
        let clock = Clock(Instant::now());
        let mut different_output = SglangBook::default();
        different_output.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        different_output.activity(clock.at(50), &[]);
        different_output.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        different_output.activity(clock.at(1_500), &[measured(1, 1_000, 400, 10_000)]);
        different_output.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(
            different_output.fill(1),
            None,
            "equal input, different output"
        );

        let mut far = SglangBook::default();
        far.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        far.activity(clock.at(50), &[]);
        far.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        far.activity(clock.at(1_500), &[measured(1, 1_000, 401, 20_000)]);
        far.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(far.fill(1), None, "duration 20 s against 10 s");

        // One second over does not agree. Two percent of 10 s is 0.2 s,
        // so the upper bound is 0.5 s.
        let mut edge = SglangBook::default();
        edge.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        edge.activity(clock.at(50), &[]);
        edge.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        edge.activity(clock.at(1_500), &[measured(1, 1_000, 401, 11_000)]);
        edge.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(edge.fill(1), None, "one second over 10 s");

        // 36_468 ms against 36.47 s agrees; two seconds past it does not.
        // The row is read before the scrape that shows the finish.
        let mut before = SglangBook::default();
        before.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        before.activity(clock.at(50), &[]);
        before.activity(clock.at(80), &[measured(1, 1_000, 401, 36_468)]);
        before.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (36.47, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(before.fill(1).expect("read before the finish").cached, 600);
        assert_eq!(before.fill(1).expect("rate").prompt_tps, Some(200.0));

        let mut past = SglangBook::default();
        past.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        past.activity(clock.at(50), &[]);
        past.activity(clock.at(80), &[measured(1, 1_000, 401, 38_470)]);
        past.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (36.47, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(past.fill(1), None, "36.47 s plus 2 s is past two percent");

        // The same finish, read after the scrape that showed it.
        let mut after = SglangBook::default();
        after.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        after.activity(clock.at(50), &[]);
        after.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (36.47, 1.0), (2.0, 1.0), 0.0)),
        );
        after.activity(clock.at(1_500), &[measured(1, 1_000, 401, 36_468)]);
        after.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(1, 1_000, 600, 401, (36.47, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(after.fill(1).expect("read after the finish").cached, 600);
    }

    /// llama-swap's duration is the engine's end-to-end time plus its own
    /// overhead: no more than 0.1 s shorter, and no more than the larger
    /// of 0.5 s and two percent longer.
    #[test]
    fn duration_slack_is_one_sided() {
        let clock = Clock(Instant::now());
        let run = |e2e: f64, duration_ms: u64| {
            let mut book = SglangBook::default();
            book.observe(
                "flash",
                clock.at(0),
                SystemTime::UNIX_EPOCH,
                Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
            );
            book.activity(clock.at(50), &[]);
            book.activity(clock.at(80), &[measured(1, 1_000, 401, duration_ms)]);
            book.observe(
                "flash",
                clock.at(1_000),
                SystemTime::UNIX_EPOCH,
                Some(totals(1, 1_000, 600, 401, (e2e, 1.0), (1.0, 1.0), 0.0)),
            );
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
            book.observe(
                "flash",
                clock.at(0),
                SystemTime::UNIX_EPOCH,
                Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
            );
            book.activity(clock.at(50), &[]);
            let finish = Totals {
                queue_sum,
                queue_count: 1.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
            };
            book.observe(
                "flash",
                clock.at(1_000),
                SystemTime::UNIX_EPOCH,
                Some(finish),
            );
            book.activity(clock.at(1_500), &[claim(1, 1_000)]);
            book.observe(
                "flash",
                clock.at(2_000),
                SystemTime::UNIX_EPOCH,
                Some(finish),
            );
            book.fill(1).expect("filled")
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
        book.observe(
            "flash",
            clock.at(9_000),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(9_100), &[claim(2, 0)]);
        book.observe(
            "flash",
            clock.at(9_200),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
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
        book.observe(
            "flash",
            clock.at(0),
            SystemTime::UNIX_EPOCH,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(50), &[]);
        // The running request left the queue after 0.02 s.
        book.observe(
            "flash",
            clock.at(1_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 0.02,
                queue_count: 1.0,
                running: 1.0,
                ..totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)
            }),
        );
        // It finishes, and the request that was waiting has already left
        // the queue. 3 s of that wait is not this finish's.
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 3.02,
                queue_count: 2.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 1.0)
            }),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(3_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 3.02,
                queue_count: 2.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 1.0)
            }),
        );
        let first = book.fill(1).expect("the request that barely waited");
        assert_eq!(first.cached, 600);
        assert_eq!(
            first.prompt_tps,
            Some(200.0),
            "its own queue time is 0.02 s"
        );
        assert_eq!(first.gen_tps, Some(50.0));

        // The waiter finishes. Nothing new was added to the histogram, so
        // a window delta of zero is not its queue time. TTFT is 4 s and
        // 3 s of it was waiting, which is more than five percent.
        book.observe(
            "flash",
            clock.at(4_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 3.02,
                queue_count: 2.0,
                ..totals(2, 1_500, 700, 602, (18.0, 2.0), (6.0, 2.0), 0.0)
            }),
        );
        book.activity(clock.at(4_500), &[measured(2, 500, 201, 8_000)]);
        book.observe(
            "flash",
            clock.at(5_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 3.02,
                queue_count: 2.0,
                ..totals(2, 1_500, 700, 602, (18.0, 2.0), (6.0, 2.0), 0.0)
            }),
        );
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
            Some(Totals {
                queue_sum: 0.01,
                queue_count: 2.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
            }),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000)]);
        book.observe(
            "flash",
            clock.at(2_000),
            SystemTime::UNIX_EPOCH,
            Some(Totals {
                queue_sum: 0.01,
                queue_count: 2.0,
                ..totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)
            }),
        );
        let fill = book.fill(1).expect("the count and the generation rate");
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
            see(
                &mut book,
                &clock,
                prev_ms,
                totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0),
            );
            book.activity(clock.at(prev_ms + 1), &[]);
            let mut row = measured(1, 1_000, 401, 10_000);
            row.timestamp = Some(stamp_secs(row_secs));
            book.activity(clock.at(prev_ms + 2), &[row]);
            see(
                &mut book,
                &clock,
                fin_ms,
                totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0),
            );
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

    /// Two finishes with the same prompt, output and a duration either row
    /// would accept. Each row's own end time picks one finish.
    #[test]
    fn equal_prompt_and_output_rows_match_on_their_end_time() {
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
        // Idle again, so the next finish's previous scrape is this one.
        see(
            &mut book,
            &clock,
            8_000,
            totals(1, 4_691, 2_467, 823, (2.35, 1.0), (1.0, 1.0), 0.0),
        );
        see(
            &mut book,
            &clock,
            9_000,
            totals(2, 9_382, 2_763, 1_646, (4.32, 2.0), (2.0, 2.0), 0.0),
        );
        let mut first = measured(1, 4_691, 823, 2_462);
        first.timestamp = Some(stamp_secs(2));
        let mut second = measured(2, 4_691, 823, 2_062);
        second.timestamp = Some(stamp_secs(9));
        book.activity(clock.at(9_500), &[first, second]);
        see(
            &mut book,
            &clock,
            10_000,
            totals(2, 9_382, 2_763, 1_646, (4.32, 2.0), (2.0, 2.0), 0.0),
        );
        let earlier = book.fill(1).expect("the row that ended at second 2");
        let later = book.fill(2).expect("the row that ended at second 9");
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
}
