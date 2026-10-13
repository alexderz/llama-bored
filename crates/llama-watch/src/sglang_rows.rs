//! One SGLang request's cached tokens and speeds, from the engine's counters (#133).
//!
//! SGLang has no per-request prefill or decode histogram. Its counters step
//! when a request finishes (`num_requests_total`, `prompt_tokens_total`,
//! `generation_tokens_total`, `cached_tokens_total`, the end-to-end latency
//! histogram), and time-to-first-token steps when the first token is
//! produced, which can be several scrapes earlier. `num_running_reqs` is
//! how many requests the engine is running.
//!
//! A window runs from one finish to the next. It is clean when, across
//! every scrape in it, exactly one request finished, exactly one
//! first-token observation happened, nothing ran beside it, no counter
//! went backwards, and no scrape was missed for longer than
//! [`crate::speeds::PENDING_FOR`]. The matched llama-swap row is the one
//! of that model whose `input_tokens` equals the prompt increase and whose
//! time falls in the same span [`crate::speeds::SpeedBook`] uses. No
//! match, or more than one, fills nothing. A value llama-swap already
//! reported is never replaced.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};

use llama_core::backend::MAX_ENGINE_TPS;

use crate::metrics::MetricsSample;
use crate::speeds::{MAX_MODELS, MAX_PENDING, MAX_ROWS, PENDING_FOR};

/// Scrapes kept per model. Past this the open window is dropped.
const HISTORY_CAP: usize = 512;
/// Untaken finishes older than this cannot meet a row.
const FINISH_FOR: Duration = Duration::from_secs(30);

/// The counters one SGLang `/metrics` scrape contributes. `None` from
/// [`Totals::from_sample`] when any of them is missing: that scrape is a
/// hole and the open window is not clean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Totals {
    /// `sglang:num_requests_total`, both streaming series summed.
    pub requests: u64,
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
    /// `sglang:num_running_reqs`.
    pub running: f64,
}

impl Totals {
    /// The counters a scrape must have before it can attribute a row.
    #[must_use]
    pub fn from_sample(sample: &MetricsSample) -> Option<Self> {
        let e2e = sample.e2e?;
        let ttft = sample.ttft?;
        let running = sample.requests_processing?;
        if ![e2e.sum, e2e.count, ttft.sum, ttft.count, running]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        Some(Self {
            requests: sample.requests_total?,
            prompt: sample.prompt_total?,
            cached: sample.cached_total?,
            generated: sample.n_decode_total?,
            e2e_sum: e2e.sum,
            e2e_count: e2e.count,
            ttft_sum: ttft.sum,
            ttft_count: ttft.count,
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
    /// llama-swap already reported a cached count or a rate. The window
    /// is consumed and nothing is stored.
    pub reported: bool,
}

#[derive(Clone, Copy, Debug)]
struct Snap {
    at: Instant,
    requests: u64,
    prompt: u64,
    cached: u64,
    generated: u64,
    e2e_sum: f64,
    e2e_count: f64,
    ttft_sum: f64,
    ttft_count: f64,
    running: f64,
}

impl Snap {
    fn new(at: Instant, totals: Totals) -> Self {
        Self {
            at,
            requests: totals.requests,
            prompt: totals.prompt,
            cached: totals.cached,
            generated: totals.generated,
            e2e_sum: totals.e2e_sum,
            e2e_count: totals.e2e_count,
            ttft_sum: totals.ttft_sum,
            ttft_count: totals.ttft_count,
            running: totals.running,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Finish {
    at: Instant,
    prompt: u64,
    cached: u64,
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
}

#[derive(Debug)]
struct Pending {
    seq: u64,
    model: String,
    input: u64,
    reported: bool,
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
    pub fn observe(&mut self, model: &str, at: Instant, totals: Option<Totals>) {
        if !self.models.contains_key(model) && self.models.len() >= MAX_MODELS {
            self.resolve(at);
            return;
        }
        let Some(totals) = totals else {
            if let Some(state) = self.models.get_mut(model) {
                state.dirty = true;
            }
            self.resolve(at);
            return;
        };
        self.models
            .entry(model.to_owned())
            .or_default()
            .note(Snap::new(at, totals));
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
                        reported: row.reported,
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
                    !finish.taken
                        && finish.at > *base_at
                        && finish.at <= *end_at
                        && finish.prompt == wait.input
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
                && !wait.reported
                && let Some(finish) = self
                    .models
                    .get(&wait.model)
                    .and_then(|model| model.finishes.get(hits[0]))
            {
                stored.push((
                    wait.seq,
                    Fill {
                        cached: finish.cached,
                        prompt_tps: finish.prompt_tps,
                        gen_tps: finish.gen_tps,
                    },
                ));
            }
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
            model.finishes.retain(|finish| {
                !finish.taken && now.saturating_duration_since(finish.at) <= FINISH_FOR
            });
        }
    }
}

impl Model {
    fn note(&mut self, snap: Snap) {
        let Some(base) = self.base else {
            self.dirty = snap.running > 1.0;
            self.base = Some(snap);
            self.remember(snap);
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
                self.record(&base, &snap);
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
        self.open(base, prev, snap);
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
        if self.dirty || base.running > 1.0 || snap.running > 1.0 {
            return false;
        }
        if self.since.iter().any(|old| old.running > 1.0) {
            return false;
        }
        if snap.requests != base.requests + 1 {
            return false;
        }
        if whole_delta(base.e2e_count, snap.e2e_count) != Some(1) {
            return false;
        }
        if whole_delta(base.ttft_count, snap.ttft_count) != Some(1) {
            return false;
        }
        let prompt = snap.prompt.saturating_sub(base.prompt);
        let cached = snap.cached.saturating_sub(base.cached);
        if cached > prompt || snap.generated < base.generated {
            return false;
        }
        snap.e2e_sum + 1e-6 >= base.e2e_sum && snap.ttft_sum + 1e-6 >= base.ttft_sum
    }

    fn record(&mut self, base: &Snap, snap: &Snap) {
        let prompt = snap.prompt - base.prompt;
        let cached = snap.cached - base.cached;
        let generated = snap.generated - base.generated;
        let ttft_s = snap.ttft_sum - base.ttft_sum;
        let e2e_s = snap.e2e_sum - base.e2e_sum;
        let (prompt_tps, gen_tps) = rates(prompt, cached, generated, ttft_s, e2e_s);
        self.finishes.push_back(Finish {
            at: snap.at,
            prompt,
            cached,
            prompt_tps,
            gen_tps,
            taken: false,
        });
        while self.finishes.len() > 64 {
            self.finishes.pop_front();
        }
    }

    fn open(&mut self, base: Snap, prev: Snap, snap: Snap) {
        let step = whole_delta(prev.ttft_count, snap.ttft_count).unwrap_or(-1);
        let from_base = whole_delta(base.ttft_count, snap.ttft_count).unwrap_or(2);
        let moved = snap.requests != prev.requests
            || snap.prompt != prev.prompt
            || snap.cached != prev.cached
            || snap.generated != prev.generated
            || !near(snap.e2e_sum, prev.e2e_sum)
            || !near(snap.e2e_count, prev.e2e_count);
        if moved
            || (step != 0 && step != 1)
            || from_base > 1
            || snap.ttft_sum + 1e-6 < prev.ttft_sum
        {
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
        && left.prompt == right.prompt
        && left.cached == right.cached
        && left.generated == right.generated
        && near(left.e2e_sum, right.e2e_sum)
        && near(left.e2e_count, right.e2e_count)
        && near(left.ttft_sum, right.ttft_sum)
        && near(left.ttft_count, right.ttft_count)
}

fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6
}

fn went_back(prev: &Snap, snap: &Snap) -> bool {
    snap.requests < prev.requests
        || snap.prompt < prev.prompt
        || snap.cached < prev.cached
        || snap.generated < prev.generated
        || snap.e2e_sum + 1e-6 < prev.e2e_sum
        || snap.e2e_count + 1e-6 < prev.e2e_count
        || snap.ttft_sum + 1e-6 < prev.ttft_sum
        || snap.ttft_count + 1e-6 < prev.ttft_count
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
            prompt,
            cached,
            generated,
            e2e_sum: e2e.0,
            e2e_count: e2e.1,
            ttft_sum: ttft.0,
            ttft_count: ttft.1,
            running,
        }
    }

    fn claim(seq: u64, input: u64, reported: bool) -> Claim {
        Claim {
            seq,
            model: "flash".to_owned(),
            input,
            reported,
        }
    }

    /// Baseline, then one request: TTFT two scrapes before the finish.
    /// Prompt 1,000, cached 600, TTFT 2 s, 401 generated, e2e 10 s.
    fn one(book: &mut SglangBook, clock: &Clock, seq: u64) -> Fill {
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(3_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(4_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(4_500), &[claim(seq, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(5_000),
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
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(7_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (3.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(8_000),
            Some(totals(2, 1_800, 800, 502, (13.0, 2.0), (3.0, 2.0), 0.0)),
        );
        book.activity(clock.at(8_500), &[claim(2, 800, false)]);
        book.observe(
            "flash",
            clock.at(9_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 500, 500, 11, (4.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 500, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        book.activity(
            clock.at(1_500),
            &[claim(1, 1_000, false), claim(2, 1_000, false)],
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2), None);
        book.observe(
            "flash",
            clock.at(3_000),
            Some(totals(2, 2_000, 1_000, 800, (20.0, 2.0), (4.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(4_000),
            Some(totals(3, 3_000, 1_600, 1_201, (30.0, 3.0), (6.0, 3.0), 0.0)),
        );
        book.activity(clock.at(4_500), &[claim(3, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(5_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 2.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(3_000),
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
            Some(totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(4, 9_000, 4_000, 800, (40.0, 4.0), (8.0, 4.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(2_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(3_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(5_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(5_500), &[claim(2, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(6_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        let late = clock.at(0) + Duration::from_secs(10) + Duration::from_millis(1);
        book.observe(
            "flash",
            late,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(late + Duration::from_millis(100), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            late + Duration::from_millis(200),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);

        // Exactly ten seconds is still inside the bound.
        let mut book = SglangBook::default();
        let start = clock.at(0);
        book.observe(
            "flash",
            start,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(start + Duration::from_millis(50), &[]);
        let edge = start + Duration::from_secs(10);
        book.observe(
            "flash",
            edge,
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(edge + Duration::from_millis(50), &[claim(7, 1_000, false)]);
        book.observe(
            "flash",
            edge + Duration::from_millis(100),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(7).expect("ten seconds").prompt_tps, Some(200.0));

        // An idle gap rebases. The request after it is filled.
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        let idle = clock.at(0) + Duration::from_secs(11);
        book.observe(
            "flash",
            idle,
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.observe(
            "flash",
            idle + Duration::from_millis(1_000),
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(
            idle + Duration::from_millis(1_100),
            &[claim(8, 1_000, false)],
        );
        book.observe(
            "flash",
            idle + Duration::from_millis(1_200),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        // The unmatched row and the row it did match share this read, so
        // the miss must not take the finish away from the match.
        book.activity(
            clock.at(1_500),
            &[claim(1, 999, false), claim(2, 1_000, false)],
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
        assert_eq!(book.fill(2).expect("the one match").cached, 600);

        // Two rows and one finish: neither is filled.
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(
            clock.at(1_500),
            &[claim(3, 1_000, false), claim(4, 1_000, false)],
        );
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(2, 1_500, 1_100, 402, (12.0, 2.0), (3.0, 2.0), 0.0)),
        );
        book.activity(
            clock.at(2_500),
            &[claim(1, 1_000, false), claim(2, 500, false)],
        );
        book.observe(
            "flash",
            clock.at(3_000),
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
    fn a_reported_row_consumes_the_window_and_stores_nothing() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000, true)]);
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
        book.activity(clock.at(2_500), &[claim(2, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(3_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(
            book.fill(2),
            None,
            "the reported row already took the finish"
        );
    }

    #[test]
    fn one_generated_token_and_a_fully_cached_prompt_omit_the_rate() {
        let clock = Clock(Instant::now());
        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 400, 1, (5.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 100, 150, 10, (4.0, 1.0), (1.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 100, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 0.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (2.0, 1.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (4.0, 2.0), 1.0)),
        );
        book.observe(
            "flash",
            clock.at(3_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (4.0, 2.0), 0.0)),
        );
        book.activity(clock.at(3_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(4_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe("flash", clock.at(500), None);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);

        let mut book = SglangBook::default();
        book.observe(
            "flash",
            clock.at(0),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        // 1_000 new tokens in 0.0005 s is 2_000_000 tok/s, past 1_000_000.
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 0, 401, (10.0, 1.0), (0.0005, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(4, 1_000, false)]);
        book.observe(
            "flash",
            clock.at(2_000),
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(100), &[]);
        book.observe(
            "flash",
            clock.at(1_000),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        book.activity(clock.at(1_500), &[claim(1, 1_000, false)]);
        book.activity(clock.at(1_500) + Duration::from_secs(11), &[]);
        assert_eq!(book.fill(1), None);
        book.observe(
            "flash",
            clock.at(1_500) + Duration::from_secs(11) + Duration::from_millis(50),
            Some(totals(1, 1_000, 600, 401, (10.0, 1.0), (2.0, 1.0), 0.0)),
        );
        assert_eq!(book.fill(1), None);
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
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        book.activity(clock.at(9_100), &[claim(2, 0, false)]);
        book.observe(
            "flash",
            clock.at(9_200),
            Some(totals(0, 0, 0, 0, (0.0, 0.0), (0.0, 0.0), 0.0)),
        );
        assert_eq!(
            book.fill(2),
            None,
            "the new process has not finished a request"
        );
    }
}
