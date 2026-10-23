//! tqLLM's monitor snapshot across reads (#180).
//!
//! tqLLM has no Prometheus `/metrics`. Its `GET /api/monitor/v1/snapshot`
//! ([`crate::metrics::parse_tqllm`]) lists every active request and the 32
//! newest finished ones, each with exact per-request numbers. Nothing in it
//! is a running total, so [`TqllmBook`] keeps one per model and hands the
//! poller a [`MetricsSample`] whose counters behave like any engine's:
//!
//! - **Counters** (prompt, cached prompt, generated, drafted, accepted)
//!   track request ids per `instance_id`. A request first seen finished
//!   counts once, whole; an active one counts by how far it advanced since
//!   the read before (the prompt by `prefill_processed_tokens` while in
//!   prefill, all of it once it decodes), and its rest when it finishes. A
//!   `failed` or `cancelled` request counts only what was already seen. The
//!   counters never go down within one instance.
//! - **Restarts.** A new `instance_id` is a new engine process: the counts
//!   start again from 0 and [`Observed::restarted`] tells the poller to
//!   reset its baselines, as on an unload.
//! - **Gaps.** A request that left the 32-entry history before it was seen
//!   finished (an id that came and went between two reads, or an active
//!   one that vanished) is a gap: its unseen tokens are never added, so
//!   there is no spike, and [`Observed::gap`] says how many for the log.
//!   Ids are told unseen by tqLLM's numbering, which counts up from 1 per
//!   instance.
//! - **Engine speeds** (SETUP and the SLOTS engine line) come from the
//!   finished requests' own `prefill_ms` / `decode_ms`, through
//!   [`crate::speeds::window`] like vLLM's per-request histograms.
//! - **Slots.** tqLLM numbers no slots, so each running request (prefill
//!   or decode) keeps the lowest free line it got, up to `engine.slots`.
//!
//! **Finished RECENT rows.** A llama-swap row of a tqLLM model may take
//! its cached tokens, and its prompt and generation tok/s where llama-swap
//! gave none, from a finished request of the snapshot only when exactly
//! one finished request has the row's prompt and output token counts and
//! ended within [`ROW_MATCH`] of the row. tqLLM's reply carries llama.cpp
//! `timings` whose `prompt_n` is the whole prompt and no `cache_n`, so
//! llama-swap's `input_tokens` is the whole prompt and its cached count is
//! unknown (#82). The snapshot has no end time: a request's end is known
//! to within one read (from its `elapsed_ms` at the read before, else
//! between the two reads), and every request whose window touches the
//! row's is a candidate. A row waits for a read stamped after its window,
//! and gives up after [`PENDING_FOR`].

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::{Instant, SystemTime};

use llama_core::backend::SpecCounts;

use crate::localtime::parse_rfc3339;
use crate::metrics::{
    EngineFacts, EngineLive, EngineRequest, EngineValues, Hist, MetricsSample, TqPhase, TqSnapshot,
};
use crate::speeds::{MAX_MODELS, MAX_PENDING, MAX_ROWS, PENDING_FOR, Phase, SpeedTotals};

/// How far a finished row's end may be from its request's, seconds.
pub const ROW_MATCH: f64 = 2.0;
/// Most SLOTS lines one tqLLM model gets.
pub const MAX_TQ_SLOTS: u32 = 64;
/// Finished requests kept per model for row fills.
const KEPT_FINISHED: usize = 64;
/// A finished request older than this can no longer meet a row.
const FINISHED_FOR: f64 = 30.0;

/// What a finished row takes from its one matching request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fill {
    /// `prefill_cached_tokens`.
    pub cached: u64,
    /// `prefill_tokens_per_second`: uncached tokens over prefill time.
    pub prompt_tps: Option<f64>,
    /// `decode_tokens_per_second`.
    pub gen_tps: Option<f64>,
}

/// One llama-swap activity row of a tqLLM model, as the fill needs it.
#[derive(Clone, Debug, PartialEq)]
pub struct RowClaim {
    /// [`crate::activity::ActivityRow::seq`].
    pub seq: u64,
    /// llama-swap model id, the key [`TqllmBook::observe`] uses.
    pub model: String,
    /// The row's whole prompt (its `input_tokens` for tqLLM).
    pub input: u64,
    /// The row's `output_tokens`.
    pub output: u64,
    /// The row's `timestamp`: its end, to the second.
    pub timestamp: Option<String>,
}

/// What one snapshot read gives the poller.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observed {
    /// Running totals since this instance was first read, and the gauges.
    pub sample: MetricsSample,
    /// ctx, SETUP values and the active requests.
    pub facts: EngineFacts,
    /// A new `instance_id` since the last read: the engine restarted.
    pub restarted: bool,
    /// Requests that left the history unseen since the last read.
    pub gap: u64,
    /// This is the instance's first gap: log it.
    pub log_gap: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counted {
    prompt: u64,
    cached: u64,
    output: u64,
    drafted: u64,
    accepted: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Seen {
    counted: Counted,
    /// Finished, failed or cancelled: nothing more is added.
    done: bool,
    /// Its start on the wall clock, seconds, from its `elapsed_ms` at the
    /// last read that saw it active: `(earliest, latest)`.
    start: Option<(f64, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct Finished {
    prompt: Option<u64>,
    output: Option<u64>,
    fill: Fill,
    /// Its end on the wall clock, seconds: `[lo, hi]`.
    end: (f64, f64),
}

#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    counted: Counted,
    prefill: Phase,
    decode: Phase,
}

#[derive(Debug, Default)]
struct Model {
    instance: String,
    seen: BTreeMap<u64, Seen>,
    max_id: Option<u64>,
    totals: Totals,
    slots: BTreeMap<u64, u32>,
    finished: VecDeque<Finished>,
    /// Wall clock before the latest read was sent, seconds.
    last_before: Option<f64>,
    gap_logged: bool,
    /// `/health`'s `fit` for this instance, once read.
    fit: Option<String>,
    health_read: bool,
}

#[derive(Clone, Debug)]
struct Pending {
    claim: RowClaim,
    seen: Instant,
}

/// tqLLM counters, slots and row fills, keyed by llama-swap model id.
#[derive(Debug, Default)]
pub struct TqllmBook {
    models: HashMap<String, Model>,
    pending: Vec<Pending>,
    rows: BTreeMap<u64, Fill>,
}

impl TqllmBook {
    /// Fold one snapshot of `model` in. `before` is the wall clock before
    /// the GET was sent and `wall` after the parse.
    pub fn observe(
        &mut self,
        model: &str,
        snap: &TqSnapshot,
        at: Instant,
        before: SystemTime,
        wall: SystemTime,
    ) -> Observed {
        if !self.models.contains_key(model) && self.models.len() >= MAX_MODELS {
            return Observed::default();
        }
        let (before, wall) = (secs(before), secs(wall).max(secs(before)));
        let mut restarted = false;
        let entry = self.models.entry(model.to_owned()).or_default();
        if entry.instance != snap.instance {
            restarted = !entry.instance.is_empty();
            *entry = Model {
                instance: snap.instance.clone(),
                ..Model::default()
            };
        }
        let first = entry.last_before.is_none();
        let mut gap = 0u64;
        let present: HashSet<u64> = snap.requests.iter().map(|req| req.id).collect();
        if !first {
            gap += entry
                .seen
                .iter()
                .filter(|(id, seen)| !seen.done && !present.contains(id))
                .count() as u64;
            if let (Some(max), Some(top)) = (entry.max_id, present.iter().max().copied())
                && top > max
            {
                let new = present.iter().filter(|id| **id > max).count() as u64;
                gap += (top - max).saturating_sub(new);
            }
        }
        let last_before = entry.last_before;
        for req in &snap.requests {
            let Some(phase) = req.phase else {
                continue;
            };
            let seen = entry.seen.entry(req.id).or_default();
            if seen.done {
                continue;
            }
            if matches!(phase, TqPhase::Failed | TqPhase::Cancelled) {
                seen.done = true;
                continue;
            }
            let now = Counted {
                prompt: req.prompt_read(),
                cached: req.cached_read(),
                output: req.output_tokens.unwrap_or(0),
                drafted: req.drafted.unwrap_or(0),
                accepted: req.accepted.unwrap_or(0),
            };
            let total = &mut entry.totals.counted;
            for (sum, old, new) in [
                (&mut total.prompt, &mut seen.counted.prompt, now.prompt),
                (&mut total.cached, &mut seen.counted.cached, now.cached),
                (&mut total.output, &mut seen.counted.output, now.output),
                (&mut total.drafted, &mut seen.counted.drafted, now.drafted),
                (
                    &mut total.accepted,
                    &mut seen.counted.accepted,
                    now.accepted,
                ),
            ] {
                if new > *old {
                    *sum = sum.saturating_add(new - *old);
                    *old = new;
                }
            }
            let elapsed = req.elapsed_ms / 1000.0;
            if phase != TqPhase::Complete {
                seen.start = Some((before - elapsed, wall - elapsed));
                continue;
            }
            seen.done = true;
            add_phase(
                &mut entry.totals.prefill,
                req.prefill_ms,
                req.prompt_read().saturating_sub(req.cached_read()),
            );
            add_phase(
                &mut entry.totals.decode,
                req.decode_ms,
                req.output_tokens.unwrap_or(0),
            );
            // Seen running before: its end is its start plus its frozen
            // elapsed time. Else it ended between the two reads. Seen only
            // finished on the first read, it ended at a time not known,
            // perhaps before llama-watch started: it is counted (above)
            // but fills no row.
            let end = match (seen.start, last_before) {
                (Some((lo, hi)), _) => (lo + elapsed, hi + elapsed),
                (None, Some(last)) => (last, wall),
                (None, None) => continue,
            };
            if entry.finished.len() >= KEPT_FINISHED {
                entry.finished.pop_front();
            }
            entry.finished.push_back(Finished {
                prompt: req.prompt_tokens,
                output: req.output_tokens,
                fill: Fill {
                    cached: req.prefill_cached.unwrap_or(0),
                    prompt_tps: req.prefill_tps,
                    gen_tps: req.decode_tps,
                },
                end,
            });
        }
        entry.seen.retain(|id, _| present.contains(id));
        entry.max_id = entry.max_id.max(present.iter().max().copied());
        entry
            .finished
            .retain(|done| wall - done.end.1 <= FINISHED_FOR);
        entry.last_before = Some(before);
        let log_gap = gap > 0 && !entry.gap_logged;
        entry.gap_logged |= gap > 0;

        // The SLOTS lines: a running request keeps its line, a new one
        // takes the lowest free one.
        let cap = snap
            .slots
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(1)
            .min(MAX_TQ_SLOTS);
        let running: Vec<u64> = snap
            .requests
            .iter()
            .filter(|req| req.phase.is_some_and(TqPhase::running))
            .map(|req| req.id)
            .collect();
        entry
            .slots
            .retain(|id, slot| running.contains(id) && *slot < cap);
        for id in &running {
            if entry.slots.contains_key(id) {
                continue;
            }
            let used: HashSet<u32> = entry.slots.values().copied().collect();
            if let Some(free) = (0..cap).find(|slot| !used.contains(slot)) {
                entry.slots.insert(*id, free);
            }
        }
        let requests: Vec<EngineRequest> = snap
            .requests
            .iter()
            .filter(|req| req.phase.is_some_and(TqPhase::active))
            .map(|req| EngineRequest {
                request: req.clone(),
                slot: entry.slots.get(&req.id).copied(),
            })
            .collect();
        let queued = snap
            .requests
            .iter()
            .filter(|req| req.phase == Some(TqPhase::Queued))
            .count();

        let totals = entry.totals;
        let sample = MetricsSample {
            n_decode_total: Some(totals.counted.output),
            requests_processing: Some(running.len() as f64),
            prompt_total: Some(totals.counted.prompt),
            queued: Some(queued as f64),
            cached_total: Some(totals.counted.cached),
            spec: Some(SpecCounts {
                drafts: None,
                draft_tokens: totals.counted.drafted,
                accepted: totals.counted.accepted.min(totals.counted.drafted),
            }),
            speeds: SpeedTotals {
                prefill: Some(totals.prefill),
                decode: Some(totals.decode),
            }
            .reported(),
            ..MetricsSample::default()
        };
        let mut values = EngineValues::new();
        if let Some(build) = &snap.build {
            values.insert("version", build.clone());
        }
        if let Some(slots) = snap.slots {
            values.insert("slots", slots.to_string());
        }
        if let Some(ctx) = snap.context_per_slot {
            values.insert("context_per_slot", ctx.to_string());
        }
        if let Some(fit) = &entry.fit {
            values.insert("fit", fit.clone());
        }
        let facts = EngineFacts {
            ctx: snap.context_per_slot.and_then(|n| u32::try_from(n).ok()),
            ctx_exact: true,
            values,
            live: Some(EngineLive {
                requests,
                request_slots: snap.slots.and_then(|n| u32::try_from(n).ok()),
                slot_ctx: snap.context_per_slot,
                instance: Some(snap.instance.clone()),
                ..EngineLive::default()
            }),
            ..EngineFacts::default()
        };
        self.resolve(at);
        Observed {
            sample,
            facts,
            restarted,
            gap,
            log_gap,
        }
    }

    /// `snap` comes from a new engine process of `model`: its
    /// `instance_id` is not the one read before. The poller then resets
    /// its baselines (and this book's state) before [`Self::observe`].
    #[must_use]
    pub fn is_restart(&self, model: &str, snap: &TqSnapshot) -> bool {
        self.models
            .get(model)
            .is_some_and(|entry| !entry.instance.is_empty() && entry.instance != snap.instance)
    }

    /// `/health` should be read for `model`: not yet for this instance.
    #[must_use]
    pub fn health_due(&self, model: &str) -> bool {
        self.models
            .get(model)
            .is_some_and(|entry| !entry.instance.is_empty() && !entry.health_read)
    }

    /// `/health` was read for `model`'s current instance: its `fit`, when
    /// the fit changed anything (`None` when it kept what was asked, or
    /// the read failed).
    pub fn set_fit(&mut self, model: &str, fit: Option<String>) {
        if let Some(entry) = self.models.get_mut(model) {
            entry.fit = fit;
            entry.health_read = true;
        }
    }

    /// New llama-swap activity rows of tqLLM models (#180).
    pub fn activity(&mut self, at: Instant, rows: &[RowClaim]) {
        for claim in rows {
            if self.pending.len() >= MAX_PENDING {
                break;
            }
            if self.models.contains_key(&claim.model) {
                self.pending.push(Pending {
                    claim: claim.clone(),
                    seen: at,
                });
            }
        }
        self.resolve(at);
    }

    /// The fill for activity row `seq`.
    #[must_use]
    pub fn fill(&self, seq: u64) -> Option<Fill> {
        self.rows.get(&seq).copied()
    }

    /// llama-swap's row ids start again (#44): the fills and the rows
    /// waiting for one are of the old ids, so they go.
    pub fn forget_rows(&mut self) {
        self.rows.clear();
        self.pending.clear();
    }

    /// `model`'s process is gone: its next one starts from 0. Rows already
    /// filled keep their numbers.
    pub fn forget(&mut self, model: &str) {
        self.models.remove(model);
        self.pending.retain(|row| row.claim.model != model);
    }

    /// Fill each pending row whose window every read has covered, or drop
    /// it after [`PENDING_FOR`].
    fn resolve(&mut self, now: Instant) {
        let models = &self.models;
        let mut done: Vec<(u64, Option<Fill>)> = Vec::new();
        self.pending.retain(|row| {
            let Some(entry) = models.get(&row.claim.model) else {
                return false;
            };
            let Some(stamp) = row.claim.timestamp.as_deref().and_then(parse_rfc3339) else {
                return false;
            };
            // The row ended in [stamp, stamp + 1).
            let lo = stamp as f64 - ROW_MATCH;
            let hi = stamp as f64 + 1.0 + ROW_MATCH;
            if entry.last_before.is_some_and(|last| last >= hi) {
                let mut matches = entry.finished.iter().filter(|done| {
                    done.prompt == Some(row.claim.input)
                        && done.output == Some(row.claim.output)
                        && done.end.0 <= hi
                        && done.end.1 >= lo
                });
                let fill = match (matches.next(), matches.next()) {
                    (Some(one), None) => Some(one.fill),
                    _ => None,
                };
                done.push((row.claim.seq, fill));
                return false;
            }
            now.saturating_duration_since(row.seen) < PENDING_FOR
        });
        for (seq, fill) in done {
            if let Some(fill) = fill {
                if self.rows.len() >= MAX_ROWS
                    && let Some(oldest) = self.rows.keys().next().copied()
                {
                    self.rows.remove(&oldest);
                }
                self.rows.insert(seq, fill);
            }
        }
    }
}

/// One finished request's phase time and tokens into `phase`. A request
/// without the time adds nothing.
fn add_phase(phase: &mut Phase, ms: Option<f64>, tokens: u64) {
    let Some(ms) = ms.filter(|ms| ms.is_finite() && *ms > 0.0) else {
        return;
    };
    phase.seconds = Hist {
        sum: phase.seconds.sum + ms / 1000.0,
        count: phase.seconds.count + 1.0,
    };
    phase.tokens = Hist {
        sum: phase.tokens.sum + tokens as f64,
        count: phase.tokens.count + 1.0,
    };
}

/// SETUP's `engine:kv_asked` for a tqLLM whose KV is not known (#180): a
/// value given that is not a token, or flags not read as tqLLM's.
pub const KV_UNKNOWN: &str = "--";

/// The KV a tqLLM launch command named (#180,
/// [`crate::sources::cmdline::TqKv::Named`]), for SETUP's
/// `engine:kv_asked`: `q8` when K and V match, else `q4/tcq2`. `None`
/// when the detail holds no KV token. A command that named none gets no
/// value, and SETUP draws the default `kv bf16` dim, since the memory fit
/// may pick a smaller one.
#[must_use]
pub fn kv_asked(detail: Option<&llama_core::detail::ModelDetail>) -> Option<String> {
    let detail = detail?;
    let token = |side: &Option<String>| {
        side.as_deref()
            .filter(|t| llama_core::detail::is_token(t))
            .map(str::to_owned)
    };
    match (token(&detail.kv_k), token(&detail.kv_v)) {
        (Some(k), Some(v)) if k == v => Some(k),
        (Some(k), Some(v)) => Some(format!("{k}/{v}")),
        (Some(one), None) | (None, Some(one)) => Some(one),
        (None, None) => None,
    }
}

/// Wall clock as seconds since the epoch; before it is 0.
fn secs(wall: SystemTime) -> f64 {
    wall.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64())
}
