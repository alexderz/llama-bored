//! Requests still running, for RECENT's in-flight rows (#75).
//!
//! Where an engine reports a request while it runs, RECENT shows it at the
//! top until llama-swap's finished row for it arrives:
//!
//! - **llama.cpp `/slots`**: one row per busy slot (`is_processing`), from
//!   `n_prompt_tokens_cache` (reused), `n_prompt_tokens_processed`
//!   (computed so far beyond the cache), `n_prompt_tokens` and
//!   `next_token[0].n_decoded`. A new `id_task` is a new request.
//!   `n_prompt_tokens` is the tokens the slot holds, not the whole prompt
//!   (#78), and `/slots` sends no prompt length, so:
//!   - in **prefill** the row shows what the slot holds (cached +
//!     processed so far) with no target: [`Flight::prompt_known`] is false
//!     and IN reads as a lower bound;
//!   - once it **decodes**, the prefill is done and the prompt is exact
//!     ([`crate::slots::whole_prompt`]); the output is `n_decoded`.
//! - **Strata `live`**: while `state` is `reading` or `generating`, from
//!   `prompt_total` (or `prompt_tokens`), `prompt_read` and `generated`.
//!   Strata reports no cached count for the live request, so all its input
//!   is new until it finishes.
//! - **Strata in parallel mode** (`live.slots[]`, #81): one row per busy
//!   slot, from its `prompt_tokens` (the whole prompt), `generated`,
//!   `elapsed_s` (the row starts that long before it was first seen) and
//!   `tok_s`. A slot whose prompt changes, or whose clock or output goes
//!   back, runs a new request. `live.prompt_read` / `prompt_total` are the
//!   newest request's only: the one reading slot whose prompt is
//!   `live.prompt_tokens` gets them, and with them the target track
//!   ([`Flight::progress`]); any other slot in prefill shows its prompt
//!   with no target.
//! - vLLM, SGLang and OpenAI-compatible servers report no per-request
//!   progress: no in-flight row; RECENT keeps marking its newest finished
//!   row `gen` while the model generates.
//!
//! A row holds the numbers of the latest poll only: no extrapolation and
//! nothing tied to the frame clock, so it changes exactly when a poll does
//! (DUR is the time from the request's first sighting to that poll).
//!
//! **Handover.** The poller numbers every activity row it keeps (`seq`).
//! A flight remembers the newest number when it started; a later row for
//! the same model with the same whole prompt (`input + cache`) is its
//! finished row and replaces it at once. A flight whose slot went idle also goes when any
//! later row of its model arrives, and after [`HANDOVER`] in any case. So
//! a request is never shown twice, and the gap between the slot going idle
//! and llama-swap's row is covered.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use llama_core::backend::Backend;

use crate::activity::ActivityRow;
use crate::poller::ModelEngineLive;
use crate::resets::ResetReason;
use crate::slots::SlotView;

/// How long a finished flight waits for llama-swap's row.
pub const HANDOVER: Duration = Duration::from_secs(3);
/// Most flights tracked at once.
pub const MAX_FLIGHTS: usize = 16;

/// One Strata slot's request as last seen (#81), to tell a new one.
#[derive(Clone, Copy, Debug, Default)]
struct SlotMark {
    prompt: u64,
    generated: u64,
    elapsed: f64,
    epoch: u32,
    /// Seen idle since: its next request is a new one.
    idle: bool,
}

/// One request still running (or just finished, waiting for its row).
#[derive(Clone, Debug, PartialEq)]
pub struct Flight {
    /// Display name of the model, as the slots and the snapshot name it.
    pub model: String,
    /// When it was first seen.
    pub started_mono: Instant,
    /// Wall clock when it was first seen, for the TIME column.
    pub started_wall: SystemTime,
    /// The poll the numbers come from.
    pub polled: Instant,
    /// Whole prompt, tokens, when [`Self::prompt_known`]; else the prompt
    /// tokens held so far (cached + processed), a lower bound (#78).
    pub prompt: u64,
    /// `prompt` is the whole prompt. False for a llama.cpp slot in prefill.
    pub prompt_known: bool,
    /// Prompt tokens reused from the cache (0 when unknown).
    pub cached: u64,
    /// Prompt tokens computed so far beyond the cache.
    pub processed: u64,
    /// [`Self::processed`] is the engine's own count against the whole
    /// prompt, so the rest of the prompt can be drawn as the target. False
    /// for a Strata slot that is not the newest request (#81).
    pub progress: bool,
    /// Tokens generated so far.
    pub decoded: u64,
    /// Decoding; else still in prefill.
    pub decoding: bool,
    /// A context reset just before it (llama.cpp): it starts from zero.
    pub reset: Option<ResetReason>,
    /// Live prefill and decode tok/s, as the engine or poller measured them.
    pub prompt_tps: Option<f64>,
    pub gen_tps: Option<f64>,
    /// The slot's context size, when it reports one.
    pub n_ctx: Option<u64>,
    /// The engine running it, which says how its finished row counts the
    /// prompt (#82).
    pub engine: Option<Backend>,
    id: FlightId,
    /// Newest activity `seq` when the flight started.
    head_seq: u64,
    /// When its slot or engine stopped reporting it.
    ended: Option<Instant>,
    /// The slot's reset total before the request.
    start_resets: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FlightId {
    Slot {
        model: String,
        slot: i64,
        task: i64,
    },
    Engine {
        model: String,
        prompt: u64,
    },
    /// A Strata batch slot's request (#81); `epoch` counts the requests
    /// the slot ran.
    EngineSlot {
        model: String,
        slot: u32,
        epoch: u32,
    },
}

/// What one poll says.
#[derive(Clone, Copy, Debug)]
pub struct Poll<'a> {
    /// llama.cpp slots.
    pub slots: &'a [SlotView],
    /// Engines' own live reports (Strata).
    pub engine_live: &'a [ModelEngineLive],
    /// RECENT's rows, newest first.
    pub activity: &'a [ActivityRow],
    /// `(display name, llama-swap key)` of each loaded model.
    pub ids: &'a [(String, String)],
    /// Prefill tok/s from `/slots`.
    pub prompt_tps: Option<f64>,
    /// Decode tok/s from the token counter.
    pub gen_tps: Option<f64>,
}

/// The flights across polls.
#[derive(Debug, Default)]
pub struct Tracker {
    flights: Vec<Flight>,
    /// Each slot's reset total at the previous poll.
    slot_resets: HashMap<(String, i64), u64>,
    /// Each Strata slot's request as last seen (#81).
    strata_slots: HashMap<(String, u32), SlotMark>,
    /// Flights their finished row replaced, newest last, at most
    /// [`MAX_REPLACED`]: never shown again (#81).
    replaced: VecDeque<FlightId>,
}

/// Most replaced flights remembered.
const MAX_REPLACED: usize = 64;
/// Most Strata slots remembered across models.
const MAX_STRATA_MARKS: usize = 256;

impl Tracker {
    /// Take a new poll's numbers.
    pub fn poll(&mut self, poll: &Poll<'_>, mono: Instant, wall: SystemTime) {
        let head_seq = poll.activity.iter().map(|row| row.seq).max().unwrap_or(0);
        let mut seen: Vec<FlightId> = Vec::new();
        for slot in poll.slots.iter().filter(|slot| slot.is_processing) {
            let id = FlightId::Slot {
                model: slot.model.clone(),
                slot: slot.id,
                task: slot.id_task,
            };
            let key = (slot.model.clone(), slot.id);
            let before = self
                .slot_resets
                .get(&key)
                .copied()
                .unwrap_or_else(|| slot.resets.total());
            // #78: the whole prompt once decoding, else what is held.
            let total = slot.prompt_total();
            let prompt = total.unwrap_or(slot.n_prompt_tokens);
            let cached = slot.prompt_cached.unwrap_or(0).min(prompt);
            let Some(flight) = self.flight(&id, &slot.model, mono, wall, head_seq, before) else {
                continue;
            };
            flight.polled = mono;
            flight.prompt = prompt;
            flight.prompt_known = total.is_some();
            flight.progress = total.is_some();
            flight.cached = cached;
            flight.processed = prompt - cached;
            flight.decoded = slot.n_decoded;
            flight.decoding = slot.n_decoded > 0;
            flight.reset = (cached == 0 && slot.resets.total() > flight.start_resets)
                .then_some(slot.last_reset)
                .flatten();
            flight.prompt_tps = poll.prompt_tps;
            flight.gen_tps = poll.gen_tps;
            flight.n_ctx = slot.n_ctx.filter(|n| *n > 0);
            flight.engine = Some(Backend::LlamaCpp);
            flight.ended = None;
            seen.push(id);
        }
        for slot in poll.slots {
            self.slot_resets
                .insert((slot.model.clone(), slot.id), slot.resets.total());
        }
        for engine in poll.engine_live {
            let live = &engine.live;
            if !live.slots().is_empty() {
                self.poll_strata_slots(engine, mono, wall, head_seq, &mut seen);
                continue;
            }
            let state = live.state.as_deref().unwrap_or("");
            if state != "reading" && state != "generating" {
                continue;
            }
            let total = live.prompt_total.or(live.prompt_tokens);
            let prompt = total.unwrap_or(0);
            let id = FlightId::Engine {
                model: engine.model.clone(),
                prompt,
            };
            let Some(flight) = self.flight(&id, &engine.model, mono, wall, head_seq, 0) else {
                continue;
            };
            let decoding = state == "generating";
            flight.polled = mono;
            flight.prompt = prompt;
            flight.prompt_known = total.is_some();
            flight.progress = total.is_some();
            flight.cached = 0;
            flight.processed = if decoding {
                prompt
            } else {
                live.prompt_read.unwrap_or(0).min(prompt)
            };
            flight.decoded = live.generated.unwrap_or(0);
            flight.decoding = decoding;
            flight.prompt_tps = live.prefill_tok_s_mean;
            flight.gen_tps = live.tok_s.or(live.tok_s_mean);
            flight.engine = Some(Backend::Strata);
            flight.ended = None;
            seen.push(id);
        }
        for flight in &mut self.flights {
            if !seen.contains(&flight.id) && flight.ended.is_none() {
                flight.ended = Some(mono);
            }
        }
        self.hand_over(poll);
        self.retire(mono);
    }

    /// Handover: the finished row replaces its flight. Oldest flight
    /// first, each row replaces at most one flight by its prompt, so two
    /// requests running together are never both taken by one row (#81).
    fn hand_over(&mut self, poll: &Poll<'_>) {
        let mut taken: Vec<u64> = Vec::new();
        let mut order: Vec<usize> = (0..self.flights.len()).collect();
        order.sort_by_key(|at| self.flights[*at].started_mono);
        let mut gone = vec![false; self.flights.len()];
        for at in order {
            let flight = &self.flights[at];
            let key = model_key(poll.ids, &flight.model);
            let later = || {
                poll.activity.iter().filter(|row| {
                    row.seq > flight.head_seq && row.model == key && !taken.contains(&row.seq)
                })
            };
            // The row's whole prompt, read per engine (#82); a prompt
            // still only a lower bound matches no row (#78).
            let finished = flight
                .prompt_known
                .then(|| {
                    later().find(|row| {
                        row.prompt_for(flight.engine).map(|split| split.whole)
                            == Some(flight.prompt)
                    })
                })
                .flatten()
                .map(|row| row.seq);
            if let Some(seq) = finished {
                taken.push(seq);
                gone[at] = true;
            } else if flight.ended.is_some() && later().next().is_some() {
                gone[at] = true;
            }
        }
        // Only ids unique to one request: a serial Strata request is known
        // by its prompt length, which the next request may share.
        for (flight, _) in self
            .flights
            .iter()
            .zip(&gone)
            .filter(|(flight, gone)| **gone && !matches!(flight.id, FlightId::Engine { .. }))
        {
            if self.replaced.len() >= MAX_REPLACED {
                self.replaced.pop_front();
            }
            self.replaced.push_back(flight.id.clone());
        }
        let mut at = 0;
        self.flights.retain(|_| {
            at += 1;
            !gone[at - 1]
        });
    }

    /// One row per busy Strata batch slot (#81).
    fn poll_strata_slots(
        &mut self,
        engine: &ModelEngineLive,
        mono: Instant,
        wall: SystemTime,
        head_seq: u64,
        seen: &mut Vec<FlightId>,
    ) {
        let live = &engine.live;
        let newest = live.newest_slot();
        if self.strata_slots.len() >= MAX_STRATA_MARKS {
            self.strata_slots.clear();
        }
        for slot in live.slots().iter().filter(|slot| !slot.busy()) {
            if let Some(mark) = self
                .strata_slots
                .get_mut(&(engine.model.clone(), slot.slot))
            {
                mark.idle = true;
            }
        }
        for slot in live.slots().iter().filter(|slot| slot.busy()) {
            let prompt = slot.prompt_tokens.unwrap_or(0);
            let generated = slot.generated.unwrap_or(0);
            let elapsed = slot
                .elapsed_s
                .filter(|s| s.is_finite() && *s >= 0.0)
                .unwrap_or(0.0);
            let mark = self
                .strata_slots
                .entry((engine.model.clone(), slot.slot))
                .or_insert(SlotMark {
                    prompt,
                    generated,
                    elapsed,
                    epoch: 0,
                    idle: false,
                });
            if mark.idle
                || mark.prompt != prompt
                || generated < mark.generated
                || elapsed + 0.05 < mark.elapsed
            {
                mark.epoch = mark.epoch.wrapping_add(1);
            }
            (mark.prompt, mark.generated, mark.elapsed, mark.idle) =
                (prompt, generated, elapsed, false);
            let id = FlightId::EngineSlot {
                model: engine.model.clone(),
                slot: slot.slot,
                epoch: mark.epoch,
            };
            // The engine's own clock: the request started this long ago.
            let ago = Duration::try_from_secs_f64(elapsed.min(86_400.0)).unwrap_or_default();
            let started = mono.checked_sub(ago).unwrap_or(mono);
            let started_wall = wall.checked_sub(ago).unwrap_or(wall);
            let Some(flight) = self.flight(&id, &engine.model, started, started_wall, head_seq, 0)
            else {
                continue;
            };
            let decoding = slot.decoding();
            let progress = !decoding && newest == Some(slot.slot);
            flight.polled = mono;
            flight.prompt = prompt;
            flight.prompt_known = slot.prompt_tokens.is_some();
            flight.progress = progress;
            flight.cached = 0;
            flight.processed = if progress {
                live.prompt_read.unwrap_or(0).min(prompt)
            } else {
                prompt
            };
            flight.decoded = generated;
            flight.decoding = decoding;
            flight.prompt_tps = if progress {
                live.prefill_tok_s_mean
            } else {
                None
            };
            flight.gen_tps = slot.tok_s;
            flight.engine = Some(Backend::Strata);
            flight.ended = None;
            seen.push(id);
        }
    }

    /// Drop flights that ended more than [`HANDOVER`] ago without a row.
    pub fn retire(&mut self, mono: Instant) {
        self.flights.retain(|flight| {
            flight
                .ended
                .is_none_or(|at| mono.saturating_duration_since(at) < HANDOVER)
        });
    }

    /// The flights, newest first.
    #[must_use]
    pub fn flights(&self) -> Vec<&Flight> {
        let mut out: Vec<&Flight> = self.flights.iter().collect();
        out.sort_by_key(|f| std::cmp::Reverse(f.started_mono));
        out
    }

    fn flight(
        &mut self,
        id: &FlightId,
        model: &str,
        mono: Instant,
        wall: SystemTime,
        head_seq: u64,
        start_resets: u64,
    ) -> Option<&mut Flight> {
        if let Some(at) = self.flights.iter().position(|f| f.id == *id) {
            return Some(&mut self.flights[at]);
        }
        // Its finished row already replaced it: an engine that still
        // reports it for a poll after does not bring it back.
        if self.replaced.contains(id) {
            return None;
        }
        if self.flights.len() >= MAX_FLIGHTS
            && let Some(oldest) = self
                .flights
                .iter()
                .enumerate()
                .min_by_key(|(_, f)| f.started_mono)
                .map(|(i, _)| i)
        {
            self.flights.remove(oldest);
        }
        self.flights.push(Flight {
            model: model.to_owned(),
            started_mono: mono,
            started_wall: wall,
            polled: mono,
            prompt: 0,
            prompt_known: false,
            cached: 0,
            processed: 0,
            progress: false,
            decoded: 0,
            decoding: false,
            reset: None,
            prompt_tps: None,
            gen_tps: None,
            n_ctx: None,
            engine: None,
            id: id.clone(),
            head_seq,
            ended: None,
            start_resets,
        });
        let last = self.flights.len() - 1;
        Some(&mut self.flights[last])
    }
}

/// The llama-swap key RECENT rows carry for a display name: from the
/// loaded models' pairs, else the name itself.
#[must_use]
pub fn model_key(ids: &[(String, String)], display: &str) -> String {
    ids.iter()
        .find(|(name, _)| name == display)
        .map_or_else(|| display.to_owned(), |(_, key)| key.clone())
}
