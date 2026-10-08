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

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

use crate::activity::ActivityRow;
use crate::poller::ModelEngineLive;
use crate::resets::ResetReason;
use crate::slots::SlotView;

/// How long a finished flight waits for llama-swap's row.
pub const HANDOVER: Duration = Duration::from_secs(3);
/// Most flights tracked at once.
pub const MAX_FLIGHTS: usize = 16;

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
    Slot { model: String, slot: i64, task: i64 },
    Engine { model: String, prompt: u64 },
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
}

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
            let flight = self.flight(&id, &slot.model, mono, wall, head_seq, before);
            flight.polled = mono;
            flight.prompt = prompt;
            flight.prompt_known = total.is_some();
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
            flight.ended = None;
            seen.push(id);
        }
        for slot in poll.slots {
            self.slot_resets
                .insert((slot.model.clone(), slot.id), slot.resets.total());
        }
        for engine in poll.engine_live {
            let live = &engine.live;
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
            let flight = self.flight(&id, &engine.model, mono, wall, head_seq, 0);
            let decoding = state == "generating";
            flight.polled = mono;
            flight.prompt = prompt;
            flight.prompt_known = total.is_some();
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
            flight.ended = None;
            seen.push(id);
        }
        for flight in &mut self.flights {
            if !seen.contains(&flight.id) && flight.ended.is_none() {
                flight.ended = Some(mono);
            }
        }
        // Handover: the finished row replaces its flight.
        self.flights.retain(|flight| {
            let key = model_key(poll.ids, &flight.model);
            let later = || {
                poll.activity
                    .iter()
                    .filter(|row| row.seq > flight.head_seq && row.model == key)
            };
            // llama.cpp's row splits the prompt into input + cache; a
            // prompt still only a lower bound matches no row (#78).
            let finished = flight.prompt_known
                && later().any(|row| {
                    let input = row.input_tokens;
                    let whole = input.map(|n| n.saturating_add(row.cached_tokens.unwrap_or(0)));
                    whole == Some(flight.prompt) || input == Some(flight.prompt)
                });
            let replaced = finished || (flight.ended.is_some() && later().next().is_some());
            !replaced
        });
        self.retire(mono);
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
    ) -> &mut Flight {
        if let Some(at) = self.flights.iter().position(|f| f.id == *id) {
            return &mut self.flights[at];
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
            decoded: 0,
            decoding: false,
            reset: None,
            prompt_tps: None,
            gen_tps: None,
            n_ctx: None,
            id: id.clone(),
            head_seq,
            ended: None,
            start_resets,
        });
        let last = self.flights.len() - 1;
        &mut self.flights[last]
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
