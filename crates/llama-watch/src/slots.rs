//! `/upstream/<model>/slots`: declared fields, tail text, prompt rate, and
//! each slot's held context and drops by reason for the snapshot (#10, #9).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use llama_core::rate::{counter_delta, delta_per_s};
use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::activity::{ActivityRow, model_key};
use crate::config::PromptView;
use crate::resets::{Displaced, PENDING_TTL, ResetCounts, ResetReason, classify};
use crate::tty::chat_template::clean;
use crate::tty::ctx_history::{MAX_SLOTS as MAX_TRACKED_SLOTS, is_drop};
use crate::tty::grid::Cell;
use crate::tty::sanitize::sanitize;

/// One slot after tail truncation and console sanitising.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotView {
    /// Sanitised display name of the model that owns the slot.
    pub model: String,
    /// Slot index from llama-server.
    pub id: i64,
    /// Task id. Input text is captured once per new value.
    pub id_task: i64,
    /// Server's processing flag.
    pub is_processing: bool,
    /// Prompt tokens for this task.
    pub n_prompt_tokens: u64,
    /// Prompt tokens processed so far.
    pub n_prompt_tokens_processed: u64,
    /// `next_token[0].n_decoded`, or 0 when the array is empty.
    pub n_decoded: u64,
    /// `n_ctx` when that key is present. `Some(0)` is a real zero.
    pub n_ctx: Option<u64>,
    /// Prompt tokens that occupy context: `n_prompt_tokens` when that key is
    /// present, otherwise `n_prompt_tokens_cache`. `None` when both are absent.
    pub ctx_prompt: Option<u64>,
    /// Sanitised prompt tail. Unchanged when this body skipped text.
    pub input: Vec<Cell>,
    /// Sanitised generated tail from this poll, or the previous tail when
    /// text was skipped.
    pub output: Vec<Cell>,
    /// Context the slot holds (#10): prompt plus decoded, and an idle slot
    /// keeps its last value (its KV cache stays). `None` until `/slots`
    /// gave a prompt count.
    pub ctx_used: Option<u64>,
    /// Context drops this slot had since the watcher started (#10), by the
    /// sparkline's rule ([`is_drop`] while busy), by reason (#9). A drop is
    /// counted when its reason is decided, up to [`PENDING_TTL`] later.
    pub resets: ResetCounts,
    /// The last decided reason (#9).
    pub last_reset: Option<ResetReason>,
}

/// The one slot that both IN and OUT show.
///
/// A busy slot always beats an idle one, so a stale idle prompt can never
/// win. Among equals the highest `id_task` wins: llama-server hands task ids
/// out in increasing order, so that is the newest busy task, or the task that
/// finished last when every slot is idle. A slot with no task reads as
/// `id_task = -1`. `None` only when there are no slots.
#[must_use]
pub fn pick_slot(slots: &[SlotView]) -> Option<&SlotView> {
    slots
        .iter()
        .max_by_key(|slot| (slot.is_processing, slot.id_task))
}

#[derive(Debug, Default)]
struct Tracked {
    model_id: String,
    model: String,
    id: i64,
    id_task: i64,
    is_processing: bool,
    n_prompt_tokens: u64,
    n_prompt_tokens_processed: u64,
    n_decoded: u64,
    n_ctx: Option<u64>,
    ctx_prompt: Option<u64>,
    input: Vec<Cell>,
    output: Vec<Cell>,
    captured_task: Option<i64>,
}

/// One slot's context across the whole watcher run (#10, #9).
#[derive(Debug, Default)]
struct CtxTrack {
    held: Option<u64>,
    resets: ResetCounts,
    last: Option<ResetReason>,
    /// A drop waiting for its evidence (#9).
    pending: Option<Pending>,
}

/// A drop whose reason is not decided yet. Numbers only.
#[derive(Debug)]
struct Pending {
    /// Context the slot held before the drop: the conversation that left.
    left: u64,
    /// The new task's prompt tokens.
    prompt: u64,
    /// `/slots` `n_prompt_tokens_cache` at the drop, when the server sent it.
    slots_cached: Option<u64>,
    /// Newest activity row id when the drop was seen. The request's own row
    /// appears after it, when the request finishes.
    after: Option<i64>,
    /// When an activity read first saw it waiting; [`PENDING_TTL`] runs from here.
    since: Option<Instant>,
}

impl CtxTrack {
    /// The [`crate::tty::ctx_history::CtxHistory::sample`] rule: only a busy
    /// slot can drop, and an idle slot that reads lower keeps its value.
    /// Returns the context held before a drop.
    fn sample(&mut self, used: Option<u64>, busy: bool) -> Option<u64> {
        let prev = self.held;
        let dropped = match (used, prev) {
            (Some(v), Some(p)) if busy && is_drop(p, v) => Some(p),
            _ => None,
        };
        self.held = match (used, prev) {
            (Some(v), Some(p)) if !busy && v < p => Some(p),
            (Some(v), _) => Some(v),
            (None, p) => p,
        };
        dropped
    }

    /// Decide the pending drop with `cached` as its evidence.
    fn decide(&mut self, cached: Option<u64>, displaced: &mut Displaced, now: Instant) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let reason = classify(pending.prompt, cached, displaced, now);
        self.resets.add(reason);
        self.last = Some(reason);
        if reason != ResetReason::Compacted {
            displaced.push(pending.left, now);
        }
    }
}

/// True when an activity row's prompt (`input + cache`, llama.cpp timings)
/// is `prompt`, give or take 0.5 % (at least 8 tokens).
fn row_matches(row: &ActivityRow, prompt: u64) -> bool {
    let (Some(input), Some(cache)) = (row.input_tokens, row.cached_tokens) else {
        return false;
    };
    input.saturating_add(cache).abs_diff(prompt) <= (prompt / 200).max(8)
}

/// Slot numbers and tails across polls.
///
/// [`Self::without_text`] builds a book that never keeps llama text: the
/// body is parsed with `prompt` and `generated` as [`IgnoredAny`], so those
/// strings are not even allocated (RR-LV1, `tty.show_text = false`).
#[derive(Debug, Default)]
pub struct SlotBook {
    text_off: bool,
    prompt_view: PromptView,
    slots: Vec<Tracked>,
    last_processed: HashMap<(String, i64), u64>,
    last_at: Option<Instant>,
    round_delta: u64,
    round_seen: bool,
    prompt_tps: Option<f64>,
    /// Held context and drop counts by `(model id, slot id)`, at most
    /// [`MAX_TRACKED_SLOTS`]. [`Self::clear`] and [`Self::retain_models`]
    /// forget the held context but keep the counts, so a counter only
    /// restarts with the watcher.
    ctx: HashMap<(String, i64), CtxTrack>,
    /// Conversations each model's slots lost, by model id (#9).
    displaced: HashMap<String, Displaced>,
    /// Newest llama-swap activity row id seen (#9).
    activity_newest: Option<i64>,
}

impl SlotBook {
    /// Start a slots phase. Deltas from [`Self::apply`] land in [`Self::finish_round`].
    pub fn begin_round(&mut self) {
        self.round_delta = 0;
        self.round_seen = false;
    }

    /// A book that keeps text and shows prompts as `view` says.
    #[must_use]
    pub fn with_prompt_view(prompt_view: PromptView) -> Self {
        Self {
            prompt_view,
            ..Self::default()
        }
    }

    /// A book that keeps slot numbers only. Input and output stay empty.
    #[must_use]
    pub fn without_text() -> Self {
        Self {
            text_off: true,
            ..Self::default()
        }
    }

    /// Drop every slot, tail, and prompt-rate baseline. The text mode stays,
    /// and so do the drop counts; held contexts are forgotten.
    pub fn clear(&mut self) {
        self.settle(|_| true);
        let mut ctx = std::mem::take(&mut self.ctx);
        for track in ctx.values_mut() {
            track.held = None;
        }
        *self = Self {
            text_off: self.text_off,
            prompt_view: self.prompt_view,
            ctx,
            activity_newest: self.activity_newest,
            ..Self::default()
        };
    }

    /// Track one slot's context (#10).
    fn note_ctx(&mut self, model_id: &str, slot: &LightSlot) {
        let key = (model_id.to_owned(), slot.id);
        if !self.ctx.contains_key(&key) && self.ctx.len() >= MAX_TRACKED_SLOTS {
            return;
        }
        let used = slot
            .ctx_prompt
            .map(|prompt| prompt.saturating_add(slot.n_decoded));
        let track = self.ctx.entry(key).or_default();
        let Some(left) = track.sample(used, slot.is_processing) else {
            return;
        };
        // A drop still waiting when the next one comes is decided now.
        if let Some(old) = &track.pending {
            let cached = old.slots_cached;
            let displaced = self.displaced.entry(model_id.to_owned()).or_default();
            track.decide(cached, displaced, Instant::now());
        }
        let prompt = if slot.n_prompt_tokens > 0 {
            slot.n_prompt_tokens
        } else {
            slot.ctx_prompt.unwrap_or(0)
        };
        track.pending = Some(Pending {
            left,
            prompt,
            slots_cached: slot.n_prompt_tokens_cache,
            after: self.activity_newest,
            since: None,
        });
    }

    /// Decide now every waiting drop of the models `gone` picks, on what
    /// `/slots` said: their slots are going away, and a drop is counted once.
    fn settle(&mut self, gone: impl Fn(&str) -> bool) {
        let now = Instant::now();
        for ((model_id, _), track) in &mut self.ctx {
            let Some(pending) = &track.pending else {
                continue;
            };
            if !gone(model_id) {
                continue;
            }
            let cached = pending.slots_cached;
            let displaced = self.displaced.entry(model_id.clone()).or_default();
            track.decide(cached, displaced, now);
        }
    }

    /// Feed one activity read (#9): a waiting drop takes the newer row of
    /// its model whose prompt matches its own, and one that waited
    /// [`PENDING_TTL`] is decided on what `/slots` said, or as unknown.
    /// A failed read passes no rows and still ages the waits.
    pub fn note_activity(&mut self, rows: &[ActivityRow], now: Instant) {
        if let Some(newest) = rows.iter().map(|row| row.id).max() {
            if self.activity_newest.is_some_and(|seen| newest < seen) {
                // llama-swap restarted: its ids start again.
                for track in self.ctx.values_mut() {
                    if let Some(pending) = &mut track.pending {
                        pending.after = None;
                    }
                }
            }
            self.activity_newest = Some(newest);
        }
        let mut taken: HashSet<i64> = HashSet::new();
        for ((model_id, _), track) in &mut self.ctx {
            let Some(pending) = &mut track.pending else {
                continue;
            };
            let since = *pending.since.get_or_insert(now);
            let key = model_key(model_id);
            let found = rows.iter().find(|row| {
                row.model == key
                    && pending.after.is_none_or(|after| row.id > after)
                    && !taken.contains(&row.id)
                    && row_matches(row, pending.prompt)
            });
            let cached = match found {
                Some(row) => {
                    taken.insert(row.id);
                    row.cached_tokens
                }
                None if now.saturating_duration_since(since) >= PENDING_TTL => pending.slots_cached,
                None => continue,
            };
            let displaced = self.displaced.entry(model_id.clone()).or_default();
            track.decide(cached, displaced, now);
        }
    }

    /// Ingest one model's body. A body that does not parse leaves this model
    /// unchanged and returns `false`.
    ///
    /// `prompt` is skipped with [`IgnoredAny`] when this slot's `id_task` was
    /// already captured, so a repeat poll does not allocate the prompt string.
    pub fn apply(
        &mut self,
        model_id: &str,
        display: &str,
        body: &[u8],
        input_tail: usize,
        output_tail: usize,
    ) -> bool {
        if self.text_off {
            let Some(parsed) = parse_numbers(body) else {
                return false;
            };
            self.round_seen = true;
            let next = parsed
                .iter()
                .map(|slot| {
                    self.note_processed(model_id, slot);
                    self.note_ctx(model_id, slot);
                    tracked(model_id, display, slot, Vec::new(), Vec::new(), None)
                })
                .collect::<Vec<_>>();
            self.slots.retain(|row| row.model_id != model_id);
            self.slots.extend(next);
            return true;
        }
        let Some(parsed) = parse_light(body) else {
            return false;
        };
        let prompts = if parsed
            .iter()
            .any(|slot| self.needs_prompt(model_id, slot.id, slot.id_task))
        {
            let Some(full) = parse_full(body) else {
                return false;
            };
            if full.len() != parsed.len() {
                return false;
            }
            full.into_iter().map(|slot| slot.prompt).collect()
        } else {
            vec![String::new(); parsed.len()]
        };
        self.round_seen = true;
        let mut next = Vec::with_capacity(parsed.len());
        for (slot, prompt) in parsed.iter().zip(prompts) {
            self.note_processed(model_id, slot);
            self.note_ctx(model_id, slot);

            let prev = self
                .slots
                .iter()
                .find(|row| row.model_id == model_id && row.id == slot.id);
            let mut captured_task = prev.and_then(|row| row.captured_task);
            let input = if captured_task == Some(slot.id_task) {
                prev.map(|row| row.input.clone()).unwrap_or_default()
            } else {
                captured_task = Some(slot.id_task);
                prompt_cells(&prompt, input_tail, self.prompt_view)
            };
            let output = tail_cells(&slot.generated, output_tail);
            next.push(tracked(
                model_id,
                display,
                slot,
                input,
                output,
                captured_task,
            ));
        }
        self.slots.retain(|row| row.model_id != model_id);
        self.slots.extend(next);
        true
    }

    fn note_processed(&mut self, model_id: &str, slot: &LightSlot) {
        let key = (model_id.to_owned(), slot.id);
        let previous = self.last_processed.get(&key).copied();
        self.round_delta = self
            .round_delta
            .saturating_add(counter_delta(previous, slot.n_prompt_tokens_processed));
        self.last_processed
            .insert(key, slot.n_prompt_tokens_processed);
    }

    fn needs_prompt(&self, model_id: &str, id: i64, id_task: i64) -> bool {
        let prev = self
            .slots
            .iter()
            .find(|row| row.model_id == model_id && row.id == id);
        prev.and_then(|row| row.captured_task) != Some(id_task)
    }

    /// Close the phase and store prompt tok/s. The first phase only sets a baseline.
    pub fn finish_round(&mut self, now: Instant) {
        if !self.round_seen {
            return;
        }
        if let Some(then) = self.last_at {
            let window = now.saturating_duration_since(then);
            self.prompt_tps = Some(delta_per_s(self.round_delta, window));
        }
        self.last_at = Some(now);
    }

    /// Drop slots whose model is no longer being polled. Their held
    /// context is forgotten; their drop counts stay.
    pub fn retain_models(&mut self, model_ids: &[&str]) {
        self.slots
            .retain(|row| model_ids.iter().any(|id| *id == row.model_id));
        self.settle(|model| !model_ids.contains(&model));
        for ((model, _), track) in &mut self.ctx {
            if !model_ids.iter().any(|id| id == model) {
                track.held = None;
            }
        }
        self.displaced
            .retain(|model, _| model_ids.iter().any(|id| id == model));
    }

    #[must_use]
    pub fn slots(&self) -> Vec<SlotView> {
        self.slots
            .iter()
            .map(|row| SlotView {
                model: row.model.clone(),
                id: row.id,
                id_task: row.id_task,
                is_processing: row.is_processing,
                n_prompt_tokens: row.n_prompt_tokens,
                n_prompt_tokens_processed: row.n_prompt_tokens_processed,
                n_decoded: row.n_decoded,
                n_ctx: row.n_ctx,
                ctx_prompt: row.ctx_prompt,
                input: row.input.clone(),
                output: row.output.clone(),
                ctx_used: self.ctx_of(row).and_then(|track| track.held),
                resets: self
                    .ctx_of(row)
                    .map(|track| track.resets)
                    .unwrap_or_default(),
                last_reset: self.ctx_of(row).and_then(|track| track.last),
            })
            .collect()
    }

    fn ctx_of(&self, row: &Tracked) -> Option<&CtxTrack> {
        self.ctx.get(&(row.model_id.clone(), row.id))
    }

    #[must_use]
    pub fn prompt_tps(&self) -> Option<f64> {
        self.prompt_tps
    }
}

fn tracked(
    model_id: &str,
    display: &str,
    slot: &LightSlot,
    input: Vec<Cell>,
    output: Vec<Cell>,
    captured_task: Option<i64>,
) -> Tracked {
    Tracked {
        model_id: model_id.to_owned(),
        model: display.to_owned(),
        id: slot.id,
        id_task: slot.id_task,
        is_processing: slot.is_processing,
        n_prompt_tokens: slot.n_prompt_tokens,
        n_prompt_tokens_processed: slot.n_prompt_tokens_processed,
        n_decoded: slot.n_decoded,
        n_ctx: slot.n_ctx,
        ctx_prompt: slot.ctx_prompt,
        input,
        output,
        captured_task,
    }
}

struct LightSlot {
    id: i64,
    id_task: i64,
    is_processing: bool,
    n_prompt_tokens: u64,
    n_prompt_tokens_processed: u64,
    n_decoded: u64,
    n_ctx: Option<u64>,
    ctx_prompt: Option<u64>,
    /// `n_prompt_tokens_cache`, when the server sends it (#9 evidence).
    n_prompt_tokens_cache: Option<u64>,
    generated: String,
}

struct FullSlot {
    prompt: String,
}

fn parse_light(body: &[u8]) -> Option<Vec<LightSlot>> {
    let slots: Vec<SlotJsonLight> = serde_json::from_slice(body).ok()?;
    Some(slots.into_iter().map(LightSlot::from).collect())
}

/// Numbers only: `prompt` and `generated` are skipped, never allocated.
fn parse_numbers(body: &[u8]) -> Option<Vec<LightSlot>> {
    let slots: Vec<SlotJsonNumbers> = serde_json::from_slice(body).ok()?;
    Some(
        slots
            .into_iter()
            .map(|slot| LightSlot {
                id: slot.id,
                id_task: slot.id_task,
                is_processing: slot.is_processing,
                n_prompt_tokens: slot.n_prompt_tokens.unwrap_or(0),
                n_prompt_tokens_processed: slot.n_prompt_tokens_processed,
                n_decoded: slot
                    .next_token
                    .first()
                    .map(|token| token.n_decoded)
                    .unwrap_or(0),
                n_ctx: slot.n_ctx,
                ctx_prompt: slot.n_prompt_tokens.or(slot.n_prompt_tokens_cache),
                n_prompt_tokens_cache: slot.n_prompt_tokens_cache,
                generated: String::new(),
            })
            .collect(),
    )
}

fn parse_full(body: &[u8]) -> Option<Vec<FullSlot>> {
    let slots: Vec<SlotJsonFull> = serde_json::from_slice(body).ok()?;
    Some(
        slots
            .into_iter()
            .map(|slot| FullSlot {
                prompt: slot.prompt,
            })
            .collect(),
    )
}

impl From<SlotJsonLight> for LightSlot {
    fn from(slot: SlotJsonLight) -> Self {
        let _ = slot.prompt;
        Self {
            id: slot.id,
            id_task: slot.id_task,
            is_processing: slot.is_processing,
            n_prompt_tokens: slot.n_prompt_tokens.unwrap_or(0),
            n_prompt_tokens_processed: slot.n_prompt_tokens_processed,
            n_decoded: slot
                .next_token
                .first()
                .map(|token| token.n_decoded)
                .unwrap_or(0),
            n_ctx: slot.n_ctx,
            ctx_prompt: slot.n_prompt_tokens.or(slot.n_prompt_tokens_cache),
            n_prompt_tokens_cache: slot.n_prompt_tokens_cache,
            generated: slot.generated,
        }
    }
}

/// The prompt tail as IN shows it. Clean mode strips template tokens from
/// the raw tail first; [`sanitize`] runs last either way (S12).
pub(crate) fn prompt_cells(text: &str, max_chars: usize, view: PromptView) -> Vec<Cell> {
    match view {
        PromptView::Raw => tail_cells(text, max_chars),
        PromptView::Clean => {
            let tail = tail_str(text, max_chars);
            let cleaned = clean(tail, tail.len() < text.len());
            let mut out = Vec::new();
            sanitize(&cleaned, &mut out);
            out
        }
    }
}

pub(crate) fn tail_cells(text: &str, max_chars: usize) -> Vec<Cell> {
    let mut out = Vec::new();
    sanitize(tail_str(text, max_chars), &mut out);
    out
}

fn tail_str(text: &str, max_chars: usize) -> &str {
    let count = text.chars().count();
    if count <= max_chars {
        return text;
    }
    let skip = count - max_chars;
    let start = text
        .char_indices()
        .nth(skip)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    &text[start..]
}

#[derive(Debug, Deserialize)]
struct SlotJsonLight {
    #[serde(default)]
    id: i64,
    #[serde(default = "missing_task")]
    id_task: i64,
    #[serde(default)]
    is_processing: bool,
    #[serde(default)]
    n_prompt_tokens: Option<u64>,
    /// Cached prompt tokens (`n_prompt_tokens_cache` on llama-server `/slots`).
    #[serde(default)]
    n_prompt_tokens_cache: Option<u64>,
    /// Context window for this slot. Absent stays `None`; a present 0 stays `Some(0)`.
    #[serde(default)]
    n_ctx: Option<u64>,
    #[serde(default)]
    n_prompt_tokens_processed: u64,
    #[serde(default)]
    next_token: Vec<NextTokenJson>,
    #[serde(default)]
    prompt: IgnoredAny,
    #[serde(default)]
    generated: String,
}

/// [`SlotJsonLight`] without `prompt` or `generated`. Used when text is off.
#[derive(Debug, Deserialize)]
struct SlotJsonNumbers {
    #[serde(default)]
    id: i64,
    #[serde(default = "missing_task")]
    id_task: i64,
    #[serde(default)]
    is_processing: bool,
    #[serde(default)]
    n_prompt_tokens: Option<u64>,
    #[serde(default)]
    n_prompt_tokens_cache: Option<u64>,
    #[serde(default)]
    n_ctx: Option<u64>,
    #[serde(default)]
    n_prompt_tokens_processed: u64,
    #[serde(default)]
    next_token: Vec<NextTokenJson>,
    // No `prompt` or `generated` field: serde skips undeclared keys without
    // building their strings.
}

#[derive(Debug, Deserialize)]
struct SlotJsonFull {
    #[serde(default)]
    prompt: String,
}

fn missing_task() -> i64 {
    -1
}

#[derive(Debug, Deserialize)]
struct NextTokenJson {
    #[serde(default)]
    n_decoded: u64,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn cells(cells: &[Cell]) -> String {
        cells.iter().map(|cell| cell.ch).collect()
    }

    fn body(id_task: i64, prompt: &str, generated: &str, processed: u64, decoded: u64) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": id_task,
            "is_processing": true,
            "n_prompt_tokens": 10,
            "n_prompt_tokens_processed": processed,
            "next_token": [{"n_decoded": decoded, "has_next_token": true, "stopped_eos": false}],
            "prompt": prompt,
            "generated": generated,
            "grammar": "DROP_ME_GRAMMAR",
        }]))
        .expect("json")
    }

    #[test]
    fn slots_tail_truncates_then_sanitises() {
        let mut book = SlotBook::default();
        let prompt = format!("{}{}", "A".repeat(300), "\u{1b}[2JTAIL");
        let generated = format!("{}{}", "B".repeat(300), "OUT\n");
        let bytes = body(4, &prompt, &generated, 3, 2);
        assert!(book.apply("m", "Model", &bytes, 256, 256));
        let slots = book.slots();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].n_decoded, 2);
        assert_eq!(slots[0].n_prompt_tokens_processed, 3);
        let mut expect_in = Vec::new();
        sanitize(tail_str(&prompt, 256), &mut expect_in);
        let mut expect_out = Vec::new();
        sanitize(tail_str(&generated, 256), &mut expect_out);
        assert_eq!(slots[0].input, expect_in);
        assert_eq!(slots[0].output, expect_out);
        let input = cells(&slots[0].input);
        assert!(input.ends_with("TAIL"), "{input}");
        assert!(
            !input
                .chars()
                .any(|ch| ch == '\u{1b}' || (ch != '\n' && ch < ' '))
        );
        assert!(!format!("{slots:?}").contains("DROP_ME_GRAMMAR"));
    }

    #[test]
    fn slots_input_is_taken_once_per_task() {
        let mut book = SlotBook::default();
        assert!(book.apply(
            "m",
            "Model",
            &body(4, "PROMPT-ONE-UNIQUE", "gen-1", 1, 1),
            256,
            256
        ));
        assert!(book.apply(
            "m",
            "Model",
            &body(4, "PROMPT-TWO-UNIQUE", "gen-2", 2, 2),
            256,
            256
        ));
        let slots = book.slots();
        let input = cells(&slots[0].input);
        let output = cells(&slots[0].output);
        assert!(input.contains("PROMPT-ONE-UNIQUE"), "{input}");
        assert!(!input.contains("PROMPT-TWO-UNIQUE"), "{input}");
        assert!(output.contains("gen-2"), "{output}");
        assert!(!output.contains("gen-1"), "{output}");

        assert!(book.apply(
            "m",
            "Model",
            &body(5, "PROMPT-THREE", "gen-3", 1, 1),
            256,
            256
        ));
        let slots = book.slots();
        let input = cells(&slots[0].input);
        assert!(input.contains("PROMPT-THREE"), "{input}");
        assert_eq!(slots[0].id_task, 5);
    }

    #[test]
    fn slots_repeat_task_keeps_input_when_prompt_is_not_a_string() {
        let mut book = SlotBook::default();
        assert!(book.apply(
            "m",
            "Model",
            &body(4, "KEEP-PROMPT", "gen-1", 1, 1),
            256,
            256
        ));
        let second = serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": 4,
            "is_processing": true,
            "n_prompt_tokens": 10,
            "n_prompt_tokens_processed": 4,
            "next_token": [{"n_decoded": 3}],
            "prompt": 1,
            "generated": "gen-2"
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &second, 256, 256));
        let slots = book.slots();
        let input = cells(&slots[0].input);
        let output = cells(&slots[0].output);
        assert!(input.contains("KEEP-PROMPT"), "{input}");
        assert!(output.contains("gen-2"), "{output}");
        assert!(!output.contains("gen-1"), "{output}");
        assert_eq!(slots[0].n_decoded, 3);
    }

    #[test]
    fn slots_ctx_prefers_prompt_tokens_over_the_cache_and_keeps_n_ctx() {
        let mut book = SlotBook::default();
        let bytes = serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": 7,
            "is_processing": true,
            "n_ctx": 262_144,
            "n_prompt_tokens": 91_000,
            "n_prompt_tokens_cache": 80_000,
            "n_prompt_tokens_processed": 91_000,
            "next_token": [{"n_decoded": 816}],
            "prompt": "p",
            "generated": "g",
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &bytes, 32, 32));
        let slots = book.slots();
        assert_eq!(slots[0].n_ctx, Some(262_144));
        assert_eq!(slots[0].ctx_prompt, Some(91_000));
        assert_eq!(slots[0].n_decoded, 816);
    }

    #[test]
    fn slots_ctx_uses_cached_prompt_tokens_when_the_prompt_count_is_absent() {
        let mut book = SlotBook::default();
        let bytes = serde_json::to_vec(&serde_json::json!([{
            "id": 1,
            "id_task": 3,
            "is_processing": true,
            "n_ctx": 4096,
            "n_prompt_tokens_cache": 80,
            "n_prompt_tokens_processed": 0,
            "next_token": [{"n_decoded": 11}],
            "prompt": "p",
            "generated": "g",
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &bytes, 32, 32));
        let slots = book.slots();
        assert_eq!(slots[0].ctx_prompt, Some(80));
        assert_eq!(slots[0].n_ctx, Some(4096));
        assert_eq!(slots[0].n_prompt_tokens, 0);
    }

    #[test]
    fn slots_ctx_keeps_a_zero_n_ctx_and_leaves_missing_keys_absent() {
        let mut book = SlotBook::default();
        let zero = serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": 1,
            "is_processing": false,
            "n_ctx": 0,
            "n_prompt_tokens": 12,
            "next_token": [{"n_decoded": 0}],
            "prompt": "p",
            "generated": "",
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &zero, 32, 32));
        assert_eq!(book.slots()[0].n_ctx, Some(0));
        assert_eq!(book.slots()[0].ctx_prompt, Some(12));

        let missing = serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": 2,
            "is_processing": false,
            "prompt": "p",
            "generated": "",
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &missing, 32, 32));
        let slots = book.slots();
        assert_eq!(slots[0].n_ctx, None);
        assert_eq!(slots[0].ctx_prompt, None);
        assert_eq!(slots[0].n_prompt_tokens, 0);
        assert_eq!(slots[0].n_decoded, 0);
    }

    #[test]
    fn text_off_keeps_numbers_and_never_the_prompt_or_generated_text() {
        let mut book = SlotBook::without_text();
        let bytes = serde_json::to_vec(&serde_json::json!([{
            "id": 0,
            "id_task": 7,
            "is_processing": true,
            "n_ctx": 262_144,
            "n_prompt_tokens": 91_000,
            "n_prompt_tokens_processed": 91_000,
            "next_token": [{"n_decoded": 816}],
            "prompt": "SECRET-PROMPT-T45",
            "generated": "SECRET-OUTPUT-T45",
        }]))
        .expect("json");
        assert!(book.apply("m", "Model", &bytes, 256, 256));
        // A second poll of the same task and a new task: still no text.
        assert!(book.apply("m", "Model", &bytes, 256, 256));
        assert!(book.apply(
            "m",
            "Model",
            &body(8, "SECRET-PROMPT-T45", "SECRET-OUTPUT-T45", 5, 1),
            256,
            256
        ));
        let slots = book.slots();
        assert_eq!(slots.len(), 1);
        assert!(slots[0].input.is_empty(), "{:?}", slots[0].input);
        assert!(slots[0].output.is_empty(), "{:?}", slots[0].output);
        assert!(slots[0].is_processing);
        assert_eq!(slots[0].id_task, 8);
        assert_eq!(slots[0].n_decoded, 1);
        let kept = format!("{book:?}{slots:?}");
        assert!(!kept.contains("SECRET"), "{kept}");

        let mut ctx = SlotBook::without_text();
        assert!(ctx.apply("m", "Model", &bytes, 256, 256));
        let slots = ctx.slots();
        assert_eq!(slots[0].n_ctx, Some(262_144));
        assert_eq!(slots[0].ctx_prompt, Some(91_000));
        assert_eq!(slots[0].n_decoded, 816);
    }

    const QWEN_TAIL: &str = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

    #[test]
    fn clean_prompt_view_strips_template_tokens_before_sanitising() {
        let mut book = SlotBook::default();
        assert!(book.apply("m", "Model", &body(1, QWEN_TAIL, "g", 1, 1), 8192, 8192));
        let input = cells(&book.slots()[0].input);
        assert_eq!(input, "-- system --\nBe brief.\n-- user --\nhi");

        let mut raw = SlotBook::with_prompt_view(PromptView::Raw);
        assert!(raw.apply("m", "Model", &body(1, QWEN_TAIL, "g", 1, 1), 8192, 8192));
        let input = cells(&raw.slots()[0].input);
        assert_eq!(input, QWEN_TAIL);
    }

    #[test]
    fn clean_prompt_view_still_sanitises_last() {
        let mut book = SlotBook::default();
        let prompt =
            "<|im_start|>user\n\u{1b}[2Jhi\u{9b}31m\u{e9}<|im_end|>\n<|im_start|>assistant\n";
        assert!(book.apply("m", "Model", &body(1, prompt, "g", 1, 1), 8192, 8192));
        let input = cells(&book.slots()[0].input);
        assert_eq!(input, "-- user --\n[2Jhi31m?");
    }

    fn ctx_body(id: i64, id_task: i64, busy: bool, prompt: u64, decoded: u64) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!([{
            "id": id,
            "id_task": id_task,
            "is_processing": busy,
            "n_ctx": 262_144,
            "n_prompt_tokens": prompt,
            "n_prompt_tokens_processed": prompt,
            "next_token": [{"n_decoded": decoded}],
            "prompt": "INVENTED-PROMPT",
            "generated": "INVENTED-OUTPUT",
        }]))
        .expect("json")
    }

    fn ctx_of(book: &SlotBook, id: i64) -> (Option<u64>, u64) {
        let slot = book
            .slots()
            .into_iter()
            .find(|slot| slot.id == id)
            .expect("slot");
        (slot.ctx_used, slot.resets.total())
    }

    /// Let every waiting drop time out.
    fn settle(book: &mut SlotBook) {
        let now = Instant::now();
        book.note_activity(&[], now);
        book.note_activity(&[], now + PENDING_TTL);
    }

    /// #10: the held context and drop count, by the sparkline's rule, with
    /// text on and off.
    #[test]
    fn slot_ctx_holds_idle_context_and_counts_drops() {
        for mut book in [SlotBook::default(), SlotBook::without_text()] {
            assert!(book.apply("m", "M", &ctx_body(0, 1, true, 80_000, 500), 64, 64));
            assert_eq!(ctx_of(&book, 0), (Some(80_500), 0));
            // Idle reads 0: the cache is held, no drop.
            assert!(book.apply("m", "M", &ctx_body(0, 1, false, 0, 0), 64, 64));
            assert_eq!(ctx_of(&book, 0), (Some(80_500), 0));
            // A small prune is not a drop; a compaction is.
            assert!(book.apply("m", "M", &ctx_body(0, 2, true, 70_000, 0), 64, 64));
            assert_eq!(ctx_of(&book, 0), (Some(70_000), 0));
            assert!(book.apply("m", "M", &ctx_body(0, 3, true, 20_000, 0), 64, 64));
            settle(&mut book);
            assert_eq!(ctx_of(&book, 0), (Some(20_000), 1));
            // A drop still waiting is decided when the next one comes.
            assert!(book.apply("m", "M", &ctx_body(0, 4, true, 12_000, 0), 64, 64));
            assert!(book.apply("m", "M", &ctx_body(0, 5, true, 1_000, 0), 64, 64));
            assert_eq!(ctx_of(&book, 0), (Some(1_000), 2));
            settle(&mut book);
            assert_eq!(ctx_of(&book, 0), (Some(1_000), 3));
        }
    }

    #[test]
    fn slot_ctx_counts_survive_clear_and_unload_but_the_context_does_not() {
        let mut book = SlotBook::default();
        assert!(book.apply("m", "M", &ctx_body(0, 1, true, 80_000, 0), 64, 64));
        assert!(book.apply("m", "M", &ctx_body(0, 2, true, 10_000, 0), 64, 64));
        assert_eq!(
            ctx_of(&book, 0),
            (Some(10_000), 0),
            "waiting for its reason"
        );
        // An unload decides the waiting drop: it is still counted once.
        book.retain_models(&["other"]);
        assert!(book.slots().is_empty());
        // The model comes back: the count carries on, the context is new,
        // and its first read is not a drop.
        assert!(book.apply("m", "M", &ctx_body(0, 1, true, 3_000, 0), 64, 64));
        assert_eq!(ctx_of(&book, 0), (Some(3_000), 1));
        book.clear();
        assert!(book.apply("m", "M", &ctx_body(0, 1, true, 50, 0), 64, 64));
        assert_eq!(ctx_of(&book, 0), (Some(50), 1));
        // Tracking is bounded.
        for id in 0..(MAX_TRACKED_SLOTS as i64 + 10) {
            assert!(book.apply("m", "M", &ctx_body(id, 1, true, 10, 0), 64, 64));
        }
        assert_eq!(book.ctx.len(), MAX_TRACKED_SLOTS);
    }

    fn row(id: i64, model: &str, input: u64, cache: Option<u64>) -> ActivityRow {
        ActivityRow {
            id,
            time: String::new(),
            source: String::new(),
            model: model.to_owned(),
            input_tokens: Some(input),
            cached_tokens: cache,
            output_tokens: Some(10),
            prompt_tps: None,
            gen_tps: None,
            duration_ms: None,
            status: Some(200),
            captured: false,
        }
    }

    fn reasons(book: &SlotBook, id: i64) -> (ResetCounts, Option<ResetReason>) {
        let slot = book
            .slots()
            .into_iter()
            .find(|slot| slot.id == id)
            .expect("slot");
        (slot.resets, slot.last_reset)
    }

    fn counts(compacted: u64, new: u64, evicted: u64, unknown: u64) -> ResetCounts {
        ResetCounts {
            compacted,
            new,
            evicted,
            unknown,
        }
    }

    /// #9: a synthetic multi-agent story on two slots, text off (no prompt
    /// text is needed to decide).
    #[test]
    fn drops_are_classified_from_the_matching_activity_row() {
        let mut book = SlotBook::without_text();
        let t0 = Instant::now();
        book.note_activity(&[row(10, "m", 5, Some(0))], t0);
        // Agent A grows to 80k in slot 0, then compacts to 20k: 14k of the
        // new prompt was cached.
        assert!(book.apply("m", "M", &ctx_body(0, 1, true, 80_000, 0), 64, 64));
        assert!(book.apply("m", "M", &ctx_body(0, 2, true, 20_000, 0), 64, 64));
        assert_eq!(reasons(&book, 0), (ResetCounts::default(), None));
        // Rows that are not this request: older, another model, another size.
        let noise = [
            row(10, "m", 6_000, Some(14_000)),
            row(11, "other", 6_000, Some(14_000)),
            row(12, "m", 9_000, Some(14_000)),
        ];
        book.note_activity(&noise, t0);
        assert_eq!(reasons(&book, 0).0.total(), 0);
        book.note_activity(&[row(13, "m", 6_000, Some(14_000))], t0);
        assert_eq!(
            reasons(&book, 0),
            (counts(1, 0, 0, 0), Some(ResetReason::Compacted))
        );

        // Agent B takes slot 0 from A (20k): a new conversation, nothing
        // cached. A's 20k is remembered as lost.
        assert!(book.apply("m", "M", &ctx_body(0, 3, true, 22_000, 0), 64, 64));
        assert!(book.apply("m", "M", &ctx_body(0, 4, true, 3_000, 0), 64, 64));
        book.note_activity(&[row(14, "m", 2_990, Some(10))], t0);
        assert_eq!(
            reasons(&book, 0),
            (counts(1, 1, 0, 0), Some(ResetReason::New))
        );

        // Slot 1 holds C at 60k; A comes back there at 23k with ~0 cached:
        // its cache was lost, the whole prompt is processed again.
        assert!(book.apply("m", "M", &ctx_body(1, 5, true, 60_000, 0), 64, 64));
        assert!(book.apply("m", "M", &ctx_body(1, 6, true, 23_000, 0), 64, 64));
        book.note_activity(&[row(15, "m", 22_800, Some(200))], t0);
        assert_eq!(
            reasons(&book, 1),
            (counts(0, 0, 1, 0), Some(ResetReason::Evicted))
        );
        // Slot 0 is untouched by slot 1's row.
        assert_eq!(book.ctx[&("m".to_owned(), 0)].resets, counts(1, 1, 0, 0));
    }

    #[test]
    fn a_drop_with_no_row_uses_slots_cache_or_is_unknown_after_the_wait() {
        let mut book = SlotBook::default();
        let t0 = Instant::now();
        let with_cache = |id_task: i64, prompt: u64, cache: Option<u64>| {
            let mut slot = serde_json::json!({
                "id": 0, "id_task": id_task, "is_processing": true,
                "n_ctx": 262_144, "n_prompt_tokens": prompt,
                "n_prompt_tokens_processed": prompt,
                "next_token": [{"n_decoded": 0}],
                "prompt": "INVENTED", "generated": "INVENTED",
            });
            if let Some(cache) = cache {
                slot["n_prompt_tokens_cache"] = cache.into();
            }
            serde_json::to_vec(&serde_json::json!([slot])).expect("json")
        };
        assert!(book.apply("m", "M", &with_cache(1, 90_000, None), 64, 64));
        assert!(book.apply("m", "M", &with_cache(2, 30_000, Some(25_000)), 64, 64));
        book.note_activity(&[], t0);
        book.note_activity(&[], t0 + PENDING_TTL - Duration::from_secs(1));
        assert_eq!(reasons(&book, 0).0.total(), 0, "still waiting");
        book.note_activity(&[], t0 + PENDING_TTL);
        assert_eq!(
            reasons(&book, 0),
            (counts(1, 0, 0, 0), Some(ResetReason::Compacted))
        );
        // No cache count anywhere: unknown.
        assert!(book.apply("m", "M", &with_cache(3, 90_000, None), 64, 64));
        assert!(book.apply("m", "M", &with_cache(4, 1_000, None), 64, 64));
        let t1 = t0 + PENDING_TTL * 2;
        book.note_activity(&[], t1);
        book.note_activity(&[], t1 + PENDING_TTL);
        assert_eq!(
            reasons(&book, 0),
            (counts(1, 0, 0, 1), Some(ResetReason::Unknown))
        );
    }

    #[test]
    fn slots_prompt_rate_is_delta_per_second() {
        let mut book = SlotBook::default();
        let t0 = Instant::now();
        book.begin_round();
        assert!(book.apply("m", "Model", &body(1, "p", "g", 10, 0), 256, 256));
        book.finish_round(t0);
        assert_eq!(book.prompt_tps(), None);

        book.begin_round();
        assert!(book.apply("m", "Model", &body(1, "p", "g", 30, 0), 256, 256));
        book.finish_round(t0 + Duration::from_secs(1));
        let rate = book.prompt_tps().expect("rate");
        assert!((rate - 20.0).abs() < 1e-6, "{rate}");

        book.begin_round();
        assert!(book.apply("m", "Model", &body(2, "p", "g", 5, 0), 256, 256));
        book.finish_round(t0 + Duration::from_secs(2));
        let reset = book.prompt_tps().expect("reset rate");
        assert!((reset - 5.0).abs() < 1e-6, "{reset}");
    }
}
