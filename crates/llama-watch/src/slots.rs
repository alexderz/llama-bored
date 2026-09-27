//! `/upstream/<model>/slots`: declared fields, tail text, prompt rate.

use std::collections::HashMap;
use std::time::Instant;

use llama_core::rate::{counter_delta, delta_per_s};
use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::config::PromptView;
use crate::tty::chat_template::clean;
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

    /// Drop every slot, tail, and prompt-rate baseline. The text mode stays.
    pub fn clear(&mut self) {
        *self = Self {
            text_off: self.text_off,
            prompt_view: self.prompt_view,
            ..Self::default()
        };
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

    /// Drop slots whose model is no longer being polled.
    pub fn retain_models(&mut self, model_ids: &[&str]) {
        self.slots
            .retain(|row| model_ids.iter().any(|id| *id == row.model_id));
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
            })
            .collect()
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
            generated: slot.generated,
        }
    }
}

/// The prompt tail as IN shows it. Clean mode strips template tokens from
/// the raw tail first; [`sanitize`] runs last either way (S12).
fn prompt_cells(text: &str, max_chars: usize, view: PromptView) -> Vec<Cell> {
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

fn tail_cells(text: &str, max_chars: usize) -> Vec<Cell> {
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
