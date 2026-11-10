//! IN and OUT scroll for a burst of text (#188).
//!
//! A burst is a whole prompt or reply that arrives in one update. The first
//! frame shows its start, then it moves down by elapsed time. The speed is
//! read every frame: the live tok/s when one is known, otherwise the average
//! of the last five completed turns of that model, then this turn's own
//! measured speed, then the whole text in two seconds. A live rate that
//! appears after the burst has started takes over, and one that disappears
//! falls back the same way. Either change keeps the characters already
//! shown, so the text does not jump.
//!
//! Characters per second are tok/s times the characters per token of the
//! whole text the panel shows a tail of: its full length over its token
//! count, else four. The panel's own text is capped (`input_tail_chars`),
//! so its length over the whole prompt's tokens would crawl: a cached
//! 32k-token prompt with an 8k-character tail moved at a quarter character
//! per token, crossing in the time a cold prefill of the whole prompt
//! takes instead of the time its tail takes (DER-451).
//!
//! IN crosses only what this request appended (DER-451): the burst starts
//! at the first new character, the text before it already passed. The new
//! part is the capture's message diff when the poller has one, else the
//! uncached tokens (`input − cached`) at the characters per token above,
//! counted back from the end, else the whole text. IN moves at prefill
//! speed, so with no live, history or own rate it uses a fast
//! [`FAST_PREFILL_TPS`] instead of the two-second crossing, and the first
//! real rate to appear takes over without a jump. OUT keeps both rules.
//!
//! A streaming update (the text grows while it is generated) keeps the tail
//! and does not jump back to the top. Text shorter than the panel is the
//! layout's concern: a scroll of zero starts at the top, and a window that
//! fits is not moved.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

/// Completed turns remembered per model.
const HISTORY: usize = 5;
/// How long a first turn with no speed takes to cross its own text.
const FALLBACK_SECS: f64 = 2.0;
/// Characters per token when the turn has no token count.
const DEFAULT_CHARS_PER_TOKEN: f64 = 4.0;
/// Prefill tok/s IN assumes until a measured rate is known (DER-451). A
/// prefill usually ends before a slow scroll would, so this is fast: an
/// 8,192-character tail crosses in about a second.
pub const FAST_PREFILL_TPS: f64 = 2_000.0;
/// Activity rows remembered so a row is counted once.
const SEEN_MAX: usize = 512;

/// One finished turn's speeds, as RECENT has them.
#[derive(Clone, Debug)]
pub struct Turn {
    pub seq: u64,
    /// Activity model key.
    pub model: String,
    /// Prefill tok/s, llama-swap's or the engine's.
    pub prompt_tps: Option<f64>,
    /// Generation tok/s, llama-swap's or the engine's.
    pub gen_tps: Option<f64>,
}

/// What the panels are showing this frame.
#[derive(Clone, Debug)]
pub struct Sample<'a> {
    /// Activity model key of the text on screen.
    pub model: &'a str,
    /// Request identity. A new id is a new turn. 0 is no request.
    pub request: u64,
    pub input: &'a [char],
    pub output: &'a [char],
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Characters of the whole prompt that [`Self::input`] is a tail of,
    /// when known. Not the length of `input`, which is capped.
    pub input_chars: Option<usize>,
    /// Characters of the whole reply that [`Self::output`] is a tail of.
    pub output_chars: Option<usize>,
    /// Of [`Self::input_tokens`], those the engine reused from its cache.
    pub input_cached: Option<u64>,
    /// Characters at the end of [`Self::input`] that this request
    /// appended, when the capture's messages tell (DER-451).
    pub input_new: Option<usize>,
    /// Live prefill tok/s (`/slots` or tqLLM). Not a held burst speed.
    pub live_prompt: Option<f64>,
    /// Live generation tok/s. Not a wall-clock spike from a burst engine.
    pub live_gen: Option<f64>,
    /// Activity rows since the watcher started. Already-seen seqs are skipped.
    pub rows: &'a [Turn],
}

/// Where each panel is. `None` is the streaming tail. `Some(n)` skips `n`
/// characters from the start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scroll {
    pub inn: Option<usize>,
    pub out: Option<usize>,
    /// Characters per second IN is moving, when it is a burst.
    pub in_cps: Option<f64>,
    /// Characters per second OUT is moving, when it is a burst.
    pub out_cps: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct Rates {
    prompt: VecDeque<f64>,
    decode: VecDeque<f64>,
    current_prompt: Option<f64>,
    current_gen: Option<f64>,
}

impl Rates {
    fn commit_current(&mut self) {
        push(&mut self.prompt, self.current_prompt);
        push(&mut self.decode, self.current_gen);
        self.current_prompt = None;
        self.current_gen = None;
    }

    fn note_current(&mut self, prompt: Option<f64>, rate: Option<f64>) {
        if prompt.is_some() {
            self.current_prompt = prompt;
        }
        if rate.is_some() {
            self.current_gen = rate;
        }
    }
}

fn push(history: &mut VecDeque<f64>, rate: Option<f64>) {
    let Some(rate) = positive(rate) else {
        return;
    };
    history.push_back(rate);
    while history.len() > HISTORY {
        history.pop_front();
    }
}

fn average(history: &VecDeque<f64>) -> Option<f64> {
    if history.is_empty() {
        return None;
    }
    let start = history.len().saturating_sub(HISTORY);
    let window = history.iter().skip(start);
    let n = history.len() - start;
    Some(window.sum::<f64>() / n as f64)
}

/// Live rate, else the last five completed turns, else this turn's own.
fn choose(live: Option<f64>, history: &VecDeque<f64>, own: Option<f64>) -> Option<f64> {
    positive(live)
        .or_else(|| average(history))
        .or_else(|| positive(own))
}

fn positive(rate: Option<f64>) -> Option<f64> {
    rate.filter(|rate| rate.is_finite() && *rate > 0.0)
}

/// Characters per token of a text of `whole` characters and `tokens`
/// tokens, else [`DEFAULT_CHARS_PER_TOKEN`].
fn per_token(whole: Option<usize>, tokens: Option<u64>) -> f64 {
    match (whole, tokens) {
        (Some(chars), Some(n)) if chars > 0 && n > 0 => chars as f64 / n as f64,
        _ => DEFAULT_CHARS_PER_TOKEN,
    }
}

/// `shown` is the panel's text, `whole` the full text it is a tail of. Its
/// tokens are the whole text's, so only `whole` is divided by them.
fn chars_per_sec(shown: usize, whole: Option<usize>, tokens: Option<u64>, tps: Option<f64>) -> f64 {
    let Some(tps) = tps else {
        return shown as f64 / FALLBACK_SECS;
    };
    tps * per_token(whole, tokens)
}

/// Uncached tokens past twice the message diff's that still read as the
/// diff (DER-451). An agent turn's uncached count also holds the re-sent
/// tool call, template tokens and the cache's block rounding: up to about
/// 2,000 tokens over its new messages on real turns. A cache miss is a
/// whole prompt.
const MISS_SLACK_TOKENS: f64 = 4_096.0;

/// Characters at the end of IN's `shown` text that this request appended
/// (DER-451): the message diff, else the uncached tokens at the text's
/// characters per token, else all of it. When the uncached tokens are far
/// more than the diff's (a cache miss: evicted, reloaded, or a rewritten
/// history), the larger estimate wins, so a miss is not under-scrolled.
fn new_input(
    shown: usize,
    diff: Option<usize>,
    whole: Option<usize>,
    tokens: Option<u64>,
    cached: Option<u64>,
) -> usize {
    let per = per_token(whole, tokens);
    let uncached = match (tokens, cached) {
        (Some(tokens), Some(cached)) if tokens > 0 => Some(tokens.saturating_sub(cached)),
        _ => None,
    };
    let estimate = uncached.map(|uncached| {
        let chars = uncached as f64 * per;
        if chars.is_finite() && chars < shown as f64 {
            chars.ceil() as usize
        } else {
            shown
        }
    });
    match (diff, uncached, estimate) {
        (Some(new), Some(uncached), Some(estimate)) => {
            let new = new.min(shown);
            if uncached as f64 > 2.0 * (new as f64 / per) + MISS_SLACK_TOKENS {
                estimate.max(new)
            } else {
                new
            }
        }
        (Some(new), ..) => new.min(shown),
        (None, _, Some(estimate)) => estimate,
        _ => shown,
    }
}

/// `next` continues `prev`: it grew in place, including a head the cap trimmed.
fn continues(prev: &[char], next: &[char]) -> bool {
    if prev.is_empty() || next.len() <= 1 {
        return false;
    }
    if next.starts_with(prev) {
        return next.len() > prev.len();
    }
    let max = prev.len().min(next.len());
    for len in (1..max).rev() {
        if prev[prev.len() - len..] == next[..len] {
            return next.len() > len;
        }
    }
    false
}

/// Speed inputs for one panel of one frame.
struct Pace<'a> {
    tokens: Option<u64>,
    /// Characters of the whole text, when known.
    chars: Option<usize>,
    /// Characters at the end of the text to cross; `None` is all of it.
    new: Option<usize>,
    /// Tok/s when no rate is known; `None` crosses in [`FALLBACK_SECS`].
    default_tps: Option<f64>,
    live: Option<f64>,
    own: Option<f64>,
    history: &'a VecDeque<f64>,
}

#[derive(Clone, Debug)]
struct Panel {
    request: u64,
    text: Vec<char>,
    mode: Mode,
}

#[derive(Clone, Debug)]
enum Mode {
    Tail,
    /// `passed` characters had already gone by at `at`. A rate change moves
    /// `at` to now and stores those characters, so the text does not jump.
    Burst {
        at: Instant,
        passed: f64,
        chars_per_sec: f64,
    },
}

fn same_rate(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1e-6
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            request: 0,
            text: Vec::new(),
            mode: Mode::Tail,
        }
    }
}

impl Panel {
    fn advance(&mut self, request: u64, text: &[char], pace: Pace<'_>, now: Instant) {
        if self.request == request && continues(&self.text, text) {
            self.text = text.to_vec();
            self.mode = Mode::Tail;
            return;
        }
        // A new request often still holds the previous reply until its own
        // output arrives. That leftover is not this turn's OUT (#188).
        if request != self.request && self.text.as_slice() == text && !text.is_empty() {
            self.request = request;
            return;
        }
        let tps = choose(pace.live, pace.history, pace.own).or(pace.default_tps);
        let cps = chars_per_sec(text.len(), pace.chars, pace.tokens, tps);
        if self.request == request && self.text.as_slice() == text {
            self.retarget(cps, now);
            return;
        }
        self.request = request;
        self.text = text.to_vec();
        if text.is_empty() {
            self.mode = Mode::Tail;
            return;
        }
        // Only the new part crosses: what came before it has passed.
        let passed = text.len() - pace.new.unwrap_or(text.len()).min(text.len());
        self.mode = Mode::Burst {
            at: now,
            passed: passed as f64,
            chars_per_sec: cps,
        };
    }

    /// Keep the characters already shown when the speed changes.
    fn retarget(&mut self, cps: f64, now: Instant) {
        let Mode::Burst {
            at,
            passed,
            chars_per_sec,
        } = self.mode
        else {
            return;
        };
        if !cps.is_finite() || cps <= 0.0 || same_rate(chars_per_sec, cps) {
            return;
        }
        let elapsed = now.saturating_duration_since(at).as_secs_f64();
        let shown = passed + elapsed * chars_per_sec;
        let shown = if shown.is_finite() {
            shown.clamp(0.0, self.text.len() as f64)
        } else {
            0.0
        };
        self.mode = Mode::Burst {
            at: now,
            passed: shown,
            chars_per_sec: cps,
        };
    }

    fn view(&self, now: Instant) -> (Option<usize>, Option<f64>) {
        match self.mode {
            Mode::Tail => (None, None),
            Mode::Burst {
                at,
                passed,
                chars_per_sec,
            } => {
                let elapsed = now.saturating_duration_since(at).as_secs_f64();
                let chars = passed + elapsed * chars_per_sec;
                let chars = if chars.is_finite() {
                    chars.clamp(0.0, self.text.len() as f64).floor()
                } else {
                    0.0
                };
                (Some(chars as usize), Some(chars_per_sec))
            }
        }
    }
}

/// Scroll state for the model on screen.
#[derive(Debug, Default)]
pub struct Motion {
    models: HashMap<String, Rates>,
    inn: Panel,
    out: Panel,
    seen: HashSet<u64>,
}

impl Motion {
    /// Advance one frame and report where IN and OUT should draw.
    pub fn update(&mut self, sample: &Sample<'_>, now: Instant) -> Scroll {
        let mut fresh: Vec<&Turn> = sample
            .rows
            .iter()
            .filter(|row| row.model == sample.model && !self.seen.contains(&row.seq))
            .collect();
        fresh.sort_by_key(|row| row.seq);
        let newest = fresh.pop();
        let older = fresh;

        let new_turn = sample.request != self.inn.request || sample.request != self.out.request;
        // This turn's own speed stays out of the average, and stays available
        // after its row has been seen, so a live rate that disappears can
        // fall back to it.
        let (prompt_hist, gen_hist, held_prompt, held_gen) = {
            let rates = self.models.entry(sample.model.to_owned()).or_default();
            if new_turn {
                rates.commit_current();
            }
            for row in &older {
                push(&mut rates.prompt, row.prompt_tps);
                push(&mut rates.decode, row.gen_tps);
            }
            (
                rates.prompt.clone(),
                rates.decode.clone(),
                rates.current_prompt,
                rates.current_gen,
            )
        };
        let own_prompt = newest
            .and_then(|row| positive(row.prompt_tps))
            .or(held_prompt);
        let own_gen = newest.and_then(|row| positive(row.gen_tps)).or(held_gen);

        self.inn.advance(
            sample.request,
            sample.input,
            Pace {
                tokens: sample.input_tokens,
                chars: sample.input_chars,
                new: Some(new_input(
                    sample.input.len(),
                    sample.input_new,
                    sample.input_chars,
                    sample.input_tokens,
                    sample.input_cached,
                )),
                default_tps: Some(FAST_PREFILL_TPS),
                live: sample.live_prompt,
                own: own_prompt,
                history: &prompt_hist,
            },
            now,
        );
        self.out.advance(
            sample.request,
            sample.output,
            Pace {
                tokens: sample.output_tokens,
                chars: sample.output_chars,
                new: None,
                default_tps: None,
                live: sample.live_gen,
                own: own_gen,
                history: &gen_hist,
            },
            now,
        );

        // The turn now on screen keeps its speed out of the average until
        // the next turn. A row that arrives after the text still counts.
        {
            let rates = self.models.entry(sample.model.to_owned()).or_default();
            if new_turn {
                rates.current_prompt = own_prompt;
                rates.current_gen = own_gen;
            } else if newest.is_some() {
                rates.note_current(own_prompt, own_gen);
            }
        }
        for row in older {
            self.seen.insert(row.seq);
        }
        if let Some(row) = newest {
            self.seen.insert(row.seq);
        }
        if self.seen.len() > SEEN_MAX {
            let keep = self.seen.iter().copied().max().unwrap_or(0);
            self.seen.retain(|seq| keep.saturating_sub(*seq) < 256);
        }

        let (inn, in_cps) = self.inn.view(now);
        let (out, out_cps) = self.out.view(now);
        Scroll {
            inn,
            out,
            in_cps,
            out_cps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(seq: u64, prompt: f64, speed: f64) -> Turn {
        Turn {
            seq,
            model: "m".to_owned(),
            prompt_tps: Some(prompt),
            gen_tps: Some(speed),
        }
    }

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    /// One character per token, so characters per second equals tok/s.
    fn sample<'a>(
        request: u64,
        input: &'a [char],
        output: &'a [char],
        rows: &'a [Turn],
        live_prompt: Option<f64>,
        live_gen: Option<f64>,
    ) -> Sample<'a> {
        Sample {
            model: "m",
            request,
            input,
            output,
            input_tokens: Some(input.len() as u64),
            output_tokens: Some(output.len() as u64),
            input_chars: Some(input.len()),
            output_chars: Some(output.len()),
            input_cached: None,
            input_new: None,
            live_prompt,
            live_gen,
            rows,
        }
    }

    #[test]
    fn the_second_burst_scrolls_at_the_first_and_starts_at_zero() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let first_in = chars("prompt-one");
        let first_out = chars("answer-one");
        let rows = [turn(1, 40.0, 10.0)];
        let first = motion.update(&sample(1, &first_in, &first_out, &rows, None, None), now);
        assert_eq!(first.out, Some(0), "the first frame is the start");
        assert_eq!(
            first.out_cps,
            Some(10.0),
            "the first turn uses its own gen speed"
        );
        assert_eq!(first.in_cps, Some(40.0));

        let second_in = chars("prompt-two");
        let second_out = chars("START of the next answer");
        let rows = [turn(1, 40.0, 10.0), turn(2, 80.0, 99.0)];
        let second = motion.update(&sample(2, &second_in, &second_out, &rows, None, None), now);
        assert_eq!(second.out, Some(0), "the new reply starts at the top");
        assert_eq!(
            second.out_cps,
            Some(10.0),
            "OUT uses the previous gen tok/s"
        );
        assert_eq!(
            second.in_cps,
            Some(40.0),
            "IN uses the previous prefill tok/s"
        );
        assert_ne!(second.out_cps, Some(99.0), "not this turn's own gen speed");
    }

    #[test]
    fn a_burst_with_no_live_rate_uses_the_five_turn_average() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let speeds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        for (i, speed) in speeds.iter().enumerate() {
            let text = chars(&format!("reply-{i}-{}", "x".repeat(8)));
            let rows: Vec<Turn> = (1..=i as u64 + 1)
                .map(|seq| turn(seq, speeds[seq as usize - 1], speeds[seq as usize - 1]))
                .collect();
            let view = motion.update(&sample(i as u64 + 1, &text, &text, &rows, None, None), now);
            let expected = if i == 0 {
                *speed
            } else {
                let start = i.saturating_sub(HISTORY);
                let window = &speeds[start..i];
                window.iter().sum::<f64>() / window.len() as f64
            };
            assert!(
                (view.out_cps.expect("cps") - expected).abs() < 1e-9,
                "turn {i}: got {:?} want {expected} (not only the previous)",
                view.out_cps
            );
        }
    }

    #[test]
    fn a_live_rate_is_used_while_it_exists() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let first = chars("one");
        let rows = [turn(1, 100.0, 100.0)];
        motion.update(&sample(1, &first, &first, &rows, None, None), now);
        let second = chars("two");
        let rows = [turn(1, 100.0, 100.0), turn(2, 200.0, 200.0)];
        let view = motion.update(
            &sample(2, &second, &second, &rows, Some(7.0), Some(9.0)),
            now,
        );
        assert_eq!(view.in_cps, Some(7.0), "live prefill, not the average");
        assert_eq!(view.out_cps, Some(9.0), "live generation, not the average");
    }

    /// The previous reply is still in the slot when the next prompt arrives.
    /// It is not scrolled again as this turn's OUT.
    #[test]
    fn a_new_turn_does_not_replay_the_previous_reply() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let prompt = chars("prompt");
        let answer = chars("previous answer that is long enough");
        let rows = [turn(1, 10.0, 10.0)];
        let first = motion.update(&sample(1, &prompt, &answer, &rows, None, None), now);
        assert_eq!(first.out, Some(0));
        let later = now + std::time::Duration::from_millis(500);
        let mid = motion.update(&sample(1, &prompt, &answer, &rows, None, None), later);
        let held = mid.out.expect("still scrolling the first reply");
        assert!(held > 0, "half a second in: {held}");
        let next_prompt = chars("next prompt");
        let rows = [turn(1, 10.0, 10.0), turn(2, 20.0, 20.0)];
        let next = motion.update(&sample(2, &next_prompt, &answer, &rows, None, None), later);
        assert_eq!(
            next.out,
            Some(held),
            "the leftover reply is not started over"
        );
        let fresh = chars("brand new answer");
        let started = motion.update(&sample(2, &next_prompt, &fresh, &rows, None, None), later);
        assert_eq!(started.out, Some(0), "the new reply starts at the top");
    }

    #[test]
    fn a_streaming_update_does_not_restart_from_the_top() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let first = chars("Hello");
        let burst = motion.update(&sample(1, &first, &first, &[], None, None), now);
        assert_eq!(burst.out, Some(0));
        let grown = chars("Hello world, and then a good deal more text");
        let stream = motion.update(&sample(1, &grown, &grown, &[], None, None), now);
        assert_eq!(stream.out, None, "growth stays on the tail");
        assert_eq!(stream.inn, None);
    }

    #[test]
    fn the_first_turn_with_no_speed_scrolls_in_two_seconds() {
        let mut motion = Motion::default();
        let start = Instant::now();
        let text = chars(&"x".repeat(100));
        let view = motion.update(&sample(1, &text, &text, &[], None, None), start);
        assert_eq!(view.out, Some(0), "the first frame is the start");
        assert!((view.out_cps.expect("cps") - 50.0).abs() < 1e-9);
        let mid = motion.update(
            &sample(1, &text, &text, &[], None, None),
            start + std::time::Duration::from_secs(1),
        );
        assert_eq!(mid.out, Some(50), "halfway at one second");
        let end = motion.update(
            &sample(1, &text, &text, &[], None, None),
            start + std::time::Duration::from_secs(2),
        );
        assert_eq!(end.out, Some(100), "the whole text at two seconds");
    }

    #[test]
    fn a_missing_token_count_uses_four_characters_per_token() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let text = chars("abcdefgh");
        let rows = [turn(1, 10.0, 10.0)];
        let mut sample = sample(1, &text, &text, &rows, None, None);
        sample.input_tokens = None;
        sample.output_tokens = None;
        let view = motion.update(&sample, now);
        assert!((view.out_cps.expect("cps") - 40.0).abs() < 1e-9);
    }

    /// An 8,192-character IN tail of a 32,178-token prompt, 29,952 of them
    /// cached, read at 3,000 tok/s (DER-451). At four characters per token
    /// the tail is about 2,048 tokens and crosses in under a second. Its
    /// own length over the whole prompt's tokens would take 10.7 s, the
    /// time to prefill all 32,178 tokens cold.
    #[test]
    fn a_capped_tail_of_a_cached_prompt_crosses_at_the_prefill_rate() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let tail = chars(&"t".repeat(8_192));
        let rows = [turn(1, 3_000.0, 30.0)];
        let mut sample = sample(1, &tail, &[], &rows, None, None);
        sample.input_tokens = Some(32_178);
        sample.input_chars = None;
        let view = motion.update(&sample, now);
        assert!((view.in_cps.expect("cps") - 12_000.0).abs() < 1e-9);
        let crossed = motion.update(&sample, now + std::time::Duration::from_millis(683));
        assert_eq!(crossed.inn, Some(8_192), "the tail is through in 0.68 s");

        // The whole prompt's length is known: its own characters per token.
        let mut motion = Motion::default();
        sample.input_chars = Some(96_534);
        let view = motion.update(&sample, now);
        assert!((view.in_cps.expect("cps") - 9_000.0).abs() < 1e-9);
    }

    /// A 24,576-character OUT tail of a 10,000-token reply that was 40,000
    /// characters in all (DER-451): four characters per token, not 2.5.
    #[test]
    fn a_capped_reply_tail_uses_the_whole_reply_per_token() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let tail = chars(&"r".repeat(24_576));
        let rows = [turn(1, 3_000.0, 30.0)];
        let mut sample = sample(1, &[], &tail, &rows, None, None);
        sample.output_tokens = Some(10_000);
        sample.output_chars = Some(40_000);
        let view = motion.update(&sample, now);
        assert!((view.out_cps.expect("cps") - 120.0).abs() < 1e-9);
    }

    /// An agent turn (DER-451): IN holds the conversation tail, and the
    /// capture's message diff says only the last 300 characters, the tool
    /// result, were appended. The burst starts at that result and crosses
    /// only it, at the prefill rate.
    #[test]
    fn a_tool_loop_turn_scrolls_only_the_appended_tool_result() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let text = chars(&format!("{}{}", "o".repeat(1_700), "n".repeat(300)));
        let rows = [turn(1, 1_000.0, 30.0)];
        let mut sample = sample(1, &text, &[], &rows, None, None);
        sample.input_new = Some(300);
        sample.input_tokens = None;
        let first = motion.update(&sample, now);
        assert_eq!(first.inn, Some(1_700), "starts at the new part");
        assert_eq!(first.in_cps, Some(4_000.0));
        let done = motion.update(&sample, now + std::time::Duration::from_millis(75));
        assert_eq!(done.inn, Some(2_000), "300 characters in 75 ms");
        // A retry appended nothing IN shows: it sits at the end.
        let mut retry = sample.clone();
        retry.request = 2;
        retry.input_new = Some(0);
        let mut motion = Motion::default();
        assert_eq!(motion.update(&retry, now).inn, Some(2_000));
    }

    /// No diff: the uncached tokens, counted back from the end, are new;
    /// with nothing cached (a new conversation) all of it is.
    #[test]
    fn without_a_diff_the_uncached_tokens_mark_the_new_tail() {
        let now = Instant::now();
        let text = chars(&"x".repeat(8_192));
        let rows = [turn(1, 1_000.0, 30.0)];
        let mut sample = sample(1, &text, &[], &rows, None, None);
        sample.input_chars = None;
        sample.input_tokens = Some(32_178);
        sample.input_cached = Some(31_678);
        let view = Motion::default().update(&sample, now);
        assert_eq!(view.inn, Some(8_192 - 2_000), "500 tokens at four each");
        sample.input_cached = Some(0);
        let view = Motion::default().update(&sample, now);
        assert_eq!(view.inn, Some(0), "a new conversation crosses everything");
        sample.input_cached = None;
        let view = Motion::default().update(&sample, now);
        assert_eq!(view.inn, Some(0), "an unknown cache crosses everything");
    }

    /// A cache miss (DER-451): the diff says one small message is new, the
    /// engine says nothing was cached. The uncached estimate wins. An agent
    /// turn whose uncached count is the diff plus about 2,000 tokens of
    /// re-sent call, template and block rounding keeps the diff.
    #[test]
    fn a_cache_miss_beats_a_small_diff_but_an_agent_turn_does_not() {
        let now = Instant::now();
        let text = chars(&"x".repeat(8_192));
        let rows = [turn(1, 1_000.0, 30.0)];
        let mut sample = sample(1, &text, &[], &rows, None, None);
        sample.input_chars = None;
        // A rewritten conversation: 100 new characters, nothing cached.
        sample.input_new = Some(100);
        sample.input_tokens = Some(26_659);
        sample.input_cached = Some(0);
        assert_eq!(Motion::default().update(&sample, now).inn, Some(0));
        // An agent turn: 884 new characters, 2,226 uncached tokens.
        sample.input_new = Some(884);
        sample.input_tokens = Some(32_178);
        sample.input_cached = Some(29_952);
        assert_eq!(
            Motion::default().update(&sample, now).inn,
            Some(8_192 - 884)
        );
        // No cached count: the diff alone.
        sample.input_new = Some(100);
        sample.input_cached = None;
        assert_eq!(
            Motion::default().update(&sample, now).inn,
            Some(8_192 - 100)
        );
    }

    /// No rate at all: IN uses the fast default, not two seconds, and the
    /// first measured rate takes over without a jump.
    #[test]
    fn an_unknown_prefill_rate_is_fast_and_a_real_one_takes_over() {
        let mut motion = Motion::default();
        let start = Instant::now();
        let text = chars(&"x".repeat(80_000));
        let mut sample = sample(1, &text, &[], &[], None, None);
        sample.input_chars = None;
        sample.input_tokens = None;
        let first = motion.update(&sample, start);
        assert_eq!(first.in_cps, Some(FAST_PREFILL_TPS * 4.0));
        let half = start + std::time::Duration::from_millis(500);
        assert_eq!(motion.update(&sample, half).inn, Some(4_000));
        let rows = [turn(1, 500.0, 30.0)];
        let mut rated = sample.clone();
        rated.rows = &rows;
        let view = motion.update(&rated, half);
        assert_eq!(view.inn, Some(4_000), "no jump");
        assert_eq!(view.in_cps, Some(2_000.0), "this turn's own rate");
        let later = motion.update(&rated, half + std::time::Duration::from_secs(1));
        assert_eq!(later.inn, Some(6_000));
    }

    /// Earlier turns' prefill rate is used before the fast default.
    #[test]
    fn the_history_prefill_rate_beats_the_default() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let first = chars("first prompt");
        let rows = [turn(1, 300.0, 30.0)];
        motion.update(&sample(1, &first, &[], &rows, None, None), now);
        let second = chars("second prompt");
        let mut next = sample(2, &second, &[], &rows, None, None);
        next.input_chars = None;
        next.input_tokens = None;
        let view = motion.update(&next, now);
        assert_eq!(view.in_cps, Some(1_200.0), "300 tok/s, four per token");
    }

    /// Text that is not capped is its own whole: the speed is what it was.
    #[test]
    fn an_uncapped_text_keeps_its_own_characters_per_token() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let text = chars(&"u".repeat(600));
        let rows = [turn(1, 100.0, 10.0)];
        let mut sample = sample(1, &text, &text, &rows, None, None);
        sample.input_tokens = Some(200);
        sample.output_tokens = Some(300);
        let view = motion.update(&sample, now);
        assert!((view.in_cps.expect("cps") - 300.0).abs() < 1e-9);
        assert!((view.out_cps.expect("cps") - 20.0).abs() < 1e-9);
    }

    /// Seven completed turns. The last burst averages only the previous
    /// five, so raising the history cap from 5 does not pass.
    #[test]
    fn seven_completed_turns_use_only_the_last_five() {
        let mut motion = Motion::default();
        let now = Instant::now();
        let speeds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut last = 0.0;
        for (i, _) in speeds.iter().enumerate() {
            let text = chars(&format!("reply-{i}-{}", "x".repeat(8)));
            let rows: Vec<Turn> = (1..=i as u64 + 1)
                .map(|seq| turn(seq, speeds[seq as usize - 1], speeds[seq as usize - 1]))
                .collect();
            let view = motion.update(&sample(i as u64 + 1, &text, &text, &rows, None, None), now);
            last = view.out_cps.expect("cps");
        }
        let last_five = (20.0 + 30.0 + 40.0 + 50.0 + 60.0) / 5.0;
        let last_six = (10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0) / 6.0;
        assert!(
            (last - last_five).abs() < 1e-9,
            "got {last}, want the last five ({last_five})"
        );
        assert!(
            (last - last_six).abs() > 1.0,
            "a cap of 6 would average {last_six}"
        );
    }

    #[test]
    fn a_live_rate_that_appears_after_the_start_does_not_jump() {
        let mut motion = Motion::default();
        let start = Instant::now();
        let first = chars(&"a".repeat(20));
        let rows = [turn(1, 10.0, 10.0)];
        motion.update(&sample(1, &first, &first, &rows, None, None), start);

        let text = chars(&"b".repeat(1_000));
        let rows = [turn(1, 10.0, 10.0), turn(2, 40.0, 40.0)];
        let begun = motion.update(&sample(2, &text, &text, &rows, None, None), start);
        assert_eq!(begun.out, Some(0));
        assert!(
            (begun.out_cps.expect("cps") - 10.0).abs() < 1e-9,
            "the average, not this turn's own 40"
        );

        let at = start + std::time::Duration::from_secs(1);
        let mid = motion.update(&sample(2, &text, &text, &rows, None, None), at);
        assert_eq!(mid.out, Some(10), "one second at 10 characters per second");

        let switched = motion.update(
            &sample(2, &text, &text, &rows, Some(100.0), Some(100.0)),
            at,
        );
        assert_eq!(
            switched.out,
            Some(10),
            "taking the live rate must not jump the text"
        );
        assert!((switched.out_cps.expect("cps") - 100.0).abs() < 1e-9);

        let later = at + std::time::Duration::from_millis(500);
        let moved = motion.update(
            &sample(2, &text, &text, &rows, Some(100.0), Some(100.0)),
            later,
        );
        assert_eq!(moved.out, Some(60), "half a second at the live rate");

        let dropped = motion.update(&sample(2, &text, &text, &rows, None, None), later);
        assert_eq!(dropped.out, Some(60), "losing the live rate must not jump");
        assert!(
            (dropped.out_cps.expect("cps") - 10.0).abs() < 1e-9,
            "back to the five-turn average, not this turn's 40"
        );
        let coast = motion.update(
            &sample(2, &text, &text, &rows, None, None),
            later + std::time::Duration::from_secs(1),
        );
        assert_eq!(coast.out, Some(70));
    }

    #[test]
    fn a_live_rate_that_disappears_on_the_first_turn_uses_its_own_speed() {
        let mut motion = Motion::default();
        let start = Instant::now();
        let text = chars(&"c".repeat(1_000));
        let rows = [turn(1, 10.0, 10.0)];
        let begun = motion.update(
            &sample(1, &text, &text, &rows, Some(80.0), Some(80.0)),
            start,
        );
        assert_eq!(begun.out, Some(0));
        assert!((begun.out_cps.expect("cps") - 80.0).abs() < 1e-9);

        let at = start + std::time::Duration::from_secs(1);
        let mid = motion.update(&sample(1, &text, &text, &rows, Some(80.0), Some(80.0)), at);
        assert_eq!(mid.out, Some(80));

        let dropped = motion.update(&sample(1, &text, &text, &[], None, None), at);
        assert_eq!(dropped.out, Some(80), "no jump back to the top");
        assert!(
            (dropped.out_cps.expect("cps") - 10.0).abs() < 1e-9,
            "this turn's own speed, not the two-second fallback"
        );
        let coast = motion.update(
            &sample(1, &text, &text, &[], None, None),
            at + std::time::Duration::from_secs(1),
        );
        assert_eq!(coast.out, Some(90));
    }
}
