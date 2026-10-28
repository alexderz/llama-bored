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

fn chars_per_sec(chars: usize, tokens: Option<u64>, tps: Option<f64>) -> f64 {
    let Some(tps) = tps else {
        return chars as f64 / FALLBACK_SECS;
    };
    let per_token = match tokens {
        Some(n) if n > 0 => chars as f64 / n as f64,
        _ => DEFAULT_CHARS_PER_TOKEN,
    };
    tps * per_token
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
        let tps = choose(pace.live, pace.history, pace.own);
        let cps = chars_per_sec(text.len(), pace.tokens, tps);
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
        self.mode = Mode::Burst {
            at: now,
            passed: 0.0,
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
