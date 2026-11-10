//! Windowed rates of per-model token counters (DER-451).
//!
//! An aggregate rate is the counter's rise over a window divided by the
//! window's wall time: what the engine did, however many requests shared
//! it. A per-request mean (`prompt_per_second` of each finished request)
//! is each one's share of the batch instead.
//!
//! The window's base is the newest sample at least `window` older than
//! the newest one, else the oldest kept: a rate over a shorter span when
//! the counter has not been read that long. A counter that goes down is a
//! new process: its history is dropped and it starts again.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Shortest span a rate is taken over. Two reads closer than this carry
/// a whole batch in a tiny gap: the wall-clock spike.
pub const MIN_SPAN: Duration = Duration::from_millis(900);

/// Most samples kept per model; at the 0.25 s metrics cadence this is
/// longer than any window asked for.
const MAX_SAMPLES: usize = 64;

#[derive(Debug, Default)]
pub struct WindowRate {
    models: HashMap<String, VecDeque<(Instant, u64)>>,
}

impl WindowRate {
    /// One read of `model`'s counter.
    pub fn observe(&mut self, model: &str, at: Instant, total: u64) {
        let samples = self.models.entry(model.to_owned()).or_default();
        if samples.back().is_some_and(|&(_, last)| total < last) {
            samples.clear();
        }
        samples.push_back((at, total));
        while samples.len() > MAX_SAMPLES {
            samples.pop_front();
        }
    }

    /// Tokens per second over the last `window`, or over the span read so
    /// far when that is shorter. `None` when the newest read is older than
    /// `fresh` or the span is under [`MIN_SPAN`].
    pub fn rate(
        &self,
        model: &str,
        now: Instant,
        window: Duration,
        fresh: Duration,
    ) -> Option<f64> {
        self.rate_over(model, now, window, fresh, MIN_SPAN)
    }

    /// [`Self::rate`] over a span of at least `min_span`.
    pub fn rate_over(
        &self,
        model: &str,
        now: Instant,
        window: Duration,
        fresh: Duration,
        min_span: Duration,
    ) -> Option<f64> {
        let samples = self.models.get(model)?;
        let &(newest_at, newest) = samples.back()?;
        if now.saturating_duration_since(newest_at) > fresh {
            return None;
        }
        let &(base_at, base) = samples
            .iter()
            .rev()
            .find(|(at, _)| newest_at.saturating_duration_since(*at) >= window)
            .or_else(|| samples.front())?;
        let span = newest_at.saturating_duration_since(base_at);
        if span < min_span.max(MIN_SPAN) {
            return None;
        }
        Some(newest.saturating_sub(base) as f64 / span.as_secs_f64())
    }

    /// Tokens the counter rose by over the last `window` (or the span read
    /// so far), `None` without two reads.
    pub fn rise(&self, model: &str, window: Duration) -> Option<u64> {
        let samples = self.models.get(model)?;
        let &(newest_at, newest) = samples.back()?;
        let &(_, base) = samples
            .iter()
            .rev()
            .find(|(at, _)| newest_at.saturating_duration_since(*at) >= window)
            .or_else(|| samples.front())?;
        (samples.len() > 1).then(|| newest.saturating_sub(base))
    }

    /// The newest read of `model`'s counter.
    #[must_use]
    pub fn newest(&self, model: &str) -> Option<(Instant, u64)> {
        self.models.get(model)?.back().copied()
    }

    /// A new episode starts: its window begins at `at`, where the counter
    /// stood at `total`, and nothing older counts.
    pub fn rebase_at(&mut self, model: &str, at: Instant, total: u64) {
        let samples = self.models.entry(model.to_owned()).or_default();
        samples.clear();
        samples.push_back((at, total));
    }

    /// `model`'s process is gone or its counter's source changed.
    pub fn forget(&mut self, model: &str) {
        self.models.remove(model);
    }
}

/// Tokens per second of busy time from a token counter and a seconds
/// counter that move together (llama.cpp's `prompt_tokens_total` and
/// `prompt_seconds_total`), over a window of reads. The last ratio is
/// held while the seconds do not move.
#[derive(Debug, Default)]
pub struct RatioWindow {
    models: HashMap<String, RatioReads>,
}

/// One model's reads of both counters, `(at, tokens, seconds)`, and its
/// held ratio.
type RatioReads = (VecDeque<(Instant, u64, f64)>, Option<f64>);

impl RatioWindow {
    /// One read of both counters.
    pub fn observe(
        &mut self,
        model: &str,
        at: Instant,
        tokens: u64,
        seconds: f64,
        window: Duration,
    ) {
        let (samples, held) = self.models.entry(model.to_owned()).or_default();
        if samples
            .back()
            .is_some_and(|&(_, last_tokens, last_seconds)| {
                tokens < last_tokens || seconds < last_seconds
            })
        {
            samples.clear();
        }
        samples.push_back((at, tokens, seconds));
        while samples.len() > MAX_SAMPLES {
            samples.pop_front();
        }
        let base = samples
            .iter()
            .rev()
            .find(|(base_at, _, _)| at.saturating_duration_since(*base_at) >= window)
            .or_else(|| samples.front())
            .copied();
        if let Some((_, base_tokens, base_seconds)) = base {
            let busy = seconds - base_seconds;
            if busy > 0.0 && busy.is_finite() {
                *held = Some(tokens.saturating_sub(base_tokens) as f64 / busy);
            }
        }
    }

    /// The last ratio, `None` before the seconds first moved.
    #[must_use]
    pub fn ratio(&self, model: &str) -> Option<f64> {
        self.models.get(model)?.1.filter(|rate| rate.is_finite())
    }

    pub fn forget(&mut self, model: &str) {
        self.models.remove(model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRESH: Duration = Duration::from_secs(1);

    fn fed(steps: &[(u64, u64)]) -> (WindowRate, Instant) {
        let t0 = Instant::now();
        let mut book = WindowRate::default();
        for &(ms, total) in steps {
            book.observe("m", t0 + Duration::from_millis(ms), total);
        }
        (book, t0)
    }

    #[test]
    fn the_rate_is_over_the_window() {
        // 100 tok/s for 10 s read every 0.25 s.
        let steps: Vec<(u64, u64)> = (0..=40).map(|i| (i * 250, i * 25)).collect();
        let (book, t0) = fed(&steps);
        let now = t0 + Duration::from_secs(10);
        assert_eq!(
            book.rate("m", now, Duration::from_secs(5), FRESH),
            Some(100.0)
        );
        assert_eq!(book.rise("m", Duration::from_secs(5)), Some(500));
    }

    #[test]
    fn a_young_counter_is_rated_over_what_was_read() {
        let (book, t0) = fed(&[(0, 0), (500, 50), (1000, 200)]);
        let now = t0 + Duration::from_secs(1);
        assert_eq!(
            book.rate("m", now, Duration::from_secs(5), FRESH),
            Some(200.0)
        );
        // Under the minimum span there is no rate: no wall-clock spike.
        let (book, t0) = fed(&[(0, 0), (250, 4000)]);
        assert_eq!(
            book.rate(
                "m",
                t0 + Duration::from_millis(250),
                Duration::from_secs(5),
                FRESH
            ),
            None
        );
    }

    #[test]
    fn a_minimum_span_and_an_episode_base() {
        let steps: Vec<(u64, u64)> = (0..=8).map(|i| (i * 250, i * 100)).collect();
        let (mut book, t0) = fed(&steps);
        let now = t0 + Duration::from_secs(2);
        let ten = Duration::from_secs(10);
        assert_eq!(
            book.rate_over("m", now, ten, FRESH, ten),
            None,
            "under 10 s read"
        );
        assert_eq!(book.rate("m", now, ten, FRESH), Some(400.0));
        book.rebase_at("m", t0 + Duration::from_millis(1500), 600);
        book.observe("m", now, 800);
        assert_eq!(
            book.rate("m", now, ten, FRESH),
            None,
            "0.5 s since the base"
        );
        book.observe("m", now + Duration::from_millis(500), 1000);
        assert_eq!(
            book.rate("m", now + Duration::from_millis(500), ten, FRESH),
            Some(400.0)
        );
        assert_eq!(
            book.newest("m"),
            Some((now + Duration::from_millis(500), 1000))
        );
    }

    #[test]
    fn busy_time_ratio_is_held_while_idle() {
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        let ten = Duration::from_secs(10);
        let mut book = RatioWindow::default();
        book.observe("m", t0, 0, 0.0, ten);
        assert_eq!(book.ratio("m"), None);
        book.observe("m", t0 + s, 3000, 1.5, ten);
        assert_eq!(book.ratio("m"), Some(2000.0));
        book.observe("m", t0 + s * 2, 3000, 1.5, ten);
        assert_eq!(book.ratio("m"), Some(2000.0), "held");
        book.observe("m", t0 + s * 20, 4000, 2.5, ten);
        assert_eq!(book.ratio("m"), Some(1000.0), "the last window only");
    }

    #[test]
    fn a_stale_counter_has_no_rate_and_a_drop_starts_again() {
        let (mut book, t0) = fed(&[(0, 0), (1000, 100), (2000, 200)]);
        assert_eq!(
            book.rate(
                "m",
                t0 + Duration::from_secs(4),
                Duration::from_secs(5),
                FRESH
            ),
            None
        );
        book.observe("m", t0 + Duration::from_secs(3), 10);
        book.observe("m", t0 + Duration::from_secs(4), 60);
        assert_eq!(
            book.rate(
                "m",
                t0 + Duration::from_secs(4),
                Duration::from_secs(5),
                FRESH
            ),
            Some(50.0)
        );
    }
}
