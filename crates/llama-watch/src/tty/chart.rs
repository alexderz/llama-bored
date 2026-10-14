//! Token-rate history for the tty chart. One ring entry per bucket.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Widest terminal the layout will draw (`term::MAX_COLS`).
pub const MAX_BUCKETS: usize = 1024;

/// Rates over one `chart_bucket_s` column.
///
/// Generation is tokens decoded in the column divided by the column width.
/// Prompt is the mean of the per-tick values. `None` is no data (llama down,
/// or a gap). `Some(0.0)` is a measured zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartBucket {
    /// Generation tok/s in this bucket (tokens in the span / `chart_bucket_s`).
    pub gen_tps: Option<f64>,
    /// Mean prompt tok/s in this bucket.
    pub prompt_tps: Option<f64>,
}

/// Fixed-size ring. The tick loop owns one; layout only reads a snapshot.
pub struct TokenChart {
    buckets: VecDeque<ChartBucket>,
    bucket_s: f64,
    start: Option<Instant>,
    /// Tokens attributed to the in-progress column.
    gen_tokens: f64,
    /// A decoded-token counter was observed in the in-progress column.
    gen_seen: bool,
    last_decoded: Option<u64>,
    last_at: Option<Instant>,
    prompt_sum: f64,
    prompt_n: u32,
}

impl TokenChart {
    /// Empty ring. `bucket_s` of 0 is treated as 1 so a bad config cannot divide
    /// by zero; validation already rejects that range.
    #[must_use]
    pub fn new(bucket_s: u64) -> Self {
        Self {
            buckets: VecDeque::new(),
            bucket_s: (bucket_s.max(1) as f64),
            start: None,
            gen_tokens: 0.0,
            gen_seen: false,
            last_decoded: None,
            last_at: None,
            prompt_sum: 0.0,
            prompt_n: 0,
        }
    }

    /// Fold the decoded-token counter and the prompt rate into the column that
    /// contains `now`.
    ///
    /// `decoded` is the counter together with the instant that view arrived
    /// (the tick `take_latest` returned it). The same instant again is not a
    /// new observation: it adds nothing and leaves the base and `last_at`
    /// alone. Rollover, the in-progress column and prompt still run every
    /// tick. A counter increase covers `(previous observation instant, this
    /// observation instant]` and is split across column boundaries in
    /// proportion to time, including columns already committed. The first
    /// observation after start or after `None`, and a step down, set the base
    /// and count nothing. Prompt is still the mean of the per-tick values.
    pub fn ingest(
        &mut self,
        now: Instant,
        decoded: Option<(Instant, u64)>,
        prompt_tps: Option<f64>,
    ) {
        if self.start.is_none() {
            self.start = Some(now);
            self.apply_decoded(decoded);
            self.add_prompt(prompt_tps);
            self.publish_current();
            return;
        }
        self.roll_to(now);
        self.apply_decoded(decoded);
        self.add_prompt(prompt_tps);
        self.publish_current();
    }

    /// Oldest first, including the in-progress bucket. At most [`MAX_BUCKETS`].
    #[must_use]
    pub fn buckets(&self) -> Vec<ChartBucket> {
        self.buckets.iter().copied().collect()
    }

    fn add_prompt(&mut self, prompt_tps: Option<f64>) {
        if let Some(v) = prompt_tps.filter(|v| v.is_finite()) {
            self.prompt_sum += v;
            self.prompt_n = self.prompt_n.saturating_add(1);
        }
    }

    fn roll_to(&mut self, now: Instant) {
        let Some(start) = self.start else {
            return;
        };
        let elapsed = now.saturating_duration_since(start).as_secs_f64();
        let width = self.bucket_s;
        if elapsed < width {
            return;
        }
        let skipped = (elapsed / width).floor();
        let skipped = if skipped.is_finite() {
            (skipped as u64).min(MAX_BUCKETS as u64)
        } else {
            MAX_BUCKETS as u64
        };
        self.commit();
        for _ in 1..skipped {
            self.push(ChartBucket {
                gen_tps: None,
                prompt_tps: None,
            });
        }
        let advance = Duration::from_secs_f64(skipped as f64 * width);
        self.start = Some(start + advance);
    }

    fn apply_decoded(&mut self, decoded: Option<(Instant, u64)>) {
        let Some((at, count)) = decoded else {
            self.last_decoded = None;
            self.last_at = None;
            return;
        };
        self.gen_seen = true;
        if self.last_at == Some(at) {
            return;
        }
        if let (Some(prev_count), Some(prev_at)) = (self.last_decoded, self.last_at)
            && count >= prev_count
        {
            self.credit(prev_at, at, (count - prev_count) as f64);
        }
        self.last_decoded = Some(count);
        self.last_at = Some(at);
    }

    /// Attribute `tokens` across ring columns whose spans overlap
    /// `(prev_at, obs_at]`. A gap longer than [`MAX_BUCKETS`] columns loses
    /// the tokens that fall outside the ring.
    fn credit(&mut self, prev_at: Instant, obs_at: Instant, tokens: f64) {
        if tokens == 0.0 {
            return;
        }
        let Some(start) = self.start else {
            return;
        };
        let width = self.bucket_s;
        let span = obs_at.saturating_duration_since(prev_at).as_secs_f64();
        debug_assert!(
            span > 0.0,
            "apply_decoded returns on a repeated instant; observation instants only advance"
        );
        let n = self.buckets.len();
        for i in 0..n {
            let age = (n - 1 - i) as f64;
            let Some(b_start) = start.checked_sub(Duration::from_secs_f64(age * width)) else {
                continue;
            };
            let b_end = b_start + Duration::from_secs_f64(width);
            let part = tokens * overlap_s(prev_at, obs_at, b_start, b_end) / span;
            if part == 0.0 {
                continue;
            }
            if i + 1 == n {
                self.gen_tokens += part;
            } else if let Some(bucket) = self.buckets.get_mut(i) {
                bucket.gen_tps = Some(bucket.gen_tps.unwrap_or(0.0) + part / width);
            }
        }
    }

    fn current(&self) -> ChartBucket {
        ChartBucket {
            gen_tps: self.gen_seen.then_some(self.gen_tokens / self.bucket_s),
            prompt_tps: mean(self.prompt_sum, self.prompt_n),
        }
    }

    fn publish_current(&mut self) {
        let bucket = self.current();
        if self.buckets.is_empty() {
            self.push(bucket);
            return;
        }
        if let Some(last) = self.buckets.back_mut() {
            *last = bucket;
        }
    }

    fn commit(&mut self) {
        self.publish_current();
        self.gen_tokens = 0.0;
        self.gen_seen = false;
        self.prompt_sum = 0.0;
        self.prompt_n = 0;
        self.push(ChartBucket {
            gen_tps: None,
            prompt_tps: None,
        });
    }

    fn push(&mut self, bucket: ChartBucket) {
        if self.buckets.len() == MAX_BUCKETS {
            self.buckets.pop_front();
        }
        self.buckets.push_back(bucket);
    }
}

fn mean(sum: f64, n: u32) -> Option<f64> {
    (n > 0).then_some(sum / f64::from(n))
}

/// Length of `(prev, now]` ∩ `[b_start, b_end)`.
fn overlap_s(prev: Instant, now: Instant, b_start: Instant, b_end: Instant) -> f64 {
    let span = now.saturating_duration_since(prev).as_secs_f64();
    let start_off = b_start.saturating_duration_since(prev).as_secs_f64();
    let end_off = b_end.saturating_duration_since(prev).as_secs_f64();
    let hi = span.min(end_off);
    (hi - start_off).max(0.0)
}

/// Map `frac` in 0..=1 onto `rows × 2` half-block levels.
///
/// eurlatgr has `▄` and `▀` (and `█`) but no eighth-blocks, so each cell is
/// empty, half, or full. `chart_glyphs = "halves"`.
#[must_use]
pub fn halves(frac: f64, rows: u16) -> u32 {
    quantise(frac, rows, 2)
}

/// Map `frac` in 0..=1 onto `rows × 8` eighth-block levels.
///
/// llama-hack-12x24 has the lower eighths U+2581–2588. `chart_glyphs =
/// "eighths"`: a 4-row half is 32 levels.
#[must_use]
pub fn eighths(frac: f64, rows: u16) -> u32 {
    quantise(frac, rows, 8)
}

fn quantise(frac: f64, rows: u16, per_row: u32) -> u32 {
    let max = u32::from(rows).saturating_mul(per_row);
    if max == 0 {
        return 0;
    }
    let n = (frac.clamp(0.0, 1.0) * f64::from(max)).round();
    if !n.is_finite() {
        return 0;
    }
    (n as u32).min(max)
}

/// One chart cell in eighths mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ink {
    /// Nothing drawn.
    Empty,
    /// The glyph in the bar colour on black.
    Fg(char),
    /// The glyph in black on the bar colour. The falling tip needs the top
    /// n/8 of a cell, and Unicode has no upper fractions other than `▀` and
    /// `▔`, so it is the lower (8 − n)/8 inverted. The console has only the
    /// eight normal background colours, so a bright bar colour shows its
    /// normal twin in that one cell.
    Inverse(char),
}

/// Lower n/8 block, n in 1..=8.
fn lower_eighth(n: u32) -> char {
    char::from_u32(0x2580 + n.clamp(1, 8)).unwrap_or('█')
}

/// One column of `rows` cells growing up from the last row (the axis), in
/// eighths. A partial cell is a lower eighth; a full cell is `█`.
#[must_use]
pub fn rise_eighths(level: u32, rows: u16) -> Vec<Ink> {
    let rows_n = usize::from(rows);
    let mut out = eighth_column(level, rows_n, |n| Ink::Fg(lower_eighth(n)));
    out.reverse();
    out
}

/// One column of `rows` cells growing down from the first row (the axis), in
/// eighths. A partial cell is an inverted lower eighth; a full cell is `█`.
#[must_use]
pub fn fall_eighths(level: u32, rows: u16) -> Vec<Ink> {
    eighth_column(level, usize::from(rows), |n| {
        Ink::Inverse(lower_eighth(8 - n))
    })
}

/// Cells ordered from the axis outwards. `tip(n)` draws a partial n/8, n in 1..=7.
fn eighth_column(level: u32, rows: usize, tip: impl Fn(u32) -> Ink) -> Vec<Ink> {
    let full = (level / 8) as usize;
    let rem = level % 8;
    (0..rows)
        .map(|i| {
            if i < full {
                Ink::Fg('█')
            } else if i == full && rem > 0 {
                tip(rem)
            } else {
                Ink::Empty
            }
        })
        .collect()
}

/// One column of `rows` cells growing up from the last row (the axis).
///
/// A half cell is `▄`; a full cell is `█`.
#[must_use]
pub fn rise_glyphs(level: u32, rows: u16) -> Vec<char> {
    column_glyphs(level, rows, '▄', true)
}

/// One column of `rows` cells growing down from the first row (the axis).
///
/// A half cell is `▀`; a full cell is `█`.
#[must_use]
pub fn fall_glyphs(level: u32, rows: u16) -> Vec<char> {
    column_glyphs(level, rows, '▀', false)
}

fn column_glyphs(level: u32, rows: u16, half: char, rise: bool) -> Vec<char> {
    let rows_n = usize::from(rows);
    let mut out = vec![' '; rows_n];
    let full = (level / 2) as usize;
    let rem = (level % 2) as usize;
    for (i, cell) in out.iter_mut().enumerate() {
        let filled = if rise { rows_n - 1 - i } else { i };
        if filled < full {
            *cell = '█';
        } else if filled == full && rem > 0 {
            *cell = half;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    fn at(origin: Instant, ms: u64) -> Instant {
        origin + Duration::from_millis(ms)
    }

    fn gens(chart: &TokenChart) -> Vec<Option<f64>> {
        chart.buckets().into_iter().map(|b| b.gen_tps).collect()
    }

    fn obs(at: Instant, count: u64) -> Option<(Instant, u64)> {
        Some((at, count))
    }

    #[test]
    fn fixed_rate_decoding_fills_full_columns_and_climbs_in_progress() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, obs(origin, 0), None);
        chart.ingest(at(origin, 500), obs(at(origin, 500), 10), None);
        assert_eq!(gens(&chart), vec![Some(5.0)]);
        chart.ingest(at(origin, 1000), obs(at(origin, 1000), 20), None);
        assert_eq!(gens(&chart), vec![Some(10.0)]);
        chart.ingest(at(origin, 1500), obs(at(origin, 1500), 30), None);
        assert_eq!(gens(&chart), vec![Some(15.0)]);
        chart.ingest(at(origin, 2000), obs(at(origin, 2000), 40), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(0.0)]);
        chart.ingest(at(origin, 2500), obs(at(origin, 2500), 50), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(5.0)]);
        chart.ingest(at(origin, 4000), obs(at(origin, 4000), 80), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(20.0), Some(0.0)]);
    }

    #[test]
    fn a_burst_that_ends_mid_column_puts_nothing_into_the_next() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, obs(origin, 0), None);
        let burst_at = at(origin, 400);
        chart.ingest(burst_at, obs(burst_at, 80), None);
        assert_eq!(gens(&chart), vec![Some(40.0)]);
        for ms in (500..=3500).step_by(100) {
            chart.ingest(at(origin, ms), obs(burst_at, 80), None);
        }
        assert_eq!(gens(&chart), vec![Some(40.0), Some(0.0)]);
    }

    #[test]
    fn a_held_poll_spreads_a_step_over_the_poll_interval() {
        // Ticks every 100 ms; the counter refreshes every 250 ms with a 50 ms
        // phase. 200 tok/s until t = 1.9 s, 2 s columns. A held poll is not a
        // new observation. The 20 tokens first seen at 2.1 s cover (1.8 s, 2.1 s]
        // and split 2:1 across the boundary. Moving last_at every tick dumps
        // them into column 1 (180 / 10 / 0).
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, obs(origin, 0), None);
        let mut llama_at = origin;
        let mut count = 0_u64;
        for ms in (100..=4000).step_by(100) {
            let tick_open = ms - 100;
            let mut poll = 50_u64;
            while poll <= ms {
                if poll > tick_open {
                    llama_at = at(origin, ms);
                    let t_s = (poll as f64 / 1000.0).min(1.9);
                    count = (200.0 * t_s).round() as u64;
                }
                poll += 250;
            }
            chart.ingest(at(origin, ms), obs(llama_at, count), None);
        }
        let got = gens(&chart);
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[2], Some(0.0), "{got:?}");
        let col0 = got[0].expect("column 0");
        let col1 = got[1].expect("column 1");
        assert!(
            (col0 - 186.67).abs() < 0.01 && (col1 - 3.33).abs() < 0.01,
            "uniform split of (1.8 s, 2.1 s], got {got:?}"
        );
        assert!(
            (col0 - 180.0).abs() >= 0.01 || (col1 - 10.0).abs() >= 0.01,
            "old per-tick last_at, got {got:?}"
        );
    }

    #[test]
    fn a_boundary_increase_is_split_and_a_reset_counts_nothing_negative() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, obs(origin, 0), None);
        chart.ingest(at(origin, 1500), obs(at(origin, 1500), 0), None);
        chart.ingest(at(origin, 2500), obs(at(origin, 2500), 40), None);
        assert_eq!(gens(&chart), vec![Some(10.0), Some(10.0)]);

        let mut reset = TokenChart::new(2);
        let origin = t0();
        reset.ingest(origin, obs(origin, 100), None);
        reset.ingest(at(origin, 500), obs(at(origin, 500), 140), None);
        assert_eq!(gens(&reset), vec![Some(20.0)]);
        reset.ingest(at(origin, 1000), obs(at(origin, 1000), 10), None);
        assert_eq!(gens(&reset), vec![Some(20.0)]);
        reset.ingest(at(origin, 1500), obs(at(origin, 1500), 30), None);
        assert_eq!(gens(&reset), vec![Some(30.0)]);
    }
}
