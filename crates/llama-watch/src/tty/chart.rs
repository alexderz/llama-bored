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
    /// A counter increase covers `(previous observation, now]` and is split
    /// across column boundaries in proportion to time. The first observation
    /// after start or after `None`, and a step down, set the base and count
    /// nothing. Prompt is still the mean of the per-tick values.
    pub fn ingest(&mut self, now: Instant, decoded: Option<u64>, prompt_tps: Option<f64>) {
        let Some(start) = self.start else {
            self.start = Some(now);
            self.remember(decoded, now);
            self.add_prompt(prompt_tps);
            self.publish_current();
            return;
        };
        let elapsed = now.saturating_duration_since(start).as_secs_f64();
        let width = self.bucket_s;
        let step = match (self.last_decoded, decoded) {
            (Some(prev), Some(next)) if next >= prev => Some((next - prev) as f64),
            _ => None,
        };
        if elapsed < width {
            if let Some(tokens) = step {
                self.gen_tokens += tokens;
            }
            self.remember(decoded, now);
            self.add_prompt(prompt_tps);
            self.publish_current();
            return;
        }
        let skipped = (elapsed / width).floor();
        let skipped = if skipped.is_finite() {
            (skipped as u64).min(MAX_BUCKETS as u64)
        } else {
            MAX_BUCKETS as u64
        };
        if let Some(tokens) = step {
            self.spread(start, now, skipped, tokens);
        } else {
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
        self.remember(decoded, now);
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

    fn remember(&mut self, decoded: Option<u64>, now: Instant) {
        if let Some(v) = decoded {
            self.last_decoded = Some(v);
            self.last_at = Some(now);
            self.gen_seen = true;
        } else {
            self.last_decoded = None;
            self.last_at = None;
        }
    }

    fn spread(&mut self, start: Instant, now: Instant, skipped: u64, tokens: f64) {
        let width = self.bucket_s;
        let prev_at = self.last_at.unwrap_or(start);
        let span = now.saturating_duration_since(prev_at).as_secs_f64();
        let share = |i: u64| -> f64 {
            let b_start = start + Duration::from_secs_f64(i as f64 * width);
            let b_end = start + Duration::from_secs_f64((i + 1) as f64 * width);
            let ov = overlap_s(prev_at, now, b_start, b_end);
            if span > 0.0 {
                tokens * ov / span
            } else if i == skipped {
                tokens
            } else {
                0.0
            }
        };
        self.gen_tokens += share(0);
        self.gen_seen = true;
        self.commit();
        for i in 1..skipped {
            if let Some(last) = self.buckets.back_mut() {
                *last = ChartBucket {
                    gen_tps: Some(share(i) / width),
                    prompt_tps: None,
                };
            }
            self.push(ChartBucket {
                gen_tps: None,
                prompt_tps: None,
            });
        }
        self.start = Some(start + Duration::from_secs_f64(skipped as f64 * width));
        self.gen_tokens = share(skipped);
        self.gen_seen = true;
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

    #[test]
    fn fixed_rate_decoding_fills_full_columns_and_climbs_in_progress() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, Some(0), None);
        chart.ingest(at(origin, 500), Some(10), None);
        assert_eq!(gens(&chart), vec![Some(5.0)]);
        chart.ingest(at(origin, 1000), Some(20), None);
        assert_eq!(gens(&chart), vec![Some(10.0)]);
        chart.ingest(at(origin, 1500), Some(30), None);
        assert_eq!(gens(&chart), vec![Some(15.0)]);
        chart.ingest(at(origin, 2000), Some(40), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(0.0)]);
        chart.ingest(at(origin, 2500), Some(50), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(5.0)]);
        chart.ingest(at(origin, 4000), Some(80), None);
        assert_eq!(gens(&chart), vec![Some(20.0), Some(20.0), Some(0.0)]);
    }

    #[test]
    fn a_burst_that_ends_mid_column_puts_nothing_into_the_next() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, Some(0), None);
        chart.ingest(at(origin, 400), Some(80), None);
        assert_eq!(gens(&chart), vec![Some(40.0)]);
        chart.ingest(at(origin, 2000), Some(80), None);
        assert_eq!(gens(&chart), vec![Some(40.0), Some(0.0)]);
        chart.ingest(at(origin, 3500), Some(80), None);
        assert_eq!(gens(&chart), vec![Some(40.0), Some(0.0)]);
    }

    #[test]
    fn a_boundary_increase_is_split_and_a_reset_counts_nothing_negative() {
        let mut chart = TokenChart::new(2);
        let origin = t0();
        chart.ingest(origin, Some(0), None);
        chart.ingest(at(origin, 1500), Some(0), None);
        chart.ingest(at(origin, 2500), Some(40), None);
        assert_eq!(gens(&chart), vec![Some(10.0), Some(10.0)]);

        let mut reset = TokenChart::new(2);
        let origin = t0();
        reset.ingest(origin, Some(100), None);
        reset.ingest(at(origin, 500), Some(140), None);
        assert_eq!(gens(&reset), vec![Some(20.0)]);
        reset.ingest(at(origin, 1000), Some(10), None);
        assert_eq!(gens(&reset), vec![Some(20.0)]);
        reset.ingest(at(origin, 1500), Some(30), None);
        assert_eq!(gens(&reset), vec![Some(30.0)]);
    }
}
