//! Token-rate history for the tty chart. One ring entry per bucket.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Widest terminal the layout will draw (`term::MAX_COLS`).
pub const MAX_BUCKETS: usize = 1024;

/// Mean rates over one `chart_bucket_s` column.
///
/// `None` is no data (llama down, or a gap). `Some(0.0)` is a measured zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartBucket {
    /// Mean generation tok/s in this bucket.
    pub gen_tps: Option<f64>,
    /// Mean prompt tok/s in this bucket.
    pub prompt_tps: Option<f64>,
}

/// Fixed-size ring. The tick loop owns one; layout only reads a snapshot.
pub struct TokenChart {
    buckets: VecDeque<ChartBucket>,
    bucket_s: f64,
    start: Option<Instant>,
    gen_sum: f64,
    prompt_sum: f64,
    gen_n: u32,
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
            gen_sum: 0.0,
            prompt_sum: 0.0,
            gen_n: 0,
            prompt_n: 0,
        }
    }

    /// Fold the live rates into the bucket that contains `now`.
    pub fn ingest(&mut self, now: Instant, gen_tps: Option<f64>, prompt_tps: Option<f64>) {
        let Some(start) = self.start else {
            self.start = Some(now);
            self.add_sample(gen_tps, prompt_tps);
            self.publish_current();
            return;
        };
        let elapsed = now.saturating_duration_since(start).as_secs_f64();
        let width = self.bucket_s;
        if elapsed < width {
            self.add_sample(gen_tps, prompt_tps);
            self.publish_current();
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
        self.add_sample(gen_tps, prompt_tps);
        self.publish_current();
    }

    /// Oldest first, including the in-progress bucket. At most [`MAX_BUCKETS`].
    #[must_use]
    pub fn buckets(&self) -> Vec<ChartBucket> {
        self.buckets.iter().copied().collect()
    }

    fn add_sample(&mut self, gen_tps: Option<f64>, prompt_tps: Option<f64>) {
        if let Some(v) = gen_tps.filter(|v| v.is_finite()) {
            self.gen_sum += v;
            self.gen_n = self.gen_n.saturating_add(1);
        }
        if let Some(v) = prompt_tps.filter(|v| v.is_finite()) {
            self.prompt_sum += v;
            self.prompt_n = self.prompt_n.saturating_add(1);
        }
    }

    fn current(&self) -> ChartBucket {
        ChartBucket {
            gen_tps: mean(self.gen_sum, self.gen_n),
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
        self.gen_sum = 0.0;
        self.prompt_sum = 0.0;
        self.gen_n = 0;
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
