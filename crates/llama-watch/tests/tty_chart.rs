//! Token-chart ring: one bucket per `chart_bucket_s`, capped at the widest tty.

use std::time::{Duration, Instant};

use llama_watch::tty::chart::{
    ChartBucket, Ink, MAX_BUCKETS, TokenChart, eighths, fall_eighths, fall_glyphs, halves,
    rise_eighths, rise_glyphs,
};

fn t0() -> Instant {
    Instant::now()
}

fn at(origin: Instant, ms: u64) -> Instant {
    origin + Duration::from_millis(ms)
}

#[test]
fn first_sample_opens_a_bucket_with_that_rate() {
    let mut chart = TokenChart::new(2);
    let origin = t0();
    chart.ingest(origin, Some(54.0), Some(1200.0));
    assert_eq!(
        chart.buckets(),
        vec![ChartBucket {
            gen_tps: Some(54.0),
            prompt_tps: Some(1200.0),
        }]
    );
}

#[test]
fn samples_in_the_same_bucket_are_the_mean() {
    let mut chart = TokenChart::new(2);
    let origin = t0();
    chart.ingest(at(origin, 0), Some(10.0), Some(100.0));
    chart.ingest(at(origin, 100), Some(30.0), Some(300.0));
    chart.ingest(at(origin, 1999), Some(20.0), None);
    assert_eq!(
        chart.buckets(),
        vec![ChartBucket {
            gen_tps: Some(20.0),
            prompt_tps: Some(200.0),
        }]
    );
}

#[test]
fn crossing_the_bucket_width_commits_and_starts_a_new_column() {
    let mut chart = TokenChart::new(2);
    let origin = t0();
    chart.ingest(origin, Some(10.0), Some(0.0));
    chart.ingest(at(origin, 2000), Some(40.0), Some(80.0));
    assert_eq!(
        chart.buckets(),
        vec![
            ChartBucket {
                gen_tps: Some(10.0),
                prompt_tps: Some(0.0),
            },
            ChartBucket {
                gen_tps: Some(40.0),
                prompt_tps: Some(80.0),
            },
        ]
    );
}

#[test]
fn a_bucket_of_only_none_is_no_data() {
    let mut chart = TokenChart::new(2);
    let origin = t0();
    chart.ingest(origin, None, None);
    chart.ingest(at(origin, 100), None, None);
    assert_eq!(
        chart.buckets(),
        vec![ChartBucket {
            gen_tps: None,
            prompt_tps: None,
        }]
    );
}

#[test]
fn measured_zero_stays_some_zero() {
    let mut chart = TokenChart::new(2);
    chart.ingest(t0(), Some(0.0), Some(0.0));
    assert_eq!(
        chart.buckets(),
        vec![ChartBucket {
            gen_tps: Some(0.0),
            prompt_tps: Some(0.0),
        }]
    );
}

#[test]
fn a_time_jump_inserts_gap_buckets() {
    let mut chart = TokenChart::new(2);
    let origin = t0();
    chart.ingest(origin, Some(5.0), Some(6.0));
    chart.ingest(at(origin, 10_000), Some(7.0), Some(8.0));
    let buckets = chart.buckets();
    assert_eq!(buckets.len(), 6, "{buckets:?}");
    assert_eq!(
        buckets[0],
        ChartBucket {
            gen_tps: Some(5.0),
            prompt_tps: Some(6.0),
        }
    );
    for gap in &buckets[1..5] {
        assert_eq!(
            *gap,
            ChartBucket {
                gen_tps: None,
                prompt_tps: None,
            }
        );
    }
    assert_eq!(
        buckets[5],
        ChartBucket {
            gen_tps: Some(7.0),
            prompt_tps: Some(8.0),
        }
    );
}

#[test]
fn ring_drops_the_oldest_past_the_widest_terminal() {
    let mut chart = TokenChart::new(1);
    let origin = t0();
    for i in 0..=MAX_BUCKETS {
        chart.ingest(at(origin, i as u64 * 1000), Some(i as f64), Some(0.0));
    }
    let buckets = chart.buckets();
    assert_eq!(buckets.len(), MAX_BUCKETS);
    assert_eq!(buckets[0].gen_tps, Some(1.0));
    assert_eq!(buckets[MAX_BUCKETS - 1].gen_tps, Some(MAX_BUCKETS as f64));
}

#[test]
fn a_huge_time_jump_fills_at_most_the_ring() {
    let mut chart = TokenChart::new(1);
    let origin = t0();
    chart.ingest(origin, Some(1.0), None);
    chart.ingest(at(origin, 1_000_000_000), Some(2.0), None);
    let buckets = chart.buckets();
    assert_eq!(buckets.len(), MAX_BUCKETS);
    assert_eq!(buckets[MAX_BUCKETS - 1].gen_tps, Some(2.0));
}

#[test]
fn halves_hit_the_quantisation_edges() {
    assert_eq!(halves(0.0, 4), 0);
    assert_eq!(halves(1.0 / 8.0, 4), 1);
    assert_eq!(halves(2.0 / 8.0, 4), 2);
    assert_eq!(halves(7.0 / 8.0, 4), 7);
    assert_eq!(halves(1.0, 4), 8);
    assert_eq!(halves(2.0, 4), 8, "clamped at the ceiling");
    assert_eq!(halves(1.0, 3), 6);
    assert_eq!(halves(1.0 / 6.0, 3), 1);
}

#[test]
fn rise_glyphs_grow_up_from_the_axis_in_half_cells() {
    assert_eq!(rise_glyphs(0, 4), vec![' ', ' ', ' ', ' ']);
    assert_eq!(rise_glyphs(1, 4), vec![' ', ' ', ' ', '▄']);
    assert_eq!(rise_glyphs(2, 4), vec![' ', ' ', ' ', '█']);
    assert_eq!(rise_glyphs(3, 4), vec![' ', ' ', '▄', '█']);
    assert_eq!(rise_glyphs(8, 4), vec!['█', '█', '█', '█']);
    assert_eq!(rise_glyphs(1, 3), vec![' ', ' ', '▄']);
    assert_eq!(rise_glyphs(6, 3), vec!['█', '█', '█']);
}

#[test]
fn fall_glyphs_grow_down_from_the_axis_in_half_cells() {
    assert_eq!(fall_glyphs(0, 4), vec![' ', ' ', ' ', ' ']);
    assert_eq!(fall_glyphs(1, 4), vec!['▀', ' ', ' ', ' ']);
    assert_eq!(fall_glyphs(2, 4), vec!['█', ' ', ' ', ' ']);
    assert_eq!(fall_glyphs(3, 4), vec!['█', '▀', ' ', ' ']);
    assert_eq!(fall_glyphs(8, 4), vec!['█', '█', '█', '█']);
    assert_eq!(fall_glyphs(1, 3), vec!['▀', ' ', ' ']);
    assert_eq!(fall_glyphs(6, 3), vec!['█', '█', '█']);
}

#[test]
fn eighths_hit_the_quantisation_edges() {
    assert_eq!(eighths(0.0, 4), 0);
    assert_eq!(eighths(1.0 / 32.0, 4), 1);
    assert_eq!(eighths(2.0 / 32.0, 4), 2);
    assert_eq!(eighths(7.0 / 32.0, 4), 7);
    assert_eq!(eighths(8.0 / 32.0, 4), 8);
    assert_eq!(eighths(17.0 / 32.0, 4), 17);
    assert_eq!(eighths(31.0 / 32.0, 4), 31);
    assert_eq!(eighths(1.0, 4), 32);
    assert_eq!(eighths(2.0, 4), 32, "clamped at the ceiling");
    assert_eq!(eighths(-1.0, 4), 0);
    assert_eq!(eighths(f64::NAN, 4), 0);
    assert_eq!(eighths(1.0, 3), 24);
    assert_eq!(eighths(1.0 / 24.0, 3), 1);
    assert_eq!(eighths(1.0, 0), 0);
}

const E: Ink = Ink::Empty;

fn fg(ch: char) -> Ink {
    Ink::Fg(ch)
}

fn inv(ch: char) -> Ink {
    Ink::Inverse(ch)
}

#[test]
fn rise_eighths_grow_up_from_the_axis_in_lower_eighths() {
    assert_eq!(rise_eighths(0, 4), vec![E, E, E, E]);
    let lower = ['▁', '▂', '▃', '▄', '▅', '▆', '▇'];
    for (n, ch) in lower.into_iter().enumerate() {
        assert_eq!(
            rise_eighths(n as u32 + 1, 4),
            vec![E, E, E, fg(ch)],
            "level {}",
            n + 1
        );
    }
    assert_eq!(rise_eighths(8, 4), vec![E, E, E, fg('█')]);
    assert_eq!(rise_eighths(9, 4), vec![E, E, fg('▁'), fg('█')]);
    assert_eq!(rise_eighths(20, 4), vec![E, fg('▄'), fg('█'), fg('█')]);
    assert_eq!(
        rise_eighths(31, 4),
        vec![fg('▇'), fg('█'), fg('█'), fg('█')]
    );
    assert_eq!(rise_eighths(32, 4), vec![fg('█'); 4]);
    assert_eq!(rise_eighths(99, 3), vec![fg('█'); 3], "clamped to the rows");
}

/// The upper fractions other than ▀ and ▔ are not in Unicode block elements,
/// so the falling tip is a lower eighth drawn in inverse video: the top n/8
/// is the background colour.
#[test]
fn fall_eighths_grow_down_from_the_axis_in_inverted_lower_eighths() {
    assert_eq!(fall_eighths(0, 4), vec![E, E, E, E]);
    let lower_of_rest = ['▇', '▆', '▅', '▄', '▃', '▂', '▁'];
    for (n, ch) in lower_of_rest.into_iter().enumerate() {
        assert_eq!(
            fall_eighths(n as u32 + 1, 4),
            vec![inv(ch), E, E, E],
            "level {}",
            n + 1
        );
    }
    assert_eq!(fall_eighths(8, 4), vec![fg('█'), E, E, E]);
    assert_eq!(fall_eighths(9, 4), vec![fg('█'), inv('▇'), E, E]);
    assert_eq!(fall_eighths(20, 4), vec![fg('█'), fg('█'), inv('▄'), E]);
    assert_eq!(fall_eighths(32, 4), vec![fg('█'); 4]);
}
