//! Tokens · 24 h: the 40-point integer cascade and its counter feed.

use kraken_lcd::tokens::{
    Bucket, GEN_TAU_S, GenRate, SLOTS, TIERS, TokenChart, TokenFeed, gen_rate_text, hold_gen_tenths,
};
use llama_core::sample::TokenReading;

fn reading(run_id: u64, seq: u64, t_ms: u64, total: Option<u64>) -> TokenReading {
    TokenReading {
        run_id,
        seq,
        t_mono_ns: t_ms * 1_000_000,
        decoded_total: total,
    }
}

#[test]
fn tiers_cover_exactly_24_hours_in_40_points() {
    let points: usize = TIERS.iter().map(|(_, cap)| cap).sum();
    assert_eq!(points, SLOTS);
    let span: u64 = TIERS
        .iter()
        .map(|(ms, cap)| u64::from(*ms) * *cap as u64)
        .sum();
    // 10×30 s + 11×5 min + 10×30 min + 9×2 h = 5 min + 55 min + 5 h + 18 h.
    assert_eq!(span, 24 * 3_600_000);
    // Carries merge whole buckets: 10 × 30 s = 5 min, 6 × 5 min = 30 min,
    // 4 × 30 min = 2 h.
    for pair in TIERS.windows(2) {
        assert_eq!(pair[1].0 % pair[0].0, 0, "{pair:?}");
    }
}

#[test]
fn rollover_sums_are_exact_and_nothing_is_rebinned() {
    let mut chart = TokenChart::default();
    let mut added = Bucket::default();
    // Two days at 10 Hz with an uneven token stream, in 100 ms steps.
    let mut state = 12_345_u64;
    for _ in 0..(2 * 24 * 3600 * 10) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let tok = (state >> 59) & 0x1F;
        chart.add(tok, 100);
        added.tok += tok;
        let held = chart.held();
        let dropped = chart.dropped();
        assert_eq!(
            held.tok + dropped.tok,
            added.tok,
            "every token is held or fell off the 24 h end"
        );
    }
    let held = chart.held();
    let dropped = chart.dropped();
    assert_eq!(
        u64::from(held.ms) + u64::from(dropped.ms),
        2 * 24 * 3600 * 1000,
        "every millisecond is accounted for"
    );
    // The held span is 24 h plus at most the carries and the live bucket.
    assert!(u64::from(held.ms) >= 24 * 3_600_000, "{}", held.ms);
    assert!(
        u64::from(held.ms) <= 24 * 3_600_000 + 30_000 + 300_000 + 1_800_000 + 7_200_000,
        "{}",
        held.ms
    );
}

#[test]
fn a_constant_rate_reads_the_same_on_every_point() {
    let mut chart = TokenChart::default();
    // 42 tok/s for 30 h, fed as 4.2 tokens per 100 ms (21 per 500 ms).
    for _ in 0..(30 * 3600 * 2) {
        chart.add(21, 500);
    }
    let points = chart.points();
    assert_eq!(points.len(), SLOTS);
    for (i, rate) in points.iter().enumerate() {
        assert!((rate - 42.0).abs() < 1e-3, "point {i} reads {rate}");
    }
}

#[test]
fn points_are_newest_first_and_the_live_bucket_waits_for_3_s() {
    let mut chart = TokenChart::default();
    // 5 min at 10 tok/s, then 5 min at 100 tok/s.
    for _ in 0..300 {
        chart.add(10, 1_000);
    }
    for _ in 0..300 {
        chart.add(100, 1_000);
    }
    let points = chart.points();
    assert!((points[0] - 100.0).abs() < 1e-3, "{points:?}");
    assert!((points.last().copied().unwrap_or(0.0) - 10.0).abs() < 1e-3);
    chart.add(50, 2_000);
    assert_eq!(
        chart.points(),
        points,
        "2 s in the live bucket is not plotted"
    );
    chart.add(50, 1_000);
    let with_live = chart.points();
    assert!((with_live[0] - 100.0 / 3.0).abs() < 1e-3, "{with_live:?}");
    assert_eq!(with_live.len(), points.len() + 1);
}

#[test]
fn memory_is_bounded_after_days_of_samples() {
    let mut chart = TokenChart::default();
    for _ in 0..(5 * 24 * 3600) {
        chart.add(7, 1_000);
    }
    // 40 buckets, 4 carries and the live one.
    assert!(chart.stored() <= 40 + 4 + 1, "{}", chart.stored());
    assert!(chart.points().len() <= SLOTS);
    // The held milliseconds do not grow without bound.
    assert!(u64::from(chart.held().ms) < 36 * 3_600_000);
}

#[test]
fn a_counter_reset_or_model_swap_is_a_gap_not_a_negative() {
    let mut feed = TokenFeed::new(2.0);
    assert_eq!(feed.accept(reading(1, 1, 1_000, Some(500))).interval, None);
    let step = feed.accept(reading(1, 2, 1_100, Some(510)));
    assert!(step.fresh);
    assert_eq!(step.interval, Some((10, 100)));
    // The counter goes down: a restart or a swapped model.
    let reset = feed.accept(reading(1, 3, 1_200, Some(3)));
    assert!(reset.fresh);
    assert_eq!(reset.interval, None, "a decrease adds nothing");
    // The next interval counts from the new baseline.
    assert_eq!(
        feed.accept(reading(1, 4, 1_300, Some(8))).interval,
        Some((5, 100))
    );
    // A missing counter is a gap on both sides.
    assert_eq!(feed.accept(reading(1, 5, 1_400, None)).interval, None);
    assert_eq!(feed.accept(reading(1, 6, 1_500, Some(20))).interval, None);
    // A new watcher run is a gap even when the counter is higher.
    assert_eq!(feed.accept(reading(2, 1, 1_600, Some(900))).interval, None);
    // A stall beyond max_gap_s is a gap.
    assert_eq!(feed.accept(reading(2, 2, 4_000, Some(950))).interval, None);
    // A re-read is not fresh and does not move the baseline.
    let reread = feed.accept(reading(2, 2, 4_000, Some(950)));
    assert!(!reread.fresh);
    assert_eq!(
        feed.accept(reading(2, 3, 4_100, Some(960))).interval,
        Some((10, 100))
    );

    // Through the chart: the reset leaves no negative anywhere.
    let mut chart = TokenChart::default();
    let mut feed = TokenFeed::new(2.0);
    let mut total = 1_000_u64;
    for seq in 0..600 {
        if seq == 300 {
            total = 0;
        }
        total += 5;
        if let Some((tok, ms)) = feed
            .accept(reading(1, seq, 10_000 + seq * 100, Some(total)))
            .interval
        {
            chart.add(tok, ms);
        }
    }
    let points = chart.points();
    assert!(!points.is_empty());
    assert!(
        points.iter().all(|rate| (*rate - 50.0).abs() < 1e-3),
        "5 tokens per 100 ms is 50 tok/s either side of the reset: {points:?}"
    );
}

#[test]
fn current_rate_text_at_each_range() {
    assert_eq!(gen_rate_text(None), "\u{2014}", "no counter or no data");
    assert_eq!(gen_rate_text(Some(0)), "0", "idle");
    assert_eq!(gen_rate_text(Some(1)), "0.1");
    assert_eq!(gen_rate_text(Some(75)), "7.5");
    assert_eq!(gen_rate_text(Some(99)), "9.9");
    assert_eq!(gen_rate_text(Some(100)), "10");
    assert_eq!(gen_rate_text(Some(1_120)), "112");
    assert_eq!(gen_rate_text(Some(9_990)), "999");
    assert_eq!(gen_rate_text(Some(10_000)), "1.0k");
    assert_eq!(gen_rate_text(Some(12_000)), "1.2k");
    assert_eq!(gen_rate_text(Some(99_000)), "9.9k");
    assert_eq!(gen_rate_text(Some(120_000)), "12k");
    assert_eq!(gen_rate_text(Some(1_250_000)), "125k");
}

/// One fresh reading every `step_ms` with `tok` new tokens.
fn feed_rate(
    rate: &mut GenRate,
    feed: &mut TokenFeed,
    start: (u64, u64),
    n: u64,
    step_ms: u64,
    tok: u64,
) -> (u64, u64) {
    let (mut seq, mut total) = start;
    for _ in 0..n {
        seq += 1;
        total += tok;
        let r = reading(1, seq, seq * step_ms, Some(total));
        rate.step(feed.accept(r), true);
    }
    (seq, total)
}

#[test]
fn current_rate_is_a_five_second_average_reset_by_a_gap() {
    let mut rate = GenRate::default();
    assert_eq!(rate.rate(), None, "no counter yet");
    let mut feed = TokenFeed::new(2.0);
    // A reading without a counter is still no counter.
    rate.step(feed.accept(reading(1, 0, 0, None)), false);
    assert_eq!(rate.rate(), None);

    // 500 ms ticks, 50 tokens each: 100 tok/s.
    let at = feed_rate(&mut rate, &mut feed, (0, 0), 1, 500, 0);
    assert_eq!(rate.rate(), Some(0.0), "the first counted reading is a gap");
    let at = feed_rate(&mut rate, &mut feed, at, 10, 500, 50);
    // After 5 s (one time constant) it has covered 1 − 1/e of the step.
    let expect = 100.0 * (1.0 - (-5.0 / GEN_TAU_S).exp()) as f32;
    let got = rate.rate().expect("seen");
    assert!((got - expect).abs() < 0.01, "{got} vs {expect}");
    let at = feed_rate(&mut rate, &mut feed, at, 120, 500, 50);
    assert!((rate.rate().expect("seen") - 100.0).abs() < 0.1);

    // Prefill: no new tokens. Under 3 s it decays with τ = 5 s...
    let at = feed_rate(&mut rate, &mut feed, at, 5, 500, 0);
    let decayed = rate.rate().expect("seen");
    let expect = 100.0 * (-2.5 / GEN_TAU_S).exp() as f32;
    assert!((decayed - expect).abs() < 0.1, "{decayed} vs {expect}");

    // A re-read changes nothing.
    let before = rate.rate();
    rate.step(feed.accept(reading(1, at.0, at.0 * 500, Some(at.1))), true);
    assert_eq!(rate.rate(), before);

    // ...and at 3 s without a token it snaps to zero.
    let at = feed_rate(&mut rate, &mut feed, at, 1, 500, 0);
    assert_eq!(rate.rate(), Some(0.0), "3 s idle");
    let at = feed_rate(&mut rate, &mut feed, at, 1, 500, 50);
    assert!(rate.rate().expect("seen") > 0.0, "tokens again");

    // A stall longer than max_gap_s resets to zero.
    let (seq, total) = at;
    rate.step(
        feed.accept(reading(1, seq + 1, seq * 500 + 3_000, Some(total + 300))),
        true,
    );
    assert_eq!(rate.rate(), Some(0.0), "stall");

    // So does a run_id change, and a snapshot without a reading.
    let at = feed_rate(&mut rate, &mut feed, (seq + 1, total + 300), 20, 500, 50);
    assert!(rate.rate().expect("seen") > 10.0);
    rate.step(feed.accept(reading(9, 0, at.0 * 500 + 500, Some(5))), true);
    assert_eq!(rate.rate(), Some(0.0), "run_id change");
    rate.step(
        feed.accept(reading(9, 1, at.0 * 500 + 1_000, Some(55))),
        true,
    );
    assert!(rate.rate().expect("seen") > 0.0);
    rate.gap();
    assert_eq!(rate.rate(), Some(0.0), "missing reading");
}

#[test]
fn current_rate_holds_inside_a_thirty_percent_margin() {
    // No shown value: round to the range's step.
    assert_eq!(hold_gen_tenths(7.46, None), 75);
    assert_eq!(hold_gen_tenths(112.4, None), 1_120);
    assert_eq!(hold_gen_tenths(1_234.0, None), 12_000);
    assert_eq!(hold_gen_tenths(12_345.0, None), 120_000);
    assert_eq!(hold_gen_tenths(0.04, None), 0);
    assert_eq!(hold_gen_tenths(f32::NAN, None), 0);
    assert_eq!(hold_gen_tenths(-3.0, None), 0);

    // Below 10: step 0.1, held within 0.08.
    assert_eq!(hold_gen_tenths(7.57, Some(75)), 75);
    assert_eq!(hold_gen_tenths(7.43, Some(75)), 75);
    assert_eq!(hold_gen_tenths(7.59, Some(75)), 76);
    assert_eq!(hold_gen_tenths(7.41, Some(75)), 74);
    // 10..999: step 1, held within 0.8.
    assert_eq!(hold_gen_tenths(112.79, Some(1_120)), 1_120);
    assert_eq!(hold_gen_tenths(111.21, Some(1_120)), 1_120);
    assert_eq!(hold_gen_tenths(112.81, Some(1_120)), 1_130);
    assert_eq!(hold_gen_tenths(111.19, Some(1_120)), 1_110);
    // From 1000: step 100, held within 80.
    assert_eq!(hold_gen_tenths(1_279.0, Some(12_000)), 12_000);
    assert_eq!(hold_gen_tenths(1_121.0, Some(12_000)), 12_000);
    assert_eq!(hold_gen_tenths(1_281.0, Some(12_000)), 13_000);
    // Across the 10 tok/s edge: 10 holds down to 9.2, then 0.1 steps.
    assert_eq!(hold_gen_tenths(9.3, Some(100)), 100);
    assert_eq!(hold_gen_tenths(9.1, Some(100)), 91);
    assert_eq!(hold_gen_tenths(9.9, Some(99)), 99);
    assert_eq!(hold_gen_tenths(10.2, Some(99)), 100);
    // Idle: 0.1 decays to 0 once below 0.02.
    assert_eq!(hold_gen_tenths(0.03, Some(1)), 1);
    assert_eq!(hold_gen_tenths(0.01, Some(1)), 0);

    // A noisy 112 ± 0.6 never changes the text.
    let mut shown = Some(hold_gen_tenths(112.0, None));
    for k in 0..200 {
        let raw = 112.0 + 0.6 * ((k as f32) * 0.7).sin();
        shown = Some(hold_gen_tenths(raw, shown));
        assert_eq!(shown, Some(1_120), "k = {k}, raw = {raw}");
    }
}

#[test]
fn current_rate_snaps_to_zero_after_three_idle_seconds() {
    let mut rate = GenRate::default();
    let mut feed = TokenFeed::new(2.0);
    // 100 ms ticks at 90 tok/s for a minute.
    let at = feed_rate(&mut rate, &mut feed, (0, 0), 600, 100, 9);
    assert!((rate.rate().expect("seen") - 90.0).abs() < 0.1);
    // Pauses shorter than 3 s, broken by tokens, never add up: the
    // average decays through them but never snaps.
    let mut at = at;
    for _ in 0..5 {
        at = feed_rate(&mut rate, &mut feed, at, 29, 100, 0);
        assert!(rate.rate().expect("seen") > 0.0, "2.9 s idle still decays");
        at = feed_rate(&mut rate, &mut feed, at, 1, 100, 9);
    }
    // Back at 90, then 2.9 s still: 90 × e^(−2.9/5) ≈ 50; at 3.0 s, zero,
    // and it stays there.
    let at = feed_rate(&mut rate, &mut feed, at, 600, 100, 9);
    let at = feed_rate(&mut rate, &mut feed, at, 29, 100, 0);
    let held = rate.rate().expect("seen");
    let expect = 90.0 * (-2.9 / GEN_TAU_S).exp() as f32;
    assert!((held - expect).abs() < 0.1, "{held} vs {expect}");
    let at = feed_rate(&mut rate, &mut feed, at, 1, 100, 0);
    assert_eq!(rate.rate(), Some(0.0), "3 s without a token");
    let at = feed_rate(&mut rate, &mut feed, at, 50, 100, 0);
    assert_eq!(rate.rate(), Some(0.0));
    // Tokens again: it climbs with τ = 5 s from zero.
    feed_rate(&mut rate, &mut feed, at, 10, 100, 9);
    let expect = 90.0 * (1.0 - (-1.0 / GEN_TAU_S).exp()) as f32;
    let got = rate.rate().expect("seen");
    assert!((got - expect).abs() < 0.01, "{got} vs {expect}");
}
