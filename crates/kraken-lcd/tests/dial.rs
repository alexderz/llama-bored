//! Activity dial: per-bar mean activity over clock-aligned tier windows.
//!
//! Tiers are 10×0.5 s, 2×5 s, 3×15 s, 4×1 min, 5×5 min. Each tier's newest
//! bar collects until its window is full, then the tier shifts by one bar.

use kraken_lcd::activity::{ActivityDial, BARS, SLOT_NS};
use kraken_lcd::config::{Config, Dial as DialCfg, InvalidConfig, Tier};

const S: u64 = 1_000_000_000;

/// Feed `value(t)` at 10 Hz for `[from_s, to_s)` seconds.
fn feed(dial: &mut ActivityDial, from_s: f64, to_s: f64, value: impl Fn(f64) -> f32) {
    let mut t = from_s;
    while t < to_s - 1e-9 {
        dial.add((t * 1e9).round() as u64, value(t));
        t += 0.1;
    }
}

fn near(got: Option<f32>, expect: f32) -> bool {
    got.is_some_and(|got| (got - expect).abs() < 1e-3)
}

#[test]
fn each_tier_starts_on_its_own_clock_grid() {
    let dial = ActivityDial::default();
    let widths = [1_u64, 10, 30, 120, 600];
    // At any time, a tier's older bars start on a multiple of its width
    // (shifted by the faster tiers' span, itself a multiple here).
    for now_s in [3_600.0, 3_601.3, 3_659.9, 3_899.5, 4_199.0] {
        let now = (now_s * 1e9) as u64;
        let windows = dial.windows(now);
        let mut bar = 0;
        for (tier, width) in [10_usize, 2, 3, 4, 5].into_iter().zip(widths) {
            for j in 0..tier {
                let (start, end) = windows[bar];
                if j > 0 {
                    assert_eq!(start % width, 0, "bar {bar} at {now_s}: {start}..{end}");
                    assert_eq!(end - start, width, "bar {bar} is one whole window");
                } else {
                    assert_eq!(start % width, 0, "newest bar {bar} starts on the grid");
                    assert!(end - start <= width && end > start, "bar {bar}");
                }
                bar += 1;
            }
        }
        assert_eq!(bar, BARS);
        // Contiguous inside a tier: each bar ends where the newer one starts.
        for pair in windows.windows(2) {
            assert!(pair[1].1 <= pair[0].0, "{pair:?} overlaps at {now_s}");
        }
    }
}

#[test]
fn the_newest_bar_of_a_tier_collects_until_its_window_is_full() {
    let dial = ActivityDial::default();
    // 5 min tier (bars 19..24), 300 s behind now. At t = 3600 + 60 s its
    // newest window has run 60 of 300 s; 240 s later it is full and a
    // second later the tier has shifted.
    let at = |s: f64| dial.windows((s * 1e9) as u64)[19];
    let slot = |s: u64| s * S / SLOT_NS;
    let (start, end) = at(3_660.0);
    assert_eq!(start, slot(3_600 - 300));
    assert_eq!(
        end - start,
        slot(60) + 1,
        "60 s collected plus the current slot"
    );
    let (start2, end2) = at(3_899.9);
    assert_eq!(start2, start, "still the same window");
    assert_eq!(end2 - start2, slot(300), "full at the edge");
    let (start3, _) = at(3_900.0);
    assert_eq!(start3, slot(3_900 - 300), "shifted by one bar");
    let fill = dial.fill((3_660.0 * 1e9) as u64);
    assert!(
        (fill[4] - 0.2).abs() < 1e-6,
        "5 min tier is 60/300 full: {fill:?}"
    );
    assert!(
        (fill[3] - 0.0).abs() < 1e-6,
        "1 min tier just rolled: {fill:?}"
    );
}

#[test]
fn bars_are_the_mean_of_their_window() {
    let mut dial = ActivityDial::default();
    // 30 min at 88, then 25 s at 60, then 5 s at 118.
    feed(&mut dial, 0.0, 1_800.0, |_| 88.0);
    feed(&mut dial, 1_800.0, 1_825.0, |_| 60.0);
    feed(&mut dial, 1_825.0, 1_830.0, |_| 118.0);
    let now = 1_830 * S - 1;
    let means = dial.means(now);
    for (bar, mean) in means.iter().enumerate().take(10) {
        assert!(near(*mean, 118.0), "0.5 s bar {bar} is the spike: {mean:?}");
    }
    // 5 s tier: 1820..1825 is all 60.
    assert!(near(means[10], 60.0), "{:?}", means[10]);
    // Oldest 5 min bars are all 88.
    assert!(near(means[23], 88.0), "{:?}", means[23]);
    // 15 s tier on its grid: 1800..1815 is all 60, 1785..1800 all 88.
    assert!(near(means[12], 60.0), "{:?}", means[12]);
    assert!(near(means[13], 88.0), "{:?}", means[13]);
    // Over 100 survives: the spike reads 118, not 100.
    assert!(means.iter().flatten().any(|mean| *mean > 100.0));
}

#[test]
fn values_clamp_to_the_125_peg_and_non_finite_is_dropped() {
    let mut dial = ActivityDial::default();
    dial.add(10 * S, 400.0);
    dial.add(10 * S + 1, f32::NAN);
    dial.add(10 * S + 2, -5.0);
    let means = dial.means(10 * S + 3);
    assert!(
        near(means[0], 62.5),
        "(125 + 0) / 2, NaN dropped: {:?}",
        means[0]
    );
}

#[test]
fn a_gap_is_no_data_below_half_coverage() {
    let mut dial = ActivityDial::default();
    feed(&mut dial, 0.0, 1_000.0, |_| 50.0);
    // 40 s stall, then 2 s of samples.
    feed(&mut dial, 1_040.0, 1_042.0, |_| 50.0);
    let now = 1_042 * S - 1;
    let means = dial.means(now);
    assert!(means[0].is_some(), "the newest 0.5 s bar has data");
    // 5 s bars 1035..1037 (collecting) and 1030..1035 sit in the stall.
    assert_eq!(means[10], None, "a stalled 5 s bar is no data");
    assert_eq!(means[11], None);
    // 5 min bars are older than the stall and stay.
    assert!(near(means[20], 50.0), "{:?}", means[20]);
    // A fresh dial has no data anywhere.
    let empty = ActivityDial::default();
    assert_eq!(empty.means(now), [None; BARS]);
}

#[test]
fn memory_is_bounded_by_the_dial_span() {
    let mut dial = ActivityDial::default();
    // Three hours at 10 Hz.
    feed(&mut dial, 0.0, 3.0 * 3_600.0, |t| (t % 100.0) as f32);
    // 1800 s span = 3600 slots, plus one widest window (600) and one.
    assert!(dial.len() <= 3_600 + 600 + 2, "{}", dial.len());
    assert!(dial.len() >= 3_600, "{}", dial.len());
}

#[test]
fn xff_and_tiers_come_from_the_writer_config() {
    let settings = DialCfg {
        tiers: vec![
            Tier {
                width_s: 0.5,
                bars: 12,
            },
            Tier {
                width_s: 1.0,
                bars: 12,
            },
        ],
        xff: 1.0,
        ..DialCfg::default()
    };
    let mut strict = ActivityDial::new(&settings);
    let widths: Vec<_> = strict.tiers().iter().map(|t| (t.width, t.bars)).collect();
    assert_eq!(widths, [(1, 12), (2, 12)]);
    let mut half = ActivityDial::new(&DialCfg {
        xff: 0.5,
        ..settings.clone()
    });
    // At 101 s the second tier's first whole 1 s bar is 94..95 s. Only its
    // first slot has a sample.
    for dial in [&mut strict, &mut half] {
        dial.add(94 * S, 40.0);
        dial.add(101 * S, 40.0);
    }
    assert!(near(strict.means(101 * S)[0], 40.0));
    assert_eq!(strict.means(101 * S)[13], None, "half covered at xff 1.0");
    assert!(
        near(half.means(101 * S)[13], 40.0),
        "half covered at xff 0.5"
    );
}

#[test]
fn tier_validation_errors_come_from_the_writer_config() {
    fn tiers(pairs: &[(f64, u32)]) -> Vec<Tier> {
        pairs
            .iter()
            .copied()
            .map(|(width_s, bars)| Tier { width_s, bars })
            .collect()
    }
    assert!(Config::default().validate().is_ok());

    let mut cfg = Config::default();
    cfg.dial.tiers = Vec::new();
    assert!(matches!(
        cfg.validate(),
        Err(InvalidConfig::TierZero { width_s: None })
    ));

    cfg.dial.tiers = tiers(&[(1.0, 24)]);
    assert!(matches!(
        cfg.validate(),
        Err(InvalidConfig::TierZero { width_s: Some(width) }) if width == 1.0
    ));

    cfg.dial.tiers = tiers(&[(0.5, 20), (0.25, 4)]);
    assert!(matches!(
        cfg.validate(),
        Err(InvalidConfig::TierOrder { .. })
    ));

    cfg.dial.tiers = tiers(&[(0.5, 20), (0.7, 4)]);
    assert!(matches!(
        cfg.validate(),
        Err(InvalidConfig::TierMultiple { .. })
    ));

    cfg.dial.tiers = tiers(&[(0.5, 23)]);
    assert!(matches!(
        cfg.validate(),
        Err(InvalidConfig::TierBars { bars: 23 })
    ));
}
