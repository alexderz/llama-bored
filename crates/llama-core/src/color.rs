//! `act_color`: one colour function for the LCD (ring, bars, readout, tokens
//! chart) and the RGB writer (llama-light). Moved here from kraken-lcd's
//! `render/color.rs` so both writers share one ramp.
//!
//! 0–100 is the token-dial ramp L1..L6 placed at 0/20/40/60/80/100 (T62:
//! deep, saturated blues at the cold end). 100–125
//! keeps heating like a black body: red → orange → bright orange-gold
//! (T62: no white at the top, it stopped reading as heat). Both are
//! piecewise-linear in sRGB over the eleven stops, input clamped to 0–125.

/// One sRGB channel triple.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Rgb {
    /// Red, 0–255.
    pub r: u8,
    /// Green, 0–255.
    pub g: u8,
    /// Blue, 0–255.
    pub b: u8,
}

const fn rgb(hex: u32) -> Rgb {
    Rgb {
        r: ((hex >> 16) & 0xFF) as u8,
        g: ((hex >> 8) & 0xFF) as u8,
        b: (hex & 0xFF) as u8,
    }
}

/// 0–100: L1..L6.
pub const ACT_STOPS: [(f32, Rgb); 6] = [
    (0.0, rgb(0x1428D8)),
    (20.0, rgb(0x3A1EF0)),
    (40.0, rgb(0x7A1FE8)),
    (60.0, rgb(0xC21CC8)),
    (80.0, rgb(0xF21E5A)),
    (100.0, rgb(0xFF2A14)),
];

/// 100–125: blackbody. The 100 stop is L6, so the two ramps join without a
/// step.
pub const BB_STOPS: [(f32, Rgb); 5] = [
    (100.0, rgb(0xFF2A14)),
    (107.0, rgb(0xFF5A12)),
    (114.0, rgb(0xFF8A1E)),
    (120.0, rgb(0xFFB030)),
    (125.0, rgb(0xFFD050)),
];

/// `#FFE6A0`: hot pale gold, the hottest touch anywhere (white-hot core,
/// halo, filament, embers). Never pure white, so 125 still reads as heat.
pub const HOT_GOLD: Rgb = rgb(0xFFE6A0);

/// `#000000`.
pub const BLACK: Rgb = rgb(0x000000);
/// `#FFFFFF`.
pub const WHITE: Rgb = rgb(0xFFFFFF);

/// Colour of activity `p` (0–125).
#[must_use]
pub fn act_color(p: f32) -> Rgb {
    let p = if p.is_finite() { p } else { 0.0 };
    if p <= 100.0 {
        ramp(&ACT_STOPS, p)
    } else {
        ramp(&BB_STOPS, p.min(125.0))
    }
}

/// Piecewise-linear colour over `stops` (positions ascending). `p` below the
/// first stop is the first colour; above the last stop, the last colour.
#[must_use]
pub fn ramp(stops: &[(f32, Rgb)], p: f32) -> Rgb {
    let Some(&(first_p, first)) = stops.first() else {
        return BLACK;
    };
    if p <= first_p {
        return first;
    }
    for pair in stops.windows(2) {
        let (p0, c0) = pair[0];
        let (p1, c1) = pair[1];
        if p <= p1 {
            return mix(c0, c1, (p - p0) / (p1 - p0));
        }
    }
    stops.last().map_or(BLACK, |stop| stop.1)
}

/// `amount` of `toward` mixed into `color`, rounded per channel.
#[must_use]
pub fn mix(color: Rgb, toward: Rgb, amount: f32) -> Rgb {
    let amount = if amount.is_finite() { amount } else { 0.0 };
    let channel = |from: u8, to: u8| -> u8 {
        let mixed = f32::from(from) + (f32::from(to) - f32::from(from)) * amount;
        mixed.round().clamp(0.0, 255.0) as u8
    };
    Rgb {
        r: channel(color.r, toward.r),
        g: channel(color.g, toward.g),
        b: channel(color.b, toward.b),
    }
}

/// `rgb` from a `0xRRGGBB` literal.
#[must_use]
pub const fn hex(value: u32) -> Rgb {
    rgb(value)
}

/// Positions on the activity ramp for the grid file's exact colours.
///
/// The console paints the nearest palette step. These functions return the
/// ramp position of the same reading, and [`act_color`] turns that into the
/// colour the grid file writes. The five spectrum bands are the five gaps
/// between [`ACT_STOPS`]: a cell at the centre of band `k` sits on stop
/// `k`. Above 100 the position continues into [`BB_STOPS`].
pub mod spectrum {
    use super::{ACT_STOPS, BB_STOPS};

    /// How far a digit's top row sits above its value, and its bottom row
    /// below it, in ramp points.
    pub const SHADE: f32 = 14.0;

    /// Five bands, one per gap in [`ACT_STOPS`].
    const BANDS: usize = ACT_STOPS.len() - 1;

    /// Ramp position of the cell at `index` in a run of `len`, measured at
    /// the cell's centre. Band centres land on the stops 0/20/40/60/80. The
    /// right edge sits between L5 and L6. An empty run is the first stop.
    #[must_use]
    pub fn place(index: usize, len: usize) -> f32 {
        if len == 0 {
            return ACT_STOPS[0].0;
        }
        let frac = (index as f32 + 0.5) / len as f32;
        from_frac(frac)
    }

    /// Ramp position of `value` on the logarithmic rate scale whose top is
    /// `ceiling` (the same scale that picks the console's step). The centre
    /// of each of the five steps lands on stops 0/20/40/60/80. At `ceiling`
    /// the position is between L5 and L6; above it the position climbs into
    /// the blackbody tail. A non-finite, non-positive value or ceiling is
    /// the first stop.
    #[must_use]
    pub fn rate(value: f64, ceiling: f64) -> f32 {
        if !value.is_finite() || !ceiling.is_finite() || value <= 0.0 || ceiling <= 0.0 {
            return ACT_STOPS[0].0;
        }
        let bands = BANDS as f64;
        let frac = ((value / ceiling).log2() + bands) / bands;
        from_frac(frac as f32)
    }

    /// `pos` shaded across `rows`: row 0 is [`SHADE`] points above `pos`,
    /// the last row is [`SHADE`] points below, linear between. One row is
    /// `pos` itself. [`act_color`] clamps the result to the ramp.
    #[must_use]
    pub fn shade(pos: f32, row: usize, rows: usize) -> f32 {
        if rows <= 1 {
            return pos;
        }
        let t = row as f32 / (rows - 1) as f32;
        pos + SHADE * (1.0 - 2.0 * t)
    }

    /// Ramp position of a hot bar's top row in the last fifth. That fifth
    /// runs from the first [`BB_STOPS`] position to the last, so the
    /// white-hot console cell continues into the blackbody tail. An empty
    /// run is the start of the tail.
    #[must_use]
    pub fn hot_top(index: usize, len: usize) -> f32 {
        let start = BB_STOPS[0].0;
        let end = BB_STOPS[BB_STOPS.len() - 1].0;
        if len == 0 {
            return start;
        }
        let frac = (index as f32 + 0.5) / len as f32;
        let fifth = (BANDS - 1) as f32 / BANDS as f32;
        let along = ((frac - fifth) / (1.0 - fifth)).clamp(0.0, 1.0);
        start + along * (end - start)
    }

    /// `frac` 0 is half a band below the first stop; `frac` 1 is half a
    /// band above the last of the five. That is what puts a band's centre
    /// on its stop.
    fn from_frac(frac: f32) -> f32 {
        let step = ACT_STOPS[1].0 - ACT_STOPS[0].0;
        frac * step * BANDS as f32 - step / 2.0
    }
}

/// Temperature colour anchored to a reading's warning and critical points.
///
/// The console still paints its palette level. The grid file writes this
/// ramp: the bright value colour at `warn - COOL_SPAN` and below, [`WARM`]
/// ten degrees under the warning, [`WARN`] at the warning, and [`CRIT`] at
/// the critical point and above.
pub mod temp {
    use super::{Rgb, WHITE, ramp};

    /// `#FFFFFF`, the colour at [`COOL_SPAN`] degrees under the warning and
    /// below. That is the bright value colour the console uses for a normal
    /// reading.
    pub const NORMAL: Rgb = WHITE;
    /// `#FFD050`, [`WARM_SPAN`] degrees under the warning.
    pub const WARM: Rgb = super::hex(0xFFD050);
    /// `#FFB030`, the warning threshold.
    pub const WARN: Rgb = super::hex(0xFFB030);
    /// `#FF2A14`, the critical threshold and above.
    pub const CRIT: Rgb = super::hex(0xFF2A14);

    /// Degrees under the warning where the colour is [`NORMAL`].
    pub const COOL_SPAN: f32 = 35.0;
    /// Degrees under the warning where the colour is [`WARM`].
    pub const WARM_SPAN: f32 = 10.0;

    /// Colour of `celsius` on the warning/critical anchors. `None` when any
    /// input is non-finite or `crit` is not above `warn`: a reading without
    /// a usable pair of thresholds keeps its palette colour.
    #[must_use]
    pub fn color(celsius: f32, warn: f32, crit: f32) -> Option<Rgb> {
        if !celsius.is_finite() || !warn.is_finite() || !crit.is_finite() || crit <= warn {
            return None;
        }
        let stops = [
            (warn - COOL_SPAN, NORMAL),
            (warn - WARM_SPAN, WARM),
            (warn, WARN),
            (crit, CRIT),
        ];
        Some(ramp(&stops, celsius))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::{Palette, role};

    fn step() -> f32 {
        ACT_STOPS[1].0 - ACT_STOPS[0].0
    }

    fn near(got: f32, want: f32) {
        assert!((got - want).abs() < 1.0e-3, "{got} vs {want}");
    }

    #[test]
    fn fifth_centres_sit_on_the_stops() {
        let bands = ACT_STOPS.len() - 1;
        for (k, stop) in ACT_STOPS.iter().enumerate().take(bands) {
            let pos = spectrum::place(k, bands);
            near(pos, stop.0);
            assert_eq!(act_color(pos), stop.1);
            assert_eq!(
                act_color(pos),
                Palette::Llama.rgb(role::SPECTRUM[k]),
                "band {k}"
            );
        }
        // The right-hand edge is halfway from L5 to L6, not on a stop. The
        // last cell of a wide bar is half a cell short of that edge.
        let edge = ACT_STOPS[bands - 1].0 + step() / 2.0;
        let wide = 1000;
        let short = step() * bands as f32 / (2.0 * wide as f32);
        near(spectrum::place(wide - 1, wide), edge - short);
        assert_eq!(spectrum::place(0, 0), ACT_STOPS[0].0);
    }

    #[test]
    fn a_wide_bar_is_close_at_each_band_centre() {
        let len = 67;
        let bands = ACT_STOPS.len() - 1;
        for k in 0..bands {
            let centre = ((k as f32 + 0.5) / bands as f32) * len as f32 - 0.5;
            let index = centre.round() as usize;
            let pos = spectrum::place(index, len);
            let want = k as f32 * step();
            assert!(
                (pos - want).abs() < step() / 10.0,
                "band {k}: cell {index} at {pos}, stop {want}"
            );
            let exact = act_color(pos);
            let slot = Palette::Llama.rgb(role::SPECTRUM[k]);
            for (got, want_c) in [(exact.r, slot.r), (exact.g, slot.g), (exact.b, slot.b)] {
                assert!(
                    got.abs_diff(want_c) <= 12,
                    "band {k} channel {got} vs {want_c}"
                );
            }
        }
    }

    #[test]
    fn rate_centres_sit_on_the_same_stops() {
        let ceiling = 250.0;
        let bands = ACT_STOPS.len() - 1;
        for (k, stop) in ACT_STOPS.iter().enumerate().take(bands) {
            // Geometric centre of step k: halfway between ceiling/32 and
            // ceiling on a log scale, one band at a time.
            let log = k as f64 - (bands as f64 - 0.5);
            let value = ceiling * 2.0_f64.powf(log);
            let pos = spectrum::rate(value, ceiling);
            near(pos, stop.0);
            assert_eq!(act_color(pos), stop.1, "step {k}");
            assert_eq!(act_color(pos), Palette::Llama.rgb(role::SPECTRUM[k]));
        }
        near(
            spectrum::rate(ceiling, ceiling),
            ACT_STOPS[bands].0 - step() / 2.0,
        );
        // Past the ceiling the position keeps climbing, and it reaches the
        // end of the blackbody tail.
        assert!(spectrum::rate(ceiling * 2.0, ceiling) > ACT_STOPS[bands].0);
        let tail = BB_STOPS[BB_STOPS.len() - 1].0;
        let mut high = ceiling;
        while spectrum::rate(high, ceiling) < tail {
            high *= 2.0;
            assert!(high < ceiling * 64.0, "the tail stays reachable");
        }
        assert_eq!(
            act_color(spectrum::rate(high, ceiling)),
            BB_STOPS[BB_STOPS.len() - 1].1
        );
        assert_eq!(spectrum::rate(0.0, ceiling), ACT_STOPS[0].0);
        assert_eq!(spectrum::rate(-1.0, ceiling), ACT_STOPS[0].0);
        assert_eq!(spectrum::rate(f64::NAN, ceiling), ACT_STOPS[0].0);
        assert_eq!(spectrum::rate(1.0, 0.0), ACT_STOPS[0].0);
        assert_eq!(spectrum::rate(1.0, f64::INFINITY), ACT_STOPS[0].0);
    }

    #[test]
    fn shade_lifts_the_top_row_and_lowers_the_bottom() {
        let pos = 40.0;
        let rows = 5;
        near(spectrum::shade(pos, 0, rows), pos + spectrum::SHADE);
        near(spectrum::shade(pos, rows - 1, rows), pos - spectrum::SHADE);
        near(spectrum::shade(pos, 2, rows), pos);
        near(spectrum::shade(pos, 0, 1), pos);
        // Clamped to the ramp: a cold value's bottom row is still the first stop.
        assert_eq!(act_color(spectrum::shade(0.0, 4, 5)), ACT_STOPS[0].1);
        assert_eq!(
            act_color(spectrum::shade(BB_STOPS[BB_STOPS.len() - 1].0, 0, 5)),
            BB_STOPS[BB_STOPS.len() - 1].1
        );
    }

    #[test]
    fn hot_top_runs_the_blackbody_tail_across_the_last_fifth() {
        let len = 5;
        let last = len - 1;
        let pos = spectrum::hot_top(last, len);
        assert!(pos > BB_STOPS[0].0 && pos < BB_STOPS[BB_STOPS.len() - 1].0);
        // A cell whose centre is on the left of the last fifth is the start
        // of the tail. The last cell is half a cell short of the right edge.
        let len = 100;
        let bands = ACT_STOPS.len() - 1;
        let left = ((bands - 1) as f32 / bands as f32) * len as f32 - 0.5;
        near(spectrum::hot_top(left.floor() as usize, len), BB_STOPS[0].0);
        let end = BB_STOPS[BB_STOPS.len() - 1].0;
        let tail = end - BB_STOPS[0].0;
        let short = (0.5 / len as f32) / (1.0 / bands as f32) * tail;
        near(spectrum::hot_top(len - 1, len), end - short);
        assert_eq!(spectrum::hot_top(0, 0), BB_STOPS[0].0);
        let colour = act_color(pos);
        assert_ne!(colour, WHITE);
        assert_eq!(colour, ramp(&BB_STOPS, pos));
    }

    #[test]
    fn temperature_anchors_are_the_named_colours() {
        let warn = 80.0;
        let crit = 90.0;
        let at = |c| temp::color(c, warn, crit).expect("thresholds");
        assert_eq!(at(warn - temp::COOL_SPAN), temp::NORMAL);
        assert_eq!(at(warn - temp::COOL_SPAN - 20.0), temp::NORMAL);
        assert_eq!(at(warn - temp::WARM_SPAN), temp::WARM);
        assert_eq!(at(warn), temp::WARN);
        assert_eq!(at(crit), temp::CRIT);
        assert_eq!(at(crit + 15.0), temp::CRIT);
        // Between warn and crit the colour leaves the warning yellow.
        let mid = at((warn + crit) / 2.0);
        assert_ne!(mid, temp::WARN);
        assert_ne!(mid, temp::CRIT);
        assert_eq!(temp::NORMAL, WHITE);
        assert_eq!(temp::WARM, BB_STOPS[BB_STOPS.len() - 1].1);
        assert_eq!(temp::WARN, BB_STOPS[3].1);
        assert_eq!(temp::CRIT, ACT_STOPS[ACT_STOPS.len() - 1].1);
        assert_eq!(temp::WARN, Palette::Llama.rgb(role::WARN));
        assert!(temp::color(70.0, 90.0, 80.0).is_none());
        assert!(temp::color(f32::NAN, warn, crit).is_none());
        assert!(temp::color(70.0, f32::NAN, crit).is_none());
        assert!(temp::color(70.0, warn, warn).is_none());
    }
}
