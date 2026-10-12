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
/// colour the grid file writes. A bar or a rate scale runs the whole ramp:
/// the first stop at its low end, the last [`ACT_STOPS`] stop at full
/// scale (#123). The console's five steps are the five gaps between the
/// stops, so the middle of a band is half a band above the step the
/// console paints there. Above 100 the position continues into
/// [`BB_STOPS`].
pub mod spectrum {
    use super::{ACT_STOPS, BB_STOPS};

    /// How far a digit's top row sits above its value, and its bottom row
    /// below it, in ramp points.
    pub const SHADE: f32 = 14.0;

    /// Five bands, one per gap in [`ACT_STOPS`].
    const BANDS: usize = ACT_STOPS.len() - 1;

    /// The ramp position of full scale: the last [`ACT_STOPS`] stop. A bar's
    /// right edge and a rate at its ceiling are here.
    pub const FULL: f32 = ACT_STOPS[BANDS].0;

    /// Ramp position of the cell at `index` in a run of `len`, measured at
    /// the cell's centre: the first stop at the run's left edge, [`FULL`] at
    /// its right edge (#123). An empty run is the first stop.
    #[must_use]
    pub fn place(index: usize, len: usize) -> f32 {
        place_to(index, len, FULL)
    }

    /// [`place`] on a scale that ends at `top` instead of [`FULL`]: a bar
    /// past full scale is the same ramp stretched to its value, so no cell
    /// jumps when the value crosses 100 and the bar has no seam. `top` is
    /// held between [`FULL`] and the end of the blackbody tail.
    #[must_use]
    pub fn place_to(index: usize, len: usize, top: f32) -> f32 {
        if len == 0 {
            return ACT_STOPS[0].0;
        }
        let top = if top.is_finite() {
            top.clamp(FULL, BB_STOPS[BB_STOPS.len() - 1].0)
        } else {
            FULL
        };
        let frac = (index as f32 + 0.5) / len as f32;
        ACT_STOPS[0].0 + frac * (top - ACT_STOPS[0].0)
    }

    /// The `top` of a level meter at `percent` of full scale, for
    /// [`place_to`]: [`FULL`] up to 100 %, the level itself above that, at
    /// most the end of the blackbody tail. Not a number is [`FULL`].
    #[must_use]
    pub fn top_of(percent: f64) -> f32 {
        if percent.is_nan() {
            FULL
        } else {
            (percent as f32).clamp(FULL, BB_STOPS[BB_STOPS.len() - 1].0)
        }
    }

    /// Ramp position of `value` on the logarithmic rate scale whose top is
    /// `ceiling` (the same scale that picks the console's step): the first
    /// stop at the bottom of the scale, [`FULL`] at `ceiling`. Above it the
    /// position climbs into the blackbody tail. A non-finite, non-positive
    /// value or ceiling is the first stop.
    #[must_use]
    pub fn rate(value: f64, ceiling: f64) -> f32 {
        if !value.is_finite() || !ceiling.is_finite() || value <= 0.0 || ceiling <= 0.0 {
            return ACT_STOPS[0].0;
        }
        let bands = BANDS as f64;
        let frac = ((value / ceiling).log2() + bands) / bands;
        ACT_STOPS[0].0 + frac as f32 * (FULL - ACT_STOPS[0].0)
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
}

/// Temperature colour anchored to a reading's warning and critical points.
///
/// The console still paints its palette level. The grid file writes this
/// ramp: the bright value colour at `warn - COOL_SPAN` and below, [`WARM`]
/// at `warn - WARM_SPAN`, [`WARN`] at the warning, and [`CRIT`] at
/// the critical point and above.
pub mod temp {
    use super::{Rgb, WHITE, ramp};

    /// `#FFFFFF`, the colour at [`COOL_SPAN`] degrees under the warning and
    /// below. That is the bright value colour the console uses for a normal
    /// reading.
    pub const NORMAL: Rgb = WHITE;
    /// The last [`super::BB_STOPS`] colour, [`WARM_SPAN`] degrees under the warning.
    pub const WARM: Rgb = super::BB_STOPS[super::BB_STOPS.len() - 1].1;
    /// The blackbody stop at 120, the warning threshold. The llama palette's
    /// warn yellow is this same colour.
    pub const WARN: Rgb = super::BB_STOPS[3].1;
    /// The activity ramp's 100 stop, at the critical threshold and above.
    /// The console's critical level stays BrightRed (`#F21E5A` in the llama
    /// palette, the stop at 80). That difference is intended: the file
    /// follows this ramp.
    pub const CRIT: Rgb = super::ACT_STOPS[super::ACT_STOPS.len() - 1].1;

    /// Degrees under the warning where the colour is [`NORMAL`].
    ///
    /// The span is short on purpose (#125): a sensor whose normal range sits
    /// close under its warning, coolant for one, must still read as normal.
    pub const COOL_SPAN: f32 = 10.0;
    /// Degrees under the warning where the colour is [`WARM`].
    pub const WARM_SPAN: f32 = 4.0;

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

    /// From the first stop to full scale.
    const FULL_SPAN: f32 = spectrum::FULL - ACT_STOPS[0].0;

    fn near(got: f32, want: f32) {
        assert!((got - want).abs() < 1.0e-3, "{got} vs {want}");
    }

    /// Largest channel difference between two colours.
    fn gap(a: Rgb, b: Rgb) -> u8 {
        a.r.abs_diff(b.r)
            .max(a.g.abs_diff(b.g))
            .max(a.b.abs_diff(b.b))
    }

    #[test]
    fn a_bar_runs_from_the_first_stop_to_the_last() {
        let wide = 1000;
        let half = FULL_SPAN / (2.0 * wide as f32);
        near(spectrum::place(0, wide), ACT_STOPS[0].0 + half);
        near(spectrum::place(wide - 1, wide), spectrum::FULL - half);
        assert_eq!(spectrum::place(0, 0), ACT_STOPS[0].0);
        assert!(gap(act_color(spectrum::place(0, wide)), ACT_STOPS[0].1) <= 1);
        assert!(
            gap(
                act_color(spectrum::place(wide - 1, wide)),
                ACT_STOPS[ACT_STOPS.len() - 1].1
            ) <= 1
        );
        // A band's centre is half a band above the console's step for that
        // band: the console paints the band's first stop, the ramp its middle.
        let bands = ACT_STOPS.len() - 1;
        for (k, stop) in ACT_STOPS.iter().enumerate().take(bands) {
            near(spectrum::place(k, bands), stop.0 + step() / 2.0);
            assert_eq!(stop.1, Palette::Llama.rgb(role::SPECTRUM[k]), "band {k}");
        }
    }

    #[test]
    fn a_bar_changes_by_small_steps_at_any_top() {
        let len = 67;
        for top in [spectrum::FULL, 100.5, 110.0, 118.0, 125.0] {
            let mut last = spectrum::place_to(0, len, top);
            for index in 1..len {
                let pos = spectrum::place_to(index, len, top);
                assert!(pos > last, "top {top}: cell {index} goes back");
                assert!(
                    gap(act_color(pos), act_color(last)) <= 16,
                    "top {top}: a seam at cell {index}"
                );
                last = pos;
            }
        }
    }

    #[test]
    fn past_full_scale_the_bar_stretches_to_the_value() {
        let end = BB_STOPS[BB_STOPS.len() - 1].0;
        assert_eq!(spectrum::top_of(0.0), spectrum::FULL);
        assert_eq!(spectrum::top_of(100.0), spectrum::FULL);
        near(spectrum::top_of(118.0), 118.0);
        assert_eq!(spectrum::top_of(500.0), end);
        assert_eq!(spectrum::top_of(f64::NAN), spectrum::FULL);
        assert_eq!(spectrum::top_of(f64::INFINITY), end);
        assert_eq!(spectrum::top_of(f64::NEG_INFINITY), spectrum::FULL);
        assert_eq!(spectrum::top_of(-5.0), spectrum::FULL);

        let len = 67;
        // No jump when the value crosses 100: just above it every cell is
        // where it was.
        for index in 0..len {
            let at = spectrum::place(index, len);
            let above = spectrum::place_to(index, len, spectrum::top_of(100.5));
            assert!((above - at).abs() <= 0.5, "cell {index}");
            assert!(gap(act_color(above), act_color(at)) <= 4, "cell {index}");
        }
        // At the end of the tail the last cell is half a cell short of it,
        // and the first cell has hardly moved.
        let half = (end - ACT_STOPS[0].0) / (2.0 * len as f32);
        near(spectrum::place_to(len - 1, len, end), end - half);
        assert!(spectrum::place_to(0, len, end) < 2.0);
        // A top outside the tail is held to it.
        assert_eq!(
            spectrum::place_to(3, len, 1000.0),
            spectrum::place_to(3, len, end)
        );
        assert_eq!(spectrum::place_to(3, len, 10.0), spectrum::place(3, len));
        assert_eq!(
            spectrum::place_to(3, len, f32::NAN),
            spectrum::place(3, len)
        );
        assert_eq!(spectrum::place_to(0, 0, end), ACT_STOPS[0].0);
    }

    #[test]
    fn a_rate_at_its_ceiling_is_full_scale() {
        let ceiling = 250.0;
        let bands = ACT_STOPS.len() - 1;
        near(spectrum::rate(ceiling, ceiling), spectrum::FULL);
        near(spectrum::rate(ceiling / 32.0, ceiling), ACT_STOPS[0].0);
        for (k, stop) in ACT_STOPS.iter().enumerate().take(bands) {
            // Geometric centre of step k: half a band above its first stop.
            let log = k as f64 - (bands as f64 - 0.5);
            let value = ceiling * 2.0_f64.powf(log);
            near(spectrum::rate(value, ceiling), stop.0 + step() / 2.0);
        }
        // Past the ceiling the position keeps climbing, and it reaches the
        // end of the blackbody tail.
        assert!(spectrum::rate(ceiling * 2.0, ceiling) > spectrum::FULL);
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
        // The values themselves, so moving a stop cannot move an anchor
        // unseen.
        assert_eq!(temp::WARM, hex(0xFFD050));
        assert_eq!(temp::WARN, hex(0xFFB030));
        assert_eq!(temp::CRIT, hex(0xFF2A14));
        assert_eq!(temp::WARN, Palette::Llama.rgb(role::WARN));
        // #125: a low warning does not tint a normal reading. Coolant that
        // warns at 45 is the normal colour at 31 and at 35.
        let coolant = |c| temp::color(c, 45.0, 55.0).expect("thresholds");
        assert_eq!(coolant(31.0), temp::NORMAL);
        assert_eq!(coolant(35.0), temp::NORMAL);
        assert_ne!(coolant(38.0), temp::NORMAL);
        assert_eq!(coolant(41.0), temp::WARM);
        assert_eq!(coolant(45.0), temp::WARN);
        assert!(temp::color(70.0, 90.0, 80.0).is_none());
        assert!(temp::color(f32::NAN, warn, crit).is_none());
        assert!(temp::color(70.0, f32::NAN, crit).is_none());
        assert!(temp::color(70.0, warn, warn).is_none());
    }
}
