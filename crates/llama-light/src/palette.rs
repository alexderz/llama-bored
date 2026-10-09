//! Value → position → colour.
//!
//! A value is placed on its entry's range as a percent: 0 % at `range.min`,
//! 100 % at `range.max`, linearly or logarithmically. Palettes are defined
//! on that percent axis. The `act` palette is llama-light's own LED ramp:
//! blue → violet → purple → magenta → crimson → pure red over 0–100 %, held
//! at red from 100–125 %. It shares the LCD's `act_color` only up to 100 %
//! in spirit (cold-to-hot, blue to red) — not its RGB values, and not its
//! white-hot blackbody tail above 100 %, which reads as weak white light on
//! LEDs rather than heat (#93). The LCD keeps `act_color` unchanged.

use llama_core::color::{Rgb, hex, ramp};

/// Percent-of-range ceiling. Above 100 only the `act` palette changes.
pub const MAX_PCT: f32 = 125.0;

/// How a value is placed on its range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    /// `(v - min) / (max - min)`.
    Linear,
    /// `ln(v / min) / ln(max / min)`. Needs `min > 0`.
    Log,
}

impl Scale {
    /// Percent of `range` for `value`, clamped to `0..=MAX_PCT`. A value
    /// at or below zero on a log scale is 0 %.
    #[must_use]
    pub fn percent(self, value: f32, range: (f32, f32)) -> f32 {
        let (min, max) = range;
        let fraction = match self {
            Scale::Linear => (value - min) / (max - min),
            Scale::Log => {
                if value <= 0.0 {
                    0.0
                } else {
                    (value / min).ln() / (max / min).ln()
                }
            }
        };
        let pct = fraction * 100.0;
        if pct.is_finite() {
            pct.clamp(0.0, MAX_PCT)
        } else {
            0.0
        }
    }
}

/// A colour ramp on the percent axis.
#[derive(Clone, Debug, PartialEq)]
pub enum Palette {
    /// llama-light's own LED ramp ([`ACT_STOPS`]): blue → violet → purple →
    /// magenta → crimson → red over 0–100 %, held at red through 125 %.
    /// Not the LCD's `act_color`, which stays blackbody on its tail above
    /// 100 % (#93).
    Act,
    /// Piecewise-linear stops, positions in percent, strictly ascending.
    Stops(Vec<(f32, Rgb)>),
}

/// `act`: llama-light's LED ramp, blue to pure red, held at red from 100 to
/// 125 % so an over-range value never fades toward white (#93). Approved
/// and already running on-device as these exact `light.toml` stops.
pub const ACT_STOPS: [(f32, Rgb); 8] = [
    (0.0, hex(0x0010FF)),
    (20.0, hex(0x3800FF)),
    (40.0, hex(0x7A00FF)),
    (60.0, hex(0xC000E0)),
    (75.0, hex(0xF00090)),
    (88.0, hex(0xFF0030)),
    (100.0, hex(0xFF0000)),
    (125.0, hex(0xFF0000)),
];

/// `thermal`: cold blue through green and yellow to red.
pub const THERMAL: [(f32, u32); 5] = [
    (0.0, 0x2040FF),
    (25.0, 0x00B0FF),
    (50.0, 0x30E060),
    (75.0, 0xFFC020),
    (100.0, 0xFF2010),
];

/// `mono`: dim white to full white.
pub const MONO: [(f32, u32); 2] = [(0.0, 0x101010), (100.0, 0xFFFFFF)];

impl Palette {
    /// A named palette: `act`, `thermal` or `mono`.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        let stops = |table: &[(f32, u32)]| {
            Palette::Stops(table.iter().map(|(p, c)| (*p, hex(*c))).collect())
        };
        match name {
            "act" => Some(Palette::Act),
            "thermal" => Some(stops(&THERMAL)),
            "mono" => Some(stops(&MONO)),
            _ => None,
        }
    }

    /// Colour at `pct` (0..=[`MAX_PCT`]).
    #[must_use]
    pub fn color(&self, pct: f32) -> Rgb {
        match self {
            Palette::Act => ramp(&ACT_STOPS, pct),
            Palette::Stops(stops) => ramp(stops, pct),
        }
    }
}

/// Parse `#RRGGBB`.
#[must_use]
pub fn parse_hex(text: &str) -> Option<Rgb> {
    let digits = text.strip_prefix('#')?;
    if digits.len() != 6 || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(digits, 16).ok().map(hex)
}

/// `color` scaled by `percent` (0..=100), rounded per channel.
#[must_use]
pub fn dim(color: Rgb, percent: f32) -> Rgb {
    let factor = (percent / 100.0).clamp(0.0, 1.0);
    let channel = |value: u8| (f32::from(value) * factor).round().clamp(0.0, 255.0) as u8;
    Rgb {
        r: channel(color.r),
        g: channel(color.g),
        b: channel(color.b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_is_blue_at_zero_and_pure_red_at_100_and_125() {
        assert_eq!(Palette::Act.color(0.0), hex(0x0010FF));
        assert_eq!(Palette::Act.color(100.0), hex(0xFF0000));
        assert_eq!(
            Palette::Act.color(125.0),
            hex(0xFF0000),
            "held at red past 100"
        );
    }

    #[test]
    fn act_never_lets_green_outshine_both_red_and_blue() {
        let mut pct = 0.0_f32;
        while pct <= MAX_PCT {
            let c = Palette::Act.color(pct);
            assert!(
                !(c.g > c.r && c.g > c.b),
                "{pct}%: green above both red and blue: {c:?}"
            );
            pct += 0.1;
        }
    }

    #[test]
    fn act_hue_climbs_from_blue_through_violet_purple_magenta_crimson_to_red() {
        // Standard RGB -> hue in degrees, 0..360 (0/360 is red). The ramp
        // never dips through green (60-180), so unwrapping only ever needs
        // to add 360 once, right at the blue -> red join.
        fn hue(c: Rgb) -> f32 {
            let (r, g, b) = (f32::from(c.r), f32::from(c.g), f32::from(c.b));
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;
            if delta == 0.0 {
                return 0.0;
            }
            let h = if max == r {
                60.0 * ((g - b) / delta).rem_euclid(6.0)
            } else if max == g {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            h.rem_euclid(360.0)
        }

        let mut prev = hue(Palette::Act.color(0.0));
        let mut pct = 0.1_f32;
        while pct <= MAX_PCT {
            let h = hue(Palette::Act.color(pct));
            let mut step = h - prev;
            if step < -180.0 {
                // The one place the arc crosses the 360/0 seam.
                step += 360.0;
            }
            assert!(
                step >= -1e-3,
                "{pct}%: hue stepped backward ({prev} -> {h})"
            );
            prev = h;
            pct += 0.1;
        }
    }

    #[test]
    fn the_lcds_act_color_is_unchanged() {
        // llama-light's `act` is its own ramp now (#93); the LCD's ramp
        // must not move. Full stop-by-stop coverage:
        // kraken-lcd/tests/v3b.rs::act_color_hits_every_stop_and_interpolates_between.
        use llama_core::color::act_color;
        assert_eq!(act_color(0.0), hex(0x1428D8));
        assert_eq!(act_color(100.0), hex(0xFF2A14));
        assert_eq!(act_color(125.0), hex(0xFFD050));
    }
}
