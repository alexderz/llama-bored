//! Value → position → colour.
//!
//! A value is placed on its entry's range as a percent: 0 % at `range.min`,
//! 100 % at `range.max`, linearly or logarithmically. Palettes are defined
//! on that percent axis. The `act` palette is the LCD's ramp (`act_color`):
//! blue → red over 0–100 %, then blackbody to white at 125 %, so a value
//! past the range max heats up exactly as on the LCD.

use llama_core::color::{Rgb, act_color, hex, ramp};

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
    /// The LCD's `act_color`: 0–100 % blue → red, 100–125 % blackbody.
    Act,
    /// Piecewise-linear stops, positions in percent, strictly ascending.
    Stops(Vec<(f32, Rgb)>),
}

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
            Palette::Act => act_color(pct),
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
