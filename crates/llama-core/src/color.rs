//! `act_color`: one colour function for the LCD (ring, bars, readout, tokens
//! chart) and the RGB writer (llama-light). Moved here from kraken-lcd's
//! `render/color.rs` so both writers share one ramp.
//!
//! 0–100 is the token-dial ramp L1..L6 placed at 0/20/40/60/80/100. 100–125
//! keeps heating like a black body: red → orange → yellow → white. Both are
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
    (0.0, rgb(0x4A55C8)),
    (20.0, rgb(0x7550D8)),
    (40.0, rgb(0xA64ACF)),
    (60.0, rgb(0xD044A8)),
    (80.0, rgb(0xF4466A)),
    (100.0, rgb(0xFF3A22)),
];

/// 100–125: blackbody.
pub const BB_STOPS: [(f32, Rgb); 5] = [
    (100.0, rgb(0xFF3A22)),
    (107.0, rgb(0xFF6E1A)),
    (114.0, rgb(0xFFB02A)),
    (120.0, rgb(0xFFEA9A)),
    (125.0, rgb(0xFFFFFF)),
];

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
