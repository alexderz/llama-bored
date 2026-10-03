//! The console palette: the RGB value of each of the 16 colour slots tty11
//! draws with (#26). One table, three users:
//!
//! - llama-watch loads it into tty11 with `ESC ] P n rrggbb` (and
//!   `ESC ] R` puts the kernel's VGA set back),
//! - llama-view paints the same RGB values as truecolor or the nearest
//!   xterm-256 colour, so an SSH viewer matches tty11 (#25),
//! - llama-cast renders its frames with it, so the TV matches too.
//!
//! Slots are in ANSI order (0 black, 1 red, ... 7 white, 8..=15 bright),
//! the order of SGR 30–37 / 90–97 and of the Linux `ESC ] P n` digit.
//!
//! # Role → slot map ([`Palette::Llama`])
//!
//! The tty layout keeps naming the 16 colours as before (`C16` in
//! llama-watch); this table only changes what each slot looks like. The heat
//! colours are the shared activity ramp ([`crate::color::ACT_STOPS`],
//! L1..L6) and its blackbody tail ([`crate::color::BB_STOPS`]).
//!
//! | slot | ANSI name      | RGB       | from          | tty roles |
//! |------|----------------|-----------|---------------|-----------|
//! | 0    | black          | `#000000` |               | background; text on a warning band |
//! | 1    | red            | `#FF2A14` | L6 (100)      | errors: stalled fan, capacity over 100 % (lower half) |
//! | 2    | green          | `#30D158` | semantic      | ok: `READY`, health `ok`/`idle`, capacity 15–40 % (lower) |
//! | 3    | yellow         | `#FFB030` | blackbody 120 | warnings: `AI DOWN`, stuck model, `ERR` badge, alarm bands (background), error count, capacity 40–70 % (lower) |
//! | 4    | blue           | `#1428D8` | L1 (0)        | spectrum step 1, capacity under 15 % (lower) |
//! | 5    | magenta        | `#7A1FE8` | L3 (40)       | spectrum step 3 |
//! | 6    | cyan           | `#FF8A1E` | blackbody 114 | spare heat colour (the layout draws nothing in slot 6 today) |
//! | 7    | white          | `#C8C8C8` | grey          | text and labels (default foreground), capacity 70–100 % (lower) |
//! | 8    | bright black   | `#686868` | grey          | dim text, rules, axis labels, empty bar track |
//! | 9    | bright red     | `#F21E5A` | L5 (80)       | spectrum step 5, live: `GENERATING`, `gen`, `>`, output cursor; reset mark `v`; capacity over 100 % (upper) |
//! | 10   | bright green   | `#7CF08C` | semantic      | capacity 15–40 % (upper), reset mark `c` |
//! | 11   | bright yellow  | `#FFD050` | blackbody 125 | capacity 40–70 % (upper), reset mark `e` |
//! | 12   | bright blue    | `#3A1EF0` | L2 (20)       | spectrum step 2, capacity under 15 % (upper) |
//! | 13   | bright magenta | `#C21CC8` | L4 (60)       | spectrum step 4 |
//! | 14   | bright cyan    | `#50C8F0` | semantic      | reset mark `n` (a cool colour, apart from the heat ones) |
//! | 15   | bright white   | `#FFFFFF` | white         | bright values, white-hot top of an over-scale spectrum cell |
//!
//! The background SGR on the console is 40–47 only, so slots 8–15 are never a
//! background; 0 (black) and 3 (warning bands) are the backgrounds the layout
//! uses.
//!
//! [`Palette::Vga`] is the kernel's default set (`default_red`,
//! `default_grn`, `default_blu` in `drivers/tty/vt/vt.c`).

use serde::Deserialize;

use crate::color::{Rgb, hex};

/// Which 16 colours tty11 shows: `[tty] palette` in watch.toml, `palette`
/// in cast.toml, `--palette` in llama-view.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Palette {
    /// llama-bored's heat colours ([`LLAMA`]). The default.
    #[default]
    Llama,
    /// The kernel's VGA colours ([`VGA`]); llama-watch writes no palette.
    Vga,
}

impl Palette {
    /// The 16 slots, ANSI order.
    #[must_use]
    pub const fn slots(self) -> &'static [Rgb; 16] {
        match self {
            Self::Llama => &LLAMA,
            Self::Vga => &VGA,
        }
    }

    /// The config spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Vga => "vga",
        }
    }

    /// `"llama"` or `"vga"`; anything else is `None`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "llama" => Some(Self::Llama),
            "vga" => Some(Self::Vga),
            _ => None,
        }
    }

    /// RGB of ANSI slot `index` (only the low four bits count).
    #[must_use]
    pub const fn rgb(self, index: u8) -> Rgb {
        self.slots()[(index & 0x0f) as usize]
    }
}

/// The slot each tty role uses. See the module table.
pub mod role {
    /// Background.
    pub const BACKGROUND: u8 = 0;
    /// Errors: stalled fan, over capacity.
    pub const ERROR: u8 = 1;
    /// Ok: `READY`, healthy services.
    pub const OK: u8 = 2;
    /// Warnings: `AI DOWN`, `ERR`, alarm bands.
    pub const WARN: u8 = 3;
    /// Text and labels.
    pub const TEXT: u8 = 7;
    /// Dim text, rules, empty bar track.
    pub const DIM: u8 = 8;
    /// Live generation: `GENERATING`, `gen`, the output cursor.
    pub const LIVE: u8 = 9;
    /// Bright values and the white-hot top of an over-scale cell.
    pub const BRIGHT: u8 = 15;
    /// The five spectrum steps of meters, digits and the chart, low to
    /// high: L1, L2, L3, L4, L5 of the activity ramp.
    pub const SPECTRUM: [u8; 5] = [4, 12, 5, 13, 9];
}

/// llama-bored's console palette, ANSI order. See the module table.
pub const LLAMA: [Rgb; 16] = [
    hex(0x000000),
    hex(0xFF2A14),
    hex(0x30D158),
    hex(0xFFB030),
    hex(0x1428D8),
    hex(0x7A1FE8),
    hex(0xFF8A1E),
    hex(0xC8C8C8),
    hex(0x686868),
    hex(0xF21E5A),
    hex(0x7CF08C),
    hex(0xFFD050),
    hex(0x3A1EF0),
    hex(0xC21CC8),
    hex(0x50C8F0),
    hex(0xFFFFFF),
];

/// The Linux console's default palette, ANSI order.
pub const VGA: [Rgb; 16] = [
    hex(0x000000),
    hex(0xAA0000),
    hex(0x00AA00),
    hex(0xAA5500),
    hex(0x0000AA),
    hex(0xAA00AA),
    hex(0x00AAAA),
    hex(0xAAAAAA),
    hex(0x555555),
    hex(0xFF5555),
    hex(0x55FF55),
    hex(0xFFFF55),
    hex(0x5555FF),
    hex(0xFF55FF),
    hex(0x55FFFF),
    hex(0xFFFFFF),
];

/// The Linux console sequence that loads `slots`: `ESC ] P n rrggbb` per
/// slot, `n` one lowercase hex digit, `rrggbb` six. No terminator: the
/// console takes exactly seven hex digits after `P`.
#[must_use]
pub fn console_load(slots: &[Rgb; 16]) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(16 * 10);
    for (n, rgb) in slots.iter().enumerate() {
        out.extend_from_slice(b"\x1b]P");
        out.push(HEX[n]);
        for byte in [rgb.r, rgb.g, rgb.b] {
            out.push(HEX[usize::from(byte >> 4)]);
            out.push(HEX[usize::from(byte & 0x0f)]);
        }
    }
    out
}

/// `ESC ] R`: the console resets its palette to the kernel default.
pub const CONSOLE_RESET: &[u8] = b"\x1b]R";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ACT_STOPS, BB_STOPS};

    fn luma(c: Rgb) -> f64 {
        let lin = |v: u8| {
            let s = f64::from(v) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b)
    }

    fn contrast_on_black(c: Rgb) -> f64 {
        (luma(c) + 0.05) / 0.05
    }

    #[test]
    fn spectrum_slots_are_the_activity_ramp_l1_to_l5() {
        for (step, slot) in role::SPECTRUM.iter().enumerate() {
            assert_eq!(LLAMA[usize::from(*slot)], ACT_STOPS[step].1, "step {step}");
        }
        assert_eq!(LLAMA[usize::from(role::ERROR)], ACT_STOPS[5].1, "L6 red");
    }

    #[test]
    fn warning_and_spare_slots_are_blackbody_stops() {
        let bb: Vec<Rgb> = BB_STOPS.iter().map(|stop| stop.1).collect();
        for slot in [3usize, 6, 11] {
            assert!(bb.contains(&LLAMA[slot]), "slot {slot} {:?}", LLAMA[slot]);
        }
    }

    #[test]
    fn semantic_roles_read_as_their_meaning() {
        let ok = LLAMA[usize::from(role::OK)];
        assert!(ok.g > ok.r && ok.g > ok.b, "ok is green: {ok:?}");
        let err = LLAMA[usize::from(role::ERROR)];
        assert!(
            err.r == 0xFF && err.g < 0x40 && err.b < 0x40,
            "error is red: {err:?}"
        );
        assert_eq!(LLAMA[usize::from(role::BACKGROUND)], hex(0));
        for slot in [role::TEXT, role::DIM, role::BRIGHT] {
            let c = LLAMA[usize::from(slot)];
            assert!(c.r == c.g && c.g == c.b, "slot {slot} is a neutral grey");
        }
        // Text is readable on black; dim text still is (WCAG 3:1 for large
        // console glyphs), and the bright tier is above the plain one.
        assert!(contrast_on_black(LLAMA[usize::from(role::TEXT)]) > 10.0);
        assert!(contrast_on_black(LLAMA[usize::from(role::DIM)]) > 3.0);
        assert!(
            luma(LLAMA[usize::from(role::BRIGHT)]) > luma(LLAMA[usize::from(role::TEXT)])
                && luma(LLAMA[usize::from(role::TEXT)]) > luma(LLAMA[usize::from(role::DIM)])
        );
        // Black text on the warning band.
        assert!(contrast_on_black(LLAMA[usize::from(role::WARN)]) > 7.0);
    }

    #[test]
    fn every_slot_is_distinct() {
        for (i, a) in LLAMA.iter().enumerate() {
            for b in &LLAMA[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn console_load_bytes() {
        let bytes = console_load(&LLAMA);
        assert_eq!(bytes.len(), 16 * 10);
        assert!(bytes.starts_with(b"\x1b]P0000000\x1b]P1ff2a14\x1b]P230d158"));
        assert!(bytes.ends_with(b"\x1b]Pdc21cc8\x1b]Pe50c8f0\x1b]Pfffffff"));
        let vga = console_load(&VGA);
        assert!(vga.starts_with(b"\x1b]P0000000\x1b]P1aa0000"));
        assert_eq!(CONSOLE_RESET, b"\x1b]R");
    }

    #[test]
    fn palette_names_and_lookup() {
        assert_eq!(Palette::default(), Palette::Llama);
        assert_eq!(Palette::parse("llama"), Some(Palette::Llama));
        assert_eq!(Palette::parse("vga"), Some(Palette::Vga));
        assert_eq!(Palette::parse("VGA"), None);
        assert_eq!(Palette::Vga.label(), "vga");
        assert_eq!(Palette::Llama.rgb(9), hex(0xF21E5A));
        assert_eq!(Palette::Vga.rgb(3), hex(0xAA5500));
        #[derive(Deserialize)]
        struct T {
            p: Palette,
        }
        let t: T = toml::from_str("p = \"vga\"").expect("vga");
        assert_eq!(t.p, Palette::Vga);
        assert!(toml::from_str::<T>("p = \"xterm\"").is_err());
    }
}
