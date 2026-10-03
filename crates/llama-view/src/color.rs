//! How a console colour becomes SGR (#25).
//!
//! The 16 basic SGR colours (30–37, 90–97, 40–47) go through the viewing
//! terminal's theme, so over SSH the dashboard can look nothing like tty11.
//! `truecolor` and `256` send the RGB tty11 actually shows instead, from the
//! same table llama-watch loads (`llama_core::palette`, `--palette`).

use llama_core::color::Rgb;
use llama_core::palette::Palette;

use crate::screen::Color;

/// `--colors` / `LLAMA_VIEW_COLORS`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ColorChoice {
    /// [`ColorMode::Truecolor`] when `COLORTERM` is `truecolor` or `24bit`,
    /// else [`ColorMode::Xterm256`].
    #[default]
    Auto,
    Truecolor,
    Xterm256,
    Ansi16,
}

impl ColorChoice {
    /// `auto`, `truecolor`, `256` or `16`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(Self::Auto),
            "truecolor" => Some(Self::Truecolor),
            "256" => Some(Self::Xterm256),
            "16" => Some(Self::Ansi16),
            _ => None,
        }
    }

    /// The mode to paint with. `colorterm` is `$COLORTERM`.
    ///
    /// `TERM` is not consulted: SSH forwards `TERM` but not `COLORTERM`, and
    /// `xterm-256color`, `tmux-256color` and `screen-256color` all take 256
    /// colours, which no theme remaps in practice.
    pub fn resolve(self, colorterm: Option<&str>) -> ColorMode {
        match self {
            Self::Auto => {
                let truecolor = colorterm.is_some_and(|value| {
                    value.eq_ignore_ascii_case("truecolor") || value.eq_ignore_ascii_case("24bit")
                });
                if truecolor {
                    ColorMode::Truecolor
                } else {
                    ColorMode::Xterm256
                }
            }
            Self::Truecolor => ColorMode::Truecolor,
            Self::Xterm256 => ColorMode::Xterm256,
            Self::Ansi16 => ColorMode::Ansi16,
        }
    }
}

/// What a colour index becomes on the wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorMode {
    /// `38;2;r;g;b` / `48;2;r;g;b` with the palette's RGB.
    Truecolor,
    /// `38;5;n` / `48;5;n`, `n` the nearest xterm colour in 16..=255.
    Xterm256,
    /// `30–37`, `90–97`, `40–47`, `100–107`: the terminal's own 16 colours.
    Ansi16,
}

impl ColorMode {
    /// The `--colors` spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::Truecolor => "truecolor",
            Self::Xterm256 => "256",
            Self::Ansi16 => "16",
        }
    }
}

/// The six levels of each axis of the xterm 6x6x6 cube (indices 16..=231).
const CUBE: [u8; 6] = [0, 95, 135, 175, 215, 255];

/// RGB of xterm-256 colour `index`, for 16..=255. 0..=15 are themed, so they
/// are never chosen; they map to black here.
pub fn xterm256_rgb(index: u8) -> Rgb {
    match index {
        16..=231 => {
            let n = index - 16;
            Rgb {
                r: CUBE[usize::from(n / 36)],
                g: CUBE[usize::from((n / 6) % 6)],
                b: CUBE[usize::from(n % 6)],
            }
        }
        232..=255 => {
            let v = 8 + 10 * (index - 232);
            Rgb { r: v, g: v, b: v }
        }
        _ => Rgb::default(),
    }
}

/// The xterm-256 colour in 16..=255 nearest to `rgb` by squared RGB
/// distance. Ties go to the lower index.
pub fn nearest_xterm256(rgb: Rgb) -> u8 {
    let dist = |c: Rgb| {
        let d = |a: u8, b: u8| {
            let x = i32::from(a) - i32::from(b);
            x * x
        };
        d(c.r, rgb.r) + d(c.g, rgb.g) + d(c.b, rgb.b)
    };
    let mut best = 16u8;
    let mut best_d = i32::MAX;
    for index in 16..=255u8 {
        let d = dist(xterm256_rgb(index));
        if d < best_d {
            best = index;
            best_d = d;
        }
    }
    best
}

/// The SGR sequence for each foreground and background colour, built once.
#[derive(Clone, Debug)]
pub struct SgrTable {
    fg: Vec<Vec<u8>>,
    bg: Vec<Vec<u8>>,
}

impl SgrTable {
    pub fn new(mode: ColorMode, palette: Palette) -> Self {
        let mut fg = Vec::with_capacity(16);
        let mut bg = Vec::with_capacity(16);
        for index in 0..16u8 {
            let color = Color::from_ansi(index);
            let rgb = palette.rgb(index);
            let (f, b) = match mode {
                ColorMode::Ansi16 => (color.fg_sgr().to_string(), color.bg_sgr().to_string()),
                ColorMode::Truecolor => (
                    format!("38;2;{};{};{}", rgb.r, rgb.g, rgb.b),
                    format!("48;2;{};{};{}", rgb.r, rgb.g, rgb.b),
                ),
                ColorMode::Xterm256 => {
                    let n = nearest_xterm256(rgb);
                    (format!("38;5;{n}"), format!("48;5;{n}"))
                }
            };
            fg.push(format!("\x1b[{f}m").into_bytes());
            bg.push(format!("\x1b[{b}m").into_bytes());
        }
        Self { fg, bg }
    }

    /// Foreground sequence of `color`.
    pub fn fg(&self, color: Color) -> &[u8] {
        &self.fg[usize::from(color.index())]
    }

    /// Background sequence of `color`.
    pub fn bg(&self, color: Color) -> &[u8] {
        &self.bg[usize::from(color.index())]
    }
}

impl Default for SgrTable {
    fn default() -> Self {
        Self::new(ColorMode::Ansi16, Palette::Llama)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_core::palette::{LLAMA, VGA};

    fn all() -> impl Iterator<Item = Color> {
        (0..16u8).map(Color::from_ansi)
    }

    #[test]
    fn sixteen_is_todays_output_for_every_colour() {
        let table = SgrTable::new(ColorMode::Ansi16, Palette::Llama);
        let want_fg = [
            30, 31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96, 97,
        ];
        let want_bg = [
            40, 41, 42, 43, 44, 45, 46, 47, 100, 101, 102, 103, 104, 105, 106, 107,
        ];
        for (i, color) in all().enumerate() {
            assert_eq!(table.fg(color), format!("\x1b[{}m", want_fg[i]).as_bytes());
            assert_eq!(table.bg(color), format!("\x1b[{}m", want_bg[i]).as_bytes());
        }
        // The palette does not change 16-colour output.
        let vga = SgrTable::new(ColorMode::Ansi16, Palette::Vga);
        for color in all() {
            assert_eq!(table.fg(color), vga.fg(color));
        }
    }

    #[test]
    fn truecolor_sends_the_palette_rgb_for_every_colour() {
        for (palette, slots) in [(Palette::Llama, LLAMA), (Palette::Vga, VGA)] {
            let table = SgrTable::new(ColorMode::Truecolor, palette);
            for (i, color) in all().enumerate() {
                let c = slots[i];
                assert_eq!(
                    table.fg(color),
                    format!("\x1b[38;2;{};{};{}m", c.r, c.g, c.b).as_bytes()
                );
                assert_eq!(
                    table.bg(color),
                    format!("\x1b[48;2;{};{};{}m", c.r, c.g, c.b).as_bytes()
                );
            }
        }
        let table = SgrTable::new(ColorMode::Truecolor, Palette::Llama);
        assert_eq!(table.fg(Color::Blue), b"\x1b[38;2;20;40;216m");
        assert_eq!(table.bg(Color::Black), b"\x1b[48;2;0;0;0m");
    }

    #[test]
    fn xterm256_table_is_the_cube_and_the_grey_ramp() {
        assert_eq!(xterm256_rgb(16), Rgb { r: 0, g: 0, b: 0 });
        assert_eq!(xterm256_rgb(21), Rgb { r: 0, g: 0, b: 255 });
        assert_eq!(xterm256_rgb(196), Rgb { r: 255, g: 0, b: 0 });
        assert_eq!(
            xterm256_rgb(231),
            Rgb {
                r: 255,
                g: 255,
                b: 255
            }
        );
        assert_eq!(xterm256_rgb(232), Rgb { r: 8, g: 8, b: 8 });
        assert_eq!(
            xterm256_rgb(255),
            Rgb {
                r: 238,
                g: 238,
                b: 238
            }
        );
        assert_eq!(
            xterm256_rgb(244),
            Rgb {
                r: 128,
                g: 128,
                b: 128
            }
        );
    }

    #[test]
    fn nearest_256_is_the_true_minimum_and_never_a_themed_index() {
        let brute = |c: Rgb| {
            (16..=255u8)
                .map(|i| {
                    let x = xterm256_rgb(i);
                    let d = (i32::from(x.r) - i32::from(c.r)).pow(2)
                        + (i32::from(x.g) - i32::from(c.g)).pow(2)
                        + (i32::from(x.b) - i32::from(c.b)).pow(2);
                    (d, i)
                })
                .min()
                .unwrap()
                .1
        };
        for c in LLAMA.iter().chain(VGA.iter()) {
            let n = nearest_xterm256(*c);
            assert!(n >= 16, "{c:?} -> {n}");
            assert_eq!(n, brute(*c), "{c:?}");
        }
        // Exact cube and ramp members map to themselves.
        assert_eq!(nearest_xterm256(Rgb { r: 0, g: 0, b: 0 }), 16);
        assert_eq!(
            nearest_xterm256(Rgb {
                r: 255,
                g: 255,
                b: 255
            }),
            231
        );
        assert_eq!(
            nearest_xterm256(Rgb {
                r: 128,
                g: 128,
                b: 128
            }),
            244
        );
    }

    #[test]
    fn xterm256_sgr_for_every_colour_of_both_palettes() {
        // Pinned so a change to the palette or the search is deliberate.
        let llama = [
            16, 196, 77, 215, 20, 92, 208, 251, 242, 197, 120, 221, 57, 128, 81, 231,
        ];
        let vga = [
            16, 124, 34, 130, 19, 127, 37, 248, 240, 203, 83, 227, 63, 207, 87, 231,
        ];
        for (palette, want) in [(Palette::Llama, llama), (Palette::Vga, vga)] {
            let table = SgrTable::new(ColorMode::Xterm256, palette);
            for (i, color) in all().enumerate() {
                assert_eq!(
                    table.fg(color),
                    format!("\x1b[38;5;{}m", want[i]).as_bytes(),
                    "{palette:?} fg {i}"
                );
                assert_eq!(
                    table.bg(color),
                    format!("\x1b[48;5;{}m", want[i]).as_bytes(),
                    "{palette:?} bg {i}"
                );
            }
        }
    }

    #[test]
    fn auto_selection_matrix() {
        // TERM is not an input: these are the TERM values the matrix covers,
        // each with COLORTERM unset, empty, truecolor, 24bit and another.
        for _term in [
            "xterm-256color",
            "tmux-256color",
            "screen-256color",
            "linux",
        ] {
            for (colorterm, want) in [
                (None, ColorMode::Xterm256),
                (Some(""), ColorMode::Xterm256),
                (Some("truecolor"), ColorMode::Truecolor),
                (Some("24bit"), ColorMode::Truecolor),
                (Some("TrueColor"), ColorMode::Truecolor),
                (Some("yes"), ColorMode::Xterm256),
            ] {
                assert_eq!(ColorChoice::Auto.resolve(colorterm), want, "{colorterm:?}");
            }
        }
        for (choice, want) in [
            (ColorChoice::Truecolor, ColorMode::Truecolor),
            (ColorChoice::Xterm256, ColorMode::Xterm256),
            (ColorChoice::Ansi16, ColorMode::Ansi16),
        ] {
            assert_eq!(choice.resolve(None), want);
            assert_eq!(choice.resolve(Some("truecolor")), want);
        }
    }

    #[test]
    fn choice_spellings() {
        assert_eq!(ColorChoice::parse("auto"), Some(ColorChoice::Auto));
        assert_eq!(
            ColorChoice::parse("truecolor"),
            Some(ColorChoice::Truecolor)
        );
        assert_eq!(ColorChoice::parse("256"), Some(ColorChoice::Xterm256));
        assert_eq!(ColorChoice::parse("16"), Some(ColorChoice::Ansi16));
        for bad in ["", "24bit", "TRUECOLOR", "8", "256color"] {
            assert_eq!(ColorChoice::parse(bad), None, "{bad}");
        }
    }
}
