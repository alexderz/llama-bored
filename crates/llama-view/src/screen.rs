//! vcsa snapshot: header, attributes, and one character per cell.

/// One of the 16 ANSI console colours. The value is the ANSI index, not the VGA nibble.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7,
    BrightBlack = 8,
    BrightRed = 9,
    BrightGreen = 10,
    BrightYellow = 11,
    BrightBlue = 12,
    BrightMagenta = 13,
    BrightCyan = 14,
    BrightWhite = 15,
}

impl Color {
    /// Foreground SGR parameter (`30–37` or `90–97`).
    pub fn fg_sgr(self) -> u8 {
        let n = self as u8;
        if n < 8 { 30 + n } else { 90 + (n - 8) }
    }

    /// Background SGR parameter (`40–47` or `100–107`).
    pub fn bg_sgr(self) -> u8 {
        let n = self as u8;
        if n < 8 { 40 + n } else { 100 + (n - 8) }
    }

    fn from_ansi(index: u8) -> Self {
        match index {
            0 => Self::Black,
            1 => Self::Red,
            2 => Self::Green,
            3 => Self::Yellow,
            4 => Self::Blue,
            5 => Self::Magenta,
            6 => Self::Cyan,
            7 => Self::White,
            8 => Self::BrightBlack,
            9 => Self::BrightRed,
            10 => Self::BrightGreen,
            11 => Self::BrightYellow,
            12 => Self::BrightBlue,
            13 => Self::BrightMagenta,
            14 => Self::BrightCyan,
            _ => Self::BrightWhite,
        }
    }
}

/// Map one VGA attribute byte to ANSI foreground and background.
///
/// The nibble is VGA order, the same order Linux stores after `color_table`:
/// bit 0 blue, bit 1 green, bit 2 red. Bit 3 is bright foreground. Bits 4–6
/// are the background. Bit 7 is blink and does not change the colour.
pub fn vga_attr(attr: u8) -> (Color, Color) {
    let fg = ansi_from_vga(attr & 0x07, attr & 0x08 != 0);
    let bg = ansi_from_vga((attr >> 4) & 0x07, false);
    (fg, bg)
}

/// Glyphs the tty11 font actually draws. Same set `llama-watch` emits
/// (`llama_watch::tty::term::GLYPHS`), including the lower eighths that
/// `chart_glyphs = "eighths"` draws with llama-hack-12x24.
const GLYPHS: &[char] = &[
    '█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·', '…', '▁', '▂', '▃', '▅', '▆', '▇',
];

/// Printable ASCII and [`GLYPHS`] stay. Every other scalar, including ESC,
/// C1, DEL, and Latin-1 such as `Û`, becomes `?`.
fn screen_char(ch: char) -> char {
    if ('\u{20}'..='\u{7e}').contains(&ch) || GLYPHS.contains(&ch) {
        ch
    } else {
        '?'
    }
}

fn char_at(glyph: char, vcs: Option<&[u8]>, vcsu: Option<&[u8]>, index: usize) -> char {
    if let Some(bytes) = vcsu {
        let start = index.saturating_mul(4);
        if let Some(chunk) = bytes.get(start..start + 4) {
            let cp = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            return char::from_u32(cp).unwrap_or('?');
        }
    }
    if let Some(byte) = vcs.and_then(|bytes| bytes.get(index).copied()) {
        return char::from(byte);
    }
    glyph
}

fn ansi_from_vga(vga: u8, bright: bool) -> Color {
    let b = vga & 1;
    let g = (vga >> 1) & 1;
    let r = (vga >> 2) & 1;
    let ansi = r | (g << 1) | (b << 2);
    let index = if bright { ansi + 8 } else { ansi };
    Color::from_ansi(index)
}

/// One filtered cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cell {
    pub ch: char,
    pub fg: Color,
    pub bg: Color,
}

impl Cell {
    pub const fn new(ch: char, fg: Color, bg: Color) -> Self {
        Self { ch, fg, bg }
    }

    /// Space, white on black. Used when the crop hangs off the screen.
    pub const fn blank() -> Self {
        Self::new(' ', Color::White, Color::Black)
    }
}

/// A decoded console snapshot. Cells are row-major and already filtered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Screen {
    pub cols: u16,
    pub rows: u16,
    pub cursor_x: u16,
    pub cursor_y: u16,
    pub cells: Vec<Cell>,
}

/// `vcsa` bytes could not be decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Fewer than four header bytes, or a short cell array.
    Truncated,
}

/// Decode a `vcsa` image.
///
/// The header is four bytes: rows, columns, cursor column, cursor row.
/// Each cell is a host-endian pair, glyph then attribute. When `vcsu` has a
/// native-endian `u32` for the cell, that code point wins. Otherwise a `vcs`
/// byte wins over the glyph. The chosen character then passes through
/// [`screen_char`].
pub fn decode_screen(
    vcsa: &[u8],
    vcs: Option<&[u8]>,
    vcsu: Option<&[u8]>,
) -> Result<Screen, DecodeError> {
    if vcsa.len() < 4 {
        return Err(DecodeError::Truncated);
    }
    let rows = u16::from(vcsa[0]);
    let cols = u16::from(vcsa[1]);
    let cursor_x = u16::from(vcsa[2]);
    let cursor_y = u16::from(vcsa[3]);
    let n = usize::from(rows).saturating_mul(usize::from(cols));
    let need = 4usize.saturating_add(n.saturating_mul(2));
    if vcsa.len() < need {
        return Err(DecodeError::Truncated);
    }
    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        let glyph = char::from(vcsa[4 + i * 2]);
        let ch = screen_char(char_at(glyph, vcs, vcsu, i));
        let (fg, bg) = vga_attr(vcsa[5 + i * 2]);
        cells.push(Cell::new(ch, fg, bg));
    }
    Ok(Screen {
        cols,
        rows,
        cursor_x,
        cursor_y,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(screen: &Screen) -> Vec<char> {
        screen.cells.iter().map(|cell| cell.ch).collect()
    }

    fn vcsa(rows: u8, cols: u8, x: u8, y: u8, cells: &[(u8, u8)]) -> Vec<u8> {
        let mut bytes = vec![rows, cols, x, y];
        for (ch, attr) in cells {
            bytes.push(*ch);
            bytes.push(*attr);
        }
        bytes
    }

    #[test]
    fn parses_header_and_cells_from_fixture_bytes() {
        // 2×3, cursor at column 1 row 0. Low byte is the glyph, high byte the attribute.
        let bytes = vcsa(
            2,
            3,
            1,
            0,
            &[
                (b'A', 0x07),
                (b'B', 0x07),
                (b'C', 0x07),
                (b'D', 0x07),
                (b'E', 0x07),
                (b'F', 0x07),
            ],
        );
        let screen = decode_screen(&bytes, None, None).expect("fixture decodes");
        assert_eq!(screen.rows, 2);
        assert_eq!(screen.cols, 3);
        assert_eq!(screen.cursor_x, 1);
        assert_eq!(screen.cursor_y, 0);
        assert_eq!(chars(&screen), ['A', 'B', 'C', 'D', 'E', 'F']);
    }

    #[test]
    fn maps_vga_attributes_to_ansi_16() {
        // Linux stores the nibble in VGA bit order (bit0 blue, bit1 green, bit2 red)
        // via `color_table`. Bit 3 is bright foreground. Bit 7 is blink, not a colour.
        let cases = [
            (0x00, Color::Black, Color::Black, 30, 40),
            (0x01, Color::Blue, Color::Black, 34, 40),
            (0x02, Color::Green, Color::Black, 32, 40),
            (0x03, Color::Cyan, Color::Black, 36, 40),
            (0x04, Color::Red, Color::Black, 31, 40),
            (0x05, Color::Magenta, Color::Black, 35, 40),
            (0x06, Color::Yellow, Color::Black, 33, 40),
            (0x07, Color::White, Color::Black, 37, 40),
            (0x08, Color::BrightBlack, Color::Black, 90, 40),
            (0x0C, Color::BrightRed, Color::Black, 91, 40),
            (0x0F, Color::BrightWhite, Color::Black, 97, 40),
            (0x10, Color::Black, Color::Blue, 30, 44),
            (0x40, Color::Black, Color::Red, 30, 41),
            (0x70, Color::Black, Color::White, 30, 47),
            (0x16, Color::Yellow, Color::Blue, 33, 44),
            (0x8F, Color::BrightWhite, Color::Black, 97, 40),
        ];
        for (attr, fg, bg, fg_sgr, bg_sgr) in cases {
            let (got_fg, got_bg) = vga_attr(attr);
            assert_eq!(got_fg, fg, "attr {attr:#04x} foreground");
            assert_eq!(got_bg, bg, "attr {attr:#04x} background");
            assert_eq!(got_fg.fg_sgr(), fg_sgr, "attr {attr:#04x} fg sgr");
            assert_eq!(got_bg.bg_sgr(), bg_sgr, "attr {attr:#04x} bg sgr");
        }
    }

    #[test]
    fn hostile_screen_bytes_become_question_marks() {
        // ESC, C1 CSI (U+009B), DEL. None of these may survive as themselves.
        let bytes = vcsa(1, 3, 0, 0, &[(0x1B, 0x07), (0x9B, 0x07), (0x7F, 0x07)]);
        let screen = decode_screen(&bytes, None, None).expect("hostile fixture decodes");
        assert_eq!(chars(&screen), ['?', '?', '?']);
    }

    #[test]
    fn vcsu_glyphs_pass_and_latin1_controls_do_not() {
        // Glyph bytes are placeholders. Unicode comes from vcsu (native u32).
        // `Û` (U+00DB) is the S12 Latin-1 case: its UTF-8 contains 0x9B, so it
        // must become `?`. Block and middle-dot stay.
        let bytes = vcsa(1, 4, 0, 0, &[(b'x', 0x07); 4]);
        let vcsu = u32s(&[0x2588, 0x00B7, 0x00DB, u32::from(b'A')]);
        let screen = decode_screen(&bytes, None, Some(&vcsu)).expect("vcsu decodes");
        assert_eq!(chars(&screen), ['█', '·', '?', 'A']);
    }

    #[test]
    fn vcsu_lower_eighths_pass_and_other_blocks_do_not() {
        // llama-watch draws U+2581-2587 in chart_glyphs = "eighths" mode.
        let bytes = vcsa(1, 9, 0, 0, &[(b'x', 0x07); 9]);
        let vcsu = u32s(&[
            0x2581, 0x2582, 0x2583, 0x2584, 0x2585, 0x2586, 0x2587, 0x2594, 0x2589,
        ]);
        let screen = decode_screen(&bytes, None, Some(&vcsu)).expect("vcsu decodes");
        assert_eq!(
            chars(&screen),
            ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '?', '?']
        );
    }

    #[test]
    fn vcs_overrides_glyph_when_vcsu_is_absent() {
        let bytes = vcsa(1, 2, 0, 0, &[(b'A', 0x07), (b'B', 0x07)]);
        let vcs = [b'Z', 0x1B];
        let screen = decode_screen(&bytes, Some(&vcs), None).expect("vcs decodes");
        assert_eq!(chars(&screen), ['Z', '?']);
    }

    fn u32s(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_ne_bytes());
        }
        out
    }
}
