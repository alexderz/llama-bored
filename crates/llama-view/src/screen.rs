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

    /// ANSI index, 0..=15: the slot in `llama_core::palette`.
    pub fn index(self) -> u8 {
        self as u8
    }

    /// The colour of ANSI slot `index` (only the low four bits count).
    pub fn from_ansi(index: u8) -> Self {
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

/// [`vga_attr`] for a console with a 512-glyph font, such as tty11 with
/// llama-hack: the kernel shifts the attribute left one bit and bit 0 is
/// glyph bit 8. Foreground `(attr >> 1) & 0xf`, background `(attr >> 5) & 7`,
/// still VGA order. There is no blink bit.
pub fn vga_attr_512(attr: u8) -> (Color, Color) {
    let fg_nibble = (attr >> 1) & 0x0f;
    let fg = ansi_from_vga(fg_nibble & 0x07, fg_nibble & 0x08 != 0);
    let bg = ansi_from_vga((attr >> 5) & 0x07, false);
    (fg, bg)
}

/// How a vcsa attribute byte is laid out. It depends on the console font,
/// which a reader of vcsa cannot ask the kernel for (that is an ioctl), so it
/// is either given (`--font-glyphs`) or inferred with [`detect_layout`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AttrLayout {
    /// A 256-glyph font: [`vga_attr`].
    #[default]
    Glyphs256,
    /// A 512-glyph font: [`vga_attr_512`].
    Glyphs512,
}

/// Infer the attribute layout from blank cells.
///
/// A space is glyph 0x20 in every console font, so with a 512-glyph font its
/// attribute has bit 0 (glyph bit 8) clear. With a 256-glyph font bit 0 is
/// the foreground's blue bit, set for the default white (and for every white,
/// cyan, magenta or blue foreground). So: any space with bit 0 set means
/// 256 glyphs; spaces that all have it clear mean 512. Only cells whose vcsa
/// glyph byte is 0x20 count, and, when `vcsu` is readable, only those whose
/// Unicode is U+0020 too. `None` when the screen has no such cell; the caller
/// keeps its previous answer.
///
/// A 256-glyph console whose every space has a black, green, red or yellow
/// foreground would read as 512; tty11 (llama-watch fills blank cells with
/// a white foreground) never does. `--font-glyphs` overrides the guess.
pub fn detect_layout(vcsa: &[u8], vcsu: Option<&[u8]>) -> Option<AttrLayout> {
    let cells = vcsa.get(4..)?;
    let mut spaces = 0usize;
    for (i, pair) in cells.as_chunks::<2>().0.iter().enumerate() {
        if pair[0] != b' ' {
            continue;
        }
        if let Some(bytes) = vcsu {
            let start = i.saturating_mul(4);
            if let Some(chunk) = bytes.get(start..start + 4) {
                let cp = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if cp != 0x20 {
                    continue;
                }
            }
        }
        if pair[1] & 1 == 1 {
            return Some(AttrLayout::Glyphs256);
        }
        spaces += 1;
    }
    (spaces > 0).then_some(AttrLayout::Glyphs512)
}

/// Glyphs the tty11 font actually draws. Same set `llama-watch` emits
/// (`llama_watch::tty::term::GLYPHS`), including the lower eighths that
/// `chart_glyphs = "eighths"` draws with llama-hack-12x24, the meter bars'
/// `▇` included (#52).
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

/// Largest console (`[tty] size` tops out at 1024x512).
pub const MAX_COLS: u16 = 1024;
/// Largest console rows.
pub const MAX_ROWS: u16 = 512;
/// Largest vcsa image: header plus every cell of the largest console.
pub const MAX_VCSA_BYTES: usize = 4 + 2 * MAX_COLS as usize * MAX_ROWS as usize;

/// Rows and columns of a whole vcsa read. The kernel truncates both header
/// bytes to 8 bits, so a console wider than 255 columns (tty11 can be up to
/// 1024x512) reads as `cols % 256`; the cell count gives the real size back,
/// the same way llama-cast does. A read longer than the header's size with no
/// wider match keeps the header's size (trailing bytes ignored).
pub fn vcsa_geometry(vcsa: &[u8]) -> Option<(u16, u16)> {
    let header = vcsa.get(..4)?;
    let (rows_lo, cols_lo) = (u16::from(header[0]), u16::from(header[1]));
    let body = vcsa.len() - 4;
    let cells = body / 2;
    if body.is_multiple_of(2) {
        let mut rows = rows_lo;
        while rows <= MAX_ROWS {
            let mut cols = cols_lo;
            while cols <= MAX_COLS {
                if usize::from(rows) * usize::from(cols) == cells {
                    return Some((rows, cols));
                }
                cols += 256;
            }
            rows += 256;
        }
    }
    (usize::from(rows_lo) * usize::from(cols_lo) <= cells).then_some((rows_lo, cols_lo))
}

/// Decode a `vcsa` image, inferring the attribute layout with
/// [`detect_layout`] (256 glyphs when it cannot tell).
pub fn decode_screen(
    vcsa: &[u8],
    vcs: Option<&[u8]>,
    vcsu: Option<&[u8]>,
) -> Result<Screen, DecodeError> {
    let layout = detect_layout(vcsa, vcsu).unwrap_or_default();
    decode_screen_with(vcsa, vcs, vcsu, layout)
}

/// Decode a `vcsa` image whose attributes have `layout`.
///
/// The header is four bytes: rows, columns, cursor column, cursor row.
/// Each cell is a host-endian pair, glyph then attribute. When `vcsu` has a
/// native-endian `u32` for the cell, that code point wins. Otherwise a `vcs`
/// byte wins over the glyph. The chosen character then passes through
/// [`screen_char`].
pub fn decode_screen_with(
    vcsa: &[u8],
    vcs: Option<&[u8]>,
    vcsu: Option<&[u8]>,
    layout: AttrLayout,
) -> Result<Screen, DecodeError> {
    let (rows, cols) = vcsa_geometry(vcsa).ok_or(DecodeError::Truncated)?;
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
        let attr = vcsa[5 + i * 2];
        let (fg, bg) = match layout {
            AttrLayout::Glyphs256 => vga_attr(attr),
            AttrLayout::Glyphs512 => vga_attr_512(attr),
        };
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
    fn maps_512_glyph_attributes_like_llama_cast() {
        // fg bits 1-4, bg bits 5-7, bit 0 is glyph bit 8 and changes nothing.
        for vga_fg in 0..16u8 {
            for vga_bg in 0..8u8 {
                let attr = (vga_bg << 5) | (vga_fg << 1);
                let want = vga_attr((vga_bg << 4) | vga_fg);
                assert_eq!(vga_attr_512(attr), want, "fg {vga_fg} bg {vga_bg}");
                assert_eq!(vga_attr_512(attr | 1), want);
            }
        }
        // tty11's default: white on black is 0x0e with a 512-glyph font.
        assert_eq!(vga_attr_512(0x0e), (Color::White, Color::Black));
        // VGA bright red (12) on VGA blue (1).
        assert_eq!(
            vga_attr_512((1 << 5) | (12 << 1) | 1),
            (Color::BrightRed, Color::Blue)
        );
    }

    #[test]
    fn detects_the_attribute_layout_from_spaces() {
        let white_256 = vcsa(1, 3, 0, 0, &[(b' ', 0x07), (b'A', 0x02), (b' ', 0x07)]);
        assert_eq!(detect_layout(&white_256, None), Some(AttrLayout::Glyphs256));
        let white_512 = vcsa(1, 3, 0, 0, &[(b' ', 0x0e), (b'A', 0x1e), (b' ', 0x0e)]);
        assert_eq!(detect_layout(&white_512, None), Some(AttrLayout::Glyphs512));
        let no_space = vcsa(1, 2, 0, 0, &[(b'A', 0x07), (b'B', 0x07)]);
        assert_eq!(detect_layout(&no_space, None), None);
        // A glyph-0x120 cell (low byte 0x20) that vcsu says is not a space
        // does not count.
        let hi = vcsa(1, 2, 0, 0, &[(b' ', 0x0f), (b' ', 0x0e)]);
        let vcsu = u32s(&[0x2588, 0x20]);
        assert_eq!(detect_layout(&hi, Some(&vcsu)), Some(AttrLayout::Glyphs512));
        assert_eq!(detect_layout(&hi, None), Some(AttrLayout::Glyphs256));
        // decode_screen uses the detected layout.
        let screen = decode_screen(&white_512, None, None).expect("decode");
        assert!(screen.cells.iter().all(|c| c.bg == Color::Black));
        assert_eq!(screen.cells[0].fg, Color::White);
        assert_eq!(screen.cells[1].fg, Color::BrightWhite);
    }

    #[test]
    fn wide_consoles_get_their_real_size_back() {
        // 49 x 320 reads as 49 x 64 in the header.
        let mut wide = vec![49u8, 64, 0, 0];
        wide.resize(4 + 2 * 49 * 320, 0);
        assert_eq!(vcsa_geometry(&wide), Some((49, 320)));
        let screen = decode_screen(&wide, None, None).expect("wide");
        assert_eq!((screen.rows, screen.cols), (49, 320));
        assert_eq!(screen.cells.len(), 49 * 320);
        // An exact small read and a short one.
        let small = vcsa(2, 3, 0, 0, &[(b'a', 7); 6]);
        assert_eq!(vcsa_geometry(&small), Some((2, 3)));
        assert_eq!(vcsa_geometry(&small[..small.len() - 2]), None);
        assert_eq!(vcsa_geometry(&small[..3]), None);
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
