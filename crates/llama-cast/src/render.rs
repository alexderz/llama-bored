//! `/dev/vcsa11` bytes to one 1920x1080 RGB24 frame.
//!
//! vcsa is a 4-byte header (rows, columns, cursor x, cursor y; rows and
//! columns are truncated to 8 bits by the kernel) and then one
//! (character, attribute) byte pair per cell. With a 512-glyph font the
//! kernel shifts the attribute left one bit and bit 0 carries glyph bit 8:
//! foreground `(attr >> 1) & 0xf`, background `(attr >> 5) & 7`. Colours are
//! in VGA order (bit 0 blue) and map to the console palette tty11 shows
//! (`llama_core::palette`, cast.toml `palette`, matching `[tty] palette`).
//! The kernel does not expose the loaded palette through vcsa, so the config
//! names it (#26).
//!
//! The console image is centred on the frame when smaller and cropped from
//! the top-left when larger, the same corner a smaller monitor shows.

use llama_core::palette::Palette;
use thiserror::Error;

use crate::font::Psf2;

pub const FRAME_WIDTH: usize = 1920;
pub const FRAME_HEIGHT: usize = 1080;
/// Bytes in one frame.
pub const FRAME_BYTES: usize = FRAME_WIDTH * FRAME_HEIGHT * 3;
/// Largest console (`[tty] size` tops out at 1024x512).
pub const MAX_COLS: usize = 1024;
pub const MAX_ROWS: usize = 512;
/// Largest vcsa read: header plus every cell of the largest console.
pub const MAX_VCSA_BYTES: usize = 4 + 2 * MAX_COLS * MAX_ROWS;

/// The 16 colours of `palette` as RGB24 triples, ANSI order (0 black, 1
/// red, ...).
#[must_use]
pub fn rgb24(palette: Palette) -> [[u8; 3]; 16] {
    palette.slots().map(|c| [c.r, c.g, c.b])
}

/// Why a vcsa dump was refused.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VcsaError {
    #[error("vcsa dump is larger than the largest console")]
    TooLarge,
    #[error("vcsa dump is shorter than its header or has an odd cell area")]
    Short,
    #[error("vcsa size does not match its rows and columns")]
    Geometry,
}

/// A parsed screen: `rows` x `cols` cells of (character, attribute).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Screen<'a> {
    pub rows: usize,
    pub cols: usize,
    cells: &'a [u8],
}

impl<'a> Screen<'a> {
    /// Parse a whole vcsa read. Rows and columns come from the header,
    /// widened past 255 by the cell count when the kernel truncated them.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, VcsaError> {
        if bytes.len() > MAX_VCSA_BYTES {
            return Err(VcsaError::TooLarge);
        }
        if bytes.len() < 4 || !(bytes.len() - 4).is_multiple_of(2) {
            return Err(VcsaError::Short);
        }
        let cells = (bytes.len() - 4) / 2;
        let (rows_lo, cols_lo) = (usize::from(bytes[0]), usize::from(bytes[1]));
        let mut found = None;
        let mut rows = rows_lo;
        while rows <= MAX_ROWS {
            let mut cols = cols_lo;
            while cols <= MAX_COLS {
                if rows > 0 && cols > 0 && rows * cols == cells {
                    found = Some((rows, cols));
                    break;
                }
                cols += 256;
            }
            if found.is_some() {
                break;
            }
            rows += 256;
        }
        let (rows, cols) = found.ok_or(VcsaError::Geometry)?;
        Ok(Self {
            rows,
            cols,
            cells: &bytes[4..],
        })
    }

    /// (character, attribute) at a cell.
    #[must_use]
    pub fn cell(&self, row: usize, col: usize) -> (u8, u8) {
        let at = 2 * (row * self.cols + col);
        (self.cells[at], self.cells[at + 1])
    }
}

/// VGA colour number (bit 0 blue, bit 2 red) to ANSI (bit 0 red).
#[must_use]
pub fn vga_to_ansi(vga: u8) -> u8 {
    (vga & 0b1010) | ((vga & 1) << 2) | ((vga & 4) >> 2)
}

/// Glyph index, foreground and background (ANSI numbers) of one cell.
#[must_use]
pub fn decode_cell(ch: u8, attr: u8, hi512: bool) -> (usize, u8, u8) {
    if hi512 {
        let glyph = usize::from(ch) | (usize::from(attr & 1) << 8);
        (
            glyph,
            vga_to_ansi((attr >> 1) & 0x0f),
            vga_to_ansi((attr >> 5) & 0x07),
        )
    } else {
        (
            usize::from(ch),
            vga_to_ansi(attr & 0x0f),
            vga_to_ansi((attr >> 4) & 0x07),
        )
    }
}

/// Render `screen` with `font` and `palette` into `frame` (`FRAME_BYTES`
/// long, RGB24). Outside the console image is black (slot 0 in both
/// palettes).
pub fn render_into(screen: &Screen<'_>, font: &Psf2, palette: Palette, frame: &mut [u8]) {
    let colours = rgb24(palette);
    frame.fill(0);
    let (gw, gh) = (font.width(), font.height());
    let hi512 = font.count() >= 512;
    let native_w = screen.cols * gw;
    let native_h = screen.rows * gh;
    let off_x = FRAME_WIDTH.saturating_sub(native_w) / 2;
    let off_y = FRAME_HEIGHT.saturating_sub(native_h) / 2;
    for row in 0..screen.rows {
        let y0 = off_y + row * gh;
        if y0 >= FRAME_HEIGHT {
            break;
        }
        for col in 0..screen.cols {
            let x0 = off_x + col * gw;
            if x0 >= FRAME_WIDTH {
                break;
            }
            let (ch, attr) = screen.cell(row, col);
            let (glyph, fg, bg) = decode_cell(ch, attr, hi512);
            let fg = colours[usize::from(fg)];
            let bg = colours[usize::from(bg)];
            for gy in 0..gh {
                let y = y0 + gy;
                if y >= FRAME_HEIGHT {
                    break;
                }
                let line = y * FRAME_WIDTH;
                for gx in 0..gw {
                    let x = x0 + gx;
                    if x >= FRAME_WIDTH {
                        break;
                    }
                    let rgb = if font.pixel(glyph, gx, gy) { fg } else { bg };
                    let at = (line + x) * 3;
                    frame[at..at + 3].copy_from_slice(&rgb);
                }
            }
        }
    }
}

/// Render into a new frame.
#[must_use]
pub fn render(screen: &Screen<'_>, font: &Psf2, palette: Palette) -> Vec<u8> {
    let mut frame = vec![0_u8; FRAME_BYTES];
    render_into(screen, font, palette, &mut frame);
    frame
}
