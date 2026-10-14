//! A grid file (`llama_core::gridfile`) painted into one 1920x1080 RGB24
//! frame. The bytes are walked in place: the frame path does not build a
//! `GridImage`. A missing, unreadable, refused or stale file does not
//! replace the last good frame; it draws a one-line notice. The clock is
//! the caller's: nothing here reads `SystemTime`.

use std::io;

use llama_core::glyph::is_showable;
use llama_core::gridfile::{
    self, CELL_LEN, GridError, HEADER_LEN, MAGIC, MAX_COLS, MAX_ROWS, STALE_AFTER,
};

use crate::config::Font;
use crate::font::Psf2;
use crate::render::{FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH};

/// Grey on black for the notice. The grid's own colours are not used, so a
/// stale frame stays readable and the line does not look like dashboard text.
pub const NOTICE_GREY: [u8; 3] = [170, 170, 170];

/// Why the picture is not a fresh grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Notice {
    Missing,
    Unreadable,
    Refused,
    /// Whole seconds since the file's `unix_ms`, already over [`STALE_AFTER`].
    Stale(u64),
}

/// What a read of the grid file produced. `Data` is the capped bytes.
#[derive(Clone, Copy, Debug)]
pub enum GridRead<'a> {
    Missing,
    Unreadable,
    Data(&'a [u8]),
}

/// The last good frame, kept across calls. Allocated once.
pub struct Hold {
    pixels: Vec<u8>,
    cols: u16,
    rows: u16,
    font_index: usize,
    good: bool,
}

impl Hold {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pixels: vec![0; FRAME_BYTES],
            cols: 0,
            rows: 0,
            font_index: 0,
            good: false,
        }
    }
}

impl Default for Hold {
    fn default() -> Self {
        Self::new()
    }
}

/// A decoded header and the cell bytes that follow it.
pub struct GridView<'a> {
    pub cols: u16,
    pub rows: u16,
    pub unix_ms: u64,
    cells: &'a [u8],
}

impl<'a> GridView<'a> {
    fn cell(&self, index: usize) -> (u32, [u8; 3], [u8; 3]) {
        let at = index * CELL_LEN;
        let raw = &self.cells[at..at + CELL_LEN];
        let scalar = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let scalar = char::from_u32(scalar)
            .filter(|&ch| is_showable(ch))
            .map_or(u32::from(gridfile::REPLACEMENT), u32::from);
        (scalar, [raw[4], raw[5], raw[6]], [raw[7], raw[8], raw[9]])
    }
}

/// Header and length checks, the same ones as `gridfile::decode`, without
/// allocating a cell per glyph.
pub fn open_bytes(bytes: &[u8]) -> Result<GridView<'_>, GridError> {
    let header = bytes.get(..HEADER_LEN).ok_or(GridError::Truncated)?;
    if &header[..8] != MAGIC.as_slice() {
        return Err(GridError::BadMagic);
    }
    let cols = u16::from_le_bytes([header[8], header[9]]);
    let rows = u16::from_le_bytes([header[10], header[11]]);
    if !(1..=MAX_COLS).contains(&cols) || !(1..=MAX_ROWS).contains(&rows) {
        return Err(GridError::BadSize { cols, rows });
    }
    let unix_ms = u64::from_le_bytes([
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ]);
    let want = gridfile::file_len(cols, rows);
    if bytes.len() != want {
        return Err(GridError::BadLength {
            want,
            got: bytes.len(),
        });
    }
    Ok(GridView {
        cols,
        rows,
        unix_ms,
        cells: &bytes[HEADER_LEN..],
    })
}

/// Whole seconds since `written_ms` when that age is over [`STALE_AFTER`],
/// else `None`. A stamp in the future is not stale. Exactly five seconds is
/// not stale.
#[must_use]
pub fn stale_secs(written_ms: u64, now_ms: u64) -> Option<u64> {
    let limit = u64::try_from(STALE_AFTER.as_millis()).unwrap_or(u64::MAX);
    let age = now_ms.saturating_sub(written_ms);
    (age > limit).then_some(age / 1000)
}

/// Largest shipped font, by cell area, whose `cols` by `rows` cells fit in
/// the frame. When none fits, the smallest, which the painter then crops
/// from the top-left. An equal area keeps the earlier font.
#[must_use]
pub fn pick_font(fonts: &[Font], cols: u16, rows: u16) -> Font {
    let index = pick_by(fonts.len(), |i| {
        let (width, height) = fonts[i].cell_size();
        let fits = usize::from(cols).saturating_mul(width) <= FRAME_WIDTH
            && usize::from(rows).saturating_mul(height) <= FRAME_HEIGHT;
        (width * height, fits)
    });
    fonts.get(index).copied().unwrap_or(Font::Hack9x18)
}

fn pick_psf(fonts: &[Psf2], cols: u16, rows: u16) -> usize {
    pick_by(fonts.len(), |i| {
        let (width, height) = (fonts[i].width(), fonts[i].height());
        let fits = usize::from(cols).saturating_mul(width) <= FRAME_WIDTH
            && usize::from(rows).saturating_mul(height) <= FRAME_HEIGHT;
        (width * height, fits)
    })
}

/// Index of the largest area that fits, else the smallest area. The first
/// font wins a tie.
fn pick_by(count: usize, area_fits: impl Fn(usize) -> (usize, bool)) -> usize {
    let mut best: Option<(usize, usize)> = None;
    let mut smallest: Option<(usize, usize)> = None;
    for index in 0..count {
        let (area, fits) = area_fits(index);
        if fits && best.is_none_or(|(have, _)| area > have) {
            best = Some((area, index));
        }
        if smallest.is_none_or(|(have, _)| area < have) {
            smallest = Some((area, index));
        }
    }
    best.or(smallest).map_or(0, |(_, index)| index)
}

fn largest_index(fonts: &[Psf2]) -> usize {
    pick_by(fonts.len(), |i| {
        (fonts[i].width() * fonts[i].height(), true)
    })
}

/// Where the console image sits: centred when it fits, top-left when it
/// does not (the part past 1920x1080 is cropped).
#[must_use]
pub fn origin(cols: u16, rows: u16, font: &Psf2) -> (usize, usize) {
    let native_w = usize::from(cols) * font.width();
    let native_h = usize::from(rows) * font.height();
    (
        FRAME_WIDTH.saturating_sub(native_w) / 2,
        FRAME_HEIGHT.saturating_sub(native_h) / 2,
    )
}

/// The notice, ASCII, into `out`. Returns how many bytes were written.
/// `waiting for grid - no update for 17 s` is the stale form (the issue's
/// text: an ASCII hyphen, no program name in front).
pub fn write_notice(notice: Notice, out: &mut [u8]) -> usize {
    const HEAD: &[u8] = b"waiting for grid - ";
    if out.len() < HEAD.len() {
        return 0;
    }
    out[..HEAD.len()].copy_from_slice(HEAD);
    let mut n = HEAD.len();
    match notice {
        Notice::Missing => copy_tail(b"missing", out, n),
        Notice::Unreadable => copy_tail(b"unreadable", out, n),
        Notice::Refused => copy_tail(b"refused", out, n),
        Notice::Stale(secs) => {
            const MID: &[u8] = b"no update for ";
            if n + MID.len() + 22 > out.len() {
                return n;
            }
            out[n..n + MID.len()].copy_from_slice(MID);
            n += MID.len();
            n += write_u64(secs, &mut out[n..]);
            out[n..n + 2].copy_from_slice(b" s");
            n + 2
        }
    }
}

fn copy_tail(tail: &[u8], out: &mut [u8], n: usize) -> usize {
    if n + tail.len() > out.len() {
        return n;
    }
    out[n..n + tail.len()].copy_from_slice(tail);
    n + tail.len()
}

fn write_u64(mut value: u64, out: &mut [u8]) -> usize {
    let mut tmp = [0_u8; 20];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + u8::try_from(value % 10).unwrap_or(0);
        value /= 10;
        if value == 0 {
            break;
        }
    }
    let digits = &tmp[i..];
    let n = digits.len().min(out.len());
    out[..n].copy_from_slice(&digits[..n]);
    n
}

fn put(frame: &mut [u8], x: usize, y: usize, rgb: [u8; 3]) {
    if x >= FRAME_WIDTH || y >= FRAME_HEIGHT {
        return;
    }
    let at = (y * FRAME_WIDTH + x) * 3;
    frame[at..at + 3].copy_from_slice(&rgb);
}

fn blit_cell(
    frame: &mut [u8],
    font: &Psf2,
    glyph: Option<usize>,
    x0: usize,
    y0: usize,
    fg: [u8; 3],
    bg: [u8; 3],
) {
    let (width, height) = (font.width(), font.height());
    for gy in 0..height {
        let y = y0 + gy;
        if y >= FRAME_HEIGHT {
            break;
        }
        for gx in 0..width {
            let x = x0 + gx;
            if x >= FRAME_WIDTH {
                break;
            }
            let on = glyph.is_some_and(|index| font.pixel(index, gx, gy));
            put(frame, x, y, if on { fg } else { bg });
        }
    }
}

fn paint_view(view: &GridView<'_>, font: &Psf2, frame: &mut [u8]) {
    frame[..FRAME_BYTES].fill(0);
    let (off_x, off_y) = origin(view.cols, view.rows, font);
    let (width, height) = (font.width(), font.height());
    let cols = usize::from(view.cols);
    let rows = usize::from(view.rows);
    for row in 0..rows {
        let y0 = off_y + row * height;
        if y0 >= FRAME_HEIGHT {
            break;
        }
        for col in 0..cols {
            let x0 = off_x + col * width;
            if x0 >= FRAME_WIDTH {
                break;
            }
            let (scalar, fg, bg) = view.cell(row * cols + col);
            blit_cell(frame, font, font.glyph_for_cell(scalar), x0, y0, fg, bg);
        }
    }
}

fn draw_text(frame: &mut [u8], font: &Psf2, x0: usize, y0: usize, text: &[u8], cols: usize) {
    let width = font.width();
    for (col, &byte) in text.iter().enumerate() {
        if col >= cols {
            break;
        }
        let x = x0 + col * width;
        if x >= FRAME_WIDTH {
            break;
        }
        blit_cell(
            frame,
            font,
            font.glyph_for_cell(u32::from(byte)),
            x,
            y0,
            NOTICE_GREY,
            [0, 0, 0],
        );
    }
}

fn draw_on_last_row(frame: &mut [u8], font: &Psf2, cols: u16, rows: u16, text: &[u8]) {
    let (off_x, off_y) = origin(cols, rows, font);
    let height = font.height();
    // The last row that fits in the frame completely. A row that runs past
    // the bottom stays the cell, so the notice is not cut in half.
    let mut row = usize::from(rows).saturating_sub(1);
    while row > 0
        && off_y
            .saturating_add(row.saturating_mul(height))
            .saturating_add(height)
            > FRAME_HEIGHT
    {
        row -= 1;
    }
    let y0 = off_y + row * height;
    let cols_n = usize::from(cols);
    for col in 0..cols_n {
        let x0 = off_x + col * font.width();
        if x0 >= FRAME_WIDTH {
            break;
        }
        blit_cell(frame, font, None, x0, y0, NOTICE_GREY, [0, 0, 0]);
    }
    draw_text(frame, font, off_x, y0, text, cols_n);
}

fn draw_centred(frame: &mut [u8], font: &Psf2, text: &[u8]) {
    let width = font.width();
    let text_w = text.len().saturating_mul(width);
    let x0 = FRAME_WIDTH.saturating_sub(text_w) / 2;
    let y0 = FRAME_HEIGHT.saturating_sub(font.height()) / 2;
    draw_text(frame, font, x0, y0, text, text.len());
}

fn font_index(fonts: &[Psf2], auto: bool, cols: u16, rows: u16) -> usize {
    if auto { pick_psf(fonts, cols, rows) } else { 0 }
}

/// Paint `read` into `frame`. A fresh grid replaces `hold`. Anything else
/// keeps `hold` and draws the notice on the last row of that grid, or
/// centred on black when there has never been a good frame. `now_ms` is the
/// clock, in unix milliseconds.
pub fn compose(
    read: GridRead<'_>,
    now_ms: u64,
    fonts: &[Psf2],
    auto: bool,
    hold: &mut Hold,
    frame: &mut [u8],
) -> io::Result<()> {
    if frame.len() < FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame is short",
        ));
    }
    if fonts.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no font"));
    }
    let mut notice = None;
    let mut size = None;
    match read {
        GridRead::Missing => notice = Some(Notice::Missing),
        GridRead::Unreadable => notice = Some(Notice::Unreadable),
        GridRead::Data(bytes) => match open_bytes(bytes) {
            Err(_) => notice = Some(Notice::Refused),
            Ok(view) => {
                size = Some((view.cols, view.rows));
                if let Some(secs) = stale_secs(view.unix_ms, now_ms) {
                    notice = Some(Notice::Stale(secs));
                } else {
                    let index = font_index(fonts, auto, view.cols, view.rows);
                    paint_view(&view, &fonts[index], frame);
                    hold.pixels.copy_from_slice(&frame[..FRAME_BYTES]);
                    hold.cols = view.cols;
                    hold.rows = view.rows;
                    hold.font_index = index;
                    hold.good = true;
                }
            }
        },
    }
    if let Some(notice) = notice {
        let mut buf = [0_u8; 64];
        let len = write_notice(notice, &mut buf);
        let text = &buf[..len];
        if hold.good {
            frame[..FRAME_BYTES].copy_from_slice(&hold.pixels);
            let font = &fonts[hold.font_index.min(fonts.len() - 1)];
            draw_on_last_row(frame, font, hold.cols, hold.rows, text);
        } else {
            frame[..FRAME_BYTES].fill(0);
            let index = match size {
                Some((cols, rows)) => font_index(fonts, auto, cols, rows),
                None if auto => largest_index(fonts),
                None => 0,
            };
            draw_centred(frame, &fonts[index], text);
        }
    }
    Ok(())
}
