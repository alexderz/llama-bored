//! Diff a cropped view into ANSI. An unchanged view writes nothing.

use std::io::{self, Write};

use crate::screen::{Cell, Color, Screen};

/// Enter the alternate screen and hide the cursor.
pub const ENTER: &[u8] = b"\x1b[?1049h\x1b[?25l";

/// Reset colours, show the cursor, and leave the alternate screen.
pub const RESTORE: &[u8] = b"\x1b[0m\x1b[?25h\x1b[?1049l";

/// Writes [`ENTER`] on creation and [`RESTORE`] on drop, including unwind.
pub struct TermGuard<W: Write> {
    out: W,
    restored: bool,
}

impl<W: Write> TermGuard<W> {
    pub fn enter(mut out: W) -> io::Result<Self> {
        Write::write_all(&mut out, ENTER)?;
        out.flush()?;
        Ok(Self {
            out,
            restored: false,
        })
    }

    pub fn writer(&mut self) -> &mut W {
        &mut self.out
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        Write::write_all(&mut self.out, RESTORE)?;
        self.out.flush()?;
        Ok(())
    }
}

impl<W: Write> Drop for TermGuard<W> {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Top-left crop. `offset_x`/`offset_y` pan the window. Cells past the source are blank.
pub fn crop(screen: &Screen, cols: u16, rows: u16, offset_x: u16, offset_y: u16) -> Screen {
    let width = usize::from(screen.cols);
    let mut cells = Vec::with_capacity(usize::from(cols).saturating_mul(usize::from(rows)));
    for row in 0..rows {
        for col in 0..cols {
            let src_row = offset_y.saturating_add(row);
            let src_col = offset_x.saturating_add(col);
            let cell = if src_row < screen.rows && src_col < screen.cols {
                let index = usize::from(src_row) * width + usize::from(src_col);
                screen.cells.get(index).copied().unwrap_or_else(Cell::blank)
            } else {
                Cell::blank()
            };
            cells.push(cell);
        }
    }
    Screen {
        cols,
        rows,
        cursor_x: 0,
        cursor_y: 0,
        cells,
    }
}

/// Remembers the last view it wrote so the next identical one can be skipped.
pub struct Renderer {
    prev: Option<Grid>,
    buf: Vec<u8>,
    fg: Option<Color>,
    bg: Option<Color>,
}

struct Grid {
    cols: u16,
    rows: u16,
    cells: Vec<Cell>,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            prev: None,
            buf: Vec::new(),
            fg: None,
            bg: None,
        }
    }

    /// Paint `cells` (`cols * rows` long, row-major).
    ///
    /// The first view, and any view whose size changed, is a full repaint.
    /// After that only changed runs are written. An equal view writes nothing.
    pub fn render(&mut self, cols: u16, rows: u16, cells: &[Cell]) -> &[u8] {
        if self
            .prev
            .as_ref()
            .is_some_and(|prev| prev.cols == cols && prev.rows == rows && prev.cells == cells)
        {
            self.buf.clear();
            return &self.buf;
        }

        self.buf.clear();
        let full = self
            .prev
            .as_ref()
            .is_none_or(|prev| prev.cols != cols || prev.rows != rows);
        if full {
            self.buf.extend_from_slice(b"\x1b[0m\x1b[2J\x1b[H");
            self.fg = None;
            self.bg = None;
            emit_all(&mut self.buf, cols, rows, cells, &mut self.fg, &mut self.bg);
        } else if let Some(prev) = self.prev.as_ref() {
            emit_diff(
                &mut self.buf,
                cols,
                rows,
                cells,
                &prev.cells,
                &mut self.fg,
                &mut self.bg,
            );
        }

        match &mut self.prev {
            Some(prev)
                if prev.cols == cols && prev.rows == rows && prev.cells.len() == cells.len() =>
            {
                prev.cells.copy_from_slice(cells);
            }
            _ => {
                self.prev = Some(Grid {
                    cols,
                    rows,
                    cells: cells.to_vec(),
                });
            }
        }
        &self.buf
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

fn emit_all(
    out: &mut Vec<u8>,
    cols: u16,
    rows: u16,
    cells: &[Cell],
    fg: &mut Option<Color>,
    bg: &mut Option<Color>,
) {
    let width = usize::from(cols);
    if width == 0 {
        return;
    }
    for row in 0..rows {
        let start = usize::from(row) * width;
        let Some(line) = cells.get(start..start + width) else {
            break;
        };
        cup(out, row, 0);
        for cell in line {
            emit_sgr(out, *cell, fg, bg);
            emit_char(out, cell.ch);
        }
    }
}

fn emit_diff(
    out: &mut Vec<u8>,
    cols: u16,
    rows: u16,
    cells: &[Cell],
    prev: &[Cell],
    fg: &mut Option<Color>,
    bg: &mut Option<Color>,
) {
    let width = usize::from(cols);
    if width == 0 {
        return;
    }
    let mut run = false;
    let mut run_row = 0u16;
    let mut expect_col = 0u16;
    for row in 0..rows {
        let start = usize::from(row) * width;
        for col in 0..cols {
            let index = start + usize::from(col);
            let (Some(cell), Some(old)) = (cells.get(index), prev.get(index)) else {
                return;
            };
            if cell == old {
                run = false;
                continue;
            }
            let contiguous = run && row == run_row && col == expect_col;
            if !contiguous {
                cup(out, row, col);
                run = true;
                run_row = row;
            }
            emit_sgr(out, *cell, fg, bg);
            emit_char(out, cell.ch);
            expect_col = col.saturating_add(1);
        }
    }
}

fn emit_sgr(out: &mut Vec<u8>, cell: Cell, fg: &mut Option<Color>, bg: &mut Option<Color>) {
    if *fg != Some(cell.fg) {
        sgr(out, cell.fg.fg_sgr());
        *fg = Some(cell.fg);
    }
    if *bg != Some(cell.bg) {
        sgr(out, cell.bg.bg_sgr());
        *bg = Some(cell.bg);
    }
}

fn emit_char(out: &mut Vec<u8>, ch: char) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
}

fn cup(out: &mut Vec<u8>, row: u16, col: u16) {
    out.extend_from_slice(b"\x1b[");
    push_u16(out, row.saturating_add(1));
    out.push(b';');
    push_u16(out, col.saturating_add(1));
    out.push(b'H');
}

fn sgr(out: &mut Vec<u8>, code: u8) {
    out.extend_from_slice(b"\x1b[");
    push_u16(out, u16::from(code));
    out.push(b'm');
}

fn push_u16(out: &mut Vec<u8>, mut n: u16) {
    let mut buf = [0u8; 5];
    let mut i = 5;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screen::{Cell, Color, Screen};

    fn row(text: &str) -> Vec<Cell> {
        text.chars()
            .map(|ch| Cell::new(ch, Color::White, Color::Black))
            .collect()
    }

    #[test]
    fn drop_restores_alternate_screen_and_cursor() {
        let mut buf = Vec::new();
        {
            let _guard = TermGuard::enter(&mut buf).expect("enter");
        }
        assert!(buf.starts_with(ENTER), "enter sequence missing: {buf:?}");
        assert!(buf.ends_with(RESTORE), "restore sequence missing: {buf:?}");
    }

    #[test]
    fn panic_restores_alternate_screen_and_cursor() {
        let mut buf = Vec::new();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = TermGuard::enter(&mut buf).expect("enter");
            panic!("boom");
        }));
        std::panic::set_hook(previous);
        assert!(panicked.is_err());
        assert!(
            buf.ends_with(RESTORE),
            "panic left the terminal unrestored: {buf:?}"
        );
    }

    #[test]
    fn unchanged_frame_emits_zero_bytes() {
        let cells = row("AB");
        let mut renderer = Renderer::new();
        let first = renderer.render(2, 1, &cells).to_vec();
        assert!(!first.is_empty(), "the first frame has to paint the screen");
        assert_eq!(renderer.render(2, 1, &cells), b"");
    }

    #[test]
    fn frame_430x90_stays_inside_the_cpu_budget() {
        let cols = 430u16;
        let rows = 90u16;
        let n = usize::from(cols) * usize::from(rows);
        let mut cells = vec![Cell::new('A', Color::White, Color::Black); n];
        let mut renderer = Renderer::new();
        let full_at = std::time::Instant::now();
        assert!(!renderer.render(cols, rows, &cells).is_empty());
        let full = full_at.elapsed();

        let same_at = std::time::Instant::now();
        let repeats = 50u32;
        for _ in 0..repeats {
            assert!(renderer.render(cols, rows, &cells).is_empty());
        }
        let same = same_at.elapsed() / repeats;

        let diff_at = std::time::Instant::now();
        for i in 0..repeats {
            cells[0].ch = if i % 2 == 0 { 'B' } else { 'A' };
            assert!(!renderer.render(cols, rows, &cells).is_empty());
        }
        let diff = diff_at.elapsed() / repeats;

        let same_pct = same.as_secs_f64() * 10.0 * 100.0;
        let diff_pct = diff.as_secs_f64() * 10.0 * 100.0;
        let full_pct = full.as_secs_f64() * 10.0 * 100.0;
        eprintln!(
            "430x90 at 10 fps: identical {same:?} ({same_pct:.4}%), one-cell {diff:?} ({diff_pct:.4}%), full {full:?} ({full_pct:.4}%)"
        );
        // Debug builds are noisier. The release measurement is what the ticket asks for.
        let limit = if cfg!(debug_assertions) {
            std::time::Duration::from_millis(5)
        } else {
            std::time::Duration::from_millis(1)
        };
        assert!(
            same < limit,
            "identical frame {same:?} is {same_pct:.3}% of a thread at 10 fps"
        );
        assert!(
            diff < limit,
            "one-cell frame {diff:?} is {diff_pct:.3}% of a thread at 10 fps"
        );
    }

    #[test]
    fn crops_from_the_top_left_with_an_offset() {
        let screen = letter_screen(4, 3);
        let view = crop(&screen, 2, 2, 1, 1);
        assert_eq!(view.cols, 2);
        assert_eq!(view.rows, 2);
        assert_eq!(text(&view), "FGJK");
        let origin = crop(&screen, 2, 1, 0, 0);
        assert_eq!(text(&origin), "AB");
        let hung = crop(&screen, 2, 1, 3, 2);
        assert_eq!(text(&hung), "L ");
    }

    #[test]
    fn hostile_bytes_never_reach_rendered_output() {
        let bytes = vcsa_bytes(1, 3, &[(0x1B, 0x07), (0x9B, 0x07), (0x7F, 0x07)]);
        let screen = crate::decode_screen(&bytes, None, None).expect("decode");
        let mut renderer = Renderer::new();
        let out = renderer
            .render(screen.cols, screen.rows, &screen.cells)
            .to_vec();
        assert_eq!(payload(&out).expect("only our sequences"), "???");
        assert!(!out.contains(&0x7F), "DEL reached the output: {out:?}");
        assert!(
            !out.windows(2)
                .any(|pair| pair == [0xC2, 0x9B] || pair == [0xC3, 0x9B]),
            "a C1 or Latin-1 encoding reached the output: {out:?}"
        );
    }

    fn vcsa_bytes(rows: u8, cols: u8, cells: &[(u8, u8)]) -> Vec<u8> {
        let mut bytes = vec![rows, cols, 0, 0];
        for (ch, attr) in cells {
            bytes.push(*ch);
            bytes.push(*attr);
        }
        bytes
    }

    /// Strip our CSI sequences and return the cell text.
    fn payload(bytes: &[u8]) -> Result<String, ()> {
        let mut text = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0x1B {
                if bytes.get(i + 1) != Some(&b'[') {
                    return Err(());
                }
                i += 2;
                while i < bytes.len() && matches!(bytes[i], b'0'..=b'9' | b';' | b'?') {
                    i += 1;
                }
                if i >= bytes.len() || !(0x40..=0x7E).contains(&bytes[i]) {
                    return Err(());
                }
                i += 1;
                continue;
            }
            text.push(bytes[i]);
            i += 1;
        }
        String::from_utf8(text).map_err(|_| ())
    }

    fn letter_screen(cols: u16, rows: u16) -> Screen {
        let mut cells = Vec::new();
        let mut ch = b'A';
        for _ in 0..rows {
            for _ in 0..cols {
                cells.push(Cell::new(char::from(ch), Color::White, Color::Black));
                ch += 1;
            }
        }
        Screen {
            cols,
            rows,
            cursor_x: 0,
            cursor_y: 0,
            cells,
        }
    }

    fn text(screen: &Screen) -> String {
        screen.cells.iter().map(|cell| cell.ch).collect()
    }
}
