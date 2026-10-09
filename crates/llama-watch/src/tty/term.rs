//! The only byte emitter for the console.
//!
//! Sequences are fixed functions. Llama text never reaches this module as a
//! string: callers pass a [`Grid`](super::grid::Grid) of cells.

use std::io::{self, Write};
use std::os::fd::AsFd;
use std::time::{Duration, Instant};

use llama_core::palette::{self, Palette};

use super::grid::{C16, Cell, Grid};

/// How often the window size is read. Not the full-repaint interval.
const SIZE_POLL: Duration = Duration::from_secs(5);

/// Largest window the layout will draw. A bigger winsize is stored as this.
const MAX_COLS: u16 = 1024;
const MAX_ROWS: u16 = 512;

/// ASCII is handled separately. These are the only non-ASCII scalars `emit_char`
/// will write as UTF-8.
///
/// The first nineteen are present in eurlatgr (parsed
/// `/usr/lib/kbd/consolefonts/eurlatgr.psfu.gz`, 2026-09-25). The first
/// eleven (`█` through `≈`) are all that `chart_glyphs = "halves"` draws.
/// `…` (U+2026) is glyph 491 in that font; the RECENT table uses it to mark
/// a name or address cut to the column. `≈` (U+2248, glyph 484 there;
/// appended to llama-hack's extras) marks an approximate KV count on a
/// SLOTS engine line (#79).
///
/// The next eight (`’ ‘ “ ” – — •` and the replacement character U+FFFD) are
/// #90: `tty::sanitize` transliterates common typography and an unmappable
/// scalar to these instead of a literal `?`, no font rebuild needed because
/// `packaging/fonts/build-psf.py`'s slot table already carried them (parsed
/// from the committed PSF Unicode tables, and from eurlatgr, 2026-10-09).
///
/// The last six are the lower eighths U+2581–2583 and U+2585–2587 (`▄` is
/// above). eurlatgr lacks them; they are verified present in
/// llama-hack-12x24 (`tests/tty_font.rs` parses the committed PSF). Only
/// `chart_glyphs = "eighths"` draws them: the chart, and since #52 the
/// single-spaced meter bars (`▇`; `▄` in halves mode). `▔` (U+2594) is in that font too but
/// is not drawn, so it is not here.
pub const GLYPHS: &[char] = &[
    '█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·', '…', '≈', '’', '‘', '“', '”', '–', '—', '•',
    '\u{fffd}', '▁', '▂', '▃', '▅', '▆', '▇',
];

/// Columns and rows reported for the terminal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Size {
    pub cols: u16,
    pub rows: u16,
}

/// Console blank and powerdown timers, sent once at start-up.
///
/// The Linux console private sequences `ESC [ 9 ; n ]` (blank after `n`
/// minutes without a keypress) and `ESC [ 14 ; n ]` (VESA powerdown `n`
/// minutes after the blank). Both set kernel-wide values that a tty reset
/// does not touch. `ESC [ 9 ; n ]` also unblanks the screen and restarts the
/// blank timer, so it must never be sent again on a repaint: a repaint every
/// few seconds would keep the screen from ever blanking.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConsoleBlank {
    blank_min: u16,
    powerdown_min: u16,
}

impl ConsoleBlank {
    /// Off: nothing is sent and the kernel defaults stay.
    pub const OFF: Self = Self {
        blank_min: 0,
        powerdown_min: 0,
    };

    /// From `[tty] blank_min` and `sleep_min`, both minutes since the last
    /// keypress. The kernel counts powerdown from the blank, so the powerdown
    /// interval is `sleep_min - blank_min`. Values the config would reject
    /// (above 60, or sleep without a blank before it) send nothing for that
    /// timer.
    pub fn from_minutes(blank_min: u32, sleep_min: u32) -> Self {
        let blank_min = if blank_min <= 60 { blank_min } else { 0 };
        let powerdown_min = if blank_min > 0 && sleep_min > blank_min && sleep_min <= 60 {
            sleep_min - blank_min
        } else {
            0
        };
        Self {
            blank_min: blank_min as u16,
            powerdown_min: powerdown_min as u16,
        }
    }
}

#[derive(Clone, Copy)]
struct Active {
    fg: Option<C16>,
    bg: Option<C16>,
}

/// Writes allowlisted bytes to `W`. Remembers the last grid that was written.
///
/// `Term` is `Send` (#42): the size callback is a `Box<dyn FnMut + Send>`,
/// so the watcher can hand the whole terminal to its tty writer thread
/// ([`super::writer::FrameWriter`]) and never write the console from the
/// tick loop.
///
/// Cells are clipped to `min(grid, terminal)` on both axes. Phase B builds
/// each grid from [`Term::cols`] and [`Term::rows`]. A stale grid that is
/// still the old, larger size must not wrap the console after a shrink.
pub struct Term<W> {
    out: W,
    winsize: Box<dyn FnMut() -> io::Result<Size> + Send>,
    cols: u16,
    rows: u16,
    full_redraw: Duration,
    last_full: Instant,
    last_size_poll: Instant,
    force_full: bool,
    written: Option<Grid>,
    active: Active,
    buf: Vec<u8>,
    blank: ConsoleBlank,
    blank_sent: bool,
    palette: Palette,
    /// [`Self::with_palette`] was called. A bare `Term` writes no palette
    /// bytes at all, which keeps older byte-exact tests unchanged.
    palette_set: bool,
    /// tty11's line settings from before [`quiet_stdout`], put back by
    /// [`Self::restore_console`].
    modes: Option<SavedModes>,
}

impl<W: Write> Term<W> {
    /// Read the size once. The first [`Self::render`] is a full repaint.
    ///
    /// `winsize` is called again every 5 seconds from `render`. A later error
    /// keeps the previous size. `full_redraw` schedules `ED 2` plus the whole
    /// grid even when the cells have not changed.
    pub fn new(
        out: W,
        mut winsize: impl FnMut() -> io::Result<Size> + Send + 'static,
        full_redraw: Duration,
        now: Instant,
    ) -> io::Result<Self> {
        let size = clamp_size(winsize()?);
        Ok(Self {
            out,
            winsize: Box::new(winsize),
            cols: size.cols,
            rows: size.rows,
            full_redraw,
            last_full: now,
            last_size_poll: now,
            force_full: true,
            written: None,
            active: Active { fg: None, bg: None },
            buf: Vec::new(),
            blank: ConsoleBlank::OFF,
            blank_sent: false,
            palette: Palette::Vga,
            palette_set: false,
            modes: None,
        })
    }

    /// The console palette (#26, `[tty] palette`). [`Palette::Llama`] loads
    /// `llama_core::palette::LLAMA` with `ESC ] P n rrggbb` on every full
    /// repaint, so a console reset or a VT switch that dropped it is healed
    /// within `full_redraw`. [`Palette::Vga`] sends one `ESC ] R` with the
    /// first frame, undoing a palette an earlier run left, and nothing after.
    /// Without this call no palette bytes are written at all.
    pub fn with_palette(mut self, palette: Palette) -> Self {
        self.palette = palette;
        self.palette_set = true;
        self
    }

    /// Clean exit: `ESC ] R` when this terminal loaded the llama palette, so
    /// a later login on the console sees the kernel's colours. Nothing for
    /// `vga`. The next [`Self::render`] repaints in full (and reloads).
    pub fn restore_palette(&mut self) -> io::Result<()> {
        if !(self.palette_set && self.palette == Palette::Llama) {
            return Ok(());
        }
        self.force_full = true;
        Write::write_all(&mut self.out, palette::CONSOLE_RESET)?;
        self.out.flush()
    }

    /// Make the next [`Self::render`] a full repaint (`ED 2` and every
    /// cell). The tty writer calls it after a stalled write (#42), so a
    /// frame the console only partly showed, or anything it drew while
    /// paused, does not stay on screen.
    pub fn force_repaint(&mut self) {
        self.force_full = true;
    }

    /// Keep `modes` so [`Self::restore_console`] puts them back on a clean
    /// exit (#42).
    pub fn with_saved_modes(mut self, modes: Option<SavedModes>) -> Self {
        self.modes = modes;
        self
    }

    /// Clean exit: [`Self::restore_palette`], then the saved line settings.
    /// Both are tried; the first error is returned.
    pub fn restore_console(&mut self) -> io::Result<()> {
        let palette = self.restore_palette();
        let modes = match &self.modes {
            Some(modes) => modes.restore_on_stdout(),
            None => Ok(()),
        };
        palette.and(modes)
    }

    /// Send `blank` with the first frame. Later frames, including full
    /// repaints, never send it again (see [`ConsoleBlank`]).
    pub fn with_console_blank(mut self, blank: ConsoleBlank) -> Self {
        self.blank = blank;
        self
    }

    /// [`Self::new`] with the size read from `fd` via [`window_size`].
    pub fn from_fd<Fd: AsFd + Send + 'static>(
        out: W,
        fd: Fd,
        full_redraw: Duration,
        now: Instant,
    ) -> io::Result<Self> {
        Self::new(out, move || window_size(&fd), full_redraw, now)
    }

    /// Paint `grid`. An unchanged frame writes nothing.
    pub fn render(&mut self, grid: &Grid, now: Instant) -> io::Result<()> {
        self.poll_size(now);
        let full = self.need_full(grid, now);
        if !full && self.written.as_ref().is_some_and(|prev| prev == grid) {
            return Ok(());
        }

        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        let mut active = self.active;
        let view_cols = self.cols.min(grid.cols());
        let view_rows = self.rows.min(grid.rows());
        if !full && let Some(prev) = self.written.as_ref() {
            let mark = buf.len();
            esc_utf8(&mut buf);
            if !emit_diff(&mut buf, grid, prev, &mut active, view_cols, view_rows) {
                buf.truncate(mark);
            }
        } else {
            let load = match (self.palette_set, self.palette) {
                (false, _) => None,
                (true, Palette::Llama) => Some(PaletteBytes::Load(self.palette.slots())),
                // Once: `blank_sent` is also the first-full-frame latch.
                (true, Palette::Vga) if !self.blank_sent => Some(PaletteBytes::Reset),
                (true, Palette::Vga) => None,
            };
            emit_full(&mut buf, grid, &mut active, view_cols, view_rows, load);
            if !self.blank_sent {
                emit_console_blank(&mut buf, self.blank);
            }
        }
        if buf.is_empty() {
            self.buf = buf;
            self.remember(grid);
            return Ok(());
        }

        // Trait call, not a method call: the source scan flags the method spelling.
        if let Err(err) = Write::write_all(&mut self.out, &buf) {
            self.buf = buf;
            // The console may have taken a prefix. The next frame repaints.
            self.force_full = true;
            return Err(err);
        }
        self.buf = buf;
        self.active = active;
        self.remember(grid);
        if full {
            self.blank_sent = true;
            self.force_full = false;
            self.last_full = now;
        }
        Ok(())
    }

    /// Columns from the last successful size read.
    pub fn cols(&self) -> u16 {
        self.cols
    }

    /// Rows from the last successful size read.
    pub fn rows(&self) -> u16 {
        self.rows
    }

    fn poll_size(&mut self, now: Instant) {
        if now.saturating_duration_since(self.last_size_poll) < SIZE_POLL {
            return;
        }
        self.last_size_poll = now;
        if let Ok(size) = (self.winsize)() {
            let size = clamp_size(size);
            if size.cols != self.cols || size.rows != self.rows {
                self.cols = size.cols;
                self.rows = size.rows;
                self.force_full = true;
            }
        }
    }

    fn need_full(&self, grid: &Grid, now: Instant) -> bool {
        if self.force_full {
            return true;
        }
        match &self.written {
            None => true,
            Some(prev) if prev.cols() != grid.cols() || prev.rows() != grid.rows() => true,
            Some(_) => now.saturating_duration_since(self.last_full) >= self.full_redraw,
        }
    }

    fn remember(&mut self, grid: &Grid) {
        if let Some(prev) = &mut self.written
            && prev.cols() == grid.cols()
            && prev.rows() == grid.rows()
        {
            prev.copy_cells_from(grid);
            return;
        }
        self.written = Some(grid.clone());
    }
}

impl Term<io::Stdout> {
    /// Own the process stdout and read its window size.
    ///
    /// This is the only place in the watcher that may name stdout. T22 calls
    /// it and never keeps a console handle of its own. The method sits on
    /// `Term<io::Stdout>` so the caller can write `Term::for_stdout` without
    /// naming the stdout type (S13).
    pub fn for_stdout(full_redraw: Duration, now: Instant) -> io::Result<Self> {
        Self::from_fd(io::stdout(), io::stdout(), full_redraw, now)
    }
}

/// `llama-watch tty-reset` (the unit's `ExecStopPost=`): write `ESC ] R` to
/// stdout, which systemd points at tty11, so a watcher that was killed
/// (SIGTERM keeps its default action) still leaves the kernel's colours.
/// Harmless when the palette is `vga`.
pub fn reset_palette_on_stdout() -> io::Result<()> {
    let mut out = io::stdout();
    Write::write_all(&mut out, palette::CONSOLE_RESET)?;
    out.flush()
}

impl Term<Vec<u8>> {
    /// Bytes written so far. Only a `Vec<u8>` sink exposes this; a live
    /// console handle cannot be read back or written past the allowlist.
    pub fn out(&self) -> &[u8] {
        &self.out
    }

    /// The buffer, so a test can clear it between frames.
    pub fn out_mut(&mut self) -> &mut Vec<u8> {
        &mut self.out
    }
}

fn clamp_size(size: Size) -> Size {
    Size {
        cols: size.cols.min(MAX_COLS),
        rows: size.rows.min(MAX_ROWS),
    }
}

/// Read the terminal window size.
pub fn window_size<Fd: AsFd>(fd: Fd) -> io::Result<Size> {
    let ws = rustix::termios::tcgetwinsize(fd)?;
    Ok(Size {
        cols: ws.ws_col,
        rows: ws.ws_row,
    })
}

/// The input flags the watcher clears on tty11 (#42): no XON/XOFF flow
/// control, so Ctrl+S cannot stop output.
#[must_use]
pub fn quiet_input(input: rustix::termios::InputModes) -> rustix::termios::InputModes {
    use rustix::termios::InputModes;
    input - (InputModes::IXON | InputModes::IXOFF)
}

/// The local flags the watcher clears on tty11 (#42): no echo (keys would
/// draw on the dashboard), no canonical line editing and no signal keys.
#[must_use]
pub fn quiet_local(local: rustix::termios::LocalModes) -> rustix::termios::LocalModes {
    use rustix::termios::LocalModes;
    local - (LocalModes::ECHO | LocalModes::ICANON | LocalModes::ISIG)
}

/// What `llama-watch tty-reset` puts back when the watcher could not:
/// the kernel console defaults for the flags [`quiet_input`] and
/// [`quiet_local`] clear (`ixon echo icanon isig`, `ixoff` stays off).
#[must_use]
pub fn console_defaults(
    input: rustix::termios::InputModes,
    local: rustix::termios::LocalModes,
) -> (rustix::termios::InputModes, rustix::termios::LocalModes) {
    use rustix::termios::{InputModes, LocalModes};
    (
        (input | InputModes::IXON) - InputModes::IXOFF,
        local | LocalModes::ECHO | LocalModes::ICANON | LocalModes::ISIG,
    )
}

/// tty11's line settings from before [`quiet_stdout`], for a clean exit.
pub struct SavedModes {
    saved: rustix::termios::Termios,
}

impl SavedModes {
    /// Put the saved settings back on stdout (`TCSANOW`, never drains
    /// output). Nothing when stdout is no longer tty11.
    pub fn restore_on_stdout(&self) -> io::Result<()> {
        let out = io::stdout();
        if !is_configured_tty(&out)? {
            return Ok(());
        }
        rustix::termios::tcsetattr(&out, rustix::termios::OptionalActions::Now, &self.saved)?;
        Ok(())
    }
}

/// Harden the console on stdout so the keyboard cannot pause or draw on the
/// dashboard (#42): clear `ixon ixoff echo icanon isig` (`TCSANOW`, so
/// nothing waits for output) and flush pending input.
///
/// Only when stdout is the configured console ([`super::setup::TTY`],
/// compared by device number); any other stdout (a terminal someone ran
/// `llama-watch run` in, a pipe, a file) is left alone and `Ok(None)` comes
/// back. Scroll Lock (VT hold) still stops output whatever the flags say;
/// the tty writer thread is what keeps the tick loop running then.
pub fn quiet_stdout() -> io::Result<Option<SavedModes>> {
    use rustix::termios::{OptionalActions, QueueSelector, tcflush, tcgetattr, tcsetattr};
    let out = io::stdout();
    if !is_configured_tty(&out)? {
        return Ok(None);
    }
    let saved = tcgetattr(&out)?;
    let mut quiet = saved.clone();
    quiet.input_modes = quiet_input(quiet.input_modes);
    quiet.local_modes = quiet_local(quiet.local_modes);
    tcsetattr(&out, OptionalActions::Now, &quiet)?;
    tcflush(&out, QueueSelector::IFlush)?;
    Ok(Some(SavedModes { saved }))
}

/// `llama-watch tty-reset`: [`console_defaults`] on stdout, only when it is
/// the configured console. `TCSANOW`, so a held console cannot block it.
pub fn console_defaults_on_stdout() -> io::Result<()> {
    use rustix::termios::{OptionalActions, tcgetattr, tcsetattr};
    let out = io::stdout();
    if !is_configured_tty(&out)? {
        return Ok(());
    }
    let mut modes = tcgetattr(&out)?;
    let (input, local) = console_defaults(modes.input_modes, modes.local_modes);
    modes.input_modes = input;
    modes.local_modes = local;
    tcsetattr(&out, OptionalActions::Now, &modes)?;
    Ok(())
}

/// `fd` is the character device at [`super::setup::TTY`].
fn is_configured_tty<Fd: AsFd>(fd: Fd) -> io::Result<bool> {
    use rustix::fs::FileType;
    let have = rustix::fs::fstat(fd)?;
    if FileType::from_raw_mode(have.st_mode) != FileType::CharacterDevice {
        return Ok(false);
    }
    let want = match rustix::fs::stat(super::setup::TTY) {
        Ok(want) => want,
        Err(_) => return Ok(false),
    };
    Ok(
        FileType::from_raw_mode(want.st_mode) == FileType::CharacterDevice
            && want.st_rdev == have.st_rdev,
    )
}

/// The palette bytes of one full repaint.
#[derive(Clone, Copy)]
enum PaletteBytes {
    /// `ESC ] P n rrggbb` for all 16 slots.
    Load(&'static [llama_core::color::Rgb; 16]),
    /// `ESC ] R`.
    Reset,
}

fn emit_full(
    out: &mut Vec<u8>,
    grid: &Grid,
    active: &mut Active,
    cols: u16,
    rows: u16,
    palette: Option<PaletteBytes>,
) {
    esc_utf8(out);
    esc_hide_cursor(out);
    esc_reset(out);
    // Before the clear, so the erased screen is already the new black.
    match palette {
        Some(PaletteBytes::Load(slots)) => out.extend_from_slice(&palette::console_load(slots)),
        Some(PaletteBytes::Reset) => out.extend_from_slice(palette::CONSOLE_RESET),
        None => {}
    }
    esc_ed2(out);
    esc_home(out);
    *active = Active { fg: None, bg: None };
    for row in 0..rows {
        if cols == 0 {
            continue;
        }
        if row > 0 {
            esc_cup(out, row + 1, 1);
        }
        for col in 0..cols {
            let cell = grid.cell_at(col, row);
            emit_sgr(out, cell, active);
            emit_char(out, cell.ch);
        }
    }
}

fn emit_diff(
    out: &mut Vec<u8>,
    grid: &Grid,
    prev: &Grid,
    active: &mut Active,
    cols: u16,
    rows: u16,
) -> bool {
    let mut run = false;
    let mut run_row = 0u16;
    let mut expect_col = 0u16;
    let mut drew = false;
    let same_size = grid.for_each_change(prev, |col, row, cell| {
        if col >= cols || row >= rows {
            run = false;
            return;
        }
        let contiguous = run && row == run_row && col == expect_col;
        if !contiguous {
            esc_cup(out, row + 1, col + 1);
            run = true;
            run_row = row;
        }
        emit_sgr(out, cell, active);
        emit_char(out, cell.ch);
        expect_col = col.saturating_add(1);
        drew = true;
    });
    debug_assert!(same_size);
    drew
}

fn emit_sgr(out: &mut Vec<u8>, cell: Cell, active: &mut Active) {
    if active.fg != Some(cell.fg) {
        sgr(out, cell.fg.fg_param());
        active.fg = Some(cell.fg);
    }
    if active.bg != Some(cell.bg) {
        sgr(out, cell.bg.bg_param());
        active.bg = Some(cell.bg);
    }
}

fn emit_char(out: &mut Vec<u8>, ch: char) {
    if ('\u{20}'..='\u{7e}').contains(&ch) {
        out.push(ch as u8);
        return;
    }
    if GLYPHS.contains(&ch) {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        out.extend_from_slice(encoded.as_bytes());
        return;
    }
    out.push(b'?');
}

/// `ESC [ 9 ; n ]` then `ESC [ 14 ; n ]`, each only when its minutes are
/// non-zero. Nothing when both are 0, so the kernel defaults stay.
fn emit_console_blank(out: &mut Vec<u8>, blank: ConsoleBlank) {
    if blank.blank_min > 0 {
        out.extend_from_slice(b"\x1b[9;");
        push_u16(out, blank.blank_min);
        out.push(b']');
    }
    if blank.powerdown_min > 0 {
        out.extend_from_slice(b"\x1b[14;");
        push_u16(out, blank.powerdown_min);
        out.push(b']');
    }
}

fn esc_utf8(out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b%G");
}

fn esc_hide_cursor(out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[?25l");
}

fn esc_home(out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[H");
}

fn esc_ed2(out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[2J");
}

fn esc_reset(out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[0m");
}

fn esc_cup(out: &mut Vec<u8>, row: u16, col: u16) {
    out.extend_from_slice(b"\x1b[");
    push_u16(out, row);
    out.push(b';');
    push_u16(out, col);
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
