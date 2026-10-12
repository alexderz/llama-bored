//! Cell grid. `ch` is a Unicode scalar; the sanitiser and the emitter decide
//! which scalars are legal. `rich` is the same cell for a full-Unicode
//! viewer (the grid file, #107); tty11 never reads it.
//!
//! Exact colours (#116) are a side plane, not fields of [`Cell`]. The
//! console layout never allocates it, so a frame with the grid file off
//! stores nothing extra per cell. The plane is eight bytes a cell when a
//! grid records exact colours.

use llama_core::color::Rgb;

/// The 16 console colours. Foreground uses all of them (SGR 30–37 and 90–97).
/// Background SGR is only 40–47, so a bright background is drawn as the
/// matching normal colour.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum C16 {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    #[default]
    White,
    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

impl C16 {
    /// ANSI slot number, 0..=15: the index into `llama_core::palette`.
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub(crate) fn fg_param(self) -> u8 {
        match self {
            Self::Black => 30,
            Self::Red => 31,
            Self::Green => 32,
            Self::Yellow => 33,
            Self::Blue => 34,
            Self::Magenta => 35,
            Self::Cyan => 36,
            Self::White => 37,
            Self::BrightBlack => 90,
            Self::BrightRed => 91,
            Self::BrightGreen => 92,
            Self::BrightYellow => 93,
            Self::BrightBlue => 94,
            Self::BrightMagenta => 95,
            Self::BrightCyan => 96,
            Self::BrightWhite => 97,
        }
    }

    /// Background SGR is 40–47. Bright colours use the matching normal colour.
    pub(crate) fn bg_param(self) -> u8 {
        match self {
            Self::Black | Self::BrightBlack => 40,
            Self::Red | Self::BrightRed => 41,
            Self::Green | Self::BrightGreen => 42,
            Self::Yellow | Self::BrightYellow => 43,
            Self::Blue | Self::BrightBlue => 44,
            Self::Magenta | Self::BrightMagenta => 45,
            Self::Cyan | Self::BrightCyan => 46,
            Self::White | Self::BrightWhite => 47,
        }
    }
}

/// One console cell.
///
/// `ch == '\n'` is an end-of-line marker from the sanitiser, not a glyph.
/// Every other `ch` the sanitiser produces is printable ASCII or `?`.
///
/// `rich` is what a full-Unicode terminal should draw instead of `ch`: the
/// original scalar when it is safe to show as one cell
/// (`llama_core::glyph::is_showable`), else `ch` itself. The console emitter
/// ignores it; it is what the grid file carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cell {
    pub ch: char,
    pub rich: char,
    pub fg: C16,
    pub bg: C16,
}

impl Cell {
    /// A cell whose `rich` is its `ch`.
    pub const fn new(ch: char, fg: C16, bg: C16) -> Self {
        Self {
            ch,
            rich: ch,
            fg,
            bg,
        }
    }

    /// A cell the console draws as `ch` and a full-Unicode viewer as `rich`.
    pub const fn with_rich(ch: char, rich: char, fg: C16, bg: C16) -> Self {
        Self { ch, rich, fg, bg }
    }

    /// Space, white on black.
    pub const fn blank() -> Self {
        Self::new(' ', C16::White, C16::Black)
    }

    /// Same console output: `ch`, `fg` and `bg` match, `rich` is ignored.
    pub const fn same_console(self, other: Self) -> bool {
        self.ch == other.ch && self.fg as u8 == other.fg as u8 && self.bg as u8 == other.bg as u8
    }

    pub const fn is_line_end(self) -> bool {
        self.ch == '\n'
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::blank()
    }
}

/// One cell's exact foreground and background, beside its palette colour.
///
/// `None` means the grid file writes the palette slot. Eight bytes: two
/// `Option<Rgb>`, each four.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Exact {
    fg: Option<Rgb>,
    bg: Option<Rgb>,
}

/// Row-major cells. `cells.len()` is always `cols * rows`.
///
/// The fields are private so a caller cannot hand the emitter a vec whose
/// length does not match the declared size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grid {
    cells: Vec<Cell>,
    cols: u16,
    rows: u16,
    /// Exact colours, one per cell, or absent. The console grid leaves this
    /// empty. [`PartialEq`] includes it, so a frame that changed only its
    /// exact colours is still rewritten to the grid file; [`Self::same_console`]
    /// ignores it, so tty11 does not.
    exact: Option<Vec<Exact>>,
}

impl Grid {
    /// A grid of blank cells. The length is `cols * rows`.
    ///
    /// If that product does not fit in `usize`, the grid is empty.
    pub fn new(cols: u16, rows: u16) -> Self {
        let Some(n) = usize::from(cols).checked_mul(usize::from(rows)) else {
            return Self {
                cells: Vec::new(),
                cols: 0,
                rows: 0,
                exact: None,
            };
        };
        Self {
            cells: vec![Cell::blank(); n],
            cols,
            rows,
            exact: None,
        }
    }

    /// [`Self::new`] plus an exact-colour plane, every channel empty.
    ///
    /// The plane is eight bytes a cell (`2 * size_of::<Option<Rgb>>()`).
    /// A console frame uses [`Self::new`] and does not allocate it.
    pub fn with_exact(cols: u16, rows: u16) -> Self {
        let mut grid = Self::new(cols, rows);
        grid.exact = Some(vec![Exact::default(); grid.cells.len()]);
        grid
    }

    /// A grid of blank cells.
    pub fn blank(cols: u16, rows: u16) -> Self {
        Self::new(cols, rows)
    }

    /// This grid has an exact-colour plane. Painters skip the ramp when it
    /// does not.
    pub(crate) fn records_exact(&self) -> bool {
        self.exact.is_some()
    }

    /// Exact foreground at `col`, `row`, if a painter set one.
    pub(crate) fn exact_fg(&self, col: u16, row: u16) -> Option<Rgb> {
        self.exact_at(col, row).and_then(|exact| exact.fg)
    }

    /// Exact background at `col`, `row`, if a painter set one.
    pub(crate) fn exact_bg(&self, col: u16, row: u16) -> Option<Rgb> {
        self.exact_at(col, row).and_then(|exact| exact.bg)
    }

    /// Record `rgb` as the exact foreground. No plane, or the unpainted
    /// margin (the last row and the last column), stores nothing.
    pub(crate) fn set_exact_fg(&mut self, col: u16, row: u16, rgb: Rgb) {
        if let Some(slot) = self.exact_slot(col, row) {
            slot.fg = Some(rgb);
        }
    }

    /// Record `rgb` as the exact background. Same limits as [`Self::set_exact_fg`].
    pub(crate) fn set_exact_bg(&mut self, col: u16, row: u16, rgb: Rgb) {
        if let Some(slot) = self.exact_slot(col, row) {
            slot.bg = Some(rgb);
        }
    }

    /// Drop any exact colour at `col`, `row`. A painter calls this when it
    /// overwrites a cell, so a later blank does not keep a stale ramp.
    /// No plane: return before the index. The console layout paints every
    /// cell through here and has no plane.
    pub(crate) fn clear_exact(&mut self, col: u16, row: u16) {
        if self.exact.is_none() {
            return;
        }
        let Some(i) = self.checked_index(col, row) else {
            return;
        };
        if let Some(plane) = self.exact.as_mut() {
            plane[i] = Exact::default();
        }
    }

    fn exact_at(&self, col: u16, row: u16) -> Option<Exact> {
        let i = self.checked_index(col, row)?;
        self.exact.as_ref()?.get(i).copied()
    }

    /// Paintable cells only: the last row and the last column stay blank.
    fn exact_slot(&mut self, col: u16, row: u16) -> Option<&mut Exact> {
        if row >= self.rows.saturating_sub(1) || col >= self.cols.saturating_sub(1) {
            return None;
        }
        let i = self.checked_index(col, row)?;
        self.exact.as_mut()?.get_mut(i)
    }

    pub fn cols(&self) -> u16 {
        self.cols
    }

    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// Store `cell`. Coordinates outside the grid are ignored.
    pub fn put(&mut self, col: u16, row: u16, cell: Cell) {
        let Some(i) = self.checked_index(col, row) else {
            return;
        };
        self.cells[i] = cell;
    }

    /// The cell at `col`, `row`, or `None` when that coordinate is outside.
    pub fn get(&self, col: u16, row: u16) -> Option<Cell> {
        self.checked_index(col, row).map(|i| self.cells[i])
    }

    /// Every cell has the same console output as `other`'s (see
    /// [`Cell::same_console`]) and the sizes match. A cell that differs only
    /// in `rich` writes nothing to tty11.
    pub(crate) fn same_console(&self, other: &Grid) -> bool {
        self.cols == other.cols
            && self.rows == other.rows
            && self
                .cells
                .iter()
                .zip(&other.cells)
                .all(|(a, b)| a.same_console(*b))
    }

    /// Cells in row-major order.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub(crate) fn cell_at(&self, col: u16, row: u16) -> Cell {
        self.cells[self.index(col, row)]
    }

    pub(crate) fn copy_cells_from(&mut self, other: &Grid) {
        self.cells.copy_from_slice(&other.cells);
        self.exact.clone_from(&other.exact);
    }

    fn index(&self, col: u16, row: u16) -> usize {
        usize::from(row) * usize::from(self.cols) + usize::from(col)
    }

    fn checked_index(&self, col: u16, row: u16) -> Option<usize> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        Some(self.index(col, row))
    }

    /// Visit cells that differ from `written`, in row-major order.
    ///
    /// Returns `false` when the sizes differ. The caller then repaints.
    pub(crate) fn for_each_change(
        &self,
        written: &Grid,
        mut visit: impl FnMut(u16, u16, Cell),
    ) -> bool {
        if self.cols != written.cols || self.rows != written.rows {
            return false;
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = self.cell_at(col, row);
                if !cell.same_console(written.cell_at(col, row)) {
                    visit(col, row, cell);
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exact_plane_is_eight_bytes_a_cell_and_absent_until_asked() {
        assert_eq!(std::mem::size_of::<Exact>(), 8);
        assert_eq!(std::mem::size_of::<Option<Rgb>>(), 4);
        let plain = Grid::new(192, 60);
        assert!(!plain.records_exact());
        assert!(plain.exact_fg(0, 0).is_none());

        let mut exact = Grid::with_exact(192, 60);
        let plane = exact.exact.as_ref().expect("plane");
        assert_eq!(plane.len(), 192 * 60);
        assert_eq!(plane.len() * std::mem::size_of::<Exact>(), 92_160);
        let rgb = Rgb { r: 1, g: 2, b: 3 };
        exact.set_exact_fg(3, 4, rgb);
        exact.set_exact_bg(3, 4, Rgb { r: 4, g: 5, b: 6 });
        assert_eq!(exact.exact_fg(3, 4), Some(rgb));
        assert_eq!(exact.exact_bg(3, 4), Some(Rgb { r: 4, g: 5, b: 6 }));
        // The last column and the last row are the unpainted margin.
        exact.set_exact_fg(191, 4, rgb);
        exact.set_exact_fg(3, 59, rgb);
        assert!(exact.exact_fg(191, 4).is_none());
        assert!(exact.exact_fg(3, 59).is_none());
        exact.clear_exact(3, 4);
        assert!(exact.exact_fg(3, 4).is_none());
        assert!(exact.exact_bg(3, 4).is_none());

        let mut copy = Grid::new(192, 60);
        copy.copy_cells_from(&plain);
        assert!(!copy.records_exact());
        let mut kept = Grid::with_exact(192, 60);
        exact.set_exact_fg(1, 1, rgb);
        kept.copy_cells_from(&exact);
        assert_eq!(kept.exact_fg(1, 1), Some(rgb));
    }
}
