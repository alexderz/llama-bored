//! Cell grid. `ch` is a Unicode scalar; the sanitiser and the emitter decide
//! which scalars are legal. `rich` is the same cell for a full-Unicode
//! viewer (the grid file, #107); tty11 never reads it.

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

/// Row-major cells. `cells.len()` is always `cols * rows`.
///
/// The fields are private so a caller cannot hand the emitter a vec whose
/// length does not match the declared size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grid {
    cells: Vec<Cell>,
    cols: u16,
    rows: u16,
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
            };
        };
        Self {
            cells: vec![Cell::blank(); n],
            cols,
            rows,
        }
    }

    /// A grid of blank cells.
    pub fn blank(cols: u16, rows: u16) -> Self {
        Self::new(cols, rows)
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
