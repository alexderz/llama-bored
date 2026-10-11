//! The grid file (#107): llama-watch's dashboard as full-Unicode cells with
//! exact RGB, for llama-view on a console that is not limited to tty11's
//! 512-glyph font. One encoder and one decoder, shared by both sides.
//!
//! Little-endian, no padding:
//!
//! | bytes | field |
//! |-------|-------|
//! | 8     | magic `LBGRID1\n` |
//! | 2     | columns (`u16`) |
//! | 2     | rows (`u16`) |
//! | 4     | sequence (`u32`), up by one per write, wrapping |
//! | 8     | unix time of the write in milliseconds (`u64`) |
//! | 10 each, row-major, `columns * rows` of them | scalar (`u32`), foreground r g b, background r g b |
//!
//! The length must be exactly `24 + 10 * columns * rows`. The reader treats
//! the file as untrusted (it is group-readable and a tmpfs file, but llama-view
//! draws it on a terminal): [`decode`] rejects a bad magic, a size over the
//! caps or a length that does not match, and maps every scalar that is
//! invalid or not [`is_showable`] to U+FFFD, so no control or escape byte can
//! reach the viewer's terminal from here.

use std::time::Duration;

use crate::color::Rgb;
use crate::glyph::is_showable;

/// First eight bytes of every grid file.
pub const MAGIC: &[u8; 8] = b"LBGRID1\n";
/// Bytes before the first cell.
pub const HEADER_LEN: usize = 24;
/// Bytes per cell.
pub const CELL_LEN: usize = 10;
/// Widest grid, columns.
pub const MAX_COLS: u16 = 1024;
/// Tallest grid, rows.
pub const MAX_ROWS: u16 = 512;
/// The writer rewrites an unchanged grid at least this often, so a reader can
/// tell a live writer from a file nobody updates.
pub const HEARTBEAT: Duration = Duration::from_secs(1);
/// A reader calls a file this old stale: no heartbeat for several periods.
pub const STALE_AFTER: Duration = Duration::from_secs(5);
/// What a scalar that cannot be shown becomes.
pub const REPLACEMENT: char = '\u{fffd}';

/// One cell: what to draw and in which colours.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridCell {
    pub ch: char,
    pub fg: Rgb,
    pub bg: Rgb,
}

/// A decoded (or about to be encoded) grid file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GridImage {
    pub seq: u32,
    /// Unix milliseconds when the writer wrote the file.
    pub unix_ms: u64,
    pub cols: u16,
    pub rows: u16,
    /// Row-major, `cols * rows` long.
    pub cells: Vec<GridCell>,
}

/// Why a grid file was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridError {
    /// Shorter than the header.
    Truncated,
    BadMagic,
    /// Zero or over [`MAX_COLS`] x [`MAX_ROWS`].
    BadSize {
        cols: u16,
        rows: u16,
    },
    /// The length does not match the header's size.
    BadLength {
        want: usize,
        got: usize,
    },
    /// [`encode`] only: `cells` is not `cols * rows` long.
    CellCount {
        want: usize,
        got: usize,
    },
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("shorter than the grid header"),
            Self::BadMagic => f.write_str("not a grid file (bad magic)"),
            Self::BadSize { cols, rows } => write!(f, "grid size {cols}x{rows} is out of range"),
            Self::BadLength { want, got } => {
                write!(f, "grid file is {got} bytes, its header says {want}")
            }
            Self::CellCount { want, got } => write!(f, "grid has {got} cells, not {want}"),
        }
    }
}

impl std::error::Error for GridError {}

fn size_ok(cols: u16, rows: u16) -> bool {
    (1..=MAX_COLS).contains(&cols) && (1..=MAX_ROWS).contains(&rows)
}

/// Exact file length for a `cols` x `rows` grid.
#[must_use]
pub fn file_len(cols: u16, rows: u16) -> usize {
    HEADER_LEN + CELL_LEN * usize::from(cols) * usize::from(rows)
}

/// Replace `out` with the file for `image`. Refuses a size over the caps or a
/// cell count that does not match; a scalar that is not [`is_showable`] is
/// written as [`REPLACEMENT`], the same as the decoder would read it.
pub fn encode(image: &GridImage, out: &mut Vec<u8>) -> Result<(), GridError> {
    let (cols, rows) = (image.cols, image.rows);
    if !size_ok(cols, rows) {
        return Err(GridError::BadSize { cols, rows });
    }
    let want = usize::from(cols) * usize::from(rows);
    if image.cells.len() != want {
        return Err(GridError::CellCount {
            want,
            got: image.cells.len(),
        });
    }
    out.clear();
    out.reserve(file_len(cols, rows));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&cols.to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&image.seq.to_le_bytes());
    out.extend_from_slice(&image.unix_ms.to_le_bytes());
    for cell in &image.cells {
        let ch = if is_showable(cell.ch) {
            cell.ch
        } else {
            REPLACEMENT
        };
        out.extend_from_slice(&u32::from(ch).to_le_bytes());
        out.extend_from_slice(&[
            cell.fg.r, cell.fg.g, cell.fg.b, cell.bg.r, cell.bg.g, cell.bg.b,
        ]);
    }
    Ok(())
}

/// Read a grid file. Every check is on the bytes, nothing is trusted.
pub fn decode(bytes: &[u8]) -> Result<GridImage, GridError> {
    let header = bytes.get(..HEADER_LEN).ok_or(GridError::Truncated)?;
    if &header[..8] != MAGIC {
        return Err(GridError::BadMagic);
    }
    let cols = u16::from_le_bytes([header[8], header[9]]);
    let rows = u16::from_le_bytes([header[10], header[11]]);
    if !size_ok(cols, rows) {
        return Err(GridError::BadSize { cols, rows });
    }
    let seq = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let unix_ms = u64::from_le_bytes([
        header[16], header[17], header[18], header[19], header[20], header[21], header[22],
        header[23],
    ]);
    let want = file_len(cols, rows);
    if bytes.len() != want {
        return Err(GridError::BadLength {
            want,
            got: bytes.len(),
        });
    }
    let cells = bytes[HEADER_LEN..]
        .as_chunks::<CELL_LEN>()
        .0
        .iter()
        .map(|raw| {
            let scalar = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            let ch = char::from_u32(scalar)
                .filter(|&c| is_showable(c))
                .unwrap_or(REPLACEMENT);
            GridCell {
                ch,
                fg: Rgb {
                    r: raw[4],
                    g: raw[5],
                    b: raw[6],
                },
                bg: Rgb {
                    r: raw[7],
                    g: raw[8],
                    b: raw[9],
                },
            }
        })
        .collect();
    Ok(GridImage {
        seq,
        unix_ms,
        cols,
        rows,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(r: u8, g: u8, b: u8) -> Rgb {
        Rgb { r, g, b }
    }

    fn image(cols: u16, rows: u16, ch: char) -> GridImage {
        let n = usize::from(cols) * usize::from(rows);
        GridImage {
            seq: 7,
            unix_ms: 1_700_000_000_123,
            cols,
            rows,
            cells: (0..n)
                .map(|i| GridCell {
                    ch: if i == 0 { ch } else { 'x' },
                    fg: rgb(1, 2, i as u8),
                    bg: rgb(4, 5, 6),
                })
                .collect(),
        }
    }

    fn bytes(image: &GridImage) -> Vec<u8> {
        let mut out = Vec::new();
        encode(image, &mut out).expect("encode");
        out
    }

    #[test]
    fn round_trips_with_the_documented_layout() {
        let img = image(3, 2, '\u{e9}');
        let raw = bytes(&img);
        assert_eq!(raw.len(), 24 + 10 * 6);
        assert_eq!(&raw[..8], b"LBGRID1\n");
        assert_eq!(&raw[8..12], &[3, 0, 2, 0]);
        assert_eq!(&raw[12..16], &7u32.to_le_bytes());
        assert_eq!(&raw[16..24], &1_700_000_000_123u64.to_le_bytes());
        assert_eq!(&raw[24..28], &0xe9u32.to_le_bytes());
        assert_eq!(&raw[28..34], &[1, 2, 0, 4, 5, 6]);
        assert_eq!(decode(&raw), Ok(img));
    }

    #[test]
    fn the_caps_are_enforced_both_ways() {
        let mut out = Vec::new();
        assert_eq!(
            encode(&image(1025, 1, 'a'), &mut out),
            Err(GridError::BadSize {
                cols: 1025,
                rows: 1
            })
        );
        assert!(encode(&image(1, 513, 'a'), &mut out).is_err());
        assert!(encode(&image(0, 1, 'a'), &mut out).is_err());
        assert!(encode(&image(1024, 512, 'a'), &mut out).is_ok());
        assert_eq!(out.len(), 24 + 10 * 1024 * 512);
        assert!(decode(&out).is_ok());
        for (cols, rows) in [
            (1025u16, 1u16),
            (1, 513),
            (0, 4),
            (4, 0),
            (u16::MAX, u16::MAX),
        ] {
            let mut raw = bytes(&image(2, 2, 'a'));
            raw[8..10].copy_from_slice(&cols.to_le_bytes());
            raw[10..12].copy_from_slice(&rows.to_le_bytes());
            assert_eq!(decode(&raw), Err(GridError::BadSize { cols, rows }));
        }
    }

    #[test]
    fn a_wrong_cell_count_is_refused_by_the_encoder() {
        let mut img = image(2, 2, 'a');
        img.cells.pop();
        let mut out = Vec::new();
        assert_eq!(
            encode(&img, &mut out),
            Err(GridError::CellCount { want: 4, got: 3 })
        );
    }

    #[test]
    fn bad_magic_and_truncation_are_refused() {
        let good = bytes(&image(2, 2, 'a'));
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(decode(&bad), Err(GridError::BadMagic));
        assert_eq!(decode(&[]), Err(GridError::Truncated));
        assert_eq!(decode(&good[..23]), Err(GridError::Truncated));
        assert_eq!(
            decode(&good[..good.len() - 1]),
            Err(GridError::BadLength { want: 64, got: 63 })
        );
        let mut long = good.clone();
        long.push(0);
        assert_eq!(
            decode(&long),
            Err(GridError::BadLength { want: 64, got: 65 })
        );
        assert_eq!(
            decode(&good[..24]),
            Err(GridError::BadLength { want: 64, got: 24 })
        );
    }

    #[test]
    fn hostile_scalars_become_the_replacement_character() {
        let mut raw = bytes(&image(4, 4, 'a'));
        let hostile = [
            0xd800u32,
            0xdfff,
            0x11_0000,
            0xffff_ffff,
            0x1b,
            0x9b,
            0x7f,
            0x0a,
            0x00,
            0x200d,
            0x202e,
            0x2066,
            0xfeff,
            0x4e2d,
            0x1f600,
            0x0301,
        ];
        for (i, scalar) in hostile.iter().enumerate() {
            let at = 24 + i * 10;
            raw[at..at + 4].copy_from_slice(&scalar.to_le_bytes());
        }
        let img = decode(&raw).expect("decode");
        for (i, cell) in img.cells.iter().enumerate() {
            assert_eq!(cell.ch, REPLACEMENT, "cell {i} ({:#x})", hostile[i]);
        }
    }

    #[test]
    fn the_encoder_writes_the_same_replacement() {
        let img = image(2, 1, '\u{1b}');
        let back = decode(&bytes(&img)).expect("decode");
        assert_eq!(back.cells[0].ch, REPLACEMENT);
        assert_eq!(back.cells[1].ch, 'x');
    }

    #[test]
    fn showable_scalars_survive() {
        for ch in [
            '\u{e9}', '\u{2192}', '\u{d7}', '\u{44f}', '\u{3b1}', ' ', '\u{2588}',
        ] {
            let back = decode(&bytes(&image(1, 1, ch))).expect("decode");
            assert_eq!(back.cells[0].ch, ch);
        }
    }
}
