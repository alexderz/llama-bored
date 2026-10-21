//! The grid source (#107): read llama-watch's grid file instead of
//! `/dev/vcsa`. The file carries the original characters and exact RGB, so
//! the cells are not limited to tty11's font. Decoding is
//! `llama_core::gridfile::decode`; this module reads the file, turns it into
//! a [`Screen`], and judges how old it is.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use llama_core::glyph;
use llama_core::gridfile::{self, GridImage, MAX_COLS, MAX_ROWS, STALE_AFTER};

use crate::screen::{Cell, Color, Screen};

/// A grid file older than this is stale (milliseconds): llama-watch
/// rewrites it at least once a second while it is running.
pub const STALE_AFTER_MS: u64 = STALE_AFTER.as_millis() as u64;

/// Open `path` afresh (the writer replaces it with a rename), read at most one
/// maximum-size grid and decode it. Gauge segments (#165) are kept; whether
/// they are drawn is [`screen_of`]'s choice. Every failure is an `io::Error` whose text
/// names the path or the reason, ready for a status line or stderr.
pub fn load_grid(path: &Path) -> io::Result<GridImage> {
    let mut file = File::open(path)
        .map_err(|err| io::Error::new(err.kind(), format!("opening {}: {err}", path.display())))?;
    let cap = u64::try_from(gridfile::file_len(MAX_COLS, MAX_ROWS)).unwrap_or(u64::MAX) + 1;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file).take(cap).read_to_end(&mut bytes)?;
    gridfile::decode_segments(&bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
}

/// Unix time now, in milliseconds.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Whole seconds since the file was written when that is over
/// [`STALE_AFTER_MS`], else `None`. A file stamped in the future (clock
/// skew) is not stale.
pub fn stale_secs(written_ms: u64, now_ms: u64) -> Option<u64> {
    let age = now_ms.saturating_sub(written_ms);
    (age > STALE_AFTER_MS).then_some(age / 1000)
}

/// The grid's cells as a screen, every colour exact. A gauge segment
/// (`llama_core::glyph::SEGMENT`, #165) stays itself with `segments`
/// (`--segments`: the terminal font has it), else it is the plain `▄`, so
/// a font without it never shows a replacement glyph.
pub fn screen_of(image: &GridImage, segments: bool) -> Screen {
    Screen {
        cols: image.cols,
        rows: image.rows,
        cursor_x: 0,
        cursor_y: 0,
        cells: image
            .cells
            .iter()
            .map(|cell| {
                Cell::new(
                    if segments {
                        cell.ch
                    } else {
                        glyph::plain(cell.ch)
                    },
                    Color::Rgb(cell.fg.r, cell.fg.g, cell.fg.b),
                    Color::Rgb(cell.bg.r, cell.bg.g, cell.bg.b),
                )
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_core::color::Rgb;
    use llama_core::gridfile::{GridCell, encode};

    fn image(ms: u64) -> GridImage {
        let cell = |ch| GridCell {
            ch,
            fg: Rgb { r: 1, g: 2, b: 3 },
            bg: Rgb { r: 4, g: 5, b: 6 },
        };
        GridImage {
            seq: 1,
            unix_ms: ms,
            cols: 2,
            rows: 1,
            cells: vec![cell('\u{e9}'), cell('\u{2192}')],
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llama-view-gridsrc-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        dir
    }

    #[test]
    fn stale_after_five_seconds_and_never_for_the_future() {
        assert_eq!(stale_secs(10_000, 10_000), None);
        assert_eq!(stale_secs(10_000, 15_000), None, "exactly 5 s is not stale");
        assert_eq!(stale_secs(10_000, 15_001), Some(5));
        assert_eq!(stale_secs(10_000, 27_900), Some(17));
        assert_eq!(stale_secs(20_000, 10_000), None, "future stamp");
    }

    #[test]
    fn screen_keeps_characters_and_exact_colours() {
        let screen = screen_of(&image(0), false);
        assert_eq!((screen.cols, screen.rows), (2, 1));
        assert_eq!(screen.cells[0].ch, '\u{e9}');
        assert_eq!(screen.cells[1].ch, '\u{2192}');
        assert_eq!(screen.cells[0].fg, Color::Rgb(1, 2, 3));
        assert_eq!(screen.cells[0].bg, Color::Rgb(4, 5, 6));
    }

    /// #165: a gauge segment survives the file and is drawn as itself only
    /// with `--segments`; otherwise it is the plain half block, never U+FFFD.
    #[test]
    fn a_segment_is_a_half_block_unless_the_font_has_it() {
        let dir = scratch("segment");
        let path = dir.join("grid");
        let mut img = image(0);
        img.cells[0].ch = glyph::SEGMENT;
        let mut bytes = Vec::new();
        encode(&img, &mut bytes).expect("encode");
        std::fs::write(&path, &bytes).expect("write");
        let loaded = load_grid(&path).expect("load");
        assert_eq!(loaded.cells[0].ch, glyph::SEGMENT);
        assert_eq!(screen_of(&loaded, false).cells[0].ch, '\u{2584}');
        assert_eq!(screen_of(&loaded, true).cells[0].ch, glyph::SEGMENT);
        assert_eq!(screen_of(&loaded, false).cells[1].ch, '\u{2192}');
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_reads_a_good_file_and_names_the_bad_ones() {
        let dir = scratch("load");
        let good = dir.join("grid");
        let mut bytes = Vec::new();
        encode(&image(42), &mut bytes).expect("encode");
        std::fs::write(&good, &bytes).expect("write");
        assert_eq!(load_grid(&good).expect("load"), image(42));

        let missing = load_grid(&dir.join("absent")).unwrap_err();
        assert_eq!(missing.kind(), io::ErrorKind::NotFound);

        let bad = dir.join("bad");
        std::fs::write(&bad, b"not a grid file at all, but long enough").expect("write");
        let err = load_grid(&bad).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("bad magic"), "{err}");

        std::fs::write(&bad, &bytes[..bytes.len() - 3]).expect("write");
        assert!(load_grid(&bad).unwrap_err().to_string().contains("bytes"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
