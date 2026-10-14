//! Grid frames: exact RGB, centre and crop, a missing glyph, the font pick,
//! and a bad file (stale, future, truncated, bad magic, missing). The stale
//! cases pass the clock in; they do not read `SystemTime`.

mod common;

use std::os::unix::fs::symlink;

use llama_cast::config::Font;
use llama_cast::font::Psf2;
use llama_cast::grid::{
    self, GridRead, Hold, NOTICE_GREY, Notice, compose, pick_font, write_notice,
};
use llama_cast::render::{FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH};
use llama_cast::source::{FrameSource, GridSource};
use llama_core::color::Rgb;
use llama_core::gridfile::{self, GridCell, GridImage};

fn pixel(frame: &[u8], x: usize, y: usize) -> [u8; 3] {
    let at = (y * FRAME_WIDTH + x) * 3;
    [frame[at], frame[at + 1], frame[at + 2]]
}

fn psf(count: usize, width: usize, height: usize, flags: u32, mark: u8, table: &[u8]) -> Psf2 {
    let row_bytes = width.div_ceil(8);
    let char_size = row_bytes * height;
    let mut glyphs = vec![0_u8; count * char_size];
    for glyph in 0..count {
        glyphs[glyph * char_size] = mark;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x864A_B572_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(char_size).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(height).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(width).unwrap().to_le_bytes());
    bytes.extend_from_slice(&glyphs);
    if flags & 1 == 1 {
        bytes.extend_from_slice(table);
    }
    Psf2::parse(&bytes).unwrap()
}

/// 8x8, no Unicode table: a scalar below 128 is its own glyph, and pixel
/// (0, 0) of every glyph is set.
fn identity() -> Psf2 {
    psf(128, 8, 8, 0, 0x80, &[])
}

/// 8x8. Glyph 0 is 'A' with the top row lit. Glyph 1 is U+FFFD with only
/// pixel (7, 7) lit.
fn letter_and_replacement() -> Psf2 {
    // 8x8, two glyphs: 'A' has its top row lit, U+FFFD has only pixel (7, 7).
    let mut glyphs = vec![0_u8; 2 * 8];
    glyphs[0] = 0xFF;
    glyphs[8 + 7] = 0x01;
    let table = [
        0x41, 0xFF, // A
        0xEF, 0xBF, 0xBD, 0xFF, // U+FFFD
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x864A_B572_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&glyphs);
    bytes.extend_from_slice(&table);
    Psf2::parse(&bytes).unwrap()
}

fn cell(ch: char, fg: [u8; 3], bg: [u8; 3]) -> GridCell {
    GridCell {
        ch,
        fg: Rgb {
            r: fg[0],
            g: fg[1],
            b: fg[2],
        },
        bg: Rgb {
            r: bg[0],
            g: bg[1],
            b: bg[2],
        },
    }
}

fn file(cols: u16, rows: u16, unix_ms: u64, cells: Vec<GridCell>) -> Vec<u8> {
    let image = GridImage {
        seq: 1,
        unix_ms,
        cols,
        rows,
        cells,
    };
    let mut out = Vec::new();
    gridfile::encode(&image, &mut out).unwrap();
    out
}

fn paint(bytes: &[u8], font: &Psf2, now_ms: u64, hold: &mut Hold) -> Vec<u8> {
    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(bytes),
        now_ms,
        std::slice::from_ref(font),
        false,
        hold,
        &mut frame,
    )
    .unwrap();
    frame
}

const RED: [u8; 3] = [0x14, 0x28, 0xD8];
const BLUE: [u8; 3] = [0x01, 0x02, 0x03];
const GREEN: [u8; 3] = [0x10, 0xB0, 0x20];

#[test]
fn a_cells_foreground_and_background_are_the_bytes_in_the_file() {
    let font = letter_and_replacement();
    let bytes = file(1, 1, 1_000, vec![cell('A', RED, BLUE)]);
    let frame = paint(&bytes, &font, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 8) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(
        pixel(&frame, x, y),
        RED,
        "a lit glyph pixel is the foreground"
    );
    assert_eq!(
        pixel(&frame, x, y + 1),
        BLUE,
        "an unset pixel is the background"
    );
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0], "outside the cell is black");
    assert_eq!(pixel(&frame, x + 7, y), RED);
}

#[test]
fn a_small_grid_is_centred_and_a_large_one_is_cropped_from_the_top_left() {
    let font = letter_and_replacement();
    let small = file(1, 1, 1_000, vec![cell('A', RED, BLUE)]);
    let frame = paint(&small, &font, 1_000, &mut Hold::new());
    let (x, y) = ((FRAME_WIDTH - 8) / 2, (FRAME_HEIGHT - 8) / 2);
    assert_eq!((x, y), (956, 536));
    assert_eq!(pixel(&frame, x, y), RED);
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0]);

    // Wider than the frame, one row tall: cropped on x, centred on y.
    let cols = 241_u16;
    let mut cells = vec![cell('A', BLUE, BLUE); usize::from(cols)];
    cells[0] = cell('A', RED, BLUE);
    let wide = file(cols, 1, 1_000, cells);
    let frame = paint(&wide, &font, 1_000, &mut Hold::new());
    let row_y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(pixel(&frame, 0, row_y), RED, "a wide grid starts at x = 0");
    assert_eq!(pixel(&frame, 8, row_y), BLUE);
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0]);

    // Taller than the frame, one column: cropped on y, centred on x.
    let rows = 136_u16;
    let mut cells = vec![cell('A', BLUE, BLUE); usize::from(rows)];
    cells[0] = cell('A', RED, BLUE);
    let tall = file(1, rows, 1_000, cells);
    let frame = paint(&tall, &font, 1_000, &mut Hold::new());
    let col_x = (FRAME_WIDTH - 8) / 2;
    assert_eq!(pixel(&frame, col_x, 0), RED, "a tall grid starts at y = 0");
    assert_eq!(pixel(&frame, col_x, 8), BLUE);
}

#[test]
fn a_missing_glyph_is_the_replacement_or_a_blank_cell() {
    let font = letter_and_replacement();
    let bytes = file(
        2,
        1,
        1_000,
        vec![cell('B', RED, BLUE), cell('\u{1F600}', GREEN, BLUE)],
    );
    let frame = paint(&bytes, &font, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 16) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(pixel(&frame, x, y), BLUE, "B is not A's top row");
    assert_eq!(pixel(&frame, x + 7, y + 7), RED, "B uses U+FFFD");
    assert_eq!(
        pixel(&frame, x + 8, y),
        BLUE,
        "an unshowable scalar is U+FFFD too"
    );
    assert_eq!(pixel(&frame, x + 8 + 7, y + 7), GREEN);

    let bare = identity();
    // Showable, and past this font's glyph count, and the font has no U+FFFD.
    let bytes = file(1, 1, 1_000, vec![cell('\u{2192}', RED, BLUE)]);
    let frame = paint(&bytes, &bare, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 8) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(
        pixel(&frame, x, y),
        BLUE,
        "no U+FFFD glyph, so the cell is background"
    );
    assert_eq!(pixel(&frame, x + 7, y + 7), BLUE);
}

#[test]
fn the_font_pick_fits_the_example_grids_and_crops_an_oversize_one() {
    let fonts = &Font::SHIPPED;
    assert_eq!(pick_font(fonts, 192, 60), Font::Hack10x18);
    assert_eq!(pick_font(fonts, 213, 60), Font::Hack9x18);
    assert_eq!(pick_font(fonts, 160, 45), Font::Hack12x24);
    assert_eq!(pick_font(fonts, 214, 60), Font::Hack9x18);
    assert_eq!(pick_font(fonts, 1024, 512), Font::Hack9x18);

    let sized = [
        psf(1, 12, 24, 0, 0, &[]),
        psf(1, 12, 22, 0, 0, &[]),
        psf(1, 10, 18, 0, 0, &[]),
        psf(1, 9, 18, 0, 0, &[]),
    ];
    let bytes = file(213, 2, 1_000, vec![cell('A', RED, BLUE); 426]);
    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(&bytes),
        1_000,
        &sized,
        true,
        &mut Hold::new(),
        &mut frame,
    )
    .unwrap();
    // 213 * 9 = 1917 and 2 * 18 = 36, so the 9x18 font is centred. A 10-wide
    // font would be cropped and start at x = 0.
    let y = (FRAME_HEIGHT - 2 * 18) / 2;
    assert_eq!(pixel(&frame, 0, y), [0, 0, 0]);
    assert_eq!(pixel(&frame, 1, y), BLUE);
}

#[test]
fn the_notice_names_the_age_and_a_future_stamp_is_fresh() {
    let mut buf = [0_u8; 64];
    let n = write_notice(Notice::Stale(17), &mut buf);
    assert_eq!(&buf[..n], b"waiting for grid - no update for 17 s");
    assert_eq!(grid::stale_secs(10_000, 27_900), Some(17));
    assert_eq!(
        grid::stale_secs(10_000, 15_000),
        None,
        "exactly 5 s is fresh"
    );
    assert_eq!(
        grid::stale_secs(50_000, 1_000),
        None,
        "a future stamp is fresh"
    );

    let font = identity();
    let fresh = file(2, 2, 10_000, vec![cell('A', RED, BLUE); 4]);
    let mut hold = Hold::new();
    let frame = paint(&fresh, &font, 10_000, &mut hold);
    let x = (FRAME_WIDTH - 16) / 2;
    let y = (FRAME_HEIGHT - 16) / 2;
    assert_eq!(pixel(&frame, x, y), RED);

    let later = file(2, 2, 10_000, vec![cell('A', GREEN, BLUE); 4]);
    let frame = paint(&later, &font, 15_000, &mut hold);
    assert_eq!(
        pixel(&frame, x, y),
        GREEN,
        "five seconds is still the new frame"
    );

    let frame = paint(&later, &font, 27_900, &mut hold);
    assert_eq!(
        pixel(&frame, x, y),
        GREEN,
        "a stale file does not replace the picture"
    );
    assert_eq!(
        pixel(&frame, x, y + 8),
        NOTICE_GREY,
        "the last row carries the notice"
    );
    assert_eq!(
        pixel(&frame, x + 1, y + 8),
        [0, 0, 0],
        "the rest of that row is cleared"
    );

    let ahead = file(2, 2, 50_000, vec![cell('A', RED, BLUE); 4]);
    let frame = paint(&ahead, &font, 1_000, &mut hold);
    assert_eq!(pixel(&frame, x, y), RED, "a future stamp is drawn");
    assert_ne!(pixel(&frame, x, y + 8), NOTICE_GREY);
}

#[test]
fn a_truncated_or_bad_file_is_a_notice_and_keeps_a_good_frame() {
    let font = identity();
    let good = file(2, 2, 1_000, vec![cell('A', RED, BLUE); 4]);
    let mut hold = Hold::new();
    let _ = paint(&good, &font, 1_000, &mut hold);
    let x = (FRAME_WIDTH - 16) / 2;
    let y = (FRAME_HEIGHT - 16) / 2;

    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(&[0_u8; 10]),
        2_000,
        std::slice::from_ref(&font),
        false,
        &mut hold,
        &mut frame,
    )
    .unwrap();
    assert_eq!(pixel(&frame, x, y), RED);
    assert_eq!(pixel(&frame, x, y + 8), NOTICE_GREY);

    let bad = vec![0_u8; 24];
    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(&bad),
        2_000,
        std::slice::from_ref(&font),
        false,
        &mut hold,
        &mut frame,
    )
    .unwrap();
    assert_eq!(
        pixel(&frame, x, y),
        RED,
        "bad magic keeps the last good frame"
    );

    let mut bare = Hold::new();
    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(&bad),
        2_000,
        std::slice::from_ref(&font),
        false,
        &mut bare,
        &mut frame,
    )
    .unwrap();
    let text = b"waiting for grid - refused";
    let nx = (FRAME_WIDTH - text.len() * 8) / 2;
    let ny = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0]);
    assert_eq!(pixel(&frame, nx, ny), NOTICE_GREY);
    assert_eq!((nx, ny), (856, 536));
}

#[test]
fn a_missing_file_and_a_symlink_do_not_end_the_stream() {
    let dir = common::scratch("grid-missing");
    let font = identity();
    let missing = GridSource::new(dir.join("absent"), vec![font.clone()], false);
    let mut frame = vec![0_u8; FRAME_BYTES];
    missing.frame(&mut frame).unwrap();
    let text = b"waiting for grid - missing";
    let x = (FRAME_WIDTH - text.len() * 8) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(pixel(&frame, x, y), NOTICE_GREY);
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0]);

    let real = dir.join("grid");
    std::fs::write(&real, file(1, 1, 1_000, vec![cell('A', RED, BLUE)])).unwrap();
    let link = dir.join("via-link");
    symlink(&real, &link).unwrap();
    let linked = GridSource::new(link, vec![font], false);
    let mut frame = vec![0_u8; FRAME_BYTES];
    linked.frame(&mut frame).unwrap();
    assert_eq!(pixel(&frame, 0, 0), [0, 0, 0], "a symlink is not followed");
    let text = b"waiting for grid - unreadable";
    let x = (FRAME_WIDTH - text.len() * 8) / 2;
    assert_eq!(pixel(&frame, x, y), NOTICE_GREY);
}
