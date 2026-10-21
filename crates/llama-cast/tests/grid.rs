//! Grid frames: exact RGB, centre and crop, a missing glyph, the font pick,
//! and a bad file (stale, future, truncated, bad magic, missing). The stale
//! cases pass the clock in; they do not read `SystemTime`, except the one
//! test that checks `GridSource` uses the real clock.

mod common;

use std::os::unix::fs::symlink;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// 8x8 with a full block `█` (glyph 0, every pixel lit) and `▄` (glyph 1,
/// rows 4-7 lit), and no gauge segment: a font from before #165.
fn blocks_without_segment() -> Psf2 {
    let mut glyphs = vec![0_u8; 2 * 8];
    glyphs[..8].fill(0xFF);
    glyphs[8 + 4..16].fill(0xFF);
    let table = [
        0xE2, 0x96, 0x88, 0xFF, // U+2588
        0xE2, 0x96, 0x84, 0xFF, // U+2584
    ];
    let mut bytes = Vec::new();
    for word in [0x864A_B572_u32, 0, 32, 1, 2, 8, 8, 8] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&glyphs);
    bytes.extend_from_slice(&table);
    Psf2::parse(&bytes).unwrap()
}

/// #165: a gauge segment is the lower half one pixel column short. With a
/// font that predates it, the cast draws the font's `▄` and blanks the last
/// column itself, so the hairline shows and the cell is never U+FFFD. With
/// the shipped 9x18 font, the font's own segment glyph draws.
#[test]
fn a_gauge_segment_draws_a_half_block_with_a_one_pixel_gap() {
    let old = blocks_without_segment();
    let seg = llama_core::glyph::SEGMENT;
    let bytes = file(
        2,
        1,
        1_000,
        vec![cell(seg, RED, BLUE), cell(seg, GREEN, BLUE)],
    );
    let frame = paint(&bytes, &old, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 16) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    for gx in 0..8 {
        assert_eq!(pixel(&frame, x + gx, y + 3), BLUE, "upper half at {gx}");
        let want = if gx == 7 { BLUE } else { RED };
        assert_eq!(pixel(&frame, x + gx, y + 4), want, "lower half at {gx}");
        let want = if gx == 7 { BLUE } else { GREEN };
        assert_eq!(
            pixel(&frame, x + 8 + gx, y + 7),
            want,
            "second cell at {gx}"
        );
    }
    // A plain `▄` keeps its last column.
    let plain = file(1, 1, 1_000, vec![cell('\u{2584}', RED, BLUE)]);
    let frame = paint(&plain, &old, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 8) / 2;
    assert_eq!(pixel(&frame, x + 7, y + 7), RED);

    let path = common::workspace_root().join("packaging/fonts/llama-hack-9x18.psfu");
    let shipped = Psf2::parse(&std::fs::read(path).unwrap()).unwrap();
    let bytes = file(1, 1, 1_000, vec![cell(seg, RED, BLUE)]);
    let frame = paint(&bytes, &shipped, 1_000, &mut Hold::new());
    let (x, y) = ((FRAME_WIDTH - 9) / 2, (FRAME_HEIGHT - 18) / 2);
    for gx in 0..9 {
        let want = if gx == 8 { BLUE } else { RED };
        assert_eq!(
            pixel(&frame, x + gx, y + 17),
            want,
            "9x18 bottom row at {gx}"
        );
        assert_eq!(pixel(&frame, x + gx, y), BLUE, "9x18 top row at {gx}");
    }
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

fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

fn notice_origin(text: &[u8]) -> (usize, usize) {
    ((FRAME_WIDTH - text.len() * 8) / 2, (FRAME_HEIGHT - 8) / 2)
}

/// Both calls must return. A blocking open keeps the worker thread, and the
/// timeout fails the test; the process exit ends that thread.
fn two_frames(source: GridSource) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut first = vec![0_u8; FRAME_BYTES];
        let mut second = vec![0_u8; FRAME_BYTES];
        let once = source.frame(&mut first);
        let again = source.frame(&mut second);
        let _ = tx.send((once, again, first));
    });
    let (once, again, frame) = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("a fifo blocked the stream");
    once.expect("frame");
    again.expect("the stream goes on");
    frame
}

#[test]
fn a_fifo_at_the_grid_path_returns_at_once_with_the_unreadable_notice() {
    let dir = common::scratch("grid-fifo");
    let path = dir.join("grid");
    let made = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo");
    assert!(made.success(), "mkfifo");
    let font = identity();
    let frame = two_frames(GridSource::new(path, vec![font.clone()], false));
    // "unreadable" is longer than "refused", so its first pixel sits further left.
    let (ux, y) = notice_origin(b"waiting for grid - unreadable");
    let (rx, _) = notice_origin(b"waiting for grid - refused");
    assert_eq!((ux, y, rx), (844, 536, 856));
    assert_eq!(
        pixel(&frame, ux, y),
        NOTICE_GREY,
        "a fifo is the unreadable notice"
    );
    assert_eq!(pixel(&frame, rx, y), [0, 0, 0]);

    // `/dev/null` reads as empty. Without the fstat, that is "refused".
    let frame = two_frames(GridSource::new("/dev/null".into(), vec![font], false));
    assert_eq!(
        pixel(&frame, ux, y),
        NOTICE_GREY,
        "a device is not a regular file"
    );
    assert_eq!(pixel(&frame, rx, y), [0, 0, 0]);
}

#[test]
fn grid_cap_reads_a_file_at_the_cap_and_refuses_one_byte_over() {
    let cap = gridfile::file_len(gridfile::MAX_COLS, gridfile::MAX_ROWS);
    assert_eq!(llama_cast::source::grid_cap(), cap);
    let dir = common::scratch("grid-cap");
    let at = dir.join("at");
    let over = dir.join("over");
    std::fs::File::create(&at)
        .unwrap()
        .set_len(u64::try_from(cap).unwrap())
        .unwrap();
    std::fs::File::create(&over)
        .unwrap()
        .set_len(u64::try_from(cap + 1).unwrap())
        .unwrap();
    let font = identity();
    let (rx, y) = notice_origin(b"waiting for grid - refused");
    let (ux, _) = notice_origin(b"waiting for grid - unreadable");
    let mut frame = vec![0_u8; FRAME_BYTES];
    GridSource::new(at, vec![font.clone()], false)
        .frame(&mut frame)
        .unwrap();
    assert_eq!(
        pixel(&frame, rx, y),
        NOTICE_GREY,
        "a file at the cap is read"
    );
    assert_eq!(pixel(&frame, ux, y), [0, 0, 0]);
    GridSource::new(over, vec![font], false)
        .frame(&mut frame)
        .unwrap();
    assert_eq!(
        pixel(&frame, ux, y),
        NOTICE_GREY,
        "one byte over the cap is refused"
    );
    assert_eq!(pixel(&frame, rx, y), [0, 0, 0]);
}

fn agrees(bytes: &[u8]) {
    match (grid::open_bytes(bytes), gridfile::decode(bytes)) {
        (Ok(view), Ok(image)) => {
            assert_eq!(view.cols, image.cols);
            assert_eq!(view.rows, image.rows);
            assert_eq!(view.unix_ms, image.unix_ms);
        }
        (Err(left), Err(right)) => assert_eq!(left, right, "{bytes:?}"),
        (Ok(view), Err(err)) => panic!("open_bytes accepted {}x{}: {err}", view.cols, view.rows),
        (Err(err), Ok(image)) => {
            panic!(
                "open_bytes refused ({err}) a {}x{} grid",
                image.cols, image.rows
            );
        }
    }
}

fn header(cols: u16, rows: u16) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(gridfile::MAGIC);
    out.extend_from_slice(&cols.to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&1_u32.to_le_bytes());
    out.extend_from_slice(&1_000_u64.to_le_bytes());
    out
}

#[test]
fn open_bytes_accepts_exactly_when_gridfile_decode_does() {
    let valid = file(2, 2, 1_700, vec![cell('A', RED, BLUE); 4]);
    for n in 0..=valid.len() {
        agrees(&valid[..n]);
    }
    let mut magic = valid.clone();
    magic[0] = b'X';
    agrees(&magic);
    agrees(&header(0, 1));
    agrees(&header(1, 0));
    agrees(&header(0, 0));
    agrees(&header(1025, 1));
    agrees(&header(1, 513));
    agrees(&[0]);
    let mut short = valid.clone();
    short.pop();
    agrees(&short);
    let mut trailing = valid.clone();
    trailing.push(0);
    agrees(&trailing);
    // The header says 3x1. The body is the four cells of the 2x2 file.
    let mut lies = header(3, 1);
    lies.extend_from_slice(&valid[gridfile::HEADER_LEN..]);
    agrees(&lies);
}

#[test]
fn an_age_of_5000_ms_is_fresh_and_5001_ms_is_stale() {
    assert_eq!(grid::stale_secs(10_000, 15_000), None);
    assert_eq!(grid::stale_secs(10_000, 15_001), Some(5));
    let font = identity();
    let bytes = file(1, 1, 10_000, vec![cell('A', RED, BLUE)]);
    let frame = paint(&bytes, &font, 15_000, &mut Hold::new());
    assert_eq!(pixel(&frame, 956, 536), RED, "5000 ms is the cell");
    let frame = paint(&bytes, &font, 15_001, &mut Hold::new());
    let mut buf = [0_u8; 64];
    let n = write_notice(Notice::Stale(5), &mut buf);
    let text = &buf[..n];
    assert_eq!(text, b"waiting for grid - no update for 5 s");
    let (x0, y) = notice_origin(text);
    let hyphen = text.iter().position(|byte| *byte == b'-').unwrap();
    assert_eq!(pixel(&frame, x0 + hyphen * 8, y), NOTICE_GREY);
    assert_ne!(pixel(&frame, 956, 536), RED, "5001 ms is not the cell");
}

#[test]
fn a_fresh_grid_file_through_grid_source_uses_the_real_clock() {
    let dir = common::scratch("grid-clock");
    let path = dir.join("grid");
    let font = identity();
    let now = unix_now_ms();
    std::fs::write(&path, file(1, 1, now, vec![cell('A', RED, BLUE)])).unwrap();
    let mut frame = vec![0_u8; FRAME_BYTES];
    GridSource::new(path.clone(), vec![font.clone()], false)
        .frame(&mut frame)
        .unwrap();
    assert_eq!(
        pixel(&frame, 956, 536),
        RED,
        "a file stamped now has no notice"
    );
    std::fs::write(
        &path,
        file(1, 1, now.saturating_sub(10_000), vec![cell('A', RED, BLUE)]),
    )
    .unwrap();
    GridSource::new(path, vec![font], false)
        .frame(&mut frame)
        .unwrap();
    assert_ne!(
        pixel(&frame, 956, 536),
        RED,
        "a file stamped ten seconds ago is the notice"
    );
}

#[test]
fn a_smaller_frame_clears_the_pixels_a_larger_grid_painted() {
    let fonts = vec![
        psf(128, 12, 24, 0, 0, &[]),
        psf(128, 12, 22, 0, 0, &[]),
        psf(128, 10, 18, 0, 0, &[]),
        psf(128, 9, 18, 0, 0, &[]),
    ];
    let dir = common::scratch("grid-resize");
    let path = dir.join("grid");
    let now = unix_now_ms();
    let big = file(213, 60, now, vec![cell('A', RED, RED); 213 * 60]);
    std::fs::write(&path, &big).unwrap();
    let source = GridSource::new(path.clone(), fonts, true);
    let mut frame = vec![0_u8; FRAME_BYTES];
    source.frame(&mut frame).unwrap();
    // Inside the 213x60 picture (origin x = 1) and outside the centred 10x3.
    assert_eq!(pixel(&frame, 100, 100), RED);
    assert_eq!(
        pixel(&frame, 0, 0),
        [0, 0, 0],
        "the margin is not the sample"
    );
    let small = file(10, 3, now, vec![cell('A', BLUE, BLUE); 30]);
    std::fs::write(&path, small).unwrap();
    source.frame(&mut frame).unwrap();
    assert_eq!(pixel(&frame, 100, 100), [0, 0, 0]);
}

#[test]
fn the_notice_uses_the_font_index_of_the_last_good_frame() {
    // 12x24 does not fit 213x60, so the held index is the 9x18. Its top-left
    // pixel is lit; the 12x24's is not. Forcing the index to 0 leaves (1, 1062) dark.
    let fonts = vec![
        psf(128, 12, 24, 0, 0x00, &[]),
        psf(128, 9, 18, 0, 0x80, &[]),
    ];
    let dir = common::scratch("grid-font-index");
    let path = dir.join("grid");
    let now = unix_now_ms();
    std::fs::write(
        &path,
        file(213, 60, now, vec![cell('A', RED, BLUE); 213 * 60]),
    )
    .unwrap();
    let source = GridSource::new(path.clone(), fonts, true);
    let mut frame = vec![0_u8; FRAME_BYTES];
    source.frame(&mut frame).unwrap();
    std::fs::remove_file(&path).unwrap();
    source.frame(&mut frame).unwrap();
    assert_eq!(pixel(&frame, 1, 1062), NOTICE_GREY);
}

#[test]
fn compose_breaks_ties_toward_the_earlier_font_and_accepts_an_exact_fit() {
    let earlier = psf(128, 8, 8, 0, 0x80, &[]);
    let later = psf(128, 16, 4, 0, 0x00, &[]);
    let bytes = file(1, 1, 1_000, vec![cell('A', RED, BLUE)]);
    let mut frame = vec![0_u8; FRAME_BYTES];
    compose(
        GridRead::Data(&bytes),
        1_000,
        &[earlier, later],
        true,
        &mut Hold::new(),
        &mut frame,
    )
    .unwrap();
    // 8x8 is centred at (956, 536). The 16x4 is centred at (952, 538), so
    // this pixel is above that cell.
    assert_eq!(pixel(&frame, 956, 536), RED);

    let exact = psf(128, 8, 8, 0, 0, &[]);
    let smaller = psf(128, 8, 4, 0, 0, &[]);
    // 135 * 8 = 1080, so the 8px font fits exactly. A strict `<` rejects it
    // and the 4px font, centred lower, leaves y = 0 black.
    let tall = file(1, 135, 1_000, vec![cell('A', RED, BLUE); 135]);
    compose(
        GridRead::Data(&tall),
        1_000,
        &[exact, smaller],
        true,
        &mut Hold::new(),
        &mut frame,
    )
    .unwrap();
    let x = (FRAME_WIDTH - 8) / 2;
    assert_eq!(pixel(&frame, x, 0), BLUE);
}

#[test]
fn the_refused_notice_does_not_say_missing() {
    let mut refused = [0_u8; 64];
    let mut missing = [0_u8; 64];
    let rn = write_notice(Notice::Refused, &mut refused);
    let mn = write_notice(Notice::Missing, &mut missing);
    assert_eq!(&refused[..rn], b"waiting for grid - refused");
    assert_eq!(&missing[..mn], b"waiting for grid - missing");
    assert_ne!(&refused[..rn], &missing[..mn]);
}

#[test]
fn an_unshowable_scalar_is_not_drawn_when_the_font_maps_it() {
    let mut glyphs = vec![0_u8; 16];
    glyphs[0] = 0xFF;
    glyphs[8 + 7] = 0x01;
    let table = [
        0x1B, 0xFF, // U+001B, not showable
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
    let font = Psf2::parse(&bytes).unwrap();

    let mut raw = header(1, 1);
    raw.extend_from_slice(&0x1B_u32.to_le_bytes());
    raw.extend_from_slice(&RED);
    raw.extend_from_slice(&BLUE);
    let frame = paint(&raw, &font, 1_000, &mut Hold::new());
    let x = (FRAME_WIDTH - 8) / 2;
    let y = (FRAME_HEIGHT - 8) / 2;
    assert_eq!(pixel(&frame, x, y), BLUE, "U+001B is not drawn");
    assert_eq!(pixel(&frame, x + 7, y + 7), RED, "the cell is U+FFFD");
}

#[test]
fn a_stale_notice_on_a_tall_grid_uses_the_last_full_row() {
    let font = psf(128, 8, 7, 0, 0, &[]);
    // 155 * 7 = 1085. Row 154 starts at y = 1078 and runs past 1080.
    // Row 153 starts at y = 1071 and ends at 1078.
    let bytes = file(1, 155, 0, vec![cell('A', RED, BLUE); 155]);
    let mut hold = Hold::new();
    let frame = paint(&bytes, &font, 0, &mut hold);
    let x = (FRAME_WIDTH - 8) / 2;
    assert_eq!(pixel(&frame, x, 1071), BLUE);
    assert_eq!(pixel(&frame, x, 1078), BLUE);
    let frame = paint(&bytes, &font, 5_001, &mut hold);
    assert_eq!(
        pixel(&frame, x, 1071),
        [0, 0, 0],
        "the last full row is cleared"
    );
    assert_eq!(pixel(&frame, x, 1078), BLUE, "a cropped row stays the cell");
}
