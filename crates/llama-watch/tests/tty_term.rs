//! Grid diff and the term byte emitter. Output is a `Vec<u8>`, never a tty.

use std::fs::File;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use llama_watch::tty::term::{ConsoleBlank, GLYPHS, Size, Term};
use llama_watch::tty::{C16, Cell, Grid};

fn start(cols: u16, rows: u16, full: Duration, now: Instant) -> Term<Vec<u8>> {
    Term::new(Vec::new(), move || Ok(Size { cols, rows }), full, now).expect("size")
}

fn paint(term: &mut Term<Vec<u8>>, grid: &Grid, now: Instant) -> Vec<u8> {
    term.out_mut().clear();
    term.render(grid, now).expect("render");
    term.out().to_vec()
}

/// Parsed `/usr/lib/kbd/consolefonts/eurlatgr.psfu.gz` on 2026-09-25:
/// U+2581–2587 and U+2594 are missing. The font has `█ ▀ ▄ ▌ ▐ ░ ▒ ▓`,
/// `·` (U+00B7) and `…` (U+2026, glyph 491). `chart_glyphs = "halves"`
/// draws with only these, for when llama-hack-12x24 failed to load.
const EURLATGR: &[char] = &['█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·', '…'];

/// Lower eighths U+2581–2587 other than `▄`. Verified present in
/// llama-hack-12x24 by `tests/tty_font.rs`; absent from eurlatgr.
const LOWER_EIGHTHS: &[char] = &['▁', '▂', '▃', '▅', '▆', '▇'];

#[test]
fn term_non_ascii_glyphs_are_eurlatgr_plus_the_lower_eighths() {
    let want: Vec<char> = EURLATGR.iter().chain(LOWER_EIGHTHS).copied().collect();
    assert_eq!(GLYPHS, want.as_slice());
}

#[test]
fn ellipsis_is_emitted_as_utf8() {
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(4, 1);
    grid.put(0, 0, Cell::new('…', C16::White, C16::Black));
    let bytes = paint(&mut term, &grid, t0);
    assert!(
        bytes.windows(3).any(|w| w == [0xe2, 0x80, 0xa6]),
        "ellipsis became '?' : {bytes:?}"
    );
}

#[test]
fn chart_glyphs_are_emitted_as_utf8() {
    let t0 = Instant::now();
    let mut term = start(12, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(12, 1);
    let glyphs = ['█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·'];
    for (i, ch) in glyphs.into_iter().enumerate() {
        grid.put(i as u16, 0, Cell::new(ch, C16::Blue, C16::Black));
    }
    let bytes = paint(&mut term, &grid, t0);
    for ch in glyphs {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf).as_bytes();
        assert!(
            bytes.windows(encoded.len()).any(|w| w == encoded),
            "missing {ch:?} ({encoded:?}) in {bytes:?}"
        );
    }
}

#[test]
fn lower_eighths_are_emitted_as_utf8() {
    let t0 = Instant::now();
    let mut term = start(8, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(8, 1);
    for (i, ch) in LOWER_EIGHTHS.iter().enumerate() {
        grid.put(i as u16, 0, Cell::new(*ch, C16::Blue, C16::Black));
    }
    let bytes = paint(&mut term, &grid, t0);
    for ch in LOWER_EIGHTHS {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf).as_bytes();
        assert!(
            bytes.windows(encoded.len()).any(|w| w == encoded),
            "missing {ch:?} ({encoded:?}) in {bytes:?}"
        );
    }
}

/// ▔ (U+2594) is in the font but nothing draws it; the rest are not in the font.
#[test]
fn other_blocks_are_not_emitted() {
    const OTHERS: [char; 6] = ['▔', '▉', '▏', '▖', '▟', '\u{1FB82}'];
    let t0 = Instant::now();
    let mut term = start(8, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(8, 1);
    for (i, ch) in OTHERS.into_iter().enumerate() {
        grid.put(i as u16, 0, Cell::new(ch, C16::White, C16::Black));
    }
    let bytes = paint(&mut term, &grid, t0);
    assert!(
        bytes.windows(6).any(|w| w == b"??????"),
        "other blocks should become ASCII '?': {bytes:?}"
    );
    for ch in OTHERS {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf).as_bytes();
        assert!(
            !bytes.windows(encoded.len()).any(|w| w == encoded),
            "block {ch:?} leaked"
        );
    }
}

#[test]
fn unchanged_frame_emits_nothing() {
    let t0 = Instant::now();
    let mut term = start(4, 2, Duration::from_secs(30), t0);
    let grid = Grid::blank(4, 2);
    let first = paint(&mut term, &grid, t0);
    assert!(!first.is_empty(), "the first frame is a full repaint");
    let second = paint(&mut term, &grid, t0 + Duration::from_secs(1));
    assert_eq!(second, b"");
}

#[test]
fn one_cell_character_change_is_one_cup_and_the_char() {
    let t0 = Instant::now();
    let mut term = start(4, 3, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(4, 3);
    paint(&mut term, &grid, t0);
    grid.put(2, 1, Cell::new('Z', C16::White, C16::Black));
    let bytes = paint(&mut term, &grid, t0 + Duration::from_millis(10));
    assert_eq!(bytes, b"\x1b%G\x1b[2;3HZ");
}

#[test]
fn one_cell_color_change_adds_sgr_only_for_the_attribute_that_changed() {
    let t0 = Instant::now();
    let mut term = start(4, 3, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(4, 3);
    paint(&mut term, &grid, t0);

    grid.put(2, 1, Cell::new(' ', C16::Red, C16::Black));
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(10)),
        b"\x1b%G\x1b[2;3H\x1b[31m "
    );

    grid.put(2, 1, Cell::new(' ', C16::Red, C16::Blue));
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(20)),
        b"\x1b%G\x1b[2;3H\x1b[44m "
    );

    grid.put(2, 1, Cell::new('Q', C16::Green, C16::Yellow));
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(30)),
        b"\x1b%G\x1b[2;3H\x1b[32m\x1b[43mQ"
    );
}

#[test]
fn sgr_is_emitted_when_the_cell_does_not_match_the_active_color() {
    let t0 = Instant::now();
    let mut term = start(2, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(2, 1);
    grid.put(1, 0, Cell::new('R', C16::Red, C16::Black));
    paint(&mut term, &grid, t0);
    grid.put(0, 0, Cell::new('Z', C16::White, C16::Black));
    let bytes = paint(&mut term, &grid, t0 + Duration::from_millis(10));
    assert_eq!(bytes, b"\x1b%G\x1b[1;1H\x1b[37mZ");
}

#[test]
fn a_run_of_changed_cells_uses_one_cup() {
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(4, 1);
    paint(&mut term, &grid, t0);
    grid.put(1, 0, Cell::new('X', C16::White, C16::Black));
    grid.put(2, 0, Cell::new('Y', C16::White, C16::Black));
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(10)),
        b"\x1b%G\x1b[1;2HXY"
    );

    grid.put(0, 0, Cell::new('A', C16::White, C16::Black));
    grid.put(2, 0, Cell::new('C', C16::Green, C16::Black));
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(20)),
        b"\x1b%G\x1b[1;1HA\x1b[1;3H\x1b[32mC"
    );
}

#[test]
fn full_repaint_at_start_uses_the_fixed_sequences() {
    let t0 = Instant::now();
    let mut term = start(1, 2, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(1, 2);
    grid.put(0, 0, Cell::new('A', C16::White, C16::Black));
    grid.put(0, 1, Cell::new('B', C16::White, C16::Black));
    let bytes = paint(&mut term, &grid, t0);
    assert_eq!(
        bytes,
        b"\x1b%G\x1b[?25l\x1b[0m\x1b[2J\x1b[H\x1b[37m\x1b[40mA\x1b[2;1HB"
    );
}

#[test]
fn full_repaint_fires_at_full_redraw_s() {
    let t0 = Instant::now();
    let mut term = start(2, 1, Duration::from_secs(5), t0);
    let grid = Grid::blank(2, 1);
    paint(&mut term, &grid, t0);
    assert_eq!(
        paint(&mut term, &grid, t0 + Duration::from_millis(4999)),
        b""
    );
    let bytes = paint(&mut term, &grid, t0 + Duration::from_secs(5));
    assert!(bytes.starts_with(b"\x1b%G"), "{bytes:?}");
    assert!(
        bytes.windows(4).any(|w| w == b"\x1b[2J"),
        "missing ED 2: {bytes:?}"
    );
    assert!(
        bytes.windows(6).any(|w| w == b"\x1b[?25l"),
        "cursor hide is part of the full repaint"
    );
}

#[test]
fn size_change_forces_a_full_repaint() {
    let t0 = Instant::now();
    let mut n = 0u32;
    let mut term = Term::new(
        Vec::new(),
        move || {
            n += 1;
            let cols = if n >= 3 { 20 } else { 10 };
            Ok(Size { cols, rows: 4 })
        },
        Duration::from_secs(60),
        t0,
    )
    .expect("size");
    assert_eq!(term.cols(), 10);
    let grid = Grid::blank(10, 4);
    paint(&mut term, &grid, t0);
    assert_eq!(paint(&mut term, &grid, t0 + Duration::from_secs(5)), b"");
    assert_eq!(term.cols(), 10);
    let bytes = paint(&mut term, &grid, t0 + Duration::from_secs(10));
    assert_eq!(term.cols(), 20);
    assert!(bytes.starts_with(b"\x1b%G"));
    assert!(bytes.windows(4).any(|w| w == b"\x1b[2J"));
}

#[test]
fn winsize_is_read_at_start_and_every_five_seconds() {
    let t0 = Instant::now();
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    let mut term = Term::new(
        Vec::new(),
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Size { cols: 8, rows: 2 })
        },
        Duration::from_secs(60),
        t0,
    )
    .expect("size");
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    let grid = Grid::blank(8, 2);
    paint(&mut term, &grid, t0);
    paint(&mut term, &grid, t0 + Duration::from_secs(4));
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    paint(&mut term, &grid, t0 + Duration::from_secs(5));
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}

#[test]
fn a_later_winsize_error_keeps_the_previous_size() {
    let t0 = Instant::now();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let mut term = Term::new(
        Vec::new(),
        move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Ok(Size { cols: 8, rows: 2 })
            } else {
                Err(io::Error::other("not a tty"))
            }
        },
        Duration::from_secs(60),
        t0,
    )
    .expect("size");
    let grid = Grid::blank(8, 2);
    paint(&mut term, &grid, t0);
    assert_eq!(paint(&mut term, &grid, t0 + Duration::from_secs(5)), b"");
    assert_eq!((term.cols(), term.rows()), (8, 2));
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "a frame at 5s must read the size again"
    );
}

#[test]
fn window_size_uses_tcgetwinsize() {
    let file = File::open("/dev/null").expect("null");
    let err = llama_watch::tty::term::window_size(&file).expect_err("not a tty");
    assert_eq!(err.raw_os_error(), Some(25), "{err}");
    let started = Term::from_fd(
        Vec::<u8>::new(),
        file,
        Duration::from_secs(5),
        Instant::now(),
    );
    let Err(err) = started else {
        panic!("start requires a size");
    };
    assert_eq!(err.raw_os_error(), Some(25), "{err}");
}

#[test]
fn glyphs_use_the_six_encodings_and_hostile_cells_become_question_marks() {
    let t0 = Instant::now();
    let mut term = start(8, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(8, 1);
    for (col, ch) in ['█', '▌', '▐', '░', '▒', '▓'].into_iter().enumerate() {
        grid.put(col as u16, 0, Cell::new(ch, C16::White, C16::Black));
    }
    grid.put(6, 0, Cell::new('\u{1b}', C16::White, C16::Black));
    grid.put(7, 0, Cell::new('Û', C16::BrightRed, C16::BrightBlue));
    let bytes = paint(&mut term, &grid, t0);
    let glyphs: [&[u8]; 6] = [
        &[0xe2, 0x96, 0x88],
        &[0xe2, 0x96, 0x8c],
        &[0xe2, 0x96, 0x90],
        &[0xe2, 0x96, 0x91],
        &[0xe2, 0x96, 0x92],
        &[0xe2, 0x96, 0x93],
    ];
    let mut expected = b"\x1b%G\x1b[?25l\x1b[0m\x1b[2J\x1b[H\x1b[37m\x1b[40m".to_vec();
    for glyph in glyphs {
        expected.extend_from_slice(glyph);
    }
    expected.extend_from_slice(b"?\x1b[91m\x1b[44m?");
    assert_eq!(bytes, expected);
    assert!(!bytes.windows(2).any(|w| w == [0xc3, 0x9b]));
}

#[test]
fn put_outside_the_grid_does_not_panic() {
    let mut grid = Grid::new(2, 1);
    assert_eq!((grid.cols(), grid.rows()), (2, 1));
    grid.put(5, 0, Cell::new('Z', C16::Red, C16::Black));
    grid.put(0, 4, Cell::new('Y', C16::Red, C16::Black));
    assert_eq!(grid.get(0, 0), Some(Cell::blank()));
    assert_eq!(grid.get(1, 0), Some(Cell::blank()));
    assert_eq!(grid.get(5, 0), None);
}

#[test]
fn cells_outside_the_terminal_are_not_emitted() {
    let t0 = Instant::now();
    let mut term = start(2, 1, Duration::from_secs(30), t0);
    let mut grid = Grid::blank(4, 2);
    grid.put(0, 0, Cell::new('A', C16::White, C16::Black));
    grid.put(1, 0, Cell::new('B', C16::White, C16::Black));
    grid.put(3, 0, Cell::new('Z', C16::White, C16::Black));
    grid.put(0, 1, Cell::new('Y', C16::White, C16::Black));
    let bytes = paint(&mut term, &grid, t0);
    assert!(bytes.contains(&b'A') && bytes.contains(&b'B'), "{bytes:?}");
    assert!(
        !bytes.contains(&b'Z'),
        "column past the terminal was emitted: {bytes:?}"
    );
    assert!(
        !bytes.contains(&b'Y'),
        "row past the terminal was emitted: {bytes:?}"
    );

    grid.put(3, 0, Cell::new('Q', C16::White, C16::Black));
    let diff = paint(&mut term, &grid, t0 + Duration::from_millis(10));
    assert!(
        !diff.contains(&b'Q'),
        "out-of-window change was emitted: {diff:?}"
    );
}

#[test]
fn a_short_write_forces_the_next_frame_to_repaint() {
    use std::sync::Mutex;

    struct Sink {
        bytes: Vec<u8>,
        fail: bool,
    }
    struct SharedSink(Arc<Mutex<Sink>>);
    impl io::Write for SharedSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut sink = self.0.lock().expect("sink");
            if sink.fail {
                sink.fail = false;
                if !buf.is_empty() {
                    sink.bytes.push(buf[0]);
                }
                return Err(io::Error::other("failed write"));
            }
            sink.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let t0 = Instant::now();
    let sink = Arc::new(Mutex::new(Sink {
        bytes: Vec::new(),
        fail: false,
    }));
    let mut term = Term::new(
        SharedSink(Arc::clone(&sink)),
        || Ok(Size { cols: 2, rows: 1 }),
        Duration::from_secs(60),
        t0,
    )
    .expect("size");
    let mut grid = Grid::blank(2, 1);
    term.render(&grid, t0).expect("first");
    grid.put(0, 0, Cell::new('Z', C16::White, C16::Black));
    sink.lock().expect("sink").fail = true;
    assert!(term.render(&grid, t0 + Duration::from_millis(10)).is_err());
    let mark = sink.lock().expect("sink").bytes.len();
    term.render(&grid, t0 + Duration::from_millis(20))
        .expect("repaint");
    let rest = sink.lock().expect("sink").bytes[mark..].to_vec();
    assert!(
        rest.windows(4).any(|window| window == b"\x1b[2J"),
        "next frame was not a full repaint: {rest:?}"
    );
}

#[test]
fn frame_cost() {
    let cols = 480u16;
    let rows = 135u16;
    let t0 = Instant::now();
    let mut term = start(cols, rows, Duration::from_secs(3600), t0);
    let mut grid = Grid::blank(cols, rows);
    for row in 0..rows {
        for col in (0..cols).step_by(80) {
            grid.put(col, row, Cell::new('█', C16::Green, C16::Black));
        }
    }
    paint(&mut term, &grid, t0);
    grid.put(10, 10, Cell::new('X', C16::White, C16::Black));
    paint(&mut term, &grid, t0 + Duration::from_millis(1));

    let full_at = t0 + Duration::from_secs(3600);
    term.out_mut().clear();
    let started = Instant::now();
    term.render(&grid, full_at).expect("full");
    let full_dt = started.elapsed();
    assert!(
        term.out().windows(4).any(|w| w == b"\x1b[2J"),
        "timed frame was not a full repaint"
    );

    grid.put(11, 10, Cell::new('Y', C16::Red, C16::Black));
    term.out_mut().clear();
    let started = Instant::now();
    term.render(&grid, full_at + Duration::from_millis(1))
        .expect("diff");
    let diff_dt = started.elapsed();
    assert!(
        term.out().windows(3).any(|w| w == b"\x1b[31m") || term.out().ends_with(b"Y"),
        "timed diff did not draw the changed cell: {:?}",
        term.out()
    );

    if !cfg!(debug_assertions) {
        let budget = Duration::from_millis(2);
        assert!(full_dt < budget, "full repaint {full_dt:?}");
        assert!(diff_dt < budget, "diff {diff_dt:?}");
    }
}

fn count(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

fn blanking(blank_min: u32, sleep_min: u32, full: Duration, now: Instant) -> Term<Vec<u8>> {
    start(4, 2, full, now).with_console_blank(ConsoleBlank::from_minutes(blank_min, sleep_min))
}

#[test]
fn console_blank_bytes_are_sent_once_at_start_up() {
    let t0 = Instant::now();
    let mut term = blanking(10, 15, Duration::from_secs(5), t0);
    let grid = Grid::blank(4, 2);
    let first = paint(&mut term, &grid, t0);
    // Example: blank at 10 min, powerdown 15 - 10 = 5 min after the blank.
    assert!(
        first.ends_with(b"\x1b[9;10]\x1b[14;5]"),
        "start-up frame must end with the blank timers: {first:?}"
    );
    assert_eq!(count(&first, b"\x1b[9;"), 1, "{first:?}");
    assert_eq!(count(&first, b"\x1b[14;"), 1, "{first:?}");
    assert!(first.starts_with(b"\x1b%G"));
}

#[test]
fn console_blank_is_not_resent_on_a_full_repaint_or_a_diff() {
    // ESC [ 9 ; n ] unblanks and restarts the kernel blank timer. Sent with
    // every full repaint (5 s here) the screen would never blank.
    let t0 = Instant::now();
    let mut term = blanking(10, 15, Duration::from_secs(5), t0);
    let mut grid = Grid::blank(4, 2);
    paint(&mut term, &grid, t0);

    grid.put(0, 0, Cell::new('A', C16::White, C16::Black));
    let diff = paint(&mut term, &grid, t0 + Duration::from_millis(100));
    assert!(!diff.is_empty());
    for (i, at) in [5u64, 10, 15, 3600].into_iter().enumerate() {
        grid.put(
            1,
            0,
            Cell::new(char::from(b'a' + i as u8), C16::White, C16::Black),
        );
        let full = paint(&mut term, &grid, t0 + Duration::from_secs(at));
        assert!(
            full.windows(4).any(|w| w == b"\x1b[2J"),
            "frame at {at}s was not a full repaint"
        );
        assert_eq!(
            count(&full, b"\x1b[9;"),
            0,
            "blank re-sent at {at}s: {full:?}"
        );
        assert_eq!(
            count(&full, b"\x1b[14;"),
            0,
            "powerdown re-sent at {at}s: {full:?}"
        );
    }
    assert_eq!(count(&diff, b"\x1b[9;"), 0);
    assert_eq!(count(&diff, b"\x1b[14;"), 0);
}

#[test]
fn console_blank_off_sends_nothing() {
    let t0 = Instant::now();
    let grid = Grid::blank(4, 2);
    let mut plain = start(4, 2, Duration::from_secs(5), t0);
    let want = paint(&mut plain, &grid, t0);
    for (blank_min, sleep_min) in [(0, 0), (0, 15), (61, 0), (10, 10), (10, 5), (10, 61)] {
        let mut term = blanking(blank_min, sleep_min, Duration::from_secs(5), t0);
        let got = paint(&mut term, &grid, t0);
        let expect_blank = (1..=60).contains(&blank_min);
        if expect_blank {
            let mut with = want.clone();
            with.extend_from_slice(format!("\x1b[9;{blank_min}]").as_bytes());
            assert_eq!(got, with, "({blank_min}, {sleep_min})");
        } else {
            assert_eq!(got, want, "({blank_min}, {sleep_min}) must send nothing");
        }
        assert_eq!(count(&got, b"\x1b[14;"), 0, "({blank_min}, {sleep_min})");
    }
}

#[test]
fn console_blank_without_sleep_sends_only_the_blank() {
    let t0 = Instant::now();
    let mut term = blanking(60, 0, Duration::from_secs(5), t0);
    let first = paint(&mut term, &Grid::blank(4, 2), t0);
    assert!(first.ends_with(b"\x1b[9;60]"), "{first:?}");
    assert_eq!(count(&first, b"\x1b[14;"), 0);
    let mut term = blanking(1, 60, Duration::from_secs(5), t0);
    let first = paint(&mut term, &Grid::blank(4, 2), t0);
    assert!(first.ends_with(b"\x1b[9;1]\x1b[14;59]"), "{first:?}");
}

#[test]
fn console_blank_is_retried_when_the_first_write_fails() {
    use std::sync::Mutex;

    struct FailOnce(Arc<Mutex<(bool, Vec<u8>)>>);
    impl io::Write for FailOnce {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut sink = self.0.lock().expect("sink");
            if sink.0 {
                sink.0 = false;
                return Err(io::Error::other("failed write"));
            }
            sink.1.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let sink = Arc::new(Mutex::new((true, Vec::new())));
    let t0 = Instant::now();
    let mut term = Term::new(
        FailOnce(Arc::clone(&sink)),
        || Ok(Size { cols: 4, rows: 2 }),
        Duration::from_secs(5),
        t0,
    )
    .expect("size")
    .with_console_blank(ConsoleBlank::from_minutes(10, 15));
    let grid = Grid::blank(4, 2);
    assert!(term.render(&grid, t0).is_err());
    term.render(&grid, t0 + Duration::from_millis(100))
        .expect("retry");
    term.render(&grid, t0 + Duration::from_secs(10))
        .expect("full");
    let bytes = sink.lock().expect("sink").1.clone();
    assert_eq!(count(&bytes, b"\x1b[2J"), 2, "{bytes:?}");
    assert_eq!(count(&bytes, b"\x1b[9;10]"), 1, "{bytes:?}");
    assert_eq!(count(&bytes, b"\x1b[14;5]"), 1, "{bytes:?}");
}

/// #26: `[tty] palette = "llama"` loads all 16 slots on every full repaint,
/// before the clear, and never on a diff frame.
#[test]
fn llama_palette_is_loaded_on_every_full_repaint() {
    use llama_core::palette::{CONSOLE_RESET, LLAMA, Palette, console_load};
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(5), t0).with_palette(Palette::Llama);
    let load = console_load(&LLAMA);
    let mut grid = Grid::blank(4, 1);
    let first = paint(&mut term, &grid, t0);
    assert_eq!(count(&first, &load), 1, "{first:?}");
    let at = first.windows(load.len()).position(|w| w == load).unwrap();
    let clear = first.windows(4).position(|w| w == b"\x1b[2J").unwrap();
    assert!(at < clear, "palette before the clear");
    assert!(first.starts_with(b"\x1b%G\x1b[?25l\x1b[0m\x1b]P0000000\x1b]P1ff2a14"));
    assert_eq!(count(&first, CONSOLE_RESET), 0);

    grid.put(0, 0, Cell::new('x', C16::White, C16::Black));
    let diff = paint(&mut term, &grid, t0 + Duration::from_secs(1));
    assert!(!diff.is_empty());
    assert_eq!(count(&diff, b"\x1b]P"), 0, "diff frames carry no palette");

    let full = paint(&mut term, &grid, t0 + Duration::from_secs(6));
    assert_eq!(
        count(&full, &load),
        1,
        "the scheduled full redraw reloads it"
    );
}

/// `vga` writes `ESC ] R` with the first frame only, and never `ESC ] P`.
#[test]
fn vga_palette_resets_once_and_loads_nothing() {
    use llama_core::palette::{CONSOLE_RESET, Palette};
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(5), t0).with_palette(Palette::Vga);
    let grid = Grid::blank(4, 1);
    let first = paint(&mut term, &grid, t0);
    assert_eq!(count(&first, CONSOLE_RESET), 1);
    assert_eq!(count(&first, b"\x1b]P"), 0);
    let full = paint(&mut term, &grid, t0 + Duration::from_secs(6));
    assert!(!full.is_empty());
    assert_eq!(count(&full, b"\x1b]"), 0, "{full:?}");
    term.out_mut().clear();
    term.restore_palette().unwrap();
    assert!(term.out().is_empty(), "vga has nothing to restore");
}

/// Clean exit puts the kernel palette back, and a later frame reloads ours.
#[test]
fn llama_palette_is_reset_on_restore() {
    use llama_core::palette::{CONSOLE_RESET, LLAMA, Palette, console_load};
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(30), t0).with_palette(Palette::Llama);
    let grid = Grid::blank(4, 1);
    paint(&mut term, &grid, t0);
    term.out_mut().clear();
    term.restore_palette().unwrap();
    assert_eq!(term.out(), CONSOLE_RESET);
    let again = paint(&mut term, &grid, t0 + Duration::from_millis(100));
    assert_eq!(count(&again, &console_load(&LLAMA)), 1);
}

/// A `Term` built without `with_palette` writes no palette bytes (the
/// older byte-exact tests depend on it).
#[test]
fn no_palette_bytes_unless_configured() {
    let t0 = Instant::now();
    let mut term = start(4, 1, Duration::from_secs(30), t0);
    let first = paint(&mut term, &Grid::blank(4, 1), t0);
    assert_eq!(count(&first, b"\x1b]"), 0);
    term.restore_palette().unwrap();
}

/// #26: `llama-watch tty-reset` (the unit's ExecStopPost) writes exactly
/// `ESC ] R` to stdout and reads no config.
#[test]
fn tty_reset_writes_only_the_palette_reset() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_llama-watch"))
        .arg("tty-reset")
        .output()
        .expect("run tty-reset");
    assert!(out.status.success(), "{out:?}");
    assert_eq!(out.stdout, b"\x1b]R");
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_llama-watch"))
        .args(["tty-reset", "--config", "/nonexistent"])
        .output()
        .expect("run tty-reset");
    assert_eq!(bad.status.code(), Some(2));
    assert!(bad.stdout.is_empty());
}
