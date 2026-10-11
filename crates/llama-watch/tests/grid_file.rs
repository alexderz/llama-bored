//! #107: the grid file. `Cell::rich` from the sanitiser, the writer's
//! decisions (changed, heartbeat, retry), the writer thread (a failing disk
//! never stops frames), and that tty11's bytes do not depend on any of it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llama_core::gridfile::{self, HEARTBEAT};
use llama_core::log::Sink;
use llama_core::palette::Palette;
use llama_watch::config::{Config, DEFAULT_GRID_PATH};
use llama_watch::grid_out::{GridTap, GridWriter};
use llama_watch::tty::sanitize::sanitize;
use llama_watch::tty::term::{GLYPHS, Size, Term};
use llama_watch::tty::{C16, Cell, Grid};

fn rich(input: &str) -> (String, String) {
    let mut cells = Vec::new();
    sanitize(input, &mut cells);
    (
        cells.iter().map(|c| c.ch).collect(),
        cells.iter().map(|c| c.rich).collect(),
    )
}

#[test]
fn rich_is_the_original_scalar_when_it_is_safe_as_one_cell() {
    for (input, ch, rich_want) in [
        ("caf\u{e9}", "cafe", "caf\u{e9}"),
        (
            "\u{201c}hi\u{201d} it\u{2019}s",
            "\u{201c}hi\u{201d} it\u{2019}s",
            "\u{201c}hi\u{201d} it\u{2019}s",
        ),
        ("a \u{2192} b", "a > b", "a \u{2192} b"),
        ("3 \u{d7} 4", "3 x 4", "3 \u{d7} 4"),
        (
            "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}",
            "\u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}",
            "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}",
        ),
        ("\u{3b1}\u{3b2}", "\u{fffd}\u{fffd}", "\u{3b1}\u{3b2}"),
        ("plain ~", "plain ~", "plain ~"),
        ("a\u{a0}b", "a b", "a b"),
    ] {
        assert_eq!(
            rich(input),
            (ch.to_string(), rich_want.to_string()),
            "{input:?}"
        );
    }
}

#[test]
fn rich_falls_back_to_ch_for_cjk_emoji_and_bidi() {
    for input in [
        "\u{4e2d}\u{6587}",
        "\u{1f600}",
        "\u{202e}x",
        "\u{2066}x\u{2069}",
        "e\u{301}",
        "\u{ff21}",
    ] {
        let (ch, rich) = rich(input);
        assert_eq!(ch, rich, "{input:?}");
        assert!(
            !rich.contains('\u{202e}') && !rich.contains('\u{4e2d}') && !rich.contains('\u{1f600}')
        );
    }
    // Zero-width characters still take no cell at all.
    assert_eq!(rich("a\u{200b}b"), ("ab".to_string(), "ab".to_string()));
}

#[test]
fn constructors_default_rich_to_ch() {
    assert_eq!(Cell::new('x', C16::Red, C16::Black).rich, 'x');
    assert_eq!(Cell::blank().rich, ' ');
    let cell = Cell::with_rich('e', '\u{e9}', C16::White, C16::Black);
    assert!(cell.same_console(Cell::new('e', C16::White, C16::Black)));
    assert_ne!(cell, Cell::new('e', C16::White, C16::Black));
}

/// Every console glyph tty11 draws is one the grid file may carry, except
/// those the shared predicate leaves out (reported by name).
#[test]
fn console_glyphs_are_showable() {
    for &ch in GLYPHS {
        assert!(
            llama_core::glyph::is_showable(ch),
            "{ch:?} U+{:04X}",
            u32::from(ch)
        );
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("grid-file-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    dir
}

fn read(path: &Path) -> gridfile::GridImage {
    gridfile::decode(&std::fs::read(path).expect("read")).expect("decode")
}

fn frame(text: &str) -> Grid {
    let mut grid = Grid::blank(12, 2);
    for (i, ch) in text.chars().enumerate() {
        let ascii = if ch.is_ascii() { ch } else { '?' };
        grid.put(
            u16::try_from(i).unwrap(),
            0,
            Cell::with_rich(ascii, ch, C16::Red, C16::Yellow),
        );
    }
    grid
}

#[test]
fn the_writer_writes_rich_characters_and_palette_rgb() {
    let dir = scratch("content");
    let path = dir.join("grid");
    let mut writer = GridWriter::new(&path, Palette::Llama);
    let t0 = Instant::now();
    assert!(
        writer
            .offer(&frame("caf\u{e9} \u{2192}"), t0, 1_000)
            .unwrap()
    );
    let image = read(&path);
    assert_eq!(
        (image.cols, image.rows, image.seq, image.unix_ms),
        (12, 2, 1, 1_000)
    );
    let text: String = image.cells[..6].iter().map(|c| c.ch).collect();
    assert_eq!(text, "caf\u{e9} \u{2192}");
    let red = Palette::Llama.rgb(1);
    let yellow = Palette::Llama.rgb(3);
    assert_eq!(image.cells[0].fg, red);
    assert_eq!(image.cells[0].bg, yellow);
    // A bright background is shown as the console shows it: the normal colour.
    let mut grid = Grid::blank(12, 2);
    grid.put(0, 0, Cell::new('x', C16::BrightRed, C16::BrightYellow));
    writer
        .offer(&grid, t0 + Duration::from_millis(100), 1_100)
        .unwrap();
    let image = read(&path);
    assert_eq!(image.cells[0].fg, Palette::Llama.rgb(9));
    assert_eq!(image.cells[0].bg, Palette::Llama.rgb(3));
    // No temp file is left, and the file is not world readable.
    assert!(!dir.join("grid.tmp").exists());
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o007,
        0
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unchanged_frames_wait_for_the_heartbeat() {
    let dir = scratch("heartbeat");
    let path = dir.join("grid");
    let mut writer = GridWriter::new(&path, Palette::Llama);
    let t0 = Instant::now();
    let grid = frame("same");
    assert!(writer.offer(&grid, t0, 10).unwrap());
    assert_eq!(writer.seq(), 1);
    for ms in [100, 400, 900] {
        let wrote = writer
            .offer(&grid, t0 + Duration::from_millis(ms), 10 + ms)
            .unwrap();
        assert!(!wrote, "{ms} ms");
    }
    assert_eq!(read(&path).seq, 1);
    // A change goes out at once.
    assert!(
        writer
            .offer(&frame("diff"), t0 + Duration::from_millis(950), 960)
            .unwrap()
    );
    assert_eq!(read(&path).seq, 2);
    // The heartbeat: the same frame again once a second has passed.
    let t1 = t0 + Duration::from_millis(950) + HEARTBEAT;
    assert!(writer.offer(&frame("diff"), t1, 2_000).unwrap());
    let image = read(&path);
    assert_eq!((image.seq, image.unix_ms), (3, 2_000));
    // And with no new frame at all.
    assert!(
        !writer
            .heartbeat(t1 + Duration::from_millis(500), 2_500)
            .unwrap()
    );
    assert!(writer.heartbeat(t1 + HEARTBEAT, 3_000).unwrap());
    assert_eq!(read(&path).seq, 4);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_write_is_an_error_retried_once_per_heartbeat() {
    let dir = scratch("fail");
    let path = dir.join("absent-dir").join("grid");
    let mut writer = GridWriter::new(&path, Palette::Llama);
    let t0 = Instant::now();
    assert!(writer.offer(&frame("a"), t0, 1).is_err());
    assert_eq!(writer.seq(), 0, "a failed write is not a write");
    // Waiting out the retry delay: no second attempt, no error.
    assert!(
        !writer
            .offer(&frame("b"), t0 + Duration::from_millis(100), 2)
            .unwrap()
    );
    assert!(writer.offer(&frame("b"), t0 + HEARTBEAT, 3).is_err());
    // The directory appears: the next attempt works.
    std::fs::create_dir_all(dir.join("absent-dir")).unwrap();
    assert!(writer.offer(&frame("b"), t0 + HEARTBEAT * 2, 4).unwrap());
    assert_eq!(read(&path).seq, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<String>>>);

impl Sink for Capture {
    fn write_line(&mut self, line: &str) {
        self.0.lock().unwrap().push(line.to_owned());
    }
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(5);
    while Instant::now() < until {
        if ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

fn seq_of(path: &Path) -> Option<u32> {
    let image = gridfile::decode(&std::fs::read(path).ok()?).ok()?;
    Some(image.seq)
}

#[test]
fn the_tap_writes_changes_and_heartbeats_and_never_blocks() {
    let dir = scratch("tap");
    let path = dir.join("grid");
    let tap = GridTap::spawn(GridWriter::new(&path, Palette::Llama), Capture::default()).unwrap();
    tap.offer(&frame("one"));
    wait_for("the first file", || seq_of(&path) == Some(1));
    // Unchanged frames at tick rate: the file stays as it is, for a while.
    let started = Instant::now();
    for _ in 0..20 {
        tap.offer(&frame("one"));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(started.elapsed() < HEARTBEAT, "test pacing");
    assert_eq!(seq_of(&path), Some(1));
    // A change.
    tap.offer(&frame("two"));
    wait_for("the change", || seq_of(&path) == Some(2));
    // With no frame at all, the thread still heartbeats.
    wait_for("a heartbeat", || seq_of(&path).is_some_and(|s| s >= 3));
    assert_eq!(read(&path).cells[0].ch, 't');
    drop(tap);
    let _ = std::fs::remove_dir_all(&dir);
}

fn euid_is_root() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .is_some_and(|text| text.trim() == "0")
}

#[test]
fn an_unwritable_directory_is_logged_once_and_frames_keep_flowing() {
    if euid_is_root() {
        eprintln!("skipping: euid is 0, so a read-only directory still takes writes");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("unwritable");
    let path = dir.join("grid");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
    let log = Capture::default();
    let tap = GridTap::spawn(GridWriter::new(&path, Palette::Llama), log.clone()).unwrap();
    let started = Instant::now();
    for i in 0..30 {
        tap.offer(&frame(&format!("f{i}")));
        std::thread::sleep(Duration::from_millis(10));
    }
    // offer() never waited on the disk.
    assert!(started.elapsed() < Duration::from_secs(2));
    wait_for("the error line", || !log.0.lock().unwrap().is_empty());
    std::thread::sleep(Duration::from_millis(100));
    let lines = log.0.lock().unwrap().clone();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("<3>grid: cannot write"), "{lines:?}");
    assert!(!path.exists());
    // The disk comes back: writing resumes by itself and says so.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    tap.offer(&frame("back"));
    wait_for("writes to resume", || seq_of(&path).is_some());
    wait_for("the recovery line", || log.0.lock().unwrap().len() == 2);
    drop(tap);
    let _ = std::fs::remove_dir_all(&dir);
}

fn console_bytes(grids: &[Grid], tap: Option<GridTap>) -> Vec<Vec<u8>> {
    let t0 = Instant::now();
    let mut term = Term::new(
        Vec::new(),
        || Ok(Size { cols: 12, rows: 2 }),
        Duration::from_secs(30),
        t0,
    )
    .expect("size");
    if let Some(tap) = tap {
        term = term.with_grid_tap(tap);
    }
    grids
        .iter()
        .map(|grid| {
            term.out_mut().clear();
            if let Some(tap) = term.grid_tap() {
                tap.offer(grid);
            }
            term.render(grid, t0).expect("render");
            term.out().to_vec()
        })
        .collect()
}

/// tty11's bytes are a function of `ch`, `fg` and `bg` only: a tap changes
/// nothing, and a frame that differs only in `rich` writes no byte.
#[test]
fn the_console_output_does_not_depend_on_rich_or_the_tap() {
    let dir = scratch("console");
    let plain = {
        let mut g = Grid::blank(12, 2);
        for (i, ch) in "cafe".chars().enumerate() {
            g.put(i as u16, 0, Cell::new(ch, C16::White, C16::Black));
        }
        g
    };
    let accented = {
        let mut g = plain.clone();
        g.put(3, 0, Cell::with_rich('e', '\u{e9}', C16::White, C16::Black));
        g
    };
    let grids = [plain.clone(), accented.clone(), plain];
    let without = console_bytes(&grids, None);
    let tap = GridTap::spawn(
        GridWriter::new(&dir.join("grid"), Palette::Llama),
        Capture::default(),
    )
    .unwrap();
    let with = console_bytes(&grids, Some(tap));
    assert_eq!(without, with);
    assert!(!without[0].is_empty());
    assert!(without[1].is_empty(), "rich-only change: {:?}", without[1]);
    assert!(without[2].is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn grid_config_defaults_parse_and_validate() {
    let config = Config::from_toml("").expect("empty");
    assert!(!config.grid.enabled);
    assert_eq!(config.grid.path, Path::new(DEFAULT_GRID_PATH));
    assert_eq!(DEFAULT_GRID_PATH, "/run/llama-watch-grid/grid");
    assert!(config.validate(8).is_ok());

    let on = Config::from_toml("[grid]\nenabled = true\npath = \"/run/x/grid\"\n").expect("on");
    assert!(on.grid.enabled);
    assert_eq!(on.grid.path, Path::new("/run/x/grid"));
    assert!(on.validate(8).is_ok());

    for bad in ["grid", "/", "", "/run/x/\u{0}grid"] {
        let text = format!("[grid]\nenabled = true\npath = {bad:?}\n");
        let Ok(config) = Config::from_toml(&text) else {
            continue;
        };
        assert!(config.validate(8).is_err(), "{bad:?}");
    }
    assert!(Config::from_toml("[grid]\nbogus = 1\n").is_err());
}
