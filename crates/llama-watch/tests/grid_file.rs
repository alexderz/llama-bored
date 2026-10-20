//! #107: the grid file. `Cell::rich` from the sanitiser, the writer's
//! decisions (changed, heartbeat, retry), the writer thread (a failing disk
//! never stops frames), and that tty11's bytes do not depend on any of it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use llama_core::gridfile::{self, GridImage, HEARTBEAT};
use llama_core::log::Sink;
use llama_core::palette::Palette;
use llama_watch::collector::LoadSource;
use llama_watch::config::{ChartGlyphs, Config, DEFAULT_GRID_PATH, GridFile, TtySize};
use llama_watch::grid_out::{GridTap, GridWriter};
use llama_watch::service::RenderStep;
use llama_watch::tty::chart::ChartBucket;
use llama_watch::tty::ctx_history::CtxPoint;
use llama_watch::tty::layout::{
    self, Activity, HealthSeg, HealthStatus, SetupItem, SetupRow, SetupView, Slot, TtyModel,
    WatchState,
};
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
    let tap = GridTap::spawn(
        GridWriter::new(&path, Palette::Llama),
        Capture::default(),
        None,
    )
    .unwrap();
    let model = busy_frame();
    tap.offer_frame(&model, &frame("one"));
    wait_for("the first file", || seq_of(&path) == Some(1));
    // Unchanged frames at tick rate: the file stays as it is, for a while.
    let started = Instant::now();
    for _ in 0..20 {
        tap.offer_frame(&model, &frame("one"));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(started.elapsed() < HEARTBEAT, "test pacing");
    assert_eq!(seq_of(&path), Some(1));
    // A change.
    tap.offer_frame(&model, &frame("two"));
    wait_for("the change", || seq_of(&path) == Some(2));
    // With no frame at all, the thread still heartbeats.
    wait_for("a heartbeat", || seq_of(&path).is_some_and(|s| s >= 3));
    assert_eq!(read(&path).cells[0].ch, 't');
    drop(tap);
    let _ = std::fs::remove_dir_all(&dir);
}

/// #115: the same model, offered again because `[grid] size` differs from
/// the console, is written once and then only on the heartbeat.
#[test]
fn an_unchanged_second_layout_waits_for_the_heartbeat() {
    let dir = scratch("idle-size");
    let path = dir.join("grid");
    let tap = GridTap::spawn(
        GridWriter::new(&path, Palette::Llama),
        Capture::default(),
        Some(TtySize {
            cols: 213,
            rows: 60,
        }),
    )
    .unwrap();
    let model = busy_frame();
    let console = frame("idle");
    tap.offer_frame(&model, &console);
    wait_for("the first second layout", || seq_of(&path) == Some(1));
    let first = read(&path);
    assert_eq!((first.cols, first.rows), (213, 60));
    let started = Instant::now();
    for _ in 0..20 {
        tap.offer_frame(&model, &console);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(started.elapsed() < HEARTBEAT, "test pacing");
    assert_eq!(
        seq_of(&path),
        Some(1),
        "an unchanged second layout wrote again"
    );
    wait_for("the heartbeat", || seq_of(&path).is_some_and(|s| s >= 2));
    let again = read(&path);
    assert_eq!(
        again.seq, 2,
        "the file was written more than once per heartbeat"
    );
    assert_eq!((again.cols, again.rows), (213, 60));
    assert_eq!(again.cells, first.cells);
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
    let tap = GridTap::spawn(GridWriter::new(&path, Palette::Llama), log.clone(), None).unwrap();
    let model = busy_frame();
    let started = Instant::now();
    for i in 0..30 {
        tap.offer_frame(&model, &frame(&format!("f{i}")));
        std::thread::sleep(Duration::from_millis(10));
    }
    // offer_frame() never waited on the disk.
    assert!(started.elapsed() < Duration::from_secs(2));
    wait_for("the error line", || !log.0.lock().unwrap().is_empty());
    std::thread::sleep(Duration::from_millis(100));
    let lines = log.0.lock().unwrap().clone();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("<3>grid: cannot write"), "{lines:?}");
    assert!(!path.exists());
    // The disk comes back: writing resumes by itself and says so.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    tap.offer_frame(&model, &frame("back"));
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
    let model = busy_frame();
    grids
        .iter()
        .map(|grid| {
            term.out_mut().clear();
            if let Some(tap) = term.grid_tap() {
                tap.offer_frame(&model, grid);
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
        None,
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
    assert!(config.grid.exact_colors);
    assert_eq!(config.grid.path, Path::new(DEFAULT_GRID_PATH));
    assert_eq!(config.grid.size, None);
    assert_eq!(DEFAULT_GRID_PATH, "/run/llama-watch-grid/grid");
    assert!(config.validate(8).is_ok());

    let on = Config::from_toml("[grid]\nenabled = true\npath = \"/run/x/grid\"\n").expect("on");
    assert!(on.grid.enabled);
    assert!(on.grid.exact_colors);
    assert_eq!(on.grid.path, Path::new("/run/x/grid"));
    assert!(on.validate(8).is_ok());

    let off = Config::from_toml("[grid]\nexact_colors = false\n").expect("exact off");
    assert!(!off.grid.exact_colors);
    assert!(off.validate(8).is_ok());

    for bad in ["grid", "/", "", "/run/x/\u{0}grid"] {
        let text = format!("[grid]\nenabled = true\npath = {bad:?}\n");
        let Ok(config) = Config::from_toml(&text) else {
            continue;
        };
        assert!(config.validate(8).is_err(), "{bad:?}");
    }
    assert!(Config::from_toml("[grid]\nbogus = 1\n").is_err());
}

/// #116: exact colours off, the file is the palette frame. On, it is the
/// ramp frame of the same model, and it is not the palette frame.
#[test]
fn exact_off_matches_the_palette_file_and_exact_on_matches_the_ramp() {
    let model = busy_frame();
    let console = layout::layout(&model, 192, 60);
    let palette = palette_cells(&console);

    let dir = scratch("exact-off");
    let path = dir.join("grid");
    let tap = GridTap::spawn(
        GridWriter::new(&path, Palette::Llama),
        Capture::default(),
        None,
    )
    .unwrap()
    .with_exact(false);
    tap.offer_frame(&model, &console);
    wait_for("the palette file", || seq_of(&path) == Some(1));
    let got = read(&path);
    assert_eq!((got.cols, got.rows), (192, 60));
    assert_eq!(got.cells, palette);
    let via_image = llama_watch::grid_out::image_of(&console, Palette::Llama, got.seq, got.unix_ms);
    assert_eq!(got.cells, via_image.cells);
    drop(tap);
    let _ = std::fs::remove_dir_all(&dir);

    let dir = scratch("exact-on");
    let path = dir.join("grid");
    let tap = GridTap::spawn(
        GridWriter::new(&path, Palette::Llama),
        Capture::default(),
        None,
    )
    .unwrap()
    .with_exact(true);
    tap.offer_frame(&model, &console);
    wait_for("the ramp file", || seq_of(&path) == Some(1));
    let got = read(&path);
    let ramp = layout::layout_exact(&model, 192, 60);
    let want = llama_watch::grid_out::image_of(&ramp, Palette::Llama, got.seq, got.unix_ms);
    assert_eq!(got.cells, want.cells);
    assert_ne!(got.cells, palette);
    let mut encoded = Vec::new();
    gridfile::encode(&want, &mut encoded).expect("encode");
    assert_eq!(std::fs::read(&path).expect("file"), encoded);
    drop(tap);
    let _ = std::fs::remove_dir_all(&dir);
}

/// #116: `[grid]` is what the tap writes. [`GridTap::spawn`] leaves exact
/// colours off, and the service does not call [`GridTap::with_exact`]
/// itself: it calls [`GridTap::from_config`]. The default config (exact
/// colours on) writes the ramp frame; `exact_colors = false` writes the
/// palette frame.
#[test]
fn the_grid_config_is_what_the_tap_writes() {
    let model = busy_frame();
    let console = layout::layout(&model, 192, 60);
    let palette = palette_cells(&console);
    let ramp = layout::layout_exact(&model, 192, 60);

    let offer = |name: &str, grid: &GridFile| {
        let dir = scratch(name);
        let path = dir.join("grid");
        let tap = GridTap::from_config(
            GridWriter::new(&path, Palette::Llama),
            Capture::default(),
            grid,
        )
        .expect("tap");
        tap.offer_frame(&model, &console);
        wait_for(name, || seq_of(&path) == Some(1));
        let got = read(&path);
        drop(tap);
        let _ = std::fs::remove_dir_all(&dir);
        got
    };

    let got = offer("config-default", &GridFile::default());
    let want = llama_watch::grid_out::image_of(&ramp, Palette::Llama, got.seq, got.unix_ms);
    assert_eq!(got.cells, want.cells, "default config writes the ramp");
    assert_ne!(
        got.cells, palette,
        "default config is not the palette frame"
    );

    let off = GridFile {
        exact_colors: false,
        ..GridFile::default()
    };
    let got = offer("config-exact-off", &off);
    assert_eq!(
        got.cells, palette,
        "exact_colors = false writes the palette"
    );
    let via = llama_watch::grid_out::image_of(&console, Palette::Llama, got.seq, got.unix_ms);
    assert_eq!(got.cells, via.cells);
}

/// Today's palette RGB, built without [`image_of`], so an exact colour
/// cannot sneak into the "off" comparison.
fn palette_cells(grid: &Grid) -> Vec<llama_core::gridfile::GridCell> {
    use llama_core::glyph::is_showable;
    use llama_core::gridfile::GridCell;
    grid.cells()
        .iter()
        .map(|cell| GridCell {
            ch: if is_showable(cell.rich) {
                cell.rich
            } else {
                cell.ch
            },
            fg: Palette::Llama.rgb(cell.fg.index()),
            bg: Palette::Llama.rgb(cell.bg.index() & 7),
        })
        .collect()
}

/// #115: `[grid] size` parses with [`TtySize`] and fails in the same words.
#[test]
fn grid_size_uses_the_tty_size_wording_and_bounds() {
    let ok = Config::from_toml("[grid]\nsize = \"213x60\"\n").expect("parse");
    assert_eq!(
        ok.grid.size,
        Some(TtySize {
            cols: 213,
            rows: 60
        })
    );
    assert!(ok.validate(8).is_ok());

    for bad in [
        "\"213X60\"",
        "\"213x\"",
        "\"x60\"",
        "\"0213x60\"",
        "\"213 x 60\"",
        "\"\"",
    ] {
        let err = Config::from_toml(&format!("[grid]\nsize = {bad}\n")).expect_err(bad);
        assert!(
            err.to_string()
                .contains("grid.size must be \"COLSxROWS\", such as \"160x49\""),
            "{bad}: {err}"
        );
    }
    let err = Config::from_toml("[grid]\nsize = \"159x60\"\n")
        .expect("parses")
        .validate(8)
        .expect_err("below the floor");
    assert_eq!(
        err.to_string(),
        "grid.size 159x60 is outside 160x26..=1024x512"
    );
    // `enabled = false` does not skip the check. Nothing is built either way
    // until the file is turned on; a bad size is still a load error.
    let err = Config::from_toml("[grid]\nenabled = false\nsize = \"213x25\"\n")
        .expect("parses")
        .validate(8)
        .expect_err("below the floor");
    assert_eq!(
        err.to_string(),
        "grid.size 213x25 is outside 160x26..=1024x512"
    );
    for (cols, rows) in [(160u16, 26u16), (1024, 512)] {
        let cfg = Config::from_toml(&format!("[grid]\nsize = \"{cols}x{rows}\"\n")).expect("parse");
        assert_eq!(cfg.grid.size, Some(TtySize { cols, rows }));
        assert!(cfg.validate(8).is_ok(), "{cols}x{rows}");
    }
    for (text, msg) in [
        ("1025x60", "grid.size 1025x60 is outside 160x26..=1024x512"),
        (
            "1024x513",
            "grid.size 1024x513 is outside 160x26..=1024x512",
        ),
    ] {
        let err = Config::from_toml(&format!("[grid]\nsize = \"{text}\"\n"))
            .expect("parses")
            .validate(8)
            .expect_err(text);
        assert_eq!(err.to_string(), msg);
    }
}

fn encoded(dir: &Path, grid: &Grid) -> GridImage {
    let path = dir.join("direct");
    let mut writer = GridWriter::new(&path, Palette::Llama);
    writer
        .offer(grid, Instant::now(), 7)
        .expect("encode the console frame");
    read(&path)
}

fn read_ok(path: &Path) -> Option<GridImage> {
    gridfile::decode(&std::fs::read(path).ok()?).ok()
}

fn wait_cells(path: &Path, want: &GridImage) {
    wait_for("the published cells", || {
        read_ok(path).is_some_and(|got| {
            got.cols == want.cols && got.rows == want.rows && got.cells == want.cells
        })
    });
}

fn first_diff(a: &Grid, b: &Grid) -> Option<(u16, u16)> {
    if a.cols() != b.cols() || a.rows() != b.rows() {
        return Some((0, 0));
    }
    for row in 0..a.rows() {
        for col in 0..a.cols() {
            if a.get(col, row) != b.get(col, row) {
                return Some((col, row));
            }
        }
    }
    None
}

/// Top-left copy: a pad when `src` is smaller, a crop when it is larger.
fn placed_like(src: &Grid, cols: u16, rows: u16) -> Grid {
    let mut out = Grid::blank(cols, rows);
    for row in 0..rows.min(src.rows()) {
        for col in 0..cols.min(src.cols()) {
            if let Some(cell) = src.get(col, row) {
                out.put(col, row, cell);
            }
        }
    }
    out
}

/// #115: unset, and a size equal to the console, publish that frame and
/// do not lay it out again. A poisoned cell would not survive a second layout.
#[test]
fn an_unset_or_matching_size_publishes_the_console_frame() {
    let model = busy_frame();
    let fresh = layout::layout(&model, 192, 60);
    let mut console = fresh.clone();
    let mark = Cell::with_rich('Z', '\u{017d}', C16::BrightMagenta, C16::BrightGreen);
    assert_ne!(
        fresh.get(4, 4),
        Some(mark),
        "the mark must not be a layout cell"
    );
    console.put(4, 4, mark);

    for (name, size) in [
        ("unset", None),
        (
            "equal",
            Some(TtySize {
                cols: 192,
                rows: 60,
            }),
        ),
    ] {
        let dir = scratch(name);
        let path = dir.join("grid");
        let want = encoded(&dir, &console);
        let tap = GridTap::spawn(
            GridWriter::new(&path, Palette::Llama),
            Capture::default(),
            size,
        )
        .expect("tap");
        tap.offer_frame(&model, &console);
        wait_cells(&path, &want);
        let got = read(&path);
        assert_eq!((got.cols, got.rows), (192, 60), "{name}");
        assert_ne!(
            encoded(&dir, &fresh).cells,
            got.cells,
            "{name}: a second layout would have dropped the marked cell"
        );
        drop(tap);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// #115: a wider file and a narrower one are real layouts at that size.
/// The console bytes stay the console's.
#[test]
fn a_wider_and_a_narrower_grid_are_laid_out_and_the_console_is_unchanged() {
    let model = busy_frame();
    let console = layout::layout(&model, 192, 60);
    assert_eq!((console.cols(), console.rows()), (192, 60));
    let plain = console_draw(&model, None);

    for (cols, rows) in [(213u16, 60u16), (160, 49)] {
        let laid = layout::layout(&model, cols, rows);
        assert!(
            first_diff(&laid, &placed_like(&console, cols, rows)).is_some(),
            "{cols}x{rows} is a pad or a crop of the console"
        );
        let dir = scratch(&format!("{cols}x{rows}"));
        let path = dir.join("grid");
        let want = encoded(&dir, &laid);
        let tap = GridTap::spawn(
            GridWriter::new(&path, Palette::Llama),
            Capture::default(),
            Some(TtySize { cols, rows }),
        )
        .expect("tap");
        tap.offer_frame(&model, &console);
        wait_cells(&path, &want);
        assert_eq!(console_draw(&model, Some(TtySize { cols, rows })), plain);
        drop(tap);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn console_draw(model: &TtyModel, size: Option<TtySize>) -> Vec<u8> {
    let label = match size {
        None => "draw-unset".to_owned(),
        Some(size) => format!("draw-{}x{}", size.cols, size.rows),
    };
    let dir = scratch(&label);
    let now = Instant::now();
    let mut term = Term::new(
        Vec::new(),
        || {
            Ok(Size {
                cols: 192,
                rows: 60,
            })
        },
        Duration::from_secs(30),
        now,
    )
    .expect("term");
    if let Some(size) = size {
        let tap = GridTap::spawn(
            GridWriter::new(&dir.join("grid"), Palette::Llama),
            Capture::default(),
            Some(size),
        )
        .expect("tap");
        term = term.with_grid_tap(tap);
    }
    RenderStep::draw(&mut term, model, now).expect("draw");
    let bytes = term.out().to_vec();
    drop(term);
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn busy_frame() -> TtyModel {
    let prose = |tag: &str| {
        let line = format!(
            "{tag} the model writes a full line of ordinary prose about tokens and context. "
        );
        let mut text = String::new();
        for _ in 0..12 {
            text.push_str(&line);
            text.push('\n');
        }
        text.push_str("caf\u{e9} \u{2192} next");
        text
    };
    let slot = |id: u32, generating: bool| Slot {
        id,
        generating,
        done: 12_000 + u64::from(id) * 1_000,
        cached: 8_000,
        total: 20_000,
        open: false,
        done_known: true,
        decoded: 400 + u64::from(id),
        ctx_prompt: Some(20_000),
        n_ctx: Some(262_144),
        ctx_history: (0u32..24)
            .map(|i| CtxPoint {
                ms: 10_000,
                max: Some(10_000 + u64::from(i) * 500),
                reset: i == 3,
                reasons: 0,
            })
            .collect(),
    };
    let row = |id: u32, live: bool| Activity {
        live,
        id,
        time: format!("18:4{}:0{}", id % 6, id % 10),
        source: "local".to_owned(),
        model: "Model".to_owned(),
        input_tok: 20_000 + u64::from(id) * 100,
        cached_tok: 8_000,
        output_tok: 400 + u64::from(id),
        prompt_tps: Some(900.0 + f64::from(id)),
        gen_tps: Some(40.0 + f64::from(id)),
        dur: format!("{id}.2s"),
        n_ctx: Some(262_144),
        ..Activity::default()
    };
    TtyModel {
        state: WatchState::Generating,
        host: "box".to_owned(),
        model_name: "Model".to_owned(),
        model_detail: "llama.cpp".to_owned(),
        model_stuck: false,
        model_loading: false,
        model_stopping: false,
        gpu_busy: false,
        slots_line: "2/3 busy".to_owned(),
        swap_line: "none".to_owned(),
        cool_c: Some(34),
        cpu_c: Some(61),
        gpu_c: Some(72),
        clock: "2026-10-11 12:00:00".to_owned(),
        cpu_pct: Some(41.0),
        cpu_cores: Some(16),
        gpu_pct: Some(88.0),
        vram_used_gb: Some(20.5),
        vram_total_gb: Some(24.0),
        mem_used_gb: Some(40.0),
        mem_total_gb: Some(64.0),
        power_w: Some(280.0),
        power_limit_w: Some(350.0),
        load_pct: Some(70.0),
        activity_pct: Some(82.0),
        activity_src: Some(LoadSource::Gpu),
        activity_w: Some(280.0),
        gen_tps: Some(54.2),
        prompt_tps: Some(1200.0),
        prompt_last: Some(4096),
        gen_last: None,
        gen_ceiling: 250.0,
        prompt_ceiling: 1500.0,
        slots: vec![slot(0, true), slot(1, true), slot(2, false)],
        backend_lines: vec!["engine running".to_owned()],
        text_note: String::new(),
        requests: (0u32..8).map(|id| row(id, id == 0)).collect(),
        in_title: "IN  slot 0".to_owned(),
        out_title: "OUT slot 0".to_owned(),
        in_lines: vec![prose("IN")],
        out_lines: vec![prose("OUT")],
        out_shown: 0,
        replay_frame: None,
        show_text: true,
        down_since: String::new(),
        health: vec![
            HealthSeg {
                name: "llama-swap".to_owned(),
                status: HealthStatus::Ok,
                note: "3ms".to_owned(),
            },
            HealthSeg {
                name: "slots".to_owned(),
                status: HealthStatus::Ok,
                note: "14ms".to_owned(),
            },
            HealthSeg {
                name: "nvml".to_owned(),
                status: HealthStatus::Ok,
                note: String::new(),
            },
        ],
        snapshot: Some(48_213),
        snapshot_age: "0.1s".to_owned(),
        errors: 0,
        uptime: "3d04h".to_owned(),
        chart: (0u32..160)
            .map(|i| ChartBucket {
                gen_tps: Some(20.0 + f64::from(i % 17)),
                prompt_tps: Some(if i % 5 == 0 { 400.0 } else { 0.0 }),
            })
            .collect(),
        chart_bucket_s: 2,
        chart_glyphs: ChartGlyphs::Eighths,
        fans: None,
        temps: None,
        ctx_history_h: 6,
        setup: Some(SetupView {
            id: "model".to_owned(),
            name: "Model".to_owned(),
            more: 1,
            rows: vec![
                SetupRow {
                    label: "ctx".to_owned(),
                    items: vec![SetupItem {
                        sep: String::new(),
                        text: "262144".to_owned(),
                        dim: false,
                        text_value: false,
                    }],
                },
                SetupRow {
                    label: "kv".to_owned(),
                    items: vec![SetupItem {
                        sep: " · ".to_owned(),
                        text: "q8_0".to_owned(),
                        dim: false,
                        text_value: false,
                    }],
                },
            ],
        }),
        job: None,
        job_note: None,
        model_id: String::new(),
        claims: Vec::new(),
        now: UNIX_EPOCH,
        tz_offset_s: 0,
    }
}

/// Report the CPU of a second layout. It prints; it does not pass or fail
/// on the numbers. Ignored so `cargo test` does not spend a release build
/// on it. The report names the profile from `debug_assertions`. A debug
/// build says the numbers are not the release profile.
///
/// Run it on the shipped release profile (fat LTO, one codegen unit):
///
/// ```sh
/// cargo test -p llama-watch --locked --release --test grid_file -- --ignored measure_second_layout_cost --exact --nocapture
/// ```
#[ignore = "prints a timing report; run it by name, with --release for the shipped profile"]
#[test]
fn measure_second_layout_cost() {
    fn clk_tck() -> f64 {
        let out = std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .expect("getconf");
        let text = String::from_utf8_lossy(&out.stdout);
        text.trim().parse().expect("CLK_TCK")
    }
    fn proc_cpu() -> u64 {
        let stat = std::fs::read_to_string("/proc/self/stat").expect("stat");
        let rest = stat.split_once(')').expect("comm").1;
        let mut fields = rest.split_whitespace();
        let utime: u64 = fields.nth(11).expect("utime").parse().expect("utime");
        let stime: u64 = fields.next().expect("stime").parse().expect("stime");
        utime + stime
    }
    /// Median CPU seconds per call, and the matching wall seconds.
    fn median(hz: f64, mut body: impl FnMut()) -> (f64, f64) {
        let mut cpu_samples = Vec::new();
        let mut wall_samples = Vec::new();
        for _ in 0..3 {
            let cpu0 = proc_cpu();
            let wall0 = Instant::now();
            let mut n = 0u32;
            loop {
                body();
                n += 1;
                if n >= 50 && wall0.elapsed() >= Duration::from_secs(1) {
                    break;
                }
            }
            let ticks = proc_cpu() - cpu0;
            let cpu = ticks as f64 / hz / f64::from(n);
            let wall = wall0.elapsed().as_secs_f64() / f64::from(n);
            eprintln!("  sample n {n} ticks {ticks} cpu {cpu:.9}s wall {wall:.9}s");
            cpu_samples.push(cpu);
            wall_samples.push(wall);
        }
        cpu_samples.sort_by(f64::total_cmp);
        wall_samples.sort_by(f64::total_cmp);
        (cpu_samples[1], wall_samples[1])
    }
    fn report(label: &str, cpu_s: f64, wall_s: f64) {
        let us = cpu_s * 1_000_000.0;
        let pct = cpu_s * 10.0 * 100.0;
        eprintln!(
            "{label}: {us:.1} µs/frame, {pct:.4}% of one core at 10 fps (wall {:.1} µs)",
            wall_s * 1_000_000.0
        );
    }
    fn prose(chars: usize) -> String {
        let unit = "tokens and context stay inside the panel. ";
        let mut out = String::new();
        while out.len() < chars {
            out.push_str(unit);
        }
        out.truncate(chars);
        out
    }
    fn with_text(in_chars: usize, out_chars: usize) -> TtyModel {
        let mut model = busy_frame();
        model.in_lines = if in_chars == 0 {
            Vec::new()
        } else {
            vec![prose(in_chars)]
        };
        model.out_lines = if out_chars == 0 {
            Vec::new()
        } else {
            vec![prose(out_chars)]
        };
        model
    }

    let hz = clk_tck();
    eprintln!("measure_second_layout_cost");
    eprintln!(
        "profile: {}",
        if cfg!(debug_assertions) {
            "debug build. debug_assertions is on, so these numbers are not the release profile"
        } else {
            "release build. debug_assertions is off"
        }
    );
    eprintln!("CLK_TCK {hz}");
    eprintln!("frames: no IN/OUT text; 8192 characters of IN; IN and OUT at 32768");
    eprintln!("each figure is the median of three one-second runs, from utime+stime");

    for (name, inn, outn) in [
        ("no IN/OUT text", 0usize, 0usize),
        ("8192 characters of IN", 8192, 0),
        ("IN and OUT at 32768", 32768, 32768),
    ] {
        let model = with_text(inn, outn);
        let wide = layout::layout(&model, 213, 60);
        let copy = wide.clone();
        assert_eq!((wide.cols(), wide.rows()), (213, 60));
        assert_eq!(wide.cells().len(), 213 * 60);
        eprintln!(
            "{name}: in {inn} chars, out {outn} chars, 213x60 cells {}",
            wide.cells().len()
        );
        let (cpu, wall) = median(hz, || {
            std::hint::black_box(std::hint::black_box(&model).clone());
        });
        report("  clone", cpu, wall);
        let (cpu, wall) = median(hz, || {
            std::hint::black_box(layout::layout(std::hint::black_box(&model), 192, 60));
        });
        report("  layout 192x60", cpu, wall);
        let (cpu, wall) = median(hz, || {
            std::hint::black_box(layout::layout_exact(std::hint::black_box(&model), 192, 60));
        });
        report("  layout_exact 192x60", cpu, wall);
        let (cpu, wall) = median(hz, || {
            std::hint::black_box(layout::layout(std::hint::black_box(&model), 213, 60));
        });
        report("  layout 213x60", cpu, wall);
        let (cpu, wall) = median(hz, || {
            std::hint::black_box(std::hint::black_box(&wide) != std::hint::black_box(&copy));
        });
        report("  compare 213x60", cpu, wall);
    }
}
