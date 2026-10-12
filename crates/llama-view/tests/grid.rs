//! #107: the grid source. A grid file's original characters and exact RGB
//! through the same renderer the vcsa path uses; the CLI; and the binary's
//! behaviour on a good, missing, bad and stale file, with and without
//! `--kiosk`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use llama_core::color::Rgb;
use llama_core::gridfile::{GridCell, GridImage, encode};
use llama_view::{
    CliError, Color, ColorMode, ColorSettings, Fit, Options, Palette, Renderer, Session, SgrTable,
    SourceKind, nearest_xterm256, now_unix_ms, parse_args, place, screen_of, source_kind,
};

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn image_of(text: &str, fg: Rgb, bg: Rgb, ms: u64) -> GridImage {
    let cells: Vec<GridCell> = text.chars().map(|ch| GridCell { ch, fg, bg }).collect();
    GridImage {
        seq: 1,
        unix_ms: ms,
        cols: u16::try_from(cells.len()).expect("short"),
        rows: 1,
        cells,
    }
}

fn session(mode: ColorMode) -> Session {
    let colors = ColorSettings {
        mode,
        palette: Palette::Llama,
    };
    Session::new(&Options::default(), colors, "box", false, false)
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("grid-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    dir
}

fn write_grid(path: &Path, image: &GridImage) {
    let mut bytes = Vec::new();
    encode(image, &mut bytes).expect("encode");
    std::fs::write(path, bytes).expect("write");
}

const ORANGE: Rgb = Rgb {
    r: 255,
    g: 138,
    b: 30,
};
const TEAL: Rgb = Rgb {
    r: 10,
    g: 99,
    b: 120,
};

#[test]
fn truecolor_sends_the_cell_rgb_and_non_console_glyphs_pass() {
    let image = image_of("caf\u{e9} \u{2192} \u{d7} \u{44f}", ORANGE, TEAL, 0);
    let mut out = Vec::new();
    session(ColorMode::Truecolor).frame_grid((40, 3), &image, &mut out);
    assert!(contains(&out, b"\x1b[38;2;255;138;30m"), "{out:?}");
    assert!(contains(&out, b"\x1b[48;2;10;99;120m"));
    assert!(contains(
        &out,
        "caf\u{e9} \u{2192} \u{d7} \u{44f}".as_bytes()
    ));
}

#[test]
fn an_unchanged_grid_writes_nothing_the_second_time() {
    let image = image_of("hello", ORANGE, TEAL, 0);
    let mut s = session(ColorMode::Truecolor);
    let mut out = Vec::new();
    s.frame_grid((40, 3), &image, &mut out);
    out.clear();
    s.frame_grid((40, 3), &image, &mut out);
    assert!(out.is_empty(), "{out:?}");
}

/// An RGB equal to a palette slot gives, in every mode and both palettes,
/// exactly the sequence the vcsa path gives for that slot.
#[test]
fn slot_equal_rgb_matches_the_vcsa_path_in_every_mode() {
    for palette in [Palette::Llama, Palette::Vga] {
        for mode in [ColorMode::Truecolor, ColorMode::Xterm256, ColorMode::Ansi16] {
            let table = SgrTable::new(mode, palette);
            for index in 0..16u8 {
                let slot = palette.rgb(index);
                let rgb = Color::Rgb(slot.r, slot.g, slot.b);
                let color = Color::from_ansi(index);
                let (mut f, mut b) = (Vec::new(), Vec::new());
                table.push_fg(&mut f, rgb);
                table.push_bg(&mut b, rgb);
                assert_eq!(f, table.fg(color), "{palette:?} {mode:?} fg {index}");
                assert_eq!(b, table.bg(color), "{palette:?} {mode:?} bg {index}");
            }
        }
    }
}

#[test]
fn other_rgb_maps_to_the_nearest_colour() {
    let rgb = Color::Rgb(ORANGE.r, ORANGE.g, ORANGE.b);
    let near = Color::Rgb(250, 140, 28);
    let mut out = Vec::new();
    SgrTable::new(ColorMode::Xterm256, Palette::Llama).push_fg(&mut out, near);
    assert_eq!(
        out,
        format!(
            "\x1b[38;5;{}m",
            nearest_xterm256(Rgb {
                r: 250,
                g: 140,
                b: 28
            })
        )
        .as_bytes()
    );
    // 16 colours: the nearest palette slot, here slot 6 (the orange cyan).
    let table16 = SgrTable::new(ColorMode::Ansi16, Palette::Llama);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    table16.push_fg(&mut a, near);
    table16.push_fg(&mut b, rgb);
    assert_eq!(a, b"\x1b[36m");
    assert_eq!(a, b);
    // Truecolor: as is.
    let mut t = Vec::new();
    SgrTable::new(ColorMode::Truecolor, Palette::Llama).push_bg(&mut t, near);
    assert_eq!(t, b"\x1b[48;2;250;140;28m");
}

/// A grid of the vcsa fixture's colours renders to the same bytes as the
/// vcsa cells do, in each mode (the characters are plain ASCII).
#[test]
fn a_grid_of_palette_colours_renders_like_the_vcsa_cells() {
    for mode in [ColorMode::Truecolor, ColorMode::Xterm256, ColorMode::Ansi16] {
        let palette = Palette::Llama;
        let mut grid_cells = Vec::new();
        let mut vcsa_cells = Vec::new();
        for index in 0..16u8 {
            let fg = palette.rgb(index);
            let bg = palette.rgb(index & 7);
            let ch = char::from(b'A' + index);
            grid_cells.push(GridCell { ch, fg, bg });
            vcsa_cells.push(llama_view::Cell::new(
                ch,
                Color::from_ansi(index),
                Color::from_ansi(index & 7),
            ));
        }
        let image = GridImage {
            seq: 1,
            unix_ms: 0,
            cols: 16,
            rows: 1,
            cells: grid_cells,
        };
        let screen = screen_of(&image);
        let a = Renderer::with_colors(mode, palette)
            .render(16, 1, &screen.cells)
            .to_vec();
        let b = Renderer::with_colors(mode, palette)
            .render(16, 1, &vcsa_cells)
            .to_vec();
        assert_eq!(a, b, "{mode:?}");
    }
}

#[test]
fn fit_center_and_the_hint_work_on_a_grid() {
    let image = GridImage {
        rows: 4,
        cols: 60,
        cells: vec![
            GridCell {
                ch: 'x',
                fg: ORANGE,
                bg: TEAL
            };
            240
        ],
        seq: 1,
        unix_ms: 0,
    };
    let opts = Options {
        fit: Fit::Center,
        ..Options::default()
    };
    let colors = ColorSettings {
        mode: ColorMode::Truecolor,
        palette: Palette::Llama,
    };
    let mut s = Session::new(&opts, colors, "box", false, false);
    let mut out = Vec::new();
    s.frame_grid((50, 3), &image, &mut out);
    assert!(contains(&out, b"middle shown"), "{out:?}");
}

#[test]
fn cli_source_flags() {
    let parse =
        |args: &[&str]| parse_args(std::iter::once("llama-view").chain(args.iter().copied()));
    let opts = parse(&[]).expect("defaults");
    assert_eq!(opts.source, None);
    assert_eq!(source_kind(&opts, no_env), Ok(SourceKind::Vcsa));
    assert_eq!(
        opts.grid_path(),
        PathBuf::from("/run/llama-watch-grid/grid")
    );

    let opts = parse(&["--source", "grid"]).expect("grid");
    assert_eq!(source_kind(&opts, no_env), Ok(SourceKind::Grid));

    let opts = parse(&["--grid", "/tmp/x/grid"]).expect("grid path");
    assert_eq!(source_kind(&opts, no_env), Ok(SourceKind::Grid), "implied");
    assert_eq!(opts.grid_path(), PathBuf::from("/tmp/x/grid"));

    let env_grid = |key: &str| (key == "LLAMA_VIEW_SOURCE").then(|| "grid".to_string());
    let plain = parse(&[]).expect("plain");
    assert_eq!(source_kind(&plain, env_grid), Ok(SourceKind::Grid));
    let flag_wins = parse(&["--source", "vcsa"]).expect("vcsa");
    assert_eq!(source_kind(&flag_wins, env_grid), Ok(SourceKind::Vcsa));
    let bad_env = |key: &str| (key == "LLAMA_VIEW_SOURCE").then(|| "tty".to_string());
    assert!(
        matches!(source_kind(&plain, bad_env), Err(CliError::Message(m)) if m.contains("LLAMA_VIEW_SOURCE"))
    );

    for bad in [
        &["--source", "tty"][..],
        &["--source"],
        &["--grid"],
        &["--source", "vcsa", "--grid", "/x"],
    ] {
        assert!(matches!(parse(bad), Err(CliError::Message(_))), "{bad:?}");
    }
}

fn run_once(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llama-view"))
        .args(args)
        .arg("--once")
        .stdin(Stdio::null())
        .output()
        .expect("run")
}

#[test]
fn the_binary_paints_a_grid_file_once() {
    let dir = scratch("once");
    let path = dir.join("grid");
    write_grid(&path, &image_of("caf\u{e9} \u{2192}", ORANGE, TEAL, 0));
    // Not kiosk: a stale file (stamp 0) is shown as it is.
    let out = run_once(&["--grid", path.to_str().unwrap(), "--colors", "truecolor"]);
    assert!(out.status.success(), "{out:?}");
    assert!(contains(&out.stdout, "caf\u{e9} \u{2192}".as_bytes()));
    assert!(contains(&out.stdout, b"\x1b[38;2;255;138;30m"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_kiosk_a_missing_or_bad_file_exits_1_before_painting() {
    let dir = scratch("errs");
    let missing = dir.join("absent");
    let out = run_once(&["--source", "grid", "--grid", missing.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty(), "terminal untouched");
    assert!(String::from_utf8_lossy(&out.stderr).contains("absent"));

    let bad = dir.join("bad");
    std::fs::write(&bad, b"definitely not a grid file, no").unwrap();
    let out = run_once(&["--grid", bad.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Run `--kiosk` for a moment and return what it wrote; it never exits by itself.
fn kiosk_output(args: &[&str]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-view"))
        .arg("--kiosk")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdout = child.stdout.take().expect("stdout");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&seen);
    let reader = std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        while let Ok(n @ 1..) = stdout.read(&mut chunk) {
            writer.lock().expect("lock").extend_from_slice(&chunk[..n]);
        }
    });
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        child.try_wait().expect("try_wait").is_none(),
        "kiosk exited"
    );
    child.kill().expect("kill");
    let _ = child.wait();
    let _ = reader.join();
    let bytes = seen.lock().expect("lock").clone();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn kiosk_waits_on_a_missing_file() {
    let dir = scratch("k-missing");
    let path = dir.join("absent");
    let out = kiosk_output(&["--grid", path.to_str().unwrap()]);
    assert!(out.contains("waiting for grid"), "{out:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kiosk_waits_on_a_bad_file() {
    let dir = scratch("k-bad");
    let path = dir.join("grid");
    std::fs::write(&path, b"garbage garbage garbage garbage").unwrap();
    let out = kiosk_output(&["--grid", path.to_str().unwrap()]);
    assert!(out.contains("waiting for grid"), "{out:?}");
    assert!(out.contains("bad magic"), "{out:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kiosk_calls_an_old_file_stale_and_shows_a_fresh_one() {
    let dir = scratch("k-stale");
    let path = dir.join("grid");
    write_grid(
        &path,
        &image_of("caf\u{e9} live", ORANGE, TEAL, now_unix_ms() - 17_000),
    );
    let out = kiosk_output(&["--grid", path.to_str().unwrap(), "--colors", "truecolor"]);
    assert!(out.contains("no update for 17 s"), "{out:?}");
    assert!(!out.contains("live"), "a stale grid is not drawn: {out:?}");

    write_grid(
        &path,
        &image_of("caf\u{e9} live", ORANGE, TEAL, now_unix_ms()),
    );
    let out = kiosk_output(&["--grid", path.to_str().unwrap(), "--colors", "truecolor"]);
    assert!(out.contains("caf\u{e9} live"), "{out:?}");
    assert!(!out.contains("waiting"), "{out:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// #115: llama-view takes the size from the file. A 213x60 grid shown in a
/// 192x60 pane says so, and a 160x49 grid is not treated as a 192-wide console.
#[test]
fn a_grid_file_is_shown_at_its_own_size() {
    let mark = GridCell {
        ch: '\u{00b6}',
        fg: ORANGE,
        bg: TEAL,
    };
    let blank = GridCell {
        ch: ' ',
        fg: ORANGE,
        bg: TEAL,
    };
    let mut wide_cells = vec![blank; 213 * 60];
    wide_cells[200] = mark;
    let wide = GridImage {
        seq: 1,
        unix_ms: 0,
        cols: 213,
        rows: 60,
        cells: wide_cells,
    };
    let screen = screen_of(&wide);
    assert_eq!((screen.cols, screen.rows), (213, 60));

    let mut out = Vec::new();
    session(ColorMode::Truecolor).frame_grid((213, 60), &wide, &mut out);
    assert!(
        contains(&out, "\u{00b6}".as_bytes()),
        "the mark is on a pane of the file's own size"
    );

    let mut cropped = Vec::new();
    session(ColorMode::Truecolor).frame_grid((192, 60), &wide, &mut cropped);
    let text = String::from_utf8_lossy(&cropped);
    assert!(text.contains("tty11 is 213x60, this pane 192x60"), "{text}");
    assert!(
        !text.contains('\u{00b6}'),
        "column 200 is past a 192-column pane"
    );

    let mut narrow_cells = vec![blank; 160 * 49];
    narrow_cells[159] = mark;
    let narrow = GridImage {
        seq: 1,
        unix_ms: 0,
        cols: 160,
        rows: 49,
        cells: narrow_cells,
    };
    let mut pane = Vec::new();
    session(ColorMode::Truecolor).frame_grid((192, 60), &narrow, &mut pane);
    let text = String::from_utf8_lossy(&pane);
    assert!(
        text.contains('\u{00b6}'),
        "the file's last column is drawn; the pane is wider than the file"
    );
    assert!(
        !text.contains("tty11 is"),
        "a larger pane does not rename the file: {text}"
    );

    let mut small = Vec::new();
    session(ColorMode::Truecolor).frame_grid((80, 24), &narrow, &mut small);
    let text = String::from_utf8_lossy(&small);
    assert!(text.contains("tty11 is 160x49, this pane 80x24"), "{text}");
}

fn session_fit(fit: Fit) -> Session {
    let opts = Options {
        fit,
        ..Options::default()
    };
    let colors = ColorSettings {
        mode: ColorMode::Truecolor,
        palette: Palette::Llama,
    };
    Session::new(&opts, colors, "box", false, false)
}

/// #115: `--fit center` uses the file's own size too. A wider file shows its
/// middle, and a narrower file is centred in the pane.
#[test]
fn a_grid_file_centres_when_fit_is_center() {
    let edge = GridCell {
        ch: '\u{00b6}',
        fg: ORANGE,
        bg: TEAL,
    };
    let inner = GridCell {
        ch: '\u{2020}',
        fg: ORANGE,
        bg: TEAL,
    };
    let blank = GridCell {
        ch: ' ',
        fg: ORANGE,
        bg: TEAL,
    };
    let mut wide_cells = vec![blank; 213 * 60];
    wide_cells[0] = edge;
    wide_cells[10] = inner;
    let wide = GridImage {
        seq: 1,
        unix_ms: 0,
        cols: 213,
        rows: 60,
        cells: wide_cells,
    };
    let centred = place(&screen_of(&wide), 192, 60, 0, 0, Fit::Center, 11);
    assert_eq!(
        centred.cells[0].ch, '\u{2020}',
        "column 10 of the file is the pane's first column"
    );
    assert!(
        centred.cells.iter().all(|cell| cell.ch != '\u{00b6}'),
        "the file's first column is outside a centred 192-wide pane"
    );
    let hint: String = centred.cells[centred.cells.len() - 192..]
        .iter()
        .map(|cell| cell.ch)
        .collect();
    assert!(hint.contains("tty11 is 213x60, this pane 192x60"), "{hint}");
    assert!(hint.contains("middle shown"), "{hint}");

    let mut out = Vec::new();
    session_fit(Fit::Center).frame_grid((192, 60), &wide, &mut out);
    let text = String::from_utf8_lossy(&out);
    assert!(text.contains("middle shown"), "{text}");
    assert!(text.contains('\u{2020}'), "{text}");
    assert!(!text.contains('\u{00b6}'), "{text}");

    let mut narrow_cells = vec![blank; 160 * 49];
    narrow_cells[0] = edge;
    let narrow = GridImage {
        seq: 1,
        unix_ms: 0,
        cols: 160,
        rows: 49,
        cells: narrow_cells,
    };
    let centred = place(&screen_of(&narrow), 192, 60, 0, 0, Fit::Center, 11);
    assert_eq!(
        centred.cells.len(),
        192 * 60,
        "a larger pane has no hint row"
    );
    assert_ne!(centred.cells[0].ch, '\u{00b6}');
    assert_eq!(centred.cells[5 * 192 + 16].ch, '\u{00b6}');

    let mut pane = Vec::new();
    session_fit(Fit::Center).frame_grid((192, 60), &narrow, &mut pane);
    let text = String::from_utf8_lossy(&pane);
    assert!(text.contains('\u{00b6}'), "{text}");
    assert!(
        !text.contains("tty11 is"),
        "a larger pane draws the file: {text}"
    );
}
