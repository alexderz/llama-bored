//! #25: a small tty11-like vcsa (512-glyph attribute layout, every
//! foreground and background) rendered in each colour mode, byte for byte.
//!
//! Regenerate after a deliberate change with
//! `LLAMA_VIEW_BLESS=1 cargo test -p llama-view --test colors_golden`, then
//! read the diff of crates/llama-view/tests/golden/ before committing it.

use std::path::PathBuf;
use std::process::Command;

use llama_view::{
    AttrLayout, ColorMode, Palette, Renderer, decode_screen, detect_layout, nearest_xterm256,
};

/// 2 x 16: row 0 is `A`..`P` in VGA foreground 0..=15 on black; row 1 is
/// eight spaces on VGA background 0..=7 (white foreground), then `bg` and
/// six spaces. A 512-glyph font shifts every attribute left one bit.
fn fixture() -> Vec<u8> {
    let mut bytes = vec![2u8, 16, 0, 0];
    for fg in 0..16u8 {
        bytes.extend_from_slice(&[b'A' + fg, fg << 1]);
    }
    for bg in 0..8u8 {
        bytes.extend_from_slice(&[b' ', (bg << 5) | (7 << 1)]);
    }
    for ch in *b"bg      " {
        bytes.extend_from_slice(&[ch, 7 << 1]);
    }
    bytes
}

fn render(mode: ColorMode, palette: Palette) -> Vec<u8> {
    let vcsa = fixture();
    let screen = decode_screen(&vcsa, None, None).expect("fixture decodes");
    let mut renderer = Renderer::with_colors(mode, palette);
    renderer
        .render(screen.cols, screen.rows, &screen.cells)
        .to_vec()
}

fn golden(name: &str, got: &[u8]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name);
    if std::env::var_os("LLAMA_VIEW_BLESS").is_some() {
        std::fs::write(&path, got).expect("bless");
    }
    let want = std::fs::read(&path).expect("golden (LLAMA_VIEW_BLESS=1 to create)");
    assert!(
        want == got,
        "{name} differs:\nwant {:?}\ngot  {:?}",
        String::from_utf8_lossy(&want),
        String::from_utf8_lossy(got)
    );
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn the_fixture_reads_as_a_512_glyph_screen() {
    assert_eq!(detect_layout(&fixture(), None), Some(AttrLayout::Glyphs512));
}

#[test]
fn truecolor_llama_golden() {
    let got = render(ColorMode::Truecolor, Palette::Llama);
    // L1 blue (row 0, VGA 1) and the L6 red error slot (VGA 4).
    assert!(contains(&got, b"\x1b[38;2;20;40;216mB"));
    assert!(contains(&got, b"\x1b[38;2;255;42;20mE"));
    golden("colors-truecolor-llama.ans", &got);
}

#[test]
fn truecolor_vga_golden() {
    golden(
        "colors-truecolor-vga.ans",
        &render(ColorMode::Truecolor, Palette::Vga),
    );
}

#[test]
fn xterm256_llama_golden() {
    let got = render(ColorMode::Xterm256, Palette::Llama);
    for n in 0..16 {
        let themed = format!("\x1b[38;5;{n}m");
        assert!(
            !got.windows(themed.len()).any(|w| w == themed.as_bytes()),
            "themed index {n}"
        );
    }
    let blue = nearest_xterm256(Palette::Llama.rgb(4));
    assert!(contains(&got, format!("\x1b[38;5;{blue}mB").as_bytes()));
    golden("colors-256-llama.ans", &got);
}

#[test]
fn ansi16_golden_is_todays_output() {
    let got = render(ColorMode::Ansi16, Palette::Llama);
    assert_eq!(got, render(ColorMode::Ansi16, Palette::Vga));
    golden("colors-16.ans", &got);
}

/// The binary reads `LLAMA_VIEW_COLORS`, and `--colors` beats it.
#[test]
fn binary_takes_the_env_and_the_flag_wins() {
    let dir = std::env::temp_dir().join(format!("llama-view-colors-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let vcsa = dir.join("vcsa11");
    std::fs::write(&vcsa, fixture()).expect("fixture");
    let run = |extra: &[&str], env: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_llama-view"))
            .args(["--device", vcsa.to_str().expect("utf-8"), "--once"])
            .args(extra)
            .env_remove("COLORTERM")
            .env_remove("LLAMA_VIEW_PALETTE")
            .env("LLAMA_VIEW_COLORS", env)
            .output()
            .expect("run");
        assert!(out.status.success(), "{out:?}");
        out.stdout
    };
    let has = |out: &[u8], needle: &[u8]| out.windows(needle.len()).any(|w| w == needle);
    let env_true = run(&[], "truecolor");
    assert!(has(&env_true, b"\x1b[38;2;20;40;216m"), "env truecolor");
    let flag_16 = run(&["--colors", "16"], "truecolor");
    assert!(!has(&flag_16, b"38;2;"), "flag 16 beats env");
    assert!(has(&flag_16, b"\x1b[34m"), "VGA blue as SGR 34");
    let bad = Command::new(env!("CARGO_BIN_EXE_llama-view"))
        .args(["--device", vcsa.to_str().expect("utf-8"), "--once"])
        .env("LLAMA_VIEW_COLORS", "lots")
        .output()
        .expect("run");
    assert_eq!(bad.status.code(), Some(2));
    assert!(
        bad.stdout.is_empty(),
        "refused before touching the terminal"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
