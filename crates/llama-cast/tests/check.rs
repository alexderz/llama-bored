//! The text `llama-cast check` prints, and the line `run` logs at start.
//! Neither test opens the machine id or spawns ffmpeg.

mod common;

use llama_cast::config::{Config, Font};
use llama_cast::encoder::Codec;
use llama_cast::font::Psf2;
use llama_cast::service::{cast_banner, check_output, font_is_auto};
use llama_core::color::Rgb;
use llama_core::gridfile::{self, GridCell, GridImage};

fn psf() -> Psf2 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x864A_B572_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&[0_u8; 8]);
    Psf2::parse(&bytes).unwrap()
}

fn config(extra: &str) -> Config {
    Config::from_toml(&format!(
        r#"
listen = "127.0.0.1:19478"
allow = ["127.0.0.1/32"]
name = "cast"
title = "live"
fps = 2
max_clients = 1
bitrate_kbps = 4000
keyframe_s = 1
preroll_s = 0
ffmpeg = "/usr/bin/ffmpeg"
{extra}"#
    ))
    .unwrap()
}

fn grid_file(dir: &std::path::Path, cols: u16, rows: u16) -> std::path::PathBuf {
    let path = dir.join("grid");
    let n = usize::from(cols) * usize::from(rows);
    let image = GridImage {
        seq: 1,
        unix_ms: 1_000,
        cols,
        rows,
        cells: vec![
            GridCell {
                ch: 'A',
                fg: Rgb { r: 1, g: 2, b: 3 },
                bg: Rgb { r: 4, g: 5, b: 6 },
            };
            n
        ],
    };
    let mut bytes = Vec::new();
    gridfile::encode(&image, &mut bytes).unwrap();
    std::fs::write(&path, bytes).unwrap();
    path
}

fn text(lines: &[String]) -> String {
    lines.join("\n")
}

#[test]
fn check_text_for_tty_mode() {
    let output = check_output(
        &config(""),
        "host",
        &[(Font::Hack12x24, psf())],
        Some(Codec::Libx264),
    );
    let body = text(&output.lines);
    assert!(body.contains("source tty"), "{body}");
    assert!(body.contains("font 12x24"), "{body}");
    assert!(
        output
            .lines
            .iter()
            .all(|line| !line.starts_with("grid ") && !line.starts_with("note:"))
    );
    assert!(output.lines.iter().any(|line| line == "encoder: libx264"));
    assert_eq!(output.status, 0);
    let banner = cast_banner(&config(""), "screen");
    assert!(
        banner.starts_with("casting tty11 as \"screen\": "),
        "{banner}"
    );
    assert!(banner.contains("source tty"), "{banner}");
}

#[test]
fn check_text_for_grid_mode_with_a_set_font() {
    let dir = common::scratch("check-font");
    let path = grid_file(&dir, 1, 1);
    let cfg = config(&format!(
        "source = \"grid\"\nfont = \"9x18\"\ngrid = \"{}\"\n",
        path.display()
    ));
    let output = check_output(
        &cfg,
        "host",
        &[(Font::Hack9x18, psf())],
        Some(Codec::Libx264),
    );
    let body = text(&output.lines);
    assert!(body.contains("source grid"), "{body}");
    assert!(body.contains("font 9x18"), "{body}");
    assert!(
        output.lines.iter().any(|line| line == "grid 1x1 font 9x18"),
        "{body}"
    );
    assert_eq!(output.status, 0);
    let banner = cast_banner(&cfg, "screen");
    assert!(
        banner.starts_with(&format!(
            "casting grid {} as \"screen\": ",
            cfg.grid.display()
        )),
        "{banner}"
    );
}

#[test]
fn check_text_for_grid_mode_auto_with_a_readable_grid() {
    let dir = common::scratch("check-auto");
    let path = grid_file(&dir, 213, 60);
    let cfg = config(&format!(
        "source = \"grid\"\ngrid = \"{}\"\n",
        path.display()
    ));
    let font = psf();
    let fonts: Vec<_> = Font::SHIPPED
        .into_iter()
        .map(|kind| (kind, font.clone()))
        .collect();
    let output = check_output(&cfg, "host", &fonts, Some(Codec::Libx264));
    let body = text(&output.lines);
    assert!(body.contains("source grid"), "{body}");
    assert!(body.contains("font auto"), "{body}");
    assert!(
        output
            .lines
            .iter()
            .any(|line| line == "grid 213x60 font auto (9x18)"),
        "{body}"
    );
    assert_eq!(output.status, 0);
}

#[test]
fn check_text_for_an_unreadable_grid_is_a_note_and_success() {
    let dir = common::scratch("check-bad");
    let path = dir.join("grid");
    std::fs::write(&path, b"not a grid").unwrap();
    let cfg = config(&format!(
        "source = \"grid\"\ngrid = \"{}\"\n",
        path.display()
    ));
    let font = psf();
    let fonts: Vec<_> = Font::SHIPPED
        .into_iter()
        .map(|kind| (kind, font.clone()))
        .collect();
    let output = check_output(&cfg, "host", &fonts, Some(Codec::Libx264));
    let body = text(&output.lines);
    assert!(
        output.lines.iter().any(|line| line.starts_with("note: ")),
        "{body}"
    );
    assert!(
        output.lines.iter().any(|line| line == "encoder: libx264"),
        "{body}"
    );
    assert_eq!(output.status, 0);
}

#[test]
fn run_takes_auto_from_the_config() {
    let src = include_str!("../src/service.rs");
    let body = src.split_once("fn run(").expect("fn run").1;
    assert!(
        body.contains("let auto = font_is_auto(config);"),
        "run must not force auto off"
    );
    assert!(
        !body.contains("let auto = false"),
        "run must not force auto off"
    );
    assert!(font_is_auto(&config("source = \"grid\"\n")));
    assert!(!font_is_auto(&config(
        "source = \"grid\"\nfont = \"9x18\"\n"
    )));
}
