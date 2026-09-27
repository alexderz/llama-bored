//! `kraken-lcd render-once` against the view fixture.
//!
//! These tests live outside `src/` so the fixture path can come from the
//! build without `std::env::var` in the scanned writer source.

mod common;

use std::path::PathBuf;
use std::process::Command;

fn bin() -> String {
    std::env::var("CARGO_BIN_EXE_kraken-lcd").expect("CARGO_BIN_EXE_kraken-lcd")
}

fn fixtures() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("fixtures");
    path
}

fn fixture_snapshot() -> String {
    fixtures()
        .join("views/working-hard.json")
        .display()
        .to_string()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "kraken-lcd-render-once-{}-{}",
        name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn render_once(args: &[String]) -> std::process::Output {
    let mut command = Command::new(bin());
    command.arg("render-once").args(args);
    command.output().expect("spawn kraken-lcd render-once")
}

#[test]
fn render_once_writes_a_png_of_the_view_fixture() {
    let out = scratch("png-out").join("frame.png");
    let output = render_once(&[
        "--snapshot".to_owned(),
        fixture_snapshot(),
        "--out".to_owned(),
        out.display().to_string(),
    ]);
    assert!(
        output.status.success(),
        "render-once failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).expect("png");
    let (width, height, corner) = common::rgba_png_pixel(&bytes, 0, 0).expect("corner");
    assert_eq!((width, height), (320, 320));
    assert_eq!(corner, [0, 0, 0, 255], "corner is opaque black");
    // 12 o'clock on the ring is 62.5 on the 0–125 scale. The fixture's 70 %
    // arc covers it in act_color magenta (#D044A8 → #F4466A), head-lightened.
    let (_, _, pixel) = common::rgba_png_pixel(&bytes, 160, 12).expect("ring pixel");
    assert!(
        pixel[0] > 0xC0 && pixel[2] > 0x90 && pixel[1] < 0x90 && pixel[3] == 255,
        "ring pixel {pixel:?} should be the magenta activity arc"
    );
}

#[test]
fn render_once_reports_a_snapshot_that_is_not_a_view() {
    let dir = scratch("bad-snapshot");
    let snapshot = dir.join("not-a-view.json");
    std::fs::write(&snapshot, b"{}").expect("fixture");
    let output = render_once(&[
        "--snapshot".to_owned(),
        snapshot.display().to_string(),
        "--out".to_owned(),
        dir.join("unused.png").display().to_string(),
    ]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let err = String::from_utf8(output.stderr).expect("utf8");
    assert!(err.contains("failed to parse snapshot"), "{err}");
}

#[test]
fn render_once_refuses_a_symlink_out() {
    let dir = scratch("symlink-out");
    let kept = dir.join("kept.png");
    std::fs::write(&kept, b"keep").expect("target");
    let link = dir.join("out.png");
    std::os::unix::fs::symlink(&kept, &link).expect("symlink");
    let output = render_once(&[
        "--snapshot".to_owned(),
        fixture_snapshot(),
        "--out".to_owned(),
        link.display().to_string(),
    ]);
    assert_eq!(
        std::fs::read(&kept).expect("target still there"),
        b"keep",
        "a symlink out path must not be followed"
    );
    assert_eq!(output.status.code(), Some(2), "{output:?}");
}

#[test]
fn render_once_refuses_dev_null() {
    let output = render_once(&[
        "--snapshot".to_owned(),
        fixture_snapshot(),
        "--out".to_owned(),
        "/dev/null".to_owned(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
}
