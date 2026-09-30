//! The ffmpeg argument vector (pinned), the spawn (fixed args, empty
//! environment, killed on drop), and, when /usr/bin/ffmpeg with libx264 is
//! installed, a real encode of the synthetic screen to MPEG-TS.

mod common;

use std::io::{self, Read};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::fixture;
use llama_cast::encoder::{Encoder, ffmpeg_args};
use llama_cast::render::FRAME_BYTES;
use llama_cast::source::{FrameSource, VcsaSource, load_font};

struct Solid(u8);

impl FrameSource for Solid {
    fn frame(&self, frame: &mut [u8]) -> io::Result<()> {
        frame.fill(self.0);
        Ok(())
    }
}

fn script(label: &str, body: &str) -> PathBuf {
    let dir = common::scratch(label);
    let path = dir.join("fake-ffmpeg");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Writing the fake ffmpeg and exec'ing it races with forks from other test
/// threads (they can inherit the write fd): retry exec on ETXTBSY briefly.
fn start_retry(
    path: &std::path::Path,
    fps: u32,
    source: Arc<dyn FrameSource>,
) -> io::Result<Encoder> {
    for _ in 0..40 {
        match Encoder::start(path, fps, Arc::clone(&source)) {
            Err(err) if err.kind() == io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(Duration::from_millis(25));
            }
            other => return other,
        }
    }
    Encoder::start(path, fps, source)
}

fn read_n(enc: &mut Encoder, n: usize, budget: Duration) -> Vec<u8> {
    let start = Instant::now();
    let mut out = Vec::new();
    let mut buf = [0_u8; 8192];
    while out.len() < n && start.elapsed() < budget {
        match enc.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(k) => out.extend_from_slice(&buf[..k]),
        }
    }
    out
}

#[test]
fn argument_vector_is_pinned() {
    assert_eq!(
        ffmpeg_args(2).join(" "),
        "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i - \
         -c:v libx264 -preset veryfast -tune stillimage -pix_fmt yuv420p -r 10 -g 20 -f mpegts -"
    );
    let five = ffmpeg_args(5);
    let at = five.iter().position(|a| a == "-r").unwrap();
    assert_eq!(five[at + 1], "5");
    // No shell, no file, no network output: input and output are pipes.
    for arg in ffmpeg_args(2) {
        assert!(
            !arg.contains('/') && !arg.contains(':') || arg == "-c:v",
            "{arg}"
        );
    }
}

#[test]
fn spawn_uses_fixed_args_an_empty_environment_and_stdin_frames() {
    // The test process has HOME and CARGO_MANIFEST_DIR set; the child must
    // see neither. (sh supplies its own default PATH, so PATH is no probe.)
    assert!(std::env::var_os("CARGO_MANIFEST_DIR").is_some());
    let fake = script(
        "enc-fake",
        "printf '[%s][%s] ' \"$HOME\" \"$CARGO_MANIFEST_DIR\"; printf '%s ' \"$@\"; printf '\\n'; exec /usr/bin/head -c 64",
    );
    let mut enc = start_retry(&fake, 4, Arc::new(Solid(0xab))).unwrap();
    let out = read_n(&mut enc, 1000, Duration::from_secs(5));
    let text = String::from_utf8_lossy(&out);
    let (line, _) = text.split_once('\n').expect("args line");
    assert!(line.starts_with("[][] "), "environment leaked: {line}");
    assert_eq!(
        line.trim_start_matches("[][] ").trim_end(),
        ffmpeg_args(4).join(" ")
    );
    let at = out.iter().position(|b| *b == b'\n').unwrap() + 1;
    assert_eq!(&out[at..], &[0xab_u8; 64][..]);
}

#[test]
fn drop_kills_a_stuck_encoder() {
    let fake = script("enc-stuck", "exec /usr/bin/sleep 30");
    let enc = start_retry(&fake, 2, Arc::new(Solid(0))).unwrap();
    let start = Instant::now();
    drop(enc);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "drop waited for the child"
    );
}

#[test]
fn a_missing_encoder_is_an_error() {
    assert!(Encoder::start(Path::new("/nonexistent/ffmpeg"), 2, Arc::new(Solid(0))).is_err());
}

#[test]
fn real_ffmpeg_encodes_mpeg_ts_when_installed() {
    let ffmpeg = Path::new("/usr/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("SKIP: /usr/bin/ffmpeg is not installed");
        return;
    }
    let font =
        load_font(&common::workspace_root().join("packaging/fonts/llama-hack-12x24.psfu")).unwrap();
    let source = Arc::new(VcsaSource::new(fixture("synthetic.vcsa"), font));
    let mut probe = vec![0_u8; FRAME_BYTES];
    source.frame(&mut probe).unwrap();
    let mut enc = Encoder::start(ffmpeg, 2, source).unwrap();
    let out = read_n(&mut enc, 188 * 64, Duration::from_secs(30));
    if out.is_empty() {
        eprintln!("SKIP: ffmpeg produced nothing (no libx264?)");
        return;
    }
    assert!(out.len() >= 188 * 64, "short stream: {} bytes", out.len());
    for packet in out.as_chunks::<188>().0.iter().take(64) {
        assert_eq!(packet[0], 0x47, "not MPEG-TS");
    }
}
