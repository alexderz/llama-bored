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
use llama_cast::encoder::{Encoder, Settings, ffmpeg_args};
use llama_cast::render::FRAME_BYTES;
use llama_cast::source::{FrameSource, VcsaSource, load_font};

/// The shipped defaults (cast.toml without the stream keys).
const DEFAULTS: Settings = Settings {
    fps: 2,
    bitrate_kbps: 4000,
    keyframe_s: 1,
    preroll_s: 3,
};

fn with_fps(fps: u32) -> Settings {
    Settings { fps, ..DEFAULTS }
}

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
    settings: Settings,
    source: Arc<dyn FrameSource>,
) -> io::Result<Encoder> {
    for _ in 0..40 {
        match Encoder::start(path, settings, Arc::clone(&source)) {
            Err(err) if err.kind() == io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(Duration::from_millis(25));
            }
            other => return other,
        }
    }
    Encoder::start(path, settings, source)
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
    // #20: low latency (zerolatency: no B-frames, no lookahead), a 1 s
    // keyframe interval, CBR with filler so a still screen still fills the
    // TV's buffer, packets flushed as muxed, PAT/PMT every 100 ms.
    assert_eq!(
        ffmpeg_args(DEFAULTS).join(" "),
        "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i - \
         -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p -r 10 -g 10 -bf 0 \
         -b:v 4000k -minrate 4000k -maxrate 4000k -bufsize 4000k -x264-params nal-hrd=cbr \
         -flush_packets 1 -pat_period 0.1 -f mpegts -"
    );
    // bitrate_kbps = 0: no rate options, x264 is quality-based.
    assert_eq!(
        ffmpeg_args(Settings {
            bitrate_kbps: 0,
            keyframe_s: 2,
            ..DEFAULTS
        })
        .join(" "),
        "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i - \
         -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p -r 10 -g 20 -bf 0 \
         -flush_packets 1 -pat_period 0.1 -f mpegts -"
    );
    let five = ffmpeg_args(with_fps(5));
    let at = five.iter().position(|a| a == "-r").unwrap();
    assert_eq!(five[at + 1], "5");
    let slow = ffmpeg_args(Settings {
        keyframe_s: 10,
        bitrate_kbps: 20_000,
        ..DEFAULTS
    });
    let at = slow.iter().position(|a| a == "-g").unwrap();
    assert_eq!(slow[at + 1], "100");
    assert!(slow.contains(&"20000k".to_owned()));
    // No shell, no file, no network output: input and output are pipes.
    for arg in ffmpeg_args(DEFAULTS) {
        assert!(
            !arg.contains('/') && !arg.contains(':') || arg == "-c:v" || arg == "-b:v",
            "{arg}"
        );
    }
    // The pre-roll: the first frame plus preroll_s seconds of it.
    assert_eq!(DEFAULTS.burst_frames(), 7);
    assert_eq!(
        Settings {
            preroll_s: 0,
            ..DEFAULTS
        }
        .burst_frames(),
        1
    );
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
    let settings = Settings {
        preroll_s: 0,
        ..with_fps(4)
    };
    let mut enc = start_retry(&fake, settings, Arc::new(Solid(0xab))).unwrap();
    let out = read_n(&mut enc, 1000, Duration::from_secs(5));
    let text = String::from_utf8_lossy(&out);
    let (line, _) = text.split_once('\n').expect("args line");
    assert!(line.starts_with("[][] "), "environment leaked: {line}");
    assert_eq!(
        line.trim_start_matches("[][] ").trim_end(),
        ffmpeg_args(settings).join(" ")
    );
    let at = out.iter().position(|b| *b == b'\n').unwrap() + 1;
    assert_eq!(&out[at..], &[0xab_u8; 64][..]);
}

#[test]
fn drop_kills_a_stuck_encoder() {
    let fake = script("enc-stuck", "exec /usr/bin/sleep 30");
    let enc = start_retry(&fake, DEFAULTS, Arc::new(Solid(0))).unwrap();
    let start = Instant::now();
    drop(enc);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "drop waited for the child"
    );
}

#[test]
fn a_missing_encoder_is_an_error() {
    assert!(
        Encoder::start(
            Path::new("/nonexistent/ffmpeg"),
            DEFAULTS,
            Arc::new(Solid(0))
        )
        .is_err()
    );
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
    let source = Arc::new(VcsaSource::new(
        fixture("synthetic.vcsa"),
        font,
        llama_cast::config::Palette::Llama,
    ));
    let mut probe = vec![0_u8; FRAME_BYTES];
    source.frame(&mut probe).unwrap();
    let mut enc = Encoder::start(ffmpeg, DEFAULTS, source).unwrap();
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

/// #20: a new viewer's encoder gets the first frame `preroll_s * fps + 1`
/// times at once, not one per `1 / fps`.
#[test]
fn preroll_writes_the_first_frame_at_once() {
    let settings = Settings {
        fps: 1,
        preroll_s: 5,
        ..DEFAULTS
    };
    let want = FRAME_BYTES * settings.burst_frames() as usize;
    let fake = script("enc-preroll", &format!("exec /usr/bin/head -c {want}"));
    let start = Instant::now();
    let mut enc = start_retry(&fake, settings, Arc::new(Solid(0x5a))).unwrap();
    let out = read_n(&mut enc, want, Duration::from_secs(20));
    let took = start.elapsed();
    assert_eq!(out.len(), want);
    assert!(out.iter().all(|b| *b == 0x5a));
    // Paced at 1 fps these six frames would take five seconds.
    assert!(
        took < Duration::from_secs(4),
        "pre-roll was paced: {took:?}"
    );
}

// --- MPEG-TS inspection for the real-encoder test ---------------------

fn pid(p: &[u8]) -> u16 {
    (u16::from(p[1] & 0x1f) << 8) | u16::from(p[2])
}

fn pusi(p: &[u8]) -> bool {
    p[1] & 0x40 != 0
}

fn payload(p: &[u8]) -> &[u8] {
    let afc = (p[3] >> 4) & 3;
    let mut at = 4;
    if afc & 2 != 0 {
        at += 1 + usize::from(p[4]);
    }
    if afc & 1 == 0 || at >= 188 {
        return &[];
    }
    &p[at..]
}

/// A PSI section's body after the pointer field and the 8-byte header.
fn section(p: &[u8]) -> &[u8] {
    let pay = payload(p);
    let start = 1 + usize::from(pay[0]);
    let s = &pay[start..];
    let len = (usize::from(s[1] & 0x0f) << 8) | usize::from(s[2]);
    &s[8..3 + len - 4]
}

/// NAL unit types in an Annex B byte stream, in order.
fn nal_types(es: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            out.push(es[i + 3] & 0x1f);
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// #20 with the real encoder: the stream starts with PAT and PMT, the
/// first video access unit is an IDR with SPS and PPS before it, PAT/PMT
/// repeat, keyframes come every second, and the pre-roll arrives faster
/// than real time at the configured bitrate.
#[test]
fn real_ffmpeg_starts_at_an_idr_and_bursts_the_preroll() {
    let ffmpeg = Path::new("/usr/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("SKIP: /usr/bin/ffmpeg is not installed");
        return;
    }
    let font =
        load_font(&common::workspace_root().join("packaging/fonts/llama-hack-12x24.psfu")).unwrap();
    let source = Arc::new(VcsaSource::new(
        fixture("synthetic.vcsa"),
        font,
        llama_cast::config::Palette::Llama,
    ));
    let settings = Settings {
        preroll_s: 8,
        ..DEFAULTS
    };
    // Four seconds of video at 4000 kbit/s.
    let want = 4 * 4000 * 1000 / 8;
    let start = Instant::now();
    let mut enc = Encoder::start(ffmpeg, settings, source).unwrap();
    let out = read_n(&mut enc, want, Duration::from_secs(60));
    let took = start.elapsed();
    if out.is_empty() {
        eprintln!("SKIP: ffmpeg produced nothing (no libx264?)");
        return;
    }
    assert!(out.len() >= want, "short stream: {} bytes", out.len());
    // Paced, four seconds of video take about 3.5 s (ffmpeg runs up to
    // half an input frame ahead); the burst brings them at once.
    assert!(
        took < Duration::from_millis(2500),
        "four seconds of video took {took:?}: no pre-roll burst"
    );
    let packets = out.as_chunks::<188>().0;
    assert!(packets.iter().all(|p| p[0] == 0x47), "not MPEG-TS");

    // PAT first (SDT may come before it), then the PMT it names.
    let pat_at = packets.iter().position(|p| pid(p) == 0).expect("a PAT");
    let pat = section(&packets[pat_at]);
    let pmt_pid = (u16::from(pat[2] & 0x1f) << 8) | u16::from(pat[3]);
    let pmt_at = packets
        .iter()
        .position(|p| pid(p) == pmt_pid)
        .expect("a PMT");
    let pmt = section(&packets[pmt_at]);
    let info_len = (usize::from(pmt[2] & 0x0f) << 8) | usize::from(pmt[3]);
    let es = &pmt[4 + info_len..];
    assert_eq!(es[0], 0x1b, "H.264 stream type");
    let video = (u16::from(es[1] & 0x1f) << 8) | u16::from(es[2]);
    let first_video = packets.iter().position(|p| pid(p) == video).expect("video");
    assert!(pat_at < pmt_at && pmt_at < first_video, "PAT, PMT, video");

    // Access units: the payloads of each PES, without the PES header.
    let mut units: Vec<Vec<u8>> = Vec::new();
    for p in packets.iter().filter(|p| pid(&p[..]) == video) {
        let pay = payload(p);
        if pusi(p) {
            let header = 9 + usize::from(pay[8]);
            units.push(pay[header..].to_vec());
        } else if let Some(last) = units.last_mut() {
            last.extend_from_slice(pay);
        }
    }
    let first = nal_types(&units[0]);
    let idr = first
        .iter()
        .position(|t| *t == 5)
        .expect("first unit is IDR");
    assert!(
        first[..idr].contains(&7) && first[..idr].contains(&8),
        "{first:?}"
    );
    assert!(!first.contains(&1), "a non-IDR slice in the first unit");
    // A keyframe every 10 output frames (1 s), no B-frames needed to join.
    let keyframes: Vec<usize> = units
        .iter()
        .enumerate()
        .filter(|(_, u)| nal_types(u).contains(&5))
        .map(|(i, _)| i)
        .collect();
    assert!(keyframes.len() >= 3, "{keyframes:?}");
    assert!(
        keyframes.windows(2).all(|w| w[1] - w[0] == 10),
        "{keyframes:?}"
    );
    // PAT/PMT repeat (every 100 ms of stream time).
    let pats = packets.iter().filter(|p| pid(&p[..]) == 0).count();
    assert!(pats >= 20, "{pats} PATs in four seconds");
}
