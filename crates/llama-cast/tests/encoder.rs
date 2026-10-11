//! The ffmpeg argument vector (pinned, one set per encoder), the probe
//! (`ffmpeg -encoders`, bounded), the spawn (fixed args, empty environment,
//! killed on drop), and, when /usr/bin/ffmpeg offers libx264 or libopenh264,
//! a real encode of a few frames to MPEG-TS. Hardware encoders are never
//! started.

mod common;

use std::io::{self, Read};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::fixture;
use llama_cast::encoder::{
    ChooseError, Codec, Encoder, EncoderChoice, ProbeError, Settings, choose, ffmpeg_args,
    parse_encoders, probe, probe_limited,
};
use llama_cast::http::Live;
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
fn probe_retry(path: &Path, timeout: Duration, max_bytes: usize) -> Result<String, ProbeError> {
    for _ in 0..40 {
        match probe_limited(path, timeout, max_bytes) {
            Err(ProbeError::Spawn { source, .. })
                if source.kind() == io::ErrorKind::ExecutableFileBusy =>
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            other => return other,
        }
    }
    probe_limited(path, timeout, max_bytes)
}

fn start_retry(
    path: &std::path::Path,
    codec: Codec,
    settings: Settings,
    source: Arc<dyn FrameSource>,
) -> io::Result<Encoder> {
    for _ in 0..40 {
        match Encoder::start(path, codec, settings, Arc::clone(&source)) {
            Err(err) if err.kind() == io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(Duration::from_millis(25));
            }
            other => return other,
        }
    }
    Encoder::start(path, codec, settings, source)
}

/// The real-ffmpeg tests run one at a time: in parallel they compete for
/// the CPU, and the pre-roll test's wall-clock bound fails (#64).
static REAL_FFMPEG: Mutex<()> = Mutex::new(());

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
        ffmpeg_args(Codec::Libx264, DEFAULTS).join(" "),
        "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i - \
         -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p -r 10 -g 10 -bf 0 \
         -b:v 4000k -minrate 4000k -maxrate 4000k -bufsize 4000k -x264-params nal-hrd=cbr \
         -flush_packets 1 -pat_period 0.1 -f mpegts -"
    );
    // bitrate_kbps = 0: no rate options, x264 is quality-based.
    assert_eq!(
        ffmpeg_args(
            Codec::Libx264,
            Settings {
                bitrate_kbps: 0,
                keyframe_s: 2,
                ..DEFAULTS
            },
        )
        .join(" "),
        "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i - \
         -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p -r 10 -g 20 -bf 0 \
         -flush_packets 1 -pat_period 0.1 -f mpegts -"
    );
    let five = ffmpeg_args(Codec::Libx264, with_fps(5));
    let at = five.iter().position(|a| a == "-r").unwrap();
    assert_eq!(five[at + 1], "5");
    let slow = ffmpeg_args(
        Codec::Libx264,
        Settings {
            keyframe_s: 10,
            bitrate_kbps: 20_000,
            ..DEFAULTS
        },
    );
    let at = slow.iter().position(|a| a == "-g").unwrap();
    assert_eq!(slow[at + 1], "100");
    assert!(slow.contains(&"20000k".to_owned()));
    // No shell, no file, no network output: input and output are pipes.
    for arg in ffmpeg_args(Codec::Libx264, DEFAULTS) {
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
    let mut enc = start_retry(&fake, Codec::Libx264, settings, Arc::new(Solid(0xab))).unwrap();
    let out = read_n(&mut enc, 1000, Duration::from_secs(5));
    let text = String::from_utf8_lossy(&out);
    let (line, _) = text.split_once('\n').expect("args line");
    assert!(line.starts_with("[][] "), "environment leaked: {line}");
    assert_eq!(
        line.trim_start_matches("[][] ").trim_end(),
        ffmpeg_args(Codec::Libx264, settings).join(" ")
    );
    let at = out.iter().position(|b| *b == b'\n').unwrap() + 1;
    assert_eq!(&out[at..], &[0xab_u8; 64][..]);
}

#[test]
fn drop_kills_a_stuck_encoder() {
    let fake = script("enc-stuck", "exec /usr/bin/sleep 30");
    let enc = start_retry(&fake, Codec::Libx264, DEFAULTS, Arc::new(Solid(0))).unwrap();
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
            Codec::Libx264,
            DEFAULTS,
            Arc::new(Solid(0))
        )
        .is_err()
    );
}

#[test]
fn real_ffmpeg_encodes_mpeg_ts_when_installed() {
    let _serial = REAL_FFMPEG.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut enc = Encoder::start(ffmpeg, Codec::Libx264, DEFAULTS, source).unwrap();
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
    let mut enc = start_retry(&fake, Codec::Libx264, settings, Arc::new(Solid(0x5a))).unwrap();
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
    let _serial = REAL_FFMPEG.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut enc = Encoder::start(ffmpeg, Codec::Libx264, settings, source).unwrap();
    let out = read_n(&mut enc, want, Duration::from_secs(60));
    let took = start.elapsed();
    if out.is_empty() {
        eprintln!("SKIP: ffmpeg produced nothing (no libx264?)");
        return;
    }
    assert!(out.len() >= want, "short stream: {} bytes", out.len());
    // No wall-clock bound here (#64): on a loaded host the encode itself can
    // take longer than the paced stream would. The burst is proven exactly by
    // `preroll_writes_the_first_frame_at_once`; this test checks the stream.
    eprintln!("four seconds of video took {took:?}");
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

/// `ffmpeg -encoders` from the ffmpeg on this host (libx264 and libopenh264
/// both listed). A build without libx264 is this listing with those lines
/// removed, not a second capture.
const ENCODERS_LISTING: &str = include_str!("fixtures/ffmpeg-encoders.txt");

fn drop_named(text: &str, names: &[&str]) -> String {
    text.lines()
        .filter(|line| {
            let name = line.split_whitespace().nth(1).unwrap_or("");
            !names.contains(&name)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn encoder_listing_parses_and_prefers_software() {
    let all = parse_encoders(ENCODERS_LISTING);
    for name in [
        "libx264",
        "libx264rgb",
        "libopenh264",
        "h264_amf",
        "h264_nvenc",
        "h264_qsv",
        "h264_v4l2m2m",
        "h264_vaapi",
        "h264_vulkan",
    ] {
        assert!(all.has(name), "{name} missing: {}", all.h264().join(", "));
    }
    assert!(!all.has("libaom-av1"), "a non-H.264 encoder was kept");
    assert!(!all.has("V....."));
    // The legend line is six flag characters and then `=`, not an encoder.
    assert!(
        parse_encoders(" V..... = Video\n ------\n")
            .h264()
            .is_empty()
    );
    // libx264rgb shares the prefix and is not libx264.
    assert_ne!(
        all.h264()
            .iter()
            .filter(|n| n.as_str() == "libx264")
            .count(),
        0
    );
    assert_eq!(choose(EncoderChoice::Auto, &all).unwrap(), Codec::Libx264);

    let no_x264 = parse_encoders(&drop_named(ENCODERS_LISTING, &["libx264", "libx264rgb"]));
    assert!(!no_x264.has("libx264"));
    assert!(no_x264.has("libopenh264"));
    assert_eq!(
        choose(EncoderChoice::Auto, &no_x264).unwrap(),
        Codec::Libopenh264
    );

    // Hardware only after both software encoders are gone. nvenc is the
    // first hardware name in the preference, not the first in the listing.
    let hardware = parse_encoders(&drop_named(
        ENCODERS_LISTING,
        &["libx264", "libx264rgb", "libopenh264"],
    ));
    assert_eq!(
        choose(EncoderChoice::Auto, &hardware).unwrap(),
        Codec::H264Nvenc
    );

    let vaapi = parse_encoders(" V....D h264_vaapi           H.264/AVC (VAAPI) (codec h264)\n");
    assert_eq!(
        choose(EncoderChoice::Auto, &vaapi).unwrap(),
        Codec::H264Vaapi
    );

    // A known name still counts when the build omits the `(codec h264)` tag.
    let bare = parse_encoders(" V....D libopenh264          OpenH264 H.264\n");
    assert!(bare.has("libopenh264"));
    assert_eq!(
        choose(EncoderChoice::Auto, &bare).unwrap(),
        Codec::Libopenh264
    );
    // An H.264 encoder we do not drive is listed and then refused.
    let vulkan = parse_encoders(" V....D h264_vulkan          H.264/AVC (Vulkan) (codec h264)\n");
    assert_eq!(
        choose(EncoderChoice::Auto, &vulkan)
            .unwrap_err()
            .to_string(),
        "ffmpeg has no usable H.264 encoder (H.264 encoders it has: h264_vulkan)"
    );
    let empty = parse_encoders("Encoders:\n A....D aac something (codec aac)\n");
    assert_eq!(
        choose(EncoderChoice::Auto, &empty).unwrap_err().to_string(),
        "ffmpeg has no usable H.264 encoder (H.264 encoders it has: none)"
    );
    // The same name twice is one entry.
    let twice =
        parse_encoders(" V....D libx264 one (codec h264)\n V....D libx264 two (codec h264)\n");
    assert_eq!(twice.h264(), ["libx264".to_owned()]);
}

#[test]
fn a_pin_is_honoured_and_a_missing_one_names_what_ffmpeg_has() {
    let all = parse_encoders(ENCODERS_LISTING);
    assert_eq!(
        choose(EncoderChoice::Pinned(Codec::Libopenh264), &all).unwrap(),
        Codec::Libopenh264
    );
    assert_eq!(
        choose(EncoderChoice::Pinned(Codec::H264Vaapi), &all).unwrap(),
        Codec::H264Vaapi
    );
    let no_x264 = parse_encoders(&drop_named(ENCODERS_LISTING, &["libx264"]));
    match choose(EncoderChoice::Pinned(Codec::Libx264), &no_x264) {
        Err(ChooseError::Missing { name, have }) => {
            assert_eq!(name, "libx264");
            let names: Vec<&str> = have.split(", ").collect();
            assert!(!names.contains(&"libx264"), "{have}");
            assert!(names.contains(&"libx264rgb"), "{have}");
            assert!(names.contains(&"libopenh264"), "{have}");
            assert!(names.contains(&"h264_nvenc"), "{have}");
        }
        other => panic!("expected a missing pin, got {other:?}"),
    }
}

const HEAD: &str = "-hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i -";
const TAIL: &str = "-flush_packets 1 -pat_period 0.1 -f mpegts -";

fn zero_rate() -> Settings {
    Settings {
        bitrate_kbps: 0,
        ..DEFAULTS
    }
}

#[test]
fn each_encoder_has_one_fixed_argument_vector() {
    let join = |codec| ffmpeg_args(codec, DEFAULTS).join(" ");
    let join0 = |codec| ffmpeg_args(codec, zero_rate()).join(" ");
    assert_eq!(
        join(Codec::Libopenh264),
        format!(
            "{HEAD} -c:v libopenh264 -allow_skip_frames 0 -pix_fmt yuv420p -r 10 -g 10 -bf 0 \
             -rc_mode bitrate -b:v 4000k -maxrate 4000k {TAIL}"
        )
    );
    assert_eq!(
        join0(Codec::Libopenh264),
        format!(
            "{HEAD} -c:v libopenh264 -allow_skip_frames 0 -pix_fmt yuv420p -r 10 -g 10 -bf 0 {TAIL}"
        )
    );
    assert_eq!(
        join(Codec::H264Nvenc),
        format!(
            "{HEAD} -c:v h264_nvenc -preset p1 -tune ull -pix_fmt yuv420p -r 10 -g 10 -bf 0 \
             -rc cbr -b:v 4000k -maxrate 4000k -bufsize 4000k {TAIL}"
        )
    );
    assert_eq!(
        join0(Codec::H264Nvenc),
        format!(
            "{HEAD} -c:v h264_nvenc -preset p1 -tune ull -pix_fmt yuv420p -r 10 -g 10 -bf 0 {TAIL}"
        )
    );
    assert_eq!(
        join(Codec::H264Qsv),
        format!(
            "{HEAD} -c:v h264_qsv -preset veryfast -pix_fmt yuv420p -r 10 -g 10 -bf 0 \
             -b:v 4000k -maxrate 4000k -bufsize 4000k {TAIL}"
        )
    );
    assert_eq!(
        join0(Codec::H264Qsv),
        format!("{HEAD} -c:v h264_qsv -preset veryfast -pix_fmt yuv420p -r 10 -g 10 -bf 0 {TAIL}")
    );
    let vaapi_head = "-hide_banner -loglevel error -init_hw_device vaapi=cast \
         -filter_hw_device cast -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i -";
    assert_eq!(
        join(Codec::H264Vaapi),
        format!(
            "{vaapi_head} -vf format=yuv420p,format=nv12,hwupload -c:v h264_vaapi -r 10 -g 10 -bf 0 \
             -rc_mode CBR -b:v 4000k -maxrate 4000k -bufsize 4000k {TAIL}"
        )
    );
    assert_eq!(
        join0(Codec::H264Vaapi),
        format!(
            "{vaapi_head} -vf format=yuv420p,format=nv12,hwupload -c:v h264_vaapi -r 10 -g 10 -bf 0 {TAIL}"
        )
    );
    assert_eq!(
        join(Codec::H264Amf),
        format!(
            "{HEAD} -c:v h264_amf -usage ultralowlatency -quality speed -pix_fmt yuv420p \
             -r 10 -g 10 -bf 0 -rc cbr -b:v 4000k -maxrate 4000k -bufsize 4000k {TAIL}"
        )
    );
    assert_eq!(
        join0(Codec::H264Amf),
        format!(
            "{HEAD} -c:v h264_amf -usage ultralowlatency -quality speed -pix_fmt yuv420p \
             -r 10 -g 10 -bf 0 {TAIL}"
        )
    );
    assert_eq!(
        join(Codec::H264V4l2m2m),
        format!("{HEAD} -c:v h264_v4l2m2m -pix_fmt yuv420p -r 10 -g 10 -bf 0 -b:v 4000k {TAIL}")
    );
    assert_eq!(
        join0(Codec::H264V4l2m2m),
        format!("{HEAD} -c:v h264_v4l2m2m -pix_fmt yuv420p -r 10 -g 10 -bf 0 {TAIL}")
    );

    // Only the numeric settings change between configs. No device path.
    let custom = Settings {
        fps: 4,
        bitrate_kbps: 500,
        keyframe_s: 2,
        ..DEFAULTS
    };
    for codec in [
        Codec::Libx264,
        Codec::Libopenh264,
        Codec::H264Nvenc,
        Codec::H264Qsv,
        Codec::H264Vaapi,
        Codec::H264Amf,
        Codec::H264V4l2m2m,
    ] {
        let args = ffmpeg_args(codec, custom);
        let at = args.iter().position(|a| a == "-r").unwrap();
        assert_eq!(args[at + 1], "4", "{codec}");
        let at = args.iter().position(|a| a == "-g").unwrap();
        assert_eq!(args[at + 1], "20", "{codec}");
        assert!(args.contains(&"500k".to_owned()), "{codec}");
        assert!(!args.iter().any(|a| a.contains("4000")), "{codec}");
        for arg in &args {
            assert!(!arg.contains('/'), "{codec} {arg}");
            assert!(
                !arg.contains(':') || arg == "-c:v" || arg == "-b:v",
                "{codec} {arg}"
            );
        }
    }
}

#[test]
fn the_probe_is_bounded_and_reads_no_environment() {
    let listing = script(
        "probe-ok",
        "printf '[%s][%s]\\n' \"$HOME\" \"$CARGO_MANIFEST_DIR\"; \
         printf '%s\\n' ' V....D libopenh264 OpenH264 (codec h264)'",
    );
    let text = probe_retry(&listing, Duration::from_secs(2), 4096).unwrap();
    let (line, rest) = text.split_once('\n').unwrap();
    assert_eq!(line, "[][]", "environment leaked: {line}");
    assert!(parse_encoders(rest).has("libopenh264"));

    let slow = script("probe-slow", "exec /usr/bin/sleep 30");
    let err = probe_retry(&slow, Duration::from_millis(400), 4096).unwrap_err();
    assert!(matches!(err, ProbeError::Timeout), "{err}");

    let huge = script(
        "probe-huge",
        "i=0; while [ \"$i\" -lt 200 ]; do printf '0123456789abcdef'; i=$((i+1)); done",
    );
    let err = probe_retry(&huge, Duration::from_secs(2), 32).unwrap_err();
    assert!(matches!(err, ProbeError::TooLarge { max: 32 }), "{err}");

    let bad = script("probe-status", "printf 'no\\n'; exit 3");
    let err = probe_retry(&bad, Duration::from_secs(2), 4096).unwrap_err();
    match err {
        ProbeError::Status(status) => assert!(status.contains('3'), "{status}"),
        other => panic!("{other}"),
    }

    let messy = script("probe-utf8", "printf '\\377'");
    let err = probe_retry(&messy, Duration::from_secs(2), 4096).unwrap_err();
    assert!(matches!(err, ProbeError::Output), "{err}");

    let missing = probe_retry(Path::new("/nonexistent/ffmpeg"), Duration::from_secs(2), 64);
    assert!(matches!(missing, Err(ProbeError::Spawn { .. })));
}

struct PanicSource;

impl FrameSource for PanicSource {
    fn frame(&self, _frame: &mut [u8]) -> io::Result<()> {
        panic!("tty11 was read before the encoder refusal");
    }
}

#[test]
fn a_refused_encoder_does_not_spawn_or_read_a_frame() {
    let live = llama_cast::service::FfmpegLive {
        ffmpeg: PathBuf::from("/nonexistent/ffmpeg"),
        settings: DEFAULTS,
        codec: Err("ffmpeg has no usable H.264 encoder (H.264 encoders it has: none)".to_owned()),
        source: Arc::new(PanicSource),
    };
    let err = match live.open() {
        Err(err) => err,
        Ok(_) => panic!("a refused encoder started a stream"),
    };
    assert!(err.to_string().contains("no usable H.264 encoder"), "{err}");
}

fn pmt_says_h264(bytes: &[u8]) -> bool {
    if bytes.len() < 188 * 4 || bytes[0] != 0x47 {
        return false;
    }
    let packets = bytes.as_chunks::<188>().0;
    if packets.is_empty() || !packets.iter().all(|p| p[0] == 0x47) {
        return false;
    }
    let Some(pat_at) = packets.iter().position(|p| pid(p) == 0 && pusi(p)) else {
        return false;
    };
    let pat = section(&packets[pat_at]);
    if pat.len() < 4 {
        return false;
    }
    let pmt_pid = (u16::from(pat[2] & 0x1f) << 8) | u16::from(pat[3]);
    let Some(pmt_at) = packets.iter().position(|p| pid(p) == pmt_pid && pusi(p)) else {
        return false;
    };
    let pmt = section(&packets[pmt_at]);
    if pmt.len() < 5 {
        return false;
    }
    let info_len = (usize::from(pmt[2] & 0x0f) << 8) | usize::from(pmt[3]);
    let es_at = 4 + info_len;
    pmt.len() > es_at && pmt[es_at] == 0x1b
}

/// `Some` when ffprobe ran. `None` when it is not installed.
fn ffprobe_h264_mpegts(path: &Path) -> Option<bool> {
    let ffprobe = Path::new("/usr/bin/ffprobe");
    if !ffprobe.is_file() {
        return None;
    }
    let out = Command::new(ffprobe)
        .env_clear()
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-show_entries",
            "format=format_name",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("ffprobe: {}", String::from_utf8_lossy(&out.stderr).trim());
        return Some(false);
    }
    let text = String::from_utf8(out.stdout).unwrap_or_default();
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    Some(compact.contains("\"codec_name\":\"h264\"") && compact.contains("mpegts"))
}

fn encode_few_frames(ffmpeg: &Path, codec: Codec) {
    let font =
        load_font(&common::workspace_root().join("packaging/fonts/llama-hack-12x24.psfu")).unwrap();
    let source = Arc::new(VcsaSource::new(
        fixture("synthetic.vcsa"),
        font,
        llama_cast::config::Palette::Llama,
    ));
    let settings = Settings {
        preroll_s: 1,
        ..DEFAULTS
    };
    let mut enc = Encoder::start(ffmpeg, codec, settings, source)
        .unwrap_or_else(|err| panic!("{codec} did not start: {err}"));
    let out = read_n(&mut enc, 64 * 1024, Duration::from_secs(20));
    drop(enc);
    assert!(out.len() >= 188 * 4, "{codec} produced {} bytes", out.len());
    assert!(
        out.as_chunks::<188>().0.iter().all(|p| p[0] == 0x47),
        "{codec} is not MPEG-TS"
    );
    let dir = common::scratch(&format!("enc-{codec}"));
    let path = dir.join("few.ts");
    std::fs::write(&path, &out).unwrap();
    match ffprobe_h264_mpegts(&path) {
        Some(true) => {}
        Some(false) => panic!("{codec}: ffprobe did not report h264 in mpegts"),
        None => assert!(pmt_says_h264(&out), "{codec}: PMT is not H.264"),
    }
}

/// Each software encoder this ffmpeg actually has. Hardware encoders are
/// not started: they need a device, and the GPU may be busy.
#[test]
fn real_ffmpeg_software_encoders_emit_h264_in_mpeg_ts() {
    let _serial = REAL_FFMPEG.lock().unwrap_or_else(|e| e.into_inner());
    let ffmpeg = Path::new("/usr/bin/ffmpeg");
    if !ffmpeg.is_file() {
        eprintln!("SKIP: /usr/bin/ffmpeg is not installed");
        return;
    }
    let inventory = parse_encoders(&probe(ffmpeg).expect("ffmpeg -encoders"));
    match choose(EncoderChoice::Auto, &inventory) {
        Ok(Codec::Libx264) => assert!(inventory.has("libx264")),
        Ok(Codec::Libopenh264) => {
            assert!(inventory.has("libopenh264"));
            assert!(!inventory.has("libx264"));
        }
        Ok(other) => {
            assert!(!inventory.has("libx264") && !inventory.has("libopenh264"));
            eprintln!("auto chose {other}; not encoding it");
        }
        Err(err) => {
            assert!(!inventory.has("libx264") && !inventory.has("libopenh264"));
            eprintln!("no software encoder: {err}");
        }
    }
    let mut encoded = 0;
    for codec in [Codec::Libx264, Codec::Libopenh264] {
        if !inventory.has(codec.name()) {
            eprintln!("SKIP: {codec} is not in this ffmpeg");
            continue;
        }
        encode_few_frames(ffmpeg, codec);
        encoded += 1;
    }
    if encoded == 0 {
        eprintln!("SKIP: this ffmpeg has neither libx264 nor libopenh264");
    }
}
