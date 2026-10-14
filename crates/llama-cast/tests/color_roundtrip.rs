//! The colour round-trip S19 used to spell out as its own libx264 command.
//! The vector here is `encoder::ffmpeg_args`, so a change of the filter or
//! the tags moves this test. `scripts/s19-ffmpeg.sh` runs it with `--ignored`.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use llama_cast::encoder::{Codec, Settings, ffmpeg_args};
use llama_cast::render::{FRAME_HEIGHT, FRAME_WIDTH};
use llama_core::color::{ACT_STOPS, BB_STOPS, Rgb};

fn program(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then_some(candidate)
    })
}

fn push_stop(out: &mut Vec<[u8; 3]>, rgb: Rgb) {
    let rgb = [rgb.r, rgb.g, rgb.b];
    if !out.contains(&rgb) {
        out.push(rgb);
    }
}

fn stops() -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    for (_, rgb) in ACT_STOPS {
        push_stop(&mut out, rgb);
    }
    for (_, rgb) in BB_STOPS {
        push_stop(&mut out, rgb);
    }
    out
}

fn paint(stops: &[[u8; 3]]) -> Vec<u8> {
    let mut frame = vec![0_u8; FRAME_WIDTH * FRAME_HEIGHT * 3];
    for (i, rgb) in stops.iter().enumerate() {
        let x0 = 64 + i * 64;
        for y in 64..64 + 32 {
            for x in x0..x0 + 32 {
                let at = (y * FRAME_WIDTH + x) * 3;
                frame[at..at + 3].copy_from_slice(rgb);
            }
        }
    }
    frame
}

fn worst(blob: &[u8], stops: &[[u8; 3]]) -> u32 {
    assert_eq!(
        blob.len(),
        FRAME_WIDTH * FRAME_HEIGHT * 3,
        "decoded frame length"
    );
    let mut worst = 0_u32;
    for (i, rgb) in stops.iter().enumerate() {
        let x0 = 64 + i * 64 + 8;
        let y0 = 64 + 8;
        for y in y0..y0 + 16 {
            for x in x0..x0 + 16 {
                let at = (y * FRAME_WIDTH + x) * 3;
                for channel in 0..3 {
                    let err =
                        (i32::from(blob[at + channel]) - i32::from(rgb[channel])).unsigned_abs();
                    worst = worst.max(err);
                }
            }
        }
    }
    worst
}

/// The same vector with the scale filter and the VUI bitstream filter removed.
fn without_color(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut skip = false;
    for arg in args {
        if skip {
            skip = false;
            continue;
        }
        if arg == "-vf" || arg == "-bsf:v" {
            skip = true;
            continue;
        }
        out.push(arg.clone());
    }
    out
}

fn one_frame(args: &[String], input: &Path, output: &Path) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len() + 3);
    out.push("-y".to_owned());
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-i" && args.get(i + 1).map(String::as_str) == Some("-") {
            out.push("-i".to_owned());
            out.push(input.display().to_string());
            i += 2;
            continue;
        }
        if i + 1 == args.len() && args[i] == "-" {
            out.push("-frames:v".to_owned());
            out.push("1".to_owned());
            out.push(output.display().to_string());
            break;
        }
        out.push(args[i].clone());
        i += 1;
    }
    out
}

fn run(ffmpeg: &Path, args: &[String]) {
    let out = Command::new(ffmpeg)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
}

fn encode(ffmpeg: &Path, args: &[String], frame: &Path, ts: &Path) {
    run(ffmpeg, &one_frame(args, frame, ts));
}

fn decode(ffmpeg: &Path, ts: &Path, rgb: &Path) {
    run(
        ffmpeg,
        &[
            "-y".to_owned(),
            "-hide_banner".to_owned(),
            "-loglevel".to_owned(),
            "error".to_owned(),
            "-i".to_owned(),
            ts.display().to_string(),
            "-vf".to_owned(),
            "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24".to_owned(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-f".to_owned(),
            "rawvideo".to_owned(),
            "-pix_fmt".to_owned(),
            "rgb24".to_owned(),
            rgb.display().to_string(),
        ],
    );
}

#[test]
#[ignore = "encodes with the system ffmpeg; scripts/s19-ffmpeg.sh runs it"]
fn colour_roundtrip_follows_ffmpeg_args() {
    let Some(ffmpeg) = program("ffmpeg") else {
        println!("S19: ffmpeg is not installed; skipping the colour round-trip");
        return;
    };
    let listed = Command::new(&ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output()
        .expect("ffmpeg -encoders");
    let text = String::from_utf8_lossy(&listed.stdout);
    if !text.split_whitespace().any(|word| word == "libx264") {
        println!("S19: this ffmpeg has no libx264; skipping the colour round-trip");
        return;
    }
    let stops = stops();
    let last = 64 + stops.len().saturating_sub(1) * 64 + 32;
    assert!(last <= FRAME_WIDTH, "the ramp patches no longer fit");
    let dir = common::scratch("color-roundtrip");
    let frame_path = dir.join("frame.rgb");
    std::fs::write(&frame_path, paint(&stops)).unwrap();
    let settings = Settings {
        fps: 2,
        bitrate_kbps: 4000,
        keyframe_s: 1,
        preroll_s: 0,
    };
    let new_args = ffmpeg_args(Codec::Libx264, settings);
    let old_args = without_color(&new_args);
    assert!(new_args.windows(2).any(|pair| pair[0] == "-vf"));
    assert!(new_args.windows(2).any(|pair| pair[0] == "-bsf:v"));
    assert!(
        !old_args
            .windows(2)
            .any(|pair| pair[0] == "-vf" || pair[0] == "-bsf:v")
    );
    let new_ts = dir.join("new.ts");
    let old_ts = dir.join("old.ts");
    encode(&ffmpeg, &new_args, &frame_path, &new_ts);
    encode(&ffmpeg, &old_args, &frame_path, &old_ts);
    let new_rgb = dir.join("new.rgb");
    let old_rgb = dir.join("old.rgb");
    decode(&ffmpeg, &new_ts, &new_rgb);
    decode(&ffmpeg, &old_ts, &old_rgb);
    let new_worst = worst(&std::fs::read(&new_rgb).unwrap(), &stops);
    let old_worst = worst(&std::fs::read(&old_rgb).unwrap(), &stops);
    let tol = 4;
    println!(
        "S19: colour round-trip new worst {new_worst} (tolerance {tol} per channel), old worst {old_worst}"
    );
    assert!(
        new_worst <= tol,
        "S19: the BT.709 vector round-trips worse than {tol}"
    );
    assert!(
        old_worst > tol,
        "S19: the old argument vector still round-trips within {tol}"
    );
    let Some(ffprobe) = program("ffprobe") else {
        println!("S19: ffprobe is not installed; skipping the tag check");
        return;
    };
    let tags = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_range,color_space,color_transfer,color_primaries",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&new_ts)
        .output()
        .expect("ffprobe");
    assert!(
        tags.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&tags.stderr).trim()
    );
    let text = String::from_utf8_lossy(&tags.stdout);
    for want in [
        "color_range=tv",
        "color_space=bt709",
        "color_transfer=bt709",
        "color_primaries=bt709",
    ] {
        assert!(
            text.lines().any(|line| line == want),
            "S19: new stream is missing {want}\n{text}"
        );
    }
}
