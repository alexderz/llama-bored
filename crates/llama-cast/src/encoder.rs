//! The one program llama-cast starts: the configured ffmpeg, with a fixed
//! argument vector, no shell, and an empty environment. One encoder per
//! client. At start the same program is run once (`-hide_banner -encoders`)
//! to see which H.264 encoder to ask for. A feeder thread writes raw RGB24
//! frames to its stdin at `fps`, after a pre-roll burst of the first frame;
//! the HTTP worker copies its stdout (MPEG-TS) to the client. Dropping the
//! [`Encoder`] kills ffmpeg and joins the feeder.

use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority};
use thiserror::Error;

use crate::render::{FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH};
use crate::source::FrameSource;

/// How long `ffmpeg -encoders` may run. The listing is small and returns
/// at once; the bound is so a stuck ffmpeg cannot hold start-up.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Most bytes kept from `ffmpeg -encoders`. A real listing is a few dozen
/// kilobytes; past this the probe is a failure, not a partial choice.
pub const PROBE_MAX_BYTES: usize = 256 * 1024;

/// `ffmpeg -encoders`, nothing else. No shell.
const PROBE_ARGS: &[&str] = &["-hide_banner", "-encoders"];

/// An H.264 encoder we know how to drive. The names are ffmpeg's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    Libx264,
    Libopenh264,
    H264Nvenc,
    H264Qsv,
    H264Vaapi,
    H264Amf,
    H264V4l2m2m,
}

impl Codec {
    /// The ffmpeg encoder name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Libx264 => "libx264",
            Self::Libopenh264 => "libopenh264",
            Self::H264Nvenc => "h264_nvenc",
            Self::H264Qsv => "h264_qsv",
            Self::H264Vaapi => "h264_vaapi",
            Self::H264Amf => "h264_amf",
            Self::H264V4l2m2m => "h264_v4l2m2m",
        }
    }

    /// `name` when it is one of [`Codec`]'s.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "libx264" => Some(Self::Libx264),
            "libopenh264" => Some(Self::Libopenh264),
            "h264_nvenc" => Some(Self::H264Nvenc),
            "h264_qsv" => Some(Self::H264Qsv),
            "h264_vaapi" => Some(Self::H264Vaapi),
            "h264_amf" => Some(Self::H264Amf),
            "h264_v4l2m2m" => Some(Self::H264V4l2m2m),
            _ => None,
        }
    }
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What `cast.toml` asked for. `auto` is the default.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncoderChoice {
    Auto,
    Pinned(Codec),
}

impl EncoderChoice {
    /// The config value, as written.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Pinned(codec) => codec.name(),
        }
    }
}

impl std::fmt::Display for EncoderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// `text` when it is `auto` or a [`Codec`] name.
pub fn parse_choice(text: &str) -> Option<EncoderChoice> {
    if text == "auto" {
        return Some(EncoderChoice::Auto);
    }
    Codec::from_name(text).map(EncoderChoice::Pinned)
}

/// H.264 encoders a probe found, in the order ffmpeg listed them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EncoderInventory {
    h264: Vec<String>,
}

impl EncoderInventory {
    /// Whether ffmpeg listed `name` as an H.264 encoder.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.h264.iter().any(|have| have == name)
    }

    /// The H.264 encoder names, in ffmpeg's order.
    #[must_use]
    pub fn h264(&self) -> &[String] {
        &self.h264
    }

    fn listed(&self) -> String {
        if self.h264.is_empty() {
            "none".to_owned()
        } else {
            self.h264.join(", ")
        }
    }
}

/// Preference for a live text screen. Software x264 gives the best text
/// quality at 2 fps, plays everywhere, and takes nothing from a GPU busy
/// with inference; hardware encoders need a device and are a last resort,
/// used only when neither software encoder exists.
const PREFERENCE: [Codec; 7] = [
    Codec::Libx264,
    Codec::Libopenh264,
    Codec::H264Nvenc,
    Codec::H264Qsv,
    Codec::H264Vaapi,
    Codec::H264Amf,
    Codec::H264V4l2m2m,
];

/// Why the configured encoder cannot be used.
#[derive(Debug, Error)]
pub enum ChooseError {
    #[error("ffmpeg has no {name} (H.264 encoders it has: {have})")]
    Missing { name: &'static str, have: String },
    #[error("ffmpeg has no usable H.264 encoder (H.264 encoders it has: {have})")]
    None { have: String },
}

/// Pick from `inventory` by [`PREFERENCE`], or honour a pin.
pub fn choose(choice: EncoderChoice, inventory: &EncoderInventory) -> Result<Codec, ChooseError> {
    match choice {
        EncoderChoice::Auto => {
            for codec in PREFERENCE {
                if inventory.has(codec.name()) {
                    return Ok(codec);
                }
            }
            Err(ChooseError::None {
                have: inventory.listed(),
            })
        }
        EncoderChoice::Pinned(codec) => {
            if inventory.has(codec.name()) {
                Ok(codec)
            } else {
                Err(ChooseError::Missing {
                    name: codec.name(),
                    have: inventory.listed(),
                })
            }
        }
    }
}

/// Why `ffmpeg -encoders` could not be read.
#[derive(Debug, Error)]
pub enum ProbeError {
    #[error("cannot run {path}: {source}")]
    Spawn {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("ffmpeg -encoders timed out")]
    Timeout,
    #[error("ffmpeg -encoders exceeded {max} bytes")]
    TooLarge { max: usize },
    #[error("ffmpeg -encoders failed ({0})")]
    Status(String),
    #[error("ffmpeg -encoders output is not utf-8")]
    Output,
}

/// Why no encoder was selected at start-up.
#[derive(Debug, Error)]
pub enum ResolveError {
    #[error("{0}")]
    Probe(#[from] ProbeError),
    #[error("{0}")]
    Choose(#[from] ChooseError),
}

/// The configured ffmpeg, no shell, empty environment.
fn command(ffmpeg: &Path) -> Command {
    let mut cmd = Command::new(ffmpeg);
    cmd.env_clear();
    cmd
}

enum Captured {
    Eof(Vec<u8>),
    Overflow,
}

fn capture(mut stdout: impl Read, limit: usize) -> Captured {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) => return Captured::Eof(buf),
            Ok(n) => {
                let room = limit.saturating_sub(buf.len());
                if n > room {
                    return Captured::Overflow;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Captured::Eof(buf),
        }
    }
}

enum Waited {
    Ok,
    Failed(String),
    TimedOut,
}

/// Reap `child` by `deadline`. On the deadline, kill it.
fn wait_child(child: &mut Child, deadline: Instant) -> io::Result<Waited> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(if status.success() {
                Waited::Ok
            } else {
                Waited::Failed(status.to_string())
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Waited::TimedOut);
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn stop_child(mut child: Child, reader: JoinHandle<()>) {
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
}

/// Run `ffmpeg -encoders` and return its stdout.
///
/// No shell, empty environment, at most `max_bytes`, and `timeout` on the
/// whole run. Past the byte cap or the deadline the process is killed and
/// the probe fails: a partial listing is not a choice.
pub fn probe_limited(
    ffmpeg: &Path,
    timeout: Duration,
    max_bytes: usize,
) -> Result<String, ProbeError> {
    let mut child = command(ffmpeg)
        .args(PROBE_ARGS)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|source| ProbeError::Spawn {
            path: ffmpeg.display().to_string(),
            source,
        })?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(ProbeError::Output);
    };
    let (tx, rx) = mpsc::channel();
    let reader = thread::spawn(move || {
        // Associated function: the spawn scan treats a method call here as a datagram.
        let _ = mpsc::Sender::send(&tx, capture(stdout, max_bytes));
    });
    let deadline = Instant::now() + timeout;
    let captured = match rx.recv_timeout(timeout) {
        Ok(captured) => captured,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            stop_child(child, reader);
            return Err(ProbeError::Timeout);
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            stop_child(child, reader);
            return Err(ProbeError::Output);
        }
    };
    let bytes = match captured {
        Captured::Overflow => {
            stop_child(child, reader);
            return Err(ProbeError::TooLarge { max: max_bytes });
        }
        Captured::Eof(bytes) => {
            let _ = reader.join();
            bytes
        }
    };
    match wait_child(&mut child, deadline) {
        Ok(Waited::Ok) => {}
        Ok(Waited::Failed(status)) => return Err(ProbeError::Status(status)),
        Ok(Waited::TimedOut) => return Err(ProbeError::Timeout),
        Err(err) => {
            return Err(ProbeError::Spawn {
                path: ffmpeg.display().to_string(),
                source: err,
            });
        }
    }
    String::from_utf8(bytes).map_err(|_| ProbeError::Output)
}

/// [`probe_limited`] with the start-up bounds.
pub fn probe(ffmpeg: &Path) -> Result<String, ProbeError> {
    probe_limited(ffmpeg, PROBE_TIMEOUT, PROBE_MAX_BYTES)
}

/// H.264 encoders in `ffmpeg -encoders` output.
///
/// A video line is six flag characters (the first `V`) and a name. It
/// counts when ffmpeg tags it `(codec h264)`, and also when the name is
/// one we drive, so a build that omits the tag still matches a pin.
#[must_use]
pub fn parse_encoders(text: &str) -> EncoderInventory {
    let mut h264 = Vec::new();
    for line in text.lines() {
        let Some((name, desc)) = encoder_line(line) else {
            continue;
        };
        let known = Codec::from_name(name).is_some();
        if !known && !desc.contains("(codec h264)") {
            continue;
        }
        if !h264.iter().any(|have| have == name) {
            h264.push(name.to_owned());
        }
    }
    EncoderInventory { h264 }
}

/// `(name, description)` of a video encoder line, or `None`.
fn encoder_line(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    let mut parts = line.split_whitespace();
    let flags = parts.next()?;
    let bytes = flags.as_bytes();
    if bytes.len() != 6 || bytes[0] != b'V' {
        return None;
    }
    if !bytes[1..]
        .iter()
        .all(|b| matches!(b, b'F' | b'S' | b'X' | b'B' | b'D' | b'.'))
    {
        return None;
    }
    let name = parts.next()?;
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return None;
    }
    let desc = line
        .split_once(name)
        .map(|(_, rest)| rest.trim())
        .unwrap_or("");
    Some((name, desc))
}

/// Probe `ffmpeg` and apply `choice`.
pub fn resolve(ffmpeg: &Path, choice: EncoderChoice) -> Result<Codec, ResolveError> {
    let text = probe(ffmpeg)?;
    choose(choice, &parse_encoders(&text)).map_err(ResolveError::from)
}

/// The output frame rate ffmpeg repeats input frames up to.
pub const OUTPUT_FPS: u32 = 10;

/// The encoder knobs from `cast.toml`, already validated. Only these
/// numbers reach the argument vector; no config text does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Settings {
    /// Frames rendered from tty11 per second (1..=5).
    pub fps: u32,
    /// Constant bitrate in kbit/s, padded with filler when the screen is
    /// still; 0 leaves x264 quality-based (a still screen is then a few
    /// kbit/s, which a TV's byte-sized start buffer fills slowly, #20).
    pub bitrate_kbps: u32,
    /// Keyframe interval in seconds (1..=10).
    pub keyframe_s: u32,
    /// Seconds of the first frame sent to a new viewer at once (0..=10).
    pub preroll_s: u32,
}

impl Settings {
    /// Keyframe interval in output frames.
    #[must_use]
    pub fn gop(self) -> u32 {
        self.keyframe_s.max(1) * OUTPUT_FPS
    }

    /// Copies of the first frame written before pacing starts: the
    /// pre-roll plus the first frame itself.
    #[must_use]
    pub fn burst_frames(self) -> u32 {
        self.preroll_s * self.fps.max(1) + 1
    }
}

/// The argument vector after the program name.
///
/// The same stream for every encoder: raw RGB24 in, MPEG-TS out, yuv420p,
/// no B-frames, GOP from `keyframe_s`, output at [`OUTPUT_FPS`], packets
/// flushed as they are muxed, PAT/PMT every 100 ms. Each viewer has its
/// own ffmpeg, so its stream begins with PAT, PMT and an IDR. Only the
/// numbers from `settings` are interpolated; every other token is fixed.
///
/// With `bitrate_kbps` set, libx264 is CBR with filler (`nal-hrd=cbr`),
/// so a still dashboard still arrives at that rate. That vector is the
/// one llama-cast has always used. The other encoders get their own
/// rate-control options for the same shape.
#[must_use]
pub fn ffmpeg_args(codec: Codec, settings: Settings) -> Vec<String> {
    let size = format!("{FRAME_WIDTH}x{FRAME_HEIGHT}");
    let fps = settings.fps.to_string();
    let output_fps = OUTPUT_FPS.to_string();
    let gop = settings.gop().to_string();
    let rate = format!("{}k", settings.bitrate_kbps);
    let mut args: Vec<&str> = vec!["-hide_banner", "-loglevel", "error"];
    // VAAPI only encodes a hardware surface. `cast` is ffmpeg's device
    // name for the default render node, not a path from the config.
    if codec == Codec::H264Vaapi {
        args.extend(["-init_hw_device", "vaapi=cast", "-filter_hw_device", "cast"]);
    }
    args.extend([
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-r", &fps, "-i", "-",
    ]);
    match codec {
        Codec::Libx264 => {
            args.extend([
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "stillimage,zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend([
                    "-b:v",
                    &rate,
                    "-minrate",
                    &rate,
                    "-maxrate",
                    &rate,
                    "-bufsize",
                    &rate,
                    "-x264-params",
                    "nal-hrd=cbr",
                ]);
            }
        }
        Codec::Libopenh264 => {
            // libopenh264 has no HRD filler (`nal-hrd=cbr` is x264's).
            // `-rc_mode bitrate` is its bitrate mode: the wrapper sets the
            // target from `-b:v` and the ceiling from `-maxrate` (equal
            // here, so the rate cannot peak above the config). Frame skip
            // stays off, so a live picture is not dropped to meet the
            // rate. The wrapper has no buffer size; a still screen is
            // smaller than the target because this encoder cannot pad.
            // Checked against `ffmpeg -h encoder=libopenh264` (rc_mode
            // bitrate / quality, allow_skip_frames default false).
            args.extend([
                "-c:v",
                "libopenh264",
                "-allow_skip_frames",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend(["-rc_mode", "bitrate", "-b:v", &rate, "-maxrate", &rate]);
            }
        }
        Codec::H264Nvenc => {
            args.extend([
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-tune",
                "ull",
                "-pix_fmt",
                "yuv420p",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend([
                    "-rc", "cbr", "-b:v", &rate, "-maxrate", &rate, "-bufsize", &rate,
                ]);
            }
        }
        Codec::H264Qsv => {
            // Equal `-b:v` and `-maxrate` is how this wrapper selects CBR.
            // Its pixel formats are nv12 and qsv, not yuv420p
            // (`ffmpeg -h encoder=h264_qsv`), so the frame is nv12.
            args.extend([
                "-c:v",
                "h264_qsv",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "nv12",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                push_cbr(&mut args, &rate);
            }
        }
        Codec::H264Vaapi => {
            // yuv420p, then nv12, the 4:2:0 layout the hardware surface
            // accepts. `-rc_mode CBR` with the buffer equal to the rate.
            args.extend([
                "-vf",
                "format=yuv420p,format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend(["-rc_mode", "CBR"]);
                push_cbr(&mut args, &rate);
            }
        }
        Codec::H264Amf => {
            args.extend([
                "-c:v",
                "h264_amf",
                "-usage",
                "ultralowlatency",
                "-quality",
                "speed",
                "-pix_fmt",
                "yuv420p",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend(["-rc", "cbr"]);
                push_cbr(&mut args, &rate);
            }
        }
        Codec::H264V4l2m2m => {
            // The wrapper takes the rate from `-b:v` and has no CBR switch
            // and no buffer size.
            args.extend([
                "-c:v",
                "h264_v4l2m2m",
                "-pix_fmt",
                "yuv420p",
                "-r",
                &output_fps,
                "-g",
                &gop,
                "-bf",
                "0",
            ]);
            if settings.bitrate_kbps > 0 {
                args.extend(["-b:v", &rate]);
            }
        }
    }
    args.extend([
        "-flush_packets",
        "1",
        "-pat_period",
        "0.1",
        "-f",
        "mpegts",
        "-",
    ]);
    args.into_iter().map(str::to_owned).collect()
}

/// `-b:v`, `-maxrate` and `-bufsize` all at `rate` (a one-second buffer).
fn push_cbr<'a>(args: &mut Vec<&'a str>, rate: &'a str) {
    args.extend(["-b:v", rate, "-maxrate", rate, "-bufsize", rate]);
}

/// A running ffmpeg and its frame feeder.
pub struct Encoder {
    child: Child,
    stdout: ChildStdout,
    stop: Arc<AtomicBool>,
    feeder: Option<JoinHandle<()>>,
}

impl Encoder {
    /// Start `ffmpeg` fed from `source`, encoding with `codec` as
    /// `settings` say.
    pub fn start(
        ffmpeg: &Path,
        codec: Codec,
        settings: Settings,
        source: Arc<dyn FrameSource>,
    ) -> io::Result<Self> {
        let mut child = command(ffmpeg)
            .args(ffmpeg_args(codec, settings))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::other("ffmpeg pipes missing"));
        };
        let stop = Arc::new(AtomicBool::new(false));
        let feeder = {
            let stop = Arc::clone(&stop);
            let period = Duration::from_secs(1) / settings.fps.max(1);
            let burst = settings.burst_frames();
            thread::spawn(move || feed(stdin, &*source, period, burst, &stop))
        };
        Ok(Self {
            child,
            stdout,
            stop,
            feeder: Some(feeder),
        })
    }
}

impl Read for Encoder {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stdout.read(buf)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(feeder) = self.feeder.take() {
            let _ = feeder.join();
        }
    }
}

/// Write the first frame `burst` times at once (the pre-roll: ffmpeg
/// encodes them as fast as it can, so a new viewer's buffer starts full),
/// then one frame per `period` until the pipe breaks, a frame fails, or
/// `stop` is set. The picture then runs the pre-roll behind tty11. Closing
/// stdin ends ffmpeg's input.
fn feed(
    mut stdin: impl Write,
    source: &dyn FrameSource,
    period: Duration,
    burst: u32,
    stop: &AtomicBool,
) {
    let mut frame = vec![0_u8; FRAME_BYTES];
    let mut extra = burst.saturating_sub(1);
    let mut next = Instant::now();
    while !stop.load(Ordering::SeqCst) {
        if let Err(err) = source.frame(&mut frame) {
            log::emit(
                &mut log::Stderr,
                Priority::Warning,
                &format!("stream: frame: {err}"),
            );
            return;
        }
        if stdin.write_all(&frame).is_err() || stdin.flush().is_err() {
            return;
        }
        if extra > 0 {
            while extra > 0 && !stop.load(Ordering::SeqCst) {
                extra -= 1;
                if stdin.write_all(&frame).is_err() || stdin.flush().is_err() {
                    return;
                }
            }
            next = Instant::now();
        }
        next += period;
        let now = Instant::now();
        if next > now {
            thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}
