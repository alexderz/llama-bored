//! The one process llama-cast starts: the configured ffmpeg, with a fixed
//! argument vector, no shell, and an empty environment. One encoder per
//! client. A feeder thread writes raw RGB24 frames to its stdin at `fps`,
//! after a pre-roll burst of the first frame; the HTTP worker copies its stdout (MPEG-TS) to the client. Dropping the
//! [`Encoder`] kills ffmpeg and joins the feeder.

use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority};

use crate::render::{FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH};
use crate::source::FrameSource;

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
/// Low latency for a live picture: `zerolatency` (no B-frames, no
/// lookahead, every frame out at once) on top of `stillimage`, a fixed
/// keyframe interval, and packets flushed to the pipe as they are muxed.
/// Each viewer has its own ffmpeg, so its stream begins with PAT, PMT and
/// an IDR; PAT/PMT then repeat every 100 ms. With `bitrate_kbps` set the
/// stream is CBR with filler (`nal-hrd=cbr`), so a still dashboard still
/// arrives at that rate.
#[must_use]
pub fn ffmpeg_args(settings: Settings) -> Vec<String> {
    let size = format!("{FRAME_WIDTH}x{FRAME_HEIGHT}");
    let fps = settings.fps.to_string();
    let output_fps = OUTPUT_FPS.to_string();
    let gop = settings.gop().to_string();
    let mut args: Vec<&str> = vec![
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        &size,
        "-r",
        &fps,
        "-i",
        "-",
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
    ];
    let rate = format!("{}k", settings.bitrate_kbps);
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

/// A running ffmpeg and its frame feeder.
pub struct Encoder {
    child: Child,
    stdout: ChildStdout,
    stop: Arc<AtomicBool>,
    feeder: Option<JoinHandle<()>>,
}

impl Encoder {
    /// Start `ffmpeg` fed from `source` as `settings` say.
    pub fn start(
        ffmpeg: &Path,
        settings: Settings,
        source: Arc<dyn FrameSource>,
    ) -> io::Result<Self> {
        let mut child = Command::new(ffmpeg)
            .args(ffmpeg_args(settings))
            .env_clear()
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
