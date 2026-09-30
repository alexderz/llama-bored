//! The one process llama-cast starts: the configured ffmpeg, with a fixed
//! argument vector, no shell, and an empty environment. One encoder per
//! client. A feeder thread writes raw RGB24 frames to its stdin at `fps`;
//! the HTTP worker copies its stdout (MPEG-TS) to the client. Dropping the
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
/// Keyframe interval, in output frames (2 s): a TV joins within 2 s.
pub const GOP: u32 = 20;

/// The argument vector after the program name.
#[must_use]
pub fn ffmpeg_args(fps: u32) -> Vec<String> {
    let size = format!("{FRAME_WIDTH}x{FRAME_HEIGHT}");
    [
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
        &fps.to_string(),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-r",
        &OUTPUT_FPS.to_string(),
        "-g",
        &GOP.to_string(),
        "-f",
        "mpegts",
        "-",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

/// A running ffmpeg and its frame feeder.
pub struct Encoder {
    child: Child,
    stdout: ChildStdout,
    stop: Arc<AtomicBool>,
    feeder: Option<JoinHandle<()>>,
}

impl Encoder {
    /// Start `ffmpeg` fed from `source` at `fps`.
    pub fn start(ffmpeg: &Path, fps: u32, source: Arc<dyn FrameSource>) -> io::Result<Self> {
        let mut child = Command::new(ffmpeg)
            .args(ffmpeg_args(fps))
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
            let period = Duration::from_secs(1) / fps.max(1);
            thread::spawn(move || feed(stdin, &*source, period, &stop))
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

/// Write one frame per `period` until the pipe breaks, a frame fails, or
/// `stop` is set. Closing stdin ends ffmpeg's input.
fn feed(mut stdin: impl Write, source: &dyn FrameSource, period: Duration, stop: &AtomicBool) {
    let mut frame = vec![0_u8; FRAME_BYTES];
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
        next += period;
        let now = Instant::now();
        if next > now {
            thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}
