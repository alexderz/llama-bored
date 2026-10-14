//! CLI and the production wiring: bind the one TCP listener and the one
//! SSDP socket, notify systemd, announce, serve.

use std::io::{self, Read};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::Duration;

use llama_core::log::{self, Priority};
use rustix::net::{AddressFamily, SocketType, ipproto, sockopt};
use sd_notify::NotifyState;

use crate::config::{self, Config, Font, Source};
use crate::discovery::{self, Advert};
use crate::dlna::Device;
use crate::encoder::{self, Codec, Encoder, Settings};
use crate::font::Psf2;
use crate::http::{self, App, Limits, Live, ServerConfig, Stats};
use crate::render::FRAME_BYTES;
use crate::source::{self, FrameSource, GridSource, VcsaSource};
use crate::ssdp;

/// Exit status for a usage or config error. The unit does not restart on it.
pub const EXIT_CONFIG: i32 = 2;

pub const USAGE: &str = "usage: llama-cast run --config PATH\n       llama-cast check --config PATH\n       llama-cast bye --config PATH";

/// Parsed command line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Announce and serve.
    Run { config: PathBuf },
    /// Validate the config and the font, and print what it would do.
    Check { config: PathBuf },
    /// Send SSDP byebye (the unit's `ExecStop=`) and exit.
    Bye { config: PathBuf },
}

/// `run|check|bye --config PATH`, nothing else.
pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let args: Vec<String> = args.into_iter().collect();
    match args.as_slice() {
        [cmd, flag, path] if flag == "--config" && !path.is_empty() => {
            let config = PathBuf::from(path);
            match cmd.as_str() {
                "run" => Ok(Command::Run { config }),
                "check" => Ok(Command::Check { config }),
                "bye" => Ok(Command::Bye { config }),
                _ => Err(USAGE.to_owned()),
            }
        }
        _ => Err(USAGE.to_owned()),
    }
}

/// The production stream: ffmpeg fed from tty11 or the grid file.
pub struct FfmpegLive {
    pub ffmpeg: PathBuf,
    pub settings: Settings,
    /// The encoder chosen at start, or the error already logged once.
    /// `Err` refuses the stream without spawning ffmpeg.
    pub codec: Result<Codec, String>,
    pub source: Arc<dyn FrameSource>,
    /// Named in the error when the first frame cannot be read.
    pub origin: String,
}

impl Live for FfmpegLive {
    fn refused(&self) -> Option<&str> {
        self.codec.as_ref().err().map(String::as_str)
    }

    fn open(&self) -> io::Result<Box<dyn Read + Send>> {
        // A missing encoder was logged once at start. Refuse here, the
        // same way a failed ffmpeg is refused, without spawning one.
        let codec = match &self.codec {
            Ok(codec) => *codec,
            Err(message) => return Err(io::Error::other(message.clone())),
        };
        // An unreadable picture is a 500 with the reason logged, not an
        // empty 200. A grid that is missing or stale still returns a frame.
        let mut probe = vec![0_u8; FRAME_BYTES];
        self.source
            .frame(&mut probe)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", self.origin)))?;
        drop(probe);
        let encoder = Encoder::start(&self.ffmpeg, codec, self.settings, Arc::clone(&self.source))?;
        Ok(Box::new(encoder))
    }
}

/// The SSDP socket: `0.0.0.0:1900` with `SO_REUSEADDR` (other SSDP stacks
/// on the host keep working), joined to the group on `interface`, sending
/// multicast from `interface`. The only UDP bind in the crate.
pub fn bind_ssdp(interface: Ipv4Addr) -> io::Result<UdpSocket> {
    let fd = rustix::net::socket(AddressFamily::INET, SocketType::DGRAM, Some(ipproto::UDP))?;
    sockopt::set_socket_reuseaddr(&fd, true)?;
    rustix::net::bind(&fd, &SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, ssdp::PORT))?;
    let socket = UdpSocket::from(fd);
    socket.join_multicast_v4(&ssdp::GROUP, &interface)?;
    sockopt::set_ip_multicast_if(&socket, &interface)?;
    Ok(socket)
}

fn err(message: &str) {
    log::emit(&mut log::Stderr, Priority::Err, message);
}

fn summary(config: &Config) -> String {
    let nets: Vec<String> = config
        .allow
        .nets()
        .iter()
        .map(ToString::to_string)
        .collect();
    format!(
        "listen {} allow [{}] interface_addr {} name {:?} title {:?} fps {} max_clients {} bitrate_kbps {} keyframe_s {} preroll_s {} ffmpeg {} encoder {} source {} font {} palette {}",
        config.listen,
        nets.join(", "),
        config.interface_addr,
        config.name,
        config.title,
        config.fps,
        config.max_clients,
        config.bitrate_kbps,
        config.keyframe_s,
        config.preroll_s,
        config.ffmpeg.display(),
        config.encoder,
        config.source.label(),
        font_summary(config),
        config.palette.label(),
    )
}

fn font_summary(config: &Config) -> &'static str {
    match (config.source, config.font) {
        (Source::Grid, None) => "auto",
        (_, Some(font)) => font.label(),
        (Source::Tty, None) => Font::Hack12x24.label(),
    }
}

/// Grid mode with no `font` picks one per frame. A set font, and tty mode,
/// do not.
#[must_use]
pub fn font_is_auto(config: &Config) -> bool {
    config.source == Source::Grid && config.font.is_none()
}

/// What `check` prints, and the status it returns. `encoder` is the probe
/// result; `None` is a failure and adds no encoder line (the caller logs
/// the reason).
#[must_use]
pub fn check_output(
    config: &Config,
    host: &str,
    fonts: &[(Font, Psf2)],
    encoder: Option<Codec>,
) -> CheckOutput {
    let mut lines = vec![
        format!("config ok: {}", summary(config)),
        format!("friendly name: {:?}", config.friendly_name(host)),
        format!("item title: {:?}", config.item_title(host)),
        format!(
            "stream url: {}{}",
            config.base_url(),
            crate::dlna::STREAM_PATH
        ),
    ];
    if config.source == Source::Grid {
        match source::inspect_grid(&config.grid, fonts, font_is_auto(config)) {
            source::GridInspect::Ready { cols, rows, font } => {
                lines.push(format!("grid {cols}x{rows} font {font}"));
            }
            source::GridInspect::Note(text) => lines.push(format!("note: {text}")),
        }
    }
    let status = if let Some(codec) = encoder {
        lines.push(format!("encoder: {codec}"));
        0
    } else {
        1
    };
    CheckOutput { lines, status }
}

/// The lines from [`check_output`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckOutput {
    pub lines: Vec<String>,
    pub status: i32,
}

/// The start-up line. Tty mode names tty11. Grid mode names the grid file.
#[must_use]
pub fn cast_banner(config: &Config, device_name: &str) -> String {
    let picture = match config.source {
        Source::Tty => "tty11".to_owned(),
        Source::Grid => format!("grid {}", config.grid.display()),
    };
    format!("casting {picture} as {device_name:?}: {}", summary(config))
}

fn load_fonts(config: &Config) -> Result<Vec<(Font, Psf2)>, (PathBuf, io::Error)> {
    let mut out = Vec::with_capacity(config.fonts_to_load().len());
    for font in config.fonts_to_load() {
        let path = source::font_path(font);
        match source::load_font(&path) {
            Ok(psf) => out.push((font, psf)),
            Err(err) => return Err((path, err)),
        }
    }
    Ok(out)
}

/// Run a parsed command. Returns the process exit status.
#[must_use]
pub fn execute(command: &Command) -> i32 {
    let path = match command {
        Command::Run { config } | Command::Check { config } | Command::Bye { config } => config,
    };
    let config = match Config::load(path) {
        Ok(config) => config,
        Err(e) => {
            err(&format!("config: {e}"));
            return EXIT_CONFIG;
        }
    };
    match command {
        Command::Check { .. } => check(&config),
        Command::Bye { .. } => bye(&config),
        Command::Run { .. } => run(&config),
    }
}

fn check(config: &Config) -> i32 {
    let fonts = match load_fonts(config) {
        Ok(fonts) => fonts,
        Err((path, error)) => {
            err(&format!("font {}: {error}", path.display()));
            return EXIT_CONFIG;
        }
    };
    let host = match source::machine_udn(Path::new(source::MACHINE_ID_PATH)) {
        Ok(udn) => config::host_label(&source::nodename(), &udn),
        Err(e) => {
            err(&format!("machine id: {e}"));
            return 1;
        }
    };
    let (encoder, failed) = match encoder::resolve(&config.ffmpeg, config.encoder) {
        Ok(codec) => (Some(codec), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let output = check_output(config, &host, &fonts, encoder);
    for line in &output.lines {
        println!("{line}");
    }
    if let Some(message) = failed {
        err(&format!("encoder: {message}"));
    }
    output.status
}

/// Probe once. The choice is logged here; a failure is logged here and
/// then every stream is refused without spawning ffmpeg.
fn open_codec(config: &Config) -> Result<Codec, String> {
    match encoder::resolve(&config.ffmpeg, config.encoder) {
        Ok(codec) => {
            log::emit(
                &mut log::Stderr,
                Priority::Info,
                &format!("encoder: {codec}"),
            );
            Ok(codec)
        }
        Err(err) => {
            let message = err.to_string();
            log::emit(
                &mut log::Stderr,
                Priority::Err,
                &format!("encoder: {message}"),
            );
            Err(message)
        }
    }
}

fn bye(config: &Config) -> i32 {
    let udn = match source::machine_udn(Path::new(source::MACHINE_ID_PATH)) {
        Ok(udn) => udn,
        Err(e) => {
            err(&format!("machine id: {e}"));
            return 1;
        }
    };
    match bind_ssdp(config.interface_addr) {
        Ok(socket) => {
            discovery::byebye(&socket, &udn);
            0
        }
        Err(e) => {
            err(&format!("ssdp socket: {e}"));
            1
        }
    }
}

fn run(config: &Config) -> i32 {
    let fonts = match load_fonts(config) {
        Ok(fonts) => fonts,
        Err((path, error)) => {
            err(&format!("font {}: {error}", path.display()));
            return EXIT_CONFIG;
        }
    };
    let udn = match source::machine_udn(Path::new(source::MACHINE_ID_PATH)) {
        Ok(udn) => udn,
        Err(e) => {
            err(&format!("machine id: {e}"));
            return 1;
        }
    };
    let (screen, origin): (Arc<dyn FrameSource>, String) = if config.source == Source::Tty {
        let Some((_, font)) = fonts.into_iter().next() else {
            err("font: none configured");
            return EXIT_CONFIG;
        };
        let screen: Arc<dyn FrameSource> = Arc::new(VcsaSource::new(
            PathBuf::from(source::VCSA_PATH),
            font,
            config.palette,
        ));
        // A missing or unreadable tty11 is logged, not fatal: each stream
        // retries, and ends at once while it stays unreadable.
        let mut probe = vec![0_u8; FRAME_BYTES];
        if let Err(e) = screen.frame(&mut probe) {
            log::emit(
                &mut log::Stderr,
                Priority::Warning,
                &format!("{}: {e}", source::VCSA_PATH),
            );
        }
        (screen, source::VCSA_PATH.to_owned())
    } else {
        // A missing grid is a notice on the picture, not a failed start.
        let auto = font_is_auto(config);
        let psfs = fonts.into_iter().map(|(_, psf)| psf).collect();
        let origin = config.grid.display().to_string();
        (
            Arc::new(GridSource::new(config.grid.clone(), psfs, auto)),
            origin,
        )
    };
    let listener = match TcpListener::bind(config.listen) {
        Ok(listener) => listener,
        Err(e) => {
            err(&format!("bind {}: {e}", config.listen));
            return 1;
        }
    };
    let ssdp_socket = match bind_ssdp(config.interface_addr) {
        Ok(socket) => socket,
        Err(e) => {
            err(&format!("ssdp socket on {}: {e}", config.interface_addr));
            return 1;
        }
    };
    let base_url = config.base_url();
    let host = config::host_label(&source::nodename(), &udn);
    let device = Device {
        name: config.friendly_name(&host),
        title: config.item_title(&host),
        udn: udn.clone(),
        base_url: base_url.clone(),
    };
    let advert = Advert {
        udn,
        base_url,
        allow: config.allow.clone(),
    };
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            if let Err(e) = discovery::run(&ssdp_socket, &advert, &stop) {
                err(&format!("ssdp: {e}"));
            }
        });
    }
    // The probe runs even when it fails: the process still announces and
    // serves the description, and streams are refused (see FfmpegLive).
    let codec = open_codec(config);
    log::emit(
        &mut log::Stderr,
        Priority::Info,
        &cast_banner(config, &device.name),
    );
    let app = Arc::new(App {
        device,
        live: Arc::new(FfmpegLive {
            ffmpeg: config.ffmpeg.clone(),
            settings: config.encode_settings(),
            codec,
            source: screen,
            origin,
        }),
    });
    let server = ServerConfig {
        allow: config.allow.clone(),
        max_clients: usize::try_from(config.max_clients).unwrap_or(1),
        limits: Limits::default(),
        poll_interval: Duration::from_secs(1),
    };
    let _ = sd_notify::notify(&[NotifyState::Ready]);
    let result = http::serve(
        listener,
        server,
        app,
        Arc::new(Stats::default()),
        stop,
        || {
            let _ = sd_notify::notify(&[NotifyState::Watchdog]);
        },
    );
    match result {
        Ok(()) => 0,
        Err(e) => {
            err(&format!("serve: {e}"));
            1
        }
    }
}
