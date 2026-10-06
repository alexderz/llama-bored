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

use crate::config::Config;
use crate::discovery::{self, Advert};
use crate::dlna::Device;
use crate::encoder::{Encoder, Settings};
use crate::http::{self, App, Limits, Live, ServerConfig, Stats};
use crate::render::FRAME_BYTES;
use crate::source::{self, FrameSource, VcsaSource};
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

/// The production stream: ffmpeg fed from tty11.
pub struct FfmpegLive {
    pub ffmpeg: PathBuf,
    pub settings: Settings,
    pub source: Arc<dyn FrameSource>,
}

impl Live for FfmpegLive {
    fn open(&self) -> io::Result<Box<dyn Read + Send>> {
        // An unreadable tty11 is a 500 with the reason logged, not an
        // empty 200.
        let mut probe = vec![0_u8; FRAME_BYTES];
        self.source
            .frame(&mut probe)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", source::VCSA_PATH)))?;
        drop(probe);
        let encoder = Encoder::start(&self.ffmpeg, self.settings, Arc::clone(&self.source))?;
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
        "listen {} allow [{}] interface_addr {} name {:?} fps {} max_clients {} bitrate_kbps {} keyframe_s {} preroll_s {} ffmpeg {} font {} palette {}",
        config.listen,
        nets.join(", "),
        config.interface_addr,
        config.name,
        config.fps,
        config.max_clients,
        config.bitrate_kbps,
        config.keyframe_s,
        config.preroll_s,
        config.ffmpeg.display(),
        config.font.label(),
        config.palette.label(),
    )
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
    let font_path = source::font_path(config.font);
    if let Err(e) = source::load_font(&font_path) {
        err(&format!("font {}: {e}", font_path.display()));
        return EXIT_CONFIG;
    }
    println!("config ok: {}", summary(config));
    println!(
        "stream url: {}{}",
        config.base_url(),
        crate::dlna::STREAM_PATH
    );
    0
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
    let font_path = source::font_path(config.font);
    let font = match source::load_font(&font_path) {
        Ok(font) => font,
        Err(e) => {
            err(&format!("font {}: {e}", font_path.display()));
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
    drop(probe);
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
    let device = Device {
        name: config.name.clone(),
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
    log::emit(
        &mut log::Stderr,
        Priority::Info,
        &format!("casting tty11: {}", summary(config)),
    );
    let app = Arc::new(App {
        device,
        live: Arc::new(FfmpegLive {
            ffmpeg: config.ffmpeg.clone(),
            settings: config.encode_settings(),
            source: screen,
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
