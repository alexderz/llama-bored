//! `cast.toml`: where to listen, who may connect, how SSDP announces the
//! server, and how the stream is encoded. The screen (`/dev/vcsa11`), the
//! machine id and the font directory are compile-time constants in
//! `source.rs`; the config file itself is the CLI argument.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

pub use llama_core::palette::Palette;

use crate::acl::{Allowlist, Cidr, CidrError};
use crate::encoder::Settings;

/// Largest config file read.
pub const MAX_CONFIG_BYTES: u64 = 64 * 1024;
/// Most allowlist entries.
pub const MAX_ALLOW: usize = 32;
/// The documented TCP port (next to llama-metrics' 19477).
pub const DEFAULT_PORT: u16 = 19478;
/// The SSDP port. The HTTP listener may not use it.
pub const SSDP_PORT: u16 = 1900;
/// Replaced in `name` and `title` by this host's name (#21).
pub const HOST_PLACEHOLDER: &str = "{host}";
/// Default friendly name, shown in the TV's source list. The host name
/// tells two servers (or two boxes) apart there (#21).
pub const DEFAULT_NAME: &str = "llama-bored ({host})";
/// Default title of the one video item.
pub const DEFAULT_TITLE: &str = "tty11 on {host}";
/// Longest friendly name or title, in characters, before and after the
/// host is put in (the expanded text is cut to this).
pub const MAX_NAME_CHARS: usize = 64;
/// Longest host name put into `{host}`, in characters.
pub const MAX_HOST_CHARS: usize = 32;
/// Default and bounds of `fps`.
pub const DEFAULT_FPS: u32 = 2;
pub const MAX_FPS: u32 = 5;
/// Default and bounds of `max_clients`.
pub const DEFAULT_MAX_CLIENTS: u32 = 2;
pub const MAX_CLIENTS: u32 = 4;
/// Default and bounds of `bitrate_kbps` (0 turns the floor off).
pub const DEFAULT_BITRATE_KBPS: u32 = 4000;
pub const MIN_BITRATE_KBPS: u32 = 500;
pub const MAX_BITRATE_KBPS: u32 = 20_000;
/// Default and bound of `keyframe_s`.
pub const DEFAULT_KEYFRAME_S: u32 = 1;
pub const MAX_KEYFRAME_S: u32 = 10;
/// Default and bound of `preroll_s`.
pub const DEFAULT_PREROLL_S: u32 = 3;
pub const MAX_PREROLL_S: u32 = 10;
/// Default encoder.
pub const DEFAULT_FFMPEG: &str = "/usr/bin/ffmpeg";
/// Longest `ffmpeg` path.
pub const MAX_PATH_BYTES: usize = 255;

/// The console font tty11 uses. Must match `[tty] font` in watch.toml.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
pub enum Font {
    #[default]
    #[serde(rename = "12x24")]
    Hack12x24,
    #[serde(rename = "12x22")]
    Hack12x22,
    #[serde(rename = "10x18")]
    Hack10x18,
}

impl Font {
    /// The file name under the installed font directory.
    #[must_use]
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Hack12x24 => "llama-hack-12x24.psfu",
            Self::Hack12x22 => "llama-hack-12x22.psfu",
            Self::Hack10x18 => "llama-hack-10x18.psfu",
        }
    }

    /// The `font` value as written in the config.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Hack12x24 => "12x24",
            Self::Hack12x22 => "12x22",
            Self::Hack10x18 => "10x18",
        }
    }
}

/// Why the config was refused. Every variant exits 2.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read the config: {0}")]
    Read(String),
    #[error("config is larger than 64 KiB")]
    TooLarge,
    #[error("config is not valid TOML for llama-cast: {0}")]
    Parse(String),
    #[error("listen {0:?} is not IPv4:PORT")]
    Listen(String),
    #[error("listen port must not be 0 or 1900 (SSDP)")]
    ListenPort,
    #[error("allow must list 1..=32 networks")]
    AllowCount,
    #[error("allow: {0}")]
    Allow(#[from] CidrError),
    #[error("allow lists {0} twice")]
    AllowDuplicate(String),
    #[error(
        "interface_addr {0:?} must be this host's IPv4 LAN address (not 0.0.0.0, multicast or broadcast)"
    )]
    InterfaceAddr(String),
    #[error("interface_addr is required when listen is 0.0.0.0")]
    InterfaceAddrMissing,
    #[error("interface_addr {0} differs from the listen address {1}")]
    InterfaceAddrMismatch(Ipv4Addr, Ipv4Addr),
    #[error("name must be 1..=64 printable characters, with braces only in {{host}}")]
    Name,
    #[error("title must be 1..=64 printable characters, with braces only in {{host}}")]
    Title,
    #[error("fps must be 1..=5")]
    Fps,
    #[error("max_clients must be 1..=4")]
    MaxClients,
    #[error("bitrate_kbps must be 0 (off) or 500..=20000")]
    Bitrate,
    #[error("keyframe_s must be 1..=10")]
    Keyframe,
    #[error("preroll_s must be 0..=10")]
    Preroll,
    #[error("ffmpeg must be an absolute, normalised path of at most 255 bytes")]
    Ffmpeg,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    listen: String,
    allow: Vec<String>,
    interface_addr: Option<String>,
    #[serde(default = "default_name")]
    name: String,
    #[serde(default = "default_title")]
    title: String,
    #[serde(default = "default_fps")]
    fps: u32,
    #[serde(default = "default_max_clients")]
    max_clients: u32,
    #[serde(default = "default_bitrate")]
    bitrate_kbps: u32,
    #[serde(default = "default_keyframe")]
    keyframe_s: u32,
    #[serde(default = "default_preroll")]
    preroll_s: u32,
    #[serde(default = "default_ffmpeg")]
    ffmpeg: String,
    #[serde(default)]
    font: Font,
    #[serde(default)]
    palette: Palette,
}

fn default_name() -> String {
    DEFAULT_NAME.to_owned()
}

fn default_title() -> String {
    DEFAULT_TITLE.to_owned()
}

fn default_fps() -> u32 {
    DEFAULT_FPS
}

fn default_max_clients() -> u32 {
    DEFAULT_MAX_CLIENTS
}

fn default_bitrate() -> u32 {
    DEFAULT_BITRATE_KBPS
}

fn default_keyframe() -> u32 {
    DEFAULT_KEYFRAME_S
}

fn default_preroll() -> u32 {
    DEFAULT_PREROLL_S
}

fn default_ffmpeg() -> String {
    DEFAULT_FFMPEG.to_owned()
}

/// A validated config.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Config {
    /// TCP address to bind (IPv4; SSDP here is IPv4 only).
    pub listen: SocketAddrV4,
    /// Peers allowed to connect or to be answered over SSDP.
    pub allow: Allowlist,
    /// The LAN address SSDP joins the multicast group on, sends from, and
    /// advertises in `LOCATION` and the stream URL.
    pub interface_addr: Ipv4Addr,
    /// Friendly name template (`{host}` is this host's name).
    pub name: String,
    /// The video item's title template (`{host}` likewise).
    pub title: String,
    /// Frames rendered per second.
    pub fps: u32,
    /// Streams served at once; more are answered 503.
    pub max_clients: u32,
    /// Constant bitrate, kbit/s (0: quality-based, no floor).
    pub bitrate_kbps: u32,
    /// Keyframe interval, seconds.
    pub keyframe_s: u32,
    /// Seconds of the first frame each new viewer gets at once.
    pub preroll_s: u32,
    /// The encoder binary.
    pub ffmpeg: PathBuf,
    /// The tty11 console font.
    pub font: Font,
    /// The colours tty11 shows; must match `[tty] palette` in watch.toml.
    pub palette: Palette,
}

impl Config {
    /// Parse and validate TOML text.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let raw: Raw = toml::from_str(text).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let listen = match raw.listen.parse::<SocketAddr>() {
            Ok(SocketAddr::V4(v4)) => v4,
            _ => return Err(ConfigError::Listen(raw.listen.clone())),
        };
        if listen.port() == 0 || listen.port() == SSDP_PORT {
            return Err(ConfigError::ListenPort);
        }
        if raw.allow.is_empty() || raw.allow.len() > MAX_ALLOW {
            return Err(ConfigError::AllowCount);
        }
        let mut nets: Vec<Cidr> = Vec::with_capacity(raw.allow.len());
        for text in &raw.allow {
            let net: Cidr = text.parse()?;
            if nets.contains(&net) {
                return Err(ConfigError::AllowDuplicate(net.to_string()));
            }
            nets.push(net);
        }
        let interface_addr = interface_addr(raw.interface_addr.as_deref(), *listen.ip())?;
        if !valid_template(&raw.name) {
            return Err(ConfigError::Name);
        }
        if !valid_template(&raw.title) {
            return Err(ConfigError::Title);
        }
        if raw.fps == 0 || raw.fps > MAX_FPS {
            return Err(ConfigError::Fps);
        }
        if raw.max_clients == 0 || raw.max_clients > MAX_CLIENTS {
            return Err(ConfigError::MaxClients);
        }
        if raw.bitrate_kbps != 0
            && !(MIN_BITRATE_KBPS..=MAX_BITRATE_KBPS).contains(&raw.bitrate_kbps)
        {
            return Err(ConfigError::Bitrate);
        }
        if raw.keyframe_s == 0 || raw.keyframe_s > MAX_KEYFRAME_S {
            return Err(ConfigError::Keyframe);
        }
        if raw.preroll_s > MAX_PREROLL_S {
            return Err(ConfigError::Preroll);
        }
        let ffmpeg = PathBuf::from(&raw.ffmpeg);
        if !clean_absolute(&raw.ffmpeg) || raw.ffmpeg.len() > MAX_PATH_BYTES {
            return Err(ConfigError::Ffmpeg);
        }
        Ok(Self {
            listen,
            allow: Allowlist::new(nets),
            interface_addr,
            name: raw.name,
            title: raw.title,
            fps: raw.fps,
            max_clients: raw.max_clients,
            bitrate_kbps: raw.bitrate_kbps,
            keyframe_s: raw.keyframe_s,
            preroll_s: raw.preroll_s,
            ffmpeg,
            font: raw.font,
            palette: raw.palette,
        })
    }

    /// Read `path` (the `--config` argument), capped at 64 KiB, and validate.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let meta = std::fs::metadata(path).map_err(|err| ConfigError::Read(err.to_string()))?;
        if !meta.is_file() {
            return Err(ConfigError::Read("not a regular file".to_owned()));
        }
        if meta.len() > MAX_CONFIG_BYTES {
            return Err(ConfigError::TooLarge);
        }
        let text =
            std::fs::read_to_string(path).map_err(|err| ConfigError::Read(err.to_string()))?;
        if text.len() as u64 > MAX_CONFIG_BYTES {
            return Err(ConfigError::TooLarge);
        }
        Self::from_toml(&text)
    }

    /// The friendly name with `host` (from [`host_label`]) put in.
    #[must_use]
    pub fn friendly_name(&self, host: &str) -> String {
        expand(&self.name, host)
    }

    /// The video item's title with `host` put in.
    #[must_use]
    pub fn item_title(&self, host: &str) -> String {
        expand(&self.title, host)
    }

    /// The encoder knobs.
    #[must_use]
    pub fn encode_settings(&self) -> Settings {
        Settings {
            fps: self.fps,
            bitrate_kbps: self.bitrate_kbps,
            keyframe_s: self.keyframe_s,
            preroll_s: self.preroll_s,
        }
    }

    /// `http://ADDR:PORT`, the base of every URL the server advertises.
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.interface_addr, self.listen.port())
    }
}

/// 1..=64 printable characters, no surrounding space, and `{`/`}` only
/// as `{host}`.
fn valid_template(text: &str) -> bool {
    let chars = text.chars().count();
    chars > 0
        && chars <= MAX_NAME_CHARS
        && !text.chars().any(char::is_control)
        && text.trim() == text
        && !text.replace(HOST_PLACEHOLDER, "").contains(['{', '}'])
}

/// `template` with every `{host}` replaced, cut to 64 characters.
#[must_use]
pub fn expand(template: &str, host: &str) -> String {
    let text = template.replace(HOST_PLACEHOLDER, host);
    let cut: String = text.chars().take(MAX_NAME_CHARS).collect();
    cut.trim_end().to_owned()
}

/// The name put into `{host}`: the first label of `nodename` (the kernel
/// host name), only ASCII letters, digits, `-` and `_` kept, at most 32
/// characters. When that leaves nothing, `host-` and the first 8 hex
/// digits of the UDN (a hash, not the machine id), still one per host.
#[must_use]
pub fn host_label(nodename: &[u8], udn: &str) -> String {
    let first = nodename.split(|b| *b == b'.').next().unwrap_or_default();
    let label: String = first
        .iter()
        .filter(|b| b.is_ascii_alphanumeric() || **b == b'-' || **b == b'_')
        .take(MAX_HOST_CHARS)
        .map(|b| char::from(*b))
        .collect();
    // "(none)" is the kernel's name for an unnamed host.
    if !label.is_empty() && first != b"(none)" {
        return label;
    }
    let hex: String = udn
        .strip_prefix("uuid:")
        .unwrap_or(udn)
        .chars()
        .filter(char::is_ascii_hexdigit)
        .take(8)
        .collect();
    format!("host-{hex}")
}

fn interface_addr(text: Option<&str>, listen: Ipv4Addr) -> Result<Ipv4Addr, ConfigError> {
    let addr = match text {
        Some(text) => match text.parse::<IpAddr>() {
            Ok(IpAddr::V4(v4)) => v4,
            _ => return Err(ConfigError::InterfaceAddr(text.to_owned())),
        },
        None if listen.is_unspecified() => return Err(ConfigError::InterfaceAddrMissing),
        None => listen,
    };
    if addr.is_unspecified() || addr.is_multicast() || addr.is_broadcast() {
        return Err(ConfigError::InterfaceAddr(addr.to_string()));
    }
    if !listen.is_unspecified() && listen != addr {
        return Err(ConfigError::InterfaceAddrMismatch(addr, listen));
    }
    Ok(addr)
}

/// Absolute, every segment non-empty and neither `.` nor `..`, no control
/// bytes, and not `/` itself.
fn clean_absolute(text: &str) -> bool {
    let Some(rest) = text.strip_prefix('/') else {
        return false;
    };
    !rest.is_empty()
        && !text.bytes().any(|b| b.is_ascii_control())
        && rest
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
}
