//! `metrics.toml`: where to listen, who may connect, and when the snapshot
//! is stale. There is no path key: the snapshot path is
//! `llama_core::wire::SNAPSHOT_PATH`, and the config file itself is the CLI
//! argument.

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

use crate::acl::{Allowlist, Cidr, CidrError};

/// Largest config file read.
pub const MAX_CONFIG_BYTES: u64 = 64 * 1024;
/// Most allowlist entries.
pub const MAX_ALLOW: usize = 32;
/// Top of `max_conns`.
pub const MAX_CONNS: u32 = 64;
/// Default `max_conns`.
pub const DEFAULT_MAX_CONNS: u32 = 16;
/// Default `stale_after_s`.
pub const DEFAULT_STALE_AFTER_S: u64 = 5;
/// Top of `stale_after_s`.
pub const MAX_STALE_AFTER_S: u64 = 300;
/// The documented port (outside the Prometheus exporter registry, which is
/// full from 9100 to 9999).
pub const DEFAULT_PORT: u16 = 19477;

/// Why the config was refused. Every variant exits 2.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read the config: {0}")]
    Read(String),
    #[error("config is larger than 64 KiB")]
    TooLarge,
    #[error("config is not valid TOML for llama-metrics: {0}")]
    Parse(String),
    #[error("listen {0:?} is not IP:PORT")]
    Listen(String),
    #[error("listen port must not be 0")]
    ListenPort,
    #[error("allow must list 1..=32 networks")]
    AllowCount,
    #[error("allow: {0}")]
    Allow(#[from] CidrError),
    #[error("allow lists {0} twice")]
    AllowDuplicate(String),
    #[error("max_conns must be 1..=64")]
    MaxConns,
    #[error("stale_after_s must be 1..=300")]
    StaleAfter,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    listen: String,
    allow: Vec<String>,
    #[serde(default = "default_max_conns")]
    max_conns: u32,
    #[serde(default = "default_stale_after_s")]
    stale_after_s: u64,
}

fn default_max_conns() -> u32 {
    DEFAULT_MAX_CONNS
}

fn default_stale_after_s() -> u64 {
    DEFAULT_STALE_AFTER_S
}

/// A validated config.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Config {
    /// Socket address to bind. `[::]:PORT` is dual-stack on Linux.
    pub listen: SocketAddr,
    /// Peers allowed to connect.
    pub allow: Allowlist,
    /// Connections served at once; more are answered 503 and closed.
    pub max_conns: u32,
    /// A snapshot older than this is stale.
    pub stale_after: Duration,
}

impl Config {
    /// Parse and validate TOML text.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let raw: Raw = toml::from_str(text).map_err(|err| ConfigError::Parse(err.to_string()))?;
        let listen: SocketAddr = raw
            .listen
            .parse()
            .map_err(|_| ConfigError::Listen(raw.listen.clone()))?;
        if listen.port() == 0 {
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
        if raw.max_conns == 0 || raw.max_conns > MAX_CONNS {
            return Err(ConfigError::MaxConns);
        }
        if raw.stale_after_s == 0 || raw.stale_after_s > MAX_STALE_AFTER_S {
            return Err(ConfigError::StaleAfter);
        }
        Ok(Self {
            listen,
            allow: Allowlist::new(nets),
            max_conns: raw.max_conns,
            stale_after: Duration::from_secs(raw.stale_after_s),
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
}
