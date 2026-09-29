//! CLI and the production wiring: bind, notify systemd, serve.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use llama_core::log::{self, Priority, Sink};
use sd_notify::NotifyState;

use crate::config::Config;
use crate::expo::{self, Rejected, Scrape};
use crate::http::{self, Body, Limits, ServerConfig, Stats};
use crate::snapshot::{self, SnapshotFile};

/// Exit status for a usage or config error. The unit does not restart on it.
pub const EXIT_CONFIG: i32 = 2;

pub const USAGE: &str =
    "usage: llama-metrics run --config PATH\n       llama-metrics check --config PATH";

/// Parsed command line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Serve `/metrics`.
    Run { config: PathBuf },
    /// Validate the config and print what it would do.
    Check { config: PathBuf },
}

/// `run --config PATH` or `check --config PATH`, nothing else.
pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let args: Vec<String> = args.into_iter().collect();
    match args.as_slice() {
        [cmd, flag, path] if flag == "--config" && !path.is_empty() => match cmd.as_str() {
            "run" => Ok(Command::Run {
                config: PathBuf::from(path),
            }),
            "check" => Ok(Command::Check {
                config: PathBuf::from(path),
            }),
            _ => Err(USAGE.to_owned()),
        },
        _ => Err(USAGE.to_owned()),
    }
}

/// Monotonic nanoseconds. Production is [`snapshot::mono_now_ns`].
pub type Clock = Box<dyn Fn() -> u64 + Send + Sync>;

/// Where [`SnapshotMetrics`] logs. Production is stderr.
pub type LogSink = Box<dyn Sink + Send>;

/// The last rejection reason that was logged, and the sink.
struct RejectLog {
    last: Option<String>,
    sink: LogSink,
}

/// Renders `/metrics` from a fresh read of the snapshot on every scrape.
pub struct SnapshotMetrics {
    file: SnapshotFile,
    stale_after: Duration,
    clock: Clock,
    log: Mutex<RejectLog>,
}

impl SnapshotMetrics {
    #[must_use]
    pub fn new(file: SnapshotFile, stale_after: Duration, clock: Clock) -> Self {
        Self {
            file,
            stale_after,
            clock,
            log: Mutex::new(RejectLog {
                last: None,
                sink: Box::new(log::Stderr),
            }),
        }
    }

    /// Log to `sink` instead of stderr. Tests capture the lines.
    #[must_use]
    pub fn with_log(self, sink: LogSink) -> Self {
        Self {
            log: Mutex::new(RejectLog { last: None, sink }),
            ..self
        }
    }

    /// `snapshot rejected: <reason>` once per distinct reason, at warning;
    /// a good read resets it, so a later failure logs again (#12).
    fn note(&self, read: &Result<llama_core::wire::WireSnapshot, snapshot::ReadError>) {
        let Ok(mut state) = self.log.lock() else {
            return;
        };
        let state = &mut *state;
        match read {
            Ok(_) => {
                if state.last.take().is_some() {
                    state
                        .sink
                        .write_line(&log::format_line(Priority::Info, "snapshot accepted"));
                }
            }
            Err(err) => {
                let reason = err.reason();
                if state.last.as_deref() != Some(reason.as_str()) {
                    state.sink.write_line(&log::format_line(
                        Priority::Warning,
                        &format!("snapshot rejected: {reason}"),
                    ));
                    state.last = Some(reason);
                }
            }
        }
    }
}

impl Body for SnapshotMetrics {
    fn metrics(&self, rejected: Rejected) -> String {
        let read = self.file.read();
        self.note(&read);
        expo::render(&Scrape {
            read: &read,
            now_ns: (self.clock)(),
            stale_after: self.stale_after,
            rejected,
        })
    }
}

/// Run a parsed command. Returns the process exit status.
#[must_use]
pub fn execute(command: &Command) -> i32 {
    let (path, run) = match command {
        Command::Run { config } => (config, true),
        Command::Check { config } => (config, false),
    };
    let config = match Config::load(path) {
        Ok(config) => config,
        Err(err) => {
            log::emit(&mut log::Stderr, Priority::Err, &format!("config: {err}"));
            return EXIT_CONFIG;
        }
    };
    let nets: Vec<String> = config
        .allow
        .nets()
        .iter()
        .map(ToString::to_string)
        .collect();
    let summary = format!(
        "listen {} allow [{}] max_conns {} stale_after_s {}",
        config.listen,
        nets.join(", "),
        config.max_conns,
        config.stale_after.as_secs()
    );
    if !run {
        println!("config ok: {summary}");
        return 0;
    }
    let listener = match TcpListener::bind(config.listen) {
        Ok(listener) => listener,
        Err(err) => {
            log::emit(
                &mut log::Stderr,
                Priority::Err,
                &format!("bind {}: {err}", config.listen),
            );
            return 1;
        }
    };
    log::emit(
        &mut log::Stderr,
        Priority::Info,
        &format!("serving /metrics: {summary}"),
    );
    let body: Arc<dyn Body> = Arc::new(SnapshotMetrics::new(
        SnapshotFile::published(),
        config.stale_after,
        Box::new(snapshot::mono_now_ns),
    ));
    let server = ServerConfig {
        allow: config.allow.clone(),
        max_conns: usize::try_from(config.max_conns).unwrap_or(1),
        limits: Limits::default(),
        poll_interval: Duration::from_secs(1),
    };
    let _ = sd_notify::notify(&[NotifyState::Ready]);
    let result = http::serve(
        listener,
        server,
        body,
        Arc::new(Stats::default()),
        Arc::new(AtomicBool::new(false)),
        || {
            let _ = sd_notify::notify(&[NotifyState::Watchdog]);
        },
    );
    match result {
        Ok(()) => 0,
        Err(err) => {
            log::emit(&mut log::Stderr, Priority::Err, &format!("serve: {err}"));
            1
        }
    }
}
