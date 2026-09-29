//! Atomic snapshot publish.
//!
//! The directory is a parameter. Production passes
//! [`llama_core::wire::SNAPSHOT_DIR`]. The directory fd is opened once with
//! `O_DIRECTORY|O_NOFOLLOW`. Each publish validates, skips a body longer than
//! the cap, writes `snapshot.json.tmp` without following a symlink, and
//! `renameat`s it onto `snapshot.json`. The temp name is unlinked first, then
//! created with `O_EXCL` and `fchmod`ed to `0640`, so a planted symlink, FIFO,
//! or hard link cannot redirect or truncate the write. There is no fsync: the
//! rename is the publication.

use std::io::Error as IoError;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use llama_core::detail::{self, MAX_FULL_NAME_CHARS};
use llama_core::log::{self, Priority, Sink};
use llama_core::names::{sanitize, sanitize_wire};
use llama_core::sample::{AiState, LlamaView, Snapshot};
use llama_core::wire::{
    self, Ai, AiWire, Host, ModelState, ModelWire, Tokens, WireError, WireSnapshot,
};
use thiserror::Error;

const TMP_NAME: &str = "snapshot.json.tmp";
const FINAL_NAME: &str = "snapshot.json";

/// Why a snapshot was not published.
#[derive(Debug, Error)]
pub enum PublishError {
    /// The directory could not be opened, or the run id could not be drawn.
    #[error("could not open the snapshot directory")]
    Directory(#[source] IoError),
    /// [`wire::validate`] or encoding failed. The previous file is unchanged.
    #[error("snapshot failed validation")]
    Invalid(#[source] WireError),
    /// The encoded body is longer than the cap. The previous file is unchanged.
    #[error("snapshot exceeds the maximum length")]
    TooLarge,
    /// The temp file could not be written or renamed. The previous file is unchanged.
    ///
    /// T22 must log [`PublishError::Write`] once per transition: one line when
    /// the failure starts, and one when a later publish succeeds. Do not log
    /// it on every tick.
    #[error("could not write the snapshot")]
    Write(#[source] IoError),
}

/// Publishes validated snapshots into one directory.
pub struct Publisher<L> {
    dir: OwnedFd,
    run_id: u64,
    seq: u64,
    max_bytes: usize,
    oversize_logged: bool,
    invalid_logged: bool,
    log: L,
}

impl<L: Sink> Publisher<L> {
    /// Open `dir` once. Fails if `dir` is a symlink or not a directory.
    pub fn open(dir: &Path, log: L) -> Result<Self, PublishError> {
        let opened = rustix::fs::open(
            dir,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::DIRECTORY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(|err| PublishError::Directory(IoError::from(err)))?;
        let run_id = random_u64().map_err(PublishError::Directory)?;
        Ok(Self {
            dir: opened,
            run_id,
            seq: 0,
            max_bytes: wire::MAX_BYTES,
            oversize_logged: false,
            invalid_logged: false,
            log,
        })
    }

    /// Reject encoded snapshots longer than `max_bytes`.
    ///
    /// The default is [`wire::MAX_BYTES`]. A valid v1 snapshot stays under that
    /// cap; tests lower it to prove the branch.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
    }

    /// Validate, then replace `snapshot.json`. A refusal leaves the previous file.
    pub fn publish(&mut self, snapshot: &Snapshot, llama: &LlamaView) -> Result<(), PublishError> {
        let seq = self.seq.saturating_add(1);
        let wire_snapshot = build(snapshot, llama, self.run_id, seq);
        if let Err(err) = wire::validate(&wire_snapshot) {
            self.log_invalid(&err);
            return Err(PublishError::Invalid(err));
        }
        let bytes = match wire::to_json(&wire_snapshot) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.log_invalid(&err);
                return Err(PublishError::Invalid(err));
            }
        };
        if bytes.len() > self.max_bytes {
            self.log_oversize();
            return Err(PublishError::TooLarge);
        }
        self.write_replace(&bytes)?;
        self.seq = seq;
        self.invalid_logged = false;
        self.oversize_logged = false;
        Ok(())
    }

    fn write_replace(&self, bytes: &[u8]) -> Result<(), PublishError> {
        match rustix::fs::unlinkat(&self.dir, TMP_NAME, rustix::fs::AtFlags::empty()) {
            Ok(()) => {}
            Err(rustix::io::Errno::NOENT) => {}
            Err(err) => return Err(PublishError::Write(IoError::from(err))),
        }
        let file = rustix::fs::openat(
            &self.dir,
            TMP_NAME,
            rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            file_mode(),
        )
        .map_err(|err| PublishError::Write(IoError::from(err)))?;
        rustix::fs::fchmod(&file, file_mode())
            .map_err(|err| PublishError::Write(IoError::from(err)))?;
        write_full(&file, bytes)?;
        drop(file);
        rustix::fs::renameat(&self.dir, TMP_NAME, &self.dir, FINAL_NAME)
            .map_err(|err| PublishError::Write(IoError::from(err)))?;
        Ok(())
    }

    fn log_oversize(&mut self) {
        if self.oversize_logged {
            return;
        }
        self.oversize_logged = true;
        log::emit(
            &mut self.log,
            Priority::Err,
            "snapshot exceeds maximum length; not published",
        );
    }

    fn log_invalid(&mut self, err: &WireError) {
        if self.invalid_logged {
            return;
        }
        self.invalid_logged = true;
        log::emit(
            &mut self.log,
            Priority::Err,
            &format!("snapshot rejected: {err}"),
        );
    }
}

fn build(snapshot: &Snapshot, llama: &LlamaView, run_id: u64, seq: u64) -> WireSnapshot {
    WireSnapshot {
        schema: wire::SCHEMA,
        run_id,
        seq,
        t_mono_ns: mono_ns(),
        t_wall_ms: wall_ms(snapshot.t_wall),
        host: Host {
            load_pct: clamp_pct(snapshot.load),
            activity_pct: clamp_activity(snapshot.activity),
            cpu_pct: clamp_pct(snapshot.cpu_pct),
            cpu_topk_pct: clamp_pct(snapshot.cpu_topk_pct),
            gpu_pct: clamp_pct(snapshot.gpu_pct),
            mem_pct: clamp_pct(snapshot.mem_pct),
            coolant_c: snapshot.coolant_c,
            cpu_c: snapshot.cpu_c,
            gpu_c: snapshot.gpu_c,
        },
        ai: ai_of(snapshot),
        tokens: Tokens {
            decoded_total: llama.decoded_total,
        },
    }
}

fn ai_of(snapshot: &Snapshot) -> Ai {
    match snapshot.ai {
        AiState::Loaded => Ai {
            state: AiWire::Loaded,
            models: snapshot
                .models
                .iter()
                .take(wire::MAX_MODELS)
                .map(|model| {
                    let name = sanitize_wire(&model.name);
                    // Gauges only for a backend without `/slots`; llama.cpp
                    // keeps its slot view on the tty.
                    let gauges = model.backend.filter(|info| !info.kind.has_slots());
                    ModelWire {
                        full_name: wire_full_name(model.full_name.as_deref(), &name),
                        detail: model.detail.clone().filter(detail::is_valid),
                        name,
                        state: model_state(&model.state),
                        backend: model.backend.map(|info| info.kind),
                        running: gauges.and_then(|info| info.running).map(cap_reqs),
                        queued: gauges.and_then(|info| info.queued).map(cap_reqs),
                        kv_fill: gauges
                            .and_then(|info| info.kv_permille)
                            .filter(|permille| *permille <= 1000)
                            .map(|permille| f32::from(permille) / 1000.0),
                    }
                })
                .collect(),
        },
        AiState::Idle => Ai {
            state: AiWire::Idle,
            models: Vec::new(),
        },
        AiState::Down | AiState::NoData => Ai {
            state: AiWire::Down,
            models: Vec::new(),
        },
    }
}

fn cap_reqs(n: u16) -> u16 {
    n.min(wire::MAX_REQS)
}

/// The full name when it adds something to `name` and is canonical.
fn wire_full_name(full: Option<&str>, name: &str) -> Option<String> {
    let full = sanitize(full?, MAX_FULL_NAME_CHARS);
    (!full.is_empty() && full != name).then_some(full)
}

fn file_mode() -> rustix::fs::Mode {
    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR | rustix::fs::Mode::RGRP
}

/// Finite host percentages enter the wire inside 0..=100. Anything else is absent.
fn clamp_pct(value: Option<f32>) -> Option<f32> {
    clamp_to(value, 100.0)
}

/// Activity keeps its redline: 0..=[`wire::ACTIVITY_MAX_PCT`].
fn clamp_activity(value: Option<f32>) -> Option<f32> {
    clamp_to(value, wire::ACTIVITY_MAX_PCT)
}

fn clamp_to(value: Option<f32>, high: f32) -> Option<f32> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    Some(value.clamp(0.0, high))
}

fn model_state(state: &str) -> ModelState {
    match state {
        "ready" => ModelState::Ready,
        "starting" => ModelState::Starting,
        "stopping" => ModelState::Stopping,
        _ => ModelState::Other,
    }
}

fn mono_ns() -> u64 {
    let timespec = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    let sec = u64::try_from(timespec.tv_sec).unwrap_or(0);
    let nsec = u64::try_from(timespec.tv_nsec).unwrap_or(0);
    sec.saturating_mul(1_000_000_000).saturating_add(nsec)
}

fn wall_ms(wall: SystemTime) -> u64 {
    match wall.duration_since(UNIX_EPOCH) {
        Ok(duration) => u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

fn random_u64() -> Result<u64, IoError> {
    let mut buf = [0u8; 8];
    let mut filled = 0;
    while filled < buf.len() {
        let n = rustix::rand::getrandom(&mut buf[filled..], rustix::rand::GetRandomFlags::empty())
            .map_err(IoError::from)?;
        if n == 0 {
            return Err(IoError::other("getrandom returned no bytes"));
        }
        filled += n;
    }
    Ok(u64::from_ne_bytes(buf))
}

fn write_full(fd: &OwnedFd, mut bytes: &[u8]) -> Result<(), PublishError> {
    while !bytes.is_empty() {
        match rustix::io::write(fd, bytes) {
            Ok(0) => {
                return Err(PublishError::Write(IoError::new(
                    std::io::ErrorKind::WriteZero,
                    "snapshot write made no progress",
                )));
            }
            Ok(n) => bytes = &bytes[n..],
            Err(rustix::io::Errno::INTR) => {}
            Err(err) => return Err(PublishError::Write(IoError::from(err))),
        }
    }
    Ok(())
}
