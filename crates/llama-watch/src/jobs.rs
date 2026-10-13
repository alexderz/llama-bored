//! Job status files for the job view.
//!
//! Read only, and only from the poller: the frame path never opens these
//! files. `jobs.dir` unset means this module does not open anything.
//! The directory is opened with `O_DIRECTORY | O_NOFOLLOW` and each file
//! with `openat` `O_NOFOLLOW`, so a symlink cannot leave the directory.
//! Nothing here is written. This file is not on the safety_scan
//! `WRITE_FILES` allowlist and must not gain a write.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use llama_core::detail::is_token;
use llama_core::log::{self, Priority, Sink};
use rustix::fd::OwnedFd;
use rustix::fs::{AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;

use crate::tty::layout::cut_text;

/// How often the directory is listed and files are stat'd.
pub(crate) const CHECK_EVERY: Duration = Duration::from_secs(1);
/// Largest file read. Anything longer is "too big" and its body is not parsed.
pub(crate) const MAX_JOB_BYTES: u64 = 64 * 1024;
/// Deepest JSON array or object nesting accepted. Deeper is "too deep".
pub(crate) const MAX_JOB_DEPTH: usize = 8;
/// Most files opened in one refresh.
pub(crate) const MAX_JOB_FILES: usize = 8;
/// Most `*.json` names collected from the directory before ranking by mtime.
pub(crate) const MAX_JOB_NAMES: usize = 512;
/// Longest file stem used as a path component.
pub(crate) const MAX_STEM: usize = 128;
/// Parsed durations longer than this are stored as this (ten years).
pub(crate) const MAX_DURATION_S: u64 = 10 * 365 * 24 * 60 * 60;
/// An mtime more than this many seconds in the future counts as stale.
pub(crate) const FUTURE_STALE_S: u64 = 60;
/// Most points kept in one series, newest retained.
pub(crate) const MAX_SERIES_POINTS: usize = 512;
/// Most `rates` kept.
pub(crate) const MAX_RATES: usize = 4;
/// Most `values` kept.
pub(crate) const MAX_VALUES: usize = 8;
/// Most `series` kept.
pub(crate) const MAX_SERIES: usize = 4;
/// Most `phases` kept.
pub(crate) const MAX_PHASES: usize = 12;

const MAX_NAME: usize = 32;
const MAX_TITLE: usize = 48;
const MAX_LABEL: usize = 16;
const MAX_UNIT: usize = 12;
const MAX_RESULT: usize = 32;

const DIR_CAUSE: &str = "jobs dir unreadable";
const TOO_MANY: &str = "more than eight files";
const UNREADABLE: &str = "unreadable";
const TOO_BIG: &str = "too big";
const NOT_JSON: &str = "not json";
const TOO_DEEP: &str = "too deep";
const NOT_FILE: &str = "not a file";
const BAD_SCHEMA: &str = "bad schema";
const BAD_STATUS: &str = "bad status";

/// `st_mtime`, its nanoseconds, `st_size` and inode, so a later stat can
/// see whether the file changed without reading it.
#[derive(Clone, Copy, Eq, PartialEq)]
struct Stamp {
    mtime: i64,
    nsec: u64,
    len: i64,
    ino: u64,
}

struct Entry {
    stamp: Option<Stamp>,
    job: Option<Job>,
    checked: Instant,
    /// Last logged fault for this stem, without the stem. `None` after a
    /// successful read or a missing file (missing is not an error).
    cause: Option<&'static str>,
}

/// One parsed status file. Text is already cleaned and capped.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    /// File stem (`<stem>.json`).
    pub file: String,
    pub name: String,
    pub kind: Option<String>,
    pub title: Option<String>,
    pub state: JobState,
    pub phase: Option<JobPhase>,
    pub step: Option<u64>,
    pub total: Option<u64>,
    pub elapsed_s: Option<u64>,
    pub eta_s: Option<u64>,
    pub rate: Option<JobRate>,
    pub rates: Vec<JobNumber>,
    pub values: Vec<JobNumber>,
    pub series: Vec<JobSeries>,
    pub phases: Vec<JobPhaseRow>,
    /// File mtime, for freshness and "most recently changed".
    pub mtime: SystemTime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobState {
    Running,
    Done,
    Failed,
}

impl JobState {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "running" => Self::Running,
            "done" => Self::Done,
            "failed" => Self::Failed,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobPhase {
    pub label: String,
    pub index: Option<u64>,
    pub of: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobRate {
    pub value: f64,
    pub unit: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobNumber {
    pub label: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobSeries {
    pub label: String,
    pub points: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobPhaseRow {
    pub label: String,
    pub mark: Option<PhaseMark>,
    pub elapsed_s: Option<u64>,
    pub result: Option<String>,
}

/// Phase row state mark, including `queued` which is not a job `state`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseMark {
    Done,
    Running,
    Queued,
    Failed,
}

impl PhaseMark {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "done" => Self::Done,
            "running" => Self::Running,
            "queued" => Self::Queued,
            "failed" => Self::Failed,
            _ => return None,
        })
    }
}

/// The jobs directory and the files read from it.
pub(crate) struct Jobs {
    dir: Option<PathBuf>,
    fd: Option<OwnedFd>,
    files: HashMap<String, Entry>,
    /// Stems of the last successful snapshot, newest mtime first.
    stems: Vec<String>,
    /// Causes already logged. A cause logs again only after it cleared.
    logged: HashSet<String>,
    /// When the directory was last opened, or the open last failed.
    dir_checked: Option<Instant>,
    /// When files were last listed and stat'd.
    listed: Option<Instant>,
    /// Directory opens, stats and file opens. Tests prove an unset
    /// directory and a hostile name never touch the filesystem.
    #[cfg_attr(not(test), allow(dead_code))]
    ops: u64,
    /// File bodies read. A stat that finds the file unchanged does not count.
    #[cfg_attr(not(test), allow(dead_code))]
    reads: u64,
}

impl Jobs {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            fd: None,
            files: HashMap::new(),
            stems: Vec::new(),
            logged: HashSet::new(),
            dir_checked: None,
            listed: None,
            ops: 0,
            reads: 0,
        }
    }

    /// Parsed jobs from the last refresh, newest mtime first, including
    /// stale and finished files. A file that failed a check is absent.
    pub(crate) fn jobs(&self) -> Vec<&Job> {
        self.stems
            .iter()
            .filter_map(|stem| self.files.get(stem).and_then(|entry| entry.job.as_ref()))
            .collect()
    }

    /// Re-list the directory and re-read files whose mtime or size changed.
    /// `now` is the poller's clock, so tests can move it without sleeping.
    /// A call inside [`CHECK_EVERY`] of the last list does nothing.
    pub(crate) fn refresh(&mut self, now: Instant, log: &mut impl Sink) {
        if self.dir.is_none() {
            return;
        }
        if self
            .listed
            .is_some_and(|at| now.saturating_duration_since(at) < CHECK_EVERY)
        {
            return;
        }
        self.listed = Some(now);
        if !self.dir_still_here() {
            self.fd = None;
        }
        if self.fd.is_none() {
            self.ops = self.ops.saturating_add(1);
            self.dir_checked = Some(now);
            let opened = self.dir.as_deref().map(open_dir);
            match opened {
                Some(Ok(fd)) => {
                    self.fd = Some(fd);
                    self.logged.remove(DIR_CAUSE);
                }
                _ => {
                    self.log_cause(DIR_CAUSE, log);
                    self.stems.clear();
                    self.files.clear();
                    return;
                }
            }
        }
        let names = match self.list_stems() {
            Ok(names) => names,
            Err(()) => {
                self.fd = None;
                self.dir_checked = Some(now);
                self.log_cause(DIR_CAUSE, log);
                self.stems.clear();
                return;
            }
        };
        if names.len() > MAX_JOB_FILES {
            self.log_cause(TOO_MANY, log);
        } else {
            self.logged.remove(TOO_MANY);
        }
        let mut ranked: Vec<(String, SystemTime)> = Vec::new();
        for name in &names {
            match self.stat_stem(name) {
                Ok(Some(stat)) => ranked.push((name.clone(), mtime_of(&stat))),
                Ok(None) => {}
                Err(()) => {
                    self.fd = None;
                    self.dir_checked = Some(now);
                    self.log_cause(DIR_CAUSE, log);
                    self.stems.clear();
                    return;
                }
            }
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(MAX_JOB_FILES);
        let keep: HashSet<&str> = ranked.iter().map(|(stem, _)| stem.as_str()).collect();
        self.files.retain(|stem, _| keep.contains(stem.as_str()));
        self.logged.retain(|cause| {
            cause == DIR_CAUSE
                || cause == TOO_MANY
                || keep
                    .iter()
                    .any(|stem| cause.starts_with(&format!("{stem}: ")))
        });
        let mut listed = Vec::new();
        for (name, _) in &ranked {
            if self.refresh_one(name, now, log) {
                self.fd = None;
                self.dir_checked = Some(now);
                self.log_cause(DIR_CAUSE, log);
                self.stems.clear();
                return;
            }
            listed.push(name.clone());
        }
        listed.sort_by(|a, b| {
            let ma = self
                .files
                .get(a)
                .and_then(|entry| entry.job.as_ref())
                .map(|job| job.mtime);
            let mb = self
                .files
                .get(b)
                .and_then(|entry| entry.job.as_ref())
                .map(|job| job.mtime);
            mb.cmp(&ma).then_with(|| a.cmp(b))
        });
        self.stems = listed;
    }

    /// `true` when the open directory handle still is the path.
    fn dir_still_here(&self) -> bool {
        let Some(fd) = self.fd.as_ref() else {
            return false;
        };
        let Some(path) = self.dir.as_deref() else {
            return false;
        };
        let Ok(fd_stat) = rustix::fs::fstat(fd) else {
            return false;
        };
        let Ok(path_stat) = rustix::fs::lstat(path) else {
            return false;
        };
        FileType::from_raw_mode(path_stat.st_mode) == FileType::Directory
            && fd_stat.st_dev == path_stat.st_dev
            && fd_stat.st_ino == path_stat.st_ino
    }

    /// Stat one `<stem>.json`. `Ok(None)` is missing. `Err` is a dead directory.
    fn stat_stem(&mut self, stem: &str) -> Result<Option<rustix::fs::Stat>, ()> {
        let name = format!("{stem}.json");
        self.ops = self.ops.saturating_add(1);
        let Some(dir) = self.fd.as_ref() else {
            return Err(());
        };
        match rustix::fs::statat(dir, name.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => Ok(Some(stat)),
            Err(Errno::NOENT) => Ok(None),
            Err(err) if dir_gone(err) => Err(()),
            Err(_) => Ok(None),
        }
    }

    /// `true` when the directory fd itself is dead and the caller should
    /// drop it.
    fn refresh_one(&mut self, stem: &str, now: Instant, log: &mut impl Sink) -> bool {
        let name = format!("{stem}.json");
        self.ops = self.ops.saturating_add(1);
        let stat = {
            let Some(dir) = self.fd.as_ref() else {
                return true;
            };
            match rustix::fs::statat(dir, name.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
                Ok(stat) => stat,
                Err(Errno::NOENT) => {
                    self.store_missing(stem, now);
                    return false;
                }
                Err(Errno::LOOP) => {
                    self.fail(stem, NOT_FILE, None, now, log);
                    return false;
                }
                Err(err) if dir_gone(err) => return true,
                Err(_) => {
                    self.fail(stem, UNREADABLE, None, now, log);
                    return false;
                }
            }
        };
        let stamp = stamp_of(&stat);
        if self.files.get(stem).and_then(|entry| entry.stamp) == Some(stamp) {
            if let Some(entry) = self.files.get_mut(stem) {
                entry.checked = now;
            }
            return false;
        }
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            self.fail(stem, NOT_FILE, Some(stamp), now, log);
            return false;
        }
        if let Err(label) = byte_len(stat.st_size) {
            self.fail(stem, label, Some(stamp), now, log);
            return false;
        }
        self.ops = self.ops.saturating_add(1);
        let file = {
            let Some(dir) = self.fd.as_ref() else {
                return true;
            };
            match rustix::fs::openat(
                dir,
                name.as_str(),
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
                Mode::empty(),
            ) {
                Ok(file) => file,
                Err(Errno::NOENT) => {
                    self.store_missing(stem, now);
                    return false;
                }
                Err(Errno::LOOP) => {
                    self.fail(stem, NOT_FILE, Some(stamp), now, log);
                    return false;
                }
                Err(err) if dir_gone(err) => return true,
                Err(_) => {
                    self.fail(stem, UNREADABLE, Some(stamp), now, log);
                    return false;
                }
            }
        };
        self.ops = self.ops.saturating_add(1);
        let checked = match rustix::fs::fstat(&file) {
            Ok(stat) => stat,
            Err(_) => {
                self.fail(stem, UNREADABLE, Some(stamp), now, log);
                return false;
            }
        };
        if FileType::from_raw_mode(checked.st_mode) != FileType::RegularFile {
            self.fail(stem, NOT_FILE, Some(stamp), now, log);
            return false;
        }
        let len = match byte_len(checked.st_size) {
            Ok(len) => len,
            Err(label) => {
                self.fail(stem, label, Some(stamp_of(&checked)), now, log);
                return false;
            }
        };
        self.reads = self.reads.saturating_add(1);
        let bytes = match read_capped(&file, len) {
            Ok(bytes) => bytes,
            Err(label) => {
                self.fail(stem, label, Some(stamp_of(&checked)), now, log);
                return false;
            }
        };
        if too_deep(&bytes) {
            self.fail(stem, TOO_DEEP, Some(stamp_of(&checked)), now, log);
            return false;
        }
        match parse_job(stem, &bytes, mtime_of(&checked)) {
            Ok(job) => self.store_ok(stem, stamp_of(&checked), job, now),
            Err(label) => self.fail(stem, label, Some(stamp_of(&checked)), now, log),
        }
        false
    }

    fn list_stems(&mut self) -> Result<Vec<String>, ()> {
        let Some(dir) = self.fd.as_ref() else {
            return Err(());
        };
        self.ops = self.ops.saturating_add(1);
        let mut listing = match rustix::fs::Dir::read_from(dir) {
            Ok(listing) => listing,
            Err(_) => return Err(()),
        };
        let mut names = Vec::new();
        loop {
            match listing.read() {
                None => break,
                Some(Err(_)) => return Err(()),
                Some(Ok(entry)) => {
                    let Ok(name) = entry.file_name().to_str() else {
                        continue;
                    };
                    if name == "." || name == ".." {
                        continue;
                    }
                    let Some(stem) = name.strip_suffix(".json") else {
                        continue;
                    };
                    if !stem_ok(stem) {
                        continue;
                    }
                    if names.len() >= MAX_JOB_NAMES {
                        continue;
                    }
                    names.push(stem.to_owned());
                }
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn store_ok(&mut self, stem: &str, stamp: Stamp, job: Job, now: Instant) {
        self.clear_cause(stem);
        self.files.insert(
            stem.to_owned(),
            Entry {
                stamp: Some(stamp),
                job: Some(job),
                checked: now,
                cause: None,
            },
        );
    }

    fn store_missing(&mut self, stem: &str, now: Instant) {
        self.clear_cause(stem);
        self.files.insert(
            stem.to_owned(),
            Entry {
                stamp: None,
                job: None,
                checked: now,
                cause: None,
            },
        );
    }

    fn fail(
        &mut self,
        stem: &str,
        label: &'static str,
        stamp: Option<Stamp>,
        now: Instant,
        log: &mut impl Sink,
    ) {
        if let Some(prev) = self.files.get(stem).and_then(|entry| entry.cause)
            && prev != label
        {
            self.logged.remove(&format!("{stem}: {prev}"));
        }
        self.log_cause(&format!("{stem}: {label}"), log);
        self.files.insert(
            stem.to_owned(),
            Entry {
                stamp,
                job: None,
                checked: now,
                cause: Some(label),
            },
        );
    }

    fn clear_cause(&mut self, stem: &str) {
        if let Some(label) = self.files.get(stem).and_then(|entry| entry.cause) {
            self.logged.remove(&format!("{stem}: {label}"));
        }
    }

    fn log_cause(&mut self, cause: &str, log: &mut impl Sink) {
        if self.logged.insert(cause.to_owned()) {
            log::emit(log, Priority::Warning, &format!("job status: {cause}"));
        }
    }
}

/// A file stem that can be one path component under the jobs directory.
fn stem_ok(stem: &str) -> bool {
    !stem.is_empty()
        && stem.len() <= MAX_STEM
        && stem != "."
        && stem != ".."
        && stem
            .bytes()
            .all(|b| b != b'/' && b != b'\\' && b != 0 && (0x20..0x7f).contains(&b))
}

fn open_dir(dir: &Path) -> Result<OwnedFd, Errno> {
    rustix::fs::open(
        dir,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
}

fn dir_gone(err: Errno) -> bool {
    matches!(err, Errno::NOTDIR | Errno::BADF)
}

fn stamp_of(stat: &rustix::fs::Stat) -> Stamp {
    Stamp {
        mtime: stat.st_mtime,
        nsec: stat.st_mtime_nsec,
        len: stat.st_size,
        ino: stat.st_ino,
    }
}

fn mtime_of(stat: &rustix::fs::Stat) -> SystemTime {
    let secs = u64::try_from(stat.st_mtime.max(0)).unwrap_or(0);
    let nanos = u32::try_from(stat.st_mtime_nsec).unwrap_or(0);
    UNIX_EPOCH + Duration::new(secs, nanos)
}

/// `Ok` is a size that may be read. `Err` is `too big` or `unreadable`.
fn byte_len(len: i64) -> Result<u64, &'static str> {
    match u64::try_from(len) {
        Ok(n) if n > MAX_JOB_BYTES => Err(TOO_BIG),
        Ok(n) => Ok(n),
        Err(_) => Err(UNREADABLE),
    }
}

/// Read at most [`MAX_JOB_BYTES`]. A longer body is [`TOO_BIG`] and the
/// extra bytes are not kept. `hint` is only the buffer size.
fn read_capped(fd: &OwnedFd, hint: u64) -> Result<Vec<u8>, &'static str> {
    let cap = usize::try_from(MAX_JOB_BYTES).unwrap_or(usize::MAX);
    let mut out = Vec::new();
    let hinted = usize::try_from(hint).unwrap_or(cap);
    out.reserve(hinted.min(cap));
    let mut buf = [0u8; 8192];
    loop {
        let n = match rustix::io::read(fd, &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(Errno::INTR) => continue,
            Err(_) => return Err(UNREADABLE),
        };
        if out.len().saturating_add(n) > cap {
            return Err(TOO_BIG);
        }
        out.extend_from_slice(&buf[..n]);
    }
    Ok(out)
}

/// `true` when `{` / `[` nesting, outside strings, passes [`MAX_JOB_DEPTH`].
fn too_deep(bytes: &[u8]) -> bool {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else {
                match byte {
                    b'\\' => escaped = true,
                    b'"' => in_string = false,
                    _ => {}
                }
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > MAX_JOB_DEPTH {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

fn parse_job(stem: &str, bytes: &[u8], mtime: SystemTime) -> Result<Job, &'static str> {
    let cleaned = rewrite_nonfinite(bytes);
    let value: serde_json::Value = serde_json::from_slice(&cleaned).map_err(|_| NOT_JSON)?;
    let obj = value.as_object().ok_or(NOT_JSON)?;
    match obj.get("schema").and_then(serde_json::Value::as_u64) {
        Some(1) => {}
        _ => return Err(BAD_SCHEMA),
    }
    let name = obj
        .get("name")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_NAME))
        .ok_or(BAD_STATUS)?;
    let state = obj
        .get("state")
        .and_then(serde_json::Value::as_str)
        .and_then(JobState::parse)
        .ok_or(BAD_STATUS)?;
    let kind = obj
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| {
            let cleaned = clean_text(text)?;
            is_token(&cleaned).then_some(cleaned)
        });
    let title = obj
        .get("title")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_TITLE));
    let phase = obj.get("phase").and_then(parse_phase);
    let step = obj.get("step").and_then(as_int);
    let total = obj.get("total").and_then(as_int);
    let elapsed_s = obj.get("elapsed_s").and_then(as_duration);
    let eta_s = obj.get("eta_s").and_then(as_duration);
    let rate = obj.get("rate").and_then(parse_rate);
    let rates = obj
        .get("rates")
        .and_then(serde_json::Value::as_array)
        .map(|items| take_numbers(items, MAX_RATES))
        .unwrap_or_default();
    let values = obj
        .get("values")
        .and_then(serde_json::Value::as_array)
        .map(|items| take_numbers(items, MAX_VALUES))
        .unwrap_or_default();
    let series = obj
        .get("series")
        .and_then(serde_json::Value::as_array)
        .map(|items| take_series(items))
        .unwrap_or_default();
    let phases = obj
        .get("phases")
        .and_then(serde_json::Value::as_array)
        .map(|items| take_phases(items))
        .unwrap_or_default();
    Ok(Job {
        file: stem.to_owned(),
        name,
        kind,
        title,
        state,
        phase,
        step,
        total,
        elapsed_s,
        eta_s,
        rate,
        rates,
        values,
        series,
        phases,
        mtime,
    })
}

fn parse_phase(value: &serde_json::Value) -> Option<JobPhase> {
    let obj = value.as_object()?;
    let label = obj
        .get("label")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_LABEL))?;
    Some(JobPhase {
        label,
        index: obj.get("index").and_then(as_int),
        of: obj.get("of").and_then(as_int),
    })
}

fn parse_rate(value: &serde_json::Value) -> Option<JobRate> {
    let obj = value.as_object()?;
    let number = as_finite(obj.get("value")?)?;
    let unit = obj
        .get("unit")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_UNIT));
    Some(JobRate {
        value: number,
        unit,
    })
}

fn take_numbers(items: &[serde_json::Value], cap: usize) -> Vec<JobNumber> {
    items.iter().filter_map(parse_number).take(cap).collect()
}

fn parse_number(value: &serde_json::Value) -> Option<JobNumber> {
    let obj = value.as_object()?;
    let label = obj
        .get("label")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_LABEL))?;
    let number = as_finite(obj.get("value")?)?;
    Some(JobNumber {
        label,
        value: number,
    })
}

fn take_series(items: &[serde_json::Value]) -> Vec<JobSeries> {
    items
        .iter()
        .filter_map(parse_series)
        .take(MAX_SERIES)
        .collect()
}

fn parse_series(value: &serde_json::Value) -> Option<JobSeries> {
    let obj = value.as_object()?;
    let label = obj
        .get("label")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_LABEL))?;
    let points = obj.get("points").and_then(serde_json::Value::as_array)?;
    let mut kept: Vec<f64> = points.iter().filter_map(as_finite).collect();
    if kept.len() > MAX_SERIES_POINTS {
        let skip = kept.len() - MAX_SERIES_POINTS;
        kept.drain(..skip);
    }
    Some(JobSeries {
        label,
        points: kept,
    })
}

fn take_phases(items: &[serde_json::Value]) -> Vec<JobPhaseRow> {
    items
        .iter()
        .filter_map(parse_phase_row)
        .take(MAX_PHASES)
        .collect()
}

fn parse_phase_row(value: &serde_json::Value) -> Option<JobPhaseRow> {
    let obj = value.as_object()?;
    let label = obj
        .get("label")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| clean_cap(text, MAX_LABEL))?;
    Some(JobPhaseRow {
        label,
        mark: obj
            .get("state")
            .and_then(serde_json::Value::as_str)
            .and_then(PhaseMark::parse),
        elapsed_s: obj.get("elapsed_s").and_then(as_duration),
        result: obj
            .get("result")
            .and_then(serde_json::Value::as_str)
            .and_then(|text| clean_cap(text, MAX_RESULT)),
    })
}

fn as_finite(value: &serde_json::Value) -> Option<f64> {
    let n = value.as_f64()?;
    n.is_finite().then_some(n)
}

/// An integer JSON number. A float such as `1.4` or `921.5` is absent.
fn as_int(value: &serde_json::Value) -> Option<u64> {
    value.as_u64()
}

fn as_duration(value: &serde_json::Value) -> Option<u64> {
    Some(as_int(value)?.min(MAX_DURATION_S))
}

/// Replace bare `NaN` / `Infinity` / `-Infinity` outside strings with `null`,
/// so a diverged loss does not reject the file. Tokens inside strings stay.
fn rewrite_nonfinite(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i += 1;
            continue;
        }
        if let Some(n) = nonfinite_token(&bytes[i..]) {
            out.extend_from_slice(b"null");
            changed = true;
            i += n;
            continue;
        }
        out.push(b);
        i += 1;
    }
    if changed {
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(bytes)
    }
}

fn nonfinite_token(bytes: &[u8]) -> Option<usize> {
    const TOKENS: &[&[u8]] = &[b"-Infinity", b"Infinity", b"NaN"];
    for token in TOKENS {
        if bytes.starts_with(token) && token_boundary(bytes, token.len()) {
            return Some(token.len());
        }
    }
    None
}

fn token_boundary(bytes: &[u8], end: usize) -> bool {
    match bytes.get(end) {
        None => true,
        Some(b) => !b.is_ascii_alphanumeric() && *b != b'_',
    }
}

/// A `text` value. Whitespace runs, including newlines and tabs, become
/// one space; leading and trailing whitespace go. A control or a
/// zero-width character is dropped, so an escape sequence or a newline
/// cannot reach the console or the grid file. Everything else is kept,
/// and the console font's mapping plus the grid file's `rich` scalar
/// happen when the row is drawn. Empty is nothing.
fn clean_text(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut spaced = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                spaced = true;
            }
            continue;
        }
        if text_skipped(ch) {
            continue;
        }
        if spaced {
            out.push(' ');
            spaced = false;
        }
        out.push(ch);
    }
    (!out.is_empty()).then_some(out)
}

fn text_skipped(ch: char) -> bool {
    matches!(
        ch,
        '\u{00ad}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'
    ) || ch.is_control()
}

fn clean_cap(raw: &str, cap: usize) -> Option<String> {
    let cleaned = clean_text(raw)?;
    Some(cut_text(&cleaned, cap))
}

/// `true` when `mtime` is older than `stale_s` seconds before `now`, or more
/// than [`FUTURE_STALE_S`] seconds in the future.
pub fn is_stale(mtime: SystemTime, now: SystemTime, stale_s: u64) -> bool {
    match now.duration_since(mtime) {
        Ok(age) => age > Duration::from_secs(stale_s),
        Err(skew) => skew.duration() > Duration::from_secs(FUTURE_STALE_S),
    }
}

/// Which job, if any, the dashboard should show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JobPick<'a> {
    /// A live (`running` and fresh) job. `serving` keeps the inference view.
    Live {
        job: &'a Job,
        more: usize,
        serving: bool,
    },
    /// A fresh `done` or `failed` job: inference stays, with a one-line note.
    Finished {
        job: &'a Job,
    },
    None,
}

/// Decide the job view from the files the poller last read, newest first.
pub fn pick_job<'a>(jobs: &'a [Job], serving: bool, stale_s: u64, now: SystemTime) -> JobPick<'a> {
    let fresh: Vec<&Job> = jobs
        .iter()
        .filter(|job| !is_stale(job.mtime, now, stale_s))
        .collect();
    let live: Vec<&Job> = fresh
        .iter()
        .copied()
        .filter(|job| job.state == JobState::Running)
        .collect();
    if let Some(job) = live.first() {
        JobPick::Live {
            job,
            more: live.len().saturating_sub(1),
            serving,
        }
    } else if let Some(job) = fresh
        .iter()
        .copied()
        .find(|job| matches!(job.state, JobState::Done | JobState::Failed))
    {
        JobPick::Finished { job }
    } else {
        JobPick::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use llama_core::log::Sink;

    struct Mem {
        lines: Vec<String>,
    }

    impl Sink for Mem {
        fn write_line(&mut self, line: &str) {
            self.lines.push(line.to_owned());
        }
    }

    impl Mem {
        fn new() -> Self {
            Self { lines: Vec::new() }
        }

        fn count(&self, needle: &str) -> usize {
            self.lines
                .iter()
                .filter(|line| line.contains(needle))
                .count()
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::var_os("CARGO_TARGET_TMPDIR").map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp"),
            PathBuf::from,
        );
        let dir = root.join(format!("job-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn nested(depth: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..depth {
            bytes.extend_from_slice(b"\x7b\"a\":");
        }
        bytes.push(b'1');
        bytes.extend(std::iter::repeat_n(b'\x7d', depth));
        bytes
    }

    fn minimal() -> &'static [u8] {
        br#"{"schema":1,"name":"example-trainer","state":"running"}"#
    }

    fn write_job(dir: &Path, stem: &str, body: &[u8]) {
        std::fs::write(dir.join(format!("{stem}.json")), body).unwrap();
    }

    #[test]
    fn depth_ignores_braces_inside_strings() {
        assert!(!too_deep(br#"{"a":"{{{{}}}}"}"#));
        assert!(!too_deep(&nested(MAX_JOB_DEPTH)));
        assert!(too_deep(&nested(MAX_JOB_DEPTH + 1)));
    }

    #[test]
    fn stems_that_are_not_file_names() {
        assert!(stem_ok("example-trainer"));
        assert!(stem_ok("a..b"));
        assert!(stem_ok(&"a".repeat(MAX_STEM)));
        for bad in [
            "", ".", "..", "../x", "a/b", "a\\b", "a\0b", "a\nb", "a\u{7f}b",
        ] {
            assert!(!stem_ok(bad), "{bad:?}");
        }
        assert!(!stem_ok(&"a".repeat(MAX_STEM + 1)));
    }

    #[test]
    fn unset_dir_opens_nothing() {
        let mut jobs = Jobs::new(None);
        let mut log = Mem::new();
        let t0 = Instant::now();
        jobs.refresh(t0, &mut log);
        jobs.refresh(t0 + CHECK_EVERY, &mut log);
        assert_eq!(jobs.ops, 0);
        assert_eq!(jobs.reads, 0);
        assert!(jobs.jobs().is_empty());
        assert!(log.lines.is_empty());
    }

    #[test]
    fn a_valid_file_is_read() {
        let dir = scratch("ok");
        write_job(&dir, "example-trainer", minimal());
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert_eq!(jobs.reads, 1);
        let job = jobs.jobs();
        assert_eq!(job.len(), 1);
        assert_eq!(job[0].name, "example-trainer");
        assert_eq!(job[0].state, JobState::Running);
        assert!(log.lines.is_empty(), "{:?}", log.lines);
    }

    #[test]
    fn bad_files_log_once_per_cause() {
        let dir = scratch("bad");
        write_job(&dir, "big", &vec![b'x'; (MAX_JOB_BYTES as usize) + 1]);
        write_job(&dir, "bad", b"\x7b");
        write_job(&dir, "deep", &nested(MAX_JOB_DEPTH + 1));
        write_job(
            &dir,
            "schema",
            br#"{"schema":2,"name":"n","state":"running"}"#,
        );
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        let t0 = Instant::now();
        jobs.refresh(t0, &mut log);
        assert_eq!(jobs.reads, 3, "too big is not read");
        assert!(jobs.jobs().is_empty());
        assert_eq!(log.count("big: too big"), 1, "{:?}", log.lines);
        assert_eq!(log.count("bad: not json"), 1, "{:?}", log.lines);
        assert_eq!(log.count("deep: too deep"), 1, "{:?}", log.lines);
        assert_eq!(log.count("schema: bad schema"), 1, "{:?}", log.lines);
        let reads = jobs.reads;
        jobs.refresh(t0 + Duration::from_millis(200), &mut log);
        assert_eq!(jobs.reads, reads, "inside 1 s nothing is listed");
        assert_eq!(log.lines.len(), 4);
        jobs.refresh(t0 + CHECK_EVERY, &mut log);
        assert_eq!(jobs.reads, reads, "unchanged mtime is not read again");
        assert_eq!(log.lines.len(), 4);
    }

    #[test]
    fn a_symlink_file_is_not_read() {
        let root = scratch("link");
        let real = root.join("real");
        let outside = root.join("outside");
        let via = root.join("via");
        std::fs::create_dir(&real).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret.json"), minimal()).unwrap();
        std::os::unix::fs::symlink(outside.join("secret.json"), real.join("model.json")).unwrap();
        std::os::unix::fs::symlink(&real, &via).unwrap();

        let mut jobs = Jobs::new(Some(via));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert_eq!(jobs.reads, 0, "a symlinked directory is not opened");
        assert!(jobs.jobs().is_empty());
        assert_eq!(log.count(DIR_CAUSE), 1, "{:?}", log.lines);

        let mut jobs = Jobs::new(Some(real));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert!(jobs.jobs().is_empty(), "a symlinked file is not read");
        assert_eq!(jobs.reads, 0);
        assert_eq!(log.count("model: not a file"), 1, "{:?}", log.lines);
        assert!(
            !log.lines
                .iter()
                .any(|line| line.contains("secret") || line.contains("example-trainer")),
            "{:?}",
            log.lines
        );
    }

    #[test]
    fn more_than_eight_files_reads_the_newest_eight() {
        let dir = scratch("many");
        let t0 = SystemTime::now() - Duration::from_secs(100);
        for i in 0..9 {
            write_job(
                &dir,
                &format!("job{i}"),
                &format!(r#"{{"schema":1,"name":"job{i}","state":"running"}}"#).into_bytes(),
            );
            let path = dir.join(format!("job{i}.json"));
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(t0 + Duration::from_secs(i as u64))
                .unwrap();
        }
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert_eq!(jobs.jobs().len(), 8);
        assert_eq!(jobs.reads, 8);
        let names: Vec<&str> = jobs.jobs().iter().map(|j| j.name.as_str()).collect();
        assert_eq!(names[0], "job8", "newest mtime first: {names:?}");
        assert!(
            !names.contains(&"job0"),
            "oldest leftover is not read: {names:?}"
        );
        assert_eq!(log.count(TOO_MANY), 1, "{:?}", log.lines);
        let n = log.lines.len();
        jobs.refresh(Instant::now() + CHECK_EVERY, &mut log);
        assert_eq!(log.lines.len(), n, "the over-eight warning logs once");
    }

    #[test]
    fn more_than_sixty_four_files_still_reads_the_newest() {
        let dir = scratch("many-names");
        let t0 = SystemTime::now() - Duration::from_secs(10_000);
        for i in 0..70 {
            write_job(
                &dir,
                &format!("old-{i:02}"),
                &format!(r#"{{"schema":1,"name":"old-{i:02}","state":"running"}}"#).into_bytes(),
            );
            let path = dir.join(format!("old-{i:02}.json"));
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(t0 + Duration::from_secs(i as u64))
                .unwrap();
        }
        write_job(
            &dir,
            "newest",
            br#"{"schema":1,"name":"newest","state":"running"}"#,
        );
        let newest = dir.join("newest.json");
        std::fs::File::options()
            .write(true)
            .open(&newest)
            .unwrap()
            .set_modified(t0 + Duration::from_secs(10_000))
            .unwrap();
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert_eq!(jobs.jobs().len(), 8);
        let names: Vec<&str> = jobs.jobs().iter().map(|j| j.name.as_str()).collect();
        assert_eq!(names[0], "newest", "newest mtime first: {names:?}");
        assert!(
            !names.contains(&"old-00"),
            "an early directory-order name is not preferred: {names:?}"
        );
    }

    #[test]
    fn non_finite_and_out_of_range_drop_the_member() {
        let body = br#"{
            "schema": 1,
            "name": "n",
            "state": "running",
            "step": -1,
            "total": 10.5,
            "elapsed_s": 1e20,
            "eta_s": 4,
            "rate": {"value": null, "unit": "s/step"},
            "values": [
                {"label": "loss", "value": 0.25},
                {"label": "bad", "value": "nan"},
                {"label": "blew", "value": NaN},
                {"label": "up", "value": Infinity},
                {"label": "down", "value": -Infinity}
            ]
        }"#;
        let job = parse_job("n", body, UNIX_EPOCH).expect("parses");
        assert!(job.step.is_none(), "negative step is absent");
        assert!(job.total.is_none(), "fractional total is absent");
        assert!(job.elapsed_s.is_none(), "float elapsed is absent");
        assert_eq!(job.eta_s, Some(4));
        assert!(job.rate.is_none());
        assert_eq!(job.values.len(), 1);
        assert_eq!(job.values[0].label, "loss");
        assert_eq!(job.values[0].value, 0.25);
    }

    #[test]
    fn schema_must_be_the_integer_one() {
        for body in [
            br#"{"schema":1.4,"name":"n","state":"running"}"#.as_slice(),
            br#"{"schema":1.0,"name":"n","state":"running"}"#.as_slice(),
            br#"{"schema":2,"name":"n","state":"running"}"#.as_slice(),
        ] {
            assert_eq!(parse_job("n", body, UNIX_EPOCH), Err(BAD_SCHEMA));
        }
        assert!(parse_job("n", minimal(), UNIX_EPOCH).is_ok());
    }

    #[test]
    fn a_huge_duration_clamps_to_ten_years() {
        let body = format!(
            r#"{{"schema":1,"name":"n","state":"running","elapsed_s":{max},"eta_s":{max},"phases":[{{"label":"p","elapsed_s":{max}}}]}}"#,
            max = u64::MAX
        );
        let job = parse_job("n", body.as_bytes(), UNIX_EPOCH).expect("parses");
        assert_eq!(job.elapsed_s, Some(MAX_DURATION_S));
        assert_eq!(job.eta_s, Some(MAX_DURATION_S));
        assert_eq!(job.phases[0].elapsed_s, Some(MAX_DURATION_S));
    }

    #[test]
    fn nan_inside_a_string_is_not_rewritten() {
        let body = br#"{"schema":1,"name":"NaN","state":"running","title":"Infinity"}"#;
        let job = parse_job("n", body, UNIX_EPOCH).expect("parses");
        assert_eq!(job.name, "NaN");
        assert_eq!(job.title.as_deref(), Some("Infinity"));
    }

    #[test]
    fn caps_keep_the_newest_series_points() {
        let points: Vec<String> = (0..600).map(|i| i.to_string()).collect();
        let body = format!(
            r#"{{"schema":1,"name":"n","state":"running","series":[{{"label":"loss","points":[{}]}}]}}"#,
            points.join(",")
        );
        let job = parse_job("n", body.as_bytes(), UNIX_EPOCH).expect("parses");
        assert_eq!(job.series[0].points.len(), MAX_SERIES_POINTS);
        assert_eq!(job.series[0].points[0], 88.0);
        assert_eq!(job.series[0].points[MAX_SERIES_POINTS - 1], 599.0);
    }

    #[test]
    fn sanitising_drops_controls_and_caps_text() {
        let body = format!(
            r#"{{"schema":1,"name":"ok\u001b[2J{}","state":"running","title":"café title extra words that overflow","kind":"training"}}"#,
            "x".repeat(40)
        );
        let job = parse_job("n", body.as_bytes(), UNIX_EPOCH).expect("parses");
        assert!(!job.name.contains('\u{1b}'));
        assert!(job.name.chars().count() <= MAX_NAME);
        assert_eq!(job.kind.as_deref(), Some("training"));
        assert!(
            job.title
                .as_ref()
                .is_some_and(|t| t.chars().count() <= MAX_TITLE)
        );
    }

    #[test]
    fn stale_is_age_against_mtime() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let fresh = UNIX_EPOCH + Duration::from_secs(900);
        let old = UNIX_EPOCH + Duration::from_secs(100);
        assert!(!is_stale(fresh, now, 180));
        assert!(is_stale(old, now, 180));
        assert!(!is_stale(now + Duration::from_secs(10), now, 180));
        assert!(!is_stale(now + Duration::from_secs(60), now, 180));
        assert!(is_stale(now + Duration::from_secs(61), now, 180));
    }

    #[test]
    fn a_stale_file_is_still_parsed() {
        let dir = scratch("stale");
        write_job(&dir, "old", minimal());
        let path = dir.join("old.json");
        let age = SystemTime::now() - Duration::from_secs(1_000);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(age)
            .unwrap();
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        jobs.refresh(Instant::now(), &mut log);
        assert_eq!(jobs.jobs().len(), 1);
        assert!(is_stale(jobs.jobs()[0].mtime, SystemTime::now(), 180));
    }

    #[test]
    fn unknown_members_are_ignored_and_optional_fields_may_be_absent() {
        let body = br#"{"schema":1,"name":"n","state":"failed","extra":{"x":1}}"#;
        let job = parse_job("n", body, UNIX_EPOCH).expect("parses");
        assert_eq!(job.state, JobState::Failed);
        assert!(job.kind.is_none());
        assert!(job.title.is_none());
        assert!(job.rate.is_none());
        assert!(job.series.is_empty());
        assert!(job.phases.is_empty());
    }

    #[test]
    fn rates_values_phases_are_capped() {
        let rates: Vec<String> = (0..6)
            .map(|i| format!(r#"{{"label":"r{i}","value":{i}}}"#))
            .collect();
        let values: Vec<String> = (0..10)
            .map(|i| format!(r#"{{"label":"v{i}","value":{i}}}"#))
            .collect();
        let phases: Vec<String> = (0..15)
            .map(|i| format!(r#"{{"label":"p{i}","state":"queued"}}"#))
            .collect();
        let series: Vec<String> = (0..6)
            .map(|i| format!(r#"{{"label":"s{i}","points":[1]}}"#))
            .collect();
        let body = format!(
            r#"{{"schema":1,"name":"n","state":"running","rates":[{}],"values":[{}],"phases":[{}],"series":[{}]}}"#,
            rates.join(","),
            values.join(","),
            phases.join(","),
            series.join(",")
        );
        let job = parse_job("n", body.as_bytes(), UNIX_EPOCH).expect("parses");
        assert_eq!(job.rates.len(), MAX_RATES);
        assert_eq!(job.values.len(), MAX_VALUES);
        assert_eq!(job.phases.len(), MAX_PHASES);
        assert_eq!(job.series.len(), MAX_SERIES);
    }

    #[test]
    fn unchanged_file_is_not_re_read() {
        let dir = scratch("mtime");
        let path = dir.join("m.json");
        write_job(&dir, "m", minimal());
        let mut jobs = Jobs::new(Some(dir));
        let mut log = Mem::new();
        let t0 = Instant::now();
        jobs.refresh(t0, &mut log);
        assert_eq!(jobs.reads, 1);
        jobs.refresh(t0 + CHECK_EVERY, &mut log);
        assert_eq!(jobs.reads, 1, "the same mtime and size are not read");
        std::fs::write(&path, br#"{"schema":1,"name":"m","state":"done"}"#).unwrap();
        jobs.refresh(t0 + CHECK_EVERY + CHECK_EVERY, &mut log);
        assert_eq!(jobs.reads, 2, "a size change is read");
        assert_eq!(jobs.jobs()[0].state, JobState::Done);
    }

    #[test]
    fn a_replaced_directory_is_reopened() {
        let root = scratch("replace");
        let dir = root.join("jobs");
        std::fs::create_dir(&dir).unwrap();
        write_job(&dir, "old", minimal());
        let mut jobs = Jobs::new(Some(dir.clone()));
        let mut log = Mem::new();
        let t0 = Instant::now();
        jobs.refresh(t0, &mut log);
        assert_eq!(jobs.jobs()[0].name, "example-trainer");
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::create_dir(&dir).unwrap();
        write_job(
            &dir,
            "new",
            br#"{"schema":1,"name":"fresh-job","state":"running"}"#,
        );
        jobs.refresh(t0 + CHECK_EVERY, &mut log);
        let names: Vec<&str> = jobs.jobs().iter().map(|j| j.name.as_str()).collect();
        assert_eq!(names, ["fresh-job"], "{names:?}");
    }

    #[test]
    fn a_directory_that_appears_later_is_picked_up() {
        let root = scratch("appear");
        let dir = root.join("jobs");
        let mut jobs = Jobs::new(Some(dir.clone()));
        let mut log = Mem::new();
        let t0 = Instant::now();
        jobs.refresh(t0, &mut log);
        assert!(jobs.jobs().is_empty());
        assert_eq!(log.count(DIR_CAUSE), 1, "{:?}", log.lines);
        std::fs::create_dir(&dir).unwrap();
        write_job(&dir, "late", minimal());
        jobs.refresh(t0 + CHECK_EVERY, &mut log);
        assert_eq!(jobs.jobs().len(), 1);
        assert_eq!(jobs.jobs()[0].name, "example-trainer");
    }

    fn job_at(name: &str, state: JobState, mtime: SystemTime) -> Job {
        Job {
            file: name.to_owned(),
            name: name.to_owned(),
            kind: Some("training".to_owned()),
            title: None,
            state,
            phase: None,
            step: None,
            total: None,
            elapsed_s: None,
            eta_s: None,
            rate: None,
            rates: Vec::new(),
            values: Vec::new(),
            series: Vec::new(),
            phases: Vec::new(),
            mtime,
        }
    }

    #[test]
    fn pick_job_table() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let fresh = now - Duration::from_secs(10);
        let old = now - Duration::from_secs(500);
        let future = now + Duration::from_secs(90);
        let running = job_at("run", JobState::Running, fresh);
        let idle_run = job_at("run", JobState::Running, fresh);
        let done = job_at("done", JobState::Done, fresh);
        let failed = job_at("fail", JobState::Failed, fresh);
        let stale = job_at("old", JobState::Running, old);
        let skew = job_at("skew", JobState::Running, future);
        let other = job_at("other", JobState::Running, fresh - Duration::from_secs(1));

        match pick_job(std::slice::from_ref(&stale), false, 180, now) {
            JobPick::None => {}
            other => panic!("stale: {other:?}"),
        }
        match pick_job(std::slice::from_ref(&skew), false, 180, now) {
            JobPick::None => {}
            other => panic!("future mtime: {other:?}"),
        }
        match pick_job(std::slice::from_ref(&running), true, 180, now) {
            JobPick::Live {
                job,
                more: 0,
                serving: true,
            } => assert_eq!(job.name, "run"),
            other => panic!("running while serving: {other:?}"),
        }
        match pick_job(std::slice::from_ref(&idle_run), false, 180, now) {
            JobPick::Live {
                job,
                more: 0,
                serving: false,
            } => assert_eq!(job.name, "run"),
            other => panic!("running while idle: {other:?}"),
        }
        match pick_job(std::slice::from_ref(&done), false, 180, now) {
            JobPick::Finished { job } => assert_eq!(job.name, "done"),
            other => panic!("done: {other:?}"),
        }
        match pick_job(std::slice::from_ref(&failed), false, 180, now) {
            JobPick::Finished { job } => assert_eq!(job.name, "fail"),
            other => panic!("failed: {other:?}"),
        }
        match pick_job(&[running, other], false, 180, now) {
            JobPick::Live {
                job,
                more: 1,
                serving: false,
            } => assert_eq!(job.name, "run"),
            other => panic!("two running: {other:?}"),
        }
        match pick_job(&[], false, 180, now) {
            JobPick::None => {}
            other => panic!("none: {other:?}"),
        }
    }
}
