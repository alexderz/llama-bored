//! Per-model entry spec files for SETUP's `spec:` rules (#92).
//!
//! Read only, and only from the poller: the frame path never opens these
//! files. `specs_dir` unset means this module does not open anything.
//! The directory is opened with `O_DIRECTORY | O_NOFOLLOW` and each file
//! with `openat` `O_NOFOLLOW`, so a symlink cannot leave the directory.
//! Nothing here is written. This file is not on the safety_scan
//! `WRITE_FILES` allowlist and must not gain a write.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use llama_core::log::{self, Priority, Sink};
use rustix::fd::OwnedFd;
use rustix::fs::{AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;

/// How often a loaded model's file is stat'd when the model set is unchanged.
pub(crate) const CHECK_EVERY: Duration = Duration::from_secs(60);
/// Largest file read. Anything longer is "too big" and its body is not parsed.
pub(crate) const MAX_SPEC_BYTES: u64 = 64 * 1024;
/// Deepest JSON array or object nesting accepted. Deeper is "too deep".
pub(crate) const MAX_SPEC_DEPTH: usize = 16;
/// Longest llama-swap model id used as a file name.
pub(crate) const MAX_MODEL_ID: usize = 128;

const DIR_CAUSE: &str = "specs dir unreadable";
const UNREADABLE: &str = "unreadable";
const TOO_BIG: &str = "too big";
const NOT_JSON: &str = "not json";
const TOO_DEEP: &str = "too deep";
const NOT_FILE: &str = "not a file";

/// `st_mtime`, its nanoseconds, and `st_size`, so a later stat can see
/// whether the file changed without reading it.
#[derive(Clone, Copy, Eq, PartialEq)]
struct Stamp {
    mtime: i64,
    nsec: u64,
    len: i64,
}

struct Entry {
    stamp: Option<Stamp>,
    value: Option<serde_json::Value>,
    checked: Instant,
    /// Last logged fault for this id, without the id. `None` after a
    /// successful read or a missing file (missing is not an error).
    cause: Option<&'static str>,
}

/// The spec directory and the files read for the models loaded now.
pub(crate) struct Specs {
    dir: Option<PathBuf>,
    fd: Option<OwnedFd>,
    /// Model ids from the last refresh, in that order.
    ids: Vec<String>,
    files: HashMap<String, Entry>,
    /// Causes already logged. A cause logs again only after it cleared.
    logged: HashSet<String>,
    /// When the directory was last opened, or the open last failed.
    dir_checked: Option<Instant>,
    /// Directory opens, stats and file opens. Tests prove an unset
    /// directory and a hostile id never touch the filesystem.
    #[cfg_attr(not(test), allow(dead_code))]
    ops: u64,
    /// File bodies read. A stat that finds the file unchanged does not count.
    #[cfg_attr(not(test), allow(dead_code))]
    reads: u64,
}

impl Specs {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            fd: None,
            ids: Vec::new(),
            files: HashMap::new(),
            logged: HashSet::new(),
            dir_checked: None,
            ops: 0,
            reads: 0,
        }
    }

    /// The parsed document for `id`, when the last read succeeded.
    pub(crate) fn value(&self, id: &str) -> Option<&serde_json::Value> {
        self.files.get(id).and_then(|entry| entry.value.as_ref())
    }

    /// Re-read the files of `ids`. `now` is the poller's clock, so tests
    /// can move it without sleeping. An unchanged model set only stats,
    /// and only about every [`CHECK_EVERY`]; a stat re-reads when the
    /// mtime or the size changed. A changed model set re-reads at once.
    pub(crate) fn refresh(&mut self, ids: &[String], now: Instant, log: &mut impl Sink) {
        if self.dir.is_none() {
            return;
        }
        let set_changed = ids != self.ids.as_slice();
        let valid: Vec<&str> = ids
            .iter()
            .map(String::as_str)
            .filter(|id| id_ok(id))
            .collect();
        if valid.is_empty() {
            // No name is safe to open. Do not open the directory either.
            if set_changed {
                self.ids = ids.to_vec();
                self.files.clear();
                self.logged.retain(|cause| cause == DIR_CAUSE);
            }
            return;
        }
        let dir_due = self.fd.is_none()
            && self
                .dir_checked
                .is_none_or(|at| now.saturating_duration_since(at) >= CHECK_EVERY);
        let files_due = self.fd.is_some()
            && valid.iter().any(|id| match self.files.get(*id) {
                None => true,
                Some(entry) => now.saturating_duration_since(entry.checked) >= CHECK_EVERY,
            });
        if !set_changed && !dir_due && !files_due {
            return;
        }
        if set_changed {
            self.ids = ids.to_vec();
            let keep: HashSet<&str> = valid.iter().copied().collect();
            self.files.retain(|id, _| keep.contains(id.as_str()));
            self.logged.retain(|cause| {
                cause == DIR_CAUSE || keep.iter().any(|id| cause.starts_with(&format!("{id}: ")))
            });
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
                    return;
                }
            }
        }
        for id in valid {
            if self.refresh_one(id, set_changed, now, log) {
                // The directory fd no longer names a directory. The next
                // refresh opens it again, at the 60 s cadence unless the
                // model set changes first.
                self.fd = None;
                self.dir_checked = Some(now);
                self.log_cause(DIR_CAUSE, log);
                return;
            }
        }
    }

    /// `true` when the directory fd itself is dead and the caller should
    /// drop it. `force` re-reads even when the mtime and size match.
    fn refresh_one(&mut self, id: &str, force: bool, now: Instant, log: &mut impl Sink) -> bool {
        if !force
            && self
                .files
                .get(id)
                .is_some_and(|entry| now.saturating_duration_since(entry.checked) < CHECK_EVERY)
        {
            return false;
        }
        let name = format!("{id}.json");
        self.ops = self.ops.saturating_add(1);
        let stat = {
            let Some(dir) = self.fd.as_ref() else {
                return true;
            };
            match rustix::fs::statat(dir, name.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
                Ok(stat) => stat,
                Err(Errno::NOENT) => {
                    self.store_missing(id, now);
                    return false;
                }
                Err(Errno::LOOP) => {
                    self.fail(id, NOT_FILE, None, now, log);
                    return false;
                }
                Err(err) if dir_gone(err) => return true,
                Err(_) => {
                    self.fail(id, UNREADABLE, None, now, log);
                    return false;
                }
            }
        };
        let stamp = stamp_of(&stat);
        if !force && self.files.get(id).and_then(|entry| entry.stamp) == Some(stamp) {
            if let Some(entry) = self.files.get_mut(id) {
                entry.checked = now;
            }
            return false;
        }
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            self.fail(id, NOT_FILE, Some(stamp), now, log);
            return false;
        }
        // Reject a huge file before opening it. The fstat below repeats
        // the check in case the file was replaced.
        if let Err(label) = byte_len(stat.st_size) {
            self.fail(id, label, Some(stamp), now, log);
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
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            ) {
                Ok(file) => file,
                Err(Errno::NOENT) => {
                    self.store_missing(id, now);
                    return false;
                }
                Err(Errno::LOOP) => {
                    self.fail(id, NOT_FILE, Some(stamp), now, log);
                    return false;
                }
                Err(err) if dir_gone(err) => return true,
                Err(_) => {
                    self.fail(id, UNREADABLE, Some(stamp), now, log);
                    return false;
                }
            }
        };
        self.ops = self.ops.saturating_add(1);
        let checked = match rustix::fs::fstat(&file) {
            Ok(stat) => stat,
            Err(_) => {
                self.fail(id, UNREADABLE, Some(stamp), now, log);
                return false;
            }
        };
        // The file can be replaced between the stat and the open.
        if FileType::from_raw_mode(checked.st_mode) != FileType::RegularFile {
            self.fail(id, NOT_FILE, Some(stamp), now, log);
            return false;
        }
        let len = match byte_len(checked.st_size) {
            Ok(len) => len,
            Err(label) => {
                self.fail(id, label, Some(stamp_of(&checked)), now, log);
                return false;
            }
        };
        self.reads = self.reads.saturating_add(1);
        let bytes = match read_capped(&file, len) {
            Ok(bytes) => bytes,
            Err(label) => {
                self.fail(id, label, Some(stamp_of(&checked)), now, log);
                return false;
            }
        };
        if too_deep(&bytes) {
            self.fail(id, TOO_DEEP, Some(stamp_of(&checked)), now, log);
            return false;
        }
        match serde_json::from_slice(&bytes) {
            Ok(value) => self.store_ok(id, stamp_of(&checked), value, now),
            Err(_) => self.fail(id, NOT_JSON, Some(stamp_of(&checked)), now, log),
        }
        false
    }

    fn store_ok(&mut self, id: &str, stamp: Stamp, value: serde_json::Value, now: Instant) {
        self.clear_cause(id);
        self.files.insert(
            id.to_owned(),
            Entry {
                stamp: Some(stamp),
                value: Some(value),
                checked: now,
                cause: None,
            },
        );
    }

    fn store_missing(&mut self, id: &str, now: Instant) {
        self.clear_cause(id);
        self.files.insert(
            id.to_owned(),
            Entry {
                stamp: None,
                value: None,
                checked: now,
                cause: None,
            },
        );
    }

    fn fail(
        &mut self,
        id: &str,
        label: &'static str,
        stamp: Option<Stamp>,
        now: Instant,
        log: &mut impl Sink,
    ) {
        if let Some(prev) = self.files.get(id).and_then(|entry| entry.cause)
            && prev != label
        {
            self.logged.remove(&format!("{id}: {prev}"));
        }
        self.log_cause(&format!("{id}: {label}"), log);
        self.files.insert(
            id.to_owned(),
            Entry {
                stamp,
                value: None,
                checked: now,
                cause: Some(label),
            },
        );
    }

    fn clear_cause(&mut self, id: &str) {
        if let Some(label) = self.files.get(id).and_then(|entry| entry.cause) {
            self.logged.remove(&format!("{id}: {label}"));
        }
    }

    fn log_cause(&mut self, cause: &str, log: &mut impl Sink) {
        if self.logged.insert(cause.to_owned()) {
            log::emit(log, Priority::Warning, &format!("setup spec: {cause}"));
        }
    }
}

/// A model id that can be one path component under the spec directory.
fn id_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_MODEL_ID
        && id != "."
        && id != ".."
        && id
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
    }
}

/// `Ok` is a size that may be read. `Err` is `too big` or `unreadable`.
fn byte_len(len: i64) -> Result<u64, &'static str> {
    match u64::try_from(len) {
        Ok(n) if n > MAX_SPEC_BYTES => Err(TOO_BIG),
        Ok(n) => Ok(n),
        Err(_) => Err(UNREADABLE),
    }
}

/// Read at most [`MAX_SPEC_BYTES`]. A longer body is [`TOO_BIG`] and the
/// extra bytes are not kept. `hint` is only the buffer size.
fn read_capped(fd: &OwnedFd, hint: u64) -> Result<Vec<u8>, &'static str> {
    let cap = usize::try_from(MAX_SPEC_BYTES).unwrap_or(usize::MAX);
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

/// `true` when `{` / `[` nesting, outside strings, passes [`MAX_SPEC_DEPTH`].
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
                if depth > MAX_SPEC_DEPTH {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
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

    /// A directory under the workspace `target/tmp`, which is where Cargo
    /// points `CARGO_TARGET_TMPDIR` for integration tests. Unit tests are
    /// not given that variable, so this builds the same path from the
    /// manifest. Nothing here uses the process temp dir.
    fn scratch(name: &str) -> PathBuf {
        let root = std::env::var_os("CARGO_TARGET_TMPDIR").map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp"),
            PathBuf::from,
        );
        let dir = root.join(format!("spec-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn nested(depth: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..depth {
            bytes.extend_from_slice(br#"{"a":"#);
        }
        bytes.push(b'1');
        for _ in 0..depth {
            bytes.push(b'}');
        }
        bytes
    }

    #[test]
    fn depth_ignores_braces_inside_strings() {
        assert!(!too_deep(br#"{"a":"{{{{}}}}"}"#));
        assert!(!too_deep(&nested(MAX_SPEC_DEPTH)));
        assert!(too_deep(&nested(MAX_SPEC_DEPTH + 1)));
    }

    #[test]
    fn model_ids_that_are_not_file_names() {
        assert!(id_ok("example-35b-q6"));
        assert!(id_ok("a..b"));
        assert!(id_ok(&"a".repeat(MAX_MODEL_ID)));
        for bad in [
            "", ".", "..", "../x", "a/b", "a\\b", "a\0b", "a\nb", "a\u{7f}b",
        ] {
            assert!(!id_ok(bad), "{bad:?}");
        }
        assert!(!id_ok(&"a".repeat(MAX_MODEL_ID + 1)));
    }

    #[test]
    fn unset_dir_opens_nothing() {
        let mut specs = Specs::new(None);
        let mut log = Mem::new();
        let t0 = Instant::now();
        specs.refresh(&ids(&["example-35b-q6"]), t0, &mut log);
        specs.refresh(&ids(&["example-35b-q6"]), t0 + CHECK_EVERY, &mut log);
        assert_eq!(specs.ops, 0);
        assert_eq!(specs.reads, 0);
        assert!(specs.value("example-35b-q6").is_none());
        assert!(log.lines.is_empty());
    }

    #[test]
    fn hostile_ids_open_nothing() {
        let dir = scratch("hostile");
        std::fs::write(dir.join("x.json"), br#"{"n":1}"#).unwrap();
        let mut specs = Specs::new(Some(dir));
        let mut log = Mem::new();
        let bad = ids(&[
            "../x",
            "a/b",
            "",
            ".",
            "..",
            "a\\b",
            "a\0b",
            &"n".repeat(MAX_MODEL_ID + 1),
        ]);
        specs.refresh(&bad, Instant::now(), &mut log);
        assert_eq!(specs.ops, 0, "no open for an id that is not a file name");
        assert_eq!(specs.reads, 0);
        assert!(log.lines.is_empty());
        assert!(specs.value("x").is_none());
    }

    #[test]
    fn a_valid_id_is_one_component_and_an_extra_file_is_ignored() {
        let dir = scratch("ok");
        std::fs::write(dir.join("a..b.json"), br#"{"n":7}"#).unwrap();
        std::fs::write(
            dir.join(&format!("{}.json", "c".repeat(MAX_MODEL_ID))),
            br#"{"n":8}"#,
        )
        .unwrap();
        // Not requested: a large neighbour must not be opened.
        std::fs::write(dir.join("extra.json"), vec![b'x'; 8]).unwrap();
        let mut specs = Specs::new(Some(dir));
        let mut log = Mem::new();
        let long = "c".repeat(MAX_MODEL_ID);
        specs.refresh(&ids(&["a..b", &long]), Instant::now(), &mut log);
        assert_eq!(specs.reads, 2);
        assert_eq!(specs.value("a..b").and_then(|v| v["n"].as_i64()), Some(7));
        assert_eq!(specs.value(&long).and_then(|v| v["n"].as_i64()), Some(8));
        assert!(specs.value("extra").is_none());
        assert!(log.lines.is_empty());
    }

    #[test]
    fn a_missing_file_is_settings_unknown_not_an_error() {
        let dir = scratch("missing");
        std::fs::write(dir.join("have.json"), br#"{"served_context":131072}"#).unwrap();
        let mut specs = Specs::new(Some(dir));
        let mut log = Mem::new();
        specs.refresh(&ids(&["have", "absent"]), Instant::now(), &mut log);
        assert!(specs.value("absent").is_none());
        assert_eq!(
            specs
                .value("have")
                .and_then(|v| v["served_context"].as_i64()),
            Some(131_072)
        );
        assert_eq!(specs.reads, 1);
        assert!(log.lines.is_empty(), "{:?}", log.lines);
    }

    #[test]
    fn bad_files_log_once_per_cause_and_a_fixed_file_logs_again_later() {
        let dir = scratch("bad");
        let big = dir.join("big.json");
        let bad = dir.join("bad.json");
        let deep = dir.join("deep.json");
        std::fs::write(&big, vec![b'x'; (MAX_SPEC_BYTES as usize) + 1]).unwrap();
        std::fs::write(&bad, b"{").unwrap();
        std::fs::write(&deep, nested(MAX_SPEC_DEPTH + 1)).unwrap();
        let mut specs = Specs::new(Some(dir));
        let mut log = Mem::new();
        let names = ids(&["big", "bad", "deep"]);
        let t0 = Instant::now();
        specs.refresh(&names, t0, &mut log);
        assert_eq!(specs.reads, 2, "too big is not read");
        assert!(specs.value("big").is_none());
        assert!(specs.value("bad").is_none());
        assert!(specs.value("deep").is_none());
        assert_eq!(log.count("big: too big"), 1, "{:?}", log.lines);
        assert_eq!(log.count("bad: not json"), 1, "{:?}", log.lines);
        assert_eq!(log.count("deep: too deep"), 1, "{:?}", log.lines);
        let reads = specs.reads;
        specs.refresh(&names, t0 + Duration::from_secs(10), &mut log);
        assert_eq!(specs.reads, reads, "inside 60 s nothing is stat'd");
        assert_eq!(log.lines.len(), 3);
        specs.refresh(&names, t0 + CHECK_EVERY, &mut log);
        assert_eq!(specs.reads, reads, "unchanged mtime is not read again");
        assert_eq!(log.lines.len(), 3);

        std::fs::write(&bad, br#"{"n":1}"#).unwrap();
        specs.refresh(&names, t0 + CHECK_EVERY + CHECK_EVERY, &mut log);
        assert_eq!(specs.value("bad").and_then(|v| v["n"].as_i64()), Some(1));
        assert_eq!(log.count("bad: not json"), 1, "a success clears the cause");

        std::fs::write(&bad, b"{").unwrap();
        specs.refresh(
            &names,
            t0 + CHECK_EVERY + CHECK_EVERY + CHECK_EVERY,
            &mut log,
        );
        assert_eq!(log.count("bad: not json"), 2, "the fault logs once more");
        assert!(specs.value("bad").is_none());
    }

    #[test]
    fn a_missing_directory_is_logged_once_until_it_is_opened() {
        let root = scratch("nodir");
        let missing = root.join("absent");
        let mut specs = Specs::new(Some(missing.clone()));
        let mut log = Mem::new();
        let t0 = Instant::now();
        specs.refresh(&ids(&["m"]), t0, &mut log);
        assert_eq!(specs.ops, 1);
        assert_eq!(specs.reads, 0);
        assert_eq!(log.count(DIR_CAUSE), 1, "{:?}", log.lines);
        specs.refresh(&ids(&["m"]), t0 + Duration::from_secs(10), &mut log);
        assert_eq!(specs.ops, 1, "a failed open waits out the interval");
        assert_eq!(log.count(DIR_CAUSE), 1);
        specs.refresh(&ids(&["m"]), t0 + CHECK_EVERY, &mut log);
        assert_eq!(specs.ops, 2);
        assert_eq!(log.count(DIR_CAUSE), 1);

        std::fs::create_dir(&missing).unwrap();
        std::fs::write(missing.join("m.json"), br#"{"n":3}"#).unwrap();
        specs.refresh(&ids(&["m"]), t0 + CHECK_EVERY + CHECK_EVERY, &mut log);
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(3));
        assert_eq!(log.count(DIR_CAUSE), 1);
    }

    #[test]
    fn symlinks_are_not_followed_out_of_the_directory() {
        let root = scratch("link");
        let real = root.join("real");
        let outside = root.join("outside");
        let via = root.join("via");
        std::fs::create_dir(&real).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret.json"), br#"{"served_context":1}"#).unwrap();
        // The model's file is a symlink out of the directory. The directory
        // itself is also reached through a symlink.
        std::os::unix::fs::symlink(outside.join("secret.json"), real.join("model.json")).unwrap();
        std::os::unix::fs::symlink(&real, &via).unwrap();

        let mut specs = Specs::new(Some(via));
        let mut log = Mem::new();
        specs.refresh(&ids(&["model"]), Instant::now(), &mut log);
        assert_eq!(specs.reads, 0, "a symlinked directory is not opened");
        assert!(specs.value("model").is_none());
        assert_eq!(log.count(DIR_CAUSE), 1, "{:?}", log.lines);

        let mut specs = Specs::new(Some(real));
        let mut log = Mem::new();
        specs.refresh(&ids(&["model"]), Instant::now(), &mut log);
        assert!(
            specs.value("model").is_none(),
            "a symlinked file is not read"
        );
        assert_eq!(specs.reads, 0);
        assert_eq!(log.count("model: not a file"), 1, "{:?}", log.lines);
        assert!(
            !log.lines
                .iter()
                .any(|line| line.contains("secret") || line.contains("served")),
            "{:?}",
            log.lines
        );
    }

    #[test]
    fn reload_follows_mtime_and_size_and_the_model_set() {
        let dir = scratch("mtime");
        let path = dir.join("m.json");
        std::fs::write(&path, br#"{"n":1}"#).unwrap();
        let mut specs = Specs::new(Some(dir));
        let mut log = Mem::new();
        let t0 = Instant::now();
        specs.refresh(&ids(&["m"]), t0, &mut log);
        assert_eq!(specs.reads, 1);
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(1));
        let ops = specs.ops;

        std::fs::write(&path, br#"{"n":22}"#).unwrap();
        specs.refresh(&ids(&["m"]), t0 + Duration::from_secs(30), &mut log);
        assert_eq!(specs.reads, 1, "a quiet window does not stat");
        assert_eq!(specs.ops, ops);
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(1));

        specs.refresh(
            &ids(&["m", "other"]),
            t0 + Duration::from_secs(30),
            &mut log,
        );
        assert_eq!(specs.reads, 2, "a new model set re-reads at once");
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(22));
        assert!(specs.value("other").is_none());
        assert!(log.lines.is_empty(), "{:?}", log.lines);

        let at = t0 + Duration::from_secs(40);
        specs.refresh(&ids(&["m"]), at, &mut log);
        assert_eq!(specs.reads, 3, "leaving the set re-reads even if unchanged");
        let ops = specs.ops;
        specs.refresh(&ids(&["m"]), at + Duration::from_secs(59), &mut log);
        assert_eq!(specs.reads, 3);
        assert_eq!(specs.ops, ops);
        specs.refresh(&ids(&["m"]), at + CHECK_EVERY, &mut log);
        assert_eq!(specs.reads, 3, "the same mtime and size are not read");
        assert!(specs.ops > ops, "the interval still stats");

        let meta = std::fs::metadata(&path).unwrap();
        let mtime = meta.modified().unwrap();
        std::fs::write(&path, br#"{"n":333}"#).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        specs.refresh(&ids(&["m"]), at + CHECK_EVERY + CHECK_EVERY, &mut log);
        assert_eq!(
            specs.reads, 4,
            "a size change is read even with the old mtime"
        );
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(333));

        let later = mtime + Duration::from_secs(30);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        specs.refresh(
            &ids(&["m"]),
            at + CHECK_EVERY + CHECK_EVERY + CHECK_EVERY,
            &mut log,
        );
        assert_eq!(specs.reads, 5, "an mtime change is read");
        assert_eq!(specs.value("m").and_then(|v| v["n"].as_i64()), Some(333));
    }
}
