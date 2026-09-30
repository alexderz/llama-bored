//! S16: llama-metrics can reach nothing sensitive.
//!
//! A text scan of `crates/llama-metrics/src` with comments removed:
//!
//! - no `/proc` (except `/proc/self`), `/sys`, `/dev`, hidraw, USB, i2c,
//!   the Kraken's vendor id (1e71), NVML, or any llama/HTTP client URL;
//! - no outbound socket, no process spawn, no `unsafe`, no libc;
//! - the only absolute path literal is the route `/metrics`; the snapshot
//!   path is `llama_core::wire::SNAPSHOT_PATH`, never a literal;
//! - file reads only in `snapshot.rs` (the snapshot) and `config.rs` (the
//!   `--config` argument, once at start); no file writes anywhere;
//! - `TcpListener` only in `http.rs` and `service.rs`, and exactly one
//!   `TcpListener::bind`, in `service.rs`;
//! - the manifest's dependencies are pinned.
//!
//! The workspace-wide half (no crate but this one and llama-cast binds or
//! listens) is in `crates/kraken-lcd/tests/safety_scan.rs`. Each rule is
//! shown to bite on a planted source below.

use std::fs;
use std::path::{Path, PathBuf};

const FS_READ_FILES: &[&str] = &["snapshot.rs", "config.rs"];
const LISTENER_FILES: &[&str] = &["http.rs", "service.rs"];
const BIND_FILE: &str = "service.rs";
const ALLOWED_PATH_LITERALS: &[&str] = &["/metrics"];
const DEPENDENCIES: &[&str] = &[
    "llama-core",
    "rustix",
    "sd-notify",
    "serde",
    "thiserror",
    "toml",
];

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read dir") {
        let path = entry.expect("entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn metrics_source_reaches_nothing_sensitive() {
    let src = src_dir();
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    let rels: Vec<String> = files
        .iter()
        .map(|p| {
            p.strip_prefix(&src)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    for required in [
        "http.rs",
        "service.rs",
        "snapshot.rs",
        "config.rs",
        "expo.rs",
        "acl.rs",
        "main.rs",
    ] {
        assert!(
            rels.iter().any(|r| r == required),
            "scan is missing {required}"
        );
    }
    let mut hits = Vec::new();
    let mut binds = 0;
    for (path, rel) in files.iter().zip(&rels) {
        let text = fs::read_to_string(path).expect("read src");
        hits.extend(scan(rel, &text));
        binds += strip_comments(&text).matches("TcpListener::bind").count();
    }
    assert_eq!(binds, 1, "expected exactly one TcpListener::bind");
    let snapshot = strip_comments(&fs::read_to_string(src.join("snapshot.rs")).unwrap());
    assert!(
        snapshot.contains("wire::SNAPSHOT_PATH"),
        "snapshot.rs must read llama_core::wire::SNAPSHOT_PATH"
    );
    assert!(
        snapshot.contains("OFlags::NOFOLLOW"),
        "snapshot open must not follow symlinks"
    );
    assert!(
        snapshot.contains("wire::MAX_BYTES"),
        "snapshot read must be capped"
    );
    assert!(
        snapshot.contains("parse_validated"),
        "snapshot must use the validated parser"
    );
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn metrics_dependencies_are_pinned() {
    let manifest =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    let mut deps = Vec::new();
    let mut on = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            on = line == "[dependencies]";
            continue;
        }
        if on && !line.is_empty() && !line.starts_with('#') {
            deps.push(line.split(['=', ' ']).next().unwrap().to_owned());
        }
    }
    assert_eq!(
        deps, DEPENDENCIES,
        "a new llama-metrics dependency needs review"
    );
    assert!(!manifest.contains("[build-dependencies]"));
    assert!(
        !manifest.contains("[target."),
        "no per-target dependency tables"
    );
}

fn scan(rel: &str, text: &str) -> Vec<String> {
    let code = strip_comments(text);
    let lower = code.to_ascii_lowercase();
    let mut hits = Vec::new();
    for token in [
        "/sys",
        "/dev",
        "hidraw",
        "usb",
        "i2c",
        "1e71",
        "nvml",
        "http://",
        "https://",
        "/slots",
        "/running",
        "/upstream",
        "/v1/",
        "/api/",
        "/completion",
        "ureq",
        ":8080",
        "localhost",
    ] {
        if lower.contains(token) {
            hits.push(format!("{rel}: names {token}"));
        }
    }
    let mut rest = code.as_str();
    while let Some(at) = rest.find("/proc") {
        if !rest[at..].starts_with("/proc/self") {
            hits.push(format!("{rel}: names /proc"));
        }
        rest = &rest[at + 5..];
    }
    for token in [
        "TcpStream::connect",
        "connect(",
        "UdpSocket",
        "UnixStream",
        "UnixListener",
        "UnixDatagram",
        "rustix::net",
        "Command::new",
        "process::Command",
        "libc",
        "include_str!",
        "include_bytes!",
        "env::var",
        "set_current_dir",
    ] {
        if code.contains(token) {
            hits.push(format!("{rel}: uses {token}"));
        }
    }
    // `std::process::exit` in main.rs is the only process API.
    if code.matches("process::").count() != code.matches("std::process::exit(").count() {
        hits.push(format!("{rel}: uses std::process"));
    }
    if code
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|ident| ident == "unsafe")
    {
        hits.push(format!("{rel}: uses unsafe"));
    }
    // No file writes, anywhere.
    for token in [
        "fs::write",
        "File::create",
        "OpenOptions",
        "OFlags::CREATE",
        "OFlags::WRONLY",
        "OFlags::RDWR",
        "OFlags::TRUNC",
        "OFlags::APPEND",
        "create_dir",
        "remove_file",
        "rename(",
        "symlink",
        "set_permissions",
        "chmod",
        "chown",
    ] {
        if code.contains(token) {
            hits.push(format!("{rel}: file write API {token}"));
        }
    }
    // File reads only where reviewed.
    if !FS_READ_FILES.contains(&rel) {
        for token in [
            "std::fs",
            "fs::",
            "File::open",
            "fs::File",
            "read_to_string",
            "read_dir",
            "metadata(",
            "canonicalize",
            "read_link",
        ] {
            if code.contains(token) {
                hits.push(format!(
                    "{rel}: file read API {token} outside snapshot.rs/config.rs"
                ));
            }
        }
    }
    if rel == "config.rs" && code.contains("rustix::fs") {
        hits.push("config.rs: rustix::fs".to_owned());
    }
    if rel == "snapshot.rs" && code.contains("std::fs") {
        hits.push("snapshot.rs: std::fs".to_owned());
    }
    if !LISTENER_FILES.contains(&rel) && code.contains("TcpListener") {
        hits.push(format!("{rel}: TcpListener outside http.rs/service.rs"));
    }
    if rel != BIND_FILE && (code.contains("::bind") || code.contains(".bind(")) {
        hits.push(format!("{rel}: bind outside service.rs"));
    }
    for literal in string_literals(&code) {
        if literal.contains("snapshot.json") {
            hits.push(format!("{rel}: snapshot path literal {literal:?}"));
        }
        if literal.contains("..") && literal.contains('/') {
            hits.push(format!("{rel}: `..` in a path literal {literal:?}"));
        }
        if literal.starts_with('/') && !ALLOWED_PATH_LITERALS.contains(&literal.as_str()) {
            hits.push(format!("{rel}: path literal {literal:?}"));
        }
    }
    hits
}

/// Length of a char literal starting at `i` (`'x'`, `'\n'`, `'\u{7f}'`),
/// or `None` for a lifetime.
fn char_literal_len(chars: &[char], i: usize) -> Option<usize> {
    if chars.get(i) != Some(&'\'') {
        return None;
    }
    if chars.get(i + 1) == Some(&'\\') {
        let end = (i + 2..chars.len().min(i + 12)).find(|&j| chars[j] == '\'')?;
        return Some(end - i + 1);
    }
    if chars.get(i + 2) == Some(&'\'') {
        return Some(3);
    }
    None
}

/// Walk code: comments removed, char literals kept whole, and each string
/// literal's contents reported to `on_string`.
fn walk(src: &str, mut on_string: impl FnMut(&str)) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i += 2;
            out.push(' ');
            continue;
        }
        if let Some(len) = char_literal_len(&chars, i) {
            out.extend(&chars[i..i + len]);
            i += len;
            continue;
        }
        if ch == '"' {
            let mut lit = String::new();
            out.push('"');
            i += 1;
            while i < chars.len() {
                let next = chars[i];
                out.push(next);
                i += 1;
                if next == '\\' {
                    lit.push(next);
                    if let Some(esc) = chars.get(i) {
                        out.push(*esc);
                        lit.push(*esc);
                        i += 1;
                    }
                    continue;
                }
                if next == '"' {
                    break;
                }
                lit.push(next);
            }
            on_string(&lit);
            continue;
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn strip_comments(src: &str) -> String {
    walk(src, |_| {})
}

/// String literal contents (escape sequences kept as written).
fn string_literals(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    walk(code, |lit| out.push(lit.to_owned()));
    out
}

fn assert_hit(rel: &str, text: &str, needle: &str) {
    let hits = scan(rel, text);
    assert!(
        hits.iter().any(|h| h.contains(needle)),
        "planted {needle:?} in {rel} was not flagged: {hits:?}"
    );
}

#[test]
fn scanner_flags_device_and_kernel_paths() {
    assert_hit("expo.rs", "let p = \"/sys/class/hwmon\";\n", "/sys");
    assert_hit("expo.rs", "let p = \"/dev/hidraw0\";\n", "/dev");
    assert_hit("expo.rs", "let p = \"/proc/stat\";\n", "/proc");
    assert_hit("expo.rs", "let id = 0x1E71;\n", "1e71");
    assert_hit("expo.rs", "use nvml_wrapper::Nvml;\n", "nvml");
    assert_hit("expo.rs", "let bus = \"i2c-5\";\n", "i2c");
    assert_hit(
        "expo.rs",
        "let u = \"http://127.0.0.1:8080/slots\";\n",
        "http://",
    );
    assert_hit("expo.rs", "fn usb_reset() {}\n", "usb");
    assert!(
        scan("expo.rs", "let p = \"/proc/self/status\";\n")
            .iter()
            .all(|h| !h.contains("names /proc"))
    );
    // A comment is not code.
    assert!(scan("expo.rs", "// reads /sys and /dev/hidraw0\nfn f() {}\n").is_empty());
}

#[test]
fn scanner_flags_sockets_processes_and_files() {
    assert_hit(
        "expo.rs",
        "let s = std::net::TcpStream::connect(a);\n",
        "connect",
    );
    assert_hit("expo.rs", "let l = TcpListener::bind(a);\n", "TcpListener");
    assert_hit("http.rs", "let l = TcpListener::bind(a);\n", "bind outside");
    assert_hit(
        "expo.rs",
        "let s = std::net::UdpSocket::bind(a);\n",
        "UdpSocket",
    );
    assert_hit(
        "expo.rs",
        "std::process::Command::new(\"sh\");\n",
        "std::process",
    );
    assert_hit(
        "expo.rs",
        "let t = std::fs::read_to_string(p);\n",
        "file read API",
    );
    assert_hit("http.rs", "let f = File::open(p);\n", "file read API");
    assert_hit("snapshot.rs", "std::fs::write(p, b);\n", "file write API");
    assert_hit(
        "config.rs",
        "let f = OpenOptions::new();\n",
        "file write API",
    );
    assert_hit(
        "snapshot.rs",
        "let f = OFlags::RDONLY | OFlags::CREATE;\n",
        "file write API",
    );
    assert_hit("snapshot.rs", "let t = std::fs::read(p);\n", "std::fs");
    assert_hit("expo.rs", "unsafe { x() }\n", "unsafe");
}

#[test]
fn scanner_flags_path_literals() {
    assert_hit(
        "snapshot.rs",
        "let p = \"/run/llama-watch/snapshot.json\";\n",
        "snapshot path literal",
    );
    assert_hit("config.rs", "let p = \"/etc/shadow\";\n", "path literal");
    assert_hit("config.rs", "let p = \"a/../b\";\n", "`..`");
    assert!(scan("http.rs", "const P: &str = \"/metrics\";\n").is_empty());
    assert!(scan("expo.rs", "let c = '\"'; let s = \"x\";\n").is_empty());
}
