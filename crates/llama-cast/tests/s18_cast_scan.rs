//! S18: llama-cast can reach only what it needs.
//!
//! A text scan of `crates/llama-cast/src` with comments removed:
//!
//! - sockets: exactly one `TcpListener::bind` and one UDP bind
//!   (`rustix::net::bind`, the SSDP socket on 1900), both in `service.rs`;
//!   `TcpListener` only in `http.rs`/`service.rs`, `UdpSocket` only in
//!   `discovery.rs`/`service.rs`, datagrams sent only from `discovery.rs`;
//!   no outbound TCP (`connect`), no Unix sockets;
//! - no `/proc` (except `/proc/self`), `/sys`, hidraw, USB, i2c, the
//!   Kraken's vendor id (1e71), NVML, llama API paths or `https://`; the
//!   only `/dev` path is `/dev/vcsa11`, in `source.rs`;
//! - file reads only in `config.rs` (the `--config` argument) and
//!   `source.rs` (vcsa11, the font, machine-id: read-only, `O_NOFOLLOW`);
//!   no file writes anywhere;
//! - one process spawn: `Command::new` once, in `encoder.rs`, with
//!   `env_clear()` and the pinned argument vector; no `unsafe`, no libc;
//! - absolute path literals only from a fixed list;
//! - the manifest's dependencies are pinned.
//!
//! The workspace-wide half (no other crate but llama-metrics listens) is in
//! `crates/kraken-lcd/tests/safety_scan.rs`. Each rule is shown to bite on a
//! planted source below.

use std::fs;
use std::path::{Path, PathBuf};

const FS_READ_FILES: &[&str] = &["config.rs", "source.rs"];
const TCP_FILES: &[&str] = &["http.rs", "service.rs"];
const UDP_FILES: &[&str] = &["discovery.rs", "service.rs"];
const BIND_FILE: &str = "service.rs";
const SPAWN_FILE: &str = "encoder.rs";
const SEND_FILE: &str = "discovery.rs";
const DEV_FILE: &str = "source.rs";
const HTTP_URL_FILES: &[&str] = &["config.rs", "dlna.rs"];
const ALLOWED_PATH_LITERALS: &[&str] = &[
    "/desc.xml",
    "/cds.xml",
    "/cms.xml",
    "/ctl/cds",
    "/ctl/cms",
    "/evt/cds",
    "/evt/cms",
    "/live.ts",
    "/dev/vcsa11",
    "/etc/machine-id",
    "/usr/local/share/llama-bored",
    "/usr/bin/ffmpeg",
];
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

fn code_of(name: &str) -> String {
    strip_comments(&fs::read_to_string(src_dir().join(name)).expect(name))
}

#[test]
fn cast_source_reaches_only_what_it_needs() {
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
        "acl.rs",
        "config.rs",
        "discovery.rs",
        "dlna.rs",
        "encoder.rs",
        "font.rs",
        "http.rs",
        "main.rs",
        "render.rs",
        "service.rs",
        "sha256.rs",
        "source.rs",
        "ssdp.rs",
    ] {
        assert!(
            rels.iter().any(|r| r == required),
            "scan is missing {required}"
        );
    }
    let mut hits = Vec::new();
    let mut tcp_binds = 0;
    let mut udp_binds = 0;
    let mut spawns = 0;
    for (path, rel) in files.iter().zip(&rels) {
        let text = fs::read_to_string(path).expect("read src");
        hits.extend(scan(rel, &text));
        let code = strip_comments(&text);
        tcp_binds += code.matches("TcpListener::bind").count();
        udp_binds += code.matches("rustix::net::bind(").count();
        spawns += code.matches("Command::new").count();
    }
    assert_eq!(tcp_binds, 1, "expected exactly one TcpListener::bind");
    assert_eq!(udp_binds, 1, "expected exactly one UDP bind");
    assert_eq!(spawns, 1, "expected exactly one Command::new");
    assert!(hits.is_empty(), "{}", hits.join("\n"));

    // The one UDP socket is the SSDP port.
    let service = code_of("service.rs");
    assert!(service.contains("SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, ssdp::PORT)"));
    assert!(code_of("ssdp.rs").contains("pub const PORT: u16 = 1900;"));
    // Production reads exactly the reviewed files.
    for needle in [
        "source::VCSA_PATH",
        "source::MACHINE_ID_PATH",
        "source::font_path(config.font)",
    ] {
        assert!(service.contains(needle), "service.rs must use {needle}");
    }
    let source = code_of("source.rs");
    for needle in [
        "OFlags::RDONLY",
        "OFlags::NOFOLLOW",
        "OFlags::NOCTTY",
        "MAX_VCSA_BYTES",
        "MAX_FONT_BYTES",
    ] {
        assert!(source.contains(needle), "source.rs must use {needle}");
    }
    // The spawn: the configured program, the pinned vector, no environment.
    let encoder = code_of("encoder.rs");
    assert!(encoder.contains("Command::new(ffmpeg)"));
    assert!(encoder.contains(".args(ffmpeg_args(settings))"));
    assert!(encoder.contains(".env_clear()"));
    for token in [
        ".arg(",
        ".env(",
        ".envs(",
        "current_dir",
        "\"sh\"",
        "\"-c\"",
    ] {
        assert!(!encoder.contains(token), "encoder.rs uses {token}");
    }
}

#[test]
fn cast_dependencies_are_pinned() {
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
        "a new llama-cast dependency needs review"
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
        "hidraw",
        "usb",
        "i2c",
        "1e71",
        "nvml",
        "https://",
        "/slots",
        "/running",
        "/upstream",
        "/v1/",
        "/api/",
        "/completion",
        "/metrics",
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
    for literal in string_literals(&code) {
        let dev = literal.contains("/dev/") || literal.ends_with("/dev");
        if dev && (rel != DEV_FILE || literal != "/dev/vcsa11") {
            hits.push(format!(
                "{rel}: names /dev outside /dev/vcsa11 in source.rs"
            ));
        }
    }
    if !HTTP_URL_FILES.contains(&rel) && lower.contains("http://") {
        hits.push(format!("{rel}: names http:// outside config.rs/dlna.rs"));
    }
    for token in [
        "connect(",
        "UnixStream",
        "UnixListener",
        "UnixDatagram",
        "libc",
        "include_str!",
        "include_bytes!",
        "env::var",
        "set_current_dir",
        "set_var",
        "Ipv6Addr",
        "SocketAddrV6",
    ] {
        if code.contains(token) {
            hits.push(format!("{rel}: uses {token}"));
        }
    }
    if code
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|ident| ident == "unsafe")
    {
        hits.push(format!("{rel}: uses unsafe"));
    }
    // Processes: only encoder.rs spawns; main.rs only exits.
    let process_uses = code.matches("process::").count();
    let exits = code.matches("std::process::exit(").count();
    if rel == SPAWN_FILE {
        if process_uses != 1
            || !code.contains("use std::process::{Child, ChildStdout, Command, Stdio};")
        {
            hits.push(format!("{rel}: unexpected std::process use"));
        }
    } else {
        if process_uses != exits {
            hits.push(format!("{rel}: uses std::process"));
        }
        if code.contains("Command::new") {
            hits.push(format!("{rel}: spawns a process outside encoder.rs"));
        }
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
            "File::from",
            "read_to_string",
            "read_dir",
            "metadata(",
            "canonicalize",
            "read_link",
            "rustix::fs",
        ] {
            if code.contains(token) {
                hits.push(format!(
                    "{rel}: file read API {token} outside config.rs/source.rs"
                ));
            }
        }
    }
    if rel == "config.rs" && code.contains("rustix::fs") {
        hits.push("config.rs: rustix::fs".to_owned());
    }
    if rel == "source.rs" && code.contains("std::fs::") && !code.contains("use std::fs::File;") {
        hits.push("source.rs: std::fs beyond File".to_owned());
    }
    if rel == "source.rs" && (code.contains("File::open") || code.contains("fs::read")) {
        hits.push("source.rs: opens without rustix O_NOFOLLOW".to_owned());
    }
    if !TCP_FILES.contains(&rel) && (code.contains("TcpListener") || code.contains("TcpStream")) {
        hits.push(format!("{rel}: TCP outside http.rs/service.rs"));
    }
    if !UDP_FILES.contains(&rel) && code.contains("UdpSocket") {
        hits.push(format!("{rel}: UdpSocket outside discovery.rs/service.rs"));
    }
    if rel != BIND_FILE && (code.contains("rustix::net") || code.contains("sockopt")) {
        hits.push(format!("{rel}: rustix::net outside service.rs"));
    }
    if rel != BIND_FILE && (code.contains("::bind") || code.contains(".bind(")) {
        hits.push(format!("{rel}: bind outside service.rs"));
    }
    if code.contains("UdpSocket::bind") {
        hits.push(format!(
            "{rel}: UdpSocket::bind (the SSDP socket is rustix::net::bind in service.rs)"
        ));
    }
    if rel != SEND_FILE && (code.contains("send_to(") || code.contains(".send(")) {
        hits.push(format!("{rel}: datagram send outside discovery.rs"));
    }
    for literal in string_literals(&code) {
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
    assert_hit("render.rs", "let p = \"/sys/class/hwmon\";\n", "/sys");
    assert_hit("render.rs", "let p = \"/dev/hidraw0\";\n", "/dev");
    assert_hit("source.rs", "let p = \"/dev/vcsa1\";\n", "/dev");
    assert_hit("source.rs", "let p = \"/dev/tty11\";\n", "/dev");
    assert_hit("http.rs", "let p = \"/dev/vcsa11\";\n", "/dev");
    assert_hit("render.rs", "let p = \"/proc/stat\";\n", "/proc");
    assert_hit("render.rs", "let id = 0x1E71;\n", "1e71");
    assert_hit("render.rs", "use nvml_wrapper::Nvml;\n", "nvml");
    assert_hit("render.rs", "let bus = \"i2c-5\";\n", "i2c");
    assert_hit("render.rs", "fn usb_reset() {}\n", "usb");
    assert_hit(
        "ssdp.rs",
        "let u = \"http://127.0.0.1:8080/slots\";\n",
        "http://",
    );
    assert_hit("dlna.rs", "let u = \"https://example\";\n", "https://");
    assert!(
        scan(
            "source.rs",
            "pub const VCSA_PATH: &str = \"/dev/vcsa11\";\n"
        )
        .is_empty()
    );
    assert!(scan("render.rs", "// reads /sys and /dev/hidraw0\nfn f() {}\n").is_empty());
}

#[test]
fn scanner_flags_sockets() {
    assert_hit("http.rs", "let s = TcpStream::connect(a);\n", "connect(");
    assert_hit("discovery.rs", "socket.connect(peer)?;\n", "connect(");
    assert_hit(
        "render.rs",
        "let l = TcpListener::bind(a);\n",
        "TCP outside",
    );
    assert_hit("http.rs", "let l = TcpListener::bind(a);\n", "bind outside");
    assert_hit("render.rs", "let s: UdpSocket = x;\n", "UdpSocket outside");
    assert_hit(
        "service.rs",
        "let s = UdpSocket::bind(a);\n",
        "UdpSocket::bind",
    );
    assert_hit(
        "discovery.rs",
        "rustix::net::bind(&fd, &a)?;\n",
        "rustix::net outside",
    );
    assert_hit("http.rs", "sock.send_to(b, a);\n", "datagram send");
    assert_hit("http.rs", "let u = UnixStream::connect(p);\n", "UnixStream");
    assert_hit("dlna.rs", "let a = Ipv6Addr::LOCALHOST;\n", "Ipv6Addr");
}

#[test]
fn scanner_flags_processes_and_files() {
    assert_hit(
        "http.rs",
        "std::process::Command::new(\"sh\");\n",
        "std::process",
    );
    assert_hit(
        "service.rs",
        "let c = Command::new(p);\n",
        "spawns a process",
    );
    assert_hit(
        "encoder.rs",
        "use std::process::{Child, ChildStdout, Command, Stdio};\nlet o = std::process::Command::new(\"sh\");\n",
        "unexpected std::process",
    );
    assert_hit(
        "render.rs",
        "let t = std::fs::read_to_string(p);\n",
        "file read API",
    );
    assert_hit("http.rs", "let f = File::open(p);\n", "file read API");
    assert_hit("source.rs", "let f = File::open(p);\n", "O_NOFOLLOW");
    assert_hit("source.rs", "std::fs::write(p, b);\n", "file write API");
    assert_hit(
        "config.rs",
        "let f = OpenOptions::new();\n",
        "file write API",
    );
    assert_hit(
        "source.rs",
        "let f = OFlags::RDONLY | OFlags::CREATE;\n",
        "file write API",
    );
    assert_hit("config.rs", "let f = rustix::fs::open(p);\n", "rustix::fs");
    assert_hit("render.rs", "unsafe { x() }\n", "unsafe");
    assert_hit(
        "encoder.rs",
        "let v = std::env::var(\"PATH\");\n",
        "env::var",
    );
}

#[test]
fn scanner_flags_path_literals() {
    assert_hit("config.rs", "let p = \"/etc/shadow\";\n", "path literal");
    assert_hit("config.rs", "let p = \"a/../b\";\n", "`..`");
    assert_hit("dlna.rs", "const P: &str = \"/metrics\";\n", "/metrics");
    assert!(scan("dlna.rs", "const P: &str = \"/live.ts\";\n").is_empty());
    assert!(scan("render.rs", "let c = '\"'; let s = \"x\";\n").is_empty());
}
