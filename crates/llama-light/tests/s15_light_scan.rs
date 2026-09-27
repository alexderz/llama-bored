//! S15: llama-light's source stays a colour-only, network-free writer.
//!
//! The writer-style scans (S11 for kraken-lcd) extended to llama-light:
//! - no network, no process control, no USB stack other than hidraw writes;
//! - no i2c/SMBus, no NZXT vendor id, no pwm/hwmon (no cooling path at all);
//! - filesystem calls only in the allowlisted files, no filesystem writes,
//!   and path literals only from a fixed list;
//! - a closed opcode table in `aura/proto.rs` with no save/commit opcode;
//! - a closed command and property table in `keyboard/proto.rs` with no
//!   firmware, reset, profile-save, key-routing or read command.
//!
//! Comments are stripped before the scan. Each check also runs against a
//! scratch tree that plants one violation, so a weakened scanner fails.
//! Method calls that a text scan cannot see are backstopped by the unit
//! sandbox (`PrivateNetwork`, `DevicePolicy=closed`, `ReadOnlyPaths=/sys`).

use std::fs;
use std::path::{Path, PathBuf};

/// Files that may touch the filesystem, relative to `src/`.
///
/// `config.rs` reads the config named on the command line; `snapshot.rs`
/// reads the published snapshot; `hidraw.rs` resolves a udev pin, reads
/// sysfs and opens the pinned hidraw node.
const FS_FILES: &[&str] = &["config.rs", "snapshot.rs", "hidraw.rs"];

/// The only file that may open anything for writing.
const WRITE_OPEN_FILE: &str = "hidraw.rs";

/// Absolute path literals allowed, and where.
/// `keyboard/keymap.rs` names the `/` key.
const PATH_LITERALS: &[(&str, &str)] = &[
    ("main.rs", "/dev"),
    ("main.rs", "/sys"),
    ("keyboard/keymap.rs", "/"),
];

/// The closed opcode table.
const OPCODES: &[u8] = &[0x35, 0x40];

/// Save/commit and config opcodes that must never appear as a literal.
const FORBIDDEN_OPCODE_LITERALS: &[&str] = &["0x3F", "0x3E", "0x36", "0xB0", "0x82"];

/// The keyboard's closed tables: commands, then write properties.
const KB_OPCODES: &[u8] = &[0x07, 0x7F];
const KB_PROPERTIES: &[u8] = &[0x05, 0x28];

/// Keyboard property and command bytes that must never appear as a literal
/// in `keyboard/proto.rs`: special function, poll rate, firmware update,
/// profile and hardware-lighting saves, 9-bit commit, key input routing,
/// and the read command. Reset (`07 02`) shares its byte with software mode
/// and the apply flag, so the property table check below catches it.
const KB_FORBIDDEN_LITERALS: &[&str] = &[
    "0x04", "0x0A", "0x0C", "0x0D", "0x0E", "0x13", "0x14", "0x15", "0x16", "0x17", "0x27", "0x40",
];
#[test]
fn llama_light_source_passes_s15() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let rels = rs_rels(&src);
    for required in [
        "main.rs",
        "aura/proto.rs",
        "aura/device.rs",
        "hidraw.rs",
        "keyboard/proto.rs",
        "keyboard/device.rs",
        "service.rs",
        "config.rs",
    ] {
        assert!(
            rels.iter().any(|rel| rel == required),
            "{required} missing from the scan"
        );
    }
    let hits = scan_tree(&src);
    assert!(hits.is_empty(), "S15:\n{}", hits.join("\n"));
}

#[test]
fn the_opcode_table_in_the_source_is_closed() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/aura/proto.rs");
    let code = strip_comments(&fs::read_to_string(src).expect("proto.rs"));
    let ops = opcode_consts(&code);
    assert_eq!(ops, OPCODES.to_vec(), "OP_* constants");
    assert!(
        code.contains("pub const OPCODES: [u8; 2] = [OP_EFFECT, OP_DIRECT];"),
        "OPCODES is not the two-entry table"
    );
    assert_eq!(llama_light::aura::proto::OPCODES.to_vec(), OPCODES.to_vec());
}

#[test]
fn the_keyboard_tables_in_the_source_are_closed() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/keyboard/proto.rs");
    let code = strip_comments(&fs::read_to_string(src).expect("proto.rs"));
    assert_eq!(consts_named(&code, "const OP_"), KB_OPCODES.to_vec());
    assert_eq!(consts_named(&code, "const PROP_"), KB_PROPERTIES.to_vec());
    assert!(code.contains("pub const OPCODES: [u8; 2] = [OP_WRITE, OP_STREAM];"));
    assert!(code.contains("pub const PROPERTIES: [u8; 2] = [PROP_LIGHTING, PROP_COMMIT];"));
    assert_eq!(
        llama_light::keyboard::proto::OPCODES.to_vec(),
        KB_OPCODES.to_vec()
    );
    assert_eq!(
        llama_light::keyboard::proto::PROPERTIES.to_vec(),
        KB_PROPERTIES.to_vec()
    );
}

// ---- plants: each must be caught ----------------------------------------

fn plant(name: &str, rel: &str, body: &str) -> Vec<String> {
    let root = scratch(name);
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(&path, body).expect("write");
    let hits = scan_tree(&root);
    let _ = fs::remove_dir_all(&root);
    hits
}

fn assert_caught(name: &str, rel: &str, body: &str, label: &str) {
    let hits = plant(name, rel, body);
    assert!(
        hits.iter().any(|hit| hit.contains(label)),
        "planted {name} was not caught as {label}: {hits:?}"
    );
}

#[test]
fn planted_network_is_caught() {
    assert_caught(
        "net",
        "service.rs",
        "use std::net::TcpStream;\n",
        "std::net",
    );
    assert_caught("tcp", "service.rs", "fn f(s: TcpStream) {}\n", "TcpStream");
    assert_caught("udp", "service.rs", "fn f(s: UdpSocket) {}\n", "UdpSocket");
    assert_caught(
        "unix",
        "service.rs",
        "fn f(s: UnixStream) {}\n",
        "UnixStream",
    );
    assert_caught(
        "rnet",
        "service.rs",
        "use rustix::net::socket;\n",
        "rustix::net",
    );
    assert_caught(
        "ureq",
        "service.rs",
        "fn f() { ureq::get(\"x\"); }\n",
        "ureq",
    );
    assert_caught(
        "split",
        "service.rs",
        "use std::/* x */net::TcpStream;\n",
        "std::net",
    );
}

#[test]
fn planted_cooling_and_bus_paths_are_caught() {
    assert_caught(
        "i2c",
        "service.rs",
        "const P: &str = \"/dev/i2c-5\";\n",
        "i2c",
    );
    assert_caught("smbus", "service.rs", "fn smbus_write() {}\n", "smbus");
    assert_caught("nzxt", "service.rs", "const V: &str = \"1e71\";\n", "1e71");
    assert_caught("nzxt-hex", "service.rs", "const V: u16 = 0x1E71;\n", "1e71");
    assert_caught("pwm", "service.rs", "const P: &str = \"pwm1\";\n", "pwm");
    assert_caught(
        "hwmon",
        "keyboard.rs",
        "const P: &str = \"class/hwmon\";\n",
        "hwmon",
    );
    assert_caught("nusb", "service.rs", "use nusb::Device;\n", "nusb");
}

#[test]
fn planted_process_control_is_caught() {
    assert_caught(
        "cmd",
        "service.rs",
        "fn f() { std::process::Command::new(\"x\"); }\n",
        "std::process",
    );
    assert_caught(
        "exit",
        "main.rs",
        "fn f() { std::process::exit(1); }\n",
        "std::process",
    );
    assert_caught("cmd2", "service.rs", "use std::process;\n", "std::process");
    assert_caught(
        "env",
        "service.rs",
        "fn f() { std::env::var(\"X\"); }\n",
        "std::env",
    );
    // The one allowed form: main returns an ExitCode.
    let hits = plant(
        "exitcode",
        "main.rs",
        "use std::process::ExitCode;\nfn main() -> ExitCode { ExitCode::SUCCESS }\n",
    );
    assert!(hits.is_empty(), "{hits:?}");
}

#[test]
fn planted_filesystem_use_outside_the_allowlist_is_caught() {
    assert_caught(
        "fs",
        "service.rs",
        "fn f() { std::fs::read(\"x\"); }\n",
        "filesystem",
    );
    assert_caught(
        "file",
        "mapping.rs",
        "fn f() { File::open(\"x\"); }\n",
        "filesystem",
    );
    assert_caught(
        "rfs",
        "service.rs",
        "fn f() { rustix::fs::stat(\"x\"); }\n",
        "filesystem",
    );
    assert_caught(
        "rio",
        "service.rs",
        "fn f() { rustix::io::write(fd, b); }\n",
        "filesystem",
    );
    assert_caught("brace", "service.rs", "use std::{fs, io};\n", "filesystem");
    assert_caught(
        "kbfs",
        "keyboard/mod.rs",
        "fn f() { std::fs::read(\"x\"); }\n",
        "filesystem",
    );
    // Allowed where listed.
    let hits = plant(
        "ok",
        "hidraw.rs",
        "fn f() { let _ = std::fs::read_to_string(p); }\n",
    );
    assert!(hits.is_empty(), "{hits:?}");
}

#[test]
fn planted_filesystem_writes_are_caught_everywhere() {
    assert_caught(
        "fsw",
        "config.rs",
        "fn f() { std::fs::write(p, b); }\n",
        "filesystem write",
    );
    assert_caught(
        "mk",
        "keyboard.rs",
        "fn f() { std::fs::create_dir_all(p); }\n",
        "filesystem write",
    );
    assert_caught(
        "rm",
        "snapshot.rs",
        "fn f() { std::fs::remove_file(p); }\n",
        "filesystem write",
    );
    assert_caught(
        "create",
        "config.rs",
        "fn f() { o.create(true); }\n",
        "filesystem write",
    );
    assert_caught(
        "wopen",
        "config.rs",
        "fn f() { o.write(true); }\n",
        "open for writing",
    );
    assert_caught(
        "wopen2",
        "snapshot.rs",
        "fn f() { OFlags::WRONLY; }\n",
        "open for writing",
    );
    let hits = plant(
        "devw",
        "hidraw.rs",
        "fn f() { let _ = std::fs::OpenOptions::new().write(true); }\n",
    );
    assert!(hits.is_empty(), "{hits:?}");
    assert_caught(
        "kbw",
        "keyboard/device.rs",
        "fn f() { let _ = std::fs::OpenOptions::new().write(true); }\n",
        "open for writing",
    );
    assert_caught(
        "auraw",
        "aura/device.rs",
        "fn f() { let _ = std::fs::OpenOptions::new().write(true); }\n",
        "open for writing",
    );
}

#[test]
fn planted_path_literals_are_caught() {
    assert_caught(
        "etc",
        "config.rs",
        "const P: &str = \"/etc/llama-bored/light.toml\";\n",
        "path literal",
    );
    assert_caught(
        "run",
        "snapshot.rs",
        "const P: &str = \"/run/llama-watch/snapshot.json\";\n",
        "path literal",
    );
    assert_caught(
        "devlit",
        "service.rs",
        "const P: &str = \"/dev\";\n",
        "path literal",
    );
    assert_caught("parent", "keyboard.rs", "const P: &str = \"../x\";\n", "..");
    assert_caught(
        "raw",
        "service.rs",
        "const P: &str = r\"/proc/self\";\n",
        "path literal",
    );
}

#[test]
fn planted_opcodes_are_caught() {
    assert_caught(
        "save",
        "aura/proto.rs",
        "pub const OP_SAVE: u8 = 0x3F;\n",
        "0x3F",
    );
    assert_caught(
        "save-lower",
        "service.rs",
        "let b = [0xEC, 0x3f, 0x55];\n",
        "0x3F",
    );
    assert_caught(
        "cfg",
        "aura/proto.rs",
        "pub const OP_CFG: u8 = 0x3E;\n",
        "0x3E",
    );
    assert_caught(
        "color",
        "aura/proto.rs",
        "pub const OP_COLOR: u8 = 0x36;\n",
        "0x36",
    );
    assert_caught("report-id", "service.rs", "let id = 0xEC;\n", "0xEC");
    assert_caught(
        "new-op",
        "aura/proto.rs",
        "pub const OP_OTHER: u8 = 0x41;\n",
        "opcode table",
    );
}

#[test]
fn planted_keyboard_commands_are_caught() {
    assert_caught(
        "kb-read",
        "keyboard/proto.rs",
        "pub const OP_READ: u8 = 0x0E;\n",
        "keyboard command table",
    );
    assert_caught(
        "kb-prop",
        "keyboard/proto.rs",
        "pub const PROP_SPECIAL: u8 = 0x04;\n",
        "keyboard property table",
    );
    assert_caught(
        "kb-fw",
        "keyboard/proto.rs",
        "fn f() { let b = [0x07, 0x0C, 0xF0]; }\n",
        "0x0C",
    );
    assert_caught(
        "kb-save",
        "keyboard/proto.rs",
        "fn f() { let b = [0x07, 0x14]; }\n",
        "0x14",
    );
    assert_caught(
        "kb-keys",
        "keyboard/proto.rs",
        "fn f() { let b = [0x07, 0x40]; }\n",
        "0x40",
    );
    assert_caught(
        "kb-reset",
        "keyboard/proto.rs",
        "pub const PROP_RESET: u8 = 0x02;\n",
        "keyboard property table",
    );
    assert_caught(
        "kb-stream-op",
        "keyboard/proto.rs",
        "pub const OP_OTHER: u8 = 0x41;\n",
        "keyboard command table",
    );
}

#[test]
fn planted_source_escapes_are_caught() {
    assert_caught("inc", "service.rs", "include!(\"x.rs\");\n", "include");
    assert_caught(
        "incb",
        "service.rs",
        "const B: &[u8] = include_bytes!(\"x\");\n",
        "include",
    );
    assert_caught(
        "path",
        "service.rs",
        "#[path = \"x.rs\"]\nmod x;\n",
        "#[path",
    );
    assert_caught(
        "mac",
        "service.rs",
        "macro_rules! m { () => {} }\n",
        "macro_rules",
    );
    assert_caught(
        "envm",
        "service.rs",
        "const X: &str = env!(\"HOME\");\n",
        "env!",
    );
}

#[test]
fn comments_do_not_count_and_strings_do_not_hide_code() {
    let hits = plant(
        "comment",
        "service.rs",
        "// std::net and 0x3F and pwm and i2c\n/* 1e71 */\nfn f() {}\n",
    );
    assert!(hits.is_empty(), "{hits:?}");
    // A quote in a char literal must not swallow the code after it.
    assert_caught(
        "quote",
        "service.rs",
        "fn f() { let q = '\"'; std::process::abort(); }\n",
        "std::process",
    );
    // A lifetime is not a char literal.
    assert_caught(
        "life",
        "service.rs",
        "fn f<'a>(x: &'a str) { std::process::abort(); }\n",
        "std::process",
    );
    // "//" inside a string is not a comment.
    assert_caught(
        "strcomment",
        "service.rs",
        "fn f() { let u = \"a//b\"; std::process::abort(); }\n",
        "std::process",
    );
}

#[test]
fn an_empty_tree_is_a_failure() {
    let root = scratch("empty");
    let hits = scan_tree(&root);
    let _ = fs::remove_dir_all(&root);
    assert!(!hits.is_empty());
}

// ---- the scanner ----------------------------------------------------------

fn scan_tree(root: &Path) -> Vec<String> {
    let rels = rs_rels(root);
    let mut hits = Vec::new();
    if rels.is_empty() {
        hits.push(format!("{}: source set is empty", root.display()));
        return hits;
    }
    for rel in &rels {
        let text = fs::read_to_string(root.join(rel)).expect("read");
        let (code, literals) = split_source(&text);
        hits.extend(scan_file(rel, &code, &literals));
    }
    hits.sort();
    hits
}

fn scan_file(rel: &str, code: &str, literals: &[String]) -> Vec<String> {
    let mut hits = Vec::new();
    let mut hit = |what: &str| hits.push(format!("{rel}: {what}"));
    let joined = squash(code);
    let lower = joined.to_ascii_lowercase();
    let lit_lower: Vec<String> = literals.iter().map(|l| l.to_ascii_lowercase()).collect();

    // Network and foreign transports.
    for (needle, label) in [
        ("std::net", "std::net"),
        ("rustix::net", "rustix::net"),
        ("std::os::unix::net", "std::net"),
    ] {
        if joined.contains(needle) {
            hit(label);
        }
    }
    for ident in [
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "UnixStream",
        "UnixListener",
        "UnixDatagram",
        "ureq",
        "nusb",
        "hidapi",
        "libusb",
        "libc",
    ] {
        if has_ident(code, ident) {
            hit(ident);
        }
    }
    // Cooling and bus paths, in code or in strings.
    for needle in ["i2c", "smbus", "1e71", "pwm", "hwmon"] {
        if lower.contains(needle) || lit_lower.iter().any(|l| l.contains(needle)) {
            hit(needle);
        }
    }
    // Process control. `main.rs` may import `std::process::ExitCode` only.
    let mut process = joined.clone();
    if rel == "main.rs" {
        process = process.replace("usestd::process::ExitCode;", "");
    }
    if process.contains("std::process") || has_ident(code, "Command") {
        hit("std::process");
    }
    let mut env = joined.clone();
    if rel == "main.rs" {
        env = env.replace("std::env::args()", "");
    }
    if env.contains("std::env") {
        hit("std::env");
    }
    // Source escapes.
    for (needle, label) in [
        ("include!", "include"),
        ("include_bytes!", "include"),
        ("include_str!", "include"),
        ("#[path", "#[path"),
        ("macro_rules!", "macro_rules"),
        ("env!", "env!"),
    ] {
        if joined.contains(needle) {
            hit(label);
        }
    }
    // Filesystem reads outside the allowlist.
    let fs_allowed = FS_FILES.contains(&rel);
    let uses_fs = joined.contains("std::fs")
        || joined.contains("rustix::fs")
        || joined.contains("rustix::io")
        || has_ident(code, "File")
        || has_ident(code, "OpenOptions")
        || brace_import(&joined, "std", "fs");
    if uses_fs && !fs_allowed {
        hit("filesystem use outside the allowlist");
    }
    // Filesystem writes: none anywhere.
    for needle in [
        "fs::write",
        "create_dir",
        "remove_file",
        "remove_dir",
        "fs::rename",
        "fs::copy",
        "set_permissions",
        "hard_link",
        "fs::symlink",
        ".create(true)",
        ".create_new(",
        ".append(true)",
        ".truncate(true)",
        "OFlags::CREATE",
        "OFlags::TRUNC",
    ] {
        if joined.contains(needle) {
            hit(&format!("filesystem write ({needle})"));
        }
    }
    // Opening for writing: only the hidraw open.
    if rel != WRITE_OPEN_FILE
        && (joined.contains(".write(true)")
            || joined.contains("OFlags::WRONLY")
            || joined.contains("OFlags::RDWR"))
    {
        hit("open for writing outside hidraw.rs");
    }
    // Path literals.
    for literal in literals {
        if literal.starts_with('/') && !PATH_LITERALS.contains(&(rel, literal.as_str())) {
            hit(&format!("path literal \"{literal}\""));
        }
        if literal.contains("../") || literal.contains("/..") {
            hit(&format!("\"..\" in literal \"{literal}\""));
        }
    }
    // Opcodes.
    for literal in FORBIDDEN_OPCODE_LITERALS {
        if has_hex_literal(&joined, literal) {
            hit(literal);
        }
    }
    if rel != "aura/proto.rs" && has_hex_literal(&joined, "0xEC") {
        hit("0xEC report id outside aura/proto.rs");
    }
    if rel == "aura/proto.rs" {
        let ops = opcode_consts(code);
        if ops.iter().any(|op| !OPCODES.contains(op)) {
            hit(&format!(
                "opcode table has {ops:02x?}, allowed {OPCODES:02x?}"
            ));
        }
    }
    if rel == "keyboard/proto.rs" {
        let ops = consts_named(code, "const OP_");
        if ops.iter().any(|op| !KB_OPCODES.contains(op)) {
            hit(&format!(
                "keyboard command table has {ops:02x?}, allowed {KB_OPCODES:02x?}"
            ));
        }
        let props = consts_named(code, "const PROP_");
        if props.iter().any(|prop| !KB_PROPERTIES.contains(prop)) {
            hit(&format!(
                "keyboard property table has {props:02x?}, allowed {KB_PROPERTIES:02x?}"
            ));
        }
        for literal in KB_FORBIDDEN_LITERALS {
            if has_hex_literal(&joined, literal) {
                hit(literal);
            }
        }
    }
    hits
}

/// Values of `const OP_*: u8 = 0x..;` in order.
fn opcode_consts(code: &str) -> Vec<u8> {
    consts_named(code, "const OP_")
}

/// Values of `<prefix>*: u8 = 0x..;` in order. An unparseable value is
/// `0xFF`, which no table holds.
fn consts_named(code: &str, prefix: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = code;
    while let Some(at) = rest.find(prefix) {
        let after = &rest[at..];
        let Some(eq) = after.find('=') else { break };
        let Some(semi) = after.find(';') else { break };
        let value = after[eq + 1..semi].trim();
        if let Some(hex) = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
        } else {
            // Unparseable: 0xFF is never in the table, so this is a hit.
            out.push(0xFF);
        }
        rest = &after[semi..];
    }
    out
}

fn has_hex_literal(code: &str, literal: &str) -> bool {
    let bytes = code.as_bytes();
    let want = literal.to_ascii_lowercase();
    let lower = code.to_ascii_lowercase();
    let mut start = 0;
    while let Some(at) = lower[start..].find(&want) {
        let abs = start + at;
        let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        let end = abs + want.len();
        let after_ok =
            end >= bytes.len() || !(bytes[end].is_ascii_hexdigit() || bytes[end] == b'_');
        if before_ok && after_ok {
            return true;
        }
        start = end;
    }
    false
}

fn brace_import(code: &str, root: &str, name: &str) -> bool {
    let mut rest = code;
    let prefix = format!("{root}::{{");
    while let Some(at) = rest.find(&prefix) {
        let inner = &rest[at + prefix.len()..];
        let end = inner.find('}').unwrap_or(inner.len());
        if inner[..end]
            .split(',')
            .any(|part| part.trim() == name || part.trim().starts_with(&format!("{name}::")))
        {
            return true;
        }
        rest = &inner[end..];
    }
    false
}

fn has_ident(code: &str, name: &str) -> bool {
    let bytes = code.as_bytes();
    let mut start = 0;
    while let Some(at) = code[start..].find(name) {
        let abs = start + at;
        let before = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        let end = abs + name.len();
        let after = end >= bytes.len() || !is_ident_byte(bytes[end]);
        if before && after {
            return true;
        }
        start = end;
    }
    false
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Whitespace removed, so `std :: net` and line breaks cannot split a token.
fn squash(code: &str) -> String {
    code.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Comments removed. Returns the code (string contents kept in place) and
/// every string literal's contents.
fn split_source(src: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = src.chars().collect();
    let mut code = String::with_capacity(src.len());
    let mut literals = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let mut depth = 1;
            i += 2;
            while i < chars.len() && depth > 0 {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            code.push(' ');
            continue;
        }
        // Raw strings: r"..", r#".."#, br"..".
        let prev_ident = i > 0 && is_ident_byte(chars[i - 1] as u8);
        if !prev_ident && (c == 'r' || (c == 'b' && next == Some('r'))) {
            let mut j = i + if c == 'b' { 2 } else { 1 };
            let mut hashes = 0;
            while chars.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if chars.get(j) == Some(&'"') {
                let body_start = j + 1;
                let mut k = body_start;
                'find: while k < chars.len() {
                    if chars[k] == '"' {
                        let mut h = 0;
                        while h < hashes && chars.get(k + 1 + h) == Some(&'#') {
                            h += 1;
                        }
                        if h == hashes {
                            break 'find;
                        }
                    }
                    k += 1;
                }
                let body: String = chars[body_start..k.min(chars.len())].iter().collect();
                code.push('"');
                code.push_str(&body);
                code.push('"');
                literals.push(body);
                i = k + 1 + hashes;
                continue;
            }
        }
        if c == '"' {
            let mut body = String::new();
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    body.push(chars[i]);
                    body.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                body.push(chars[i]);
                i += 1;
            }
            i += 1;
            code.push('"');
            code.push_str(&body);
            code.push('"');
            literals.push(body);
            continue;
        }
        if c == '\'' {
            // Char literal: 'x', '\n', '\u{..}'. A lifetime has no closing quote.
            if next == Some('\\') {
                let mut j = i + 2;
                while j < chars.len() && chars[j] != '\'' {
                    j += 1;
                }
                code.push_str("' '");
                i = j + 1;
                continue;
            }
            if chars.get(i + 2) == Some(&'\'') {
                code.push_str("' '");
                i += 3;
                continue;
            }
        }
        code.push(c);
        i += 1;
    }
    (code, literals)
}

fn strip_comments(src: &str) -> String {
    split_source(src).0
}

fn rs_rels(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect(root, root, &mut out);
    out.sort();
    out
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(
                path.strip_prefix(root)
                    .expect("rel")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
}

fn scratch(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "llama-light-s15-{label}-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("scratch");
    path
}
