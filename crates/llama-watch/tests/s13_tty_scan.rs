//! S13: the watcher source may ask for the window size and nothing else.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn watcher_source_only_reads_the_window_size() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    assert!(
        files.iter().any(|path| path.ends_with("tty/term.rs")),
        "tty/term.rs is missing from the scan"
    );
    let term_src = fs::read_to_string(src.join("tty/term.rs")).expect("term.rs");
    assert!(
        term_src.contains("fn for_stdout"),
        "Term::for_stdout must live in term.rs"
    );

    let mut hits = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path).expect("read source");
        let code = strip_comments(&text);
        let rel = path
            .strip_prefix(&src)
            .unwrap_or(path)
            .display()
            .to_string();
        for token in ["ioctl", "TIOCSCTTY", "VT_", "stdin", "Stdin"] {
            if code.contains(token) {
                hits.push(format!("{rel} contains {token}"));
            }
        }
        if contains_kd_ioctl(&code) {
            hits.push(format!("{rel} contains a KD ioctl"));
        }
        let term = rel == "tty/term.rs";
        if !term {
            hits.extend(stdout_emitter_hits(&rel, &text));
        }
        // #26: the console palette (`ESC ] P n rrggbb`, `ESC ] R`) is the one
        // addition to the tty output since S13 was written. The bytes come
        // from `llama_core::palette` and only the emitter may write them; no
        // ioctl (PIO_CMAP) is used for it.
        if !term
            && (code.contains("console_load")
                || code.contains("CONSOLE_RESET")
                || code.contains("\\x1b]"))
        {
            hits.push(format!("{rel} emits a console palette sequence"));
        }
        if code.contains("termios") && !term {
            hits.push(format!("{rel} mentions termios"));
        }
        if code.contains("tcgetwinsize") && !term {
            hits.push(format!("{rel} calls tcgetwinsize"));
        }
        if term {
            if !code.contains("rustix::termios::tcgetwinsize") {
                hits.push("tty/term.rs does not call rustix::termios::tcgetwinsize".to_string());
            }
            let mut rest = code.as_str();
            while let Some(at) = rest.find("termios::") {
                let after = &rest[at + "termios::".len()..];
                let name: String = after
                    .chars()
                    .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                    .collect();
                if name != "tcgetwinsize" {
                    hits.push(format!("tty/term.rs uses termios::{name}"));
                }
                rest = &after[name.len()..];
            }
            for mac in [
                "write!",
                "writeln!",
                "print!",
                "println!",
                "eprint!",
                "eprintln!",
            ] {
                if code.contains(mac) {
                    hits.push(format!("tty/term.rs uses {mac}"));
                }
            }
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn scan_rejects_stdout_emitters_outside_term() {
    let cases = [
        ("src/main.rs", "fn main(){ print!(\"x\"); }\n"),
        ("src/main.rs", "fn main(){ println!(\"x\"); }\n"),
        ("src/main.rs", "fn main(){ let _ = std::io::stdout(); }\n"),
        ("src/main.rs", "fn main(){ let _ = io::stdout(); }\n"),
        ("src/main.rs", "fn main(){ write!(stdout, \"x\").ok(); }\n"),
        (
            "src/main.rs",
            "fn main(){ writeln!(io::stdout(), \"x\").ok(); }\n",
        ),
        (
            "src/main.rs",
            "use std::io::{stdout, Write};\nfn f(){ let _ = stdout(); }\n",
        ),
        ("src/main.rs", "fn f(out: Stdout) {}\n"),
        ("src/main.rs", "fn f(out: StdoutLock<'_>) {}\n"),
    ];
    for (rel, src) in cases {
        let hits = stdout_emitter_hits(rel, src);
        assert!(!hits.is_empty(), "missed stdout emitter: {src}");
    }
    assert!(
        stdout_emitter_hits(
            "src/log.rs",
            "fn f(){ eprint!(\"err\"); eprintln!(\"err\"); }\n"
        )
        .is_empty(),
        "stderr logging is not a stdout emitter"
    );
    assert!(
        stdout_emitter_hits("src/log.rs", "fn f(){ write!(buf, \"x\").ok(); }\n").is_empty(),
        "write! into a buffer is not stdout"
    );
}

#[test]
fn tty_handoff_marker_does_not_excuse_stdout() {
    let once =
        "// tty-handoff: T22 passes the inherited stdout to Term\nlet out = std::io::stdout();\n";
    assert!(
        !stdout_emitter_hits("src/main.rs", once).is_empty(),
        "stdout outside term.rs is rejected, marker or not"
    );
}

#[test]
fn kd_scan_matches_ioctl_names_only() {
    assert!(kd_ioctl_hits("fn background() { let packed = 1; }\n").is_empty());
    assert!(
        kd_ioctl_hits("let label = \"KD\";\n").is_empty(),
        "a bare KD is not an ioctl name"
    );
    assert!(!kd_ioctl_hits("let mode = KDSETMODE;\n").is_empty());
    assert!(!kd_ioctl_hits("let mode = KDGKBMODE;\n").is_empty());
    assert!(kd_ioctl_hits("// KDSETMODE stays in a comment\nfn ok() {}\n").is_empty());
}

fn stdout_emitter_hits(rel: &str, raw: &str) -> Vec<String> {
    if rel.ends_with("tty/term.rs") {
        return Vec::new();
    }
    let code = strip_comments(raw);
    let mut hits = Vec::new();
    if has_macro(&code, "print!") {
        hits.push(format!("{rel} uses print!"));
    }
    if has_macro(&code, "println!") {
        hits.push(format!("{rel} uses println!"));
    }
    for mac in ["write!", "writeln!", "eprint!", "eprintln!"] {
        if macro_targets_stdout(&code, mac) {
            hits.push(format!("{rel} uses {mac} on stdout"));
        }
    }
    if code.contains("io::stdout") || has_call(&code, "stdout") {
        hits.push(format!("{rel} calls stdout"));
    }
    if has_ident(&code, "StdoutLock") || has_ident(&code, "Stdout") {
        hits.push(format!("{rel} names Stdout"));
    }
    hits
}

fn has_call(code: &str, name: &str) -> bool {
    let mut start = 0;
    while let Some(at) = code[start..].find(name) {
        let abs = start + at;
        let before = abs == 0 || !is_ident_byte(code.as_bytes()[abs - 1]);
        if before {
            let after = code[abs + name.len()..].trim_start();
            if after.starts_with('(') {
                return true;
            }
        }
        start = abs + name.len();
    }
    false
}

fn has_ident(code: &str, name: &str) -> bool {
    let mut start = 0;
    while let Some(at) = code[start..].find(name) {
        let abs = start + at;
        let before = abs == 0 || !is_ident_byte(code.as_bytes()[abs - 1]);
        let end = abs + name.len();
        let after = end >= code.len() || !is_ident_byte(code.as_bytes()[end]);
        if before && after {
            return true;
        }
        start = abs + name.len();
    }
    false
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn kd_ioctl_hits(raw: &str) -> Vec<String> {
    let code = strip_comments(raw);
    if contains_kd_ioctl(&code) {
        vec!["KD ioctl".to_string()]
    } else {
        Vec::new()
    }
}

fn contains_kd_ioctl(code: &str) -> bool {
    let bytes = code.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if boundary && bytes[i] == b'K' && bytes[i + 1] == b'D' && bytes[i + 2].is_ascii_uppercase()
        {
            return true;
        }
        i += 1;
    }
    false
}

fn has_macro(code: &str, name: &str) -> bool {
    let mut start = 0;
    while let Some(at) = code[start..].find(name) {
        let abs = start + at;
        let boundary = abs == 0 || !code.as_bytes()[abs - 1].is_ascii_alphanumeric();
        if boundary {
            return true;
        }
        start = abs + name.len();
    }
    false
}

fn macro_targets_stdout(code: &str, name: &str) -> bool {
    let mut start = 0;
    while let Some(at) = code[start..].find(name) {
        let abs = start + at;
        let boundary = abs == 0 || !code.as_bytes()[abs - 1].is_ascii_alphanumeric();
        if !boundary {
            start = abs + name.len();
            continue;
        }
        let after = &code[abs + name.len()..];
        let end = after.find(';').unwrap_or(after.len());
        if first_arg_has_stdout(&after[..end]) {
            return true;
        }
        start = abs + name.len() + end;
    }
    false
}

fn first_arg_has_stdout(call: &str) -> bool {
    let Some(open) = call.find('(') else {
        return false;
    };
    let args = &call[open + 1..];
    let first = args.split(',').next().unwrap_or("");
    first.contains("stdout")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            while let Some(next) = chars.next() {
                if next == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    break;
                }
            }
            out.push(' ');
            continue;
        }
        if ch == '"' {
            out.push('"');
            while let Some(next) = chars.next() {
                out.push(next);
                if next == '\\' {
                    if let Some(escaped) = chars.next() {
                        out.push(escaped);
                    }
                    continue;
                }
                if next == '"' {
                    break;
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// #7: only the root pre-step (`tty/setup.rs`, `llama-watch tty-setup`)
/// starts programs, and only setfont and stty, never a shell. The watcher
/// (`run`) spawns nothing.
#[test]
fn only_tty_setup_spawns_and_only_setfont_and_stty() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    let mut hits = Vec::new();
    for path in &files {
        let full = strip_comments(&fs::read_to_string(path).expect("read source"));
        // Unit tests may spawn helpers (service.rs re-execs under taskset).
        // The test module sits at the end of each file.
        let code = full.split("#[cfg(test)]").next().unwrap_or_default();
        let rel = path
            .strip_prefix(&src)
            .unwrap_or(path)
            .display()
            .to_string();
        let spawns = [
            "process::Command",
            "Command::new",
            "execv",
            "posix_spawn",
            "fork",
        ]
        .iter()
        .any(|token| code.contains(token));
        if spawns && rel != "tty/setup.rs" {
            hits.push(format!("{rel} starts a program"));
        }
        if rel == "tty/setup.rs" {
            for shell in ["/bin/sh", "/usr/bin/sh", "bash", "\"sh\"", "\"-c\""] {
                if code.contains(shell) {
                    hits.push(format!("tty/setup.rs names {shell}"));
                }
            }
            let programs: Vec<&str> = code
                .match_indices("\"/")
                .map(|(at, _)| {
                    let rest = &code[at + 1..];
                    &rest[..rest.find('"').unwrap_or(rest.len())]
                })
                .collect();
            for program in &programs {
                let known = [
                    "/dev/tty11",
                    "/usr/local/share/llama-bored",
                    "/usr/bin/setfont",
                    "/usr/bin/stty",
                    "/",
                ];
                if !known.contains(program) {
                    hits.push(format!("tty/setup.rs names path {program}"));
                }
            }
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}
