//! S14 (T52): the fan source and the hwmon helpers it uses never write.
//!
//! `fanN_*` and `pwmN*` are writable on sysfs and drive the cooling. The
//! watcher only reads them. This fence greps the production code of
//! `sources/fans.rs` and `sources/hwmon.rs` (everything before a
//! `#[cfg(test)]`, comments stripped) for any write-capable file API.

use std::path::PathBuf;

/// Tokens that open a file for writing or change the filesystem.
const WRITE_TOKENS: [&str; 16] = [
    "write",
    "Write",
    "OpenOptions",
    "File::create",
    "create_new",
    "append(",
    "truncate(",
    "set_len",
    "set_permissions",
    "remove_file",
    "remove_dir",
    "rename(",
    "hard_link",
    "symlink",
    "create_dir",
    "copy(",
];

fn source(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// Production part only: cut at the first `#[cfg(test)]`, drop comments.
fn production(text: &str) -> String {
    let cut = text.find("#[cfg(test)]").unwrap_or(text.len());
    text[..cut]
        .lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn write_hits(rel: &str, text: &str) -> Vec<String> {
    let code = production(text);
    let mut hits = Vec::new();
    for token in WRITE_TOKENS {
        if code.contains(token) {
            hits.push(format!("{rel} uses {token}"));
        }
    }
    // `File::open` is read-only, but a `File` handle invites `write_all`
    // later. The fan source reads with `read_to_string` alone.
    if code.contains("File::open") || code.contains("fs::File") {
        hits.push(format!("{rel} opens a File handle"));
    }
    hits
}

#[test]
fn fan_and_hwmon_sources_have_no_write_path() {
    let mut hits = Vec::new();
    for rel in ["sources/fans.rs", "sources/hwmon.rs"] {
        hits.extend(write_hits(rel, &source(rel)));
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
    let fans = production(&source("sources/fans.rs"));
    assert!(
        fans.contains("std::fs::read_to_string"),
        "fans.rs must read with read_to_string"
    );
}

#[test]
fn fence_catches_each_write_shape() {
    let cases = [
        "fn f(p:&Path){ std::fs::write(p, \"0\").ok(); }",
        "fn f(p:&Path){ let _ = OpenOptions::new().write(true).open(p); }",
        "fn f(p:&Path){ let _ = std::fs::File::create(p); }",
        "fn f(p:&Path){ let mut f = std::fs::File::open(p).unwrap(); }",
        "fn f(p:&Path){ let _ = std::fs::set_permissions(p, perm); }",
        "fn f(p:&Path){ let _ = std::fs::rename(p, q); }",
    ];
    for case in cases {
        assert!(!write_hits("case.rs", case).is_empty(), "missed: {case}");
    }
    // Test fixtures after #[cfg(test)] and comments are not production.
    assert!(write_hits("ok.rs", "// std::fs::write\nfn f(){}\n").is_empty());
    assert!(
        write_hits(
            "ok.rs",
            "fn f(){}\n#[cfg(test)]\nmod t { fn g(){ std::fs::write(\"a\",\"b\"); } }"
        )
        .is_empty()
    );
}
