//! S1 and the S3 source scan. Plain text over every crate's `src/`, comments
//! removed. A hit is a forbidden token, a write outside the allowlist, or a
//! write API in a file that also names `/sys` or `/proc`.

use std::fs;
use std::path::{Path, PathBuf};

const WRITE_FILES: &[&str] = &[
    "crates/kraken-lcd/src/device/hid.rs",
    "crates/kraken-lcd/src/device/guard.rs",
    "crates/kraken-lcd/src/main.rs",
    // T57/T64: llama-light's one hidraw open-for-write, shared by the Aura
    // controller and the keyboard. S15 in crates/llama-light/tests fences
    // that crate further.
    "crates/llama-light/src/hidraw.rs",
    // T58: llama-metrics writes HTTP responses to its sockets. S16
    // (crates/llama-metrics/tests) forbids any file write in that crate.
    "crates/llama-metrics/src/http.rs",
    // llama-cast writes HTTP responses and the stream to its sockets, and
    // raw frames to ffmpeg's stdin pipe. S18 (crates/llama-cast/tests)
    // forbids any file write in that crate.
    "crates/llama-cast/src/http.rs",
    "crates/llama-cast/src/encoder.rs",
];
/// S16: llama-metrics and llama-cast are the only crates that may name a
/// listening or datagram socket, or bind one. Every other crate
/// (llama-watch, kraken-lcd, llama-view, llama-light, ...) keeps no
/// listening port. Inside each, its own scan (S16 for llama-metrics, S18
/// for llama-cast) scopes the sockets to reviewed files: llama-metrics one
/// TCP listener; llama-cast one TCP listener and one UDP socket on 1900.
const LISTENER_CRATES: &[&str] = &["crates/llama-metrics/", "crates/llama-cast/"];
const BULK_FILE: &str = "crates/kraken-lcd/src/device/bulk.rs";
const GPU_FILE: &str = "crates/llama-watch/src/sources/gpu.rs";

const NVML_CALLS: &[&str] = &[
    "as_ref",
    "builder",
    "default",
    "device",
    "device_by_index",
    "init",
    "lib_path",
    "enforced_power_limit",
    "map_err",
    "memory_info",
    "new",
    "power_usage",
    "temperature",
    "to_string",
    "utilization_rates",
];

const NVML_PATH: &[&str] = &[
    "nvml_wrapper",
    "enum_wrappers",
    "device",
    "TemperatureSensor",
    "Device",
    "Nvml",
    "error",
    "NvmlError",
];

#[test]
fn src_tree_passes_s1_and_the_s3_scan() {
    let rels = src_rels();
    assert!(!rels.is_empty(), "safety scan saw no Rust sources");
    for required in [
        "crates/kraken-lcd/src/device/hid.rs",
        "crates/kraken-lcd/src/device/bulk.rs",
        "crates/llama-watch/src/sources/gpu.rs",
        "crates/llama-metrics/src/http.rs",
        "crates/llama-metrics/src/service.rs",
        "crates/llama-cast/src/http.rs",
        "crates/llama-cast/src/service.rs",
        "crates/llama-cast/src/discovery.rs",
        "crates/llama-cast/src/encoder.rs",
        // S16 covers the RGB writer too: it must keep no listening socket.
        "crates/llama-light/src/service.rs",
        "crates/llama-light/src/aura/device.rs",
        "crates/llama-light/src/hidraw.rs",
        "crates/llama-light/src/keyboard/device.rs",
    ] {
        assert!(
            rels.iter().any(|rel| rel == required),
            "safety scan file set is missing {required}"
        );
    }
    let hits = scan_tree();
    assert!(hits.is_empty(), "{}", hits.join("\n"));
    let layout = layout_hits(&workspace_root());
    assert!(layout.is_empty(), "{}", layout.join("\n"));
}

#[test]
fn scanner_flags_forbidden_tokens_and_sys_writes() {
    let nusb = scan_source(
        "crates/kraken-lcd/src/device/mod.rs",
        "let _ = nusb::list_devices();\n",
    );
    assert!(nusb.iter().any(|hit| hit.contains("nusb")), "{nusb:?}");
    let detach = scan_source(
        "crates/kraken-lcd/src/device/bulk.rs",
        "dev.detach_and_claim_interface(0);\n",
    );
    assert!(
        detach
            .iter()
            .any(|hit| hit.contains("detach_and_claim_interface")),
        "{detach:?}"
    );
    let sys = scan_source(
        "crates/kraken-lcd/src/device/guard.rs",
        "std::fs::write(\"/sys/class/hwmon/hwmon0/pwm1\", \"1\");\n",
    );
    assert!(sys.iter().any(|hit| hit.contains("S3")), "{sys:?}");
    let allowed = scan_source(
        "crates/kraken-lcd/src/device/hid.rs",
        "file.write(report.as_bytes())?;\n",
    );
    assert!(allowed.is_empty(), "{allowed:?}");
}

#[test]
fn s16_only_llama_metrics_and_llama_cast_may_listen() {
    for (rel, planted) in [
        (
            "crates/llama-watch/src/service.rs",
            "let l = std::net::TcpListener::bind(a)?;\n",
        ),
        (
            "crates/kraken-lcd/src/main.rs",
            "let s = UdpSocket::bind(a);\n",
        ),
        (
            "crates/llama-view/src/main.rs",
            "let l = UnixListener::bind(p);\n",
        ),
        (
            "crates/llama-light/src/service.rs",
            "rustix::net::listen(&fd, 1)?;\n",
        ),
        (
            "crates/llama-metrics-lookalike/src/lib.rs",
            "let l = TcpListener::bind(a);\n",
        ),
        (
            "crates/llama-cast-lookalike/src/lib.rs",
            "let s = UdpSocket::bind(a);\n",
        ),
    ] {
        let hits = scan_source(rel, planted);
        assert!(
            hits.iter().any(|hit| hit.contains("S16")),
            "{rel}: planted listener was allowed: {hits:?}"
        );
    }
    for rel in [
        "crates/llama-metrics/src/http.rs",
        "crates/llama-metrics/src/service.rs",
        "crates/llama-cast/src/service.rs",
    ] {
        let hits = scan_source(rel, "let l = TcpListener::bind(a)?;\n");
        assert!(hits.iter().all(|hit| !hit.contains("S16")), "{hits:?}");
    }
    // `binding(` and a comment are not a bind.
    assert!(
        scan_source(
            "crates/kraken-lcd/src/device/mod.rs",
            "binding(root, &node);\n// listen here\n"
        )
        .is_empty()
    );
}

#[test]
fn discovery_lists_every_directory_under_crates() {
    let root = scratch_workspace("discover");
    write_crate(&root, "alpha");
    write_crate(&root, "beta");
    std::fs::write(root.join("crates/alpha/src/lib.rs"), "fn alpha() {}\n").unwrap();
    std::fs::write(root.join("crates/beta/src/lib.rs"), "fn beta() {}\n").unwrap();
    assert_eq!(
        discovered_crate_names(&root.join("crates")),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    let rels = src_rels_under(&root);
    assert!(
        rels.iter().any(|rel| rel == "crates/alpha/src/lib.rs"),
        "{rels:?}"
    );
    assert!(
        rels.iter().any(|rel| rel == "crates/beta/src/lib.rs"),
        "{rels:?}"
    );
    assert!(layout_hits(&root).is_empty(), "{:?}", layout_hits(&root));
}

#[test]
fn layout_rejects_a_crate_without_workspace_lints() {
    let root = scratch_workspace("no-lints");
    write_crate(&root, "good");
    let bad = root.join("crates/bad");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("Cargo.toml"), "[package]\nname = \"bad\"\n").unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter()
            .any(|hit| hit.contains("crates/bad") && hit.contains("[lints]")),
        "{hits:?}"
    );
}

#[test]
fn layout_rejects_workspace_lints_set_false() {
    let root = scratch_workspace("lints-false");
    let dir = root.join("crates/half");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"half\"\n\n[lints]\nworkspace = false\n",
    )
    .unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter()
            .any(|hit| hit.contains("crates/half") && hit.contains("[lints]")),
        "{hits:?}"
    );
}

#[test]
fn layout_rejects_a_package_build_key() {
    let root = scratch_workspace("package-build");
    write_crate(&root, "good");
    let dir = root.join("crates/scripted");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"scripted\"\nbuild = \"custom-build.rs\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter().any(|hit| hit.contains("package.build")),
        "{hits:?}"
    );
}

#[test]
fn layout_rejects_package_build_set_false() {
    let root = scratch_workspace("package-build-false");
    let dir = root.join("crates/off");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"off\"\nbuild = false\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter().any(|hit| hit.contains("package.build")),
        "{hits:?}"
    );
}

#[test]
fn layout_rejects_a_package_build_table() {
    let root = scratch_workspace("package-build-table");
    let dir = root.join("crates/tabled");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"tabled\"\n\n[package.build]\nscript = \"x.rs\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter().any(|hit| hit.contains("package.build")),
        "{hits:?}"
    );
}

#[test]
fn layout_allows_a_build_dependencies_table() {
    let root = scratch_workspace("build-deps");
    let dir = root.join("crates/linked");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"linked\"\n\n[lints]\nworkspace = true\n\n[build-dependencies]\ncc = \"1\"\n",
    )
    .unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter().all(|hit| !hit.contains("package.build")),
        "{hits:?}"
    );
}

#[test]
fn layout_rejects_build_rs_anywhere_outside_build_output() {
    let root = scratch_workspace("build-rs");
    write_crate(&root, "good");
    std::fs::create_dir_all(root.join("scripts/nested")).unwrap();
    std::fs::write(root.join("build.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("scripts/nested/build.rs"), "fn main() {}\n").unwrap();
    std::fs::create_dir_all(root.join("target/debug")).unwrap();
    std::fs::write(root.join("target/debug/build.rs"), "fn main() {}\n").unwrap();
    let hits = layout_hits(&root);
    assert!(
        hits.iter()
            .any(|hit| hit.contains("build.rs") && !hit.contains("target/")),
        "{hits:?}"
    );
    assert!(
        hits.iter()
            .any(|hit| hit.contains("scripts/nested/build.rs")),
        "{hits:?}"
    );
    assert!(hits.iter().all(|hit| !hit.contains("target/")), "{hits:?}");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn discovered_crate_names(crates_dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(crates_dir).expect("crates/") {
        let entry = entry.expect("dir entry");
        if entry.path().is_dir()
            && let Some(name) = entry.file_name().to_str()
        {
            names.push(name.to_owned());
        }
    }
    names.sort();
    names
}

fn src_rels_under(root: &Path) -> Vec<String> {
    let mut paths = Vec::new();
    for name in discovered_crate_names(&root.join("crates")) {
        let src = root.join("crates").join(&name).join("src");
        if src.is_dir() {
            collect_rs(&src, &mut paths);
        }
    }
    let mut rels: Vec<String> = paths
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .expect("src file under workspace")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    rels.sort();
    rels
}

fn layout_hits(root: &Path) -> Vec<String> {
    let mut hits = Vec::new();
    let crates_dir = root.join("crates");
    if !crates_dir.is_dir() {
        hits.push("crates/: missing crate directory".to_owned());
    } else {
        for name in discovered_crate_names(&crates_dir) {
            let manifest = crates_dir.join(&name).join("Cargo.toml");
            match fs::read_to_string(&manifest) {
                Err(err) => hits.push(format!("crates/{name}/Cargo.toml: {err}")),
                Ok(text) if !lints_workspace_true(&text) => hits.push(format!(
                    "crates/{name}/Cargo.toml: missing [lints] workspace = true"
                )),
                Ok(text) if package_build_key(&text) => hits.push(format!(
                    "crates/{name}/Cargo.toml: package.build is not allowed"
                )),
                Ok(_) => {}
            }
        }
    }
    hits.extend(build_rs_hits(root));
    hits.sort();
    hits
}

/// `package.build`, including `build = false` and a `[package.build]` table.
/// `[build-dependencies]` is a different table and is not a build script key.
fn package_build_key(toml: &str) -> bool {
    let mut in_package = false;
    for raw in toml.lines() {
        let line = strip_toml_comment(raw).trim().to_owned();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let header = line.trim_matches(['[', ']']).trim();
            if header == "package.build" || header.starts_with("package.build.") {
                return true;
            }
            in_package = header == "package";
            continue;
        }
        if in_package && line.split('=').next().unwrap_or("").trim() == "build" {
            return true;
        }
    }
    false
}

/// `[lints]` table contains `workspace = true`. A later table ends the search.
fn lints_workspace_true(toml: &str) -> bool {
    let mut in_lints = false;
    for raw in toml.lines() {
        let line = strip_toml_comment(raw).trim().to_owned();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            if in_lints {
                return false;
            }
            in_lints = line == "[lints]";
            continue;
        }
        if in_lints && line.replace(' ', "") == "workspace=true" {
            return true;
        }
    }
    false
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_string = false;
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => in_string = !in_string,
            b'#' if !in_string => return &line[..index],
            _ => {}
        }
        index += 1;
    }
    line
}

fn build_rs_hits(root: &Path) -> Vec<String> {
    let mut hits = Vec::new();
    collect_build_rs(root, root, &mut hits);
    hits
}

fn collect_build_rs(root: &Path, dir: &Path, hits: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            // Cargo output and the git dir are not the workspace source.
            if name == "target" || name == ".git" {
                continue;
            }
            collect_build_rs(root, &path, hits);
            continue;
        }
        if name == "build.rs" {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            hits.push(format!("{rel}: build.rs is not allowed"));
        }
    }
}

fn scratch_workspace(label: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("safety-layout-{label}"));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("crates")).unwrap();
    root
}

fn write_crate(root: &Path, name: &str) {
    let dir = root.join("crates").join(name);
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        format!("[package]\nname = \"{name}\"\n\n[lints]\nworkspace = true\n"),
    )
    .unwrap();
}

fn src_rels() -> Vec<String> {
    src_rels_under(&workspace_root())
}

fn scan_tree() -> Vec<String> {
    let root = workspace_root();
    let mut hits = Vec::new();
    for rel in src_rels() {
        let text = fs::read_to_string(root.join(&rel)).expect("read src");
        hits.extend(scan_source(&rel, &text));
        if rel != GPU_FILE && text.contains("nvml_wrapper") {
            hits.push(format!("{rel}: nvml_wrapper is only allowed in {GPU_FILE}"));
        }
    }
    hits.extend(scan_gpu(
        &fs::read_to_string(root.join(GPU_FILE)).expect("gpu.rs"),
    ));
    hits
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

fn scan_source(rel: &str, text: &str) -> Vec<String> {
    // `#[cfg(test)]` modules build fake trees. The scan covers the service code.
    let code = strip_comments(&strip_cfg_test(text));
    let mut hits = Vec::new();
    for token in [
        "detach_and_claim_interface",
        "set_configuration",
        "control_in",
        "control_out",
        "set_alt_setting",
        "libc::",
        ".reset(",
    ] {
        if code.contains(token) {
            hits.push(format!("{rel}: forbidden token {token}"));
        }
    }
    if identifiers(&code).iter().any(|ident| ident == "unsafe") {
        hits.push(format!("{rel}: forbidden token unsafe"));
    }
    if identifiers(&code).iter().any(|ident| ident == "ioctl") {
        hits.push(format!("{rel}: forbidden token ioctl"));
    }
    if rel != BULK_FILE && code.contains("nusb") {
        hits.push(format!("{rel}: nusb is only allowed in {BULK_FILE}"));
    }
    if !LISTENER_CRATES.iter().any(|c| rel.starts_with(c)) {
        hits.extend(listener_hits(rel, &code));
    }
    let writes = write_apis(&code);
    if writes && !WRITE_FILES.contains(&rel) {
        hits.push(format!(
            "{rel}: write API outside hid.rs, guard.rs, main.rs, llama-light hidraw.rs, llama-metrics http.rs, and llama-cast http.rs/encoder.rs"
        ));
    }
    if writes && (code.contains("\"/sys") || code.contains("\"/proc")) {
        hits.push(format!(
            "{rel}: S3 write API in a file that names /sys or /proc"
        ));
    }
    // Test modules may build scratch trees. A write aimed at /sys or /proc
    // is still rejected, including inside `#[cfg(test)]`.
    if s3_path_write(&strip_comments(text)) {
        hits.push(format!("{rel}: S3 write API next to a /sys or /proc path"));
    }
    hits
}

/// S16: a listening socket, a datagram socket, or a bind.
fn listener_hits(rel: &str, code: &str) -> Vec<String> {
    let mut hits = Vec::new();
    let idents = identifiers(code);
    for token in [
        "TcpListener",
        "UdpSocket",
        "UnixListener",
        "UnixDatagram",
        "bind",
        "listen",
        "SocketAddrV4",
        "SocketAddrV6",
    ] {
        if idents.iter().any(|ident| ident == token) {
            hits.push(format!("{rel}: S16 listening-socket token {token}"));
        }
    }
    for token in [
        "rustix::net",
        "socket2",
        "unix::net",
        "tokio",
        "hyper",
        "tiny_http",
    ] {
        if code.contains(token) {
            hits.push(format!("{rel}: S16 listening-socket token {token}"));
        }
    }
    hits
}

fn s3_path_write(code: &str) -> bool {
    let lines: Vec<&str> = code.lines().collect();
    for index in 0..lines.len() {
        let start = index.saturating_sub(2);
        let end = (index + 3).min(lines.len());
        let window = lines[start..end].join(" ");
        if write_apis(&window) && (window.contains("\"/sys") || window.contains("\"/proc")) {
            return true;
        }
    }
    false
}

fn write_apis(code: &str) -> bool {
    [
        ".write(",
        ".write_all(",
        ".append(true)",
        ".create(true)",
        ".truncate(true)",
        "std::fs::write",
        "fs::write(",
        "File::create",
    ]
    .iter()
    .any(|token| code.contains(token))
}

fn scan_gpu(text: &str) -> Vec<String> {
    let production = text
        .split_once("#[cfg(test)]")
        .map(|(head, _)| head)
        .unwrap_or(text);
    let code = strip_comments(production);
    let mut hits = Vec::new();
    for path in nvml_paths(&code) {
        for segment in path.split("::") {
            if !segment.is_empty() && !NVML_PATH.contains(&segment) {
                hits.push(format!("{GPU_FILE}: NVML path segment {segment}"));
            }
        }
    }
    let blocks = impl_blocks(
        &code,
        &[
            "impl GpuBackend for NvidiaGpu",
            "impl NvidiaGpu",
            "impl From<nvml_wrapper::error::NvmlError> for GpuError",
        ],
    );
    if blocks.len() != 3 {
        hits.push(format!(
            "{GPU_FILE}: expected 3 NVML impl blocks, found {}",
            blocks.len()
        ));
    }
    let mut calls = Vec::new();
    for block in &blocks {
        calls.extend(calls_in(block));
    }
    for call in &calls {
        if !NVML_CALLS.contains(&call.as_str()) {
            hits.push(format!(
                "{GPU_FILE}: NVML call {call} is outside the allowlist"
            ));
        }
    }
    for required in [
        "device_by_index",
        "utilization_rates",
        "temperature",
        "builder",
        "memory_info",
        "power_usage",
        "enforced_power_limit",
    ] {
        if !calls.iter().any(|call| call == required) {
            hits.push(format!("{GPU_FILE}: missing NVML call {required}"));
        }
    }
    hits
}

fn impl_blocks(code: &str, headers: &[&str]) -> Vec<String> {
    let mut blocks = Vec::new();
    for header in headers {
        let Some(start) = code.find(header) else {
            continue;
        };
        let Some(brace) = code[start..].find('{') else {
            continue;
        };
        let body_at = start + brace;
        let Some(end) = matching_brace(code, body_at) else {
            continue;
        };
        blocks.push(code[start..=end].to_owned());
    }
    blocks
}

fn matching_brace(code: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (offset, ch) in code[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn calls_in(code: &str) -> Vec<String> {
    let chars: Vec<char> = code.chars().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let call = if chars[index] == '.' {
            Some(index + 1)
        } else if chars[index] == ':' && index + 1 < chars.len() && chars[index + 1] == ':' {
            Some(index + 2)
        } else {
            None
        };
        if let Some(start) = call {
            let mut end = start;
            if end < chars.len() && is_ident_start(chars[end]) {
                end += 1;
                while end < chars.len() && is_ident_continue(chars[end]) {
                    end += 1;
                }
                if end < chars.len() && chars[end] == '(' {
                    out.push(chars[start..end].iter().collect());
                }
            }
        }
        index += 1;
    }
    out
}

fn nvml_paths(code: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut rest = code;
    while let Some(at) = rest.find("nvml_wrapper") {
        let tail = &rest[at..];
        let mut end = 0;
        let chars: Vec<char> = tail.chars().collect();
        while end < chars.len() {
            let ch = chars[end];
            if is_ident_continue(ch) || ch == ':' {
                end += 1;
            } else {
                break;
            }
        }
        paths.push(chars[..end].iter().collect());
        rest = &tail[end..];
    }
    paths
}

fn identifiers(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in code.chars() {
        if is_ident_continue(ch) && (is_ident_start(ch) || !current.is_empty()) {
            current.push(ch);
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn strip_cfg_test(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    const MARK: &str = "#[cfg(test)]";
    while let Some(at) = rest.find(MARK) {
        out.push_str(&rest[..at]);
        let after = &rest[at + MARK.len()..];
        let trimmed = after.trim_start();
        if let Some(end) = cfg_test_item_end(trimmed) {
            rest = &trimmed[end..];
            continue;
        }
        out.push_str(MARK);
        rest = after;
    }
    out.push_str(rest);
    out
}

fn cfg_test_item_end(trimmed: &str) -> Option<usize> {
    let item = trimmed.trim_start_matches(|ch: char| ch == '#' || ch.is_whitespace());
    let skipped = trimmed.len() - item.len();
    if item.starts_with("use ") {
        let end = item.find(';')?;
        return Some(skipped + end + 1);
    }
    if !(item.starts_with("mod ")
        || item.starts_with("fn ")
        || item.starts_with("impl ")
        || item.starts_with("struct ")
        || item.starts_with("enum ")
        || item.starts_with("const ")
        || item.starts_with("static "))
    {
        return None;
    }
    if let Some(brace) = item.find('{') {
        let end = matching_brace(item, brace)?;
        return Some(skipped + end + 1);
    }
    let end = item.find(';')?;
    Some(skipped + end + 1)
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
