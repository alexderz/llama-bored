//! S11: the writer source reads the snapshot constant and Kraken sysfs only.
//!
//! Comments are stripped before the scan. Each check is also run against a
//! scratch tree that plants one violation, so a weakened scanner fails here.
//!
//! `Path::read_dir`, `metadata`, `symlink_metadata`, and `exists` are method
//! calls. This text scan does not see them. The unit's kernel sandbox
//! (`ProtectSystem`, `ProcSubset`, `PrivateNetwork`) is the backstop.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Files that may touch the filesystem, relative to the crate `src/`.
///
/// `config.rs` reads the root-owned config file. The path is the CLI argument,
/// not a key in the file. `service.rs` is the latch and the read-only `/sys`
/// probe. Device files are the Kraken sysfs and the LCD open.
/// Public functions in these files are fixed names under a root parameter.
const FS_FILES: &[&str] = &[
    "device/sysfs.rs",
    "device/guard.rs",
    "device/hid.rs",
    "snapshot_reader.rs",
    "service.rs",
    "main.rs",
    "config.rs",
];

/// Reviewed public items of the filesystem-allowlisted files.
/// `(file, kind, name)`. Kinds include `impl` (`Trait for Type`),
/// `trait_fn` (`Trait::fn`), and `field` (`Struct.field`).
/// A new row is a review.
///
/// `fn` rows are fixed names under a root parameter.
/// Only an exception row carries its own comment.
const PUBLIC_SURFACE: &[(&str, &str, &str)] = &[
    ("config.rs", "enum", "ConfigError"),
    ("config.rs", "enum", "InvalidConfig"),
    ("config.rs", "enum", "UploadMode"),
    ("config.rs", "enum", "Variant"),
    ("config.rs", "field", "Bands.enter"),
    ("config.rs", "field", "Bands.fill_min_coverage"),
    ("config.rs", "field", "Bands.margin"),
    ("config.rs", "field", "Config.bands"),
    ("config.rs", "field", "Config.dial"),
    ("config.rs", "field", "Config.display"),
    ("config.rs", "field", "Config.hysteresis"),
    ("config.rs", "field", "Config.snapshot"),
    ("config.rs", "field", "Config.upload"),
    ("config.rs", "field", "Config.writer"),
    ("config.rs", "field", "Dial.ceiling_tps"),
    ("config.rs", "field", "Dial.max_gap_s"),
    ("config.rs", "field", "Dial.tiers"),
    ("config.rs", "field", "Dial.xff"),
    ("config.rs", "field", "DisplayCfg.rotate_deg"),
    ("config.rs", "field", "DisplayCfg.variant"),
    ("config.rs", "field", "Hysteresis.percent"),
    ("config.rs", "field", "Hysteresis.ring"),
    ("config.rs", "field", "Hysteresis.temp"),
    ("config.rs", "field", "Snapshot.stale_after_s"),
    ("config.rs", "field", "Snapshot.watch_down_restore_min_s"),
    ("config.rs", "field", "Snapshot.watch_down_stock_after_s"),
    ("config.rs", "field", "StepMargin.margin"),
    ("config.rs", "field", "StepMargin.step"),
    ("config.rs", "field", "Tier.bars"),
    ("config.rs", "field", "Tier.width_s"),
    ("config.rs", "field", "Upload.fail_limit"),
    ("config.rs", "field", "Upload.min_interval_s"),
    ("config.rs", "field", "Upload.mode"),
    ("config.rs", "field", "Upload.stream_fps"),
    ("config.rs", "field", "Writer.tick_s"),
    ("config.rs", "fn", "Config::from_toml"),
    // exception: CLI path; call sites are already fenced
    ("config.rs", "fn", "Config::load_validated"),
    ("config.rs", "fn", "Config::validate"),
    ("config.rs", "fn", "band_margin"),
    ("config.rs", "fn", "ceiling_tps"),
    ("config.rs", "fn", "enter"),
    ("config.rs", "fn", "fail_limit"),
    ("config.rs", "fn", "fill_min_coverage"),
    ("config.rs", "fn", "max_gap_s"),
    ("config.rs", "fn", "min_interval_s"),
    ("config.rs", "fn", "percent"),
    ("config.rs", "fn", "ring"),
    ("config.rs", "fn", "rotate_deg"),
    ("config.rs", "fn", "stale_after_s"),
    ("config.rs", "fn", "stream_fps"),
    ("config.rs", "fn", "temp"),
    ("config.rs", "fn", "tick_s"),
    ("config.rs", "fn", "tiers"),
    ("config.rs", "fn", "watch_down_restore_min_s"),
    ("config.rs", "fn", "watch_down_stock_after_s"),
    ("config.rs", "fn", "xff"),
    ("config.rs", "impl", "Default for Bands"),
    ("config.rs", "impl", "Default for Dial"),
    ("config.rs", "impl", "Default for DisplayCfg"),
    ("config.rs", "impl", "Default for Hysteresis"),
    ("config.rs", "impl", "Default for Snapshot"),
    ("config.rs", "impl", "Default for Upload"),
    ("config.rs", "impl", "Default for Writer"),
    ("config.rs", "impl", "Deref for ValidConfig"),
    ("config.rs", "impl", "Deserialize for FlexF64"),
    ("config.rs", "impl", "Deserialize for StepMargin"),
    ("config.rs", "impl", "Deserialize for Tier"),
    ("config.rs", "impl", "Display for ParseSummary"),
    ("config.rs", "impl", "Visitor for Visit"),
    ("config.rs", "struct", "Bands"),
    ("config.rs", "struct", "Config"),
    ("config.rs", "struct", "Dial"),
    ("config.rs", "struct", "DisplayCfg"),
    ("config.rs", "struct", "Hysteresis"),
    ("config.rs", "struct", "ParseSummary"),
    ("config.rs", "struct", "Snapshot"),
    ("config.rs", "struct", "StepMargin"),
    ("config.rs", "struct", "Tier"),
    ("config.rs", "struct", "Upload"),
    ("config.rs", "struct", "ValidConfig"),
    ("config.rs", "struct", "Writer"),
    ("device/guard.rs", "const", "FOLLOW_UP_AFTER"),
    ("device/guard.rs", "const", "PUMP_BAND_FLOOR_RPM"),
    ("device/guard.rs", "const", "PUMP_BAND_PERCENT"),
    ("device/guard.rs", "enum", "Deviation"),
    ("device/guard.rs", "field", "CoolingSnapshot.bootloader"),
    ("device/guard.rs", "field", "CoolingSnapshot.devnum"),
    ("device/guard.rs", "field", "CoolingSnapshot.fan1_rpm"),
    ("device/guard.rs", "field", "CoolingSnapshot.pwm1"),
    ("device/guard.rs", "field", "CoolingSnapshot.pwm1_enable"),
    ("device/guard.rs", "field", "CoolingSnapshot.pwm2"),
    ("device/guard.rs", "field", "CoolingSnapshot.pwm2_enable"),
    ("device/guard.rs", "field", "CoolingSnapshot.z53"),
    ("device/guard.rs", "fn", "CoolingGuard::capture"),
    ("device/guard.rs", "fn", "CoolingGuard::check"),
    ("device/guard.rs", "fn", "CoolingGuard::latch_present"),
    ("device/guard.rs", "fn", "CoolingGuard::new"),
    ("device/guard.rs", "fn", "CoolingGuard::read"),
    ("device/guard.rs", "fn", "CoolingGuard::write_latch"),
    ("device/guard.rs", "fn", "find_z53"),
    ("device/guard.rs", "fn", "follow_up_due"),
    ("device/guard.rs", "fn", "format_halt"),
    ("device/guard.rs", "fn", "log_halt"),
    ("device/guard.rs", "fn", "pump_band_rpm"),
    // exception: W_OK probe for show-image pre-flight; path is the state dir.
    // Call sites are fenced to device/ (state_dir_writable_hits).
    ("device/guard.rs", "fn", "state_dir_writable"),
    ("device/guard.rs", "struct", "CoolingGuard"),
    ("device/guard.rs", "struct", "CoolingSnapshot"),
    ("device/hid.rs", "const", "REPLY_BUDGET"),
    ("device/hid.rs", "const", "REPLY_LIMIT"),
    ("device/hid.rs", "enum", "ExchangeError"),
    // exception: pub(in crate::device). Call sites are fenced to device/.
    // The path is the hidraw node pre_open resolved.
    ("device/hid.rs", "fn", "HidLink::open"),
    ("device/hid.rs", "fn", "transact"),
    ("device/hid.rs", "impl", "Display for ExchangeError"),
    ("device/hid.rs", "impl", "Error for ExchangeError"),
    ("device/hid.rs", "impl", "From for ExchangeError"),
    ("device/hid.rs", "impl", "HidPort for HidLink"),
    ("device/hid.rs", "struct", "HidLink"),
    ("device/hid.rs", "trait", "HidPort"),
    ("device/hid.rs", "trait_fn", "HidPort::drain"),
    ("device/hid.rs", "trait_fn", "HidPort::read_report"),
    ("device/hid.rs", "trait_fn", "HidPort::send"),
    ("device/sysfs.rs", "fn", "binding"),
    ("device/sysfs.rs", "fn", "bootloader_present_at"),
    ("device/sysfs.rs", "fn", "interface0_usbfs"),
    ("device/sysfs.rs", "fn", "post_open"),
    ("device/sysfs.rs", "fn", "pre_open"),
    ("device/sysfs.rs", "fn", "read_coolant_c"),
    ("device/sysfs.rs", "field", "KrakenNode.busnum"),
    ("device/sysfs.rs", "field", "KrakenNode.device_dir"),
    ("device/sysfs.rs", "field", "KrakenNode.devnum"),
    ("device/sysfs.rs", "field", "KrakenNode.hidraw"),
    ("device/sysfs.rs", "field", "KrakenNode.name"),
    ("device/sysfs.rs", "struct", "KrakenNode"),
    ("service.rs", "const", "BACKOFF_CAP"),
    ("service.rs", "const", "BACKOFF_START"),
    ("service.rs", "const", "WALL_JUMP_TICKS"),
    ("service.rs", "enum", "CheckError"),
    ("service.rs", "enum", "LoopExit"),
    ("service.rs", "field", "CheckEnv.euid_is_root"),
    ("service.rs", "field", "CheckEnv.state_dir"),
    ("service.rs", "field", "CheckEnv.state_dir_writable"),
    ("service.rs", "field", "CheckEnv.sys_is_readonly"),
    ("service.rs", "field", "CheckEnv.sys_root"),
    ("service.rs", "field", "CheckEnv.z53_present"),
    ("service.rs", "field", "LoopInput.assets"),
    ("service.rs", "field", "LoopInput.clock"),
    ("service.rs", "field", "LoopInput.config"),
    ("service.rs", "field", "LoopInput.latch_at_start"),
    ("service.rs", "field", "LoopInput.log"),
    ("service.rs", "field", "LoopInput.notify"),
    ("service.rs", "field", "LoopInput.open"),
    ("service.rs", "field", "LoopInput.sampler"),
    ("service.rs", "field", "LoopInput.stop"),
    ("service.rs", "fn", "CheckEnv::production"),
    ("service.rs", "fn", "LoopExit::code"),
    ("service.rs", "fn", "halt_latch_path"),
    ("service.rs", "fn", "is_wall_resume"),
    ("service.rs", "fn", "latch_present"),
    ("service.rs", "fn", "next_backoff"),
    ("service.rs", "fn", "restore_stock"),
    ("service.rs", "fn", "restore_stock_on"),
    ("service.rs", "fn", "run"),
    ("service.rs", "fn", "run_loop"),
    ("service.rs", "fn", "self_check"),
    ("service.rs", "fn", "tick_duration"),
    ("service.rs", "fn", "wall_delta"),
    ("service.rs", "fn", "wall_jump_limit"),
    ("service.rs", "fn", "z53_exists"),
    ("service.rs", "impl", "Clock for &mut C"),
    ("service.rs", "impl", "Clock for RealClock"),
    ("service.rs", "impl", "Notifier for &mut N"),
    ("service.rs", "impl", "Notifier for SdNotify"),
    ("service.rs", "impl", "Sampler for &mut S"),
    ("service.rs", "impl", "Stop for &S"),
    ("service.rs", "impl", "Stop for NeverStop"),
    ("service.rs", "struct", "CheckEnv"),
    ("service.rs", "struct", "LoopInput"),
    ("service.rs", "struct", "NeverStop"),
    ("service.rs", "struct", "RealClock"),
    ("service.rs", "struct", "SdNotify"),
    ("service.rs", "trait", "Clock"),
    ("service.rs", "trait", "Notifier"),
    ("service.rs", "trait", "Sampler"),
    ("service.rs", "trait", "Stop"),
    ("service.rs", "trait_fn", "Clock::mono"),
    ("service.rs", "trait_fn", "Clock::sleep"),
    ("service.rs", "trait_fn", "Clock::wall"),
    ("service.rs", "trait_fn", "Notifier::ready"),
    ("service.rs", "trait_fn", "Notifier::stopping"),
    ("service.rs", "trait_fn", "Notifier::watchdog"),
    ("service.rs", "trait_fn", "Sampler::sample"),
    ("service.rs", "trait_fn", "Stop::requested"),
    ("snapshot_reader.rs", "fn", "ManualMono::new"),
    ("snapshot_reader.rs", "fn", "SnapshotReader::new"),
    ("snapshot_reader.rs", "fn", "SnapshotReader::open"),
    ("snapshot_reader.rs", "fn", "SnapshotReader::set_now"),
    ("snapshot_reader.rs", "struct", "HostMono"),
    ("snapshot_reader.rs", "struct", "ManualMono"),
    ("snapshot_reader.rs", "struct", "SnapshotReader"),
    ("snapshot_reader.rs", "trait", "MonoNow"),
    ("snapshot_reader.rs", "impl", "MonoNow for HostMono"),
    ("snapshot_reader.rs", "impl", "MonoNow for ManualMono"),
    (
        "snapshot_reader.rs",
        "impl",
        "Sampler for SnapshotReader<L, C>",
    ),
    ("snapshot_reader.rs", "trait_fn", "MonoNow::now_ns"),
];

#[test]
fn planted_std_net_is_rejected() {
    let root = scratch("std-net");
    fs::write(root.join("net.rs"), "use std::net::TcpStream;\n").unwrap();
    assert_hit(&root, "std::net");
}

#[test]
fn planted_std_net_split_by_a_block_comment_is_rejected() {
    let root = scratch("std-net-comment");
    fs::write(root.join("net.rs"), "use std::/**/net::TcpStream;\n").unwrap();
    assert_hit(&root, "std::net");
}

#[test]
fn planted_tcp_stream_is_rejected() {
    let root = scratch("tcp");
    fs::write(
        root.join("net.rs"),
        "fn open() -> TcpStream { unreachable!() }\n",
    )
    .unwrap();
    assert_hit(&root, "TcpStream");
}

#[test]
fn planted_udp_socket_is_rejected() {
    let root = scratch("udp");
    fs::write(
        root.join("net.rs"),
        "fn open() -> UdpSocket { unreachable!() }\n",
    )
    .unwrap();
    assert_hit(&root, "UdpSocket");
}

#[test]
fn planted_nvml_is_rejected() {
    let root = scratch("nvml");
    fs::write(root.join("gpu.rs"), "use nvml_wrapper::Nvml;\n").unwrap();
    assert_hit(&root, "nvml");
}

#[test]
fn planted_brace_import_of_fs_and_net_is_rejected() {
    let root = scratch("brace-both");
    fs::write(root.join("present.rs"), "use std::{fs, net};\n").unwrap();
    let hits = scan_dir(&root);
    assert!(
        hits.iter().any(|hit| hit.contains("std::fs")),
        "brace fs import must fail, got {hits:?}"
    );
    assert!(
        hits.iter().any(|hit| hit.contains("std::net")),
        "brace net import must fail, got {hits:?}"
    );
}

#[test]
fn planted_brace_import_of_fs_then_read_is_rejected() {
    let root = scratch("brace-fs-read");
    fs::write(
        root.join("present.rs"),
        "use std::{fs};\nfn go(p: &str) { let _ = fs::read(p); }\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_nested_brace_import_of_fs_file_is_rejected() {
    let root = scratch("brace-nested");
    fs::write(root.join("present.rs"), "use std::{io::Read, fs::File};\n").unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_rustix_fs_outside_the_allowlist_is_rejected() {
    let root = scratch("rustix-fs");
    fs::write(
        root.join("present.rs"),
        "fn stat() { let _ = rustix::fs::statvfs(\"/sys\"); }\n",
    )
    .unwrap();
    assert_hit(&root, "rustix::fs");
}

#[test]
fn planted_rustix_io_read_outside_the_allowlist_is_rejected() {
    let root = scratch("rustix-read");
    fs::write(
        root.join("present.rs"),
        "fn pull(fd: i32) { let _ = rustix::io::read(fd, &mut []); }\n",
    )
    .unwrap();
    assert_hit(&root, "rustix::io");
}

#[test]
fn planted_rustix_io_pread_outside_the_allowlist_is_rejected() {
    let root = scratch("rustix-pread");
    fs::write(
        root.join("present.rs"),
        "fn pull(fd: i32) { let _ = rustix::io::pread(fd, &mut [], 0); }\n",
    )
    .unwrap();
    assert_hit(&root, "rustix::io");
}

#[test]
fn planted_renamed_std_is_rejected_even_in_an_allowlisted_file() {
    let root = scratch("alias-std");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(root.join("device/sysfs.rs"), "use std as x;\n").unwrap();
    assert_hit(&root, "std alias");
}

#[test]
fn planted_extern_crate_std_alias_is_rejected() {
    let root = scratch("extern-std");
    fs::write(root.join("lib.rs"), "extern crate std as x;\n").unwrap();
    assert_hit(&root, "std alias");
}

#[test]
fn planted_absolute_std_path_is_rejected() {
    let root = scratch("abs-std");
    fs::write(root.join("lib.rs"), "use ::std::collections::HashMap;\n").unwrap();
    assert_hit(&root, "std alias");
}

#[test]
fn planted_tcp_listener_is_rejected() {
    let root = scratch("listener");
    fs::write(
        root.join("net.rs"),
        "fn open() -> TcpListener { unreachable!() }\n",
    )
    .unwrap();
    assert_hit(&root, "TcpListener");
}

#[test]
fn planted_unix_sockets_are_rejected() {
    let root = scratch("unix");
    fs::write(
        root.join("net.rs"),
        "fn open() -> (UnixStream, UnixDatagram, UnixListener) { unreachable!() }\n",
    )
    .unwrap();
    for name in ["UnixStream", "UnixDatagram", "UnixListener"] {
        assert_hit(&root, name);
    }
}

#[test]
fn planted_std_os_unix_net_is_rejected() {
    let root = scratch("unix-net");
    fs::write(root.join("net.rs"), "use std::os::unix::net::UnixStream;\n").unwrap();
    assert_hit(&root, "std::os::unix::net");
}

#[test]
fn planted_roots_type_is_rejected() {
    let root = scratch("roots");
    fs::write(root.join("service.rs"), "use llama_core::sample::Roots;\n").unwrap();
    assert_hit(&root, "Roots");
}

#[test]
fn a_quote_char_literal_does_not_hide_a_proc_literal() {
    let root = scratch("char-quote");
    fs::write(
        root.join("read.rs"),
        "let q = '\"';\nlet path = \"/proc/stat\";\n",
    )
    .unwrap();
    assert_hit(&root, "/proc");
}

#[test]
fn a_quote_char_does_not_keep_a_commented_proc_literal() {
    let root = scratch("char-comment");
    fs::write(
        root.join("read.rs"),
        "let q = '\"';\n// \"/proc/stat\"\nlet x = 1;\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_fs_is_rejected_when_the_allowlist_is_empty() {
    let root = scratch("empty-allow");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/sysfs.rs"),
        "fn read(p: &str) { let _ = std::fs::read(p); }\n",
    )
    .unwrap();
    let hits = scan_tree(&root, &CORE_FENCE);
    assert!(
        hits.iter().any(|hit| hit.contains("std::fs")),
        "empty allowlist must reject std::fs, got {hits:?}"
    );
}

#[test]
fn planted_ureq_is_rejected() {
    let root = scratch("ureq");
    fs::write(root.join("http.rs"), "fn agent() { let _ = ureq::get; }\n").unwrap();
    assert_hit(&root, "ureq");
}

#[test]
fn planted_proc_literal_is_rejected() {
    let root = scratch("proc");
    fs::write(root.join("read.rs"), "let path = \"/proc/stat\";\n").unwrap();
    assert_hit(&root, "/proc");
}

#[test]
fn planted_snapshot_json_literal_is_rejected() {
    let root = scratch("snap-lit");
    fs::write(
        root.join("reader.rs"),
        "let path = \"/run/llama-watch/snapshot.json\";\n",
    )
    .unwrap();
    assert_hit(&root, "snapshot.json");
}

#[test]
fn planted_parent_segment_inside_cfg_test_is_still_rejected() {
    let root = scratch("cfg-dot");
    fs::write(
        root.join("main.rs"),
        "#[cfg(test)]\nfn fixtures() { let _ = \"../../fixtures\"; }\n",
    )
    .unwrap();
    assert_hit(&root, "..");
}

#[test]
fn planted_parent_segment_in_a_path_literal_is_rejected() {
    let root = scratch("dotdot");
    fs::write(root.join("read.rs"), "let path = \"../secret\";\n").unwrap();
    assert_hit(&root, "..");
}

#[test]
fn planted_include_that_escapes_the_crate_is_rejected() {
    let root = scratch("include-escape");
    fs::create_dir_all(root.join("render")).unwrap();
    fs::write(
        root.join("render/mod.rs"),
        "let bytes = include_bytes!(\"../../../../etc/passwd\");\n",
    )
    .unwrap();
    assert_hit(&root, "..");
}

#[test]
fn planted_std_fs_outside_the_allowlist_is_rejected() {
    let root = scratch("fs-present");
    fs::write(root.join("present.rs"), "std::fs::read_to_string(path)?;\n").unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_std_fs_split_by_a_block_comment_is_rejected() {
    let root = scratch("fs-comment");
    fs::write(
        root.join("present.rs"),
        "let _ = std::/**/fs::read(path);\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_file_open_outside_the_allowlist_is_rejected() {
    let root = scratch("file-open");
    fs::write(root.join("history.rs"), "let f = File::open(path)?;\n").unwrap();
    assert_hit(&root, "File::open");
}

#[test]
fn planted_open_options_outside_the_allowlist_is_rejected() {
    let root = scratch("open-options");
    fs::write(root.join("history.rs"), "let _ = OpenOptions::new();\n").unwrap();
    assert_hit(&root, "OpenOptions");
}

#[test]
fn planted_config_path_field_is_rejected() {
    let root = scratch("cfg-path");
    fs::write(
        root.join("config.rs"),
        "#[derive(Deserialize)]\npub struct Config {\n    pub snapshot_path: String,\n}\n",
    )
    .unwrap();
    assert_hit(&root, "snapshot_path");
}

#[test]
fn planted_config_pathbuf_field_is_rejected() {
    let root = scratch("cfg-pathbuf");
    fs::write(
        root.join("config.rs"),
        "#[derive(Deserialize)]\npub struct Snapshot {\n    pub location: PathBuf,\n}\n",
    )
    .unwrap();
    assert_hit(&root, "PathBuf");
}

#[test]
fn planted_sink_without_blocked_is_rejected() {
    let root = scratch("sink");
    fs::write(
        root.join("device.rs"),
        "impl LcdSink for Silent {\n    fn show(&mut self) {}\n}\n",
    )
    .unwrap();
    assert_hit(&root, "blocked");
}

#[test]
fn an_empty_source_tree_is_rejected() {
    let root = scratch("empty");
    assert_hit(&root, "empty");
}

#[test]
fn a_commented_forbidden_token_is_not_a_hit() {
    let root = scratch("comment-ok");
    fs::write(
        root.join("note.rs"),
        "// use std::net::TcpStream\n/* \"/proc/stat\" ureq nvml snapshot.json \"../x\" */\nlet x = 1;\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn a_range_inside_a_string_is_not_a_parent_segment() {
    let root = scratch("range-ok");
    fs::write(root.join("config.rs"), "let msg = \"outside 0.1..=2.0\";\n").unwrap();
    assert_clean(&root);
}

#[test]
fn std_fs_in_an_allowlisted_file_is_accepted() {
    let root = scratch("sysfs-ok");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/sysfs.rs"),
        "fn read(p: &str) { let _ = std::fs::read_dir(p); }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn include_bytes_that_stays_inside_the_crate_is_accepted() {
    let root = scratch("include-ok");
    fs::create_dir_all(root.join("render")).unwrap();
    fs::write(
        root.join("render/mod.rs"),
        "let _ = include_bytes!(\"../../assets/fonts/Inter-Bold.ttf\");\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_include_macro_is_rejected() {
    let root = scratch("include-macro");
    fs::write(root.join("lib.rs"), "include!(\"generated.rs\");\n").unwrap();
    assert_hit_fences(&root, "include!");
}

#[test]
fn planted_path_attribute_is_rejected() {
    let root = scratch("path-attr");
    fs::write(
        root.join("lib.rs"),
        "#[path = \"elsewhere.rs\"]\nmod hidden;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "#[path");
}

#[test]
fn planted_cfg_attr_path_is_rejected() {
    let root = scratch("cfg-path-attr");
    fs::write(
        root.join("lib.rs"),
        "#[cfg_attr(unix, path = \"elsewhere.rs\")]\nmod hidden;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "#[path");
}

#[test]
fn planted_env_macro_is_rejected() {
    let root = scratch("env-macro");
    fs::write(
        root.join("main.rs"),
        "let _ = env!(\"CARGO_MANIFEST_DIR\");\n",
    )
    .unwrap();
    assert_hit_fences(&root, "env!");
}

#[test]
fn planted_option_env_macro_is_rejected() {
    let root = scratch("option-env");
    fs::write(root.join("lib.rs"), "let _ = option_env!(\"SECRET\");\n").unwrap();
    assert_hit_fences(&root, "option_env!");
}

#[test]
fn planted_absolute_include_bytes_is_rejected() {
    let root = scratch("include-abs");
    fs::write(
        root.join("lib.rs"),
        "let _ = include_bytes!(\"/etc/passwd\");\n",
    )
    .unwrap();
    assert_hit_fences(&root, "absolute");
}

#[test]
fn planted_include_bytes_concat_is_rejected() {
    let root = scratch("include-concat");
    fs::create_dir_all(root.join("render")).unwrap();
    fs::write(
        root.join("render/mod.rs"),
        "let _ = include_bytes!(concat!(\"../../assets/\", \"fonts/a.ttf\"));\n",
    )
    .unwrap();
    assert_hit_fences(&root, "concat!");
}

#[test]
fn planted_include_bytes_env_argument_is_rejected() {
    let root = scratch("include-env-arg");
    fs::write(
        root.join("lib.rs"),
        "let _ = include_bytes!(env!(\"OUT_DIR\"));\n",
    )
    .unwrap();
    assert_hit_fences(&root, "include argument");
}

#[test]
fn planted_include_outside_assets_is_rejected() {
    let root = scratch("include-not-assets");
    fs::write(
        root.join("lib.rs"),
        "let _ = include_bytes!(\"secret.bin\");\n",
    )
    .unwrap();
    assert_hit_fences(&root, "outside assets");
}

#[test]
fn planted_include_str_outside_assets_is_rejected() {
    let root = scratch("include-str-out");
    fs::write(
        root.join("lib.rs"),
        "let _ = include_str!(\"../notes.txt\");\n",
    )
    .unwrap();
    assert_hit_fences(&root, "outside assets");
}

#[test]
fn include_str_under_assets_is_accepted() {
    let root = scratch("include-str-ok");
    fs::create_dir_all(root.join("render")).unwrap();
    fs::write(
        root.join("render/mod.rs"),
        "let _ = include_str!(\"../../assets/fonts/license.txt\");\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_macro_rules_is_rejected() {
    let root = scratch("macro-rules");
    fs::write(
        root.join("lib.rs"),
        "macro_rules! m { ($a:ident,$b:ident) => { $a::$b::read(p) } }\nfn go(p: &str) { m!(std, fs); }\n",
    )
    .unwrap();
    assert_hit_fences(&root, "macro_rules!");
}

#[test]
fn planted_pub_use_of_std_fs_is_rejected_on_the_allowlist() {
    let root = scratch("reexport-std-fs");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(root.join("device/sysfs.rs"), "pub use std::fs::read;\n").unwrap();
    assert_hit_fences(&root, "re-export std::fs");
}

#[test]
fn planted_pub_crate_use_of_rustix_fs_is_rejected_on_the_allowlist() {
    let root = scratch("reexport-rustix-fs");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/guard.rs"),
        "pub(crate) use rustix::fs::open;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "re-export rustix::fs");
}

#[test]
fn planted_pub_super_use_of_rustix_io_is_rejected_on_the_allowlist() {
    let root = scratch("reexport-rustix-io");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/sysfs.rs"),
        "pub(super) use rustix::io::Errno;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "re-export rustix::io");
}

#[test]
fn planted_pub_use_of_rustix_net_is_rejected_on_the_allowlist() {
    let root = scratch("reexport-rustix-net");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/sysfs.rs"),
        "pub use rustix::net::SocketAddr;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "re-export rustix::net");
}

#[test]
fn planted_rustix_alias_is_rejected_on_the_allowlist() {
    let root = scratch("alias-rustix");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(root.join("device/sysfs.rs"), "use rustix as r;\n").unwrap();
    assert_hit_fences(&root, "rustix alias");
}

#[test]
fn planted_extern_crate_rustix_alias_is_rejected() {
    let root = scratch("extern-rustix");
    fs::write(root.join("lib.rs"), "extern crate rustix as r;\n").unwrap();
    assert_hit_fences(&root, "rustix alias");
}

#[test]
fn planted_absolute_rustix_alias_is_rejected() {
    let root = scratch("abs-rustix");
    fs::write(root.join("lib.rs"), "use ::rustix as r;\n").unwrap();
    assert_hit_fences(&root, "rustix alias");
}

#[test]
fn planted_rustix_io_module_outside_the_allowlist_is_rejected() {
    let root = scratch("rustix-io-dup");
    fs::write(
        root.join("present.rs"),
        "fn pull(fd: i32) { let _ = rustix::io::dup(fd); }\n",
    )
    .unwrap();
    assert_hit_fences(&root, "rustix::io");
}

#[test]
fn planted_rustix_net_is_rejected_on_the_allowlist() {
    let root = scratch("rustix-net");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/guard.rs"),
        "fn sock() { let _ = rustix::net::socket(); }\n",
    )
    .unwrap();
    assert_hit_fences(&root, "rustix::net");
}

#[test]
fn planted_std_process_id_in_main_is_rejected() {
    let root = scratch("process-id");
    fs::write(root.join("main.rs"), "let _ = std::process::id();\n").unwrap();
    assert_hit_fences(&root, "std::process");
}

#[test]
fn planted_std_process_exit_outside_main_is_rejected() {
    let root = scratch("process-exit");
    fs::write(root.join("service.rs"), "std::process::exit(1);\n").unwrap();
    assert_hit_fences(&root, "std::process");
}

#[test]
fn planted_std_process_command_is_rejected() {
    let root = scratch("process-cmd");
    fs::write(
        root.join("lib.rs"),
        "fn go() { let _ = std::process::Command::new(\"sh\"); }\n",
    )
    .unwrap();
    assert_hit_fences(&root, "std::process");
}

#[test]
fn std_process_exit_in_main_is_accepted() {
    let root = scratch("process-ok");
    fs::write(
        root.join("main.rs"),
        "fn stop(code: std::process::ExitCode) { std::process::exit(1); let _ = code; }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_load_validated_outside_service_and_main_is_rejected() {
    let root = scratch("load-caller");
    fs::write(
        root.join("present.rs"),
        "fn go(config_path: &Path) { let _ = Config::load_validated(config_path); }\n",
    )
    .unwrap();
    assert_hit(&root, "load_validated");
}

#[test]
fn planted_load_validated_with_a_path_literal_is_rejected() {
    let root = scratch("load-literal");
    fs::write(
        root.join("service.rs"),
        "fn go() { let _ = Config::load_validated(\"/etc/llama-bored/config.toml\"); }\n",
    )
    .unwrap();
    assert_hit(&root, "load_validated");
}

#[test]
fn load_validated_of_the_cli_path_in_service_is_accepted() {
    let root = scratch("load-ok");
    fs::write(
        root.join("service.rs"),
        "fn run(config_path: &Path) { let _ = Config::load_validated(config_path); }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_two_step_reexport_of_std_fs_is_rejected() {
    let root = scratch("reexport-two-step");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(root.join("device/sysfs.rs"), "use std::fs;\npub use fs;\n").unwrap();
    assert_hit_fences(&root, "re-export std::fs");
}

#[test]
fn planted_renamed_reexport_of_rustix_fs_is_rejected() {
    let root = scratch("reexport-renamed");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/guard.rs"),
        "use rustix::fs as x;\npub(crate) use x;\n",
    )
    .unwrap();
    assert_hit_fences(&root, "re-export rustix::fs");
}

#[test]
fn planted_pub_use_self_fs_is_a_new_public_item() {
    let root = scratch("surface-self");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/sysfs.rs"),
        "use std::fs;\npub use self::fs;\n",
    )
    .unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_pub_use_through_crate_path_is_a_new_public_item() {
    let root = scratch("surface-crate");
    fs::write(
        root.join("service.rs"),
        "pub use crate::device::sysfs::fs;\n",
    )
    .unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_pub_type_alias_of_file_is_a_new_public_item() {
    let root = scratch("surface-type");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(root.join("device/hid.rs"), "pub type F = std::fs::File;\n").unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_pub_const_fn_pointer_is_a_new_public_item() {
    let root = scratch("surface-const");
    fs::write(
        root.join("snapshot_reader.rs"),
        "pub const R: fn(&Path) -> std::io::Result<Vec<u8>> = std::fs::read;\n",
    )
    .unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_pub_static_fn_pointer_is_a_new_public_item() {
    let root = scratch("surface-static");
    fs::write(
        root.join("main.rs"),
        "pub static R: fn(&Path) -> std::io::Result<Vec<u8>> = std::fs::read;\n",
    )
    .unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_pub_fn_read_any_is_a_new_public_item() {
    let root = scratch("surface-fn");
    fs::write(
        root.join("config.rs"),
        "pub fn read_any(p: &Path) -> Vec<u8> { vec![] }\n",
    )
    .unwrap();
    assert_hit(
        &root,
        "new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE",
    );
}

#[test]
fn planted_trait_default_method_is_a_new_public_item() {
    let root = scratch("surface-trait-fn");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/hid.rs"),
        "pub trait HidPort {\n    fn send(&mut self);\n    fn slurp(&self, p: &Path) -> Vec<u8> { vec![] }\n}\n",
    )
    .unwrap();
    assert_hit(&root, "trait_fn HidPort::slurp");
}

#[test]
fn planted_impl_from_path_for_check_env_is_a_new_public_item() {
    let root = scratch("surface-impl");
    fs::write(
        root.join("service.rs"),
        "impl From<&Path> for CheckEnv {\n    fn from(_: &Path) -> Self { unimplemented!() }\n}\n",
    )
    .unwrap();
    assert_hit(&root, "impl From for CheckEnv");
}

#[test]
fn planted_fn_pointer_field_on_a_pinned_struct_is_rejected() {
    let root = scratch("surface-field");
    fs::write(
        root.join("service.rs"),
        "pub struct CheckEnv {\n    pub sys_root: PathBuf,\n    pub reader: fn(&Path) -> std::io::Result<Vec<u8>>,\n}\n",
    )
    .unwrap();
    assert_hit(&root, "public fn-pointer field CheckEnv.reader");
}

#[test]
fn planted_open_request_outside_the_open_path_is_rejected() {
    let root = scratch("open-request");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &Path) { let _ = OpenRequest { sys_root: p, state_dir: p, rotate_deg: 0, trace_hid: false }; }\n",
    )
    .unwrap();
    assert_hit(&root, "OpenRequest");
}

#[test]
fn planted_snapshot_new_of_etc_is_rejected() {
    let root = scratch("snap-new");
    fs::write(
        root.join("present.rs"),
        "fn go() { SnapshotReader::new(Path::new(\"/etc/x\"), std::time::Duration::from_secs(1), Path::new(\"/sys\"), (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_snapshot_new_alias_is_rejected() {
    let root = scratch("snap-alias");
    fs::write(
        root.join("present.rs"),
        "use crate::snapshot_reader::SnapshotReader as S;\nfn go() { S::new(Path::new(\"/etc/x\"), std::time::Duration::from_secs(1), Path::new(\"/sys\"), (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_snapshot_new_ufcs_is_rejected() {
    let root = scratch("snap-ufcs");
    fs::write(
        root.join("present.rs"),
        "fn go() { <SnapshotReader>::new(Path::new(\"/etc/x\"), std::time::Duration::from_secs(1), Path::new(\"/sys\"), (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_snapshot_turbofish_new_is_rejected() {
    let root = scratch("snap-turbofish");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &Path) { SnapshotReader::<crate::log::Stderr>::new(p, std::time::Duration::from_secs(1), p, (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_snapshot_path_ufcs_new_is_rejected() {
    let root = scratch("snap-path-ufcs");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &Path) { <crate::snapshot_reader::SnapshotReader>::new(p, std::time::Duration::from_secs(1), p, (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_snapshot_type_alias_new_is_rejected() {
    let root = scratch("snap-alias-type");
    fs::write(
        root.join("present.rs"),
        "type T = SnapshotReader<crate::log::Stderr, crate::snapshot_reader::HostMono>;\nfn go(p: &Path) { T::new(p, std::time::Duration::from_secs(1), p, (), ()); }\n",
    )
    .unwrap();
    assert_hit(&root, "SnapshotReader");
}

#[test]
fn planted_open_request_renamed_is_rejected() {
    let root = scratch("open-request-alias");
    fs::write(
        root.join("present.rs"),
        "use crate::device::OpenRequest as Q;\nfn go(p: &Path) { let _ = Q { sys_root: p, state_dir: p, rotate_deg: 0, trace_hid: false }; }\n",
    )
    .unwrap();
    assert_hit(&root, "OpenRequest");
}

#[test]
fn planted_hid_open_outside_device_is_rejected() {
    let root = scratch("hid-open");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &Path) { let _ = HidLink::open(p); }\n",
    )
    .unwrap();
    assert_hit(&root, "HidLink::open");
}

#[test]
fn planted_state_dir_writable_call_outside_device_is_rejected() {
    let root = scratch("state-dir-writable");
    fs::write(
        root.join("present.rs"),
        "fn go(d: &Path) -> bool { guard::state_dir_writable(d) }\n",
    )
    .unwrap();
    assert_hit(&root, "state_dir_writable");
}

#[test]
fn planted_state_dir_writable_import_outside_device_is_rejected() {
    let root = scratch("state-dir-writable-use");
    fs::write(
        root.join("present.rs"),
        "use crate::device::guard::state_dir_writable as w;\n",
    )
    .unwrap();
    assert_hit(&root, "state_dir_writable");
}

#[test]
fn a_state_dir_writable_field_outside_device_is_accepted() {
    let root = scratch("state-dir-writable-field");
    fs::write(
        root.join("service.rs"),
        "pub struct CheckEnv { pub state_dir_writable: bool }\n\
         fn go(e: &CheckEnv) -> bool { e.state_dir_writable }\n\
         fn make() -> CheckEnv { CheckEnv { state_dir_writable: true } }\n",
    )
    .unwrap();
    let hits = scan_dir(&root);
    assert!(
        !hits.iter().any(|hit| hit.contains("state_dir_writable")),
        "a field of the same name is not a call, got {hits:?}"
    );
}

#[test]
fn a_state_dir_writable_call_under_device_is_accepted() {
    let root = scratch("state-dir-writable-device");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/mod.rs"),
        "fn go(d: &Path) -> bool { guard::state_dir_writable(d) }\n",
    )
    .unwrap();
    let hits = scan_dir(&root);
    assert!(
        !hits.iter().any(|hit| hit.contains("state_dir_writable")),
        "device/ may call it, got {hits:?}"
    );
}

#[test]
fn planted_cooling_guard_new_outside_the_open_path_is_rejected() {
    let root = scratch("guard-new");
    fs::write(
        root.join("present.rs"),
        "fn go(d: &Path) { let _ = CoolingGuard::new(d, d, d); }\n",
    )
    .unwrap();
    assert_hit(&root, "CoolingGuard::new");
}

#[test]
fn a_for_loop_named_r_does_not_hide_std_fs() {
    let root = scratch("for-r");
    fs::write(
        root.join("present.rs"),
        "fn go(v: &[u8]) { for r in v { let s = \"a\\\"b\"; let _ = r; } std::fs::read(\"x\"); }\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn a_for_loop_named_r_does_not_keep_a_comment_inside_a_string() {
    let root = scratch("for-r-clean");
    fs::write(
        root.join("present.rs"),
        "fn go(v: &[u8]) { for r in v { let s = \"a\\\"b // c\"; let _ = r; } let t = 1; }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn a_raw_identifier_does_not_hide_std_net() {
    let root = scratch("raw-ident");
    fs::write(
        root.join("present.rs"),
        "fn go() { let r#type = 1; let s = \"a\\\"b\"; std::net::TcpStream; }\n",
    )
    .unwrap();
    assert_hit(&root, "std::net");
}

#[test]
fn a_raw_identifier_does_not_treat_a_string_as_a_comment() {
    let root = scratch("raw-ident-clean");
    fs::write(
        root.join("present.rs"),
        "fn go() { let r#type = \"x\\\" // y\"; let z = r#type; }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_raw_string_quote_does_not_hide_std_fs() {
    let root = scratch("raw-quote");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &str) { let a = r#\"x\"y\"#; let s = \"http://h\"; std::fs::read(p); }\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_raw_string_backslash_does_not_hide_std_fs() {
    let root = scratch("raw-backslash");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &str) { let a = r\"C:\\\"; let s = \"http://h\"; std::fs::read(p); }\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_byte_raw_string_does_not_hide_std_fs() {
    let root = scratch("raw-byte");
    fs::write(
        root.join("present.rs"),
        "fn go(p: &str) { let a = br##\"a\"#b\"##; let s = \"http://h\"; std::fs::read(p); }\n",
    )
    .unwrap();
    assert_hit(&root, "std::fs");
}

#[test]
fn planted_std_env_var_is_rejected_on_the_allowlist() {
    let root = scratch("env-var");
    fs::write(
        root.join("snapshot_reader.rs"),
        "fn p() -> String { std::env::var(\"KD_SNAP\").unwrap() }\n",
    )
    .unwrap();
    assert_hit_fences(&root, "std::env::var");
}

#[test]
fn planted_std_env_var_alias_is_rejected() {
    let root = scratch("env-alias");
    fs::write(
        root.join("present.rs"),
        "use std::env::{var as snap, temp_dir};\n",
    )
    .unwrap();
    assert_hit_fences(&root, "std::env::var");
}

#[test]
fn std_env_args_in_main_and_temp_dir_anywhere_are_accepted() {
    let root = scratch("env-ok");
    fs::write(
        root.join("main.rs"),
        "fn main() { let _ = std::env::args(); let _ = std::env::temp_dir(); }\n",
    )
    .unwrap();
    fs::write(
        root.join("service.rs"),
        "fn dir() -> std::path::PathBuf { std::env::temp_dir() }\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn planted_load_validated_function_pointer_is_rejected() {
    let root = scratch("load-fnptr");
    fs::write(
        root.join("present.rs"),
        "fn go() { let f = Config::load_validated; let _ = f; }\n",
    )
    .unwrap();
    assert_hit(&root, "load_validated");
}

#[test]
fn a_lifetime_does_not_keep_a_commented_proc_literal() {
    let root = scratch("lifetime-comment");
    fs::write(
        root.join("read.rs"),
        "fn f<'static>() {\n    // \"/proc/stat\"\n    let x = 1;\n}\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn an_apostrophe_in_a_string_does_not_keep_a_commented_proc_literal() {
    let root = scratch("string-apostrophe");
    fs::write(
        root.join("read.rs"),
        "fn f<'a>() {\n    let s = \"it's\";\n    // \"/proc/stat\"\n    let x = 1;\n}\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn a_string_apostrophe_does_not_hide_a_proc_literal() {
    let root = scratch("string-apostrophe-hit");
    fs::write(
        root.join("read.rs"),
        "let s = \"it's\";\nlet path = \"/proc/stat\";\n",
    )
    .unwrap();
    assert_hit(&root, "/proc");
}

#[test]
fn a_pathbuf_error_field_is_not_a_config_key() {
    let root = scratch("err-path");
    fs::write(
        root.join("config.rs"),
        "struct ConfigError {\n    path: PathBuf,\n}\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn a_sink_that_implements_blocked_is_accepted() {
    let root = scratch("sink-ok");
    fs::write(
        root.join("device.rs"),
        "impl<H, B> LcdSink for KrakenLcd<H, B> {\n    fn blocked(&mut self) -> bool { false }\n}\n",
    )
    .unwrap();
    assert_clean(&root);
}

#[test]
fn writer_source_passes_s11_and_the_file_set_is_non_empty() {
    let src = writer_src();
    let rels = rs_rels(&src);
    assert!(!rels.is_empty(), "S11 saw no writer sources");
    for required in FS_FILES {
        assert!(
            rels.iter().any(|rel| rel == required),
            "S11 file set is missing {required}: {rels:?}"
        );
    }
    let hits = scan_dir(&src);
    assert!(hits.is_empty(), "{}", hits.join("\n"));
    let mut found = BTreeSet::new();
    for rel in FS_FILES {
        let text = fs::read_to_string(src.join(rel)).expect("read allowlisted source");
        for (kind, name) in public_items(&strip_comments(&text)) {
            found.insert(((*rel).to_string(), kind, name));
        }
    }
    let expected: BTreeSet<_> = PUBLIC_SURFACE
        .iter()
        .map(|(file, kind, name)| {
            (
                (*file).to_string(),
                (*kind).to_string(),
                (*name).to_string(),
            )
        })
        .collect();
    let missing: Vec<_> = expected.difference(&found).cloned().collect();
    let extra: Vec<_> = found.difference(&expected).cloned().collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "PUBLIC_SURFACE drifted\nmissing {missing:?}\nextra {extra:?}"
    );
    let mut saw_constant = false;
    for rel in &rels {
        let text = fs::read_to_string(src.join(rel)).expect("read writer source");
        if strip_comments(&text).contains("llama_core::wire::SNAPSHOT_PATH") {
            saw_constant = true;
        }
    }
    assert!(
        saw_constant,
        "the snapshot path must appear as llama_core::wire::SNAPSHOT_PATH"
    );
}

#[test]
fn llama_core_source_has_no_net_proc_or_filesystem() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../llama-core/src");
    let rels = rs_rels(&src);
    assert!(!rels.is_empty(), "core scan saw no sources");
    let hits = scan_tree(&src, &CORE_FENCE);
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn rustix_fs_in_an_allowlisted_file_is_accepted() {
    let root = scratch("rustix-ok");
    fs::create_dir_all(root.join("device")).unwrap();
    fs::write(
        root.join("device/guard.rs"),
        "fn sync(file: &File) { rustix::fs::fsync(file).ok(); }\n",
    )
    .unwrap();
    assert_clean(&root);
}

fn assert_hit(root: &Path, needle: &str) {
    let hits = scan_dir(root);
    assert!(
        hits.iter().any(|hit| hit.contains(needle)),
        "planted {needle} must fail the scan, got {hits:?}"
    );
}

fn assert_hit_fences(root: &Path, needle: &str) {
    assert_hit(root, needle);
    let hits = scan_tree(root, &CORE_FENCE);
    assert!(
        hits.iter().any(|hit| hit.contains(needle)),
        "core fence missed planted {needle}: {hits:?}"
    );
}

fn assert_clean(root: &Path) {
    let hits = scan_dir(root);
    assert!(hits.is_empty(), "{hits:?}");
}

fn scratch(label: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("s11-{label}"));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

fn writer_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rs_rels(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    collect_rs(root, &mut files);
    let mut rels: Vec<String> = files
        .iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    rels.sort();
    rels
}

struct Fence {
    fs_allow: &'static [&'static str],
    writer: bool,
}

const WRITER_FENCE: Fence = Fence {
    fs_allow: FS_FILES,
    writer: true,
};

const CORE_FENCE: Fence = Fence {
    fs_allow: &[],
    writer: false,
};

fn scan_dir(root: &Path) -> Vec<String> {
    scan_tree(root, &WRITER_FENCE)
}

fn scan_tree(root: &Path, fence: &Fence) -> Vec<String> {
    let rels = rs_rels(root);
    let mut hits = Vec::new();
    if rels.is_empty() {
        hits.push(format!("{}: writer source set is empty", root.display()));
        return hits;
    }
    let crate_root = root.parent().unwrap_or(root);
    for rel in &rels {
        let text = match fs::read_to_string(root.join(rel)) {
            Ok(text) => text,
            Err(err) => {
                hits.push(format!("{rel}: {err}"));
                continue;
            }
        };
        let code = strip_comments(&text);
        let fs_allowed = fence.fs_allow.contains(&rel.as_str());
        hits.extend(token_hits(rel, &code));
        hits.extend(import_hits(rel, &code, fs_allowed));
        hits.extend(renamed_reexport_hits(rel, &code));
        hits.extend(outside_hits(rel, &code, root, crate_root));
        hits.extend(process_hits(rel, &code));
        hits.extend(env_hits(rel, &code));
        if fs_allowed {
            hits.extend(public_surface_hits(rel, &code));
        }
        if !fs_allowed {
            hits.extend(fs_hits(rel, &code));
        }
        if fence.writer {
            hits.extend(literal_hits(rel, &code, root, crate_root));
            if rel == "config.rs" || rel.ends_with("/config.rs") {
                hits.extend(config_path_hits(rel, &code));
            }
            hits.extend(sink_hits(rel, &code));
            hits.extend(load_validated_hits(rel, &code));
            hits.extend(fn_pointer_field_hits(rel, &code));
            hits.extend(open_request_hits(rel, &code));
            hits.extend(snapshot_new_hits(rel, &code));
            hits.extend(hid_open_hits(rel, &code));
            hits.extend(state_dir_writable_hits(rel, &code));
            hits.extend(cooling_guard_new_hits(rel, &code));
            hits.extend(resolved_hid_hits(rel, &code));
            if has_ident(&code, "Roots") {
                hits.push(format!("{rel}: Roots"));
            }
        }
    }
    hits.sort();
    hits
}

fn token_hits(rel: &str, code: &str) -> Vec<String> {
    let mut hits = Vec::new();
    if has_std_net(code) || contains_qualified(code, &["std", "os", "unix", "net"]) {
        let label = if contains_qualified(code, &["std", "os", "unix", "net"]) {
            "std::os::unix::net"
        } else {
            "std::net"
        };
        hits.push(format!("{rel}: {label}"));
    }
    if has_std_net(code) && contains_qualified(code, &["std", "os", "unix", "net"]) {
        hits.push(format!("{rel}: std::net"));
    }
    for name in [
        "TcpStream",
        "UdpSocket",
        "TcpListener",
        "UnixStream",
        "UnixDatagram",
        "UnixListener",
    ] {
        if has_ident(code, name) {
            hits.push(format!("{rel}: {name}"));
        }
    }
    if has_keyword_seq(code, &["use", "std", "as"])
        || has_keyword_seq(code, &["extern", "crate", "std", "as"])
        || has_absolute_name(code, "std")
    {
        hits.push(format!("{rel}: std alias"));
    }
    if has_keyword_seq(code, &["use", "rustix", "as"])
        || has_keyword_seq(code, &["extern", "crate", "rustix", "as"])
        || has_absolute_name(code, "rustix")
    {
        hits.push(format!("{rel}: rustix alias"));
    }
    if contains_qualified(code, &["rustix", "net"]) {
        hits.push(format!("{rel}: rustix::net"));
    }
    if code.contains("nvml") {
        hits.push(format!("{rel}: nvml"));
    }
    if code.contains("ureq") {
        hits.push(format!("{rel}: ureq"));
    }
    // "/proc is matched as a prefix only; relative or assembled paths rely on the unit sandbox (ProcSubset=pid)."
    if code.contains("\"/proc") {
        hits.push(format!("{rel}: \"/proc literal"));
    }
    hits
}

fn import_hits(rel: &str, code: &str, fs_allowed: bool) -> Vec<String> {
    let mut hits = Vec::new();
    for (item, reexport) in use_items(code) {
        let absolute = item.trim_start().starts_with("::");
        let body = item.trim().trim_start_matches("::");
        for path in expand_use_tree(body, &[]) {
            if path.first().map(String::as_str) == Some("std") && (absolute || path.len() == 1) {
                hits.push(format!("{rel}: std alias"));
            }
            if path.first().map(String::as_str) == Some("rustix") && (absolute || path.len() == 1) {
                hits.push(format!("{rel}: rustix alias"));
            }
            if reexport {
                for (prefix, label) in [
                    (&["std", "fs"][..], "re-export std::fs"),
                    (&["rustix", "fs"][..], "re-export rustix::fs"),
                    (&["rustix", "io"][..], "re-export rustix::io"),
                    (&["rustix", "net"][..], "re-export rustix::net"),
                ] {
                    if path_starts(&path, prefix) {
                        hits.push(format!("{rel}: {label}"));
                    }
                }
            }
            if !fs_allowed && path_starts(&path, &["std", "fs"]) {
                hits.push(format!("{rel}: std::fs"));
            }
            if !fs_allowed && path_starts(&path, &["rustix", "fs"]) {
                hits.push(format!("{rel}: rustix::fs"));
            }
            if !fs_allowed && path_starts(&path, &["rustix", "io"]) {
                hits.push(format!("{rel}: rustix::io"));
            }
            if path_starts(&path, &["rustix", "net"]) {
                hits.push(format!("{rel}: rustix::net"));
            }
            if path_starts(&path, &["std", "net"]) {
                hits.push(format!("{rel}: std::net"));
            }
            if path_starts(&path, &["std", "os", "unix", "net"]) {
                hits.push(format!("{rel}: std::os::unix::net"));
            }
        }
    }
    hits
}

fn use_items(code: &str) -> Vec<(String, bool)> {
    let chars: Vec<char> = code.chars().collect();
    let mut items = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, "use") {
            let reexport = use_is_reexport(&chars, index);
            let mut cursor = skip_ws(&chars, index + 3);
            let start = cursor;
            let mut depth = 0i32;
            while cursor < chars.len() {
                match chars[cursor] {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    ';' if depth == 0 => {
                        items.push((chars[start..cursor].iter().collect(), reexport));
                        break;
                    }
                    _ => {}
                }
                cursor += 1;
            }
            index = cursor;
            continue;
        }
        index += 1;
    }
    items
}

/// `pub use`, `pub(crate) use`, `pub(super) use`, and any other `pub(...) use`.
fn use_is_reexport(chars: &[char], use_at: usize) -> bool {
    let mut index = use_at;
    while index > 0 && chars[index - 1].is_whitespace() {
        index -= 1;
    }
    if index > 0 && chars[index - 1] == ')' {
        let mut depth = 1i32;
        index -= 1;
        while index > 0 && depth > 0 {
            index -= 1;
            match chars[index] {
                ')' => depth += 1,
                '(' => depth -= 1,
                _ => {}
            }
        }
        if depth != 0 {
            return false;
        }
        while index > 0 && chars[index - 1].is_whitespace() {
            index -= 1;
        }
    }
    index >= 3 && ident_at(chars, index - 3, "pub")
}

fn public_surface_hits(rel: &str, code: &str) -> Vec<String> {
    let mut hits = Vec::new();
    for (kind, name) in public_items(code) {
        let known = PUBLIC_SURFACE.iter().any(|(file, item_kind, item_name)| {
            *file == rel && *item_kind == kind && *item_name == name
        });
        if !known {
            hits.push(format!(
                "{rel}: new public item in fs-allowlisted file: review and add to PUBLIC_SURFACE ({kind} {name})"
            ));
        }
    }
    hits
}

struct SurfaceWalk {
    items: Vec<(String, String)>,
    fn_pointer_fields: Vec<String>,
}

fn public_items(code: &str) -> Vec<(String, String)> {
    walk_surface(code).items
}

fn fn_pointer_field_hits(rel: &str, code: &str) -> Vec<String> {
    walk_surface(code)
        .fn_pointer_fields
        .into_iter()
        .map(|name| format!("{rel}: public fn-pointer field {name}"))
        .collect()
}

/// `impl Trait for Type`, trait-body fns (no `pub` required), and `pub` fields.
fn walk_surface(code: &str) -> SurfaceWalk {
    let chars: Vec<char> = code.chars().collect();
    let mut walked = SurfaceWalk {
        items: Vec::new(),
        fn_pointer_fields: Vec::new(),
    };
    let mut index = 0;
    let mut depth = 0i32;
    let mut impls: Vec<(i32, String)> = Vec::new();
    let mut traits: Vec<(i32, String)> = Vec::new();
    let mut structs: Vec<(i32, String)> = Vec::new();
    let mut pending_impl: Option<(usize, String)> = None;
    let mut pending_trait: Option<(usize, String)> = None;
    let mut pending_struct: Option<(usize, String)> = None;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(&chars, index);
            index = next.max(index + 1);
            continue;
        }
        if chars[index] == '\'' {
            index = skip_quoted(&chars, index).max(index + 1);
            continue;
        }
        if chars[index] == '{' {
            depth += 1;
            take_pending(&mut pending_impl, index, depth, &mut impls);
            take_pending(&mut pending_trait, index, depth, &mut traits);
            take_pending(&mut pending_struct, index, depth, &mut structs);
            index += 1;
            continue;
        }
        if chars[index] == '}' {
            depth -= 1;
            impls.retain(|(body, _)| *body <= depth);
            traits.retain(|(body, _)| *body <= depth);
            structs.retain(|(body, _)| *body <= depth);
            index += 1;
            continue;
        }
        if ident_at(&chars, index, "impl") && at_item_pos(&chars, index) {
            let after = index + "impl".len();
            if let Some((brace, ty)) = parse_impl_header(&chars, after) {
                if let Some(row) = impl_trait_for_row(&chars, after, brace) {
                    walked.items.push(("impl".to_owned(), row));
                }
                pending_impl = Some((brace, ty));
            }
            index = after;
            continue;
        }
        if ident_at(&chars, index, "pub")
            && at_item_pos(&chars, index)
            && let Some(next) = note_pub_item(
                &mut PubCtx {
                    chars: &chars,
                    depth,
                    impls: &impls,
                    traits: &traits,
                    structs: &structs,
                    pending_struct: &mut pending_struct,
                    pending_trait: &mut pending_trait,
                    walked: &mut walked,
                },
                index,
            )
        {
            index = next.max(index + 1);
            continue;
        }
        if ident_at(&chars, index, "struct") && at_item_pos(&chars, index) {
            let after = index + "struct".len();
            if let Some((brace, name)) = pending_body(&chars, after) {
                pending_struct = Some((brace, name));
            }
            index = after;
            continue;
        }
        if ident_at(&chars, index, "trait") && at_item_pos(&chars, index) {
            let after = index + "trait".len();
            if let Some((brace, name)) = pending_body(&chars, after) {
                pending_trait = Some((brace, name));
            }
            index = after;
            continue;
        }
        if let Some(trait_name) = body_name(&traits, depth)
            && at_item_pos(&chars, index)
            && let Some((name, next)) = trait_fn_at(&chars, index)
        {
            walked
                .items
                .push(("trait_fn".to_owned(), format!("{trait_name}::{name}")));
            index = next.max(index + 1);
            continue;
        }
        index += 1;
    }
    walked
}

fn take_pending(
    pending: &mut Option<(usize, String)>,
    brace: usize,
    depth: i32,
    stack: &mut Vec<(i32, String)>,
) {
    if pending.as_ref().is_some_and(|(at, _)| *at == brace) {
        let name = pending.take().map(|(_, name)| name).unwrap_or_default();
        stack.push((depth, name));
    }
}

fn body_name(stack: &[(i32, String)], depth: i32) -> Option<&str> {
    stack
        .iter()
        .rev()
        .find(|(body, _)| *body == depth)
        .map(|(_, name)| name.as_str())
}

fn pending_body(chars: &[char], after_keyword: usize) -> Option<(usize, String)> {
    let name_at = skip_ws(chars, after_keyword);
    let name = read_ident(chars, name_at)?;
    let brace = next_body_brace(chars, name_at + name.len())?;
    Some((brace, name))
}

struct PubCtx<'a> {
    chars: &'a [char],
    depth: i32,
    impls: &'a [(i32, String)],
    traits: &'a [(i32, String)],
    structs: &'a [(i32, String)],
    pending_struct: &'a mut Option<(usize, String)>,
    pending_trait: &'a mut Option<(usize, String)>,
    walked: &'a mut SurfaceWalk,
}

/// Returns the index to resume from when `pub` began an item or a field.
fn note_pub_item(ctx: &mut PubCtx<'_>, pub_at: usize) -> Option<usize> {
    if let Some((kind, names, next)) = parse_pub_item(ctx.chars, pub_at) {
        if kind == "struct"
            && let Some(name) = names.first()
            && let Some(brace) = next_body_brace(ctx.chars, next)
        {
            *ctx.pending_struct = Some((brace, name.clone()));
        }
        if kind == "trait"
            && let Some(name) = names.first()
            && let Some(brace) = next_body_brace(ctx.chars, next)
        {
            *ctx.pending_trait = Some((brace, name.clone()));
        }
        for name in names {
            if kind == "fn" {
                if let Some(trait_name) = body_name(ctx.traits, ctx.depth) {
                    ctx.walked
                        .items
                        .push(("trait_fn".to_owned(), format!("{trait_name}::{name}")));
                } else {
                    ctx.walked
                        .items
                        .push(("fn".to_owned(), qualify_fn(ctx.impls, ctx.depth, &name)));
                }
            } else {
                ctx.walked.items.push((kind.clone(), name));
            }
        }
        return Some(next);
    }
    let struct_name = body_name(ctx.structs, ctx.depth)?;
    let field = parse_pub_field(ctx.chars, pub_at)?;
    let qualified = format!("{struct_name}.{}", field.name);
    if field_type_is_fn_pointer(&field.ty) {
        ctx.walked.fn_pointer_fields.push(qualified.clone());
    }
    ctx.walked.items.push(("field".to_owned(), qualified));
    Some(field.end)
}

struct ParsedField {
    name: String,
    ty: String,
    end: usize,
}

fn parse_pub_field(chars: &[char], pub_at: usize) -> Option<ParsedField> {
    let mut cursor = pub_at + "pub".len();
    cursor = skip_ws(chars, cursor);
    if cursor < chars.len() && chars[cursor] == '(' {
        cursor = skip_balanced(chars, cursor, '(', ')')? + 1;
    }
    cursor = skip_ws(chars, cursor);
    let name = read_ident(chars, cursor)?;
    if is_item_kind(&name) {
        return None;
    }
    cursor = skip_ws(chars, cursor + name.len());
    if cursor >= chars.len() || chars[cursor] != ':' {
        return None;
    }
    cursor += 1;
    let end = field_type_end(chars, cursor);
    let ty: String = chars[cursor..end].iter().collect();
    Some(ParsedField { name, ty, end })
}

fn is_item_kind(name: &str) -> bool {
    matches!(
        name,
        "fn" | "type"
            | "const"
            | "static"
            | "use"
            | "mod"
            | "struct"
            | "enum"
            | "trait"
            | "union"
            | "unsafe"
            | "async"
    )
}

fn field_type_end(chars: &[char], start: usize) -> usize {
    let mut angles = 0i32;
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut index = start;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(chars, index);
            index = next;
            continue;
        }
        if chars[index] == '\'' {
            index = skip_quoted(chars, index);
            continue;
        }
        match chars[index] {
            '<' => angles += 1,
            '>' => angles -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            ',' | '}' | '{' if angles == 0 && parens == 0 && brackets == 0 => return index,
            _ => {}
        }
        index += 1;
    }
    index
}

/// `fn(`, `dyn Fn`, `impl Fn`, or `Box<dyn Fn`, ignoring whitespace.
fn field_type_is_fn_pointer(ty: &str) -> bool {
    let compact: String = ty.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact.contains("fn(")
        || compact.contains("dynFn")
        || compact.contains("implFn")
        || compact.contains("Box<dynFn")
}

fn trait_fn_at(chars: &[char], index: usize) -> Option<(String, usize)> {
    let mut cursor = index;
    loop {
        cursor = skip_ws(chars, cursor);
        if ident_at(chars, cursor, "unsafe") || ident_at(chars, cursor, "async") {
            let word = if ident_at(chars, cursor, "unsafe") {
                "unsafe"
            } else {
                "async"
            };
            cursor += word.len();
            continue;
        }
        if ident_at(chars, cursor, "const") {
            let after = skip_ws(chars, cursor + "const".len());
            if ident_at(chars, after, "fn") {
                cursor += "const".len();
                continue;
            }
            return None;
        }
        break;
    }
    if !ident_at(chars, cursor, "fn") {
        return None;
    }
    cursor = skip_ws(chars, cursor + "fn".len());
    let name = read_ident(chars, cursor)?;
    let next = cursor + name.len();
    Some((name, next))
}

fn next_body_brace(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start;
    let mut angles = 0i32;
    let mut parens = 0i32;
    let mut brackets = 0i32;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(chars, index);
            index = next;
            continue;
        }
        if chars[index] == '\'' {
            index = skip_quoted(chars, index);
            continue;
        }
        match chars[index] {
            '<' => angles += 1,
            '>' => angles -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' if angles == 0 && parens == 0 && brackets == 0 => return Some(index),
            ';' if angles == 0 && parens == 0 && brackets == 0 => return None,
            _ => {}
        }
        index += 1;
    }
    None
}

fn impl_trait_for_row(chars: &[char], start: usize, brace: usize) -> Option<String> {
    let header: String = chars[start..brace].iter().collect();
    let header_chars: Vec<char> = header.chars().collect();
    let mut angles = 0i32;
    let mut index = 0;
    while index < header_chars.len() {
        match header_chars[index] {
            '<' => angles += 1,
            '>' => angles -= 1,
            _ if angles == 0 && ident_at(&header_chars, index, "for") => {
                let trait_name = last_ident_outside_angles(&header_chars[..index])?;
                let after = header[index + "for".len()..].trim();
                let type_name = normalize_ws(cut_where_clause(after));
                if type_name.is_empty() {
                    return None;
                }
                return Some(format!("{trait_name} for {type_name}"));
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn last_ident_outside_angles(chars: &[char]) -> Option<String> {
    let mut angles = 0i32;
    let mut index = 0;
    let mut last = None;
    while index < chars.len() {
        match chars[index] {
            '<' => {
                angles += 1;
                index += 1;
            }
            '>' => {
                angles -= 1;
                index += 1;
            }
            '\'' if angles == 0 => {
                index += 1;
                if let Some(name) = read_ident(chars, index) {
                    index += name.len();
                }
            }
            _ if angles == 0 && is_ident_start(chars[index]) => {
                let name = read_ident(chars, index)?;
                index += name.len();
                last = Some(name);
            }
            _ => index += 1,
        }
    }
    last
}

fn cut_where_clause(text: &str) -> &str {
    let chars: Vec<char> = text.chars().collect();
    let mut angles = 0i32;
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '<' => angles += 1,
            '>' => angles -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            _ if angles == 0
                && parens == 0
                && brackets == 0
                && ident_at(&chars, index, "where") =>
            {
                return text[..index].trim();
            }
            _ => {}
        }
        index += 1;
    }
    text.trim()
}

fn normalize_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn qualify_fn(impls: &[(i32, String)], depth: i32, name: &str) -> String {
    if let Some((_, ty)) = impls.iter().rev().find(|(body, _)| *body == depth) {
        format!("{ty}::{name}")
    } else {
        name.to_owned()
    }
}

fn at_item_pos(chars: &[char], index: usize) -> bool {
    let mut cursor = index;
    loop {
        while cursor > 0 && chars[cursor - 1].is_whitespace() {
            cursor -= 1;
        }
        if cursor == 0 {
            return true;
        }
        if chars[cursor - 1] == ']'
            && let Some(hash) = attribute_hash_before(chars, cursor - 1)
        {
            cursor = hash;
            continue;
        }
        // A comma separates fields. The second `pub` of a struct is an item.
        return matches!(chars[cursor - 1], ';' | '{' | '}' | ',');
    }
}

fn attribute_hash_before(chars: &[char], close: usize) -> Option<usize> {
    let mut depth = 1i32;
    let mut index = close;
    while index > 0 && depth > 0 {
        index -= 1;
        match chars[index] {
            ']' => depth += 1,
            '[' => depth -= 1,
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    if index > 0 && chars[index - 1] == '#' {
        Some(index - 1)
    } else {
        None
    }
}

fn parse_impl_header(chars: &[char], start: usize) -> Option<(usize, String)> {
    let mut index = start;
    let mut angles = 0i32;
    let mut parens = 0i32;
    let mut header = String::new();
    while index < chars.len() {
        let ch = chars[index];
        if ch == '"' {
            let (_, next) = take_string(chars, index);
            header.extend(&chars[index..next]);
            index = next;
            continue;
        }
        match ch {
            '<' => angles += 1,
            '>' => angles -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            ';' if angles == 0 && parens == 0 => return None,
            '{' if angles == 0 && parens == 0 => {
                return Some((index, impl_type_name(&header)));
            }
            _ => {}
        }
        header.push(ch);
        index += 1;
    }
    None
}

fn impl_type_name(header: &str) -> String {
    let body = split_impl_for(header).unwrap_or(header);
    first_type_ident(body).unwrap_or_else(|| "impl".to_owned())
}

fn split_impl_for(header: &str) -> Option<&str> {
    let chars: Vec<char> = header.chars().collect();
    let mut angles = 0i32;
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '<' => angles += 1,
            '>' => angles -= 1,
            _ if angles == 0 && ident_at(&chars, index, "for") => {
                return Some(header[index + "for".len()..].trim());
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn first_type_ident(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if chars[index].is_whitespace() || chars[index] == '&' {
            index += 1;
            continue;
        }
        if chars[index] == '\'' {
            index = skip_quoted(&chars, index);
            continue;
        }
        if chars[index] == '<' {
            index = skip_angles(&chars, index);
            continue;
        }
        if ident_at(&chars, index, "mut") {
            index += "mut".len();
            continue;
        }
        if let Some(name) = read_ident(&chars, index) {
            return Some(name);
        }
        return None;
    }
    None
}

fn skip_angles(chars: &[char], start: usize) -> usize {
    let mut depth = 0i32;
    let mut index = start;
    while index < chars.len() {
        match chars[index] {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return index;
                }
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    index
}

fn parse_pub_item(chars: &[char], pub_at: usize) -> Option<(String, Vec<String>, usize)> {
    let mut cursor = pub_at + "pub".len();
    cursor = skip_ws(chars, cursor);
    if cursor < chars.len() && chars[cursor] == '(' {
        cursor = skip_balanced(chars, cursor, '(', ')')? + 1;
    }
    loop {
        cursor = skip_ws(chars, cursor);
        if ident_at(chars, cursor, "unsafe") || ident_at(chars, cursor, "async") {
            let word = if ident_at(chars, cursor, "unsafe") {
                "unsafe"
            } else {
                "async"
            };
            cursor += word.len();
            continue;
        }
        if ident_at(chars, cursor, "const") {
            let after = skip_ws(chars, cursor + "const".len());
            if ident_at(chars, after, "fn") {
                cursor += "const".len();
                continue;
            }
        }
        break;
    }
    let kind = read_ident(chars, cursor)?;
    if !matches!(
        kind.as_str(),
        "fn" | "type" | "const" | "static" | "use" | "mod" | "struct" | "enum" | "trait" | "union"
    ) {
        return None;
    }
    cursor = skip_ws(chars, cursor + kind.len());
    if kind == "use" {
        let start = cursor;
        let mut end = cursor;
        let mut depth = 0i32;
        while end < chars.len() {
            match chars[end] {
                '{' => depth += 1,
                '}' => depth -= 1,
                ';' if depth == 0 => break,
                _ => {}
            }
            end += 1;
        }
        let body: String = chars[start..end].iter().collect();
        let names = bound_uses(&body)
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>();
        let next = if end < chars.len() { end + 1 } else { end };
        return Some((kind, names, next));
    }
    let name = read_ident(chars, cursor)?;
    Some((kind, vec![name.clone()], cursor + name.len()))
}

fn skip_balanced(chars: &[char], open: usize, left: char, right: char) -> Option<usize> {
    if chars.get(open) != Some(&left) {
        return None;
    }
    let mut depth = 0i32;
    let mut index = open;
    while index < chars.len() {
        if chars[index] == left {
            depth += 1;
        } else if chars[index] == right {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

/// `use std::fs; pub use fs` and `use rustix::fs as x; pub(crate) use x`.
/// The local name is recorded from the import, including brace groups and
/// `as` renames, then any later pub-visibility `use` of that name fails.
fn renamed_reexport_hits(rel: &str, code: &str) -> Vec<String> {
    let mut tainted: Vec<(String, &'static str)> = Vec::new();
    let mut globs: Vec<&'static str> = Vec::new();
    let mut reexports: Vec<Vec<String>> = Vec::new();
    for (item, reexport) in use_items(code) {
        for (path, name) in bound_uses(&item) {
            if reexport {
                reexports.push(path);
                continue;
            }
            let Some(source) = forbidden_fs_source(&path) else {
                continue;
            };
            if name == "*" || path.last().is_some_and(|part| part == "*") {
                globs.push(source);
            } else {
                tainted.push((name, source));
            }
        }
    }
    let mut hits = Vec::new();
    for path in reexports {
        let Some(root_name) = path.first() else {
            continue;
        };
        if let Some(source) = tainted
            .iter()
            .find(|(name, _)| name == root_name)
            .map(|(_, source)| *source)
        {
            hits.push(format!("{rel}: re-export {source}"));
        } else if let Some(source) = globs.first() {
            hits.push(format!("{rel}: re-export {source}"));
        }
    }
    hits
}

fn forbidden_fs_source(path: &[String]) -> Option<&'static str> {
    if path_starts(path, &["std", "fs"]) {
        Some("std::fs")
    } else if path_starts(path, &["rustix", "fs"]) {
        Some("rustix::fs")
    } else if path_starts(path, &["rustix", "io"]) {
        Some("rustix::io")
    } else if path_starts(path, &["rustix", "net"]) {
        Some("rustix::net")
    } else {
        None
    }
}

fn bound_uses(input: &str) -> Vec<(Vec<String>, String)> {
    bound_uses_prefixed(input, &[])
}

fn bound_uses_prefixed(input: &str, prefix: &[String]) -> Vec<(Vec<String>, String)> {
    let text = input.trim();
    if text.is_empty() {
        return Vec::new();
    }
    if text.starts_with('{') {
        let Some(inner) = brace_interior(text) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for part in split_top_commas(inner) {
            out.extend(bound_uses_prefixed(part, prefix));
        }
        return out;
    }
    let (head, alias) = split_binding_as(text);
    if let Some(brace_at) = head.find('{') {
        let path_head = head[..brace_at].trim().trim_end_matches("::").trim();
        let mut next = prefix.to_vec();
        next.extend(path_segments(path_head));
        let Some(inner) = brace_interior(&head[brace_at..]) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for part in split_top_commas(inner) {
            out.extend(bound_uses_prefixed(part, &next));
        }
        return out;
    }
    let mut path = prefix.to_vec();
    path.extend(path_segments(&head));
    if path.is_empty() {
        return Vec::new();
    }
    let name = alias.unwrap_or_else(|| path.last().cloned().unwrap_or_default());
    vec![(path, name)]
}

fn split_binding_as(text: &str) -> (String, Option<String>) {
    let Some(at) = text.rfind(" as ") else {
        return (text.trim().to_owned(), None);
    };
    if text[..at].contains('{') {
        return (text.trim().to_owned(), None);
    }
    let name = text[at + 4..].trim();
    if plain_ident(name) {
        (text[..at].trim().to_owned(), Some(name.to_owned()))
    } else {
        (text.trim().to_owned(), None)
    }
}

fn plain_ident(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if is_ident_start(first) => chars.all(is_ident_continue),
        _ => false,
    }
}

fn env_hits(rel: &str, code: &str) -> Vec<String> {
    let main = rel == "main.rs" || rel.ends_with("/main.rs");
    let mut hits = Vec::new();
    for name in env_qualified_names(code) {
        if env_name_banned(&name, main) {
            hits.push(format!("{rel}: std::env::{name}"));
        }
    }
    for (item, _) in use_items(code) {
        for (path, _) in bound_uses(&item) {
            if let Some(name) = env_import_banned(&path, main) {
                hits.push(format!("{rel}: std::env::{name}"));
            }
        }
    }
    hits
}

fn env_qualified_names(code: &str) -> Vec<String> {
    let chars: Vec<char> = code.chars().collect();
    let mut names = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, "std") {
            let mut cursor = skip_ws(&chars, index + "std".len());
            if consume_colons(&chars, cursor) {
                cursor = skip_ws(&chars, cursor + 2);
                if ident_at(&chars, cursor, "env") {
                    cursor = skip_ws(&chars, cursor + "env".len());
                    if consume_colons(&chars, cursor) {
                        cursor = skip_ws(&chars, cursor + 2);
                        if chars.get(cursor) == Some(&'{') {
                            if let Some(end) = matching_brace_at(&chars, cursor) {
                                let inner: String = chars[cursor + 1..end].iter().collect();
                                for (path, _) in bound_uses_prefixed(&inner, &[]) {
                                    if let Some(name) = path.first() {
                                        names.push(name.clone());
                                    }
                                }
                            }
                        } else if let Some(name) = read_ident(&chars, cursor) {
                            names.push(name);
                        }
                    }
                }
            }
        }
        index += 1;
    }
    names
}

fn env_import_banned(path: &[String], main: bool) -> Option<&'static str> {
    if path.len() == 2 && path[0] == "std" && path[1] == "env" {
        return Some("var");
    }
    if path_starts(path, &["std", "env"]) && path.len() >= 3 && env_name_banned(&path[2], main) {
        return Some(env_ban_label(&path[2]));
    }
    None
}

fn env_name_banned(name: &str, main: bool) -> bool {
    matches!(name, "var" | "var_os" | "vars" | "vars_os" | "*") || (name == "args" && !main)
}

fn env_ban_label(name: &str) -> &'static str {
    match name {
        "var_os" => "var_os",
        "vars" => "vars",
        "vars_os" => "vars_os",
        "args" => "args",
        "*" => "var",
        _ => "var",
    }
}

fn expand_use_tree(input: &str, prefix: &[String]) -> Vec<Vec<String>> {
    let text = input.trim();
    if text.is_empty() {
        return Vec::new();
    }
    if text.starts_with('{') {
        let Some(inner) = brace_interior(text) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for part in split_top_commas(inner) {
            out.extend(expand_use_tree(part, prefix));
        }
        return out;
    }
    let text = strip_rename(text);
    if let Some(brace_at) = text.find('{') {
        let head = text[..brace_at].trim().trim_end_matches("::").trim();
        let mut next = prefix.to_vec();
        next.extend(path_segments(head));
        let Some(inner) = brace_interior(&text[brace_at..]) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for part in split_top_commas(inner) {
            out.extend(expand_use_tree(part, &next));
        }
        return out;
    }
    let mut path = prefix.to_vec();
    path.extend(path_segments(text));
    if path.is_empty() {
        Vec::new()
    } else {
        vec![path]
    }
}

fn path_segments(text: &str) -> Vec<String> {
    text.split("::")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn strip_rename(text: &str) -> &str {
    let Some(at) = text.rfind(" as ") else {
        return text;
    };
    if text[..at].contains('{') {
        return text;
    }
    let name = text[at + 4..].trim();
    if !name.is_empty()
        && name.chars().all(is_ident_continue)
        && is_ident_start(name.chars().next().unwrap_or('_'))
    {
        return text[..at].trim();
    }
    text
}

fn brace_interior(text: &str) -> Option<&str> {
    let text = text.trim();
    let open = text.find('{')?;
    let end = matching_brace(text, open)?;
    Some(&text[open + 1..end])
}

fn split_top_commas(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn path_starts(path: &[String], prefix: &[&str]) -> bool {
    path.len() >= prefix.len()
        && prefix
            .iter()
            .zip(path.iter())
            .all(|(expect, got)| got == expect)
}

fn has_keyword_seq(code: &str, words: &[&str]) -> bool {
    let chars: Vec<char> = code.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, words[0]) {
            let mut cursor = index + words[0].len();
            let mut matched = true;
            for word in &words[1..] {
                cursor = skip_ws(&chars, cursor);
                if !ident_at(&chars, cursor, word) {
                    matched = false;
                    break;
                }
                cursor += word.len();
            }
            if matched {
                return true;
            }
        }
        index += 1;
    }
    false
}

fn has_absolute_name(code: &str, name: &str) -> bool {
    let needle = format!("::{name}");
    let bytes = code.as_bytes();
    let mut index = 0;
    while let Some(at) = code[index..].find(&needle) {
        let abs = index + at;
        let after = abs + needle.len();
        let boundary = after >= bytes.len() || !is_ident_continue(bytes[after] as char);
        if boundary {
            return true;
        }
        index = abs + 2;
    }
    false
}

fn literal_hits(rel: &str, code: &str, src_root: &Path, crate_root: &Path) -> Vec<String> {
    let mut hits = Vec::new();
    for literal in string_literals(code) {
        if literal.value.contains("snapshot.json") {
            hits.push(format!("{rel}: snapshot.json literal"));
        }
        if !path_has_parent(&literal.value) {
            continue;
        }
        if literal.include && include_under_assets(src_root, crate_root, rel, &literal.value) {
            continue;
        }
        hits.push(format!("{rel}: `..` in a path literal"));
    }
    hits
}

fn fs_hits(rel: &str, code: &str) -> Vec<String> {
    let mut hits = Vec::new();
    if contains_qualified(code, &["std", "fs"]) || contains_qualified(code, &["rustix", "fs"]) {
        hits.push(format!(
            "{rel}: {}",
            if contains_qualified(code, &["rustix", "fs"]) {
                "rustix::fs"
            } else {
                "std::fs"
            }
        ));
    }
    if contains_qualified(code, &["std", "fs"]) && contains_qualified(code, &["rustix", "fs"]) {
        hits.push(format!("{rel}: std::fs"));
    }
    if contains_qualified(code, &["rustix", "io"]) {
        hits.push(format!("{rel}: rustix::io"));
    }
    if contains_qualified(code, &["File", "open"]) {
        hits.push(format!("{rel}: File::open"));
    }
    if has_ident(code, "OpenOptions") {
        hits.push(format!("{rel}: OpenOptions"));
    }
    hits
}

fn config_path_hits(rel: &str, code: &str) -> Vec<String> {
    let lines: Vec<&str> = code.lines().collect();
    let mut hits = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let is_item = trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("enum ");
        if !is_item || !recent_deserialize(&lines, index) {
            continue;
        }
        let Some(open) = code_from_line(&lines, index).find('{') else {
            continue;
        };
        let body_src = code_from_line(&lines, index);
        let Some(end) = matching_brace(&body_src, open) else {
            continue;
        };
        let body = &body_src[open..=end];
        for field_line in body.lines() {
            let Some((name, ty)) = field_decl(field_line) else {
                continue;
            };
            if path_field_name(&name) {
                hits.push(format!("{rel}: config field {name} names a path"));
            } else if type_is_path(&ty) {
                hits.push(format!(
                    "{rel}: config field {name} has type {ty} (PathBuf)"
                ));
            }
        }
    }
    hits
}

fn code_from_line(lines: &[&str], index: usize) -> String {
    lines[index..].join("\n")
}

/// Attributes sitting immediately above an item. A previous item, which ends
/// in `;` or `}`, stops the walk so a nearby `Deserialize` cannot leak.
fn recent_deserialize(lines: &[&str], struct_at: usize) -> bool {
    let mut buf = Vec::new();
    let mut index = struct_at;
    while index > 0 {
        index -= 1;
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            if buf.is_empty() {
                continue;
            }
            break;
        }
        if trimmed.ends_with('}') || (trimmed.ends_with(';') && !trimmed.starts_with('#')) {
            break;
        }
        buf.push(trimmed);
    }
    let attrs = buf.join("\n");
    attrs.contains("derive") && attrs.contains("Deserialize")
}

fn field_decl(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim().trim_end_matches(',');
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed == "{" || trimmed == "}" {
        return None;
    }
    let rest = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
    let (name, after) = rest.split_once(':')?;
    let name = name.trim();
    if name.is_empty()
        || !name.chars().all(is_ident_continue)
        || !is_ident_start(name.chars().next()?)
    {
        return None;
    }
    let ty = after.trim().trim_end_matches(',').trim().to_owned();
    if ty.is_empty() {
        return None;
    }
    Some((name.to_owned(), ty))
}

fn path_field_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    const EXACT: &[&str] = &[
        "path", "dir", "file", "url", "uri", "addr", "address", "socket", "endpoint",
    ];
    if EXACT.contains(&name.as_str()) {
        return true;
    }
    [
        "_path",
        "_dir",
        "_file",
        "_url",
        "_uri",
        "_addr",
        "_socket",
        "_endpoint",
        "path_",
        "dir_",
        "file_",
        "url_",
    ]
    .iter()
    .any(|part| name.contains(part))
}

fn type_is_path(ty: &str) -> bool {
    ty.contains("PathBuf") || has_ident(ty, "Path")
}

fn sink_hits(rel: &str, code: &str) -> Vec<String> {
    let masked = mask_strings(code);
    let mut hits = Vec::new();
    let mut rest = masked.as_str();
    while let Some(at) = rest.find("LcdSink for") {
        let header = &rest[..at];
        if !header_has_impl(header) {
            rest = &rest[at + "LcdSink for".len()..];
            continue;
        }
        let after = &rest[at..];
        let Some(brace) = after.find('{') else {
            break;
        };
        let Some(end) = matching_brace(after, brace) else {
            break;
        };
        let body = &after[brace..=end];
        if !has_fn_blocked(body) {
            hits.push(format!("{rel}: LcdSink impl does not override fn blocked"));
        }
        rest = &after[end + 1..];
    }
    hits
}

fn header_has_impl(header: &str) -> bool {
    header.rsplit(['\n', ';', '}']).next().is_some_and(|tail| {
        tail.split_whitespace()
            .any(|word| word == "impl" || word.starts_with("impl<"))
    })
}

fn has_fn_blocked(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, "fn") {
            let cursor = skip_ws(&chars, index + 2);
            if ident_at(&chars, cursor, "blocked") {
                return true;
            }
        }
        index += 1;
    }
    false
}

fn include_under_assets(src_root: &Path, crate_root: &Path, rel: &str, literal: &str) -> bool {
    if literal.is_empty() || Path::new(literal).is_absolute() {
        return false;
    }
    let file = src_root.join(rel);
    let start = file.parent().unwrap_or(src_root);
    let normal = normalize(&start.join(literal));
    let assets = normalize(&crate_root.join("assets"));
    normal.starts_with(&assets) && normal != assets
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn path_has_parent(literal: &str) -> bool {
    literal.split(['/', '\\']).any(|segment| segment == "..")
}

struct Lit {
    value: String,
    include: bool,
}

fn string_literals(code: &str) -> Vec<Lit> {
    let chars: Vec<char> = code.chars().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '"' {
            let include = is_include_arg(&chars, index);
            let (value, next) = take_string(&chars, index);
            out.push(Lit { value, include });
            index = next;
            continue;
        }
        index += 1;
    }
    out
}

fn is_include_arg(chars: &[char], quote: usize) -> bool {
    let before: String = chars[..quote].iter().collect();
    let before = before.trim_end();
    before.ends_with("include_bytes!(") || before.ends_with("include_str!(")
}

fn take_string(chars: &[char], quote: usize) -> (String, usize) {
    if let Some(hashes) = raw_hashes_before(chars, quote) {
        return take_raw_body(chars, quote, hashes);
    }
    let mut value = String::new();
    let mut index = quote + 1;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '\\' {
            value.push(ch);
            index += 1;
            if index < chars.len() {
                value.push(chars[index]);
                index += 1;
            }
            continue;
        }
        if ch == '"' {
            return (value, index + 1);
        }
        value.push(ch);
        index += 1;
    }
    (value, chars.len())
}

/// Hashes of a raw string whose opening quote is `quote`. `r"`, `r#"…"#`,
/// `br"…"`, `br#"…"#`, and the `c` prefix, with any hash count.
fn raw_hashes_of_output(out: &str) -> Option<usize> {
    let bytes = out.as_bytes();
    let mut index = bytes.len();
    let mut hashes = 0usize;
    while index > 0 && bytes[index - 1] == b'#' {
        hashes += 1;
        index -= 1;
    }
    if index == 0 || bytes[index - 1] != b'r' {
        return None;
    }
    let r_at = index - 1;
    if r_at > 0 {
        let prev = bytes[r_at - 1] as char;
        if prev == 'b' || prev == 'c' {
            if r_at >= 2 && is_ident_continue(bytes[r_at - 2] as char) {
                return None;
            }
        } else if is_ident_continue(prev) {
            return None;
        }
    }
    Some(hashes)
}

fn copy_raw_body(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    out: &mut String,
    hashes: usize,
) {
    while let Some(next) = chars.next() {
        if next != '"' {
            out.push(next);
            continue;
        }
        let mut marks = String::new();
        let mut closed = true;
        for _ in 0..hashes {
            match chars.next() {
                Some('#') => marks.push('#'),
                Some(other) => {
                    marks.push(other);
                    closed = false;
                    break;
                }
                None => {
                    closed = false;
                    break;
                }
            }
        }
        out.push('"');
        out.push_str(&marks);
        if closed {
            break;
        }
    }
}

fn raw_hashes_before(chars: &[char], quote: usize) -> Option<usize> {
    let mut index = quote;
    let mut hashes = 0usize;
    while index > 0 && chars[index - 1] == '#' {
        hashes += 1;
        index -= 1;
    }
    if index == 0 || chars[index - 1] != 'r' {
        return None;
    }
    let r_at = index - 1;
    if r_at > 0 {
        let prev = chars[r_at - 1];
        if prev == 'b' || prev == 'c' {
            if r_at >= 2 && is_ident_continue(chars[r_at - 2]) {
                return None;
            }
        } else if is_ident_continue(prev) {
            return None;
        }
    }
    Some(hashes)
}

fn take_raw_body(chars: &[char], quote: usize, hashes: usize) -> (String, usize) {
    let mut index = quote + 1;
    let mut value = String::new();
    while index < chars.len() {
        if chars[index] == '"' && hash_run(chars, index + 1, hashes) {
            return (value, index + 1 + hashes);
        }
        value.push(chars[index]);
        index += 1;
    }
    (value, chars.len())
}

fn hash_run(chars: &[char], start: usize, hashes: usize) -> bool {
    start + hashes <= chars.len() && chars[start..start + hashes].iter().all(|ch| *ch == '#')
}

fn mask_strings(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(&chars, index);
            for ch in &chars[index..next] {
                out.push(if *ch == '\n' { '\n' } else { ' ' });
            }
            index = next;
            continue;
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn has_std_net(code: &str) -> bool {
    contains_qualified(code, &["std", "net"])
}

fn contains_qualified(code: &str, parts: &[&str]) -> bool {
    let chars: Vec<char> = code.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, parts[0]) {
            let mut cursor = index + parts[0].len();
            let mut matched = true;
            for part in &parts[1..] {
                cursor = skip_ws(&chars, cursor);
                if !consume_colons(&chars, cursor) {
                    matched = false;
                    break;
                }
                cursor += 2;
                cursor = skip_ws(&chars, cursor);
                if !ident_at(&chars, cursor, part) {
                    matched = false;
                    break;
                }
                cursor += part.len();
            }
            if matched {
                return true;
            }
        }
        index += 1;
    }
    false
}

fn has_ident(code: &str, name: &str) -> bool {
    let chars: Vec<char> = code.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, name) {
            return true;
        }
        index += 1;
    }
    false
}

fn ident_at(chars: &[char], index: usize, name: &str) -> bool {
    let name: Vec<char> = name.chars().collect();
    if index + name.len() > chars.len() {
        return false;
    }
    if chars[index..index + name.len()] != name[..] {
        return false;
    }
    if index > 0 && is_ident_continue(chars[index - 1]) {
        return false;
    }
    let end = index + name.len();
    if end < chars.len() && is_ident_continue(chars[end]) {
        return false;
    }
    true
}

fn skip_ws(chars: &[char], mut index: usize) -> usize {
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    index
}

fn consume_colons(chars: &[char], index: usize) -> bool {
    index + 1 < chars.len() && chars[index] == ':' && chars[index + 1] == ':'
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn outside_hits(rel: &str, code: &str, src_root: &Path, crate_root: &Path) -> Vec<String> {
    let chars: Vec<char> = code.chars().collect();
    let mut hits = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(&chars, index);
            index = next;
            continue;
        }
        if chars[index] == '\'' {
            index = skip_quoted(&chars, index);
            continue;
        }
        if macro_bang(&chars, index, "include") {
            hits.push(format!("{rel}: include!"));
            index += "include".len();
            continue;
        }
        if macro_bang(&chars, index, "include_bytes") {
            hits.extend(include_macro_hits(
                rel,
                "include_bytes",
                &chars,
                index,
                src_root,
                crate_root,
            ));
            index += "include_bytes".len();
            continue;
        }
        if macro_bang(&chars, index, "include_str") {
            hits.extend(include_macro_hits(
                rel,
                "include_str",
                &chars,
                index,
                src_root,
                crate_root,
            ));
            index += "include_str".len();
            continue;
        }
        if macro_bang(&chars, index, "option_env") {
            hits.push(format!("{rel}: option_env!"));
            index += "option_env".len();
            continue;
        }
        if macro_bang(&chars, index, "env") {
            hits.push(format!("{rel}: env!"));
            index += "env".len();
            continue;
        }
        if macro_bang(&chars, index, "macro_rules") {
            hits.push(format!("{rel}: macro_rules!"));
            index += "macro_rules".len();
            continue;
        }
        if chars[index] == '#'
            && let Some(end) = attribute_end(&chars, index)
        {
            if attribute_sets_path(&chars[index..end]) {
                hits.push(format!("{rel}: #[path"));
            }
            index = end;
            continue;
        }
        index += 1;
    }
    hits
}

fn macro_bang(chars: &[char], index: usize, name: &str) -> bool {
    if !ident_at(chars, index, name) {
        return false;
    }
    let bang = skip_ws(chars, index + name.len());
    bang < chars.len() && chars[bang] == '!'
}

fn include_macro_hits(
    rel: &str,
    name: &str,
    chars: &[char],
    index: usize,
    src_root: &Path,
    crate_root: &Path,
) -> Vec<String> {
    let mut hits = Vec::new();
    let bang = skip_ws(chars, index + name.len());
    let mut cursor = skip_ws(chars, bang + 1);
    if bang >= chars.len() || chars[bang] != '!' || cursor >= chars.len() || chars[cursor] != '(' {
        hits.push(format!("{rel}: {name}! argument"));
        return hits;
    }
    cursor = skip_ws(chars, cursor + 1);
    if include_arg_is_macro(chars, cursor, "concat") {
        hits.push(format!("{rel}: {name}! concat!"));
        return hits;
    }
    if include_arg_is_macro(chars, cursor, "env")
        || include_arg_is_macro(chars, cursor, "option_env")
    {
        hits.push(format!("{rel}: {name}! include argument"));
        return hits;
    }
    match take_include_literal(chars, cursor) {
        Some((value, _)) if Path::new(&value).is_absolute() => {
            hits.push(format!("{rel}: {name}! absolute"));
        }
        Some((value, _)) if !include_under_assets(src_root, crate_root, rel, &value) => {
            hits.push(format!("{rel}: {name}! outside assets"));
        }
        Some(_) => {}
        None => hits.push(format!("{rel}: {name}! argument")),
    }
    hits
}

fn include_arg_is_macro(chars: &[char], index: usize, name: &str) -> bool {
    if !ident_at(chars, index, name) {
        return false;
    }
    let bang = skip_ws(chars, index + name.len());
    bang < chars.len() && chars[bang] == '!'
}

fn take_include_literal(chars: &[char], index: usize) -> Option<(String, usize)> {
    if index >= chars.len() {
        return None;
    }
    if chars[index] == 'r' {
        return take_raw_string(chars, index);
    }
    if chars[index] != '"' {
        return None;
    }
    let (raw, next) = take_string(chars, index);
    Some((unescape_rust(&raw)?, next))
}

fn take_raw_string(chars: &[char], index: usize) -> Option<(String, usize)> {
    let mut cursor = index + 1;
    let mut hashes = 0usize;
    while cursor < chars.len() && chars[cursor] == '#' {
        hashes += 1;
        cursor += 1;
    }
    if cursor >= chars.len() || chars[cursor] != '"' {
        return None;
    }
    cursor += 1;
    let start = cursor;
    while cursor < chars.len() {
        if chars[cursor] == '"' {
            let tail = &chars[cursor + 1..];
            if tail.len() >= hashes && tail[..hashes].iter().all(|ch| *ch == '#') {
                let value: String = chars[start..cursor].iter().collect();
                return Some((value, cursor + 1 + hashes));
            }
        }
        cursor += 1;
    }
    None
}

fn unescape_rust(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next()? {
            '\\' => out.push('\\'),
            '"' => out.push('"'),
            '\'' => out.push('\''),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '0' => out.push('\0'),
            'u' => {
                if chars.next() != Some('{') {
                    return None;
                }
                let mut hex = String::new();
                for h in chars.by_ref() {
                    if h == '}' {
                        break;
                    }
                    hex.push(h);
                }
                let code = u32::from_str_radix(&hex, 16).ok()?;
                out.push(char::from_u32(code)?);
            }
            'x' => {
                let mut hex = String::new();
                hex.push(chars.next()?);
                hex.push(chars.next()?);
                let code = u32::from_str_radix(&hex, 16).ok()?;
                out.push(char::from_u32(code)?);
            }
            other => out.push(other),
        }
    }
    Some(out)
}

fn attribute_end(chars: &[char], hash: usize) -> Option<usize> {
    let mut index = skip_ws(chars, hash + 1);
    if index >= chars.len() || chars[index] != '[' {
        return None;
    }
    let mut depth = 0i32;
    while index < chars.len() {
        match chars[index] {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            '"' => {
                let (_, next) = take_string(chars, index);
                index = next;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn attribute_sets_path(chars: &[char]) -> bool {
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(chars, index);
            index = next;
            continue;
        }
        if ident_at(chars, index, "path") {
            let after = skip_ws(chars, index + "path".len());
            if after < chars.len() && chars[after] == '=' {
                return true;
            }
            index += "path".len();
            continue;
        }
        index += 1;
    }
    false
}

fn process_hits(rel: &str, code: &str) -> Vec<String> {
    let main = rel == "main.rs" || rel.ends_with("/main.rs");
    if process_disallowed(code, main) {
        vec![format!("{rel}: std::process")]
    } else {
        Vec::new()
    }
}

fn process_disallowed(code: &str, main: bool) -> bool {
    let chars: Vec<char> = code.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if ident_at(&chars, index, "std") {
            let mut cursor = skip_ws(&chars, index + "std".len());
            if consume_colons(&chars, cursor) {
                cursor = skip_ws(&chars, cursor + 2);
                if ident_at(&chars, cursor, "process")
                    && !process_tail_allowed(&chars, cursor, main)
                {
                    return true;
                }
            }
        }
        index += 1;
    }
    false
}

fn process_tail_allowed(chars: &[char], process_at: usize, main: bool) -> bool {
    let mut cursor = skip_ws(chars, process_at + "process".len());
    if !consume_colons(chars, cursor) {
        return false;
    }
    cursor = skip_ws(chars, cursor + 2);
    if cursor < chars.len() && chars[cursor] == '{' {
        if !main {
            return false;
        }
        let Some(end) = matching_brace_at(chars, cursor) else {
            return false;
        };
        let inner: String = chars[cursor + 1..end].iter().collect();
        return process_names_allowed(&inner);
    }
    let Some(name) = read_ident(chars, cursor) else {
        return false;
    };
    main && (name == "exit" || name == "ExitCode")
}

fn process_names_allowed(inner: &str) -> bool {
    let mut saw = false;
    for part in split_top_commas(inner) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let name = part.split_whitespace().next().unwrap_or("");
        if name != "exit" && name != "ExitCode" {
            return false;
        }
        saw = true;
    }
    saw
}

fn matching_brace_at(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (offset, ch) in chars[open..].iter().enumerate() {
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

fn read_ident(chars: &[char], index: usize) -> Option<String> {
    if index >= chars.len() || !is_ident_start(chars[index]) {
        return None;
    }
    let mut end = index + 1;
    while end < chars.len() && is_ident_continue(chars[end]) {
        end += 1;
    }
    Some(chars[index..end].iter().collect())
}

/// Any spelling of `OpenRequest` outside `service.rs`, `main.rs`, and `device/`
/// is a caller-chosen sysfs root or state directory. Aliases and turbofish
/// still contain the name.
fn open_request_hits(rel: &str, code: &str) -> Vec<String> {
    if rel == "service.rs"
        || rel == "main.rs"
        || rel.starts_with("device/")
        || rel.ends_with("/service.rs")
        || rel.ends_with("/main.rs")
        || rel.contains("/device/")
    {
        return Vec::new();
    }
    if has_ident(code, "OpenRequest") {
        vec![format!("{rel}: OpenRequest")]
    } else {
        Vec::new()
    }
}

/// `SnapshotReader` is only named in its definition file and in `service.rs`
/// (`open` reads the snapshot constant). `main.rs` does not name it.
fn snapshot_new_hits(rel: &str, code: &str) -> Vec<String> {
    if rel == "snapshot_reader.rs"
        || rel == "service.rs"
        || rel.ends_with("/snapshot_reader.rs")
        || rel.ends_with("/service.rs")
    {
        return Vec::new();
    }
    if has_ident(code, "SnapshotReader") {
        vec![format!("{rel}: SnapshotReader")]
    } else {
        Vec::new()
    }
}

/// `open_resolved_hid` opens the path it is given. Only `service.rs` and
/// `device/` may name it; `device/` passes the node `pre_open` resolved.
fn resolved_hid_hits(rel: &str, code: &str) -> Vec<String> {
    if rel == "service.rs"
        || rel.starts_with("device/")
        || rel.ends_with("/service.rs")
        || rel.contains("/device/")
    {
        return Vec::new();
    }
    if has_ident(code, "open_resolved_hid") {
        vec![format!("{rel}: open_resolved_hid")]
    } else {
        Vec::new()
    }
}

/// `HidLink::open` is only named under `device/`. Callers outside that module
/// go through the pre-open wrappers.
fn hid_open_hits(rel: &str, code: &str) -> Vec<String> {
    if rel.starts_with("device/") || rel.contains("/device/") {
        return Vec::new();
    }
    if contains_qualified(code, &["HidLink", "open"]) {
        vec![format!("{rel}: HidLink::open")]
    } else {
        Vec::new()
    }
}

/// `guard::state_dir_writable` (a `W_OK` probe) is only named under
/// `device/`. `CheckEnv.state_dir_writable` is a field of the same name, so a
/// field declaration, initialiser (`name:` but not `name::`) or access
/// (`.name`) is not a hit.
fn state_dir_writable_hits(rel: &str, code: &str) -> Vec<String> {
    const NAME: &str = "state_dir_writable";
    if rel.starts_with("device/") || rel.contains("/device/") {
        return Vec::new();
    }
    let chars: Vec<char> = code.chars().collect();
    let len = NAME.chars().count();
    for index in 0..chars.len() {
        if !ident_at(&chars, index, NAME) {
            continue;
        }
        let mut before = index;
        while before > 0 && chars[before - 1].is_whitespace() {
            before -= 1;
        }
        let field_access = before > 0 && chars[before - 1] == '.';
        let after = skip_ws(&chars, index + len);
        let field_colon = chars.get(after) == Some(&':') && chars.get(after + 1) != Some(&':');
        if !field_access && !field_colon {
            return vec![format!("{rel}: {NAME}")];
        }
    }
    Vec::new()
}

/// `CoolingGuard::new` is built on the LCD open path (`device/mod.rs`).
/// `service.rs` and `main.rs` are allowed; every other scanned file fails.
/// The real constructors live in `device/mod.rs`, so that file is allowed too.
fn cooling_guard_new_hits(rel: &str, code: &str) -> Vec<String> {
    if rel == "device/mod.rs"
        || rel == "service.rs"
        || rel == "main.rs"
        || rel.ends_with("/device/mod.rs")
        || rel.ends_with("/service.rs")
        || rel.ends_with("/main.rs")
    {
        return Vec::new();
    }
    if contains_qualified(code, &["CoolingGuard", "new"]) {
        vec![format!("{rel}: CoolingGuard::new")]
    } else {
        Vec::new()
    }
}

fn load_validated_hits(rel: &str, code: &str) -> Vec<String> {
    let service_or_main = rel == "service.rs"
        || rel == "main.rs"
        || rel.ends_with("/service.rs")
        || rel.ends_with("/main.rs");
    let config = rel == "config.rs" || rel.ends_with("/config.rs");
    if !service_or_main && !config {
        if has_ident(code, "load_validated") {
            return vec![format!("{rel}: load_validated")];
        }
        return Vec::new();
    }
    if config {
        return Vec::new();
    }
    let chars: Vec<char> = code.chars().collect();
    let mut hits = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(&chars, index);
            index = next;
            continue;
        }
        if ident_at(&chars, index, "load_validated") && !preceded_by_fn(&chars, index) {
            let mut cursor = skip_ws(&chars, index + "load_validated".len());
            if consume_colons(&chars, cursor) {
                let after = skip_ws(&chars, cursor + 2);
                if after < chars.len()
                    && chars[after] == '<'
                    && let Some(end) = matching_angle(&chars, after)
                {
                    cursor = skip_ws(&chars, end + 1);
                }
            }
            if cursor < chars.len() && chars[cursor] == '(' {
                let arg_ok = matching_paren(&chars, cursor).is_some_and(|end| {
                    let arg: String = chars[cursor + 1..end].iter().collect();
                    cli_path_arg(&arg)
                });
                if !arg_ok {
                    hits.push(format!("{rel}: load_validated"));
                }
                if let Some(end) = matching_paren(&chars, cursor) {
                    index = end + 1;
                    continue;
                }
            }
        }
        index += 1;
    }
    hits
}

fn preceded_by_fn(chars: &[char], index: usize) -> bool {
    let mut cursor = index;
    while cursor > 0 && chars[cursor - 1].is_whitespace() {
        cursor -= 1;
    }
    cursor >= 2 && ident_at(chars, cursor - 2, "fn")
}

fn cli_path_arg(arg: &str) -> bool {
    let arg = arg.trim();
    let arg = arg.strip_prefix('&').unwrap_or(arg).trim();
    let mut chars = arg.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    is_ident_start(first) && chars.all(is_ident_continue)
}

fn matching_paren(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut index = open;
    while index < chars.len() {
        if chars[index] == '"' {
            let (_, next) = take_string(chars, index);
            index = next;
            continue;
        }
        match chars[index] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn matching_angle(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut index = open;
    while index < chars.len() {
        match chars[index] {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn skip_quoted(chars: &[char], quote: usize) -> usize {
    if quote + 1 >= chars.len() {
        return chars.len();
    }
    if chars[quote + 1] == '\\' {
        let mut index = quote + 2;
        if index < chars.len()
            && chars[index] == 'u'
            && index + 1 < chars.len()
            && chars[index + 1] == '{'
        {
            index += 2;
            while index < chars.len() && chars[index] != '}' {
                index += 1;
            }
            index = (index + 1).min(chars.len());
        } else if index < chars.len() && chars[index] == 'x' {
            index = (index + 3).min(chars.len());
        } else {
            index = (index + 1).min(chars.len());
        }
        if index < chars.len() && chars[index] == '\'' {
            return index + 1;
        }
        return index;
    }
    if quote + 2 < chars.len() && chars[quote + 2] == '\'' {
        return quote + 3;
    }
    if is_ident_start(chars[quote + 1]) {
        let mut index = quote + 2;
        while index < chars.len() && is_ident_continue(chars[index]) {
            index += 1;
        }
        return index;
    }
    quote + 2
}

/// Block comments are deleted, so `std::/**/net` stays one path. Line comments
/// stop at the newline. String bodies are kept. A char literal is one scalar
/// (or one escape) between quotes. A lifetime (`'a`, `'static`) has no closing
/// quote, so it is copied as an ident and cannot swallow the next token.
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
            continue;
        }
        if ch == '\'' {
            push_quote(&mut chars, &mut out);
            continue;
        }
        if ch == '"' {
            if let Some(hashes) = raw_hashes_of_output(&out) {
                out.push('"');
                copy_raw_body(&mut chars, &mut out, hashes);
                continue;
            }
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

fn push_quote(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, out: &mut String) {
    out.push('\'');
    let Some(next) = chars.next() else {
        return;
    };
    if next == '\\' {
        out.push(next);
        push_char_escape(chars, out);
        if let Some(close) = chars.next_if(|ch| *ch == '\'') {
            out.push(close);
        }
        return;
    }
    if chars.peek() == Some(&'\'') {
        out.push(next);
        if let Some(close) = chars.next() {
            out.push(close);
        }
        return;
    }
    if is_ident_start(next) {
        out.push(next);
        while let Some(cont) = chars.next_if(|ch| is_ident_continue(*ch)) {
            out.push(cont);
        }
        return;
    }
    out.push(next);
}

fn push_char_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, out: &mut String) {
    let Some(kind) = chars.next() else {
        return;
    };
    out.push(kind);
    if kind == 'u' && chars.peek() == Some(&'{') {
        if let Some(open) = chars.next() {
            out.push(open);
        }
        for ch in chars.by_ref() {
            out.push(ch);
            if ch == '}' {
                break;
            }
        }
        return;
    }
    if kind == 'x' {
        for _ in 0..2 {
            if let Some(hex) = chars.next() {
                out.push(hex);
            }
        }
    }
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
