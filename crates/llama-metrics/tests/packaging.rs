//! The unit and the example config agree, and the unit keeps its sandbox.
//! scripts/stage.sh pins the unit's full directive set.

mod common;

use std::collections::BTreeSet;

use llama_metrics::config::{Config, DEFAULT_PORT};

fn unit() -> String {
    std::fs::read_to_string(common::workspace_root().join("packaging/llama-metrics.service"))
        .expect("unit")
}

fn example() -> Config {
    let text =
        std::fs::read_to_string(common::workspace_root().join("packaging/metrics.example.toml"))
            .expect("example");
    Config::from_toml(&text).expect("example parses")
}

fn directives(unit: &str) -> Vec<(String, String)> {
    unit.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('['))
        .map(|l| {
            let (k, v) = l.split_once('=').expect("key=value");
            (k.to_owned(), v.to_owned())
        })
        .collect()
}

fn values(unit: &str, key: &str) -> Vec<String> {
    directives(unit)
        .into_iter()
        .filter(|(k, _)| k == key)
        .map(|(_, v)| v)
        .collect()
}

#[test]
fn ip_address_allow_equals_the_config_allowlist() {
    let unit = unit();
    let from_unit: BTreeSet<String> = values(&unit, "IPAddressAllow")
        .iter()
        .flat_map(|v| v.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
        .collect();
    let from_config: BTreeSet<String> = example()
        .allow
        .nets()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        from_unit, from_config,
        "IPAddressAllow= and metrics.example.toml `allow` must list the same networks"
    );
    assert_eq!(values(&unit, "IPAddressDeny"), ["any"]);
}

#[test]
fn socket_bind_allow_equals_the_listen_port() {
    let unit = unit();
    let cfg = example();
    assert_eq!(cfg.listen.port(), DEFAULT_PORT);
    assert_eq!(
        values(&unit, "SocketBindAllow"),
        [format!("tcp:{}", cfg.listen.port())]
    );
    assert_eq!(values(&unit, "SocketBindDeny"), ["any"]);
}

#[test]
fn unit_is_sandboxed_and_device_free() {
    let unit = unit();
    for (key, want) in [
        ("User", "llama-metrics"),
        ("Group", "llama-metrics"),
        ("SupplementaryGroups", "llama-watch"),
        ("DevicePolicy", "closed"),
        ("PrivateDevices", "yes"),
        ("InaccessiblePaths", "/sys"),
        ("ProtectSystem", "strict"),
        ("ProtectHome", "yes"),
        ("ProcSubset", "pid"),
        ("ProtectProc", "invisible"),
        ("NoNewPrivileges", "yes"),
        ("CapabilityBoundingSet", ""),
        ("AmbientCapabilities", ""),
        ("RestrictAddressFamilies", "AF_INET AF_INET6 AF_UNIX"),
        ("MemoryDenyWriteExecute", "yes"),
        ("Type", "notify"),
        ("WatchdogSec", "10"),
        (
            "ExecStart",
            "/usr/local/libexec/llama-bored/llama-metrics run --config /etc/llama-bored/metrics.toml",
        ),
    ] {
        assert_eq!(values(&unit, key), [want], "{key}");
    }
    assert!(values(&unit, "SystemCallFilter").contains(&"@system-service".to_owned()));
    for absent in [
        "DeviceAllow",
        "ReadWritePaths",
        "ReadOnlyPaths",
        "BindPaths",
        "StateDirectory",
        "RuntimeDirectory",
        "PrivateNetwork",
        "ExecStartPre",
        "ExecStopPost",
    ] {
        assert!(
            values(&unit, absent).is_empty(),
            "{absent} in the metrics unit"
        );
    }
    let lower = unit.to_ascii_lowercase();
    for word in ["hidraw", "1e71", "i2c", "usb", "nvidia", "tty"] {
        assert!(!lower.contains(word), "the metrics unit names {word}");
    }
}

#[test]
fn example_config_says_it_is_lan_exposed_and_textless() {
    let text =
        std::fs::read_to_string(common::workspace_root().join("packaging/metrics.example.toml"))
            .unwrap();
    assert!(text.contains("LAN-exposed by design"));
    assert!(text.contains("no prompt or output text"));
    assert!(text.contains("IPAddressAllow="));
}

#[test]
fn sysusers_has_the_metrics_user_and_no_extra_groups() {
    let text =
        std::fs::read_to_string(common::workspace_root().join("packaging/llama-bored.sysusers"))
            .unwrap();
    assert!(
        text.lines()
            .any(|l| l == "u llama-metrics - \"Llama Metrics exporter\" / /usr/sbin/nologin")
    );
    assert!(
        !text
            .lines()
            .any(|l| l.starts_with('m') && l.contains("llama-metrics")),
        "group membership comes from SupplementaryGroups= only"
    );
}
