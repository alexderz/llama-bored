//! The unit and the example config agree, and the unit keeps its sandbox.
//! scripts/stage.sh pins the unit's full directive set.

mod common;

use std::collections::BTreeSet;

use llama_cast::config::{Config, DEFAULT_PORT, SSDP_PORT};

const GROUP_CIDR: &str = "239.255.255.250/32";

fn unit() -> String {
    std::fs::read_to_string(common::workspace_root().join("packaging/llama-cast.service"))
        .expect("unit")
}

fn example_text() -> String {
    std::fs::read_to_string(common::workspace_root().join("packaging/cast.example.toml"))
        .expect("example")
}

fn example() -> Config {
    Config::from_toml(&example_text()).expect("example parses")
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
fn ip_address_allow_is_the_allowlist_plus_the_ssdp_group() {
    let unit = unit();
    let from_unit: BTreeSet<String> = values(&unit, "IPAddressAllow")
        .iter()
        .flat_map(|v| v.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
        .collect();
    let mut want: BTreeSet<String> = example()
        .allow
        .nets()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert!(!want.contains(GROUP_CIDR));
    want.insert(GROUP_CIDR.to_owned());
    assert_eq!(
        from_unit, want,
        "IPAddressAllow= must be cast.example.toml `allow` plus {GROUP_CIDR}"
    );
    assert_eq!(values(&unit, "IPAddressDeny"), ["any"]);
}

#[test]
fn socket_bind_allow_is_the_listen_port_and_ssdp() {
    let unit = unit();
    let cfg = example();
    assert_eq!(cfg.listen.port(), DEFAULT_PORT);
    assert_eq!(
        values(&unit, "SocketBindAllow"),
        [
            format!("tcp:{}", cfg.listen.port()),
            format!("udp:{SSDP_PORT}")
        ]
    );
    assert_eq!(values(&unit, "SocketBindDeny"), ["any"]);
}

#[test]
fn unit_is_sandboxed_and_reads_only_vcsa11() {
    let unit = unit();
    for (key, want) in [
        ("User", "llama-cast"),
        ("Group", "llama-cast"),
        ("SupplementaryGroups", "llama-view"),
        ("DevicePolicy", "closed"),
        ("DeviceAllow", "/dev/vcsa11 r"),
        ("InaccessiblePaths", "/sys"),
        ("ProtectSystem", "strict"),
        ("ProtectHome", "yes"),
        ("PrivateTmp", "yes"),
        ("ProcSubset", "pid"),
        ("ProtectProc", "invisible"),
        ("NoNewPrivileges", "yes"),
        ("CapabilityBoundingSet", ""),
        ("AmbientCapabilities", ""),
        ("RestrictAddressFamilies", "AF_INET AF_UNIX"),
        ("MemoryDenyWriteExecute", "yes"),
        ("Type", "notify"),
        ("WatchdogSec", "10"),
        ("RestartPreventExitStatus", "2"),
        (
            "ExecStart",
            "/usr/local/libexec/llama-bored/llama-cast run --config /etc/llama-bored/cast.toml",
        ),
        (
            "ExecStop",
            "/usr/local/libexec/llama-bored/llama-cast bye --config /etc/llama-bored/cast.toml",
        ),
    ] {
        assert_eq!(values(&unit, key), [want], "{key}");
    }
    assert!(values(&unit, "SystemCallFilter").contains(&"@system-service".to_owned()));
    for absent in [
        "PrivateDevices",
        "ReadWritePaths",
        "ReadOnlyPaths",
        "BindPaths",
        "BindReadOnlyPaths",
        "StateDirectory",
        "RuntimeDirectory",
        "PrivateNetwork",
        "ExecStartPre",
        "ExecStopPost",
        "Environment",
        "EnvironmentFile",
    ] {
        assert!(
            values(&unit, absent).is_empty(),
            "{absent} in the cast unit"
        );
    }
    let lower = unit.to_ascii_lowercase();
    for word in [
        "hidraw", "1e71", "i2c", "usb", "nvidia", "tty0", "vcsa1 ", "char-",
    ] {
        assert!(!lower.contains(word), "the cast unit names {word}");
    }
}

#[test]
fn example_config_says_it_is_lan_exposed() {
    let text = example_text();
    assert!(text.contains("LAN-exposed by design"));
    assert!(text.contains("no prompt or output text"));
    assert!(text.contains("IPAddressAllow="));
    assert!(text.contains("CHANGE THIS"));
    let cfg = example();
    assert_eq!(cfg.name, "llama-bored");
    assert_eq!(cfg.fps, 2);
    assert_eq!(cfg.max_clients, 2);
    assert_eq!(cfg.bitrate_kbps, 4000);
    assert_eq!(cfg.keyframe_s, 1);
    assert_eq!(cfg.preroll_s, 3);
    assert_eq!(cfg.ffmpeg.to_str(), Some("/usr/bin/ffmpeg"));
}

#[test]
fn sysusers_has_the_cast_user_and_no_extra_groups() {
    let text =
        std::fs::read_to_string(common::workspace_root().join("packaging/llama-bored.sysusers"))
            .unwrap();
    assert!(
        text.lines()
            .any(|l| l == "u llama-cast - \"Llama Cast DLNA streamer\" / /usr/sbin/nologin")
    );
    assert!(
        !text
            .lines()
            .any(|l| l.starts_with('m') && l.contains("llama-cast")),
        "group membership comes from SupplementaryGroups= only"
    );
}

#[test]
fn the_vcsa_rule_grants_llama_view_read_only() {
    let rule =
        std::fs::read_to_string(common::workspace_root().join("packaging/72-llama-view.rules"))
            .unwrap();
    assert!(rule.contains("KERNEL==\"vcsa11|vcs11|vcsu11\", GROUP=\"llama-view\", MODE=\"0640\""));
}
