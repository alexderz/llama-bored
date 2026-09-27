//! Unit and udev-rule goldens for llama-light. scripts/stage.sh pins the
//! same text (with planted-directive checks); this keeps `cargo test` honest.

fn packaging(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../../packaging/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect(name)
}

/// Directives only: comments and blank lines dropped, trailing space trimmed.
fn directives(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim_end)
        .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

const UNIT_GOLDEN: &str = "\
[Unit]
Description=Llama Light - RGB lighting from the llama-watch snapshot (colour only, never cooling)
StartLimitIntervalSec=600
StartLimitBurst=5
[Service]
Type=notify
NotifyAccess=main
User=llama-light
Group=llama-light
SupplementaryGroups=llama-watch
ExecStart=/usr/local/libexec/llama-bored/llama-light run --config /etc/llama-bored/light.toml
ExecStopPost=-/usr/local/libexec/llama-bored/llama-light restore --config /etc/llama-bored/light.toml
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=10
WatchdogSec=10
TimeoutStopSec=10
UMask=0077
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
RemoveIPC=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
ReadOnlyPaths=/sys
DevicePolicy=closed
DeviceAllow=/dev/llama-light/aura rw
PrivateNetwork=yes
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
[Install]
WantedBy=multi-user.target";

const RULE_GOLDEN: &str = r#"SUBSYSTEM=="hidraw", KERNELS=="0003:0B05:18F3.*", GROUP="llama-light", MODE="0660", TAG-="uaccess", TAG-="udev-acl", SYMLINK+="llama-light/aura""#;

#[test]
fn the_unit_is_the_golden() {
    let got = directives(&packaging("llama-light.service"));
    let want: Vec<String> = UNIT_GOLDEN.lines().map(str::to_owned).collect();
    assert_eq!(got, want);
}

#[test]
fn the_unit_has_one_device_and_no_network_or_capabilities() {
    let got = directives(&packaging("llama-light.service"));
    let devices: Vec<&String> = got
        .iter()
        .filter(|l| l.starts_with("DeviceAllow="))
        .collect();
    assert_eq!(devices, ["DeviceAllow=/dev/llama-light/aura rw"]);
    assert!(got.contains(&"DevicePolicy=closed".to_owned()));
    assert!(got.contains(&"PrivateNetwork=yes".to_owned()));
    assert!(got.contains(&"CapabilityBoundingSet=".to_owned()));
    assert!(!got.iter().any(|l| l.starts_with("ReadWritePaths")));
    let groups: Vec<&String> = got
        .iter()
        .filter(|l| l.starts_with("SupplementaryGroups="))
        .collect();
    assert_eq!(groups, ["SupplementaryGroups=llama-watch"]);
}

#[test]
fn the_udev_rule_is_the_golden() {
    let got = directives(&packaging("94-llama-light-hidraw.rules"));
    assert_eq!(got, [RULE_GOLDEN]);
}

#[test]
fn sysusers_has_the_light_user_with_no_extra_groups() {
    let text = packaging("llama-bored.sysusers");
    assert!(
        text.lines()
            .any(|l| l == r#"u llama-light - "Llama Light RGB writer" / /usr/sbin/nologin"#)
    );
    assert!(!text.lines().any(|l| l.starts_with("m llama-light")));
}
