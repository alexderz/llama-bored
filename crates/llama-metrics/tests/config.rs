//! metrics.toml bounds, and the CLI surface.

mod common;

use std::time::Duration;

use llama_metrics::config::{Config, ConfigError};
use llama_metrics::service::{Command, parse_args};

const GOOD: &str = r#"
listen = "0.0.0.0:19477"
allow = ["192.168.0.0/22", "127.0.0.1/32"]
max_conns = 16
stale_after_s = 5
"#;

fn with(line_from: &str, line_to: &str) -> Result<Config, ConfigError> {
    assert!(GOOD.contains(line_from), "{line_from}");
    Config::from_toml(&GOOD.replace(line_from, line_to))
}

#[test]
fn the_documented_config_parses() {
    let cfg = Config::from_toml(GOOD).expect("good config");
    assert_eq!(cfg.listen.to_string(), "0.0.0.0:19477");
    assert_eq!(cfg.allow.nets().len(), 2);
    assert_eq!(cfg.max_conns, 16);
    assert_eq!(cfg.stale_after, Duration::from_secs(5));
}

#[test]
fn defaults_fill_max_conns_and_stale_after() {
    let cfg =
        Config::from_toml("listen = \"127.0.0.1:19477\"\nallow = [\"127.0.0.1/32\"]\n").unwrap();
    assert_eq!(cfg.max_conns, 16);
    assert_eq!(cfg.stale_after, Duration::from_secs(5));
}

#[test]
fn ipv6_listen_is_accepted() {
    let cfg = with("0.0.0.0:19477", "[::]:19477").unwrap();
    assert!(cfg.listen.is_ipv6());
}

#[test]
fn listen_and_allow_are_required() {
    assert!(matches!(
        Config::from_toml("allow = [\"127.0.0.1/32\"]\n"),
        Err(ConfigError::Parse(_))
    ));
    assert!(matches!(
        Config::from_toml("listen = \"0.0.0.0:19477\"\n"),
        Err(ConfigError::Parse(_))
    ));
}

#[test]
fn bounds_are_enforced() {
    assert!(matches!(
        with("\"0.0.0.0:19477\"", "\"0.0.0.0\""),
        Err(ConfigError::Listen(_))
    ));
    assert!(matches!(
        with("\"0.0.0.0:19477\"", "\"llama-host:19477\""),
        Err(ConfigError::Listen(_))
    ));
    assert!(matches!(
        with("0.0.0.0:19477", "0.0.0.0:0"),
        Err(ConfigError::ListenPort)
    ));
    assert!(matches!(
        with("[\"192.168.0.0/22\", \"127.0.0.1/32\"]", "[]"),
        Err(ConfigError::AllowCount)
    ));
    let many: Vec<String> = (0..33).map(|i| format!("\"10.{i}.0.0/16\"")).collect();
    assert!(matches!(
        with(
            "[\"192.168.0.0/22\", \"127.0.0.1/32\"]",
            &format!("[{}]", many.join(", "))
        ),
        Err(ConfigError::AllowCount)
    ));
    assert!(matches!(
        with("\"127.0.0.1/32\"", "\"0.0.0.0/0\""),
        Err(ConfigError::Allow(_))
    ));
    assert!(matches!(
        with("\"127.0.0.1/32\"", "\"192.168.0.0/22\""),
        Err(ConfigError::AllowDuplicate(_))
    ));
    assert!(matches!(
        with("max_conns = 16", "max_conns = 0"),
        Err(ConfigError::MaxConns)
    ));
    assert!(matches!(
        with("max_conns = 16", "max_conns = 65"),
        Err(ConfigError::MaxConns)
    ));
    assert!(with("max_conns = 16", "max_conns = 64").is_ok());
    assert!(with("max_conns = 16", "max_conns = 1").is_ok());
    assert!(matches!(
        with("stale_after_s = 5", "stale_after_s = 0"),
        Err(ConfigError::StaleAfter)
    ));
    assert!(matches!(
        with("stale_after_s = 5", "stale_after_s = 301"),
        Err(ConfigError::StaleAfter)
    ));
    assert!(matches!(
        with("stale_after_s = 5", "stale_after_s = -1"),
        Err(ConfigError::Parse(_))
    ));
    assert!(with("stale_after_s = 5", "stale_after_s = 300").is_ok());
}

#[test]
fn unknown_keys_and_path_keys_are_refused() {
    for extra in [
        "snapshot = \"/tmp/x.json\"",
        "path = \"/run/llama-watch/snapshot.json\"",
        "url = \"http://127.0.0.1:8080\"",
        "[tls]",
    ] {
        let text = format!("{GOOD}\n{extra}\n");
        assert!(
            matches!(Config::from_toml(&text), Err(ConfigError::Parse(_))),
            "{extra} accepted"
        );
    }
}

#[test]
fn load_caps_the_file_size() {
    let dir = common::scratch("config-load");
    let path = dir.join("metrics.toml");
    std::fs::write(&path, GOOD).unwrap();
    assert!(Config::load(&path).is_ok());
    let big = format!("{GOOD}\n#{}\n", "x".repeat(70 * 1024));
    std::fs::write(&path, big).unwrap();
    assert!(matches!(Config::load(&path), Err(ConfigError::TooLarge)));
    assert!(matches!(Config::load(&dir), Err(ConfigError::Read(_))));
}

#[test]
fn cli_takes_run_or_check_with_a_config() {
    let args = |v: &[&str]| parse_args(v.iter().map(|s| (*s).to_owned()));
    assert_eq!(
        args(&["run", "--config", "/etc/llama-bored/metrics.toml"]),
        Ok(Command::Run {
            config: "/etc/llama-bored/metrics.toml".into()
        })
    );
    assert!(matches!(
        args(&["check", "--config", "x"]),
        Ok(Command::Check { .. })
    ));
    for bad in [
        &[][..],
        &["run"][..],
        &["run", "--config"][..],
        &["run", "--config", ""][..],
        &["serve", "--config", "x"][..],
        &["run", "--config", "x", "--listen", "0.0.0.0:1"][..],
    ] {
        assert!(args(bad).is_err(), "{bad:?}");
    }
}
