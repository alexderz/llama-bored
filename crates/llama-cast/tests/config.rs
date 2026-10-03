//! cast.toml bounds, and the CLI surface.

mod common;

use std::net::Ipv4Addr;

use llama_cast::acl::CidrError;
use llama_cast::config::{Config, ConfigError, Font, Palette};
use llama_cast::service::{Command, parse_args};

const GOOD: &str = r#"
listen = "0.0.0.0:19478"
allow = ["192.168.1.0/24", "127.0.0.1/32"]
interface_addr = "192.168.1.20"
name = "llama-bored"
fps = 2
max_clients = 2
ffmpeg = "/usr/bin/ffmpeg"
font = "12x24"
palette = "llama"
"#;

fn with(from: &str, to: &str) -> Result<Config, ConfigError> {
    assert!(GOOD.contains(from), "{from}");
    Config::from_toml(&GOOD.replace(from, to))
}

#[test]
fn the_documented_config_parses() {
    let cfg = Config::from_toml(GOOD).expect("good config");
    assert_eq!(cfg.listen.to_string(), "0.0.0.0:19478");
    assert_eq!(cfg.allow.nets().len(), 2);
    assert_eq!(cfg.interface_addr, Ipv4Addr::new(192, 168, 1, 20));
    assert_eq!(cfg.name, "llama-bored");
    assert_eq!(cfg.fps, 2);
    assert_eq!(cfg.max_clients, 2);
    assert_eq!(cfg.ffmpeg.to_str(), Some("/usr/bin/ffmpeg"));
    assert_eq!(cfg.font, Font::Hack12x24);
    assert_eq!(cfg.palette, Palette::Llama);
    assert_eq!(cfg.base_url(), "http://192.168.1.20:19478");
}

#[test]
fn defaults_fill_everything_but_listen_and_allow() {
    let cfg = Config::from_toml("listen = \"192.168.1.20:19478\"\nallow = [\"192.168.1.0/24\"]\n")
        .unwrap();
    // interface_addr defaults to the listen address.
    assert_eq!(cfg.interface_addr, Ipv4Addr::new(192, 168, 1, 20));
    assert_eq!(cfg.name, "llama-bored");
    assert_eq!(cfg.fps, 2);
    assert_eq!(cfg.max_clients, 2);
    assert_eq!(cfg.ffmpeg.to_str(), Some("/usr/bin/ffmpeg"));
    assert_eq!(cfg.font, Font::Hack12x24);
    assert_eq!(cfg.font.file_name(), "llama-hack-12x24.psfu");
    assert_eq!(
        cfg.palette,
        Palette::Llama,
        "matches the [tty] palette default"
    );
}

/// #26: `palette` names tty11's colours, the same words as `[tty] palette`.
#[test]
fn palette_parses_llama_and_vga_only() {
    assert_eq!(
        with("palette = \"llama\"", "palette = \"vga\"")
            .unwrap()
            .palette,
        Palette::Vga
    );
    for bad in ["\"xterm\"", "\"VGA\"", "1"] {
        assert!(
            matches!(
                with("palette = \"llama\"", &format!("palette = {bad}")),
                Err(ConfigError::Parse(_))
            ),
            "accepted {bad}"
        );
    }
}

#[test]
fn listen_and_allow_are_required() {
    assert!(matches!(
        Config::from_toml("allow = [\"127.0.0.1/32\"]\n"),
        Err(ConfigError::Parse(_))
    ));
    assert!(matches!(
        Config::from_toml("listen = \"127.0.0.1:19478\"\n"),
        Err(ConfigError::Parse(_))
    ));
}

#[test]
fn unknown_keys_are_refused() {
    assert!(matches!(
        Config::from_toml(&format!("{GOOD}\nport = 1\n")),
        Err(ConfigError::Parse(_))
    ));
}

#[test]
fn listen_must_be_ipv4_with_a_real_port() {
    for bad in ["\"[::]:19478\"", "\"0.0.0.0\"", "\"host:19478\"", "\"\""] {
        assert!(
            matches!(with("\"0.0.0.0:19478\"", bad), Err(ConfigError::Listen(_))),
            "{bad}"
        );
    }
    assert!(matches!(
        with("0.0.0.0:19478", "0.0.0.0:0"),
        Err(ConfigError::ListenPort)
    ));
    assert!(matches!(
        with("0.0.0.0:19478", "0.0.0.0:1900"),
        Err(ConfigError::ListenPort)
    ));
}

#[test]
fn allowlist_is_validated_like_llama_metrics() {
    assert!(matches!(
        with("[\"192.168.1.0/24\", \"127.0.0.1/32\"]", "[]"),
        Err(ConfigError::AllowCount)
    ));
    let many: Vec<String> = (0..33).map(|i| format!("\"10.0.{i}.0/24\"")).collect();
    assert!(matches!(
        with(
            "[\"192.168.1.0/24\", \"127.0.0.1/32\"]",
            &format!("[{}]", many.join(", "))
        ),
        Err(ConfigError::AllowCount)
    ));
    for (bad, want) in [
        ("\"0.0.0.0/0\"", CidrError::Prefix("0.0.0.0/0".into())),
        ("\"::/0\"", CidrError::Prefix("::/0".into())),
        (
            "\"192.168.1.5/24\"",
            CidrError::HostBits("192.168.1.5/24".into()),
        ),
        ("\"192.168.1.0\"", CidrError::Syntax("192.168.1.0".into())),
        (
            "\"192.168.1.0/33\"",
            CidrError::Prefix("192.168.1.0/33".into()),
        ),
    ] {
        match with("\"192.168.1.0/24\"", bad) {
            Err(ConfigError::Allow(got)) => assert_eq!(got, want, "{bad}"),
            other => panic!("{bad}: {other:?}"),
        }
    }
    assert!(matches!(
        with("\"127.0.0.1/32\"", "\"192.168.1.0/24\""),
        Err(ConfigError::AllowDuplicate(_))
    ));
}

#[test]
fn interface_addr_rules() {
    // Required when listening on every address.
    assert!(matches!(
        with("interface_addr = \"192.168.1.20\"\n", ""),
        Err(ConfigError::InterfaceAddrMissing)
    ));
    for bad in [
        "0.0.0.0",
        "239.255.255.250",
        "255.255.255.255",
        "::1",
        "lan",
    ] {
        assert!(
            matches!(
                with("\"192.168.1.20\"", &format!("\"{bad}\"")),
                Err(ConfigError::InterfaceAddr(_))
            ),
            "{bad}"
        );
    }
    // A specific listen address must be the advertised one.
    let text = GOOD.replace("0.0.0.0:19478", "192.168.1.21:19478");
    assert!(matches!(
        Config::from_toml(&text),
        Err(ConfigError::InterfaceAddrMismatch(_, _))
    ));
}

#[test]
fn name_fps_clients_font_and_ffmpeg_bounds() {
    for bad in [
        "\"\"",
        "\" padded\"",
        "\"tab\\there\"",
        &format!("\"{}\"", "x".repeat(65)),
    ] {
        assert!(
            matches!(with("\"llama-bored\"", bad), Err(ConfigError::Name)),
            "{bad}"
        );
    }
    assert!(with("\"llama-bored\"", "\"Den <tty11> & co\"").is_ok());
    assert!(matches!(with("fps = 2", "fps = 0"), Err(ConfigError::Fps)));
    assert!(matches!(with("fps = 2", "fps = 6"), Err(ConfigError::Fps)));
    assert!(with("fps = 2", "fps = 5").is_ok());
    assert!(matches!(
        with("max_clients = 2", "max_clients = 0"),
        Err(ConfigError::MaxClients)
    ));
    assert!(matches!(
        with("max_clients = 2", "max_clients = 5"),
        Err(ConfigError::MaxClients)
    ));
    assert!(with("max_clients = 2", "max_clients = 4").is_ok());
    assert_eq!(
        with("\"12x24\"", "\"12x22\"").unwrap().font,
        Font::Hack12x22
    );
    assert!(matches!(
        with("\"12x24\"", "\"8x16\""),
        Err(ConfigError::Parse(_))
    ));
    for bad in [
        "ffmpeg",
        "./ffmpeg",
        "/usr/bin/../bin/ffmpeg",
        "/usr/./bin/ffmpeg",
        "/usr//bin/ffmpeg",
        "/usr/bin/",
        "/",
        "",
    ] {
        assert!(
            matches!(
                with("\"/usr/bin/ffmpeg\"", &format!("\"{bad}\"")),
                Err(ConfigError::Ffmpeg)
            ),
            "{bad}"
        );
    }
    assert!(with("\"/usr/bin/ffmpeg\"", "\"/opt/ffmpeg/bin/ffmpeg\"").is_ok());
}

#[test]
fn load_caps_the_file_and_wants_a_regular_file() {
    let dir = common::scratch("config-load");
    let path = dir.join("cast.toml");
    std::fs::write(&path, GOOD).unwrap();
    assert!(Config::load(&path).is_ok());
    std::fs::write(&path, format!("{GOOD}#{}\n", "x".repeat(70 * 1024))).unwrap();
    assert!(matches!(Config::load(&path), Err(ConfigError::TooLarge)));
    assert!(matches!(Config::load(&dir), Err(ConfigError::Read(_))));
    assert!(matches!(
        Config::load(&dir.join("absent.toml")),
        Err(ConfigError::Read(_))
    ));
}

#[test]
fn cli_is_run_check_or_bye_with_a_config() {
    let args = |a: &[&str]| parse_args(a.iter().map(|s| (*s).to_owned()));
    assert_eq!(
        args(&["run", "--config", "/etc/llama-bored/cast.toml"]),
        Ok(Command::Run {
            config: "/etc/llama-bored/cast.toml".into()
        })
    );
    assert_eq!(
        args(&["check", "--config", "c.toml"]),
        Ok(Command::Check {
            config: "c.toml".into()
        })
    );
    assert_eq!(
        args(&["bye", "--config", "c.toml"]),
        Ok(Command::Bye {
            config: "c.toml".into()
        })
    );
    for bad in [
        &["run"][..],
        &["run", "--config"],
        &["run", "--config", ""],
        &["serve", "--config", "c.toml"],
        &["run", "-c", "c.toml"],
        &["run", "--config", "c.toml", "extra"],
    ] {
        assert!(args(bad).is_err(), "{bad:?}");
    }
}
