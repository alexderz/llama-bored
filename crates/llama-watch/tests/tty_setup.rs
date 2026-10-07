//! #7: `llama-watch tty-setup`, the root pre-step that loads the tty11 font
//! and sets its size from `[tty] font` / `[tty] size`. Tests use a recording
//! runner: nothing here opens a tty or runs setfont or stty.

use std::path::PathBuf;

use llama_watch::config::{Tty, TtyFont, TtySize};
use llama_watch::tty::setup::{
    FONT_DIR, Runner, SETFONT, STTY, SetupUsage, Step, TTY, parse_args, run_setup, steps,
};

#[derive(Default)]
struct Record {
    ran: Vec<Step>,
    fail: Option<&'static str>,
    /// setfont fails this many times, then works (the boot race).
    font_fails: usize,
    nudges: usize,
    paused: std::time::Duration,
}

impl Runner for Record {
    fn run(&mut self, step: &Step) -> Result<(), String> {
        self.ran.push(step.clone());
        if step.program == SETFONT && self.font_fails > 0 {
            self.font_fails -= 1;
            return Err("setfont: console not ready".to_owned());
        }
        if self.fail == Some(step.program) {
            Err(format!("{} failed", step.program))
        } else {
            Ok(())
        }
    }

    fn nudge(&mut self) -> Result<(), String> {
        assert!(self.ran.is_empty(), "the nudge comes before any program");
        self.nudges += 1;
        Ok(())
    }

    fn pause(&mut self, d: std::time::Duration) {
        self.paused += d;
    }
}

struct TempConfig(PathBuf);

impl TempConfig {
    fn new(label: &str, text: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "llama-watch-tty-setup-{label}-{}.toml",
            std::process::id()
        ));
        std::fs::write(&path, text).expect("write config");
        Self(path)
    }
}

impl Drop for TempConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn setfont(file: &str) -> Step {
    Step {
        program: SETFONT,
        args: vec!["-C".into(), TTY.into(), format!("{FONT_DIR}/{file}")],
    }
}

fn stty(cols: &str, rows: &str) -> Step {
    Step {
        program: STTY,
        args: vec![
            "-F".into(),
            TTY.into(),
            "cols".into(),
            cols.into(),
            "rows".into(),
            rows.into(),
        ],
    }
}

#[test]
fn fixed_programs_and_paths() {
    assert_eq!(TTY, "/dev/tty11");
    assert_eq!(FONT_DIR, "/usr/local/share/llama-bored");
    assert_eq!(SETFONT, "/usr/bin/setfont");
    assert_eq!(STTY, "/usr/bin/stty");
}

#[test]
fn default_config_loads_the_12x24_font_and_keeps_the_size() {
    assert_eq!(
        steps(&Tty::default()),
        [setfont("llama-hack-12x24.psfu")],
        "same as the old unit's setfont pre-step"
    );
}

#[test]
fn font_then_size_from_the_config() {
    let tty = Tty {
        font: TtyFont::Hack12x22,
        size: Some(TtySize {
            cols: 160,
            rows: 49,
        }),
        ..Tty::default()
    };
    assert_eq!(
        steps(&tty),
        [setfont("llama-hack-12x22.psfu"), stty("160", "49")]
    );
}

/// #73: the 10x18 font for a 1080p screen viewed up close.
#[test]
fn ten_by_eighteen_font_then_192x60() {
    let tty = Tty {
        font: TtyFont::Hack10x18,
        size: Some(TtySize {
            cols: 192,
            rows: 60,
        }),
        ..Tty::default()
    };
    assert_eq!(
        steps(&tty),
        [setfont("llama-hack-10x18.psfu"), stty("192", "60")]
    );
}

#[test]
fn run_setup_validates_then_runs_font_and_size() {
    let cfg = TempConfig::new("ok", "[tty]\nfont = \"12x22\"\nsize = \"160x49\"\n");
    let mut rec = Record::default();
    assert_eq!(run_setup(&cfg.0, 8, &mut rec), 0);
    assert_eq!(
        rec.ran,
        [setfont("llama-hack-12x22.psfu"), stty("160", "49")]
    );
}

#[test]
fn an_invalid_or_missing_config_runs_nothing() {
    for (label, text) in [
        ("small", "[tty]\nsize = \"120x30\"\n"),
        ("shell", "[tty]\nsize = \"160x49; reboot\"\n"),
        ("font", "[tty]\nfont = \"/tmp/evil.psfu\"\n"),
        ("other", "[tty]\nfps = 0\n"),
    ] {
        let cfg = TempConfig::new(label, text);
        let mut rec = Record::default();
        assert_eq!(run_setup(&cfg.0, 8, &mut rec), 2, "{label}");
        assert!(rec.ran.is_empty(), "{label}: ran {:?}", rec.ran);
    }
    let mut rec = Record::default();
    let missing = std::env::temp_dir().join("llama-watch-tty-setup-no-such-file.toml");
    assert_eq!(run_setup(&missing, 8, &mut rec), 2);
    assert!(rec.ran.is_empty());
}

#[test]
fn a_failed_font_skips_the_size_chosen_for_it() {
    let cfg = TempConfig::new("fontfail", "[tty]\nsize = \"160x45\"\n");
    let mut rec = Record {
        fail: Some(SETFONT),
        ..Record::default()
    };
    assert_eq!(run_setup(&cfg.0, 8, &mut rec), 1);
    // #22: the font is retried until FONT_WAIT; the size never runs.
    assert!(!rec.ran.is_empty());
    assert!(
        rec.ran
            .iter()
            .all(|s| *s == setfont("llama-hack-12x24.psfu"))
    );
    let mut rec = Record {
        fail: Some(STTY),
        ..Record::default()
    };
    assert_eq!(run_setup(&cfg.0, 8, &mut rec), 1);
    assert_eq!(rec.ran.len(), 2);
}

#[test]
fn usage_is_exactly_config_path() {
    let ok: Vec<String> = vec!["--config".into(), "/etc/llama-bored/watch.toml".into()];
    assert_eq!(
        parse_args(&ok),
        Ok(PathBuf::from("/etc/llama-bored/watch.toml"))
    );
    for bad in [
        vec![],
        vec!["--config".to_string()],
        vec!["--config".to_string(), String::new()],
        vec!["--config".to_string(), "--no-text".to_string()],
        vec!["--config".to_string(), "a".to_string(), "b".to_string()],
        vec!["--font".to_string(), "12x22".to_string()],
    ] {
        assert_eq!(parse_args(&bad), Err(SetupUsage), "{bad:?}");
    }
}

/// The binary dispatches `tty-setup` before `run`: a usage error exits 2 and
/// spawns nothing (no config, no tty).
#[test]
fn binary_tty_setup_usage_exits_2() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_llama-watch"))
        .arg("tty-setup")
        .output()
        .expect("spawn llama-watch");
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("llama-watch tty-setup --config PATH"), "{err}");
}

/// At boot fbcon may still defer its take-over: tty-setup nudges tty11 first,
/// then retries the font until it loads, then sets the size.
#[test]
fn setup_nudges_then_retries_the_font_until_the_console_is_ready() {
    let cfg = TempConfig::new("boot-race", "[tty]\nfont = \"12x22\"\nsize = \"160x49\"\n");
    let mut rec = Record {
        font_fails: 3,
        ..Record::default()
    };
    assert_eq!(run_setup(&cfg.0, 8, &mut rec), 0);
    assert_eq!(rec.nudges, 1);
    let fonts = rec.ran.iter().filter(|s| s.program == SETFONT).count();
    assert_eq!(fonts, 4, "three failures, then success");
    assert_eq!(rec.ran.last().map(|s| s.program), Some(STTY));
    assert_eq!(rec.paused, llama_watch::tty::setup::FONT_RETRY * 3);
}

/// A font that never loads gives up after FONT_WAIT and skips the size.
#[test]
fn setup_gives_up_on_the_font_after_the_wait_and_skips_the_size() {
    let cfg = TempConfig::new("never", "[tty]\nfont = \"12x22\"\nsize = \"160x49\"\n");
    let mut rec = Record {
        fail: Some(SETFONT),
        ..Record::default()
    };
    assert_eq!(run_setup(&cfg.0, 8, &mut rec), 1);
    assert!(rec.ran.iter().all(|s| s.program == SETFONT));
    assert!(rec.paused < llama_watch::tty::setup::FONT_WAIT);
    assert!(rec.paused >= llama_watch::tty::setup::FONT_WAIT - llama_watch::tty::setup::FONT_RETRY);
}
