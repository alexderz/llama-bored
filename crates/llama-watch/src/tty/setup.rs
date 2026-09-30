//! `llama-watch tty-setup --config PATH`: the unit's root pre-step (#7).
//!
//! `llama-watch.service` runs it as `ExecStartPre=-+`: as root, outside the
//! sandbox, before the watcher, and a failure does not stop the watcher. It
//! validates the config like `run` does, then runs two fixed programs on
//! tty11:
//!
//! ```text
//! /usr/bin/setfont -C /dev/tty11 /usr/local/share/llama-bored/<font file>
//! /usr/bin/stty -F /dev/tty11 cols <C> rows <R>     (only with tty.size)
//! ```
//!
//! No shell runs and no text from the file reaches an argument: the font is
//! a fixed file name per [`TtyFont`] variant and the size is two validated
//! integers. The environment is cleared. The watcher itself still only asks
//! tty11 for its window size (S13); it never sets it.

use std::path::{Path, PathBuf};

use crate::config::{Config, Tty, TtyFont, TtySize};

/// The console the watcher draws on.
pub const TTY: &str = "/dev/tty11";
/// Where install.sh puts the bundled fonts.
pub const FONT_DIR: &str = "/usr/local/share/llama-bored";
/// kbd's setfont.
pub const SETFONT: &str = "/usr/bin/setfont";
/// coreutils' stty.
pub const STTY: &str = "/usr/bin/stty";

/// One program and its arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Step {
    /// Absolute path of the program: [`SETFONT`] or [`STTY`].
    pub program: &'static str,
    /// Arguments, built from fixed words and validated numbers only.
    pub args: Vec<String>,
}

/// Font first: a new font makes the kernel refit the console to the screen,
/// so the size goes after it. No size keeps the kernel's refit.
#[must_use]
pub fn steps(tty: &Tty) -> Vec<Step> {
    let mut out = vec![font_step(tty.font)];
    if let Some(size) = tty.size {
        out.push(size_step(size));
    }
    out
}

fn font_step(font: TtyFont) -> Step {
    Step {
        program: SETFONT,
        args: vec![
            "-C".to_owned(),
            TTY.to_owned(),
            format!("{FONT_DIR}/{}", font.file_name()),
        ],
    }
}

fn size_step(size: TtySize) -> Step {
    Step {
        program: STTY,
        args: vec![
            "-F".to_owned(),
            TTY.to_owned(),
            "cols".to_owned(),
            size.cols.to_string(),
            "rows".to_owned(),
            size.rows.to_string(),
        ],
    }
}

/// Runs a [`Step`]. Tests pass a recorder; the service passes [`System`].
pub trait Runner {
    /// `Err` carries a one-line reason for the journal.
    fn run(&mut self, step: &Step) -> Result<(), String>;
}

/// Spawns the program with an empty environment and waits for it.
pub struct System;

impl Runner for System {
    fn run(&mut self, step: &Step) -> Result<(), String> {
        let status = std::process::Command::new(step.program)
            .args(&step.args)
            .env_clear()
            .current_dir("/")
            .status()
            .map_err(|err| format!("{}: {err}", step.program))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{} exited with {status}", step.program))
        }
    }
}

/// `tty-setup` was not given exactly `--config PATH`.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
#[error("usage: llama-watch tty-setup --config PATH")]
pub struct SetupUsage;

/// The arguments after `tty-setup`.
pub fn parse_args(args: &[String]) -> Result<PathBuf, SetupUsage> {
    match args {
        [flag, path] if flag == "--config" && !path.is_empty() && !path.starts_with("--") => {
            Ok(PathBuf::from(path))
        }
        _ => Err(SetupUsage),
    }
}

/// Validate the config at `path` against `nproc` and run its steps.
///
/// Exit 2: usage or config error, nothing run. Exit 1: a step failed; when
/// the font fails the size is not set, since it was chosen for that font.
/// Exit 0: every step ran.
pub fn run_setup(path: &Path, nproc: u32, runner: &mut impl Runner) -> i32 {
    let config = match Config::load_validated(path, nproc) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("tty-setup: {err}");
            return 2;
        }
    };
    for step in steps(&config.tty) {
        if let Err(reason) = runner.run(&step) {
            eprintln!("tty-setup: {reason}");
            return 1;
        }
    }
    0
}

/// `llama-watch tty-setup ARGS`, from `main`.
pub fn main(args: &[String]) -> i32 {
    let path = match parse_args(args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let roots = crate::sources::Roots::default();
    let nproc = crate::service::host_nproc(&roots.proc);
    run_setup(&path, nproc, &mut System)
}
