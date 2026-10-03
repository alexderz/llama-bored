//! Command line. No files are opened here.

use std::path::PathBuf;

use llama_core::palette::Palette;

use crate::color::{ColorChoice, ColorMode};
use crate::pane::Fit;
use crate::screen::AttrLayout;

/// What the mirror should open and how fast it should poll.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    /// Virtual terminal number. Ignored for the vcsa path when [`Self::device`] is set.
    pub vt: u8,
    /// Explicit vcsa path. `vcs` and `vcsu` are the sibling names when the file starts with `vcsa`.
    pub device: Option<PathBuf>,
    /// Frames per second, `1..=20`.
    pub fps: u8,
    pub offset_x: u16,
    pub offset_y: u16,
    /// Paint one frame and return, so a test can observe restore on exit.
    pub once: bool,
    /// `--colors`. `None`: `LLAMA_VIEW_COLORS`, else `auto`.
    pub colors: Option<ColorChoice>,
    /// `--palette`. `None`: `LLAMA_VIEW_PALETTE`, else `llama`.
    pub palette: Option<Palette>,
    /// `--font-glyphs`. `None` (`auto`): inferred from the screen.
    pub font_glyphs: Option<AttrLayout>,
    /// `--fit`: what a pane smaller than the console shows.
    pub fit: Fit,
    /// `--tmux-window-name`. Also `LLAMA_VIEW_TMUX_WINDOW`; only inside tmux.
    pub tmux_window_name: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            vt: 11,
            device: None,
            fps: 10,
            offset_x: 0,
            offset_y: 0,
            once: false,
            colors: None,
            palette: None,
            font_glyphs: None,
            fit: Fit::Crop,
            tmux_window_name: false,
        }
    }
}

/// `--help`, or a message for stderr.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliError {
    Help,
    Message(String),
}

pub const HELP: &str = "\
llama-view — read-only mirror of a Linux virtual console

Keys: q or Ctrl-C quits and restores the terminal.

  --vt N          console number (default 11, range 0..=63)
  --device PATH   vcsa path for tests; vcs and vcsu are sibling names
  --fps N         poll rate (default 10, range 1..=20)
  --offset-x N    pan the view right, in columns
  --offset-y N    pan the view down, in rows
  --fit MODE      crop (default) or center: what a pane smaller than the console
                  shows, the top-left (header, meters, RECENT) or the middle; the
                  last row then names both sizes
  --colors MODE   auto (default), truecolor, 256 or 16; env LLAMA_VIEW_COLORS
                  truecolor: the exact RGB tty11 shows (SGR 38;2 / 48;2)
                  256: the nearest xterm colour 16-255, which themes leave alone
                  16: the terminal's own 16 colours, through its theme
                  auto: truecolor when COLORTERM is truecolor or 24bit, else 256
                  (SSH does not forward COLORTERM: pass --colors truecolor or
                  set LLAMA_VIEW_COLORS=truecolor if your terminal has it)
  --palette NAME  llama (default) or vga: tty11's [tty] palette in watch.toml;
                  env LLAMA_VIEW_PALETTE
  --font-glyphs N auto (default), 256 or 512: tty11's font size in glyphs, which
                  decides how vcsa stores colours; auto infers it from blanks
  --tmux-window-name
                  inside tmux, also name the window after the model (ESC k);
                  needs `set -g allow-rename on`; env LLAMA_VIEW_TMUX_WINDOW=1
  --once          paint one frame, restore the terminal, and exit
  --help          show this text

A flag wins over its environment variable.

In tmux: the pane title (#T) is `llama-view: <model> · <host>`. Frames are
synchronized (no flicker on tmux 3.4+). With `set -g focus-events on` the
mirror drops to 1 fps while its pane is unfocused or the client detached.
Truecolor needs `set -as terminal-features ',*:RGB'` (tmux 3.2+); without it
tmux turns 24-bit colour into its own 256.

Signals: SIGWINCH is not caught (the size is read every frame). SIGTERM and
SIGHUP end llama-view without a restore; the next run, or `reset`, puts the
terminal back.
";

/// `--tmux-window-name` or `LLAMA_VIEW_TMUX_WINDOW` (`1`, `yes`, `true`,
/// `on`), and only when `TMUX` says this is a tmux pane.
pub fn tmux_window_name<F>(opts: &Options, env: F) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    let wanted = opts.tmux_window_name
        || env("LLAMA_VIEW_TMUX_WINDOW")
            .is_some_and(|v| matches!(v.as_str(), "1" | "yes" | "true" | "on"));
    wanted && env("TMUX").is_some_and(|v| !v.is_empty())
}

/// Colours resolved from the flags, the environment and `COLORTERM`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColorSettings {
    pub mode: ColorMode,
    pub palette: Palette,
}

/// Resolve `--colors` and `--palette`: the flag, else `LLAMA_VIEW_COLORS` /
/// `LLAMA_VIEW_PALETTE`, else `auto` / `llama`; `auto` then reads
/// `COLORTERM`. `env` looks a variable up (the process environment in
/// `main`, a table in tests). An unknown value in a variable is an error,
/// the same as in a flag.
pub fn color_settings<F>(opts: &Options, env: F) -> Result<ColorSettings, CliError>
where
    F: Fn(&str) -> Option<String>,
{
    let choice = match opts.colors {
        Some(choice) => choice,
        None => match env("LLAMA_VIEW_COLORS") {
            Some(text) => ColorChoice::parse(&text).ok_or_else(|| {
                CliError::Message(format!(
                    "LLAMA_VIEW_COLORS={text} is not auto, truecolor, 256 or 16"
                ))
            })?,
            None => ColorChoice::Auto,
        },
    };
    let palette = match opts.palette {
        Some(palette) => palette,
        None => match env("LLAMA_VIEW_PALETTE") {
            Some(text) => Palette::parse(&text).ok_or_else(|| {
                CliError::Message(format!("LLAMA_VIEW_PALETTE={text} is not llama or vga"))
            })?,
            None => Palette::Llama,
        },
    };
    let colorterm = env("COLORTERM");
    Ok(ColorSettings {
        mode: choice.resolve(colorterm.as_deref()),
        palette,
    })
}

/// Parse `args`, including argv0.
pub fn parse_args<I, S>(args: I) -> Result<Options, CliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    let _argv0 = args.next();
    let mut opts = Options::default();
    while let Some(arg) = args.next() {
        let arg = arg.as_ref();
        match arg {
            "--help" | "-h" => return Err(CliError::Help),
            "--vt" => {
                let value = need(&mut args, "--vt")?;
                opts.vt = parse_vt(&value)?;
            }
            "--device" => {
                let value = need(&mut args, "--device")?;
                opts.device = Some(PathBuf::from(value));
            }
            "--fps" => {
                let value = need(&mut args, "--fps")?;
                opts.fps = parse_fps(&value)?;
            }
            "--offset-x" => {
                let value = need(&mut args, "--offset-x")?;
                opts.offset_x = parse_u16(&value, "--offset-x")?;
            }
            "--offset-y" => {
                let value = need(&mut args, "--offset-y")?;
                opts.offset_y = parse_u16(&value, "--offset-y")?;
            }
            "--once" => opts.once = true,
            "--fit" => {
                let value = need(&mut args, "--fit")?;
                opts.fit = Fit::parse(&value).ok_or_else(|| {
                    CliError::Message(format!("--fit {value} is not crop or center"))
                })?;
            }
            "--tmux-window-name" => opts.tmux_window_name = true,
            "--colors" => {
                let value = need(&mut args, "--colors")?;
                opts.colors = Some(ColorChoice::parse(&value).ok_or_else(|| {
                    CliError::Message(format!(
                        "--colors {value} is not auto, truecolor, 256 or 16"
                    ))
                })?);
            }
            "--palette" => {
                let value = need(&mut args, "--palette")?;
                opts.palette = Some(Palette::parse(&value).ok_or_else(|| {
                    CliError::Message(format!("--palette {value} is not llama or vga"))
                })?);
            }
            "--font-glyphs" => {
                let value = need(&mut args, "--font-glyphs")?;
                opts.font_glyphs = match value.as_str() {
                    "auto" => None,
                    "256" => Some(AttrLayout::Glyphs256),
                    "512" => Some(AttrLayout::Glyphs512),
                    _ => {
                        return Err(CliError::Message(format!(
                            "--font-glyphs {value} is not auto, 256 or 512"
                        )));
                    }
                };
            }
            other => {
                return Err(CliError::Message(format!("unknown argument {other}")));
            }
        }
    }
    Ok(opts)
}

/// vcsa path, then optional vcs and vcsu siblings.
pub fn device_paths(opts: &Options) -> (PathBuf, Option<PathBuf>, Option<PathBuf>) {
    if let Some(vcsa) = &opts.device {
        let vcs = sibling(vcsa, "vcsa", "vcs");
        let vcsu = sibling(vcsa, "vcsa", "vcsu");
        (vcsa.clone(), vcs, vcsu)
    } else {
        let n = opts.vt;
        (
            PathBuf::from(format!("/dev/vcsa{n}")),
            Some(PathBuf::from(format!("/dev/vcs{n}"))),
            Some(PathBuf::from(format!("/dev/vcsu{n}"))),
        )
    }
}

fn sibling(path: &std::path::Path, from: &str, to: &str) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix(from)?;
    Some(path.with_file_name(format!("{to}{rest}")))
}

fn need<I, S>(args: &mut I, flag: &str) -> Result<String, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    match args.next() {
        Some(value) => Ok(value.as_ref().to_string()),
        None => Err(CliError::Message(format!("{flag} needs a value"))),
    }
}

fn parse_vt(value: &str) -> Result<u8, CliError> {
    let n: u8 = value
        .parse()
        .map_err(|_| CliError::Message(format!("--vt {value} is not a console number")))?;
    if n > 63 {
        return Err(CliError::Message(format!("--vt {value} is outside 0..=63")));
    }
    Ok(n)
}

fn parse_fps(value: &str) -> Result<u8, CliError> {
    let n: u8 = value
        .parse()
        .map_err(|_| CliError::Message(format!("--fps {value} is not a number")))?;
    if !(1..=20).contains(&n) {
        return Err(CliError::Message(format!(
            "--fps {value} is outside 1..=20"
        )));
    }
    Ok(n)
}

fn parse_u16(value: &str, flag: &str) -> Result<u16, CliError> {
    value
        .parse()
        .map_err(|_| CliError::Message(format!("{flag} {value} is not a column or row")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_tty11_at_10fps() {
        let opts = parse_args(["llama-view"]).expect("defaults");
        assert_eq!(opts.vt, 11);
        assert_eq!(opts.fps, 10);
        assert_eq!(opts.offset_x, 0);
        assert_eq!(opts.offset_y, 0);
        assert!(!opts.once);
        assert_eq!(opts.device, None);
        let (vcsa, vcs, vcsu) = device_paths(&opts);
        assert_eq!(vcsa, PathBuf::from("/dev/vcsa11"));
        assert_eq!(vcs.unwrap(), PathBuf::from("/dev/vcs11"));
        assert_eq!(vcsu.unwrap(), PathBuf::from("/dev/vcsu11"));
    }

    #[test]
    fn fps_and_vt_stay_in_range() {
        let opts = parse_args(["llama-view", "--fps", "1", "--vt", "0"]).expect("low");
        assert_eq!(opts.fps, 1);
        assert_eq!(opts.vt, 0);
        let opts = parse_args(["llama-view", "--fps", "20", "--vt", "63"]).expect("high");
        assert_eq!(opts.fps, 20);
        assert_eq!(opts.vt, 63);
        assert!(parse_args(["llama-view", "--fps", "0"]).is_err());
        assert!(parse_args(["llama-view", "--fps", "21"]).is_err());
        assert!(parse_args(["llama-view", "--vt", "64"]).is_err());
    }

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }

    #[test]
    fn colors_default_to_auto_and_llama() {
        let opts = parse_args(["llama-view"]).unwrap();
        assert_eq!(opts.colors, None);
        assert_eq!(opts.palette, None);
        assert_eq!(opts.font_glyphs, None);
        let got = color_settings(&opts, env_of(&[])).unwrap();
        assert_eq!(got.mode, ColorMode::Xterm256, "auto without COLORTERM");
        assert_eq!(got.palette, Palette::Llama);
        let got = color_settings(&opts, env_of(&[("COLORTERM", "truecolor")])).unwrap();
        assert_eq!(got.mode, ColorMode::Truecolor);
    }

    #[test]
    fn auto_matrix_over_term_and_colorterm() {
        let opts = parse_args(["llama-view"]).unwrap();
        for term in ["xterm-256color", "tmux-256color", "screen-256color"] {
            let unset = color_settings(&opts, env_of(&[("TERM", term)])).unwrap();
            assert_eq!(unset.mode, ColorMode::Xterm256, "{term} without COLORTERM");
            for colorterm in ["truecolor", "24bit"] {
                let set =
                    color_settings(&opts, env_of(&[("TERM", term), ("COLORTERM", colorterm)]))
                        .unwrap();
                assert_eq!(set.mode, ColorMode::Truecolor, "{term} {colorterm}");
            }
        }
    }

    #[test]
    fn flag_beats_env_and_env_beats_auto() {
        let env = env_of(&[
            ("LLAMA_VIEW_COLORS", "16"),
            ("LLAMA_VIEW_PALETTE", "vga"),
            ("COLORTERM", "truecolor"),
        ]);
        let bare = parse_args(["llama-view"]).unwrap();
        let got = color_settings(&bare, &env).unwrap();
        assert_eq!(got.mode, ColorMode::Ansi16, "env beats auto");
        assert_eq!(got.palette, Palette::Vga);
        let flags = parse_args(["llama-view", "--colors", "256", "--palette", "llama"]).unwrap();
        let got = color_settings(&flags, &env).unwrap();
        assert_eq!(got.mode, ColorMode::Xterm256, "flag beats env");
        assert_eq!(got.palette, Palette::Llama);
        let auto = parse_args(["llama-view", "--colors", "auto"]).unwrap();
        let got = color_settings(&auto, &env).unwrap();
        assert_eq!(
            got.mode,
            ColorMode::Truecolor,
            "--colors auto beats the env"
        );
        let truecolor = env_of(&[("LLAMA_VIEW_COLORS", "truecolor")]);
        assert_eq!(
            color_settings(&bare, truecolor).unwrap().mode,
            ColorMode::Truecolor,
            "env truecolor without COLORTERM"
        );
    }

    #[test]
    fn bad_color_values_are_refused() {
        for args in [
            &["llama-view", "--colors", "24bit"][..],
            &["llama-view", "--colors"][..],
            &["llama-view", "--palette", "xterm"][..],
            &["llama-view", "--font-glyphs", "1024"][..],
        ] {
            assert!(parse_args(args.iter().copied()).is_err(), "{args:?}");
        }
        let bare = parse_args(["llama-view"]).unwrap();
        assert!(color_settings(&bare, env_of(&[("LLAMA_VIEW_COLORS", "lots")])).is_err());
        assert!(color_settings(&bare, env_of(&[("LLAMA_VIEW_PALETTE", "VGA")])).is_err());
        let glyphs = parse_args(["llama-view", "--font-glyphs", "512"]).unwrap();
        assert_eq!(glyphs.font_glyphs, Some(AttrLayout::Glyphs512));
        let glyphs = parse_args(["llama-view", "--font-glyphs", "auto"]).unwrap();
        assert_eq!(glyphs.font_glyphs, None);
    }

    #[test]
    fn fit_and_tmux_window_flags() {
        let opts = parse_args(["llama-view"]).unwrap();
        assert_eq!(opts.fit, Fit::Crop);
        assert!(!opts.tmux_window_name);
        let opts = parse_args(["llama-view", "--fit", "center", "--tmux-window-name"]).unwrap();
        assert_eq!(opts.fit, Fit::Center);
        assert!(opts.tmux_window_name);
        assert!(parse_args(["llama-view", "--fit", "zoom"]).is_err());
        let tmux = ("TMUX", "/tmp/tmux-1000/default,1234,0");
        assert!(tmux_window_name(&opts, env_of(&[tmux])));
        assert!(!tmux_window_name(&opts, env_of(&[])), "not inside tmux");
        assert!(!tmux_window_name(&opts, env_of(&[("TMUX", "")])));
        let bare = parse_args(["llama-view"]).unwrap();
        assert!(!tmux_window_name(&bare, env_of(&[tmux])), "opt-in only");
        assert!(tmux_window_name(
            &bare,
            env_of(&[tmux, ("LLAMA_VIEW_TMUX_WINDOW", "1")])
        ));
        assert!(!tmux_window_name(
            &bare,
            env_of(&[tmux, ("LLAMA_VIEW_TMUX_WINDOW", "0")])
        ));
        assert!(!tmux_window_name(
            &bare,
            env_of(&[("LLAMA_VIEW_TMUX_WINDOW", "1")])
        ));
    }

    #[test]
    fn help_documents_the_tmux_settings() {
        for needle in [
            "--fit",
            "--colors",
            "--palette",
            "--tmux-window-name",
            "allow-rename on",
            "focus-events on",
            "terminal-features ',*:RGB'",
            "LLAMA_VIEW_COLORS",
            "q or Ctrl-C",
        ] {
            assert!(HELP.contains(needle), "{needle}");
        }
    }

    #[test]
    fn device_and_offset_pan_a_fixture() {
        let opts = parse_args([
            "llama-view",
            "--device",
            "/tmp/fix/vcsa11",
            "--offset-x",
            "4",
            "--offset-y",
            "9",
        ])
        .expect("fixture");
        assert_eq!(opts.offset_x, 4);
        assert_eq!(opts.offset_y, 9);
        let (vcsa, vcs, vcsu) = device_paths(&opts);
        assert_eq!(vcsa, PathBuf::from("/tmp/fix/vcsa11"));
        assert_eq!(vcs.unwrap(), PathBuf::from("/tmp/fix/vcs11"));
        assert_eq!(vcsu.unwrap(), PathBuf::from("/tmp/fix/vcsu11"));
    }
}
