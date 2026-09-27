//! Command line. No files are opened here.

use std::path::PathBuf;

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

  --vt N          console number (default 11, range 0..=63)
  --device PATH   vcsa path for tests; vcs and vcsu are sibling names
  --fps N         poll rate (default 10, range 1..=20)
  --offset-x N    pan the view right, in columns
  --offset-y N    pan the view down, in rows
  --once          paint one frame, restore the terminal, and exit
  --help          show this text
";

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
