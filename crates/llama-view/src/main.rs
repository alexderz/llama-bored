//! Read `/dev/vcsaN` and paint the changed cells on the local terminal.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;
use std::thread;

use llama_view::{
    CliError, ColorSettings, FOCUS_ON, HELP, MAX_VCSA_BYTES, Next, Options, RESTORE, RawInput,
    Session, TermGuard, Wait, color_settings, device_paths, parse_args, restored,
    tmux_window_wanted, vcsa_geometry, wait_input,
};

fn main() -> ExitCode {
    match parse_args(std::env::args()) {
        Err(CliError::Help) => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
        Err(CliError::Message(message)) => {
            eprintln!("llama-view: {message}");
            ExitCode::from(2)
        }
        Ok(opts) => match mirror_with_env(opts) {
            Ok(()) => ExitCode::SUCCESS,
            Err(MirrorError::Usage(CliError::Message(message))) => {
                eprintln!("llama-view: {message}");
                ExitCode::from(2)
            }
            Err(MirrorError::Usage(CliError::Help)) => ExitCode::from(2),
            Err(MirrorError::Io(err)) => {
                eprintln!("llama-view: {err}");
                ExitCode::from(1)
            }
        },
    }
}

fn mirror_with_env(opts: Options) -> Result<(), MirrorError> {
    let env = |key: &str| std::env::var(key).ok();
    let colors = color_settings(&opts, env).map_err(MirrorError::Usage)?;
    let tmux_window = tmux_window_wanted(&opts, env);
    mirror(opts, colors, tmux_window).map_err(MirrorError::Io)
}

enum MirrorError {
    Usage(CliError),
    Io(io::Error),
}

/// The host name, read once for the title (sanitised by the session).
fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|text| text.trim().to_string())
        .unwrap_or_default()
}

fn mirror(opts: Options, colors: ColorSettings, tmux_window: bool) -> io::Result<()> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_restore_stdout();
        previous(info);
    }));

    let (vcsa_path, vcs_path, vcsu_path) = device_paths(&opts);
    let mut vcsa = open_read(&vcsa_path)?;
    let mut vcs = open_optional(vcs_path.as_deref());
    let mut vcsu = open_optional(vcsu_path.as_deref());
    // Keyboard and focus reports, only when stdin is a terminal. Dropped
    // after the guard (declared first), so the screen is restored before the
    // line discipline.
    let mut input = RawInput::enter(io::stdin())?;
    let mut guard = TermGuard::enter(io::stdout())?;
    if input.is_some() {
        Write::write_all(guard.writer(), FOCUS_ON)?;
    }
    let mut session = Session::new(&opts, colors, &host_name(), tmux_window);
    let mut vcsa_buf = Vec::new();
    let mut vcs_buf = Vec::new();
    let mut vcsu_buf = Vec::new();
    let mut out = Vec::new();

    loop {
        // Every frame: the pane and tty11 can both change size at any time.
        let size = terminal_size();
        read_vcsa(&mut vcsa, &mut vcsa_buf)?;
        let cells = vcsa_geometry(&vcsa_buf)
            .map_or(0, |(rows, cols)| usize::from(rows) * usize::from(cols));
        let vcs_bytes = read_optional(&mut vcs, cells, 1, &mut vcs_buf);
        let vcsu_bytes = read_optional(&mut vcsu, cells, 4, &mut vcsu_buf);
        out.clear();
        session
            .frame(size, &vcsa_buf, vcs_bytes, vcsu_bytes, &mut out)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("vcsa: {err:?}")))?;
        if !out.is_empty() {
            Write::write_all(guard.writer(), &out)?;
            guard.writer().flush()?;
        }
        if opts.once {
            break;
        }

        let period = session.period();
        let Some(raw) = input.as_mut() else {
            thread::sleep(period);
            continue;
        };
        match wait_input(raw.fd(), period)? {
            Wait::Timeout => {}
            Wait::Closed => break,
            Wait::Readable => {
                let mut bytes = [0u8; 64];
                let n = rustix::io::read(raw.fd(), &mut bytes)?;
                if n == 0 || session.input(&bytes[..n]) == Next::Quit {
                    break;
                }
            }
        }
    }
    guard.restore()?;
    if let Some(raw) = input.as_mut() {
        raw.restore()?;
    }
    Ok(())
}

fn terminal_size() -> (u16, u16) {
    match rustix::termios::tcgetwinsize(io::stdout()) {
        Ok(size) if size.ws_col > 0 && size.ws_row > 0 => (size.ws_col, size.ws_row),
        _ => (80, 24),
    }
}

fn open_read(path: &Path) -> io::Result<File> {
    File::open(path).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("opening {} read-only: {err}", path.display()),
        )
    })
}

fn open_optional(path: Option<&Path>) -> Option<File> {
    let path = path?;
    File::open(path).ok()
}

/// The whole vcsa image, capped at the largest console. Its length gives
/// the real size of a console wider than 255 columns (see `vcsa_geometry`).
fn read_vcsa(file: &mut File, buf: &mut Vec<u8>) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    buf.clear();
    let cap = u64::try_from(MAX_VCSA_BYTES).unwrap_or(u64::MAX);
    Read::by_ref(file).take(cap).read_to_end(buf)?;
    if buf.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "vcsa is shorter than its header",
        ));
    }
    Ok(())
}

fn read_optional<'a>(
    file: &mut Option<File>,
    cells: usize,
    width: usize,
    buf: &'a mut Vec<u8>,
) -> Option<&'a [u8]> {
    let file = file.as_mut()?;
    let n = cells.saturating_mul(width);
    if file.seek(SeekFrom::Start(0)).is_err() {
        return None;
    }
    buf.resize(n, 0);
    if file.read_exact(buf).is_err() {
        return None;
    }
    Some(buf.as_slice())
}

/// The panic path (release builds abort, so no destructor runs): the screen
/// sequences, then canonical input and echo back on stdin if it is a tty.
fn write_restore_stdout() {
    let mut out = io::stdout();
    let _ = Write::write_all(&mut out, RESTORE);
    let _ = out.flush();
    let stdin = io::stdin();
    if rustix::termios::isatty(&stdin)
        && let Ok(found) = rustix::termios::tcgetattr(&stdin)
    {
        let _ = rustix::termios::tcsetattr(
            &stdin,
            rustix::termios::OptionalActions::Now,
            &restored(found),
        );
    }
}
