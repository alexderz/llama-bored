//! Read `/dev/vcsaN` and paint the changed cells on the local terminal.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use llama_view::{
    CliError, ColorSettings, FOCUS_ON, HELP, MAX_VCSA_BYTES, Next, Options, RESTORE, RawInput,
    Session, SourceWatch, TermGuard, Wait, color_settings, device_paths, kiosk_wanted, parse_args,
    restored, source_name, tmux_window_wanted, vcsa_geometry, wait_input, wants_focus_reports,
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
    let kiosk = kiosk_wanted(&opts, env);
    mirror(opts, colors, tmux_window, kiosk).map_err(MirrorError::Io)
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

fn mirror(opts: Options, colors: ColorSettings, tmux_window: bool, kiosk: bool) -> io::Result<()> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_restore_stdout();
        previous(info);
    }));

    let (vcsa_path, vcs_path, vcsu_path) = device_paths(&opts);
    // Outside kiosk mode the source has to be there from the start, exactly
    // as before: a failure here exits before the terminal is touched. In
    // kiosk mode it can be missing at any time (#104: an llama-watch
    // restart frees tty11 for a moment), so opening happens in the loop and
    // a failure is a status line, not an exit.
    let mut source = if kiosk {
        None
    } else {
        Some(open_source(
            &vcsa_path,
            vcs_path.as_deref(),
            vcsu_path.as_deref(),
        )?)
    };

    // Keyboard and focus reports, only when stdin is a terminal. Dropped
    // after the guard (declared first), so the screen is restored before the
    // line discipline.
    let mut input = RawInput::enter(io::stdin())?;
    let mut guard = match TermGuard::enter(io::stdout()) {
        Ok(guard) => guard,
        Err(err) => return terminal_gone(err, kiosk),
    };
    if wants_focus_reports(kiosk, input.is_some()) {
        Write::write_all(guard.writer(), FOCUS_ON)?;
    }
    let mut session = Session::new(&opts, colors, &host_name(), tmux_window, kiosk);
    let mut watch = SourceWatch::default();
    let source_name = source_name(opts.vt, opts.device.as_deref());
    let mut vcsa_buf = Vec::new();
    let mut vcs_buf = Vec::new();
    let mut vcsu_buf = Vec::new();
    let mut out = Vec::new();

    loop {
        // Every frame: the pane and tty11 can both change size at any time.
        let size = terminal_size();
        if kiosk {
            kiosk_frame(
                &vcsa_path,
                vcs_path.as_deref(),
                vcsu_path.as_deref(),
                &mut source,
                &mut watch,
                &source_name,
                &mut session,
                size,
                &mut vcsa_buf,
                &mut vcs_buf,
                &mut vcsu_buf,
                &mut out,
            );
        } else {
            let (vcsa, vcs, vcsu) = source
                .as_mut()
                .expect("opened above, never dropped outside kiosk");
            read_vcsa(vcsa, &mut vcsa_buf)?;
            let cells = vcsa_geometry(&vcsa_buf)
                .map_or(0, |(rows, cols)| usize::from(rows) * usize::from(cols));
            let vcs_bytes = read_optional(vcs, cells, 1, &mut vcs_buf);
            let vcsu_bytes = read_optional(vcsu, cells, 4, &mut vcsu_buf);
            out.clear();
            session
                .frame(size, &vcsa_buf, vcs_bytes, vcsu_bytes, &mut out)
                .map_err(|err| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("vcsa: {err:?}"))
                })?;
        }
        if !out.is_empty() {
            let wrote =
                Write::write_all(guard.writer(), &out).and_then(|()| guard.writer().flush());
            if let Err(err) = wrote {
                return terminal_gone(err, kiosk);
            }
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
    if let Err(err) = guard.restore() {
        return terminal_gone(err, kiosk);
    }
    if let Some(raw) = input.as_mut() {
        raw.restore()?;
    }
    Ok(())
}

/// One kiosk-mode frame (#104): read and decode the console source when it
/// is open, (re)open it when it is due for a retry, and show a status
/// frame instead of exiting when it is down. `out` is cleared and filled
/// with this frame's bytes either way.
#[allow(clippy::too_many_arguments)]
fn kiosk_frame(
    vcsa_path: &Path,
    vcs_path: Option<&Path>,
    vcsu_path: Option<&Path>,
    source: &mut Option<(File, Option<File>, Option<File>)>,
    watch: &mut SourceWatch,
    source_name: &str,
    session: &mut Session,
    size: (u16, u16),
    vcsa_buf: &mut Vec<u8>,
    vcs_buf: &mut Vec<u8>,
    vcsu_buf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) {
    let now = Instant::now();
    if source.is_none() && watch.due(now) {
        match open_source(vcsa_path, vcs_path, vcsu_path) {
            Ok(opened) => *source = Some(opened),
            Err(err) => {
                watch.failed(source_name, &err.to_string(), now);
            }
        }
    }

    let mut failed: Option<String> = None;
    out.clear();
    if let Some((vcsa, vcs, vcsu)) = source.as_mut() {
        match read_vcsa(vcsa, vcsa_buf) {
            Ok(()) => {
                let cells = vcsa_geometry(vcsa_buf)
                    .map_or(0, |(rows, cols)| usize::from(rows) * usize::from(cols));
                let vcs_bytes = read_optional(vcs, cells, 1, vcs_buf);
                let vcsu_bytes = read_optional(vcsu, cells, 4, vcsu_buf);
                if watch.recovered() {
                    session.invalidate();
                }
                if let Err(err) = session.frame(size, vcsa_buf, vcs_bytes, vcsu_bytes, out) {
                    failed = Some(format!("{err:?}"));
                }
            }
            Err(err) => failed = Some(err.to_string()),
        }
    }
    if let Some(reason) = failed {
        *source = None;
        out.clear();
        let line = watch.failed(source_name, &reason, now).to_string();
        session.status_frame(size, &line, out);
    } else if source.is_none() {
        out.clear();
        if let Some(line) = watch.status_line() {
            session.status_frame(size, line, out);
        }
    }
}

/// The terminal is gone (a write failed): in kiosk mode that is not fatal
/// (exit 0, the supervisor restarts llama-view), otherwise it propagates as
/// before (exit 1).
fn terminal_gone(err: io::Error, kiosk: bool) -> io::Result<()> {
    if kiosk { Ok(()) } else { Err(err) }
}

/// Open the vcsa source (required) and its vcs/vcsu siblings (best effort,
/// same as always): used both at startup and, in kiosk mode, on every retry.
fn open_source(
    vcsa_path: &Path,
    vcs_path: Option<&Path>,
    vcsu_path: Option<&Path>,
) -> io::Result<(File, Option<File>, Option<File>)> {
    let vcsa = open_read(vcsa_path)?;
    let vcs = open_optional(vcs_path);
    let vcsu = open_optional(vcsu_path);
    Ok((vcsa, vcs, vcsu))
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
