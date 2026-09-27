//! Read `/dev/vcsaN` and paint the changed cells on the local terminal.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;
use std::thread;
use std::time::Duration;

use llama_view::{
    CliError, HELP, Options, RESTORE, TermGuard, crop, decode_screen, device_paths, parse_args,
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
        Ok(opts) => match mirror(opts) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("llama-view: {err}");
                ExitCode::from(1)
            }
        },
    }
}

fn mirror(opts: Options) -> io::Result<()> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_restore_stdout();
        previous(info);
    }));

    let (vcsa_path, vcs_path, vcsu_path) = device_paths(&opts);
    let mut vcsa = open_read(&vcsa_path)?;
    let mut vcs = open_optional(vcs_path.as_deref());
    let mut vcsu = open_optional(vcsu_path.as_deref());
    let mut guard = TermGuard::enter(io::stdout())?;
    let mut renderer = llama_view::Renderer::new();
    let period = frame_period(opts.fps);
    let mut vcsa_buf = Vec::new();
    let mut vcs_buf = Vec::new();
    let mut vcsu_buf = Vec::new();

    loop {
        // Every frame, so a resize is picked up on the next poll.
        let size = terminal_size();
        read_vcsa(&mut vcsa, &mut vcsa_buf)?;
        let vcs_bytes = read_optional(&mut vcs, vcsa_buf[0], vcsa_buf[1], 1, &mut vcs_buf);
        let vcsu_bytes = read_optional(&mut vcsu, vcsa_buf[0], vcsa_buf[1], 4, &mut vcsu_buf);
        let screen = decode_screen(&vcsa_buf, vcs_bytes, vcsu_bytes)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("vcsa: {err:?}")))?;
        let view = crop(&screen, size.0, size.1, opts.offset_x, opts.offset_y);
        let frame = renderer.render(view.cols, view.rows, &view.cells);
        if !frame.is_empty() {
            Write::write_all(guard.writer(), frame)?;
            guard.writer().flush()?;
        }
        if opts.once {
            break;
        }
        thread::sleep(period);
    }
    guard.restore()?;
    Ok(())
}

fn frame_period(fps: u8) -> Duration {
    Duration::from_nanos(1_000_000_000 / u64::from(fps.max(1)))
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

fn read_vcsa(file: &mut File, buf: &mut Vec<u8>) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    buf.resize(4, 0);
    file.read_exact(&mut buf[..4])?;
    let cells = usize::from(buf[0]).saturating_mul(usize::from(buf[1]));
    let total = 4usize.saturating_add(cells.saturating_mul(2));
    buf.resize(total, 0);
    file.read_exact(&mut buf[4..])?;
    Ok(())
}

fn read_optional<'a>(
    file: &mut Option<File>,
    rows: u8,
    cols: u8,
    width: usize,
    buf: &'a mut Vec<u8>,
) -> Option<&'a [u8]> {
    let file = file.as_mut()?;
    let n = usize::from(rows)
        .saturating_mul(usize::from(cols))
        .saturating_mul(width);
    if file.seek(SeekFrom::Start(0)).is_err() {
        return None;
    }
    buf.resize(n, 0);
    if file.read_exact(buf).is_err() {
        return None;
    }
    Some(buf.as_slice())
}

fn write_restore_stdout() {
    let mut out = io::stdout();
    let _ = Write::write_all(&mut out, RESTORE);
    let _ = out.flush();
}
