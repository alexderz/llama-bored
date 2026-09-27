use std::io::{self, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_command(&args) {
        Some(Command::Run) => match parse_service_flags(&args[1..]) {
            Ok((config, trace_hid)) => {
                std::process::exit(kraken_lcd::service::run(&config, trace_hid));
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        },
        Some(Command::RestoreStock) => match parse_service_flags(&args[1..]) {
            Ok((config, trace_hid)) => {
                std::process::exit(kraken_lcd::service::restore_stock(&config, trace_hid));
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        },
        Some(Command::ClearHalt) => {
            if args.len() != 1 {
                eprintln!("usage: kraken-lcd clear-halt");
                std::process::exit(2);
            }
            if let Err(err) = clear_halt() {
                eprintln!("{err}");
                std::process::exit(1);
            }
        }
        Some(Command::RenderOnce) => {
            if let Err(err) = render_once(&args[1..]) {
                eprintln!("{err}");
                std::process::exit(exit_code(&err));
            }
        }
        Some(Command::QueryBuckets) => match parse_query_flags(&args[1..]) {
            Ok(trace_hid) => std::process::exit(query_buckets(trace_hid)),
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        },
        Some(Command::ShowImage) => std::process::exit(show_image(&args[1..])),
        Some(Command::BenchUpload) => std::process::exit(bench_upload(&args[1..])),
        None => {
            eprintln!(
                "usage: kraken-lcd <run|restore-stock|clear-halt|render-once|show-image|bench-upload|--query-buckets>"
            );
            std::process::exit(2);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Command {
    Run,
    RestoreStock,
    ClearHalt,
    RenderOnce,
    QueryBuckets,
    ShowImage,
    BenchUpload,
}

fn parse_command(args: &[String]) -> Option<Command> {
    match args.first().map(String::as_str) {
        Some("run") => Some(Command::Run),
        Some("restore-stock") => Some(Command::RestoreStock),
        Some("clear-halt") => Some(Command::ClearHalt),
        Some("render-once") => Some(Command::RenderOnce),
        Some("--query-buckets") => Some(Command::QueryBuckets),
        Some("show-image") => Some(Command::ShowImage),
        Some("bench-upload") => Some(Command::BenchUpload),
        _ => None,
    }
}

#[derive(Debug, Error)]
enum ServiceFlagsError {
    #[error("usage: kraken-lcd <run|restore-stock> --config PATH")]
    Usage,
}

#[derive(Debug, Error)]
enum QueryFlagsError {
    #[error("usage: kraken-lcd --query-buckets [--trace-hid]")]
    Usage,
}

fn parse_query_flags(args: &[String]) -> Result<bool, QueryFlagsError> {
    match args {
        [] => Ok(false),
        [flag] if flag == "--trace-hid" => Ok(true),
        _ => Err(QueryFlagsError::Usage),
    }
}

fn query_refused_as_root(is_root: bool) -> bool {
    is_root
}

/// Hidraw-only `QueryBucket(0..=15)` inside the cooling guard. No bulk port.
fn query_buckets(trace_hid: bool) -> i32 {
    use kraken_lcd::device::{HidLink, KrakenLcd, NoBulk, SYS_ROOT, interface0_usbfs};

    if query_refused_as_root(rustix::process::geteuid().is_root()) {
        eprintln!("query-buckets: refusing to run as root");
        return 1;
    }
    match interface0_usbfs(Path::new(SYS_ROOT)) {
        Ok(true) => {
            eprintln!("query-buckets: refusing; interface 0 is bound to usbfs");
            return 1;
        }
        Ok(false) => {}
        Err(err) => {
            eprintln!("query-buckets: {err}");
            return query_exit(err);
        }
    }
    match KrakenLcd::<HidLink, NoBulk>::query_buckets_resolved(trace_hid) {
        Ok(rows) => {
            print!("{}", kraken_lcd::device::format_bucket_table(&rows));
            0
        }
        Err(err) => {
            eprintln!("query-buckets: {err}");
            query_exit(err)
        }
    }
}

fn query_exit(err: kraken_lcd::device::SinkError) -> i32 {
    match err {
        kraken_lcd::device::SinkError::Fence => 2,
        _ => 1,
    }
}

struct ShowImageFlags {
    view: PathBuf,
    bucket: u8,
    trace_hid: bool,
}

fn parse_show_image(args: &[String]) -> Result<ShowImageFlags, CliError> {
    let mut view = None;
    let mut bucket = None;
    let mut trace_hid = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--view" => {
                let Some(path) = args.get(index + 1) else {
                    return Err(CliError::ShowImageUsage);
                };
                if view.is_some() {
                    return Err(CliError::ShowImageUsage);
                }
                view = Some(PathBuf::from(path));
                index += 2;
            }
            "--bucket" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliError::ShowImageUsage);
                };
                if bucket.is_some() {
                    return Err(CliError::ShowImageUsage);
                }
                let parsed: u8 = value.parse().map_err(|_| CliError::ShowImageUsage)?;
                if parsed >= kraken_lcd::device::proto::SLOT_COUNT {
                    return Err(CliError::ShowImageUsage);
                }
                bucket = Some(parsed);
                index += 2;
            }
            "--trace-hid" => {
                if trace_hid {
                    return Err(CliError::ShowImageUsage);
                }
                trace_hid = true;
                index += 1;
            }
            _ => return Err(CliError::ShowImageUsage),
        }
    }
    match view {
        Some(view) => Ok(ShowImageFlags {
            view,
            bucket: bucket.unwrap_or(0),
            trace_hid,
        }),
        None => Err(CliError::ShowImageUsage),
    }
}

/// Render `--view` then take the same one-frame device path as `run`.
///
/// The view file is opened before any device step. A guard trip or error
/// uses the same HALTED latch as `run` and does not retry.
fn show_image(args: &[String]) -> i32 {
    use kraken_lcd::device::{KrakenLcd, SYS_ROOT, interface0_usbfs};

    let flags = match parse_show_image(args) {
        Ok(flags) => flags,
        Err(err) => {
            eprintln!("{err}");
            return exit_code(&err);
        }
    };
    let view = match read_view(&flags.view) {
        Ok(view) => view,
        Err(err) => {
            eprintln!("{err}");
            return exit_code(&err);
        }
    };
    let mut assets = match kraken_lcd::render::Assets::load() {
        Ok(assets) => assets,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let frame = kraken_lcd::render::render(
        &view,
        &kraken_lcd::config::DisplayCfg::default(),
        &mut assets,
    );
    let slot = match kraken_lcd::device::proto::SlotId::try_new(flags.bucket) {
        Ok(slot) => slot,
        Err(_) => {
            eprintln!("{}", CliError::ShowImageUsage);
            return 2;
        }
    };
    if query_refused_as_root(rustix::process::geteuid().is_root()) {
        eprintln!("show-image: refusing to run as root");
        return 1;
    }
    match interface0_usbfs(Path::new(SYS_ROOT)) {
        Ok(true) => {
            eprintln!("show-image: refusing; interface 0 is bound to usbfs");
            return 1;
        }
        Ok(false) => {}
        Err(err) => {
            eprintln!("show-image: {err}");
            return query_exit(err);
        }
    }
    match KrakenLcd::show_image_resolved(slot, &frame, flags.trace_hid) {
        Ok(rows) => {
            print!("{}", kraken_lcd::device::format_bucket_table(&rows));
            0
        }
        Err(err) => {
            eprintln!("show-image: {err}");
            query_exit(err)
        }
    }
}

struct BenchUploadFlags {
    view: PathBuf,
    spec: kraken_lcd::device::BenchSpec,
    trace_hid: bool,
}

fn parse_bench_upload(args: &[String]) -> Result<BenchUploadFlags, CliError> {
    let mut view = None;
    let mut count = None;
    let mut fps = None;
    let mut slots = None;
    let mut trace_hid = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--view" => {
                let Some(path) = args.get(index + 1) else {
                    return Err(CliError::BenchUploadUsage);
                };
                if view.is_some() {
                    return Err(CliError::BenchUploadUsage);
                }
                view = Some(PathBuf::from(path));
                index += 2;
            }
            "--count" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliError::BenchUploadUsage);
                };
                if count.is_some() {
                    return Err(CliError::BenchUploadUsage);
                }
                let parsed: u32 = value.parse().map_err(|_| CliError::BenchUploadUsage)?;
                count = Some(parsed);
                index += 2;
            }
            "--fps" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliError::BenchUploadUsage);
                };
                if fps.is_some() {
                    return Err(CliError::BenchUploadUsage);
                }
                let parsed: f64 = value.parse().map_err(|_| CliError::BenchUploadUsage)?;
                fps = Some(parsed);
                index += 2;
            }
            "--slots" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(CliError::BenchUploadUsage);
                };
                if slots.is_some() {
                    return Err(CliError::BenchUploadUsage);
                }
                let parsed: u8 = value.parse().map_err(|_| CliError::BenchUploadUsage)?;
                slots = Some(parsed);
                index += 2;
            }
            "--trace-hid" => {
                if trace_hid {
                    return Err(CliError::BenchUploadUsage);
                }
                trace_hid = true;
                index += 1;
            }
            _ => return Err(CliError::BenchUploadUsage),
        }
    }
    let (Some(view), Some(count), Some(fps)) = (view, count, fps) else {
        return Err(CliError::BenchUploadUsage);
    };
    let spec = kraken_lcd::device::BenchSpec::try_new(count, fps, slots.unwrap_or(2))
        .map_err(|_| CliError::BenchUploadUsage)?;
    Ok(BenchUploadFlags {
        view,
        spec,
        trace_hid,
    })
}

/// Render `--view` then ping-pong the `show-image` one-frame path `count` times.
///
/// The view file is opened before any device step. A guard trip latches. An
/// upload failure exits 1 without latching and without retrying.
fn bench_upload(args: &[String]) -> i32 {
    use kraken_lcd::device::{KrakenLcd, SYS_ROOT, interface0_usbfs};

    let flags = match parse_bench_upload(args) {
        Ok(flags) => flags,
        Err(err) => {
            eprintln!("{err}");
            return exit_code(&err);
        }
    };
    let view = match read_view(&flags.view) {
        Ok(view) => view,
        Err(err) => {
            eprintln!("{err}");
            return exit_code(&err);
        }
    };
    let mut assets = match kraken_lcd::render::Assets::load() {
        Ok(assets) => assets,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let frame = kraken_lcd::render::render(
        &view,
        &kraken_lcd::config::DisplayCfg::default(),
        &mut assets,
    );
    if query_refused_as_root(rustix::process::geteuid().is_root()) {
        eprintln!("bench-upload: refusing to run as root");
        return 1;
    }
    match interface0_usbfs(Path::new(SYS_ROOT)) {
        Ok(true) => {
            eprintln!("bench-upload: refusing; interface 0 is bound to usbfs");
            return 1;
        }
        Ok(false) => {}
        Err(err) => {
            eprintln!("bench-upload: {err}");
            return query_exit(err);
        }
    }
    match KrakenLcd::bench_upload_resolved(&frame, flags.spec, flags.trace_hid, &mut io::stdout()) {
        Ok(_) => 0,
        Err(err) => {
            eprintln!("bench-upload: {err}");
            query_exit(err)
        }
    }
}

fn parse_service_flags(args: &[String]) -> Result<(PathBuf, bool), ServiceFlagsError> {
    let mut config = None;
    let mut trace_hid = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                let Some(path) = args.get(index + 1) else {
                    return Err(ServiceFlagsError::Usage);
                };
                if config.is_some() {
                    return Err(ServiceFlagsError::Usage);
                }
                config = Some(PathBuf::from(path));
                index += 2;
            }
            "--trace-hid" => {
                trace_hid = true;
                index += 1;
            }
            _ => return Err(ServiceFlagsError::Usage),
        }
    }
    match config {
        Some(config) => Ok((config, trace_hid)),
        None => Err(ServiceFlagsError::Usage),
    }
}

#[derive(Debug, Error)]
enum ClearHaltError {
    #[error("clear-halt refuses unless euid is 0")]
    NotRoot,
    #[error("confirmation was not YES")]
    NotConfirmed,
    #[error("failed to remove {path}: {source}")]
    Remove {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to read confirmation: {0}")]
    Stdin(#[from] io::Error),
}

fn halt_latch() -> PathBuf {
    kraken_lcd::service::halt_latch_path()
}

fn clear_halt() -> Result<(), ClearHaltError> {
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    clear_halt_at(
        &halt_latch(),
        rustix::process::geteuid().is_root(),
        &mut stdin,
        &mut io::stderr(),
    )
}

fn clear_halt_at(
    path: &Path,
    is_root: bool,
    stdin: &mut impl io::BufRead,
    stderr: &mut impl Write,
) -> Result<(), ClearHaltError> {
    if !is_root {
        return Err(ClearHaltError::NotRoot);
    }
    match path.symlink_metadata() {
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            writeln!(stderr, "no halt latch at {}", path.display())?;
            return Ok(());
        }
        Err(err) => writeln!(stderr, "latch at {}: ({err})", path.display())?,
        Ok(_) => {
            let body = std::fs::read_to_string(path).unwrap_or_else(|err| format!("({err})"));
            writeln!(stderr, "latch at {}:\n{body}", path.display())?;
        }
    }
    write!(stderr, "type YES to remove: ")?;
    stderr.flush()?;
    let mut line = String::new();
    stdin.read_line(&mut line)?;
    if line.trim() != "YES" {
        return Err(ClearHaltError::NotConfirmed);
    }
    std::fs::remove_file(path).map_err(|source| ClearHaltError::Remove {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

struct RenderOncePaths {
    snapshot: PathBuf,
    out: PathBuf,
}

#[derive(Debug, Error)]
enum CliError {
    #[error("usage: kraken-lcd render-once --snapshot FILE.json --out FILE.png")]
    Usage,
    #[error(
        "usage: kraken-lcd show-image --view FILE.json [--bucket N] [--trace-hid]\n  rotation is fixed at 0\n  for the power-cycle test, use fixtures/views/test-card.json"
    )]
    ShowImageUsage,
    #[error(
        "usage: kraken-lcd bench-upload --view FILE --count N --fps F [--slots 2] [--trace-hid]\n  1 ≤ N ≤ 36000, 0 < F ≤ 30, slots must be 2\n  rotation is fixed at 0"
    )]
    BenchUploadUsage,
    #[error("failed to read snapshot {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse snapshot {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("view {path} is a symlink")]
    ViewSymlink { path: PathBuf },
    #[error("view {path} is too large")]
    ViewTooLarge { path: PathBuf },
    #[error("view {path} is not a regular file")]
    ViewNotRegular { path: PathBuf },
    #[error("failed to load assets: {0}")]
    Assets(#[from] kraken_lcd::render::AssetError),
    #[error("failed to write png {path}: {detail}")]
    Write { path: PathBuf, detail: String },
    /// `--out` exists and is a symlink or not a regular file.
    #[error("--out {path} {reason}")]
    OutPath { path: PathBuf, reason: &'static str },
}

fn parse_render_once(args: &[String]) -> Result<RenderOncePaths, CliError> {
    let mut snapshot = None;
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        let slot = match args[index].as_str() {
            "--snapshot" => &mut snapshot,
            "--out" => &mut out,
            _ => return Err(CliError::Usage),
        };
        let Some(path) = args.get(index + 1) else {
            return Err(CliError::Usage);
        };
        if slot.is_some() {
            return Err(CliError::Usage);
        }
        *slot = Some(PathBuf::from(path));
        index += 2;
    }
    match (snapshot, out) {
        (Some(snapshot), Some(out)) => Ok(RenderOncePaths { snapshot, out }),
        _ => Err(CliError::Usage),
    }
}

/// Read a serialised [`kraken_lcd::present::View`], render it, and write one PNG.
/// No device and no start-up self-checks beyond decoding the bundled assets.
/// The still shows the peg's smoke as it looks four seconds in.
fn render_once(args: &[String]) -> Result<(), CliError> {
    let paths = parse_render_once(args)?;
    let mut view = read_view(&paths.snapshot)?;
    kraken_lcd::present::warm_still(&mut view);
    let mut assets = kraken_lcd::render::Assets::load()?;
    let frame = kraken_lcd::render::render(
        &view,
        &kraken_lcd::config::DisplayCfg::default(),
        &mut assets,
    );
    write_png(&paths.out, &frame)
}

/// Largest view JSON accepted before parsing. Matches the snapshot cap.
const VIEW_MAX_BYTES: usize = 16 * 1024;

fn read_view(path: &Path) -> Result<kraken_lcd::present::View, CliError> {
    let bytes = read_view_bytes(path)?;
    serde_json::from_slice(&bytes).map_err(|source| CliError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

fn read_view_bytes(path: &Path) -> Result<Vec<u8>, CliError> {
    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .map_err(|err| map_view_open(path, err))?;
    let stat = rustix::fs::fstat(&fd).map_err(|err| view_io(path, err))?;
    let kind = rustix::fs::FileType::from_raw_mode(stat.st_mode);
    if !kind.is_file() {
        return Err(CliError::ViewNotRegular {
            path: path.to_path_buf(),
        });
    }
    let max = i64::try_from(VIEW_MAX_BYTES).unwrap_or(i64::MAX);
    if stat.st_size < 0 || stat.st_size > max {
        return Err(CliError::ViewTooLarge {
            path: path.to_path_buf(),
        });
    }
    let mut buf = vec![0_u8; VIEW_MAX_BYTES + 1];
    let mut filled = 0_usize;
    while filled < buf.len() {
        match rustix::io::read(&fd, &mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(err) if err == rustix::io::Errno::INTR => continue,
            Err(err) => return Err(view_io(path, err)),
        }
    }
    if filled > VIEW_MAX_BYTES {
        return Err(CliError::ViewTooLarge {
            path: path.to_path_buf(),
        });
    }
    buf.truncate(filled);
    Ok(buf)
}

fn map_view_open(path: &Path, err: rustix::io::Errno) -> CliError {
    if err == rustix::io::Errno::LOOP {
        CliError::ViewSymlink {
            path: path.to_path_buf(),
        }
    } else {
        view_io(path, err)
    }
}

fn view_io(path: &Path, err: rustix::io::Errno) -> CliError {
    CliError::Read {
        path: path.to_path_buf(),
        source: std::io::Error::from_raw_os_error(err.raw_os_error()),
    }
}

fn exit_code(err: &CliError) -> i32 {
    match err {
        CliError::Usage
        | CliError::ShowImageUsage
        | CliError::BenchUploadUsage
        | CliError::OutPath { .. } => 2,
        _ => 1,
    }
}

fn write_png(path: &Path, frame: &kraken_lcd::render::Frame) -> Result<(), CliError> {
    // Render fills opaque black and later layers composite onto it, so every
    // pixel's alpha is 255. The premultiplied buffer is therefore straight RGBA.
    debug_assert!(
        frame.0.pixels().iter().all(|pixel| pixel.alpha() == 255),
        "render paints opaque pixels"
    );
    let file = open_out(path)?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        frame.0.width(),
        frame.0.height(),
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|err| CliError::Write {
        path: path.to_path_buf(),
        detail: err.to_string(),
    })?;
    writer
        .write_image_data(frame.0.data())
        .map_err(|err| CliError::Write {
            path: path.to_path_buf(),
            detail: err.to_string(),
        })?;
    writer.finish().map_err(|err| CliError::Write {
        path: path.to_path_buf(),
        detail: err.to_string(),
    })?;
    Ok(())
}

/// Create a new file, or truncate an existing regular file. A symlink or any
/// other existing node is refused before it is opened.
fn open_out(path: &Path) -> Result<std::fs::File, CliError> {
    match std::fs::symlink_metadata(path) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|err| write_err(path, err)),
        Err(err) => Err(write_err(path, err)),
        Ok(meta) if meta.file_type().is_symlink() => Err(CliError::OutPath {
            path: path.to_path_buf(),
            reason: "is a symlink",
        }),
        Ok(meta) if !meta.file_type().is_file() => Err(CliError::OutPath {
            path: path.to_path_buf(),
            reason: "is not a regular file",
        }),
        Ok(_) => std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|err| write_err(path, err)),
    }
}

fn write_err(path: &Path, err: std::io::Error) -> CliError {
    CliError::Write {
        path: path.to_path_buf(),
        detail: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn args(cmd: &str) -> Vec<String> {
        vec![cmd.to_owned()]
    }

    #[test]
    fn known_commands_are_recognized() {
        assert_eq!(parse_command(&args("run")), Some(Command::Run));
        assert_eq!(
            parse_command(&args("restore-stock")),
            Some(Command::RestoreStock)
        );
        assert_eq!(parse_command(&args("clear-halt")), Some(Command::ClearHalt));
        assert_eq!(
            parse_command(&args("render-once")),
            Some(Command::RenderOnce)
        );
        assert_eq!(
            parse_command(&args("--query-buckets")),
            Some(Command::QueryBuckets)
        );
        assert_eq!(parse_command(&args("show-image")), Some(Command::ShowImage));
        assert_eq!(
            parse_command(&args("bench-upload")),
            Some(Command::BenchUpload)
        );
    }

    #[test]
    fn query_buckets_accepts_only_the_hidden_trace_flag() {
        assert!(!parse_query_flags(&[]).expect("bare command"));
        assert!(parse_query_flags(&["--trace-hid".to_owned()]).expect("trace"));
        assert!(parse_query_flags(&["--config".to_owned(), "a.toml".to_owned()]).is_err());
        assert!(parse_query_flags(&["--trace-hid".to_owned(), "--trace-hid".to_owned()]).is_err());
    }

    #[test]
    fn query_buckets_refuses_to_run_as_root() {
        assert!(query_refused_as_root(true));
        assert!(!query_refused_as_root(false));
    }

    #[test]
    fn query_buckets_fence_exits_2_and_other_failures_exit_1() {
        use kraken_lcd::device::SinkError;
        assert_eq!(query_exit(SinkError::Fence), 2);
        assert_eq!(query_exit(SinkError::Halted), 1);
        assert_eq!(query_exit(SinkError::DeviceUnavailable), 1);
        assert_eq!(query_exit(SinkError::DeviceInBootloader), 1);
    }

    #[test]
    fn unknown_command_is_none() {
        assert_eq!(parse_command(&args("upload")), None);
        assert_eq!(parse_command(&[]), None);
    }

    #[test]
    fn halt_latch_is_under_the_state_dir() {
        assert_eq!(
            halt_latch(),
            PathBuf::from(kraken_lcd::device::STATE_DIR).join("halted")
        );
    }

    #[test]
    fn clear_halt_with_no_latch_exits_ok_without_prompting() {
        let missing = std::env::temp_dir().join(format!(
            "t14-no-latch-{}-missing",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_file(&missing);
        let mut stdin = io::Cursor::new(Vec::<u8>::new());
        let mut stderr = Vec::new();
        clear_halt_at(&missing, true, &mut stdin, &mut stderr).expect("no latch");
        let text = String::from_utf8(stderr).expect("utf8");
        assert!(text.contains("no halt latch"), "{text}");
        assert!(!text.contains("type YES"), "{text}");
        assert_eq!(stdin.position(), 0, "stdin was not read");
    }

    #[test]
    fn run_requires_config_and_accepts_hidden_trace_hid() {
        let parsed = parse_service_flags(&[
            "--trace-hid".to_owned(),
            "--config".to_owned(),
            "/etc/llama-bored/config.toml".to_owned(),
        ])
        .expect("flags");
        assert_eq!(parsed.0, PathBuf::from("/etc/llama-bored/config.toml"));
        assert!(parsed.1);
        assert!(parse_service_flags(&[]).is_err());
        assert!(parse_service_flags(&["--trace-hid".to_owned()]).is_err());
        assert!(
            parse_service_flags(&[
                "--config".to_owned(),
                "a.toml".to_owned(),
                "--device".to_owned()
            ])
            .is_err()
        );
    }

    #[test]
    fn render_once_accepts_snapshot_and_out_in_either_order() {
        let parsed = parse_render_once(&[
            "--out".to_owned(),
            "frame.png".to_owned(),
            "--snapshot".to_owned(),
            "view.json".to_owned(),
        ])
        .expect("flags");
        assert_eq!(parsed.snapshot, PathBuf::from("view.json"));
        assert_eq!(parsed.out, PathBuf::from("frame.png"));
    }

    #[test]
    fn render_once_rejects_missing_unknown_and_duplicate_flags() {
        assert!(parse_render_once(&[]).is_err());
        assert!(parse_render_once(&["--snapshot".to_owned(), "view.json".to_owned()]).is_err());
        assert!(
            parse_render_once(&[
                "--snapshot".to_owned(),
                "view.json".to_owned(),
                "--out".to_owned(),
                "frame.png".to_owned(),
                "--device".to_owned(),
            ])
            .is_err()
        );
        assert!(
            parse_render_once(&[
                "--snapshot".to_owned(),
                "a.json".to_owned(),
                "--snapshot".to_owned(),
                "b.json".to_owned(),
                "--out".to_owned(),
                "frame.png".to_owned(),
            ])
            .is_err()
        );
    }

    #[test]
    fn show_image_accepts_view_bucket_and_trace_in_any_order() {
        let parsed = parse_show_image(&[
            "--trace-hid".to_owned(),
            "--bucket".to_owned(),
            "3".to_owned(),
            "--view".to_owned(),
            "view.json".to_owned(),
        ])
        .expect("flags");
        assert_eq!(parsed.view, PathBuf::from("view.json"));
        assert_eq!(parsed.bucket, 3);
        assert!(parsed.trace_hid);
    }

    #[test]
    fn show_image_defaults_bucket_to_the_first_rotation_slot() {
        let parsed =
            parse_show_image(&["--view".to_owned(), "view.json".to_owned()]).expect("flags");
        assert_eq!(parsed.bucket, 0);
        assert!(!parsed.trace_hid);
    }

    #[test]
    fn show_image_refuses_a_bucket_outside_the_rotation_before_any_device_step() {
        assert!(
            parse_show_image(&[
                "--view".to_owned(),
                "view.json".to_owned(),
                "--bucket".to_owned(),
                "8".to_owned(),
            ])
            .is_err()
        );
        assert!(
            parse_show_image(&[
                "--view".to_owned(),
                "view.json".to_owned(),
                "--bucket".to_owned(),
                "16".to_owned(),
            ])
            .is_err()
        );
        assert!(parse_show_image(&["--bucket".to_owned(), "0".to_owned()]).is_err());
    }

    fn view_scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kraken-lcd-show-image-{name}-{}",
            rustix::process::getpid().as_raw_nonzero().get()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn show_image_refuses_a_symlink_view_before_any_device_step() {
        let dir = view_scratch("symlink");
        let target = dir.join("view.json");
        std::fs::write(&target, br#"{"ring_pct":null,"ring_band":null,"blocks":["Quiet","Quiet","Quiet"],"coolant_c":null,"cpu_c":null,"gpu_c":null,"cpu_pct":null,"mem_pct":null,"ai":"Down","models":[],"model_count":0}"#).expect("target");
        let link = dir.join("link.json");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let err = read_view(&link).expect_err("symlink");
        assert!(matches!(err, CliError::ViewSymlink { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn show_image_refuses_an_oversize_view_before_any_device_step() {
        let dir = view_scratch("oversize");
        let path = dir.join("view.json");
        let body = vec![b'x'; VIEW_MAX_BYTES + 1];
        std::fs::write(&path, body).expect("oversize");
        let err = read_view(&path).expect_err("oversize");
        assert!(matches!(err, CliError::ViewTooLarge { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bench_upload_accepts_view_count_fps_slots_and_trace() {
        let parsed = parse_bench_upload(&[
            "--trace-hid".to_owned(),
            "--slots".to_owned(),
            "2".to_owned(),
            "--fps".to_owned(),
            "10".to_owned(),
            "--count".to_owned(),
            "8".to_owned(),
            "--view".to_owned(),
            "view.json".to_owned(),
        ])
        .expect("flags");
        assert_eq!(parsed.view, PathBuf::from("view.json"));
        assert_eq!(parsed.spec.count, 8);
        assert_eq!(parsed.spec.fps, 10.0);
        assert_eq!(parsed.spec.slots, 2);
        assert!(parsed.trace_hid);
    }

    #[test]
    fn bench_upload_defaults_slots_to_two() {
        let parsed = parse_bench_upload(&[
            "--view".to_owned(),
            "view.json".to_owned(),
            "--count".to_owned(),
            "1".to_owned(),
            "--fps".to_owned(),
            "0.2".to_owned(),
        ])
        .expect("flags");
        assert_eq!(parsed.spec.slots, 2);
        assert!(!parsed.trace_hid);
    }

    #[test]
    fn bench_upload_rejects_out_of_range_count_fps_and_slots() {
        let view_count_fps = |count: &str, fps: &str, slots: Option<&str>| {
            let mut args = vec![
                "--view".to_owned(),
                "view.json".to_owned(),
                "--count".to_owned(),
                count.to_owned(),
                "--fps".to_owned(),
                fps.to_owned(),
            ];
            if let Some(slots) = slots {
                args.push("--slots".to_owned());
                args.push(slots.to_owned());
            }
            parse_bench_upload(&args)
        };
        assert!(view_count_fps("0", "10", None).is_err());
        assert!(view_count_fps("36001", "10", None).is_err());
        assert!(view_count_fps("1", "0", None).is_err());
        assert!(view_count_fps("1", "30.1", None).is_err());
        assert!(view_count_fps("1", "10", Some("8")).is_err());
        assert!(view_count_fps("1", "10", Some("1")).is_err());
        assert!(view_count_fps("36000", "30", Some("2")).is_ok());
        assert!(parse_bench_upload(&["--count".to_owned(), "1".to_owned()]).is_err());
    }

    #[test]
    fn show_image_usage_recommends_the_test_card_and_fixed_rotation() {
        let text = CliError::ShowImageUsage.to_string();
        assert!(text.contains("fixtures/views/test-card.json"), "{text}");
        assert!(text.contains("rotation is fixed at 0"), "{text}");
    }

    #[test]
    fn show_image_refuses_a_bad_view_before_any_device_step() {
        let dir = view_scratch("bad-json");
        let path = dir.join("view.json");
        std::fs::write(&path, b"{}").expect("fixture");
        let err = read_view(&path).expect_err("not a view");
        assert!(matches!(err, CliError::Parse { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
