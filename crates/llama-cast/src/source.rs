//! Every file llama-cast reads besides its config: the tty11 screen, the
//! grid file (grid mode only), the console font, and the machine id (hashed
//! into the UDN). Each is opened read-only, `O_NOFOLLOW | O_NOCTTY |
//! O_CLOEXEC`, and read with a size cap. The grid is also opened
//! non-blocking and kept only when `fstat` on that descriptor says it is a
//! regular file, so a FIFO there cannot stall the stream. vcsa is a
//! character device and stays on the plain open. Also the host name, from
//! `uname(2)` (no file), for the friendly name. The grid's clock is read
//! here and passed into `grid::compose`; that function does not read the
//! time.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use llama_core::gridfile;
use llama_core::palette::Palette;
use rustix::fs::{FileType, Mode, OFlags};

use crate::config::Font;
use crate::font::{MAX_FONT_BYTES, Psf2};
use crate::grid::{self, GridRead, Hold};
use crate::render::{self, MAX_VCSA_BYTES, Screen};

/// tty11's screen and attributes (`root:llama-view 0640`, udev rule
/// 72-llama-view). Read-only; the unit allows only this device.
pub const VCSA_PATH: &str = "/dev/vcsa11";
/// Hashed into the UDN.
pub const MACHINE_ID_PATH: &str = "/etc/machine-id";
/// Where install.sh puts the llama-hack console fonts.
pub const FONT_DIR: &str = "/usr/local/share/llama-bored";
/// `/etc/machine-id` is 33 bytes; anything past this is refused.
pub const MAX_MACHINE_ID_BYTES: usize = 64;

/// Read at most `cap` bytes of `path`; a longer file is an error.
pub fn read_capped(path: &Path, cap: usize) -> io::Result<Vec<u8>> {
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NOCTTY | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    read_limited(File::from(fd), path, cap)
}

/// The grid file only. Non-blocking, then refused unless `fstat` on the
/// opened descriptor says it is a regular file. No second path lookup.
/// Anything else is an error, which the frame path turns into the
/// unreadable notice. A FIFO opened for reading the ordinary way never
/// returns.
fn read_grid(path: &Path) -> io::Result<Vec<u8>> {
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NOCTTY | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )?;
    let stat = rustix::fs::fstat(&fd)?;
    if !FileType::from_raw_mode(stat.st_mode).is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a regular file",
        ));
    }
    read_limited(File::from(fd), path, grid_cap())
}

fn read_limited(file: File, path: &Path, cap: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let limit = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
    file.take(limit).read_to_end(&mut out)?;
    if out.len() > cap {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is larger than {cap} bytes", path.display()),
        ));
    }
    Ok(out)
}

/// The installed font file for `font`.
#[must_use]
pub fn font_path(font: Font) -> PathBuf {
    Path::new(FONT_DIR).join(font.file_name())
}

/// Load and validate a PSF2 font.
pub fn load_font(path: &Path) -> io::Result<Psf2> {
    let bytes = read_capped(path, MAX_FONT_BYTES)?;
    Psf2::parse(&bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// The UDN for this host, from `/etc/machine-id`.
pub fn machine_udn(path: &Path) -> io::Result<String> {
    let bytes = read_capped(path, MAX_MACHINE_ID_BYTES)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "machine id is not text"))?;
    if !crate::ssdp::valid_machine_id(&text) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "machine id is not 32 hex digits",
        ));
    }
    Ok(crate::ssdp::udn_from_machine_id(&text))
}

/// The kernel's host name (`uname(2)` nodename), raw. The caller keeps
/// only a sanitised first label of it (`config::host_label`).
#[must_use]
pub fn nodename() -> Vec<u8> {
    rustix::system::uname().nodename().to_bytes().to_vec()
}

/// Produces one RGB24 frame per call.
pub trait FrameSource: Send + Sync + 'static {
    /// Fill `frame` (`render::FRAME_BYTES` long).
    fn frame(&self, frame: &mut [u8]) -> io::Result<()>;
}

/// A vcsa file rendered with a console font and palette.
pub struct VcsaSource {
    path: PathBuf,
    font: Psf2,
    palette: Palette,
}

impl VcsaSource {
    #[must_use]
    pub fn new(path: PathBuf, font: Psf2, palette: Palette) -> Self {
        Self {
            path,
            font,
            palette,
        }
    }
}

impl FrameSource for VcsaSource {
    fn frame(&self, frame: &mut [u8]) -> io::Result<()> {
        let bytes = read_capped(&self.path, MAX_VCSA_BYTES)?;
        let screen =
            Screen::parse(&bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        render::render_into(&screen, &self.font, self.palette, frame);
        Ok(())
    }
}

/// Largest grid file the grid reader will accept: the `gridfile` caps.
#[must_use]
pub fn grid_cap() -> usize {
    gridfile::file_len(gridfile::MAX_COLS, gridfile::MAX_ROWS)
}

fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

/// The grid file, painted with one or more console fonts. `auto` picks a
/// font from the file's size on each fresh frame. A bad file returns `Ok`
/// and the picture carries a notice, so the stream keeps going.
pub struct GridSource {
    path: PathBuf,
    fonts: Vec<Psf2>,
    auto: bool,
    hold: Mutex<Hold>,
}

impl GridSource {
    #[must_use]
    pub fn new(path: PathBuf, fonts: Vec<Psf2>, auto: bool) -> Self {
        Self {
            path,
            fonts,
            auto,
            hold: Mutex::new(Hold::new()),
        }
    }
}

impl FrameSource for GridSource {
    fn frame(&self, frame: &mut [u8]) -> io::Result<()> {
        let now = unix_now_ms();
        let read_result = read_grid(&self.path);
        let read = match &read_result {
            Ok(bytes) => GridRead::Data(bytes),
            Err(err) if err.kind() == io::ErrorKind::NotFound => GridRead::Missing,
            Err(_) => GridRead::Unreadable,
        };
        let mut hold = self.hold.lock().unwrap_or_else(|err| err.into_inner());
        grid::compose(read, now, &self.fonts, self.auto, &mut hold, frame)
    }
}

/// What `check` says about a grid file. An unreadable file is a note, not
/// a failed config: llama-watch may be down while `cast.toml` is fine.
pub enum GridInspect {
    Ready { cols: u16, rows: u16, font: String },
    Note(String),
}

/// Read the grid once. A file that decodes, stale or not, reports its size
/// and the font that would be used. Anything else is [`GridInspect::Note`].
pub fn inspect_grid(path: &Path, fonts: &[(Font, Psf2)], auto: bool) -> GridInspect {
    let bytes = match read_grid(path) {
        Ok(bytes) => bytes,
        Err(err) => return GridInspect::Note(format!("grid {}: {err}", path.display())),
    };
    let view = match grid::open_bytes(&bytes) {
        Ok(view) => view,
        Err(err) => return GridInspect::Note(format!("grid {}: {err}", path.display())),
    };
    let index = if auto {
        let kinds: Vec<Font> = fonts.iter().map(|(font, _)| *font).collect();
        let picked = grid::pick_font(&kinds, view.cols, view.rows);
        fonts
            .iter()
            .position(|(font, _)| *font == picked)
            .unwrap_or(0)
    } else {
        0
    };
    let label = fonts
        .get(index)
        .map(|(font, _)| font.label())
        .unwrap_or("12x24");
    let font = if auto {
        format!("auto ({label})")
    } else {
        label.to_owned()
    };
    GridInspect::Ready {
        cols: view.cols,
        rows: view.rows,
        font,
    }
}
