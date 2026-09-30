//! Every file llama-cast reads besides its config: the tty11 screen, the
//! console font, and the machine id (hashed into the UDN). Each is opened
//! read-only, `O_NOFOLLOW | O_NOCTTY | O_CLOEXEC`, and read with a size cap.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use rustix::fs::{Mode, OFlags};

use crate::config::Font;
use crate::font::{MAX_FONT_BYTES, Psf2};
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
    let file = File::from(fd);
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

/// Produces one RGB24 frame per call.
pub trait FrameSource: Send + Sync + 'static {
    /// Fill `frame` (`render::FRAME_BYTES` long).
    fn frame(&self, frame: &mut [u8]) -> io::Result<()>;
}

/// A vcsa file rendered with a console font.
pub struct VcsaSource {
    path: PathBuf,
    font: Psf2,
}

impl VcsaSource {
    #[must_use]
    pub fn new(path: PathBuf, font: Psf2) -> Self {
        Self { path, font }
    }
}

impl FrameSource for VcsaSource {
    fn frame(&self, frame: &mut [u8]) -> io::Result<()> {
        let bytes = read_capped(&self.path, MAX_VCSA_BYTES)?;
        let screen =
            Screen::parse(&bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        render::render_into(&screen, &self.font, frame);
        Ok(())
    }
}
