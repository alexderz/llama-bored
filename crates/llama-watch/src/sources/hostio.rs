//! Disk, network, swap and pressure from a proc and sys root.
//!
//! The collector reads these on its existing tick. Rates are taken over a
//! trailing window ([`RateWindow`]); the render path only copies numbers
//! that are already in hand. A missing file is absent, not a failure: test
//! roots and a kernel without one of the files must not light `errors`. So
//! is a file with nothing in it to count (no whole disk, no physical
//! interface). A file that is present but not the format the kernel writes
//! is a failure.
//!
//! Nothing here opens the real `/proc` or `/sys`. Callers pass [`Roots`].

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::time::{Duration, Instant};

use super::Roots;

/// Longest gap between two samples a rate may cross. Longer than this (a
/// stall, a sleep) starts a new baseline instead of publishing a spike.
const MAX_RATE_GAP: Duration = Duration::from_secs(5);

/// Longest span a rate may cover: a [`RateWindow`] of up to 5 s
/// (`collector.cpu_window_s` is at most 5) plus one tick of up to
/// [`MAX_RATE_GAP`]. A longer `dt` is not a window but a stall.
const MAX_RATE_SPAN: Duration = Duration::from_secs(10);

/// 1 Gbit/s in bytes/s. The NET ceiling of an up link without a positive
/// `speed` (Wi-Fi reports -1), and of a host with no link up.
pub const ONE_GBIT_BPS: f64 = 125_000_000.0;

/// How often the link ceiling is read again. `speed` can take the RTNL
/// lock on some drivers, and a link changes speed rarely; a new or gone
/// interface re-reads at once.
const LINK_REFRESH: Duration = Duration::from_secs(5);

/// Linux `EOPNOTSUPP`: a pressure file on a kernel booted with `psi=0`.
/// std leaves it uncategorised, so the raw number is checked as well.
const EOPNOTSUPP: i32 = 95;

/// One host read. `Missing` is "not there"; `Failed` is "there, but wrong".
#[derive(Clone, Debug, PartialEq)]
pub enum Probe<T> {
    Ready(T),
    Missing,
    Failed(String),
}

/// Summed sector counters of whole disks (`nvme0n1`, `sda`, …).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DiskCounters {
    pub read_sectors: u64,
    pub write_sectors: u64,
}

/// Summed byte counters of physical interfaces, plus their ceiling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NetCounters {
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    /// Sum of the up links' ceilings, bytes/s ([`LinkSpeeds`]).
    pub ceiling_bps: Option<u64>,
}

/// Swap from `SwapTotal` and `SwapFree`, bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SwapBytes {
    pub used: u64,
    pub total: u64,
}

/// PSI `some avg10`, percent, for the three pressure files.
///
/// A file that is not there leaves that field `None`. `Missing` is only
/// when none of the three files are there.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PsiSome {
    pub cpu: Option<f32>,
    pub io: Option<f32>,
    pub memory: Option<f32>,
}

/// Whether the tty health word is ok, down, or left off the failure list.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceHealth {
    #[default]
    Absent,
    Ok,
    Down,
}

/// What one collector tick knows about disk, net, swap and PSI.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostIo {
    pub disk_read_bps: Option<f64>,
    pub disk_write_bps: Option<f64>,
    pub net_rx_bps: Option<f64>,
    pub net_tx_bps: Option<f64>,
    /// Link-speed ceiling, bytes/s. `None` until a physical interface is seen.
    pub net_ceiling_bps: Option<f64>,
    pub swap_used_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub cpu_psi: Option<f32>,
    pub io_psi: Option<f32>,
    pub mem_psi: Option<f32>,
    pub disk: SourceHealth,
    pub net: SourceHealth,
    pub psi: SourceHealth,
}

/// Whole-disk sector counters from `{proc}/diskstats`.
pub fn read_disk(roots: &Roots) -> Probe<DiskCounters> {
    let text = match read_file(&roots.proc.join("diskstats")) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Probe::Missing,
        Err(err) => return Probe::Failed(err.to_string()),
    };
    match parse_diskstats(&text) {
        Ok(Some(counters)) => Probe::Ready(counters),
        // No whole disk to count: absent, not a zero rate.
        Ok(None) => Probe::Missing,
        Err(err) => Probe::Failed(err),
    }
}

/// The NET ceiling, re-read at most every [`LINK_REFRESH`] or when the
/// set of physical interfaces changes.
#[derive(Clone, Debug, Default)]
pub struct LinkSpeeds {
    /// `(read at, physical interface names, summed ceiling)`.
    last: Option<(Instant, Vec<String>, u64)>,
}

impl LinkSpeeds {
    fn ceiling(&mut self, roots: &Roots, names: Vec<String>, mono: Instant) -> u64 {
        if let Some((at, seen, ceiling)) = &self.last
            && *seen == names
            && *at <= mono
            && mono.duration_since(*at) < LINK_REFRESH
        {
            return *ceiling;
        }
        let ceiling = links_ceiling_bps(roots, &names);
        self.last = Some((mono, names, ceiling));
        ceiling
    }
}

/// Physical-interface byte counters from `{proc}/net/dev` and `{sys}`.
/// `links` keeps the ceiling between ticks.
pub fn read_net(roots: &Roots, links: &mut LinkSpeeds, mono: Instant) -> Probe<NetCounters> {
    let text = match read_file(&roots.proc.join("net/dev")) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Probe::Missing,
        Err(err) => return Probe::Failed(err.to_string()),
    };
    let rows = match parse_netdev(&text) {
        Ok(rows) => rows,
        Err(err) => return Probe::Failed(err),
    };
    let mut total = NetCounters::default();
    let mut physical = Vec::new();
    for (name, rx, tx) in rows {
        if !has_device_link(&roots.sys.join("class/net").join(&name)) {
            continue;
        }
        total.rx_bytes = total.rx_bytes.saturating_add(rx);
        total.tx_bytes = total.tx_bytes.saturating_add(tx);
        physical.push(name);
    }
    if physical.is_empty() {
        // No physical interface (no sysfs, a container): absent, not a
        // zero rate.
        return Probe::Missing;
    }
    total.ceiling_bps = Some(links.ceiling(roots, physical, mono));
    Probe::Ready(total)
}

/// Swap used and total from `{proc}/meminfo`. Missing keys are absent.
pub fn read_swap(roots: &Roots) -> Probe<SwapBytes> {
    let text = match read_file(&roots.proc.join("meminfo")) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Probe::Missing,
        Err(err) => return Probe::Failed(err.to_string()),
    };
    match parse_swap(&text) {
        Ok(Some(swap)) => Probe::Ready(swap),
        Ok(None) => Probe::Missing,
        Err(err) => Probe::Failed(err),
    }
}

/// PSI `some avg10` from `{proc}/pressure/{cpu,io,memory}`.
pub fn read_psi(roots: &Roots) -> Probe<PsiSome> {
    let mut any = false;
    let mut failed: Option<String> = None;
    let mut out = PsiSome::default();
    for (name, slot) in [
        ("cpu", &mut out.cpu),
        ("io", &mut out.io),
        ("memory", &mut out.memory),
    ] {
        match read_file(&roots.proc.join("pressure").join(name)) {
            Ok(text) => {
                any = true;
                match parse_some_avg10(&text) {
                    Ok(value) => *slot = Some(value),
                    Err(err) => failed = Some(format!("{name}: {err}")),
                }
            }
            Err(err) if psi_absent(&err) => {}
            Err(err) => failed = Some(format!("{name}: {err}")),
        }
    }
    if let Some(err) = failed {
        return Probe::Failed(err);
    }
    if !any {
        Probe::Missing
    } else {
        Probe::Ready(out)
    }
}

/// Not there, or there but switched off (`psi=0` answers `EOPNOTSUPP`):
/// either way PSI is absent, not down.
fn psi_absent(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::Unsupported
    ) || err.raw_os_error() == Some(EOPNOTSUPP)
}

/// Counters a [`RateWindow`] rates.
pub trait Counters: Copy {
    /// `true` when any counter is below `prev`: a device went away or a
    /// counter was reset.
    fn behind(&self, prev: &Self) -> bool;
}

impl Counters for DiskCounters {
    fn behind(&self, prev: &Self) -> bool {
        self.read_sectors < prev.read_sectors || self.write_sectors < prev.write_sectors
    }
}

impl Counters for NetCounters {
    fn behind(&self, prev: &Self) -> bool {
        self.rx_bytes < prev.rx_bytes || self.tx_bytes < prev.tx_bytes
    }
}

/// Byte rates over a trailing window, as CPU percent is: the baseline is
/// the newest stored sample at least `window` old. One 0.1 s tick of
/// writeback would swing the digits between 0 and 900 MB/s; a second's
/// worth reads steady, and the exported gauge is the same number.
#[derive(Clone, Debug)]
pub struct RateWindow<T> {
    window: Duration,
    /// `(mono, counters)` from oldest to newest.
    history: VecDeque<(Instant, T)>,
}

impl<T: Counters> RateWindow<T> {
    #[must_use]
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            history: VecDeque::new(),
        }
    }

    /// Forget every sample. The next [`Self::push`] is a first sample.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Store `now` and return `(elapsed, baseline)`, or `None` while no
    /// stored sample is `window` old. A counter below the previous sample
    /// drops the history first, so a rate never spans a reset.
    pub fn push(&mut self, mono: Instant, now: T) -> Option<(Duration, T)> {
        self.history.retain(|(stamp, _)| *stamp <= mono);
        // A counter reset, or a stall (a sleep, a hung read) longer than
        // MAX_RATE_GAP since the last sample: start a new baseline rather
        // than spread or spike across it.
        if self.history.back().is_some_and(|(stamp, last)| {
            now.behind(last) || mono.saturating_duration_since(*stamp) > MAX_RATE_GAP
        }) {
            self.history.clear();
        }
        let window = self.window;
        let baseline = self
            .history
            .iter()
            .rev()
            .find(|(stamp, _)| mono.saturating_duration_since(*stamp) >= window)
            .copied();
        self.history.push_back((mono, now));
        if let Some(keep) = self
            .history
            .iter()
            .rposition(|(stamp, _)| mono.saturating_duration_since(*stamp) >= window)
        {
            self.history.drain(..keep);
        }
        baseline.map(|(at, prev)| (mono.saturating_duration_since(at), prev))
    }
}

/// Bytes/s between two counters. `None` when the gap is unusable or the
/// counter went backwards: the caller keeps the new sample as a baseline.
pub fn counter_rate(prev: u64, next: u64, dt: Duration) -> Option<f64> {
    if dt.is_zero() || dt > MAX_RATE_SPAN {
        return None;
    }
    let delta = next.checked_sub(prev)?;
    let secs = dt.as_secs_f64();
    if !(secs.is_finite() && secs > 0.0) {
        return None;
    }
    Some(delta as f64 / secs)
}

/// Bytes/s from a sector counter. One sector is 512 bytes.
pub fn sector_rate(prev: u64, next: u64, dt: Duration) -> Option<f64> {
    if dt.is_zero() || dt > MAX_RATE_SPAN {
        return None;
    }
    let delta = next.checked_sub(prev)?;
    let bytes = delta.saturating_mul(512);
    let secs = dt.as_secs_f64();
    if !(secs.is_finite() && secs > 0.0) {
        return None;
    }
    Some(bytes as f64 / secs)
}

fn read_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Ok(text)
}

/// `true` for a whole disk: `nvme0n1`, `sda`, `vda`, `hda`, `xvda`, `mmcblk0`.
/// Partitions, loop, zram and device-mapper are not summed.
fn whole_disk(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("nvme") {
        let Some((ctrl, ns)) = rest.split_once('n') else {
            return false;
        };
        return !ctrl.is_empty()
            && ctrl.bytes().all(|b| b.is_ascii_digit())
            && !ns.is_empty()
            && ns.bytes().all(|b| b.is_ascii_digit());
    }
    for prefix in ["xvd", "sd", "vd", "hd"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_lowercase());
        }
    }
    if let Some(rest) = name.strip_prefix("mmcblk") {
        return !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

/// Summed whole-disk counters, or `None` when no line is a whole disk.
fn parse_diskstats(text: &str) -> Result<Option<DiskCounters>, String> {
    let mut total = DiskCounters::default();
    let mut any = false;
    for (nth, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        // A short line the sum ignores (a partition, loop) does not fail
        // the whole file; a short whole-disk line does.
        if fields.len() >= 3 && !whole_disk(fields[2]) {
            continue;
        }
        if fields.len() < 14 {
            return Err(format!(
                "line {} has {} fields, need 14",
                nth + 1,
                fields.len()
            ));
        }
        let name = fields[2];
        any = true;
        let read = parse_counter(name, "sectors read", fields[5])?;
        let written = parse_counter(name, "sectors written", fields[9])?;
        total.read_sectors = total.read_sectors.saturating_add(read);
        total.write_sectors = total.write_sectors.saturating_add(written);
    }
    Ok(any.then_some(total))
}

fn parse_counter(name: &str, field: &str, text: &str) -> Result<u64, String> {
    text.parse::<u64>()
        .map_err(|_| format!("{name} {field} is not a counter"))
}

fn parse_netdev(text: &str) -> Result<Vec<(String, u64, u64)>, String> {
    let mut rows = Vec::new();
    for (nth, line) in text.lines().enumerate() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        // The kernel refuses only `/`, `:` and whitespace in a name.
        if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(format!("line {} has a bad interface name", nth + 1));
        }
        let fields: Vec<&str> = rest.split_whitespace().collect();
        if fields.len() < 9 {
            return Err(format!(
                "{name} has {} fields after the colon, need 9",
                fields.len()
            ));
        }
        let rx = parse_counter(name, "rx bytes", fields[0])?;
        let tx = parse_counter(name, "tx bytes", fields[8])?;
        rows.push((name.to_owned(), rx, tx));
    }
    Ok(rows)
}

/// A `device` symlink (or directory) marks a physical interface. The link
/// is not followed, so a fixture cannot escape its sys root.
fn has_device_link(iface: &Path) -> bool {
    match std::fs::symlink_metadata(iface.join("device")) {
        Ok(meta) => meta.file_type().is_symlink() || meta.is_dir(),
        Err(_) => false,
    }
}

/// Sum of the up links' ceilings. A link that is down (no cable, Wi-Fi
/// off) adds nothing; with no link up the ceiling is 1 Gbit.
fn links_ceiling_bps(roots: &Roots, names: &[String]) -> u64 {
    let mut ceiling = 0u64;
    let mut up = false;
    for name in names {
        let iface = roots.sys.join("class/net").join(name);
        if !link_up(&iface) {
            continue;
        }
        up = true;
        ceiling = ceiling.saturating_add(link_ceiling_bps(&iface));
    }
    if up { ceiling } else { ONE_GBIT_BPS as u64 }
}

/// `operstate` `up`, or `unknown` (a driver without carrier detection,
/// which still passes traffic). An unreadable `operstate` reads as unknown.
fn link_up(iface: &Path) -> bool {
    match read_file(&iface.join("operstate")) {
        Ok(text) => matches!(text.trim(), "up" | "unknown"),
        Err(_) => true,
    }
}

/// `speed` is Mbit/s. Non-positive (Wi-Fi's -1, a missing file) is 1 Gbit.
fn link_ceiling_bps(iface: &Path) -> u64 {
    let text = read_file(&iface.join("speed")).unwrap_or_default();
    let mbit = text.trim().parse::<i64>().unwrap_or(-1);
    if mbit > 0 {
        (mbit as u64).saturating_mul(1_000_000) / 8
    } else {
        ONE_GBIT_BPS as u64
    }
}

fn parse_swap(text: &str) -> Result<Option<SwapBytes>, String> {
    let mut total_kb = None;
    let mut free_kb = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key != "SwapTotal" && key != "SwapFree" {
            continue;
        }
        let Some(token) = rest.split_whitespace().next() else {
            return Err(format!("bad {key} value"));
        };
        let kb = token
            .parse::<u64>()
            .map_err(|_| format!("bad {key} value"))?;
        if key == "SwapTotal" {
            total_kb = Some(kb);
        } else {
            free_kb = Some(kb);
        }
    }
    match (total_kb, free_kb) {
        (Some(total), Some(free)) => {
            let total_b = total.saturating_mul(1024);
            let free_b = free.saturating_mul(1024).min(total_b);
            Ok(Some(SwapBytes {
                used: total_b - free_b,
                total: total_b,
            }))
        }
        (None, None) => Ok(None),
        _ => Err("meminfo has only one of SwapTotal and SwapFree".to_owned()),
    }
}

fn parse_some_avg10(text: &str) -> Result<f32, String> {
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("some") {
            continue;
        }
        for field in parts {
            let Some(value) = field.strip_prefix("avg10=") else {
                continue;
            };
            let pct = value
                .parse::<f32>()
                .map_err(|_| format!("bad avg10 {value:?}"))?;
            if !pct.is_finite() || pct < 0.0 {
                return Err(format!("bad avg10 {value}"));
            }
            return Ok(pct);
        }
    }
    Err("no some avg10".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_disk_names_keep_nvme_and_sd_and_drop_parts() {
        assert!(whole_disk("nvme0n1"));
        assert!(whole_disk("nvme1n1"));
        assert!(whole_disk("sda"));
        assert!(whole_disk("vda"));
        assert!(whole_disk("mmcblk0"));
        assert!(!whole_disk("nvme0n1p1"));
        assert!(!whole_disk("sda1"));
        assert!(!whole_disk("loop0"));
        assert!(!whole_disk("zram0"));
        assert!(!whole_disk("dm-0"));
    }

    #[test]
    fn psi_switched_off_is_absent_and_other_errors_are_not() {
        assert!(psi_absent(&io::Error::from(io::ErrorKind::NotFound)));
        assert!(psi_absent(&io::Error::from(io::ErrorKind::Unsupported)));
        assert!(psi_absent(&io::Error::from_raw_os_error(EOPNOTSUPP)));
        assert!(!psi_absent(&io::Error::from(
            io::ErrorKind::PermissionDenied
        )));
        assert!(!psi_absent(&io::Error::from_raw_os_error(5)));
    }
}
