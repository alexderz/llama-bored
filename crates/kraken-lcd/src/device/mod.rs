//! LCD sink: open, upload, and restore, inside the cooling guard.
//!
//! [`KrakenLcd`] is generic over the HID and bulk ports so tests inject fakes.
//! [`HidLink`] and [`BulkLink`] are the real ports. Tests do not construct them.

mod bulk;
mod guard;
mod hid;
pub mod proto;
mod sysfs;

use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

/// `true` when exactly one hwmon under `sys_root` is named `z53`.
///
/// The root is canonicalized first so the guard's directory check sees the
/// same path the sensor directories canonicalize to.
#[must_use]
pub fn z53_exists(sys_root: &Path) -> bool {
    let Ok(root) = sys_root.canonicalize() else {
        return false;
    };
    guard::find_z53(&root).is_some()
}

use thiserror::Error;

pub use bulk::{BulkLink, BulkPort, NoBulk};
pub(crate) use guard::follow_up_due;
pub use guard::{
    CoolingSnapshot, Deviation, PUMP_BAND_FLOOR_RPM, PUMP_BAND_PERCENT, pump_band_rpm,
};
pub use hid::{HidLink, HidPort};
pub(crate) use sysfs::read_coolant_c;
pub use sysfs::{KrakenNode, binding, interface0_usbfs, post_open, pre_open};

use crate::log::{self, Priority};
use crate::render::Frame;
use guard::{CoolingGuard, format_halt, log_halt};
use hid::ExchangeError;
use proto::{BucketId, Cmd, SlotId};

pub(crate) fn log_at(priority: Priority, message: &str) {
    log::emit(&mut log::Stderr, priority, message);
}

/// Where an upload step failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum UploadFailed {
    #[error("no reply at {0}")]
    NoReply(Step),
    #[error("device refused {0}")]
    Refused(Step),
    #[error("protocol error at {0}")]
    Protocol(Step),
}

/// A named step in an open or an upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Step {
    PreTransfer,
    DeleteBucket,
    SetupBucket,
    WriteStart,
    Bulk,
    WriteEnd,
    ShowSlot,
    ShowLiquid,
    QueryBucket,
    LcdInfo,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PreTransfer => "pre-transfer",
            Self::DeleteBucket => "delete-bucket",
            Self::SetupBucket => "setup-bucket",
            Self::WriteStart => "write-start",
            Self::Bulk => "bulk",
            Self::WriteEnd => "write-end",
            Self::ShowSlot => "show-slot",
            Self::ShowLiquid => "show-liquid",
            Self::QueryBucket => "query-bucket",
            Self::LcdInfo => "lcd-info",
        })
    }
}

/// Failure from [`LcdSink::show`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum SinkError {
    #[error("upload failed: {0}")]
    UploadFailed(UploadFailed),
    #[error("device unavailable")]
    DeviceUnavailable,
    #[error("transfer aborted")]
    TransferAborted,
    #[error("command fence")]
    Fence,
    #[error("device is in the bootloader")]
    DeviceInBootloader,
    #[error("cooling guard halted")]
    Halted,
    /// Post-open sysfs check failed. The service exits non-zero and does not retry.
    #[error("fatal: {0}")]
    Fatal(&'static str),
}

/// Injected clock for [`KrakenLcd::bench_upload`]. Production sleeps; tests advance ns.
pub trait BenchClock {
    /// Monotonic nanoseconds from an arbitrary origin.
    fn now_ns(&self) -> u64;
    /// Wait `d`, or advance a fake clock by `d`.
    fn sleep(&self, d: Duration);
}

struct InstantNs {
    origin: Instant,
}

impl BenchClock for InstantNs {
    fn now_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// Paced ping-pong parameters. `try_new` is the only constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchSpec {
    /// Frames to upload, `1..=36000`.
    pub count: u32,
    /// Target frames per second, `0 < fps ≤ 30`.
    pub fps: f64,
    /// Rotation slots to ping-pong, fixed at 2.
    pub slots: u8,
}

/// Why [`BenchSpec::try_new`] refused the arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum BenchBounds {
    #[error("count must be 1..=36000")]
    Count,
    #[error("fps must be > 0 and ≤ 30")]
    Fps,
    #[error("slots must be 2")]
    Slots,
}

impl BenchSpec {
    /// `count` in `1..=36000`, `0 < fps ≤ 30`, `slots == 2`.
    pub fn try_new(count: u32, fps: f64, slots: u8) -> Result<Self, BenchBounds> {
        if !(1..=36_000).contains(&count) {
            return Err(BenchBounds::Count);
        }
        if !(fps > 0.0 && fps <= 30.0) {
            return Err(BenchBounds::Fps);
        }
        if slots != 2 {
            return Err(BenchBounds::Slots);
        }
        Ok(Self { count, fps, slots })
    }

    fn period_ns(self) -> u64 {
        (1_000_000_000.0 / self.fps).round() as u64
    }
}

/// One paced upload: milliseconds, with signed slack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchSample {
    pub seq: u32,
    pub slot: u8,
    pub upload_ms: f64,
    pub switch_ms: f64,
    pub total_ms: f64,
    pub slack_ms: f64,
}

/// Per-step samples plus the fps measured from first start to last start.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchSummary {
    pub samples: Vec<BenchSample>,
    pub achieved_fps: f64,
}

/// Slack in milliseconds: `period - total`. Negative when the frame overran.
#[must_use]
pub fn bench_slack_ms(period_ns: u64, total_ns: u64) -> f64 {
    (period_ns as i128 - total_ns as i128) as f64 / 1_000_000.0
}

/// Host sysfs root used by [`KrakenLcd::connect`]. Tests pass a different root.
pub const SYS_ROOT: &str = "/sys";

/// Host state directory used by [`KrakenLcd::connect`]. Tests pass a different directory.
pub const STATE_DIR: &str = "/var/lib/kraken-lcd";

/// Linux `ENODEV`: the device node is gone.
const ENODEV: i32 = 19;
/// Linux `EPIPE`: the endpoint closed.
const EPIPE: i32 = 32;
/// Linux `ETIMEDOUT`: the transfer timed out.
const ETIMEDOUT: i32 = 110;

/// Failure from a hidraw or bulk port.
#[derive(Debug)]
pub struct PortError {
    unavailable: bool,
    detail: String,
}

impl PortError {
    /// The device is gone, stalled, or the wait elapsed.
    #[must_use]
    pub fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            unavailable: true,
            detail: detail.into(),
        }
    }

    /// The port failed for a reason other than the device disappearing.
    #[must_use]
    pub fn failed(detail: impl Into<String>) -> Self {
        Self {
            unavailable: false,
            detail: detail.into(),
        }
    }

    pub(crate) fn from_io(err: std::io::Error) -> Self {
        let unavailable = matches!(
            err.kind(),
            std::io::ErrorKind::NotFound
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::UnexpectedEof
        ) || matches!(err.raw_os_error(), Some(ENODEV | EPIPE | ETIMEDOUT));
        if unavailable {
            Self::unavailable(err.to_string())
        } else {
            Self::failed(err.to_string())
        }
    }

    /// `true` for a disconnected device or a timed-out transfer.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        self.unavailable
    }
}

impl std::fmt::Display for PortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for PortError {}

/// One `QueryBucket` reply, in bucket order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BucketRow {
    /// Bucket index, `0..=15`.
    pub id: u8,
    /// Parsed start and size. `empty` is the device's own marker.
    pub table: proto::BucketTable,
}

/// One line per bucket: `bucket=N empty=yes|no start_kib=N size_kib=N`.
#[must_use]
pub fn format_bucket_table(rows: &[BucketRow]) -> String {
    let mut out = String::new();
    for row in rows {
        let empty = if row.table.empty { "yes" } else { "no" };
        out.push_str(&format!(
            "bucket={} empty={} start_kib={} size_kib={}\n",
            row.id, empty, row.table.start_kib, row.table.size_kib
        ));
    }
    out
}

/// Roots and the rotation applied on top of the panel orientation.
#[derive(Clone, Copy, Debug)]
pub struct OpenRequest<'a> {
    pub sys_root: &'a Path,
    pub state_dir: &'a Path,
    pub rotate_deg: u16,
    pub trace_hid: bool,
}

/// What [`FakeLcd`] recorded. `Cmd` values are the sink-level operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Record {
    Cmd(Cmd),
    Bulk(Vec<u8>),
}

/// In-memory sink for the service tests. Script the next `show` result with [`FakeLcd::script`].
#[derive(Debug)]
pub struct FakeLcd {
    records: Vec<Record>,
    script: VecDeque<Result<(), SinkError>>,
    tick_script: VecDeque<Result<(), SinkError>>,
    reupload: bool,
    halted: bool,
    next_slot: u8,
}

impl FakeLcd {
    /// Empty log, not halted, no scripted reply.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            script: VecDeque::new(),
            tick_script: VecDeque::new(),
            reupload: false,
            halted: false,
            next_slot: 0,
        }
    }

    /// The next [`LcdSink::show`] returns `result` instead of success.
    pub fn script(&mut self, result: Result<(), SinkError>) {
        self.script.push_back(result);
    }

    /// The next [`LcdSink::tick`] returns `result`.
    pub fn script_tick(&mut self, result: Result<(), SinkError>) {
        self.tick_script.push_back(result);
    }

    /// Commands and bulk payloads recorded since construction.
    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// `true` after [`LcdSink::needs_reupload`].
    #[must_use]
    pub fn reupload_pending(&self) -> bool {
        self.reupload
    }

    /// `true` after a scripted [`SinkError::Halted`] or [`FakeLcd::halt`].
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Stop recording. Later `show` and `restore_stock` send nothing.
    pub fn halt(&mut self) {
        self.halted = true;
    }
}

impl Default for FakeLcd {
    fn default() -> Self {
        Self::new()
    }
}

/// The seam the service uses. `needs_reupload` marks the next upload, it does not query.
pub trait LcdSink {
    /// Upload one frame.
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError>;

    /// Ask the panel to draw the stock coolant screen. Best effort, and it does not panic.
    fn restore_stock(&mut self);

    /// The next policy tick should upload.
    fn needs_reupload(&mut self);

    /// One cooling-guard check for the operation that just finished.
    fn tick(&mut self) -> Result<(), SinkError>;

    /// Read-only latch check. The service calls this every tick in OURS,
    /// separate from the delayed follow-up in [`Self::tick`].
    ///
    /// There is no default. A sink that forgot the latch would otherwise
    /// report "not blocked" and keep writing.
    fn blocked(&mut self) -> bool;

    /// Upload `frame` into one rotation slot.
    ///
    /// Stream mode passes `0` then `1`. The default ignores the slot and calls
    /// [`Self::show`], which keeps the eight-slot ring. [`FakeLcd`] and
    /// [`KrakenLcd`] override it.
    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        let _ = slot;
        self.show(frame)
    }

    /// Stream-mode cooling check, at least once a second.
    ///
    /// The default is [`Self::tick`]. [`KrakenLcd`] also stores a new baseline
    /// so the next check can still see a drift after this one consumes it.
    fn pace_guard(&mut self) -> Result<(), SinkError> {
        self.tick()
    }
}

impl FakeLcd {
    fn record_show(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        if self.halted {
            return Err(SinkError::Halted);
        }
        let result = self.script.pop_front().unwrap_or(Ok(()));
        if matches!(result, Err(SinkError::Halted)) {
            self.halted = true;
            return Err(SinkError::Halted);
        }
        if matches!(result, Err(SinkError::TransferAborted)) {
            self.records.push(Record::Cmd(Cmd::ShowLiquid));
            return Err(SinkError::TransferAborted);
        }
        if result.is_ok() {
            if let Ok(slot) = SlotId::try_new(slot) {
                self.records.push(Record::Cmd(Cmd::ShowSlot(slot)));
            }
            if let Ok(packed) = proto::pack(frame, 0, 0) {
                self.records.push(Record::Bulk(packed));
            }
        }
        result
    }
}

impl LcdSink for FakeLcd {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        let slot = self.next_slot;
        let result = self.record_show(slot, frame);
        if result.is_ok() {
            self.next_slot = self.next_slot.wrapping_add(1) % proto::SLOT_COUNT;
        }
        result
    }

    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        self.record_show(slot, frame)
    }

    fn tick(&mut self) -> Result<(), SinkError> {
        if self.halted {
            return Err(SinkError::Halted);
        }
        let result = self.tick_script.pop_front().unwrap_or(Ok(()));
        if matches!(result, Err(SinkError::Halted)) {
            self.halted = true;
        }
        result
    }

    fn restore_stock(&mut self) {
        if self.halted {
            return;
        }
        self.records.push(Record::Cmd(Cmd::ShowLiquid));
    }

    fn needs_reupload(&mut self) {
        self.reupload = true;
    }

    fn blocked(&mut self) -> bool {
        self.halted
    }
}

/// Open Kraken LCD. `H` and `B` are the ports; production uses [`HidLink`] and [`BulkLink`].
pub struct KrakenLcd<H, B> {
    hid: Option<H>,
    bulk: Option<B>,
    guard: CoolingGuard,
    active: Option<SlotId>,
    orientation: Option<u8>,
    rotate_deg: u16,
    needs_flag: bool,
    halted: bool,
    trace: bool,
    baseline: Option<CoolingSnapshot>,
}

impl<H, B> std::fmt::Debug for KrakenLcd<H, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KrakenLcd")
            .field("halted", &self.halted)
            .field("orientation", &self.orientation)
            .field("active", &self.active)
            .field("reupload", &self.needs_flag)
            .finish()
    }
}

impl KrakenLcd<HidLink, BulkLink> {
    /// Open the real hidraw node and bulk endpoint on `/sys`.
    ///
    /// The state directory is `/var/lib/kraken-lcd`. Tests must not call this;
    /// they use [`KrakenLcd::open`] with a fake root.
    pub fn connect(rotate_deg: u16, trace_hid: bool) -> Result<Self, SinkError> {
        let request = OpenRequest {
            sys_root: Path::new(SYS_ROOT),
            state_dir: Path::new(STATE_DIR),
            rotate_deg,
            trace_hid,
        };
        Self::open(&request, BulkLink::open, HidLink::open)
    }

    /// One-frame upload on [`SYS_ROOT`] and [`STATE_DIR`], then `QueryBucket(0..=15)`.
    ///
    /// Rotation stays `0`, matching the CLI. Tests use [`KrakenLcd::show_image`].
    /// Never sends `ShowLiquid`.
    pub fn show_image_resolved(
        slot: SlotId,
        frame: &Frame,
        trace_hid: bool,
    ) -> Result<Vec<BucketRow>, SinkError> {
        let request = OpenRequest {
            sys_root: Path::new(SYS_ROOT),
            state_dir: Path::new(STATE_DIR),
            rotate_deg: 0,
            trace_hid,
        };
        Self::show_image(
            &request,
            slot,
            frame,
            BulkLink::open,
            HidLink::open,
            std::thread::sleep,
        )
    }

    /// Paced ping-pong on [`SYS_ROOT`] and [`STATE_DIR`]. Rotation stays `0`.
    ///
    /// Tests use [`KrakenLcd::bench_upload`]. Never sends `ShowLiquid` on the
    /// happy path. Aborts on the first upload failure (stricter than the
    /// service `fail_limit`).
    pub fn bench_upload_resolved(
        frame: &Frame,
        spec: BenchSpec,
        trace_hid: bool,
        out: &mut impl Write,
    ) -> Result<BenchSummary, SinkError> {
        let request = OpenRequest {
            sys_root: Path::new(SYS_ROOT),
            state_dir: Path::new(STATE_DIR),
            rotate_deg: 0,
            trace_hid,
        };
        Self::bench_upload(
            &request,
            frame,
            spec,
            BulkLink::open,
            HidLink::open,
            &InstantNs {
                origin: Instant::now(),
            },
            out,
        )
    }
}

/// Opens the hidraw path `pre_open` already resolved. Not a free-path API:
/// [`KrakenLcd::open_restore`] and [`KrakenLcd::query_buckets`] pass
/// `node.hidraw`, and S11 only allows this name in `service.rs` and `device/`.
pub(crate) fn open_resolved_hid(path: &Path) -> Result<HidLink, PortError> {
    HidLink::open(path)
}

impl KrakenLcd<HidLink, NoBulk> {
    /// Hidraw-only `QueryBucket(0..=15)` on [`SYS_ROOT`] and [`STATE_DIR`].
    ///
    /// Rotation stays `0`, matching the CLI. Tests use [`KrakenLcd::query_buckets`].
    pub fn query_buckets_resolved(trace_hid: bool) -> Result<Vec<BucketRow>, SinkError> {
        let request = OpenRequest {
            sys_root: Path::new(SYS_ROOT),
            state_dir: Path::new(STATE_DIR),
            rotate_deg: 0,
            trace_hid,
        };
        Self::query_buckets(&request, open_resolved_hid)
    }
}

impl<H: HidPort> KrakenLcd<H, NoBulk> {
    /// Hidraw-only open, binding check, then one `ShowLiquid`.
    ///
    /// No bulk claim, no bucket queries, and no LCD-info query. A present latch
    /// skips the open. Errors are logged. This does not panic.
    pub fn open_restore(
        request: &OpenRequest<'_>,
        open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
    ) -> Self {
        let mut lcd = Self::closed(request);
        if lcd.guard.latch_present() {
            lcd.halted = true;
            log_at(
                Priority::Crit,
                "restore skipped; cooling guard latch is present",
            );
            return lcd;
        }
        let node = match pre_open(request.sys_root) {
            Ok(node) => node,
            Err(err) => {
                log_at(Priority::Err, &format!("restore pre-open: {err}"));
                return lcd;
            }
        };
        lcd.guard = CoolingGuard::new(request.sys_root, &node.device_dir, request.state_dir);
        match open_hid(&node.hidraw) {
            Ok(hid) => lcd.hid = Some(hid),
            Err(err) => {
                log_at(Priority::Err, &format!("restore hid open: {err}"));
                return lcd;
            }
        }
        if let Err(err) = binding(request.sys_root, &node) {
            log_at(Priority::Crit, &format!("restore binding: {err}"));
            lcd.hid.take();
            return lcd;
        }
        lcd.restore_stock();
        lcd
    }

    /// Pre-open checks, hidraw only, then `QueryBucket` for buckets `0..=15`.
    ///
    /// No bulk claim, no delete, and no `ShowLiquid`. Each query is inside
    /// the cooling guard. A deviation latches and sends nothing further.
    /// A latch that is already present skips the open.
    pub fn query_buckets(
        request: &OpenRequest<'_>,
        open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
    ) -> Result<Vec<BucketRow>, SinkError> {
        if reject_root(rustix::process::geteuid().is_root()) {
            log_at(Priority::Err, "query-buckets refused; euid is 0");
            return Err(SinkError::DeviceUnavailable);
        }
        match interface0_usbfs(request.sys_root) {
            Ok(true) => {
                log_at(
                    Priority::Err,
                    "query-buckets refused; interface 0 is bound to usbfs",
                );
                return Err(SinkError::DeviceUnavailable);
            }
            Ok(false) => {}
            Err(err) => {
                log_at(Priority::Err, &format!("query-buckets pre-open: {err}"));
                return Err(err);
            }
        }
        let mut lcd = Self::closed(request);
        if lcd.guard.latch_present() {
            lcd.halted = true;
            log_at(
                Priority::Crit,
                "query-buckets skipped; cooling guard latch is present",
            );
            return Err(SinkError::Halted);
        }
        let node = match pre_open(request.sys_root) {
            Ok(node) => node,
            Err(err) => {
                log_at(Priority::Err, &format!("query-buckets pre-open: {err}"));
                return Err(err);
            }
        };
        lcd.guard = CoolingGuard::new(request.sys_root, &node.device_dir, request.state_dir);
        let baseline = match lcd.guard.capture() {
            Ok(snap) => snap,
            Err(dev) => {
                lcd.enter_halt(None, dev);
                return Err(SinkError::Halted);
            }
        };
        match open_hid(&node.hidraw) {
            Ok(hid) => lcd.hid = Some(hid),
            Err(err) => {
                log_at(Priority::Err, &format!("query-buckets hid open: {err}"));
                return Err(kept_or_halt(
                    &mut lcd,
                    &baseline,
                    SinkError::DeviceUnavailable,
                ));
            }
        }
        if let Err(err) = binding(request.sys_root, &node) {
            log_at(Priority::Crit, &format!("query-buckets binding: {err}"));
            lcd.hid.take();
            return Err(kept_or_halt(&mut lcd, &baseline, err));
        }
        lcd.collect_bucket_table(&baseline)
    }
}

/// Root must not open the cooler. Every device entry point
/// (`query_buckets`, `show_image`, `bench_upload`) checks this first.
pub(crate) fn reject_root(is_root: bool) -> bool {
    is_root
}

/// Guard check after a query step. A deviation replaces `err` with [`SinkError::Halted`].
fn kept_or_halt<H: HidPort, B: BulkPort>(
    lcd: &mut KrakenLcd<H, B>,
    baseline: &CoolingSnapshot,
    err: SinkError,
) -> SinkError {
    lcd.finish_guard(baseline, Err(err)).err().unwrap_or(err)
}

impl<H: HidPort, B: BulkPort> KrakenLcd<H, B> {
    /// Pre-open checks, bulk, HID, post-open checks, slot hygiene, then LCD info.
    pub fn open(
        request: &OpenRequest<'_>,
        open_bulk: impl FnOnce(u8, u8) -> Result<B, PortError>,
        open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
    ) -> Result<Self, SinkError> {
        let gate = CoolingGuard::new(request.sys_root, Path::new(""), request.state_dir);
        if gate.latch_present() {
            return Err(SinkError::Halted);
        }
        let node = pre_open(request.sys_root)?;
        let mut lcd = Self {
            hid: None,
            bulk: None,
            guard: CoolingGuard::new(request.sys_root, &node.device_dir, request.state_dir),
            active: None,
            orientation: None,
            rotate_deg: request.rotate_deg,
            needs_flag: false,
            halted: false,
            trace: request.trace_hid,
            baseline: None,
        };
        // Baseline before the claim, so a cooling change caused by opening
        // the ports is visible at the post-open check.
        let baseline = match lcd.guard.capture() {
            Ok(snap) => snap,
            Err(dev) => {
                lcd.enter_halt(None, dev);
                return Err(SinkError::Halted);
            }
        };
        let bulk = match open_bulk(node.busnum, node.devnum) {
            Ok(bulk) => bulk,
            Err(_) => return Err(SinkError::DeviceUnavailable),
        };
        let hid = match open_hid(&node.hidraw) {
            Ok(hid) => hid,
            Err(_) => {
                drop(bulk);
                return Err(SinkError::DeviceUnavailable);
            }
        };
        if let Err(err) = post_open(request.sys_root, &node) {
            drop(hid);
            drop(bulk);
            return Err(err);
        }
        lcd.hid = Some(hid);
        lcd.bulk = Some(bulk);
        lcd.baseline = Some(baseline.clone());
        lcd.finish_guard(&baseline, Ok(()))?;
        let opened = lcd.hygiene().and_then(|()| lcd.read_info());
        match lcd.finish_guard(&baseline, opened) {
            Ok(()) => {
                lcd.needs_flag = true;
                Ok(lcd)
            }
            Err(err) => {
                lcd.hid.take();
                lcd.bulk.take();
                Err(err)
            }
        }
    }

    /// One-frame upload, then hidraw `QueryBucket(0..=15)`. Never `ShowLiquid`.
    ///
    /// Reuses [`Self::open`] (latch, `pre_open`, binding, cooling guard, slot
    /// hygiene) and the same upload sequence as [`LcdSink::show`].
    ///
    /// After the bucket table, waits [`guard::FOLLOW_UP_AFTER`] then runs one
    /// more cooling check. Tests inject `sleep` so that wait is not a real 2 s.
    pub fn show_image(
        request: &OpenRequest<'_>,
        slot: SlotId,
        frame: &Frame,
        open_bulk: impl FnOnce(u8, u8) -> Result<B, PortError>,
        open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
        mut sleep: impl FnMut(Duration),
    ) -> Result<Vec<BucketRow>, SinkError> {
        if reject_root(rustix::process::geteuid().is_root()) {
            log_at(Priority::Err, "show-image refused; euid is 0");
            return Err(SinkError::DeviceUnavailable);
        }
        if !z53_exists(request.sys_root) {
            log_at(Priority::Err, "show-image: no z53 hwmon");
            return Err(SinkError::DeviceUnavailable);
        }
        if !guard::state_dir_writable(request.state_dir) {
            log_at(Priority::Err, "show-image: state directory is not writable");
            return Err(SinkError::DeviceUnavailable);
        }
        match interface0_usbfs(request.sys_root) {
            Ok(true) => {
                log_at(
                    Priority::Err,
                    "show-image refused; interface 0 is bound to usbfs",
                );
                return Err(SinkError::DeviceUnavailable);
            }
            Ok(false) => {}
            Err(err) => {
                log_at(Priority::Err, &format!("show-image pre-open: {err}"));
                return Err(err);
            }
        }
        let mut lcd = Self::open(request, open_bulk, open_hid)?;
        lcd.show_at(Some(slot), frame)?;
        let baseline = match lcd.guard.capture() {
            Ok(snap) => snap,
            Err(dev) => {
                lcd.enter_halt(None, dev);
                return Err(SinkError::Halted);
            }
        };
        let rows = lcd.collect_bucket_table(&baseline)?;
        sleep(guard::FOLLOW_UP_AFTER);
        lcd.tick()?;
        Ok(rows)
    }

    /// Open on the `show-image` path, then ping-pong `spec.count` uploads.
    ///
    /// Reuses [`Self::open`] and [`Self::show_at`]. Ticks the cooling guard at
    /// least once a second and runs the 2 s follow-up after the last upload.
    pub fn bench_upload(
        request: &OpenRequest<'_>,
        frame: &Frame,
        spec: BenchSpec,
        open_bulk: impl FnOnce(u8, u8) -> Result<B, PortError>,
        open_hid: impl FnOnce(&Path) -> Result<H, PortError>,
        clock: &impl BenchClock,
        out: &mut impl Write,
    ) -> Result<BenchSummary, SinkError> {
        if reject_root(rustix::process::geteuid().is_root()) {
            log_at(Priority::Err, "bench-upload refused; euid is 0");
            return Err(SinkError::DeviceUnavailable);
        }
        if !z53_exists(request.sys_root) {
            log_at(Priority::Err, "bench-upload: no z53 hwmon");
            return Err(SinkError::DeviceUnavailable);
        }
        if !guard::state_dir_writable(request.state_dir) {
            log_at(
                Priority::Err,
                "bench-upload: state directory is not writable",
            );
            return Err(SinkError::DeviceUnavailable);
        }
        match interface0_usbfs(request.sys_root) {
            Ok(true) => {
                log_at(
                    Priority::Err,
                    "bench-upload refused; interface 0 is bound to usbfs",
                );
                return Err(SinkError::DeviceUnavailable);
            }
            Ok(false) => {}
            Err(err) => {
                log_at(Priority::Err, &format!("bench-upload pre-open: {err}"));
                return Err(err);
            }
        }
        let mut lcd = Self::open(request, open_bulk, open_hid)?;
        let period_ns = spec.period_ns();
        let mut samples = Vec::with_capacity(spec.count as usize);
        let mut last_tick = clock.now_ns();
        let mut first_start = None;
        let mut last_start = None;
        for seq in 0..spec.count {
            let slot_u8 = u8::try_from(seq % u32::from(spec.slots)).unwrap_or(0);
            let slot = match SlotId::try_new(slot_u8) {
                Ok(slot) => slot,
                Err(_) => return Err(SinkError::Fence),
            };
            let t0 = clock.now_ns();
            first_start.get_or_insert(t0);
            last_start = Some(t0);
            let timing = lcd.show_timed(Some(slot), frame, clock)?;
            let upload_ms = ns_to_ms(timing.upload_ns);
            let switch_ms = ns_to_ms(timing.switch_ns);
            let total_ns = timing.upload_ns.saturating_add(timing.switch_ns);
            let total_ms = ns_to_ms(total_ns);
            let slack_ms = bench_slack_ms(period_ns, total_ns);
            let _ = writeln!(
                out,
                "seq={seq} upload_ms={upload_ms:.3} switch_ms={switch_ms:.3} total_ms={total_ms:.3} slack_ms={slack_ms:.3}"
            );
            samples.push(BenchSample {
                seq,
                slot: slot_u8,
                upload_ms,
                switch_ms,
                total_ms,
                slack_ms,
            });
            tick_if_due(&mut lcd, clock, &mut last_tick)?;
            if seq + 1 < spec.count {
                let slack_ns = period_ns.saturating_sub(total_ns);
                wait_with_ticks(&mut lcd, clock, slack_ns, &mut last_tick)?;
            }
        }
        wait_with_ticks(
            &mut lcd,
            clock,
            u64::try_from(guard::FOLLOW_UP_AFTER.as_nanos()).unwrap_or(u64::MAX),
            &mut last_tick,
        )?;
        write_bench_summary(out, &samples);
        let achieved_fps = achieved_fps(spec.count, first_start, last_start, &samples);
        let _ = writeln!(out, "achieved_fps={achieved_fps:.3}");
        Ok(BenchSummary {
            samples,
            achieved_fps,
        })
    }

    fn collect_bucket_table(
        &mut self,
        baseline: &CoolingSnapshot,
    ) -> Result<Vec<BucketRow>, SinkError> {
        let mut rows = Vec::with_capacity(usize::from(proto::BUCKET_COUNT));
        for id in 0..proto::BUCKET_COUNT {
            let bucket = match BucketId::try_new(id) {
                Ok(bucket) => bucket,
                Err(_) => return Err(kept_or_halt(self, baseline, SinkError::Fence)),
            };
            let reply = match self.exchange(Cmd::QueryBucket(bucket), Step::QueryBucket) {
                Ok(reply) => reply,
                Err(err) => return Err(kept_or_halt(self, baseline, err)),
            };
            let table = match proto::parse_bucket_table(&reply) {
                Ok(table) => table,
                Err(_) => {
                    return Err(kept_or_halt(
                        self,
                        baseline,
                        SinkError::UploadFailed(UploadFailed::Protocol(Step::QueryBucket)),
                    ));
                }
            };
            rows.push(BucketRow { id, table });
            self.finish_guard(baseline, Ok(()))?;
        }
        Ok(rows)
    }

    /// One check of the baseline taken before the last device operation.
    ///
    /// The baseline is consumed. A later tick, with no new operation, does
    /// not halt: re-enumeration while idle is not a cooling failure.
    pub fn tick(&mut self) -> Result<(), SinkError> {
        self.poll_cooling()
    }

    fn poll_cooling(&mut self) -> Result<(), SinkError> {
        if self.cooling_blocked() {
            self.halted = true;
            return Err(SinkError::Halted);
        }
        let Some(baseline) = self.baseline.take() else {
            return Ok(());
        };
        match self.guard.check(&baseline) {
            Ok(()) => Ok(()),
            Err(dev) => {
                self.enter_halt(Some(&baseline), dev);
                Err(SinkError::Halted)
            }
        }
    }

    /// `true` after a successful open, or after [`LcdSink::needs_reupload`].
    #[must_use]
    pub fn reupload_pending(&self) -> bool {
        self.needs_flag
    }

    /// Clear the reupload mark.
    pub fn clear_reupload(&mut self) {
        self.needs_flag = false;
    }

    /// `true` once the guard has latched or the latch file is present.
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Orientation byte from LCD info, once open has read it.
    #[must_use]
    pub fn orientation(&self) -> Option<u8> {
        self.orientation
    }

    /// Slot shown by the last successful upload.
    #[must_use]
    pub fn active_slot(&self) -> Option<SlotId> {
        self.active
    }

    fn closed(request: &OpenRequest<'_>) -> Self {
        Self {
            hid: None,
            bulk: None,
            guard: CoolingGuard::new(request.sys_root, Path::new(""), request.state_dir),
            active: None,
            orientation: None,
            rotate_deg: request.rotate_deg,
            needs_flag: false,
            halted: false,
            trace: request.trace_hid,
            baseline: None,
        }
    }

    fn cooling_blocked(&self) -> bool {
        self.halted || self.guard.latch_present()
    }

    fn hygiene(&mut self) -> Result<(), SinkError> {
        for id in 0..proto::BUCKET_COUNT {
            let bucket = BucketId::try_new(id).map_err(|_| SinkError::Fence)?;
            let reply = self.exchange(Cmd::QueryBucket(bucket), Step::QueryBucket)?;
            let table = proto::parse_bucket_table(&reply)
                .map_err(|_| SinkError::UploadFailed(UploadFailed::Protocol(Step::QueryBucket)))?;
            if table.empty {
                continue;
            }
            let remove = if id < proto::SLOT_COUNT {
                let start = u16::from(id) * proto::SLOT_UNITS;
                table.start_kib != start || table.size_kib != proto::SLOT_UNITS
            } else {
                overlaps_slot_region(table.start_kib, table.size_kib)
            };
            if remove {
                log_at(
                    Priority::Info,
                    &format!(
                        "delete foreign bucket {id} start {} size {}",
                        table.start_kib, table.size_kib
                    ),
                );
                self.exchange(Cmd::DeleteBucket(bucket), Step::DeleteBucket)?;
            }
        }
        Ok(())
    }

    fn read_info(&mut self) -> Result<(), SinkError> {
        let reply = self.exchange(Cmd::LcdInfo, Step::LcdInfo)?;
        let info = proto::parse_lcd_info(&reply)
            .map_err(|_| SinkError::UploadFailed(UploadFailed::Protocol(Step::LcdInfo)))?;
        log_at(
            Priority::Info,
            &format!("lcd brightness {}", info.brightness),
        );
        self.orientation = Some(info.orientation);
        Ok(())
    }

    fn prepare(&mut self, slot: SlotId) -> Result<(), SinkError> {
        self.exchange(Cmd::PreTransfer, Step::PreTransfer)?;
        let bucket = BucketId::try_new(slot.get()).map_err(|_| SinkError::Fence)?;
        match self.exchange(Cmd::DeleteBucket(bucket), Step::DeleteBucket) {
            Ok(_) => {}
            Err(SinkError::UploadFailed(UploadFailed::NoReply(_) | UploadFailed::Refused(_))) => {
                self.exchange(Cmd::DeleteBucket(bucket), Step::DeleteBucket)?;
            }
            Err(err) => return Err(err),
        }
        self.exchange(Cmd::SetupBucket { slot }, Step::SetupBucket)?;
        Ok(())
    }

    fn exchange(&mut self, cmd: Cmd, step: Step) -> Result<[u8; proto::REPORT_LEN], SinkError> {
        let hid = self.hid.as_mut().ok_or(SinkError::DeviceUnavailable)?;
        hid::transact(hid, &cmd, self.trace).map_err(|err| map_exchange(step, err))
    }

    fn finish_guard(
        &mut self,
        baseline: &CoolingSnapshot,
        result: Result<(), SinkError>,
    ) -> Result<(), SinkError> {
        match self.guard.check(baseline) {
            Ok(()) => result,
            Err(dev) => {
                self.enter_halt(Some(baseline), dev);
                Err(SinkError::Halted)
            }
        }
    }

    fn enter_halt(&mut self, baseline: Option<&CoolingSnapshot>, dev: Deviation) {
        self.halted = true;
        self.hid.take();
        self.bulk.take();
        let current = self.guard.read();
        let body = format_halt(baseline, &current, dev);
        log_halt(&body);
        if let Err(err) = self.guard.write_latch(&body) {
            log_at(Priority::Crit, &format!("halt latch: {err}"));
        }
    }

    fn show_at(&mut self, forced: Option<SlotId>, frame: &Frame) -> Result<(), SinkError> {
        self.show_timed(
            forced,
            frame,
            &InstantNs {
                origin: Instant::now(),
            },
        )
        .map(|_| ())
    }

    fn tick_and_hold(&mut self) -> Result<(), SinkError> {
        self.tick()?;
        match self.guard.capture() {
            Ok(snap) => {
                self.baseline = Some(snap);
                Ok(())
            }
            Err(dev) => {
                self.enter_halt(None, dev);
                Err(SinkError::Halted)
            }
        }
    }

    fn show_timed(
        &mut self,
        forced: Option<SlotId>,
        frame: &Frame,
        clock: &impl BenchClock,
    ) -> Result<UploadTiming, SinkError> {
        if self.cooling_blocked() {
            self.halted = true;
            return Err(SinkError::Halted);
        }
        let baseline = match self.guard.capture() {
            Ok(snap) => snap,
            Err(dev) => {
                self.enter_halt(None, dev);
                return Err(SinkError::Halted);
            }
        };
        self.baseline = Some(baseline.clone());
        if self.hid.is_none() || self.bulk.is_none() {
            return self
                .finish_guard(&baseline, Err(SinkError::DeviceUnavailable))
                .map(|()| UploadTiming::ZERO);
        }
        let orientation = match self.orientation {
            Some(value) => value,
            None => {
                return self
                    .finish_guard(
                        &baseline,
                        Err(SinkError::UploadFailed(UploadFailed::Protocol(
                            Step::LcdInfo,
                        ))),
                    )
                    .map(|()| UploadTiming::ZERO);
            }
        };
        let pixels = match proto::pack(frame, orientation, self.rotate_deg) {
            Ok(pixels) => pixels,
            Err(_) => {
                return self
                    .finish_guard(
                        &baseline,
                        Err(SinkError::UploadFailed(UploadFailed::Protocol(Step::Bulk))),
                    )
                    .map(|()| UploadTiming::ZERO);
            }
        };
        let slot = match forced {
            Some(slot) => slot,
            None => match self.active {
                Some(active) => active.next(),
                None => match SlotId::try_new(0) {
                    Ok(slot) => slot,
                    Err(_) => {
                        return self
                            .finish_guard(&baseline, Err(SinkError::Fence))
                            .map(|()| UploadTiming::ZERO);
                    }
                },
            },
        };
        let t0 = clock.now_ns();
        if let Err(err) = self.prepare(slot) {
            return self
                .finish_guard(&baseline, Err(err))
                .map(|()| UploadTiming::ZERO);
        }
        let Some(hid) = self.hid.take() else {
            return self
                .finish_guard(&baseline, Err(SinkError::DeviceUnavailable))
                .map(|()| UploadTiming::ZERO);
        };
        let Some(bulk) = self.bulk.take() else {
            self.hid = Some(hid);
            return self
                .finish_guard(&baseline, Err(SinkError::DeviceUnavailable))
                .map(|()| UploadTiming::ZERO);
        };
        let mut transfer = Transfer {
            hid: Some(hid),
            bulk: Some(bulk),
            guard: self.guard.clone(),
            baseline: baseline.clone(),
            trace: self.trace,
            armed: true,
            halted: false,
        };
        match transfer.write_then_switch(slot, &pixels, clock, t0) {
            Ok(timing) => {
                transfer.armed = false;
                self.hid = transfer.hid.take();
                self.bulk = transfer.bulk.take();
                self.active = Some(slot);
                self.finish_guard(&baseline, Ok(()))?;
                Ok(timing)
            }
            Err(failure) => {
                log_at(
                    Priority::Err,
                    &format!("transfer failed at {}: {}", failure.step, failure.detail),
                );
                let halted = transfer.recover();
                self.hid = None;
                self.bulk = None;
                if halted {
                    self.halted = true;
                    Err(SinkError::Halted)
                } else {
                    Err(SinkError::TransferAborted)
                }
            }
        }
    }
}

impl<H: HidPort, B: BulkPort> LcdSink for KrakenLcd<H, B> {
    fn show(&mut self, frame: &Frame) -> Result<(), SinkError> {
        self.show_at(None, frame)
    }

    fn show_slot(&mut self, slot: u8, frame: &Frame) -> Result<(), SinkError> {
        match SlotId::try_new(slot) {
            Ok(slot) => self.show_at(Some(slot), frame),
            Err(_) => Err(SinkError::Fence),
        }
    }

    fn pace_guard(&mut self) -> Result<(), SinkError> {
        self.tick_and_hold()
    }

    fn restore_stock(&mut self) {
        if self.cooling_blocked() {
            self.halted = true;
            return;
        }
        let baseline = match self.guard.capture() {
            Ok(snap) => snap,
            Err(dev) => {
                self.enter_halt(None, dev);
                return;
            }
        };
        self.baseline = Some(baseline.clone());
        if self.hid.is_none() {
            log_at(Priority::Info, "restore skipped; hid is closed");
            return;
        }
        let sent = self.exchange(Cmd::ShowLiquid, Step::ShowLiquid).map(|_| ());
        let _ = self.finish_guard(&baseline, sent);
    }

    fn needs_reupload(&mut self) {
        self.needs_flag = true;
    }

    fn tick(&mut self) -> Result<(), SinkError> {
        self.poll_cooling()
    }

    fn blocked(&mut self) -> bool {
        self.cooling_blocked()
    }
}

/// Owns the ports after `WriteStart`. On drop while still armed, the guard
/// runs before any `ShowLiquid`: a cooling change sends nothing and latches.
struct Transfer<H: HidPort, B: BulkPort> {
    hid: Option<H>,
    bulk: Option<B>,
    guard: CoolingGuard,
    baseline: CoolingSnapshot,
    trace: bool,
    armed: bool,
    halted: bool,
}

impl<H: HidPort, B: BulkPort> Transfer<H, B> {
    fn write_then_switch(
        &mut self,
        slot: SlotId,
        pixels: &[u8],
        clock: &impl BenchClock,
        t0: u64,
    ) -> Result<UploadTiming, TransferFail> {
        self.hid_step(Cmd::WriteStart(slot), Step::WriteStart)?;
        self.bulk_step(proto::BULK_HEADER.as_slice(), Step::Bulk)?;
        for chunk in pixels.chunks(512) {
            self.bulk_step(chunk, Step::Bulk)?;
        }
        self.hid_step(Cmd::WriteEnd, Step::WriteEnd)?;
        let t1 = clock.now_ns();
        self.hid_step(Cmd::ShowSlot(slot), Step::ShowSlot)?;
        let t2 = clock.now_ns();
        Ok(UploadTiming {
            upload_ns: t1.saturating_sub(t0),
            switch_ns: t2.saturating_sub(t1),
        })
    }

    fn hid_step(&mut self, cmd: Cmd, step: Step) -> Result<(), TransferFail> {
        let Some(hid) = self.hid.as_mut() else {
            return Err(TransferFail {
                step,
                detail: "hid port is closed".to_owned(),
            });
        };
        hid::transact(hid, &cmd, self.trace)
            .map(|_| ())
            .map_err(|err| TransferFail {
                step,
                detail: err.to_string(),
            })
    }

    fn bulk_step(&mut self, data: &[u8], step: Step) -> Result<(), TransferFail> {
        let Some(bulk) = self.bulk.as_mut() else {
            return Err(TransferFail {
                step,
                detail: "bulk port is closed".to_owned(),
            });
        };
        bulk.write_chunk(data).map_err(|err| TransferFail {
            step,
            detail: err.to_string(),
        })
    }

    /// Guard first. A deviation latches and sends nothing. Otherwise one
    /// `ShowLiquid`, then the guard runs again.
    fn recover(&mut self) -> bool {
        if !self.armed {
            return self.halted;
        }
        self.armed = false;
        match self.guard.check(&self.baseline) {
            Ok(()) => {
                if let Some(hid) = self.hid.as_mut() {
                    let _ = hid::transact(hid, &Cmd::ShowLiquid, self.trace);
                }
                if let Err(dev) = self.guard.check(&self.baseline) {
                    self.latch(dev);
                }
            }
            Err(dev) => self.latch(dev),
        }
        self.hid.take();
        self.bulk.take();
        self.halted
    }

    fn latch(&mut self, dev: Deviation) {
        self.halted = true;
        let current = self.guard.read();
        let body = format_halt(Some(&self.baseline), &current, dev);
        log_halt(&body);
        if let Err(err) = self.guard.write_latch(&body) {
            log_at(Priority::Crit, &format!("halt latch: {err}"));
        }
    }
}

struct TransferFail {
    step: Step,
    detail: String,
}

impl<H: HidPort, B: BulkPort> Drop for Transfer<H, B> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.recover();
        }
    }
}

fn overlaps_slot_region(start: u16, size: u16) -> bool {
    if size == 0 {
        return false;
    }
    let start = u32::from(start);
    let end = start + u32::from(size);
    start < u32::from(proto::SLOT_REGION.end) && end > u32::from(proto::SLOT_REGION.start)
}

fn map_exchange(step: Step, err: ExchangeError) -> SinkError {
    match err {
        ExchangeError::NoReply => SinkError::UploadFailed(UploadFailed::NoReply(step)),
        ExchangeError::Refused => SinkError::UploadFailed(UploadFailed::Refused(step)),
        ExchangeError::Protocol => SinkError::UploadFailed(UploadFailed::Protocol(step)),
        ExchangeError::Fence => SinkError::Fence,
        ExchangeError::Unavailable | ExchangeError::Failed => SinkError::DeviceUnavailable,
    }
}

struct UploadTiming {
    upload_ns: u64,
    switch_ns: u64,
}

impl UploadTiming {
    const ZERO: Self = Self {
        upload_ns: 0,
        switch_ns: 0,
    };
}

const TICK_NS: u64 = 1_000_000_000;

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn tick_if_due<H: HidPort, B: BulkPort>(
    lcd: &mut KrakenLcd<H, B>,
    clock: &impl BenchClock,
    last_tick: &mut u64,
) -> Result<(), SinkError> {
    if clock.now_ns().saturating_sub(*last_tick) >= TICK_NS {
        lcd.tick_and_hold()?;
        *last_tick = clock.now_ns();
    }
    Ok(())
}

fn wait_with_ticks<H: HidPort, B: BulkPort>(
    lcd: &mut KrakenLcd<H, B>,
    clock: &impl BenchClock,
    wait_ns: u64,
    last_tick: &mut u64,
) -> Result<(), SinkError> {
    let deadline = clock.now_ns().saturating_add(wait_ns);
    while clock.now_ns() < deadline {
        tick_if_due(lcd, clock, last_tick)?;
        let now = clock.now_ns();
        if now >= deadline {
            break;
        }
        let until_deadline = deadline - now;
        let since_tick = now.saturating_sub(*last_tick);
        let until_tick = TICK_NS.saturating_sub(since_tick);
        let slice = until_deadline.min(if until_tick == 0 { TICK_NS } else { until_tick });
        clock.sleep(Duration::from_nanos(slice));
    }
    tick_if_due(lcd, clock, last_tick)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last = sorted.len() - 1;
    let idx = ((last as f64) * p).round() as usize;
    sorted[idx.min(last)]
}

fn write_bench_summary(out: &mut impl Write, samples: &[BenchSample]) {
    write_stat_line(out, "upload_ms", samples, |s| s.upload_ms);
    write_stat_line(out, "switch_ms", samples, |s| s.switch_ms);
    write_stat_line(out, "total_ms", samples, |s| s.total_ms);
    write_stat_line(out, "slack_ms", samples, |s| s.slack_ms);
}

fn write_stat_line(
    out: &mut impl Write,
    name: &str,
    samples: &[BenchSample],
    pick: impl Fn(&BenchSample) -> f64,
) {
    let mut values: Vec<f64> = samples.iter().map(pick).collect();
    values.sort_by(|a, b| a.total_cmp(b));
    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(0.0);
    let median = percentile(&values, 0.5);
    let p95 = percentile(&values, 0.95);
    let _ = writeln!(
        out,
        "summary {name} min={min:.3} median={median:.3} p95={p95:.3} max={max:.3}"
    );
}

fn achieved_fps(count: u32, first: Option<u64>, last: Option<u64>, samples: &[BenchSample]) -> f64 {
    if count > 1
        && let (Some(first), Some(last)) = (first, last)
        && last > first
    {
        return f64::from(count - 1) / ns_to_ms(last - first) * 1000.0;
    }
    samples
        .first()
        .and_then(|sample| (sample.total_ms > 0.0).then(|| 1000.0 / sample.total_ms))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::reject_root;

    /// One test for the shared guard. That each entry point calls it first
    /// is checked in `tests/show_image.rs`.
    #[test]
    fn reject_root_refuses_only_euid_zero() {
        assert!(reject_root(true));
        assert!(!reject_root(false));
    }
}
