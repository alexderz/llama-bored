//! Shared fake sysfs tree and fake ports for the device integration tests.
//!
//! Nothing here opens a real device node or writes outside the scratch directory.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kraken_lcd::device::proto::{self, Cmd, EncodedReport};
use kraken_lcd::device::{BulkPort, HidPort, KrakenLcd, OpenRequest, PortError, SinkError};

pub const BUSNUM: u8 = 3;
pub const DEVNUM: u32 = 2;
pub const PUMP_RPM: u32 = 1304;

pub struct Tree {
    pub root: PathBuf,
    pub sys: PathBuf,
    pub state: PathBuf,
}

type SendHook = Box<dyn FnMut(&[u8; 64]) + Send>;

pub struct SharedHid {
    state: Mutex<HidState>,
    on_send: Mutex<Option<SendHook>>,
    pub alive: AtomicUsize,
}

struct HidState {
    sent: Vec<[u8; 64]>,
    replies: VecDeque<[u8; 64]>,
    stale: VecDeque<[u8; 64]>,
    timeouts: Vec<Duration>,
    fail_send: Option<(u8, u8)>,
    read_unavailable: bool,
}

pub struct FakeHid {
    shared: Arc<SharedHid>,
}

pub struct SharedBulk {
    state: Mutex<BulkState>,
    pub alive: AtomicUsize,
}

struct BulkState {
    chunks: Vec<Vec<u8>>,
    fail_at: Option<usize>,
}

pub struct FakeBulk {
    shared: Arc<SharedBulk>,
}

pub struct Opened {
    pub lcd: KrakenLcd<FakeHid, FakeBulk>,
    pub hid: Arc<SharedHid>,
    pub bulk: Arc<SharedBulk>,
    pub tree: Tree,
}

impl Tree {
    pub fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("t12-{label}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sys = root.join("sys");
        let state = root.join("state");
        std::fs::create_dir_all(&sys).expect("sys root");
        std::fs::create_dir_all(&state).expect("state dir");
        Self { root, sys, state }
    }

    pub fn request(&self, rotate_deg: u16, trace_hid: bool) -> OpenRequest<'_> {
        OpenRequest {
            sys_root: &self.sys,
            state_dir: &self.state,
            rotate_deg,
            trace_hid,
        }
    }

    pub fn latch(&self) -> PathBuf {
        self.state.join("halted")
    }

    pub fn dev_dir(&self) -> PathBuf {
        self.sys.join("bus/usb/devices/3-5")
    }

    pub fn hwmon(&self) -> PathBuf {
        self.sys.join("class/hwmon/hwmon4")
    }
}

pub fn install_kraken(tree: &Tree) {
    write_kraken_usb(tree);
    write_hwmon(tree, PUMP_RPM, 89, 89, 0, 0);
    write_other_usb(tree);
}

fn write_kraken_usb(tree: &Tree) {
    let name = "3-5";
    let dev = tree.sys.join("bus/usb/devices").join(name);
    std::fs::create_dir_all(dev.join(format!("{name}:1.0"))).expect("iface 0");
    write(&dev.join("idVendor"), "1e71");
    write(&dev.join("idProduct"), "3008");
    write(&dev.join("busnum"), &BUSNUM.to_string());
    write(&dev.join("devnum"), &DEVNUM.to_string());
    write(&dev.join("bConfigurationValue"), "1");
    let hid = dev.join(format!("{name}:1.1")).join("0003:1E71:3008.0001");
    std::fs::create_dir_all(hid.join("hidraw/hidraw0")).expect("hidraw");
    write(
        &hid.join("uevent"),
        "HID_ID=0003:00001E71:00003008\nHID_NAME=NZXT Kraken Z\n",
    );
    write(&hid.join("hidraw/hidraw0/dev"), "236:0");
    std::os::unix::fs::symlink(
        "../../../../../../bus/hid/drivers/nzxt_kraken3",
        hid.join("driver"),
    )
    .expect("hid driver link");
}

fn write_other_usb(tree: &Tree) {
    let dev = tree.sys.join("bus/usb/devices/1-2");
    std::fs::create_dir_all(&dev).expect("other usb");
    write(&dev.join("idVendor"), "046d");
    write(&dev.join("idProduct"), "c52b");
    write(&dev.join("busnum"), "1");
    write(&dev.join("devnum"), "4");
    write(&dev.join("bConfigurationValue"), "1");
}

pub fn write_hwmon(tree: &Tree, fan: u32, pwm1: u32, pwm2: u32, en1: u32, en2: u32) {
    let dir = tree.hwmon();
    std::fs::create_dir_all(&dir).expect("hwmon");
    write(&dir.join("name"), "z53");
    write(&dir.join("fan1_input"), &fan.to_string());
    write(&dir.join("pwm1"), &pwm1.to_string());
    write(&dir.join("pwm2"), &pwm2.to_string());
    write(&dir.join("pwm1_enable"), &en1.to_string());
    write(&dir.join("pwm2_enable"), &en2.to_string());
}

pub fn claim_usbfs(tree: &Tree) {
    let iface = tree.dev_dir().join("3-5:1.0");
    std::fs::create_dir_all(&iface).expect("iface 0");
    let link = iface.join("driver");
    if link.symlink_metadata().is_ok() {
        return;
    }
    std::os::unix::fs::symlink("../../../../bus/usb/drivers/usbfs", &link).expect("usbfs link");
}

pub fn add_bootloader(tree: &Tree) {
    let dev = tree.sys.join("bus/usb/devices/9-1");
    std::fs::create_dir_all(&dev).expect("bootloader");
    write(&dev.join("idVendor"), "1e71");
    write(&dev.join("idProduct"), "3011");
    write(&dev.join("busnum"), "9");
    write(&dev.join("devnum"), "3");
    write(&dev.join("bConfigurationValue"), "1");
}

fn write(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    std::fs::write(path, text).expect("write fixture");
}

impl SharedHid {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(HidState {
                sent: Vec::new(),
                replies: VecDeque::new(),
                stale: VecDeque::new(),
                timeouts: Vec::new(),
                fail_send: None,
                read_unavailable: false,
            }),
            on_send: Mutex::new(None),
            alive: AtomicUsize::new(0),
        })
    }

    pub fn push(&self, reply: [u8; 64]) {
        self.state.lock().expect("hid").replies.push_back(reply);
    }

    pub fn push_stale(&self, reply: [u8; 64]) {
        self.state.lock().expect("hid").stale.push_back(reply);
    }

    pub fn sent(&self) -> Vec<[u8; 64]> {
        self.state.lock().expect("hid").sent.clone()
    }

    pub fn timeouts(&self) -> Vec<Duration> {
        self.state.lock().expect("hid").timeouts.clone()
    }

    pub fn on_send(&self, hook: impl FnMut(&[u8; 64]) + Send + 'static) {
        *self.on_send.lock().expect("hook") = Some(Box::new(hook));
    }

    pub fn fail_send(&self, first: u8, second: u8) {
        self.state.lock().expect("hid").fail_send = Some((first, second));
    }
}

impl FakeHid {
    pub fn attach(shared: Arc<SharedHid>) -> Self {
        shared.alive.fetch_add(1, Ordering::Relaxed);
        Self { shared }
    }
}

impl Drop for FakeHid {
    fn drop(&mut self) {
        self.shared.alive.fetch_sub(1, Ordering::Relaxed);
    }
}

impl HidPort for FakeHid {
    fn send(&mut self, report: &EncodedReport) -> Result<(), PortError> {
        let bytes = *report.as_bytes();
        if let Some(hook) = self.shared.on_send.lock().expect("hook").as_mut() {
            hook(&bytes);
        }
        let mut state = self.shared.state.lock().expect("hid");
        if let Some((first, second)) = state.fail_send
            && bytes[0] == first
            && bytes[1] == second
        {
            return Err(PortError::unavailable("injected send failure"));
        }
        state.sent.push(bytes);
        Ok(())
    }

    fn drain(&mut self) -> Result<Vec<[u8; 64]>, PortError> {
        let mut state = self.shared.state.lock().expect("hid");
        Ok(state.stale.drain(..).collect())
    }

    fn read_report(&mut self, timeout: Duration) -> Result<Option<[u8; 64]>, PortError> {
        let mut state = self.shared.state.lock().expect("hid");
        state.timeouts.push(timeout);
        if state.read_unavailable {
            return Err(PortError::unavailable("injected read failure"));
        }
        if let Some(stale) = state.stale.pop_front() {
            return Ok(Some(stale));
        }
        Ok(state.replies.pop_front())
    }
}

impl SharedBulk {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BulkState {
                chunks: Vec::new(),
                fail_at: None,
            }),
            alive: AtomicUsize::new(0),
        })
    }

    pub fn chunks(&self) -> Vec<Vec<u8>> {
        self.state.lock().expect("bulk").chunks.clone()
    }

    pub fn fail_at(&self, index: usize) {
        self.state.lock().expect("bulk").fail_at = Some(index);
    }
}

impl FakeBulk {
    pub fn attach(shared: Arc<SharedBulk>) -> Self {
        shared.alive.fetch_add(1, Ordering::Relaxed);
        Self { shared }
    }
}

impl Drop for FakeBulk {
    fn drop(&mut self) {
        self.shared.alive.fetch_sub(1, Ordering::Relaxed);
    }
}

impl BulkPort for FakeBulk {
    fn write_chunk(&mut self, data: &[u8]) -> Result<(), PortError> {
        let mut state = self.shared.state.lock().expect("bulk");
        if state.fail_at == Some(state.chunks.len()) {
            return Err(PortError::unavailable("injected bulk failure"));
        }
        state.chunks.push(data.to_vec());
        Ok(())
    }
}

pub fn ack(prefix: [u8; 2]) -> [u8; 64] {
    let mut reply = [0u8; 64];
    reply[0] = prefix[0];
    reply[1] = prefix[1];
    reply[14] = 0x01;
    reply
}

pub fn empty_bucket() -> [u8; 64] {
    ack([0x31, 0x04])
}

pub fn occupied(start: u16, size: u16) -> [u8; 64] {
    let mut reply = ack([0x31, 0x04]);
    reply[15] = 0x01;
    let start = start.to_le_bytes();
    let size = size.to_le_bytes();
    reply[17] = start[0];
    reply[18] = start[1];
    reply[19] = size[0];
    reply[20] = size[1];
    reply
}

pub fn lcd_info(orientation: u8) -> [u8; 64] {
    let mut reply = ack([0x31, 0x01]);
    reply[0x18] = 0x32;
    reply[0x1A] = orientation;
    reply
}

pub fn junk() -> [u8; 64] {
    let mut reply = [0u8; 64];
    reply[0] = 0x75;
    reply[1] = 0x01;
    reply
}

pub fn script_empty_open(hid: &SharedHid, orientation: u8) {
    for _ in 0..16 {
        hid.push(empty_bucket());
    }
    hid.push(lcd_info(orientation));
}

pub fn script_show(hid: &SharedHid) {
    for cmd in [
        Cmd::PreTransfer,
        Cmd::DeleteBucket(bucket(0)),
        Cmd::SetupBucket { slot: slot(0) },
        Cmd::WriteStart(slot(0)),
        Cmd::WriteEnd,
        Cmd::ShowSlot(slot(0)),
    ] {
        hid.push(ack(proto::expected_prefix(&cmd)));
    }
}

pub fn script_show_slot(hid: &SharedHid, id: u8) {
    let bucket = bucket(id);
    let slot = slot(id);
    for cmd in [
        Cmd::PreTransfer,
        Cmd::DeleteBucket(bucket),
        Cmd::SetupBucket { slot },
        Cmd::WriteStart(slot),
        Cmd::WriteEnd,
        Cmd::ShowSlot(slot),
    ] {
        hid.push(ack(proto::expected_prefix(&cmd)));
    }
}

fn slot(id: u8) -> proto::SlotId {
    proto::SlotId::try_new(id).expect("slot")
}

fn bucket(id: u8) -> proto::BucketId {
    proto::BucketId::try_new(id).expect("bucket")
}

pub fn open_lcd(
    tree: &Tree,
    hid: Arc<SharedHid>,
    bulk: Arc<SharedBulk>,
    rotate_deg: u16,
    trace_hid: bool,
) -> Result<KrakenLcd<FakeHid, FakeBulk>, SinkError> {
    let sys = tree.sys.clone();
    KrakenLcd::open(
        &tree.request(rotate_deg, trace_hid),
        move |bus, dev| {
            assert_eq!(bus, BUSNUM, "bulk open bus");
            assert_eq!(u32::from(dev), DEVNUM, "bulk open devnum");
            claim_usbfs_at(&sys);
            Ok(FakeBulk::attach(bulk))
        },
        move |path| {
            assert_eq!(path, Path::new("/dev/hidraw0"));
            Ok(FakeHid::attach(hid))
        },
    )
}

fn claim_usbfs_at(sys: &Path) {
    let iface = sys.join("bus/usb/devices/3-5/3-5:1.0");
    std::fs::create_dir_all(&iface).expect("iface");
    let link = iface.join("driver");
    if link.symlink_metadata().is_err() {
        std::os::unix::fs::symlink("../../../../bus/usb/drivers/usbfs", link).expect("usbfs");
    }
}

pub fn opened(label: &str) -> Opened {
    let tree = Tree::new(label);
    install_kraken(&tree);
    let hid = SharedHid::new();
    let bulk = SharedBulk::new();
    script_empty_open(&hid, 0);
    let lcd = open_lcd(&tree, Arc::clone(&hid), Arc::clone(&bulk), 0, false).expect("open");
    Opened {
        lcd,
        hid,
        bulk,
        tree,
    }
}

pub fn count_prefix(sent: &[[u8; 64]], prefix: &[u8]) -> usize {
    sent.iter()
        .filter(|report| report.starts_with(prefix))
        .count()
}

pub fn encoded(cmd: &Cmd) -> [u8; 64] {
    *proto::encode(cmd).expect("encode").as_bytes()
}
