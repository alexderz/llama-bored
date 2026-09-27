//! Read-only USB sysfs checks that run before and after the LCD ports open.
//!
//! Callers pass a sysfs root. Production uses `/sys`. Tests pass a fake tree.
//! This module never writes.

use std::path::{Path, PathBuf};

use super::{SinkError, log_at};
use crate::log::Priority;

const VENDOR: &str = "1e71";
const PRODUCT: &str = "3008";
const BOOTLOADER_PRODUCT: &str = "3011";
const HID_ID_LINE: &str = "HID_ID=0003:00001E71:00003008";
const HID_DRIVER: &str = "nzxt_kraken3";
const USBFS_DRIVER: &str = "usbfs";
/// T44 / RR7: the udev symlink the unit's `DeviceAllow=` names, relative to
/// the `dev` directory beside the sysfs root (`/sys` -> `/dev`).
const PIN_DEV_DIR: &str = "dev";
const PIN_REL: &str = "kraken-lcd/hid";

/// The Kraken `1e71:3008` node discovered under a sysfs root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KrakenNode {
    /// Directory name under `bus/usb/devices`, such as `3-5`.
    pub name: String,
    pub busnum: u8,
    pub devnum: u8,
    /// `/dev/hidrawN` resolved from the bound HID device. Not opened here.
    pub hidraw: PathBuf,
    /// Canonical directory of the USB device inside the sysfs root.
    pub device_dir: PathBuf,
}

/// Pre-open checks. A bootloader device fails before any port is opened.
pub fn pre_open(sys_root: &Path) -> Result<KrakenNode, SinkError> {
    let root = canonical_root(sys_root)?;
    if bootloader_present(&root)? {
        log_at(
            Priority::Crit,
            "kraken reports product 3011; refusing to open",
        );
        return Err(SinkError::DeviceInBootloader);
    }
    let mut found = Vec::new();
    for dir in usb_device_dirs(&root)? {
        if vendor_product(&root, &dir)? == Some((VENDOR, PRODUCT)) {
            found.push(dir);
        }
    }
    let device_dir = match found.len() {
        1 => found.remove(0),
        n => {
            let others = if n == 0 {
                other_nzxt_products(&root)
            } else {
                Vec::new()
            };
            log_at(Priority::Err, &unsupported_message(n, &others));
            return Err(SinkError::DeviceUnavailable);
        }
    };
    let name = device_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(SinkError::DeviceUnavailable)?
        .to_owned();
    if read_u32(&root, &device_dir.join("bConfigurationValue"))? != 1 {
        log_at(Priority::Err, "kraken configuration is not 1");
        return Err(SinkError::DeviceUnavailable);
    }
    if link_name(&device_dir.join(format!("{name}:1.0")).join("driver"))?.is_some() {
        log_at(Priority::Err, "kraken interface 0 already has a driver");
        return Err(SinkError::DeviceUnavailable);
    }
    let hidraw = resolve_hidraw(&root, &device_dir, &name)?;
    check_pin(&root, &hidraw)?;
    let busnum = u8_field(&root, &device_dir.join("busnum"))?;
    let devnum = u8_field(&root, &device_dir.join("devnum"))?;
    Ok(KrakenNode {
        name,
        busnum,
        devnum,
        hidraw,
        device_dir,
    })
}

/// `true` when the Kraken's interface 0 (`name:1.0`) is bound to `usbfs`.
///
/// That binding is the running service's bulk claim. A one-shot query must
/// not open the device beside it. A missing Kraken is [`SinkError::DeviceUnavailable`].
/// A bootloader device is [`SinkError::DeviceInBootloader`].
pub fn interface0_usbfs(sys_root: &Path) -> Result<bool, SinkError> {
    let root = canonical_root(sys_root)?;
    if bootloader_present(&root)? {
        log_at(
            Priority::Crit,
            "kraken reports product 3011; refusing to open",
        );
        return Err(SinkError::DeviceInBootloader);
    }
    let mut found = Vec::new();
    for dir in usb_device_dirs(&root)? {
        if vendor_product(&root, &dir)? == Some((VENDOR, PRODUCT)) {
            found.push(dir);
        }
    }
    let device_dir = match found.len() {
        1 => found.remove(0),
        _ => return Err(SinkError::DeviceUnavailable),
    };
    let name = device_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(SinkError::DeviceUnavailable)?;
    let driver = link_name(&device_dir.join(format!("{name}:1.0")).join("driver"))?;
    Ok(driver.as_deref() == Some(USBFS_DRIVER))
}

/// Post-open checks: interface 0 is `usbfs`, and the HID driver is unchanged.
///
/// A miss is [`SinkError::Fatal`]: the caller logs it and exits. It is not retried.
pub fn post_open(sys_root: &Path, node: &KrakenNode) -> Result<(), SinkError> {
    let root = canonical_root(sys_root)
        .map_err(|_| SinkError::Fatal("post-open sysfs root is unreadable"))?;
    if binding(&root, node).is_err() {
        return Err(SinkError::Fatal("interface 1 is not bound to nzxt_kraken3"));
    }
    let driver = match link_name(
        &node
            .device_dir
            .join(format!("{}:1.0", node.name))
            .join("driver"),
    ) {
        Ok(driver) => driver,
        Err(_) => {
            log_at(
                Priority::Crit,
                "post-open check: interface 0 driver is not usbfs",
            );
            return Err(SinkError::Fatal("interface 0 driver is not usbfs"));
        }
    };
    if driver.as_deref() != Some(USBFS_DRIVER) {
        log_at(
            Priority::Crit,
            "post-open check: interface 0 driver is not usbfs",
        );
        return Err(SinkError::Fatal("interface 0 driver is not usbfs"));
    }
    Ok(())
}

/// HID binding check used by the hidraw-only restore path.
pub fn binding(sys_root: &Path, node: &KrakenNode) -> Result<(), SinkError> {
    let root = canonical_root(sys_root)?;
    if resolve_hidraw(&root, &node.device_dir, &node.name).is_err() {
        log_at(
            Priority::Crit,
            "post-open check: interface 1 is not bound to nzxt_kraken3",
        );
        return Err(SinkError::DeviceUnavailable);
    }
    Ok(())
}

/// `true` when any USB device under `sys_root` is `1e71:3011`.
pub fn bootloader_present_at(sys_root: &Path) -> Result<bool, SinkError> {
    let root = canonical_root(sys_root)?;
    bootloader_present(&root)
}

/// Log text when there is not exactly one `1e71:3008` device.
///
/// `others` are product ids of other NZXT (`1e71`) USB devices. They are named
/// so a user with another Kraken sees why nothing happens; they are never opened.
fn unsupported_message(found: usize, others: &[String]) -> String {
    if found > 1 {
        return format!(
            "found {found} NZXT Kraken Z devices (1e71:3008); kraken-lcd drives exactly one and opens none"
        );
    }
    if others.is_empty() {
        return "no NZXT Kraken Z LCD (1e71:3008) found; kraken-lcd supports only that device \
                (tested on the Kraken Z53)"
            .to_owned();
    }
    let ids: Vec<String> = others.iter().map(|id| format!("1e71:{id}")).collect();
    format!(
        "found NZXT device {} but no Kraken Z LCD (1e71:3008); that model is not supported \
         and is not opened (kraken-lcd is tested on the Kraken Z53 only)",
        ids.join(", ")
    )
}

/// Product ids of NZXT USB devices that are neither `3008` nor the `3011`
/// bootloader. Read-only; only `idVendor` and `idProduct` are read. An id
/// that is not four hex digits is skipped, so nothing odd reaches the log.
fn other_nzxt_products(root: &Path) -> Vec<String> {
    let Ok(dirs) = usb_device_dirs(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(vendor) = read_trim(root, &dir.join("idVendor")) else {
            continue;
        };
        if !vendor.eq_ignore_ascii_case(VENDOR) {
            continue;
        }
        let Ok(product) = read_trim(root, &dir.join("idProduct")) else {
            continue;
        };
        let product = product.to_ascii_lowercase();
        if product.len() != 4 || !product.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        if product == PRODUCT || product == BOOTLOADER_PRODUCT {
            continue;
        }
        out.push(product);
    }
    out.sort();
    out.dedup();
    out
}

fn canonical_root(sys_root: &Path) -> Result<PathBuf, SinkError> {
    sys_root
        .canonicalize()
        .map_err(|_| SinkError::DeviceUnavailable)
}

fn bootloader_present(root: &Path) -> Result<bool, SinkError> {
    for dir in usb_device_dirs(root)? {
        if vendor_product(root, &dir)? == Some((VENDOR, BOOTLOADER_PRODUCT)) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn usb_device_dirs(root: &Path) -> Result<Vec<PathBuf>, SinkError> {
    let dir = root.join("bus/usb/devices");
    let entries = std::fs::read_dir(&dir).map_err(|_| SinkError::DeviceUnavailable)?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|_| SinkError::DeviceUnavailable)?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.contains(':') || name.starts_with("usb") {
            continue;
        }
        let Ok(path) = entry.path().canonicalize() else {
            continue;
        };
        if path.starts_with(root) {
            out.push(path);
        }
    }
    Ok(out)
}

fn vendor_product(
    root: &Path,
    dir: &Path,
) -> Result<Option<(&'static str, &'static str)>, SinkError> {
    let vendor = match read_trim(root, &dir.join("idVendor")) {
        Ok(value) => value.to_ascii_lowercase(),
        Err(SinkError::DeviceUnavailable) => return Ok(None),
        Err(err) => return Err(err),
    };
    let product = match read_trim(root, &dir.join("idProduct")) {
        Ok(value) => value.to_ascii_lowercase(),
        Err(SinkError::DeviceUnavailable) => return Ok(None),
        Err(err) => return Err(err),
    };
    let vendor = match vendor.as_str() {
        VENDOR => VENDOR,
        _ => return Ok(None),
    };
    let product = match product.as_str() {
        PRODUCT => PRODUCT,
        BOOTLOADER_PRODUCT => BOOTLOADER_PRODUCT,
        _ => return Ok(None),
    };
    Ok(Some((vendor, product)))
}

fn resolve_hidraw(root: &Path, device_dir: &Path, name: &str) -> Result<PathBuf, SinkError> {
    let iface = device_dir.join(format!("{name}:1.1"));
    let entries = std::fs::read_dir(&iface).map_err(|_| SinkError::DeviceUnavailable)?;
    let mut node = None;
    for entry in entries {
        let entry = entry.map_err(|_| SinkError::DeviceUnavailable)?;
        let path = match entry.path().canonicalize() {
            Ok(path) if path.starts_with(root) => path,
            _ => continue,
        };
        let uevent = match read_trim(root, &path.join("uevent")) {
            Ok(text) => text,
            Err(_) => continue,
        };
        if !uevent.lines().any(|line| line == HID_ID_LINE) {
            continue;
        }
        if link_name(&path.join("driver"))?.as_deref() != Some(HID_DRIVER) {
            return Err(SinkError::DeviceUnavailable);
        }
        let hidraw = hidraw_node(root, &path.join("hidraw"))?;
        if node.replace(hidraw).is_some() {
            return Err(SinkError::DeviceUnavailable);
        }
    }
    node.ok_or(SinkError::DeviceUnavailable)
}

/// T44 / RR7: when the udev pin `/dev/kraken-lcd/hid` exists, it must
/// resolve to the hidraw node sysfs gave. The unit's `DeviceAllow=` names the
/// pin, so a pin on another node means the cgroup and the code disagree about
/// which device is ours; refuse the open. An absent pin is not checked here:
/// without it the unit admits no hidraw node and the open fails as
/// "device unavailable". Metadata, `readlink` and `realpath` only; the pin is
/// never opened. `<sys_root>/../dev` is `/dev` in production.
fn check_pin(root: &Path, hidraw: &Path) -> Result<(), SinkError> {
    let dev = root
        .parent()
        .map(|parent| parent.join(PIN_DEV_DIR))
        .ok_or(SinkError::DeviceUnavailable)?;
    let pin = dev.join(PIN_REL);
    match pin.symlink_metadata() {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Ok(meta) if meta.file_type().is_symlink() => {}
        _ => {
            log_at(
                Priority::Err,
                &format!(
                    "{} is not a symlink; refusing to open the kraken",
                    pin.display()
                ),
            );
            return Err(SinkError::DeviceUnavailable);
        }
    }
    let want = hidraw
        .file_name()
        .and_then(|name| dev.canonicalize().ok().map(|dev| dev.join(name)));
    match (pin.canonicalize(), want) {
        (Ok(got), Some(want)) if got == want => Ok(()),
        (got, _) => {
            let got = got.map_or_else(|_| "nothing".to_owned(), |path| format!("{path:?}"));
            log_at(
                Priority::Err,
                &format!(
                    "{} resolves to {got} but sysfs gives {hidraw:?}; refusing to open the kraken",
                    pin.display()
                ),
            );
            Err(SinkError::DeviceUnavailable)
        }
    }
}

fn hidraw_node(root: &Path, dir: &Path) -> Result<PathBuf, SinkError> {
    let entries = std::fs::read_dir(dir).map_err(|_| SinkError::DeviceUnavailable)?;
    let mut found = None;
    for entry in entries {
        let entry = entry.map_err(|_| SinkError::DeviceUnavailable)?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(index) = name.strip_prefix("hidraw") else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let path = entry
            .path()
            .canonicalize()
            .map_err(|_| SinkError::DeviceUnavailable)?;
        if !path.starts_with(root) {
            return Err(SinkError::DeviceUnavailable);
        }
        if found
            .replace(PathBuf::from(format!("/dev/{name}")))
            .is_some()
        {
            return Err(SinkError::DeviceUnavailable);
        }
    }
    found.ok_or(SinkError::DeviceUnavailable)
}

fn u8_field(root: &Path, path: &Path) -> Result<u8, SinkError> {
    let value = read_u32(root, path)?;
    u8::try_from(value).map_err(|_| SinkError::DeviceUnavailable)
}

fn read_u32(root: &Path, path: &Path) -> Result<u32, SinkError> {
    let text = read_trim(root, path)?;
    text.parse().map_err(|_| SinkError::DeviceUnavailable)
}

fn read_trim(root: &Path, path: &Path) -> Result<String, SinkError> {
    let canon = path
        .canonicalize()
        .map_err(|_| SinkError::DeviceUnavailable)?;
    if !canon.starts_with(root) {
        return Err(SinkError::DeviceUnavailable);
    }
    let text = std::fs::read_to_string(canon).map_err(|_| SinkError::DeviceUnavailable)?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Err(SinkError::DeviceUnavailable)
    } else {
        Ok(trimmed.to_owned())
    }
}

/// Coolant °C from the single `z53` hwmon `temp1_input`, or `None` on failure.
///
/// The directory is the one the cooling guard resolves: exactly one `hwmonN`
/// whose `name` is `z53`, canonicalized and still inside `sys_root`. The
/// read is read-only. Millidegrees in the file are divided by 1000.
#[must_use]
pub fn read_coolant_c(sys_root: &Path) -> Option<f32> {
    let root = sys_root.canonicalize().ok()?;
    let dir = super::guard::find_z53(&root)?;
    let path = dir.join("temp1_input");
    let canon = path.canonicalize().ok()?;
    if !canon.starts_with(&root) {
        return None;
    }
    let text = std::fs::read_to_string(canon).ok()?;
    let milli: f32 = text.trim().parse().ok()?;
    let celsius = milli / 1_000.0;
    // Same window as the snapshot wire: finite and −20..=150 °C.
    if celsius.is_finite() && (-20.0..=150.0).contains(&celsius) {
        Some(celsius)
    } else {
        None
    }
}

/// `Ok(None)` when the symlink is absent. A non-symlink at the path is an error.
fn link_name(path: &Path) -> Result<Option<String>, SinkError> {
    match path.symlink_metadata() {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(SinkError::DeviceUnavailable),
        Ok(meta) if !meta.file_type().is_symlink() => Err(SinkError::DeviceUnavailable),
        Ok(_) => {
            let target = std::fs::read_link(path).map_err(|_| SinkError::DeviceUnavailable)?;
            let name = target
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or(SinkError::DeviceUnavailable)?;
            Ok(Some(name.to_owned()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Fake sysfs root. Tests remove it with [`Scratch::finish`]; no `Drop`
    /// impl, so S11's public-surface scan of this file is unchanged.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path =
                std::env::temp_dir().join(format!("kraken-lcd-sysfs-unit-{label}-{nanos}-{n}"));
            std::fs::create_dir_all(path.join("bus/usb/devices")).expect("scratch");
            Self(path)
        }

        fn finish(self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }

        fn usb(&self, name: &str, vendor: &str, product: &str) {
            let dir = self.0.join("bus/usb/devices").join(name);
            std::fs::create_dir_all(&dir).expect("usb dir");
            std::fs::write(dir.join("idVendor"), format!("{vendor}\n")).expect("vendor");
            std::fs::write(dir.join("idProduct"), format!("{product}\n")).expect("product");
        }
    }

    #[test]
    fn other_nzxt_products_lists_only_unsupported_nzxt_ids() {
        let scratch = Scratch::new("others");
        scratch.usb("1-1", "1e71", "300c");
        scratch.usb("1-2", "1E71", "2007");
        scratch.usb("1-3", "046d", "c52b");
        scratch.usb("1-4", "1e71", "zz\u{1b}[2J");
        let root = scratch.0.canonicalize().expect("root");
        let found = other_nzxt_products(&root);
        scratch.finish();
        assert_eq!(found, vec!["2007", "300c"]);
    }

    #[test]
    fn a_supported_kraken_is_not_listed_as_other() {
        let scratch = Scratch::new("supported");
        scratch.usb("1-1", "1e71", "3008");
        let root = scratch.0.canonicalize().expect("root");
        let found = other_nzxt_products(&root);
        scratch.finish();
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn missing_device_message_names_the_supported_model() {
        let none = unsupported_message(0, &[]);
        assert!(none.contains("1e71:3008"), "{none}");
        assert!(none.contains("Z53"), "{none}");
        assert!(none.contains("tested"), "{none}");

        let other = unsupported_message(0, &["300c".to_owned()]);
        assert!(other.contains("1e71:300c"), "{other}");
        assert!(other.contains("not supported"), "{other}");
        assert!(other.contains("not opened"), "{other}");

        let two = unsupported_message(2, &[]);
        assert!(two.contains("2"), "{two}");
        assert!(two.contains("exactly one"), "{two}");
    }
}
