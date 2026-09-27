//! Corsair STRAFE RGB MK.2 (`1b1c:1b48`): detection only.
//!
//! This backend has no protocol writes yet (a follow-up ticket). It reads
//! `idVendor`/`idProduct` under `<sys>/bus/usb/devices` and reports the
//! keyboard present or absent. It opens no device node.

use std::path::{Path, PathBuf};

use llama_core::color::Rgb;

use crate::backend::Backend;

/// USB vendor id of the keyboard.
pub const KEYBOARD_VENDOR: &str = "1b1c";
/// USB product id of the keyboard.
pub const KEYBOARD_PRODUCT: &str = "1b48";

/// Key names a `keyboard.keys[...]` target may use, in row order. A range
/// `"F1".."F12"` is every key between the two in this order.
pub const KEY_ORDER: &[&str] = &[
    "Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "1", "2",
    "3", "4", "5", "6", "7", "8", "9", "0", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A",
    "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M",
];

/// Index of `name` in [`KEY_ORDER`].
#[must_use]
pub fn key_index(name: &str) -> Option<usize> {
    KEY_ORDER.iter().position(|key| *key == name)
}

/// Detection-only keyboard backend.
pub struct KeyboardStub {
    sys_root: PathBuf,
    found: bool,
}

impl KeyboardStub {
    /// `sys_root` is `/sys` in production.
    #[must_use]
    pub fn new(sys_root: impl Into<PathBuf>) -> Self {
        Self {
            sys_root: sys_root.into(),
            found: false,
        }
    }
}

/// Whether a `1b1c:1b48` USB device is listed under `sys_root`. Read-only.
#[must_use]
pub fn keyboard_listed(sys_root: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(sys_root.join("bus/usb/devices")) else {
        return false;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let vendor = read_id(&dir.join("idVendor"));
        let product = read_id(&dir.join("idProduct"));
        if vendor.as_deref() == Some(KEYBOARD_VENDOR)
            && product.as_deref() == Some(KEYBOARD_PRODUCT)
        {
            return true;
        }
    }
    false
}

fn read_id(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let id = text.trim().to_ascii_lowercase();
    (id.len() == 4).then_some(id)
}

impl Backend for KeyboardStub {
    fn name(&self) -> &'static str {
        "keyboard"
    }

    fn is_open(&self) -> bool {
        self.found
    }

    fn probe(&mut self) -> Result<(), String> {
        self.found = keyboard_listed(&self.sys_root);
        if self.found {
            Ok(())
        } else {
            Err("no 1b1c:1b48 on the USB bus".to_owned())
        }
    }

    /// No keyboard protocol yet: nothing is ever written.
    fn show(&mut self, _frame: &[Rgb]) -> Result<bool, String> {
        self.found = keyboard_listed(&self.sys_root);
        if self.found {
            Ok(false)
        } else {
            Err("keyboard unplugged".to_owned())
        }
    }
}
