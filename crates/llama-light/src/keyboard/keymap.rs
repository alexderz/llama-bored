//! Key names and LED slots of the Corsair STRAFE RGB MK.2 (ANSI).
//!
//! Names are OpenRGB's LED names without the `"Key: "` prefix, as the
//! OpenRGB 1.0 bench test listed them for this keyboard (2026-09-27), so a
//! config written while looking at OpenRGB uses the same words.
//!
//! LED slots: the legacy protocol carries 144 slots per colour channel.
//! Slots run column by column, 12 per column, and each position in a column
//! is one physical row or block: 0 F-row, 1 number row, 2 Q row, 3 A row,
//! 4 Z row, 5 bottom row, 6 and 7 the navigation block and row ends,
//! 8 the media/lock row above the numpad, 9 the numpad column. So `F1` is
//! slot 12, `F2` slot 24, and so on; `F12` sits in the first column (6),
//! as do `=` (7) and `Windows Lock` (8).
//!
//! Evidence: the bench test's LED list (in slot order) matches this grid
//! for the 105 names it printed; `tests/keyboard.rs` pins that order. The
//! last 11 names of the bench list were lost in capture. Of those, `'`,
//! `/`, `Right Arrow`, `Number Pad Enter` and `Number Pad .` follow from
//! the grid with confidence. `Brightness` at slot 137 is the least certain
//! entry and is listed for the on-device check. The bench zone has 116
//! LEDs; the others (such as a logo LED) have no name here and are
//! written black.
//!
//! Order: [`KEYS`] is visual order, row by row, left to right. A range
//! `"F1".."F12"` in a config is every key between the two in this order.

/// Colour slots per channel in one frame.
pub const LED_SLOTS: usize = 144;

/// One named key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Key {
    /// OpenRGB's name without `"Key: "`.
    pub name: &'static str,
    /// Slot in the 144-slot channel buffer.
    pub led: u8,
}

const fn k(name: &'static str, led: u8) -> Key {
    Key { name, led }
}

/// Every named key, in visual order: rows top to bottom, left to right.
pub const KEYS: &[Key] = &[
    // Row 0: function row, then the lock/brightness and media keys.
    k("Escape", 0),
    k("F1", 12),
    k("F2", 24),
    k("F3", 36),
    k("F4", 48),
    k("F5", 60),
    k("F6", 72),
    k("F7", 84),
    k("F8", 96),
    k("F9", 108),
    k("F10", 120),
    k("F11", 132),
    k("F12", 6),
    k("Print Screen", 18),
    k("Scroll Lock", 30),
    k("Pause/Break", 42),
    k("Brightness", 137),
    k("Windows Lock", 8),
    k("Media Stop", 32),
    k("Media Previous", 44),
    k("Media Play/Pause", 56),
    k("Media Next", 68),
    k("Media Mute", 20),
    // Row 1: number row.
    k("`", 1),
    k("1", 13),
    k("2", 25),
    k("3", 37),
    k("4", 49),
    k("5", 61),
    k("6", 73),
    k("7", 85),
    k("8", 97),
    k("9", 109),
    k("0", 121),
    k("-", 133),
    k("=", 7),
    k("Backspace", 31),
    k("Insert", 54),
    k("Home", 66),
    k("Page Up", 78),
    k("Num Lock", 80),
    k("Number Pad /", 92),
    k("Number Pad *", 104),
    k("Number Pad -", 116),
    // Row 2: Q row.
    k("Tab", 2),
    k("Q", 14),
    k("W", 26),
    k("E", 38),
    k("R", 50),
    k("T", 62),
    k("Y", 74),
    k("U", 86),
    k("I", 98),
    k("O", 110),
    k("P", 122),
    k("[", 134),
    k("]", 90),
    k("\\ (ANSI)", 102),
    k("Delete", 43),
    k("End", 55),
    k("Page Down", 67),
    k("Number Pad 7", 9),
    k("Number Pad 8", 21),
    k("Number Pad 9", 33),
    k("Number Pad +", 128),
    // Row 3: A row.
    k("Caps Lock", 3),
    k("A", 15),
    k("S", 27),
    k("D", 39),
    k("F", 51),
    k("G", 63),
    k("H", 75),
    k("J", 87),
    k("K", 99),
    k("L", 111),
    k(";", 123),
    k("'", 135),
    k("Enter", 126),
    k("Number Pad 4", 57),
    k("Number Pad 5", 69),
    k("Number Pad 6", 81),
    // Row 4: Z row.
    k("Left Shift", 4),
    k("Z", 28),
    k("X", 40),
    k("C", 52),
    k("V", 64),
    k("B", 76),
    k("N", 88),
    k("M", 100),
    k(",", 112),
    k(".", 124),
    k("/", 136),
    k("Right Shift", 79),
    k("Up Arrow", 103),
    k("Number Pad 1", 93),
    k("Number Pad 2", 105),
    k("Number Pad 3", 117),
    k("Number Pad Enter", 140),
    // Row 5: bottom row.
    k("Left Control", 5),
    k("Left Windows", 17),
    k("Left Alt", 29),
    k("Space", 53),
    k("Right Alt", 89),
    k("Right Windows", 101),
    k("Menu", 113),
    k("Right Control", 91),
    k("Left Arrow", 115),
    k("Down Arrow", 127),
    k("Right Arrow", 139),
    k("Number Pad 0", 129),
    k("Number Pad .", 141),
];

/// Position of `name` in [`KEYS`]. Exact match.
#[must_use]
pub fn key_index(name: &str) -> Option<usize> {
    KEYS.iter().position(|key| key.name == name)
}

/// A known name that differs from `name` only in letter case, for the
/// "did you mean" hint.
#[must_use]
pub fn suggest(name: &str) -> Option<&'static str> {
    let bare = name.strip_prefix("Key: ").unwrap_or(name);
    KEYS.iter()
        .find(|key| key.name.eq_ignore_ascii_case(bare))
        .map(|key| key.name)
}
