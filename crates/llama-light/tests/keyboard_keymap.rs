//! The keyboard key map: names, slots, order, and the bench evidence.

use llama_core::color::Rgb;
use llama_light::keyboard::keymap::{KEYS, LED_SLOTS, key_index, suggest};
use llama_light::keyboard::slots;

/// OpenRGB 1.0's LED list for this keyboard (bench test, 2026-09-27), in
/// the order it printed them: the device's slot order. Names without
/// `"Key: "`. The last 11 of the 116 were lost in capture.
const BENCH_ORDER: &[&str] = &[
    "Escape",
    "`",
    "Tab",
    "Caps Lock",
    "Left Shift",
    "Left Control",
    "F12",
    "=",
    "Windows Lock",
    "Number Pad 7",
    "F1",
    "1",
    "Q",
    "A",
    "Left Windows",
    "Print Screen",
    "Media Mute",
    "Number Pad 8",
    "F2",
    "2",
    "W",
    "S",
    "Z",
    "Left Alt",
    "Scroll Lock",
    "Backspace",
    "Media Stop",
    "Number Pad 9",
    "F3",
    "3",
    "E",
    "D",
    "X",
    "Pause/Break",
    "Delete",
    "Media Previous",
    "F4",
    "4",
    "R",
    "F",
    "C",
    "Space",
    "Insert",
    "End",
    "Media Play/Pause",
    "Number Pad 4",
    "F5",
    "5",
    "T",
    "G",
    "V",
    "Home",
    "Page Down",
    "Media Next",
    "Number Pad 5",
    "F6",
    "6",
    "Y",
    "H",
    "B",
    "Page Up",
    "Right Shift",
    "Num Lock",
    "Number Pad 6",
    "F7",
    "7",
    "U",
    "J",
    "N",
    "Right Alt",
    "]",
    "Right Control",
    "Number Pad /",
    "Number Pad 1",
    "F8",
    "8",
    "I",
    "K",
    "M",
    "Right Windows",
    "\\ (ANSI)",
    "Up Arrow",
    "Number Pad *",
    "Number Pad 2",
    "F9",
    "9",
    "O",
    "L",
    ",",
    "Menu",
    "Left Arrow",
    "Number Pad -",
    "Number Pad 3",
    "F10",
    "0",
    "P",
    ";",
    ".",
    "Enter",
    "Down Arrow",
    "Number Pad +",
    "Number Pad 0",
    "F11",
    "-",
    "[",
];

#[test]
fn the_bench_names_are_known_and_in_slot_order() {
    let mut last = None;
    for name in BENCH_ORDER {
        let index = key_index(name).unwrap_or_else(|| panic!("{name} missing from KEYS"));
        let led = KEYS[index].led;
        if let Some(prev) = last {
            assert!(led > prev, "{name} (slot {led}) is not after slot {prev}");
        }
        last = Some(led);
    }
}

#[test]
fn slots_follow_the_twelve_per_column_grid() {
    // Position in a column is the physical row/block.
    let at = |name: &str| KEYS[key_index(name).expect(name)].led;
    let f_row = [
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
    ];
    for (column, name) in f_row.iter().enumerate() {
        assert_eq!(usize::from(at(name)), 12 * (column + 1), "{name}");
    }
    let numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"];
    for (column, name) in numbers.iter().enumerate() {
        assert_eq!(usize::from(at(name)), 12 * (column + 1) + 1, "{name}");
    }
    let numpad = [
        "Number Pad 7",
        "Number Pad 8",
        "Number Pad 9",
        "Number Pad 4",
        "Number Pad 5",
        "Number Pad 6",
        "Number Pad 1",
        "Number Pad 2",
        "Number Pad 3",
        "Number Pad 0",
        "Number Pad .",
    ];
    let columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11];
    for (name, column) in numpad.iter().zip(columns) {
        assert_eq!(usize::from(at(name)), 12 * column + 9, "{name}");
    }
    assert_eq!(at("Escape"), 0);
    assert_eq!(at("F12"), 6);
    assert_eq!(at("Windows Lock"), 8);
    assert_eq!(at("W"), 26);
    assert_eq!(at("A"), 15);
    assert_eq!(at("S"), 27);
    assert_eq!(at("D"), 39);
    assert_eq!(at("'"), 135);
    assert_eq!(at("/"), 136);
    assert_eq!(at("Right Arrow"), 139);
    assert_eq!(at("Number Pad Enter"), 140);
}

#[test]
fn every_key_has_its_own_slot_and_name() {
    // 105 from the bench list, 5 from the grid, and Brightness.
    assert_eq!(KEYS.len(), 111);
    let mut seen = [false; LED_SLOTS];
    for key in KEYS {
        let slot = usize::from(key.led);
        assert!(slot < LED_SLOTS, "{}", key.name);
        assert!(!seen[slot], "slot {slot} used twice ({})", key.name);
        seen[slot] = true;
        assert_eq!(key_index(key.name).map(|i| KEYS[i].name), Some(key.name));
        assert!(!key.name.contains('"'), "names are quoted in config");
        assert!(!key.name.starts_with("Key: "));
    }
}

#[test]
fn visual_order_puts_the_f_row_number_row_and_wasd_together() {
    let f1 = key_index("F1").expect("F1");
    let names: Vec<&str> = KEYS[f1..f1 + 12].iter().map(|k| k.name).collect();
    assert_eq!(
        names,
        [
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"
        ]
    );
    let one = key_index("1").expect("1");
    let names: Vec<&str> = KEYS[one..one + 12].iter().map(|k| k.name).collect();
    assert_eq!(
        names,
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="]
    );
}

#[test]
fn unknown_names_get_a_hint() {
    assert_eq!(key_index("f1"), None);
    assert_eq!(suggest("f1"), Some("F1"));
    assert_eq!(suggest("Key: Escape"), Some("Escape"));
    assert_eq!(suggest("space"), Some("Space"));
    assert_eq!(suggest("Esc"), None);
}

#[test]
fn a_frame_lands_on_the_right_slots_and_the_rest_are_black() {
    let mut frame = vec![Rgb { r: 0, g: 0, b: 0 }; KEYS.len()];
    let w = key_index("W").expect("W");
    frame[w] = Rgb { r: 9, g: 8, b: 7 };
    let out = slots(&frame).expect("slots");
    assert_eq!(out[26], Rgb { r: 9, g: 8, b: 7 });
    assert_eq!(
        out.iter()
            .filter(|c| **c != Rgb { r: 0, g: 0, b: 0 })
            .count(),
        1
    );
    let white = vec![
        Rgb {
            r: 255,
            g: 255,
            b: 255
        };
        KEYS.len()
    ];
    let out = slots(&white).expect("slots");
    // Unnamed slots (unused grid positions, unnamed LEDs) stay black.
    assert_eq!(out.iter().filter(|c| c.r == 255).count(), KEYS.len());
    assert_eq!(out[10], Rgb { r: 0, g: 0, b: 0 });
    assert!(slots(&white[1..]).is_err(), "a short frame is refused");
}
