//! Keyboard protocol: byte-for-byte fixtures built from the protocol facts
//! in `src/keyboard/proto.rs`, the closed command table, and the forbidden
//! commands.

use llama_core::color::Rgb;
use llama_light::keyboard::keymap::LED_SLOTS;
use llama_light::keyboard::proto::{
    Channel, Cmd, EncodeError, OPCODES, PROPERTIES, REPORT_LEN, check_raw, encode, frame_reports,
};

/// A 65-byte hidraw write: `head` then zeros.
fn fixture(head: &[u8]) -> [u8; REPORT_LEN] {
    let mut out = [0u8; REPORT_LEN];
    out[..head.len()].copy_from_slice(head);
    out
}

/// A stream fixture: `00 7F <n> <len> 00` then `data`.
fn stream(packet: u8, data: &[u8]) -> [u8; REPORT_LEN] {
    let mut head = vec![0x00, 0x7F, packet, data.len() as u8, 0x00];
    head.extend_from_slice(data);
    fixture(&head)
}

#[test]
fn software_mode_is_07_05_02_00_03() {
    let report = encode(&Cmd::SoftwareMode).expect("encode");
    assert_eq!(
        report.as_bytes(),
        &fixture(&[0x00, 0x07, 0x05, 0x02, 0x00, 0x03])
    );
}

#[test]
fn hardware_mode_is_07_05_01_00_03() {
    let report = encode(&Cmd::HardwareMode).expect("encode");
    assert_eq!(
        report.as_bytes(),
        &fixture(&[0x00, 0x07, 0x05, 0x01, 0x00, 0x03])
    );
}

#[test]
fn commits_are_07_28_channel_03_with_apply_on_blue() {
    let cases = [
        (Channel::Red, [0x00, 0x07, 0x28, 0x01, 0x03, 0x01]),
        (Channel::Green, [0x00, 0x07, 0x28, 0x02, 0x03, 0x01]),
        (Channel::Blue, [0x00, 0x07, 0x28, 0x03, 0x03, 0x02]),
    ];
    for (channel, head) in cases {
        let report = encode(&Cmd::Commit(channel)).expect("encode");
        assert_eq!(report.as_bytes(), &fixture(&head), "{channel:?}");
    }
}

#[test]
fn stream_packets_are_60_60_24() {
    let data: Vec<u8> = (0..60).collect();
    let report = encode(&Cmd::Stream {
        packet: 1,
        data: &data,
    })
    .expect("encode");
    let mut head = vec![0x00, 0x7F, 0x01, 0x3C, 0x00];
    head.extend(0..60u8);
    assert_eq!(head.len(), REPORT_LEN, "60 bytes fill the report exactly");
    assert_eq!(report.as_bytes(), &fixture(&head));
    let tail: Vec<u8> = (0..24).collect();
    let report = encode(&Cmd::Stream {
        packet: 3,
        data: &tail,
    })
    .expect("encode");
    let mut head = vec![0x00, 0x7F, 0x03, 0x18, 0x00];
    head.extend(0..24u8);
    assert_eq!(report.as_bytes(), &fixture(&head));
}

#[test]
fn stream_packets_outside_the_table_are_refused() {
    let sixty = [0u8; 60];
    let short = [0u8; 24];
    for (packet, data) in [
        (0u8, &sixty[..]),
        (4, &sixty[..]),
        (1, &short[..]),
        (3, &sixty[..]),
        (2, &sixty[..59]),
    ] {
        assert_eq!(
            encode(&Cmd::Stream { packet, data }),
            Err(EncodeError::Range),
            "packet {packet} len {}",
            data.len()
        );
    }
}

#[test]
fn a_frame_is_twelve_reports_red_green_blue_each_three_streams_and_a_commit() {
    let mut slots = [Rgb { r: 0, g: 0, b: 0 }; LED_SLOTS];
    for (index, slot) in slots.iter_mut().enumerate() {
        let i = index as u8;
        *slot = Rgb {
            r: i,
            g: i.wrapping_add(1),
            b: 255 - i,
        };
    }
    let reports = frame_reports(&slots).expect("frame");
    assert_eq!(reports.len(), 12);
    let channel_of = |pick: fn(u8) -> u8| -> Vec<u8> { (0..144u8).map(pick).collect() };
    let channels = [
        (channel_of(|i| i), [0x00, 0x07, 0x28, 0x01, 0x03, 0x01]),
        (
            channel_of(|i| i.wrapping_add(1)),
            [0x00, 0x07, 0x28, 0x02, 0x03, 0x01],
        ),
        (
            channel_of(|i| 255 - i),
            [0x00, 0x07, 0x28, 0x03, 0x03, 0x02],
        ),
    ];
    for (c, (values, commit)) in channels.iter().enumerate() {
        let at = c * 4;
        assert_eq!(reports[at].as_bytes(), &stream(1, &values[0..60]));
        assert_eq!(reports[at + 1].as_bytes(), &stream(2, &values[60..120]));
        assert_eq!(reports[at + 2].as_bytes(), &stream(3, &values[120..144]));
        assert_eq!(reports[at + 3].as_bytes(), &fixture(commit));
    }
}

#[test]
fn every_report_a_frame_can_produce_passes_the_allowlist() {
    let slots = [Rgb {
        r: 255,
        g: 255,
        b: 255,
    }; LED_SLOTS];
    for report in frame_reports(&slots).expect("frame") {
        assert!(check_raw(*report.as_bytes()).is_ok());
    }
}

#[test]
fn the_command_table_is_closed() {
    assert_eq!(OPCODES, [0x07, 0x7F]);
    assert_eq!(PROPERTIES, [0x05, 0x28]);
    // Every (command, property) pair other than the four shapes is refused,
    // whatever follows. 07 05 and 07 28 pass only with their exact args.
    for command in 0..=255u8 {
        for property in 0..=255u8 {
            let bytes = fixture(&[0x00, command, property, 0x01, 0x03, 0x01]);
            let ok = check_raw(bytes).is_ok();
            let expect = matches!((command, property), (0x07, 0x28));
            assert_eq!(ok, expect, "{command:02x} {property:02x}");
        }
    }
    // A non-zero report number is refused.
    assert!(check_raw(fixture(&[0x01, 0x07, 0x05, 0x02, 0x00, 0x03])).is_err());
    // Lighting control with any mode other than 01/02, or other args.
    for mode in 0..=255u8 {
        let ok = check_raw(fixture(&[0x00, 0x07, 0x05, mode, 0x00, 0x03])).is_ok();
        assert_eq!(ok, mode == 1 || mode == 2, "mode {mode:02x}");
    }
    assert!(check_raw(fixture(&[0x00, 0x07, 0x05, 0x02, 0x01, 0x03])).is_err());
    assert!(check_raw(fixture(&[0x00, 0x07, 0x05, 0x02, 0x00, 0x04])).is_err());
    assert!(check_raw(fixture(&[0x00, 0x07, 0x05, 0x02, 0x00, 0x03, 0x01])).is_err());
    // Commit: blue must apply, red and green must not; three packets only.
    assert!(check_raw(fixture(&[0x00, 0x07, 0x28, 0x03, 0x03, 0x01])).is_err());
    assert!(check_raw(fixture(&[0x00, 0x07, 0x28, 0x01, 0x03, 0x02])).is_err());
    assert!(check_raw(fixture(&[0x00, 0x07, 0x28, 0x04, 0x03, 0x02])).is_err());
    assert!(check_raw(fixture(&[0x00, 0x07, 0x28, 0x01, 0x04, 0x01])).is_err());
    // Stream: a byte past the declared length is refused.
    let mut bytes = stream(3, &[7u8; 24]);
    bytes[5 + 24] = 1;
    assert!(check_raw(bytes).is_err());
    let mut bytes = stream(3, &[7u8; 24]);
    bytes[3] = 60;
    assert!(check_raw(bytes).is_err());
}

/// Commands that must never reach the keyboard. Each is refused by the
/// allowlist whatever its arguments, and no `Cmd` produces its bytes.
#[test]
fn forbidden_commands_are_refused() {
    let forbidden: &[(&str, &[u8])] = &[
        ("reset / bootloader", &[0x07, 0x02]),
        ("reset to bootloader", &[0x07, 0x02, 0xF0]),
        ("special function control", &[0x07, 0x04, 0x02]),
        ("poll rate", &[0x07, 0x0A]),
        ("firmware update start", &[0x07, 0x0C, 0xF0]),
        ("firmware update data commit", &[0x07, 0x0D]),
        ("hardware profile write", &[0x07, 0x13]),
        ("profile save", &[0x07, 0x14]),
        ("profile name / lighting save", &[0x07, 0x15]),
        ("hardware lighting save", &[0x07, 0x16]),
        ("hardware mode save", &[0x07, 0x17]),
        ("9-bit colour commit", &[0x07, 0x27, 0x01]),
        ("key input routing", &[0x07, 0x40, 0x1E]),
        ("read firmware info", &[0x0E, 0x01]),
        ("read anything", &[0x0E, 0x05]),
    ];
    let mut every = Vec::new();
    for cmd in [
        Cmd::SoftwareMode,
        Cmd::HardwareMode,
        Cmd::Commit(Channel::Red),
        Cmd::Commit(Channel::Green),
        Cmd::Commit(Channel::Blue),
    ] {
        every.push(*encode(&cmd).expect("encode").as_bytes());
    }
    for (name, head) in forbidden {
        for tail in [0x00u8, 0x01, 0x02, 0x03, 0xFF] {
            let mut bytes = vec![0x00];
            bytes.extend_from_slice(head);
            bytes.push(tail);
            let raw = fixture(&bytes);
            assert!(check_raw(raw).is_err(), "{name} passed the allowlist");
        }
        for sent in &every {
            assert_ne!(&sent[1..3], &head[..2], "{name} is built by an encoder");
        }
    }
}
