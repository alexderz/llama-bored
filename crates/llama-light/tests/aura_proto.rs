//! Aura protocol: byte-for-byte fixtures built from the protocol facts in
//! `src/aura/proto.rs`, the closed opcode table, and the forbidden opcodes.

use llama_core::color::Rgb;
use llama_light::aura::proto::{
    self, Cmd, EncodeError, LEDS_PER_REPORT, MAX_LEDS, OPCODES, REPORT_LEN, check_raw, encode,
    frame_reports,
};

fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    Rgb { r, g, b }
}

/// A 65-byte report: `head` then zeros.
fn fixture(head: &[u8]) -> [u8; REPORT_LEN] {
    let mut out = [0u8; REPORT_LEN];
    out[..head.len()].copy_from_slice(head);
    out
}

#[test]
fn enter_direct_is_ec_35_header1_mode_ff() {
    let report = encode(&Cmd::EnterDirect).expect("encode");
    assert_eq!(
        report.as_bytes(),
        &fixture(&[0xEC, 0x35, 0x01, 0x00, 0x00, 0xFF])
    );
}

#[test]
fn a_mirrored_six_led_frame_is_one_apply_report() {
    let frame = [
        rgb(1, 2, 3),
        rgb(4, 5, 6),
        rgb(7, 8, 9),
        rgb(10, 11, 12),
        rgb(13, 14, 15),
        rgb(16, 17, 18),
    ];
    let reports = frame_reports(&frame).expect("frame");
    assert_eq!(reports.len(), 1);
    let mut head = vec![0xEC, 0x40, 0x80, 0x00, 0x06];
    head.extend(1..=18u8);
    assert_eq!(reports[0].as_bytes(), &fixture(&head));
}

#[test]
fn a_thirty_led_chain_is_twenty_then_ten_with_apply_on_the_last() {
    let frame: Vec<Rgb> = (0..30u8).map(|i| rgb(i, i + 100, 200 - i)).collect();
    let reports = frame_reports(&frame).expect("frame");
    assert_eq!(reports.len(), 2);

    let mut first = vec![0xEC, 0x40, 0x00, 0x00, 20];
    for i in 0..20u8 {
        first.extend([i, i + 100, 200 - i]);
    }
    assert_eq!(first.len(), REPORT_LEN, "20 LEDs fill the report exactly");
    assert_eq!(reports[0].as_bytes(), &fixture(&first));

    let mut second = vec![0xEC, 0x40, 0x80, 20, 10];
    for i in 20..30u8 {
        second.extend([i, i + 100, 200 - i]);
    }
    assert_eq!(reports[1].as_bytes(), &fixture(&second));
}

#[test]
fn direct_ranges_are_closed() {
    let led = [rgb(1, 1, 1)];
    assert_eq!(
        encode(&Cmd::Direct {
            start: 0,
            leds: &[],
            apply: true
        }),
        Err(EncodeError::Range)
    );
    let too_many = vec![rgb(0, 0, 0); LEDS_PER_REPORT + 1];
    assert_eq!(
        encode(&Cmd::Direct {
            start: 0,
            leds: &too_many,
            apply: true
        }),
        Err(EncodeError::Range)
    );
    assert_eq!(
        encode(&Cmd::Direct {
            start: MAX_LEDS as u8,
            leds: &led,
            apply: true
        }),
        Err(EncodeError::Range)
    );
    assert_eq!(frame_reports(&[]), Err(EncodeError::Range));
    assert_eq!(
        frame_reports(&vec![rgb(0, 0, 0); MAX_LEDS + 1]),
        Err(EncodeError::Range)
    );
}

/// S2-style table: the allowlist holds exactly two opcodes, and every other
/// value of byte 1 is refused whatever follows it.
#[test]
fn the_opcode_table_is_exactly_effect_and_direct() {
    assert_eq!(OPCODES.as_slice(), &[0x35, 0x40]);
    for op in 0..=255u8 {
        let accepted = check_raw(fixture(&[0xEC, op, 0x01, 0x00, 0x00, 0xFF])).is_ok()
            || check_raw(fixture(&[0xEC, op, 0x80, 0x00, 0x01, 1, 2, 3])).is_ok();
        assert_eq!(
            accepted,
            OPCODES.contains(&op),
            "opcode {op:#04x} allowlist mismatch"
        );
    }
    // Wrong report id.
    assert!(check_raw(fixture(&[0xED, 0x35, 0x01, 0x00, 0x00, 0xFF])).is_err());
}

/// Save / commit / config writes. None may ever leave the encoder.
const FORBIDDEN: &[(&[u8], &str)] = &[
    (&[0xEC, 0x3F, 0x55], "commit current effect to flash (save)"),
    (&[0xEC, 0x3F, 0xAA], "commit variant"),
    (
        &[0xEC, 0x3E, 0x52, 0x53],
        "header configuration write (Gen1/Gen2)",
    ),
    (&[0xEC, 0x36, 0x00, 0xFF, 0x00, 1, 2, 3], "effect colour"),
    (&[0xEC, 0xB0], "config table request"),
    (&[0xEC, 0x82], "firmware version request"),
    // Effect opcode with anything but Direct on header 1.
    (
        &[0xEC, 0x35, 0x00, 0x00, 0x00, 0x00],
        "mainboard channel off",
    ),
    (&[0xEC, 0x35, 0x01, 0x00, 0x00, 0x01], "header 1 static"),
    (
        &[0xEC, 0x35, 0x00, 0x00, 0x00, 0xFF],
        "mainboard channel direct",
    ),
    (
        &[0xEC, 0x35, 0x01, 0x00, 0x01, 0xFF],
        "shutdown-effect byte set",
    ),
    // Direct opcode on another channel, or malformed.
    (
        &[0xEC, 0x40, 0x81, 0x00, 0x01, 1, 2, 3],
        "direct on header 2",
    ),
    (
        &[0xEC, 0x40, 0x84, 0x00, 0x01, 1, 2, 3],
        "direct on the mainboard channel",
    ),
    (&[0xEC, 0x40, 0x80, 0x00, 0x00], "zero LEDs"),
    (&[0xEC, 0x40, 0x80, 0x00, 21], "21 LEDs"),
    (
        &[0xEC, 0x40, 0x80, 119, 2, 1, 1, 1, 1, 1, 1],
        "past LED 120",
    ),
];

#[test]
fn forbidden_opcodes_never_pass_the_fence() {
    for (head, what) in FORBIDDEN {
        assert!(
            check_raw(fixture(head)).is_err(),
            "{what} ({head:02x?}) passed the allowlist"
        );
    }
    // Trailing garbage after a valid direct payload is refused too.
    let mut tail = fixture(&[0xEC, 0x40, 0x80, 0x00, 0x01, 1, 2, 3]);
    tail[20] = 0x3F;
    assert!(check_raw(tail).is_err());
    let mut tail = fixture(&[0xEC, 0x35, 0x01, 0x00, 0x00, 0xFF]);
    tail[6] = 0x55;
    assert!(check_raw(tail).is_err());
}

#[test]
fn no_encodable_command_carries_a_forbidden_opcode() {
    let leds: Vec<Rgb> = (0..LEDS_PER_REPORT as u8).map(|i| rgb(i, i, i)).collect();
    let mut all = vec![encode(&Cmd::EnterDirect).expect("enter")];
    for start in [0u8, 20, 100] {
        for n in [1usize, 6, 20] {
            for apply in [false, true] {
                all.push(
                    encode(&Cmd::Direct {
                        start,
                        leds: &leds[..n],
                        apply,
                    })
                    .expect("direct"),
                );
            }
        }
    }
    for report in all {
        let bytes = report.as_bytes();
        assert_eq!(bytes[0], proto::REPORT_ID);
        assert!(OPCODES.contains(&bytes[1]), "{bytes:02x?}");
        assert!(![0x3E, 0x3F].contains(&bytes[1]));
    }
}
