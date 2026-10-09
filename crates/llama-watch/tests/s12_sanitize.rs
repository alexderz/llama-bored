//! S12: llama text becomes cells, and a frame of those cells stays on the
//! term allowlist.

use std::time::{Duration, Instant};

use llama_watch::tty::sanitize::sanitize;
use llama_watch::tty::term::{ConsoleBlank, GLYPHS, Size, Term};
use llama_watch::tty::{C16, Cell, Grid};

fn chars(input: &str) -> String {
    let mut cells = Vec::new();
    sanitize(input, &mut cells);
    cells.iter().map(|cell| cell.ch).collect()
}

/// #90: the distinct "this console cannot show it" placeholder. Never `?`.
const UNKNOWN: char = '\u{fffd}';

/// Printable ASCII, `?`, the end-of-line marker, or [`UNKNOWN`]. Nothing else.
fn assert_cells_are_safe(input: &str) {
    let mut cells = Vec::new();
    sanitize(input, &mut cells);
    for cell in cells {
        let ch = cell.ch;
        assert!(
            ch == '\n' || ch == '?' || ch == UNKNOWN || ('\u{20}'..='\u{7e}').contains(&ch),
            "input {input:?} produced {ch:?}"
        );
    }
}

#[test]
fn keeps_printable_ascii() {
    assert_eq!(chars("Hello, world ~"), "Hello, world ~");
    assert_eq!(chars(""), "");
    assert_eq!(chars(" "), " ");
}

#[test]
fn tab_expands_to_the_next_4_column_stop() {
    assert_eq!(chars("\t"), "    ");
    assert_eq!(chars("a\t"), "a   ");
    assert_eq!(chars("ab\t"), "ab  ");
    assert_eq!(chars("abc\t"), "abc ");
    assert_eq!(chars("abcd\t"), "abcd    ");
}

#[test]
fn newline_marks_the_end_of_the_line_and_resets_tabs() {
    assert_eq!(chars("ab\ncd"), "ab\ncd");
    assert_eq!(chars("abcd\n\t"), "abcd\n    ");
    assert_eq!(chars("\n\n"), "\n\n");
}

#[test]
fn carriage_return_is_dropped_and_does_not_reset_the_column() {
    assert_eq!(chars("\r"), "");
    assert_eq!(chars("a\rb"), "ab");
    assert_eq!(chars("ab\r\t"), "ab  ");
    assert_eq!(chars("a\r\nb"), "a\nb");
}

#[test]
fn escape_payloads_lose_controls_and_keep_printable_text() {
    let cases = [
        ("\u{1b}]0;x\u{7}", "]0;x"),
        ("\u{1b}[2J", "[2J"),
        ("\u{9b}31m", "31m"),
        ("\u{1b}", ""),
        ("\u{1b}\u{1b}", ""),
        ("ok\u{1b}[31mX", "ok[31mX"),
    ];
    for (input, expect) in cases {
        assert_eq!(chars(input), expect, "input {input:?}");
        assert_cells_are_safe(input);
    }
}

/// #90: accented Latin transliterates to its base letter (the `.nfkd()`
/// result for the ~170 of these with a real decomposition; a hand-picked
/// stand-in for the dozen or so, like `Æ`/`ß`/`Ø`, that have none) instead of
/// becoming a question mark or the unknown placeholder.
#[test]
fn accented_latin_becomes_its_base_ascii_letter() {
    assert_eq!(chars("Û"), "U");
    assert_eq!(chars("\u{db}"), "U");
    assert_eq!(chars("aÛb"), "aUb");
    assert_eq!(chars("\u{ff}"), "y"); // ÿ
    assert_eq!(chars("Calliope's"), "Calliope's"); // plain ASCII: untouched
    assert_eq!(chars("café"), "cafe");
    assert_eq!(chars("naïve"), "naive");
    assert_eq!(chars("Zürich"), "Zurich");
    // No Unicode decomposition at all: ligatures and "stroke" letters, by
    // hand, one ASCII letter each.
    for (input, want) in [
        ("Æ", "A"),
        ("æ", "a"),
        ("ß", "s"),
        ("Ø", "O"),
        ("ø", "o"),
        ("Ð", "D"),
        ("ð", "d"),
        ("Þ", "T"),
        ("þ", "t"),
        ("Đ", "D"),
        ("đ", "d"),
        ("Ħ", "H"),
        ("ħ", "h"),
        ("ı", "i"),
        ("Ł", "L"),
        ("ł", "l"),
        ("Ŋ", "N"),
        ("ŋ", "n"),
        ("Œ", "O"),
        ("œ", "o"),
        ("Ŧ", "T"),
        ("ŧ", "t"),
    ] {
        assert_eq!(chars(input), want, "input {input:?}");
    }
    assert_eq!(chars("Û\t"), "U   ");
}

/// #90's typography table: the issue's named quotes, dashes, ellipsis,
/// bullet, arrows, multiplication sign, and space variants. The quotes,
/// dashes, ellipsis, and bullet are real glyphs already in `term::GLYPHS`
/// (and in all three committed fonts, and in eurlatgr) so they pass through
/// as themselves; the rest have no font slot and fall back to one ASCII
/// character each.
#[test]
fn typography_table_prefers_the_real_glyph_then_ascii() {
    for (input, want) in [
        ("\u{2019}", "\u{2019}"), // ’ right single quote: real glyph
        ("\u{2018}", "\u{2018}"), // ‘ left single quote: real glyph
        ("\u{201b}", "'"),        // ‛ single high-reversed-9: no glyph
        ("\u{2032}", "'"),        // ′ prime: no glyph
        ("\u{201c}", "\u{201c}"), // “ real glyph
        ("\u{201d}", "\u{201d}"), // ” real glyph
        ("\u{201e}", "\""),       // „ double low-9: no glyph
        ("\u{2033}", "\""),       // ″ double prime: no glyph
        ("\u{2013}", "\u{2013}"), // – en dash: real glyph
        ("\u{2014}", "\u{2014}"), // — em dash: real glyph
        ("\u{2010}", "-"),        // ‐ Unicode hyphen: no glyph
        ("\u{2011}", "-"),        // ‑ non-breaking hyphen: no glyph
        ("\u{2015}", "-"),        // ― horizontal bar: no glyph
        ("\u{2026}", "\u{2026}"), // … real glyph
        ("\u{2022}", "\u{2022}"), // • real glyph
        ("\u{2192}", ">"),        // →
        ("\u{2190}", "<"),        // ←
        ("\u{2191}", "^"),        // ↑
        ("\u{2193}", "v"),        // ↓
        ("\u{21d2}", ">"),        // ⇒
        ("\u{d7}", "x"),          // ×
        ("\u{a0}", " "),          // NBSP
        ("\u{2007}", " "),        // figure space
        ("\u{2009}", " "),        // thin space
        ("\u{202f}", " "),        // narrow NBSP
        ("\u{fb01}", "f"),        // ﬁ
        ("\u{fb02}", "l"),        // ﬂ
    ] {
        assert_eq!(chars(input), want, "input {input:?}");
    }
    assert_eq!(
        chars("Calliope\u{2019}s \u{201c}quoted\u{201d} text \u{2014} with an ellipsis\u{2026}"),
        "Calliope\u{2019}s \u{201c}quoted\u{201d} text \u{2014} with an ellipsis\u{2026}"
    );
}

/// #90: zero-width format characters and the BOM carry no column of their
/// own and are dropped (zero cells), not shown as anything. Bidi direction
/// overrides are a different case (next test): those stay visible.
#[test]
fn zero_width_and_bom_are_dropped() {
    for mark in ["\u{ad}", "\u{200b}", "\u{200c}", "\u{200d}", "\u{feff}"] {
        assert_eq!(chars(mark), "", "input {mark:?}");
    }
    assert_eq!(chars("a\u{200d}b"), "ab");
    // A ZWJ emoji sequence: the joiner vanishes, each emoji is unmapped.
    assert_eq!(
        chars("\u{1f468}\u{200d}\u{1f469}"),
        format!("{UNKNOWN}{UNKNOWN}")
    );
}

/// #90: anything with no mapping — CJK, emoji, bidi direction overrides,
/// a standalone combining mark, Latin-1 symbols this table does not cover —
/// becomes the distinct placeholder, never a literal `?`. A `?` the model
/// actually sent passes straight through, so it stays unambiguous.
#[test]
fn unknown_scalars_become_the_placeholder_not_a_literal_question_mark() {
    assert_eq!(chars("?"), "?");
    assert_eq!(chars("a?b?"), "a?b?");
    assert_eq!(chars("🔥"), UNKNOWN.to_string());
    assert_eq!(
        chars("e\u{301}\u{302}"),
        format!("e{UNKNOWN}{UNKNOWN}"),
        "bare combining marks are not a precomposed letter"
    );
    assert_eq!(chars("÷"), UNKNOWN.to_string(), "division sign: no mapping");
    // Bidi direction overrides: Trojan-Source-style marks must stay visible
    // rather than vanish like the zero-width marks above.
    for mark in [
        "\u{200e}", "\u{200f}", "\u{202a}", "\u{202b}", "\u{202c}", "\u{202d}", "\u{202e}",
        "\u{2066}", "\u{2067}", "\u{2068}", "\u{2069}",
    ] {
        assert_eq!(chars(mark), UNKNOWN.to_string(), "bidi {mark:?}");
    }
    // A real `?` next to the placeholder stays tellable apart.
    assert_eq!(chars("?🔥?"), format!("?{UNKNOWN}?"));
}

/// #90: every transliterated or placeholder scalar is exactly one cell, so
/// column alignment never drifts. `ﬁ`/`ﬂ` are the issue's one named
/// exception allowed to widen; this sanitiser chooses not to, so the rule
/// holds with no exception at all.
#[test]
fn every_mapped_scalar_is_exactly_one_cell() {
    let inputs = [
        "\u{2019}", "\u{2018}", "\u{201c}", "\u{201d}", "\u{201b}", "\u{2032}", "\u{201e}",
        "\u{2033}", "\u{2013}", "\u{2014}", "\u{2010}", "\u{2026}", "\u{2022}", "\u{2192}",
        "\u{d7}", "\u{a0}", "\u{fb01}", "\u{fb02}", "Û", "\u{ff}", "🔥", "÷",
    ];
    for input in inputs {
        let mut cells = Vec::new();
        sanitize(input, &mut cells);
        assert_eq!(
            cells.len(),
            1,
            "input {input:?} ({} chars) produced {} cells, not 1",
            input.chars().count(),
            cells.len()
        );
    }
    // A run of them keeps column 1:1 with the input's scalar count.
    let text = "Calliope\u{2019}s \u{201c}quoted\u{201d} text \u{2014} with an ellipsis\u{2026}";
    let mut cells = Vec::new();
    sanitize(text, &mut cells);
    assert_eq!(cells.len(), text.chars().count());
}

#[test]
fn appends_and_uses_white_on_black() {
    let mut cells = vec![Cell::new('Z', C16::Red, C16::Blue)];
    sanitize("A\t", &mut cells);
    assert_eq!(cells[0].ch, 'Z');
    let text: String = cells[1..].iter().map(|cell| cell.ch).collect();
    assert_eq!(text, "A   ");
    assert!(
        cells[1..]
            .iter()
            .all(|cell| cell.fg == C16::White && cell.bg == C16::Black)
    );
}

#[test]
fn ten_megabyte_input_stays_inside_the_cell_set() {
    const N: usize = 10 * 1024 * 1024;
    let mut input = String::new();
    input.push_str("\u{1b}]0;x\u{7}Û\u{9b}\u{7f}\u{202e}\n");
    while input.len() < N {
        input.push('A');
    }
    let mut end = N;
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    input.truncate(end);
    assert!(input.len() + 3 >= N, "input is only {} bytes", input.len());

    let mut cells = Vec::new();
    sanitize(&input, &mut cells);
    assert!(
        cells.len() > N / 2,
        "expected millions of cells, got {}",
        cells.len()
    );
    assert!(cells.iter().all(|cell| {
        let ch = cell.ch;
        ch == '\n' || ch == '?' || ch == UNKNOWN || ('\u{20}'..='\u{7e}').contains(&ch)
    }));
}

#[test]
fn c0_del_and_c1_are_dropped() {
    for cp in 0u32..=0x1f {
        let ch = char::from_u32(cp).unwrap();
        let got = chars(&ch.to_string());
        match ch {
            '\t' => assert_eq!(got, "    ", "tab"),
            '\n' => assert_eq!(got, "\n", "newline"),
            '\r' => assert_eq!(got, "", "cr"),
            _ => assert_eq!(got, "", "C0 U+{cp:04X}"),
        }
    }
    assert_eq!(chars("\u{7f}"), "");
    assert_eq!(chars("a\u{7f}b"), "ab");
    for cp in 0x80u32..=0x9f {
        let ch = char::from_u32(cp).unwrap();
        assert_eq!(chars(&ch.to_string()), "", "C1 U+{cp:04X}");
    }
}

/// #90: every Latin-1 byte either drops (C1, the soft hyphen), becomes a
/// plain space (NBSP), becomes `x` (the multiplication sign), or stays
/// exactly one cell as either an ASCII letter (`accented_latin_becomes_its_base_ascii_letter`
/// checks the specific mapping) or the placeholder. None of them is a
/// literal `?` any more, and none of them produces more or fewer cells than
/// scalars going in.
#[test]
fn every_latin1_byte_is_dropped_mapped_or_the_placeholder() {
    for cp in 0x80u32..=0x9f {
        let ch = char::from_u32(cp).unwrap();
        assert_eq!(chars(&ch.to_string()), "", "C1 U+{cp:04X}");
    }
    for cp in 0xa0u32..=0xff {
        let ch = char::from_u32(cp).unwrap();
        let got = chars(&ch.to_string());
        match cp {
            0xa0 => assert_eq!(got, " ", "NBSP"),
            0xad => assert_eq!(got, "", "soft hyphen"),
            0xd7 => assert_eq!(got, "x", "multiplication sign"),
            _ => {
                assert_eq!(
                    got.chars().count(),
                    1,
                    "U+{cp:04X} must stay exactly one cell, got {got:?}"
                );
                let out = got.chars().next().unwrap();
                assert!(
                    out.is_ascii_alphabetic() || out == UNKNOWN,
                    "U+{cp:04X} produced {out:?}, want an ASCII letter or the placeholder"
                );
            }
        }
    }
    assert_cells_are_safe("Û\u{9b}\u{1b}]0;x\u{7}\u{7f}\u{202e}\u{200d}e\u{301}");
}

#[test]
fn allowlist_parser_rejects_sequences_we_do_not_emit() {
    assert!(check_allowlist(b"\x1b%G").is_ok());
    assert!(check_allowlist(b"\x1b%G\x1b[31mA\x1b[40m ").is_ok());
    assert!(check_allowlist(b"\x1b%G\x1b[2J\x1b[?25l\x1b[0m\x1b[H\x1b[1;1H").is_ok());
    for good in [
        b"\x1b%G\x1b[9;10]\x1b[14;5]".as_slice(),
        b"\x1b%G\x1b[9;1]".as_slice(),
        b"\x1b%G\x1b[9;60]".as_slice(),
        b"\x1b%G\x1b[14;1]".as_slice(),
        b"\x1b%G\x1b[14;59]".as_slice(),
        b"\x1b%G\x1b[9;10H".as_slice(), // still a CUP, not a blank
    ] {
        assert!(check_allowlist(good).is_ok(), "rejected {good:?}");
    }
    assert_eq!(
        ALLOWED_GLYPHS, GLYPHS,
        "S12 must track the term.rs glyph set exactly"
    );
    for ch in ALLOWED_GLYPHS {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        let mut frame = b"\x1b%G".to_vec();
        frame.extend_from_slice(encoded.as_bytes());
        assert!(check_allowlist(&frame).is_ok(), "{frame:?}");
    }
    for bad in [
        b"\x1b[31m".as_slice(),
        b"\x1b%G\x1b[31;41m".as_slice(),
        b"\x1b%G\x1b".as_slice(),
        b"\x1b%G\x9b".as_slice(),
        b"\x1b%G\x1b[101m".as_slice(),
        b"\x1b%G\x1b[38;5;1m".as_slice(),
        b"\x1b%G\x00".as_slice(),
        b"\x1b%G\xc3\x9b".as_slice(),
        b"\x1b%G\x7f".as_slice(),
        b"\x1b%G\xe2\x96\x94".as_slice(), // ▔, in the font but never drawn
        b"\x1b%G\xe2\x96\x89".as_slice(), // ▉, left 7/8
        b"\x1b%G\xe2\x96\x88\xe2\x96".as_slice(), // █ then a cut ▁
        // T61 neighbours of the two console blank sequences.
        b"\x1b%G\x1b[9;x]".as_slice(),
        b"\x1b%G\x1b[9;1x]".as_slice(),
        b"\x1b%G\x1b[9;]".as_slice(),
        b"\x1b%G\x1b[9]".as_slice(),
        b"\x1b%G\x1b[9;0]".as_slice(),
        b"\x1b%G\x1b[9;010]".as_slice(),
        b"\x1b%G\x1b[9;61]".as_slice(),
        b"\x1b%G\x1b[9;99999]".as_slice(),
        b"\x1b%G\x1b[9;10".as_slice(),
        b"\x1b%G\x1b[9;10;5]".as_slice(),
        b"\x1b%G\x1b[9;-1]".as_slice(),
        b"\x1b%G\x1b[14;x]".as_slice(),
        b"\x1b%G\x1b[14;0]".as_slice(),
        b"\x1b%G\x1b[14;61]".as_slice(),
        b"\x1b%G\x1b[8;1]".as_slice(),
        b"\x1b%G\x1b[10;5]".as_slice(),
        b"\x1b%G\x1b[11;200]".as_slice(),
        b"\x1b%G\x1b[12;1]".as_slice(),
        b"\x1b%G\x1b[13]".as_slice(),
        b"\x1b%G\x1b[13;1]".as_slice(),
        b"\x1b%G\x1b[15;1]".as_slice(),
        b"\x1b%G\x1b[19;1]".as_slice(),
        b"\x1b%G\x1b[09;1]".as_slice(),
        b"\x1b%G\x9b9;1]".as_slice(),
    ] {
        assert!(check_allowlist(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn hostile_frame_stays_on_the_allowlist_for_diff_and_full_repaint() {
    let mut text = String::from("\u{1b}]0;x\u{7}\u{1b}[2J\u{9b}31m\u{db}");
    for cp in 0x80u32..=0xff {
        text.push(char::from_u32(cp).expect("latin-1"));
    }
    for cp in 0u32..=0x1f {
        text.push(char::from_u32(cp).expect("c0"));
    }
    text.push('\u{7f}');
    text.push_str("\u{202e}\u{200d}e\u{301}\u{202a}\u{2066}\u{200e}\u{200f}");

    let mut cells = Vec::new();
    sanitize(&text, &mut cells);
    assert!(cells.iter().all(|cell| {
        let ch = cell.ch;
        ch == '\n' || ch == '?' || ch == UNKNOWN || ('\u{20}'..='\u{7e}').contains(&ch)
    }));

    let mut planted = vec![
        Cell::new('\u{1b}', C16::White, C16::Black),
        Cell::new('\u{9b}', C16::White, C16::Black),
        Cell::new('\u{db}', C16::White, C16::Black),
        Cell::new('\u{7f}', C16::White, C16::Black),
    ];
    planted.extend(cells.into_iter().filter(|cell| !cell.is_line_end()));
    for ch in ALLOWED_GLYPHS.iter().copied().chain(['▔']) {
        planted.push(Cell::new(ch, C16::White, C16::Black));
    }

    let cols = 40u16;
    let rows = u16::try_from(planted.len() / 40 + 1).expect("rows");
    let mut grid = Grid::blank(cols, rows);
    for (i, cell) in planted.iter().enumerate() {
        grid.put((i % 40) as u16, (i / 40) as u16, *cell);
    }

    let t0 = Instant::now();
    let mut term = Term::new(
        Vec::new(),
        move || Ok(Size { cols, rows }),
        Duration::from_secs(30),
        t0,
    )
    .expect("size");
    let blank = Grid::blank(cols, rows);
    term.render(&blank, t0).expect("blank");
    term.out_mut().clear();
    term.render(&grid, t0 + Duration::from_millis(1))
        .expect("diff");
    check_allowlist(term.out()).expect("diff path");
    assert!(
        term.out().windows(3).any(|w| w == [0xe2, 0x96, 0x88]),
        "diff path dropped the block glyph"
    );
    for (name, bytes) in [
        ("▀", [0xe2, 0x96, 0x80].as_slice()),
        ("▄", [0xe2, 0x96, 0x84].as_slice()),
        ("·", [0xc2, 0xb7].as_slice()),
        ("▁", [0xe2, 0x96, 0x81].as_slice()),
        ("▇", [0xe2, 0x96, 0x87].as_slice()),
    ] {
        assert!(
            term.out().windows(bytes.len()).any(|w| w == bytes),
            "diff path dropped {name}"
        );
    }
    assert!(
        !term.out().windows(3).any(|w| w == [0xe2, 0x96, 0x94]),
        "upper eighth ▔ leaked into the byte stream"
    );

    term.out_mut().clear();
    term.render(&grid, t0 + Duration::from_secs(30))
        .expect("full");
    check_allowlist(term.out()).expect("full repaint");
    assert!(term.out().windows(4).any(|w| w == b"\x1b[2J"));
    assert!(
        !term.out().windows(3).any(|w| w == [0xe2, 0x96, 0x94]),
        "upper eighth ▔ leaked on full repaint"
    );
}

#[test]
fn llama_text_cannot_produce_the_console_blank_sequences() {
    // Hostile llama text asks to blank, power down, unblank and switch VT.
    let text = "\u{1b}[9;1]\u{1b}[14;1]\u{1b}[13]\u{1b}[12;1]\u{9b}9;1]\u{1b}\u{1b}[9;1]";
    let mut cells = Vec::new();
    sanitize(text, &mut cells);
    assert!(
        cells
            .iter()
            .all(|cell| cell.ch != '\u{1b}' && cell.ch != '\u{9b}')
    );

    let cols = 40u16;
    let rows = 2u16;
    let mut grid = Grid::blank(cols, rows);
    for (i, cell) in cells.iter().filter(|cell| !cell.is_line_end()).enumerate() {
        grid.put((i % 40) as u16, (i / 40) as u16, *cell);
    }
    let count =
        |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).filter(|w| *w == needle).count();

    // Off: llama text alone never reaches ESC [ n ].
    let t0 = Instant::now();
    let mut off = Term::new(
        Vec::new(),
        move || Ok(Size { cols, rows }),
        Duration::from_secs(5),
        t0,
    )
    .expect("size");
    off.render(&grid, t0).expect("start-up");
    off.render(&grid, t0 + Duration::from_secs(5))
        .expect("full");
    check_allowlist(off.out()).expect("off");
    for seq in [b"\x1b[9;".as_slice(), b"\x1b[14;", b"\x1b[13", b"\x1b[12"] {
        assert_eq!(count(off.out(), seq), 0, "{seq:?} in {:?}", off.out());
    }

    // On: the only blank sequences are ours, once, at start-up.
    let mut on = Term::new(
        Vec::new(),
        move || Ok(Size { cols, rows }),
        Duration::from_secs(5),
        t0,
    )
    .expect("size")
    .with_console_blank(ConsoleBlank::from_minutes(10, 15));
    let blank = Grid::blank(cols, rows);
    on.render(&blank, t0).expect("start-up");
    check_allowlist(on.out()).expect("start-up");
    assert!(
        on.out().ends_with(b"\x1b[9;10]\x1b[14;5]"),
        "{:?}",
        on.out()
    );
    on.out_mut().clear();
    on.render(&grid, t0 + Duration::from_millis(1))
        .expect("diff");
    check_allowlist(on.out()).expect("diff");
    on.render(&grid, t0 + Duration::from_secs(5)).expect("full");
    check_allowlist(on.out()).expect("full");
    assert!(count(on.out(), b"\x1b[2J") >= 1, "no full repaint");
    for seq in [b"\x1b[9;".as_slice(), b"\x1b[14;", b"\x1b[13", b"\x1b[12"] {
        assert_eq!(
            count(on.out(), seq),
            0,
            "{seq:?} after start-up in {:?}",
            on.out()
        );
    }
}

fn check_allowlist(bytes: &[u8]) -> Result<(), String> {
    if !bytes.starts_with(b"\x1b%G") {
        return Err(format!("does not start with ESC % G: {}", preview(bytes)));
    }
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0x1b {
            let Some(n) = match_escape(&bytes[i..]) else {
                return Err(format!(
                    "ESC at {i} is not allowlisted: {}",
                    preview(&bytes[i..])
                ));
            };
            i += n;
        } else if b >= 0x80 {
            let Some(n) = match_glyph(&bytes[i..]) else {
                return Err(format!(
                    "byte 0x{b:02X} at {i} is outside the glyph allowlist"
                ));
            };
            if bytes[i..i + n].iter().any(|byte| *byte < 0x80) {
                return Err(format!("glyph at {i} contains a low byte"));
            }
            i += n;
        } else if (0x20..=0x7e).contains(&b) {
            i += 1;
        } else {
            return Err(format!("byte 0x{b:02X} at {i} is not allowlisted"));
        }
    }
    Ok(())
}

fn match_escape(seq: &[u8]) -> Option<usize> {
    const FIXED: [&[u8]; 5] = [b"\x1b%G", b"\x1b[?25l", b"\x1b[2J", b"\x1b[0m", b"\x1b[H"];
    for fixed in FIXED {
        if seq.starts_with(fixed) {
            return Some(fixed.len());
        }
    }
    match_cup(seq)
        .or_else(|| match_sgr(seq))
        .or_else(|| match_console_blank(seq))
}

/// T61: exactly `ESC [ 9 ; n ]` (blank) and `ESC [ 14 ; n ]` (powerdown),
/// n in 1..=60 with no leading zero. No other `ESC [ … ]` console private
/// sequence (13 unblank, 12 switch VT, 10/11 bell, ...) is allowed.
fn match_console_blank(seq: &[u8]) -> Option<usize> {
    let head = if seq.starts_with(b"\x1b[9;") {
        4
    } else if seq.starts_with(b"\x1b[14;") {
        5
    } else {
        return None;
    };
    let (minutes, used) = parse_number(&seq[head..])?;
    if !(1..=60).contains(&minutes) {
        return None;
    }
    let end = head + used;
    if seq.get(end) != Some(&b']') {
        return None;
    }
    Some(end + 1)
}

fn match_cup(seq: &[u8]) -> Option<usize> {
    if seq.len() < 5 || seq[0] != 0x1b || seq[1] != b'[' {
        return None;
    }
    let (row, used) = parse_number(&seq[2..])?;
    if row == 0 {
        return None;
    }
    let semi = 2 + used;
    if seq.get(semi) != Some(&b';') {
        return None;
    }
    let (col, used) = parse_number(&seq[semi + 1..])?;
    if col == 0 {
        return None;
    }
    let end = semi + 1 + used;
    if seq.get(end) != Some(&b'H') {
        return None;
    }
    Some(end + 1)
}

fn match_sgr(seq: &[u8]) -> Option<usize> {
    if seq.len() < 4 || seq[0] != 0x1b || seq[1] != b'[' {
        return None;
    }
    let (code, used) = parse_number(&seq[2..])?;
    let end = 2 + used;
    if seq.get(end) != Some(&b'm') {
        return None;
    }
    let ok = (30..=37).contains(&code) || (90..=97).contains(&code) || (40..=47).contains(&code);
    if !ok {
        return None;
    }
    Some(end + 1)
}

fn parse_number(bytes: &[u8]) -> Option<(u16, usize)> {
    if bytes.is_empty() || !bytes[0].is_ascii_digit() {
        return None;
    }
    if bytes[0] == b'0' && bytes.len() > 1 && bytes[1].is_ascii_digit() {
        return None;
    }
    let mut value: u32 = 0;
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        if i >= 5 {
            return None;
        }
        value = value * 10 + u32::from(bytes[i] - b'0');
        if value > u32::from(u16::MAX) {
            return None;
        }
        i += 1;
    }
    Some((u16::try_from(value).ok()?, i))
}

/// Written out here, not taken from `term::GLYPHS`, so a change to the
/// allowlist has to change this test too.
const ALLOWED_GLYPHS: &[char] = &[
    '█', '▌', '▐', '░', '▒', '▓', '▀', '▄', '·', '…', '≈', '’', '‘', '“', '”', '–', '—', '•',
    '\u{fffd}', '▁', '▂', '▃', '▅', '▆', '▇',
];

fn match_glyph(bytes: &[u8]) -> Option<usize> {
    for ch in ALLOWED_GLYPHS {
        let mut buf = [0u8; 4];
        let glyph = ch.encode_utf8(&mut buf);
        if bytes.starts_with(glyph.as_bytes()) {
            return Some(glyph.len());
        }
    }
    None
}

fn preview(bytes: &[u8]) -> String {
    bytes.iter().take(24).fold(String::new(), |mut acc, byte| {
        acc.push_str(&format!("\\x{byte:02X}"));
        acc
    })
}
