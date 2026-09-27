//! Display-name sanitiser shared by the watcher and the writer.
//!
//! Printable ASCII, spaces collapsed and trimmed, then truncated.
//! Tab, newline, and CR become a space before other controls are dropped.
//! The ellipsis counts toward `max_chars`. An upstream U+2026 is kept, so a
//! truncated name is a fixed point: `sanitize(sanitize(raw, n), n)` equals
//! `sanitize(raw, n)`. [`sanitize_wire`] always uses the snapshot width.

/// Collapse whitespace, keep printable ASCII and `…`, and truncate to `max_chars`.
///
/// Zero keeps nothing. A name longer than `max_chars` keeps `max_chars - 1`
/// scalars and a trailing `…`. U+2026 already in the input is kept, so
/// `sanitize(sanitize(raw, n), n) == sanitize(raw, n)`.
#[must_use]
pub fn sanitize(raw: &str, max_chars: usize) -> String {
    let mut cleaned = String::new();
    let mut pending_space = false;
    for c in raw.chars() {
        let c = if matches!(c, '\t' | '\n' | '\r') {
            ' '
        } else {
            c
        };
        if c != '\u{2026}' && !is_printable_ascii(c) {
            continue;
        }
        if c == ' ' {
            if !cleaned.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            cleaned.push(' ');
            pending_space = false;
        }
        cleaned.push(c);
    }
    truncate_name(&cleaned, max_chars)
}

/// Sanitise a name that will be stored in the snapshot.
///
/// The width is [`crate::wire::CANONICAL_NAME_CHARS`] (12, including `…`),
/// not the watcher config. A non-empty result passes [`crate::wire::validate`].
#[must_use]
pub fn sanitize_wire(raw: &str) -> String {
    sanitize(raw, crate::wire::CANONICAL_NAME_CHARS)
}

fn is_printable_ascii(c: char) -> bool {
    c.is_ascii() && !c.is_ascii_control()
}

fn truncate_name(cleaned: &str, max_chars: usize) -> String {
    if cleaned.chars().count() <= max_chars {
        return cleaned.to_owned();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut out: String = cleaned.chars().take(max_chars - 1).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_printable_ascii_collapses_space_and_truncates() {
        let cases = [
            ("Qwen 35B", 12, "Qwen 35B"),
            ("123456789012", 12, "123456789012"),
            ("1234567890123", 12, "12345678901…"),
            ("hello   world", 20, "hello world"),
            ("  hello   world  ", 20, "hello world"),
            ("Qwen 三五B", 20, "Qwen B"),
            ("héllo", 20, "hllo"),
            ("a\tb\nc", 20, "a b c"),
            ("a \t b", 20, "a b"),
            ("Qwen\t35B", 20, "Qwen 35B"),
            ("Qwen\r\n35B", 20, "Qwen 35B"),
            ("~", 12, "~"),
            ("\u{7f}x", 12, "x"),
            ("", 12, ""),
            ("🔥", 12, ""),
            ("abcdefghij", 5, "abcd…"),
            ("abc", 0, ""),
            ("abcdef", 1, "…"),
            ("Qwen-35B_a3b", 20, "Qwen-35B_a3b"),
            ("ab", 2, "ab"),
            ("abc", 2, "a…"),
        ];
        for (raw, max, expect) in cases {
            assert_eq!(sanitize(raw, max), expect, "raw={raw:?} max={max}");
        }
    }

    #[test]
    fn truncated_name_is_a_fixed_point() {
        let once = sanitize("1234567890123", 12);
        assert_eq!(once, "12345678901…");
        assert_eq!(sanitize(&once, 12), once);
        assert_eq!(sanitize("a…b", 12), "a…b");
        assert_eq!(sanitize("héllo…", 12), "hllo…");
    }
}
