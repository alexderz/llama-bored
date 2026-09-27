//! Llama text to cells. Bytes are not emitted here.

use super::grid::{C16, Cell};

/// Append sanitised cells for `input`.
///
/// Printable ASCII is kept. `\n` appends an end-of-line marker and resets the
/// tab column. `\t` expands to the next multiple of 4 columns (1 to 4 spaces).
/// `\r` is dropped and does not move the column. Other C0, DEL, and C1 are
/// dropped. Every remaining scalar (Latin-1 included) becomes `?` and counts
/// as one column. Each call starts at column 0; cells already in `out` stay.
pub fn sanitize(input: &str, out: &mut Vec<Cell>) {
    let mut col = 0usize;
    for ch in input.chars() {
        match ch {
            '\n' => {
                push(out, '\n');
                col = 0;
            }
            '\t' => {
                let spaces = 4 - (col % 4);
                for _ in 0..spaces {
                    push(out, ' ');
                    col += 1;
                }
            }
            '\r' => {}
            c if ('\u{20}'..='\u{7e}').contains(&c) => {
                push(out, c);
                col += 1;
            }
            c if c.is_control() => {}
            _ => {
                push(out, '?');
                col += 1;
            }
        }
    }
}

fn push(out: &mut Vec<Cell>, ch: char) {
    out.push(Cell::new(ch, C16::White, C16::Black));
}
