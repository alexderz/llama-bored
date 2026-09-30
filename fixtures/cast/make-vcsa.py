#!/usr/bin/env python3
"""Write synthetic.vcsa: a made-up 10x40 tty11 dump for llama-cast's render
test. No real screen content. The attribute bytes use the 512-glyph layout
(bit 0 = glyph bit 8, fg = bits 1-4, bg = bits 5-7, VGA colour order), as
the kernel stores them with the llama-hack fonts loaded."""
import sys

ROWS, COLS = 10, 40
ANSI_TO_VGA = [0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15]


def attr(fg, bg, glyph):
    return (ANSI_TO_VGA[bg & 7] << 5) | (ANSI_TO_VGA[fg] << 1) | ((glyph >> 8) & 1)


cells = [[(32, attr(7, 0, 32))] * COLS for _ in range(ROWS)]


def put(row, col, text, fg, bg):
    for i, ch in enumerate(text):
        g = ord(ch)
        cells[row][col + i] = (g & 0xFF, attr(fg, bg, g))


put(0, 0, " llama-bored cast fixture ".ljust(COLS), 15, 4)
put(2, 1, "fg:", 7, 0)
for c in range(16):
    cells[2][5 + c] = (ord("#"), attr(c, 0, ord("#")))
put(4, 1, "bg:", 7, 0)
for c in range(8):
    cells[4][5 + c] = (32, attr(7, c, 32))
put(6, 1, "hi:", 7, 0)
# Glyph bit 8 set (the shipped fonts leave 256..511 blank): the cyan
# background shows the colour bits still decode with bit 0 taken.
for i in range(8):
    g = 256 + 40 + i
    cells[6][5 + i] = (g & 0xFF, attr(11, 6, g))
put(8, 1, "GEN 123.4 t/s  CTX 42%", 10, 0)
put(9, 39, "X", 9, 1)

out = bytearray([ROWS, COLS, 0, 0])
for row in cells:
    for ch, at in row:
        out += bytes([ch, at])
sys.stdout.buffer.write(out)
