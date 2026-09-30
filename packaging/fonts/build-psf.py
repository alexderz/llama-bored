#!/usr/bin/env python3
"""Build the tty11 console font (PSF2 with a Unicode table) from a TTF.

Usage:
  build-psf.py TTF WxH OUT          write OUT
  build-psf.py --check TTF WxH OUT  rebuild in memory, compare with OUT byte
                                    for byte, exit 1 on any difference
  build-psf.py --self-test          rebuild llama-hack-12x24.psfu and
                                    llama-hack-12x22.psfu from Hack Regular
                                    and compare each; skip (exit 0 with a
                                    note) when Pillow or the TTF is absent

The committed fonts are built with:

  packaging/fonts/build-psf.py \\
    /usr/share/fonts/source-foundry-hack-fonts/Hack-Regular.ttf 12x24 \\
    packaging/fonts/llama-hack-12x24.psfu
  packaging/fonts/build-psf.py \\
    /usr/share/fonts/source-foundry-hack-fonts/Hack-Regular.ttf 12x22 \\
    packaging/fonts/llama-hack-12x22.psfu

12x22 is `[tty] font = "12x22"`: two more rows on a 1080-line screen
(160x49 at 1920x1080 instead of 160x45).

Reference inputs (2026-09-25): Hack 3.003
(source-foundry-hack-fonts-3.003-7.fc44), Pillow 12.3.0, FreeType 2.14.3.
Hack is MIT with Bitstream Vera portions; see LICENSE.Hack.

Layout:
- ASCII (U+0020-007E) and Latin-1 (U+00A0-00FF) sit at their own slots.
  setfont wants slot 32 blank: the console erases with it.
- Every other glyph fills slots 0-31, then 0x7F-0x9F, then 0x100 upward, in
  the order of EXTRA below.
- Block elements U+2580-2595 are drawn as exact rectangles, so the eighth
  blocks the tty chart uses tile a cell with no gaps. Other glyphs are
  rendered from the TTF at the largest pixel size whose ascent + descent fits
  the cell height and whose advance fits the width, then thresholded.

The output depends only on the TTF bytes, the cell size and the Pillow /
FreeType versions: no timestamps, no dict ordering, no randomness.
"""

import hashlib
import os
import struct
import sys

PSF2_MAGIC = 0x864AB572
SLOTS = 512

# Non-Latin-1 glyphs, in slot order. Adding one moves every glyph after it.
EXTRA = (
    [chr(c) for c in range(0x2580, 0x25A0)]
    + list("─│┌┐└┘├┤┬┴┼═║•…–—‘’“”�")
)

HERE = os.path.dirname(os.path.abspath(__file__))
SELF_TEST_TTF = "/usr/share/fonts/source-foundry-hack-fonts/Hack-Regular.ttf"
SELF_TEST_TTF_SHA256 = "15f55cc0c85a2988d2b4b3a8cdb5d77fdfbaf319e1bb5309d725db9818fb7125"
# Every committed font, rebuilt and compared by --self-test.
SELF_TEST_FONTS = (
    ((12, 24), os.path.join(HERE, "llama-hack-12x24.psfu")),
    ((12, 22), os.path.join(HERE, "llama-hack-12x22.psfu")),
)


def slot_table():
    """Glyph for each of the 512 slots, or None for an empty slot."""
    glyphs = [None] * SLOTS
    for c in range(0x20, 0x7F):
        glyphs[c] = chr(c)
    for c in range(0xA0, 0x100):
        glyphs[c] = chr(c)
    free = [i for i in range(SLOTS) if glyphs[i] is None]
    if len(EXTRA) > len(free):
        raise SystemExit(f"build-psf: {len(EXTRA)} extra glyphs, {len(free)} free slots")
    for ch, i in zip(EXTRA, free):
        glyphs[i] = ch
    return glyphs


def pick_font(ImageFont, ttf, width, height):
    size = height
    while size > 4:
        font = ImageFont.truetype(ttf, size)
        asc, desc = font.getmetrics()
        if asc + desc <= height and font.getlength("M") <= width:
            return font, size
        size -= 1
    raise SystemExit(f"build-psf: no size of {ttf} fits {width}x{height}")


def block_image(Image, ImageDraw, ch, width, height):
    """Exact-cell block element, or None when `ch` is not one we draw."""
    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    c = ord(ch)

    def fill(x0, y0, x1, y1):
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=1)

    if c == 0x2580:  # upper half
        fill(0, 0, width, height // 2)
    elif 0x2581 <= c <= 0x2588:  # lower n/8, n = 1..8
        n = c - 0x2580
        fill(0, height - round(height * n / 8), width, height)
    elif 0x2589 <= c <= 0x258F:  # left n/8, n = 7..1
        n = 0x2590 - c
        fill(0, 0, round(width * n / 8), height)
    elif c == 0x2590:  # right half
        fill(width // 2, 0, width, height)
    elif c in (0x2591, 0x2592, 0x2593):  # light / medium / dark shade
        for y in range(height):
            for x in range(width):
                if c == 0x2592:
                    on = (x + y) % 2 == 0
                else:
                    k = (x + 2 * y) % 4
                    on = k == 0 if c == 0x2591 else k != 0
                if on:
                    img.putpixel((x, y), 1)
    elif c == 0x2594:  # upper 1/8
        fill(0, 0, width, max(1, height // 8))
    elif c == 0x2595:  # right 1/8
        fill(width - max(1, width // 8), 0, width, height)
    else:
        return None
    return img


def build(ttf, width, height):
    """PSF2 bytes. Imports Pillow here so --self-test can skip without it."""
    from PIL import Image, ImageDraw, ImageFont

    glyphs = slot_table()
    font, size = pick_font(ImageFont, ttf, width, height)
    asc, desc = font.getmetrics()
    top = (height - (asc + desc)) // 2
    left = int(round((width - font.getlength("M")) / 2))

    def text_image(ch):
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).text((left, top), ch, font=font, fill=255)
        return img.point(lambda v: 1 if v >= 110 else 0, mode="1")

    rowbytes = (width + 7) // 8
    data = bytearray()
    for ch in glyphs:
        img = None
        if ch is not None:
            img = block_image(Image, ImageDraw, ch, width, height) or text_image(ch)
        for y in range(height):
            bits = 0
            for x in range(width):
                if img is not None and img.getpixel((x, y)):
                    bits |= 1 << (rowbytes * 8 - 1 - x)
            data += bits.to_bytes(rowbytes, "big")

    table = bytearray()
    for ch in glyphs:
        if ch is not None:
            table += ch.encode("utf-8")
        table += b"\xff"

    # magic, version 0, header size 32, flags 1 (has Unicode table),
    # glyph count, bytes per glyph, height, width.
    header = struct.pack("<8I", PSF2_MAGIC, 0, 32, 1, SLOTS, rowbytes * height, height, width)
    count = sum(g is not None for g in glyphs)
    return bytes(header + data + table), size, count


def parse_cell(text):
    try:
        w, h = text.lower().split("x")
        width, height = int(w), int(h)
    except ValueError:
        raise SystemExit(f"build-psf: cell size must be WxH, got {text!r}")
    if not (4 <= width <= 32 and 4 <= height <= 64):
        raise SystemExit(f"build-psf: cell size {width}x{height} is out of range")
    return width, height


def check(ttf, width, height, out):
    got, _, _ = build(ttf, width, height)
    with open(out, "rb") as f:
        want = f.read()
    if got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print(
            f"build-psf: rebuilt {width}x{height} from {ttf} differs from {out} "
            f"(sizes {len(got)} vs {len(want)}, first difference at byte {first})",
            file=sys.stderr,
        )
        return 1
    print(f"build-psf: {out} matches a rebuild from {ttf} ({len(got)} bytes)")
    return 0


def self_test():
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("build-psf self-test: SKIP, Pillow is not installed; the committed PSF was not rebuilt")
        return 0
    ttf = os.environ.get("LLAMA_HACK_TTF", SELF_TEST_TTF)
    if not os.path.isfile(ttf):
        print(f"build-psf self-test: SKIP, {ttf} is absent; the committed PSF was not rebuilt")
        return 0
    with open(ttf, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SELF_TEST_TTF_SHA256:
        print(
            f"build-psf self-test: SKIP, {ttf} is not the reference Hack 3.003 "
            f"(sha256 {digest}); the committed PSF was not rebuilt"
        )
        return 0
    failed = 0
    for (width, height), out in SELF_TEST_FONTS:
        if not os.path.isfile(out):
            print(f"build-psf self-test: {out} is missing", file=sys.stderr)
            failed = 1
            continue
        failed |= check(ttf, width, height, out)
    return failed


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 4 and argv[0] == "--check":
        width, height = parse_cell(argv[2])
        return check(argv[1], width, height, argv[3])
    if len(argv) == 3 and not argv[0].startswith("-"):
        width, height = parse_cell(argv[1])
        data, size, count = build(argv[0], width, height)
        tmp = argv[2] + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, argv[2])
        print(f"{argv[2]}: {width}x{height}, {size}px, {count} glyphs")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
