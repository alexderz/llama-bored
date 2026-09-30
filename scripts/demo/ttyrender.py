import struct, zlib, sys
from PIL import Image
FONT = open('/usr/local/share/llama-bored/llama-hack-12x24.psfu','rb').read()
_, _, hdr, _, NG, CB, GH, GW = struct.unpack('<8I', FONT[:32])
PAL = [tuple(int(h[i:i+2],16) for i in (0,2,4)) for h in
       '000000 aa0000 00aa00 aa5500 0000aa aa00aa 00aaaa aaaaaa 555555 ff5555 55ff55 ffff55 5555ff ff55ff 55ffff ffffff'.split()]
ROWS, COLS = 60, 286
V2A = [(v & 0b1010) | ((v & 1) << 2) | ((v & 4) >> 2) for v in range(16)]
glyph_cache = {}
def glyph(i):
    g = glyph_cache.get(i)
    if g is None:
        data = FONT[hdr + i*CB: hdr + (i+1)*CB]
        rb = (GW + 7)//8
        g = Image.new('1', (GW, GH))
        px = g.load()
        for y in range(GH):
            row = int.from_bytes(data[y*rb:(y+1)*rb], 'big')
            for x in range(GW):
                if row >> (rb*8 - 1 - x) & 1: px[x, y] = 1
        glyph_cache[i] = g
    return g
def render(vcsa, hi512=True):
    img = Image.new('RGB', (COLS*GW, ROWS*GH))
    cells = vcsa[4:]
    for r in range(ROWS):
        for c in range(COLS):
            k = 2*(r*COLS + c)
            ch, at = cells[k], cells[k+1]
            fg, bg = at & 0x0f, (at >> 4) & 0x07
            if hi512 and NG == 512:
                # 512-glyph font: vt shifts the attribute left one bit and
                # bit 0 carries glyph bit 8 (vc_hi_font_mask == 0x100).
                ch |= (at & 1) << 8
                fg, bg = (at >> 1) & 0x0f, (at >> 5) & 0x07
            fg, bg = V2A[fg], V2A[bg]
            x, y = c*GW, r*GH
            if bg: img.paste(PAL[bg], (x, y, x+GW, y+GH))
            if ch != 32:
                img.paste(PAL[fg], (x, y), glyph(ch))
    return img
def frames(path):
    d = open(path, 'rb').read(); p = 0
    while p + 13 <= len(d):
        kind = d[p:p+1]; ts, n = struct.unpack('<dI', d[p+1:p+13]); p += 13
        if p + n > len(d): break
        yield kind, ts, zlib.decompress(d[p:p+n]); p += n
if __name__ == '__main__':
    last = None
    for kind, ts, blob in frames(sys.argv[1]):
        if kind == b'V': last = blob
    print(len(last), last[:4])
    render(last).save(sys.argv[2])
