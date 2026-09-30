#!/usr/bin/env python3
"""Procedural top-down scene of an AIO pump head on a motherboard (Pillow only)."""
import json, math, os, random
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 960, 720
CX, CY, LCD_D = 440, 360, 320
LCD_R = LCD_D // 2
S = 2  # supersample for shape drawing
random.seed(7)

def sc(v): return int(round(v * S))

def radial(size, center, radius, inner=255, outer=0):
    """L image: `inner` at center fading to `outer` at radius (linear), clamped."""
    g = Image.radial_gradient("L")  # 0 centre -> 255 edge at r=128
    g = ImageOps.invert(g)          # 255 centre -> 0 edge
    d = int(radius * 2)
    g = g.resize((d, d), Image.BILINEAR)
    if inner != 255 or outer != 0:
        g = g.point(lambda v: int(outer + (inner - outer) * v / 255))
    im = Image.new("L", size, outer)
    im.paste(g, (int(center[0] - radius), int(center[1] - radius)))
    return im

def linear(size, angle_deg, a=0, b=255):
    """L image linear gradient along angle."""
    g = Image.linear_gradient("L")  # top 0 -> bottom 255
    big = int(math.hypot(*size)) + 4
    g = g.resize((big, big), Image.BILINEAR).rotate(angle_deg, resample=Image.BILINEAR)
    g = g.crop(((big - size[0]) // 2, (big - size[1]) // 2, (big - size[0]) // 2 + size[0], (big - size[1]) // 2 + size[1]))
    return g.point(lambda v: int(a + (b - a) * v / 255))

def tint(base, color, mask):
    """Composite flat `color` over `base` using L mask."""
    layer = Image.new("RGB", base.size, color)
    return Image.composite(layer, base, mask)

def add(base, color, mask):
    """Additive light: base + color*mask."""
    layer = Image.new("RGB", base.size, color)
    layer = Image.composite(layer, Image.new("RGB", base.size, (0, 0, 0)), mask)
    return ImageChops.add(base, layer)

def bezier(pts, n=60):
    out = []
    for i in range(n + 1):
        t = i / n
        p = list(pts)
        while len(p) > 1:
            p = [((1 - t) * p[k][0] + t * p[k + 1][0], (1 - t) * p[k][1] + t * p[k + 1][1]) for k in range(len(p) - 1)]
        out.append(p[0])
    return out

SW, SH = W * S, H * S
# ---------------------------------------------------------------- motherboard
pcb = Image.new("RGB", (SW, SH), (12, 14, 18))
d = ImageDraw.Draw(pcb)

# subtle PCB stripes (black/dark-grey diagonal panel like the reference board)
for x in range(-SH, SW, sc(46)):
    d.polygon([(x, 0), (x + sc(18), 0), (x + sc(18) + SH, SH), (x + SH, SH)], fill=(16, 18, 22))

# traces
for _ in range(90):
    x0 = random.randint(0, SW); y0 = random.randint(0, SH)
    L = random.randint(sc(40), sc(220))
    ang = random.choice([0, 90, 45, 135])
    x1 = x0 + L * math.cos(math.radians(ang)); y1 = y0 - L * math.sin(math.radians(ang))
    d.line([(x0, y0), (x1, y1)], fill=(26, 30, 36), width=sc(1.5))

# socket / CPU retention frame under the cap (mostly hidden)
d.rectangle([sc(300), sc(215), sc(590), sc(505)], outline=(40, 44, 50), width=sc(3), fill=(18, 20, 24))
for (x, y) in [(305, 220), (585, 220), (305, 500), (585, 500)]:
    d.ellipse([sc(x - 9), sc(y - 9), sc(x + 9), sc(y + 9)], fill=(58, 62, 68))
    d.ellipse([sc(x - 4), sc(y - 4), sc(x + 4), sc(y + 4)], fill=(22, 24, 28))

# VRM heatsink top-left: finned block
def heatsink(x, y, w, h, fins, vertical=True):
    d.rectangle([sc(x), sc(y), sc(x + w), sc(y + h)], fill=(44, 48, 54))
    n = fins
    for i in range(n):
        if vertical:
            fx = x + (i + 0.5) * w / n
            d.rectangle([sc(fx - w / n * 0.28), sc(y + 4), sc(fx + w / n * 0.28), sc(y + h - 4)], fill=(70, 76, 84))
            d.line([(sc(fx - w / n * 0.28), sc(y + 4)), (sc(fx - w / n * 0.28), sc(y + h - 4))], fill=(110, 118, 128), width=sc(1))
        else:
            fy = y + (i + 0.5) * h / n
            d.rectangle([sc(x + 4), sc(fy - h / n * 0.28), sc(x + w - 4), sc(fy + h / n * 0.28)], fill=(70, 76, 84))
            d.line([(sc(x + 4), sc(fy - h / n * 0.28)), (sc(x + w - 4), sc(fy - h / n * 0.28))], fill=(110, 118, 128), width=sc(1))

heatsink(70, 130, 130, 470, 11, vertical=True)     # left VRM
heatsink(250, 40, 420, 90, 16, vertical=False)     # top VRM (horizontal fins)
# top heatsink: make the fins vertical strips instead (looks like the reference)
d.rectangle([sc(250), sc(40), sc(670), sc(130)], fill=(44, 48, 54))
for i in range(22):
    fx = 250 + (i + 0.5) * 420 / 22
    d.rectangle([sc(fx - 5), sc(46), sc(fx + 5), sc(124)], fill=(72, 78, 86))
    d.line([(sc(fx - 5), sc(46)), (sc(fx - 5), sc(124))], fill=(112, 120, 130), width=sc(1))

# RAM slots on the right: 4 slots, two populated with dark sticks
for i in range(4):
    x = 700 + i * 44
    d.rectangle([sc(x), sc(150), sc(x + 16), sc(600)], fill=(24, 26, 30), outline=(52, 56, 62), width=sc(1))
    for y in range(160, 590, 8):
        d.line([(sc(x + 3), sc(y)), (sc(x + 13), sc(y))], fill=(38, 40, 46), width=sc(1))
    # latches
    d.rectangle([sc(x - 4), sc(140), sc(x + 20), sc(158)], fill=(60, 64, 70))
    d.rectangle([sc(x - 4), sc(596), sc(x + 20), sc(614)], fill=(60, 64, 70))
for i in (1, 3):
    x = 700 + i * 44
    d.rectangle([sc(x - 6), sc(120), sc(x + 22), sc(632)], fill=(30, 32, 36))
    d.rectangle([sc(x - 2), sc(128), sc(x + 18), sc(624)], fill=(52, 55, 60))  # heatspreader
    for y in range(140, 620, 24):
        d.line([(sc(x - 2), sc(y)), (sc(x + 18), sc(y + 10))], fill=(70, 74, 80), width=sc(2))

# capacitors and small chips
for (x, y) in [(230, 640), (262, 640), (294, 640), (326, 640), (620, 200), (650, 200), (620, 560), (650, 560), (210, 170), (210, 200)]:
    d.ellipse([sc(x - 9), sc(y - 9), sc(x + 9), sc(y + 9)], fill=(34, 36, 42), outline=(66, 70, 78), width=sc(1))
    d.ellipse([sc(x - 5), sc(y - 5), sc(x + 5), sc(y + 5)], fill=(50, 52, 58))
for (x, y, w, h) in [(380, 600, 60, 40), (470, 610, 90, 30), (560, 640, 40, 40), (120, 640, 70, 26)]:
    d.rectangle([sc(x), sc(y), sc(x + w), sc(y + h)], fill=(20, 22, 26), outline=(48, 50, 56), width=sc(1))
    for k in range(int(w / 6)):
        d.line([(sc(x + 3 + k * 6), sc(y - 3)), (sc(x + 3 + k * 6), sc(y))], fill=(80, 84, 90), width=sc(1))
        d.line([(sc(x + 3 + k * 6), sc(y + h)), (sc(x + 3 + k * 6), sc(y + h + 3))], fill=(80, 84, 90), width=sc(1))

# a few cables at the bottom-right
for k in range(6):
    pts = bezier([(sc(560 + k * 8), sc(720)), (sc(600 + k * 8), sc(600)), (sc(760 + k * 6), sc(690)), (sc(960), sc(650 + k * 6))])
    d.line(pts, fill=(22 + k, 24 + k, 28 + k), width=sc(7))
    d.line(pts, fill=(36 + k, 38 + k, 42 + k), width=sc(2))

# ------------------------------------------------ lighting on the motherboard
pcb = pcb.resize((W, H), Image.LANCZOS)
pcb = pcb.filter(ImageFilter.GaussianBlur(2.2))  # shallow depth of field
# teal key light from top-left, magenta from bottom-right (multiplicative shading)
teal = linear((W, H), 45, 150, 40)   # bright top-left
shade = Image.new("RGB", (W, H), (255, 255, 255))
shade = Image.composite(Image.new("RGB", (W, H), (120, 220, 235)), Image.new("RGB", (W, H), (150, 60, 170)), linear((W, H), 45, 255, 0))
pcb = ImageChops.multiply(pcb, shade)
pcb = ImageChops.add(pcb, ImageChops.multiply(pcb, Image.new("RGB", (W, H), (60, 60, 60))))  # lift
# cap-centred vignette so the screen is the hero
vig = radial((W, H), (CX, CY), 640, inner=255, outer=20)
vig = vig.point(lambda v: int(18 + 237 * (v / 255) ** 1.6))
pcb = ImageChops.multiply(pcb, Image.merge("RGB", (vig, vig, vig)))
# colored glows: teal top-left, magenta bottom-right
pcb = add(pcb, (0, 48, 56), radial((W, H), (60, 60), 520, 255, 0).point(lambda v: int((v / 255) ** 2.2 * 255)))
pcb = add(pcb, (80, 10, 70), radial((W, H), (930, 700), 520, 255, 0).point(lambda v: int((v / 255) ** 2.2 * 255)))

# ---------------------------------------------------------------- tubes
tubes = Image.new("RGB", (SW, SH), (0, 0, 0))
tmask = Image.new("L", (SW, SH), 0)
td = ImageDraw.Draw(tubes); tm = ImageDraw.Draw(tmask)
paths = [
    bezier([(CX + 120, CY + 40), (CX + 330, CY + 95), (CX + 340, CY - 120), (W + 40, -60)]),
    bezier([(CX + 120, CY + 90), (CX + 380, CY + 165), (CX + 400, CY - 90), (W + 90, -20)]),
]
TW = 40
for p in paths:
    pp = [(sc(x), sc(y)) for x, y in p]
    tm.line(pp, fill=255, width=sc(TW), joint="curve")
    td.line(pp, fill=(16, 17, 19), width=sc(TW), joint="curve")
    # cylindrical shading: brighter stripe offset toward the light (up-left)
    for k, (w, col) in enumerate([(TW * 0.72, (30, 32, 36)), (TW * 0.45, (46, 50, 56)), (TW * 0.2, (64, 70, 78))]):
        off = -(TW * 0.5 - w * 0.5) * 0.55
        q = [(x + sc(off * 0.7), y + sc(off * 0.7)) for x, y in pp]
        td.line(q, fill=col, width=sc(w), joint="curve")
tubes = tubes.resize((W, H), Image.LANCZOS)
tmask = tmask.resize((W, H), Image.LANCZOS)
# braid texture: diagonal cross-hatch clipped to the tube mask
braid = Image.new("L", (W, H), 0)
bd = ImageDraw.Draw(braid)
for x in range(-H, W + H, 6):
    bd.line([(x, 0), (x + H, H)], fill=60, width=1)
    bd.line([(x + H, 0), (x, H)], fill=60, width=1)
braid = ImageChops.multiply(braid, tmask).filter(ImageFilter.GaussianBlur(0.4))
tubes = add(tubes, (60, 62, 70), braid)
# tint tubes with teal/magenta rim
tubes = ImageChops.multiply(tubes, shade.point(lambda v: min(255, v + 60)))
tubes = tubes.filter(ImageFilter.GaussianBlur(0.6))
# tube shadow onto board
tshadow = tmask.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.75))
tshadow = ImageChops.offset(tshadow, 10, 14)
pcb = ImageChops.multiply(pcb, Image.merge("RGB", [ImageOps.invert(tshadow)] * 3))
scene = Image.composite(tubes, pcb, tmask)

# ---------------------------------------------------------------- pump cap
CAP_R = 218
# fittings (short cylinders emerging under the cap toward the tubes)
fit = Image.new("RGB", (SW, SH), (0, 0, 0)); fmask = Image.new("L", (SW, SH), 0)
fd = ImageDraw.Draw(fit); fm = ImageDraw.Draw(fmask)
for path in paths:
    # fitting follows the tube direction where it emerges from under the cap
    pts = [(x, y) for x, y in path if 150 <= math.hypot(x - CX, y - CY) <= 236]
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    ang = math.atan2(y1 - y0, x1 - x0); nx, ny = -math.sin(ang), math.cos(ang)
    fm.line([(sc(x0), sc(y0)), (sc(x1), sc(y1))], fill=255, width=sc(46))
    fd.line([(sc(x0), sc(y0)), (sc(x1), sc(y1))], fill=(20, 21, 24), width=sc(46))
    fd.line([(sc(x0 - nx * 9), sc(y0 - ny * 9)), (sc(x1 - nx * 9), sc(y1 - ny * 9))], fill=(46, 48, 54), width=sc(14))
    fd.line([(sc(x0 - nx * 14), sc(y0 - ny * 14)), (sc(x1 - nx * 14), sc(y1 - ny * 14))], fill=(84, 90, 100), width=sc(3))
    # collar ring at the tube end
    for k in (0.55, 0.8):
        px, py = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
        fd.line([(sc(px + nx * 23), sc(py + ny * 23)), (sc(px - nx * 23), sc(py - ny * 23))], fill=(60, 64, 72), width=sc(3))
fit = fit.resize((W, H), Image.LANCZOS); fmask = fmask.resize((W, H), Image.LANCZOS)
scene = Image.composite(fit, scene, fmask)

# drop shadow
sh = Image.new("L", (W, H), 0)
ImageDraw.Draw(sh).ellipse([CX - CAP_R - 8, CY - CAP_R - 8, CX + CAP_R + 8, CY + CAP_R + 8], fill=200)
sh = ImageChops.offset(sh.filter(ImageFilter.GaussianBlur(22)), 14, 20)
scene = ImageChops.multiply(scene, Image.merge("RGB", [ImageOps.invert(sh)] * 3))

# cap body: matte black with soft top-left lift
cap = Image.new("RGB", (SW, SH), (0, 0, 0)); cmask = Image.new("L", (SW, SH), 0)
cd = ImageDraw.Draw(cap); cm = ImageDraw.Draw(cmask)
cm.ellipse([sc(CX - CAP_R), sc(CY - CAP_R), sc(CX + CAP_R), sc(CY + CAP_R)], fill=255)
cd.ellipse([sc(CX - CAP_R), sc(CY - CAP_R), sc(CX + CAP_R), sc(CY + CAP_R)], fill=(22, 23, 26))
cap = cap.resize((W, H), Image.LANCZOS); cmask = cmask.resize((W, H), Image.LANCZOS)
# surface shading: gentle gradient (lighter up-left), rim glow rings
capshade = linear((W, H), 45, 255, 150)
cap = ImageChops.multiply(cap, Image.merge("RGB", [capshade] * 3))
cap = add(cap, (14, 30, 34), radial((W, H), (CX - 120, CY - 120), 260, 255, 0))       # teal lift
cap = add(cap, (24, 6, 22), radial((W, H), (CX + 150, CY + 150), 220, 255, 0))       # magenta lift
# bevel: outer edge ring (bright arc top-left teal, magenta bottom-right)
bev = Image.new("RGB", (SW, SH), (0, 0, 0)); bm = Image.new("L", (SW, SH), 0)
bd_ = ImageDraw.Draw(bev); bmd = ImageDraw.Draw(bm)
box = [sc(CX - CAP_R + 1), sc(CY - CAP_R + 1), sc(CX + CAP_R - 1), sc(CY + CAP_R - 1)]
bmd.arc(box, 150, 330, fill=255, width=sc(3))
bd_.arc(box, 150, 330, fill=(120, 200, 210), width=sc(3))
bmd.arc(box, 330, 150, fill=140, width=sc(2))
bd_.arc(box, 330, 150, fill=(200, 80, 170), width=sc(2))
# inner bevel step (a faint ring a bit inside the edge)
box2 = [sc(CX - CAP_R + 14), sc(CY - CAP_R + 14), sc(CX + CAP_R - 14), sc(CY + CAP_R - 14)]
bmd.ellipse(box2, outline=70, width=sc(1.5))
bd_.ellipse(box2, outline=(90, 96, 104), width=sc(1.5))
bev = bev.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
bm = bm.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
cap = Image.composite(bev, cap, bm)

scene = Image.composite(cap, scene, cmask)

# bezel: near-black ring around the LCD, glass edge highlight
GL_R = LCD_R + 16
bz = Image.new("RGB", (SW, SH), (0, 0, 0)); bzm = Image.new("L", (SW, SH), 0)
zd = ImageDraw.Draw(bz); zm = ImageDraw.Draw(bzm)
gbox = [sc(CX - GL_R), sc(CY - GL_R), sc(CX + GL_R), sc(CY + GL_R)]
zm.ellipse(gbox, fill=255); zd.ellipse(gbox, fill=(6, 7, 9))
zd.ellipse(gbox, outline=(40, 44, 50), width=sc(1.2))
zd.arc(gbox, 170, 320, fill=(110, 160, 170), width=sc(1.2))
zd.arc(gbox, 0, 120, fill=(120, 60, 110), width=sc(1))
bz = bz.resize((W, H), Image.LANCZOS); bzm = bzm.resize((W, H), Image.LANCZOS)
scene = Image.composite(bz, scene, bzm)

# ---------------------------------------------------------------- LCD hole
hole = Image.new("L", (W * 4, H * 4), 255)
ImageDraw.Draw(hole).ellipse([(CX - LCD_R) * 4, (CY - LCD_R) * 4, (CX + LCD_R) * 4, (CY + LCD_R) * 4], fill=0)
alpha = hole.resize((W, H), Image.BOX)
out = scene.convert("RGBA"); out.putalpha(alpha)
out.save(os.path.join(OUT, "kraken-scene.png"), optimize=True)

# ---------------------------------------------------------------- glass overlay
glass = Image.new("RGBA", (W, H), (0, 0, 0, 0))
g = Image.new("L", (W * 2, H * 2), 0); gd = ImageDraw.Draw(g)
# broad soft diagonal sheen across the upper-left half of the disc
band = Image.new("L", (W * 2, H * 2), 0)
ImageDraw.Draw(band).polygon([(0, 0), (W * 2, 0), (W * 2, H * 0.55), (0, H * 1.75)], fill=255)
band = band.filter(ImageFilter.GaussianBlur(90))
sheen = ImageChops.multiply(band, linear((W * 2, H * 2), 45, 255, 0))
g = sheen.point(lambda v: int(v * 0.08))
# specular arc near the top-left edge
spec = Image.new("L", (W * 2, H * 2), 0)
ImageDraw.Draw(spec).arc([(CX - LCD_R + 10) * 2, (CY - LCD_R + 10) * 2, (CX + LCD_R - 10) * 2, (CY + LCD_R - 10) * 2], 195, 285, fill=255, width=14)
spec = spec.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.12))
g = ImageChops.add(g, spec)
# clip to LCD disc (alpha 0 outside)
disc = Image.new("L", (W * 2, H * 2), 0)
ImageDraw.Draw(disc).ellipse([(CX - LCD_R) * 2, (CY - LCD_R) * 2, (CX + LCD_R) * 2, (CY + LCD_R) * 2], fill=255)
g = ImageChops.multiply(g, disc).resize((W, H), Image.LANCZOS)
glass = Image.merge("RGBA", (Image.new("L", (W, H), 255),) * 3 + (g,))
glass.save(os.path.join(OUT, "kraken-glass.png"), optimize=True)

json.dump({"width": W, "height": H, "lcd_cx": CX, "lcd_cy": CY, "lcd_d": LCD_D},
          open(os.path.join(OUT, "kraken-scene.json"), "w"))

# ---------------------------------------------------------------- preview
frame_path = os.path.join(OUT, "sample-lcd.png")
base = Image.new("RGBA", (W, H), (0, 0, 0, 255))
if os.path.exists(frame_path):
    fr = Image.open(frame_path).convert("RGBA").resize((LCD_D, LCD_D))
else:
    fr = Image.new("RGBA", (LCD_D, LCD_D), (8, 8, 10, 255))
    ImageDraw.Draw(fr).ellipse([12, 12, LCD_D - 12, LCD_D - 12], outline=(230, 70, 160), width=10)
base.alpha_composite(fr, (CX - LCD_R, CY - LCD_R))
base.alpha_composite(out)
base.alpha_composite(glass)
base.convert("RGB").save(os.path.join(OUT, "kraken-preview.png"))
print("ok", CX, CY)
