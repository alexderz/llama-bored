"""Build tty11.gif and kraken-lcd.gif for one window of a demo recording.

usage: build.py REC FROM_S TO_S LCD_FRAMES_DIR SCENE_DIR OUT_DIR
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import ttyrender as t

rec, s, e, lcd_dir, scene_dir, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
os.makedirs(out, exist_ok=True)

# tty11: one frame per 100 ms of the window, rendered as the console draws it
vf = [(ts, b) for k, ts, b in t.frames(rec) if k == b'V']
t0 = vf[0][0]
pick, nxt = [], s
for ts, b in vf:
    if ts - t0 >= nxt and ts - t0 < e:
        pick.append(b); nxt += 0.1
imgs = [t.render(b) for b in pick]
w, h = imgs[0].size
small = [im.resize((w // 2, h // 2), Image.LANCZOS) for im in imgs]
pal = small[len(small) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in small]
q[0].save(os.path.join(out, 'tty11.gif'), save_all=True, append_images=q[1:], duration=100, loop=0, optimize=True)
imgs[len(imgs) // 2].save(os.path.join(out, 'tty11-mid.png'))
print('tty11', len(q), 'frames', w // 2, 'x', h // 2)

# Kraken LCD: replayed 320x320 RGBA frames under the scene, glass on top
meta = json.load(open(os.path.join(scene_dir, 'kraken-scene.json')))
scene = Image.open(os.path.join(scene_dir, 'kraken-scene.png')).convert('RGBA')
glass = Image.open(os.path.join(scene_dir, 'kraken-glass.png')).convert('RGBA')
cx, cy = meta['lcd_cx'], meta['lcd_cy']
from PIL import ImageDraw
d = meta.get('lcd_d', 320)
mask = Image.new('L', (320 * 4, 320 * 4), 0)
ImageDraw.Draw(mask).ellipse([(160 - d / 2) * 4, (160 - d / 2) * 4, (160 + d / 2) * 4 - 1, (160 + d / 2) * 4 - 1], fill=255)
mask = mask.resize((320, 320), Image.LANCZOS)  # the panel is round
files = sorted(glob.glob(os.path.join(lcd_dir, '*.rgba')))
comp = []
for f in files:
    fr = Image.frombytes('RGBA', (320, 320), open(f, 'rb').read())
    fr.putalpha(mask)
    base = scene.copy()
    base.alpha_composite(fr, (cx - 160, cy - 160))
    base.alpha_composite(glass)
    comp.append(base.convert('RGB'))
pal = comp[len(comp) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in comp]
q[0].save(os.path.join(out, 'kraken-lcd.gif'), save_all=True, append_images=q[1:], duration=100, loop=0, optimize=True)
comp[len(comp) // 2].save(os.path.join(out, 'kraken-lcd-mid.png'))
print('lcd', len(q), 'frames', scene.size)
