"""Stylized STRAFE RGB MK.2 drawing with simulated keycap bleed (after
Fable's keyboard-c.html). Reads replay_keyboard frames, writes a GIF."""
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

L = []
def K(i, x, y, w=1, h=1, label=None): L.append((i, x, y, w, h, i if label is None else label))
K('Windows Lock',2.2,0,1,.55,'WIN'); K('Media Mute',15.25,0,1,.55,'MUTE')
for x,n,l in ((18.5,'Media Stop','STOP'),(19.5,'Media Previous','PREV'),(20.5,'Media Play/Pause','PLAY'),(21.5,'Media Next','NEXT')): K(n,x,0,1,.55,l)
K('Escape',0,.95,1,1,'Esc')
for x,n in [(2,'F1'),(3,'F2'),(4,'F3'),(5,'F4'),(6.5,'F5'),(7.5,'F6'),(8.5,'F7'),(9.5,'F8'),(11,'F9'),(12,'F10'),(13,'F11'),(14,'F12')]: K(n,x,.95)
K('Print Screen',15.25,.95,1,1,'PrtSc'); K('Scroll Lock',16.25,.95,1,1,'ScrLk'); K('Pause/Break',17.25,.95,1,1,'Pause')
R1=2.15; K('`',0,R1)
for i,n in enumerate('1234567890'): K(n,1+i,R1)
K('-',11,R1); K('=',12,R1); K('Backspace',13,R1,2,1,'Bksp')
K('Insert',15.25,R1,1,1,'Ins'); K('Home',16.25,R1); K('Page Up',17.25,R1,1,1,'PgUp')
K('Num Lock',18.5,R1,1,1,'Num'); K('Number Pad /',19.5,R1,1,1,'/'); K('Number Pad *',20.5,R1,1,1,'*'); K('Number Pad -',21.5,R1,1,1,'-')
R2=3.15; K('Tab',0,R2,1.5)
for i,n in enumerate('QWERTYUIOP'): K(n,1.5+i,R2)
K('[',11.5,R2); K(']',12.5,R2); K('\\ (ANSI)',13.5,R2,1.5,1,'\\')
K('Delete',15.25,R2,1,1,'Del'); K('End',16.25,R2); K('Page Down',17.25,R2,1,1,'PgDn')
K('Number Pad 7',18.5,R2,1,1,'7'); K('Number Pad 8',19.5,R2,1,1,'8'); K('Number Pad 9',20.5,R2,1,1,'9'); K('Number Pad +',21.5,R2,1,2,'+')
R3=4.15; K('Caps Lock',0,R3,1.75,1,'Caps')
for i,n in enumerate('ASDFGHJKL'): K(n,1.75+i,R3)
K(';',10.75,R3); K("'",11.75,R3); K('Enter',12.75,R3,2.25)
K('Number Pad 4',18.5,R3,1,1,'4'); K('Number Pad 5',19.5,R3,1,1,'5'); K('Number Pad 6',20.5,R3,1,1,'6')
R4=5.15; K('Left Shift',0,R4,2.25,1,'Shift')
for i,n in enumerate('ZXCVBNM'): K(n,2.25+i,R4)
K(',',9.25,R4); K('.',10.25,R4); K('/',11.25,R4); K('Right Shift',12.25,R4,2.75,1,'Shift')
K('Up Arrow',16.25,R4,1,1,'^')
K('Number Pad 1',18.5,R4,1,1,'1'); K('Number Pad 2',19.5,R4,1,1,'2'); K('Number Pad 3',20.5,R4,1,1,'3'); K('Number Pad Enter',21.5,R4,1,2,'Ent')
R5=6.15; K('Left Control',0,R5,1.25,1,'Ctrl'); K('Left Windows',1.25,R5,1.25,1,'Win'); K('Left Alt',2.5,R5,1.25,1,'Alt'); K('Space',3.75,R5,6.25,1,'')
K('Right Alt',10,R5,1.25,1,'Alt'); K('Right Windows',11.25,R5,1.25,1,'Win'); K('Menu',12.5,R5,1.25,1,'Menu'); K('Right Control',13.75,R5,1.25,1,'Ctrl')
K('Left Arrow',15.25,R5,1,1,'<'); K('Down Arrow',16.25,R5,1,1,'v'); K('Right Arrow',17.25,R5,1,1,'>')
K('Number Pad 0',18.5,R5,2,1,'0'); K('Number Pad .',20.5,R5,1,1,'.')

U, PAD = 40, 14
W, H = int(22.5*U + 2*PAD), int(7.15*U + 2*PAD)
BG, CASE, EDGE = (10, 11, 16), (21, 23, 30), (38, 42, 54)
FONT = {s: ImageFont.truetype('/usr/share/fonts/source-foundry-hack-fonts/Hack-Regular.ttf', s) for s in (8, 9, 11)}
def lift(c): return tuple(round(255*(v/255)**0.6) for v in c)
def hexc(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))

def draw(colors):
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, W-3, H-3), 12, fill=CASE, outline=EDGE)
    glow = Image.new('RGB', (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    rects = []
    for i, x, y, w, h, lab in L:
        c = lift(colors.get(i, (0, 0, 0)))
        r = (PAD+x*U+2, PAD+y*U+2, PAD+(x+w)*U-2, PAD+(y+h)*U-2)
        rects.append((r, c, lab, h))
        g.rounded_rectangle((r[0]-2, r[1]-2, r[2]+2, r[3]+2), 8, fill=c)
    wide = glow.filter(ImageFilter.GaussianBlur(U*0.45)).point(lambda v: v*0.9)
    near = glow.filter(ImageFilter.GaussianBlur(U*0.12)).point(lambda v: v*0.7)
    img = ImageChops.screen(ImageChops.screen(img, wide), near)
    caps = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    k = ImageDraw.Draw(caps)
    for r, c, lab, h in rects:
        k.rounded_rectangle(r, 5, fill=c + (199,), outline=(0, 0, 0, 115))
    img = Image.alpha_composite(img.convert('RGBA'), caps)
    t = ImageDraw.Draw(img)
    for r, c, lab, h in rects:
        if not lab: continue
        f = FONT[8 if h < .8 else (9 if len(lab) > 3 else 11)]
        t.text(((r[0]+r[2])/2, (r[1]+r[3])/2), lab, font=f, fill=(255, 255, 255, 158), anchor='mm')
    return img.convert('RGB')

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    frames = []
    for line in open(src):
        parts = line.rstrip('\n').split('\t')
        frames.append(draw({k: hexc(v) for k, v in (p.rsplit('=', 1) for p in parts[1:])}))
    frames[len(frames)//2].save(out.replace('.gif', '-mid.png'))
    strip = Image.new('RGB', (W, H*4))
    for n, i in enumerate((0, len(frames)//3, 2*len(frames)//3, -1)): strip.paste(frames[i], (0, H*n))
    pal = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=100, loop=0, optimize=True, disposal=1)
    print(len(frames))
