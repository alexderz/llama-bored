import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttyrender as t

def text_rows(v):
    rows, cols = v[0], v[1]
    cells = v[4:]
    out = []
    for r in range(rows):
        out.append(bytes(cells[2 * (r * cols + c)] for c in range(cols)))
    return out

frames = [(ts, b) for kind, ts, b in t.frames(sys.argv[1]) if kind == b'V']
t0 = frames[0][0]
# OUT panel: from the row starting "OUT (" to the health row, left 110 columns
changes = []
prev = None
for ts, v in frames:
    rows = text_rows(v)
    try:
        start = next(i for i, r in enumerate(rows) if r.startswith(b'OUT'))
    except StopIteration:
        start = None
    body = b''.join(r[:110] for r in rows[start:start + 10]) if start is not None else b''
    gen = any(b' gen ' in r for r in rows[8:20])
    changes.append((ts - t0, body != prev and prev is not None, gen))
    prev = body
win = 15.0
best = None
for i, (s, _, _) in enumerate(changes):
    e = s + win
    sel = [c for c in changes if s <= c[0] < e]
    if not sel or sel[-1][0] < e - 0.5:
        continue
    score = sum(1 for c in sel if c[1])
    gen = sum(1 for c in sel if c[2]) / len(sel)
    if best is None or score > best[0]:
        best = (score, s, gen)
print(f"frames {len(frames)} span {changes[-1][0]:.1f}s")
print("best window: start %.1fs, OUT changes %d, generating %.0f%%" % (best[1], best[0], best[2] * 100))
