import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ttyrender import frames
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
idx = open(os.path.join(out, 'index.tsv'), 'w'); n = 0
for kind, ts, blob in frames(src):
    if kind == b'S':
        name = f's{n:06d}.json'; open(os.path.join(out, name), 'wb').write(blob)
        idx.write(f'{ts:.3f}\t{name}\n'); n += 1
print(n)
