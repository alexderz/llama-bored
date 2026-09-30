import time, zlib, struct, sys
out = open(sys.argv[1], 'wb')
end = time.monotonic() + float(sys.argv[2])
t0 = time.monotonic(); n = 0
while time.monotonic() < end:
    t = time.monotonic()
    try:
        snap = open('/run/llama-watch/snapshot.json', 'rb').read()
        scr = open('/dev/vcsa11', 'rb').read()
    except OSError:
        time.sleep(0.1); continue
    ts = time.time()
    for kind, blob in ((b'S', snap), (b'V', scr)):
        z = zlib.compress(blob, 6)
        out.write(kind + struct.pack('<dI', ts, len(z)) + z)
    n += 1
    if n % 50 == 0: out.flush()
    time.sleep(max(0, 0.1 - (time.monotonic() - t)))
out.close()
