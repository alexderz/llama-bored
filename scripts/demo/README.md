# README demo GIFs

How `docs/media/kraken-lcd.gif`, `docs/media/tty11.gif` and `docs/media/keyboard.gif` are made. Dev
tooling only; nothing here is installed.

1. Record (root, reads the snapshot and `/dev/vcsa11` at 10 Hz):
   `sudo python3 record.py demo.rec 3600`
2. Export the snapshots: `python3 export.py demo.rec snaps/`
   (writes `index.tsv` and one JSON per snapshot).
3. Pick a window (seconds from the start) and replay it through the real
   writer loop, stream mode, fake LCD. Start the replay early enough that
   the history dial has filled:
   `REPLAY_IN=snaps REPLAY_OUT=lcd REPLAY_FROM_S=207.5 REPLAY_TO_S=219.5 cargo test --release -p kraken-lcd --test replay_demo -- --ignored --nocapture`
4. The scene: `python3 make_scene.py` draws `kraken-scene.png`,
   `kraken-glass.png` and `kraken-scene.json` (the 320 px LCD hole).
   Paste each 320x320 RGBA frame at (cx-160, cy-160) under the scene, the
   glass on top, one shared 255-colour palette, 100 ms per frame.
5. tty11: `ttyrender.render(vcsa_bytes)` draws a recorded `/dev/vcsa11`
   buffer at the size in its header, with the console font
   (`TTYRENDER_FONT`, default the bundled 10x18) and the llama palette read
   from llama-core (`TTYRENDER_PALETTE=vga` for the kernel's). 512 glyphs:
   the attribute is shifted and bit 0 is glyph bit 8; colours are in VGA
   order. Scale to 50 %, 64-colour palette.

   Pick the window: `python3 scan.py demo.rec` prints the 15 s with the
   most OUT changes while generating (live text needs llama.cpp with
   `LLAMA_SERVER_SLOTS_DEBUG=1`). Then
   `python3 make_scene.py` (in a scratch copy) and
   `python3 build.py demo.rec FROM TO lcd-frames/ scene-dir/ out/` write
   `tty11.gif` and `kraken-lcd.gif` (the LCD frame masked to the round
   panel).
6. Keyboard: run the same snapshots through the real llama-light renderer
   with the `keyboard-c` example (plus `[aura] enabled = false`):
   `REPLAY_IN=snaps REPLAY_CONFIG=kbc1.toml REPLAY_OUT=kb-frames.tsv REPLAY_FROM_S=207.5 REPLAY_TO_S=219.5 cargo test --release -p llama-light --test replay_keyboard -- --ignored`
   then `python3 kbdraw.py kb-frames.tsv keyboard.gif` draws the stylized
   STRAFE with simulated keycap bleed.
