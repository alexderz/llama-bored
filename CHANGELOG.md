# Changelog

All notable changes to llama-bored. Versions follow
[Semantic Versioning](https://semver.org/); the process is in
[docs/RELEASING.md](docs/RELEASING.md).

## Unreleased

### Fixed

- tty11 SLOTS ctx sparklines are no longer wiped on every model swap (#45). A history is kept per (model display name, slot) and dropped only when its slot has not been on `/slots` for 30 min (or `tty.ctx_history_h`, if shorter), or to make room at the 64-slot cap (the slot away longest goes); a second model loading (A→A+B) leaves A's line alone, and A→B→A within that time resumes A's line, the time away shown as a gap. Only slots on `/slots` are drawn. A slot that comes back after the set of `/running` models changed gets the existing reset marker and drops its held context (the reloaded model's cache is gone); new lines after a swap are marked as before. The conversations a model's slots lost (#9's `evicted`) now outlive an unload and a llama-swap restart for their own 1 h, 16 per model, at most 64 models (oldest loss dropped first), so an eviction can be told across a swap. Drop counts behave as before. No golden changed
- tty11 RECENT no longer empties when llama-swap restarts (#44). llama-watch keeps its own ring of the newest 32 rows: each `/api/metrics/activity` read is merged in (new rows on top, rows already held updated in place) instead of replacing what is shown, so RECENT and its engine-measured speeds (#35) survive a restart and a model swap; RECENT still shows 8 rows with the text panels, up to 32 without. A llama-swap restart is now told by a lower top id, an empty list after rows, a held id whose fingerprint (model, timestamp, input and output tokens, duration) changed, or `/running` having been down with no held row on the page; its rows are a new generation and all count (previously a restart whose new top id reached the old one was skipped for fallback tokens, prompt-cache counters and speeds, and a reused id could take an old row's speeds). Old rows stay with their speeds; per-row state (speeds, reset-reason matching) is keyed by llama-watch's own row number, and a capture (IN/OUT) is fetched only for a row of the page just read, keyed by generation and id, so an old row's id is never fetched. Logged once per restart: `activity: llama-swap restarted; its row ids start again`
- llama-watch: engine and counter state no longer carries over or freezes across llama-swap restarts and engine changes (#46). A failed `/running` read now forgets the previously ready models' engine windows and attributed-speed reads at once (TTFT, ITL, spec and speeds from the old process stopped showing until a new window); the first good read after it still compares against the last good ready list. A model that leaves `ready`, or whose server changes under the same llama-swap id, counts its next process's decode and prompt counters from 0 (a reload already past the old value was undercounted by it) and goes back to activity rows for its prompt and cached counters until `/metrics` gives both again (a vLLM/SGLang → llama.cpp switch froze them). When `/running` is refused (llama-swap gone with its servers) the counters restart at once; a timeout keeps their baselines, so a busy llama-swap does not count a live server's totals twice
- llama-watch is no longer killed by its watchdog when a write to tty11 blocks (#42). The likely trigger on Titan was waking the blanked console with a key (Space, no XOFF): `con_write` waits on the console lock while fbcon and nvidia-drm unblank and modeset the monitor, 1 s to over 10 s, and the frame was written on the thread that publishes the snapshot and pings the watchdog (kraken-lcd and llama-light logged `snapshot stale` first). Frames now go to a `tty-writer` thread through a one-frame slot: the newest frame replaces a waiting one, publish and `WATCHDOG=1` never wait, a draw that took 1 s or more is followed by a full repaint, `tty: output stalled` / `tty: output resumed` are logged once each, and a clean stop waits at most 500 ms for a stuck writer. tty11 is also quietened at start: `ixon ixoff echo icanon isig` cleared and input flushed, only when stdout is `/dev/tty11` (Ctrl+S cannot pause it, keys do not echo onto it); a clean stop puts the saved flags back and `tty-reset` restores the console defaults before the palette. Scroll Lock still holds output (the dashboard freezes, then repaints); S13 allows `tcgetattr` / `tcsetattr` / `tcflush` in `term.rs`

## 0.3.4 — 2026-10-03

### Fixed

- tty11's IN no longer sticks on the first user message during an agent's tool loop (#38). For a llama-swap capture whose last message is not `user`, IN shows the `user` and `tool` messages after the last `assistant` one, in order, newest last, tool results marked `[tool]`, and the title says what they are: `IN (last request · 3 tool results)`, `IN (last request · user + 2 tool results)`. With no `assistant` message the run is every message but `system` / `developer`. A plain chat (last message `user`) and a completions prompt look as before. The walk still streams one message at a time and keeps at most 32,768 characters (the largest `llama.input_tail_chars`) of IN text; panel titles now draw the `·` separator. No golden changed
- tty11: a long model name and engine detail no longer run into the clock. The header detail drops whole trailing ` · item`s to keep two blank cells before the clock (at 160 columns `vLLM · kv kvarn_k4v2_g128 · block 128`, was `… · prefix2026-10-03 21:59:44`); the `slots` and `swap` fields are left out when they would reach it (#39)

## 0.3.3 — 2026-10-03

### Added

- The active inference engine is always named (#33). tty11's header detail
  starts with the shown model's engine (`llama.cpp · 256k · kv f16 · Q6_K`;
  `SGLang`, `vLLM`, `Strata`, `OpenAI-compatible`; a llama.cpp fork stays
  `llama.cpp`) and says `engine --` with no model. The LCD leads the detail
  line under a single model's name with the short name (`llama.cpp`,
  `sglang`, `vllm`, `strata`, `openai`), kept while quant, kv and
  `spec 78 %` drop for width. llama-view's pane title is
  `llama-view: <model> (<engine>) · <host>` when it fits the 40-character
  cap. The exporter is unchanged (`backend` label on
  `llamabored_model_loaded`).
- vLLM requests in RECENT get prompt and generation speeds measured by the
  engine (#35). llama-swap reports `-1` for them, so llama-watch reads vLLM's
  per-request histograms (`request_prefill_time_seconds`,
  `request_prefill_kv_computed_tokens`, `request_decode_time_seconds`,
  `request_generation_tokens`) on each metrics poll: prefill tok/s is
  uncached prompt tokens over prefill time, decode tok/s is generated tokens
  after the first over decode time. A new row gets the speeds of the
  requests that finished around it (their average if several), drawn
  `~2,134` / `~41.3` with a `~ = engine-measured` legend note; no match
  stays `--`, and llama.cpp rows are unchanged. SGLang has no per-request
  phase times and gets none. Snapshot: optional `engine.prefill_tps` /
  `engine.decode_tps` (schema 1, additive). llama-metrics:
  `llamabored_model_prefill_tokens_per_second` and
  `llamabored_model_decode_tokens_per_second`. tty11's backend line ends
  `· prefill 2,134/s · decode 41.2/s` where it fits.

## 0.3.2 — 2026-10-03

### Added

- A model whose launch command does not name its server (for example a
  container image that runs `vllm serve` itself) is recognised by its
  `/metrics` names, read once per load and only while llama-swap reports it
  `ready` (#31).
- vLLM: speculative-decoding acceptance rate and mean accepted length (tty11,
  the LCD and the exporter), mean TTFT, inter-token and request latency,
  preemptions, engine sleep, and the KV dtype, block size and prefix caching
  from `cache_config_info` for the tuning line. SGLang's spec-decode gauges
  and latency histograms are read too (#31).

## 0.3.1 — 2026-10-03

### Added

- tty11 loads llama-bored's own 16-colour palette, built from the heat ramp
  the LCD and RGB use (`[tty] palette = "llama"`, the default; `"vga"`
  keeps the kernel's colours). Only tty11 changes; `llama-watch tty-reset`
  restores the console palette when the unit stops. llama-cast renders with
  the same palette (`palette` in `cast.toml`) (#26).
- llama-view sends exact colours instead of the terminal theme's 16:
  `--colors auto|truecolor|256|16` (`LLAMA_VIEW_COLORS`) and
  `--palette llama|vga` (`LLAMA_VIEW_PALETTE`); `auto` uses truecolor when
  `COLORTERM` says so, else the nearest fixed xterm-256 colour (#25).
- llama-view is a better tmux resident: it follows the pane's and tty11's
  size (`--fit crop|center`, `--offset-x/-y`, a hint line when the pane is
  smaller), draws each frame as one synchronized update, drops to 1 fps
  while its pane is unfocused, sets the pane title to
  `llama-view: <model> · <host>` (and, opt-in, the tmux window name with
  `--tmux-window-name`), restores the terminal on `q`/Ctrl-C, and never
  rings the bell or captures the mouse (#27).

### Fixed

- llama-view decodes tty11's colours correctly with a 512-glyph console font
  (the bundled fonts); `--font-glyphs auto|256|512` overrides the detection (#28).
- The configured tty11 `[tty] font` and `size` now survive a reboot: at boot
  the framebuffer console's take-over can still be deferred, so
  `llama-watch tty-setup` writes one space to tty11 first and retries
  `setfont` for up to 5 s before setting the size (#22).

## 0.3.0 — 2026-09-30

### Added

- **llama-cast: watch tty11 on a TV.** A DLNA/UPnP media server that streams
  the tty11 dashboard as live 1920x1080 H.264 (MPEG-TS, rendered from the
  console at 2 fps with the bundled Hack font) to players on your LAN.
  Tested with Roku Media Player on TCL Roku TVs: open it, choose
  **llama-bored**, then **llama-bored live**. Off by default; its own user,
  an in-process CIDR allowlist plus `IPAddressAllow=`, pinned ports (a TCP
  stream port and UDP 1900 for SSDP), strict HTTP, one `ffmpeg` per viewer
  with a fixed argument list, no outbound connections, and a new source
  fence (S18). Setup: README "Watch tty11 on a TV".
- **Strata backend** ([Strata](https://github.com/Niko1221/Strata),
  `serve/server.py --engine strata`): detected from the launch command;
  tok/s, running and queued requests, prompt and cached-prompt counters and
  the context size from its JSON `/metrics`. Snapshot readers now read an
  unknown backend word as `openai` instead of rejecting the snapshot.

### Changed

- Only llama-metrics and llama-cast may listen (S16 now allows both, each
  under its own fence). The printed upgrade step restarts llama-cast too.

## 0.2.3 — 2026-09-30

### Fixed

- kraken-lcd: right after it opens the device, the Kraken can refuse bucket
  commands for several seconds (seen after a restart, while the previous
  process's stock screen settles). Until the first successful upload, for
  up to 60 s, refusals now back off without counting toward `fail_limit`;
  after that, failures count as before and persistent ones still restore
  the stock screen (#14).

## 0.2.2 — 2026-09-29

### Added

- Install without an NZXT Kraken Z: with none attached, the installer sets
  up everything except kraken-lcd's unit, udev rules and state directory,
  and says how to add it later; more than one still refuses (#6).
- tty11 on small or mixed screens: `[tty] size = "COLSxROWS"` and
  `[tty] font = "12x24" | "12x22"` (a second bundled Hack font), applied by
  a root pre-step `llama-watch tty-setup` with no shell; below 48 rows the
  layout drops IN/OUT, then FANS, then shrinks the chart instead of
  refusing, down to 160x26 (#7).
- SLOTS labels each context drop by best-guess reason (compacted, new,
  evicted, unknown) from token counts, with a legend (#9).
- llama-metrics: `llamabored_slot_ctx_used_tokens`,
  `llamabored_slot_ctx_resets_total{reason}`,
  `llamabored_model_prompt_tokens_total` and
  `llamabored_model_prompt_cached_tokens_total` (#10).
- IN/OUT for SGLang, vLLM and other servers without `/slots`: the last
  finished exchange from llama-swap's request captures
  (`/api/captures/<id>`, up to 2 MiB, never with `show_text = false`),
  pinned by a new source fence (S17) (#5).
- `scripts/demo/` and two ignored replay tests regenerate the README GIFs
  from recorded snapshots.

### Changed

- tty11 RECENT: DURATION sits beside the other numbers, left of the rate
  bar (#8).
- `llama-watch.service` loads the console font through `llama-watch
  tty-setup` instead of calling `setfont` directly.

### Fixed

- kraken-lcd stream mode backs off after a refused upload (2 s, doubling to
  30 s) instead of retrying on the next frame, so restarting it together
  with llama-watch no longer spends `fail_limit` in 300 ms and exits (#14).

## 0.2.1 — 2026-09-29

### Added

- llama-metrics exports everything the dashboards show: GPU and CPU power
  and the GPU power limit, VRAM and system memory in bytes, a prompt-token
  counter, llama.cpp slots busy and total, SGLang/vLLM running and queued
  requests, KV cache fill and cache hit ratio, model lifecycle state, fan
  rpm and pwm, and per-source health and latency. A test fails when a
  snapshot field is not exported (#11).

### Fixed

- Readers ignore unknown snapshot fields within the same schema number
  (known fields stay strictly validated), so upgrading llama-watch no
  longer blinds an older llama-metrics or llama-light; both now log once
  why they reject a snapshot. install.sh's printed restart step covers all
  four units (#12).

## 0.2.0 — 2026-09-29

### Added

- Backends beyond llama.cpp: SGLang, vLLM and any other OpenAI-compatible
  server behind llama-swap, told apart by the launch command or
  `[llama.backends]` in `watch.toml`. SGLang (start it with
  `--enable-metrics`) and vLLM metrics give tok/s, running and queued
  requests, KV fill and cache hit rate; other servers fall back to token
  counts from llama-swap's request log. The tuning line reads SGLang and
  vLLM flags. The snapshot gains optional `backend`, `running`, `queued`
  and `kv_fill` per model.
- tty11 shows the backend, a `running · queued · KV · hit` line in SLOTS
  for servers without `/slots`, and flags a model that stays `stopping`
  for over a minute.
- Continuous integration: GitHub Actions run `scripts/check.sh` on every
  push and pull request, a weekly advisory audit, and a release workflow
  that checks a `vX.Y.Z` tag against the crate versions and the changelog
  and publishes the release notes. Actions are pinned by commit; Dependabot
  keeps them current.

### Changed

- **Upgrading:** a llama.cpp server started through a wrapper script, with
  no `llama-server` in its llama-swap command, is now treated as a generic
  OpenAI-compatible server (no `/slots`, no llama.cpp metrics). Add
  `"model-id" = "llamacpp"` under `[llama.backends]` in `watch.toml`.
- **Upgrading:** `llamabored_model_loaded` has a new `backend` label
  (`llamacpp`, `sglang`, `vllm` or `openai`). Queries or alerts that match
  its exact label set need the label added or ignored.

## 0.1.2 — 2026-09-27

### Added

- `docs/RELEASING.md`: how versions, the changelog and releases work.
- README demo GIFs of the Kraken LCD, tty11 and the keyboard (`docs/media/`).
- README says up front what llama-bored needs: Linux with systemd and one
  NZXT Kraken Z (`1e71:3008`); NVIDIA, llama-swap, Aura and the keyboard are
  optional. Requirements list the tools the build and install scripts use
  (`cc`, `nm`, `ldd`, `sha256sum`, `setfacl`/`getfacl`).
- The `cpu_topk` light metric (the busiest cores) is documented in the
  README and the install skill.

### Changed

- Comments in the shipped units, udev rules and scripts no longer cite
  internal ticket numbers; risk ids name their row in `docs/SAFETY.md`.
- The install skill's frontmatter name is `install`, matching its folder,
  and it clones from `https://github.com/alexderz/llama-bored`.

### Fixed

- README, `docs/SAFETY.md` and `docs/ARCHITECTURE.md` describe device access
  exactly: kraken-lcd's usbfs grant is class-wide and narrowed by udev (new
  risk row RR7c), llama-light is pinned to two nodes (Aura and keyboard), and
  the unit hardening is listed per unit.
- The install skill's `pgrep -f` checks no longer match their own shell
  when run through `bash -c`.
- README and the skill name the log lines to look for: `cooling guard
  halted` / `cooling guard latch is present` (`journalctl -u kraken-lcd -p
  crit`) and llama-light's `aura absent (...)`. `check` commands are shown
  with their full path and `--config`.
- README: the demo generates at about 160 tok/s, not 80; tty11 is
  Ctrl+Alt+F11 from a graphical session; `listen` and `allow` in
  `metrics.toml` are required; the full uninstall also removes the
  `llama-metrics.service.d` drop-in directory.
- `docs/SAFETY.md`: the HALTED latch file is owned by kraken-lcd and
  cleared by an explicit root command; the development-only
  `--query-buckets` and `bench-upload` commands are guarded too.

### Security

- `llama-metrics.service` adds `InaccessiblePaths=/sys`, so the exporter
  cannot read sysfs even through a bug; it never needed it.

## 0.1.1 — 2026-09-27

### Added

- **llama-light drives the Corsair STRAFE RGB MK.2** (`1b1c:1b48`) per key,
  on its vendor lighting interface only (typing is untouched). A closed
  encoder with four report shapes (software mode, hardware mode, colour
  stream, 24-bit commit) and no profile, firmware or flash writes; stopping
  hands the keyboard back to its own lighting. An unplugged keyboard is
  logged once and looked for every 10 s. udev pins it to
  `/dev/llama-light/keyboard`, the unit's second `DeviceAllow=`.
- Keyboard targets: key ranges (`keyboard.keys["F1".."F12"]`), key lists,
  `keyboard.all`, single keys by name, and `led:N`.
- An optional frame pipeline for llama-light: `[engine]` (`tick_hz`,
  `target_hz`, `tween_fps`, `tween_s`) recomputes targets at a slow rate and
  tweens between them; `[base]` colour under every entry; named
  `[palette.NAME]` stops.
- New `[[light]]` options: `ladder`, `gate` and `peak` styles,
  `edge = "fractional"`, `gradient = "position"`, `attack_s` (asymmetric
  smoothing), `rate_window_s` (windowed tokens/s) and `brightness = [lo, hi]`.
- `packaging/light.example.toml` has a complete keyboard layout
  (`keyboard-c`), made for translucent keycaps where light bleeds between
  keys: activity on F1–F12, GPU and busiest cores as two bars each two rows
  tall (keys interleaved along the row stagger, so they grow in half-key
  steps), each bar one colour chosen by its value, GPU temperature on the
  navigation keys, and a tokens/s dial on the number pad around "5".

### Changed

- Colours: a deeper, more saturated cold end, and the top of the redline
  is a bright hot orange-gold instead of white. LCD, tty11 and RGB share it.
- tty11 level meters are spectra: each cell is coloured by its position.

## 0.1.0 — 2026-09-26

First public release. For watching when you're bored waiting on your agent.

Built with Claude Code and Grok: Claude (Opus, Fable) for design, review,
security and much of the code; Grok CLI for many of the builds.

### Added

- **llama-watch**, the single reader of `/proc`, hwmon, NVML and llama-swap,
  publishing a validated snapshot to `/run/llama-watch/snapshot.json` at
  10 Hz, and the **tty11 dashboard**:
  - the bundled Hack 12x24 console font and an eighth-block token chart with
    configurable rate ceilings (`gen_ceiling_tps`, `prompt_ceiling_tps`);
  - an ACTIVITY row that names its source (`gpu`, `cpu` or `util`);
  - SLOTS with per-slot context fill and a context-history sparkline with
    reset markers;
  - RECENT across the full width with full timestamps;
  - IN and OUT for the busy slot, with a clean prompt view that strips
    chat-template tokens;
  - `show_text = false` / `--no-text` to hide and stop keeping all llama text;
  - an optional read-only FANS panel for a Super-I/O hwmon chosen by name;
  - the full model name and a tuning line (ctx, KV cache, quant, CPU MoE).
- **Activity** as the bottleneck device's power headroom: GPU via NVML, CPU
  via zenergy, with `nominal_frac` and auto-learned idle floors that never
  rise; falls back to utilisation without power sensors.
- **llama-view**, a read-only mirror of tty11 for tmux or SSH (group
  `llama-view`).
- **kraken-lcd**, a network-free LCD writer for the NZXT Kraken Z53
  (320x320) with the "V3b Plasma Blackbody" face: a 270° activity gauge with
  a 100–125 redline, a 30-minute activity dial, a tokens·24h chart, model
  name and tuning line, temperatures, and distinct "AI down", "no model" and
  "no data" states. Change mode (default) and opt-in stream mode (10 fps).
  Its own minimal Kraken Z3 LCD protocol with a closed command set, a cooling
  guard, a persistent HALTED latch, stock restore on stop, and a guarded
  `show-image` test card.
- **llama-light**, a colour-only RGB writer for the ASUS Aura USB controller
  (`0b05:18f3`): `[[light]]` mappings from any snapshot metric to solid,
  ring, bar or pulse styles with palettes or colour stops, mirrored or
  daisy-chained fans, a brightness cap, and live reload of `light.toml`. A
  two-opcode encoder with no save-to-flash. Corsair keyboard support was
  detection only in this release.
- **llama-metrics**, a Prometheus exporter on TCP 19477 (`llamabored_*`
  series) for a CIDR allowlist, with strict HTTP limits; the only process
  that listens.
- **llama-core**, the shared snapshot schema v1, validator and sanitiser.
- Hardened systemd units, udev rules pinning each writer to one device node
  (`/dev/kraken-lcd/hid`, `/dev/llama-light/aura`), sysusers, and an
  installer anchored to `main` that verifies the build, never starts the LCD
  writer or enables the exporter, prints the next steps, and keeps a per-run
  rollback.
- Safety fences S1–S3 and S6–S16 in `scripts/check.sh` and the test suite.
- An agent install skill (`skills/install/SKILL.md`).

### Hardware

- Tested on the Kraken Z53 and an ASUS Aura USB controller. The Kraken Z63
  and Z73 share USB id `1e71:3008`, so they are accepted, but untested.
