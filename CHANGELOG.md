# Changelog

All notable changes to llama-bored. Versions follow
[Semantic Versioning](https://semver.org/); the process is in
[docs/RELEASING.md](docs/RELEASING.md).

## Unreleased

### Fixed

- SGLang and vLLM `/metrics` may be up to 1 MiB: SGLang's is about 70 KiB of
  latency histograms, over the 64 KiB llama.cpp limit, so its gauges were
  dropped. The fallback log line now names the reason.

### Added

- Backends beyond llama.cpp: SGLang, vLLM and any other OpenAI-compatible
  server behind llama-swap, told apart by the launch command or
  `[llama.backends]`. SGLang and vLLM metrics give tok/s, running and
  queued requests, KV fill and cache hit rate; others fall back to token
  counts from llama-swap's request log. The tuning line reads SGLang and
  vLLM flags. The snapshot gains optional `backend`, `running`, `queued`
  and `kv_fill` per model, and `llamabored_model_loaded` a `backend` label.
- tty11 flags a model that stays `stopping` for over a minute.

- Continuous integration: GitHub Actions run `scripts/check.sh` on every
  push and pull request, a weekly advisory audit, and a release workflow
  that checks a `vX.Y.Z` tag against the crate versions and the changelog
  and publishes the release notes. Actions are pinned by commit; Dependabot
  keeps them current.

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
