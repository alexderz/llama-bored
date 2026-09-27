# Changelog

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
  two-opcode encoder with no save-to-flash. Corsair keyboard support is
  detection only for now.
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
