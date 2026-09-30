# Architecture

llama-bored is five small Rust binaries and a shared library in one Cargo
workspace. They are split so that each process can reach only what its job
needs: the one that reads everything cannot reach a device, the ones that
write to a device have no network and read almost nothing, and the one that
listens on the network opens no device.

## Processes

| Binary | Runs as | Reads | Writes | Network |
|---|---|---|---|---|
| `llama-watch` | system unit, user `llama-watch` | `/proc`, hwmon (incl. an opt-in Super-I/O fan chip), NVML, llama-swap on loopback | `/run/llama-watch/snapshot.json` (10 Hz), tty11 | Loopback client only (`IPAddressAllow=localhost`) |
| `kraken-lcd` | system unit, user `kraken-lcd` | The snapshot; the cooler's own sysfs (guard) | The Kraken LCD (hidraw + usbfs bulk), `/var/lib/kraken-lcd/halted` | **None** (`PrivateNetwork=yes`) |
| `llama-light` | system unit, user `llama-light` | The snapshot; `light.toml`; the Aura node's sysfs ids | The Aura controller's and the keyboard lighting interface's hidraw nodes (colour only) | **None** (`PrivateNetwork=yes`) |
| `llama-metrics` | system unit, user `llama-metrics` | The snapshot; `metrics.toml` | HTTP responses | **Listens** on TCP 19477 for a CIDR allowlist; dials nothing |
| `llama-view` | any user in group `llama-view` | `/dev/vcsa11` | Its own terminal | None |
| `llama-cast` | system unit, user `llama-cast` (group `llama-view`) | `/dev/vcsa11`, the console font, `cast.toml` | An H.264 MPEG-TS stream from one `ffmpeg` per viewer | **Listens** on a TCP port and UDP 1900 (SSDP) for a CIDR allowlist; dials nothing |

**Only `llama-metrics` and `llama-cast` listen (S16, S18).** No other crate
may name a listening socket, a datagram socket or `bind`; the exporter has
exactly one `TcpListener::bind`, and llama-cast one TCP listener and one UDP
socket on port 1900. The three snapshot readers (`kraken-lcd`, `llama-light`,
`llama-metrics`) get the snapshot through `SupplementaryGroups=llama-watch`
and parse it with the same validator in `llama-core`.

The workspace crates:

| Crate | Contents | Notable dependencies |
|---|---|---|
| `llama-core` | Snapshot schema v1 and its validator, the name sanitiser, the model-detail type and its line builders, the reset-aware rate helper, logging | `serde`, `serde_json`, `toml` |
| `llama-watch` | Sources (including the launch-command parser and the fan reader), collector and activity, llama poller, publisher, tty model and renderer | `nvml-wrapper` (dlopen), `ureq` |
| `kraken-lcd` | Snapshot reader, activity dial and tokens·24h history, presentation, V3b renderer and animation, upload policy, device layer, service | `nusb`, `tiny-skia`, `fontdue`. **No** `ureq` or `nvml-wrapper` (S10) |
| `llama-light` | Config and `[[light]]` mappings, metric and palette maths, the closed Aura and keyboard encoders, the pinned hidraw opens, the frame engine (tweening), service with live reload | `rustix`, `toml`. Dependency allowlist (S10/S15) |
| `llama-metrics` | CIDR allowlist, bounded HTTP/1.1 server, text exposition, service | `rustix`, `toml`. Dependency allowlist (S10/S16) |
| `llama-view` | vcsa reader and terminal painter | `rustix` |
| `llama-cast` | Config and allowlist, vcsa reader and PSF renderer, SSDP responder, DLNA MediaServer (description, ContentDirectory), bounded HTTP server, `ffmpeg` encoder process | `rustix`, `toml`. Dependency allowlist (S10/S18) |

`unsafe` is forbidden workspace-wide. Release builds use fat LTO and
`panic = "abort"`.

## Data flow

```
 /proc/stat, /proc/meminfo ────────────────┐
 hwmon: z53, k10temp/coretemp, zenergy ────┤
 hwmon: Super-I/O fans (tty only, opt-in) ─┤
 NVML (GPU util, temp, power, VRAM) ───────┼──► llama-watch ──► /run/llama-watch/snapshot.json (10 Hz, atomic rename, 0640)
 llama-swap on loopback ───────────────────┘         │                         │
   /running (+ launch cmd), /upstream/<m>/metrics,   │                         │ read-only (group llama-watch)
   /upstream/<m>/slots, /api/metrics/activity        ▼                         ▼
                                              tty11 (Alt+F11)            kraken-lcd writer
                                                     │                   activity dial + tokens·24h ─► present ─► render ─► policy
                                                     ▼                         │                                              │
                                              llama-view (vcsa11)              │ cooling guard reads z53 sysfs                ▼
                                                                               └──────────────────────────────────► Kraken LCD (hidraw + usbfs EP 0x02)

 snapshot.json ──► llama-light ──► [[light]] mappings ──► Aura USB controller (hidraw, direct colour, RAM only)
                                                         └──► Corsair keyboard lighting interface (hidraw, per-key colour, RAM only)
 snapshot.json ──► llama-metrics ──► GET /metrics on :19477 ──► Prometheus on the allowed LAN
```

- **Watcher cadences:** local reads at 10 Hz (CPU % over a sliding 1 s
  window). `/running` at 2 Hz. `/metrics` at 4 Hz, only while a model is ready,
  because an upstream call must never trigger a model load. `/slots` at 1 Hz,
  only while a request is processing. tty11 at about 10 fps.
- **Composite load** = `max(gpu_util, mean of the k busiest CPUs)` (k = 8 by
  default). One stray thread cannot peg the ring, and an all-core job still
  reads 100 %.
- **Activity** = the bottleneck device's power headroom. For each device,
  `frac = (watts − idle) / (nominal_frac × limit − idle)`, clamped to
  0–1.25, and `activity = max(gpu_frac, cpu_frac)`, smoothed (`smooth_s`). The
  GPU uses NVML power and its enforced limit; the CPU uses zenergy socket
  energy and `[load] cpu_limit_w`. `nominal_frac` (0.8) of a limit reads
  100 %, so 100 is sustained heavy load and spikes reach the 125 peg. With
  `idle = "auto"` the idle floors are the lower of the configured watts and the
  lowest draw seen since start; they never rise. A device without watts drops
  out; with neither, activity is `max(GPU util, CPU mean)`. The tty ACTIVITY
  row names the source (`gpu`, `cpu` or `util`).
- **Model detail.** The launch command in `/running` is untrusted text. It is
  parsed once inside the deserialiser into a `ModelDetail` (context size, KV
  cache types, a GGUF quant tag, CPU MoE layers, flash attention), keeping only
  numbers and short `[A-Za-z0-9_.+-]` tokens, never a path. The command string
  is dropped.
- **Tokens cross the boundary as a counter**, not a rate: a monotonic
  `decoded_total` per watcher run. The writer turns deltas into a 40-point
  integer cascade (10 × 30 s, 11 × 5 min, 10 × 30 min, 9 × 2 h = 24 h) for the
  tokens·24h chart. A counter drop, a `run_id` change or a stall longer than
  `max_gap_s` is a gap, never a negative.
- **Writer tick** is 0.5 s. It reads the snapshot, feeds the activity history
  and the token cascade, and builds a quantised View with hysteresis. *Change*
  mode uploads when the View changes, at most once per `min_interval_s` (floor
  10 s). *Stream* mode renders and uploads every tick at `stream_fps` (1–12,
  default 10), smoothing the gauge per frame and animating the redline.
- **LCD (V3b Plasma Blackbody).** A 270° gauge, 0–125 from 7:30 to 4:30,
  coloured by position, with a 100–125 redline; above 100 the head goes hot
  orange-gold, and at the peg deterministic smoke and embers rise. Inside it,
  the activity dial: 24 bars in five tiers (10 × 0.5 s, 2 × 5 s, 3 × 15 s,
  4 × 1 min, 5 × 5 min = 30 min), each the mean activity of its window, with
  tier ticks and labels as a time scale. A bar below `xff` coverage draws
  nothing, never zero. The centre has the model name (up to two lines, head and
  tail kept), the detail line fitted to the chord, the tokens·24h chart (A1),
  temperatures and CPU/MEM.
- **tty only.** Per-slot context history (a max-per-bucket cascade out to
  `ctx_history_h`, with reset flags), the fan readings, request history and
  llama text stay in the watcher. The snapshot does not carry them.
- **llama-light** reads the snapshot at up to `fps` (1–20, default 10),
  evaluates each `[[light]]` entry in order (later entries draw over
  earlier ones), caps every LED at `brightness_max`, and sends a frame only
  when it changes. `tokens_rate` is derived from the `decoded_total` counter.
  `light.toml` is re-read within 2 s of a change; an invalid file is logged
  once and the running config stays. A missing metric shows `idle_color` or
  a dim grey. On a splitter (`fans = "mirrored"`) the frame is
  `leds_per_fan` long and every fan shows it; on a daisy chain
  (`fans = "chain"`) fan k is LEDs `k × leds_per_fan` onwards.
- **llama-metrics** reads the snapshot per scrape and renders gauges
  (`llamabored_activity_pct`, `_load_pct`, `_cpu_pct`, `_cpu_topk_pct`,
  `_gpu_pct`, `_mem_pct`, `_coolant_celsius`, `_cpu_celsius`,
  `_gpu_celsius`, `_ai_state`, `_model_loaded`, `_model_ctx_size_tokens`),
  the counter `llamabored_tokens_decoded_total`, and health series
  (`_snapshot_up`, `_snapshot_stale`, `_snapshot_age_seconds`,
  `_snapshot_seq`, `_exporter_build_info`,
  `_exporter_rejected_connections_total`). A stale snapshot drops the value
  series. Labels carry only model display names and allowlisted tuning
  tokens.
- **No persistence.** History and the dial live in memory and refill after a
  restart. The snapshot lives on tmpfs.

## Snapshot schema v1

`/run/llama-watch/snapshot.json` is a compile-time constant path. The
watcher writes `snapshot.json.tmp` with `O_NOFOLLOW` and mode 0640, then
renames it over the old file.

```json
{
  "schema": 1,
  "run_id": 17461234567890123,
  "seq": 48213,
  "t_mono_ns": 912345678901234,
  "t_wall_ms": 1790200000123,
  "host": {
    "load_pct": 71.2, "activity_pct": 108.5, "cpu_pct": 12.5, "cpu_topk_pct": 41.0,
    "gpu_pct": 71.2, "mem_pct": 11.4, "coolant_c": 34.1, "cpu_c": 79.3, "gpu_c": 75.0
  },
  "ai": { "state": "loaded", "models": [ {
    "name": "Qwen3 35B A3B", "state": "ready",
    "full_name": "Qwen3 35B A3B Instruct",
    "detail": { "ctx": 262144, "ncmoe": 16, "kv_k": "q8_0", "kv_v": "q8_0", "quant": "UD-Q4_K_M", "fa": true }
  } ] },
  "tokens": { "decoded_total": 1234567 }
}
```

| Field | Type | Rule the writer enforces |
|---|---|---|
| `schema` | u8 | Must be `1` |
| `run_id` | u64 | Random per watcher start. A change starts a new token epoch |
| `seq` | u64 | Strictly increasing within a `run_id` |
| `t_mono_ns` | u64 | `CLOCK_MONOTONIC`, used for staleness. Not in the future |
| `t_wall_ms` | u64 | Logs only |
| `host.*_pct` | f32 or null | 0–100 |
| `host.activity_pct` | f32, null or absent | 0–125 (100 = nominal sustained load; spikes above). Absent from an older watcher |
| `host.*_c` | f32 or null | −20 to 150 |
| `ai.state` | enum | `down`, `idle` or `loaded` |
| `ai.models` | array, ≤ 8 | Empty unless `loaded`. `name` is printable ASCII, ≤ 13 chars, re-sanitised by the writer. `state` is `ready`, `starting`, `stopping` or `other` |
| `ai.models[].full_name` | string, optional | Untruncated display name, ≤ 48 printable ASCII chars; omitted when equal to `name` |
| `ai.models[].detail` | object, optional | `ctx` (u32), `ncmoe` (u16; 65535 = `--cpu-moe`), `kv_k`/`kv_v`/`quant` (tokens of `[A-Za-z0-9_.+-]`, ≤ 16 chars), `fa` (bool). All fields optional, unknown fields refused |
| `tokens.decoded_total` | u64 or null | Monotonic within a `run_id` |

The writer opens it with `O_NOFOLLOW|O_NONBLOCK`, requires a regular file of at
most 16 KiB, and parses with `deny_unknown_fields` on every struct, then range
checks. Anything invalid is treated as stale. **No llama text, error strings or
free-form data cross this boundary**; model names and allowlisted detail tokens are the only strings.

**Staleness.** A snapshot older than `stale_after_s` (default 1 s) means
*watch down*. The LCD shows "no data" (distinct from "AI down"), and history
and the dial record gaps, not zeros. After `watch_down_stock_after_s` (30 s)
the writer hands the screen back to the stock readout, at most once per
`watch_down_restore_min_s`. That restore counts against the same device-write
budget as uploads.

## Trust boundaries

| Boundary | What crosses | Controls |
|---|---|---|
| **B1** llama-swap → watcher | HTTP JSON and text from loopback. Prompt text is attacker-influenced | Loopback-only URL (validated), unit `IPAddressAllow=localhost`. Per-endpoint timeouts and byte caps (up to 4 MiB for `/slots`). Declared fields only. No redirects or proxies. Text is cut to a tail and sanitised inside the poller. The launch command is reduced to numbers and allowlisted tokens and dropped |
| **B2** watcher → tty11 | Console cells | Only one module writes bytes, from a fixed escape allowlist. llama text is reduced to printable ASCII before it becomes cells. `ESC % G` is sent on every repaint. The only ioctl is `tcgetwinsize`. llama text never reaches the journal. `prompt_view = "clean"` strips chat-template tokens before the sanitiser, which still runs last. With `[tty] show_text = false` (or `--no-text`) the watcher does not keep llama text at all |
| **B3** watcher → writer | The snapshot file, across two uids | The schema and checks above. A hostile watcher can at worst show wrong numbers or "no data". It has no path to the device, the guard or the latch, and flapping cannot raise the write rate |
| **B4** writer → cooler | 64-byte HID reports and bulk frame data | The closed command set, the cooling guard and the HALTED latch. See [SAFETY.md](SAFETY.md) |
| **B5** writers ↔ network | Nothing | `kraken-lcd` and `llama-light`: `PrivateNetwork=yes`, `RestrictAddressFamilies=AF_UNIX` (sd_notify only), `IPAddressDeny=any`, and no network crates |
| **B6** watcher → llama-light | The snapshot | The same validator as B3. A hostile snapshot can at worst pick wrong colours. Every LED is capped at `brightness_max`, and the frame rate at `fps` |
| **B7** llama-light → Aura controller | 65-byte HID output reports | A closed encoder with two opcodes (`0x35` direct mode on channel 1, `0x40` colours on direct channel 0) and no save/commit. Write-only open of the one node behind `/dev/llama-light/aura`, checked against sysfs before and `fstat` after. See [SAFETY.md](SAFETY.md) |
| **B7b** llama-light → keyboard lighting interface | 64-byte HID output reports | A closed encoder with four report shapes (software and hardware mode, a colour stream packet, a 24-bit commit) and no profile, firmware or stored-lighting writes, and no reads. Write-only open of the one node behind `/dev/llama-light/keyboard` (USB interface 1, usage page `0xFFC2`), checked against sysfs before and `fstat` after. Typing is on another interface and untouched. See [SAFETY.md](SAFETY.md) |
| **B8** LAN → llama-metrics | HTTP requests | In-process CIDR allowlist checked before a byte is read, and the same list in the unit's `IPAddressAllow=`; `SocketBindAllow=` pins the port. `GET /metrics` only; caps on request line, head size and header count, one deadline for the head (slowloris gets `408`), `max_conns` workers then `503`. No request body is read. The response holds numbers, model names and allowlisted tokens only |

| Actor | May write | May read |
|---|---|---|
| `llama-watch` | `/run/llama-watch/`, tty11 (an fd inherited from systemd), the journal | `/proc`, hwmon (the `[fans]` chip only by reading `fanN_input`, `pwmN`, `pwmN_enable`), NVML devices, loopback llama-swap. It **cannot** open the cooler's nodes |
| `kraken-lcd` | The cooler's LCD, `/var/lib/kraken-lcd/halted`, the journal | The snapshot, the cooler's own sysfs (guard and open checks), its hidraw and usbfs nodes |
| `llama-light` | The Aura controller's and the keyboard's colours, the journal | The snapshot, `light.toml`, the Aura node's sysfs ids and the keyboard's sysfs ids and usage page. No Kraken, no hwmon, no i2c |
| `llama-metrics` | Its sockets, the journal | The snapshot and `metrics.toml`. No `/proc` (beyond its own), `/sys` or `/dev` |
| `llama-view` users | Their own terminal | `/dev/vcsa11` (group `llama-view`, 0640) |

**Device access.** udev gives the cooler's usbfs node and hidraw node to group
`kraken-lcd`, mode 0660, and strips `uaccess` ACLs from the hidraw node. It
also adds a stable symlink, `/dev/kraken-lcd/hid`. The writer unit's only
hidraw grant is `DeviceAllow=/dev/kraken-lcd/hid`, so it cannot open any
other hidraw device (keyboards and so on), whatever their file modes. Its
usbfs grant, `DeviceAllow=char-usb_device rw`, is class-wide: there the
kernel's cgroup check allows every USB device node, and what narrows it to
the cooler is the udev group and mode (other USB nodes stay root-owned with
their distro modes) and the writer opening only the bus and device number it
reads from the cooler's sysfs directory (SAFETY.md RR7c). systemd
resolves that path when the unit starts. After a USB re-enumeration the node can
get a new number, and the writer then needs a restart to reach it. The writer
claims **only USB interface 0** (bulk) and never detaches a driver, changes the
configuration or resets the device. The `nzxt-kraken3` hwmon driver stays bound
to the HID interface.

The Aura controller and the keyboard's lighting interface get the same
treatment: `94-llama-light-hidraw.rules` gives their hidraw nodes to group
`llama-light`, mode 0660, strips `uaccess`, and adds `/dev/llama-light/aura`
and `/dev/llama-light/keyboard`, the llama-light unit's only two
`DeviceAllow=` entries.

**Units.** All four system units use `NoNewPrivileges`, an empty capability set,
`ProtectSystem=strict`, `ProtectKernelTunables`, `DevicePolicy=closed`, a
`@system-service` syscall filter, `MemoryDenyWriteExecute`, and a systemd
watchdog (LCD writer 30 s, the others 10 s). Per unit:

- `llama-watch`: `ReadOnlyPaths=/sys`; `DeviceAllow=` for `/dev/nvidiactl`,
  `/dev/nvidia0` and `/dev/tty11`; `IPAddressAllow=localhost`. Two
  `ExecStartPre=-+` steps run as root, outside the sandbox (the `+` prefix),
  before the service starts, and their failure is ignored (the `-`): `setfont`
  loads the bundled font on tty11 and `setterm` sets its power-down mode.
- `kraken-lcd`: `ReadOnlyPaths=/sys`, `ProcSubset=pid`, `PrivateNetwork=yes`;
  `DeviceAllow=char-usb_device` and `DeviceAllow=/dev/kraken-lcd/hid`;
  `restore-stock` in `ExecStopPost`.
- `llama-light`: `ReadOnlyPaths=/sys`, `ProcSubset=pid`, `PrivateNetwork=yes`;
  `DeviceAllow=/dev/llama-light/aura` and `/dev/llama-light/keyboard`;
  `restore` (a neutral colour, never saved) in `ExecStopPost`.
- `llama-metrics`: `InaccessiblePaths=/sys`, `PrivateDevices=yes` and no
  `DeviceAllow=`, `ProcSubset=pid`, `IPAddressAllow=` (its allowlist),
  `IPAddressDeny=any` and `SocketBindAllow=tcp:19477`.

`scripts/stage.sh` pins the full directive set of the writer, watcher and
exporter units, and checks that the exporter's `IPAddressAllow=` equals
`allow` in `packaging/metrics.example.toml`. The writer does not depend on the watcher:
it must run, and show "no data", when the watcher is absent.

## Safety fences

Automated checks in `scripts/check.sh` and the test suite. They run on every
build.

| Fence | What it checks |
|---|---|
| **S1** source scan | Only the HID module writes to a device file, and only the bulk module names `nusb`. No `detach_and_claim_interface`, `set_configuration`, `reset`, control transfers, `ioctl` or `unsafe` anywhere. File writes only at approved sites (hidraw, the latch, `clear-halt`, `render-once --out`) |
| **S2** command table | Every constructible command encodes to an allowlisted two-byte prefix. An out-of-table prefix is refused at runtime. No report can start with `70` (init), `72` (pump/fan duty) or `30 02` (brightness/orientation) |
| **S3** read-only roots | No writes to `/sys` or `/proc`. A full cycle succeeds on read-only fake roots |
| **S6** binary symbols | The release writer contains no USB detach, configure, reset or control-transfer symbols, and links only glibc. A secondary check; S1 is the control |
| **S7** cooling guard | Every deviation halts and latches. After HALTED the fake device records zero further bytes, across crashes and restarts. Firmware-driven `pwm` value changes do not halt |
| **S8** abort after WriteStart | A failure mid-transfer sends exactly one "show stock" and then nothing, and drops all handles |
| **S9** bucket hygiene | Only foreign or overlapping buckets are deleted. `restore-stock` sends exactly one command |
| **S10** writer dependencies | The LCD writer's dependency tree has no `ureq`, `nvml-wrapper`, `libloading`, `rustls`, `native-tls` or `http` (`cargo tree` and `cargo deny`). `llama-light` and `llama-metrics` must match exact allowlists (`scripts/s15-light-allow.txt`, `scripts/s16-metrics-allow.txt`) |
| **S11** writer reads nothing | The writer source has no `std::net`, no `/proc` literal and no NVML. Files are opened only in allowlisted modules, and the snapshot path is a constant |
| **S12** console sanitiser | Hostile text (ESC/CSI/OSC, C1, UTF-8 that could decode to C1, bidi, 10 MB inputs) yields only printable ASCII cells, and every high byte in the output is one of our own glyphs |
| **S13** tty syscalls | The tty code's only termios or ioctl call is `tcgetwinsize`. No `VT_`, `KD` or `TIOCSCTTY` |
| **S14** fans read-only | The fan source and its hwmon helpers contain no write-capable file API (`write`, `OpenOptions`, `File::create`, rename, remove, ...) |
| **S15** llama-light colour-only | No network, process control, i2c/SMBus, NZXT id, pwm or hwmon; filesystem calls only in allowlisted files, no filesystem writes, fixed path literals; the opcode table in `aura/proto.rs` is exactly `0x35` and `0x40`, and save/config opcodes (`0x3F`, `0x3E`, ...) never appear. Each rule is shown to bite on a planted violation |
| **S18** llama-cast | One TCP listener and one UDP socket on port 1900, no outbound connection, no Unix sockets, no `/sys`, `/proc`, hidraw, USB, NVML or llama API paths; reads only `/dev/vcsa11`, the font, the machine id and its config; no file writes; exactly one `Command::new` (the configured `ffmpeg`, fixed arguments, `env_clear()`); fixed path literals and pinned dependencies |
| **S16** listeners | Only `llama-metrics` and `llama-cast` name a listening socket, datagram socket or bind, with exactly one `TcpListener::bind`; it has no `/sys`, `/dev`, hidraw, USB, NVML, outbound socket, process spawn, `unsafe` or file write, and its dependencies are pinned |

S4 and S5 are unused numbers. Also in the gate: `cargo fmt`, `clippy -D
warnings`, `cargo audit`, the installer and packaging self-tests, and a
byte-for-byte rebuild of the console font.

## Device protocol (summary)

The writer implements its own minimal Kraken Z3 LCD protocol. It was written
from captured traffic, with liquidctl and KrakenZPlayground read as protocol
references; no code was ported. HID reports are exactly 64 bytes on hidraw.
Replies are matched by prefix, and anything else (the device's periodic status
reports) is skipped. Frames are 320x320x4 bytes, sent as 512-byte bulk
transfers on endpoint `0x02`. Change mode rotates through device buckets 0–7,
and stream mode ping-pongs between buckets 0 and 1. The full command set:

| Command | Purpose |
|---|---|
| `30 01` | LCD info (orientation) |
| `30 04 b` | Query bucket `b` (read-only) |
| `32 01 …` / `32 02 b` | Set up / delete a bucket |
| `36 01 s` / `36 02` / `36 03` | Write start / end / pre-transfer |
| `38 01 04 s` | Show bucket `s` |
| `38 01 02 00` | Show the stock liquid screen |

If the setup is refused, no bulk data is sent. Any failure after a write has
started sends only "show stock" once, then drops every handle.
