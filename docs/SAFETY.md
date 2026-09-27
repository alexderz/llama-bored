# Safety

llama-bored writes to two devices that sit next to cooling. The NZXT Kraken
Z's USB device drives the LCD **and** the pump and fans of the CPU cooler;
kraken-lcd only wants the LCD. The ASUS Aura USB controller drives the
lighting on the fan headers; llama-light only wants the colours. This
document explains how both stay away from cooling, how the one networked
process (llama-metrics) is contained, what happens when something looks
wrong, and which risks remain.

## What "LCD-only" means

- **A closed command set.** The writer knows nine commands: LCD info, bucket
  query, bucket setup and delete, write start, end and pre-transfer, show a
  bucket, and show the stock screen. There is **no code, enum variant or
  constant** for pump or fan duty (`0x72`), init (`0x70`),
  brightness/orientation (`30 02`), built-in animations or firmware.
- **One way to build a report.** A HID report can only be built by the
  protocol encoder, which checks it against a const allowlist. The HID link
  accepts only that type, so there is no raw-bytes write path.
- **No device tricks.** The writer never detaches a kernel driver, changes the
  USB configuration, resets the device or sends control transfers. It claims
  only interface 0 (bulk frame data). The `nzxt-kraken3` hwmon driver stays
  bound and keeps reporting.
- **Never writes sysfs.** Cooling state is read from hwmon (read-only, and
  `/sys` is read-only inside the units). Nothing in the project writes
  `pwm*` or fan curves. The optional tty11 FANS panel reads `fanN_input`,
  `pwmN` and `pwmN_enable` of one motherboard hwmon, chosen by name, and fence
  S14 checks that its code has no way to write them.
- **No liquidctl at runtime**, and nothing calls `initialize`.
- **Checked on every build** by safety fences S1–S3 and S6–S16 (see
  [ARCHITECTURE.md](ARCHITECTURE.md#safety-fences)).

Cooling never depends on kraken-lcd. The cooler's firmware and the kernel
driver regulate the pump and fans whether it runs or not, so stopping or
crashing it changes nothing there.

## The cooling guard

Every device operation (upload, open-time checks, restore to stock, and the
one-shot `show-image`) is wrapped by a read-only guard.

**Baseline**, taken just before the operation:

- the `z53` hwmon directory exists (found by name, not by number);
- `pwm1_enable` and `pwm2_enable` (the pump and fan **mode**);
- `fan1_input` (pump rpm);
- the cooler's USB device number;
- no `1e71:3011` (bootloader) device present.

`pwm1`/`pwm2` **values** are logged but not compared. The firmware's own curve
moves them all the time, and comparing them would cause false halts.

**Checks**, right after the operation (success or failure) and again about 2 s
later:

- `z53` still present;
- `pwm*_enable` identical;
- pump rpm within **max(15 %, 150 rpm)** of the baseline. This band is a
  constant, not a config value;
- the same USB device number;
- still no bootloader device.

## The HALTED latch

Any failed check moves the writer to **HALTED**:

1. It logs CRITICAL with both snapshots.
2. It **stops all device I/O immediately**: no retries, and not even a "show
   stock" command.
3. It writes the latch file `/var/lib/kraken-lcd/halted` (fsynced).
4. It keeps running and feeding the systemd watchdog, so systemd does not
   restart it into another attempt.

The latch **persists** across crashes, restarts and reboots. While it exists,
`run` does no device I/O, and `restore-stock` (including the one in
`ExecStopPost`) exits without touching the device.

**Clearing it is a human decision.** After you have checked the cooler:

```sh
cat /var/lib/kraken-lcd/halted                                  # time and both snapshots
sudo /usr/local/libexec/llama-bored/kraken-lcd clear-halt      # root only; asks to confirm
sudo systemctl restart kraken-lcd
```

An optional user unit, `kraken-lcd-halt.path`, sends a desktop notification
when the latch appears (`sudo systemctl --global enable kraken-lcd-halt.path`).

## The RGB writer (llama-light)

- **Colour only.** llama-light reads the snapshot and writes lighting frames.
  It has no code path to fan speed, pump, pwm, hwmon, i2c/SMBus or the Kraken
  (fence S15), and its unit has `ReadOnlyPaths=/sys` and no network.
- **A closed opcode table.** The Aura encoder can emit exactly two opcodes:
  `0x35` (set effect channel 1 to Direct mode) and `0x40` (stream up to 20
  LED colours on direct channel 0). **There is no save-to-flash**: the commit
  opcode (`EC 3F 55`) and header configuration (`EC 3E`) are never built, and
  S15 fails if they, or `0x36`, `0xB0` or `0x82`, appear in the source.
- **Nothing is stored on the board.** Direct colours live in the
  controller's RAM. On stop, `ExecStopPost` runs `llama-light restore`, which
  sends one neutral static frame, again without a commit; the board's own
  stored effect returns at the next power cycle.
- **One pinned node.** udev (`94-llama-light-hidraw.rules`) gives the
  controller's hidraw node to group `llama-light`, mode 0660, strips
  `uaccess`, and adds `/dev/llama-light/aura`, the unit's only
  `DeviceAllow=`. Before opening, llama-light checks in sysfs that the node is
  HID `0003:0B05:18F3`; after opening (write-only) it checks with `fstat`
  that it opened that character device.
- **Bounded output.** Every LED is capped at `brightness_max` (default 80 %),
  which also limits header current, and frames are sent at most `fps` times
  a second, only when they change.
- **Keyboard: a closed encoder on the lighting interface.** udev pins only
  the STRAFE RGB MK.2's interface 1 (vendor usage page `0xFFC2`) to
  `/dev/llama-light/keyboard`; llama-light re-checks the usage page before
  opening it. Typing uses another interface and evdev, and is untouched. The
  encoder can emit four report shapes: software mode and hardware mode
  (`07 05`, a RAM mode switch), a colour stream packet (`7F`) and a 24-bit
  channel commit (`07 28`). Reset, special-function, firmware, poll-rate,
  key-routing, hardware-profile and stored-lighting writes, and all reads,
  are never built, and S15 fails if their bytes appear. On stop,
  `llama-light restore` sends hardware mode, so the keyboard shows its own
  lighting again. Unplugged: logged once, retried every 10 s.

## The network exporter (llama-metrics)

- **Only llama-metrics listens (S16).** No other crate names a listening
  socket or `bind`. The exporter binds exactly once, and the unit pins the
  port with `SocketBindAllow=tcp:19477` / `SocketBindDeny=any`.
- **Two copies of the allowlist.** The config's `allow` is checked in process
  before a byte is read; the unit's `IPAddressAllow=` (with
  `IPAddressDeny=any`) makes the kernel enforce the same list. `/0` and
  networks with host bits set are refused. The shipped example admits
  `192.168.0.0/16` and loopback; narrow it to your LAN.
- **Nothing sensitive to reach.** It reads only the snapshot (which carries no
  llama text) and its config. It has `PrivateDevices=yes`, no `/proc` beyond
  its own, no `/sys`, no hidraw or USB, no outbound connections and no file
  writes (S16), and runs as its own user.
- **Strict HTTP.** `GET /metrics` only; request-line, head-size and
  header-count caps; one deadline for the whole head; a fixed worker pool
  (`max_conns`) with `503` beyond it. No body is read.
- **Off by default.** The installer never enables it and never touches the
  firewall.

## Other failure behaviour

| Event | What happens | The screen shows |
|---|---|---|
| Writer stopped, crashes or panics | `ExecStopPost` runs `restore-stock` | Stock coolant readout |
| Writer hangs | Watchdog (30 s) kills it, then `ExecStopPost` | Stock within about 30 s |
| Upload refused before data is sent | Counted; after `fail_limit` (3) in a row: stock, exit, bounded restarts | Stock |
| Failure mid-transfer | One "show stock", all handles dropped, reopen with backoff | Stock until the next good upload |
| Device unplugged, re-enumerated, resumed | Reopen with backoff, then a fresh upload. If the hidraw node got a new number, the open keeps failing safely until `systemctl restart kraken-lcd` | Ours again, or stock until the restart |
| Bootloader (`1e71:3011`) | No device I/O until restart; recovery is a full power-off | Whatever the device shows |
| Watcher gone or stale | "no data", then stock after 30 s. Never trips the guard | "no data", then stock |
| llama-light stopped or crashed | `ExecStopPost` sends one neutral static frame (no save) | Neutral colour; board effect at next power cycle |
| Aura controller absent or re-enumerated | One log line; looks again periodically without more logs. A new node number needs `systemctl restart llama-light` | Whatever the board shows |
| `light.toml` edited to something invalid | Logged once, ignored; the running mapping stays | Unchanged |
| Snapshot stale (exporter) | Value series dropped, `llamabored_snapshot_stale 1` | n/a |

## Tested finding: frame memory is RAM

The Kraken stores uploaded frames in on-device "buckets". If those were
flash, a high upload rate would wear it out. So before any higher rate was
allowed, a power-cycle test was run on a Z53. One frame was uploaded, the PSU
was switched off for 30 s, and the bucket table was queried read-only after
boot.

**All 16 buckets were empty, and the LCD booted to the stock screen.** Bucket
memory is RAM. Wear is not a concern. Stream mode (10 fps) was then validated
with a watched ramp (about 63 ms per full frame, so a ceiling near 15 fps) and a
6,000-frame soak at 10 fps. A 1 Hz cooling monitor ran throughout and found no
anomalies. Nothing on the device needs cleaning up after uninstall.

## Risks and their status

| Id | Risk | Mitigation | Status |
|---|---|---|---|
| RR1 | "LCD-only" is enforced by our code; the kernel does not filter what a holder of the hidraw/usbfs handle sends | Closed command enum, a single encoder, no raw write path, no `unsafe` or `ioctl`, fences S1/S2, the sandbox, a group-only udev grant, and the cooling guard | **Accepted** |
| RR2 | Some distros leave the cooler's hidraw node world-writable | The installer narrows it to `0660 root:kraken-lcd`, removes ACLs, and verifies | **Closed** |
| RR3 | The device is opened before safety checks | All sysfs checks run before any open; no detach, configure or reset call sites | **Closed** |
| RR4 | Bucket memory might be flash (wear) | Power-cycle test: it is RAM | **Retired** |
| RR5 | A broken runtime could stop `ExecStopPost` from restoring stock | One native binary on glibc; `restore-stock` uses no optional libraries | **Closed** |
| RR6 | The build tree and toolchain are writable by the build user, whose build root installs | The installer refuses a dirty tree, an unreleased HEAD, or a hash mismatch, and root never runs git or cargo. This protects against accidents, not against a compromised build account | **Accepted** |
| RR7 | A class-wide `DeviceAllow=char-hidraw` would let the writer open every hidraw node (keyboards and so on) | Closed by pinning: udev adds `/dev/kraken-lcd/hid` for the cooler's node only, the writer unit's single hidraw grant is `DeviceAllow=/dev/kraken-lcd/hid`, and the installer checks that the symlink resolves to that node. The writer cross-checks the node against the cooler's sysfs path before opening it (S1). Trade-off: after a USB re-enumeration the writer needs a restart | **Closed** |
| RR7b | The Aura controller's and the keyboard lighting interface's hidraw nodes may also be world-writable or `uaccess`-tagged by distro or OpenRGB rules | `94-llama-light-hidraw.rules` sorts after them, sets `0660 root:llama-light` and strips `uaccess`; the unit is pinned to `/dev/llama-light/aura`, checked against sysfs and `fstat` | **Closed** |
| RR8 | While HALTED, our last frame can stay on screen and look live | Deliberate: no device I/O beats a fresh screen once cooling looks wrong. CRITICAL log, `systemctl status`, optional desktop alert | **Accepted** |
| RR-LV1 | Prompts and outputs are readable on tty11 (and through llama-view) by anyone at the console, KVM or remote console, or in the `llama-view` group | `[tty] show_text = false` (or `llama-watch run --no-text`) removes the IN/OUT panels, the watcher stops keeping llama text at all, and the header shows "text off". The install skill asks the operator. Text appears only if `LLAMA_SERVER_SLOTS_DEBUG=1` is on. Tails only, no scrollback, and the VT is cleared on stop | **Operator's choice** (default: shown) |
| RR-LV2 | With `LLAMA_SERVER_SLOTS_DEBUG=1`, llama-server's `/slots` returns whole prompts to **anyone who can reach llama-swap** | Keep llama-swap bound to loopback, or put a proxy in front that blocks any `slots` path for other clients. llama-watch must reach llama-swap directly | **Operator's responsibility** |
| RR-LV3 | The watcher unit ends any other session on tty11 | tty11 is reserved for the dashboard (`Conflicts=` the tty11 getty) | **Accepted** |
| RR-LV4 | LCD numbers other than coolant, the RGB colours and the exported metrics come from the watcher; a compromised watcher could show plausible false values | Separate sandboxed uid with no device access. Coolant, the guard and the latch use only the writer's own reads; colours are capped by `brightness_max` and `fps` | **Accepted** |
| RR-L1 | An RGB write path next to fan headers could be misused to change fan behaviour or persist settings | Colour-only crate (S15), two-opcode encoder with no save/commit, no i2c/hwmon/pwm, `ReadOnlyPaths=/sys`, one pinned node, restore without commit | **Closed** |
| RR-L2 | Direct mode overrides the board's stored effect while llama-light runs | By design; nothing is saved, and the stored effect returns at the next power cycle | **Accepted** |
| RR-M1 | llama-metrics is reachable from the LAN | Off by default; in-process allowlist plus `IPAddressAllow=`; pinned port; strict HTTP limits; no device, `/sys` or outbound access (S16). Opening the firewall is the operator's step | **Operator's choice** |
| RR-M2 | Scrapers learn model names, tuning tokens and activity patterns | Numbers and allowlisted names only, never prompt or output text (the snapshot has none). Restrict `allow` to the Prometheus host if that matters | **Accepted** |

## If you are changing the code

- Adding a device command is a design change, not a patch. It needs a new
  safety review, and fence S2 will fail until the allowlist is changed on
  purpose.
- The writer crates must never gain a network or NVML dependency (S10, S11,
  S15). Adding an Aura opcode is a safety review, like a Kraken command.
- Only llama-metrics may listen (S16).
- The snapshot schema is frozen at v1. Any change bumps `schema`.
- Record `pwm*_enable`, pump rpm and coolant before and after any
  device-facing test. `scripts/cooling-snapshot.sh` prints one line, and
  `scripts/ramp-monitor.sh` is a 1 Hz monitor for longer runs. Both are
  read-only.
