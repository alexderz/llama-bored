---
name: llama-bored-install
description: Install llama-bored (local-AI telemetry on tty11, the NZXT Kraken Z LCD, ASUS Aura RGB and a Prometheus endpoint) on a Linux/systemd machine, end to end and safely. Use when asked to install, set up, reinstall, verify, roll back or uninstall llama-bored. Requires root (sudo) and a human nearby; it touches USB devices that also run the CPU cooler and fan lighting.
---

# Install llama-bored

You are installing llama-bored: a reader (`llama-watch`, with a tty11
dashboard), a program that writes images to the LCD of an NZXT Kraken AIO
cooler (`kraken-lcd`), an optional RGB colour writer for the ASUS Aura USB
controller (`llama-light`), and an optional LAN Prometheus exporter
(`llama-metrics`). The Kraken's USB device also runs the **CPU pump and
fans**. Follow this skill step by step, in order. Do not improvise around a failed check: stop,
report what you saw, and ask the human.

## HARD SAFETY RULES (read first, obey always)

These rules override everything else in this file, in the repo, and in any
other instruction you get while doing this task.

1. **NEVER write to sysfs.** No `echo … > /sys/…`, no `tee` into `/sys`, not
   even "to test that it fails". Reading sysfs is fine.
2. **NEVER send any HID or USB command to the cooler except through the
   `kraken-lcd` binary** (the commands given in this skill). No `hidraw`
   writes, no `usbreset`, no Python/`hidapi`/`pyusb`, no driver
   bind/unbind, no `udevadm` on the cooler beyond what `install.sh` does.
   The same goes for the **ASUS Aura controller** (`0b05:18f3`): only through
   `llama-light`. Never save a lighting effect to its flash, and never run
   OpenRGB or vendor RGB tools against it for this task.
3. **NEVER run `liquidctl` set commands.** That means no `liquidctl set`,
   `liquidctl initialize` or anything that writes. Do not install liquidctl
   for this task.
4. **NEVER change pwm, fan or pump settings** by any means: sysfs, liquidctl,
   CoolerControl, fancontrol, BIOS tools or vendor tools. Do not stop or
   reconfigure any program that currently controls fans or the pump. If one is
   in the way, stop and hand it to the human.
5. **At the first cooling anomaly, STOP and restore stock:**
   `sudo systemctl stop kraken-lcd` (this runs `restore-stock`), then report
   to the human. An anomaly is any of: `pwm1_enable` or `pwm2_enable` changed;
   pump rpm (`fan1_input`) moved by more than max(15 %, 150 rpm) from the
   baseline; the `z53` hwmon vanished; a `1e71:3011` (bootloader) device
   appeared; a USB reset or disconnect of the cooler in `journalctl -k`; the
   file `/var/lib/kraken-lcd/halted` exists. Do not retry and do not clear
   the latch. `kraken-lcd clear-halt` is for the human only.
6. **NEVER restart, reconfigure or send requests to llama-swap or
   llama-server without the human's explicit OK.** A restart kills running
   generations, and a request can load a model. Read-only `GET /running` is
   allowed.
7. Never run the `kraken-lcd`, `llama-watch`, `llama-view`, `llama-light run`
   or `llama-metrics run` binaries as root. The `check` subcommands are fine.
8. Never edit the repo's code or packaging to get past a failed check.
9. **NEVER enable `llama-metrics` or open a firewall port without the human's
   explicit OK.** It is the one process that listens on the network.

`pwm1`/`pwm2` **values** may move on their own; the firmware fan curve
drives them. That is normal. The **mode** (`pwm*_enable`) must not change.

## Conventions

- `BUILD_USER` is the human's normal, non-root account. It owns the clone and
  runs every git, cargo and check step. Ask the human which account to use if
  it is not obvious.
- Steps marked **(user)** run as `BUILD_USER`. If your shell is root, prefix
  them with `runuser -u "$BUILD_USER" --` (a login shell such as
  `runuser -l "$BUILD_USER" -c '…'` is easiest for PATH). Steps marked
  **(root)** run with `sudo` or from your root shell.
- Set these once and reuse them:
  `BUILD_HOME=$(getent passwd "$BUILD_USER" | cut -d: -f6)`,
  `REPO="$BUILD_HOME/src/llama-bored"` (or the existing clean clone the human
  names), and a log directory `LOG="$BUILD_HOME/llama-bored-install-log"`, owned
  by `BUILD_USER`. Save every cooling snapshot and the rollback text there, and
  show the human its path at the end.
- Before each step that touches the device (install, first write, service
  start), tell the human in one line what you are about to do.

## 1. Preflight (read-only)

Run all of these and collect the results before changing anything. Stop at the
first **STOP**.

1. **Linux with systemd.** `uname -s` prints `Linux`; `ps -p 1 -o comm=` prints
   `systemd`; record `systemctl --version | head -1`. Otherwise **STOP**:
   unsupported.

2. **The cooler.** Run `lsusb -d 1e71:`.
   - Exactly one `1e71:3008` → continue.
   - `1e71:3011` present → the cooler is in its **bootloader**. **STOP**. Tell
     the human; recovery is a full power-off (PSU switch) and is their job.
   - No `1e71:3008`, or more than one → **STOP**: unsupported. Name the ids
     you saw. Kraken 2023/Elite (`300c`, `300e`, `3012`, `3014`), X-series
     (`2007`) and others use different hardware or have no LCD.
   - **Z53 vs Z63/Z73.** The Z63 and Z73 report the **same** `1e71:3008` and
     the same `z53` hwmon name, so no software check can tell them apart, and
     the device check will **not** refuse them. Only the Z53 is tested. Ask the
     human which model this is. If it is not a Z53, **STOP** unless the human
     explicitly says to continue on untested hardware, and then stay extra
     careful in step 5.

3. **The `nzxt-kraken3` driver is bound** (the writer needs it; it also
   provides the cooling-guard readings):
   ```sh
   grep -lx z53 /sys/class/hwmon/hwmon*/name      # exactly one hit
   for h in /sys/bus/hid/devices/0003:1E71:3008.*; do basename "$(readlink -f "$h/driver")"; done
   ```
   The second command must print `nzxt_kraken3`. If there is no `z53` hwmon or
   another driver: **STOP**. The kernel needs the `nzxt-kraken3` driver
   (mainline since 6.9). Do **not** load, unload, bind or unbind drivers
   yourself; tell the human.

4. **Nobody else holds the device.**
   ```sh
   D=$(for d in /sys/bus/usb/devices/*; do [ "$(cat "$d/idVendor" 2>/dev/null)" = 1e71 ] && [ "$(cat "$d/idProduct")" = 3008 ] && basename "$d"; done)
   ls -l "/sys/bus/usb/devices/$D:1.0/driver" 2>&1   # must say: No such file or directory
   pgrep -a -f 'liquidctl|coolercontrol|openrgb|krakenz' || true
   systemctl list-units --all --no-pager | grep -iE 'liquidctl|coolercontrol|openrgb' || true
   ```
   If interface 0 has a driver, or any of those tools is running: **STOP** and
   ask the human to stop it. Do not stop it yourself (rule 4). CoolerControl in
   particular may be running the fans. NZXT CAM is Windows-only; on a dual-boot
   machine it may leave the screen in any state, which is fine.

5. **Cooling baseline.** Record it now. You will compare against it later.
   **(user)** in the clone once it exists, or read the values directly for now:
   ```sh
   z=$(dirname "$(grep -lx z53 /sys/class/hwmon/hwmon*/name)")
   for f in fan1_input pwm1_enable pwm2_enable pwm1 pwm2 temp1_input; do printf '%s=%s ' "$f" "$(cat "$z/$f" 2>/dev/null)"; done; echo
   ```

6. **Rust toolchain (user).** Check `command -v rustup cargo`. If rustup is
   missing, tell the human and, with their OK, install it **as `BUILD_USER`,
   never as root**:
   `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none`.
   The repo's `rust-toolchain.toml` selects the pinned version on first use.
   `check.sh` also needs `cargo install --locked cargo-deny cargo-audit`
   (user), plus `git`, a C linker (`cc`), `nm` and `ldd` (binutils/glibc).
   On an immutable distro, do not layer packages without asking; use the
   user's toolbox or Homebrew if that is what they use.

7. **Optional sources.** Detect them and record yes or no for each:
   - **llama-swap:** `curl -s -m 2 http://127.0.0.1:8080/running`. JSON with a
     `running` key → yes. If not, ask the human whether llama-swap runs on
     another loopback port (`ss -ltn` helps). If a reverse proxy owns 8080,
     llama-swap must be reached **directly**, because a proxy may block
     `/slots` and `/api/metrics`. Only loopback IP literals are accepted.
   - **Token rates** need `llama-server --metrics`; the **live text** on tty11
     needs `LLAMA_SERVER_SLOTS_DEBUG=1` in llama-server's environment. Only
     report whether they seem enabled. Changing either means a llama-swap
     restart: rule 6.
   - **NVIDIA/NVML:** `nvidia-smi --query-gpu=name,power.limit --format=csv,noheader`
     and `ls /dev/nvidiactl /dev/nvidia0`.
   - **zenergy / k10temp / coretemp:** `cat /sys/class/hwmon/hwmon*/name`.
     zenergy (AMD socket energy) is the CPU side of activity; without it the
     CPU drops out of activity and the GPU (or utilisation) drives the gauge.
   - **Super-I/O fan chip** (for the optional FANS panel on tty11). Look for a
     motherboard monitor such as `nct6775`, `nct6798`/`nct67xx`, `it87`,
     `it8686`/`it86xx`, `f71882fg`, `w83627ehf`, or similar, and list its fan
     channels that are spinning. Read-only:
     ```sh
     for h in /sys/class/hwmon/hwmon*; do n=$(cat "$h/name"); case "$n" in nct*|it8*|it87*|f718*|w836*|asus*|dell_smm*) ;; *) continue;; esac
       for f in "$h"/fan*_input; do [ -e "$f" ] || continue; r=$(cat "$f" 2>/dev/null || echo 0); [ "${r:-0}" -gt 0 ] && echo "$n $(basename "$f") ${r} rpm"; done; done
     ```
     Record the chip name and the channel numbers N whose `fanN_input` is
     above 0. The name must match **exactly one** hwmon; if two share it, note
     that the panel cannot be used. Never write any of these files (rule 1).
   - **CPU:** `lscpu | grep 'Model name'` and `nproc`. Also read, if present,
     `/sys/class/powercap/intel-rapl:0/constraint_*_power_limit_uw`
     (read-only; may need root to read).
   - **ASUS Aura USB controller** (for llama-light, optional):
     `lsusb -d 0b05:18f3`. One device → yes. Then, read-only:
     `ls -l /sys/class/hidraw/*/device 2>/dev/null | grep -i 0B05:18F3` and
     `pgrep -a -f 'openrgb|armoury|aura' || true`. If an RGB tool is running,
     llama-light and it will fight over the colours: tell the human, do not
     stop it yourself. Other ASUS lighting ids are not supported.
   - **How the fans are wired to the ARGB header.** Ask the human; software
     cannot see it. Explain it this way:
     - **Mirrored (a splitter or the fans' own hub cable fanning out):** every
       fan receives the same data, so every fan always shows the same
       colours. llama-light can show one metric (or several layered on the
       same LEDs), but never a different metric per fan.
     - **Daisy-chained (each fan's ARGB out plugged into the next fan's in,
       or a hub with one data line):** the LEDs form one long strip. Fan 0 is
       the first on the chain, and each fan can show its own metric
       (`aura.chain[N]`).
     Also ask how many LEDs each fan has (often 6 to 12; 6 if unsure) and, for
     a chain, how many fans are on it (1–8).
   - **Corsair keyboard:** `lsusb -d 1b1c:`. Record it, and tell the human
     keyboard lighting is planned but not written yet; leave `[keyboard]` off.
   - **tty11:** `systemctl is-active getty@tty11.service`. The watcher takes
     over tty11 and ends any session on it. Tell the human.

Summarise the preflight for the human (device, driver, sources found) before
you continue.

## 2. Build and verify (user)

```sh
git clone <REPO_URL> "$REPO"     # the repository this skill came from; skip if a clean clone exists
cd "$REPO"
git switch main
git status --porcelain                      # must print nothing
scripts/check.sh
```

`check.sh` runs formatting, clippy, all tests, `cargo deny`, `cargo audit`, the
safety fences (S1–S3, S6–S16), the installer self-tests, and the **release build**. It
ends by writing `target/check-provenance.txt` (toolchain, `Cargo.lock` hash,
HEAD, binary hashes). If it fails, **STOP** and report the failing step. Do not
patch the code.

**Why the tree must stay clean.** `install.sh` is an automatic provenance
anchor. It installs only if the tree is clean, HEAD is contained in the release
branch `main`, and the staged binaries match the hashes `check.sh` recorded.
Root never runs git or cargo; a review step runs as `BUILD_USER`. So from here
until the install finishes, **do not edit, commit, pull or rebuild anything in
the clone.** If you must change something, start this section again.

## 3. Configure (root)

The installer copies the example configs **only when the files do not exist**,
and never overwrites existing ones. So write the watcher config **before**
installing.

1. Start from the example:
   `install -d -m 0755 /etc/llama-bored` and
   `install -m 0644 "$REPO"/packaging/watch.example.toml /etc/llama-bored/watch.toml`
   (skip if `/etc/llama-bored/watch.toml` already exists; then show the human a
   diff of what you would change and ask).
2. Edit `/etc/llama-bored/watch.toml` from the preflight:
   - **`[llama]`:** `enabled = true` and `url = "http://127.0.0.1:<port>"` if
     llama-swap was found (llama-swap itself, not a proxy in front of it);
     otherwise `enabled = false`.
   - **`[load] cpu_limit_w`: set it from the CPU.** It is the CPU's
     full-load socket power in watts; `nominal_frac` (0.8) of it reads 100 %
     on the gauge. Use, in order: AMD PPT for the model found by `lscpu`
     (usually 1.35 x TDP, for example 142 for a 105 W part, 88 for 65 W, 230
     for 170 W); Intel PL2, or the RAPL limit read in preflight; otherwise the
     TDP, and tell the human it can be tuned later. Keep it finite, above 0 and
     above `cpu_idle_w / nominal_frac`. It only matters with zenergy; tell the
     human if zenergy was not found. Leave `idle = "auto"` and the idle watts
     as shipped.
   - **`[models.aliases]`:** offer short names for long model ids seen in
     `/running`. Optional; the LCD shows the full name on two lines anyway.
   - **`[tty] blank_min` / `sleep_min` (optional burn-in guard): ask the human.**
     Say: "Blank the tty11 dashboard monitor after N minutes without a
     keypress, and power it down at M minutes (M > N; 0 = off)?" Write
     `blank_min = N` and `sleep_min = M` under `[tty]`, or leave both unset.
     The kernel timer resets only on keyboard input, so the monitor blanks even
     while the dashboard updates; any key or VT switch wakes it. The timers are
     kernel-wide (all text VTs).
   - **`[tty] show_text`: ask the human.** Say: "tty11 can show the live
     prompt and output text of your local models. Anyone at this machine's
     console, or on a KVM or remote console, can read it. Show it?" Set
     `show_text = true` or `false` from the answer (the example ships
     `true`). With `false` the watcher does not keep llama text at all, the
     IN/OUT panels are gone, and the header shows "text off". Also tell the
     human that text appears only while `LLAMA_SERVER_SLOTS_DEBUG=1` is on for
     llama-server. Leave `prompt_view = "clean"` unless they ask for raw.
   - **`[tty] chart_glyphs = "eighths"`** as shipped (the watcher unit loads
     the bundled font). `ctx_history_h = 6` as shipped.
   - **`[fans]`: offer it** if preflight found a Super-I/O chip with spinning
     fans. Say which chip and channels you found and that the panel only
     reads them. If the human agrees, uncomment the block and set:
     ```toml
     [fans]
     enabled = true
     hwmon = "<chip name>"           # the hwmon *name*, never hwmonN
     channels = [<N>, ...]           # fanN_input > 0; 1..=16, at most 8
     labels = ["<label>", ...]       # optional, same length, <= 10 chars each
     ```
     Ask the human for labels (for example `front`, `rear`, `top`) or leave
     `labels` out. If no chip was found, leave `[fans]` commented out.
3. **`/etc/llama-bored/light.toml`** (only if the Aura controller was found
   and the human wants RGB; otherwise let the installer copy the example and
   leave llama-light off). Start from `packaging/light.example.toml` the same
   way as `watch.toml`, then set `[aura]` from preflight:
   ```toml
   [aura]
   enabled = true
   leds_per_fan = <N>              # 1..=20
   fans = "mirrored"               # or "chain" for a daisy chain
   # chain_len = <fans>            # fans = "chain" only, 1..=8
   brightness_max = 80             # cap on every LED; lower it if the human asks
   ```
   Ask the human what they want to see. The shipped default (no `[[light]]`
   entries) shows `activity` on every fan on the LCD's colour ramp. On a
   **chain**, offer one metric per fan, for example:
   ```toml
   [[light]]
   target = "aura.chain[0]"
   metric = "gpu"

   [[light]]
   target = "aura.chain[1]"
   metric = "tokens_rate"
   range = [1, 200]
   scale = "log"
   style = "pulse"
   ```
   On **mirrored** fans use `target = "aura.fans"` (the default) and layer
   entries instead (a dim `activity` base with a `ring` gauge on top, say);
   `aura.chain[...]` is refused there. Metrics: `activity`, `gpu`, `cpu`,
   `load`, `mem`, `tokens_rate`, `coolant`, `gpu_temp`, `cpu_temp`. The
   example file documents every key and has whole-file examples. Check it
   **(user)** without touching any device:
   `"$REPO"/target/release/llama-light check --config /etc/llama-bored/light.toml`.
   Record whether the human wants llama-light enabled for boot.
4. **`/etc/llama-bored/metrics.toml`** (only if the human wants the
   Prometheus exporter; rule 9). Ask: "llama-metrics serves numbers (no
   prompt text) on TCP 19477 to the networks you allow. Which network or
   Prometheus host should reach it?" Write it from the example with
   `allow = ["<their CIDR>", "127.0.0.1/32"]` (host bits zero, no /0). The
   shipped `192.168.0.0/16` is only an example. The unit keeps the kernel's
   copy of the same list: after installing, add a drop-in with
   `systemctl edit llama-metrics`:
   ```ini
   [Service]
   IPAddressAllow=
   IPAddressAllow=<their CIDR> 127.0.0.1/32
   ```
   Both lists must match, or scrapes are refused. If the human declines, let
   the installer copy the example and leave the exporter disabled.
5. Leave `/etc/llama-bored/config.toml` to the installer (it copies
   `config.example.toml`). **Change mode is the default** (`[upload] mode =
   "change"`, at most one upload per 60 s). **Offer stream mode:** tell the
   human it redraws the LCD at 10 fps with the animated gauge, and ask whether
   they want it. Record the answer. Either way the install and the first
   write run in change mode; if they said yes, switch after verification with
   section 8.

## 4. Install (root, human informed)

1. Cooling snapshot **(user)**:
   `scripts/cooling-snapshot.sh | tee -a "$LOG/cooling.txt"`.
2. Run the installer through sudo from `BUILD_USER`, from the clone:
   `sudo scripts/install.sh`, or `sudo scripts/install.sh --enable-light`
   if the human wants llama-light enabled for boot (it is never started).
   It needs `SUDO_USER` to name the repo owner and refuses a bare root shell.
   If you only have a root shell, the equivalent is
   `env SUDO_USER="$BUILD_USER" "$REPO"/scripts/install.sh`.
3. **Read the whole output.** Copy the block starting
   `install.sh: how to roll back` into `$LOG/rollback-<date>.txt` and tell the
   human where it is. On failure the installer prints the rollback and also
   saves it to `/usr/local/libexec/llama-bored/ROLLBACK_PENDING`. Report it and
   **STOP**; do not retry blindly.
4. The installer has created the users `kraken-lcd`, `llama-watch`,
   `llama-light` and `llama-metrics` and the group `llama-view`. It has
   installed the udev rules, narrowed the cooler's hidraw node to
   `0660 root:kraken-lcd` with no ACL, and checked that `/dev/kraken-lcd/hid`
   points at that node (the writer unit may open only that path). If the
   Aura controller is attached, its node is `0660 root:llama-light` behind
   `/dev/llama-light/aura`. It has created `/var/lib/kraken-lcd` and enabled (not
   started) `llama-watch`. It **never** starts or enables the LCD writer. It
   ends by printing next steps; this skill follows the same order.
5. Start the watcher: `systemctl start llama-watch`. Its `ExecStartPre` loads
   the Hack console font on tty11. Check `systemctl status llama-watch`
   (active) and ask the human to glance at Alt+F11.
6. Cooling snapshot again **(user)** and compare with the baseline. Installing
   runs `udevadm trigger` on the cooler, which is expected to be
   cooling-neutral. Any anomaly → rule 5.

## 5. First LCD write (root, human watching)

Tell the human: "I will now upload one test image to the LCD."

```sh
install -d -o kraken-lcd -g kraken-lcd -m 0755 /var/lib/kraken-lcd   # no-op: install.sh created it
d=$(mktemp -d) && chmod 0755 "$d"
install -m 0644 "$REPO"/fixtures/views/test-card.json "$d/test-card.json"
runuser -u kraken-lcd -- /usr/local/libexec/llama-bored/kraken-lcd show-image --view "$d/test-card.json"
```

- Exit 0 and a bucket table printed → the LCD should show a **"TEST IMAGE"**
  card. Ask the human to confirm they see it.
- `show-image` runs the cooling guard itself (before, right after, and a 2 s
  follow-up). It refuses root, a missing `z53`, an unwritable state dir, or
  interface 0 already claimed (the writer running).
- Take a cooling snapshot **(user)** and compare: `pwm*_enable` identical, pump
  rpm within the band, no `/var/lib/kraken-lcd/halted`, and nothing new about
  `1e71` in `journalctl -k --since -5min`. Any anomaly → rule 5.

Only then start the writer:

```sh
systemctl enable --now kraken-lcd
```

Within a minute the LCD shows the live dashboard. Offer the desktop alert for
a HALTED latch: `systemctl --global enable kraken-lcd-halt.path` (needs
`notify-send` in the user's session).

## 5b. RGB and the exporter (only what the human chose)

**llama-light.** It changes colours only: it never saves to the controller's
flash, and never touches fan speed or the pump. Tell the human the fans'
lighting is about to change, then:

```sh
/usr/local/libexec/llama-bored/llama-light check --config /etc/llama-bored/light.toml
systemctl start llama-light
```

Ask the human whether the fans show what they expected. If every fan shows
the same colour although `aura.chain[N]` entries differ, the fans are on a
splitter: set `fans = "mirrored"` and layer entries instead. llama-light
re-reads `light.toml` within 2 s of a change; no restart is needed. Take a
cooling snapshot **(user)**: nothing about the pump or `pwm*_enable` may
change (rule 5). `systemctl enable llama-light` keeps it across reboots, if
`--enable-light` was not used. Stopping it leaves a neutral colour in RAM;
the board's own effect returns at the next power cycle.

**llama-metrics** (rule 9: only with the human's OK). After the drop-in from
section 3:

```sh
/usr/local/libexec/llama-bored/llama-metrics check --config /etc/llama-bored/metrics.toml
systemctl enable --now llama-metrics
curl -s http://127.0.0.1:19477/metrics | head
```

**Firewall note.** The installer never touches the firewall. If the host runs
one, the port stays closed to the LAN until opened. Show the human the
command for their setup and let them decide, for example with firewalld
(pick the zone that holds the LAN interface, `firewall-cmd
--get-active-zones`):
`firewall-cmd --permanent --zone=<zone> --add-port=19477/tcp && firewall-cmd --reload`.
Open it only to the same network as `allow`. Then give them the scrape
config: `targets: ["<this-host>:19477"]`, job name `llamabored`.

## 6. Verify

1. `systemctl status llama-watch kraken-lcd --no-pager`: both `active
   (running)`. No `HALTED` or `CRITICAL` in
   `journalctl -u kraken-lcd -b --no-pager`.
2. **The snapshot is updating** (10 Hz):
   ```sh
   grep -o '"seq":[0-9]*' /run/llama-watch/snapshot.json; sleep 1; grep -o '"seq":[0-9]*' /run/llama-watch/snapshot.json
   ```
   The second number is about 10 higher.
3. `systemd-analyze security llama-watch.service kraken-lcd.service` (plus
   `llama-light.service` and `llama-metrics.service` if enabled): each
   exposure is **≤ 3.0** (reference: watcher about 1.3, writer about 0.6).
4. **No llama text in the journal.**
   `journalctl -u llama-watch -u kraken-lcd --since -1h -o cat --no-pager` shows
   only status and state-change lines, no prompt or output text. For a strict
   check, and only with the human's OK (rule 6), they send one short prompt
   containing a unique word such as `zebra-4417`; then
   `journalctl -u llama-watch -u kraken-lcd --since -10min | grep -c zebra-4417`
   must print `0`.
5. **tty11 content.** Ask the human to check Alt+F11: the ACTIVITY row names
   its source (`gpu`, `cpu`, or `util` when no power sensor is readable); if
   `[fans]` is on, the FANS panel shows the chosen channels; with
   `show_text = false` the header says "text off". Confirm the fan panel did
   not change anything: `pwmN_enable` of that chip is as before.
6. **llama-view:** `usermod -aG llama-view "$BUILD_USER"`, then (in a new
   login session of that user) `llama-view --once` paints one frame of tty11
   and exits. Plain `llama-view` runs live until Ctrl+C.
7. If enabled: `systemctl status llama-light llama-metrics --no-pager`, and
   from the Prometheus host `curl -s http://<this-host>:19477/metrics |
   grep llamabored_snapshot_up` prints `1`. `ss -ltnp` shows 19477 as the
   only new listening port.
8. Final cooling snapshot **(user)** compared with the baseline.
9. If the human chose stream mode in section 3, do section 8 now.
10. Report to the human: the versions, the sources found, the config choices
   (`show_text`, llama URL, `cpu_limit_w`, `[fans]`, upload mode, the light
   layout and mappings, the metrics `allow` and firewall state), the verify
   results, and the paths of the log and rollback files.

## 7. Rollback and uninstall

- **Undo one install run:** first `systemctl disable --now kraken-lcd` if you
  enabled it (the screen returns to stock). Then run the saved rollback block
  line by line as root. It restores exactly what that run replaced and removes
  what it created, in a safe order, and ends by removing `ROLLBACK_PENDING`.
- **Full uninstall:** follow "Uninstall and rollback" in the README. Stop and
  disable every unit first, so `ExecStopPost` puts the stock screen back
  before any file is removed, and close any firewall port you opened.
- Nothing on the device needs undoing. The LCD frame memory is RAM and
  clears at power-off.

## 8. Stream mode (only if the human chose it)

1. Tell the human that stream mode uploads about 10 frames per second. It was
   validated on a Z53 with a watched ramp (about 63 ms per frame, a ceiling
   near 15 fps) and a 6,000-frame soak at 10 fps, with the cooling guard
   active throughout.
2. **(user)** Start `scripts/ramp-monitor.sh --out "$LOG/stream-monitor.tsv"` in
   a second terminal. It is a read-only 1 Hz cooling monitor. It prints
   `ABORT …` on a real anomaly and `NOTE pwmN a->b` for normal firmware curve
   steps.
3. **(root)** Set `[upload] mode = "stream"` (keep `stream_fps = 10`) in
   `/etc/llama-bored/config.toml`, then `systemctl restart kraken-lcd`.
4. Watch for 10 minutes. On any `ABORT` → rule 5, then set `mode = "change"`
   again.

## When things go wrong

| Situation | Do this |
|---|---|
| Any cooling anomaly | Rule 5: `systemctl stop kraken-lcd`, report, stop |
| `/var/lib/kraken-lcd/halted` exists | Do not clear it. Show the human its contents; they decide and run `kraken-lcd clear-halt` |
| `install.sh` refuses the tree or provenance | `git status`, `git switch main`, re-run `scripts/check.sh`, and install without touching the tree in between |
| `ROLLBACK_PENDING` exists | Show it to the human, run it with their OK, then retry |
| Watcher fails to start | `journalctl -u llama-watch -b`. Usually a `watch.toml` value out of range; fix it and restart |
| Writer logs "no NZXT Kraken Z LCD" | Re-run preflight step 2 |
| LCD shows "no data" | The watcher is down or stale; check it first |
| llama-light: "no Aura controller" | Expected without `0b05:18f3`; leave it disabled |
| llama-light exits with status 2 | `light.toml` failed its check at start; run `llama-light check` and fix the key it names |
| Fans all show one colour with per-fan entries | The fans are on a splitter: `fans = "mirrored"` |
| Prometheus cannot scrape | `allow` in `metrics.toml`, the `IPAddressAllow=` drop-in, and the firewall must all admit the scraper |
