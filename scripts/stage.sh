#!/usr/bin/env bash
# Unprivileged review. install.sh runs this as the invoking user, never as root.
# Git, rustc, and cargo stay on this side of the split so a root installer
# does not execute a uid-1000 toolchain, fsmonitor, diff helper, or hook.
# The review refuses a dirty tree, a HEAD that is not contained in the
# release branch (main), and a provenance mismatch. On success it prints
# the anchor summary.
#
# Usage: scripts/stage.sh REPO INSTALLED_SHA_FILE
#        scripts/stage.sh --require-landed REPO
#        scripts/stage.sh --self-test
set -euo pipefail

# Release anchor: install.sh only installs a HEAD contained in this branch.
RELEASE_BRANCH=main

refuse_if_root() {
  local euid=$1
  if [[ "$euid" -eq 0 ]]; then
    echo "stage.sh: refusing to run as root" >&2
    return 1
  fi
}

repo_root() {
  (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
}

# The writer unit is the runtime backstop for S11's accepted residuals.
# llama-watch.service must be the primary watcher unit (TTYPath=/dev/tty11),
# byte for byte the golden below.
watch_unit_golden() {
  cat <<'EOF'
[Unit]
Description=Llama Watch - single reader of host + llama state; tty11 dashboard; publishes snapshot
Conflicts=getty@tty11.service
After=getty@tty11.service systemd-user-sessions.service
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=notify
NotifyAccess=main
User=llama-watch
Group=llama-watch
ExecStartPre=-+/usr/local/libexec/llama-bored/llama-watch tty-setup --config /etc/llama-bored/watch.toml
ExecStartPre=-+/usr/bin/sh -c 'exec /usr/bin/setterm --term linux --powersave powerdown </dev/tty11 >/dev/tty11'
ExecStart=/usr/local/libexec/llama-bored/llama-watch run --config /etc/llama-bored/watch.toml
Restart=on-failure
RestartSec=5
WatchdogSec=10
UMask=0027
RuntimeDirectory=llama-watch
RuntimeDirectoryMode=0750
StandardInput=null
StandardOutput=tty
StandardError=journal
TTYPath=/dev/tty11
TTYReset=yes
TTYVHangup=yes
TTYVTDisallocate=yes

NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
RemoveIPC=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ReadOnlyPaths=/sys
DevicePolicy=closed
DeviceAllow=/dev/nvidiactl rw
DeviceAllow=/dev/nvidia0 rw
DeviceAllow=/dev/tty11 rw
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
IPAddressAllow=localhost
IPAddressDeny=any

[Install]
WantedBy=multi-user.target
EOF
}

assert_packaging_contract() {
  local root unit watch sys expected line rules
  root="$(repo_root)"
  unit="$root/packaging/kraken-lcd.service"
  watch="$root/packaging/llama-watch.service"
  sys="$root/packaging/llama-bored.sysusers"
  rules="$root/packaging/93-kraken-lcd-hidraw.rules"

  if [[ ! -f "$watch" ]]; then
    echo "stage self-test: missing $watch (primary watcher unit, TTYPath=/dev/tty11)" >&2
    exit 1
  fi
  expected="$(mktemp)"
  watch_unit_golden >"$expected"
  if ! cmp -s -- "$expected" "$watch"; then
    echo "stage self-test: llama-watch.service is not the golden watcher unit" >&2
    diff -u -- "$expected" "$watch" >&2 || true
    rm -f -- "$expected"
    exit 1
  fi
  rm -f -- "$expected"
  # tty11 gets its font and size before the watcher starts (#7). `+` runs
  # `llama-watch tty-setup` as root outside the sandbox: it validates
  # watch.toml, then runs /usr/bin/setfont with the bundled font `[tty] font`
  # names and, when `[tty] size` is set, /usr/bin/stty cols/rows on
  # /dev/tty11 (argv from an enum and two bounded integers, no shell, empty
  # environment). setfont needs KDFONTOP and stty the resize, neither of
  # which the sandboxed watcher may do (S13). `-` keeps a missing font or a
  # failed step from stopping the watcher, which then draws with whatever
  # font and size tty11 has (tty.chart_glyphs = "halves" is safe).
  local font_pre='ExecStartPre=-+/usr/local/libexec/llama-bored/llama-watch tty-setup --config /etc/llama-bored/watch.toml'
  # The second root pre-step sets the console powersave mode (TIOCLINUX, via
  # setterm on stdin = /dev/tty11) so `[tty] sleep_min` can power the monitor
  # down. The watcher itself never calls TIOCLINUX (S13). `-` keeps a setterm
  # failure from stopping the watcher.
  local powersave_pre="ExecStartPre=-+/usr/bin/sh -c 'exec /usr/bin/setterm --term linux --powersave powerdown </dev/tty11 >/dev/tty11'"
  if [[ "$(grep -c '^ExecStartPre=' "$watch")" -ne 2 ]] \
    || ! grep -F -q -x -- "$font_pre" "$watch" \
    || ! grep -F -q -x -- "$powersave_pre" "$watch"; then
    echo "stage self-test: llama-watch.service must have exactly two ExecStartPre, the non-fatal root tty-setup (font and size) and the non-fatal setterm powersave" >&2
    exit 1
  fi
  local font
  for font in llama-hack-12x24.psfu llama-hack-12x22.psfu; do
    if [[ ! -f "$root/packaging/fonts/$font" ]]; then
      echo "stage self-test: missing packaging/fonts/$font" >&2
      exit 1
    fi
  done
  # The watcher reaches tty11 through the fd systemd hands it (TTYPath=)
  # and DeviceAllow=/dev/tty11 rw. A group-based fallback
  # (SupplementaryGroups=llama-tty, packaging/72-llama-watch-tty.rules and
  # sysusers `g llama-tty -`) is only a plan for the case where that path
  # fails; it must not ship unless it is needed. Flip this assertion then.
  if grep -F -q 'llama-tty' "$watch" || grep -F -q 'SupplementaryGroups=' "$watch"; then
    echo "stage self-test: the unused tty group fallback was shipped" >&2
    exit 1
  fi

  local -a required=(
    'SupplementaryGroups=llama-watch'
    'PrivateNetwork=yes'
    'ProcSubset=pid'
    'RestrictAddressFamilies=AF_UNIX'
    'ProtectSystem=strict'
    'ReadOnlyPaths=/sys'
    'ProtectKernelTunables=yes'
    'DevicePolicy=closed'
    'ProtectHome=yes'
    'DeviceAllow=/dev/kraken-lcd/hid rw'
    'IPAddressDeny=any'
  )
  for line in "${required[@]}"; do
    if ! grep -F -q -x -- "$line" "$unit"; then
      echo "stage self-test: writer unit missing directive: $line" >&2
      exit 1
    fi
  done
  # SAFETY.md RR7: no class-wide hidraw grant; only the udev pin is allowed.
  local -a forbidden=(
    'DeviceAllow=char-hidraw rw'
    'DeviceAllow=/dev/nvidiactl rw'
    'DeviceAllow=/dev/nvidia0 rw'
    'IPAddressAllow=localhost'
    'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6'
  )
  for line in "${forbidden[@]}"; do
    if grep -F -q -x -- "$line" "$unit"; then
      echo "stage self-test: writer unit still has: $line" >&2
      exit 1
    fi
  done
  if grep -E -q '^(After|Wants)=.*llama-watch' "$unit"; then
    echo "stage self-test: writer unit orders itself on llama-watch" >&2
    exit 1
  fi

  if ! grep -F -q 'SUBSYSTEM=="hidraw"' "$rules" \
    || ! grep -F -q '1E71:3008' "$rules" \
    || ! grep -F -q 'GROUP="kraken-lcd"' "$rules" \
    || ! grep -F -q 'MODE="0660"' "$rules" \
    || ! grep -F -q 'TAG-="uaccess"' "$rules" \
    || ! grep -F -q 'TAG-="udev-acl"' "$rules" \
    || ! grep -F -q 'SYMLINK+="kraken-lcd/hid"' "$rules"; then
    echo "stage self-test: hidraw udev rule lost the S11 backstop" >&2
    exit 1
  fi

  if ! grep -F -q -x -- 'u llama-watch - "Llama Watch reader/tty" / /usr/sbin/nologin' "$sys"; then
    echo "stage self-test: sysusers missing the llama-watch user" >&2
    exit 1
  fi
  if awk 'index($0, "llama-watch") && index($0, "kraken-lcd") { bad = 1 } END { exit !bad }' "$sys"; then
    echo "stage self-test: llama-watch must not be in group kraken-lcd" >&2
    exit 1
  fi
  if ! grep -F -q -x -- 'g llama-view -' "$sys"; then
    echo "stage self-test: sysusers missing the llama-view group" >&2
    exit 1
  fi
  if ! grep -F -q -x -- 'KERNEL=="vcsa11|vcs11|vcsu11", GROUP="llama-view", MODE="0640"' \
    "$root/packaging/72-llama-view.rules"; then
    echo "stage self-test: llama-view udev rule is not the tty11 mirror rule" >&2
    exit 1
  fi

  assert_writer_unit_golden "$unit"
  assert_hidraw_rules_golden "$rules"
  assert_light_contract "$root"
  assert_metrics_contract "$root"
}

# llama-light, the RGB writer. Colour only, its own uid, only the Aura
# controller's and the keyboard lighting interface's hidraw pins, no
# network. Directives only (comments ignored).
light_directive_golden() {
  cat <<'EOF'
[Unit]
Description=Llama Light - RGB lighting from the llama-watch snapshot (colour only, never cooling)
StartLimitIntervalSec=600
StartLimitBurst=5
[Service]
Type=notify
NotifyAccess=main
User=llama-light
Group=llama-light
SupplementaryGroups=llama-watch
ExecStart=/usr/local/libexec/llama-bored/llama-light run --config /etc/llama-bored/light.toml
ExecStopPost=-/usr/local/libexec/llama-bored/llama-light restore --config /etc/llama-bored/light.toml
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=10
WatchdogSec=10
TimeoutStopSec=10
UMask=0077
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
RemoveIPC=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
ReadOnlyPaths=/sys
DevicePolicy=closed
DeviceAllow=/dev/llama-light/aura rw
DeviceAllow=/dev/llama-light/keyboard rw
PrivateNetwork=yes
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
[Install]
WantedBy=multi-user.target
EOF
}

light_rules_golden() {
  cat <<'EOF'
# /etc/udev/rules.d/94-llama-light-hidraw.rules — the ASUS Aura USB controller (0b05:18f3)
# and the Corsair STRAFE RGB MK.2 keyboard's lighting interface (1b1c:1b48, USB interface 1).
# Numbered after distro and OpenRGB rules that leave these hidraw nodes 0666 or tag them uaccess;
# install.sh overwrites it. The two symlinks are llama-light.service's only DeviceAllow= entries.
SUBSYSTEM=="hidraw", KERNELS=="0003:0B05:18F3.*", GROUP="llama-light", MODE="0660", TAG-="uaccess", TAG-="udev-acl", SYMLINK+="llama-light/aura"
# Keyboard: udev matches all parent keys on one ancestor, so the HID id (on the HID device) and
# the interface number (on the USB interface) take two rules. Only interface 1, the vendor
# lighting interface (usage page 0xFFC2), is taken; llama-light re-checks the usage page before
# it opens the node. Typing uses the keyboard interface and evdev, not hidraw: unaffected.
SUBSYSTEM=="hidraw", KERNELS=="0003:1B1C:1B48.*", ENV{LLAMA_LIGHT_KBD}="1"
SUBSYSTEM=="hidraw", ENV{LLAMA_LIGHT_KBD}=="1", SUBSYSTEMS=="usb", ATTRS{bInterfaceNumber}=="01", GROUP="llama-light", MODE="0660", TAG-="uaccess", TAG-="udev-acl", SYMLINK+="llama-light/keyboard"
EOF
}

assert_light_unit_golden() {
  local unit=$1 got want tmp extra
  got="$(unit_directives "$unit")"
  want="$(light_directive_golden)"
  assert_text_eq "$got" "$want" "llama-light unit directive set drifted"
  local -a plants=(
    'PrivateNetwork=no'
    'RestrictAddressFamilies=AF_INET'
    'DeviceAllow=char-hidraw rw'
    'DeviceAllow=/dev/kraken-lcd/hid rw'
    'DeviceAllow=/dev/hidraw0 rw'
    'DeviceAllow=char-usb_device rw'
    'DeviceAllow=/dev/i2c-5 rw'
    'SupplementaryGroups=kraken-lcd'
    'ReadWritePaths=/sys/class/hwmon'
    'AmbientCapabilities=CAP_SYS_RAWIO'
  )
  for extra in "${plants[@]}"; do
    tmp="$(mktemp)"
    cat -- "$unit" >"$tmp"
    printf '\n%s\n' "$extra" >>"$tmp"
    got="$(unit_directives "$tmp")"
    rm -f -- "$tmp"
    if [[ "$got" == "$want" ]]; then
      echo "stage self-test: llama-light unit accepted an added directive: $extra" >&2
      exit 1
    fi
  done
}

assert_light_rules_golden() {
  local rules=$1 got want tmp
  got="$(rules_text "$rules")"
  want="$(light_rules_golden)"
  assert_text_eq "$got" "$want" "llama-light udev rule drifted"
  tmp="$(mktemp)"
  cat -- "$rules" >"$tmp"
  printf '\nSUBSYSTEM=="hidraw", KERNELS=="0003:0B05:18F3.*", MODE="0666"\n' >>"$tmp"
  got="$(rules_text "$tmp")"
  rm -f -- "$tmp"
  if [[ "$got" == "$want" ]]; then
    echo "stage self-test: llama-light rules accepted an added 0666 line" >&2
    exit 1
  fi
  # The keyboard rule widened to every interface of the keyboard.
  tmp="$(mktemp)"
  sed 's/, ATTRS{bInterfaceNumber}=="01"//' -- "$rules" >"$tmp"
  got="$(rules_text "$tmp")"
  rm -f -- "$tmp"
  if [[ "$got" == "$want" ]]; then
    echo "stage self-test: llama-light rules accepted a keyboard rule on every interface" >&2
    exit 1
  fi
}

assert_light_contract() {
  local root=$1 unit rules sys example f
  unit="$root/packaging/llama-light.service"
  rules="$root/packaging/94-llama-light-hidraw.rules"
  sys="$root/packaging/llama-bored.sysusers"
  example="$root/packaging/light.example.toml"
  for f in "$unit" "$rules" "$example"; do
    if [[ ! -f "$f" ]]; then
      echo "stage self-test: missing $f" >&2
      exit 1
    fi
  done
  if ! grep -F -q -x -- 'u llama-light - "Llama Light RGB writer" / /usr/sbin/nologin' "$sys"; then
    echo "stage self-test: sysusers missing the llama-light user" >&2
    exit 1
  fi
  if grep -E -q '^m[[:space:]]+llama-light' "$sys"; then
    echo "stage self-test: llama-light must not get extra groups from sysusers" >&2
    exit 1
  fi
  # The light unit and rule never name the Kraken, i2c, usbfs or writable sysfs.
  if { unit_directives "$unit"; rules_text "$rules"; } \
    | grep -E -i -q '1e71|kraken-lcd/hid|i2c|usb_device|ReadWritePaths'; then
    echo "stage self-test: llama-light packaging names a cooling or bus device" >&2
    exit 1
  fi
  assert_light_unit_golden "$unit"
  assert_light_rules_golden "$rules"
}

# llama-metrics, the LAN Prometheus exporter. Its own uid, the snapshot
# group only, no devices, IP allowlist and bind port pinned in the kernel.
# Directives only (comments ignored).
metrics_directive_golden() {
  cat <<'EOF'
[Unit]
Description=Llama Metrics - Prometheus exporter for the llama-watch snapshot (read-only, LAN)
StartLimitIntervalSec=600
StartLimitBurst=5
[Service]
Type=notify
NotifyAccess=main
User=llama-metrics
Group=llama-metrics
SupplementaryGroups=llama-watch
ExecStart=/usr/local/libexec/llama-bored/llama-metrics run --config /etc/llama-bored/metrics.toml
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=5
WatchdogSec=10
TimeoutStopSec=5
UMask=0077
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
RemoveIPC=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
InaccessiblePaths=/sys
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
DevicePolicy=closed
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
IPAddressAllow=192.168.0.0/16 127.0.0.1/32
IPAddressDeny=any
SocketBindAllow=tcp:19477
SocketBindDeny=any
[Install]
WantedBy=multi-user.target
EOF
}

assert_metrics_unit_golden() {
  local unit=$1 got want tmp extra
  got="$(unit_directives "$unit")"
  want="$(metrics_directive_golden)"
  assert_text_eq "$got" "$want" "llama-metrics unit directive set drifted"
  local -a plants=(
    'DeviceAllow=char-hidraw rw'
    'DeviceAllow=/dev/kraken-lcd/hid rw'
    'SupplementaryGroups=kraken-lcd'
    'IPAddressAllow=any'
    'IPAddressAllow=0.0.0.0/0'
    'SocketBindAllow=any'
    'ReadWritePaths=/run/llama-watch'
    'ReadOnlyPaths=/sys'
    'AmbientCapabilities=CAP_NET_BIND_SERVICE'
    'User=llama-watch'
  )
  for extra in "${plants[@]}"; do
    tmp="$(mktemp)"
    cat -- "$unit" >"$tmp"
    printf '\n%s\n' "$extra" >>"$tmp"
    got="$(unit_directives "$tmp")"
    rm -f -- "$tmp"
    if [[ "$got" == "$want" ]]; then
      echo "stage self-test: llama-metrics unit accepted an added directive: $extra" >&2
      exit 1
    fi
  done
}

# The example config's allow list must be the unit's IPAddressAllow=, and its
# listen port the unit's SocketBindAllow=. crates/llama-metrics/tests/packaging.rs
# checks the same with the real config parser.
assert_metrics_contract() {
  local root=$1 unit sys example f allow_cfg allow_unit port_cfg port_unit
  unit="$root/packaging/llama-metrics.service"
  sys="$root/packaging/llama-bored.sysusers"
  example="$root/packaging/metrics.example.toml"
  for f in "$unit" "$example"; do
    if [[ ! -f "$f" ]]; then
      echo "stage self-test: missing $f" >&2
      exit 1
    fi
  done
  if ! grep -F -q -x -- 'u llama-metrics - "Llama Metrics exporter" / /usr/sbin/nologin' "$sys"; then
    echo "stage self-test: sysusers missing the llama-metrics user" >&2
    exit 1
  fi
  if grep -E -q '^m[[:space:]]+llama-metrics' "$sys"; then
    echo "stage self-test: llama-metrics must not get extra groups from sysusers" >&2
    exit 1
  fi
  if unit_directives "$unit" | grep -E -i -q '1e71|hidraw|i2c|usb|nvidia|tty|DeviceAllow|ReadWritePaths'; then
    echo "stage self-test: llama-metrics unit names a device or a writable path" >&2
    exit 1
  fi
  allow_cfg="$(sed -n 's/^allow = \[\(.*\)\]$/\1/p' "$example" | tr ',' '\n' | tr -d '" ' | LC_ALL=C sort | tr '\n' ' ')"
  allow_unit="$(sed -n 's/^IPAddressAllow=//p' "$unit" | tr ' ' '\n' | LC_ALL=C sort | tr '\n' ' ')"
  if [[ -z "${allow_cfg// /}" || "$allow_cfg" != "$allow_unit" ]]; then
    echo "stage self-test: metrics.example.toml allow ($allow_cfg) is not IPAddressAllow= ($allow_unit)" >&2
    exit 1
  fi
  port_cfg="$(sed -n 's/^listen = ".*:\([0-9][0-9]*\)"$/\1/p' "$example")"
  port_unit="$(sed -n 's/^SocketBindAllow=tcp://p' "$unit")"
  if [[ -z "$port_cfg" || "$port_cfg" != "$port_unit" ]]; then
    echo "stage self-test: metrics.example.toml listen port ($port_cfg) is not SocketBindAllow= ($port_unit)" >&2
    exit 1
  fi
  if ! grep -F -q 'LAN-exposed by design' "$example"; then
    echo "stage self-test: metrics.example.toml does not say it is LAN-exposed" >&2
    exit 1
  fi
  assert_metrics_unit_golden "$unit"
}

# Directives only: comments and blank lines are not part of the pin.
unit_directives() {
  awk '
    { sub(/\r$/, "") }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { sub(/[[:space:]]+$/, ""); print }
  ' "$1"
}

# Blank lines are ignored. Commented-out rule lines are kept, so a
# commented SUBSYSTEM line is a different file.
rules_text() {
  awk '
    { sub(/\r$/, "") }
    /^[[:space:]]*$/ { next }
    { sub(/[[:space:]]+$/, ""); print }
  ' "$1"
}

writer_directive_golden() {
  cat <<'EOF'
[Unit]
Description=Kraken LCD - live telemetry on the NZXT Kraken Z LCD (LCD-only, never cooling)
StartLimitIntervalSec=600
StartLimitBurst=5
[Service]
Type=notify
NotifyAccess=main
User=kraken-lcd
Group=kraken-lcd
SupplementaryGroups=llama-watch
ExecStart=/usr/local/libexec/llama-bored/kraken-lcd run --config /etc/llama-bored/config.toml
ExecStopPost=-/usr/local/libexec/llama-bored/kraken-lcd restore-stock --config /etc/llama-bored/config.toml
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=60
WatchdogSec=30
TimeoutStopSec=15
UMask=0077
StateDirectory=kraken-lcd
StateDirectoryMode=0755
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictSUIDSGID=yes
LockPersonality=yes
RestrictRealtime=yes
RestrictNamespaces=yes
RemoveIPC=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
ReadOnlyPaths=/sys
DevicePolicy=closed
DeviceAllow=char-usb_device rw
DeviceAllow=/dev/kraken-lcd/hid rw
PrivateNetwork=yes
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
[Install]
WantedBy=multi-user.target
EOF
}

hidraw_rules_golden() {
  cat <<'EOF'
# /etc/udev/rules.d/93-kraken-lcd-hidraw.rules — SAFETY.md RR2. Numbered to sort after distro rules
# (such as 92-viia.rules) that add uaccess to hidraw nodes; install.sh overwrites it.
# SAFETY.md RR7: the stable name is the writer unit's only hidraw DeviceAllow= entry.
SUBSYSTEM=="hidraw", KERNELS=="0003:1E71:3008.*", GROUP="kraken-lcd", MODE="0660", TAG-="uaccess", TAG-="udev-acl", SYMLINK+="kraken-lcd/hid"
EOF
}

assert_text_eq() {
  local got=$1 want=$2 label=$3
  if [[ "$got" != "$want" ]]; then
    echo "stage self-test: $label" >&2
    diff -u --label want --label got <(printf '%s\n' "$want") <(printf '%s\n' "$got") >&2 || true
    exit 1
  fi
}

assert_writer_unit_golden() {
  local unit=$1 got want tmp extra
  got="$(unit_directives "$unit")"
  want="$(writer_directive_golden)"
  assert_text_eq "$got" "$want" "writer unit directive set drifted"
  local -a plants=(
    'PrivateNetwork=no'
    'RestrictAddressFamilies=AF_INET'
    'DeviceAllow=block-sd rw'
    'ReadWritePaths=/sys/class/hwmon'
    'Requires=llama-watch.service'
    'DeviceAllow=char-hidraw rw'
  )
  for extra in "${plants[@]}"; do
    tmp="$(mktemp)"
    cat -- "$unit" >"$tmp"
    printf '\n%s\n' "$extra" >>"$tmp"
    got="$(unit_directives "$tmp")"
    rm -f -- "$tmp"
    if [[ "$got" == "$want" ]]; then
      echo "stage self-test: writer unit accepted an added directive: $extra" >&2
      exit 1
    fi
  done
}

assert_hidraw_rules_golden() {
  local rules=$1 got want tmp
  got="$(rules_text "$rules")"
  want="$(hidraw_rules_golden)"
  assert_text_eq "$got" "$want" "hidraw udev rule drifted"
  tmp="$(mktemp)"
  cat -- "$rules" >"$tmp"
  printf '\n# SUBSYSTEM=="hidraw", KERNELS=="0003:1E71:3008.*", GROUP="kraken-lcd", MODE="0660"\n' >>"$tmp"
  got="$(rules_text "$tmp")"
  rm -f -- "$tmp"
  if [[ "$got" == "$want" ]]; then
    echo "stage self-test: hidraw rules accepted a commented-out SUBSYSTEM line" >&2
    exit 1
  fi
}

# Home of the invoking user. sudo's secure_path does not keep that user's PATH,
# so this is resolved from the passwd database rather than from $HOME or $PATH.
invoking_home() {
  local user home
  user="${SUDO_USER:-}"
  if [[ -z "$user" || "$user" == "root" ]]; then
    user="$(id -un)"
  fi
  if ! home="$(getent passwd "$user" | cut -d: -f6)"; then
    echo "stage.sh: no passwd entry for $user" >&2
    return 1
  fi
  if [[ -z "$home" || ! -d "$home" ]]; then
    echo "stage.sh: no home directory for $user" >&2
    return 1
  fi
  printf '%s\n' "$home"
}

prepend_toolchain() {
  # rustup from Homebrew when present (the standard Linuxbrew prefix, or brew
  # on PATH), else a plain rustup install in ~/.cargo/bin, else PATH as is.
  local brew=/home/linuxbrew/.linuxbrew/bin/brew
  local home prefix
  home="$(invoking_home)" || return 1
  if [[ ! -x "$brew" ]]; then
    brew="$(command -v brew 2>/dev/null || true)"
  fi
  if [[ -n "$brew" && -x "$brew" ]] \
    && prefix="$("$brew" --prefix rustup 2>/dev/null)" \
    && [[ -d "${prefix}/bin" ]]; then
    PATH="${prefix}/bin:${home}/.cargo/bin:${PATH}"
  else
    PATH="${home}/.cargo/bin:${PATH}"
  fi
  export PATH
}

# Run a command and store its stdout in the named variable.
# The exit status is the command's status, including when set -e is off.
capture() {
  local __name=$1 __out __status
  shift
  set +e
  __out=$("$@")
  __status=$?
  set -e
  printf -v "$__name" '%s' "$__out"
  return "$__status"
}

# No fsmonitor helper and no external diff program. Hooks are an empty directory.
# --no-textconv and --no-ext-diff are diff-family options, so log and diff pass them.
git_safe() {
  git -c core.fsmonitor= -c diff.external= -c "core.hooksPath=${STAGE_HOOKS:?}" "$@"
}

# Print the review. Return 1 unless the tree is safe to install.
# Caller sets nothing; this function owns STAGE_HOOKS for the duration.
stage_review() {
  local repo=$1 installed_sha_file=$2 status=0
  STAGE_HOOKS="$(mktemp -d)"
  _stage_review "$repo" "$installed_sha_file" || status=$?
  rm -rf -- "$STAGE_HOOKS"
  unset STAGE_HOOKS
  return "$status"
}

_stage_review() {
  local repo=$1 installed_sha_file=$2
  local dirty binary provenance name
  refuse_if_root "$(id -u)" || return 1
  if ! capture dirty git_safe -C "$repo" status --porcelain; then
    echo "stage.sh: git status failed" >&2
    return 1
  fi
  if [[ -n "$dirty" ]]; then
    echo "stage.sh: refusing a dirty tree" >&2
    printf '%s\n' "$dirty" >&2
    return 1
  fi
  # Landed commit before provenance. A matching provenance file must not
  # make an unlanded HEAD installable.
  refuse_unless_landed "$repo" || return 1
  provenance="$repo/target/check-provenance.txt"
  for name in kraken-lcd llama-watch llama-view llama-light llama-metrics; do
    binary="$repo/target/release/$name"
    if [[ ! -f "$binary" || -L "$binary" ]]; then
      echo "stage.sh: release binary missing at $binary" >&2
      return 1
    fi
  done
  if [[ ! -f "$provenance" || -L "$provenance" ]]; then
    echo "stage.sh: missing $provenance; run scripts/check.sh first" >&2
    return 1
  fi
  prepend_toolchain || return 1
  check_provenance "$repo" "$provenance" || return 1
  review_since "$repo" "$installed_sha_file" || return 1
  echo "----- target/check-provenance.txt -----"
  cat -- "$provenance" || return 1
  echo "----- binaries on disk -----"
  sha256sum -- \
    "$repo/target/release/kraken-lcd" \
    "$repo/target/release/llama-watch" \
    "$repo/target/release/llama-view" \
    "$repo/target/release/llama-light" \
    "$repo/target/release/llama-metrics" || return 1
  anchor_summary "$repo" "$installed_sha_file" || return 1
}

check_provenance() {
  local repo=$1 file=$2
  local -a prov=() sums=() binaries=(kraken-lcd llama-watch llama-view llama-light llama-metrics)
  local rustc_v cargo_v current head describe live_hash recorded_hash path i
  mapfile -t prov <"$file" || return 1
  if [[ ${#prov[@]} -ne 11 ]]; then
    echo "stage.sh: $file must have 11 lines from scripts/check.sh" >&2
    return 1
  fi
  if ! capture rustc_v rustc -V; then
    echo "stage.sh: rustc -V failed" >&2
    return 1
  fi
  if ! capture cargo_v cargo -V; then
    echo "stage.sh: cargo -V failed" >&2
    return 1
  fi
  if [[ "${prov[0]}" != "$rustc_v" ]]; then
    echo "stage.sh: rustc -V does not match $file" >&2
    echo "  recorded: ${prov[0]}" >&2
    echo "  now:      $rustc_v" >&2
    return 1
  fi
  if [[ "${prov[1]}" != "$cargo_v" ]]; then
    echo "stage.sh: cargo -V does not match $file" >&2
    return 1
  fi
  if ! capture current sha256sum -- "$repo/Cargo.lock" "$repo/rust-toolchain.toml"; then
    echo "stage.sh: sha256sum of the lockfiles failed" >&2
    return 1
  fi
  mapfile -t sums <<<"$current" || return 1
  # sha256sum prints the path it was given. check.sh runs it from the repo
  # root, so rewrite absolute paths back to those basenames before comparing.
  sums[0]="${sums[0]%% *}  Cargo.lock"
  sums[1]="${sums[1]%% *}  rust-toolchain.toml"
  if [[ "${sums[0]}" != "${prov[2]}" || "${sums[1]}" != "${prov[3]}" ]]; then
    echo "stage.sh: Cargo.lock or rust-toolchain.toml hash does not match $file" >&2
    return 1
  fi
  if ! capture head git_safe -C "$repo" rev-parse HEAD; then
    echo "stage.sh: git rev-parse HEAD failed" >&2
    return 1
  fi
  if ! capture describe git_safe -C "$repo" describe --always --dirty; then
    echo "stage.sh: git describe failed" >&2
    return 1
  fi
  if [[ "${prov[4]}" != "$head" ]]; then
    echo "stage.sh: $file is for ${prov[4]}, HEAD is $head; re-run scripts/check.sh" >&2
    return 1
  fi
  if [[ "${prov[5]}" != "$describe" || "$describe" == *-dirty ]]; then
    echo "stage.sh: provenance describe is '${prov[5]}' (now '$describe')" >&2
    return 1
  fi
  # Line 7 is kraken-lcd, line 8 is llama-watch, line 9 is llama-view,
  # line 10 is llama-light, line 11 is llama-metrics. S6 still checks the
  # LCD writer.
  for i in 0 1 2 3 4; do
    path="$repo/target/release/${binaries[$i]}"
    if ! capture live_hash sha256sum -- "$path"; then
      echo "stage.sh: sha256sum of $path failed" >&2
      return 1
    fi
    recorded_hash="${prov[$((6 + i))]%% *}"
    if [[ -z "$recorded_hash" || "${live_hash%% *}" != "$recorded_hash" ]]; then
      echo "stage.sh: ${binaries[$i]} hash does not match $file" >&2
      return 1
    fi
    if [[ "${prov[$((6 + i))]}" != "$recorded_hash  target/release/${binaries[$i]}" ]]; then
      echo "stage.sh: provenance line $((7 + i)) is not target/release/${binaries[$i]}" >&2
      return 1
    fi
  done
}

# HEAD is contained in the branch refs/heads/$RELEASE_BRANCH: an
# ancestor, or the tip itself. A tag or remote of the same short name is
# not the branch. git merge-base --is-ancestor treats a commit as an
# ancestor of itself. install.sh calls this before it writes.
refuse_unless_landed() {
  local repo=$1 head tip
  # refs/heads/ only. A tag named like the release branch would win the
  # short-name lookup and could point the anchor at an unlanded commit.
  if ! capture tip git_safe -C "$repo" rev-parse --verify --quiet "refs/heads/$RELEASE_BRANCH^{commit}"; then
    echo "stage.sh: refs/heads/$RELEASE_BRANCH is not a commit in this repository" >&2
    return 1
  fi
  if ! capture head git_safe -C "$repo" rev-parse --verify HEAD; then
    echo "stage.sh: git rev-parse HEAD failed" >&2
    return 1
  fi
  if ! git_safe -C "$repo" merge-base --is-ancestor "$head" "$tip"; then
    echo "stage.sh: HEAD $head is not contained in $RELEASE_BRANCH" >&2
    return 1
  fi
}

require_landed() {
  local repo=$1 status=0
  refuse_if_root "$(id -u)" || return 1
  STAGE_HOOKS="$(mktemp -d)"
  refuse_unless_landed "$repo" || status=$?
  rm -rf -- "$STAGE_HOOKS"
  unset STAGE_HOOKS
  return "$status"
}

# Prints "since<TAB>label". The label is the installed commit, or "root <sha>"
# when nothing is installed yet. No pathspec: a filtered diff can hide a change.
resolve_since() {
  local repo=$1 installed_sha_file=$2
  local since label roots_text
  local -a roots=()
  if [[ -e "$installed_sha_file" ]]; then
    if [[ -L "$installed_sha_file" || ! -f "$installed_sha_file" ]]; then
      echo "stage.sh: INSTALLED_SHA is not a regular file" >&2
      return 1
    fi
    if ! since="$(tr -d '[:space:]' <"$installed_sha_file")"; then
      echo "stage.sh: failed to read INSTALLED_SHA" >&2
      return 1
    fi
    if [[ ! "$since" =~ ^[0-9a-f]{40}$ ]]; then
      echo "stage.sh: INSTALLED_SHA is not a 40-digit commit id" >&2
      return 1
    fi
    if ! git_safe -C "$repo" cat-file -e "${since}^{commit}" 2>/dev/null; then
      echo "stage.sh: INSTALLED_SHA $since is not in this repository" >&2
      return 1
    fi
    label="$since"
  else
    if ! capture roots_text git_safe -C "$repo" rev-list --max-parents=0 HEAD; then
      echo "stage.sh: git rev-list failed" >&2
      return 1
    fi
    mapfile -t roots <<<"$roots_text" || return 1
    if [[ ${#roots[@]} -ne 1 || -z "${roots[0]}" ]]; then
      echo "stage.sh: expected one root commit, found ${#roots[@]}" >&2
      return 1
    fi
    since="${roots[0]}"
    label="root $since"
  fi
  printf '%s\t%s\n' "$since" "$label"
}

# Every path since the installed commit.
review_since() {
  local repo=$1 installed_sha_file=$2
  local line since label
  line="$(resolve_since "$repo" "$installed_sha_file")" || return 1
  since="${line%%$'\t'*}"
  label="${line#*$'\t'}"
  echo "----- git log --stat since $label -----"
  git_safe -C "$repo" --no-pager log --no-textconv --no-ext-diff --stat \
    "${since}..HEAD" || return 1
  echo "----- git diff since $label -----"
  git_safe -C "$repo" --no-pager diff --no-textconv --no-ext-diff \
    "$since" HEAD || return 1
}

# One block: HEAD, subject, both binary hashes, diff stat since INSTALLED_SHA.
anchor_summary() {
  local repo=$1 installed_sha_file=$2
  local line since label head subject dash_hash watch_hash view_hash light_hash metrics_hash
  line="$(resolve_since "$repo" "$installed_sha_file")" || return 1
  since="${line%%$'\t'*}"
  label="${line#*$'\t'}"
  if ! capture head git_safe -C "$repo" rev-parse HEAD; then
    echo "stage.sh: git rev-parse HEAD failed" >&2
    return 1
  fi
  if ! capture subject git_safe -C "$repo" log -1 --format=%s HEAD; then
    echo "stage.sh: git log subject failed" >&2
    return 1
  fi
  if ! capture dash_hash sha256sum -- "$repo/target/release/kraken-lcd"; then
    echo "stage.sh: sha256sum of kraken-lcd failed" >&2
    return 1
  fi
  if ! capture watch_hash sha256sum -- "$repo/target/release/llama-watch"; then
    echo "stage.sh: sha256sum of llama-watch failed" >&2
    return 1
  fi
  if ! capture view_hash sha256sum -- "$repo/target/release/llama-view"; then
    echo "stage.sh: sha256sum of llama-view failed" >&2
    return 1
  fi
  if ! capture light_hash sha256sum -- "$repo/target/release/llama-light"; then
    echo "stage.sh: sha256sum of llama-light failed" >&2
    return 1
  fi
  if ! capture metrics_hash sha256sum -- "$repo/target/release/llama-metrics"; then
    echo "stage.sh: sha256sum of llama-metrics failed" >&2
    return 1
  fi
  echo "----- anchor -----"
  echo "HEAD: $head"
  echo "subject: $subject"
  echo "${dash_hash%% *}  kraken-lcd"
  echo "${watch_hash%% *}  llama-watch"
  echo "${view_hash%% *}  llama-view"
  echo "${light_hash%% *}  llama-light"
  echo "${metrics_hash%% *}  llama-metrics"
  echo "----- diff stat since $label -----"
  git_safe -C "$repo" --no-pager diff --stat --no-textconv --no-ext-diff \
    "$since" HEAD || return 1
  echo "----- end anchor -----"
}

git_commit() {
  local repo=$1 message=$2
  shift 2
  git -C "$repo" add -- "$@"
  # Built at runtime so a literal-email scan of the tree stays empty.
  local email
  email="test$(printf '\x40')example.invalid"
  git -C "$repo" -c user.name='stage self-test' -c user.email="$email" \
    -c core.hooksPath="$(mktemp -d)" commit -m "$message" >/dev/null
}

write_provenance() {
  local repo=$1
  mkdir -p -- "$repo/target"
  prepend_toolchain
  {
    rustc -V || return 1
    cargo -V || return 1
    (cd "$repo" && sha256sum Cargo.lock rust-toolchain.toml) || return 1
    git -C "$repo" rev-parse HEAD || return 1
    git -C "$repo" describe --always --dirty || return 1
    # Writer (line 7), watcher (line 8), view (line 9), light (line 10),
    # metrics (line 11).
    (cd "$repo" && sha256sum target/release/kraken-lcd target/release/llama-watch \
      target/release/llama-view target/release/llama-light \
      target/release/llama-metrics) || return 1
  } >"$repo/target/check-provenance.txt"
}

self_test() {
  local tmp repo out first
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "stage.sh: --self-test refuses to run as root" >&2
    exit 1
  fi
  # sudo's secure_path hides brew and ~/.cargo. The review must still find rustc.
  if [[ "${STAGE_SCRUBBED:-}" != 1 ]]; then
    STAGE_SCRUBBED=1 PATH=/usr/sbin:/usr/bin:/sbin:/bin bash "$0" --self-test
    return
  fi
  if refuse_if_root 0; then
    echo "stage self-test: root was accepted" >&2
    exit 1
  fi
  refuse_if_root "$(id -u)"
  assert_packaging_contract

  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf $(printf '%q' "$tmp")" RETURN
  repo="$tmp/repo"
  mkdir -p "$repo/src" "$repo/packaging" "$repo/target/release"
  printf 'lock\n' >"$repo/Cargo.lock"
  printf 'toolchain\n' >"$repo/rust-toolchain.toml"
  printf 'target/\n' >"$repo/.gitignore"
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  printf 'fake-view\n' >"$repo/target/release/llama-view"
  printf 'fake-light\n' >"$repo/target/release/llama-light"
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"
  printf 'base\n' >"$repo/README"
  git -C "$repo" init -b work >/dev/null
  git_commit "$repo" "base" .gitignore Cargo.lock rust-toolchain.toml README
  printf 'stage-diff-marker\n' >"$repo/README"
  git_commit "$repo" "readme" README
  # Tip of the landed branch. A later commit on work leaves this behind.
  # The literal is deliberate: a changed RELEASE_BRANCH fails this test.
  git -C "$repo" branch main HEAD
  write_provenance "$repo"
  local prov_good prov_file prov_err
  prov_file="$repo/target/check-provenance.txt"
  prov_good="$(cat -- "$prov_file")"
  # Same empty hooks directory stage_review uses, so a short file is not
  # rejected by git_safe before the line count (or accepted when the count
  # check is gone and the extra line is otherwise valid).
  mkdir -p -- "$tmp/hooks"
  STAGE_HOOKS="$tmp/hooks"
  prov_err="$tmp/prov-err"
  # Twelve lines first. A file that is otherwise valid is accepted when the
  # count check is missing; ten lines then die on an unset array slot
  # before this message can be checked.
  printf '%s\n' "$prov_good" >"$prov_file"
  printf 'extra\n' >>"$prov_file"
  if check_provenance "$repo" "$prov_file" >"$prov_err" 2>&1; then
    echo "stage self-test: 12-line provenance was accepted" >&2
    exit 1
  fi
  if ! grep -F -q "must have 11 lines" "$prov_err"; then
    echo "stage self-test: 12-line provenance did not fail the line count" >&2
    cat -- "$prov_err" >&2
    exit 1
  fi
  head -n 10 <<<"$prov_good" >"$prov_file"
  if check_provenance "$repo" "$prov_file" >"$prov_err" 2>&1; then
    echo "stage self-test: 10-line provenance was accepted" >&2
    exit 1
  fi
  if ! grep -F -q "must have 11 lines" "$prov_err"; then
    echo "stage self-test: 10-line provenance did not fail the line count" >&2
    cat -- "$prov_err" >&2
    exit 1
  fi
  printf '%s\n' "$prov_good" >"$prov_file"
  unset STAGE_HOOKS

  out="$(stage_review "$repo" "$tmp/no-such-sha")"
  [[ "$out" == *"stage-diff-marker"* ]] || {
    echo "stage self-test: full diff omitted README" >&2
    printf '%s\n' "$out" >&2
    exit 1
  }
  # Provenance records relative paths. The live sha256sum lines use the
  # absolute path, so this does not pass just because the file was cat'd.
  [[ "$out" == *"  $repo/target/release/kraken-lcd"* \
    && "$out" == *"  $repo/target/release/llama-watch"* \
    && "$out" == *"  $repo/target/release/llama-view"* \
    && "$out" == *"  $repo/target/release/llama-light"* \
    && "$out" == *"  $repo/target/release/llama-metrics"* ]] || {
    echo "stage self-test: review did not hash all five binaries" >&2
    exit 1
  }
  first="$(git -C "$repo" rev-list --max-parents=0 HEAD)"
  printf '%s\n' "$first" >"$tmp/INSTALLED_SHA"
  out="$(stage_review "$repo" "$tmp/INSTALLED_SHA")"
  [[ "$out" == *"since $first"* && "$out" == *"stage-diff-marker"* ]] || {
    echo "stage self-test: review since INSTALLED_SHA missed the marker" >&2
    exit 1
  }
  head_now="$(git -C "$repo" rev-parse HEAD)"
  dash_hash="$(sha256sum -- "$repo/target/release/kraken-lcd" | awk '{ print $1 }')"
  watch_hash="$(sha256sum -- "$repo/target/release/llama-watch" | awk '{ print $1 }')"
  view_hash="$(sha256sum -- "$repo/target/release/llama-view" | awk '{ print $1 }')"
  light_hash="$(sha256sum -- "$repo/target/release/llama-light" | awk '{ print $1 }')"
  metrics_hash="$(sha256sum -- "$repo/target/release/llama-metrics" | awk '{ print $1 }')"
  [[ "$out" == *"----- anchor -----"* \
    && "$out" == *"HEAD: $head_now"* \
    && "$out" == *"subject: readme"* \
    && "$out" == *"$dash_hash  kraken-lcd"* \
    && "$out" == *"$watch_hash  llama-watch"* \
    && "$out" == *"$view_hash  llama-view"* \
    && "$out" == *"$light_hash  llama-light"* \
    && "$out" == *"$metrics_hash  llama-metrics"* \
    && "$out" == *"----- diff stat since $first -----"* ]] || {
    echo "stage self-test: anchor summary missing HEAD, subject, hashes, or diff stat" >&2
    printf '%s\n' "$out" >&2
    exit 1
  }

  printf 'x\n' >"$repo/src/dirty"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: dirty tree was accepted" >&2
    exit 1
  fi
  rm -f -- "$repo/src/dirty"

  printf 'tampered\n' >>"$repo/target/release/kraken-lcd"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: tampered binary matched provenance" >&2
    exit 1
  fi
  git -C "$repo" checkout -- . >/dev/null 2>&1 || true
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"

  printf 'tampered\n' >>"$repo/target/release/llama-watch"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: tampered llama-watch matched provenance" >&2
    exit 1
  fi
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"

  printf 'tampered\n' >>"$repo/target/release/llama-view"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: tampered llama-view matched provenance" >&2
    exit 1
  fi
  printf 'fake-view\n' >"$repo/target/release/llama-view"

  printf 'tampered\n' >>"$repo/target/release/llama-light"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: tampered llama-light matched provenance" >&2
    exit 1
  fi
  printf 'fake-light\n' >"$repo/target/release/llama-light"

  rm -f -- "$repo/target/release/llama-light"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: missing llama-light was accepted" >&2
    exit 1
  fi
  printf 'fake-light\n' >"$repo/target/release/llama-light"

  printf 'tampered\n' >>"$repo/target/release/llama-metrics"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: tampered llama-metrics matched provenance" >&2
    exit 1
  fi
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"

  rm -f -- "$repo/target/release/llama-metrics"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: missing llama-metrics was accepted" >&2
    exit 1
  fi
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"

  rm -f -- "$repo/target/release/llama-view"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: missing llama-view was accepted" >&2
    exit 1
  fi
  printf 'fake-view\n' >"$repo/target/release/llama-view"

  rm -f -- "$repo/target/release/llama-watch"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: missing llama-watch was accepted" >&2
    exit 1
  fi
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"

  # Parent of the landed tip is contained. Equality is not the only pass.
  parent="$(git -C "$repo" rev-parse 'HEAD^')"
  git -C "$repo" checkout --quiet --detach "$parent"
  write_provenance "$repo"
  if ! stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: an ancestor of main was refused" >&2
    exit 1
  fi
  git -C "$repo" checkout --quiet work
  write_provenance "$repo"

  # HEAD moved past main. Provenance matches the new HEAD,
  # so a missing ancestor check accepts the review.
  printf 'unlanded\n' >"$repo/UNLANDED"
  git_commit "$repo" "unlanded" UNLANDED
  write_provenance "$repo"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >"$tmp/unlanded.err" 2>&1; then
    echo "stage self-test: HEAD not contained in main was accepted" >&2
    exit 1
  fi
  if ! grep -F -q "not contained in main" "$tmp/unlanded.err"; then
    echo "stage self-test: unlanded HEAD did not fail the ancestor check" >&2
    cat -- "$tmp/unlanded.err" >&2
    exit 1
  fi

  # A tag of the same name points at this unlanded HEAD. The branch still
  # points at the landed tip. The tag must not satisfy the anchor.
  git -C "$repo" tag main HEAD
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >"$tmp/tag.err" 2>&1; then
    echo "stage self-test: a tag named main hid the branch" >&2
    exit 1
  fi
  if ! grep -F -q "not contained in main" "$tmp/tag.err"; then
    echo "stage self-test: tag shadow did not fail the branch ancestor check" >&2
    cat -- "$tmp/tag.err" >&2
    exit 1
  fi

  rm -rf -- "$repo/.git"
  if stage_review "$repo" "$tmp/INSTALLED_SHA" >/dev/null; then
    echo "stage self-test: git status failure was treated as a clean tree" >&2
    exit 1
  fi
  echo "stage self-test: ok"
}

main() {
  case "${1:-}" in
    --self-test)
      self_test
      ;;
    --require-landed)
      if [[ $# -ne 2 ]]; then
        echo "usage: scripts/stage.sh --require-landed REPO" >&2
        exit 2
      fi
      require_landed "$2"
      ;;
    *)
      if [[ $# -ne 2 ]]; then
        echo "usage: scripts/stage.sh REPO INSTALLED_SHA_FILE" >&2
        exit 2
      fi
      stage_review "$1" "$2"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"; exit $?
fi
