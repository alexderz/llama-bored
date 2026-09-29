#!/usr/bin/env bash
# Install kraken-lcd, llama-watch, llama-view, llama-light and
# llama-metrics. Run with sudo. It is idempotent.
#
# Root never runs git, cargo, rustc, or brew. scripts/stage.sh does that
# review as the repo owner ($SUDO_USER). Root's PATH is only
# /usr/sbin:/usr/bin:/sbin:/bin.
# Bytes are copied into a 0700 staging directory, and only those copies are
# installed. The provenance gate (SAFETY.md RR6) is an automatic anchor: a clean tree,
# a HEAD contained in the release branch (main), and hashes that match the staged
# copies. There is no typed-SHA prompt.
# After the files are in place, llama-watch is enabled for boot.
# kraken-lcd is never started, restarted, or enabled. The next step is
# printed for the operator. llama-light (RGB lighting) is installed but
# enabled for boot only with --enable-light; it is never started here.
# llama-metrics (the LAN Prometheus exporter) is installed but never enabled
# or started here, and the firewall is never touched: enabling it is a
# printed operator step.
#
# `--self-test` refuses root and does not call sudo, udevadm,
# systemd-sysusers, setfacl, or systemctl.
#
# Usage: sudo scripts/install.sh [--enable-light]
#        scripts/install.sh --self-test
#
# The body is one compound command. Bash reads through the closing brace
# before it runs anything, so bytes appended while it runs are not executed.
# `exit` then stops the shell from reading further.
{
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_PATH=/usr/sbin:/usr/bin:/sbin:/bin

dest_path() {
  local root=$1 rel=$2
  rel="${rel#/}"
  if [[ -z "$root" ]]; then
    printf '/%s\n' "$rel"
  else
    printf '%s/%s\n' "${root%/}" "$rel"
  fi
}

# shellcheck disable=SC2329 # called as `must place_file`
place_file() {
  local mode=$1 src=$2 dest=$3
  local dest_dir
  dest_dir="$(dirname -- "$dest")"
  if [[ -L "$dest" || -L "$dest_dir" ]]; then
    echo "install.sh: refusing to follow a symlink at $dest" >&2
    return 1
  fi
  mkdir -p -- "$dest_dir" || return 1
  if [[ "$(id -u)" -eq 0 ]]; then
    install -o root -g root -m "$mode" -- "$src" "$dest" || return 1
  elif [[ "${INSTALL_ALLOW_UNPRIV_COPY:-}" == 1 ]]; then
    install -m "$mode" -- "$src" "$dest" || return 1
    printf 'chown root:root %s\n' "$dest" >>"${INSTALL_LOG:?}"
  else
    echo "install.sh: not root, refusing to copy $dest" >&2
    return 1
  fi
}

# shellcheck disable=SC2329 # called as `must host_cmd`
host_cmd() {
  if [[ "${INSTALL_DRY:-}" == 1 ]]; then
    printf '%q ' "$@" >>"${INSTALL_LOG:?}"
    printf '\n' >>"${INSTALL_LOG:?}"
    return 0
  fi
  "$@"
}

copy_regular() {
  local src=$1 dest=$2
  if [[ ! -f "$src" || -L "$src" ]]; then
    echo "install.sh: $src is not a regular file" >&2
    return 1
  fi
  cp -- "$src" "$dest" || return 1
}

# Root-owned (when install_real is root) 0700 directory of the bytes to install.
# Prints the directory. Copies the provenance file too, so later checks do not
# re-read a user-writable path.
freeze_staging() {
  local repo=$1 dest_root=$2
  local dir config_dest config_new watch_dest light_dest metrics_dest
  dir="$(mktemp -d)"
  chmod 0700 "$dir" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/release/kraken-lcd" "$dir/kraken-lcd" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/release/llama-watch" "$dir/llama-watch" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/release/llama-view" "$dir/llama-view" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/release/llama-light" "$dir/llama-light" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/llama-light.service" "$dir/llama-light.service" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/94-llama-light-hidraw.rules" "$dir/94-llama-light-hidraw.rules" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/release/llama-metrics" "$dir/llama-metrics" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/llama-metrics.service" "$dir/llama-metrics.service" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/kraken-lcd.service" "$dir/kraken-lcd.service" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/llama-watch.service" "$dir/llama-watch.service" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/llama-bored.sysusers" "$dir/llama-bored.sysusers" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/71-kraken-lcd.rules" "$dir/71-kraken-lcd.rules" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/93-kraken-lcd-hidraw.rules" "$dir/93-kraken-lcd-hidraw.rules" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/72-llama-view.rules" "$dir/72-llama-view.rules" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/fonts/llama-hack-12x24.psfu" "$dir/llama-hack-12x24.psfu" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/user/kraken-lcd-halt.path" "$dir/kraken-lcd-halt.path" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/packaging/user/kraken-lcd-halt-notify.service" \
    "$dir/kraken-lcd-halt-notify.service" || {
    rm -rf -- "$dir"
    return 1
  }
  copy_regular "$repo/target/check-provenance.txt" "$dir/check-provenance.txt" || {
    rm -rf -- "$dir"
    return 1
  }
  config_dest="$(dest_path "$dest_root" /etc/llama-bored/config.toml)"
  if [[ -e "$config_dest" ]]; then
    if [[ -L "$config_dest" || ! -f "$config_dest" ]]; then
      echo "install.sh: $config_dest is not a regular file" >&2
      rm -rf -- "$dir"
      return 1
    fi
    copy_regular "$config_dest" "$dir/config.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'keep\n' >"$dir/config.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  else
    copy_regular "$repo/packaging/config.example.toml" "$dir/config.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'install\n' >"$dir/config.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  fi
  config_new="$(dest_path "$dest_root" /etc/llama-bored/config.toml.new)"
  if [[ -e "$config_new" ]]; then
    if [[ -L "$config_new" || ! -f "$config_new" ]]; then
      echo "install.sh: $config_new is not a regular file" >&2
      rm -rf -- "$dir"
      return 1
    fi
    copy_regular "$config_new" "$dir/config.toml.new" || {
      rm -rf -- "$dir"
      return 1
    }
  fi
  watch_dest="$(dest_path "$dest_root" /etc/llama-bored/watch.toml)"
  if [[ -e "$watch_dest" ]]; then
    if [[ -L "$watch_dest" || ! -f "$watch_dest" ]]; then
      echo "install.sh: $watch_dest is not a regular file" >&2
      rm -rf -- "$dir"
      return 1
    fi
    copy_regular "$watch_dest" "$dir/watch.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'keep\n' >"$dir/watch.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  else
    copy_regular "$repo/packaging/watch.example.toml" "$dir/watch.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'install\n' >"$dir/watch.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  fi
  # light.toml: the operator's file is kept; the example is installed only
  # when there is none.
  light_dest="$(dest_path "$dest_root" /etc/llama-bored/light.toml)"
  if [[ -e "$light_dest" ]]; then
    if [[ -L "$light_dest" || ! -f "$light_dest" ]]; then
      echo "install.sh: $light_dest is not a regular file" >&2
      rm -rf -- "$dir"
      return 1
    fi
    copy_regular "$light_dest" "$dir/light.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'keep\n' >"$dir/light.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  else
    copy_regular "$repo/packaging/light.example.toml" "$dir/light.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'install\n' >"$dir/light.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  fi
  # metrics.toml: the operator's file is kept; the example is installed only
  # when there is none.
  metrics_dest="$(dest_path "$dest_root" /etc/llama-bored/metrics.toml)"
  if [[ -e "$metrics_dest" ]]; then
    if [[ -L "$metrics_dest" || ! -f "$metrics_dest" ]]; then
      echo "install.sh: $metrics_dest is not a regular file" >&2
      rm -rf -- "$dir"
      return 1
    fi
    copy_regular "$metrics_dest" "$dir/metrics.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'keep\n' >"$dir/metrics.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  else
    copy_regular "$repo/packaging/metrics.example.toml" "$dir/metrics.toml" || {
      rm -rf -- "$dir"
      return 1
    }
    printf 'install\n' >"$dir/metrics.mode" || {
      rm -rf -- "$dir"
      return 1
    }
  fi
  printf '%s\n' "$dir"
}

show_staging_hashes() {
  local staging=$1
  local -a files=()
  echo "----- staged sha256 (these bytes are what will be installed) -----"
  files=(
    "$staging/kraken-lcd"
    "$staging/llama-watch"
    "$staging/llama-view"
    "$staging/llama-light"
    "$staging/llama-metrics"
    "$staging/kraken-lcd.service"
    "$staging/llama-watch.service"
    "$staging/llama-light.service"
    "$staging/llama-metrics.service"
    "$staging/llama-bored.sysusers"
    "$staging/71-kraken-lcd.rules"
    "$staging/72-llama-view.rules"
    "$staging/93-kraken-lcd-hidraw.rules"
    "$staging/94-llama-light-hidraw.rules"
    "$staging/llama-hack-12x24.psfu"
    "$staging/config.toml"
    "$staging/light.toml"
    "$staging/metrics.toml"
  )
  # watch.toml is always staged. config.toml.new is the operator's hand-off, listed
  # when he prepared one, so the hashes cover the bytes that will be installed.
  [[ -f "$staging/watch.toml" ]] && files+=("$staging/watch.toml")
  [[ -f "$staging/config.toml.new" ]] && files+=("$staging/config.toml.new")
  files+=(
    "$staging/kraken-lcd-halt.path"
    "$staging/kraken-lcd-halt-notify.service"
  )
  sha256sum -- "${files[@]}" || return 1
}

verify_staged_binary() {
  local staging=$1 got want count i name line
  local -a names=(kraken-lcd llama-watch llama-view llama-light llama-metrics)
  local -a lines=(7 8 9 10 11)
  # Line 7 is the writer, line 8 the watcher, line 9 the view, line 10 the
  # light writer, line 11 the metrics exporter. Same order as check.sh.
  count="$(awk 'END { print NR }' "$staging/check-provenance.txt")"
  if [[ "$count" -ne 11 ]]; then
    echo "install.sh: staged provenance must have 11 lines" >&2
    return 1
  fi
  for i in 0 1 2 3 4; do
    name="${names[$i]}"
    line="${lines[$i]}"
    if ! got="$(sha256sum -- "$staging/$name")"; then
      echo "install.sh: sha256sum of the staged $name failed" >&2
      return 1
    fi
    got="${got%% *}"
    if ! want="$(awk -v n="$line" 'NR==n { print $1; exit }' "$staging/check-provenance.txt")"; then
      echo "install.sh: failed to read staged provenance" >&2
      return 1
    fi
    if [[ -z "$want" || "$got" != "$want" ]]; then
      echo "install.sh: staged $name hash $got does not match check-provenance.txt line $line (${want:-missing})" >&2
      return 1
    fi
  done
}

# After stage.sh returns, the files it reviewed must still be the staged bytes.
verify_staged_matches_disk() {
  local staging=$1 repo=$2 dest_root=$3
  local config_mode config_src config_new watch_mode watch_src light_mode light_src pair staged live
  local metrics_mode metrics_src
  local -a pairs=(
    kraken-lcd:target/release/kraken-lcd
    llama-watch:target/release/llama-watch
    llama-view:target/release/llama-view
    llama-light:target/release/llama-light
    llama-light.service:packaging/llama-light.service
    94-llama-light-hidraw.rules:packaging/94-llama-light-hidraw.rules
    llama-metrics:target/release/llama-metrics
    llama-metrics.service:packaging/llama-metrics.service
    kraken-lcd.service:packaging/kraken-lcd.service
    llama-watch.service:packaging/llama-watch.service
    llama-bored.sysusers:packaging/llama-bored.sysusers
    71-kraken-lcd.rules:packaging/71-kraken-lcd.rules
    72-llama-view.rules:packaging/72-llama-view.rules
    93-kraken-lcd-hidraw.rules:packaging/93-kraken-lcd-hidraw.rules
    llama-hack-12x24.psfu:packaging/fonts/llama-hack-12x24.psfu
    kraken-lcd-halt.path:packaging/user/kraken-lcd-halt.path
    kraken-lcd-halt-notify.service:packaging/user/kraken-lcd-halt-notify.service
    check-provenance.txt:target/check-provenance.txt
  )
  for pair in "${pairs[@]}"; do
    staged="${pair%%:*}"
    live="${pair#*:}"
    if ! cmp -s -- "$staging/$staged" "$repo/$live"; then
      echo "install.sh: staged $staged does not match the reviewed $live" >&2
      return 1
    fi
  done
  if ! config_mode="$(tr -d '[:space:]' <"$staging/config.mode")"; then
    echo "install.sh: staged config mode is unreadable" >&2
    return 1
  fi
  if [[ "$config_mode" == "install" ]]; then
    config_src="$repo/packaging/config.example.toml"
  else
    config_src="$(dest_path "$dest_root" /etc/llama-bored/config.toml)"
  fi
  if ! cmp -s -- "$staging/config.toml" "$config_src"; then
    echo "install.sh: staged config does not match the reviewed config" >&2
    return 1
  fi
  if ! watch_mode="$(tr -d '[:space:]' <"$staging/watch.mode")"; then
    echo "install.sh: staged watch mode is unreadable" >&2
    return 1
  fi
  if [[ "$watch_mode" == "install" ]]; then
    watch_src="$repo/packaging/watch.example.toml"
  else
    watch_src="$(dest_path "$dest_root" /etc/llama-bored/watch.toml)"
  fi
  if ! cmp -s -- "$staging/watch.toml" "$watch_src"; then
    echo "install.sh: staged watch.toml does not match the reviewed watch.toml" >&2
    return 1
  fi
  if ! light_mode="$(tr -d '[:space:]' <"$staging/light.mode")"; then
    echo "install.sh: staged light mode is unreadable" >&2
    return 1
  fi
  if [[ "$light_mode" == "install" ]]; then
    light_src="$repo/packaging/light.example.toml"
  else
    light_src="$(dest_path "$dest_root" /etc/llama-bored/light.toml)"
  fi
  if ! cmp -s -- "$staging/light.toml" "$light_src"; then
    echo "install.sh: staged light.toml does not match the reviewed light.toml" >&2
    return 1
  fi
  if ! metrics_mode="$(tr -d '[:space:]' <"$staging/metrics.mode")"; then
    echo "install.sh: staged metrics mode is unreadable" >&2
    return 1
  fi
  if [[ "$metrics_mode" == "install" ]]; then
    metrics_src="$repo/packaging/metrics.example.toml"
  else
    metrics_src="$(dest_path "$dest_root" /etc/llama-bored/metrics.toml)"
  fi
  if ! cmp -s -- "$staging/metrics.toml" "$metrics_src"; then
    echo "install.sh: staged metrics.toml does not match the reviewed metrics.toml" >&2
    return 1
  fi
  config_new="$(dest_path "$dest_root" /etc/llama-bored/config.toml.new)"
  if [[ -f "$staging/config.toml.new" ]]; then
    if ! cmp -s -- "$staging/config.toml.new" "$config_new"; then
      echo "install.sh: staged config.toml.new does not match the reviewed file" >&2
      return 1
    fi
  elif [[ -e "$config_new" ]]; then
    echo "install.sh: config.toml.new appeared after staging" >&2
    return 1
  fi
}

provenance_head() {
  local file=$1 head
  if ! head="$(awk 'NR==5 { print; exit }' "$file")"; then
    echo "install.sh: failed to read staged provenance HEAD" >&2
    return 1
  fi
  if [[ ! "$head" =~ ^[0-9a-f]{40}$ ]]; then
    echo "install.sh: staged provenance HEAD is not a commit id" >&2
    return 1
  fi
  printf '%s\n' "$head"
}

min_interval_of() {
  local file=$1 raw
  if [[ ! -f "$file" || -L "$file" ]]; then
    echo "install.sh: config is missing: $file" >&2
    return 1
  fi
  if ! raw="$(
    awk '
      function trim(s) {
        sub(/^[ \t]+/, "", s)
        sub(/[ \t]+$/, "", s)
        return s
      }
      /^[ \t]*#/ { next }
      /^[ \t]*\[/ {
        section = $0
        sub(/[ \t]*#.*/, "", section)
        section = trim(section)
        next
      }
      section != "[upload]" { next }
      {
        line = $0
        sub(/[ \t]*#.*/, "", line)
        if (line ~ /^[ \t]*min_interval_s[ \t]*=/) {
          sub(/^[ \t]*min_interval_s[ \t]*=[ \t]*/, "", line)
          sub(/[ \t].*/, "", line)
          value = line
        }
      }
      END {
        if (value == "") print "60"
        else print value
      }
    ' "$file"
  )"; then
    echo "install.sh: failed to read $file" >&2
    return 1
  fi
  if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
    echo "install.sh: upload.min_interval_s is not an integer ('$raw')" >&2
    return 1
  fi
  if [[ "$raw" -lt 10 ]]; then
    echo "install.sh: upload.min_interval_s $raw is below the floor of 10" >&2
    return 1
  fi
  printf '%s\n' "$raw"
}

hidraw_mode_group() {
  local node=$1
  if [[ -n "${INSTALL_FAKE_STAT+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_STAT"
    return 0
  fi
  stat -c '%a %G' -- "$node"
}

hidraw_has_user_acl() {
  local node=$1 text
  if [[ -n "${INSTALL_FAKE_GETFACL+x}" ]]; then
    text="$INSTALL_FAKE_GETFACL"
  else
    if ! text="$(getfacl --absolute-names -- "$node")"; then
      echo "install.sh: getfacl failed for $node" >&2
      return 1
    fi
  fi
  grep -E -q '(^|:)user:[^:]+:' <<<"$text"
}

# SAFETY.md RR7: where the udev symlink from 93-kraken-lcd-hidraw.rules points.
# Prints nothing when it is absent. readlink only; the node is not opened.
hidraw_pin_target() {
  if [[ -n "${INSTALL_FAKE_PIN+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_PIN"
    return 0
  fi
  readlink -e -- /dev/kraken-lcd/hid || true
}

# The Aura controller's pin from 94-llama-light-hidraw.rules.
# readlink only; the node is not opened.
aura_pin_target() {
  if [[ -n "${INSTALL_FAKE_AURA_PIN+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_AURA_PIN"
    return 0
  fi
  readlink -e -- /dev/llama-light/aura || true
}

aura_mode_group() {
  local node=$1
  if [[ -n "${INSTALL_FAKE_AURA_STAT+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_AURA_STAT"
    return 0
  fi
  stat -c '%a %G' -- "$node"
}

# /dev/hidrawN of the ASUS Aura USB controller (HID 0003:0B05:18F3), or
# nothing when it is not attached. More than one is an error. sysfs reads only.
resolve_aura_hidraw() {
  local sys=$1 dir uevent
  local -a found=()
  [[ -d "$sys/class/hidraw" ]] || return 0
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    uevent="$dir/device/uevent"
    [[ -f "$uevent" ]] || continue
    if grep -q -x 'HID_ID=0003:00000B05:000018F3' "$uevent"; then
      found+=("$(basename -- "$dir")")
    fi
  done < <(find "$sys/class/hidraw" -mindepth 1 -maxdepth 1 -name 'hidraw[0-9]*' -print 2>/dev/null || true)
  if [[ ${#found[@]} -gt 1 ]]; then
    echo "install.sh: expected at most one Aura (0b05:18f3) hidraw node, found ${#found[@]}" >&2
    return 1
  fi
  if [[ ${#found[@]} -eq 1 ]]; then
    printf '/dev/%s\n' "${found[0]}"
  fi
}

# The keyboard's lighting pin from 94-llama-light-hidraw.rules.
# readlink only; the node is not opened.
keyboard_pin_target() {
  if [[ -n "${INSTALL_FAKE_KBD_PIN+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_KBD_PIN"
    return 0
  fi
  readlink -e -- /dev/llama-light/keyboard || true
}

keyboard_mode_group() {
  local node=$1
  if [[ -n "${INSTALL_FAKE_KBD_STAT+x}" ]]; then
    printf '%s\n' "$INSTALL_FAKE_KBD_STAT"
    return 0
  fi
  stat -c '%a %G' -- "$node"
}

# /dev/hidrawN of the Corsair STRAFE RGB MK.2's lighting interface (HID
# 0003:1B1C:1B48 on USB interface 01, as the udev rule picks it), or nothing
# when the keyboard is not attached. More than one is an error. sysfs reads
# only; the keyboard's other interfaces (typing) are left alone.
resolve_keyboard_hidraw() {
  local sys=$1 dir uevent hid iface
  local -a found=()
  [[ -d "$sys/class/hidraw" ]] || return 0
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    uevent="$dir/device/uevent"
    [[ -f "$uevent" ]] || continue
    grep -q -x 'HID_ID=0003:00001B1C:00001B48' "$uevent" || continue
    hid="$(readlink -f -- "$dir/device")" || continue
    iface="$(dirname -- "$hid")/bInterfaceNumber"
    [[ -f "$iface" ]] || continue
    if [[ "$(tr -d '[:space:]' <"$iface")" == "01" ]]; then
      found+=("$(basename -- "$dir")")
    fi
  done < <(find "$sys/class/hidraw" -mindepth 1 -maxdepth 1 -name 'hidraw[0-9]*' -print 2>/dev/null || true)
  if [[ ${#found[@]} -gt 1 ]]; then
    echo "install.sh: expected at most one keyboard lighting (1b1c:1b48 interface 01) hidraw node, found ${#found[@]}" >&2
    return 1
  fi
  if [[ ${#found[@]} -eq 1 ]]; then
    printf '/dev/%s\n' "${found[0]}"
  fi
}

resolve_kraken_hidraw() {
  local sys=$1
  local devices="$sys/bus/usb/devices"
  local dev base vendor product device name iface hid node target driver_name raw
  local -a found=() nodes=() hidraws=()
  if [[ ! -d "$devices" ]]; then
    echo "install.sh: $devices is not a directory" >&2
    return 1
  fi
  while IFS= read -r dev; do
    [[ -n "$dev" ]] || continue
    base="$(basename -- "$dev")"
    [[ "$base" == *:* || "$base" == usb* ]] && continue
    [[ -d "$dev" && -f "$dev/idVendor" && -f "$dev/idProduct" ]] || continue
    vendor="$(tr -d '[:space:]' <"$dev/idVendor")"
    product="$(tr -d '[:space:]' <"$dev/idProduct")"
    vendor="${vendor,,}"
    product="${product,,}"
    if [[ "$vendor" == "1e71" && "$product" == "3008" ]]; then
      found+=("$dev")
    fi
  done < <(find "$devices" -mindepth 1 -maxdepth 1 -print 2>/dev/null || true)
  if [[ ${#found[@]} -ne 1 ]]; then
    echo "install.sh: expected one 1e71:3008 device, found ${#found[@]}" >&2
    return 1
  fi
  device="${found[0]}"
  name="$(basename -- "$device")"
  iface="$device/${name}:1.1"
  [[ -d "$iface" ]] || {
    echo "install.sh: $iface is missing" >&2
    return 1
  }
  while IFS= read -r hid; do
    [[ -n "$hid" && -f "$hid/uevent" ]] || continue
    if grep -q -x 'HID_ID=0003:00001E71:00003008' "$hid/uevent"; then
      nodes+=("$hid")
    fi
  done < <(find "$iface" -mindepth 1 -maxdepth 1 -print 2>/dev/null || true)
  if [[ ${#nodes[@]} -ne 1 ]]; then
    echo "install.sh: expected one Kraken HID node, found ${#nodes[@]}" >&2
    return 1
  fi
  node="${nodes[0]}"
  if [[ ! -L "$node/driver" ]]; then
    echo "install.sh: Kraken HID driver link is missing" >&2
    return 1
  fi
  target="$(readlink -- "$node/driver")"
  driver_name="$(basename -- "$target")"
  if [[ "$driver_name" != "nzxt_kraken3" ]]; then
    echo "install.sh: Kraken HID driver is $driver_name, expected nzxt_kraken3" >&2
    return 1
  fi
  if [[ ! -d "$node/hidraw" ]]; then
    echo "install.sh: $node/hidraw is missing" >&2
    return 1
  fi
  while IFS= read -r raw; do
    [[ -n "$raw" ]] || continue
    base="$(basename -- "$raw")"
    [[ "$base" =~ ^hidraw[0-9]+$ ]] || continue
    hidraws+=("$base")
  done < <(find "$node/hidraw" -mindepth 1 -maxdepth 1 -print 2>/dev/null || true)
  if [[ ${#hidraws[@]} -ne 1 ]]; then
    echo "install.sh: expected one hidraw node, found ${#hidraws[@]}" >&2
    return 1
  fi
  printf '/dev/%s\n' "${hidraws[0]}"
}

# A hand-written trial rule that gives the Kraken (vendor 1e71) to one login
# user with OWNER= must be removed first: it would bypass the kraken-lcd group.
trial_rule_files() {
  local dir hit
  for dir in "$@"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r hit; do
      [[ -n "$hit" ]] || continue
      if grep -E -q 'OWNER[[:space:]]*:?=' "$hit" 2>/dev/null; then
        printf '%s\n' "$hit"
      fi
    # udev only loads files ending in .rules; our own .bak-<sha> backups are inert.
    done < <(grep -R -i -l -F --include='*.rules' '1e71' "$dir" 2>/dev/null || true)
  done
}

# shellcheck disable=SC2329 # called as `must rewrite_restart_sec`
rewrite_restart_sec() {
  local unit=$1 sec=$2 tmp
  tmp="$(mktemp)"
  if ! awk -v sec="$sec" '
    /^RestartSec=/ { count++; print "RestartSec=" sec; next }
    { print }
    END { if (count != 1) exit 1 }
  ' "$unit" >"$tmp"; then
    rm -f -- "$tmp"
    echo "install.sh: $unit has no single RestartSec= line" >&2
    return 1
  fi
  place_file 0644 "$tmp" "$unit" || {
    rm -f -- "$tmp"
    return 1
  }
  rm -f -- "$tmp"
}

# Paths in the printed rollback are host paths (`/etc`, `/usr`). A self-test
# dest root is stripped here and put back when the text is executed.
display_path() {
  local path=$1 root="${INSTALL_DEST_ROOT%/}"
  if [[ -n "$root" && "$path" == "$root/"* ]]; then
    printf '%s\n' "${path#"$root"}"
  else
    printf '%s\n' "$path"
  fi
}

# One line in this run's record. `kind` is created, created-dir or replaced.
# A replaced file names the backup this run would restore; a created file or
# directory has no backup.
rollback_record() {
  local kind=$1 path=$2 bak=${3:-}
  local shown
  ROLLBACK_KIND+=("$kind")
  ROLLBACK_PATH+=("$path")
  ROLLBACK_BAK+=("$bak")
  shown="$(display_path "$path")"
  if [[ "$kind" == "replaced" ]]; then
    mark_write "replaced ${shown} (backup at $(display_path "$bak"))"
  elif [[ "$kind" == "created-dir" ]]; then
    mark_write "created directory ${shown}"
  else
    mark_write "created ${shown}"
  fi
}

# shellcheck disable=SC2329 # called from commit_new_file
rollback_note_rules() {
  local base
  base="$(basename -- "$1")"
  if [[ "$base" == "71-kraken-lcd.rules" || "$base" == "93-kraken-lcd-hidraw.rules" \
    || "$base" == "72-llama-view.rules" || "$base" == "94-llama-light-hidraw.rules" ]]; then
    INSTALL_RULES_CHANGED=1
  fi
}

# "Created llama-watch" means both the binary and the unit were absent
# before this run, and this run created at least one of them. An existing
# INSTALLED_SHA (from an older LCD-writer-only install) does not change that.
# shellcheck disable=SC2329 # called from commit_new_file
rollback_note_watch() {
  local base
  base="$(basename -- "$1")"
  if [[ "$base" == "llama-watch" || "$base" == "llama-watch.service" ]]; then
    if [[ "${INSTALL_WATCH_WAS_ABSENT:-}" == 1 ]]; then
      INSTALL_CREATED_WATCH=1
    fi
  fi
}

# "Created llama-metrics": its binary and unit were both absent before this
# run. The installer never enables it, but the operator may have since, so
# the rollback disables it before removing its unit.
# shellcheck disable=SC2329 # called from commit_new_file
rollback_note_metrics() {
  local base
  base="$(basename -- "$1")"
  if [[ "$base" == "llama-metrics" || "$base" == "llama-metrics.service" ]]; then
    if [[ "${INSTALL_METRICS_WAS_ABSENT:-}" == 1 ]]; then
      INSTALL_CREATED_METRICS=1
    fi
  fi
}

# A failed run with a non-empty record leaves this file. The next install
# refuses until the printed rollback removes it.
rollback_marker_path() {
  dest_path "${INSTALL_DEST_ROOT}" /usr/local/libexec/llama-bored/ROLLBACK_PENDING
}

# Refuse while a previous failure's rollback is still pending. A symlink is
# refused on its own, before any install write.
refuse_rollback_pending() {
  local marker dir
  marker="$(rollback_marker_path)"
  dir="$(dirname -- "$marker")"
  if [[ -L "$marker" || -L "$dir" ]]; then
    echo "install.sh: refusing to follow a symlink at $marker" >&2
    return 1
  fi
  if [[ -e "$marker" ]]; then
    echo "install.sh: a previous install failed; run the rollback in $marker first (or remove the marker if you have already rolled back)" >&2
    return 1
  fi
}

# mktemp in the marker's directory (same filesystem), then mv into place.
# As root the temp file is root-owned. A symlink is not followed.
# shellcheck disable=SC2329 # called from the EXIT hook
save_rollback_marker() {
  local text=$1
  local dir marker tmp
  [[ ${#ROLLBACK_KIND[@]} -gt 0 ]] || return 0
  marker="$(rollback_marker_path)"
  dir="$(dirname -- "$marker")"
  if [[ -L "$marker" || -L "$dir" ]]; then
    echo "install.sh: refusing to follow a symlink at $marker" >&2
    return 1
  fi
  # Self-test only: install_real unsets this before a real run.
  if [[ -n "${INSTALL_FAIL_MARKER:-}" ]]; then
    return 1
  fi
  mkdir -p -- "$dir" || return 1
  tmp="$(mktemp "$dir/ROLLBACK_PENDING.XXXXXX")" || return 1
  chmod 0644 -- "$tmp" || {
    rm -f -- "$tmp"
    return 1
  }
  if [[ "$(id -u)" -eq 0 ]]; then
    chown root:root -- "$tmp" || {
      rm -f -- "$tmp"
      return 1
    }
  fi
  printf '%s\n' "$text" >"$tmp" || {
    rm -f -- "$tmp"
    return 1
  }
  mv -f -- "$tmp" "$marker" || {
    rm -f -- "$tmp"
    return 1
  }
  echo "install.sh: wrote the rollback to $marker" >&2
}

# Printed on success and from the EXIT hook. The lines undo exactly the
# files this run recorded. They do not look at whatever .bak files are on disk.
# The marker rm is the last step, including when this run did not write one.
print_rollback() {
  local prev="${INSTALL_PREV_SHA:-none}"
  local i path bak shown shown_bak shown_new n marker_shown
  echo "install.sh: how to roll back to the previous INSTALLED_SHA (${prev}):"
  if [[ "${INSTALL_CREATED_WATCH:-0}" == 1 ]]; then
    echo "  systemctl disable --now llama-watch.service"
  fi
  if [[ "${INSTALL_ENABLED_LIGHT:-0}" == 1 ]]; then
    echo "  systemctl disable --now llama-light.service"
  fi
  if [[ "${INSTALL_CREATED_METRICS:-0}" == 1 ]]; then
    echo "  systemctl disable --now llama-metrics.service"
  fi
  n=${#ROLLBACK_KIND[@]}
  for ((i = n - 1; i >= 0; i--)); do
    path="${ROLLBACK_PATH[$i]}"
    shown="$(display_path "$path")"
    if [[ "${ROLLBACK_KIND[$i]}" == "replaced" ]]; then
      bak="${ROLLBACK_BAK[$i]}"
      shown_bak="$(display_path "$bak")"
      # Keep the migrated writer config. This runs before the backup is moved
      # back over config.toml, so the bytes are the ones just installed.
      if [[ "$(basename -- "$path")" == "config.toml" ]]; then
        shown_new="$(display_path "${path}.new")"
        printf '  cp -p -- %q %q\n' "$shown" "$shown_new"
      fi
      printf '  mv -f -- %q %q\n' "$shown_bak" "$shown"
    elif [[ "${ROLLBACK_KIND[$i]}" == "created-dir" ]]; then
      # rmdir refuses a non-empty directory, so writer state (such as a
      # HALTED latch) written after this run is never deleted.
      printf '  rmdir -- %q\n' "$shown"
    else
      printf '  rm -f -- %q\n' "$shown"
    fi
  done
  echo "  systemctl daemon-reload"
  if [[ "${INSTALL_RULES_CHANGED:-0}" == 1 ]]; then
    echo "  udevadm control --reload"
    echo "  udevadm trigger --action=change --attr-match=idVendor=1e71 --attr-match=idProduct=3008"
    echo "  udevadm trigger --action=change --sysname-match=vcsa11"
    echo "  udevadm trigger --action=change --sysname-match=vcs11"
    echo "  udevadm trigger --action=change --sysname-match=vcsu11"
    echo "  udevadm trigger --action=change --attr-match=idVendor=0b05 --attr-match=idProduct=18f3"
    echo "  udevadm trigger --action=change --attr-match=idVendor=1b1c --attr-match=idProduct=1b48"
    echo "  udevadm trigger --action=change --subsystem-match=hidraw"
  fi
  echo "  # INSTALLED_SHA stays the previous value (${prev})."
  marker_shown="$(display_path "$(rollback_marker_path)")"
  printf '  rm -f -- %q\n' "$marker_shown"
}

print_phase_b() {
  cat <<'EOF'
install.sh: files are installed. llama-watch is enabled for boot but not
started. kraken-lcd is neither started nor enabled. This script never runs
the steps below; run them yourself, as root, one at a time.

  # 1. llama-watch is enabled; start it (tty11 dashboard, Ctrl+Alt+F11 to look):
  systemctl start llama-watch

  # 2. Create the state dir if needed (systemd would on the first start):
  install -d -o kraken-lcd -g kraken-lcd -m 0755 /var/lib/kraken-lcd

  # 3. Test the LCD with the writer stopped, as the kraken-lcd user.
  #    <test-card> is a copy of fixtures/views/test-card.json that the
  #    kraken-lcd user can read. Compare scripts/cooling-snapshot.sh
  #    before and after.
  runuser -u kraken-lcd -- /usr/local/libexec/llama-bored/kraken-lcd show-image --view <test-card>

  # 4. Then start the writer:
  systemctl enable --now kraken-lcd

Upgrading an install that already runs? Move the running units onto the new
binaries. try-restart restarts only units that are running now (here any of
llama-watch, kraken-lcd, llama-light, llama-metrics); a stopped unit stays
stopped and nothing is enabled:
  systemctl try-restart llama-watch kraken-lcd llama-light llama-metrics

Optional:
  systemctl --global enable kraken-lcd-halt.path   # desktop alert on HALTED
  usermod -aG llama-view $SUDO_USER   # mirror tty11 with llama-view; log in again
EOF
  print_light_steps
  print_metrics_steps
  print_rollback
}

print_metrics_steps() {
  cat <<'EOF'

Prometheus exporter (llama-metrics) is installed, not enabled. It is
LAN-exposed by design and serves numbers only (no prompt or output text).
To turn it on:
  /usr/local/libexec/llama-bored/llama-metrics check --config /etc/llama-bored/metrics.toml
  systemctl enable --now llama-metrics
  curl -s http://127.0.0.1:19477/metrics | head
  Keep IPAddressAllow= and SocketBindAllow= in llama-metrics.service equal
  to allow and the listen port in /etc/llama-bored/metrics.toml.
  Open the port in the firewall zone that holds your LAN yourself, for example:
  firewall-cmd --permanent --zone=internal --add-port=19477/tcp && firewall-cmd --reload
EOF
}

print_light_steps() {
  if [[ "${INSTALL_ENABLED_LIGHT:-0}" == 1 ]]; then
    cat <<'EOF'

RGB lighting (llama-light) is enabled for boot but not started:
  /usr/local/libexec/llama-bored/llama-light check --config /etc/llama-bored/light.toml
  systemctl start llama-light
EOF
  else
    cat <<'EOF'

RGB lighting (llama-light) is installed, not enabled. To try it:
  /usr/local/libexec/llama-bored/llama-light check --config /etc/llama-bored/light.toml
  systemctl start llama-light      # stop: systemctl stop llama-light
  systemctl enable llama-light     # keep it across reboots
EOF
  fi
}

# Writer top-level names, matching Config in crates/kraken-lcd/src/config.rs.
# A [collector] inside a multi-line string can false-positive; that fails safe.
config_scan() {
  local file=$1
  awk '
    function top_name(s) {
      gsub(/[ \t\[\]"\047]/, "", s)
      sub(/\..*/, "", s)
      return s
    }
    function strip_comment(s,    i, c, q) {
      q = ""
      for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (q == "" && c == "#") return substr(s, 1, i - 1)
        if (c == "\"" || c == "\047") {
          if (q == c) q = ""
          else if (q == "") q = c
        }
      }
      return s
    }
    function allowed(n) {
      return n == "writer" || n == "snapshot" || n == "dial" || n == "upload" || n == "bands" || n == "hysteresis" || n == "display"
    }
    function bracket_delta(s,    i, c, q, d) {
      q = ""
      d = 0
      for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (q == "" && c == "#") break
        if (c == "\"" || c == "\047") {
          if (q == c) q = ""
          else if (q == "") q = c
          continue
        }
        if (q != "") continue
        if (c == "[") d++
        else if (c == "]") d--
      }
      return d
    }
    BEGIN { in_table = 0; table_move = 0; value_depth = 0 }
    {
      raw = $0
      sub(/\r$/, "", raw)
      if (raw ~ /^[ \t]*#/) next
      if (raw ~ /^[ \t]*$/) next
      t = strip_comment(raw)
      sub(/^[ \t]+/, "", t)
      sub(/[ \t]+$/, "", t)
      if (t == "") next
      # A multi-line array value (`tiers = [` then `  [0.5, 10],`) is still
      # inside the table. Bracket depth keeps those lines from becoming keys.
      if (value_depth > 0) {
        value_depth += bracket_delta(t)
        if (value_depth < 0) value_depth = 0
        next
      }
      if (t ~ /^\[/) {
        name = top_name(t)
        in_table = 1
        if (name == "" || allowed(name)) {
          table_move = 0
        } else if (name == "collector" || name == "models") {
          table_move = 1
          print "MOVE\t" name
        } else {
          table_move = 0
          print "UNKNOWN\t" name
        }
        next
      }
      if (t !~ /=/) next
      key = t
      sub(/[ \t]*=.*/, "", key)
      sub(/^[ \t]+/, "", key)
      sub(/[ \t]+$/, "", key)
      sub(/\r$/, "", key)
      if (in_table) {
        value_depth += bracket_delta(t)
        if (value_depth < 0) value_depth = 0
        if (table_move && key != "") print "KEY\t" key
        next
      }
      value_depth += bracket_delta(t)
      if (value_depth < 0) value_depth = 0
      name = top_name(key)
      if (name == "" || allowed(name)) next
      if (name == "collector" || name == "models") {
        print "MOVE\t" name
        rest = key
        gsub(/[ \t\[\]"\047]/, "", rest)
        if (index(rest, ".") > 0) {
          sub(/^[^.]*\./, "", rest)
          sub(/\..*/, "", rest)
          if (rest != "") print "KEY\t" rest
        }
      } else {
        print "UNKNOWN\t" name
      }
    }
  ' "$file"
}

config_is_clean() {
  local file=$1 out
  if ! out="$(config_scan "$file")"; then
    echo "install.sh: failed to read $file" >&2
    return 1
  fi
  [[ -z "$out" ]]
}

explain_problems() {
  local file=$1 dest_root=$2
  local kind name
  local -a move=() unknown=()
  local seen_move=$'\n' seen_unknown=$'\n'
  while IFS=$'\t' read -r kind name || [[ -n "$kind" ]]; do
    [[ -n "$kind" ]] || continue
    name="${name//$'\r'/}"
    case "$kind" in
      MOVE|KEY)
        if [[ "$seen_move" != *$'\n'"$name"$'\n'* ]]; then
          move+=("$name")
          seen_move+=$'\n'"$name"$'\n'
        fi
        ;;
      UNKNOWN)
        if [[ "$seen_unknown" != *$'\n'"$name"$'\n'* ]]; then
          unknown+=("$name")
          seen_unknown+=$'\n'"$name"$'\n'
        fi
        ;;
    esac
  done < <(config_scan "$file" || true)
  if [[ ${#move[@]} -gt 0 ]]; then
    echo "install.sh: refusing config.toml; move these to watch.toml:" >&2
    printf '%s\n' "${move[@]}" >&2
    echo "install.sh: write the migrated writer config to $(dest_path "$dest_root" /etc/llama-bored/config.toml.new)" >&2
  fi
  if [[ ${#unknown[@]} -gt 0 ]]; then
    echo "install.sh: refusing config.toml; unknown top-level key:" >&2
    printf '%s\n' "${unknown[@]}" >&2
  fi
}

# Sets CONFIG_ACTION (keep|swap|install) and CONFIG_SOURCE. Returns 1
# without writing when a legacy live config has no valid config.toml.new.
plan_config() {
  local staging=$1 dest_root=$2
  local mode live_bad=0 new_bad=0 have_new=0
  CONFIG_ACTION=keep
  CONFIG_SOURCE=""
  if ! mode="$(tr -d '[:space:]' <"$staging/config.mode")"; then
    echo "install.sh: staged config mode is unreadable" >&2
    return 1
  fi
  if [[ -f "$staging/config.toml.new" ]]; then
    have_new=1
    if ! config_is_clean "$staging/config.toml.new"; then
      new_bad=1
    fi
  fi
  if [[ "$mode" == "keep" ]]; then
    if ! config_is_clean "$staging/config.toml"; then
      live_bad=1
    fi
  fi
  if [[ "$live_bad" == 1 ]]; then
    if [[ "$have_new" != 1 || "$new_bad" == 1 ]]; then
      explain_problems "$staging/config.toml" "$dest_root"
      if [[ "$have_new" == 1 && "$new_bad" == 1 ]]; then
        explain_problems "$staging/config.toml.new" "$dest_root"
      fi
      return 1
    fi
    CONFIG_ACTION=swap
    CONFIG_SOURCE="$staging/config.toml.new"
    return 0
  fi
  if [[ "$have_new" == 1 ]]; then
    if [[ "$new_bad" == 1 ]]; then
      explain_problems "$staging/config.toml.new" "$dest_root"
      return 1
    fi
    CONFIG_ACTION=swap
    CONFIG_SOURCE="$staging/config.toml.new"
    return 0
  fi
  if [[ "$mode" == "install" ]]; then
    if ! config_is_clean "$staging/config.toml"; then
      explain_problems "$staging/config.toml" "$dest_root"
      return 1
    fi
    CONFIG_ACTION=install
    CONFIG_SOURCE="$staging/config.toml"
  fi
  return 0
}

previous_installed_sha() {
  local dest_root=$1 path prev
  path="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  prev=""
  if [[ -f "$path" && ! -L "$path" ]]; then
    prev="$(tr -d '[:space:]' <"$path")"
  fi
  if [[ ! "$prev" =~ ^[0-9a-f]{40}$ ]]; then
    prev="none"
  fi
  printf '%s\n' "$prev"
}

# Copy the live writer config aside. Never overwrites an existing backup.
# The name is config.toml.bak-<previous INSTALLED_SHA>, or bak-none.
# shellcheck disable=SC2329 # called as `must backup_live_config`
backup_live_config() {
  local staging=$1 dest_root=$2 prev=$3
  local mode bak
  if ! mode="$(tr -d '[:space:]' <"$staging/config.mode")"; then
    echo "install.sh: staged config mode is unreadable" >&2
    return 1
  fi
  if [[ "$mode" != "keep" ]]; then
    return 0
  fi
  if [[ ! "$prev" =~ ^([0-9a-f]{40}|none)$ ]]; then
    echo "install.sh: refusing to name a config backup '$prev'" >&2
    return 1
  fi
  bak="$(dest_path "$dest_root" "/etc/llama-bored/config.toml.bak-$prev")"
  if [[ -L "$bak" ]]; then
    echo "install.sh: refusing to follow a symlink at $bak" >&2
    return 1
  fi
  INSTALL_CONFIG_BAK_NEW=0
  INSTALL_CONFIG_BAK_PATH=""
  if [[ -e "$bak" ]]; then
    return 0
  fi
  place_file 0644 "$staging/config.toml" "$bak" || return 1
  INSTALL_CONFIG_BAK_NEW=1
  INSTALL_CONFIG_BAK_PATH=$bak
}

mark_write() {
  INSTALL_DONE+=("$1")
  if [[ "${INSTALL_ARMED:-}" == 1 ]]; then
    return 0
  fi
  INSTALL_ARMED=1
  INSTALL_HOOK_RAN=0
  # No ERR trap. Without set -E (errtrace) an ERR trap does not fire inside
  # functions, so it would never see a failed step. The EXIT hook runs on
  # the way out, including after `exit 1` from a failed step.
  trap install_on_exit EXIT
}

restore_exit_trap() {
  INSTALL_ARMED=0
  if [[ -n "${INSTALL_SAVED_EXIT:-}" ]]; then
    eval "$INSTALL_SAVED_EXIT"
  else
    trap - EXIT
  fi
}

# shellcheck disable=SC2329 # installed as the EXIT trap
install_on_exit() {
  local status=$? rollback_text marker_failed=0
  trap - EXIT
  if [[ "$status" -ne 0 && "${INSTALL_ARMED:-}" == 1 && "${INSTALL_HOOK_RAN:-}" != 1 ]]; then
    INSTALL_HOOK_RAN=1
    rollback_text="$(print_rollback)"
    if [[ ${#ROLLBACK_KIND[@]} -gt 0 ]]; then
      save_rollback_marker "$rollback_text" || marker_failed=1
    fi
    {
      echo "install.sh: failed after these steps:"
      if [[ ${#INSTALL_DONE[@]} -eq 0 ]]; then
        echo "  (none recorded)"
      else
        printf '  %s\n' "${INSTALL_DONE[@]}"
      fi
      printf '%s\n' "$rollback_text"
      if [[ "$marker_failed" == 1 ]]; then
        echo "install.sh: ROLLBACK_PENDING was not written; run the rollback above before any retry"
      fi
    } >&2
    if [[ ${#INSTALL_NEW_PATHS[@]} -gt 0 ]]; then
      rm -f -- "${INSTALL_NEW_PATHS[@]}"
    fi
  fi
  # The root install's staging directory is only known here. Chain the
  # cleanup that was installed before this hook, so a failure does not
  # leave that directory behind.
  if [[ -n "${INSTALL_STAGING_DIR:-}" && -d "${INSTALL_STAGING_DIR}" ]]; then
    rm -rf -- "$INSTALL_STAGING_DIR"
  fi
  if [[ -n "${INSTALL_SAVED_EXIT:-}" ]]; then
    eval "$INSTALL_SAVED_EXIT"
  fi
  exit "$status"
}

fail_install() {
  if [[ "${INSTALL_ARMED:-}" == 1 ]]; then
    exit 1
  fi
  return 1
}

# The writer's state directory, as the unit's StateDirectory=kraken-lcd
# with StateDirectoryMode=0755 would create it. systemd only creates it on
# the first service start; show-image needs it before that. An existing
# directory is left as it is.
# shellcheck disable=SC2329 # called as `must ensure_state_dir`
ensure_state_dir() {
  local dest_root=$1 dir parent
  dir="$(dest_path "$dest_root" /var/lib/kraken-lcd)"
  parent="$(dirname -- "$dir")"
  if [[ -L "$dir" || -L "$parent" ]]; then
    echo "install.sh: refusing to follow a symlink at $dir" >&2
    return 1
  fi
  if [[ -e "$dir" ]]; then
    if [[ ! -d "$dir" ]]; then
      echo "install.sh: $dir exists and is not a directory" >&2
      return 1
    fi
    return 0
  fi
  mkdir -p -- "$parent" || return 1
  mkdir -m 0755 -- "$dir" || return 1
  rollback_record created-dir "$dir" ""
  host_cmd chown kraken-lcd:kraken-lcd -- "$dir" || return 1
}

must() {
  if "$@"; then
    return 0
  fi
  fail_install
  return 1
}

# Write dest.new. The swap into dest happens in commit_new_file.
# shellcheck disable=SC2329 # called as `must stage_new_file`
stage_new_file() {
  local mode=$1 src=$2 dest=$3 label=$4
  local prev="${INSTALL_PREV_SHA:-none}"
  local new="$dest.new" bak="${dest}.bak-${prev}"
  if [[ -L "$dest" || -L "$new" || -L "$bak" ]]; then
    echo "install.sh: refusing to follow a symlink at $dest" >&2
    return 1
  fi
  if [[ ! -f "$src" || -L "$src" ]]; then
    echo "install.sh: $src is not a regular file" >&2
    return 1
  fi
  place_file "$mode" "$src" "$new" || return 1
  INSTALL_NEW_PATHS+=("$new")
  mark_write "$label"
}

# Swap dest.new into dest. An identical file is left in place and recorded
# as nothing. A different file is backed up to <dest>.bak-<previous sha>
# (never a bare .bak, and never over an existing backup). A missing file is
# recorded as created.
# shellcheck disable=SC2329 # called as `must commit_new_file`
commit_new_file() {
  local dest=$1
  local prev="${INSTALL_PREV_SHA:-none}"
  local new="$dest.new" bak="${dest}.bak-${prev}"
  local base
  base="$(basename -- "$dest")"
  if [[ "${INSTALL_FAIL_AT:-}" == "commit:${base}" ]]; then
    echo "install.sh: injected failure at commit:${base}" >&2
    return 1
  fi
  if [[ ! -f "$new" || -L "$new" ]]; then
    echo "install.sh: missing staged file $new" >&2
    return 1
  fi
  if [[ -L "$dest" || -L "$bak" ]]; then
    echo "install.sh: refusing to follow a symlink at $dest" >&2
    return 1
  fi
  if [[ -e "$dest" ]] && cmp -s -- "$new" "$dest"; then
    rm -f -- "$new"
    return 0
  fi
  if [[ -e "$dest" ]]; then
    if [[ ! -e "$bak" ]]; then
      cp -a -- "$dest" "$bak" || return 1
    fi
    mv -f -- "$new" "$dest" || return 1
    rollback_record replaced "$dest" "$bak"
    rollback_note_rules "$dest"
    return 0
  fi
  mv -f -- "$new" "$dest" || return 1
  rollback_record created "$dest" ""
  rollback_note_rules "$dest"
  rollback_note_watch "$dest"
  rollback_note_metrics "$dest"
}

# shellcheck disable=SC2329 # called as `must commit_installed_sha`
commit_installed_sha() {
  local dest=$1 head=$2
  local prev="${INSTALL_PREV_SHA:-none}"
  local bak="${dest}.bak-${prev}"
  local tmp
  if [[ -L "$dest" || -L "$bak" ]]; then
    echo "install.sh: refusing to follow a symlink at $dest" >&2
    return 1
  fi
  tmp="$(mktemp)"
  if ! printf '%s\n' "$head" >"$tmp"; then
    rm -f -- "$tmp"
    return 1
  fi
  if [[ -e "$dest" ]] && cmp -s -- "$tmp" "$dest"; then
    rm -f -- "$tmp"
    return 0
  fi
  if [[ -e "$dest" ]]; then
    if [[ ! -e "$bak" ]]; then
      cp -a -- "$dest" "$bak" || {
        rm -f -- "$tmp"
        return 1
      }
    fi
    place_file 0644 "$tmp" "$dest" || {
      rm -f -- "$tmp"
      return 1
    }
    rm -f -- "$tmp"
    rollback_record replaced "$dest" "$bak"
    return 0
  fi
  place_file 0644 "$tmp" "$dest" || {
    rm -f -- "$tmp"
    return 1
  }
  rm -f -- "$tmp"
  rollback_record created "$dest" ""
}

# Install only from `staging`. `head` is the full provenance HEAD, written last.
# Binaries, units and config land only after the hidraw check. kraken-lcd is
# never started, restarted or enabled.
apply_from_staging() {
  local staging=$1 dest_root=$2 sys_root=$3 head=$4
  local config_dest min_interval unit watch_unit sysusers rule71 rule93
  local binary_dest watch_binary view_binary view_rule watch_mode watch_dest user_dir font_dest
  local node hidraw_name mode_group pin sha_dest prev interval_src config_bak
  local light_binary light_unit light_rule light_mode light_dest aura_node aura_pin aura_mode
  local kbd_node kbd_pin kbd_mode
  local metrics_binary metrics_unit metrics_mode metrics_dest
  local -a scan=() trial=()
  ROLLBACK_KIND=()
  ROLLBACK_PATH=()
  ROLLBACK_BAK=()
  INSTALL_DONE=()
  INSTALL_NEW_PATHS=()
  INSTALL_ARMED=0
  INSTALL_HOOK_RAN=0
  INSTALL_PREV_SHA=none
  INSTALL_CREATED_WATCH=0
  INSTALL_ENABLED_LIGHT=0
  INSTALL_WATCH_WAS_ABSENT=0
  INSTALL_CREATED_METRICS=0
  INSTALL_METRICS_WAS_ABSENT=0
  INSTALL_RULES_CHANGED=0
  INSTALL_CONFIG_BAK_NEW=0
  INSTALL_CONFIG_BAK_PATH=""
  INSTALL_DEST_ROOT="${dest_root%/}"
  INSTALL_SAVED_EXIT="$(trap -p EXIT || true)"
  refuse_rollback_pending || return 1
  if [[ ! "$head" =~ ^[0-9a-f]{40}$ ]]; then
    echo "install.sh: refusing to record a HEAD that is not 40 hex digits" >&2
    return 1
  fi
  if [[ -z "$dest_root" ]]; then
    scan=(/etc/udev/rules.d /usr/lib/udev/rules.d /run/udev/rules.d)
  else
    scan=("$(dest_path "$dest_root" /etc/udev/rules.d)")
  fi
  mapfile -t trial < <(trial_rule_files "${scan[@]}")
  if [[ ${#trial[@]} -ne 0 ]]; then
    echo "install.sh: a udev rule still gives the Kraken (1e71) to a user with OWNER=:" >&2
    printf '%s\n' "${trial[@]}" >&2
    return 1
  fi

  plan_config "$staging" "$dest_root" || return 1
  if [[ -n "$CONFIG_SOURCE" ]]; then
    interval_src="$CONFIG_SOURCE"
  else
    interval_src="$staging/config.toml"
  fi
  min_interval="$(min_interval_of "$interval_src")" || return 1
  prev="$(previous_installed_sha "$dest_root")"
  INSTALL_PREV_SHA="$prev"

  if ! watch_mode="$(tr -d '[:space:]' <"$staging/watch.mode")"; then
    echo "install.sh: staged watch mode is unreadable" >&2
    return 1
  fi
  if ! light_mode="$(tr -d '[:space:]' <"$staging/light.mode")"; then
    echo "install.sh: staged light mode is unreadable" >&2
    return 1
  fi
  if ! metrics_mode="$(tr -d '[:space:]' <"$staging/metrics.mode")"; then
    echo "install.sh: staged metrics mode is unreadable" >&2
    return 1
  fi

  binary_dest="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/kraken-lcd)"
  watch_binary="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/llama-watch)"
  view_binary="$(dest_path "$dest_root" /usr/local/bin/llama-view)"
  view_rule="$(dest_path "$dest_root" /etc/udev/rules.d/72-llama-view.rules)"
  # tty11 console font, loaded by the watcher unit's ExecStartPre.
  font_dest="$(dest_path "$dest_root" /usr/local/share/llama-bored/llama-hack-12x24.psfu)"
  watch_unit="$(dest_path "$dest_root" /etc/systemd/system/llama-watch.service)"
  light_binary="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/llama-light)"
  light_unit="$(dest_path "$dest_root" /etc/systemd/system/llama-light.service)"
  light_rule="$(dest_path "$dest_root" /etc/udev/rules.d/94-llama-light-hidraw.rules)"
  light_dest="$(dest_path "$dest_root" /etc/llama-bored/light.toml)"
  if [[ ! -e "$watch_binary" && ! -e "$watch_unit" ]]; then
    INSTALL_WATCH_WAS_ABSENT=1
  fi
  metrics_binary="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/llama-metrics)"
  metrics_unit="$(dest_path "$dest_root" /etc/systemd/system/llama-metrics.service)"
  metrics_dest="$(dest_path "$dest_root" /etc/llama-bored/metrics.toml)"
  if [[ ! -e "$metrics_binary" && ! -e "$metrics_unit" ]]; then
    INSTALL_METRICS_WAS_ABSENT=1
  fi

  # Writes start here. The hook arms on the first one.
  must backup_live_config "$staging" "$dest_root" "$prev" || return 1
  if [[ "${INSTALL_CONFIG_BAK_NEW:-0}" == 1 ]]; then
    rollback_record created "$INSTALL_CONFIG_BAK_PATH" ""
  fi
  must stage_new_file 0755 "$staging/kraken-lcd" "$binary_dest" "staged kraken-lcd.new" || return 1
  must stage_new_file 0755 "$staging/llama-watch" "$watch_binary" "staged llama-watch.new" || return 1
  must stage_new_file 0755 "$staging/llama-view" "$view_binary" "staged llama-view.new" || return 1
  must stage_new_file 0755 "$staging/llama-light" "$light_binary" "staged llama-light.new" || return 1
  must stage_new_file 0644 "$staging/llama-light.service" "$light_unit" "staged llama-light.service.new" || return 1
  if [[ "$light_mode" == "install" ]]; then
    must stage_new_file 0644 "$staging/light.toml" "$light_dest" "staged light.toml.new" || return 1
  fi
  must stage_new_file 0755 "$staging/llama-metrics" "$metrics_binary" "staged llama-metrics.new" || return 1
  must stage_new_file 0644 "$staging/llama-metrics.service" "$metrics_unit" \
    "staged llama-metrics.service.new" || return 1
  if [[ "$metrics_mode" == "install" ]]; then
    must stage_new_file 0644 "$staging/metrics.toml" "$metrics_dest" "staged metrics.toml.new" || return 1
  fi
  must stage_new_file 0644 "$staging/llama-hack-12x24.psfu" "$font_dest" \
    "staged llama-hack-12x24.psfu.new" || return 1

  unit="$(dest_path "$dest_root" /etc/systemd/system/kraken-lcd.service)"
  must stage_new_file 0644 "$staging/kraken-lcd.service" "$unit" "staged kraken-lcd.service.new" || return 1
  must rewrite_restart_sec "$unit.new" "$min_interval" || return 1
  must stage_new_file 0644 "$staging/llama-watch.service" "$watch_unit" "staged llama-watch.service.new" || return 1

  watch_dest="$(dest_path "$dest_root" /etc/llama-bored/watch.toml)"
  if [[ "$watch_mode" == "install" ]]; then
    must stage_new_file 0644 "$staging/watch.toml" "$watch_dest" "staged watch.toml.new" || return 1
  fi

  user_dir="$(dest_path "$dest_root" /etc/systemd/user)"
  must stage_new_file 0644 "$staging/kraken-lcd-halt.path" \
    "$user_dir/kraken-lcd-halt.path" "staged halt.path.new" || return 1
  must stage_new_file 0644 "$staging/kraken-lcd-halt-notify.service" \
    "$user_dir/kraken-lcd-halt-notify.service" "staged halt-notify.service.new" || return 1

  # udev and sysusers have to be in place before the hidraw check.
  sysusers="$(dest_path "$dest_root" /etc/sysusers.d/llama-bored.conf)"
  rule71="$(dest_path "$dest_root" /etc/udev/rules.d/71-kraken-lcd.rules)"
  rule93="$(dest_path "$dest_root" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)"
  must stage_new_file 0644 "$staging/llama-bored.sysusers" "$sysusers" "staged sysusers.new" || return 1
  must commit_new_file "$sysusers" || return 1
  must stage_new_file 0644 "$staging/71-kraken-lcd.rules" "$rule71" "staged 71-rules.new" || return 1
  must commit_new_file "$rule71" || return 1
  must stage_new_file 0644 "$staging/93-kraken-lcd-hidraw.rules" "$rule93" "staged 93-rules.new" || return 1
  must commit_new_file "$rule93" || return 1
  must stage_new_file 0644 "$staging/72-llama-view.rules" "$view_rule" "staged 72-rules.new" || return 1
  must commit_new_file "$view_rule" || return 1
  must stage_new_file 0644 "$staging/94-llama-light-hidraw.rules" "$light_rule" "staged 94-rules.new" || return 1
  must commit_new_file "$light_rule" || return 1

  must host_cmd systemd-sysusers || return 1
  mark_write "systemd-sysusers"
  must host_cmd udevadm control --reload || return 1
  mark_write "udev reload"
  must host_cmd udevadm trigger --action=change \
    --attr-match=idVendor=1e71 --attr-match=idProduct=3008 || return 1
  must host_cmd udevadm trigger --action=change --sysname-match=vcsa11 || return 1
  must host_cmd udevadm trigger --action=change --sysname-match=vcs11 || return 1
  must host_cmd udevadm trigger --action=change --sysname-match=vcsu11 || return 1
  # The Aura controller may be absent; a trigger that matches nothing is fine.
  must host_cmd udevadm trigger --action=change \
    --attr-match=idVendor=0b05 --attr-match=idProduct=18f3 || return 1
  # The keyboard may be absent too.
  must host_cmd udevadm trigger --action=change \
    --attr-match=idVendor=1b1c --attr-match=idProduct=1b48 || return 1
  # A USB parent's change event does not re-run rules on its hidraw child, so
  # re-apply the hidraw rules directly (idempotent; seen on a test machine with the
  # Aura node staying 0666 after install).
  must host_cmd udevadm trigger --action=change --subsystem-match=hidraw || return 1
  node="$(resolve_kraken_hidraw "$sys_root")" || fail_install || return 1
  hidraw_name="${node#/dev/}"
  must host_cmd udevadm trigger --action=change "$sys_root/class/hidraw/$hidraw_name" || return 1
  must host_cmd udevadm settle || return 1
  mark_write "udev settle"
  must host_cmd setfacl -b -- "$node" || return 1
  if ! mode_group="$(hidraw_mode_group "$node")"; then
    echo "install.sh: stat of $node failed" >&2
    fail_install || return 1
  fi
  if [[ "$mode_group" != "660 kraken-lcd" ]]; then
    echo "install.sh: $node is '$mode_group', expected '660 kraken-lcd'" >&2
    fail_install || return 1
  fi
  if hidraw_has_user_acl "$node"; then
    echo "install.sh: $node still has a named user ACL after setfacl -b" >&2
    fail_install || return 1
  fi
  # The writer unit's only hidraw DeviceAllow= is this symlink (SAFETY.md RR7).
  pin="$(hidraw_pin_target)"
  if [[ "$pin" != "$node" ]]; then
    echo "install.sh: /dev/kraken-lcd/hid resolves to '${pin:-nothing}', expected $node" >&2
    fail_install || return 1
  fi
  mark_write "hidraw check passed"

  # The Aura controller's node is llama-light's alone (0660, no
  # uaccess ACL), and its pin names it. Not attached: nothing to check;
  # llama-light logs it absent once and looks again every 10 s.
  aura_node="$(resolve_aura_hidraw "$sys_root")" || fail_install || return 1
  if [[ -n "$aura_node" ]]; then
    must host_cmd setfacl -b -- "$aura_node" || return 1
    if ! aura_mode="$(aura_mode_group "$aura_node")"; then
      echo "install.sh: stat of $aura_node failed" >&2
      fail_install || return 1
    fi
    if [[ "$aura_mode" != "660 llama-light" ]]; then
      echo "install.sh: Aura node $aura_node is '$aura_mode', expected '660 llama-light'" >&2
      fail_install || return 1
    fi
    if hidraw_has_user_acl "$aura_node"; then
      echo "install.sh: $aura_node still has a named user ACL after setfacl -b" >&2
      fail_install || return 1
    fi
    aura_pin="$(aura_pin_target)"
    if [[ "$aura_pin" != "$aura_node" ]]; then
      echo "install.sh: /dev/llama-light/aura resolves to '${aura_pin:-nothing}', expected $aura_node" >&2
      fail_install || return 1
    fi
    mark_write "aura hidraw check passed"
  else
    echo "install.sh: no Aura controller (0b05:18f3) attached; llama-light will report it absent" >&2
  fi

  # The keyboard's lighting interface is llama-light's alone (0660, no
  # uaccess ACL; typing uses another interface and evdev), and its pin names
  # it. Not attached: nothing to check; llama-light logs it absent once.
  kbd_node="$(resolve_keyboard_hidraw "$sys_root")" || fail_install || return 1
  if [[ -n "$kbd_node" ]]; then
    must host_cmd setfacl -b -- "$kbd_node" || return 1
    if ! kbd_mode="$(keyboard_mode_group "$kbd_node")"; then
      echo "install.sh: stat of $kbd_node failed" >&2
      fail_install || return 1
    fi
    if [[ "$kbd_mode" != "660 llama-light" ]]; then
      echo "install.sh: keyboard lighting node $kbd_node is '$kbd_mode', expected '660 llama-light'" >&2
      fail_install || return 1
    fi
    if hidraw_has_user_acl "$kbd_node"; then
      echo "install.sh: $kbd_node still has a named user ACL after setfacl -b" >&2
      fail_install || return 1
    fi
    kbd_pin="$(keyboard_pin_target)"
    if [[ "$kbd_pin" != "$kbd_node" ]]; then
      echo "install.sh: /dev/llama-light/keyboard resolves to '${kbd_pin:-nothing}', expected $kbd_node" >&2
      fail_install || return 1
    fi
    mark_write "keyboard hidraw check passed"
  else
    echo "install.sh: no Corsair keyboard (1b1c:1b48) attached; llama-light will report it absent" >&2
  fi

  # Swap binaries, units and config only after the device check.
  must commit_new_file "$binary_dest" || return 1
  must commit_new_file "$watch_binary" || return 1
  must commit_new_file "$view_binary" || return 1
  must commit_new_file "$light_binary" || return 1
  must commit_new_file "$light_unit" || return 1
  if [[ "$light_mode" == "install" ]]; then
    must commit_new_file "$light_dest" || return 1
  fi
  must commit_new_file "$metrics_binary" || return 1
  must commit_new_file "$metrics_unit" || return 1
  if [[ "$metrics_mode" == "install" ]]; then
    must commit_new_file "$metrics_dest" || return 1
  fi
  must commit_new_file "$font_dest" || return 1
  must commit_new_file "$unit" || return 1
  must commit_new_file "$watch_unit" || return 1
  if [[ "$watch_mode" == "install" ]]; then
    must commit_new_file "$watch_dest" || return 1
  fi
  must commit_new_file "$user_dir/kraken-lcd-halt.path" || return 1
  must commit_new_file "$user_dir/kraken-lcd-halt-notify.service" || return 1

  config_dest="$(dest_path "$dest_root" /etc/llama-bored/config.toml)"
  config_bak="$(dest_path "$dest_root" "/etc/llama-bored/config.toml.bak-$prev")"
  if [[ "$CONFIG_ACTION" == "swap" || "$CONFIG_ACTION" == "install" ]]; then
    if [[ "$CONFIG_ACTION" == "swap" && -e "$config_dest" ]]; then
      must place_file 0644 "$CONFIG_SOURCE" "$config_dest" || return 1
      rollback_record replaced "$config_dest" "$config_bak"
      rm -f -- "$(dest_path "$dest_root" /etc/llama-bored/config.toml.new)"
    else
      must place_file 0644 "$CONFIG_SOURCE" "$config_dest" || return 1
      if [[ "$CONFIG_ACTION" == "swap" ]]; then
        rm -f -- "$(dest_path "$dest_root" /etc/llama-bored/config.toml.new)"
      fi
      rollback_record created "$config_dest" ""
    fi
  fi

  must ensure_state_dir "$dest_root" || return 1

  if [[ "${INSTALL_FAIL_AT:-}" == "daemon-reload" ]]; then
    echo "install.sh: injected failure at daemon-reload" >&2
    fail_install || return 1
  fi
  must host_cmd systemctl daemon-reload || return 1
  mark_write "daemon-reload"
  # Enable the watcher for boot. Do not start it. Never start or enable the writer.
  must host_cmd systemctl enable llama-watch.service || return 1
  mark_write "enabled llama-watch"
  # llama-light is enabled only on the operator's --enable-light. Never started.
  if [[ "${LIGHT_ENABLE:-0}" == 1 ]]; then
    must host_cmd systemctl enable llama-light.service || return 1
    INSTALL_ENABLED_LIGHT=1
    mark_write "enabled llama-light"
  fi
  # llama-metrics is never enabled or started here, and the firewall is
  # never touched. print_metrics_steps prints the operator's steps.

  sha_dest="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  must commit_installed_sha "$sha_dest" "$head" || return 1
  restore_exit_trap
  print_phase_b
}

# Run a review command as the repo owner. Root uses runuser so git stays off
# root's PATH. The self-test is the owner and runs the command directly.
owner_exec() {
  if [[ "${INSTALL_OWNER_MODE:-direct}" == "runuser" ]]; then
    runuser -u "${SUDO_USER:?}" -- env \
      "PATH=${INSTALL_OWNER_PATH:?}" \
      "HOME=${INSTALL_OWNER_HOME:?}" \
      "USER=${SUDO_USER}" \
      "LOGNAME=${SUDO_USER}" \
      "$@"
  else
    "$@"
  fi
}

# Self-test only. install_real unsets INSTALL_TAMPER_LIVE. After the review
# has accepted the tree, append one line so the staged-bytes compare is the
# check that refuses, before any destination write.
apply_review_tamper() {
  local repo=$1 rel="${INSTALL_TAMPER_LIVE:-}"
  [[ -n "$rel" ]] || return 0
  case "$rel" in
    /*|*..*)
      echo "install.sh: tamper path is not a repo-relative file" >&2
      return 1
      ;;
  esac
  if [[ ! -f "$repo/$rel" || -L "$repo/$rel" ]]; then
    echo "install.sh: tamper path is not a regular file" >&2
    return 1
  fi
  printf '\n# tampered-after-review\n' >>"$repo/$rel"
}

# Review, then install from the frozen copies. No prompt. INSTALL_DID_STAGE
# is 1 only after freeze_staging has created the staging directory.
# --require-landed resolves refs/heads/main^{commit} (RELEASE_BRANCH in stage.sh).
anchored_install() {
  local repo=$1 dest_root=$2 sys_root=$3
  local installed staging head status
  INSTALL_DEST_ROOT="${dest_root%/}"
  INSTALL_DID_STAGE=0
  INSTALL_STAGING_DIR=""
  refuse_rollback_pending || return 1
  # Before any staging write. The full review repeats this after the copy.
  if ! owner_exec /usr/bin/bash "$ROOT/scripts/stage.sh" --require-landed "$repo"; then
    return 1
  fi
  installed="$(dest_path "$dest_root" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  staging="$(freeze_staging "$repo" "$dest_root")" || return 1
  INSTALL_DID_STAGE=1
  INSTALL_STAGING_DIR="$staging"
  status=0
  if ! show_staging_hashes "$staging"; then
    status=1
  elif ! verify_staged_binary "$staging"; then
    status=1
  elif ! owner_exec /usr/bin/bash "$ROOT/scripts/stage.sh" "$repo" "$installed"; then
    echo "install.sh: unprivileged review failed" >&2
    status=1
  elif ! apply_review_tamper "$repo"; then
    status=1
  elif ! verify_staged_matches_disk "$staging" "$repo" "$dest_root"; then
    status=1
  elif ! show_staging_hashes "$staging"; then
    status=1
  elif ! head="$(provenance_head "$staging/check-provenance.txt")"; then
    status=1
  elif ! plan_config "$staging" "$dest_root"; then
    status=1
  elif ! apply_from_staging "$staging" "$dest_root" "$sys_root" "$head"; then
    status=1
  fi
  rm -rf -- "$staging"
  INSTALL_STAGING_DIR=""
  return "$status"
}

install_real() {
  local user_path="$PATH"
  local sudo_user="${SUDO_USER:-}"
  local user_home
  export PATH="$ROOT_PATH"
  hash -r
  unset INSTALL_DRY INSTALL_ALLOW_UNPRIV_COPY INSTALL_FAKE_STAT INSTALL_FAKE_GETFACL INSTALL_FAKE_PIN INSTALL_LOG INSTALL_FAIL_AT INSTALL_FAIL_MARKER
  unset INSTALL_FAKE_AURA_PIN INSTALL_FAKE_AURA_STAT INSTALL_FAKE_KBD_PIN INSTALL_FAKE_KBD_STAT
  unset INSTALL_OWNER_MODE INSTALL_OWNER_PATH INSTALL_OWNER_HOME INSTALL_TAMPER_LIVE
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "install.sh: must be run as root (with sudo)" >&2
    exit 1
  fi
  if [[ -z "$sudo_user" || "$sudo_user" == "root" ]]; then
    echo "install.sh: run via sudo from the installing user, not a root shell" >&2
    exit 1
  fi
  if [[ ! -x /usr/sbin/runuser && ! -x /usr/bin/runuser ]]; then
    echo "install.sh: runuser is required" >&2
    exit 1
  fi
  if ! user_home="$(getent passwd "$sudo_user" | cut -d: -f6)"; then
    echo "install.sh: no passwd entry for $sudo_user" >&2
    exit 1
  fi
  if [[ -z "$user_home" || ! -d "$user_home" ]]; then
    echo "install.sh: no home directory for $sudo_user" >&2
    exit 1
  fi
  INSTALL_OWNER_MODE=runuser
  INSTALL_OWNER_PATH="$user_path"
  INSTALL_OWNER_HOME="$user_home"
  anchored_install "$ROOT" "" /sys || exit 1
}

seed_kraken_sysfs() {
  local sys=$1
  local dev="$sys/bus/usb/devices/3-5"
  local hid="$dev/3-5:1.1/0003:1E71:3008.0001"
  mkdir -p "$hid/hidraw/hidraw0" "$sys/bus/hid/drivers/nzxt_kraken3"
  printf '1e71\n' >"$dev/idVendor"
  printf '3008\n' >"$dev/idProduct"
  printf 'HID_ID=0003:00001E71:00003008\nHID_NAME=NZXT Kraken Z\n' >"$hid/uevent"
  ln -s ../../../../bus/hid/drivers/nzxt_kraken3 "$hid/driver"
}

write_real_provenance() {
  local repo=$1 rustc_v cargo_v
  mkdir -p -- "$repo/target"
  # rustc/cargo stay outside the fixture. Its rust-toolchain.toml is not a
  # real toolchain file, and rustup would refuse to start if cwd were there.
  rustc_v="$(rustc -V)" || return 1
  cargo_v="$(cargo -V)" || return 1
  {
    printf '%s\n' "$rustc_v"
    printf '%s\n' "$cargo_v"
    (cd "$repo" && sha256sum Cargo.lock rust-toolchain.toml) || return 1
    git -C "$repo" rev-parse HEAD || return 1
    git -C "$repo" describe --always --dirty || return 1
    (cd "$repo" && sha256sum target/release/kraken-lcd target/release/llama-watch \
      target/release/llama-view target/release/llama-light \
      target/release/llama-metrics) || return 1
  } >"$repo/target/check-provenance.txt"
}

write_fixture_provenance() {
  local repo=$1 head=$2 hash watch_hash view_hash light_hash metrics_hash
  hash="$(sha256sum -- "$repo/target/release/kraken-lcd" | awk '{ print $1 }')"
  watch_hash="$(sha256sum -- "$repo/target/release/llama-watch" | awk '{ print $1 }')"
  view_hash="$(sha256sum -- "$repo/target/release/llama-view" | awk '{ print $1 }')"
  light_hash="$(sha256sum -- "$repo/target/release/llama-light" | awk '{ print $1 }')"
  metrics_hash="$(sha256sum -- "$repo/target/release/llama-metrics" | awk '{ print $1 }')"
  mkdir -p -- "$repo/target"
  {
    echo "rustc-fixture"
    echo "cargo-fixture"
    echo "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  Cargo.lock"
    echo "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  rust-toolchain.toml"
    echo "$head"
    echo "fixture"
    echo "$hash  target/release/kraken-lcd"
    echo "$watch_hash  target/release/llama-watch"
    echo "$view_hash  target/release/llama-view"
    echo "$light_hash  target/release/llama-light"
    echo "$metrics_hash  target/release/llama-metrics"
  } >"$repo/target/check-provenance.txt"
}

assert_eq() {
  local got=$1 want=$2 label=$3
  if [[ "$got" != "$want" ]]; then
    echo "install self-test: $label" >&2
    echo "got:  $got" >&2
    echo "want: $want" >&2
    exit 1
  fi
}

run_expect_ok() {
  set +e
  (
    set -euo pipefail
    "$@"
  )
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "install self-test: expected success ($status): $*" >&2
    exit 1
  fi
}

run_expect_fail() {
  set +e
  (
    set -euo pipefail
    "$@"
  )
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "install self-test: expected failure: $*" >&2
    exit 1
  fi
}

log_has() {
  local pattern=$1
  grep -F -q -- "$pattern" "$INSTALL_LOG"
}

# A dry-run line that names kraken-lcd and also starts, restarts, or enables it.
log_mentions_writer_lifecycle() {
  local line
  [[ -f "${INSTALL_LOG:-}" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *kraken-lcd* ]] || continue
    if [[ "$line" =~ (^|[^[:alnum:]_])(reload-or-restart|try-restart|restart|start|enable)([^[:alnum:]_]|$) ]]; then
      printf '%s\n' "$line" >&2
      return 0
    fi
  done <"$INSTALL_LOG"
  return 1
}

log_line_number() {
  local pattern=$1
  grep -n -F -- "$pattern" "$INSTALL_LOG" | head -n 1 | cut -d: -f1
}

file_manifest() {
  local root=$1
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -type f -printf '%P\n' | LC_ALL=C sort
}

assert_same_tree() {
  local got=$1 want=$2 label=$3
  local got_list want_list rel
  got_list="$(file_manifest "$got")"
  want_list="$(file_manifest "$want")"
  if [[ "$got_list" != "$want_list" ]]; then
    echo "install self-test: $label file list differs" >&2
    diff -u <(printf '%s\n' "$want_list") <(printf '%s\n' "$got_list") >&2 || true
    exit 1
  fi
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    if ! cmp -s -- "$got/$rel" "$want/$rel"; then
      echo "install self-test: $label content differs: $rel" >&2
      exit 1
    fi
  done <<<"$got_list"
}

snapshot_tree() {
  local src=$1 dest=$2
  rm -rf -- "$dest"
  mkdir -p -- "$dest"
  if [[ -d "$src" ]]; then
    cp -a -- "$src/." "$dest/"
  fi
}

assert_no_new() {
  local root=$1 hits
  hits="$(find "$root" -name '*.new' ! -name 'config.toml.new' -print 2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    echo "install self-test: .new files left under $root" >&2
    printf '%s\n' "$hits" >&2
    exit 1
  fi
}

rollback_section() {
  local text=$1 line on=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == *"how to roll back"* ]]; then
      on=1
    fi
    if [[ "$on" == 1 ]]; then
      printf '%s\n' "$line"
    fi
  done <<<"$text"
}

assert_disable_present() {
  local text=$1 want=$2 section
  section="$(rollback_section "$text")"
  if [[ "$want" == 1 ]]; then
    [[ "$section" == *"systemctl disable --now llama-watch.service"* ]] || {
      echo "install self-test: rollback omitted disable --now llama-watch" >&2
      printf '%s\n' "$section" >&2
      exit 1
    }
  elif [[ "$section" == *"disable --now llama-watch"* ]]; then
    echo "install self-test: rollback disabled llama-watch for a run that did not create it" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
}

assert_rollback_order() {
  local text=$1
  local section line cmd n=0 disable_n=0 remove_n=0 reload_n=0
  local udev_n=0 trigger_n=0 marker_n=0 file_after=0
  section="$(rollback_section "$text")"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == "  "* ]] || continue
    cmd="${line#"  "}"
    [[ -z "$cmd" || "$cmd" == "#"* ]] && continue
    n=$((n + 1))
    if [[ "$cmd" == "systemctl disable --now llama-watch.service" ]]; then
      disable_n=$n
    fi
    if [[ "$cmd" == *"llama-watch.service"* && "$cmd" != *disable* ]]; then
      remove_n=$n
    fi
    if [[ "$cmd" == "systemctl daemon-reload" ]]; then
      reload_n=$n
    fi
    if [[ "$cmd" == "udevadm control --reload" ]]; then
      udev_n=$n
    fi
    if [[ "$cmd" == udevadm\ trigger\ * ]]; then
      trigger_n=$n
    fi
    if [[ "$cmd" == "rm -f -- /usr/local/libexec/llama-bored/ROLLBACK_PENDING" ]]; then
      marker_n=$n
    fi
    if [[ "$reload_n" -gt 0 && "$n" -gt "$reload_n" && "$cmd" != udevadm* \
      && "$cmd" != "rm -f -- /usr/local/libexec/llama-bored/ROLLBACK_PENDING" ]]; then
      file_after=1
    fi
  done <<<"$section"
  if [[ "$disable_n" -gt 0 && "$remove_n" -gt 0 && "$disable_n" -ge "$remove_n" ]]; then
    echo "install self-test: rollback disables llama-watch after removing its unit" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
  if [[ "$reload_n" -eq 0 || "$file_after" == 1 ]]; then
    echo "install self-test: rollback reload is not after the file restores" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
  if [[ "$marker_n" -eq 0 || "$marker_n" -ne "$n" ]]; then
    echo "install self-test: rollback does not remove ROLLBACK_PENDING last" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
  if [[ "$udev_n" -gt 0 || "$trigger_n" -gt 0 ]]; then
    if [[ "$udev_n" -eq 0 || "$trigger_n" -eq 0 || "$reload_n" -ge "$udev_n" \
      || "$udev_n" -ge "$trigger_n" || "$trigger_n" -ge "$marker_n" ]]; then
      echo "install self-test: udev reload is not after daemon-reload" >&2
      printf '%s\n' "$section" >&2
      exit 1
    fi
  fi
  if printf '%s\n' "$section" | grep -E -q '\.bak([^[:alnum:]-]|$)'; then
    echo "install self-test: rollback names a bare .bak" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
}

assert_udev_reload() {
  local text=$1 section
  section="$(rollback_section "$text")"
  [[ "$section" == *"udevadm control --reload"* \
    && "$section" == *"udevadm trigger --action=change --attr-match=idVendor=1e71 --attr-match=idProduct=3008"* ]] || {
    echo "install self-test: rollback omitted the udev reload" >&2
    printf '%s\n' "$section" >&2
    exit 1
  }
  assert_rollback_order "$text"
}

assert_disable_first() {
  local text=$1 section line cmd first=""
  section="$(rollback_section "$text")"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == "  "* ]] || continue
    cmd="${line#"  "}"
    [[ -z "$cmd" || "$cmd" == "#"* ]] && continue
    first="$cmd"
    break
  done <<<"$section"
  [[ "$first" == "systemctl disable --now llama-watch.service" ]] || {
    echo "install self-test: rollback does not disable llama-watch first" >&2
    printf '%s\n' "$section" >&2
    exit 1
  }
}

# The migrated config is copied aside before the live file is restored, and
# that restore happens before the backup this run created is removed.
assert_config_restore_order() {
  local text=$1 section cp_line mv_line rm_line
  section="$(rollback_section "$text")"
  cp_line="$(printf '%s\n' "$section" | grep -n -m 1 -F \
    'cp -p -- /etc/llama-bored/config.toml /etc/llama-bored/config.toml.new' || true)"
  mv_line="$(printf '%s\n' "$section" | grep -n -m 1 -E \
    'mv -f -- /etc/llama-bored/config\.toml\.bak-[[:alnum:]]+ /etc/llama-bored/config\.toml$' || true)"
  rm_line="$(printf '%s\n' "$section" | grep -n -m 1 -E \
    'rm -f -- /etc/llama-bored/config\.toml\.bak-[[:alnum:]]+$' || true)"
  cp_line="${cp_line%%:*}"
  mv_line="${mv_line%%:*}"
  rm_line="${rm_line%%:*}"
  if [[ -z "$cp_line" || -z "$mv_line" || -z "$rm_line" \
    || "$cp_line" -ge "$mv_line" || "$mv_line" -ge "$rm_line" ]]; then
    echo "install self-test: config restore is not cp, then mv, then rm of the backup" >&2
    printf '%s\n' "$section" >&2
    exit 1
  fi
}

execute_printed_rollback() {
  local text=$1 dest=$2
  local section line cmd rewritten
  section="$(rollback_section "$text")"
  : >"$INSTALL_LOG"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == "  "* ]] || continue
    cmd="${line#"  "}"
    [[ -z "$cmd" || "$cmd" == "#"* ]] && continue
    if [[ "$cmd" == systemctl* || "$cmd" == udevadm* ]]; then
      printf '%s\n' "$cmd" >>"$INSTALL_LOG"
      continue
    fi
    rewritten="${cmd//\/var\/lib\//${dest}/var/lib/}"
    rewritten="${rewritten//\/etc\//${dest}/etc/}"
    rewritten="${rewritten//\/usr\//${dest}/usr/}"
    if [[ "$rewritten" == systemctl* || "$rewritten" == udevadm* ]]; then
      echo "install self-test: host command was rewritten into a file command" >&2
      exit 1
    fi
    bash -c "$rewritten" || {
      echo "install self-test: rollback command failed: $rewritten" >&2
      exit 1
    }
  done <<<"$section"
}

# Only the self-test process that set the trap cleans up. $$ does not
# change in a subshell; BASHPID does, so a command substitution that
# inherits the trap cannot delete the fixture tree.
# shellcheck disable=SC2329 # installed as the EXIT trap
self_test_cleanup() {
  [[ "${SELF_TEST_PID:-}" == "$BASHPID" ]] || return 0
  [[ -n "${SELF_TEST_TMP:-}" ]] && rm -rf -- "$SELF_TEST_TMP"
  [[ -n "${SELF_TEST_NONTEMP:-}" ]] && rm -f -- "$SELF_TEST_NONTEMP"
  [[ -n "${SELF_TEST_REPO_PROBE:-}" ]] && rm -f -- "$SELF_TEST_REPO_PROBE"
  return 0
}

self_test() {
  local tmp repo dest sys head staging mode out body
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "install.sh: --self-test refuses to run as root" >&2
    exit 1
  fi
  bash "$ROOT/scripts/stage.sh" --self-test

  tmp="$(mktemp -d)"
  SELF_TEST_TMP="$tmp"
  SELF_TEST_PID=$BASHPID
  trap self_test_cleanup EXIT
  local copy marker nontemp nontemp_before nontemp_after real_before real_after
  local repo_probe repo_before repo_after root_real
  copy="$tmp/install-copy.sh"
  marker="$tmp/append-ran"
  cp -- "$ROOT/scripts/install.sh" "$copy"
  chmod +x "$copy"
  bash "$copy" --append-probe "$marker"
  if [[ -e "$marker" ]]; then
    echo "install self-test: text appended after start was executed" >&2
    exit 1
  fi
  # A copy outside a temp directory must be refused, and left unmodified.
  # The repo script is checked only after that, so a missing guard cannot
  # append a line to scripts/install.sh.
  nontemp="$(mktemp -p "$HOME" llama-install-probe.XXXXXX)"
  SELF_TEST_NONTEMP="$nontemp"
  cp -- "$ROOT/scripts/install.sh" "$nontemp"
  chmod +x "$nontemp"
  nontemp_before="$(sha256sum -- "$nontemp")"
  if bash "$nontemp" --append-probe "$marker"; then
    echo "install self-test: --append-probe accepted a non-temporary copy" >&2
    exit 1
  fi
  nontemp_after="$(sha256sum -- "$nontemp")"
  if [[ "$nontemp_before" != "$nontemp_after" ]]; then
    echo "install self-test: --append-probe modified a non-temporary copy" >&2
    exit 1
  fi
  rm -f -- "$nontemp"
  SELF_TEST_NONTEMP=""
  # A checkout under /tmp or /var/tmp makes the repo script look like a
  # throwaway copy, so the probe would append to scripts/install.sh.
  # Skip that case. The self-test never writes "$ROOT"/scripts itself.
  root_real="$(readlink -f -- "$ROOT")"
  case "$root_real" in
    /tmp/*|/var/tmp/*)
      echo "install self-test: skipping repo-script append probe; ROOT is under a temporary directory"
      ;;
    *)
      real_before="$(sha256sum -- "$ROOT/scripts/install.sh")"
      if bash "$ROOT/scripts/install.sh" --append-probe "$marker"; then
        echo "install self-test: --append-probe accepted the repo script" >&2
        exit 1
      fi
      real_after="$(sha256sum -- "$ROOT/scripts/install.sh")"
      if [[ "$real_before" != "$real_after" ]]; then
        echo "install self-test: --append-probe modified the repo script" >&2
        exit 1
      fi
      real_before="$(sha256sum -- "$ROOT/scripts/install.sh")"
      if TMPDIR=/ bash "$ROOT/scripts/install.sh" --append-probe "$marker"; then
        echo "install self-test: --append-probe accepted TMPDIR=/" >&2
        exit 1
      fi
      real_after="$(sha256sum -- "$ROOT/scripts/install.sh")"
      if [[ "$real_before" != "$real_after" ]]; then
        echo "install self-test: --append-probe with TMPDIR=/ modified the repo script" >&2
        exit 1
      fi
      ;;
  esac
  mkdir -p -- "$ROOT/target"
  repo_probe="$ROOT/target/append-probe-copy.sh"
  case "$(readlink -f -- "$ROOT/target")" in
    /tmp/*|/var/tmp/*)
      echo "install self-test: skipping target append probe; the copy would be under a temporary directory"
      ;;
    *)
      SELF_TEST_REPO_PROBE="$repo_probe"
      cp -- "$ROOT/scripts/install.sh" "$repo_probe"
      chmod +x "$repo_probe"
      repo_before="$(sha256sum -- "$repo_probe")"
      if TMPDIR="$ROOT/target" bash "$repo_probe" --append-probe "$marker"; then
        echo "install self-test: --append-probe accepted TMPDIR under the repo" >&2
        exit 1
      fi
      repo_after="$(sha256sum -- "$repo_probe")"
      if [[ "$repo_before" != "$repo_after" ]]; then
        echo "install self-test: --append-probe with TMPDIR under the repo modified the copy" >&2
        exit 1
      fi
      rm -f -- "$repo_probe"
      SELF_TEST_REPO_PROBE=""
      ;;
  esac
  repo="$tmp/repo"
  dest="$tmp/dest"
  sys="$tmp/sys"
  head="0123456789abcdef0123456789abcdef01234567"
  mkdir -p "$repo/packaging/user" "$repo/packaging/fonts" "$repo/target/release"
  cp -- "$ROOT/packaging/kraken-lcd.service" "$repo/packaging/kraken-lcd.service"
  cp -- "$ROOT/packaging/llama-bored.sysusers" "$repo/packaging/llama-bored.sysusers"
  cp -- "$ROOT/packaging/71-kraken-lcd.rules" "$repo/packaging/71-kraken-lcd.rules"
  cp -- "$ROOT/packaging/93-kraken-lcd-hidraw.rules" "$repo/packaging/93-kraken-lcd-hidraw.rules"
  cp -- "$ROOT/packaging/72-llama-view.rules" "$repo/packaging/72-llama-view.rules"
  cp -- "$ROOT/packaging/llama-light.service" "$repo/packaging/llama-light.service"
  cp -- "$ROOT/packaging/94-llama-light-hidraw.rules" "$repo/packaging/94-llama-light-hidraw.rules"
  cp -- "$ROOT/packaging/light.example.toml" "$repo/packaging/light.example.toml"
  cp -- "$ROOT/packaging/llama-metrics.service" "$repo/packaging/llama-metrics.service"
  cp -- "$ROOT/packaging/metrics.example.toml" "$repo/packaging/metrics.example.toml"
  cp -- "$ROOT/packaging/fonts/llama-hack-12x24.psfu" "$repo/packaging/fonts/llama-hack-12x24.psfu"
  cp -- "$ROOT/packaging/config.example.toml" "$repo/packaging/config.example.toml"
  cp -- "$ROOT/packaging/watch.example.toml" "$repo/packaging/watch.example.toml"
  cp -- "$ROOT/packaging/llama-watch.service" "$repo/packaging/llama-watch.service"
  cp -- "$ROOT/packaging/user/kraken-lcd-halt.path" "$repo/packaging/user/kraken-lcd-halt.path"
  cp -- "$ROOT/packaging/user/kraken-lcd-halt-notify.service" \
    "$repo/packaging/user/kraken-lcd-halt-notify.service"
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  printf 'fake-view\n' >"$repo/target/release/llama-view"
  printf 'fake-light\n' >"$repo/target/release/llama-light"
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"
  write_fixture_provenance "$repo" "$head"

  body="$(awk '/^install_real\(\)/{f=1} f{print} f && /^}$/{exit}' "$ROOT/scripts/install.sh")"
  if grep -E -q '(^|[^[:alnum:]_])(git|cargo|rustc|brew)([^[:alnum:]_]|$)' <<<"$body"; then
    echo "install self-test: root install path names a user toolchain" >&2
    exit 1
  fi
  # The dollar sign is text in the script, not an expansion.
  # shellcheck disable=SC2016
  if ! grep -F -q 'export PATH="$ROOT_PATH"' <<<"$body"; then
    echo "install self-test: root install path does not lock PATH" >&2
    exit 1
  fi
  local anchor_body refuse_at freeze_at landed_at
  anchor_body="$(awk '/^anchored_install\(\)/{f=1} f{print} f && /^}$/{exit}' "$ROOT/scripts/install.sh")"
  refuse_at="$(grep -n -F -m 1 'refuse_rollback_pending' <<<"$anchor_body" | cut -d: -f1)"
  landed_at="$(grep -n -F -m 1 -- '--require-landed' <<<"$anchor_body" | cut -d: -f1)"
  freeze_at="$(grep -n -F -m 1 'freeze_staging' <<<"$anchor_body" | cut -d: -f1)"
  if [[ -z "$refuse_at" || -z "$landed_at" || -z "$freeze_at" \
    || "$refuse_at" -ge "$landed_at" || "$landed_at" -ge "$freeze_at" ]]; then
    echo "install self-test: rollback and landed-commit checks are not before staging" >&2
    exit 1
  fi
  [[ "$ROOT_PATH" == "/usr/sbin:/usr/bin:/sbin:/bin" ]] || {
    echo "install self-test: ROOT_PATH is not the system directories" >&2
    exit 1
  }
  if grep -F -q 'Type the short HEAD SHA' <<<"$body$anchor_body"; then
    echo "install self-test: typed SHA prompt is still present" >&2
    exit 1
  fi
  if grep -F -q 'refuse_noninteractive' <<<"$body$anchor_body"; then
    echo "install self-test: non-interactive refusal is still present" >&2
    exit 1
  fi
  local owner_mode_at owner_call_at
  owner_mode_at="$(grep -n -F -m 1 'INSTALL_OWNER_MODE=runuser' <<<"$body" | cut -d: -f1 || true)"
  owner_call_at="$(grep -n -F -m 1 'anchored_install ' <<<"$body" | cut -d: -f1 || true)"
  if [[ -z "$owner_mode_at" || -z "$owner_call_at" || "$owner_mode_at" -ge "$owner_call_at" ]]; then
    echo "install self-test: install_real does not run the owner git calls through runuser" >&2
    exit 1
  fi

  run_expect_fail install_real

  local min_file="$tmp/min.toml"
  printf '[upload]\nmin_interval_s = 45 # seconds\n' >"$min_file"
  assert_eq "$(min_interval_of "$min_file")" "45" "parsed min_interval"
  printf '[collector]\nmin_interval_s = 1\n[upload]\nmin_interval_s = 10\n' >"$min_file"
  assert_eq "$(min_interval_of "$min_file")" "10" "min_interval only in upload"
  printf '[upload]\n# min_interval_s = 1\n' >"$min_file"
  assert_eq "$(min_interval_of "$min_file")" "60" "absent key defaults to 60"
  printf '[upload]\nmin_interval_s = 9\n' >"$min_file"
  run_expect_fail min_interval_of "$min_file"

  unit="$(cat -- "$ROOT/packaging/kraken-lcd.service")"
  [[ "$unit" == *"RestartPreventExitStatus=2"* ]] || {
    echo "install self-test: unit missing RestartPreventExitStatus=2" >&2
    exit 1
  }
  [[ "$unit" == *$'\nRestartSec=60\n'* ]] || {
    echo "install self-test: packaged RestartSec is not 60" >&2
    exit 1
  }
  grep -F -q 'GROUP="kraken-lcd", MODE="0660"' "$ROOT/packaging/71-kraken-lcd.rules"
  grep -F -q 'KERNELS=="0003:1E71:3008.*"' "$ROOT/packaging/93-kraken-lcd-hidraw.rules"
  if grep -E -q 'OWNER[[:space:]]*:?=' "$ROOT/packaging/"*.rules; then
    echo "install self-test: packaged rules set OWNER" >&2
    exit 1
  fi

  seed_kraken_sysfs "$sys"
  export INSTALL_DRY=1 INSTALL_ALLOW_UNPRIV_COPY=1
  unset INSTALL_FAIL_AT
  export INSTALL_LOG="$tmp/log"
  export INSTALL_FAKE_STAT='660 kraken-lcd'
  export INSTALL_FAKE_GETFACL=$'# file: /dev/hidraw0\n# owner: root\n# group: kraken-lcd\nuser::rw-\ngroup::rw-\nother::---\n'
  export INSTALL_FAKE_PIN=/dev/hidraw0
  : >"$INSTALL_LOG"

  mkdir -p "$(dest_path "$dest" /etc/llama-bored)" \
    "$(dest_path "$dest" /etc/udev/rules.d)" \
    "$(dest_path "$dest" /usr/local/libexec/llama-bored)"
  printf 'custom config\n[upload]\nmin_interval_s = 45\n' >"$(dest_path "$dest" /etc/llama-bored/config.toml)"
  printf 'MODE="0600"\n' >"$(dest_path "$dest" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)"
  printf 'previous\n' >"$(dest_path "$dest" /usr/local/libexec/llama-bored/kraken-lcd)"

  staging="$(freeze_staging "$repo" "$dest")"
  assert_eq "$(stat -c '%a' -- "$staging")" "700" "staging mode"
  out="$(show_staging_hashes "$staging")"
  [[ "$out" == *"$(sha256sum -- "$staging/kraken-lcd" | awk '{ print $1 }')"* ]] || {
    echo "install self-test: staged hash was not shown" >&2
    exit 1
  }
  run_expect_ok verify_staged_binary "$staging"
  local prov_good="$tmp/prov.good"
  cp -- "$staging/check-provenance.txt" "$prov_good"
  head -n 7 "$prov_good" >"$staging/check-provenance.txt"
  run_expect_fail verify_staged_binary "$staging"
  cp -- "$prov_good" "$staging/check-provenance.txt"
  printf 'extra\n' >>"$staging/check-provenance.txt"
  run_expect_fail verify_staged_binary "$staging"
  cp -- "$prov_good" "$staging/check-provenance.txt"
  run_expect_ok verify_staged_binary "$staging"
  run_expect_ok verify_staged_matches_disk "$staging" "$repo" "$dest"
  printf 'tampered\n' >"$repo/target/release/kraken-lcd"
  run_expect_fail verify_staged_matches_disk "$staging" "$repo" "$dest"
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'tampered-watch\n' >"$repo/target/release/llama-watch"
  run_expect_fail verify_staged_matches_disk "$staging" "$repo" "$dest"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  printf '\n# tampered\n' >>"$repo/packaging/llama-watch.service"
  run_expect_fail verify_staged_matches_disk "$staging" "$repo" "$dest"
  cp -- "$ROOT/packaging/llama-watch.service" "$repo/packaging/llama-watch.service"
  printf 'x' >>"$repo/packaging/fonts/llama-hack-12x24.psfu"
  run_expect_fail verify_staged_matches_disk "$staging" "$repo" "$dest"
  cp -- "$ROOT/packaging/fonts/llama-hack-12x24.psfu" "$repo/packaging/fonts/llama-hack-12x24.psfu"
  [[ "$(show_staging_hashes "$staging")" == *"$(sha256sum -- "$ROOT/packaging/fonts/llama-hack-12x24.psfu" | awk '{ print $1 }')  $staging/llama-hack-12x24.psfu"* ]] || {
    echo "install self-test: staged font hash was not shown" >&2
    exit 1
  }
  out="$(apply_from_staging "$staging" "$dest" "$sys" "$head")"
  assert_eq "$(cat -- "$(dest_path "$dest" /usr/local/libexec/llama-bored/kraken-lcd)")" \
    "fake-binary" "installed the staged binary"
  assert_eq "$(cat -- "$(dest_path "$dest" /usr/local/libexec/llama-bored/kraken-lcd.bak-none)")" \
    "previous" "previous binary kept"
  apply_from_staging "$staging" "$dest" "$sys" "$head" >/dev/null
  assert_eq "$(cat -- "$(dest_path "$dest" /usr/local/libexec/llama-bored/kraken-lcd.bak-none)")" \
    "previous" "same-version reinstall kept the older backup"
  assert_eq "$(cat -- "$(dest_path "$dest" /etc/llama-bored/config.toml)")" \
    "$(printf 'custom config\n[upload]\nmin_interval_s = 45\n')" \
    "existing config preserved"
  assert_eq "$(cat -- "$(dest_path "$dest" /usr/local/libexec/llama-bored/INSTALLED_SHA)")" \
    "$head" "INSTALLED_SHA"
  mode="$(stat -c '%a' -- "$(dest_path "$dest" /usr/local/libexec/llama-bored/kraken-lcd)")"
  assert_eq "$mode" "755" "binary mode"
  grep -q '^RestartSec=45$' "$(dest_path "$dest" /etc/systemd/system/kraken-lcd.service)"
  cmp -s "$ROOT/packaging/93-kraken-lcd-hidraw.rules" \
    "$(dest_path "$dest" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)"
  cmp -s "$ROOT/packaging/user/kraken-lcd-halt.path" \
    "$(dest_path "$dest" /etc/systemd/user/kraken-lcd-halt.path)"
  local font_installed
  font_installed="$(dest_path "$dest" /usr/local/share/llama-bored/llama-hack-12x24.psfu)"
  cmp -s -- "$ROOT/packaging/fonts/llama-hack-12x24.psfu" "$font_installed" || {
    echo "install self-test: console font was not installed from staging" >&2
    exit 1
  }
  assert_eq "$(stat -c '%a' -- "$font_installed")" "644" "font mode"
  # The first apply created the font, so its rollback removes it.
  [[ "$(rollback_section "$out")" == *"rm -f -- /usr/local/share/llama-bored/llama-hack-12x24.psfu"* ]] || {
    echo "install self-test: rollback record does not remove the console font this run created" >&2
    exit 1
  }
  cmp -s "$ROOT/packaging/user/kraken-lcd-halt-notify.service" \
    "$(dest_path "$dest" /etc/systemd/user/kraken-lcd-halt-notify.service)"
  # Generic, public next steps. install.sh prints them and runs none of them
  # (the log checks below and log_mentions_writer_lifecycle).
  local step
  # shellcheck disable=SC2016 # $SUDO_USER is printed literally
  for step in \
    "llama-watch is enabled; start it" \
    "systemctl start llama-watch" \
    "Create the state dir if needed" \
    "install -d -o kraken-lcd -g kraken-lcd -m 0755 /var/lib/kraken-lcd" \
    "runuser -u kraken-lcd -- /usr/local/libexec/llama-bored/kraken-lcd show-image --view <test-card>" \
    "systemctl enable --now kraken-lcd" \
    "systemctl try-restart llama-watch kraken-lcd llama-light llama-metrics" \
    "systemctl --global enable kraken-lcd-halt.path" \
    'usermod -aG llama-view $SUDO_USER' \
    "how to roll back"; do
    [[ "$out" == *"$step"* ]] || {
      echo "install self-test: next steps are missing: $step" >&2
      exit 1
    }
  done
  if printf '%s\n' "${out%%how to roll back*}" | grep -E -q 'RR-?[A-Z0-9]|\bT[0-9]+\b|Phase B|--trace-hid|was restarted'; then
    echo "install self-test: next steps use internal ticket jargon or claim a restart" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  local state_dir
  state_dir="$(dest_path "$dest" /var/lib/kraken-lcd)"
  [[ -d "$state_dir" && ! -L "$state_dir" ]] || {
    echo "install self-test: state dir /var/lib/kraken-lcd was not created" >&2
    exit 1
  }
  assert_eq "$(stat -c '%a' -- "$state_dir")" "755" "state dir mode"
  log_has "chown kraken-lcd:kraken-lcd -- $state_dir" || {
    echo "install self-test: state dir was not given to kraken-lcd" >&2
    exit 1
  }
  [[ "$(rollback_section "$out")" == *"rmdir -- /var/lib/kraken-lcd"* ]] || {
    echo "install self-test: rollback record does not remove the state dir this run created" >&2
    exit 1
  }
  log_has 'systemd-sysusers'
  log_has 'udevadm control --reload'
  log_has 'udevadm trigger --action=change --attr-match=idVendor=1e71 --attr-match=idProduct=3008'
  log_has "udevadm trigger --action=change $sys/class/hidraw/hidraw0"
  log_has 'udevadm settle'
  log_has 'setfacl -b -- /dev/hidraw0'
  log_has 'systemctl daemon-reload'
  if log_has 'enable --now' || log_has 'property-match'; then
    echo "install self-test: root commands still start the service or OR property matches" >&2
    exit 1
  fi
  if log_mentions_writer_lifecycle; then
    echo "install self-test: dry-run log starts, restarts, or enables kraken-lcd" >&2
    exit 1
  fi
  local sysusers_line udev_line reload_line enable_line
  sysusers_line="$(log_line_number 'systemd-sysusers')"
  udev_line="$(log_line_number 'udevadm')"
  reload_line="$(log_line_number 'systemctl daemon-reload')"
  enable_line="$(log_line_number 'systemctl enable llama-watch.service')"
  if [[ -z "$sysusers_line" || -z "$udev_line" || -z "$reload_line" || -z "$enable_line" \
    || "$sysusers_line" -ge "$udev_line" || "$udev_line" -ge "$reload_line" \
    || "$reload_line" -ge "$enable_line" ]]; then
    echo "install self-test: host command order is not sysusers, udev, daemon-reload, enable llama-watch" >&2
    cat -- "$INSTALL_LOG" >&2
    exit 1
  fi
  rm -rf -- "$staging"

  local low="$tmp/low-dest" low_stage
  mkdir -p "$(dest_path "$low" /etc/llama-bored)"
  printf '[upload]\nmin_interval_s = 9\n' >"$(dest_path "$low" /etc/llama-bored/config.toml)"
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  write_fixture_provenance "$repo" "$head"
  low_stage="$(freeze_staging "$repo" "$low")"
  : >"$INSTALL_LOG"
  run_expect_fail apply_from_staging "$low_stage" "$low" "$sys" "$head"
  if [[ -e "$(dest_path "$low" /usr/local/libexec/llama-bored/kraken-lcd)" ]]; then
    echo "install self-test: invalid config still copied the binary" >&2
    exit 1
  fi
  rm -rf -- "$low_stage"

  local bad="$tmp/bad-stat" bad_stage
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  write_fixture_provenance "$repo" "$head"
  bad_stage="$(freeze_staging "$repo" "$bad")"
  mkdir -p -- "$bad"
  snapshot_tree "$bad" "$tmp/snap-hidraw"
  : >"$INSTALL_LOG"
  INSTALL_FAKE_STAT='600 root'
  local bad_err bad_status hid_section
  set +e
  bad_err="$(apply_from_staging "$bad_stage" "$bad" "$sys" "$head" 2>&1 >/dev/null)"
  bad_status=$?
  set -e
  if [[ "$bad_status" -eq 0 ]]; then
    echo "install self-test: hidraw mode failure was accepted" >&2
    exit 1
  fi
  [[ "$bad_err" == *"how to roll back"* && "$bad_err" == *"failed after these steps:"* ]] || {
    echo "install self-test: hidraw failure did not print rollback" >&2
    printf '%s\n' "$bad_err" >&2
    exit 1
  }
  if [[ -e "$(dest_path "$bad" /usr/local/libexec/llama-bored/kraken-lcd)" \
    || -e "$(dest_path "$bad" /usr/local/libexec/llama-bored/llama-watch)" ]]; then
    echo "install self-test: hidraw failure swapped a binary" >&2
    exit 1
  fi
  if [[ -e "$(dest_path "$bad" /usr/local/libexec/llama-bored/INSTALLED_SHA)" ]]; then
    echo "install self-test: INSTALLED_SHA was written before verify succeeded" >&2
    exit 1
  fi
  if log_has 'systemctl daemon-reload'; then
    echo "install self-test: daemon-reload ran before the hidraw check passed" >&2
    exit 1
  fi
  log_has 'udevadm settle' || {
    echo "install self-test: settle did not run before the mode check" >&2
    exit 1
  }
  assert_no_new "$bad"
  assert_disable_present "$bad_err" 0
  assert_rollback_order "$bad_err"
  hid_section="$(rollback_section "$bad_err")"
  [[ "$hid_section" == *"/etc/sysusers.d/llama-bored.conf"* \
    && "$hid_section" == *"/etc/udev/rules.d/71-kraken-lcd.rules"* \
    && "$hid_section" == *"/etc/udev/rules.d/93-kraken-lcd-hidraw.rules"* \
    && "$hid_section" != *"llama-watch"* ]] || {
    echo "install self-test: hidraw rollback did not list only the committed files" >&2
    printf '%s\n' "$hid_section" >&2
    exit 1
  }
  execute_printed_rollback "$bad_err" "$bad"
  assert_same_tree "$bad" "$tmp/snap-hidraw" "hidraw rollback"
  assert_no_new "$bad"
  rm -rf -- "$bad_stage"

  local acl="$tmp/bad-acl" acl_stage
  : >"$INSTALL_LOG"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  acl_stage="$(freeze_staging "$repo" "$acl")"
  INSTALL_FAKE_GETFACL=$'user::rw-\nuser:someone:rw-\n' \
    run_expect_fail apply_from_staging "$acl_stage" "$acl" "$sys" "$head"
  if [[ -e "$(dest_path "$acl" /usr/local/libexec/llama-bored/INSTALLED_SHA)" ]]; then
    echo "install self-test: a user ACL still recorded INSTALLED_SHA" >&2
    exit 1
  fi
  assert_no_new "$acl"
  rm -rf -- "$acl_stage"

  # SAFETY.md RR7: the udev pin must name the node sysfs gives, and must exist.
  local pin_case pin_dest pin_stage pin_err
  for pin_case in /dev/hidraw3 ''; do
    pin_dest="$tmp/bad-pin-${pin_case##*/}"
    : >"$INSTALL_LOG"
    INSTALL_FAKE_STAT='660 kraken-lcd'
    pin_stage="$(freeze_staging "$repo" "$pin_dest")"
    set +e
    pin_err="$(INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n' INSTALL_FAKE_PIN="$pin_case" \
      apply_from_staging "$pin_stage" "$pin_dest" "$sys" "$head" 2>&1 >/dev/null)"
    local pin_status=$?
    set -e
    if [[ "$pin_status" -eq 0 ]]; then
      echo "install self-test: udev pin '${pin_case:-absent}' was accepted" >&2
      exit 1
    fi
    [[ "$pin_err" == *"/dev/kraken-lcd/hid"* ]] || {
      echo "install self-test: udev pin failure did not name /dev/kraken-lcd/hid" >&2
      printf '%s\n' "$pin_err" >&2
      exit 1
    }
    if [[ -e "$(dest_path "$pin_dest" /usr/local/libexec/llama-bored/kraken-lcd)" ]] \
      || log_has 'systemctl daemon-reload'; then
      echo "install self-test: udev pin failure still swapped the binary or reloaded units" >&2
      exit 1
    fi
    assert_no_new "$pin_dest"
    rm -rf -- "$pin_stage"
  done

  local trial_root="$tmp/trial-dest" trial_stage kept
  mkdir -p "$(dest_path "$trial_root" /etc/udev/rules.d)"
  # An OWNER rule that does not name the Kraken is not a trial rule.
  mkdir -p "$tmp/unrelated-rules"
  printf 'SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", OWNER="someone"\n' >"$tmp/unrelated-rules/50-other.rules"
  printf 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1e71", OWNER="someone"\n' >"$tmp/unrelated-rules/93-kraken-lcd-hidraw.rules.bak-none"
  if [[ -n "$(trial_rule_files "$tmp/unrelated-rules")" ]]; then
    echo "install self-test: an unrelated OWNER rule was treated as a trial rule" >&2
    exit 1
  fi
  printf 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1E71", OWNER="someone"\n' >"$(dest_path "$trial_root" /etc/udev/rules.d/71-kraken-lcd.rules)"
  kept="$(cat -- "$(dest_path "$trial_root" /etc/udev/rules.d/71-kraken-lcd.rules)")"
  trial_stage="$(freeze_staging "$repo" "$trial_root")"
  : >"$INSTALL_LOG"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  run_expect_fail apply_from_staging "$trial_stage" "$trial_root" "$sys" "$head"
  assert_eq "$(cat -- "$(dest_path "$trial_root" /etc/udev/rules.d/71-kraken-lcd.rules)")" \
    "$kept" "trial rule was overwritten"
  if log_has 'udevadm'; then
    echo "install self-test: trial rule still triggered udev" >&2
    exit 1
  fi
  if [[ -e "$(dest_path "$trial_root" /usr/local/libexec/llama-bored/kraken-lcd)" ]]; then
    echo "install self-test: trial rule still installed the binary" >&2
    exit 1
  fi
  rm -rf -- "$trial_stage"

  local fresh="$tmp/fresh" fresh_stage
  : >"$INSTALL_LOG"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  fresh_stage="$(freeze_staging "$repo" "$fresh")"
  run_expect_ok apply_from_staging "$fresh_stage" "$fresh" "$sys" "$head"
  cmp -s "$repo/packaging/config.example.toml" "$(dest_path "$fresh" /etc/llama-bored/config.toml)"
  grep -q '^RestartSec=60$' "$(dest_path "$fresh" /etc/systemd/system/kraken-lcd.service)"
  if [[ -e "$(dest_path "$fresh" /usr/local/libexec/llama-bored/kraken-lcd.bak)" \
    || -e "$(dest_path "$fresh" /usr/local/libexec/llama-bored/kraken-lcd.bak-none)" ]]; then
    echo "install self-test: a first install invented a binary backup" >&2
    exit 1
  fi
  rm -rf -- "$fresh_stage"

  local linkdest="$tmp/link-dest" target="$tmp/keep" link_stage
  mkdir -p "$(dest_path "$linkdest" /usr/local/libexec/llama-bored)"
  printf 'keep\n' >"$target"
  ln -s "$target" "$(dest_path "$linkdest" /usr/local/libexec/llama-bored/kraken-lcd)"
  link_stage="$(freeze_staging "$repo" "$linkdest")"
  run_expect_fail apply_from_staging "$link_stage" "$linkdest" "$sys" "$head"
  assert_eq "$(cat -- "$target")" "keep" "symlink target overwritten"
  rm -rf -- "$link_stage" "$staging" 2>/dev/null || true

  # Both binaries, watch.toml, the config backup, legacy refusal, enable/restart.
  local both both_stage both_hashes legacy legacy_stage legacy_err legacy_status
  local models_only models_stage models_err models_status fresh_watch fresh_stage
  local bak sentinel
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  cp -- "$ROOT/packaging/llama-watch.service" "$repo/packaging/llama-watch.service"
  cp -- "$ROOT/packaging/watch.example.toml" "$repo/packaging/watch.example.toml"
  write_fixture_provenance "$repo" "$head"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'

  both="$tmp/both"
  mkdir -p "$(dest_path "$both" /etc/llama-bored)" "$(dest_path "$both" /etc/udev/rules.d)"
  printf 'custom config\n[upload]\nmin_interval_s = 30\n' >"$(dest_path "$both" /etc/llama-bored/config.toml)"
  printf 'custom watch\n' >"$(dest_path "$both" /etc/llama-bored/watch.toml)"
  : >"$INSTALL_LOG"
  both_stage="$(freeze_staging "$repo" "$both")"
  if [[ ! -f "$both_stage/llama-watch" || ! -f "$both_stage/llama-watch.service" ]]; then
    echo "install self-test: llama-watch binary and unit were not staged" >&2
    exit 1
  fi
  both_hashes="$(show_staging_hashes "$both_stage")"
  [[ "$both_hashes" == *"$(sha256sum -- "$both_stage/kraken-lcd")"* \
    && "$both_hashes" == *"$(sha256sum -- "$both_stage/llama-watch")"* \
    && "$both_hashes" == *"$(sha256sum -- "$both_stage/watch.toml")"* \
    && "$both_hashes" != *"config.toml.new"* ]] || {
    echo "install self-test: staged hashes omitted a binary or watch.toml" >&2
    exit 1
  }
  run_expect_ok verify_staged_binary "$both_stage"
  printf 'tampered-watch\n' >"$both_stage/llama-watch"
  run_expect_fail verify_staged_binary "$both_stage"
  printf 'fake-watch\n' >"$both_stage/llama-watch"
  apply_from_staging "$both_stage" "$both" "$sys" "$head" >/dev/null
  assert_eq "$(cat -- "$(dest_path "$both" /usr/local/libexec/llama-bored/kraken-lcd)")" \
    "fake-binary" "installed kraken-lcd"
  assert_eq "$(cat -- "$(dest_path "$both" /usr/local/libexec/llama-bored/llama-watch)")" \
    "fake-watch" "installed llama-watch"
  assert_eq "$(cat -- "$(dest_path "$both" /etc/llama-bored/watch.toml)")" \
    "custom watch" "watch.toml not overwritten"
  bak="$(dest_path "$both" /etc/llama-bored/config.toml.bak-none)"
  assert_eq "$(cat -- "$bak")" \
    "$(printf 'custom config\n[upload]\nmin_interval_s = 30\n')" \
    "live config backed up"
  cmp -s -- "$ROOT/packaging/llama-watch.service" \
    "$(dest_path "$both" /etc/systemd/system/llama-watch.service)" || {
    echo "install self-test: llama-watch.service was not installed verbatim" >&2
    exit 1
  }
  grep -q '^RestartSec=30$' "$(dest_path "$both" /etc/systemd/system/kraken-lcd.service)"
  grep -q '^RestartSec=5$' "$(dest_path "$both" /etc/systemd/system/llama-watch.service)"
  log_has 'systemctl enable llama-watch.service' || {
    echo "install self-test: llama-watch was not enabled" >&2
    exit 1
  }
  if log_mentions_writer_lifecycle; then
    echo "install self-test: dry-run log starts, restarts, or enables kraken-lcd" >&2
    exit 1
  fi
  if log_has 'systemctl start llama-watch'; then
    echo "install self-test: llama-watch was started" >&2
    exit 1
  fi
  # The second apply reads INSTALLED_SHA=$head, so the backup it would
  # write is bak-$head. A sentinel on bak-none is not that file.
  local sentinel_bak
  sentinel_bak="$(dest_path "$both" "/etc/llama-bored/config.toml.bak-$head")"
  printf 'sentinel-backup\n' >"$sentinel_bak"
  apply_from_staging "$both_stage" "$both" "$sys" "$head" >/dev/null
  assert_eq "$(cat -- "$sentinel_bak")" "sentinel-backup" "config backup was overwritten"
  assert_eq "$(cat -- "$bak")" \
    "$(printf 'custom config\n[upload]\nmin_interval_s = 30\n')" \
    "first config backup changed on the second apply"
  rm -rf -- "$both_stage"

  fresh_watch="$tmp/fresh-watch"
  : >"$INSTALL_LOG"
  fresh_stage="$(freeze_staging "$repo" "$fresh_watch")"
  apply_from_staging "$fresh_stage" "$fresh_watch" "$sys" "$head" >/dev/null
  cmp -s -- "$ROOT/packaging/watch.example.toml" \
    "$(dest_path "$fresh_watch" /etc/llama-bored/watch.toml)" || {
    echo "install self-test: absent watch.toml was not installed from the example" >&2
    exit 1
  }
  if [[ -e "$(dest_path "$fresh_watch" "/etc/llama-bored/config.toml.bak-$head")" ]]; then
    echo "install self-test: a first install invented a config backup" >&2
    exit 1
  fi
  rm -rf -- "$fresh_stage"

  legacy="$tmp/legacy"
  mkdir -p "$(dest_path "$legacy" /etc/llama-bored)"
  cp -- "$ROOT/fixtures/config/legacy-config.toml" \
    "$(dest_path "$legacy" /etc/llama-bored/config.toml)"
  printf 'sentinel\n' >"$(dest_path "$legacy" /etc/llama-bored/sentinel)"
  sentinel="$(cat -- "$(dest_path "$legacy" /etc/llama-bored/config.toml)")"
  : >"$INSTALL_LOG"
  legacy_stage="$(freeze_staging "$repo" "$legacy")"
  set +e
  legacy_err="$(apply_from_staging "$legacy_stage" "$legacy" "$sys" "$head" 2>&1 >/dev/null)"
  legacy_status=$?
  set -e
  if [[ "$legacy_status" -eq 0 ]]; then
    echo "install self-test: legacy config was accepted" >&2
    exit 1
  fi
  [[ "$legacy_err" == *"move these to watch.toml:"* && "$legacy_err" == *$'\ncollector\n'* \
    && "$legacy_err" == *"tick_s"* && "$legacy_err" == *"llama_swap_url"* \
    && "$legacy_err" == *"http_timeout_s"* && "$legacy_err" == *"cpu_top_k"* \
    && "$legacy_err" == *$'\nmodels\n'* && "$legacy_err" == *"max_name_chars"* \
    && "$legacy_err" == *"qwen3.6-35b-a3b"* \
    && "$legacy_err" == *"/etc/llama-bored/config.toml.new"* ]] || {
    echo "install self-test: legacy refusal did not print the keys to move" >&2
    printf '%s\n' "$legacy_err" >&2
    exit 1
  }
  if [[ "$legacy_err" == *$'\r'* ]]; then
    echo "install self-test: legacy refusal kept a carriage return" >&2
    exit 1
  fi
  assert_eq "$(cat -- "$(dest_path "$legacy" /etc/llama-bored/config.toml)")" \
    "$sentinel" "legacy config was edited"
  if [[ -e "$(dest_path "$legacy" /usr/local/libexec/llama-bored/kraken-lcd)" \
    || -e "$(dest_path "$legacy" /usr/local/libexec/llama-bored/llama-watch)" \
    || -e "$(dest_path "$legacy" "/etc/llama-bored/config.toml.bak-$head")" \
    || -e "$(dest_path "$legacy" /etc/llama-bored/config.toml.bak-none)" \
    || -e "$(dest_path "$legacy" /etc/llama-bored/watch.toml)" \
    || -e "$(dest_path "$legacy" /etc/systemd/system/kraken-lcd.service)" ]]; then
    echo "install self-test: legacy refusal still wrote install files" >&2
    exit 1
  fi
  assert_eq "$(cat -- "$(dest_path "$legacy" /etc/llama-bored/sentinel)")" \
    "sentinel" "legacy refusal touched another file"
  if log_has 'systemctl' || log_has 'udevadm'; then
    echo "install self-test: legacy refusal still ran host commands" >&2
    exit 1
  fi
  rm -rf -- "$legacy_stage"

  models_only="$tmp/models-only"
  mkdir -p "$(dest_path "$models_only" /etc/llama-bored)"
  printf '[upload]\nmin_interval_s = 60\n[models]\nmax_name_chars = 12\n' \
    >"$(dest_path "$models_only" /etc/llama-bored/config.toml)"
  : >"$INSTALL_LOG"
  models_stage="$(freeze_staging "$repo" "$models_only")"
  set +e
  models_err="$(apply_from_staging "$models_stage" "$models_only" "$sys" "$head" 2>&1 >/dev/null)"
  models_status=$?
  set -e
  if [[ "$models_status" -eq 0 ]]; then
    echo "install self-test: [models] alone was accepted" >&2
    exit 1
  fi
  [[ "$models_err" == *"move these to watch.toml:"* && "$models_err" == *$'\nmodels\n'* \
    && "$models_err" == *"max_name_chars"* \
    && "$models_err" == *"/etc/llama-bored/config.toml.new"* ]] || {
    echo "install self-test: [models] refusal did not name the key" >&2
    printf '%s\n' "$models_err" >&2
    exit 1
  }
  if [[ -e "$(dest_path "$models_only" /usr/local/libexec/llama-bored/kraken-lcd)" ]]; then
    echo "install self-test: [models] refusal still copied the binary" >&2
    exit 1
  fi
  rm -rf -- "$models_stage"

  # Allowlist: every legacy spelling is a top-level collector or models name.
  local variant variant_file scan_out
  for variant in \
    '[ collector ]'$'\n''tick_s = 1' \
    '[[collector]]'$'\n''tick_s = 1' \
    '["collector"]'$'\n''tick_s = 1' \
    'collector.tick_s = 1' \
    'collector = { tick_s = 1 }'
  do
    variant_file="$tmp/variant.toml"
    printf '%s\n' "$variant" >"$variant_file"
    scan_out="$(config_scan "$variant_file")"
    [[ "$scan_out" == *"MOVE"$'\t'"collector"* ]] || {
      echo "install self-test: allowlist missed collector variant: $variant" >&2
      printf '%s\n' "$scan_out" >&2
      exit 1
    }
  done
  printf '[ models . aliases ]\n"qwen3.6-35b-a3b" = "Qwen 35B"\n' >"$tmp/variant.toml"
  scan_out="$(config_scan "$tmp/variant.toml")"
  [[ "$scan_out" == *"MOVE"$'\t'"models"* ]] || {
    echo "install self-test: allowlist missed [ models . aliases ]" >&2
    printf '%s\n' "$scan_out" >&2
    exit 1
  }
  printf '[collector]\r\ntick_s = 1\r\n' >"$tmp/variant.toml"
  scan_out="$(config_scan "$tmp/variant.toml")"
  [[ "$scan_out" == *"MOVE"$'\t'"collector"* && "$scan_out" == *"KEY"$'\t'"tick_s"* ]] || {
    echo "install self-test: CRLF collector was not normalised" >&2
    printf '%s\n' "$scan_out" >&2
    exit 1
  }
  if [[ "$scan_out" == *$'\r'* ]]; then
    echo "install self-test: scan kept a carriage return" >&2
    exit 1
  fi
  printf 'not_a_real = 1\n' >"$tmp/variant.toml"
  scan_out="$(config_scan "$tmp/variant.toml")"
  [[ "$scan_out" == *"UNKNOWN"$'\t'"not_a_real"* ]] || {
    echo "install self-test: unknown root key was accepted" >&2
    printf '%s\n' "$scan_out" >&2
    exit 1
  }
  if ! config_is_clean "$ROOT/packaging/config.example.toml"; then
    echo "install self-test: writer example config is not on the allowlist" >&2
    exit 1
  fi
  printf '[dial]\ntiers = [\n  [0.5, 10],\n  [1.0, 20],\n]\n' >"$tmp/dial.toml"
  if ! config_is_clean "$tmp/dial.toml"; then
    echo "install self-test: multi-line [dial] array was read as a top-level key" >&2
    config_scan "$tmp/dial.toml" >&2 || true
    exit 1
  fi
  printf '[dial]\ntiers = [\n  [0.5, 10],\n]\n[collector]\ntick_s = 1\n' >"$tmp/dial.toml"
  scan_out="$(config_scan "$tmp/dial.toml")"
  [[ "$scan_out" == *"MOVE"$'\t'"collector"* && "$scan_out" == *"KEY"$'\t'"tick_s"* \
    && "$scan_out" != *"UNKNOWN"* ]] || {
    echo "install self-test: multi-line array hid a later collector table" >&2
    printf '%s\n' "$scan_out" >&2
    exit 1
  }

  # Legacy live config plus a good config.toml.new swaps in after backup.
  local migrated migrated_stage prev_sha
  prev_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  migrated="$tmp/migrated"
  mkdir -p "$(dest_path "$migrated" /etc/llama-bored)" \
    "$(dest_path "$migrated" /usr/local/libexec/llama-bored)"
  cp -- "$ROOT/fixtures/config/legacy-config.toml" \
    "$(dest_path "$migrated" /etc/llama-bored/config.toml)"
  cp -- "$ROOT/packaging/config.example.toml" \
    "$(dest_path "$migrated" /etc/llama-bored/config.toml.new)"
  printf '%s\n' "$prev_sha" >"$(dest_path "$migrated" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  : >"$INSTALL_LOG"
  migrated_stage="$(freeze_staging "$repo" "$migrated")"
  local mig_hashes
  mig_hashes="$(show_staging_hashes "$migrated_stage")"
  [[ "$mig_hashes" == *"$(sha256sum -- "$migrated_stage/config.toml.new")"* \
    && "$mig_hashes" == *"$(sha256sum -- "$migrated_stage/watch.toml")"* ]] || {
    echo "install self-test: staged hashes omitted config.toml.new or watch.toml" >&2
    printf '%s\n' "$mig_hashes" >&2
    exit 1
  }
  apply_from_staging "$migrated_stage" "$migrated" "$sys" "$head" >/dev/null
  assert_eq "$(cat -- "$(dest_path "$migrated" "/etc/llama-bored/config.toml.bak-$prev_sha")")" \
    "$(cat -- "$ROOT/fixtures/config/legacy-config.toml")" \
    "backup is the previous live config"
  cmp -s -- "$ROOT/packaging/config.example.toml" \
    "$(dest_path "$migrated" /etc/llama-bored/config.toml)" || {
    echo "install self-test: config.toml.new was not moved into place" >&2
    exit 1
  }
  if [[ -e "$(dest_path "$migrated" /etc/llama-bored/config.toml.new)" ]]; then
    echo "install self-test: config.toml.new was left beside the live config" >&2
    exit 1
  fi
  rm -rf -- "$migrated_stage"

  # A bad .new is refused and nothing is written.
  local badnew="$tmp/bad-new" badnew_stage badnew_err badnew_status kept_live
  mkdir -p "$(dest_path "$badnew" /etc/llama-bored)"
  cp -- "$ROOT/fixtures/config/legacy-config.toml" \
    "$(dest_path "$badnew" /etc/llama-bored/config.toml)"
  printf '[collector]\ntick_s = 1\n' >"$(dest_path "$badnew" /etc/llama-bored/config.toml.new)"
  kept_live="$(cat -- "$(dest_path "$badnew" /etc/llama-bored/config.toml)")"
  : >"$INSTALL_LOG"
  badnew_stage="$(freeze_staging "$repo" "$badnew")"
  set +e
  badnew_err="$(apply_from_staging "$badnew_stage" "$badnew" "$sys" "$head" 2>&1 >/dev/null)"
  badnew_status=$?
  set -e
  if [[ "$badnew_status" -eq 0 ]]; then
    echo "install self-test: legacy config with a bad .new was accepted" >&2
    exit 1
  fi
  [[ "$badnew_err" == *"config.toml.new"* && "$badnew_err" == *"collector"* ]] || {
    echo "install self-test: bad .new refusal did not name config.toml.new" >&2
    printf '%s\n' "$badnew_err" >&2
    exit 1
  }
  assert_eq "$(cat -- "$(dest_path "$badnew" /etc/llama-bored/config.toml)")" \
    "$kept_live" "bad .new edited the live config"
  if [[ -e "$(dest_path "$badnew" /usr/local/libexec/llama-bored/kraken-lcd)" \
    || -e "$(dest_path "$badnew" /etc/llama-bored/config.toml.bak-none)" ]]; then
    echo "install self-test: bad .new still wrote install files" >&2
    exit 1
  fi
  rm -rf -- "$badnew_stage"

  # Watcher copy fails after the writer .new is staged: neither binary is swapped.
  local miss="$tmp/miss-watch" miss_stage miss_err miss_status
  mkdir -p "$(dest_path "$miss" /etc/udev/rules.d)"
  : >"$INSTALL_LOG"
  miss_stage="$(freeze_staging "$repo" "$miss")"
  rm -f -- "$miss_stage/llama-watch"
  set +e
  miss_err="$(apply_from_staging "$miss_stage" "$miss" "$sys" "$head" 2>&1 >/dev/null)"
  miss_status=$?
  set -e
  if [[ "$miss_status" -eq 0 ]]; then
    echo "install self-test: missing watcher binary was installed" >&2
    exit 1
  fi
  [[ "$miss_err" == *"failed after these steps:"* && "$miss_err" == *"how to roll back"* \
    && "$miss_err" == *"staged kraken-lcd.new"* ]] || {
    echo "install self-test: watcher copy failure did not print rollback" >&2
    printf '%s\n' "$miss_err" >&2
    exit 1
  }
  if [[ -e "$(dest_path "$miss" /usr/local/libexec/llama-bored/kraken-lcd)" \
    || -e "$(dest_path "$miss" /usr/local/libexec/llama-bored/llama-watch)" ]]; then
    echo "install self-test: watcher copy failure swapped a binary" >&2
    exit 1
  fi
  assert_no_new "$miss"
  rm -rf -- "$miss_stage"

  # Execute the printed rollback. systemctl and udevadm go to the dry-run log.
  unset INSTALL_FAIL_AT
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'

  # (a) successful first install.
  local roll_a roll_a_stage roll_a_out
  roll_a="$tmp/roll-a"
  mkdir -p -- "$roll_a"
  snapshot_tree "$roll_a" "$tmp/snap-a"
  : >"$INSTALL_LOG"
  roll_a_stage="$(freeze_staging "$repo" "$roll_a")"
  roll_a_out="$(apply_from_staging "$roll_a_stage" "$roll_a" "$sys" "$head")"
  assert_disable_present "$roll_a_out" 1
  assert_udev_reload "$roll_a_out"
  [[ -d "$(dest_path "$roll_a" /var/lib/kraken-lcd)" ]] || {
    echo "install self-test: first install did not create the state dir" >&2
    exit 1
  }
  execute_printed_rollback "$roll_a_out" "$roll_a"
  assert_same_tree "$roll_a" "$tmp/snap-a" "first-install rollback"
  if [[ -e "$(dest_path "$roll_a" /var/lib/kraken-lcd)" ]]; then
    echo "install self-test: first-install rollback left the state dir it created" >&2
    exit 1
  fi
  assert_no_new "$roll_a"
  rm -rf -- "$roll_a_stage"

  # (b) second install: existing watch.toml, unchanged watcher binary and unit.
  local roll_b roll_b_stage roll_b_out roll_b_section
  roll_b="$tmp/roll-b"
  mkdir -p "$(dest_path "$roll_b" /etc/llama-bored)" \
    "$(dest_path "$roll_b" /usr/local/libexec/llama-bored)" \
    "$(dest_path "$roll_b" /etc/systemd/system)"
  printf 'custom watch\n' >"$(dest_path "$roll_b" /etc/llama-bored/watch.toml)"
  printf 'fake-watch\n' >"$(dest_path "$roll_b" /usr/local/libexec/llama-bored/llama-watch)"
  cp -- "$ROOT/packaging/llama-watch.service" \
    "$(dest_path "$roll_b" /etc/systemd/system/llama-watch.service)"
  printf '%s\n' "$head" >"$(dest_path "$roll_b" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  # An existing state dir (with a latch in it) is not this run's to remove.
  mkdir -p -- "$(dest_path "$roll_b" /var/lib/kraken-lcd)"
  printf 'latched\n' >"$(dest_path "$roll_b" /var/lib/kraken-lcd/halted)"
  snapshot_tree "$roll_b" "$tmp/snap-b"
  : >"$INSTALL_LOG"
  roll_b_stage="$(freeze_staging "$repo" "$roll_b")"
  roll_b_out="$(apply_from_staging "$roll_b_stage" "$roll_b" "$sys" "$head")"
  assert_eq "$(cat -- "$(dest_path "$roll_b" /etc/llama-bored/watch.toml)")" \
    "custom watch" "second install kept watch.toml"
  assert_eq "$(cat -- "$(dest_path "$roll_b" /usr/local/libexec/llama-bored/llama-watch)")" \
    "fake-watch" "second install kept the watcher binary"
  cmp -s -- "$ROOT/packaging/llama-watch.service" \
    "$(dest_path "$roll_b" /etc/systemd/system/llama-watch.service)" || {
    echo "install self-test: second install rewrote an unchanged watcher unit" >&2
    exit 1
  }
  roll_b_section="$(rollback_section "$roll_b_out")"
  if [[ "$roll_b_section" == *"watch.toml"* || "$roll_b_section" == *"llama-watch.service"* \
    || "$roll_b_section" == *"/llama-watch"* || "$roll_b_section" == *"rmdir"* ]]; then
    echo "install self-test: rollback recorded an unchanged watcher file or state dir" >&2
    printf '%s\n' "$roll_b_section" >&2
    exit 1
  fi
  assert_disable_present "$roll_b_out" 0
  assert_rollback_order "$roll_b_out"
  execute_printed_rollback "$roll_b_out" "$roll_b"
  assert_same_tree "$roll_b" "$tmp/snap-b" "second-install rollback"
  assert_eq "$(cat -- "$(dest_path "$roll_b" /etc/llama-bored/watch.toml)")" \
    "custom watch" "rollback removed watch.toml"
  assert_no_new "$roll_b"
  rm -rf -- "$roll_b_stage"

  # (d) failure at the watcher commit. The writer binary is already in place.
  local roll_d roll_d_stage roll_d_err roll_d_status roll_d_section
  roll_d="$tmp/roll-d"
  mkdir -p -- "$roll_d"
  snapshot_tree "$roll_d" "$tmp/snap-d"
  : >"$INSTALL_LOG"
  roll_d_stage="$(freeze_staging "$repo" "$roll_d")"
  INSTALL_FAIL_AT=commit:llama-watch
  set +e
  roll_d_err="$(apply_from_staging "$roll_d_stage" "$roll_d" "$sys" "$head" 2>&1 >/dev/null)"
  roll_d_status=$?
  set -e
  unset INSTALL_FAIL_AT
  if [[ "$roll_d_status" -eq 0 ]]; then
    echo "install self-test: watcher commit failure was accepted" >&2
    exit 1
  fi
  assert_no_new "$roll_d"
  assert_disable_present "$roll_d_err" 0
  assert_rollback_order "$roll_d_err"
  roll_d_section="$(rollback_section "$roll_d_err")"
  [[ "$roll_d_section" == *"/usr/local/libexec/llama-bored/kraken-lcd"* \
    && "$roll_d_section" != *"/usr/local/libexec/llama-bored/llama-watch"* \
    && "$roll_d_section" != *"llama-watch.service"* ]] || {
    echo "install self-test: watcher-commit rollback did not stop at the writer" >&2
    printf '%s\n' "$roll_d_section" >&2
    exit 1
  }
  execute_printed_rollback "$roll_d_err" "$roll_d"
  assert_same_tree "$roll_d" "$tmp/snap-d" "watcher-commit rollback"
  assert_no_new "$roll_d"
  rm -rf -- "$roll_d_stage"

  # (e) failure at daemon-reload. The rules file changes. A retry is refused
  # until the printed rollback removes ROLLBACK_PENDING.
  local roll_e roll_e_stage roll_e_err roll_e_status roll_e_marker
  local roll_e_retry_err roll_e_retry_status roll_e_out2
  roll_e="$tmp/roll-e"
  mkdir -p "$(dest_path "$roll_e" /etc/udev/rules.d)"
  printf 'OLD-RULE\n' >"$(dest_path "$roll_e" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)"
  snapshot_tree "$roll_e" "$tmp/snap-e"
  : >"$INSTALL_LOG"
  roll_e_stage="$(freeze_staging "$repo" "$roll_e")"
  INSTALL_FAIL_AT=daemon-reload
  set +e
  roll_e_err="$(apply_from_staging "$roll_e_stage" "$roll_e" "$sys" "$head" 2>&1 >/dev/null)"
  roll_e_status=$?
  set -e
  unset INSTALL_FAIL_AT
  if [[ "$roll_e_status" -eq 0 ]]; then
    echo "install self-test: daemon-reload failure was accepted" >&2
    exit 1
  fi
  assert_no_new "$roll_e"
  assert_disable_present "$roll_e_err" 1
  assert_udev_reload "$roll_e_err"
  if [[ -e "$(dest_path "$roll_e" /usr/local/libexec/llama-bored/INSTALLED_SHA)" ]]; then
    echo "install self-test: daemon-reload failure recorded INSTALLED_SHA" >&2
    exit 1
  fi
  if cmp -s -- "$ROOT/packaging/93-kraken-lcd-hidraw.rules" \
    "$(dest_path "$roll_e" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)"; then
    :
  else
    echo "install self-test: daemon-reload failure did not replace the rules file" >&2
    exit 1
  fi
  roll_e_marker="$(dest_path "$roll_e" /usr/local/libexec/llama-bored/ROLLBACK_PENDING)"
  [[ -f "$roll_e_marker" && ! -L "$roll_e_marker" ]] || {
    echo "install self-test: daemon-reload failure did not write ROLLBACK_PENDING" >&2
    exit 1
  }
  [[ "$roll_e_err" == *"wrote the rollback to $roll_e_marker"* ]] || {
    echo "install self-test: failure did not say where ROLLBACK_PENDING is" >&2
    exit 1
  }
  assert_eq "$(cat -- "$roll_e_marker")" "$(rollback_section "$roll_e_err")" \
    "ROLLBACK_PENDING is not the printed rollback"
  snapshot_tree "$roll_e" "$tmp/snap-e-failed"
  set +e
  roll_e_retry_err="$(apply_from_staging "$roll_e_stage" "$roll_e" "$sys" "$head" 2>&1 >/dev/null)"
  roll_e_retry_status=$?
  set -e
  if [[ "$roll_e_retry_status" -eq 0 ]]; then
    echo "install self-test: retry after a failed install was accepted" >&2
    exit 1
  fi
  [[ "$roll_e_retry_err" == *"a previous install failed; run the rollback in $roll_e_marker first (or remove the marker if you have already rolled back)"* ]] || {
    echo "install self-test: retry did not name the pending rollback" >&2
    printf '%s\n' "$roll_e_retry_err" >&2
    exit 1
  }
  assert_same_tree "$roll_e" "$tmp/snap-e-failed" "retry wrote after a pending rollback"
  execute_printed_rollback "$roll_e_err" "$roll_e"
  assert_same_tree "$roll_e" "$tmp/snap-e" "daemon-reload rollback"
  [[ ! -e "$roll_e_marker" && ! -L "$roll_e_marker" ]] || {
    echo "install self-test: rollback left ROLLBACK_PENDING" >&2
    exit 1
  }
  assert_eq "$(cat -- "$(dest_path "$roll_e" /etc/udev/rules.d/93-kraken-lcd-hidraw.rules)")" \
    "OLD-RULE" "rollback did not restore the previous rules"
  assert_no_new "$roll_e"
  roll_e_out2="$(apply_from_staging "$roll_e_stage" "$roll_e" "$sys" "$head")"
  assert_udev_reload "$roll_e_out2"
  execute_printed_rollback "$roll_e_out2" "$roll_e"
  assert_same_tree "$roll_e" "$tmp/snap-e" "retry rollback after the marker was cleared"
  assert_no_new "$roll_e"
  rm -rf -- "$roll_e_stage"

  # A symlink at the marker path is refused before any install write.
  local roll_link roll_link_stage roll_link_err roll_link_status roll_link_target
  roll_link="$tmp/roll-link"
  roll_link_target="$tmp/roll-link-target"
  printf 'sentinel\n' >"$roll_link_target"
  mkdir -p "$(dest_path "$roll_link" /usr/local/libexec/llama-bored)"
  ln -s "$roll_link_target" \
    "$(dest_path "$roll_link" /usr/local/libexec/llama-bored/ROLLBACK_PENDING)"
  roll_link_stage="$(freeze_staging "$repo" "$roll_link")"
  set +e
  roll_link_err="$(apply_from_staging "$roll_link_stage" "$roll_link" "$sys" "$head" 2>&1 >/dev/null)"
  roll_link_status=$?
  set -e
  if [[ "$roll_link_status" -eq 0 ]]; then
    echo "install self-test: a symlinked ROLLBACK_PENDING was accepted" >&2
    exit 1
  fi
  [[ "$roll_link_err" == *"refusing to follow a symlink at $(dest_path "$roll_link" /usr/local/libexec/llama-bored/ROLLBACK_PENDING)"* ]] || {
    echo "install self-test: symlink marker was not refused" >&2
    printf '%s\n' "$roll_link_err" >&2
    exit 1
  }
  assert_eq "$(cat -- "$roll_link_target")" "sentinel" "symlink marker was followed"
  if [[ -e "$(dest_path "$roll_link" /usr/local/libexec/llama-bored/kraken-lcd)" \
    || -e "$(dest_path "$roll_link" /etc/sysusers.d/llama-bored.conf)" ]]; then
    echo "install self-test: symlink marker refusal still wrote install files" >&2
    exit 1
  fi
  rm -rf -- "$roll_link_stage"

  # A failed save_rollback_marker says so after the printed rollback.
  local roll_m roll_m_stage roll_m_err roll_m_status roll_m_marker
  roll_m="$tmp/roll-m"
  snapshot_tree "$roll_m" "$tmp/snap-m"
  : >"$INSTALL_LOG"
  roll_m_stage="$(freeze_staging "$repo" "$roll_m")"
  INSTALL_FAIL_AT=daemon-reload
  INSTALL_FAIL_MARKER=1
  set +e
  roll_m_err="$(apply_from_staging "$roll_m_stage" "$roll_m" "$sys" "$head" 2>&1 >/dev/null)"
  roll_m_status=$?
  set -e
  unset INSTALL_FAIL_AT INSTALL_FAIL_MARKER
  if [[ "$roll_m_status" -eq 0 ]]; then
    echo "install self-test: marker-failure run was accepted" >&2
    exit 1
  fi
  roll_m_marker="$(dest_path "$roll_m" /usr/local/libexec/llama-bored/ROLLBACK_PENDING)"
  [[ ! -e "$roll_m_marker" && ! -L "$roll_m_marker" ]] || {
    echo "install self-test: injected marker failure still wrote ROLLBACK_PENDING" >&2
    exit 1
  }
  local roll_m_last
  roll_m_last="$(printf '%s\n' "$roll_m_err" | tail -n 1)"
  assert_eq "$roll_m_last" \
    "install.sh: ROLLBACK_PENDING was not written; run the rollback above before any retry" \
    "a failed save_rollback_marker was not reported after the rollback"
  execute_printed_rollback "$roll_m_err" "$roll_m"
  assert_same_tree "$roll_m" "$tmp/snap-m" "marker-failure rollback"
  rm -rf -- "$roll_m_stage"

  # save_rollback_marker itself refuses a symlinked marker or marker dir
  # (the EXIT hook path, after refuse_rollback_pending has already passed).
  local roll_s roll_s_target roll_s_dir roll_s_err
  roll_s="$tmp/roll-s"
  roll_s_target="$tmp/roll-s-target"
  roll_s_dir="$(dest_path "$roll_s" /usr/local/libexec/llama-bored)"
  printf 'sentinel\n' >"$roll_s_target"
  mkdir -p "$roll_s_dir"
  ln -s "$roll_s_target" "$roll_s_dir/ROLLBACK_PENDING"
  set +e
  roll_s_err="$(
    INSTALL_DEST_ROOT="$roll_s"
    ROLLBACK_KIND=(file)
    save_rollback_marker "ROLLBACK-TEXT" 2>&1
  )"
  roll_m_status=$?
  set -e
  [[ "$roll_m_status" -ne 0 \
    && "$roll_s_err" == *"refusing to follow a symlink at $roll_s_dir/ROLLBACK_PENDING"* ]] || {
    echo "install self-test: save_rollback_marker followed a symlinked marker" >&2
    exit 1
  }
  assert_eq "$(cat -- "$roll_s_target")" "sentinel" "save_rollback_marker wrote through a symlink"
  rm -f -- "$roll_s_dir/ROLLBACK_PENDING"
  mkdir -p "$tmp/roll-s-realdir"
  rmdir -- "$roll_s_dir"
  ln -s "$tmp/roll-s-realdir" "$roll_s_dir"
  set +e
  roll_s_err="$(
    INSTALL_DEST_ROOT="$roll_s"
    ROLLBACK_KIND=(file)
    save_rollback_marker "ROLLBACK-TEXT" 2>&1
  )"
  roll_m_status=$?
  set -e
  [[ "$roll_m_status" -ne 0 && "$roll_s_err" == *"refusing to follow a symlink"* ]] || {
    echo "install self-test: save_rollback_marker followed a symlinked marker dir" >&2
    exit 1
  }
  if [[ -n "$(find "$tmp/roll-s-realdir" -mindepth 1 -print -quit)" ]]; then
    echo "install self-test: save_rollback_marker wrote into a symlinked dir" >&2
    exit 1
  fi

  # (f) INSTALLED_SHA from an older LCD-writer-only install, and no llama-watch binary or unit yet.
  local roll_f roll_f_stage roll_f_out old_sha
  old_sha="dddddddddddddddddddddddddddddddddddddddd"
  roll_f="$tmp/roll-f"
  mkdir -p "$(dest_path "$roll_f" /usr/local/libexec/llama-bored)"
  printf 'old-writer\n' >"$(dest_path "$roll_f" /usr/local/libexec/llama-bored/kraken-lcd)"
  printf '%s\n' "$old_sha" >"$(dest_path "$roll_f" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  snapshot_tree "$roll_f" "$tmp/snap-f"
  : >"$INSTALL_LOG"
  roll_f_stage="$(freeze_staging "$repo" "$roll_f")"
  roll_f_out="$(apply_from_staging "$roll_f_stage" "$roll_f" "$sys" "$head")"
  [[ "$roll_f_out" == *"previous INSTALLED_SHA (${old_sha})"* ]] || {
    echo "install self-test: writer-only rollback did not name the previous INSTALLED_SHA" >&2
    exit 1
  }
  assert_disable_present "$roll_f_out" 1
  assert_rollback_order "$roll_f_out"
  assert_eq "$(cat -- "$(dest_path "$roll_f" /usr/local/libexec/llama-bored/INSTALLED_SHA)")" \
    "$head" "writer-only upgrade did not record the new INSTALLED_SHA"
  execute_printed_rollback "$roll_f_out" "$roll_f"
  assert_same_tree "$roll_f" "$tmp/snap-f" "writer-only rollback"
  assert_eq "$(cat -- "$(dest_path "$roll_f" /usr/local/libexec/llama-bored/kraken-lcd)")" \
    "old-writer" "writer-only rollback did not restore the writer"
  assert_no_new "$roll_f"
  rm -rf -- "$roll_f_stage"

  # (g) Legacy migration: an older INSTALLED_SHA, legacy config, the operator's .new and
  # watch.toml, and no llama-watch yet.
  local roll_g roll_g_stage roll_g_out old_sha_g
  old_sha_g="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
  roll_g="$tmp/roll-g"
  mkdir -p "$(dest_path "$roll_g" /etc/llama-bored)" \
    "$(dest_path "$roll_g" /usr/local/libexec/llama-bored)"
  cp -- "$ROOT/fixtures/config/legacy-config.toml" \
    "$(dest_path "$roll_g" /etc/llama-bored/config.toml)"
  cp -- "$ROOT/packaging/config.example.toml" \
    "$(dest_path "$roll_g" /etc/llama-bored/config.toml.new)"
  printf 'operator watch\n' >"$(dest_path "$roll_g" /etc/llama-bored/watch.toml)"
  printf '%s\n' "$old_sha_g" >"$(dest_path "$roll_g" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  snapshot_tree "$roll_g" "$tmp/snap-g"
  : >"$INSTALL_LOG"
  roll_g_stage="$(freeze_staging "$repo" "$roll_g")"
  roll_g_out="$(apply_from_staging "$roll_g_stage" "$roll_g" "$sys" "$head")"
  assert_disable_first "$roll_g_out"
  assert_config_restore_order "$roll_g_out"
  assert_udev_reload "$roll_g_out"
  assert_eq "$(cat -- "$(dest_path "$roll_g" /etc/llama-bored/watch.toml)")" \
    "operator watch" "migration overwrote watch.toml"
  if [[ -e "$(dest_path "$roll_g" /etc/llama-bored/config.toml.new)" ]]; then
    echo "install self-test: migration left config.toml.new in place" >&2
    exit 1
  fi
  execute_printed_rollback "$roll_g_out" "$roll_g"
  assert_same_tree "$roll_g" "$tmp/snap-g" "legacy migration rollback"
  cmp -s -- "$ROOT/packaging/config.example.toml" \
    "$(dest_path "$roll_g" /etc/llama-bored/config.toml.new)" || {
    echo "install self-test: rollback did not keep the migrated config bytes" >&2
    exit 1
  }
  assert_no_new "$roll_g"
  rm -rf -- "$roll_g_stage"

  # Automatic anchor on the dry-run fixture. stdin is closed: a leftover
  # prompt would fail the success case.
  local anchor_repo anchor_ok anchor_out anchor_status anchor_err
  local anchor_first anchor_head anchor_dirty anchor_prov anchor_ahead
  local prov_file dash_hash watch_hash view_hash metrics_hash anchor_email
  anchor_repo="$repo"
  # Built at runtime so a literal-email scan of the tree stays empty.
  anchor_email="test$(printf '\x40')example.invalid"
  printf 'fake-binary\n' >"$anchor_repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$anchor_repo/target/release/llama-watch"
  printf 'fake-view\n' >"$anchor_repo/target/release/llama-view"
  printf 'fake-light\n' >"$anchor_repo/target/release/llama-light"
  printf 'fake-metrics\n' >"$anchor_repo/target/release/llama-metrics"
  printf 'lock\n' >"$anchor_repo/Cargo.lock"
  printf 'toolchain\n' >"$anchor_repo/rust-toolchain.toml"
  printf 'target/\n' >"$anchor_repo/.gitignore"
  # Work on a side branch; main is the release anchor. The literal "main"
  # here is deliberate: a changed RELEASE_BRANCH fails these checks.
  git -C "$anchor_repo" init -b work >/dev/null
  git -C "$anchor_repo" add -- .gitignore Cargo.lock rust-toolchain.toml packaging
  git -C "$anchor_repo" -c user.name='install self-test' -c user.email="$anchor_email" \
    -c core.hooksPath="$(mktemp -d)" commit -m "base" >/dev/null
  printf 'anchor-marker\n' >"$anchor_repo/ANCHOR_MARKER"
  git -C "$anchor_repo" add -- ANCHOR_MARKER
  git -C "$anchor_repo" -c user.name='install self-test' -c user.email="$anchor_email" \
    -c core.hooksPath="$(mktemp -d)" commit -m "anchor-subject" >/dev/null
  git -C "$anchor_repo" branch main HEAD
  write_real_provenance "$anchor_repo"
  anchor_first="$(git -C "$anchor_repo" rev-list --max-parents=0 HEAD)"
  anchor_head="$(git -C "$anchor_repo" rev-parse HEAD)"

  anchor_ok="$tmp/anchor-ok"
  mkdir -p "$(dest_path "$anchor_ok" /usr/local/libexec/llama-bored)"
  printf '%s\n' "$anchor_first" \
    >"$(dest_path "$anchor_ok" /usr/local/libexec/llama-bored/INSTALLED_SHA)"
  : >"$INSTALL_LOG"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  INSTALL_OWNER_MODE=direct
  anchor_err="$tmp/anchor-ok.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_ok" "$sys" </dev/null >"$tmp/anchor-ok.out" 2>"$anchor_err"
  anchor_status=$?
  set -e
  anchor_out="$(cat -- "$tmp/anchor-ok.out")"
  if [[ "$anchor_status" -ne 0 ]]; then
    echo "install self-test: non-interactive dry-run failed" >&2
    cat -- "$anchor_err" >&2
    printf '%s\n' "$anchor_out" >&2
    exit 1
  fi
  dash_hash="$(sha256sum -- "$anchor_repo/target/release/kraken-lcd" | awk '{ print $1 }')"
  watch_hash="$(sha256sum -- "$anchor_repo/target/release/llama-watch" | awk '{ print $1 }')"
  view_hash="$(sha256sum -- "$anchor_repo/target/release/llama-view" | awk '{ print $1 }')"
  metrics_hash="$(sha256sum -- "$anchor_repo/target/release/llama-metrics" | awk '{ print $1 }')"
  [[ "$anchor_out" == *"----- anchor -----"* \
    && "$anchor_out" == *"HEAD: $anchor_head"* \
    && "$anchor_out" == *"subject: anchor-subject"* \
    && "$anchor_out" == *"$dash_hash  kraken-lcd"* \
    && "$anchor_out" == *"$watch_hash  llama-watch"* \
    && "$anchor_out" == *"$view_hash  llama-view"* \
    && "$anchor_out" == *"$metrics_hash  llama-metrics"* \
    && "$anchor_out" == *"----- diff stat since $anchor_first -----"* \
    && "$anchor_out" == *"ANCHOR_MARKER"* ]] || {
    echo "install self-test: dry-run summary missing HEAD, subject, hashes, or diff stat" >&2
    printf '%s\n' "$anchor_out" >&2
    exit 1
  }
  assert_eq "$(cat -- "$(dest_path "$anchor_ok" /usr/local/libexec/llama-bored/kraken-lcd)")" \
    "fake-binary" "dry-run installed the staged binary"
  if log_mentions_writer_lifecycle; then
    echo "install self-test: dry-run log starts, restarts, or enables kraken-lcd" >&2
    exit 1
  fi

  # No branch: the explicit ref check refuses before staging.
  local anchor_missing
  anchor_missing="$tmp/anchor-missing"
  mkdir -p -- "$anchor_missing"
  snapshot_tree "$anchor_missing" "$tmp/snap-anchor-missing"
  git -C "$anchor_repo" branch -D main >/dev/null
  : >"$INSTALL_LOG"
  INSTALL_DID_STAGE=0
  anchor_err="$tmp/anchor-missing.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_missing" "$sys" </dev/null >/dev/null 2>"$anchor_err"
  anchor_status=$?
  set -e
  if [[ "$anchor_status" -eq 0 ]]; then
    echo "install self-test: missing main branch was installed" >&2
    exit 1
  fi
  if ! grep -F -q "refs/heads/main is not a commit in this repository" "$anchor_err"; then
    echo "install self-test: missing branch did not fail the explicit ref check" >&2
    cat -- "$anchor_err" >&2
    exit 1
  fi
  if [[ "${INSTALL_DID_STAGE:-0}" != 0 ]]; then
    echo "install self-test: missing branch was refused after staging" >&2
    exit 1
  fi
  assert_same_tree "$anchor_missing" "$tmp/snap-anchor-missing" "missing branch wrote install files"
  git -C "$anchor_repo" branch main HEAD

  # Change a reviewed unit after the review. verify_staged_matches_disk
  # refuses before any destination write.
  local anchor_tamper
  anchor_tamper="$tmp/anchor-tamper"
  mkdir -p -- "$anchor_tamper"
  snapshot_tree "$anchor_tamper" "$tmp/snap-anchor-tamper"
  INSTALL_TAMPER_LIVE=packaging/kraken-lcd.service
  : >"$INSTALL_LOG"
  anchor_err="$tmp/anchor-tamper.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_tamper" "$sys" </dev/null >/dev/null 2>"$anchor_err"
  anchor_status=$?
  set -e
  unset INSTALL_TAMPER_LIVE
  cp -- "$ROOT/packaging/kraken-lcd.service" "$anchor_repo/packaging/kraken-lcd.service"
  if [[ "$anchor_status" -eq 0 ]]; then
    echo "install self-test: staged bytes were installed after the reviewed file changed" >&2
    exit 1
  fi
  if ! grep -F -q "staged kraken-lcd.service does not match the reviewed packaging/kraken-lcd.service" "$anchor_err"; then
    echo "install self-test: changed unit was not caught by verify_staged_matches_disk" >&2
    cat -- "$anchor_err" >&2
    exit 1
  fi
  assert_same_tree "$anchor_tamper" "$tmp/snap-anchor-tamper" "changed unit wrote the destination"

  anchor_dirty="$tmp/anchor-dirty"
  mkdir -p -- "$anchor_dirty"
  snapshot_tree "$anchor_dirty" "$tmp/snap-anchor-dirty"
  mkdir -p -- "$anchor_repo/src"
  printf 'x\n' >"$anchor_repo/src/dirty"
  : >"$INSTALL_LOG"
  anchor_err="$tmp/anchor-dirty.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_dirty" "$sys" </dev/null >/dev/null 2>"$anchor_err"
  anchor_status=$?
  set -e
  if [[ "$anchor_status" -eq 0 ]]; then
    echo "install self-test: dirty tree was installed" >&2
    exit 1
  fi
  if ! grep -F -q "refusing a dirty tree" "$anchor_err"; then
    echo "install self-test: dirty tree did not fail the clean-tree check" >&2
    cat -- "$anchor_err" >&2
    exit 1
  fi
  assert_same_tree "$anchor_dirty" "$tmp/snap-anchor-dirty" "dirty tree wrote install files"
  rm -f -- "$anchor_repo/src/dirty"

  anchor_prov="$tmp/anchor-prov"
  mkdir -p -- "$anchor_prov"
  snapshot_tree "$anchor_prov" "$tmp/snap-anchor-prov"
  prov_file="$anchor_repo/target/check-provenance.txt"
  awk 'NR==5 { print "0123456789abcdef0123456789abcdef01234567"; next } { print }' \
    "$prov_file" >"$prov_file.tmp"
  mv -f -- "$prov_file.tmp" "$prov_file"
  : >"$INSTALL_LOG"
  anchor_err="$tmp/anchor-prov.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_prov" "$sys" </dev/null >/dev/null 2>"$anchor_err"
  anchor_status=$?
  set -e
  if [[ "$anchor_status" -eq 0 ]]; then
    echo "install self-test: provenance/HEAD mismatch was installed" >&2
    exit 1
  fi
  if ! grep -F -q "HEAD is" "$anchor_err"; then
    echo "install self-test: provenance/HEAD mismatch did not fail the provenance check" >&2
    cat -- "$anchor_err" >&2
    exit 1
  fi
  assert_same_tree "$anchor_prov" "$tmp/snap-anchor-prov" "provenance mismatch wrote install files"
  write_real_provenance "$anchor_repo"

  anchor_ahead="$tmp/anchor-ahead"
  mkdir -p -- "$anchor_ahead"
  snapshot_tree "$anchor_ahead" "$tmp/snap-anchor-ahead"
  printf 'ahead\n' >"$anchor_repo/AHEAD"
  git -C "$anchor_repo" add -- AHEAD
  git -C "$anchor_repo" -c user.name='install self-test' -c user.email="$anchor_email" \
    -c core.hooksPath="$(mktemp -d)" commit -m "not-landed" >/dev/null
  write_real_provenance "$anchor_repo"
  : >"$INSTALL_LOG"
  INSTALL_DID_STAGE=0
  anchor_err="$tmp/anchor-ahead.err"
  set +e
  anchored_install "$anchor_repo" "$anchor_ahead" "$sys" </dev/null >/dev/null 2>"$anchor_err"
  anchor_status=$?
  set -e
  if [[ "$anchor_status" -eq 0 ]]; then
    echo "install self-test: HEAD not contained in main was installed" >&2
    exit 1
  fi
  if ! grep -F -q "not contained in main" "$anchor_err"; then
    echo "install self-test: unlanded HEAD did not fail the ancestor check" >&2
    cat -- "$anchor_err" >&2
    exit 1
  fi
  if [[ "${INSTALL_DID_STAGE:-0}" != 0 ]]; then
    echo "install self-test: unlanded HEAD was refused after staging" >&2
    exit 1
  fi
  assert_same_tree "$anchor_ahead" "$tmp/snap-anchor-ahead" "unlanded HEAD wrote install files"

  # llama-light. Installed and hashed with the rest; enabled only with
  # --enable-light (LIGHT_ENABLE=1); never started; its udev rule, unit and
  # config are in the rollback record; the Aura node is checked when present.
  local light light_stage light_out light_section light_log
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  printf 'fake-view\n' >"$repo/target/release/llama-view"
  printf 'fake-light\n' >"$repo/target/release/llama-light"
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"
  write_fixture_provenance "$repo" "$head"
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  unset INSTALL_FAIL_AT INSTALL_FAKE_AURA_PIN INSTALL_FAKE_AURA_STAT INSTALL_FAKE_KBD_PIN INSTALL_FAKE_KBD_STAT

  light="$tmp/light-default"
  mkdir -p -- "$light"
  snapshot_tree "$light" "$tmp/snap-light-default"
  : >"$INSTALL_LOG"
  light_stage="$(freeze_staging "$repo" "$light")"
  for f in llama-light llama-light.service 94-llama-light-hidraw.rules light.toml; do
    [[ -f "$light_stage/$f" ]] || {
      echo "install self-test: $f was not staged" >&2
      exit 1
    }
    [[ "$(show_staging_hashes "$light_stage")" == *"$(sha256sum -- "$light_stage/$f" | awk '{ print $1 }')  $light_stage/$f"* ]] || {
      echo "install self-test: staged hash of $f was not shown" >&2
      exit 1
    }
  done
  run_expect_ok verify_staged_binary "$light_stage"
  printf 'tampered-light\n' >"$light_stage/llama-light"
  run_expect_fail verify_staged_binary "$light_stage"
  printf 'fake-light\n' >"$light_stage/llama-light"
  run_expect_ok verify_staged_matches_disk "$light_stage" "$repo" "$light"
  printf '\n# tampered\n' >>"$repo/packaging/94-llama-light-hidraw.rules"
  run_expect_fail verify_staged_matches_disk "$light_stage" "$repo" "$light"
  cp -- "$ROOT/packaging/94-llama-light-hidraw.rules" "$repo/packaging/94-llama-light-hidraw.rules"
  printf '\n# tampered\n' >>"$repo/packaging/llama-light.service"
  run_expect_fail verify_staged_matches_disk "$light_stage" "$repo" "$light"
  cp -- "$ROOT/packaging/llama-light.service" "$repo/packaging/llama-light.service"
  printf '\n# tampered\n' >>"$repo/packaging/light.example.toml"
  run_expect_fail verify_staged_matches_disk "$light_stage" "$repo" "$light"
  cp -- "$ROOT/packaging/light.example.toml" "$repo/packaging/light.example.toml"

  LIGHT_ENABLE=0
  light_out="$(apply_from_staging "$light_stage" "$light" "$sys" "$head")"
  assert_eq "$(cat -- "$(dest_path "$light" /usr/local/libexec/llama-bored/llama-light)")" \
    "fake-light" "installed llama-light"
  assert_eq "$(stat -c '%a' -- "$(dest_path "$light" /usr/local/libexec/llama-bored/llama-light)")" \
    "755" "llama-light mode"
  cmp -s -- "$ROOT/packaging/llama-light.service" \
    "$(dest_path "$light" /etc/systemd/system/llama-light.service)" || {
    echo "install self-test: llama-light.service was not installed verbatim" >&2
    exit 1
  }
  cmp -s -- "$ROOT/packaging/94-llama-light-hidraw.rules" \
    "$(dest_path "$light" /etc/udev/rules.d/94-llama-light-hidraw.rules)" || {
    echo "install self-test: 94-llama-light-hidraw.rules was not installed verbatim" >&2
    exit 1
  }
  cmp -s -- "$ROOT/packaging/light.example.toml" \
    "$(dest_path "$light" /etc/llama-bored/light.toml)" || {
    echo "install self-test: absent light.toml was not installed from the example" >&2
    exit 1
  }
  log_has 'udevadm trigger --action=change --attr-match=idVendor=0b05 --attr-match=idProduct=18f3' || {
    echo "install self-test: the Aura udev trigger did not run" >&2
    exit 1
  }
  log_has 'udevadm trigger --action=change --attr-match=idVendor=1b1c --attr-match=idProduct=1b48' || {
    echo "install self-test: the keyboard udev trigger did not run" >&2
    exit 1
  }
  light_log="$(cat -- "$INSTALL_LOG")"
  if [[ "$light_log" =~ (enable|start|restart)[^$'\n']*llama-light ]]; then
    echo "install self-test: llama-light was enabled or started without --enable-light" >&2
    printf '%s\n' "$light_log" >&2
    exit 1
  fi
  [[ "$light_out" == *"llama-light) is installed, not enabled"* \
    && "$light_out" == *"llama-light check --config /etc/llama-bored/light.toml"* \
    && "$light_out" == *"systemctl start llama-light"* ]] || {
    echo "install self-test: next steps do not explain how to try llama-light" >&2
    printf '%s\n' "$light_out" >&2
    exit 1
  }
  light_section="$(rollback_section "$light_out")"
  [[ "$light_section" == *"rm -f -- /usr/local/libexec/llama-bored/llama-light"* \
    && "$light_section" == *"rm -f -- /etc/systemd/system/llama-light.service"* \
    && "$light_section" == *"rm -f -- /etc/udev/rules.d/94-llama-light-hidraw.rules"* \
    && "$light_section" == *"rm -f -- /etc/llama-bored/light.toml"* \
    && "$light_section" == *"--attr-match=idVendor=0b05 --attr-match=idProduct=18f3"* \
    && "$light_section" == *"--attr-match=idVendor=1b1c --attr-match=idProduct=1b48"* \
    && "$light_section" == *"udevadm trigger --action=change --subsystem-match=hidraw"* \
    && "$light_section" != *"disable --now llama-light"* ]] || {
    echo "install self-test: rollback record does not cover the llama-light files" >&2
    printf '%s\n' "$light_section" >&2
    exit 1
  }
  assert_rollback_order "$light_out"
  execute_printed_rollback "$light_out" "$light"
  assert_same_tree "$light" "$tmp/snap-light-default" "llama-light default rollback"
  rm -rf -- "$light_stage"

  # --enable-light: enabled, not started; rollback disables it.
  light="$tmp/light-enabled"
  mkdir -p "$(dest_path "$light" /etc/llama-bored)"
  printf '[aura]\nbrightness_max = 40\n' >"$(dest_path "$light" /etc/llama-bored/light.toml)"
  snapshot_tree "$light" "$tmp/snap-light-enabled"
  : >"$INSTALL_LOG"
  light_stage="$(freeze_staging "$repo" "$light")"
  assert_eq "$(tr -d '[:space:]' <"$light_stage/light.mode")" "keep" "existing light.toml kept"
  LIGHT_ENABLE=1
  light_out="$(apply_from_staging "$light_stage" "$light" "$sys" "$head")"
  LIGHT_ENABLE=0
  assert_eq "$(cat -- "$(dest_path "$light" /etc/llama-bored/light.toml)")" \
    "$(printf '[aura]\nbrightness_max = 40\n')" "operator light.toml preserved"
  log_has 'systemctl enable llama-light.service' || {
    echo "install self-test: --enable-light did not enable llama-light" >&2
    exit 1
  }
  if log_has 'start llama-light' || log_has 'enable --now llama-light'; then
    echo "install self-test: llama-light was started by the installer" >&2
    exit 1
  fi
  if [[ "$(log_line_number 'systemctl daemon-reload')" -ge "$(log_line_number 'systemctl enable llama-light.service')" ]]; then
    echo "install self-test: llama-light was enabled before daemon-reload" >&2
    exit 1
  fi
  [[ "$light_out" == *"llama-light) is enabled for boot but not started"* ]] || {
    echo "install self-test: next steps do not say llama-light is enabled" >&2
    exit 1
  }
  light_section="$(rollback_section "$light_out")"
  [[ "$light_section" == *"systemctl disable --now llama-light.service"* \
    && "$light_section" != *"light.toml"* ]] || {
    echo "install self-test: --enable-light rollback does not disable llama-light, or touches the kept light.toml" >&2
    printf '%s\n' "$light_section" >&2
    exit 1
  }
  assert_rollback_order "$light_out"
  execute_printed_rollback "$light_out" "$light"
  assert_same_tree "$light" "$tmp/snap-light-enabled" "llama-light enabled rollback"
  log_has 'systemctl disable --now llama-light.service'
  rm -rf -- "$light_stage"

  # An attached Aura controller: its node must end 0660 llama-light, with
  # no user ACL, and the pin must name it. Otherwise nothing is swapped.
  local aura_sys aura_case aura_dest aura_err aura_status
  aura_sys="$tmp/sys-aura"
  cp -a -- "$sys" "$aura_sys"
  mkdir -p "$aura_sys/class/hidraw/hidraw2/device"
  printf 'DRIVER=hid-generic\nHID_ID=0003:00000B05:000018F3\n' \
    >"$aura_sys/class/hidraw/hidraw2/device/uevent"
  assert_eq "$(resolve_aura_hidraw "$aura_sys")" "/dev/hidraw2" "Aura hidraw from sysfs"
  assert_eq "$(resolve_aura_hidraw "$sys")" "" "no Aura in sysfs"

  aura_dest="$tmp/aura-ok"
  : >"$INSTALL_LOG"
  light_stage="$(freeze_staging "$repo" "$aura_dest")"
  INSTALL_FAKE_AURA_STAT='660 llama-light' INSTALL_FAKE_AURA_PIN=/dev/hidraw2 \
    run_expect_ok apply_from_staging "$light_stage" "$aura_dest" "$aura_sys" "$head"
  log_has 'setfacl -b -- /dev/hidraw2' || {
    echo "install self-test: the Aura node's ACL was not cleared" >&2
    exit 1
  }
  rm -rf -- "$light_stage"

  for aura_case in 'stat:666 root' 'stat:660 root' 'pin:' 'pin:/dev/hidraw7' 'acl:'; do
    aura_dest="$tmp/aura-bad-${aura_case//[^a-z0-9]/-}"
    : >"$INSTALL_LOG"
    light_stage="$(freeze_staging "$repo" "$aura_dest")"
    set +e
    case "$aura_case" in
      stat:*)
        aura_err="$(INSTALL_FAKE_AURA_STAT="${aura_case#stat:}" INSTALL_FAKE_AURA_PIN=/dev/hidraw2 \
          apply_from_staging "$light_stage" "$aura_dest" "$aura_sys" "$head" 2>&1 >/dev/null)"
        ;;
      pin:*)
        aura_err="$(INSTALL_FAKE_AURA_STAT='660 llama-light' INSTALL_FAKE_AURA_PIN="${aura_case#pin:}" \
          apply_from_staging "$light_stage" "$aura_dest" "$aura_sys" "$head" 2>&1 >/dev/null)"
        ;;
      acl:*)
        aura_err="$(INSTALL_FAKE_AURA_STAT='660 llama-light' INSTALL_FAKE_AURA_PIN=/dev/hidraw2 \
          INSTALL_FAKE_GETFACL=$'user::rw-\nuser:someone:rw-\n' \
          apply_from_staging "$light_stage" "$aura_dest" "$aura_sys" "$head" 2>&1 >/dev/null)"
        ;;
    esac
    aura_status=$?
    set -e
    if [[ "$aura_status" -eq 0 ]]; then
      echo "install self-test: Aura case '$aura_case' was accepted" >&2
      exit 1
    fi
    [[ "$aura_err" == *"how to roll back"* ]] || {
      echo "install self-test: Aura case '$aura_case' did not print the rollback" >&2
      printf '%s\n' "$aura_err" >&2
      exit 1
    }
    if [[ -e "$(dest_path "$aura_dest" /usr/local/libexec/llama-bored/llama-light)" \
      || -e "$(dest_path "$aura_dest" /usr/local/libexec/llama-bored/kraken-lcd)" ]] \
      || log_has 'systemctl daemon-reload'; then
      echo "install self-test: Aura case '$aura_case' still swapped a binary or reloaded units" >&2
      exit 1
    fi
    assert_no_new "$aura_dest"
    rm -rf -- "$light_stage"
  done
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'

  # An attached keyboard. Only its interface-01 hidraw node (lighting)
  # is checked; the typing interface's node is not touched. It must end
  # 0660 llama-light, with no user ACL, and the pin must name it.
  local kbd_sys kbd_case kbd_dest kbd_err kbd_status kbd_usb
  kbd_sys="$tmp/sys-kbd"
  cp -a -- "$sys" "$kbd_sys"
  kbd_usb="$kbd_sys/devices/pci0000:00/usb1/1-4"
  mkdir -p "$kbd_usb/1-4:1.0/0003:1B1C:1B48.0003" "$kbd_usb/1-4:1.1/0003:1B1C:1B48.0004"
  printf '00\n' >"$kbd_usb/1-4:1.0/bInterfaceNumber"
  printf '01\n' >"$kbd_usb/1-4:1.1/bInterfaceNumber"
  printf 'DRIVER=hid-generic\nHID_ID=0003:00001B1C:00001B48\n' \
    >"$kbd_usb/1-4:1.0/0003:1B1C:1B48.0003/uevent"
  printf 'DRIVER=hid-generic\nHID_ID=0003:00001B1C:00001B48\n' \
    >"$kbd_usb/1-4:1.1/0003:1B1C:1B48.0004/uevent"
  mkdir -p "$kbd_sys/class/hidraw/hidraw3" "$kbd_sys/class/hidraw/hidraw4"
  ln -s -- "$kbd_usb/1-4:1.0/0003:1B1C:1B48.0003" "$kbd_sys/class/hidraw/hidraw3/device"
  ln -s -- "$kbd_usb/1-4:1.1/0003:1B1C:1B48.0004" "$kbd_sys/class/hidraw/hidraw4/device"
  assert_eq "$(resolve_keyboard_hidraw "$kbd_sys")" "/dev/hidraw4" "keyboard lighting hidraw from sysfs"
  assert_eq "$(resolve_keyboard_hidraw "$sys")" "" "no keyboard in sysfs"

  kbd_dest="$tmp/kbd-ok"
  : >"$INSTALL_LOG"
  light_stage="$(freeze_staging "$repo" "$kbd_dest")"
  INSTALL_FAKE_KBD_STAT='660 llama-light' INSTALL_FAKE_KBD_PIN=/dev/hidraw4 \
    run_expect_ok apply_from_staging "$light_stage" "$kbd_dest" "$kbd_sys" "$head"
  log_has 'setfacl -b -- /dev/hidraw4' || {
    echo "install self-test: the keyboard lighting node's ACL was not cleared" >&2
    exit 1
  }
  if log_has 'setfacl -b -- /dev/hidraw3'; then
    echo "install self-test: the keyboard's typing interface node was touched" >&2
    exit 1
  fi
  rm -rf -- "$light_stage"

  for kbd_case in 'stat:666 root' 'stat:660 root' 'pin:' 'pin:/dev/hidraw3' 'acl:'; do
    kbd_dest="$tmp/kbd-bad-${kbd_case//[^a-z0-9]/-}"
    : >"$INSTALL_LOG"
    light_stage="$(freeze_staging "$repo" "$kbd_dest")"
    set +e
    case "$kbd_case" in
      stat:*)
        kbd_err="$(INSTALL_FAKE_KBD_STAT="${kbd_case#stat:}" INSTALL_FAKE_KBD_PIN=/dev/hidraw4 \
          apply_from_staging "$light_stage" "$kbd_dest" "$kbd_sys" "$head" 2>&1 >/dev/null)"
        ;;
      pin:*)
        kbd_err="$(INSTALL_FAKE_KBD_STAT='660 llama-light' INSTALL_FAKE_KBD_PIN="${kbd_case#pin:}" \
          apply_from_staging "$light_stage" "$kbd_dest" "$kbd_sys" "$head" 2>&1 >/dev/null)"
        ;;
      acl:*)
        kbd_err="$(INSTALL_FAKE_KBD_STAT='660 llama-light' INSTALL_FAKE_KBD_PIN=/dev/hidraw4 \
          INSTALL_FAKE_GETFACL=$'user::rw-\nuser:someone:rw-\n' \
          apply_from_staging "$light_stage" "$kbd_dest" "$kbd_sys" "$head" 2>&1 >/dev/null)"
        ;;
    esac
    kbd_status=$?
    set -e
    if [[ "$kbd_status" -eq 0 ]]; then
      echo "install self-test: keyboard case '$kbd_case' was accepted" >&2
      exit 1
    fi
    [[ "$kbd_err" == *"how to roll back"* ]] || {
      echo "install self-test: keyboard case '$kbd_case' did not print the rollback" >&2
      printf '%s\n' "$kbd_err" >&2
      exit 1
    }
    if [[ -e "$(dest_path "$kbd_dest" /usr/local/libexec/llama-bored/llama-light)" \
      || -e "$(dest_path "$kbd_dest" /usr/local/libexec/llama-bored/kraken-lcd)" ]] \
      || log_has 'systemctl daemon-reload'; then
      echo "install self-test: keyboard case '$kbd_case' still swapped a binary or reloaded units" >&2
      exit 1
    fi
    assert_no_new "$kbd_dest"
    rm -rf -- "$light_stage"
  done
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  unset INSTALL_FAKE_KBD_PIN INSTALL_FAKE_KBD_STAT

  # llama-metrics. Staged and hashed with the rest; its unit and
  # config land verbatim; it is never enabled or started, and no firewall
  # command runs; the rollback removes what this run created and disables
  # the unit first (the operator may have enabled it since).
  local metrics metrics_stage metrics_out metrics_section metrics_log f
  printf 'fake-binary\n' >"$repo/target/release/kraken-lcd"
  printf 'fake-watch\n' >"$repo/target/release/llama-watch"
  printf 'fake-view\n' >"$repo/target/release/llama-view"
  printf 'fake-light\n' >"$repo/target/release/llama-light"
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"
  write_fixture_provenance "$repo" "$head"
  unset INSTALL_FAKE_AURA_PIN INSTALL_FAKE_AURA_STAT
  LIGHT_ENABLE=0
  INSTALL_FAKE_STAT='660 kraken-lcd'
  INSTALL_FAKE_GETFACL=$'user::rw-\ngroup::rw-\n'
  unset INSTALL_FAIL_AT

  metrics="$tmp/metrics-fresh"
  mkdir -p -- "$metrics"
  snapshot_tree "$metrics" "$tmp/snap-metrics-fresh"
  : >"$INSTALL_LOG"
  metrics_stage="$(freeze_staging "$repo" "$metrics")"
  for f in llama-metrics llama-metrics.service metrics.toml; do
    [[ -f "$metrics_stage/$f" ]] || {
      echo "install self-test: $f was not staged" >&2
      exit 1
    }
    [[ "$(show_staging_hashes "$metrics_stage")" == *"$(sha256sum -- "$metrics_stage/$f" | awk '{ print $1 }')  $metrics_stage/$f"* ]] || {
      echo "install self-test: staged hash of $f was not shown" >&2
      exit 1
    }
  done
  assert_eq "$(tr -d '[:space:]' <"$metrics_stage/metrics.mode")" "install" "absent metrics.toml is installed"
  run_expect_ok verify_staged_binary "$metrics_stage"
  printf 'tampered-metrics\n' >"$metrics_stage/llama-metrics"
  run_expect_fail verify_staged_binary "$metrics_stage"
  printf 'fake-metrics\n' >"$metrics_stage/llama-metrics"
  run_expect_ok verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  printf 'tampered\n' >"$repo/target/release/llama-metrics"
  run_expect_fail verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  printf 'fake-metrics\n' >"$repo/target/release/llama-metrics"
  printf '\n# tampered\n' >>"$repo/packaging/llama-metrics.service"
  run_expect_fail verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  cp -- "$ROOT/packaging/llama-metrics.service" "$repo/packaging/llama-metrics.service"
  printf '\n# tampered\n' >>"$repo/packaging/metrics.example.toml"
  run_expect_fail verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  cp -- "$ROOT/packaging/metrics.example.toml" "$repo/packaging/metrics.example.toml"
  run_expect_ok verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"

  metrics_out="$(apply_from_staging "$metrics_stage" "$metrics" "$sys" "$head")"
  assert_eq "$(cat -- "$(dest_path "$metrics" /usr/local/libexec/llama-bored/llama-metrics)")" \
    "fake-metrics" "installed llama-metrics"
  assert_eq "$(stat -c '%a' -- "$(dest_path "$metrics" /usr/local/libexec/llama-bored/llama-metrics)")" \
    "755" "llama-metrics mode"
  cmp -s -- "$ROOT/packaging/llama-metrics.service" \
    "$(dest_path "$metrics" /etc/systemd/system/llama-metrics.service)" || {
    echo "install self-test: llama-metrics.service was not installed verbatim" >&2
    exit 1
  }
  cmp -s -- "$ROOT/packaging/metrics.example.toml" \
    "$(dest_path "$metrics" /etc/llama-bored/metrics.toml)" || {
    echo "install self-test: absent metrics.toml was not installed from the example" >&2
    exit 1
  }
  assert_eq "$(stat -c '%a' -- "$(dest_path "$metrics" /etc/llama-bored/metrics.toml)")" \
    "644" "metrics.toml mode"
  metrics_log="$(cat -- "$INSTALL_LOG")"
  if [[ "$metrics_log" =~ (enable|start|restart|reload-or)[^$'\n']*llama-metrics ]]; then
    echo "install self-test: the installer enabled or started llama-metrics" >&2
    printf '%s\n' "$metrics_log" >&2
    exit 1
  fi
  if [[ "$metrics_log" == *firewall* || "$metrics_log" == *nft* || "$metrics_log" == *iptables* ]]; then
    echo "install self-test: the installer ran a firewall command" >&2
    printf '%s\n' "$metrics_log" >&2
    exit 1
  fi
  [[ "$metrics_out" == *"llama-metrics) is installed, not enabled"* \
    && "$metrics_out" == *"llama-metrics check --config /etc/llama-bored/metrics.toml"* \
    && "$metrics_out" == *"systemctl enable --now llama-metrics"* \
    && "$metrics_out" == *"--add-port=19477/tcp"* \
    && "$metrics_out" == *"IPAddressAllow="* ]] || {
    echo "install self-test: next steps do not explain how to enable llama-metrics" >&2
    printf '%s\n' "$metrics_out" >&2
    exit 1
  }
  metrics_section="$(rollback_section "$metrics_out")"
  [[ "$metrics_section" == *"systemctl disable --now llama-metrics.service"* \
    && "$metrics_section" == *"rm -f -- /usr/local/libexec/llama-bored/llama-metrics"* \
    && "$metrics_section" == *"rm -f -- /etc/systemd/system/llama-metrics.service"* \
    && "$metrics_section" == *"rm -f -- /etc/llama-bored/metrics.toml"* ]] || {
    echo "install self-test: rollback record does not cover the llama-metrics files" >&2
    printf '%s\n' "$metrics_section" >&2
    exit 1
  }
  local disable_at remove_at
  disable_at="$(grep -n -F -m 1 'systemctl disable --now llama-metrics.service' <<<"$metrics_section" | cut -d: -f1)"
  remove_at="$(grep -n -F -m 1 'rm -f -- /etc/systemd/system/llama-metrics.service' <<<"$metrics_section" | cut -d: -f1)"
  if [[ "$disable_at" -ge "$remove_at" ]]; then
    echo "install self-test: rollback removes the llama-metrics unit before disabling it" >&2
    exit 1
  fi
  assert_disable_first "$metrics_out"
  assert_rollback_order "$metrics_out"
  execute_printed_rollback "$metrics_out" "$metrics"
  assert_same_tree "$metrics" "$tmp/snap-metrics-fresh" "llama-metrics fresh rollback"
  rm -rf -- "$metrics_stage"

  # An existing metrics.toml is the operator's: kept, and never in the
  # rollback. A reinstall over an existing exporter does not disable it.
  metrics="$tmp/metrics-keep"
  mkdir -p "$(dest_path "$metrics" /etc/llama-bored)" \
    "$(dest_path "$metrics" /usr/local/libexec/llama-bored)" \
    "$(dest_path "$metrics" /etc/systemd/system)"
  printf 'listen = "0.0.0.0:19477"\nallow = ["127.0.0.1/32"]\n' \
    >"$(dest_path "$metrics" /etc/llama-bored/metrics.toml)"
  printf 'old-metrics\n' >"$(dest_path "$metrics" /usr/local/libexec/llama-bored/llama-metrics)"
  cp -- "$ROOT/packaging/llama-metrics.service" \
    "$(dest_path "$metrics" /etc/systemd/system/llama-metrics.service)"
  snapshot_tree "$metrics" "$tmp/snap-metrics-keep"
  : >"$INSTALL_LOG"
  metrics_stage="$(freeze_staging "$repo" "$metrics")"
  assert_eq "$(tr -d '[:space:]' <"$metrics_stage/metrics.mode")" "keep" "existing metrics.toml kept"
  run_expect_ok verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  printf '# edited after staging\n' >>"$(dest_path "$metrics" /etc/llama-bored/metrics.toml)"
  run_expect_fail verify_staged_matches_disk "$metrics_stage" "$repo" "$metrics"
  cp -- "$tmp/snap-metrics-keep/etc/llama-bored/metrics.toml" \
    "$(dest_path "$metrics" /etc/llama-bored/metrics.toml)"
  metrics_out="$(apply_from_staging "$metrics_stage" "$metrics" "$sys" "$head")"
  assert_eq "$(cat -- "$(dest_path "$metrics" /etc/llama-bored/metrics.toml)")" \
    "$(printf 'listen = "0.0.0.0:19477"\nallow = ["127.0.0.1/32"]\n')" "operator metrics.toml preserved"
  assert_eq "$(cat -- "$(dest_path "$metrics" /usr/local/libexec/llama-bored/llama-metrics)")" \
    "fake-metrics" "llama-metrics upgraded"
  metrics_section="$(rollback_section "$metrics_out")"
  [[ "$metrics_section" != *"metrics.toml"* \
    && "$metrics_section" != *"disable --now llama-metrics"* \
    && "$metrics_section" == *"/usr/local/libexec/llama-bored/llama-metrics.bak-"* ]] || {
    echo "install self-test: upgrade rollback touches the kept metrics.toml, disables the exporter, or loses the old binary" >&2
    printf '%s\n' "$metrics_section" >&2
    exit 1
  }
  assert_rollback_order "$metrics_out"
  execute_printed_rollback "$metrics_out" "$metrics"
  assert_eq "$(cat -- "$(dest_path "$metrics" /usr/local/libexec/llama-bored/llama-metrics)")" \
    "old-metrics" "rollback restored the previous llama-metrics"
  rm -rf -- "$metrics_stage"

  unset INSTALL_DRY INSTALL_ALLOW_UNPRIV_COPY INSTALL_FAKE_STAT INSTALL_FAKE_GETFACL INSTALL_FAKE_PIN INSTALL_LOG INSTALL_FAIL_AT INSTALL_FAIL_MARKER
  unset INSTALL_OWNER_MODE INSTALL_TAMPER_LIVE
  echo "install self-test: ok"
}

append_probe() {
  local marker=$1 script
  if [[ -z "$marker" ]]; then
    echo "install.sh: append probe needs a marker path" >&2
    exit 2
  fi
  # Only a throwaway copy under /tmp or /var/tmp may be modified.
  # TMPDIR is ignored: TMPDIR=/ would otherwise match every absolute path.
  script="$(readlink -f -- "${BASH_SOURCE[0]}")"
  case "$script" in
    /tmp/*|/var/tmp/*) ;;
    *)
      echo "install.sh: --append-probe refuses unless it runs from a temporary copy" >&2
      exit 1
      ;;
  esac
  # Append after this process has started. A shell that is still reading
  # the file would execute the new line as root.
  printf '\ntouch %q\n' "$marker" >>"$script"
  sleep 0.05
  if [[ -e "$marker" ]]; then
    echo "install.sh: text appended after start was executed" >&2
    exit 1
  fi
}

main() {
  case "${1:-}" in
    "")
      LIGHT_ENABLE=0
      install_real
      ;;
    --enable-light)
      if [[ $# -ne 1 ]]; then
        echo "usage: sudo scripts/install.sh [--enable-light]" >&2
        exit 2
      fi
      LIGHT_ENABLE=1
      install_real
      ;;
    --self-test)
      self_test
      ;;
    --append-probe)
      append_probe "${2:-}"
      ;;
    *)
      echo "usage: sudo scripts/install.sh [--enable-light]" >&2
      exit 2
      ;;
  esac
}

main "$@"; exit $?
}
