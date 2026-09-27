#!/usr/bin/env bash
# Read-only cooling evidence: z53 fan/temp/pwm files and k10temp Tctl.
# One timestamped line on stdout. Never writes sysfs.
# Usage: scripts/cooling-snapshot.sh [--sys-root DIR | --self-test]
set -euo pipefail

read_attr() {
  local dir=$1 name=$2
  local path="$dir/$name" canon root value
  if [[ ! -f "$path" || -L "$path" ]]; then
    printf 'missing'
    return 0
  fi
  canon=$(readlink -f "$path") || {
    printf 'missing'
    return 0
  }
  root=$(readlink -f "$dir") || {
    printf 'missing'
    return 0
  }
  case "$canon" in
    "$root" | "$root"/*) ;;
    *)
      printf 'missing'
      return 0
      ;;
  esac
  value=$(tr -d '[:space:]' <"$path")
  if [[ -z "$value" ]]; then
    printf 'missing'
  else
    printf '%s' "$value"
  fi
}

# Print the canonical hwmon directory whose name file equals `sensor`.
# Status 1: none. Status 2: more than one. A symlink that leaves `root` is ignored.
find_named() {
  local root=$1 sensor=$2
  local class="$root/class/hwmon" rootcanon entry base canon name
  local -a found=()
  rootcanon=$(readlink -f "$root") || return 1
  if [[ ! -d "$class" ]]; then
    return 1
  fi
  while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue
    base=$(basename "$entry")
    [[ "$base" =~ ^hwmon[0-9]+$ ]] || continue
    canon=$(readlink -f "$entry") || continue
    case "$canon" in
      "$rootcanon"/*) ;;
      *) continue ;;
    esac
    [[ -f "$canon/name" && ! -L "$canon/name" ]] || continue
    name=$(tr -d '[:space:]' <"$canon/name")
    if [[ "$name" == "$sensor" ]]; then
      found+=("$canon")
    fi
  done < <(find "$class" -mindepth 1 -maxdepth 1 -name 'hwmon[0-9]*' -print 2>/dev/null || true)
  if [[ ${#found[@]} -eq 1 ]]; then
    printf '%s\n' "${found[0]}"
    return 0
  fi
  if [[ ${#found[@]} -eq 0 ]]; then
    return 1
  fi
  return 2
}

read_tctl() {
  local dir=$1
  local label_path base label index value hits=0 tctl_raw=""
  local -a labels=()
  while IFS= read -r label_path; do
    [[ -n "$label_path" ]] || continue
    labels+=("$label_path")
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -name 'temp*_label' -print 2>/dev/null | sort || true)
  if [[ ${#labels[@]} -eq 0 ]]; then
    return 1
  fi
  for label_path in "${labels[@]}"; do
    base=$(basename "$label_path")
    [[ "$base" =~ ^temp[0-9]+_label$ ]] || continue
    [[ -L "$label_path" ]] && continue
    label=$(tr -d '[:space:]' <"$label_path")
    if [[ "$label" == "Tctl" ]]; then
      index=${base#temp}
      index=${index%_label}
      value=$(read_attr "$dir" "temp${index}_input")
      hits=$((hits + 1))
      tctl_raw=$value
    fi
  done
  if [[ $hits -eq 1 ]]; then
    printf '%s' "$tctl_raw"
    return 0
  fi
  return 1
}

pwm_enables() {
  local dir=$1 path base first=1
  local -a names=()
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    base=$(basename "$path")
    [[ "$base" =~ ^pwm[0-9]+_enable$ ]] || continue
    names+=("$base")
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -name 'pwm*_enable' -print 2>/dev/null | sort || true)
  if [[ ${#names[@]} -eq 0 ]]; then
    printf 'pwm_enable=missing'
    return 0
  fi
  for base in "${names[@]}"; do
    if [[ $first -eq 0 ]]; then
      printf ' '
    fi
    first=0
    printf '%s=%s' "$base" "$(read_attr "$dir" "$base")"
  done
}

field_or_missing() {
  local dir=$1 name=$2
  if [[ -z "$dir" ]]; then
    printf 'missing'
  else
    read_attr "$dir" "$name"
  fi
}

# Print one evidence line. Exit 1 when z53 or Tctl cannot be read.
snapshot() {
  local root=$1
  local ts z53_dir="" z53_label k10_dir="" k10_label tctl status=0
  ts=$(date --iso-8601=seconds)
  if z53_dir=$(find_named "$root" z53); then
    z53_label=$(basename "$z53_dir")
  else
    case $? in
      2) z53_label=ambiguous ;;
      *) z53_label=missing ;;
    esac
    z53_dir=""
    status=1
  fi
  if k10_dir=$(find_named "$root" k10temp); then
    k10_label=$(basename "$k10_dir")
    if ! tctl=$(read_tctl "$k10_dir"); then
      tctl=missing
      status=1
    fi
  else
    case $? in
      2) k10_label=ambiguous ;;
      *) k10_label=missing ;;
    esac
    k10_dir=""
    tctl=missing
    status=1
  fi

  local fan1 fan2 temp pwm1 pwm2 enables
  fan1=$(field_or_missing "$z53_dir" fan1_input)
  fan2=$(field_or_missing "$z53_dir" fan2_input)
  temp=$(field_or_missing "$z53_dir" temp1_input)
  pwm1=$(field_or_missing "$z53_dir" pwm1)
  pwm2=$(field_or_missing "$z53_dir" pwm2)
  if [[ -n "$z53_dir" ]]; then
    enables=$(pwm_enables "$z53_dir")
  else
    enables="pwm_enable=missing"
  fi
  if [[ "$fan1" == missing || "$fan2" == missing || "$temp" == missing || "$pwm1" == missing || "$pwm2" == missing ]]; then
    status=1
  fi

  printf '%s z53_hwmon=%s fan1_input=%s fan2_input=%s temp1_input=%s pwm1=%s pwm2=%s %s k10temp_hwmon=%s Tctl=%s\n' \
    "$ts" "$z53_label" "$fan1" "$fan2" "$temp" "$pwm1" "$pwm2" "$enables" "$k10_label" "$tctl"
  return "$status"
}

write_sensor() {
  local dir=$1
  shift
  mkdir -p "$dir"
  local name value
  while [[ $# -ge 2 ]]; do
    name=$1
    value=$2
    printf '%s\n' "$value" >"$dir/$name"
    shift 2
  done
}

self_test() {
  local tmp root line want
  tmp=$(mktemp -d)
  # Expand now: a RETURN trap runs after `local tmp` is gone.
  # shellcheck disable=SC2064
  trap "rm -rf $(printf '%q' "$tmp")" RETURN
  root="$tmp/sys"
  write_sensor "$root/devices/platform/hwmon/hwmon5" \
    name z53 \
    fan1_input 1304 \
    fan2_input 0 \
    temp1_input 37700 \
    pwm1 89 \
    pwm2 77 \
    pwm1_enable 0 \
    pwm2_enable 0
  write_sensor "$root/devices/platform/hwmon/hwmon3" \
    name k10temp \
    temp1_input 45000 \
    temp1_label Tdie \
    temp2_input 49125 \
    temp2_label Tctl
  mkdir -p "$root/class/hwmon"
  ln -s ../../devices/platform/hwmon/hwmon5 "$root/class/hwmon/hwmon5"
  ln -s ../../devices/platform/hwmon/hwmon3 "$root/class/hwmon/hwmon3"
  write_sensor "$tmp/outside" name z53 fan1_input 1
  ln -s "$tmp/outside" "$root/class/hwmon/hwmon9"

  line=$(snapshot "$root")
  want="z53_hwmon=hwmon5 fan1_input=1304 fan2_input=0 temp1_input=37700 pwm1=89 pwm2=77 pwm1_enable=0 pwm2_enable=0 k10temp_hwmon=hwmon3 Tctl=49125"
  [[ "$line" == *" $want" ]] || {
    echo "cooling-snapshot self-test: line mismatch" >&2
    echo "got:  $line" >&2
    echo "want: *$want" >&2
    exit 1
  }
  [[ "$line" != *hwmon9* ]] || {
    echo "cooling-snapshot self-test: escaped symlink was used: $line" >&2
    exit 1
  }

  write_sensor "$root/devices/platform/hwmon/hwmon6" name z53 fan1_input 1
  ln -s ../../devices/platform/hwmon/hwmon6 "$root/class/hwmon/hwmon6"
  if snapshot "$root" >/dev/null; then
    echo "cooling-snapshot self-test: two z53 sensors were accepted" >&2
    exit 1
  fi
  rm "$root/class/hwmon/hwmon6"

  rm "$root/class/hwmon/hwmon5"
  line=$(snapshot "$root" || true)
  [[ "$line" == *"z53_hwmon=missing"* ]] || {
    echo "cooling-snapshot self-test: missing z53: $line" >&2
    exit 1
  }
  echo "cooling-snapshot self-test: ok"
}

main() {
  case "${1:-}" in
    "")
      snapshot /sys
      ;;
    --self-test)
      self_test
      ;;
    --sys-root)
      if [[ $# -ne 2 || -z "${2:-}" ]]; then
        echo "usage: scripts/cooling-snapshot.sh --sys-root DIR" >&2
        exit 2
      fi
      snapshot "$2"
      ;;
    *)
      echo "usage: scripts/cooling-snapshot.sh [--sys-root DIR | --self-test]" >&2
      exit 2
      ;;
  esac
}

main "$@"
