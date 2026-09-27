#!/usr/bin/env bash
# Read-only 1 Hz cooling monitor for the live-ring fps ramp.
# Samples z53 pump/fan rpm, coolant temp, pwm* and pwm*_enable.
# Watches kernel logs for USB resets and the 1e71:3011 bootloader id.
# Never writes sysfs. Run as uid 1000.
# Usage: scripts/ramp-monitor.sh [--sys-root DIR] [--log-file FILE] [--out FILE]
#                                 [--interval SEC] [--baseline-samples N]
#                                 [--max-samples N] [--self-test]
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

find_z53() {
  local root=$1
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
    if [[ "$name" == "z53" ]]; then
      found+=("$canon")
    fi
  done < <(find "$class" -mindepth 1 -maxdepth 1 -name 'hwmon[0-9]*' -print 2>/dev/null || true)
  if [[ ${#found[@]} -eq 1 ]]; then
    printf '%s\n' "${found[0]}"
    return 0
  fi
  return 1
}

list_pwm_files() {
  local dir=$1 kind=$2
  local path base
  local -a names=()
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    base=$(basename "$path")
    case "$kind" in
      pwm)
        [[ "$base" =~ ^pwm[0-9]+$ ]] || continue
        ;;
      enable)
        [[ "$base" =~ ^pwm[0-9]+_enable$ ]] || continue
        ;;
    esac
    names+=("$base")
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -name 'pwm*' -print 2>/dev/null | sort || true)
  printf '%s\n' "${names[@]}"
}

# Print: pump fan coolant then pwmN=val ... then pwmN_enable=val ...
sample_z53() {
  local root=$1 dir
  if ! dir=$(find_z53 "$root"); then
    printf 'missing missing missing\n'
    return 1
  fi
  local pump fan coolant pwm_line="" enable_line="" name value
  pump=$(read_attr "$dir" fan1_input)
  fan=$(read_attr "$dir" fan2_input)
  coolant=$(read_attr "$dir" temp1_input)
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    value=$(read_attr "$dir" "$name")
    pwm_line+="${pwm_line:+ }${name}=${value}"
  done < <(list_pwm_files "$dir" pwm)
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    value=$(read_attr "$dir" "$name")
    enable_line+="${enable_line:+ }${name}=${value}"
  done < <(list_pwm_files "$dir" enable)
  printf '%s %s %s %s %s\n' "$pump" "$fan" "$coolant" "${pwm_line:-pwm=missing}" "${enable_line:-pwm_enable=missing}"
}

kernel_abort_reason() {
  local text=$1
  if grep -qiE '1e71:3011|idvendor=1e71.*idproduct=3011|idproduct=3011.*idvendor=1e71' <<<"$text"; then
    printf 'bootloader'
    return 0
  fi
  if grep -qiE 'usb[^[:space:]]*[[:space:]].*reset|[[:space:]]reset[[:space:]].*usb' <<<"$text"; then
    printf 'usb-reset'
    return 0
  fi
  return 1
}

read_kernel_log() {
  local file=${1:-}
  if [[ -n "$file" ]]; then
    cat "$file" 2>/dev/null || true
    return 0
  fi
  if command -v journalctl >/dev/null 2>&1; then
    journalctl -k -n 40 --no-pager -o short-iso 2>/dev/null || true
  fi
  if command -v dmesg >/dev/null 2>&1; then
    dmesg -T 2>/dev/null || dmesg 2>/dev/null || true
  fi
}

# Abort rules (T43)
# -----------------
# The Kraken firmware runs its own pump curve: pwm1 moves with coolant
# temperature (115 -> 102 as it cools) while pwm*_enable stays put. A pwm
# value change is therefore logged as `NOTE pwm1 115->102`, not an abort.
#
# The monitor aborts on:
#   - a USB reset or the 1e71:3011 bootloader id in the kernel log;
#   - an unreadable pump, fan, coolant or pwm value, or the set of pwm*
#     files changing;
#   - any pwm*_enable change against the first sample;
#   - stall: rpm 0 while that channel's pwm > 0, once the channel has been
#     seen spinning (a header with nothing attached reads 0 and is exempt).
#     Checked on every sample, with no settle delay;
#   - rpm-drift: rpm outside +-RPM_BAND_PCT of the rpm learned for the
#     current pwm value.
#
# The per-pwm table: channel pump is pwm1/fan1_input, channel fan is
# pwm2/fan2_input. rpm lags a pwm step by a few seconds, so a sample is
# "settled" only after the pwm value has been unchanged for SETTLE_SAMPLES
# samples. The first LEARN_SAMPLES settled samples at a pwm value (baseline or
# later) set the expected rpm as their mean; later settled samples at that
# value are checked against it. Samples that are not settled are never learned
# or band-checked; the stall rule covers them. A pwm value seen for the first
# time is learned, not checked, which is why the stall rule is kept as a
# backstop. The band checks start after --baseline-samples.
SETTLE_SAMPLES=5
LEARN_SAMPLES=3
RPM_BAND_PCT=15

MON_N=0
MON_BASELINE_N=10
MON_ENABLES=""
MON_PWM_NAMES=""
declare -A MON_LAST_PWM=()
declare -A MON_STEADY=()
declare -A MON_SPUN=()
declare -A MON_LEARN_SUM=()
declare -A MON_LEARN_CNT=()
declare -A MON_EXPECT=()
STEP_REASON=""
STEP_NOTES=()

monitor_reset() {
  MON_BASELINE_N=$1
  MON_N=0
  MON_ENABLES=""
  MON_PWM_NAMES=""
  MON_LAST_PWM=()
  MON_STEADY=()
  MON_SPUN=()
  MON_LEARN_SUM=()
  MON_LEARN_CNT=()
  MON_EXPECT=()
  STEP_REASON=""
  STEP_NOTES=()
}

# Value of NAME in a "k=v k=v" list, or empty.
kv_get() {
  local list=$1 name=$2 item
  for item in $list; do
    if [[ "${item%%=*}" == "$name" ]]; then
      printf '%s' "${item#*=}"
      return 0
    fi
  done
  return 0
}

kv_names() {
  local list=$1 item
  for item in $list; do
    printf '%s\n' "${item%%=*}"
  done | sort | tr '\n' ' '
}

# First NAME whose value differs between two "k=v" lists, or whose presence
# differs. Returns 1 when they match.
kv_first_diff() {
  local a=$1 b=$2 item name
  for item in $a $b; do
    name=${item%%=*}
    if [[ "$(kv_get "$a" "$name")" != "$(kv_get "$b" "$name")" ]]; then
      printf '%s' "$name"
      return 0
    fi
  done
  return 1
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

# Integer rpm only (checked by is_uint), so plain shell arithmetic.
outside_band() {
  local now=$1 expect=$2 diff
  diff=$((now - expect))
  ((diff < 0)) && diff=$((-diff))
  ((diff * 100 > expect * RPM_BAND_PCT))
}

# One channel's rpm rules. Sets STEP_REASON and returns 2 on abort.
check_channel() {
  local ch=$1 pwm=$2 rpm=$3
  local key="$ch:$pwm"
  if [[ "$rpm" -gt 0 ]]; then
    MON_SPUN[$ch]=1
  fi
  if [[ "$rpm" -eq 0 && "$pwm" -gt 0 && -n "${MON_SPUN[$ch]:-}" ]]; then
    STEP_REASON="stall $ch"
    return 2
  fi
  if [[ "${MON_STEADY[$ch]}" -lt $SETTLE_SAMPLES ]]; then
    return 0
  fi
  if [[ -n "${MON_EXPECT[$key]+x}" ]]; then
    if [[ $MON_N -gt $MON_BASELINE_N ]] && outside_band "$rpm" "${MON_EXPECT[$key]}"; then
      STEP_REASON="rpm-drift $ch"
      return 2
    fi
    return 0
  fi
  MON_LEARN_SUM[$key]=$((${MON_LEARN_SUM[$key]:-0} + rpm))
  MON_LEARN_CNT[$key]=$((${MON_LEARN_CNT[$key]:-0} + 1))
  if [[ ${MON_LEARN_CNT[$key]} -ge $LEARN_SAMPLES ]]; then
    MON_EXPECT[$key]=$(((MON_LEARN_SUM[$key] + MON_LEARN_CNT[$key] / 2) / MON_LEARN_CNT[$key]))
  fi
  return 0
}

# One sample. Runs in the caller's shell (it keeps state), so it does not
# print: STEP_NOTES gets "pwm1 115->102" lines and STEP_REASON the abort
# reason. Returns 2 when the monitor must stop.
monitor_step() {
  local pump=$1 fan=$2 coolant=$3 pwms=$4 enables=$5 log_text=$6
  local reason name ch pwm rpm prev
  STEP_REASON=""
  STEP_NOTES=()
  if reason=$(kernel_abort_reason "$log_text"); then
    STEP_REASON=$reason
    return 2
  fi
  MON_N=$((MON_N + 1))
  if [[ $MON_N -eq 1 ]]; then
    MON_ENABLES=$enables
    MON_PWM_NAMES=$(kv_names "$pwms")
  fi
  if ! is_uint "$pump" || ! is_uint "$fan" || [[ "$coolant" == missing || "$pwms" == *=missing* ]]; then
    STEP_REASON="unreadable"
    return 2
  fi
  if name=$(kv_first_diff "$MON_ENABLES" "$enables"); then
    STEP_REASON="pwm-enable $name"
    return 2
  fi
  if [[ "$(kv_names "$pwms")" != "$MON_PWM_NAMES" ]]; then
    STEP_REASON="pwm-set"
    return 2
  fi
  for ch in pump fan; do
    if [[ $ch == pump ]]; then
      name=pwm1
      rpm=$pump
    else
      name=pwm2
      rpm=$fan
    fi
    pwm=$(kv_get "$pwms" "$name")
    is_uint "$pwm" || continue
    prev=${MON_LAST_PWM[$ch]:-}
    if [[ -n "$prev" && "$prev" != "$pwm" ]]; then
      STEP_NOTES+=("$name $prev->$pwm")
      MON_STEADY[$ch]=1
    else
      MON_STEADY[$ch]=$((${MON_STEADY[$ch]:-0} + 1))
    fi
    MON_LAST_PWM[$ch]=$pwm
    check_channel "$ch" "$pwm" "$rpm" || return 2
  done
  return 0
}

abort() {
  local reason=$1
  printf 'ABORT %s\n' "$reason"
  exit 2
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

make_z53_tree() {
  local root=$1 pump=$2 fan=$3 coolant=$4 pwm1=$5 pwm2=$6 en1=$7 en2=$8
  write_sensor "$root/devices/platform/hwmon/hwmon5" \
    name z53 \
    fan1_input "$pump" \
    fan2_input "$fan" \
    temp1_input "$coolant" \
    pwm1 "$pwm1" \
    pwm2 "$pwm2" \
    pwm1_enable "$en1" \
    pwm2_enable "$en2"
  mkdir -p "$root/class/hwmon"
  ln -sfn ../../devices/platform/hwmon/hwmon5 "$root/class/hwmon/hwmon5"
}

run_monitor() {
  local sys_root=$1 log_file=$2 out_file=$3 interval=$4 baseline_n=$5 max_samples=$6
  local line pump fan coolant pwms enables ts note log_text pwm_and_en tok
  monitor_reset "$baseline_n"
  if [[ -n "$out_file" ]]; then
    printf 'ts\tpump_rpm\tfan_rpm\tcoolant_mc\tpwms\tenables\n' >"$out_file"
  else
    printf 'ts\tpump_rpm\tfan_rpm\tcoolant_mc\tpwms\tenables\n'
  fi
  while true; do
    ts=$(date --iso-8601=seconds)
    if ! line=$(sample_z53 "$sys_root"); then
      abort "z53-missing"
    fi
    pwm_and_en=""
    read -r pump fan coolant pwm_and_en <<<"$line"
    pwms=""
    enables=""
    for tok in $pwm_and_en; do
      if [[ "$tok" == *_enable=* ]]; then
        enables+="${enables:+ }$tok"
      else
        pwms+="${pwms:+ }$tok"
      fi
    done
    if [[ -n "$out_file" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$pump" "$fan" "$coolant" "$pwms" "$enables" >>"$out_file"
    else
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$pump" "$fan" "$coolant" "$pwms" "$enables"
    fi
    log_text=$(read_kernel_log "$log_file")
    if monitor_step "$pump" "$fan" "$coolant" "$pwms" "$enables" "$log_text"; then
      for note in "${STEP_NOTES[@]}"; do
        printf 'NOTE %s\n' "$note"
      done
    else
      for note in "${STEP_NOTES[@]}"; do
        printf 'NOTE %s\n' "$note"
      done
      abort "$STEP_REASON"
    fi
    if [[ $max_samples -gt 0 && $MON_N -ge $max_samples ]]; then
      return 0
    fi
    if awk -v i="$interval" 'BEGIN { exit !(i > 0) }'; then
      sleep "$interval"
    fi
  done
}

expect_abort() {
  local got=$1 want=$2 label=$3
  if [[ "$got" != "$want" ]]; then
    echo "ramp-monitor self-test: $label: got '$got' want '$want'" >&2
    exit 1
  fi
}

# Feed N identical samples; any abort fails the self-test.
feed_steps() {
  local n=$1 pump=$2 fan=$3 pwms=$4 enables=$5 label=$6 i
  for ((i = 0; i < n; i++)); do
    monitor_step "$pump" "$fan" 37700 "$pwms" "$enables" "" || {
      echo "ramp-monitor self-test: $label: unexpected ABORT $STEP_REASON" >&2
      exit 1
    }
  done
}

expect_step_abort() {
  local want=$1 pump=$2 fan=$3 pwms=$4 enables=$5 log_text=$6 label=$7 status=0
  monitor_step "$pump" "$fan" 37700 "$pwms" "$enables" "$log_text" || status=$?
  [[ $status -eq 2 ]] || {
    echo "ramp-monitor self-test: $label did not abort (status=$status)" >&2
    exit 1
  }
  expect_abort "$STEP_REASON" "$want" "$label"
}

# run_monitor for 3 samples in a subshell whose `sleep` is replaced by a
# function that rewrites pwm1 to 80, so the second sample sees a pwm step.
run_monitor_with_step() {
  local root=$1 log=$2 out=$3
  (
    sleep() {
      printf '80\n' >"$root/devices/platform/hwmon/hwmon5/pwm1"
    }
    run_monitor "$root" "$log" "$out" 1 10 3
  )
}

self_test() {
  local tmp root log out line reason status
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf $(printf '%q' "$tmp")" RETURN
  root="$tmp/sys"
  log="$tmp/kernel.log"
  out="$tmp/monitor.tsv"
  : >"$log"

  make_z53_tree "$root" 1304 0 37700 89 77 0 0
  write_sensor "$tmp/outside" name z53 fan1_input 1
  ln -s "$tmp/outside" "$root/class/hwmon/hwmon9"

  run_monitor "$root" "$log" "$out" 0 10 15
  line=$(sample_z53 "$root")
  [[ "$line" != *hwmon9* ]] || {
    echo "ramp-monitor self-test: escaped symlink was used: $line" >&2
    exit 1
  }
  [[ -s "$out" ]] || {
    echo "ramp-monitor self-test: TSV was not written" >&2
    exit 1
  }
  grep -q $'^ts\tpump_rpm\tfan_rpm\tcoolant_mc\tpwms\tenables$' "$out" || {
    echo "ramp-monitor self-test: TSV header mismatch" >&2
    exit 1
  }
  local rows
  rows=$(grep -c $'\t1304\t' "$out" || true)
  [[ "$rows" -eq 15 ]] || {
    echo "ramp-monitor self-test: expected 15 data rows, got $rows" >&2
    exit 1
  }

  chmod a-w "$root/devices/platform/hwmon/hwmon5/pwm1" \
    "$root/devices/platform/hwmon/hwmon5/fan1_input"
  sample_z53 "$root" >/dev/null
  chmod u+w "$root/devices/platform/hwmon/hwmon5/pwm1" \
    "$root/devices/platform/hwmon/hwmon5/fan1_input"

  local en="pwm1_enable=0 pwm2_enable=0"

  # A firmware pump-curve step (pwm1 115->102, enable unchanged, rpm
  # following a few samples later) is a NOTE, not an abort.
  monitor_reset 10
  feed_steps 10 2000 0 "pwm1=115 pwm2=77" "$en" "baseline"
  monitor_step 2000 0 36900 "pwm1=102 pwm2=77" "$en" "" || {
    echo "ramp-monitor self-test: firmware curve step aborted: $STEP_REASON" >&2
    exit 1
  }
  expect_abort "${STEP_NOTES[*]}" "pwm1 115->102" "pwm note"
  feed_steps 2 1900 0 "pwm1=102 pwm2=77" "$en" "rpm settling"
  feed_steps 12 1800 0 "pwm1=102 pwm2=77" "$en" "settled at 102"
  feed_steps 2 1700 0 "pwm1=102 pwm2=77" "$en" "inside the band at 102"
  # Back to 115: the rpm learned there still holds.
  feed_steps 8 2050 0 "pwm1=115 pwm2=77" "$en" "back at 115"
  expect_step_abort "rpm-drift pump" 1500 0 "pwm1=115 pwm2=77" "$en" "" "pump drift at 115"

  # A pump stall aborts at once, even right after a pwm step.
  monitor_reset 10
  feed_steps 10 2000 0 "pwm1=115 pwm2=77" "$en" "baseline"
  expect_step_abort "stall pump" 0 0 "pwm1=102 pwm2=77" "$en" "" "pump stall"

  # An enable change aborts even when nothing else moves.
  monitor_reset 10
  feed_steps 10 2000 0 "pwm1=115 pwm2=77" "$en" "baseline"
  expect_step_abort "pwm-enable pwm1_enable" 2000 0 "pwm1=115 pwm2=77" \
    "pwm1_enable=1 pwm2_enable=0" "" "pwm enable"

  # A fan header that read 0 in the baseline and now spins is drift.
  monitor_reset 10
  feed_steps 10 2000 0 "pwm1=115 pwm2=77" "$en" "baseline"
  expect_step_abort "rpm-drift fan" 2000 80 "pwm1=115 pwm2=77" "$en" "" "fan drift"

  # A pwm file that disappears aborts.
  monitor_reset 10
  feed_steps 2 2000 0 "pwm1=115 pwm2=77" "$en" "baseline"
  expect_step_abort "pwm-set" 2000 0 "pwm1=115" "$en" "" "pwm set"

  monitor_reset 10
  expect_step_abort "usb-reset" 2000 0 "pwm1=115 pwm2=77" "$en" \
    'usb 3-5: reset high-speed USB device number 2 using xhci_hcd' "usb reset"
  monitor_reset 10
  expect_step_abort "bootloader" 2000 0 "pwm1=115 pwm2=77" "$en" \
    'usb 3-6: New USB device found, idVendor=1e71, idProduct=3011' "bootloader"

  # Live path: a pwm change between two runs' trees prints a NOTE.
  make_z53_tree "$root" 1304 0 37700 89 77 0 0
  : >"$log"
  local note_out
  note_out=$(run_monitor_with_step "$root" "$log" "$tmp/note.tsv") || {
    echo "ramp-monitor self-test: live pwm change aborted: $note_out" >&2
    exit 1
  }
  grep -qx 'NOTE pwm1 89->80' <<<"$note_out" || {
    echo "ramp-monitor self-test: live pwm change did not print a NOTE: $note_out" >&2
    exit 1
  }

  status=0
  reason=$(kernel_abort_reason $'usb 3-6: 1e71:3011 bootloader\n') || status=$?
  [[ $status -eq 0 && "$reason" == bootloader ]] || {
    echo "ramp-monitor self-test: compact 1e71:3011 id was missed" >&2
    exit 1
  }

  make_z53_tree "$root" 1304 0 37700 89 77 0 0
  : >"$log"
  printf 'usb 3-5: reset high-speed USB device number 2 using xhci_hcd\n' >"$log"
  status=0
  (run_monitor "$root" "$log" "$tmp/reset.tsv" 0 1 1) && status=0 || status=$?
  [[ $status -eq 2 ]] || {
    echo "ramp-monitor self-test: live usb-reset path exit $status" >&2
    exit 1
  }

  echo "ramp-monitor self-test: ok"
}

usage() {
  echo "usage: scripts/ramp-monitor.sh [--sys-root DIR] [--log-file FILE] [--out FILE] [--interval SEC] [--baseline-samples N] [--max-samples N] [--self-test]" >&2
  exit 2
}

main() {
  local sys_root=/sys log_file="" out_file="" interval=1 baseline_n=10 max_samples=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test)
        if [[ $# -ne 1 ]]; then
          usage
        fi
        self_test
        return 0
        ;;
      --sys-root)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        sys_root=$2
        shift 2
        ;;
      --log-file)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        log_file=$2
        shift 2
        ;;
      --out)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        out_file=$2
        shift 2
        ;;
      --interval)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        interval=$2
        shift 2
        ;;
      --baseline-samples)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        baseline_n=$2
        shift 2
        ;;
      --max-samples)
        [[ $# -ge 2 && -n "${2:-}" ]] || usage
        max_samples=$2
        shift 2
        ;;
      *)
        usage
        ;;
    esac
  done
  run_monitor "$sys_root" "$log_file" "$out_file" "$interval" "$baseline_n" "$max_samples"
}

main "$@"
