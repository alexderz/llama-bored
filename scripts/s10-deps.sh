#!/usr/bin/env bash
# S10: kraken-lcd's normal dependency tree is pinned in scripts/s10-allow.txt.
# A new package fails the run so someone reviews it before the pin moves.
# The denylist below is a second check: those names are rejected even if
# they are later added to the pin by mistake. `cargo deny` also bans ureq
# and nvml-wrapper outside llama-watch.
set -euo pipefail
export LC_ALL=C

forbidden=(
  ureq
  nvml-wrapper
  libloading
  rustls
  native-tls
  http
  llama-watch
)

# S15: llama-light (the RGB writer) has its own pin,
# scripts/s15-light-allow.txt, and a longer denylist: no network, no USB
# stack (it writes one hidraw node through std), no i2c, and not the LCD
# writer crate.
light_forbidden=(
  "${forbidden[@]}"
  nusb
  rusb
  hidapi
  i2cdev
  i2c-linux
  kraken-lcd
)

# S16: llama-metrics (the LAN exporter) has its own pin,
# scripts/s16-metrics-allow.txt, and a longer denylist: no HTTP client or
# server framework, no TLS, no async runtime, no device crates, and neither
# the watcher, the LCD writer, nor the RGB writer.
metrics_forbidden=(
  "${forbidden[@]}"
  kraken-lcd
  llama-light
  nusb
  rusb
  hidapi
  nvml-wrapper-sys
  tokio
  hyper
  axum
  tiny_http
  prometheus
  socket2
  mio
)

# Print tree lines whose package name is forbidden. Empty output is a pass.
# The name is the first field of `cargo tree --prefix none`, so `http-body`
# is not `http` and `ureq-proto` is not `ureq`.
# With an argument, that array name is the denylist (default: forbidden).
s10_hits() {
  local line name crate
  local -n deny="${1:-forbidden}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    name="${line%%[[:space:]]*}"
    for crate in "${deny[@]}"; do
      if [[ "${name}" == "${crate}" ]]; then
        printf '%s\n' "${line}"
        break
      fi
    done
  done
}

# Sorted unique package names from a `cargo tree --prefix none` listing.
s10_names() {
  local line name
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    name="${line%%[[:space:]]*}"
    printf '%s\n' "${name}"
  done | sort -u
}

# Package names present in the tree and absent from the allow file.
s10_added_names() {
  local allowfile="$1"
  comm -13 "${allowfile}" <(s10_names)
}

# A planted tree must be rejected, once per forbidden name. Lookalike names
# must be allowed, so a substring search cannot replace the name check.
# An allowlist must reject a package that is not pinned, and accept the pin.
s10_self_test() {
  local crate planted hits added
  for crate in "${forbidden[@]}"; do
    planted=$(printf '%s\n' "kraken-lcd v0.1.0" "${crate} v9.9.9" "llama-core v0.1.0")
    hits="$(s10_hits <<<"${planted}")"
    if [[ -z "${hits}" ]]; then
      echo "S10 self-test: planted ${crate} was allowed" >&2
      exit 1
    fi
  done

  planted=$(printf '%s\n' \
    "kraken-lcd v0.1.0" \
    "llama-core v0.1.0" \
    "http-body v1.0.0" \
    "ureq-proto v0.1.0" \
    "rustls-pki-types v1.0.0" \
    "native-tls-vendored v0.1.0" \
    "libloading-fork v0.1.0" \
    "nvml-wrapper-sys v0.1.0" \
    "serde v1.0.0")
  hits="$(s10_hits <<<"${planted}")"
  if [[ -n "${hits}" ]]; then
    echo "S10 self-test: lookalike crates were rejected:" >&2
    printf '%s\n' "${hits}" >&2
    exit 1
  fi

  allow="$(mktemp)"
  trap 'rm -f "${allow}"' EXIT
  printf '%s\n' kraken-lcd llama-core serde > "${allow}"
  added="$(s10_added_names "${allow}" <<<"${planted}")"
  if [[ "${added}" != *http-body* ]]; then
    echo "S10 self-test: unpinned http-body was not reported" >&2
    printf '%s\n' "${added}" >&2
    exit 1
  fi
  planted=$(printf '%s\n' "llama-core v0.1.0" "kraken-lcd v0.1.0" "serde v1.0.0")
  added="$(s10_added_names "${allow}" <<<"${planted}")"
  if [[ -n "${added}" ]]; then
    echo "S10 self-test: the pinned set was rejected:" >&2
    printf '%s\n' "${added}" >&2
    exit 1
  fi

  for crate in "${light_forbidden[@]}"; do
    planted=$(printf '%s\n' "llama-light v0.1.0" "${crate} v9.9.9" "llama-core v0.1.0")
    hits="$(s10_hits light_forbidden <<<"${planted}")"
    if [[ -z "${hits}" ]]; then
      echo "S15 self-test: planted ${crate} in llama-light was allowed" >&2
      exit 1
    fi
  done
  planted=$(printf '%s\n' "llama-light v0.1.0" "nusb-lookalike v0.1.0" "rustix v1.0.0")
  hits="$(s10_hits light_forbidden <<<"${planted}")"
  if [[ -n "${hits}" ]]; then
    echo "S15 self-test: lookalike crates were rejected in llama-light:" >&2
    printf '%s\n' "${hits}" >&2
    exit 1
  fi

  for crate in "${metrics_forbidden[@]}"; do
    planted=$(printf '%s\n' "llama-metrics v0.1.0" "${crate} v9.9.9" "llama-core v0.1.0")
    hits="$(s10_hits metrics_forbidden <<<"${planted}")"
    if [[ -z "${hits}" ]]; then
      echo "S16 self-test: planted ${crate} in llama-metrics was allowed" >&2
      exit 1
    fi
  done
  planted=$(printf '%s\n' "llama-metrics v0.1.0" "tokio-lookalike v0.1.0" "rustix v1.0.0")
  hits="$(s10_hits metrics_forbidden <<<"${planted}")"
  if [[ -n "${hits}" ]]; then
    echo "S16 self-test: lookalike crates were rejected in llama-metrics:" >&2
    printf '%s\n' "${hits}" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "--self-test" ]]; then
  s10_self_test
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: scripts/s10-deps.sh [--self-test]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

tree="$(cargo tree -p kraken-lcd -e normal --locked --prefix none)"
hits="$(s10_hits <<<"${tree}")"
if [[ -n "${hits}" ]]; then
  echo "S10: kraken-lcd normal dependency tree contains a forbidden crate:" >&2
  printf '%s\n' "${hits}" >&2
  exit 1
fi

allow="${root}/scripts/s10-allow.txt"
added="$(s10_added_names "${allow}" <<<"${tree}")"
if [[ -n "${added}" ]]; then
  echo "S10: kraken-lcd gained a normal dependency that is not in scripts/s10-allow.txt; review it before adding:" >&2
  printf '%s\n' "${added}" >&2
  exit 1
fi

light_tree="$(cargo tree -p llama-light -e normal --locked --prefix none)"
hits="$(s10_hits light_forbidden <<<"${light_tree}")"
if [[ -n "${hits}" ]]; then
  echo "S15: llama-light normal dependency tree contains a forbidden crate:" >&2
  printf '%s\n' "${hits}" >&2
  exit 1
fi
light_allow="${root}/scripts/s15-light-allow.txt"
added="$(s10_added_names "${light_allow}" <<<"${light_tree}")"
if [[ -n "${added}" ]]; then
  echo "S15: llama-light gained a normal dependency that is not in scripts/s15-light-allow.txt; review it before adding:" >&2
  printf '%s\n' "${added}" >&2
  exit 1
fi

metrics_tree="$(cargo tree -p llama-metrics -e normal --locked --prefix none)"
hits="$(s10_hits metrics_forbidden <<<"${metrics_tree}")"
if [[ -n "${hits}" ]]; then
  echo "S16: llama-metrics normal dependency tree contains a forbidden crate:" >&2
  printf '%s\n' "${hits}" >&2
  exit 1
fi
metrics_allow="${root}/scripts/s16-metrics-allow.txt"
added="$(s10_added_names "${metrics_allow}" <<<"${metrics_tree}")"
if [[ -n "${added}" ]]; then
  echo "S16: llama-metrics gained a normal dependency that is not in scripts/s16-metrics-allow.txt; review it before adding:" >&2
  printf '%s\n' "${added}" >&2
  exit 1
fi
