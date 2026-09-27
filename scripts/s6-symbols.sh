#!/usr/bin/env bash
# S6: the release binary must not carry USB control symbols, and it must
# link only the platform C runtime (glibc, the dynamic linker, libgcc_s).
set -euo pipefail

# USB resets stay forbidden. A bare `::reset` also matches unrelated methods
# such as png's ZlibStream::reset, so the reset alternatives are USB-scoped.
forbidden_re='detach_and_claim_interface|set_configuration|nusb::[^ ]*reset|[Uu]sb[^ ]*::reset|USBDEVFS_RESET|21780|0x5514|reset_device|control_out|control_in|detach_kernel_driver'

s6_rejects() {
  grep -E -q "$forbidden_re" <<<"$1"
}

# Same expression as the nm scan below. Runs before the release build.
s6_self_test() {
  local line expect got
  while IFS=$'\t' read -r line expect; do
    [[ -z "$line" ]] && continue
    if s6_rejects "$line"; then
      got=reject
    else
      got=allow
    fi
    if [[ "$got" != "$expect" ]]; then
      echo "S6 self-test: '$line' expected $expect, got $got" >&2
      exit 1
    fi
  done <<'EOF'
nusb::device::Device::reset	reject
<png::decoder::zlib::ZlibStream>::reset	allow
21780	reject
0x5514	reject
EOF
}

s6_self_test

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# rustup from Homebrew when brew has it, else ~/.cargo/bin, else PATH as is.
if command -v brew >/dev/null 2>&1 \
  && rustup_prefix="$(brew --prefix rustup 2>/dev/null)" \
  && [[ -d "${rustup_prefix}/bin" ]]; then
  export PATH="${rustup_prefix}/bin:${HOME}/.cargo/bin:${PATH}"
else
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

cargo build --release --locked

symbols="$(nm -C target/release/kraken-lcd)"
# An empty or fully stripped table must fail. `main` is linked even when
# fat LTO drops the uncalled LCD paths; `run` is what pulls those in.
if ! grep -q 'kraken_lcd::main' <<<"$symbols"; then
  echo "S6: required symbol kraken_lcd::main is absent" >&2
  exit 1
fi
# `run` drives the device through KrakenLcd. Fat LTO would drop `device/` if that call
# were missing, so a real HID transact symbol is required, not drop glue.
if ! grep -E -q 'kraken_lcd::device::hid::.*transact' <<<"$symbols"; then
  echo "S6: required hid::transact symbol is absent; run does not wire the sink" >&2
  exit 1
fi
if grep -E "$forbidden_re" <<<"$symbols"; then
  echo "S6: forbidden symbol in target/release/kraken-lcd" >&2
  exit 1
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # ldd indents with a tab. The first field is the soname, or the loader path.
  read -r name _ <<<"$line"
  base="${name##*/}"
  case "$base" in
    linux-vdso.so.*|ld-linux*.so*|libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libgcc_s.so.*) ;;
    *)
      echo "S6: unexpected dynamic library: $line" >&2
      exit 1
      ;;
  esac
done < <(ldd target/release/kraken-lcd)

echo "S6: no forbidden symbols; dynamic libraries are the platform C runtime"
