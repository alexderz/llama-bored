#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo deny --locked check
scripts/s10-deps.sh
scripts/s10-deps.sh --self-test
cargo audit
scripts/s6-symbols.sh

# The committed tty11 font is a byte-for-byte rebuild from Hack. The
# script skips with a note when Pillow or the reference TTF is absent.
# tests/tty_font.rs (cargo test above) checks its glyph coverage.
if command -v python3 >/dev/null 2>&1; then
  python3 packaging/fonts/build-psf.py --self-test
else
  echo "check.sh: SKIP font rebuild, python3 is not installed"
fi

# Packaging self-test after the release build, so a unit fence or install
# spec drift fails this gate. It refuses root and runs stage.sh --self-test.
if [[ "$(id -u)" -eq 0 ]]; then
  echo "check.sh: refusing to run install self-test as root" >&2
  exit 1
fi
scripts/install.sh --self-test
scripts/ramp-monitor.sh --self-test

# Eleven lines, in this order. scripts/stage.sh and the root installer
# both parse this file. The last five lines are the release binaries.
# S6 still checks only the LCD writer (kraken-lcd); S15 (cargo test) and
# scripts/s10-deps.sh fence llama-light; S16 (cargo test) and
# scripts/s10-deps.sh fence llama-metrics.
mkdir -p target
{
  rustc -V
  cargo -V
  sha256sum Cargo.lock rust-toolchain.toml
  git rev-parse HEAD
  git describe --always --dirty
  sha256sum target/release/kraken-lcd target/release/llama-watch target/release/llama-view \
    target/release/llama-light target/release/llama-metrics
} > target/check-provenance.txt
