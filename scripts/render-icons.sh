#!/usr/bin/env bash
# Dev-only. Rasterise the brain SVGs to the 40 px sprites Assets::load decodes.
# resvg is a user-level CLI (~/.cargo/bin). It is not a crate build dependency.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.cargo/bin:${PATH}"

if ! command -v resvg >/dev/null 2>&1; then
  echo "render-icons: resvg is not on PATH (expected ${HOME}/.cargo/bin/resvg)" >&2
  exit 1
fi

mkdir -p "${root}/assets/icons"

for name in awake sleeping dead; do
  resvg -w 40 -h 40 \
    "${root}/docs/mockups/icons/brain-${name}.svg" \
    "${root}/assets/icons/brain-${name}-40.png"
done
