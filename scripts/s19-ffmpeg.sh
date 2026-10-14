#!/usr/bin/env bash
# S19: ffmpeg stays a program the operator installs. The repository must
# not gain ffmpeg, x264, openh264 or another encoder's source or binary,
# and no Cargo dependency may link one. llama-cast must keep spawning the
# external program (Command::new on the configured path, empty environment).
set -euo pipefail
export LC_ALL=C

# Exact crate names. A lookalike (ffmpeg-next-tools, my-ffmpeg-next) is
# not one of these and must be allowed.
s19_crates=(
  ffmpeg-sys
  ffmpeg-sys-next
  ffmpeg-next
  ffmpeg-next-sys
  ac-ffmpeg
  rsmpeg
  x264
  x264-sys
  openh264
  openh264-sys
  openh264-rs
  rav1e
  dav1d
  dav1d-sys
  libvpx-sys
  vpx-sys
  aom-sys
  opus-sys
  svt-av1
  svt-av1-sys
)

# A path that is an encoder tree, source file or binary. A test fixture
# that only lists encoder names (ffmpeg-encoders.txt) is not one of these.
s19_path_bad() {
  local path="$1" part lower base
  local -a parts=()
  IFS='/' read -ra parts <<< "$path"
  local i
  for part in "${parts[@]}"; do
    lower=$(printf '%s' "$part" | tr '[:upper:]' '[:lower:]')
    case "$lower" in
      ffmpeg|ffprobe|ffplay|x264|x265|openh264|libav|libavcodec|libavformat|libavutil|libavdevice|libavfilter|libswscale|libswresample)
        return 0
        ;;
    esac
  done
  # A directory whose name is, or starts with, one of those projects
  # (`ffmpeg-src/`, `x264-master/`). The last component is the file, so a
  # note named ffmpeg-notes.md is not a vendored tree.
  if ((${#parts[@]} > 1)); then
    for ((i = 0; i < ${#parts[@]} - 1; i++)); do
      lower=$(printf '%s' "${parts[$i]}" | tr '[:upper:]' '[:lower:]')
      if s19_dir_bad "$lower"; then
        return 0
      fi
    done
  fi
  base="${path##*/}"
  lower=$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    ffmpeg.exe|ffprobe.exe|ffplay.exe) return 0 ;;
  esac
  if [[ "$lower" =~ ^ffmpeg[-_].*\.(tar|tar\.gz|tar\.xz|tar\.bz2|tgz|zip)$ ]]; then
    return 0
  fi
  if [[ "$lower" =~ ^(x264|x265|openh264)[-_].*\.(tar|tar\.gz|tar\.xz|tar\.bz2|tgz|zip)$ ]]; then
    return 0
  fi
  if [[ "$lower" =~ ^lib(x264|openh264|avcodec|avformat|avutil|avdevice|avfilter|swscale|swresample)\.(so|so\..*|a|dll|dylib)$ ]]; then
    return 0
  fi
  if [[ "$lower" =~ ^(x264|openh264|ffmpeg|ffprobe|ffplay)\.(c|h|cc|cpp|s|asm|rs)$ ]]; then
    return 0
  fi
  return 1
}

# Directory component: the project name itself, or a longer name that
# starts with it (`ffmpeg-src`, `x264-master`).
s19_dir_bad() {
  local name="$1" banned
  local projects=(
    ffmpeg ffprobe ffplay x264 x265 openh264
    libavcodec libavformat libavutil libavdevice libavfilter
    libswscale libswresample libav
  )
  for banned in "${projects[@]}"; do
    if [[ "$name" == "$banned" || "$name" == "$banned"* ]]; then
      return 0
    fi
  done
  return 1
}

s19_crate_forbidden() {
  local name="$1" crate
  for crate in "${s19_crates[@]}"; do
    if [[ "$name" == "$crate" ]]; then
      return 0
    fi
  done
  return 1
}

s19_emit_crate() {
  if s19_crate_forbidden "$1"; then
    printf '%s\n' "$1"
  fi
  return 0
}

# Crate names from Cargo.toml / Cargo.lock lines. Exact match only:
# `name = "foo"`, `foo = ...` (including `foo.workspace = true`), a
# table `[dependencies.foo]` (also dev-, build-, workspace and
# target-specific tables), and a rename `x = { package = "foo" }`.
s19_crate_hits() {
  local line key
  while IFS= read -r line || [[ -n "$line" ]]; do
    key=""
    if [[ "$line" =~ ^[[:space:]]*(name|package)[[:space:]]*=[[:space:]]*\"([A-Za-z0-9_+.-]+)\" ]]; then
      key="${BASH_REMATCH[2]}"
      s19_emit_crate "$key"
    elif [[ "$line" =~ ^[[:space:]]*([A-Za-z0-9_+.-]+)[[:space:]]*= ]]; then
      key="${BASH_REMATCH[1]}"
      key="${key%.workspace}"
      s19_emit_crate "$key"
    fi
    if [[ "$line" =~ package[[:space:]]*=[[:space:]]*\"([A-Za-z0-9_+.-]+)\" ]]; then
      s19_emit_crate "${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ ^[[:space:]]*\[((dev|build)-dependencies|dependencies|workspace\.dependencies)\.([A-Za-z0-9_+.-]+)\] ]]; then
      s19_emit_crate "${BASH_REMATCH[3]}"
    elif [[ "$line" =~ ^[[:space:]]*\[[Tt]arget\.[^]]*\.(dev-dependencies|build-dependencies|dependencies)\.([A-Za-z0-9_+.-]+)\] ]]; then
      s19_emit_crate "${BASH_REMATCH[2]}"
    fi
  done | sort -u
}

# encoder.rs must spawn the configured program and clear its environment,
# and must not call an in-process encoder.
s19_spawn_ok() {
  local text="$1"
  [[ "$text" == *'Command::new(ffmpeg)'* ]] || return 1
  [[ "$text" == *'.env_clear()'* ]] || return 1
  if [[ "$text" == *'ffmpeg_sys::'* || "$text" == *'openh264::'* || "$text" == *'x264_encoder_'* ]]; then
    return 1
  fi
  return 0
}

# A binary (a NUL in the file) that carries an encoder marker. A text
# listing of encoder names has no NUL and is not a hit. grep cannot search
# for a NUL (it ends the pattern string), so the NUL check is `tr` against
# `cmp`.
s19_binary_hit() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  if cmp -s <(LC_ALL=C tr -d '\0' < "$file") "$file"; then
    return 1
  fi
  if LC_ALL=C grep -a -q -m 1 -E 'libx264|libopenh264|FFmpeg developers|libavcodec' -- "$file"; then
    return 0
  fi
  return 1
}

s19_expect_path() {
  local path="$1" expect="$2" got
  if s19_path_bad "$path"; then
    got=reject
  else
    got=allow
  fi
  if [[ "$got" != "$expect" ]]; then
    echo "S19 self-test: path '$path' expected $expect, got $got" >&2
    exit 1
  fi
}

s19_self_test() {
  local tmp planted hits good bad linked
  s19_expect_path "third_party/ffmpeg/libavcodec/utils.c" reject
  s19_expect_path "vendor/libx264.so.164" reject
  s19_expect_path "vendor/libopenh264.so" reject
  s19_expect_path "vendor/libavcodec.a" reject
  s19_expect_path "x264/x264.c" reject
  s19_expect_path "openh264/codec/api/wels/codec_api.h" reject
  s19_expect_path "dist/ffmpeg-8.1.tar.xz" reject
  s19_expect_path "dist/openh264-2.4.1.tar.gz" reject
  s19_expect_path "crates/llama-cast/src/encoder.rs" allow
  s19_expect_path "crates/llama-cast/tests/fixtures/ffmpeg-encoders.txt" allow
  s19_expect_path "README.md" allow
  s19_expect_path "scripts/s19-ffmpeg.sh" allow
  s19_expect_path "docs/ffmpeg-notes.md" allow
  s19_expect_path "ffmpeg-src/libavcodec/utils.c" reject
  s19_expect_path "x264-master/common/common.c" reject
  s19_expect_path "openh264-build/src/codec.c" reject
  s19_expect_path "vendor/not-x264/file.c" allow
  s19_expect_path "notes/my-ffmpeg-next/readme.md" allow

  planted=$(printf '%s\n' \
    'name = "ffmpeg-next"' \
    'name = "ffmpeg-next-tools"' \
    'ffmpeg-sys = "0.1"' \
    'my-ffmpeg-next = "0.1"' \
    'x264 = { version = "0.1" }' \
    'rav1e = "0.7"' \
    'openh264.workspace = true' \
    'serde = "1"' \
    'name = "serde"' \
    '[dependencies.ffmpeg-next]' \
    'version = "1"' \
    '[dev-dependencies.dav1d]' \
    '[build-dependencies.x264-sys]' \
    "[target.'cfg(unix)'.dependencies.openh264-sys]" \
    '[target."cfg(windows)".build-dependencies.rav1e]' \
    'helper = { package = "ffmpeg-next" }' \
    'other = { package = "serde" }' \
    '[dependencies.ffmpeg-next-tools]' \
    '[workspace.dependencies.aom-sys]')
  hits="$(s19_crate_hits <<<"$planted")"
  if [[ "$hits" != $'aom-sys\ndav1d\nffmpeg-next\nffmpeg-sys\nopenh264\nopenh264-sys\nrav1e\nx264\nx264-sys' ]]; then
    echo "S19 self-test: crate hits were:" >&2
    printf '%s\n' "$hits" >&2
    exit 1
  fi

  good=$'fn command(ffmpeg: &Path) -> Command {\n    let mut cmd = Command::new(ffmpeg);\n    cmd.env_clear();\n    cmd\n}\n'
  if ! s19_spawn_ok "$good"; then
    echo "S19 self-test: a real spawn was rejected" >&2
    exit 1
  fi
  bad=$'Command::new("ffmpeg")\n.env_clear()\n'
  if s19_spawn_ok "$bad"; then
    echo "S19 self-test: Command::new(\\\"ffmpeg\\\") was allowed" >&2
    exit 1
  fi
  linked=$'Command::new(ffmpeg);\ncmd.env_clear();\nffmpeg_sys::codec();\n'
  if s19_spawn_ok "$linked"; then
    echo "S19 self-test: an in-process encoder was allowed" >&2
    exit 1
  fi
  linked=$'Command::new(ffmpeg);\ncmd.env_clear();\nopenh264::encode();\n'
  if s19_spawn_ok "$linked"; then
    echo "S19 self-test: openh264:: was allowed" >&2
    exit 1
  fi

  tmp=$(mktemp -d)
  printf 'libx264 in a comment\n' > "$tmp/text.txt"
  if s19_binary_hit "$tmp/text.txt"; then
    echo "S19 self-test: a text mention of libx264 was rejected" >&2
    rm -rf -- "$tmp"
    exit 1
  fi
  printf 'hello\0libx264\n' > "$tmp/bin.so"
  if ! s19_binary_hit "$tmp/bin.so"; then
    echo "S19 self-test: a binary carrying libx264 was allowed" >&2
    rm -rf -- "$tmp"
    exit 1
  fi
  printf 'hello\0plain\n' > "$tmp/plain.bin"
  if s19_binary_hit "$tmp/plain.bin"; then
    echo "S19 self-test: a binary with no encoder marker was rejected" >&2
    rm -rf -- "$tmp"
    exit 1
  fi
  rm -rf -- "$tmp"
}

if [[ "${1:-}" == "--self-test" ]]; then
  s19_self_test
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: scripts/s19-ffmpeg.sh [--self-test]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "S19: not a git worktree" >&2
  exit 1
fi

mapfile -t files < <(git ls-files && git ls-files --others --exclude-standard)
paths=""
crates=""
binaries=""
for path in "${files[@]}"; do
  [[ -z "$path" ]] && continue
  if s19_path_bad "$path"; then
    paths+="${path}"$'\n'
  fi
  case "$path" in
    Cargo.toml|*/Cargo.toml|Cargo.lock|*/Cargo.lock)
      if [[ -f "$path" ]]; then
        hit="$(s19_crate_hits < "$path")"
        if [[ -n "$hit" ]]; then
          crates+="${path}:"$'\n'"${hit}"$'\n'
        fi
      fi
      ;;
  esac
  if [[ -f "$path" ]] && s19_binary_hit "$path"; then
    binaries+="${path}"$'\n'
  fi
done

encoder="crates/llama-cast/src/encoder.rs"
if [[ ! -f "$encoder" ]] || ! s19_spawn_ok "$(cat -- "$encoder")"; then
  echo "S19: llama-cast does not spawn an external ffmpeg (Command::new(ffmpeg) and env_clear, no in-process encoder)" >&2
  exit 1
fi

if [[ -n "$paths" || -n "$crates" || -n "$binaries" ]]; then
  echo "S19: ffmpeg, x264, openh264 or another encoder is inside the repository" >&2
  if [[ -n "$paths" ]]; then
    echo "paths:" >&2
    printf '%s' "$paths" >&2
  fi
  if [[ -n "$crates" ]]; then
    echo "cargo:" >&2
    printf '%s' "$crates" >&2
  fi
  if [[ -n "$binaries" ]]; then
    echo "binaries:" >&2
    printf '%s' "$binaries" >&2
  fi
  exit 1
fi

# The round-trip builds its libx264 vector with encoder::ffmpeg_args
# (crates/llama-cast/tests/color_roundtrip.rs) and encodes that vector, then
# the same vector with -vf and -bsf:v removed. Decode assumes a 1080p player
# (BT.709, limited range, back to full-range RGB). A plain decode of the
# untagged stream would pass the old vector, because ffmpeg's default matrix
# for this size is BT.601. The test prints the worst error and fails unless
# the new vector is within 4 per channel and the old one is not.
s19_color_roundtrip() {
  local ffmpeg encoders
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "S19: ffmpeg is not installed; skipping the colour round-trip"
    return 0
  fi
  ffmpeg="$(command -v ffmpeg)"
  # Read the whole list. grep -q would close the pipe early, and pipefail
  # would then treat ffmpeg's SIGPIPE as "no libx264".
  encoders="$("$ffmpeg" -hide_banner -encoders 2>/dev/null || true)"
  if ! grep -q -w libx264 <<<"$encoders"; then
    echo "S19: this ffmpeg has no libx264; skipping the colour round-trip"
    return 0
  fi
  cargo test -p llama-cast --locked --test color_roundtrip -- --ignored --nocapture
}

s19_color_roundtrip
