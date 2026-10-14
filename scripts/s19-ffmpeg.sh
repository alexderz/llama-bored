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

# The new vector matches crates/llama-cast/src/encoder.rs for libx264 at the
# default settings (1920x1080, fps 2, 4000 kbit/s, keyframe 1 s, output 10 fps).
# The old vector is that command without -vf and without -bsf:v. Decode assumes
# a 1080p player (BT.709, limited range, back to full-range RGB). A plain decode
# of the untagged stream would pass the old vector, because ffmpeg's default
# matrix for this size is BT.601. Patches are 32x32 (4:2:0) and are the ramp
# stops in llama_core::color (ACT_STOPS, then the BB_STOPS that are not L6).
# Tolerance is 4 levels per channel on the inner 16x16 of each patch.
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
  if ! command -v python3 >/dev/null 2>&1; then
    echo "S19: python3 is not installed; skipping the colour round-trip"
    return 0
  fi
  # The encode and decode run in a subshell so a failure still removes the
  # frames. The trap is this subshell's EXIT, not the caller's.
  (
    set -euo pipefail
    tmp="$(mktemp -d)"
    trap 'rm -rf -- "$tmp"' EXIT
  cat >"$tmp/roundtrip.py" <<'PY'
import sys

W, H = 1920, 1080
# llama_core::color ACT_STOPS, then BB_STOPS after the shared L6 stop.
STOPS = (
    (0x14, 0x28, 0xD8),
    (0x3A, 0x1E, 0xF0),
    (0x7A, 0x1F, 0xE8),
    (0xC2, 0x1C, 0xC8),
    (0xF2, 0x1E, 0x5A),
    (0xFF, 0x2A, 0x14),
    (0xFF, 0x5A, 0x12),
    (0xFF, 0x8A, 0x1E),
    (0xFF, 0xB0, 0x30),
    (0xFF, 0xD0, 0x50),
)
PATCH = 32
GAP = 64
ORIGIN = 64
INNER = 8

def paint():
    frame = bytearray(W * H * 3)
    for i, (r, g, b) in enumerate(STOPS):
        x0 = ORIGIN + i * GAP
        px = bytes((r, g, b))
        for y in range(ORIGIN, ORIGIN + PATCH):
            row = (y * W + x0) * 3
            frame[row : row + PATCH * 3] = px * PATCH
    return frame

def worst(blob):
    if len(blob) != W * H * 3:
        sys.stderr.write(f"decoded frame is {len(blob)} bytes, want {W * H * 3}\n")
        return None
    worst_err = 0
    for i, (r, g, b) in enumerate(STOPS):
        x0 = ORIGIN + i * GAP + INNER
        y0 = ORIGIN + INNER
        for y in range(y0, y0 + PATCH - 2 * INNER):
            for x in range(x0, x0 + PATCH - 2 * INNER):
                o = (y * W + x) * 3
                worst_err = max(
                    worst_err,
                    abs(blob[o] - r),
                    abs(blob[o + 1] - g),
                    abs(blob[o + 2] - b),
                )
    return worst_err

mode = sys.argv[1]
if mode == "write":
    with open(sys.argv[2], "wb") as out:
        out.write(paint())
elif mode == "score":
    with open(sys.argv[2], "rb") as f:
        new = worst(f.read())
    with open(sys.argv[3], "rb") as f:
        old = worst(f.read())
    if new is None or old is None:
        sys.exit(1)
    print(f"{new} {old}")
else:
    sys.stderr.write(f"unknown mode {mode}\n")
    sys.exit(2)
PY
  python3 "$tmp/roundtrip.py" write "$tmp/frame.rgb"
  # New: BT.709 limited, then the VUI tags. -pix_fmt matches the filter output.
  "$ffmpeg" -y -hide_banner -loglevel error \
    -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i "$tmp/frame.rgb" \
    -vf 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p' \
    -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p \
    -r 10 -g 10 -bf 0 \
    -b:v 4000k -minrate 4000k -maxrate 4000k -bufsize 4000k -x264-params nal-hrd=cbr \
    -bsf:v 'h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0' \
    -frames:v 1 -flush_packets 1 -pat_period 0.1 -f mpegts "$tmp/new.ts"
  # Old: the same command with neither the matrix filter nor the tags.
  "$ffmpeg" -y -hide_banner -loglevel error \
    -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 2 -i "$tmp/frame.rgb" \
    -c:v libx264 -preset veryfast -tune stillimage,zerolatency -pix_fmt yuv420p \
    -r 10 -g 10 -bf 0 \
    -b:v 4000k -minrate 4000k -maxrate 4000k -bufsize 4000k -x264-params nal-hrd=cbr \
    -frames:v 1 -flush_packets 1 -pat_period 0.1 -f mpegts "$tmp/old.ts"
    # -i before the output filter: a 1080p player reads the stream as BT.709
    # limited and expands it to full-range RGB.
    "$ffmpeg" -y -hide_banner -loglevel error -i "$tmp/new.ts" \
      -vf 'scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24' \
      -frames:v 1 -f rawvideo -pix_fmt rgb24 "$tmp/new.rgb"
    "$ffmpeg" -y -hide_banner -loglevel error -i "$tmp/old.ts" \
      -vf 'scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24' \
      -frames:v 1 -f rawvideo -pix_fmt rgb24 "$tmp/old.rgb"
    read -r new_worst old_worst < <(python3 "$tmp/roundtrip.py" score "$tmp/new.rgb" "$tmp/old.rgb")
    tol=4
    echo "S19: colour round-trip new worst ${new_worst} (tolerance ${tol} per channel), old worst ${old_worst}"
    if (( new_worst > tol )); then
      echo "S19: the BT.709 vector round-trips worse than ${tol}" >&2
      exit 1
    fi
    if (( old_worst <= tol )); then
      echo "S19: the old argument vector still round-trips within ${tol}" >&2
      exit 1
    fi
    if command -v ffprobe >/dev/null 2>&1; then
      tags="$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=color_range,color_space,color_transfer,color_primaries \
        -of default=noprint_wrappers=1 "$tmp/new.ts")"
      for want in color_range=tv color_space=bt709 color_transfer=bt709 color_primaries=bt709; do
        if ! grep -F -x -q -- "$want" <<<"$tags"; then
          echo "S19: new stream is missing ${want}" >&2
          printf '%s\n' "$tags" >&2
          exit 1
        fi
      done
    else
      echo "S19: ffprobe is not installed; skipping the tag check"
    fi
  )
}

s19_color_roundtrip
