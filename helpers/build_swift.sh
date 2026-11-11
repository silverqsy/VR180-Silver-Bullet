#!/usr/bin/env bash
# Build the macOS Swift helpers shipped alongside vr180-gui.
#
# Usage:
#   ./helpers/build_swift.sh                 build all into helpers/bin/
#   ./helpers/build_swift.sh --install DIR   build, then copy the helpers into DIR
#                                            (e.g. "dist/VR180 Silver Bullet.app/Contents/MacOS")
#   ./helpers/build_swift.sh --check DIR     verify DIR's helpers were built from the
#                                            CURRENT sources (exit 1 if any is stale/missing)
#   ./helpers/build_swift.sh --hash NAME     print the source hash of one helper
#
# Every helper answers `--version` with "<name> <sha256 of its .swift source>":
# the source is compiled as main.swift next to a generated version.swift that
# defines HELPER_SOURCE_HASH, under -D HELPER_VERSION_EMBEDDED. That is what
# `--check` compares, so a release bundle can never silently carry a helper
# built from older sources (the 2026-10-10 APAC fix shipped in source only
# until the bundle's copy was rebuilt — hence this guard).
#
# The Rust pipeline locates the binaries via:
#   1. helpers/bin/<name> (development)
#   2. <dir of the running vr180-gui exe>/<name> (release bundle)
# See `crates/vr180-pipeline/src/helpers.rs::locate_helper`. `crates/vr180-gui/
# build.rs` runs this script (plain build) on macOS whenever a source changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/swift"
OUT_DIR="$SCRIPT_DIR/bin"
HELPERS=(mvhevc_encode vt_denoise apac_encode)

frameworks_for() {
    case "$1" in
        mvhevc_encode) echo AVFoundation VideoToolbox CoreMedia CoreVideo ;;
        vt_denoise)    echo AVFoundation VideoToolbox CoreMedia CoreVideo ;;
        apac_encode)   echo AVFoundation CoreMedia AudioToolbox ;;
        *) echo "unknown helper: $1" >&2; return 1 ;;
    esac
}

# Oldest macOS a helper must start on. A helper without one is built for the
# build machine's own macOS: its `#available` checks compile away and symbols
# newer than the user's system are bound at launch, so it cannot start there.
min_macos_for() {
    case "$1" in
        mvhevc_encode) echo 14.0 ;;
        *) echo "" ;;
    esac
}

source_hash() {
    shasum -a 256 "$SRC_DIR/$1.swift" | cut -d' ' -f1
}

build_helper() {
    local name="$1"
    local src="$SRC_DIR/$name.swift"
    local out="$OUT_DIR/$name"
    [[ -f "$src" ]] || { echo "✗ source not found: $src" >&2; return 1; }
    local hash; hash="$(source_hash "$name")"
    local tmp; tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    cp "$src" "$tmp/main.swift"
    printf 'let HELPER_SOURCE_HASH = "%s"\n' "$hash" > "$tmp/version.swift"
    local fw_args=()
    for fw in $(frameworks_for "$name"); do fw_args+=("-framework" "$fw"); done
    local target_args=()
    local min_macos; min_macos="$(min_macos_for "$name")"
    [[ -n "$min_macos" ]] && target_args=(-target "$(uname -m)-apple-macosx$min_macos")
    echo "→ building $name ($hash)"
    # (`${a[@]+…}`: an empty array is "unbound" to the system bash 3.2 under `set -u`.)
    swiftc -O -D HELPER_VERSION_EMBEDDED ${target_args[@]+"${target_args[@]}"} \
        -o "$out" "$tmp/main.swift" "$tmp/version.swift" "${fw_args[@]}"
    local reported; reported="$("$out" --version | awk '{print $2}')"
    [[ "$reported" == "$hash" ]] || { echo "✗ $name reports $reported, expected $hash" >&2; return 1; }
    echo "  ✓ $out"
}

check_dir() {
    local dir="$1" ok=1
    for name in "${HELPERS[@]}"; do
        local want; want="$(source_hash "$name")"
        local have="missing"
        if [[ -x "$dir/$name" ]]; then
            # (an old helper without --version exits non-zero: keep going)
            have="$("$dir/$name" --version 2>/dev/null | awk 'NR==1{print $2}' || true)"
            [[ -n "$have" ]] || have="no --version (built before 2026-10-10)"
        fi
        if [[ "$have" == "$want" ]]; then
            echo "  ✓ $name is current ($want)"
        else
            echo "  ✗ $name is STALE: bundle has '$have', sources are $want"
            ok=0
        fi
    done
    [[ $ok -eq 1 ]]
}

mode="${1:-build}"
case "$mode" in
    --hash) source_hash "$2"; exit 0 ;;
    --check)
        [[ -d "${2:-}" ]] || { echo "usage: $0 --check DIR" >&2; exit 2; }
        echo "Checking helpers in $2 against $SRC_DIR:"
        check_dir "$2"
        exit $? ;;
esac

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Swift helpers are macOS-only — skipping." >&2
    exit 0
fi
command -v swiftc >/dev/null || { echo "✗ swiftc not found (install the Xcode command line tools)" >&2; exit 1; }

mkdir -p "$OUT_DIR"
for name in "${HELPERS[@]}"; do build_helper "$name"; done
echo
echo "✓ All helpers built into $OUT_DIR"

if [[ "$mode" == "--install" ]]; then
    dest="${2:-}"
    [[ -d "$dest" ]] || { echo "usage: $0 --install DIR (DIR must exist)" >&2; exit 2; }
    for name in "${HELPERS[@]}"; do cp -f "$OUT_DIR/$name" "$dest/$name"; done
    echo "✓ Installed into $dest"
    check_dir "$dest"
fi
