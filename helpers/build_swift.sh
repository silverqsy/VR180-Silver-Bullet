#!/usr/bin/env bash
# Build the macOS Swift helpers shipped alongside vr180-render.
#
# Usage: ./helpers/build_swift.sh
# Output: helpers/bin/{mvhevc_encode,vt_denoise,apac_encode}
#
# The Rust pipeline locates these via:
#   1. helpers/bin/<name> (development)
#   2. <dir of running vr180-render exe>/<name> (release)
#
# See `crates/vr180-pipeline/src/helpers.rs::locate_helper`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/swift"
OUT_DIR="$SCRIPT_DIR/bin"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Swift helpers are macOS-only — skipping." >&2
    exit 0
fi

mkdir -p "$OUT_DIR"

build_helper() {
    local name="$1"
    shift
    local frameworks=("$@")
    local src="$SRC_DIR/$name.swift"
    local out="$OUT_DIR/$name"

    if [[ ! -f "$src" ]]; then
        echo "✗ source not found: $src" >&2
        return 1
    fi

    local fw_args=()
    for fw in "${frameworks[@]}"; do
        fw_args+=("-framework" "$fw")
    done

    echo "→ building $name"
    swiftc -O -o "$out" "$src" "${fw_args[@]}"
    echo "  ✓ $out"
}

build_helper mvhevc_encode \
    AVFoundation VideoToolbox CoreMedia CoreVideo Accelerate

build_helper vt_denoise \
    AVFoundation VideoToolbox CoreMedia CoreVideo

build_helper apac_encode \
    AVFoundation CoreMedia AudioToolbox

echo
echo "✓ All helpers built into $OUT_DIR"
