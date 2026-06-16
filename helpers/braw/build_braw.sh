#!/bin/bash
# Build braw_helper from Blackmagic RAW SDK.
#
# Output lands in helpers/bin/ (alongside the Swift helpers), which is
# .gitignore'd. The source lives in helpers/braw/braw_helper.cpp and is
# committed. vr180-pipeline::helpers::locate_helper finds the binary at
# runtime — either in helpers/bin/ during dev or next to the
# vr180-render exe in a release bundle.
#
# Requires the Blackmagic RAW SDK installed via the official installer
# from https://www.blackmagicdesign.com/support/ (macOS Mac → "Blackmagic
# RAW Speed Test" download). The SDK is NOT redistributable, so we link
# at build time against the user's installed copy.

set -e

SDK_BASE="/Applications/Blackmagic RAW/Blackmagic RAW SDK/Mac"
SDK_INCLUDE="$SDK_BASE/Include"
SDK_LIBS="$SDK_BASE/Libraries"

if [ ! -d "$SDK_LIBS/BlackmagicRawAPI.framework" ]; then
    echo "ERROR: Blackmagic RAW SDK not found at $SDK_BASE"
    echo ""
    echo "Install it from:"
    echo "  https://www.blackmagicdesign.com/support/"
    echo ""
    echo "Look for 'Blackmagic RAW' under Capture and Playback Software."
    echo "The SDK ships inside the 'Blackmagic RAW' installer package."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HELPERS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$HELPERS_DIR/bin"
mkdir -p "$BIN_DIR"

echo "Compiling braw_helper..."
clang++ -std=c++17 -O2 \
    "$SCRIPT_DIR/braw_helper.cpp" \
    -I"$SDK_INCLUDE" \
    -F"$SDK_LIBS" \
    -rpath "$SDK_LIBS" \
    -framework BlackmagicRawAPI \
    -framework CoreFoundation \
    -framework CoreServices \
    -include "$SDK_INCLUDE/BlackmagicRawAPIDispatch.cpp" \
    -o "$BIN_DIR/braw_helper"

echo "Build successful: $BIN_DIR/braw_helper"
