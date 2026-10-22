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
OBJ="$(mktemp -d)"
# The helper itself is plain C++ (the SDK dispatch source is included into
# this one translation unit only); the zero-copy pool is Objective-C++
# (Metal + IOSurface) under ARC.
clang++ -std=c++17 -O2 -c \
    "$SCRIPT_DIR/braw_helper.cpp" \
    -I"$SDK_INCLUDE" \
    -include "$SDK_INCLUDE/BlackmagicRawAPIDispatch.cpp" \
    -o "$OBJ/braw_helper.o"
clang++ -std=c++17 -O2 -fobjc-arc -c \
    "$SCRIPT_DIR/braw_zc.mm" \
    -o "$OBJ/braw_zc.o"
clang++ "$OBJ/braw_helper.o" "$OBJ/braw_zc.o" \
    -F"$SDK_LIBS" \
    -rpath "$SDK_LIBS" \
    -framework BlackmagicRawAPI \
    -framework CoreFoundation \
    -framework CoreServices \
    -framework Foundation \
    -framework Metal \
    -framework IOSurface \
    -o "$BIN_DIR/braw_helper"
rm -rf "$OBJ"

echo "Build successful: $BIN_DIR/braw_helper"
