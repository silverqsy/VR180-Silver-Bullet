#!/bin/bash
# Convert PNG icons to macOS .icns format

set -e

echo "Creating .icns file for macOS..."

# Create iconset directory
ICONSET="AppIcon.iconset"
rm -rf "$ICONSET"
mkdir "$ICONSET"

# Copy and rename icons to Apple's naming convention
cp icon_16x16.png "$ICONSET/icon_16x16.png"
cp icon_32x32.png "$ICONSET/icon_16x16@2x.png"
cp icon_32x32.png "$ICONSET/icon_32x32.png"
cp icon_64x64.png "$ICONSET/icon_32x32@2x.png"
cp icon_128x128.png "$ICONSET/icon_128x128.png"
cp icon_256x256.png "$ICONSET/icon_128x128@2x.png"
cp icon_256x256.png "$ICONSET/icon_256x256.png"
cp icon_512x512.png "$ICONSET/icon_256x256@2x.png"
cp icon_512x512.png "$ICONSET/icon_512x512.png"
cp icon_1024x1024.png "$ICONSET/icon_512x512@2x.png"

# Convert to .icns
iconutil -c icns "$ICONSET" -o icon.icns

# Clean up
rm -rf "$ICONSET"
rm -f icon_*.png

echo "✓ Created icon.icns"
ls -lh icon.icns
