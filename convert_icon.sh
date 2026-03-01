#!/bin/bash
# Convert the downloaded icon.png to .icns format

set -e

echo "Converting icon.png to .icns format..."

SOURCE_ICON="/Users/siyangqi/Downloads/icon.png"

# Create iconset directory
ICONSET="AppIcon.iconset"
rm -rf "$ICONSET"
mkdir "$ICONSET"

# Generate all required sizes using sips
sips -z 16 16 "$SOURCE_ICON" --out "$ICONSET/icon_16x16.png" > /dev/null 2>&1
sips -z 32 32 "$SOURCE_ICON" --out "$ICONSET/icon_16x16@2x.png" > /dev/null 2>&1
sips -z 32 32 "$SOURCE_ICON" --out "$ICONSET/icon_32x32.png" > /dev/null 2>&1
sips -z 64 64 "$SOURCE_ICON" --out "$ICONSET/icon_32x32@2x.png" > /dev/null 2>&1
sips -z 128 128 "$SOURCE_ICON" --out "$ICONSET/icon_128x128.png" > /dev/null 2>&1
sips -z 256 256 "$SOURCE_ICON" --out "$ICONSET/icon_128x128@2x.png" > /dev/null 2>&1
sips -z 256 256 "$SOURCE_ICON" --out "$ICONSET/icon_256x256.png" > /dev/null 2>&1
sips -z 512 512 "$SOURCE_ICON" --out "$ICONSET/icon_256x256@2x.png" > /dev/null 2>&1
sips -z 512 512 "$SOURCE_ICON" --out "$ICONSET/icon_512x512.png" > /dev/null 2>&1
sips -z 1024 1024 "$SOURCE_ICON" --out "$ICONSET/icon_512x512@2x.png" > /dev/null 2>&1

echo "✓ Generated all icon sizes"

# Convert to .icns
iconutil -c icns "$ICONSET" -o icon.icns

# Clean up
rm -rf "$ICONSET"

echo "✓ Created icon.icns"
ls -lh icon.icns
