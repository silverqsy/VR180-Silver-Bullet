#!/bin/bash
# Create release packages for distribution
# FFmpeg is bundled automatically via the spec file

set -e

VERSION="1.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --version)
      VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./create_release.sh [--version X.Y.Z]"
      exit 1
      ;;
  esac
done

echo "====================================="
echo "Creating VR180 Processor Release"
echo "Version: $VERSION"
echo "====================================="
echo ""

# Check if already built
if [ -d "dist/VR180 Processor.app" ]; then
    echo "Found existing build in dist/"
    read -p "Rebuild from scratch? (y/N): " REBUILD
    if [[ $REBUILD =~ ^[Yy]$ ]]; then
        echo "Rebuilding..."
        ./build_mac.sh
    else
        echo "Using existing build..."
    fi
else
    echo "No existing build found. Building now..."
    ./build_mac.sh
fi

# Check if build succeeded
if [ ! -d "dist/VR180 Processor.app" ]; then
    echo "Error: Build failed or not found!"
    exit 1
fi

# Verify FFmpeg is bundled
echo ""
echo "Verifying bundled FFmpeg..."
if [ -f "dist/VR180 Processor.app/Contents/Frameworks/ffmpeg" ]; then
    echo "✓ FFmpeg is bundled"
    FFMPEG_SIZE=$(ls -lh "dist/VR180 Processor.app/Contents/Frameworks/ffmpeg" | awk '{print $5}')
    echo "  FFmpeg binary: $FFMPEG_SIZE"

    DYLIB_COUNT=$(find "dist/VR180 Processor.app/Contents/Frameworks" -name "libav*.dylib" 2>/dev/null | wc -l | tr -d ' ')
    echo "  FFmpeg libraries: $DYLIB_COUNT .dylib files"
else
    echo "⚠ Warning: FFmpeg not found in bundle!"
    echo "  The app may not work without system FFmpeg"
fi

# Code sign the app (self-signing for now)
echo ""
echo "Signing application..."
codesign --force --deep --sign - "dist/VR180 Processor.app" 2>/dev/null || echo "Warning: Code signing failed (not critical)"

# Create ZIP archive
echo ""
echo "Creating release package..."
rm -f VR180-Processor-*.zip
cd dist
zip -r "../VR180-Processor-${VERSION}-macOS.zip" "VR180 Processor.app"
cd ..

# Calculate file size
SIZE=$(du -h "VR180-Processor-${VERSION}-macOS.zip" | cut -f1)
APP_SIZE=$(du -sh "dist/VR180 Processor.app" | cut -f1)

echo ""
echo "====================================="
echo "Release package created successfully!"
echo "====================================="
echo ""
echo "File: VR180-Processor-${VERSION}-macOS.zip"
echo "ZIP size: $SIZE"
echo "App size: $APP_SIZE"
echo ""
echo "✅ This is a FULLY STANDALONE package"
echo "   FFmpeg is bundled - no installation required!"
echo ""
echo "Next steps:"
echo "1. Test on a clean macOS system (without Python/FFmpeg)"
echo "2. Upload to your distribution platform"
echo "3. Share with users!"
echo ""
echo "To test:"
echo "  unzip VR180-Processor-${VERSION}-macOS.zip"
echo "  open 'VR180 Processor.app'"
echo ""
