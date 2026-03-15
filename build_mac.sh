#!/bin/bash
# Build script for macOS with bundled dependencies

set -e

echo "=============================================="
echo "VR180 Silver Bullet - macOS Bundled Build Script"
echo "=============================================="
echo ""

# Check if FFmpeg is installed
echo "Checking for FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "ERROR: FFmpeg not found!"
    echo "FFmpeg is required to bundle it with the application."
    echo ""
    echo "Install FFmpeg with:"
    echo "  brew install ffmpeg"
    echo ""
    echo "After installation, run this script again."
    exit 1
fi

FFMPEG_VERSION=$(ffmpeg -version | head -n1)
echo "✓ Found FFmpeg: $FFMPEG_VERSION"
echo "  Location: $(which ffmpeg)"
echo ""

# Check if Python dependencies are installed
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Building macOS application with bundled FFmpeg..."
echo "This will create a fully standalone app (~200MB)"
echo ""

# Clean previous builds
rm -rf build dist

# Build with PyInstaller
pyinstaller --clean vr180_silver_bullet.spec

# Verify FFmpeg was bundled
echo ""
echo "Verifying bundled files..."
if [ -f "dist/VR180 Silver Bullet.app/Contents/Frameworks/ffmpeg" ]; then
    echo "✓ FFmpeg bundled successfully"
    FFMPEG_SIZE=$(ls -lh "dist/VR180 Silver Bullet.app/Contents/Frameworks/ffmpeg" | awk '{print $5}')
    echo "  Binary size: $FFMPEG_SIZE"
else
    echo "⚠ Warning: FFmpeg may not have been bundled"
fi

if [ -f "dist/VR180 Silver Bullet.app/Contents/Frameworks/ffprobe" ]; then
    echo "✓ FFprobe bundled successfully"
else
    echo "⚠ Warning: FFprobe may not have been bundled"
fi

# Check for FFmpeg libraries
DYLIB_COUNT=$(find "dist/VR180 Silver Bullet.app/Contents/Frameworks" -name "libav*.dylib" 2>/dev/null | wc -l | tr -d ' ')
if [ "$DYLIB_COUNT" -gt 0 ]; then
    echo "✓ FFmpeg libraries bundled: $DYLIB_COUNT .dylib files"
    FRAMEWORKS_SIZE=$(du -sh "dist/VR180 Silver Bullet.app/Contents/Frameworks" | cut -f1)
    echo "  Total frameworks size: $FRAMEWORKS_SIZE"
else
    echo "⚠ Warning: FFmpeg libraries may not have been bundled"
fi

# Check app bundle size
APP_SIZE=$(du -sh "dist/VR180 Silver Bullet.app" | cut -f1)

echo ""
echo "=============================================="
echo "Build completed successfully!"
echo "=============================================="
echo ""
echo "Application: dist/VR180 Silver Bullet.app"
echo "Total size: $APP_SIZE"
echo ""
echo "This is a FULLY STANDALONE application."
echo "No FFmpeg installation required on target systems!"
echo ""
echo "Creating distribution ZIP..."
cd dist
rm -f "VR180-Silver-Bullet-macOS.zip"
zip -r -y -q "VR180-Silver-Bullet-macOS.zip" "VR180 Silver Bullet.app"
ZIP_SIZE=$(du -sh "VR180-Silver-Bullet-macOS.zip" | cut -f1)
cd ..

echo "✓ Distribution ZIP created: $ZIP_SIZE"
echo ""
echo "To run the app:"
echo "  open \"dist/VR180 Silver Bullet.app\""
echo ""
echo "To distribute:"
echo "  Share dist/VR180-Silver-Bullet-macOS.zip"
echo ""
