# VR180 Silver Bullet - Windows Build Instructions

## Quick Build Guide

### Prerequisites
1. **Python 3.11 or 3.12** - https://www.python.org/downloads/windows/
   - ✓ Check "Add Python to PATH" during installation

2. **FFmpeg** - https://www.gyan.dev/ffmpeg/builds/
   - Download `ffmpeg-release-essentials.zip`
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to PATH

### Install Dependencies
```cmd
pip install PyQt6 numpy Pillow pyinstaller
```

### Build
```cmd
pyinstaller --clean vr180_processor.spec
```

### Run
```
dist\VR180 Silver Bullet\VR180Processor.exe
```

## What's Different on Windows

### ✅ Everything Works Except Vision Pro
- All VR180 adjustments
- H.265 encoding (8-bit & 10-bit)
- ProRes encoding (all profiles)
- Hardware acceleration (NVENC)
- Color grading & LUT support
- All preview modes
- YouTube VR180 metadata

### ❌ Not Available (macOS Only)
- Vision Pro mode selector is hidden on Windows
- No hvc1 tag option
- No MV-HEVC conversion

The Windows version is identical to macOS except the Vision Pro UI is automatically hidden.

## Files Included
- `icon.ico` - Windows application icon (automatically used)
- `icon.png` - Source icon (967 KB)

## Distributing
Zip the entire `dist\VR180 Silver Bullet\` folder - users just run the .exe!
