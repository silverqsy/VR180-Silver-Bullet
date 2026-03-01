# Building VR180 Processor

This document explains how to build distributable packages for macOS and Windows.

## Prerequisites

### All Platforms
- Python 3.9 or higher
- FFmpeg installed and in PATH

### macOS
- macOS 10.13 or higher
- Xcode Command Line Tools

### Windows
- Windows 10 or higher
- Visual C++ Redistributable (for PyInstaller)

## Installation

First, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Building for macOS

### Quick Build
Run the build script:
```bash
./build_mac.sh
```

### Manual Build
```bash
pyinstaller --clean vr180_processor.spec
```

### Output
The macOS application bundle will be created at:
```
dist/VR180 Processor.app
```

### Running the App
```bash
open "dist/VR180 Processor.app"
```

### Distribution
To distribute the app:
1. Compress the app: `cd dist && zip -r "VR180 Processor.zip" "VR180 Processor.app"`
2. Users must have FFmpeg installed: `brew install ffmpeg`

**Note:** For distribution on macOS, you may want to code-sign the application to avoid Gatekeeper warnings.

## Building for Windows

### Quick Build
Run the build script:
```cmd
build_windows.bat
```

### Manual Build
```cmd
pyinstaller --clean --onedir --windowed ^
    --name "VR180Processor" ^
    --hidden-import PyQt6.QtCore ^
    --hidden-import PyQt6.QtGui ^
    --hidden-import PyQt6.QtWidgets ^
    vr180_gui.py
```

### Output
The Windows application will be created at:
```
dist\VR180Processor\VR180Processor.exe
```

### Distribution
To distribute the app:
1. Compress the entire `dist\VR180Processor\` folder
2. Users must have FFmpeg installed and in PATH
3. Download FFmpeg from: https://ffmpeg.org/download.html

## FFmpeg Dependency

**IMPORTANT:** This application requires FFmpeg to be installed on the system where it runs.

### macOS
```bash
brew install ffmpeg
```

### Windows
1. Download from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to PATH environment variable

### Alternative: Bundle FFmpeg (Advanced)

To bundle FFmpeg with your application:

#### macOS
```bash
# Copy ffmpeg binaries to the app bundle
cp $(which ffmpeg) "dist/VR180 Processor.app/Contents/MacOS/"
cp $(which ffprobe) "dist/VR180 Processor.app/Contents/MacOS/"
```

#### Windows
1. Download FFmpeg static build
2. Copy `ffmpeg.exe` and `ffprobe.exe` to `dist\VR180Processor\`

**Note:** Bundling FFmpeg significantly increases the app size (~100MB).

## Testing the Build

### macOS
```bash
# Test the app
open "dist/VR180 Processor.app"

# Check FFmpeg is accessible
"dist/VR180 Processor.app/Contents/MacOS/VR180Processor" &
# Then test by loading a video
```

### Windows
```cmd
# Run the app
dist\VR180Processor\VR180Processor.exe
```

## Troubleshooting

### macOS: "App is damaged" error
This happens when the app is not code-signed. Users can bypass this by:
```bash
xattr -cr "VR180 Processor.app"
```

Or you can code-sign during build:
```bash
codesign --force --deep --sign - "dist/VR180 Processor.app"
```

### Windows: Missing DLL errors
Ensure all Visual C++ redistributables are installed. PyInstaller should bundle these automatically.

### "FFmpeg not found" errors
Make sure FFmpeg is installed and accessible in PATH, or bundle it with the application.

## File Size Optimization

The default build creates a ~150-200MB application. To reduce size:

1. Use `--onefile` instead of `--onedir` (slower startup, but single executable)
2. Exclude unnecessary packages
3. Use UPX compression (already enabled in spec file)

## Creating an Installer

### macOS DMG
```bash
# Install create-dmg
brew install create-dmg

# Create DMG
create-dmg \
  --volname "VR180 Processor" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --app-drop-link 600 185 \
  "VR180-Processor-Installer.dmg" \
  "dist/VR180 Processor.app"
```

### Windows Installer
Use Inno Setup or NSIS to create a Windows installer:

1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create a script that packages `dist\VR180Processor\`
3. Include instructions for FFmpeg installation

## License

When distributing, ensure you comply with:
- PyQt6 GPL license (or purchase commercial license)
- FFmpeg LGPL/GPL license
- Your own code's MIT license
