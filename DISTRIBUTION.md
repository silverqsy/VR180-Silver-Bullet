# VR180 Processor - Distribution Guide

## What's Included

After building, you'll have:

### macOS
- `dist/VR180 Processor.app` - Double-click to run (86MB)

### Windows
- `dist/VR180Processor/` folder containing:
  - `VR180Processor.exe` - Main executable
  - Supporting libraries and dependencies (~150MB total)

## Prerequisites for End Users

**IMPORTANT:** Users must have FFmpeg installed on their system.

### macOS Users
Install FFmpeg via Homebrew:
```bash
brew install ffmpeg
```

Or download from: https://ffmpeg.org/download.html

### Windows Users
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH

**Quick PATH setup (Windows):**
```cmd
setx PATH "%PATH%;C:\ffmpeg\bin"
```
(Restart terminal after this)

## Distribution Options

### Option 1: Simple ZIP Distribution

**macOS:**
```bash
cd dist
zip -r "VR180-Processor-macOS.zip" "VR180 Processor.app"
```

**Windows:**
```cmd
cd dist
# Use 7-Zip or WinRAR to compress the VR180Processor folder
```

Pros:
- Simple and quick
- Small download size

Cons:
- Users must install FFmpeg separately

### Option 2: Bundle FFmpeg (Recommended for ease of use)

**macOS:**
```bash
# Copy FFmpeg to app bundle
cp $(which ffmpeg) "dist/VR180 Processor.app/Contents/MacOS/"
cp $(which ffprobe) "dist/VR180 Processor.app/Contents/MacOS/"

# Then zip
cd dist
zip -r "VR180-Processor-macOS-bundled.zip" "VR180 Processor.app"
```

**Windows:**
1. Download FFmpeg static build from https://www.gyan.dev/ffmpeg/builds/
2. Extract `ffmpeg.exe` and `ffprobe.exe`
3. Copy them to `dist\VR180Processor\`
4. Compress the entire `VR180Processor` folder

Pros:
- Works immediately without user setup
- Better user experience

Cons:
- Larger download (~200MB instead of ~100MB)

### Option 3: Professional Installer

**macOS DMG:**
```bash
# Install create-dmg
brew install create-dmg

# Create professional DMG
create-dmg \
  --volname "VR180 Processor" \
  --volicon "icon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "VR180 Processor.app" 200 190 \
  --hide-extension "VR180 Processor.app" \
  --app-drop-link 600 185 \
  "VR180-Processor-Installer.dmg" \
  "dist/VR180 Processor.app"
```

**Windows Installer (using Inno Setup):**
1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create installer script (example below)
3. Compile to create a single .exe installer

Example Inno Setup script:
```ini
[Setup]
AppName=VR180 Processor
AppVersion=1.0.0
DefaultDirName={pf}\VR180Processor
DefaultGroupName=VR180 Processor
OutputBaseFilename=VR180-Processor-Setup

[Files]
Source: "dist\VR180Processor\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\VR180 Processor"; Filename: "{app}\VR180Processor.exe"
Name: "{commondesktop}\VR180 Processor"; Filename: "{app}\VR180Processor.exe"

[Run]
Filename: "{app}\VR180Processor.exe"; Description: "Launch VR180 Processor"; Flags: postinstall nowait skipifsilent
```

## What to Include with Distribution

Create a `README.txt` for end users:

```
VR180 Processor v1.0.0

INSTALLATION:
1. Extract all files to a folder
2. Install FFmpeg (required):
   - macOS: brew install ffmpeg
   - Windows: Download from https://ffmpeg.org/download.html

USAGE:
- macOS: Double-click "VR180 Processor.app"
- Windows: Double-click "VR180Processor.exe"

FEATURES:
- Fix split-eye VR180 videos
- Adjust yaw, pitch, roll for both eyes
- Real-time preview with multiple viewing modes
- Export to H.265 or ProRes

SUPPORT:
Report issues at: https://github.com/yourusername/vr180_processor

LICENSE:
MIT License - See LICENSE file
```

## Code Signing (Important for macOS)

For distribution outside the App Store, sign your app to avoid Gatekeeper warnings:

```bash
# Self-sign (for testing)
codesign --force --deep --sign - "dist/VR180 Processor.app"

# With Apple Developer ID
codesign --force --deep --sign "Developer ID Application: Your Name" "dist/VR180 Processor.app"

# Notarize (required for macOS 10.15+)
xcrun notarytool submit "VR180-Processor-macOS.zip" --apple-id "your@email.com" --team-id "TEAMID" --wait
xcrun stapler staple "dist/VR180 Processor.app"
```

## Testing Checklist

Before distributing, test on a clean system:

### macOS
- [ ] App opens without errors
- [ ] FFmpeg is detected (or bundled version works)
- [ ] Can load and preview video
- [ ] Can process and export video
- [ ] No Gatekeeper warnings (if signed)

### Windows
- [ ] App opens without errors
- [ ] FFmpeg is detected (or bundled version works)
- [ ] Can load and preview video
- [ ] Can process and export video
- [ ] No missing DLL errors

## File Size Reference

| Configuration | macOS | Windows |
|---------------|-------|---------|
| Without FFmpeg | ~86MB | ~150MB |
| With FFmpeg bundled | ~200MB | ~250MB |
| DMG/Installer | ~90MB | ~80MB |

## License Compliance

When distributing, ensure compliance with:

1. **PyQt6 License:**
   - GPL v3 (open source)
   - OR purchase commercial license from Riverbank Computing

2. **FFmpeg License:**
   - LGPL (if using shared libraries)
   - Include FFmpeg license text if bundling

3. **Your Code:**
   - MIT License (as specified in your README)

Include a `LICENSES` folder with all relevant license files.

## Upload Platforms

Consider hosting on:
- GitHub Releases (free, great for open source)
- Your own website
- SourceForge (for broader reach)

Example GitHub release command:
```bash
gh release create v1.0.0 \
  --title "VR180 Processor v1.0.0" \
  --notes "Initial release" \
  VR180-Processor-macOS.zip \
  VR180-Processor-Windows.zip
```
