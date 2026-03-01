# VR180 Processor - Build Instructions

## macOS Build (Completed ✓)

The macOS build has been completed successfully!

### Build Output
- **Location**: `dist/VR180 Processor.app`
- **Size**: 156 MB (app bundle) / 209 MB (zipped)
- **Distribution Package**: `dist/VR180-Processor-macOS.zip`
- **FFmpeg**: Bundled (no external dependencies required)
- **Architecture**: Universal (runs on both Intel and Apple Silicon)

### To Run on macOS
```bash
open "dist/VR180 Processor.app"
```

### To Distribute
Share the file: `dist/VR180-Processor-macOS.zip`

Users can simply:
1. Download and unzip the file
2. Drag "VR180 Processor.app" to Applications folder
3. Double-click to run

---

## Windows Build Instructions

To build the Windows version, you need to run the build script on a Windows machine.

### Prerequisites (Windows)

1. **Python 3.8 or newer**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **FFmpeg for Windows**
   - Download from: https://www.gyan.dev/ffmpeg/builds/
   - Download the "ffmpeg-release-full.7z" file
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to your PATH environment variable

### Build Steps (Windows)

1. Open Command Prompt or PowerShell

2. Navigate to the project directory:
   ```cmd
   cd path\to\vr180_processor
   ```

3. Run the build script:
   ```cmd
   build_windows.bat
   ```

### Windows Build Output
- **Location**: `dist\VR180Processor\`
- **Executable**: `dist\VR180Processor\VR180Processor.exe`
- **Expected Size**: ~250 MB
- **FFmpeg**: Bundled (no external dependencies required)

### To Distribute (Windows)
1. Compress the entire `dist\VR180Processor\` folder to a ZIP file
2. Share the ZIP file with users

Users can simply:
1. Download and extract the ZIP file
2. Run `VR180Processor.exe`

---

## Rebuilding After Code Changes

### macOS
```bash
./build_mac.sh
```

### Windows
```cmd
build_windows.bat
```

---

## Features Included in Build

- ✓ VR180 video processing with panomap adjustment controls
- ✓ Global shift (horizontal pixel shift)
- ✓ Global adjustments (yaw, pitch, roll)
- ✓ Stereo offset adjustments (left/right eye offset)
- ✓ Real-time preview with filter application
- ✓ LUT support (.cube files) with intensity control
- ✓ Multiple output formats:
  - H.265 (HEVC) with quality or bitrate mode
  - ProRes 422 (Proxy/LT/Standard/HQ)
  - Auto (matches input codec)
- ✓ Hardware acceleration (VideoToolbox on macOS, NVENC on Windows)
- ✓ FFmpeg bundled (no external installation required)
- ✓ Real-time FFmpeg output display
- ✓ Progress tracking with FPS and percentage

---

## Troubleshooting

### macOS: "App is damaged and can't be opened"
This happens due to Gatekeeper security. Fix with:
```bash
xattr -cr "dist/VR180 Processor.app"
```

### Windows: "Windows protected your PC"
Click "More info" then "Run anyway". This is normal for unsigned applications.

### Build Fails: "FFmpeg not found"
Make sure FFmpeg is installed and in your PATH:
- macOS: `brew install ffmpeg`
- Windows: Follow the FFmpeg installation steps above

### Build Fails: "PyInstaller not found"
Install requirements:
```bash
pip install -r requirements.txt
```

---

## Technical Details

### Dependencies Bundled
- Python 3.13 runtime
- PyQt6 (GUI framework)
- NumPy
- FFmpeg + FFprobe binaries
- All required system libraries

### Build System
- PyInstaller 6.15.0
- Spec file: `vr180_processor.spec`
- Build scripts: `build_mac.sh` (macOS), `build_windows.bat` (Windows)

---

## Version Information

**Current Version**: 1.0.0

### Changelog
- Initial release with full VR180 processing capabilities
- LUT support with intensity control
- Real-time preview
- Hardware-accelerated encoding
- Standalone builds for macOS and Windows
