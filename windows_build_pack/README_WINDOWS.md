# VR180 Silver Bullet - Windows Edition

Professional VR180 video processing tool for Windows.

## Quick Start

### 1. Install Prerequisites

**Python 3.11 or 3.12**
- Download: https://www.python.org/downloads/windows/
- ✓ Check "Add Python to PATH" during installation

**FFmpeg**
- Download: https://www.gyan.dev/ffmpeg/builds/
- Get `ffmpeg-release-essentials.zip`
- Extract to `C:\ffmpeg`
- Add `C:\ffmpeg\bin` to System PATH

### 2. Build the Application

Simply double-click: `BUILD.bat`

Or manually:
```cmd
pip install PyQt6 numpy Pillow pyinstaller
pyinstaller --clean vr180_processor.spec
```

### 3. Run

```
dist\VR180 Silver Bullet\VR180Processor.exe
```

## What's Included

All VR180 processing features work on Windows:

✅ **VR180 Adjustments**
- Global shift, yaw, pitch, roll
- Stereo offset corrections
- Timeline preview with pan/zoom

✅ **Video Codecs**
- H.265 (8-bit & 10-bit)
- ProRes (all profiles)
- Hardware acceleration (NVENC)

✅ **Color Grading**
- Gamma, white/black points
- LUT file support (.cube)
- Real-time preview

✅ **Preview Modes**
- Side by Side, Anaglyph
- Single Eye with toggle
- Difference, Checkerboard

✅ **Export Options**
- YouTube VR180 metadata
- Multiple quality presets
- Batch processing ready

❌ **Not Available**
- Vision Pro features (macOS only)
- Apple hvc1 tag (macOS only)

## Files Included

- `vr180_gui.py` - Main application
- `vr180_processor.spec` - Build configuration
- `icon.ico` - Windows icon (126 KB)
- `icon.png` - Source icon (959 KB)
- `requirements.txt` - Python dependencies
- `BUILD.bat` - Automated build script
- `BUILD_WINDOWS.md` - Detailed instructions

## Distribution

To share the application:
1. Build using `BUILD.bat`
2. Zip the entire folder: `dist\VR180 Silver Bullet\`
3. Users just extract and run `VR180Processor.exe`
4. No Python or dependencies required!

## Troubleshooting

**"Python not found"**
- Reinstall Python and check "Add to PATH"
- Restart Command Prompt

**"FFmpeg not found"**
- Add `C:\ffmpeg\bin` to System PATH
- Restart Command Prompt
- Verify: `ffmpeg -version`

**"Build failed"**
- Check Python version: `python --version` (should be 3.11 or 3.12)
- Update pip: `python -m pip install --upgrade pip`
- Reinstall dependencies: `pip install --upgrade PyQt6 numpy Pillow pyinstaller`

**"Application won't start"**
- Run from Command Prompt to see errors
- Check if running from correct location
- Verify all DLL files are present in dist folder

## Hardware Acceleration

Windows uses NVIDIA NVENC for hardware acceleration:
- Requires NVIDIA GPU (GTX 900 series or newer)
- Automatically detected and enabled
- Falls back to software encoding if not available

## Performance

**H.265 Encoding (4K VR180):**
- With NVENC: ~0.5-1x realtime
- Software: ~0.1-0.3x realtime

**ProRes Encoding (4K VR180):**
- Software only: ~0.1-0.3x realtime
- Note: ProRes is slower on Windows (no hardware support)

## Support

For issues or questions:
- Check `BUILD_WINDOWS.md` for detailed instructions
- Verify FFmpeg is properly installed
- Ensure Python 3.11 or 3.12 is used

## Version Info

- Application: VR180 Silver Bullet
- Platform: Windows 10/11
- Architecture: x64
- Build System: PyInstaller 6.x
