# VR180 Silver Bullet - Windows Build Kit

**Version 1.4.3** - January 19, 2026

This package contains everything needed to build the VR180 Silver Bullet application for Windows.

## 📦 What's Included

- `vr180_gui.py` - Main application source code
- `vr180_processor.spec` - PyInstaller build specification
- `requirements.txt` - Python dependencies
- `icon.ico` - Application icon
- `build_windows.bat` - Automated build script
- `spatialmedia/` - VR180 metadata injection module
- This README

## 🔧 Prerequisites

### 1. Python 3.10 or newer
Download from: https://www.python.org/downloads/

**IMPORTANT:** During installation, check **"Add Python to PATH"**

### 2. FFmpeg (full build with all libraries)
Download from: https://www.gyan.dev/ffmpeg/builds/

1. Download the **"ffmpeg-release-full.7z"** file (NOT essentials)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your System PATH:
   - Right-click "This PC" → Properties → Advanced System Settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click Edit
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK on all dialogs

4. Verify installation:
   ```cmd
   ffmpeg -version
   ffprobe -version
   ```

## 🚀 Quick Build Instructions

### Option 1: Automated Build (Recommended)

1. Double-click `build_windows.bat`
2. Wait for the build to complete (3-5 minutes)
3. Find your application in: `dist\VR180 Silver Bullet\`
4. Run: `dist\VR180 Silver Bullet\VR180Processor.exe`

### Option 2: Manual Build

Open Command Prompt in this folder and run:

```cmd
# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Build application
python -m PyInstaller --clean vr180_processor.spec
```

## 📋 Default Settings

The application comes with these optimized defaults:

- **Encoding Mode:** Bitrate (200 Mbps)
- **H.265 Bit Depth:** 8-bit (compatible)
- **10-bit Input Handling:**
  - Auto-converts to 8-bit when outputting 8-bit H.265
  - Preserves 10-bit when outputting 10-bit H.265 or ProRes

## 🎯 Build Output

After successful build:

```
dist\
└── VR180 Silver Bullet\
    ├── VR180Processor.exe     # Main application (double-click to run)
    ├── ffmpeg.exe             # Bundled FFmpeg
    ├── ffprobe.exe            # Bundled FFprobe
    ├── spatialmedia\          # VR180 metadata module
    └── [many DLL and support files]
```

**Total size:** ~250-300 MB (fully standalone, no installation needed)

## 📤 Distribution

To share the application:

1. Compress the entire `dist\VR180 Silver Bullet\` folder to a ZIP
2. Share the ZIP file
3. Users extract and run `VR180Processor.exe` - no installation required!

## 🔍 Troubleshooting

### "Python not found"
- Reinstall Python and check "Add Python to PATH"
- Or run in Command Prompt: `python --version`

### "FFmpeg not found"
- Make sure `C:\ffmpeg\bin` is in your PATH
- Restart Command Prompt after changing PATH
- Run: `where ffmpeg` to verify

### "Module not found" errors during build
```cmd
python -m pip install --upgrade pip
python -m pip install --force-reinstall -r requirements.txt
```

### Build completes but missing FFmpeg
The build script will automatically copy FFmpeg if PyInstaller doesn't bundle it correctly.

### Application crashes on startup
- Make sure you're using the FULL FFmpeg build (not essentials)
- The full build includes required DLLs like avcodec, avformat, etc.

## 💻 System Requirements

**For Building:**
- Windows 10/11 (64-bit)
- Python 3.10+
- 2 GB free disk space
- Internet connection (for downloading dependencies)

**For Running (end users):**
- Windows 10/11 (64-bit)
- No Python installation needed
- No FFmpeg installation needed
- Everything is bundled!

## 📝 Features

- **VR180 Video Processing**
  - Global shift adjustment (-3840 to +3840 pixels)
  - Global rotation (yaw/pitch/roll)
  - Stereo offset adjustment (convergence/IPD)

- **Output Formats**
  - H.265/HEVC (8-bit or 10-bit)
  - ProRes (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
  - Hardware acceleration (NVIDIA NVENC on supported GPUs)

- **Color Grading**
  - Gamma correction
  - White/black point adjustment
  - LUT support (.cube format)
  - LUT intensity control

- **Preview Modes**
  - Side by Side
  - Anaglyph (Red/Cyan)
  - Overlay 50%
  - Single Eye Mode
  - Difference Mode
  - Checkerboard Mode

- **VR180 Metadata**
  - YouTube VR180 metadata injection
  - Vision Pro spatial video support (hvc1, MV-HEVC)

## 📚 Documentation

For more information:
- GitHub: [Your GitHub URL]
- Issues: [Your GitHub Issues URL]

## 📄 License

[Your License Information]

## 🙏 Credits

- FFmpeg: https://ffmpeg.org
- PyQt6: https://www.riverbankcomputing.com/software/pyqt/
- Google Spatial Media: https://github.com/google/spatial-media

---

**Version:** 1.4.1
**Last Updated:** January 2026
**Built with:** Python 3.13, PyQt6, FFmpeg 8.0

## 🆕 What's New in 1.4.1

**Critical Fix:** Windows ProRes encoding now works correctly! Fixed FFmpeg error 4294967274 that prevented ProRes output on Windows systems.
