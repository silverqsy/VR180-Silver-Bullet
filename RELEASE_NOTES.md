# Silver's VR180 Tool v1.0.0

Professional VR180 video processing application with real-time preview, LUT support, and advanced panomap adjustment controls.

## 🎉 Features

### Core Processing
- ✅ VR180 side-by-side half-equirectangular video processing
- ✅ Global pixel shift for stereo alignment
- ✅ Global adjustments (yaw, pitch, roll) for both eyes
- ✅ Independent stereo offset controls for left/right eyes
- ✅ Real-time preview with multiple viewing modes

### Color Grading
- ✅ LUT support (.cube files)
- ✅ LUT intensity slider (0-100%) with real-time preview
- ✅ Smooth blending between original and LUT

### Output Options
- ✅ H.265 (HEVC) with quality (CRF) or bitrate modes
- ✅ ProRes 422 (Proxy, LT, Standard, HQ)
- ✅ Auto codec matching
- ✅ Hardware acceleration (VideoToolbox on macOS, NVENC on Windows)

### User Experience
- ✅ Real-time FFmpeg output display
- ✅ FPS and percentage progress tracking
- ✅ Settings persistence
- ✅ No console windows on Windows (runs silently)

## 📥 Downloads

### macOS (Ready to Use)
**File**: `Silvers-VR180-Tool-macOS.zip` (63 MB)

- Fully standalone application
- No installation required
- FFmpeg bundled
- Works on macOS 10.14+

**Installation**:
1. Download and unzip
2. Drag to Applications folder
3. Right-click → Open (first time only)

### Windows (Build Package)
**File**: `Silvers-VR180-Tool-Windows-BuildPackage.zip` (38 KB)

- Complete source code
- Automated build scripts
- Comprehensive troubleshooting guides
- Build on Windows 10/11

**Requirements**:
- Python 3.9, 3.10, or 3.11
- FFmpeg
- Visual C++ Runtime

**Quick Start**:
1. Extract the package
2. Read `README_FIRST.txt`
3. Run `try_different_pyqt6.bat` (recommended)

## 🐛 Bug Fixes

- Fixed LUT intensity direction (higher = more LUT applied)
- Fixed LUT slider freezing during drag
- Fixed Browse/Clear button text overflow
- Fixed console windows popping up on Windows during slider adjustments
- Fixed NumPy compatibility with Python 3.11
- Fixed PyQt6 DLL loading issues on Windows

## 🔧 Technical Details

### Bundled Components
- FFmpeg 8.0.1
- FFprobe
- Python runtime
- PyQt6 6.4.0+
- NumPy 1.26.4

### System Requirements
- **macOS**: 10.14+, 4GB RAM, 500MB disk space
- **Windows**: 10/11, 4GB RAM, 500MB disk space
- For 4K/8K: 8GB+ RAM recommended

## 📚 Documentation

- **README.md** - User guide and features
- **BUILD_INSTRUCTIONS.md** - How to build from source
- **WINDOWS_BUILD_FIX.md** - Windows troubleshooting
- **DLL_ERROR_FIX.txt** - Fix PyQt6 DLL errors
- **PYTHON_DOWNGRADE_GUIDE.txt** - Python version management

## 🙏 Credits

- **Developer**: Silver
- **Built with**: Python, PyQt6, FFmpeg, PyInstaller
- **FFmpeg**: FFmpeg team
- **PyInstaller**: PyInstaller team

## 📝 License

Proprietary - All rights reserved

## 🐞 Known Issues

### Windows
- First build may require trying multiple PyQt6 versions (automated script included)
- May need Visual C++ Runtime installation
- Python 3.12+ not supported (use 3.9-3.11)

### macOS
- First launch shows security warning (normal for unsigned apps)
- May need to allow app in System Preferences → Security & Privacy

## 🔮 Future Plans

- Batch processing
- VR180 photo support
- Custom FFmpeg parameters
- Additional LUT formats
- Preset system for common adjustments

---

**Full Changelog**: Initial release v1.0.0
