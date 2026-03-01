# VR180 Silver Bullet v1.4.1 - Windows Release Notes

**Release Date:** January 17, 2026
**Platform:** Windows 10/11 (64-bit)
**Type:** Critical Bug Fix Release

---

## 🐛 Critical Fix: Windows ProRes Encoding

This release fixes a critical bug that prevented ProRes encoding on Windows systems.

### The Problem
- **Error:** FFmpeg returned error code 4294967274 (EINVAL -22)
- **Symptom:** ProRes encoding failed immediately with "FFmpeg error" dialog
- **Scope:** Affected all ProRes profiles (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
- **Impact:** Windows users could not export to ProRes format

### The Fix
Added explicit pixel format specification to the ProRes encoder:
- **ProRes Proxy/LT/Standard/HQ:** Now use `yuv422p10le` (10-bit 4:2:2 chroma)
- **ProRes 4444/4444 XQ:** Now use `yuv444p10le` (10-bit 4:4:4 chroma)

### What's Fixed
✅ ProRes Proxy encoding
✅ ProRes LT encoding
✅ ProRes Standard encoding
✅ ProRes HQ encoding
✅ ProRes 4444 encoding
✅ ProRes 4444 XQ encoding
✅ MV-HEVC workflow (uses ProRes HQ intermediate)

---

## 📥 Download & Installation

### For End Users (No Build Required)

**Download:** `VR180_Silver_Bullet_Windows_v1.4.1.zip` from Releases

**Installation:**
1. Extract the ZIP file to any folder
2. Run `VR180Processor.exe`
3. No installation, no admin rights needed!

**Size:** ~250-300 MB (includes FFmpeg and all dependencies)

### For Developers (Build Kit)

**Download:** `VR180_Silver_Bullet_Windows_BuildKit_v1.4.1.zip`

**Build Instructions:**
1. Install Python 3.10+ and FFmpeg (see README.md)
2. Double-click `build_windows.bat`
3. Find built app in `dist\VR180 Silver Bullet\`

---

## 🔧 System Requirements

**Minimum:**
- Windows 10 (64-bit) or Windows 11
- 4 GB RAM
- 500 MB free disk space (for app)
- Additional space for video processing (~40 GB per minute of 4K video)

**Recommended:**
- Windows 11 (64-bit)
- 16 GB RAM
- NVIDIA GPU with NVENC support (for hardware encoding)
- SSD for video processing

---

## ✨ Features (All Versions)

### Video Processing
- VR180 side-by-side video correction
- Global shift adjustment (-3840 to +3840 pixels)
- Per-eye rotation control (yaw, pitch, roll)
- IPD/convergence adjustment
- Timeline scrubbing with preview

### Output Formats
- **H.265/HEVC:** 8-bit or 10-bit, CRF or bitrate mode
- **ProRes:** All profiles (Proxy, LT, Standard, HQ, 4444, 4444 XQ) ✨ FIXED in v1.4.1
- **Hardware Acceleration:** NVIDIA NVENC (when available)

### Color Grading
- Gamma correction
- White/black point adjustment
- 3D LUT support (.cube format)
- LUT intensity control (0-100%)

### Preview Modes
- Side-by-Side
- Anaglyph (Red/Cyan)
- Overlay (50% blend)
- Left/Right eye only
- Difference mode
- Checkerboard mode

### VR180 Features
- YouTube VR180 metadata injection
- Vision Pro spatial video (hvc1 tag)
- MV-HEVC support (requires spatial tool - macOS only)

---

## 🆚 Version Comparison

| Feature | v1.4.0 | v1.4.1 |
|---------|--------|--------|
| ProRes on Windows | ❌ Broken | ✅ Fixed |
| MV-HEVC workflow | ❌ Broken on Windows | ✅ Fixed |
| H.265 encoding | ✅ Works | ✅ Works |
| 10-bit preservation | ✅ Works | ✅ Works |
| All other features | ✅ Works | ✅ Works |

---

## 🔄 Upgrading from v1.4.0

**No migration needed!** Simply:
1. Delete old `VR180 Silver Bullet` folder
2. Extract v1.4.1 ZIP to new location
3. Your settings are NOT preserved (stored separately in user folder)

**Settings Location:**
- Windows: `%APPDATA%\VR180SilverBullet\settings.json`
- Settings are preserved between versions automatically

---

## 🐛 Known Issues

### Windows
- Hardware decode in preview may crash on some older systems (disabled by default)
- Some NVIDIA GPUs require latest drivers for NVENC support
- Very large files (>100 GB) may cause memory issues

### General
- MV-HEVC conversion is macOS-only (requires Apple VideoToolbox)
- Some fisheye lens types not fully supported
- Batch processing not yet available

---

## 📚 Documentation

- **Quick Start:** See `QUICK_START.txt` in the application folder
- **Full README:** See `README.md` in the build kit
- **Changelog:** See `CHANGELOG.md` for version history

---

## 🙏 Credits

This tool builds upon several excellent projects:

### spatial by Mike Swanson
- MV-HEVC encoding for Vision Pro
- https://blog.mikeswanson.com/spatial/

### FFmpeg
- Video processing powerhouse
- https://ffmpeg.org/

### OpenCV
- Video analysis and frame processing
- https://opencv.org/

### Python Libraries
- PyQt6 (GUI framework)
- NumPy (numerical processing)
- Pillow (image processing)

Special thanks to Mike Swanson for documenting the MV-HEVC workflow and creating the spatial tool!

---

## 📞 Support

**Issues & Bug Reports:**
- GitHub Issues: [Your Repository URL]

**Questions:**
- GitHub Discussions: [Your Repository URL]

**Documentation:**
- See included README and documentation files

---

## 📄 License

[Your License Here - e.g., MIT, GPL, etc.]

**Included Components:**
- FFmpeg: LGPL 2.1+ / GPL 2+
- PyQt6: GPL v3 / Commercial
- OpenCV: Apache 2.0
- spatial tool: Check Mike Swanson's license terms

---

## 🗓️ Roadmap

### Planned for v1.5.0
- Batch processing support
- Preset save/load system
- Advanced auto-alignment tools
- Enhanced preview rendering options

### Under Consideration
- GPU-accelerated filters
- Advanced stabilization
- Color space conversion tools
- Plugin system for custom workflows

---

**Thank you for using VR180 Silver Bullet!**

*Built with ❤️ for the VR180 community*
