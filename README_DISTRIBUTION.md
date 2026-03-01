# VR180 Silver Bullet v1.4.2 - Distribution Files

**Release Date:** January 18, 2026

## 📦 Available Downloads

### Windows Build Kit (62 MB)
**File:** `VR180_Silver_Bullet_v1.4.2_Windows_BuildKit.zip`

Build VR180 Silver Bullet from source on Windows.

**What's included:**
- Complete source code (vr180_gui.py)
- Build scripts (build_windows.bat)
- Requirements and dependencies
- Full documentation
- spatialmedia module

**Who should use this:**
- Windows users who want to build from source
- Developers who want to customize the app
- Users who want to verify the code

**Quick start:**
1. Extract the zip file
2. Install Python 3.10+ and FFmpeg
3. Run `build_windows.bat`
4. App will be in `dist\VR180 Silver Bullet\`

**See:** README.md inside the BuildKit for full instructions

---

### macOS Application (336 MB)
**File:** `VR180_Silver_Bullet_v1.4.2_macOS.zip`

Ready-to-run application for macOS.

**What's included:**
- VR180 Silver Bullet.app (complete standalone app)
- All dependencies bundled (no installation needed)
- Native Apple Silicon support

**Who should use this:**
- macOS users who want to use the app immediately
- Users who don't want to build from source
- Anyone who prefers ready-to-run software

**Quick start:**
1. Extract the zip file
2. Drag `VR180 Silver Bullet.app` to Applications folder
3. Right-click → Open (first time only)
4. Start using the app!

**Requirements:**
- macOS 10.15 (Catalina) or newer
- 8 GB RAM minimum (16 GB recommended)

---

## 🐛 What's New in v1.4.2

### Bug Fixes
1. **Fixed drag-and-drop crash on macOS** - No more crashes when dropping video files
2. **Fixed drag-and-drop preview loading** - Preview now loads automatically
3. **Improved spinbox controls** - Arrow buttons now increment by 0.1° (100x faster)

### Technical Changes
- Updated drag/drop event handlers with better error handling
- Fixed method resolution in drop event
- Improved UI responsiveness for rotation controls

**Full changelog:** See CHANGELOG.md in Windows BuildKit

---

## 🚀 Installation

### Windows (Build from Source)

**Prerequisites:**
- Python 3.10 or newer
- FFmpeg (full build, not essentials)

**Steps:**
```cmd
1. Extract VR180_Silver_Bullet_v1.4.2_Windows_BuildKit.zip
2. Open terminal in extracted folder
3. Run: build_windows.bat
4. Wait for build to complete (3-5 minutes)
5. Run: dist\VR180 Silver Bullet\VR180Processor.exe
```

### macOS (Ready to Run)

**Prerequisites:**
- macOS 10.15 or newer
- No other requirements!

**Steps:**
```
1. Extract VR180_Silver_Bullet_v1.4.2_macOS.zip
2. Drag "VR180 Silver Bullet.app" to Applications folder
3. First launch: Right-click → Open (to bypass Gatekeeper)
4. Subsequent launches: Double-click to open
```

---

## ⚙️ Features

- **VR180 Processing** - Adjust side-by-side VR180 videos
- **Global Controls** - Horizontal shift, yaw, pitch, roll adjustments
- **Stereo Offset** - Control convergence and IPD
- **Preview Modes** - 6 different preview modes (SBS, anaglyph, overlay, etc.)
- **Color Grading** - LUT support, gamma, white/black point
- **Multiple Formats** - H.265, ProRes, MV-HEVC (Vision Pro)
- **Hardware Acceleration** - VideoToolbox (macOS), NVENC (Windows)

---

## 📖 Documentation

**In Windows BuildKit:**
- `README.md` - Full build instructions
- `CHANGELOG.md` - Complete version history
- `RELEASE_NOTES_v1.4.2.md` - Detailed release notes
- `WHATS_NEW_v1.4.2.txt` - Quick summary
- `QUICK_START.txt` - Quick reference guide

**This Directory:**
- `DISTRIBUTION_PACKAGE_v1.4.2.txt` - Complete package information
- `README_DISTRIBUTION.md` - This file

---

## 🆘 Support

### Getting Help
1. Check documentation in Windows BuildKit
2. Review known issues below
3. Search GitHub issues
4. Open new issue with details

### Known Issues

**Windows:**
- Hardware decode may crash on some systems (disabled by default)
- NVIDIA GPUs may need latest drivers for NVENC

**macOS:**
- First launch requires right-click → Open (Gatekeeper security)
- App is not notarized (no developer account)
- MV-HEVC requires external spatial-media-kit-tool

**All Platforms:**
- Very large files (>100GB) may cause memory issues
- Some fisheye lens types not yet supported

---

## 📊 File Information

| File | Size | Platform | Type |
|------|------|----------|------|
| `VR180_Silver_Bullet_v1.4.2_Windows_BuildKit.zip` | 62 MB | Windows 10/11 | Source + Build Scripts |
| `VR180_Silver_Bullet_v1.4.2_macOS.zip` | 336 MB | macOS 10.15+ | Application Bundle |

**Both files created:** January 18, 2026
**Both files tested:** ✅ Verified working

---

## 🔄 Upgrading from Previous Versions

### From v1.4.1
- Settings preserved automatically
- Drag-and-drop now works reliably
- Spinbox controls more responsive
- No breaking changes

### From v1.4.0 or Earlier
- Most settings preserved
- Review hardware acceleration settings
- Check CHANGELOG.md for all changes

---

## 🎯 Quick Comparison

| Feature | Windows BuildKit | macOS App |
|---------|-----------------|-----------|
| Ready to run | ❌ (requires building) | ✅ Yes |
| Source code included | ✅ Yes | ❌ No |
| Customizable | ✅ Yes | ❌ No |
| File size | 62 MB | 336 MB |
| Dependencies | Manual install | ✅ Bundled |
| Build time | ~3-5 minutes | N/A |
| Best for | Developers, customizers | End users |

---

## 📝 License

[Your License Here]

---

## 🙏 Credits

- FFmpeg - Video processing
- Python - Runtime
- PyQt6 - GUI framework
- OpenCV - Image processing
- spatialmedia - VR180 metadata

---

**Version:** 1.4.2
**Released:** January 18, 2026
**Maintained by:** VR180 Silver Bullet Team
