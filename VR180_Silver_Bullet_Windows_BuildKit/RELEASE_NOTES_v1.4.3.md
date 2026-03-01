# VR180 Silver Bullet - Release Notes v1.4.3

**Release Date:** January 19, 2026
**Version:** 1.4.3
**Type:** Critical Bug Fix (Windows)

---

## 🚨 Critical Windows Bug Fix

This release fixes a **critical bug** that prevented the Windows version from working at all.

### What Was Broken

Windows users experienced this error when trying to import videos:
```
Command '['Y:\...\ffprobe.EXE', '-v', 'quiet', ...]' returned non-zero exit status 1.
```

**Impact:** Video import was completely broken on Windows. The app couldn't probe video files, making it unusable.

### What Was Fixed

Fixed the executable path detection for Windows:
- `ffmpeg.exe` is now found correctly
- `ffprobe.exe` is now found correctly
- `spatial.exe` is now found correctly

**Result:** Windows builds now work properly. Video import, preview, and processing all function correctly.

---

## 🔧 Technical Details

### Root Cause

The path detection functions were looking for executables without the `.exe` extension:
- Looked for `ffprobe` instead of `ffprobe.exe`
- Looked for `ffmpeg` instead of `ffmpeg.exe`

On Windows, executables MUST have the `.exe` extension. `Path('ffprobe').exists()` returned `False` even though `ffprobe.exe` was present in the bundle.

### The Fix

Added platform-specific executable names in three functions:

**Before (Broken):**
```python
ffprobe_path = base_path / 'ffprobe'  # ❌ Fails on Windows
```

**After (Fixed):**
```python
ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
ffprobe_path = base_path / ffprobe_name  # ✅ Works on Windows
```

Applied to:
- `get_ffmpeg_path()` (Line 67)
- `get_ffprobe_path()` (Line 102)
- `get_spatial_path()` (Line 137)

---

## 📋 Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `get_ffmpeg_path()` | Added `.exe` for Windows | ✅ FFmpeg found correctly |
| `get_ffprobe_path()` | Added `.exe` for Windows | ✅ Video probing works |
| `get_spatial_path()` | Added `.exe` for Windows | ✅ Future-proof for spatial tool |

---

## ⚠️ Who Should Update

### Windows Users: **MUST UPDATE**
If you're on Windows and have v1.4.2 or earlier:
- Video import doesn't work
- "ffprobe.EXE error" appears
- App is completely unusable

**You MUST upgrade to v1.4.3.**

### macOS Users: **Optional**
If you're on macOS:
- This bug doesn't affect you
- v1.4.2 works fine on macOS
- You can upgrade for consistency, but it's not required

---

## 🚀 How to Update

### Windows Build Kit Users

1. Download the new BuildKit: `VR180_Silver_Bullet_v1.4.3_Windows_BuildKit.zip`
2. Extract the files
3. Rebuild the application:
   ```cmd
   build_windows.bat
   ```
4. The built app in `dist/` will now work correctly

### macOS App Bundle Users

A new macOS app bundle will be released with this fix included.

---

## 📦 What's Included in v1.4.3

### New (from v1.4.3)
- ✅ **Critical Fix:** Windows executable path detection

### Carried Over (from v1.4.2)
- ✅ Drag-and-drop crash fix (macOS)
- ✅ Drag-and-drop preview loading fix
- ✅ Improved spinbox step size (0.1° increments)

### Carried Over (from v1.4.1)
- ✅ Windows ProRes encoding fix

---

## 🐛 Known Issues

### Windows
- Hardware decode in preview may crash on some systems (disabled by default)
- Some NVIDIA GPUs require latest drivers for NVENC

### macOS
- First launch requires right-click → Open (Gatekeeper)
- App is not notarized (no developer account)
- MV-HEVC requires external spatial-media-kit-tool

### All Platforms
- Very large files (>100GB) may cause memory issues
- Some fisheye lens types not yet supported

---

## 📖 Full Feature List

### Core Features
- VR180 side-by-side video processing
- Global horizontal shift adjustment
- Global panomap adjustment (yaw/pitch/roll)
- Stereo offset control (convergence/IPD)
- Real-time preview with 6 modes

### Output Formats
- H.265 (HEVC) - 8-bit or 10-bit
- ProRes (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
- MV-HEVC (Vision Pro spatial video) - macOS only

### Color Grading
- LUT support (.cube format)
- LUT intensity control
- Gamma correction
- White/black point adjustment

### Performance
- Hardware acceleration (VideoToolbox/NVENC)
- Frame caching for instant preview
- Multi-threaded encoding
- Smart 10-bit preservation

---

## 💻 System Requirements

### Windows (Build Kit)
- Python 3.10+ (3.13 recommended)
- FFmpeg (full build with all libraries)
- Windows 10 or 11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- Optional: NVIDIA GPU for NVENC

### macOS (App Bundle)
- macOS 10.15 (Catalina) or newer
- 8 GB RAM minimum (16 GB recommended)
- No other requirements (all bundled)

---

## 📝 Version History

- **v1.4.3** (January 19, 2026) - Critical Windows path detection fix
- **v1.4.2** (January 18, 2026) - Drag-and-drop and UI fixes
- **v1.4.1** (January 18, 2026) - Windows ProRes encoding fix
- **v1.4.0** (January 2026) - Major feature updates

---

## 🆘 Support

### Build Issues (Windows)
Check that you have:
- Python 3.10+ installed and in PATH
- FFmpeg full build installed and in PATH
- Extracted the BuildKit to a folder without spaces in the path

### Runtime Issues
- **"ffprobe error"**: Make sure you're using v1.4.3 (this version)
- **"Can't find FFmpeg"**: Add FFmpeg to PATH or place in app folder
- **NVENC fails**: Update NVIDIA drivers to latest version

### Getting Help
- Check README.md in BuildKit
- Review CHANGELOG.md for known issues
- See BUGFIX_v1.4.3.md for technical details

---

## 🎉 Summary

**v1.4.3 is a critical bug fix release for Windows users.**

If you're on Windows and have been experiencing the "ffprobe.EXE error", this update fixes it completely. The app should now work exactly as intended on Windows.

For macOS users, this update doesn't change anything, but you can upgrade for consistency if desired.

Thank you for your patience while we fixed this critical issue!

---

**VR180 Silver Bullet v1.4.3**
Critical Windows Bug Fix
Released: January 19, 2026
