# Changelog - VR180 Silver Bullet

## Version 1.4.3 (January 19, 2026)

### 🐛 Critical Bug Fix - Windows Executable Path Detection

**Fixed:** Windows builds unable to find bundled ffprobe.exe and ffmpeg.exe

**Problem:**
- Windows builds failed with "ffprobe.EXE returned non-zero exit status 1"
- Path detection functions looked for executables without `.exe` extension
- `Path('ffprobe').exists()` returned False even though `ffprobe.exe` was present
- Video import completely broken on Windows

**Solution:**
- Updated `get_ffmpeg_path()` to look for `ffmpeg.exe` on Windows
- Updated `get_ffprobe_path()` to look for `ffprobe.exe` on Windows
- Updated `get_spatial_path()` to look for `spatial.exe` on Windows
- Added platform-specific executable name detection using `sys.platform`

**Impact:**
- ✅ **CRITICAL FIX**: Windows builds can now find bundled executables
- ✅ Video import works correctly
- ✅ FFprobe can probe video files
- ✅ FFmpeg can process videos
- ✅ No impact on macOS (still works as before)

**Files Changed:**
- `vr180_gui.py` - Lines 67, 102, 137

---

## Version 1.4.2 (January 18, 2026)

### 🐛 Bug Fixes

**Drag and Drop Crash Fixed (macOS)**
- Fixed crash when dragging video files into the application window
- Changed `event.acceptProposedAction()` to `event.accept()` to prevent Qt fatal errors
- Added exception handling to prevent app crashes on drag/drop errors

**Drag and Drop Preview Loading**
- Fixed issue where drag and drop didn't start preview
- Corrected method call from non-existent `_load_video()` to `_load_video_info()`
- Drag and drop now works identically to Browse button

**UI Improvements**
- Changed spinbox step size from 0.001° to 0.1° for easier adjustment
- Up/down arrow buttons now increment by 0.1° instead of 0.001°
- Display precision remains at 3 decimal places for fine control
- Affects Global Panomap Adjustment and Stereo Offset controls

---

## Version 1.4.1 (January 2026)

### 🐛 Critical Bug Fix - Windows ProRes Encoding
**Fixed:** Windows ProRes encoding error (FFmpeg return code 4294967274)

**Problem:**
- ProRes encoding failed on Windows with "invalid argument" error
- Affected all ProRes profiles (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
- Error code -22 (EINVAL) appeared as 4294967274 in Windows

**Solution:**
- Added explicit pixel format specification for `prores_ks` encoder
- ProRes Proxy/LT/Standard/HQ now use `yuv422p10le` (10-bit 4:2:2)
- ProRes 4444/4444 XQ now use `yuv444p10le` (10-bit 4:4:4)
- Applied fix to both direct ProRes output and MV-HEVC intermediate encoding

**Impact:**
- ✅ ProRes encoding now works correctly on Windows
- ✅ All ProRes profiles fully functional
- ✅ MV-HEVC workflow fixed (uses ProRes HQ intermediate)
- ✅ No impact on macOS (uses VideoToolbox, unaffected)

---

## Version 1.4.0 (January 2026)

### 🎯 Default Settings Changed
- **H.265 Encoding Mode**: Changed default from CRF (Quality) to **Bitrate Mode**
- **Default Bitrate**: Changed from 50 Mbps to **200 Mbps** for higher quality output
- **Bit Depth**: Default remains 8-bit for maximum compatibility

### 🔧 10-bit Input Handling (Major Improvement)
**Previous Behavior:**
- Always converted 10-bit input to 8-bit (caused quality loss)
- Processing was slow due to unnecessary conversion

**New Behavior:**
- **Smart 10-bit Preservation**: Only converts when necessary
  - Output 8-bit H.265 → Converts 10-bit to 8-bit
  - Output 10-bit H.265 → **Preserves original 10-bit** ✨
  - Output ProRes → **Preserves original 10-bit** ✨
- Maintains color depth and quality when possible
- Faster processing for 10-bit workflows

### 📊 Status Messages
- Added informative messages during processing:
  - "Converting 10-bit input to 8-bit for output..." (when downconverting)
  - "Detected 10-bit input - preserving 10-bit for output..." (when preserving)

### 🐛 Bug Fixes
- Fixed icon path in Windows build specification
- Updated build scripts for consistency

---

## Version 1.3.0 (January 2026)

### ✨ Features
- Added Vision Pro spatial video support (MV-HEVC)
- Integrated spatial-media-kit-tool for MV-HEVC conversion
- Added VR180 metadata injection
- Improved preview performance with hardware decode

### 🎨 Preview Improvements
- Added 6 preview modes (Side-by-Side, Anaglyph, Overlay, Single Eye, Difference, Checkerboard)
- Zoom controls with pan support
- Timeline scrubbing
- Real-time adjustment preview

### 🚀 Performance
- Hardware acceleration support (VideoToolbox on macOS, NVENC on Windows)
- Optimized 10-bit video processing
- Frame caching for faster preview updates

---

## Version 1.2.0 (December 2025)

### 🎨 Color Grading
- Added LUT support (.cube format)
- LUT intensity control (0-100%)
- Gamma correction
- White/black point adjustment
- Pre-LUT color adjustments

### 📦 Packaging
- Fully standalone builds (no FFmpeg installation needed)
- Bundled all dependencies
- Reduced build size with optimized compression

---

## Version 1.1.0 (November 2025)

### ✨ Features
- Added stereo offset adjustment (convergence/IPD)
- Global rotation controls (yaw/pitch/roll)
- Multiple ProRes profiles support
- Encoder speed presets (Fast/Medium/Slow)

### 🔧 Improvements
- Better error handling and user feedback
- Progress bar with time estimates
- Console output for FFmpeg debugging

---

## Version 1.0.0 (October 2025)

### 🎉 Initial Release
- VR180 video processing
- Global horizontal shift adjustment
- H.265 and ProRes output
- Basic preview functionality
- Cross-platform support (Windows/macOS)

---

## Upcoming Features (Roadmap)

### Planned for 1.5.0
- Batch processing support
- Preset save/load system
- Auto-alignment tools
- Enhanced preview rendering

### Under Consideration
- GPU-accelerated filters
- Advanced stabilization
- Color space conversion tools
- Plugin system

---

## Known Issues

### Windows
- Hardware decode in preview may cause crashes on some systems (disabled by default)
- Some NVIDIA GPUs may require driver updates for NVENC support

### macOS
- MV-HEVC conversion requires external spatial-media-kit-tool
- Some older Macs don't support VideoToolbox 10-bit encoding

### All Platforms
- Very large files (>100GB) may cause memory issues during preview
- Some fisheye lens types not yet supported in VR180 mode

---

## Support

Report issues at: [Your GitHub Issues URL]
Documentation: [Your Docs URL]
