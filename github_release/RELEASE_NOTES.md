# VR180 Silver Bullet v1.0.0

Professional VR180 video processing application for macOS and Windows.

## 🎉 What's New

First official release of VR180 Silver Bullet - a powerful tool for processing and fixing VR180 videos.

## 📦 Downloads

### macOS (Apple Silicon & Intel)
- **File:** `VR180 Silver Bullet macOS.zip` (211 MB)
- **Requires:** macOS 11.0 or later
- **Installation:** Extract and run the `.app` file

### Windows (64-bit)
- **File:** `VR180 Silver Bullet Win.zip` (207 MB)
- **Requires:** Windows 10/11
- **Installation:** Extract and run `VR180Processor.exe`

## ✨ Key Features

### Video Processing
- Adjust global shift, yaw, pitch, and roll
- Fix stereo offset issues
- Real-time preview with multiple viewing modes
- Timeline scrubbing with preserved pan/zoom

### Encoding Options
- **H.265:** 8-bit and 10-bit encoding with hardware acceleration
- **ProRes:** All profiles (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
- **Quality Control:** CRF or bitrate targeting
- **YouTube Ready:** Automatic VR180 metadata injection

### Color Grading
- Gamma adjustment (10-300%)
- White and black point control
- LUT file support (.cube) with intensity blending
- Real-time preview of adjustments

### Preview Modes
- Side by Side
- Anaglyph (Red/Cyan 3D)
- Overlay 50%
- **NEW:** Single Eye Mode with L/R toggle
- Difference (stereo mismatch detection)
- Checkerboard pattern

### Platform-Specific Features

**macOS Exclusive:**
- Vision Pro MV-HEVC spatial video export
- Apple hvc1 tag for compatibility
- VideoToolbox hardware acceleration
- Fast ProRes encoding

**Windows:**
- NVIDIA NVENC hardware acceleration
- All core features available
- Self-contained executable

## 🚀 Quick Start

1. Download the appropriate version for your platform
2. Extract the ZIP file
3. Run the application
4. Load your VR180 video and start processing!

## 📋 System Requirements

**macOS:**
- macOS 11.0 (Big Sur) or later
- 4 GB RAM (8 GB recommended)
- 500 MB disk space + working space

**Windows:**
- Windows 10/11 (64-bit)
- 4 GB RAM (8 GB recommended)
- 500 MB disk space + working space
- NVIDIA GPU recommended for hardware acceleration

## 🐛 Known Issues

- Large video files may require significant processing time
- Windows ProRes encoding is software-based (slower than macOS)
- First launch on macOS may require security approval

## 📝 What's Included

Both versions include:
- Complete application (no installation needed)
- Bundled FFmpeg binaries
- All processing features
- No external dependencies

## 🙏 Feedback

Please report any issues or suggestions on GitHub.

---

**Full Changelog:** See CHANGELOG.md for complete release notes.
