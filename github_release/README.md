# VR180 Silver Bullet

Professional VR180 video processing application for macOS and Windows.

## 📦 Downloads

### macOS (Apple Silicon & Intel)
**File:** `VR180 Silver Bullet macOS.zip` (211 MB)
- Extract and run `VR180 Silver Bullet.app`
- Requires: macOS 11.0 or later
- Includes bundled FFmpeg

### Windows (64-bit)
**File:** `VR180 Silver Bullet Win.zip` (207 MB)
- Extract and run `VR180Processor.exe`
- Requires: Windows 10/11
- Includes bundled FFmpeg

## ✨ Features

### VR180 Processing
- **Global Adjustments**: Shift, yaw, pitch, roll
- **Stereo Offset**: Fix left/right eye alignment
- **Timeline Preview**: Scrub to any frame with pan/zoom
- **Real-time Preview**: See changes before processing

### Video Codecs
- **H.265 (HEVC)**: 8-bit and 10-bit encoding
- **ProRes**: All profiles (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
- **Hardware Acceleration**:
  - macOS: VideoToolbox
  - Windows: NVIDIA NVENC

### Color Grading
- **Gamma Adjustment**: 10-300% range
- **White Point Control**: Highlight management
- **Black Point Control**: Shadow management
- **LUT Support**: .cube files with intensity blending

### Preview Modes
- Side by Side
- Anaglyph (Red/Cyan 3D)
- Overlay 50%
- Single Eye Mode (with L/R toggle)
- Difference (stereo mismatch detection)
- Checkerboard pattern

### Export Options
- **YouTube VR180**: Automatic metadata injection
- **Quality Modes**: CRF (quality) or bitrate targeting
- **Multiple Formats**: MP4, MOV
- **Vision Pro** (macOS only): MV-HEVC spatial video

## 🚀 Quick Start

### macOS
1. Download `VR180 Silver Bullet macOS.zip`
2. Extract the ZIP file
3. Double-click `VR180 Silver Bullet.app`
4. If macOS blocks it, go to System Settings → Privacy & Security → Open Anyway

### Windows
1. Download `VR180 Silver Bullet Win.zip`
2. Extract the ZIP file
3. Run `VR180Processor.exe`
4. If Windows SmartScreen appears, click "More info" → "Run anyway"

## 📋 System Requirements

### macOS
- macOS 11.0 (Big Sur) or later
- Apple Silicon (M1/M2/M3) or Intel processor
- 4 GB RAM minimum, 8 GB recommended
- 500 MB disk space for app
- Additional space for video processing

### Windows
- Windows 10 or Windows 11 (64-bit)
- Intel or AMD processor (SSE2 support required)
- 4 GB RAM minimum, 8 GB recommended
- 500 MB disk space for app
- NVIDIA GPU recommended for hardware acceleration
- Additional space for video processing

## 🎯 Usage

### Basic Workflow
1. **Load Video**: Click "Browse" to select your VR180 video
2. **Adjust**: Use sliders to fix alignment issues
3. **Preview**: Scrub timeline and check different preview modes
4. **Color Grade**: Apply gamma, white/black points, or LUT files
5. **Export**: Choose codec and quality, then click "Start Processing"

### Tips
- Use hardware acceleration when available (enabled by default)
- For YouTube upload: Enable "Inject VR180 Metadata"
- Preview pan/zoom is preserved while adjusting parameters
- Single Eye Mode has L/R toggle for easy comparison

## 🆚 Platform Differences

| Feature | macOS | Windows |
|---------|-------|---------|
| H.265 Encoding | ✅ | ✅ |
| ProRes Encoding | ✅ Fast (HW) | ✅ Slow (SW) |
| Hardware Accel | VideoToolbox | NVENC |
| VR180 Metadata | ✅ | ✅ |
| Vision Pro MV-HEVC | ✅ | ❌ |
| Apple hvc1 Tag | ✅ | ❌ |

## 🛠️ Troubleshooting

### macOS
**"App is damaged"**
- Run: `xattr -cr "/Applications/VR180 Silver Bullet.app"`
- Or: System Settings → Privacy & Security → Open Anyway

**Slow performance**
- Enable Hardware Acceleration
- Close other applications
- Ensure sufficient disk space

### Windows
**"FFmpeg not found"**
- FFmpeg is bundled - this shouldn't happen
- Try extracting to a different location
- Run as Administrator

**Hardware acceleration not working**
- Requires NVIDIA GPU (GTX 900 series or newer)
- Update NVIDIA drivers
- Falls back to software encoding automatically

**Slow ProRes encoding**
- ProRes on Windows uses software encoding (expected)
- Consider using H.265 instead for faster encoding

## 📄 License

This software is provided as-is for video processing purposes.

## 🙏 Credits

Built with:
- **Python** - Programming language
- **PyQt6** - GUI framework
- **FFmpeg** - Video processing engine
- **NumPy** - Numerical computations

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

## 📝 Version Info

**Version:** 1.0.0
**Release Date:** January 2026
**Platforms:** macOS 11+, Windows 10/11

---

© 2026 VR180 Silver Bullet - Professional VR180 Video Processing
