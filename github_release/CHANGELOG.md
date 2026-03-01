# Changelog

## Version 1.0.0 (January 2026)

### Initial Release

#### Features
- ✨ VR180 video processing with full adjustment controls
- 🎨 Color grading with gamma, white/black points, and LUT support
- 📹 H.265 (8-bit/10-bit) and ProRes encoding
- ⚡ Hardware acceleration (VideoToolbox on macOS, NVENC on Windows)
- 👁️ Multiple preview modes including new Single Eye Mode
- 🎬 Timeline scrubbing with pan/zoom preservation
- 📤 YouTube VR180 metadata injection
- 🍎 Vision Pro MV-HEVC export (macOS only)

#### Improvements
- Removed VR180 mask feature (simplified UI)
- Show H.265 bit depth only when H.265 is selected
- Wider "Switch Eyes" button for better usability
- Vision Pro UI automatically hidden on Windows
- Preview pan/zoom position preserved during adjustments

#### Platform Support
- macOS 11+ (Apple Silicon & Intel)
- Windows 10/11 (64-bit)

#### Known Limitations
- Windows: No Vision Pro features (macOS only)
- Windows: ProRes encoding is software-based (slower)
- Large files may require significant processing time

#### Bundle Information
- Includes FFmpeg binaries
- No external dependencies required
- Self-contained application
