# Release Notes - v1.3.0

## VR180 Silver Bullet v1.3.0
**Release Date**: January 7, 2026

### 🎉 Major Features

#### Vision Pro MV-HEVC Support
- Full MV-HEVC encoding with Vision Pro spatial metadata
- Integrated Mike Swanson's `spatial` CLI tool
- Proper APMP metadata (cdist=65mm, hfov=180°, VEXU boxes)
- Full bitrate control (up to 500+ Mbps)

#### Lossless Workflow
- ProRes HQ intermediate prevents double lossy compression
- Single encoding pass to MV-HEVC maintains maximum quality
- Automatic cleanup of temporary files

#### Auto Mode Enhancement
- Codec "Auto" now uses 350 Mbps for MV-HEVC
- H.265 mode respects user bitrate settings
- Bit depth control (8-bit or 10-bit) when H.265 selected

### 🔧 Technical Improvements

- **Encoding Pipeline**: FFmpeg → ProRes → spatial make → MV-HEVC
- **Hardware Acceleration**: Uses Apple VideoToolbox for fast encoding
- **Metadata Accuracy**: Correct VR180 parameters for Vision Pro
- **Simplified Code**: Removed complex x265 multiview paths

### 📦 Bundled Tools

- **FFmpeg 8.0.1**: Video processing and ProRes encoding
- **spatial v0.6.2**: MV-HEVC encoding by Mike Swanson
- All tools bundled in macOS app

### 🐛 Bug Fixes

- Fixed MV-HEVC bitrate limitations (was ~62-87 Mbps, now unlimited)
- Corrected camera baseline (63mm → 65mm)
- Corrected horizontal FOV (90° → 180°)
- ProRes intermediate prevents quality degradation

### 📖 Documentation

- Added MV-HEVC quick guide
- Complete technical workflow documentation
- GitHub release instructions
- Proper attribution to Mike Swanson's spatial tool

### ⚠️ Known Issues

- MV-HEVC encoding requires ~40GB disk space per minute of 4K video (temporary)
- Windows version doesn't support Vision Pro features (macOS only)
- Encoding speed is moderate (0.5-1x realtime for MV-HEVC)

### 🙏 Credits

Special thanks to **Mike Swanson** for the `spatial` CLI tool that makes high-quality Vision Pro MV-HEVC encoding possible.

---

## Version History

### v1.3.0 (January 7, 2026)
- Vision Pro MV-HEVC support with spatial tool integration
- ProRes lossless intermediate workflow
- Auto mode = 350 Mbps for MV-HEVC
- Proper VEXU metadata (cdist=65, hfov=180)

### v1.2.0 (January 7, 2026)
- Explored x265 multiview encoding
- Added dual-mode encoding support

### v1.1.0 (January 7, 2026)
- ProRes intermediate to prevent double compression
- Improved bitrate handling

### v1.0.1 (January 6, 2026)
- Initial spatial CLI integration
- Full bitrate control

### v1.0.0 (January 5, 2026)
- Initial release as "VR180 Silver Bullet"
- IPD adjustment, per-eye rotation, color grading
- LUT support with intensity control
- H.265 and ProRes output

---

## Upgrade Notes

### From v1.0.x to v1.3.0

**Breaking Changes**: None

**New Features**:
- Vision Pro MV-HEVC mode now available
- Auto codec defaults to 350 Mbps for MV-HEVC

**Recommended Actions**:
- For Vision Pro videos, use "Vision Pro MV-HEVC" mode
- Ensure 40GB+ free disk space for processing

### System Requirements Changes

No changes to system requirements.

---

## Future Roadmap

Planned features for future releases:
- Batch processing support
- Preset management system
- Advanced color grading tools
- Custom metadata parameters
- Progress estimation improvements

---

## Download

Get the latest version from [GitHub Releases](https://github.com/YOUR_USERNAME/vr180-silver-bullet/releases)

**Files**:
- `VR180-Silver-Bullet-macOS-v1.3.0.zip` - macOS app bundle
- `VR180-Silver-Bullet-Windows-v1.3.0.zip` - Windows executable (if available)

---

## Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/YOUR_USERNAME/vr180-silver-bullet/issues)
- **Documentation**: See repository README and docs
- **Discussions**: Use [GitHub Discussions](https://github.com/YOUR_USERNAME/vr180-silver-bullet/discussions)
