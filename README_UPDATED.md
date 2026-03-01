# VR180 Silver Bullet

Professional VR180 video processor with Vision Pro MV-HEVC support.

## Features

### Vision Pro MV-HEVC Encoding
- Full bitrate control (up to 500+ Mbps)
- Proper APMP metadata for Vision Pro compatibility
- Automatic spatial video tagging
- Hardware-accelerated encoding

### VR180 Adjustments
- **IPD (Convergence) Control**: Adjust stereo eye separation
- **Per-Eye Rotation**: Independent yaw, pitch, roll for each eye
- **Color Grading**: Gamma, white point, black point adjustments
- **LUT Support**: Apply 3D LUTs with intensity control
- **Preview Mode**: Real-time preview with eye switching

### Output Formats
- **H.265 (HEVC)**: 8-bit or 10-bit, CRF or bitrate mode
- **ProRes**: Proxy, LT, Standard, HQ, 4444, 4444 XQ
- **Vision Pro MV-HEVC**: Spatial video with APMP metadata

## Quick Start

### macOS

1. Download `VR180-Silver-Bullet-macOS-v1.3.0.zip` from [Releases](../../releases)
2. Extract and open `VR180 Silver Bullet.app`
3. Load your VR180 video
4. Adjust settings as needed
5. Select output format
6. Click **Start**!

### For Vision Pro Spatial Video

1. **Codec**: Select "Auto" (uses 350 Mbps) or "H.265" (custom bitrate)
2. **Vision Pro Mode**: Select "Vision Pro MV-HEVC (full APMP - slow)"
3. **Process**: The app will create MV-HEVC with proper metadata
4. **Result**: Videos play as spatial video on Vision Pro

## System Requirements

### macOS
- macOS 11.0 (Big Sur) or later
- Apple Silicon (M1/M2/M3) or Intel
- ~40GB free disk space per minute of 4K video (temporary)

### Windows
- Windows 10/11 64-bit
- Visual C++ Runtime 2015-2022
- Note: Vision Pro features are macOS-only

## MV-HEVC Encoding Details

The app uses a lossless workflow to ensure maximum quality:

```
Source Video
    ↓
FFmpeg (apply adjustments) → ProRes HQ (lossless)
    ↓
spatial make → MV-HEVC with Vision Pro metadata
    ↓
Final Output (ready for Vision Pro)
```

**Metadata included:**
- Camera baseline: 65mm
- Horizontal FOV: 180°
- Projection: Half-equirectangular (VR180)
- Proper VEXU boxes for Vision Pro recognition

See [MV-HEVC Guide](README_MVHEVC.md) for details.

## Building from Source

### macOS

```bash
# Install dependencies
pip3 install -r requirements.txt

# Install FFmpeg and spatial
brew install ffmpeg
brew install mikeswanson/spatial/spatial-media-kit-tool

# Build app
./build_mac.sh

# App will be in dist/ folder
```

### Windows

See [Windows Build Guide](windows_build_pack/BUILD.bat)

## Documentation

- [MV-HEVC Quick Guide](README_MVHEVC.md)
- [Technical Workflow](FINAL_MV-HEVC_WORKFLOW.md)
- [Release Notes](RELEASE_NOTES.md)

## Credits

This application builds upon and integrates several excellent open-source tools:

### spatial by Mike Swanson

The Vision Pro MV-HEVC encoding capability is powered by Mike Swanson's `spatial` CLI tool.

- **Author**: Mike Swanson
- **Project**: https://blog.mikeswanson.com/spatial/
- **Documentation**: https://blog.mikeswanson.com/spatial_docs/
- **Purpose**: MV-HEVC encoding with Apple VideoToolbox and Vision Pro APMP metadata
- **Bundled**: Yes (included in macOS app)

The `spatial` tool provides professional-grade MV-HEVC encoding using Apple's VideoToolbox framework and automatically injects the proper VEXU metadata boxes required for Vision Pro spatial video recognition.

### FFmpeg

Video processing and intermediate ProRes encoding provided by the FFmpeg project.

- **Project**: https://ffmpeg.org/
- **License**: LGPL 2.1+ / GPL 2+
- **Purpose**: Video processing, format conversion, filter application
- **Bundled**: Yes (included in app)

### Python Libraries

- **PyQt6**: GUI framework
- **NumPy**: Video frame processing
- **OpenCV**: Video analysis

## Acknowledgments

Special thanks to:

- **Mike Swanson** for creating the `spatial` CLI tool and documenting the MV-HEVC encoding workflow. His work made high-quality Vision Pro spatial video encoding accessible.
- **FFmpeg developers** for the powerful media processing framework
- **x265 project** for HEVC encoding technology
- **PyQt team** for the cross-platform GUI framework

## License

[Your License Here - Choose: MIT, GPL, Apache, etc.]

This project uses:
- `spatial` - Check Mike Swanson's license terms
- `FFmpeg` - LGPL 2.1+ / GPL 2+

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Documentation**: See docs in repository
- **Discussions**: [GitHub Discussions](../../discussions)

## Roadmap

Future features being considered:
- Batch processing
- Preset management
- Advanced color grading
- GPU acceleration for filters

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Disclaimer

This is an independent project and is not affiliated with, endorsed by, or sponsored by Apple Inc., Mike Swanson, or the FFmpeg project.

Apple, Vision Pro, ProRes, and VideoToolbox are trademarks of Apple Inc.

## Version

**Current Version**: 1.3.0
**Release Date**: January 2026
**Compatibility**: macOS 11.0+, Windows 10/11
