# VR180 Silver Bullet - Platform Feature Comparison

## Summary: What Works on Windows

**Good news:** Almost all features work on Windows! The only limitations are macOS-specific Vision Pro features.

## Complete Feature Matrix

### ✅ Core VR180 Processing (Both Platforms)
| Feature | Windows | macOS | Notes |
|---------|---------|-------|-------|
| Global Shift | ✅ | ✅ | Horizontal alignment correction |
| Global Yaw/Pitch/Roll | ✅ | ✅ | 3D rotation adjustments |
| Stereo Offset | ✅ | ✅ | Left/right eye alignment |
| Timeline Scrubbing | ✅ | ✅ | Preview any frame |
| Pan/Zoom Preservation | ✅ | ✅ | Keeps your view when adjusting |

### 🎨 Color Grading (Both Platforms)
| Feature | Windows | macOS | Notes |
|---------|---------|-------|-------|
| Gamma Adjustment | ✅ | ✅ | 10-300% range |
| White Point | ✅ | ✅ | Highlight control |
| Black Point | ✅ | ✅ | Shadow control |
| LUT Support (.cube) | ✅ | ✅ | Professional color grading |
| LUT Intensity Blend | ✅ | ✅ | 0-100% mixing |

### 📹 Video Codecs
| Codec | Windows | macOS | Performance |
|-------|---------|-------|-------------|
| H.265 (HEVC) | ✅ | ✅ | Windows: Good with NVENC<br>macOS: Good with VideoToolbox |
| H.265 8-bit | ✅ | ✅ | Best compatibility |
| H.265 10-bit | ✅ | ✅ | Better gradients |
| ProRes Proxy | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |
| ProRes LT | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |
| ProRes Standard | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |
| ProRes HQ | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |
| ProRes 4444 | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |
| ProRes 4444 XQ | ✅ | ✅ | Windows: Slow (software)<br>macOS: Fast (hardware) |

### 🔧 Encoding Options
| Feature | Windows | macOS | Notes |
|---------|---------|-------|-------|
| CRF Quality Mode | ✅ | ✅ | Constant quality |
| Bitrate Mode | ✅ | ✅ | Target bitrate |
| Hardware Acceleration | ✅ NVENC | ✅ VideoToolbox | Requires compatible GPU |
| Software Encoding | ✅ | ✅ | Fallback option |

### 👁️ Preview Modes
| Mode | Windows | macOS | Description |
|------|---------|-------|-------------|
| Side by Side | ✅ | ✅ | Full stereo view |
| Anaglyph | ✅ | ✅ | Red/Cyan 3D glasses |
| Overlay 50% | ✅ | ✅ | Alignment checking |
| Single Eye Mode | ✅ | ✅ | With eye toggle (L/R) |
| Difference | ✅ | ✅ | Stereo mismatch detection |
| Checkerboard | ✅ | ✅ | Pattern-based comparison |

### 📱 Platform-Specific Features
| Feature | Windows | macOS | Tool Required |
|---------|---------|-------|---------------|
| YouTube VR180 Metadata | ✅ | ✅ | FFmpeg |
| Apple hvc1 Tag | ❌ | ✅ | macOS MP4Box/avconvert |
| Vision Pro MV-HEVC | ❌ | ✅ | macOS avconvert |

## Why Some Features Are macOS-Only

### Vision Pro MV-HEVC
- Requires Apple's `avconvert` tool
- Uses proprietary APMP metadata format
- Only available on macOS 13.0+
- Creates MV-HEVC with spatial video metadata

### Apple hvc1 Tag
- Requires specific MP4 atom modification
- Apple's tools ensure compatibility
- Alternative: Can manually tag on Windows with MP4Box (not included)

## Performance Comparison

### H.265 Encoding Speed
**Windows (NVIDIA RTX 3060):**
- 4K VR180: ~0.5-1x realtime with NVENC
- Software: ~0.1-0.3x realtime

**macOS (M1/M2/M3):**
- 4K VR180: ~1-2x realtime with VideoToolbox
- Software: ~0.2-0.5x realtime

### ProRes Encoding Speed
**Windows:**
- 4K VR180: ~0.1-0.3x realtime (software only)
- No hardware acceleration available

**macOS:**
- 4K VR180: ~2-5x realtime (VideoToolbox)
- Significantly faster than Windows

## Recommended Workflows

### For Windows Users
**Best Quality + Speed:**
1. Codec: H.265
2. Bit Depth: 10-bit
3. Quality: CRF 20-23
4. Hardware Acceleration: ON (if NVIDIA GPU)
5. Upload: YouTube VR180

**Maximum Quality (Slow):**
1. Codec: ProRes HQ
2. Hardware Acceleration: N/A
3. Use for: Editing, archival
4. Note: Very slow on Windows

### For macOS Users
**Best Quality + Speed:**
1. Codec: ProRes HQ
2. Hardware Acceleration: ON
3. Use for: Editing in Final Cut Pro

**Vision Pro:**
1. Codec: H.265 10-bit
2. Vision Pro Mode: MV-HEVC
3. Hardware Acceleration: ON
4. Creates: Spatial video for Vision Pro

**YouTube Upload:**
1. Codec: H.265
2. Bit Depth: 10-bit
3. Quality: CRF 18-20
4. VR180 Metadata: ON
5. Vision Pro Mode: Standard

## Windows Build Instructions

See `WINDOWS_BUILD_GUIDE.md` for detailed build instructions.

**Quick Summary:**
1. Install Python 3.11+ and FFmpeg
2. Run: `pip install PyQt6 numpy Pillow pyinstaller`
3. Run: `pyinstaller --clean vr180_processor.spec`
4. Find app in: `dist\VR180 Silver Bullet\`

## File Compatibility

Both platforms support:
- Input: MP4, MOV, MKV
- LUT: .cube files
- Output: MP4, MOV (depending on codec)

## Conclusion

**Windows users get 95% of features!** The only missing features are:
- Vision Pro MV-HEVC export (requires Apple tools)
- Apple hvc1 tag (can be added with third-party tools)

All core VR180 processing, color grading, and export features work perfectly on Windows.
