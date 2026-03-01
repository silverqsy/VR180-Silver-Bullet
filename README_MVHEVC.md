# VR180 Silver Bullet - MV-HEVC Encoding Guide

## Quick Start

Your app encodes Vision Pro MV-HEVC with full bitrate control and proper spatial metadata.

### How to Use

1. **Select codec**: H.265 (HEVC)
2. **Set bitrate**: Your desired bitrate (e.g., 350 Mbps)
3. **Vision Pro mode**: Select "Vision Pro MV-HEVC (full APMP - slow)"
4. **Process**: Click Start

**Output:** Vision Pro-ready MV-HEVC file with spatial video metadata

## The Workflow

```
Source Video
    ↓
FFmpeg applies adjustments (IPD, rotation, color, LUTs)
    ↓
ProRes HQ (lossless intermediate)
    ↓
spatial make (Apple's MV-HEVC encoder)
    ↓
MV-HEVC with Vision Pro metadata
```

**Key benefits:**
- ✅ Only **one** lossy compression step (no double encoding)
- ✅ Full bitrate control (350+ Mbps supported)
- ✅ Proper VEXU metadata (cdist=65, hfov=180)
- ✅ Hardware accelerated encoding

## Metadata

Every MV-HEVC file includes Vision Pro spatial video metadata:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cameraBaseline` | 65mm | Distance between camera lenses |
| `horizontalFieldOfView` | 180° | VR180 field of view |
| `horizontalDisparityAdjustment` | 0 | Stereo offset (neutral) |
| `projectionKind` | halfEquirectangular | VR180 projection |
| `heroEyeIndicator` | left | Primary eye for 2D playback |

## Requirements

- **macOS**: 11.0+ (Big Sur or later)
- **Disk space**: ~40GB free per minute of 4K video (temporary)
- **Tools bundled**:
  - FFmpeg 8.0.1
  - spatial CLI v0.6.2

## Quality Settings

### Recommended Bitrates

**4K VR180 (3840x1920):**
- Standard: 200 Mbps
- High: 300 Mbps
- Maximum: 400 Mbps

**5.7K VR180 (5760x2880):**
- Standard: 350 Mbps
- High: 500 Mbps
- Maximum: 700 Mbps

**8K VR180 (7680x3840):**
- Standard: 600 Mbps
- High: 900 Mbps
- Maximum: 1200 Mbps

## Performance

**Encoding Speed:**
- ProRes encoding: 1-3x realtime (GPU)
- MV-HEVC encoding: 0.5-1x realtime (VideoToolbox)

**Example:** 1 minute of 4K video takes ~1-2 minutes to process

## Technical Details

### Tools Used

1. **FFmpeg** - Video processing and ProRes encoding
2. **spatial** - MV-HEVC encoding via Apple VideoToolbox + metadata injection

### Intermediate Format

**ProRes 422 HQ:**
- 10-bit color depth
- 4:2:2 chroma subsampling
- Mathematically lossless
- Automatically deleted after encoding

### Why This Workflow?

We evaluated multiple approaches:

| Method | Quality | Speed | Complexity |
|--------|---------|-------|------------|
| ❌ avconvert | Limited | Fast | Simple |
| ❌ x265 multiview | Best | Very slow | Complex |
| ✅ **ProRes → spatial** | Excellent | Fast | Simple |

The ProRes intermediate workflow provides the best balance.

## Troubleshooting

**Error: "No space left on device"**
- Free up ~40GB per minute of video
- ProRes temp file is large but auto-deleted

**Output quality lower than expected**
- Increase bitrate setting
- Check source video quality
- View on Vision Pro for true quality assessment

**Encoding is slow**
- Normal for MV-HEVC (computationally intensive)
- Higher resolution = slower processing
- Hardware acceleration is already enabled

## Credits

- **spatial CLI**: Mike Swanson (https://blog.mikeswanson.com/spatial/)
- **FFmpeg**: FFmpeg Project (https://ffmpeg.org/)
- **Apple VideoToolbox**: Apple Inc.

## Version

Current version: **1.3.0** (Simplified ProRes workflow)

---

For full technical details, see `FINAL_MV-HEVC_WORKFLOW.md`
