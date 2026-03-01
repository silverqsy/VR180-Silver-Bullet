# Final MV-HEVC Workflow - Simplified

## The Optimal Solution

Your VR180 Silver Bullet app now uses the **best available workflow** for MV-HEVC encoding:

```
FFmpeg (filters) → ProRes HQ → spatial make → MV-HEVC + Vision Pro metadata
  (adjustments)     (lossless)   (Apple encoder + VEXU boxes)
```

## Why This Workflow

✅ **No double lossy compression** - ProRes is lossless
✅ **Full bitrate control** - Set any bitrate you want (350+ Mbps)
✅ **Proper Vision Pro metadata** - cdist=65, hfov=180, all VEXU boxes
✅ **Hardware accelerated** - Fast ProRes encoding on Apple Silicon
✅ **Works out of the box** - No building from source required
✅ **Industry standard** - Apple's official MV-HEVC encoder

## How It Works

### Step 1: FFmpeg Processing
```
Input video → Apply adjustments → Output ProRes HQ
```
**What happens:**
- IPD shift (horizontal convergence)
- V360 rotation (yaw, pitch, roll per eye)
- Color grading (gamma, white point, black point)
- LUT application with intensity blending

**Output:** Lossless ProRes HQ (temporary file)

### Step 2: MV-HEVC Encoding with spatial
```
ProRes HQ → spatial make → MV-HEVC with metadata
```
**What happens:**
- Encodes to MV-HEVC using Apple VideoToolbox
- Injects Vision Pro APMP metadata automatically
- Your specified bitrate (e.g., 350 Mbps)
- Deletes ProRes temp file

**Output:** Final MV-HEVC file ready for Vision Pro

## Metadata Injected

Every MV-HEVC file gets proper Vision Pro spatial video metadata:

```
vexu:cameraBaseline = 65mm           (your camera's IPD)
vexu:horizontalFieldOfView = 180°    (VR180 field of view)
vexu:horizontalDisparityAdjustment = 0 (neutral stereo)
vexu:projectionKind = halfEquirectangular (VR180)
vexu:heroEyeIndicator = left         (primary eye for 2D)
```

## Quality Comparison

### Old Method (Before ProRes Intermediate)
```
Source → H.265 (lossy) → MV-HEVC (lossy again)
Result: Double compression, visible artifacts
```

### New Method (Current)
```
Source → ProRes (lossless) → MV-HEVC (lossy once)
Result: Maximum quality, no generational loss
```

**Quality difference most visible in:**
- Fine details and textures
- Color gradients (less banding)
- High-motion scenes (fewer compression artifacts)
- Shadow and highlight detail

## Performance

### Encoding Speed
- **FFmpeg → ProRes**: 1-3x realtime (hardware accelerated)
- **spatial make**: 0.5-1x realtime (VideoToolbox)
- **Total**: Similar to old workflow, much better quality

### Disk Space (Temporary)
- **4K VR180**: ~20-40 GB per minute
- **8K VR180**: ~80-150 GB per minute
- ProRes temp file is **automatically deleted** after encoding

## Why We Didn't Use x265 Multiview

We explored using x265 multiview encoding, but decided against it:

❌ **Requires building from source** - Complex setup
❌ **Much slower** - CPU only, 0.1-0.3x realtime
❌ **Huge temp files** - Raw YUV adds 40+ GB
❌ **Additional complexity** - More things that can break
❌ **Minimal quality benefit** - Apple's encoder is excellent

✅ **Current workflow is simpler and just as good**

## Why We Didn't Use FFmpeg MV-HEVC Encoding

FFmpeg 8.0 has MV-HEVC **decoding** but NOT encoding:

❌ FFmpeg can decode MV-HEVC (read spatial videos)
❌ FFmpeg CANNOT encode MV-HEVC (create spatial videos)
✅ Only way to encode: x265 multiview or Apple's encoder
✅ We use Apple's encoder via `spatial` CLI (optimal)

## Using the App

### Vision Pro MV-HEVC Mode

1. **Select H.265 codec**
2. **Set your bitrate** (e.g., 350 Mbps)
   - Or use CRF quality mode
3. **Choose "Vision Pro MV-HEVC" mode**
4. **Process!**

**What happens:**
```
Processing... → (FFmpeg applies filters to ProRes)
Encoding MV-HEVC at 350Mbps... → (spatial make)
✓ Complete! (Vision Pro MV-HEVC)
```

### Output

Your video will:
- ✅ Play as spatial video on Vision Pro
- ✅ Have proper stereoscopic 3D
- ✅ Show correct field of view (180°)
- ✅ Maintain high quality at your bitrate
- ✅ Work in Photos app, Files app, Safari
- ✅ Be ready for distribution

## Technical Details

### Tools Used

1. **FFmpeg 8.0.1**
   - Applies all video filters
   - Outputs to ProRes HQ

2. **spatial CLI (v0.6.2)**
   - Mike Swanson's open-source tool
   - Uses Apple VideoToolbox via AVFoundation
   - Injects VEXU metadata boxes

3. **Bundled in app:**
   - ✅ FFmpeg binary
   - ✅ FFprobe binary
   - ✅ spatial binary

### File Formats

**Intermediate:** ProRes 422 HQ
- 10-bit color depth
- 4:2:2 chroma subsampling
- Intraframe compression
- Mathematically lossless

**Final:** MV-HEVC in MOV container
- Multi-view HEVC codec
- APMP metadata (Apple Projected Media Profile)
- VEXU boxes (Video Extended Usage)
- hvc1 tag for compatibility

## Bitrate Recommendations

### 4K VR180 (3840x1920)
- **Low**: 100-150 Mbps
- **Medium**: 200-250 Mbps
- **High**: 300-350 Mbps
- **Maximum**: 400-500 Mbps

### 5.7K VR180 (5760x2880)
- **Low**: 200-250 Mbps
- **Medium**: 300-400 Mbps
- **High**: 450-600 Mbps
- **Maximum**: 700-1000 Mbps

### 8K VR180 (7680x3840)
- **Low**: 300-400 Mbps
- **Medium**: 500-700 Mbps
- **High**: 800-1200 Mbps
- **Maximum**: 1500+ Mbps

## Troubleshooting

### "spatial CLI tool not found"
**Solution:** The app bundles spatial automatically. This shouldn't happen unless the build failed.

### "No space left on device"
**Solution:** Free up disk space. ProRes needs ~40GB per minute for 4K.

### Output quality not as expected
**Check:**
- Bitrate setting - Higher = better quality
- Source quality - Can't improve bad source
- View on Vision Pro - Desktop preview may not show full quality

### Encoding is slow
**Normal!** MV-HEVC encoding is computationally intensive:
- ProRes: Fast (1-3x realtime)
- spatial make: Moderate (0.5-1x realtime)
- Higher resolution = slower

## Version History

- **v1.0.0**: Initial release (avconvert, limited bitrate)
- **v1.0.1**: spatial CLI integration (full bitrate)
- **v1.1.0**: ProRes intermediate (no double compression)
- **v1.2.0**: Explored x265 multiview (too complex)
- **v1.3.0**: Final simplified workflow ⭐ YOU ARE HERE

## Credits

- **spatial CLI**: Mike Swanson - https://blog.mikeswanson.com/spatial/
- **FFmpeg**: FFmpeg developers - https://ffmpeg.org/
- **Apple VideoToolbox**: Apple Inc.
- **ProRes codec**: Apple Inc.

## Summary

The current workflow is the **optimal solution** for MV-HEVC encoding:

✅ Maximum quality (lossless intermediate)
✅ Full bitrate control (350+ Mbps)
✅ Proper Vision Pro metadata
✅ Fast and reliable
✅ Simple and maintainable
✅ Works out of the box

No further optimization needed!
