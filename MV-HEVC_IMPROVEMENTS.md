# MV-HEVC Quality Improvements - Version 1.1

## What Changed

### ✅ Fixed Metadata Values
- **cdist**: Changed from 63mm → **65mm** (matches your camera)
- **hfov**: Changed from 90° → **180°** (full VR180 field of view)

### ✅ Eliminated Double Lossy Compression

**Previous Workflow (Quality Loss):**
1. FFmpeg: Apply adjustments → Encode to **H.265** (lossy)
2. spatial: Re-encode H.265 → **MV-HEVC** (lossy again)
3. **Result**: Double compression = quality degradation

**New Workflow (Lossless Intermediate):**
1. FFmpeg: Apply adjustments → Encode to **ProRes HQ** (lossless)
2. spatial: Encode ProRes → **MV-HEVC** (single lossy compression)
3. **Result**: Maximum quality preservation!

## Why This Matters

### Quality Comparison

**Old Method:**
- Source → H.265 (loses quality) → MV-HEVC (loses more quality)
- Each compression step introduces artifacts
- Final output has generational loss

**New Method:**
- Source → ProRes (lossless) → MV-HEVC (loses quality once)
- Only one lossy compression step
- Final output has no generational loss

### File Size Impact

The ProRes intermediate file is **temporary** and deleted after conversion:
- ProRes intermediate: ~10-50 GB (temporary)
- Final MV-HEVC: Your specified bitrate (e.g., 350 Mbps)

## Technical Details

### ProRes HQ Profile

The app now uses **ProRes 422 HQ (profile 3)** as the intermediate:
- 10-bit color depth (preserves gradients)
- 4:2:2 chroma subsampling (better color fidelity)
- Intraframe compression (fast decode for spatial processing)
- Mathematically lossless for most content

### Encoder Selection

- **macOS with hardware accel**: `prores_videotoolbox` (fast GPU encoding)
- **Software fallback**: `prores_ks` (CPU encoding, slower but works everywhere)

## Workflow Steps

When you process a video with Vision Pro MV-HEVC mode:

1. **FFmpeg Processing** (Step 1)
   - Applies all your adjustments (IPD, rotation, color, LUTs)
   - Outputs to ProRes HQ intermediate
   - Status: "Processing... X% - Y fps"

2. **MV-HEVC Encoding** (Step 2)
   - `spatial` encodes ProRes → MV-HEVC
   - Injects Vision Pro APMP metadata
   - Uses your specified bitrate (e.g., 350 Mbps)
   - Status: "Encoding MV-HEVC at 350Mbps..."

3. **Cleanup**
   - Deletes temporary ProRes file
   - Final MV-HEVC file ready!

## Performance Notes

### Speed

- **ProRes encoding**: Very fast (often >2x realtime with VideoToolbox)
- **MV-HEVC encoding**: Slower (~0.5-1x realtime)
- **Total time**: Similar to before, but much better quality!

### Disk Space

During processing, you need temporary space for ProRes:
- **4K VR180**: ~20-40 GB per minute
- **8K VR180**: ~80-150 GB per minute

Make sure you have enough free space before processing long videos.

## Comparing Old vs New

### Example: 1-minute 4K VR180 video

**Old Method:**
```
Source (H.264, 100 Mbps)
  ↓ FFmpeg (lossy)
H.265 (350 Mbps, generation 1 loss)
  ↓ spatial (lossy)
MV-HEVC (350 Mbps, generation 2 loss) ← Visible artifacts
```

**New Method:**
```
Source (H.264, 100 Mbps)
  ↓ FFmpeg (lossless)
ProRes HQ (~2000 Mbps, no loss)
  ↓ spatial (lossy)
MV-HEVC (350 Mbps, generation 1 loss) ← Much cleaner!
```

## When to Use This

**Recommended for:**
- ✅ High-quality archival encodes
- ✅ Professional VR180 content
- ✅ Maximum quality for Vision Pro viewing

**Not needed for:**
- ❌ Quick previews/tests (use regular H.265 mode)
- ❌ Very low bitrates (<100 Mbps) where difference is minimal

## Troubleshooting

**Error: "No space left on device"**
- Free up disk space before processing
- ProRes needs ~40 GB per minute for 4K
- Temporary file is deleted after conversion

**Processing seems slower**
- ProRes encoding is actually faster
- But file size is larger, so disk I/O takes time
- Overall speed should be similar to before

**Output quality doesn't look different**
- Quality difference most visible in:
  - Fine details and textures
  - Color gradients (less banding)
  - High-motion scenes (fewer compression artifacts)
- Compare at full resolution on Vision Pro

## Version History

- **v1.0.0**: Initial MV-HEVC support with avconvert (limited bitrate)
- **v1.0.1**: Switched to spatial CLI (full bitrate control)
- **v1.1.0**: ProRes intermediate + fixed metadata (cdist=65, hfov=180) ⭐ YOU ARE HERE

## Credits

- **Lossless intermediate technique**: Industry standard practice
- **ProRes codec**: Apple (royalty-free for software encoding)
- **spatial tool**: Mike Swanson
