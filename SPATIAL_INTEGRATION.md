# Spatial Integration for High-Bitrate MV-HEVC

## What Changed

Your VR180 Silver Bullet app now uses Mike Swanson's **spatial CLI tool** instead of Apple's `avconvert` for MV-HEVC encoding. This gives you:

✅ **Full bitrate control** - Set ANY bitrate you want (350+ Mbps)
✅ **Vision Pro APMP metadata** - Automatic spatial video metadata injection
✅ **No Apple limitations** - Not restricted to ~87 Mbps anymore

## How It Works

### Previous Method (avconvert)
- Limited to ~60-100 Mbps regardless of settings
- Used preset-based quality (no direct bitrate control)
- Apple's hard-coded limitations

### New Method (spatial)
- Direct bitrate control: `--bitrate 350M`
- Uses VideoToolbox encoder with proper metadata
- Injects Vision Pro APMP metadata automatically
- Same quality as using spatial CLI manually

## What's Bundled

The app now includes:
- **FFmpeg** (video processing)
- **FFprobe** (video analysis)
- **spatial** (MV-HEVC encoding with metadata) ⭐ NEW!

Location: `VR180 Silver Bullet.app/Contents/Frameworks/spatial`

## Using the App

### Vision Pro MV-HEVC Mode

1. **Select codec**: H.265 (HEVC)
2. **Set bitrate**: Choose your desired bitrate (e.g., 350 Mbps)
   - OR use CRF quality mode (will estimate bitrate)
3. **Vision Pro Mode**: Select "Vision Pro MV-HEVC (full APMP - slow)"
4. **Process**: The app will:
   - Encode video with your adjustments using H.265
   - Re-encode to MV-HEVC using `spatial` at your specified bitrate
   - Inject Vision Pro APMP metadata automatically

### Bitrate Examples

- **62 Mbps** (old): What you were getting with avconvert
- **350 Mbps** (new): High-quality preservation
- **500+ Mbps** (new): Maximum quality for archival

The bitrate you set in the H.265 settings will be used for MV-HEVC encoding!

## Technical Details

### Metadata Injected

The `spatial` tool automatically injects:
- `vexu:cameraBaseline` - 63mm (typical VR180 camera)
- `vexu:horizontalFieldOfView` - 90° (VR180 FOV)
- `vexu:horizontalDisparityAdjustment` - 0 (neutral stereo)
- `vexu:projectionKind` - halfEquirectangular (VR180)
- `vexu:heroEyeIndicator` - left (primary eye for 2D)

### Command Example

What the app runs internally:
```bash
spatial make \
  --input processed_video.mov \
  --output final_mvhevc.mov \
  --format sbs \
  --bitrate 350M \
  --cdist 63 \
  --hfov 90 \
  --hadjust 0 \
  --projection halfEquirect \
  --hero left \
  --faststart \
  --overwrite
```

## Requirements

- **macOS**: 11.0 or later (Big Sur+)
- **Architecture**: Apple Silicon (M1/M2/M3) or Intel
- **spatial tool**: Already bundled in the app!

## Notes

- MV-HEVC encoding is **slower** than regular H.265 (expect ~0.5-1x realtime)
- Output files will be **larger** with higher bitrates (as expected)
- Vision Pro will recognize these files as native spatial video
- YouTube VR180 metadata is separate (not compatible with MV-HEVC mode)

## Troubleshooting

**Error: "spatial CLI tool not found"**
- This means the bundled spatial binary isn't being found
- Shouldn't happen with the new build, but if it does:
  - Install manually: `brew install mikeswanson/spatial/spatial-media-kit-tool`

**Encoding is very slow**
- MV-HEVC is computationally intensive
- Higher bitrates = larger files = longer encoding
- This is normal behavior

**Output bitrate doesn't match exactly**
- VideoToolbox may adjust slightly based on content complexity
- Should be very close to your target (within 5-10%)

## Credits

- **Mike Swanson** - Creator of the `spatial` CLI tool
- **spatial version**: 0.6.2
- **Project**: https://blog.mikeswanson.com/spatial
- **GitHub**: https://github.com/mikeswanson/SpatialPlayer

## Version History

- **v1.0.0** - Initial release with avconvert (limited bitrate)
- **v1.1.0** - Integrated spatial CLI (full bitrate control) ⭐ YOU ARE HERE
