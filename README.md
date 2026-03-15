# VR180 Silver Bullet

The companion processing software for the [GoPro Max 2 VR180 Mod](https://www.facebook.com/share/p/1HFBj2kauf/). Converts .360 footage into VR180 side-by-side output with gyro stabilization, rolling shutter correction, real-time preview, LUT support, and hardware-accelerated export.

Also works with any standard equirectangular VR180 footage for panoramic remapping adjustments, stereo alignment, color grading, VR180 metadata injection, and Apple Vision Pro spatial video export.

---

## Features

### Input Support
- **[GoPro Max 2 VR180 Mod](https://www.facebook.com/share/p/1HFBj2kauf/) .360**: Dual-HEVC EAC cross assembly with full gyro stabilization pipeline
- **Standard VR180**: Any side-by-side half-equirectangular stereo video — panoramic yaw/pitch/roll adjustment, stereo alignment, LUT grading, metadata injection, and Vision Pro MV-HEVC export

### Gyro Stabilization
- **CORI/IORI Orientation**: Parsed from GoPro GPMF telemetry
- **Split-Axis Smoothing**: Independent heading (yaw+pitch) and roll windows
- **Horizon Lock**: Drift-free roll correction via per-frame GRAV accelerometer data
- **Rolling Shutter Correction**: 800Hz GYRO-based per-scanline correction with fisheye time map
- **Gravity Alignment**: GRAV-based world-frame alignment for accurate horizon

### Adjustments
- **Global Shift**: Horizontal pixel alignment for stereo matching
- **Global Yaw/Pitch/Roll**: Corrections applied to both eyes
- **Stereo Offset**: Independent left/right eye adjustments for parallax tuning

### Color Grading
- **LUT Support**: Apply .cube LUT files with 0-100% intensity blending
- **Live Preview**: See all adjustments and LUTs in real-time

### Output Options
- **H.265 (HEVC)**: Quality (CRF) or Bitrate mode
- **ProRes 422**: Proxy, LT, Standard, or HQ profiles
- **Apple Vision Pro**: MV-HEVC spatial video export via [spatial](https://blog.mikeswanson.com/spatial) CLI tool (macOS)
- **Hardware Acceleration**: VideoToolbox (macOS), NVENC with multipass (Windows), MLX (Apple Silicon GPU)
- **Up to 8K Output**: Configurable resolution

### User Experience
- **Real-time Preview**: Multiple modes (Left/Right/SBS/Anaglyph/Overlay/Difference)
- **Progress Tracking**: FPS counter and time estimation
- **Settings Persistence**: Remembers your preferences

---

## Downloads

Pre-built binaries are available on the [Releases](../../releases) page.

### macOS
1. Download `VR180-Silver-Bullet-v1.0.0-macOS.zip`
2. Unzip and drag **VR180 Silver Bullet.app** to Applications
3. Right-click the app and select "Open" (first time only)

If you get a "damaged app" warning:
```bash
xattr -cr "/Applications/VR180 Silver Bullet.app"
```

### Windows
1. Download `VR180-Silver-Bullet-v1.0.0-Windows.zip`
2. Extract the folder and run **VR180 Silver Bullet.exe**
3. Install [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive) for NVIDIA GPU acceleration (any 12.x version works; must be below 13)

---

## Quick Start

1. **Load Video**: Click "Browse" to select your VR180 video
2. **Make Adjustments**: Use the sliders to adjust global shift, yaw/pitch/roll
3. **Add LUT** (optional): Browse for a .cube file and adjust intensity
4. **Preview**: Click "Update Preview" to see changes
5. **Export**: Choose output format and click "Start Processing"

---

## Controls

### Global Shift
Horizontal pixel shift for aligning left/right eyes. Useful for correcting horizontal misalignment in the stereo pair.

### Global Adjustments
- **Yaw**: Horizontal rotation (left/right)
- **Pitch**: Vertical rotation (up/down)
- **Roll**: Tilt rotation (clockwise/counterclockwise)

Applied to both eyes equally.

### Stereo Offset
Fine-tune the difference between left and right eyes:
- **Yaw Offset**: Adjust convergence/divergence
- **Pitch Offset**: Vertical parallax adjustment
- **Roll Offset**: Tilt difference between eyes

### Preview Modes
- **Left Eye**: Show only left eye
- **Right Eye**: Show only right eye
- **Side by Side**: Full stereo view
- **Anaglyph**: Red/Cyan 3D glasses view
- **Overlay 50%**: Blend both eyes equally
- **Difference**: Highlight differences between eyes

---

## Output Settings

### H.265 (HEVC)
Modern codec with excellent compression:
- **Quality Mode**: CRF 0-51 (lower = better quality, default 18)
- **Bitrate Mode**: Set Mbps directly (default 50)
- **Hardware Accel**: Uses VideoToolbox (Mac) or NVENC (Windows) if available

### ProRes 422
Professional codec for editing:
- **Proxy**: Smallest, for offline editing
- **LT**: Light, good for most editing
- **Standard**: Full quality (default)
- **HQ**: Highest quality

---

## LUT Support

### Supported Formats
- .cube files (most common)
- .3dl files

### Using LUTs
1. Click "Browse" next to LUT File
2. Select your .cube file
3. Adjust "LUT Intensity" slider (0-100%)
4. Click "Update Preview" to see the result
5. Export with LUT applied

### LUT Intensity
- **0%**: Original image, no LUT
- **50%**: 50/50 blend of original and LUT
- **100%**: Full LUT applied

---

## Technical Details

### Bundled Components
- FFmpeg 8.0.1 with full codec support
- FFprobe for media inspection
- [spatial](https://blog.mikeswanson.com/spatial) CLI tool for Vision Pro MV-HEVC (macOS)
- Python runtime and all dependencies
- No external installation required

### Video Processing
- Uses FFmpeg's v360 filter for transformations
- Lossless filter application (no quality loss from adjustments)
- Hardware acceleration when available
- Preserves audio and metadata

### System Requirements

| | macOS | Windows |
|---|---|---|
| **OS** | macOS 13 Ventura or later | Windows 10/11 |
| **Processor** | Apple Silicon (M1 or later) | Any modern x64 CPU |
| **RAM** | 8 GB minimum (16 GB recommended for 8K) | 8 GB minimum (16 GB recommended for 8K) |
| **GPU** | Built-in (MLX acceleration on Apple Silicon) | NVIDIA GPU recommended (GTX 1060+ / RTX series) |
| **Disk** | 500 MB free space | 500 MB free space |
| **CUDA** | — | [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive) required for GPU export (must be below version 13; any 12.x from the [archive](https://developer.nvidia.com/cuda-toolkit-archive) works) |

---

## Troubleshooting

### macOS

**"App is damaged and can't be opened"**
```bash
xattr -cr "/Applications/VR180 Silver Bullet.app"
```

**Preview is black**
- Check that your video file is valid
- Try clicking "Update Preview" again
- Check FFmpeg output window for errors

### Windows

**NumPy import error**
See `WINDOWS_BUILD_FIX.md` for detailed solutions.

**Missing DLL errors**
Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Building from Source

### Requirements
- Python 3.11+
- FFmpeg (installed and in PATH)
- See `requirements.txt` for Python dependencies

### macOS
```bash
pip install -r requirements.txt
./build_mac.sh
```

### Windows
```bash
pip install -r requirements.txt
build_windows.bat
```

---

## Acknowledgments

- **[Gyroflow](https://gyroflow.xyz/)** — The gyro stabilization and rolling shutter correction in this project were heavily inspired by Gyroflow, an incredible open-source tool for video stabilization using gyroscope data. Huge thanks to the Gyroflow team for pioneering accessible gyro-based stabilization.
- **[spatial](https://blog.mikeswanson.com/spatial)** by Mike Swanson — The `spatial` CLI tool enables MV-HEVC encoding for Apple Vision Pro spatial video playback.

---

**Version**: 1.0.0
**Built With**: Python, PyQt6, FFmpeg, OpenCV, NumPy, SciPy
