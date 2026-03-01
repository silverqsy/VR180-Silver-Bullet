# Silver's VR180 Tool

A professional VR180 video processing application with real-time preview, LUT support, and advanced panomap adjustment controls.

---

## Features

### Core Processing
- **VR180 Format Support**: Process side-by-side half-equirectangular stereo videos
- **Global Shift**: Horizontal pixel alignment for stereo matching
- **Global Adjustments**: Yaw, pitch, and roll corrections for both eyes
- **Stereo Offset**: Independent left/right eye adjustments for parallax tuning

### Color Grading
- **LUT Support**: Apply .cube LUT files for color grading
- **LUT Intensity Control**: Blend from 0-100% with real-time preview
- **Live Preview**: See all adjustments and LUTs before export

### Output Options
- **H.265 (HEVC)**: Quality (CRF) or Bitrate mode
- **ProRes 422**: Proxy, LT, Standard, or HQ profiles
- **Auto Mode**: Matches input codec
- **Hardware Acceleration**: VideoToolbox (macOS), NVENC (Windows)

### User Experience
- **Real-time Preview**: See changes immediately
- **Progress Tracking**: FPS counter and percentage display
- **FFmpeg Output Display**: Monitor encoding in real-time
- **Settings Persistence**: Remembers your preferences

---

## Downloads

### macOS
**File**: `Silvers-VR180-Tool-macOS.zip` (62 MB)

**Requirements**: macOS 10.14 or later

**Installation**:
1. Download and unzip the file
2. Drag "Silver's VR180 Tool.app" to Applications
3. Right-click the app and select "Open" (first time only)
4. Click "Open" in the security dialog

If you get a "damaged app" warning:
```bash
xattr -cr "/Applications/Silver's VR180 Tool.app"
```

### Windows
**Build Instructions**: See `WINDOWS_BUILD_FIX.md`

The Windows build requires building on a Windows machine. All necessary files and scripts are included.

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
- Python runtime and all dependencies
- No external installation required

### Video Processing
- Uses FFmpeg's v360 filter for transformations
- Lossless filter application (no quality loss from adjustments)
- Hardware acceleration when available
- Preserves audio and metadata

### System Requirements
- **macOS**: 10.14 Mojave or later, 4GB RAM, 500MB free space
- **Windows**: Windows 10/11, 4GB RAM, 500MB free space
- For 4K/8K: 8GB+ RAM recommended

---

## Troubleshooting

### macOS

**"App is damaged and can't be opened"**
```bash
xattr -cr "/path/to/Silver's VR180 Tool.app"
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

## Build Information

**Version**: 1.0.0
**App Name**: Silver's VR180 Tool
**Built With**: Python, PyQt6, FFmpeg

---

## Support Files

- `README.md` - This file
- `BUILD_INSTRUCTIONS.md` - How to build from source
- `WINDOWS_BUILD_FIX.md` - Windows-specific troubleshooting
- `fix_windows_build.bat` - Automated Windows build fix script
- `build_mac.sh` - macOS build script
- `build_windows.bat` - Windows build script
