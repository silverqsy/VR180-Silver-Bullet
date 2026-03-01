# VR180 Processor - Panomap Adjustment Tool

A professional GUI tool for processing VR180 side-by-side (SBS) half-equirectangular videos with real-time preview and precise alignment controls.

![VR180 Processor](screenshot.png)

## Features

### 🔧 Global Horizontal Shift (Split-Eye Fix)
Fix videos where one eye is centered while the other is split across the left/right edges of the frame. The global shift wraps the frame horizontally to properly align both eyes into standard SBS format.

### 🎯 Panomap Adjustments
- **Global adjustments**: Apply yaw, pitch, and roll corrections to both eyes simultaneously
- **Per-eye adjustments**: Fine-tune each eye individually for precise stereo alignment
- **Real-time preview**: See changes instantly without processing the entire video

### 👓 Alignment Preview Modes
- **Side by Side**: View the standard SBS output
- **Anaglyph (Red/Cyan)**: Classic 3D anaglyph for checking stereo alignment (use red/cyan glasses)
- **Overlay 50%**: Semi-transparent overlay to compare eye positions
- **Overlay Blend**: Average blend of both eyes
- **Left/Right Only**: View each eye independently
- **Difference**: Highlight differences between eyes (brighter = more misalignment)
- **Checkerboard**: Alternating blocks from each eye for edge alignment

### 🎬 Codec Support
- **H.265 (HEVC)**: High efficiency codec with quality control (CRF 0-51)
- **ProRes**: Professional codec with profile selection (Proxy to 4444 XQ)
- **Auto-detect**: Automatically matches output codec to input

## Installation

### Prerequisites
- Python 3.9+
- FFmpeg with libx265 and ProRes support

### Install FFmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Launch the GUI
```bash
python vr180_gui.py
```

### Workflow

#### Step 1: Load Video
Click "Browse..." to select your VR180 video file. Supported formats include MP4, MOV, MKV, and AVI.

#### Step 2: Fix Split-Eye Frames (if needed)
If your video has one eye in the center and another split across edges:
1. Use the **Global Horizontal Shift** slider
2. Use quick-shift buttons for common offsets (-1920, -960, etc.)
3. Preview in **Anaglyph** or **Overlay** mode to verify alignment
4. Adjust until both eyes are properly positioned side-by-side

#### Step 3: Global Panomap Adjustment
Correct the overall orientation:
- **Yaw**: Rotate left/right (horizon centering)
- **Pitch**: Rotate up/down (horizon leveling)
- **Roll**: Tilt correction

#### Step 4: Per-Eye Fine Tuning
For stereo alignment issues:
- Switch to the **Left Eye** or **Right Eye** tab
- Make small adjustments (typically < 1°)
- Use **Anaglyph** preview to check convergence
- Use **Difference** mode to spot vertical/horizontal misalignment

#### Step 5: Configure Output
- **Codec**: Auto (match input), H.265, or ProRes
- **Quality**: CRF for H.265 (18 = visually lossless, lower = better)
- **ProRes Profile**: Choose based on your workflow needs

#### Step 6: Process
Click "▶ Process Video" to render the adjusted video.

## Preview Modes Explained

| Mode | Best For |
|------|----------|
| **Side by Side** | Viewing final output layout |
| **Anaglyph** | Checking stereo depth and convergence |
| **Overlay 50%** | Comparing eye positions directly |
| **Overlay Blend** | Smooth comparison of both eyes |
| **Difference** | Finding misalignment (bright = different) |
| **Checkerboard** | Checking edge alignment on fine details |

## Tips

### Finding the Right Shift Value
- Frame width is typically 7680px (3840 per eye)
- Common shifts: ±960, ±1920, ±2880
- Use the timeline to check multiple frames

### Convergence Adjustment
- Positive left yaw + negative right yaw = eyes converge (closer objects)
- Negative left yaw + positive right yaw = eyes diverge (farther objects)

### Quality Settings
- **CRF 18**: Visually lossless, good balance
- **CRF 14-16**: High quality for mastering
- **CRF 20-23**: Good quality, smaller files

## Command Line Alternative

For batch processing, use the included CLI tool:

```bash
python vr180_cli.py input.mp4 output.mp4 \
    --shift 1920 \
    --yaw 2 --pitch -1 --roll 0.5 \
    --left-yaw 0.3 --right-yaw -0.3
```

## Technical Details

### Input Specifications
- Resolution: 7680×3840 (or any 2:1 SBS format)
- Projection: Half-equirectangular (180° × 180° per eye)
- Layout: Side-by-side (left eye left half, right eye right half)

### Processing Pipeline
1. **Global Shift**: Horizontal wrap using FFmpeg's `scroll` filter
2. **Eye Separation**: Crop into left/right halves
3. **Panomap Rotation**: FFmpeg's `v360` filter with Lanczos interpolation
4. **Recombination**: Horizontal stack back to SBS format

### Supported Codecs
- **Input**: H.264, H.265, ProRes, VP9, and more
- **Output**: H.265 (libx265), ProRes (prores_ks)

## Troubleshooting

### "FFmpeg not found"
Ensure FFmpeg is installed and in your system PATH.

### Preview is slow
- The preview extracts frames on-demand
- Larger resolution videos take longer
- Consider using proxy files for initial alignment

### Colors look wrong
The tool preserves color metadata (HDR, color space). If output looks different, check your playback software's color management.

## License

MIT License - See LICENSE file for details.

## Contributing

Pull requests welcome! Please open an issue first to discuss major changes.
