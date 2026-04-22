# VR180 Silver Bullet

A professional VR180 stereo video processor for the **GoPro Max 2 VR180 Mod**. Converts .360 EAC dual-track footage into side-by-side VR180 with gyro stabilization, rolling shutter correction, color grading, and Apple Vision Pro spatial video output.

## Features

### True 10-bit End-to-End Pipeline
- Full uint16 processing — decode, remap, LUT, sharpen, saturation, color grading all at 16-bit
- PyAV hardware decode (VideoToolbox on macOS, NVDEC on Windows)
- No 8-bit truncation anywhere in the pipeline

### GoPro .360 Processing
- Dual HEVC stream decode (s0 + s4) with EAC cross assembly
- Per-lens GEOC calibration (FRNT/BACK KLNS from camera metadata)
- Half-equirectangular and native fisheye output projections
- Multi-segment recording support with automatic chapter detection

### Gyro Stabilization & Rolling Shutter Correction
- CORI/IORI quaternion parsing from GPMF metadata
- 800Hz GYRO-based rolling shutter correction with per-pixel fisheye time mapping
- GRAV-based gravity alignment for horizon lock
- No-ERS firmware support — VQF 9-axis IMU fusion for cameras with disabled electronic rolling shutter
- Configurable smoothing, max correction, and per-axis RS factors

### Apple Vision Pro Spatial Video
- **APMP metadata** — SBS exports tagged with vexu/eyes/proj/pack/cams/hfov atoms for native visionOS 26+ playback
- **MV-HEVC direct encoder** (macOS only) — Built-in VideoToolbox spatial video encoding, no external dependencies
- 65mm stereo baseline matching the GoPro Max 2 VR180 Mod lens separation

### Temporal Denoise (macOS only)
- VTTemporalNoiseFilter with genuine 10-bit processing via 64RGBALE intermediate
- Hardware-accelerated multi-frame motion-compensated noise reduction
- Requires macOS 26+ on Apple Silicon

### Color Grading
- .cube 3D LUT support with intensity control
- ASC CDL: Lift, Gamma, Gain, Saturation
- Bundled Recommended LUT for GoPro LOG (auto-loads for .360 input)

### Output
- H.265 10-bit (VideoToolbox / NVENC / libx265)
- ProRes 422 (Proxy through 4444XQ)
- MV-HEVC spatial video for Apple Vision Pro
- Proper BT.709 color tagging on all codecs
- Trim with frame-accurate in/out points

## Downloads

### macOS
Download the signed and notarized app from [Releases](https://github.com/silverqsy/VR180-Silver-Bullet/releases).

**Requirements**: macOS 14+ (Sonnet), Apple Silicon recommended. Temporal denoise requires macOS 26+.

### Windows
Clone the repo and use the build kit:
```
cd VR180_Silver_Bullet_Windows_BuildKit
build_windows.bat
```
**Requirements**: Python 3.10+, FFmpeg in PATH, NVIDIA GPU recommended for NVDEC/NVENC.

## Building from Source

### macOS
```bash
pip install -r requirements.txt
# Build Swift helpers
swiftc -O -o mvhevc_encode mvhevc_encode.swift \
    -framework AVFoundation -framework VideoToolbox \
    -framework CoreMedia -framework CoreVideo -framework Accelerate
swiftc -O -o vt_denoise vt_denoise.swift \
    -framework AVFoundation -framework VideoToolbox \
    -framework CoreMedia -framework CoreVideo
# Build app bundle
python -m PyInstaller --clean vr180_silver_bullet.spec
```

### Windows
```
pip install -r requirements.txt
python -m PyInstaller --clean vr180_silver_bullet.spec
```

## GPU Acceleration

| Backend | Platform | Used For |
|---------|----------|----------|
| MLX Metal | macOS (Apple Silicon) | EAC remap, 3D LUT, sharpen |
| CUDA/Numba | Windows (NVIDIA) | EAC remap, 3D LUT |
| VideoToolbox | macOS | Decode, H.265/ProRes encode, MV-HEVC, denoise |
| NVENC/NVDEC | Windows | Decode, H.265 encode |
| Numba CPU | All | Fallback for remap, LUT, sharpen |

## License

MIT
