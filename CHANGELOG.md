# Changelog

## 2.0.0

The `2.0` clean-room rewrite of VR180 Silver Bullet — a native Rust + wgpu
application replacing the Python/PyQt6 app. One self-contained binary per
platform, no Python runtime, no system `ffmpeg`. Runs on **macOS (Apple
Silicon)** and **Windows (NVIDIA)**.

### Cameras & formats
- **DJI Osmo 360** (`.osv`) — primary path, with exact per-lens factory
  dewarp loaded from the file (5-coefficient Kannala-Brandt + Brown-Conrady
  tangential, reverse-engineered and bit-matched to DJI Studio).
- **GoPro Max** (`.360`, EAC) — full GPU pipeline: zero-copy decode,
  noise reduction, and **automatic firmware vs no-firmware rolling-shutter
  detection** from the CORI stream (manual override retained).
- **SBS fisheye** (Insta360 / Vuze XR / QooCam / Canon RF / generic rigs).
- **Blackmagic RAW** (`.braw`) with VQF 6D stabilization.

### Engine
- GPU-first: `wgpu` compute (Metal / DX12 / Vulkan) with WGSL shaders.
- In-process video I/O via `ffmpeg-next` 8.1 (no subprocess).
- **macOS:** VideoToolbox zero-copy P010 decode/encode through IOSurface,
  HEVC and hardware ProRes.
- **Windows:** GPU-resident export — NVDEC → wgpu → CUDA → NVENC, no CPU
  readback (~36 fps @ 8K on a 4090), libx265 fallback.
- 10-bit end-to-end (Rgba16Unorm intermediates) when 10-bit output is
  selected — decode, projection, color stack, and encode all hold ≥10-bit.

### Stabilization & rolling shutter
- Camera-lock and velocity-dampened soft-stab (adaptive smoothing with a
  **Response** slider and a soft elastic correction limit).
- Per-scanline rolling-shutter correction from measured sensor-readout
  timing; gravity/horizon alignment.
- OSV IMU timing reverse-engineered from DJI Studio (SROT, IMU phase).
- **New:** GoPro `.360` firmware-RS mode is auto-detected per clip from the
  CORI signal (the toggle still overrides).

### Color
- CDL, 3D LUT (DJI D-LogM→Rec.709 bundled + autoloaded), white balance,
  saturation, sharpen, mid-detail — identical stack in preview and export,
  matched to the Python app.

### Noise reduction
- Temporal NR via Apple `VTTemporalNoiseFilter`, ported **in-process** (objc2
  FFI, no Swift helper), fully 10-bit, GPU-resident zero-copy for OSV and
  `.360`. Export-only; macOS-only (auto-hidden where unsupported).

### Output & delivery
- Half-equirect VR180 SBS or normalized **195° equidistant fisheye** SBS.
- Native or **8192×4096 (8K)** resolution.
- H.265 or ProRes; **Vision Pro (APMP)** and **YouTube VR180** metadata
  injection; **APAC spatial** / ambisonic / stereo audio; OSV audio
  passthrough.
- Trim-accurate exports (video + audio aligned to the trim range).

### App / UX
- Native desktop GUI (eframe/egui + egui-wgpu + wgpu 29).
- **Unified batch + export:** one queue for single or many clips, a
  persistent bottom export bar with overall progress + ETA, per-clip
  multi-select, and a completion notification. (The separate batch and
  export-options windows were removed.)
- Preview modes (SBS / anaglyph / 50% overlay / single eye), zoom magnifier
  with a native-resolution still, per-eye view adjustment, upside-down mount.
- **Localized UI: English / 简体中文** (live toggle, bundled CJK font).
- Settings persist per-OS; RS mode and IMU phase are per-clip.

### Packaging
- macOS: signed + notarized `.app` / `.dmg`.
- Windows: Inno Setup installer (per-user, Start Menu shortcut).
