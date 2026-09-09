# Changelog

## 2.5.0

### Reframed output mode (new)
- **Format → "Reframed (rectilinear)"**: a pinhole-style side-by-side view
  of each eye instead of the VR180 half-equirect — zoom (horizontal FOV),
  pan / tilt / roll, a **Defish** blend from rectilinear to a fisheye look,
  and a 1:1 or 16:9 per-eye frame. Stabilization, stereo offsets, per-row
  rolling-shutter correction and the lens override all still apply.
  Available for DJI OSMO, Insta360 X6, Blackmagic and GoPro sources.
- Drag the preview to pan, scroll or pinch to zoom, double-click to
  recenter. The preview renders from the native frame and a paused frame
  shows the native-resolution still.
- Export writes the exact viewport at 1080 / 1440 / 2160 lines per eye —
  2:1 square or **32:9 side-by-side** (3840×1080 … 7680×2160) for AR
  glasses — with no VR180 metadata. The BeyondVR hack does not apply.

### Stabilization
- **Timing is derived from the file, no more IMU phase slider.** Each
  frame's pose is sampled at the centre row's mid-exposure using the
  file's sensor readout, exposure record and timestamps, so dark and
  bright clips and every frame rate are timed right automatically
  (verified on 25 / 30 / 50 fps DJI clips, consistent with DJI Studio's
  output).
- **Insta360 X6 (`.insv`)**: factory lens model, gyro stabilization
  matched to Insta360 Studio's output, per-sensor exposure timing.
- **OSMO 360 II** support and **Auto align** stereo alignment; GoPro
  chapter-safe firmware-RS detection.

### Since 2.0.0
- Seamless auto-update (2.1.0), `.360` lens calibration override, ProRes
  4:2:2 with GPU compose, 8K export default, Matching Eyes white-balance
  trim, BeyondVR Hack (VR180 output only), frame-exact multi-segment seams.

## 2.0.0

The `2.0` clean-room rewrite of VR180 Silver Bullet — a native Rust + wgpu
application replacing the Python/PyQt6 app. **The headline addition is full
support for the OSMO VR180 Mod** (`.osv`). One self-contained binary
per platform, no Python runtime, no system `ffmpeg`. Runs on **macOS (Apple
Silicon)** and **Windows (NVIDIA)**.

### Cameras & formats
- **OSMO VR180 Mod** (`.osv`) — **the headline of 2.0.** Exact
  per-lens factory dewarp loaded from the file (5-coefficient Kannala-Brandt
  + Brown-Conrady tangential), with output on par with DJI Studio.
- **GoPro Max 2 VR180 Mod** (`.360`, EAC) — full GPU pipeline: zero-copy decode,
  noise reduction, and **automatic firmware vs no-firmware rolling-shutter
  detection** from the CORI stream (manual override retained).

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
- Precise OSV IMU stabilization + rolling-shutter timing (SROT, IMU phase), matched to DJI Studio.
- **New:** GoPro Max 2 VR180 Mod (`.360`) firmware-RS mode is auto-detected
  per clip from the CORI signal (the toggle still overrides).

### Color
- CDL, 3D LUT (DJI D-LogM→Rec.709 bundled + autoloaded), white balance,
  saturation, sharpen, mid-detail — identical stack in preview and export,
  matched to the Python app.

### Noise reduction
- Temporal NR via Apple `VTTemporalNoiseFilter`, ported **in-process** (objc2
  FFI, no Swift helper), fully 10-bit, GPU-resident zero-copy for OSV and
  `.360`. Export-only; macOS-only (auto-hidden where unsupported).

### Output & delivery
- Half-equirect VR180 SBS, or a normalized equidistant fisheye SBS matched
  to the lens — **195°** for the OSMO VR180 Mod, **185°** for the
  GoPro Max 2 VR180 Mod.
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
