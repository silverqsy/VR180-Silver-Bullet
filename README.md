# VR180 Silver Bullet

A native, GPU-first VR180 processor written in Rust, built for the VR180
camera mods. It reads the camera's own dual-fisheye recording, dewarps each
lens with the exact factory calibration stored in the file, stabilizes from
the camera's gyro, and exports stereoscopic VR180 — or a reframed Flat 3D
view for AR glasses — with a real-time preview of the full pipeline the
whole way.

One self-contained binary per platform, **macOS (Apple Silicon)** and
**Windows (NVIDIA)**. No Python, no bundled runtimes, no system `ffmpeg`.
**Free and open source** under the MIT license.

**[Download the latest release][releases]** (2.5.1). Installed copies from
2.1 onward update themselves.

## Supported cameras

| Camera | File | Read from the file |
|---|---|---|
| **[OSMO VR180 Mod][osmo]** — DJI Osmo 360 and Osmo 360 II | `.osv` | Per-lens factory calibration, IMU, rolling-shutter timing |
| **Insta360 X6 VR180 Mod** | `.insv` | Factory lens model, ~1 kHz gyro, per-sensor exposure timing |
| **[GoPro Max 2 VR180 Mod][gopro]** | `.360` | EAC dual-fisheye, gyro, firmware rolling-shutter auto-detect |

Each camera's official log-to-Rec.709 LUT (DJI D-Log M, Insta360 I-Log,
GoPro GP-Log) is bundled and applied automatically on load. Dewarp output
matches the vendor's own software (DJI Studio, Insta360 Studio).

## Features

- **Two output modes.** **VR180 SBS** (half-equirect, or a fisheye
  projection matched to the lens) with Vision Pro (APMP) or YouTube
  metadata — or **Reframed (Flat 3D)**, a rectilinear side-by-side view for
  AR glasses and 3D displays with zoom, pan / tilt / roll and a Defish
  blend. Drag the preview to pan, scroll to zoom; export 2:1 or 32:9 up to
  7680×2160.
- **Real-time preview** of the exact 10-bit stack the export uses. SBS,
  anaglyph, 50% overlay and single-eye views, a native-resolution
  magnifier, audio playback.
- **Stabilization** from the camera's gyro, with timing derived from the
  file: velocity-dampened soft-stab with a Response slider, a Camera lock
  toggle, and per-scanline rolling-shutter correction.
- **Stereo alignment.** Auto align fits the rig's pitch, roll and yaw from
  the footage itself; per-eye view adjustment and stereo offset for manual
  work; Matching Eyes white-balance trim.
- **Color**, 10-bit end to end: CDL, 3D LUT, white balance, saturation,
  sharpen, mid-detail. Temporal noise reduction on macOS (export only).
- **Export**: hardware H.265 or ProRes 4:2:2, up to 8K (8192×4096), on a
  zero-copy GPU path (VideoToolbox on macOS, NVDEC → CUDA → NVENC on
  Windows). Audio is muxed inline — stereo, ambisonic, or APAC spatial on
  macOS. Multi-segment recordings join frame-exact. Optional BeyondVR hack
  for headset clarity.
- **Batch**: load many clips, tune each (or apply one setup to all), and
  export from one queue with progress, ETA and a completion notification.
- **English / 简体中文**, live toggle.

## Quick start

1. **Load** — drop `.osv`, `.insv` or `.360` files onto the window. Several
   files make a batch.
2. **Adjust** — press play and scrub; every control applies live. Click
   **Auto align** first (View adjustment), then grade, stabilize and trim
   (`I` / `O`). With a numeric field selected, **↑ / ↓** steps it precisely.
3. **Export** — pick the output folder and **Format** (VR180 SBS or
   Reframed, resolution, codec, metadata, audio) in the bottom bar, then
   **Export selected** or **Export all**.

## Build from source

```sh
# macOS (Apple Silicon)
brew install ffmpeg pkg-config
cargo build --release -p vr180-gui
./target/release/vr180-gui
```

```pwsh
# Windows — FFmpeg 8.1+ dev libs required; see docs/WINDOWS_BUILD.md
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:FFMPEG_DIR    = "C:\path\to\ffmpeg-8.x-dev"
cargo build --release -p vr180-gui
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"; .\target\release\vr180-gui.exe
```

Build `-p vr180-gui`, not the whole workspace. The first build takes a few
minutes (ffmpeg bindgen); incrementals are seconds. Video I/O is in-process
libav via `ffmpeg-next`; GPU work is `wgpu` (Metal / DX12 / Vulkan) with
WGSL shaders. Release packaging — signed and notarized macOS bundle,
Windows installer — is described in [docs/BUILD.md](docs/BUILD.md).

## Docs

- [CHANGELOG.md](CHANGELOG.md) — what changed in each release
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate layout and GPU
  pipeline shape
- [docs/BUILD.md](docs/BUILD.md) and
  [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — toolchains and packaging
- [docs/AUTO-UPDATE.md](docs/AUTO-UPDATE.md) — how the updater works
- [CLAUDE.md](CLAUDE.md) — current status and the load-bearing engineering
  decisions (start here if you're working on the code)

## License

MIT.

[releases]: https://github.com/silverqsy/VR180-Silver-Bullet/releases/latest
[osmo]: https://www.facebook.com/share/p/1J1WBwKhfy/
[gopro]: https://www.facebook.com/share/p/1QUDsLvWS8/
