# VR180 Silver Bullet 2.0

A native, GPU-first VR180 processor written in Rust. One self-contained
binary per platform — no Python, no bundled runtimes, and it never shells
out to a system `ffmpeg`. This is the `2.0` clean-room rewrite of the
Python/PyQt6 [VR180 Silver Bullet](../vr180_processor/) app, running
natively on **macOS (Apple Silicon)** and **Windows (NVIDIA)**.

Load a clip → preview with live controls → export VR180 SBS.

## Supported cameras & formats

| Source | Notes |
|---|---|
| **DJI Osmo 360** — `.osv` | Primary path. Dual-stream fisheye, exact factory lens dewarp from the file. |
| **GoPro Max** — `.360` | EAC dual-fisheye. Zero-copy GPU decode, firmware-RS auto-detect, noise reduction. |
| **SBS fisheye** — `.mp4` / `.mov` | Insta360, Vuze XR, QooCam, Canon RF dual-fisheye, generic dual-camera rigs. |
| **Blackmagic RAW** — `.braw` | Pyxis 12K, URSA Cine Immersive (VQF 6D stabilization). |

## Features

- **Exact lens dewarp** — for the DJI Osmo 360, loads the per-lens factory
  calibration straight from the `.osv` (fx/fy, principal point, 5-coefficient
  Kannala-Brandt radial + Brown-Conrady tangential — the full model DJI
  Studio uses, reverse-engineered and bit-matched). Per-eye manual override
  with file-seeded sliders when you want to hand-tune.
- **Stabilization from the camera IMU** — camera-lock or velocity-dampened
  soft-stab (Gyroflow-style adaptive smoothing that relaxes ahead of fast
  motion, soft elastic correction limit, **Response** slider), plus
  per-scanline rolling-shutter correction from measured sensor-readout
  timing. For GoPro `.360`, **firmware rolling-shutter is auto-detected**
  from the CORI stream (firmware vs no-firmware), with a manual override.
- **Color pipeline, 10-bit end-to-end** — CDL, 3D LUT (DJI D-LogM→Rec.709
  bundled and autoloaded), white balance, saturation, sharpen, mid-detail;
  the identical stack runs in preview and export.
- **Noise reduction** — temporal NR via Apple VideoToolbox
  (`VTTemporalNoiseFilter`), run **in-process** (objc2 FFI, no helper
  binary), fully 10-bit and GPU-resident zero-copy. Export-only; macOS-only
  (auto-hidden on Windows / unsupported hardware).
- **Output projections** — half-equirect VR180 SBS (the standard), or a
  normalized **195° equidistant fisheye** SBS.
- **Export** — H.265 (VideoToolbox on macOS; GPU-resident NVDEC→wgpu→CUDA→
  NVENC on Windows, with a libx265 software fallback) or ProRes, at native
  resolution or **8192×4096 (8K)**, with **Vision Pro (APMP)** or **YouTube
  VR180** metadata injection, **APAC spatial** / ambisonic / stereo audio,
  and OSV stereo-audio passthrough.
- **Batch + unified export** — load many clips, tune each independently,
  then export all (or a checked subset) through one queue with a persistent
  progress + ETA bar and a completion notification.
- **Preview** — SBS / anaglyph / 50%-overlay / single-eye view modes, zoom
  magnifier with a native-resolution still, per-eye view adjustment (pano +
  stereo offset), upside-down-mount support, audio playback.
- **Localized UI** — English / 简体中文, live toggle.

## Workspace layout

```
crates/
├── vr180-core/      # pure Rust: gyro/quat math, EAC dims, .cube LUT
│                    # parse, GEOC calib — fully portable
├── vr180-fisheye/   # camera presets + fisheye calibration (KB model,
│                    # OSV protobuf parse, Gyroflow profile import)
├── vr180-pipeline/  # the engine: decode (ffmpeg-next 8.1, in-process),
│                    # wgpu compute kernels (WGSL), DJI IMU stab + RS,
│                    # in-process VT noise reduction, export pipeline,
│                    # encoders, mp4 atom injection
├── vr180-gui/       # the product: eframe/egui app (vr180-gui binary)
└── vr180-render/    # legacy CLI (currently not building; ignore)

helpers/swift/       # macOS APAC spatial-audio helper (spawned only when
                     # exporting Vision Pro spatial audio). Noise reduction
                     # and decode/encode are all in-process now.
docs/                # build + architecture + Windows notes
installer/           # Windows Inno Setup script
```

GPU work is `wgpu` everywhere (Metal / DX12 / Vulkan picked at runtime);
shaders are WGSL. Video I/O is in-process libav via `ffmpeg-next` — the
app never shells out to a system `ffmpeg`.

## Build & run

```sh
# macOS (Apple Silicon)
brew install ffmpeg pkg-config
cargo build --release -p vr180-gui
./target/release/vr180-gui
```

```pwsh
# Windows — see docs/WINDOWS_BUILD.md for the full setup
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:FFMPEG_DIR    = "C:\path\to\ffmpeg-7.x-dev"
cargo build --release -p vr180-gui
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"; .\target\release\vr180-gui.exe
```

Build `-p vr180-gui` (not the whole workspace — `vr180-render` is a legacy
CLI that intentionally isn't kept building). First build takes a few minutes
(ffmpeg-sys bindgen); incrementals are seconds.

A signed/notarized macOS app bundle and a Windows installer are produced for
releases — see [docs/BUILD.md](docs/BUILD.md) and
[installer/windows.iss](installer/windows.iss).

## Docs

- [CLAUDE.md](CLAUDE.md) — current status + the load-bearing engineering
  decisions (start here if you're working on the code)
- [CHANGELOG.md](CHANGELOG.md) — what's in 2.0.0
- [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — Windows toolchain setup
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate boundaries, GPU
  pipeline shape
- [docs/BUILD.md](docs/BUILD.md) — FFmpeg / `FFMPEG_DIR` details, packaging
- [docs/ROADMAP.md](docs/ROADMAP.md) — historical phased build log

## License

MIT.
