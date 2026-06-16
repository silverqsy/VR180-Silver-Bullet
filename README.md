# VR180 Silver Bullet 2.0

A native, GPU-first VR180 processor written in Rust. **The headline of 2.0
is full support for the DJI Osmo 360 VR180 mod:** it reads the camera's
`.osv` dual-fisheye recordings, dewarps each lens with its exact factory
calibration, applies precise IMU stabilization and per-scanline
rolling-shutter correction, and exports stereoscopic VR180 SBS — with output
quality on par with DJI Studio and a real-time preview the whole way. (GoPro
Max `.360` is fully supported too.)

One self-contained binary per platform — no Python, no bundled runtimes, and
it never shells out to a system `ffmpeg` — running natively on **macOS (Apple
Silicon)** and **Windows (NVIDIA)**. This is the ground-up rewrite of the
Python/PyQt6 [VR180 Silver Bullet](../vr180_processor/) (1.1) app.

Load a clip → preview with live controls → export VR180 SBS.

## Supported cameras & formats

| Source | Notes |
|---|---|
| **DJI Osmo 360** (VR180 mod) — `.osv` | **The headline of 2.0.** Dual-stream fisheye, exact factory lens dewarp loaded from the file. |
| **GoPro Max** — `.360` | EAC dual-fisheye. Zero-copy GPU decode, firmware-RS auto-detect, noise reduction. |

## Features

- **Real-time, WYSIWYG preview** — scrub and tune color *and* stabilization
  with the full pipeline applied live; the preview runs the exact 10-bit
  stack the export uses, so what you see is what you ship. SBS / anaglyph /
  50%-overlay / single-eye view modes, a zoom magnifier with a
  native-resolution still, per-eye view adjustment (pano + stereo offset),
  upside-down-mount support, and audio playback.
- **Fast, hardware-accelerated export** — an end-to-end zero-copy GPU
  pipeline, **up to 2× as fast as the previous version at 10-bit**. H.265
  (VideoToolbox zero-copy on macOS; GPU-resident NVDEC→wgpu→CUDA→NVENC on
  Windows, libx265 fallback) or ProRes, at native resolution or
  **8192×4096 (8K)**, with **Vision Pro (APMP)** / **YouTube VR180** metadata,
  **APAC spatial** / ambisonic / stereo audio, and OSV stereo-audio
  passthrough.
- **Batch processing** — load many clips, tune each independently, then
  export all (or a checked subset) from one queue with a persistent
  progress + ETA bar and a completion notification.
- **Exact lens dewarp (DJI Osmo 360 VR180 mod)** — loads the per-lens factory
  calibration straight from the `.osv` (fx/fy, principal point, 5-coefficient
  Kannala-Brandt radial + Brown-Conrady tangential), with dewarp output on
  par with DJI Studio. Per-eye manual override with file-seeded sliders.
- **IMU stabilization + rolling shutter** — camera-lock or velocity-dampened
  soft-stab (Gyroflow-style adaptive smoothing with a **Response** slider and
  a soft elastic correction limit), plus per-scanline rolling-shutter
  correction from measured sensor-readout timing. For GoPro `.360`, firmware
  rolling-shutter is **auto-detected** from the CORI stream (firmware vs
  no-firmware), with a manual override.
- **Color pipeline, 10-bit end-to-end** — CDL, 3D LUT (DJI D-LogM→Rec.709
  bundled and autoloaded), white balance, saturation, sharpen, mid-detail;
  the identical stack runs in preview and export.
- **Noise reduction** — temporal NR via Apple VideoToolbox
  (`VTTemporalNoiseFilter`), run in-process and fully 10-bit, GPU-resident
  zero-copy. Export-only; macOS-only (auto-hidden where unsupported).
- **Output projections** — half-equirect VR180 SBS, or a normalized
  equidistant fisheye SBS matched to the lens (**195°** for the Osmo 360,
  **185°** for the GoPro Max).
- **Localized UI** — English / 简体中文, live toggle.

## Using it

1. **Load** — drag a `.osv` (DJI Osmo 360) or `.360` (GoPro Max) onto the
   window, or click **Load video**. Drop several to build a batch.
2. **Preview & adjust** — press play and scrub; every control applies live.
   - **Align the stereo first** — scrub to a part of the clip with a
     **far-away subject**, switch the **view** to **50% overlay** or
     **anaglyph**, and adjust the **stereo offset** (in *View adjustment*)
     until that distant subject's two eyes line up. Do this before grading
     or stabilizing.

     > **Tip:** with a numeric field selected, press **↑ / ↓** to step its
     > value precisely.
   - **Color** — CDL, 3D LUT (the DJI D-LogM→Rec.709 LUT autoloads for OSV),
     white balance, saturation, sharpen.
   - **Stabilization** + **Rolling shutter** — camera-lock or velocity-
     dampened soft-stab (tune the **Response** slider); RS mode auto-detects
     for `.360`.
   - **Noise Reduction** — temporal NR (export-only; macOS).
   - Use the **SBS / single-eye** views and zoom in to check sharpness, then
     set **Mark In / Mark Out** (`I` / `O`) to trim.
3. **Export** — in the bottom bar, choose the output folder and **Format**
   (resolution incl. 8K, codec, bit depth, VR180 metadata target — Vision Pro
   APMP or YouTube — and audio), then **Export selected** or **Export all**.
   Progress + ETA show in the bar.

**Batch** — load several clips, tune each (or set one up and **Apply settings
to all** of the same camera type), tick the ones you want, and **Export all**.

**Language** — toggle **EN / 中文** in the top bar.

## Workspace layout

```
crates/
├── vr180-core/      # pure Rust: gyro/quat math, EAC dims, .cube LUT
│                    # parse, GEOC calib — fully portable
├── vr180-fisheye/   # fisheye lens calibration (Kannala-Brandt model,
│                    # DJI OSV protobuf parse)
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
