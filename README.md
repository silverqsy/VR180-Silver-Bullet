# VR180 Silver Bullet 2.0

A native, GPU-first VR180 processor for the **DJI Osmo 360** (`.osv`
dual-fisheye), written in Rust. One self-contained binary per platform —
no Python, no bundled runtimes. This is the `2.0` clean-room rewrite of
the Python/PyQt6 [VR180 Silver Bullet](../vr180_processor/) app.

## What it does

Load a `.osv` → preview with live controls → export VR180 SBS.

- **Exact DJI lens dewarp** — loads the per-lens factory calibration from
  the OSV itself (fx/fy, principal point, 5-coefficient Kannala-Brandt
  radial + Brown-Conrady tangential — the full model DJI Studio uses,
  reverse-engineered and bit-matched). Per-eye manual override with
  file-seeded sliders when you want to hand-tune.
- **Stabilization from the camera IMU** — camera-lock or velocity-dampened
  soft-stab (Gyroflow-style: adaptive smoothing that relaxes ahead of fast
  motion, soft elastic correction limit), plus per-scanline rolling-shutter
  correction using measured sensor-readout timing (18.301 ms @ 30 fps,
  16.228 ms @ 50 fps).
- **Color pipeline, 10-bit end-to-end** — CDL, 3D LUT (DJI D-LogM→Rec.709
  bundled, autoloaded), white balance, saturation, sharpen, mid-detail;
  identical stack in preview and export.
- **Output projections** — half-equirect VR180 SBS (the standard), or a
  normalized **195° equidistant fisheye** SBS.
- **Export** — H.265 (VideoToolbox on macOS; GPU-resident NVDEC→wgpu→CUDA→
  NVENC on Windows, libx265 fallback) or ProRes, at native resolution or
  8192×4096, with Vision Pro (APMP) or YouTube VR180 metadata injection
  and OSV stereo-audio passthrough.
- **Preview niceties** — SBS / anaglyph / 50%-overlay / single-eye view
  modes, zoom magnifier with native-res still, per-eye view adjustment
  (pano + stereo offset), upside-down-mount support, audio.

Also reads SBS fisheye, Blackmagic `.braw` (VQF 6D stab), and GoPro Max
`.360` (EAC) as a legacy path.

## Workspace layout

```
crates/
├── vr180-core/      # pure Rust: gyro/quat math, EAC dims, .cube LUT
│                    # parse, GEOC calib — fully portable
├── vr180-fisheye/   # camera presets + fisheye calibration (KB model,
│                    # OSV protobuf parse, Gyroflow profile import)
├── vr180-pipeline/  # the engine: decode (ffmpeg-next 8.1, in-process),
│                    # wgpu compute kernels (WGSL), DJI IMU stab + RS,
│                    # export pipeline, encoders, mp4 atom injection
├── vr180-gui/       # the product: eframe/egui app (vr180-gui binary)
└── vr180-render/    # legacy CLI (currently not building; ignore)

helpers/swift/       # macOS AVFoundation helpers (MV-HEVC spatial video,
                     # APAC spatial audio, VT denoise) — spawned as
                     # external processes when used
docs/                # build + architecture + Windows handoff docs
```

GPU work is `wgpu` everywhere (Metal / DX12 / Vulkan picked at runtime);
shaders are WGSL. Video I/O is in-process libav via `ffmpeg-next` — the
app never shells out to a system `ffmpeg`.

## Build & run

```sh
# macOS
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

Build `-p vr180-gui` (not the whole workspace). First build takes a few
minutes (ffmpeg-sys bindgen); incrementals are seconds.

## Docs

- [CLAUDE.md](CLAUDE.md) — current status + the load-bearing engineering
  decisions (start here)
- [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — Windows toolchain setup
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate boundaries, GPU
  pipeline shape
- [docs/BUILD.md](docs/BUILD.md) — FFmpeg / `FFMPEG_DIR` details
- [docs/ROADMAP.md](docs/ROADMAP.md) — historical phased plan

## License

MIT.
