# CLAUDE.md — orientation for Claude Code sessions

Auto-loaded at session start. Read this first; deeper detail is in the
`docs/` pointers at the bottom.

## ⚡ Status: Windows build works — now a macOS verification handoff

The Windows native build of `vr180-gui` is **up and running**: it builds,
plays OSV, and exports H.265 via **NVENC, GPU-resident** (NVDEC→wgpu→CUDA,
no CPU readback — see "Windows GPU-resident export" below). A round of
Windows-developed fixes touched **cross-platform display/color/decode code**.
The macOS machine should rebuild and **verify these on Metal** (they were
developed against Vulkan and couldn't be checked on macOS):

1. **Preview gamma — most important.** The egui display-view gamma was
   corrected for **egui-wgpu 0.34** (see Lesson #2): the preview SBS is now
   handed to egui with a **plain `Rgba8Unorm` view** on every platform, not
   an sRGB view. **Verify the macOS preview matches the export** (not too
   dark, not washed). This very likely *fixes* a latent too-dark preview on
   macOS that the 0.28→0.34 egui upgrade introduced — the old sRGB-view
   trick was only correct under 0.28. Do **not** revert it to an sRGB view.
2. **Export color range.** Both export paths now emit **video-range
   (limited / `AVCOL_RANGE_MPEG`) Rec.709** (was full-range on the Windows
   GPU-resident path — a band-aid for the gamma bug, now removed). Verify
   the macOS VideoToolbox/ProRes export still looks right and matches the
   (now gamma-correct) preview.
3. **Fisheye output mode** now previews **and** exports as a fisheye SBS
   (was half-equirect-only in the zero-copy preview + GPU-resident export).
   Verify on macOS.

Build/run quick-ref (build `-p vr180-gui`, **not** the workspace):
```pwsh
# Windows
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"   # LLVM 17+
$env:FFMPEG_DIR    = "C:\path\to\ffmpeg-7.x-dev"   # avbuild dev distribution
cargo build --release -p vr180-gui
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"; .\target\release\vr180-gui.exe
```
```sh
# macOS
cargo build --release -p vr180-gui && ./target/release/vr180-gui
```
Full Windows step-by-step (toolchain, FFmpeg, gotchas):
[docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md).

> ⚠️ The Windows-only additions (`nvenc_cuda`, the `cudarc` dep, D3D11→CUDA
> interop) are `#[cfg(target_os = "windows")]` / `[target.'cfg(windows)']`
> gated — they don't build on macOS. The secondary `vr180-render` CLI still
> has a pre-existing non-exhaustive-match error; ignore it (build the GUI).
> The verification examples under `crates/vr180-pipeline/examples/` are
> gitignored local scaffolding (several are Windows/CUDA-only).

## What this is

**VR180 Silver Bullet 2.0** — a clean-room Rust rewrite of a VR180
camera-mod processor. Single self-contained native binary per platform,
no Python. The reference implementation is the Python/PyQt6 app at
`../vr180_processor/vr180_gui.py` (the `main` branch / sibling worktree)
— **we read it to port algorithms, we don't modify it.**

- **Shipped:** a native desktop GUI, `vr180-gui` (**eframe 0.34 + egui +
  egui-wgpu + wgpu 29** — note: NOT 0.28/0.20; that stale version line hid a
  display-gamma bug, see Lesson #2). This is the product.
- **Primary source format:** DJI Osmo 360 `.osv` (dual-stream fisheye).
  Also handles SBS fisheye, Blackmagic `.braw`, and GoPro Max `.360`
  (EAC) as a legacy path.
- **GPU:** `wgpu` only (Metal on macOS, DX12/Vulkan on Windows). Shaders
  are WGSL in `crates/vr180-pipeline/src/shaders/`.
- **Video I/O:** in-process libav via `ffmpeg-next 8.1` (no shelling out
  to a system `ffmpeg`).

## Build & run

| | macOS | Windows |
|---|---|---|
| Build the GUI | `cargo build --release -p vr180-gui` | same (see WINDOWS_BUILD.md for env) |
| Run | `./target/release/vr180-gui` | `.\target\release\vr180-gui.exe` (+ ffmpeg DLLs on PATH) |
| FFmpeg | `brew install ffmpeg pkg-config` (pkg-config auto-finds it) | `FFMPEG_DIR` → avbuild dev dist + `LIBCLANG_PATH` |

First build ~3-5 min (ffmpeg-sys bindgen); incremental builds are seconds.

To launch + verify after a change, the dev loop on macOS is:
`pkill -9 -f target/release/vr180-gui; cargo build --release -p vr180-gui && ./target/release/vr180-gui &`

## Cross-platform stance — the #1 rule

Code must compile and run on **both macOS and Windows**. Two platform
paths exist on purpose, behind `#[cfg(target_os = "...")]`:

- **macOS only:** `interop_macos` (IOSurface↔Metal↔wgpu zero-copy),
  `ZeroCopyDualStreamFisheyeIter` (VT P010 zero-copy decode), the P010
  IOSurface export path, VideoToolbox/ProRes-VT encode. The crates
  `metal` / `core-foundation` / `objc` / `foreign-types` are
  `[target.'cfg(target_os = "macos")'.dependencies]` — they never build
  on Windows.
- **Cross-platform + Windows fast paths:** `d3d11va` (NVDEC) decode —
  `DualStreamFisheyeIter` for stills, `D3d11SharedDualStreamIter` for the
  **zero-copy live preview** (D3D11 P010→RGBA16→Vulkan, no CPU readback);
  `hevc_nvenc` (H.265, the **Windows default**, GPU-resident — see below)
  with a `libx265` software fallback; `prores_ks` (ProRes); the whole wgpu
  compute pipeline; GUI; audio (cpal→WASAPI); file picker (rfd). Encoder
  backend auto-selected per-OS in `app.rs::commit_export`. Windows now has
  its OWN zero-copy fast paths — it's no longer "macOS fast, Windows slow."

Rules: every cfg gate matters; `pub use` from a macOS module must itself
be gated (see `vr180-pipeline/src/lib.rs`); use `Path`/`PathBuf` (never
slash string literals); `include_str!`'d shaders + `assets/` are portable.

## Architecture (where things live)

- `crates/vr180-core` — 100% portable: gyro/quaternion math, EAC dims,
  `.cube` LUT parse, GEOC calib, segments.
- `crates/vr180-pipeline` — the engine: `decode.rs`, `fisheye_decode.rs`,
  `gpu.rs` (all wgpu kernels + the color stack), `fisheye_export.rs`
  (export pipeline + the zero-copy P010 fast path), `encode.rs` (encoder
  backends), `dji_imu.rs` (OSV stabilization + rolling-shutter),
  `interop_macos.rs` (macOS-only), `panomap.rs`, `shaders/*.wgsl`.
- `crates/vr180-gui` — the app: `app.rs` (egui UI + the full-res/zoom
  "DetailCache" + export wiring), `decoder.rs` (`Settings`, the live
  decode worker, `DetailCache`, color-stack plan, persistence),
  `audio_player.rs`, `assets/` (bundled LUT).
- `crates/vr180-fisheye` — camera presets + fisheye calib (Kannala-Brandt).
- `crates/vr180-render` — legacy CLI (currently doesn't compile; ignore).

## Color pipeline — match the Python app

The color tools (CDL, 3D LUT, white balance, saturation, sharpen,
mid-detail) were just aligned to the Python app **exactly**. Order
(`gpu.rs::record_color_stack` / `apply_color_stack_per_eye_16`):

> **CDL → 3D LUT → sharpen → temp/tint → mid-detail → saturation**

- temp/tint is **post-LUT** (critical for log footage: it must act on the
  LUT's Rec.709 output, not the log input). White-balance and saturation
  are split passes of the one `color_grade` shader.
- CDL clips to `[0,1]` **before** gamma (matches `build_color_1d_lut`).
- Math constants match Python: temp/tint = `R×(1+0.3t)`,`G×(1−0.3·tint)`,
  `B×(1−0.3t)`; saturation = `lerp(BT.601-luma, color, sat)`;
  shadow/highlight = smoothstep masks ×0.6 about pivot 0.5.
- The DJI Osmo 360 **D-LogM→Rec.709 LUT is bundled** (`include_str!`,
  `crates/vr180-gui/assets/`), autoloaded for OSV, cached per-thread.

## 10-bit end-to-end (when 10-bit output is selected) — non-negotiable

Every stage must hold ≥10-bit precision. NOTE: intermediates are
**`Rgba16Unorm`** (not `Rgba16Float` — the old CLAUDE.md was wrong).
- macOS: VT P010 IOSurface decode → 16-bit project → 16-bit color stack →
  P010 IOSurface encode (Main10), zero-copy.
- Windows: NVDEC P010 decode → D3D11 P010→RGBA16 → 16-bit project → 16-bit
  color stack → P010 compose → **GPU-resident CUDA NVENC** (`--profile
  main10`), zero-copy (see "Windows GPU-resident export"). Fallback:
  P010/RGB48 readback → `hevc_nvenc`/`libx265`.
- The **preview** runs the same 16-bit color stack as the export, then
  composes to an `Rgba8Unorm` SBS for egui. Don't add an 8-bit color
  shortcut into the graded path.

## Windows GPU-resident H.265 export (the fast path)

On Windows, 10-bit H.265 export runs **fully GPU-resident** — no CPU
readback — at the NVENC ceiling (~36 fps @ 8K on a 4090):

> NVDEC P010 decode → D3D11 P010→RGBA16 → wgpu 16-bit project/color/compose
> → P010 plane textures → **exportable Vulkan images → CUDA external memory
> → `cuMemcpy2D` (DtoD) into NVENC's CUDA frame** → `hevc_nvenc`.

Lives in `nvenc_cuda.rs` (`#[cfg(target_os = "windows")]`), dispatched from
`fisheye_export.rs`. `cudarc` is a `[target.'cfg(windows)']` dep. The export
uses a **dedicated** wgpu device (`gpu.rs::new_dedicated_from_adapter`) so it
can't deadlock eframe's renderer (Lesson #1). Falls back to a P010/RGB48
readback `hevc_nvenc`/`libx265` path. Both `HalfEquirect` and `Fisheye`
output modes take the fast path.

**Color range:** both the GPU-resident and readback paths compose
**video-range** YCbCr and tag `AVCOL_RANGE_MPEG` (Rec.709) — the
distribution standard (YouTube VR180, headsets), matching the source and the
gamma-correct preview. `compose_sbs_to_p010_textures(…, full_range: bool)`
takes a flag but **both callers pass `false`**; the brief full-range
experiment was a band-aid for the preview-gamma bug (Lesson #2) and is gone.

## Hard-won lessons — DON'T regress these

1. **Never do wgpu GPU work on a non-main thread.** The wgpu
   `Device`/`Queue`/swapchain is shared with eframe's renderer; GPU
   submits + `Maintain::Wait` from a second thread wedge Metal's drawable
   queue against the presenting main thread → permanent 0% CPU deadlock.
   `DetailCache` decodes on a background thread (CPU/VideoToolbox only —
   it returns `Vec<u8>` pairs, never touches wgpu) and does the GPU
   project/compose on the **main** thread. Keep it that way.
2. **Preview gamma (egui-wgpu 0.34 — corrected):** egui-wgpu's fragment
   shader expects display textures that are **NOT sRGB-aware** — it samples
   them as gamma/raw bytes (and on an sRGB swapchain linearizes at the very
   end so the hardware re-encode is identity). Our SBS already holds
   Rec.709-gamma bytes, so register it with a **plain `Rgba8Unorm` view** on
   every platform (`preview_view_format` in `app.rs`; used at both
   `register_native_texture` calls). An **sRGB view double-decodes** → the
   preview shows `sRGB⁻¹(v)`, too dark. Windows' non-sRGB `Bgra8Unorm`
   swapchain made it obvious; on macOS's sRGB swapchain it's wrong via a
   different path. The old "register with an sRGB view" advice was correct
   only under **egui-wgpu 0.28** and is now stale — **do not restore it.**
3. **Detail still = exact frame, decode-forward skip:** the native decoder
   seeks to a keyframe and **decodes forward** to the requested pts (else the
   still shows a keyframe up to a GOP away). The OSV dual-stream path uses
   `DualStreamFisheyeIter::decode_forward_to`: it decodes the intermediate
   GOP frames on the HW engine but runs the HW→CPU download + swscale only on
   the **target** frame (≈11 s → ≈0.6 s for a deep seek at 8K). Same
   decode-forward in the live worker's seek.
4. **OSV gyro timing:** per-frame stab samples the IMU 8.5 ms after
   frame-start (fps-independent); rolling-shutter uses a 19 ms (30 fps) /
   16.23 ms (50 fps) readout window centered on it. Verified vs DJI
   Studio. See `dji_imu.rs`.
5. **Settings persistence** lives in a per-OS config dir (macOS App
   Support / Windows `%APPDATA%` / Linux XDG) — `Settings::config_path`.
   It survives both relaunch and loading a new clip (only trim resets).

## Porting algorithms from Python

1. Read `../vr180_processor/vr180_gui.py` (the working reference).
2. Port the math exactly; bit-validate against it when feasible. The
   camera-specific constants are already validated against real footage —
   don't re-derive.
3. NOTE for the Windows session: the deep project memory the macOS dev
   used lives in that machine's `~/.claude/...` and is **not on this PC**.
   The essential facts are summarized here and in `docs/`; if you need
   more on GEOC calibration / GoPro stream conventions / VQF, read the
   Python source directly.

## Don't

- Don't run wgpu/GPU work off the main thread (see lesson #1).
- `wgpu` is the GPU answer for compute/render. The ONE sanctioned exception
  is the Windows NVENC export, which uses `cudarc` purely to feed NVENC
  (Windows-gated); don't add CUDA/OpenCL/Numba anywhere else.
- Don't shell out to a system `ffmpeg` — use `ffmpeg-next` in-process.
- Don't remove a `#[cfg(target_os = "...")]` gate without confirming the
  API exists on the other platform; the Mac-only paths are FAST paths
  with deliberate cross-platform fallbacks.
- Don't push to `main` (this is the `2.0` branch).
- Don't drop the graded path to 8-bit; don't re-introduce the off-thread GPU
  deadlock; don't switch the preview back to an sRGB texture view (Lesson #2)
  or the H.265 export back to full-range.

## Doc pointers

- [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — **the Windows handoff guide (start here on Windows).**
- [docs/BUILD.md](docs/BUILD.md) — original cross-platform build notes.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate boundaries, GPU pipeline shape.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan (note: predates the GUI; treat as historical).
