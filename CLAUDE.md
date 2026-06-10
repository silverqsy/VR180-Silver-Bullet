# CLAUDE.md — orientation for Claude Code sessions

Auto-loaded at session start. Read this first; deeper detail is in the
`docs/` pointers at the bottom.

## ⚡ Status: macOS feature batch done — Windows build + verify handoff

(The previous handoff — Windows GPU-resident NVENC export + the egui-0.34
gamma fix — is **verified on macOS** and landed.) Since then a large batch
of features was developed and verified **on macOS**; the Windows session
should pull `2.0`, build, and verify them on DX12/Vulkan + NVENC. All of it
is cross-platform code (shared Rust + WGSL — the RGBA16 Windows variants are
format-patched from the same WGSL sources at pipeline creation), so expect
it to Just Work; the checklist is what to *confirm*, not to port:

1. **DJI lens model is now exact** (reverse-engineered from DJI Studio —
   verified against its export). Auto mode loads the per-lens FACTORY
   calibration from the OSV protobuf: fx/fy, raw cx/cy (top-left, y-down,
   NO flip), k1–k4, **k5 (field 15)**, and **Brown-Conrady tangential
   p1/p2 (field 20)**. All KB shaders run the 5-coeff radial + tangential
   (zeros ⟹ byte-identical to the old 4-coeff path, so non-OSV sources are
   unchanged). Verify: load an OSV → dewarp matches DJI Studio.
2. **Fisheye SBS output is now a normalized 195° equidistant fisheye**
   (`FISHEYE_OUT_FULL_FOV_DEG` in `fisheye_export.rs`) — the projection is
   canonical (source-lens distortion removed), disk edge = 97.5° off-axis.
   Both preview + export, incl. the GPU-resident NVENC path. Was: the
   source lens's own projection.
3. **Soft-stab rewritten** — velocity-dampened Gyroflow-style smoothing
   ported from the Python app (`dji_imu.rs::smooth_quats_velocity_dampened`):
   velocity-adaptive τ with ±200 ms look-ahead, symmetric fwd/bwd passes,
   SOFT elastic max-corr limit (no snap). New **Response** slider
   (0.2–3.0). Camera-lock (`smooth_ms = 0`) keeps the hard clamp. Defaults:
   smooth 1200 ms, max corr 15°.
4. **Eye orientation**: "Swap L↔R" now actually applies live (toggling
   auto-reloads the clip — the dual-stream iterators bind eye order at
   open), and the per-eye factory calib **follows the stream** (was a bug:
   swapped view used the wrong lens's calib). New **"Upside-down mount
   (180°)"** checkbox = exact `roll+180°` in `ViewAdjust` + implicit eye
   swap (`Settings::effective_swap_eyes()`).
5. **8K export option** (export dialog → Resolution → 8192×4096): renders
   the PROJECTION at 4096²/eye (single resample from the native source —
   not a last-step upscale). Verify NVENC accepts 8192-wide @ Main10 on
   the target GPU (4090: OK).
6. **UI/perf**: View-adjustment panel above Stabilization (default-open);
   "KB parameters" collapsible (k1–k5 + p1/p2 sliders, OSV-gated); Override
   auto-reseeds from each new clip's in-file calib; camera-preset/Gyroflow
   pickers hidden for OSV; sidebar slider tracks fixed at 150 px (do NOT
   size sliders from `available_width()` — feedback loop, panel swallows
   the preview; panel capped at 420 px); zoomed paused-frame drag re-render
   throttled to 120 ms (was per pointer event on the UI thread).

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

## DJI OSV lens model (exact — don't simplify)

Reverse-engineered from DJI Studio's binary and bit-matched at runtime.
Per-lens calib block in the OSV protobuf (`vr180-fisheye/src/dji_osv.rs`):
`1=fx 2=fy 3=cx 4=cy 5..8=k1..k4 10=W 11=H 15=k5 20=[p1,p2]
21=mount_quat`. Projection (in every KB WGSL shader's `project_kb`):

```
θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹ + k5·θ¹¹      # 5-coeff Kannala-Brandt
u' = u + 2·p1·u·v + p2·(r²+2u²)                        # Brown-Conrady tangential
v' = v + p1·(r²+2v²) + 2·p2·u·v                        #   (r² = θ_d²)
src = (cx + fx·u',  cy − fy·v')                        # raw cx/cy, NO flip
```

k5 keeps the radial map monotonic past ~90° (the 4-coeff poly folds at
~87–90°); the tangential is small (~2–4 px at the rim) but it's what
finally matched DJI. Both resolvers (`decoder.rs::resolve_fisheye_calib_pair`
and `fisheye_export.rs::resolve_calib_pair`) must stay in lockstep — every
calib change goes in BOTH or preview ≠ export.

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
4. **OSV gyro timing:** measured SROT (sensor readout) is **18.301 ms @
   30 fps / 16.228 ms @ 50 fps** (`dji_osmo_readout_ms_for_fps`). The
   per-frame stab IMU sample point defaults to **SROT/2 after
   frame-start** (9.15 / 8.11 ms) and is exposed as the live "IMU phase
   (ms)" slider — re-seeded to SROT/2 on every clip load, NOT persisted
   (`#[serde(skip)]`). The rolling-shutter readout window is centered on
   the same point (coupled, as DJI does). See `dji_imu.rs`.
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
