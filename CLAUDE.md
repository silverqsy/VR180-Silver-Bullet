# CLAUDE.md — orientation for Claude Code sessions

Auto-loaded at session start. Read this first; deeper detail is in the
`docs/` pointers at the bottom.

## ⚡ If you're on the Windows PC: your job is the Windows build

The immediate task for this handoff is to **build and run the GUI app
(`vr180-gui`) natively on Windows**, then fix anything platform-specific
that the macOS-only dev hasn't been able to test.

**Start here: [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md)** — it's the
full, current step-by-step (toolchain, FFmpeg, build, run, gotchas).

Quick version:
```pwsh
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"   # LLVM 17+
$env:FFMPEG_DIR    = "C:\path\to\ffmpeg-7.x-dev"   # avbuild dev distribution
cargo build --release -p vr180-gui                 # build ONLY the GUI (see below)
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"
.\target\release\vr180-gui.exe
```

Most likely Windows issue: a recently-added line references a macOS-only
API without a `#[cfg(target_os = "macos")]` gate. The error looks like
`unresolved import crate::interop_macos` or `cannot find function
create_p010_encode_buffer`. Fix = gate that block (with a
`#[cfg(not(target_os = "macos"))]` fallback or `unreachable!()` where a
value is needed — see `fisheye_export.rs` `use_zero_copy_*` for the
pattern). The big macOS modules are already gated; expect only small gaps.

> ⚠️ Build `-p vr180-gui`, **not** the whole workspace. The secondary
> `vr180-render` CLI has a pre-existing non-exhaustive-match error
> (`EncoderBackend::ProResVideoToolbox`/`ProResKs` arms missing in
> `crates/vr180-render/src/main.rs`). It's unrelated to the GUI; fix it
> only if you actually need the CLI.

## What this is

**VR180 Silver Bullet Neo** — a clean-room Rust rewrite of a VR180
camera-mod processor. Single self-contained native binary per platform,
no Python. The reference implementation is the Python/PyQt6 app at
`../vr180_processor/vr180_gui.py` (the `main` branch / sibling worktree)
— **we read it to port algorithms, we don't modify it.**

- **Shipped:** a native desktop GUI, `vr180-gui` (eframe 0.28 + egui +
  egui-wgpu + wgpu 0.20). This is the product. (The old CLAUDE.md said
  "don't build the GUI yet / Tauri later" — obsolete; the GUI exists and
  is the focus.)
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
- **Cross-platform (the Windows path):** ffmpeg software/`d3d11va` decode
  (`DualStreamFisheyeIter`), `libx265` (H.265) + `prores_ks` (ProRes)
  software encode, the whole wgpu compute pipeline, the GUI, audio
  (cpal→WASAPI), file picker (rfd). The encoder backend is auto-selected
  per-OS in `app.rs::commit_export`.

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
- Windows: ffmpeg 16-bit decode → 16-bit project → 16-bit color stack →
  RGB48 readback → `libx265 --profile main10`. Still true 10-bit, just
  not zero-copy.
- The **preview** runs the same 16-bit color stack as the export, then
  composes to an `Rgba8Unorm` SBS for egui. Don't add an 8-bit color
  shortcut into the graded path.

## Hard-won lessons — DON'T regress these

1. **Never do wgpu GPU work on a non-main thread.** The wgpu
   `Device`/`Queue`/swapchain is shared with eframe's renderer; GPU
   submits + `Maintain::Wait` from a second thread wedge Metal's drawable
   queue against the presenting main thread → permanent 0% CPU deadlock.
   `DetailCache` decodes on a background thread (CPU/VideoToolbox only —
   it returns `Vec<u8>` pairs, never touches wgpu) and does the GPU
   project/compose on the **main** thread. Keep it that way.
2. **Preview gamma:** egui treats sampled textures as *linear* and
   re-applies the OETF. Our SBS holds Rec.709-gamma RGB, so preview
   textures are registered with an **sRGB view** to cancel the double
   encode (see the `register_native_texture` calls in `app.rs`). Without
   it the preview looks washed-out vs the export.
3. **Detail still = exact frame:** the native decoder seeks to a keyframe
   and **decodes forward** to the requested pts; without it the still
   shows a keyframe up to a GOP away and the image jumps. (Same fix in
   the live decode worker's seek.)
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
- Don't add CUDA / OpenCL / Numba deps — `wgpu` is the GPU answer.
- Don't shell out to a system `ffmpeg` — use `ffmpeg-next` in-process.
- Don't remove a `#[cfg(target_os = "...")]` gate without confirming the
  API exists on the other platform; the Mac-only paths are FAST paths
  with deliberate cross-platform fallbacks.
- Don't push to `main` (this is the `neo` branch).
- Don't drop the graded path to 8-bit; don't reintroduce the preview
  double-gamma or off-thread GPU deadlock.

## Doc pointers

- [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — **the Windows handoff guide (start here on Windows).**
- [docs/BUILD.md](docs/BUILD.md) — original cross-platform build notes.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate boundaries, GPU pipeline shape.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan (note: predates the GUI; treat as historical).
