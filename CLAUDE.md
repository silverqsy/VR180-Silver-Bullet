# CLAUDE.md — orientation for Claude Code sessions

This file is auto-loaded by Claude Code at session start. Read this
first; the depth lives in the doc pointers below.

## What this is

**VR180 Silver Bullet Neo** — a clean-room Rust rewrite of the
GoPro Max VR180 mod processor that lives on the `main` branch
([../vr180_processor/](../vr180_processor/)).
**Cross-platform: macOS + Windows.** Single self-contained binary
per platform, no Python.

Architecture mirrors [SLRStudioNeo](../SLRStudioNeo/) (which mirrors
Gyroflow): pure-Rust headless core, in-process libav for video I/O
(`ffmpeg-next 8.1`), `wgpu` for GPU compute. One GPU backend
abstraction (`wgpu`), no fallback chain.

## Cross-platform stance — IMPORTANT

The project owner **builds and ships on Windows.** Development
happens on macOS (and Windows). Code must compile and run on both.

**Two platform-specific code paths exist on purpose** and stay
behind `#[cfg(target_os = "...")]` gates:

- **macOS only**: `interop_macos` (IOSurface↔Metal↔wgpu zero-copy),
  the Swift helpers in `helpers/swift/` (mvhevc_encode, vt_denoise,
  apac_encode), VideoToolbox hwaccel decode.
- **Windows only**: future `interop_windows` (CUDA↔Vulkan via cudarc
  + wgpu-hal Vulkan escape — Phase 0.6.8), NVENC hwaccel encode.
- **Cross-platform**: everything else. The `vr180-core` crate is
  100% portable Rust. The `vr180-pipeline` crate's video decode/
  encode + wgpu compute work on both — only the *hwaccel* and
  *interop* fast paths are platform-specific.

### Rules for cross-platform code

1. **Every `#[cfg(target_os = "...")]` gate matters.** Removing one,
   or forgetting to add one for a Mac-only / Win-only API, breaks
   the other platform's build silently until CI catches it. Always
   double-check after touching `vr180-pipeline`.

2. **`pub use` re-exports from platform-specific modules** must
   themselves be cfg-gated (see `vr180-pipeline/src/lib.rs`).

3. **`metal`, `core-foundation`, `objc`, `foreign-types`** are
   in the workspace `[workspace.dependencies]` but only consumed by
   `vr180-pipeline::interop_macos` (which is `#[cfg(target_os = "macos")]`).
   They compile-skip on Windows builds. Same will be true for
   `cudarc`, `ash`, `windows-sys` when the Windows interop lands.

4. **CLI flags that depend on platform features** error out with a
   helpful message on the wrong platform — they don't silently
   no-op. See `--zero-copy` on Linux/Windows for the pattern.

5. **`assets/`** + **`docs/`** + the `*.wgsl` shaders are 100%
   cross-platform. WGSL → SPIR-V/MSL/HLSL translation is wgpu's job.

6. **Path handling**: use `std::path::Path` / `PathBuf` everywhere.
   Never use forward-slash string literals for paths.

7. **Subprocess spawn**: when we add Swift helper spawning (Phase
   0.8.6), gate it `#[cfg(target_os = "macos")]`. Windows users will
   never reach that code path; the `mvhevc_encode` etc. binaries
   aren't built on Windows.

## Current state

**Phase 0.6.6 complete.** End-to-end Rust + wgpu pipeline:
`VT decode → IOSurface → wgpu (NV12→EAC cross → equirect → LUT) →
libx265 → mp4`. Pixel-identical to the CPU path, ~2× faster
end-to-end (decode + assembly + projection no longer block the
encoder). See [docs/ROADMAP.md](docs/ROADMAP.md) for the phased
plan; each phase's commit message contains its validation result.

What works today, by platform:

| Capability | macOS | Windows |
|---|---|---|
| Workspace `cargo build --release` | ✅ | ⏳ should work — needs validation |
| GPMF + CORI/IORI parse (Phase 0.2) | ✅ | ✅ pure Rust |
| VQF 9D fusion (Phase 0.3) | ✅ | ✅ pure Rust |
| EAC dims / assembly math (Phase 0.4) | ✅ | ✅ pure Rust |
| Software HEVC decode (Phase 0.4) | ✅ | ✅ via ffmpeg-next |
| Software EAC assembly → equirect (Phases 0.4+0.5) | ✅ | ✅ via wgpu |
| 3D LUT (Phase 0.7) | ✅ | ✅ via wgpu |
| H.265 software encode (Phase 0.8) | ✅ | ✅ via libx265 |
| VideoToolbox hwaccel decode (Phase 0.6) | ✅ | ❌ Mac-only API |
| IOSurface↔Metal zero-copy (Phases 0.6.5+0.6.6) | ✅ | ❌ Mac-only API |
| `--hw-accel vt` / `--zero-copy` flag | ✅ | ❌ errors with helpful message |
| **CUDA↔Vulkan zero-copy** (Phase 0.6.8) | ❌ | ⏳ planned |
| **NVENC HW encode** (Phase 0.8.5) | ❌ | ⏳ planned |
| **VideoToolbox HW encode** (Phase 0.8.5) | ⏳ planned | ❌ Mac-only API |

## Relationship to the Python app on `main`

The Python app on the `main` branch (in the sibling worktree,
`vr180_processor/`) is **the working reference**, not a thing to
modify. We don't touch it. The Rust app on this `neo` branch is the
eventual replacement — the two branches stay divergent until Neo
reaches parity, then `main` will adopt Neo.

When porting an algorithm:

1. Read the Python implementation in `../vr180_processor/vr180_gui.py`
   (or `parse_gyro_raw.py`, `pyvqf.py`).
2. Cross-reference any project memory in
   `~/.claude/projects/-Users-siyangqi-Downloads-vr180-processor/memory/`
   — that's where the hard-won facts live (GEOC calibration,
   GoPro stream conventions, gyro pipeline, VQF MNOR mode, etc.).
3. Port the logic. Don't re-derive — the camera-specific math is
   already validated against real footage.

## Key Rust conventions

- **No `unwrap()` in production code paths.** Use `?` to propagate
  `anyhow::Result` (binaries) / `Result<_, vr180_core::Error>` (libs).
- **No hardcoded dimensions.** Anything camera-format-derived
  (stream width/height, EAC tile size, GEOC sensor dim) lives in
  a struct probed at runtime. The Python app's biggest class of
  bugs was hardcoded `5952×1920` everywhere; we don't repeat it.
- **GPU pipeline is wgpu-only.** No Metal-direct, no CUDA-direct,
  no OpenCL. wgpu picks Metal on macOS, Vulkan on Linux, DX12 on
  Windows. Shaders are WGSL in `crates/vr180-pipeline/src/shaders/`.
- **Swift helpers stay Swift, called from Rust.** `mvhevc_encode`,
  `apac_encode`, `vt_denoise` in `helpers/swift/` — they're
  macOS-native, already optimal, and the Rust pipeline spawns them
  as external processes. Don't reimplement them in Rust.
- **One blast-radius rule for changes:** if you touch a kernel,
  also add a unit test against a known input/output frame.
- **Bit-validate against Python** when porting algorithms. Every
  algorithmic phase so far has produced a bit-identical match
  against the Python pipeline on a real test file. See the commit
  messages for the specific reference values (CORI[0] quaternion,
  VQF bias, EAC tile_w for various stream widths, etc.).

## Doc pointers

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — workspace boundaries,
  why each crate exists, GPU pipeline shape.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan with explicit
  "done when" checkboxes per phase. Already at Phase 0.6.6.
- [docs/BUILD.md](docs/BUILD.md) — FFmpeg / FFMPEG_DIR /
  LIBCLANG_PATH prereqs for both macOS and Windows. Read this
  before attempting a Windows build.

## Don't do

- Don't push to `main` from this worktree (this is the `neo` branch).
- Don't add a CUDA / OpenCL / Numba dependency. `wgpu` is the answer.
- Don't shell out to system `ffmpeg` for decode/encode. We use
  `ffmpeg-next` (in-process libav). The CLI binary spawns the
  Swift helpers, that's it.
- Don't port the PyQt6 GUI to Rust early. The phased plan ships
  the `vr180-render` CLI first so the existing Python GUI can
  shell out to it for big speed wins, without losing the polished
  UX. Tauri UI is Phase 1.0.
- Don't remove `#[cfg(target_os = "macos")]` gates without first
  verifying the underlying API exists on the other platforms. The
  Mac-only paths are FAST paths; the cross-platform fallbacks
  exist deliberately.
