# CLAUDE.md — orientation for Claude Code sessions

This file is auto-loaded by Claude Code at session start. Read this
first; the depth lives in the doc pointers below.

## What this is

**VR180 Silver Bullet Neo** — a clean-room Rust rewrite of the
GoPro Max 2 VR180 mod processor that lives on the `main` branch
([../vr180_processor/](../vr180_processor/)). Cross-platform
(macOS + Windows), single self-contained binary, no Python.

Architecture mirrors [SLRStudioNeo](../SLRStudioNeo/) (which mirrors
Gyroflow): pure-Rust headless core, in-process libav for video I/O
(`ffmpeg-next 8.1`), `wgpu` for GPU compute. One backend, no fallbacks.

## Current state

**Phase 0.1 — skeleton only.** Workspace compiles with the three
crates as empty stubs. No decode, no GPU, no CLI args yet. See
[docs/ROADMAP.md](docs/ROADMAP.md) for the phased plan
(0.1 skeleton → 0.9 export parity with the Python app).

## Relationship to the Python app on `main`

The Python app on the `main` branch (in the parent directory's
sibling worktree, `vr180_processor/`) is **the working reference**,
not a thing to modify. We don't touch it. The Rust app on this
`neo` branch is the eventual replacement — the two branches stay
divergent until Neo reaches parity, then `main` will adopt Neo.

When porting an algorithm:

1. Read the Python implementation in `../vr180_processor/vr180_gui.py`
   (or `parse_gyro_raw.py`, `pyvqf.py`).
2. Cross-reference any project memory in `~/.claude/projects/-Users-siyangqi-Downloads-vr180-processor/memory/`
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
  Windows. Shaders are WGSL in `crates/vr180-core/src/gpu/shaders/`.
- **Swift helpers stay Swift.** `mvhevc_encode`, `apac_encode`,
  `vt_denoise` in `helpers/swift/` — they're macOS-native, already
  optimal, and the Rust pipeline spawns them as external processes.
  Don't reimplement them in Rust.
- **One blast-radius rule for changes:** if you touch a kernel,
  also add a unit test against a known input/output frame.

## Doc pointers

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — workspace boundaries,
  why each crate exists, GPU pipeline shape.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan with explicit
  "done when" checkboxes per phase.
- [docs/BUILD.md](docs/BUILD.md) — FFmpeg avbuild prereqs (the
  one painful setup step; one-time per machine).

## Don't do

- Don't push to `main` from this worktree (this is the `neo` branch).
- Don't add a CUDA / OpenCL / Numba dependency. `wgpu` is the answer.
- Don't shell out to system `ffmpeg` for decode/encode. We use
  `ffmpeg-next` (in-process libav). The CLI binary spawns the
  Swift helpers, that's it.
- Don't port the PyQt6 GUI to Rust early. The phased plan ships
  the `vr180-render` CLI first so the existing Python GUI can
  shell out to it for big speed wins, without losing the polished
  UX. Tauri UI is Phase 0.7+.
