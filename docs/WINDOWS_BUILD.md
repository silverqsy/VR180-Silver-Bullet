# Building the Windows version (handoff guide)

This is a self-contained guide for building **`vr180-gui`** (the GUI app)
on a Windows PC. Development happens on macOS; this doc gets a Windows
machine from a clean checkout to a running `vr180-gui.exe`.

> The app is **cross-platform by design**, and Windows has its OWN fast
> paths now — it is NOT a software-fallback build: NVDEC (d3d11va) decode,
> a zero-copy D3D11→Vulkan live preview, and a **GPU-resident NVENC H.265
> export** (NVDEC→wgpu→CUDA→NVENC, no CPU readback; `libx265` fallback).
> Only the Apple-specific blocks (VideoToolbox, IOSurface/Metal interop,
> ProRes-VT) are compiled out, replaced by the Windows equivalents.

> **Current state (June 2026):** the Windows build has run before (NVENC
> export verified on a 4090). Since then a large macOS-developed feature
> batch landed on `2.0` — exact DJI lens model (k5 + tangential), 195°
> equidistant fisheye output, velocity-dampened soft-stab, eye-swap /
> upside-down fixes, 8K export. It's all shared code; see the **Status**
> section of [CLAUDE.md](../CLAUDE.md) for the verify-on-Windows checklist.

---

## Step 0 — Get the code

Everything the build needs is committed on the `2.0` branch (assets,
shaders, `Cargo.lock` — the lockfile pins the exact dependency versions
the Windows build will use):

```pwsh
git clone https://github.com/silverqsy/VR180-Silver-Bullet.git
cd VR180-Silver-Bullet
git checkout 2.0
```

---

## Step 1 — Install the toolchain (Windows)

| Tool | How | Why |
|---|---|---|
| **Rust** (1.79+, MSVC) | <https://rustup.rs/> → `rustup default stable` | Compiler. The MSVC toolchain is the default on Windows; install "Desktop development with C++" from the Visual Studio Build Tools when rustup prompts. |
| **LLVM / libclang** (17+) | `winget install LLVM.LLVM` or the installer at <https://releases.llvm.org/> | `ffmpeg-sys-next` runs `bindgen`, which needs `libclang.dll`. |
| **FFmpeg dev libs** (7.x) | a *dev* distribution with `.lib`s + headers + DLLs (see Step 2) | `ffmpeg-next = 8.1` links libav in-process. |

After installing LLVM, point Rust's bindgen at it (PowerShell, current session):

```pwsh
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
```

## Step 2 — FFmpeg dev distribution

Get a **dev** build of FFmpeg 7.x (with `include\`, `lib\`, `bin\`). Good sources:

- <https://www.gyan.dev/ffmpeg/builds/> — the `ffmpeg-release-full-shared` / `dev` variant
- <https://github.com/ShiftMediaProject/FFmpeg/releases>
- <https://github.com/m-ab-s/media-autobuild_suite>

Extract somewhere stable and point the build at it:

```pwsh
$env:FFMPEG_DIR = "C:\path\to\ffmpeg-7.x-dev"
# Expected layout:
#   %FFMPEG_DIR%\include\libavformat\avformat.h   (+ libavcodec, libavutil, libswscale)
#   %FFMPEG_DIR%\lib\avformat.lib                 (+ the rest)
#   %FFMPEG_DIR%\bin\avformat-*.dll               (runtime DLLs)
```

The version of FFmpeg must match the `ffmpeg-next = "8.1"` pin's expected
ABI (FFmpeg 7.x). If you hit `avformat-XX.lib not found` or symbol
mismatches, the dev build's major version is wrong.

## Step 3 — Build the GUI

```pwsh
cargo build --release -p vr180-gui
```

> Build **`-p vr180-gui`** specifically. Don't `cargo build --release`
> (whole workspace) — the secondary `vr180-render` CLI currently has an
> unrelated non-exhaustive-match compile error and isn't needed for the
> GUI. (If you want the CLI too, that match in `crates/vr180-render/src/main.rs`
> needs the `ProResVideoToolbox` / `ProResKs` arms added.)

First build is ~3-5 min (ffmpeg-sys bindgen + ffmpeg-next are the slow
crates); incremental builds are seconds.

## Step 4 — Run

The ffmpeg DLLs must be findable at runtime. Either prepend the bin dir to
PATH for the session, or copy the DLLs next to the exe:

```pwsh
# Option A — PATH (dev):
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"
.\target\release\vr180-gui.exe

# Option B — copy DLLs next to the exe (distributable):
Copy-Item "$env:FFMPEG_DIR\bin\*.dll" .\target\release\
.\target\release\vr180-gui.exe
```

Load a `.osv` / `.mp4` / `.braw` via the file picker and confirm the
preview renders, the color/LUT panel works, and an export produces a file.

---

## What's different on Windows vs macOS

| Area | macOS | Windows | Notes |
|---|---|---|---|
| GPU backend | Metal | **DX12 / Vulkan** (wgpu auto-selects) | No code change — wgpu picks the backend. |
| H.265 encode | `hevc_videotoolbox` (HW) | **`libx265`** (SW) | Auto-selected in `commit_export`. 10-bit = libx265 Main10. Slower than HW, same quality knobs. |
| ProRes encode | `prores_videotoolbox` (HW) | **`prores_ks`** (SW) | Auto-selected. |
| 10-bit export path | P010 IOSurface zero-copy | **16-bit project → RGB48 readback → libx265 Main10** | Still 10-bit end-to-end; just not zero-copy. The whole `interop_macos` / IOSurface layer is `#[cfg(target_os = "macos")]`. |
| Decode | VideoToolbox P010 zero-copy | **ffmpeg software decode** (or d3d11va via `HwDecode::Auto`) | OSV dual-stream decode is ffmpeg-based and portable. |
| Preview, color stack, LUT (bundled), zoom/detail cache, audio (cpal→WASAPI), file picker (rfd), settings persistence | ✓ | ✓ | Fully cross-platform. Settings now save to `%APPDATA%\VR180SilverBullet2.0\settings.json`. |

**Net effect:** Windows exports work and look identical (same 16-bit color
math, same LUT order), but **export is slower** because encode/decode run
in software instead of on Apple's media engine.

### wgpu 16-bit requirement

The device requests `TEXTURE_FORMAT_16BIT_NORM` (needed for the
`Rgba16Unorm` 16-bit color stack). This is supported on essentially all
DX12 GPUs and modern Vulkan drivers. On an adapter that lacks it, the
16-bit paths degrade — if the preview/export look wrong, check the startup
log for the requested features (`main.rs` requests it best-effort).

---

## Verifying the platform gates (first Windows build)

Because day-to-day development is on macOS, the Windows target hasn't been
compiled after every recent change. The big macOS-only modules
(`interop_macos`, the zero-copy decode/export, VideoToolbox encode) are
properly gated (`#[cfg(target_os = "macos")]`) and the macOS-only crates
(`metal`, `core-foundation`, `objc`, `foreign-types`) are
`[target.'cfg(target_os = "macos")'.dependencies]` so they never build on
Windows. But if a recent edit referenced macOS-only code without a gate,
you'll see it as a compile error like:

```
error[E0432]: unresolved import `crate::interop_macos`
error: cannot find function `create_p010_encode_buffer` ...
```

Fix = wrap the offending block in `#[cfg(target_os = "macos")]` (with a
`#[cfg(not(target_os = "macos"))]` fallback or `unreachable!()` where a
value is required — see existing examples in `fisheye_export.rs` around the
`use_zero_copy_*` branches). Then re-run `cargo build --release -p vr180-gui`.

Quick sanity command after touching anything in `vr180-pipeline`:

```pwsh
cargo check -p vr180-gui
```

---

## Optional polish (do after it builds & runs)

- **Hide the console window.** A Windows GUI exe currently also opens a
  console (where logs print). To suppress it in release builds, add to the
  top of `crates/vr180-gui/src/main.rs`:
  ```rust
  #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
  ```
  Leave it off until the port is verified — the console is where you'll see
  `tracing` logs and panics during the first build.
- **Bundle the DLLs** next to the exe (Step 4, Option B) for a portable
  folder you can zip and share.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: unable to find libclang` | `$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"` (dir containing `libclang.dll`). |
| linker can't find `avformat-*.lib` | `$env:FFMPEG_DIR` must point at the **dev** distribution root (has `lib\*.lib`). |
| runtime: "avcodec-XX.dll not found" | Copy `%FFMPEG_DIR%\bin\*.dll` next to the exe, or add that bin dir to PATH. |
| missing `audio_player.rs` / a shader / the `.cube` | The handoff didn't include untracked files — redo **Step 0** (commit `-A` and push, or copy the full working tree). |
| `cannot find ... metal / core-foundation / interop_macos` | A macOS-only reference slipped past a cfg gate — wrap it (see "Verifying the platform gates"). |
| whole-workspace build fails on `vr180-render` | Build `-p vr180-gui` only; the CLI has a separate pre-existing match error. |
| build hangs on `ffmpeg-sys-next` | First bindgen run is 30-60 s; if longer, confirm `clang --version` works. |

See `docs/BUILD.md` for the original cross-platform build notes.
