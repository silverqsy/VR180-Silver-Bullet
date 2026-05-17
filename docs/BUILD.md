# Build

This project is **cross-platform**: development happens on macOS;
release builds run on both macOS and Windows. Linux should work
(it's a stretch for the GPU backends but the core compiles) — not
tested in CI yet.

## Prerequisites by platform

| Tool | macOS | Windows | Notes |
|---|---|---|---|
| Rust toolchain | 1.79+ | 1.79+ | `rustup default stable` |
| FFmpeg dev libs | `brew install ffmpeg pkg-config` | avbuild distribution, see below | Linked by `ffmpeg-next 8.1` |
| Clang / libclang | shipped with Xcode CLT | LLVM 17+ installer | Needed by `ffmpeg-sys-next` bindgen |
| Swift toolchain | Xcode CLT | — | For `helpers/build_swift.sh`, macOS only |
| CUDA toolkit | — | optional (12.x, future Phase 0.6.8) | For zero-copy CUDA↔Vulkan interop on Windows |

## macOS

```sh
brew install ffmpeg pkg-config
cargo build --release
./target/release/vr180-render --help
```

`ffmpeg-sys-next` finds the FFmpeg headers via `pkg-config` — no
environment variables needed. First build is ~3-5 min (ffmpeg-next
is the slow crate); incremental builds are seconds.

To build the macOS-only Swift helpers (only needed when working on
the spatial video / spatial audio / temporal denoise paths):

```sh
./helpers/build_swift.sh
```

Output goes to `helpers/bin/`, which is `.gitignore`d.

## Windows

### 1. Rust + LLVM

```pwsh
# Rustup installer from https://rustup.rs/
rustup default stable

# LLVM 17+ — needed by ffmpeg-sys-next bindgen for the FFI bindings.
# Use the official Windows installer from https://releases.llvm.org/
# OR `winget install LLVM.LLVM`.
# Then point Rust at it:
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
```

### 2. FFmpeg avbuild

Get a dev distribution (with `.lib`s + headers) from
<https://github.com/m-ab-s/media-autobuild_suite>,
<https://github.com/ShiftMediaProject/FFmpeg/releases>, or
<https://www.gyan.dev/ffmpeg/builds/> (the `dev` variant).
Extract somewhere stable, then:

```pwsh
$env:FFMPEG_DIR = "C:\path\to\ffmpeg-7.x-dev"
# avbuild expects this layout:
#   $FFMPEG_DIR\include\libavformat\avformat.h  (and friends)
#   $FFMPEG_DIR\lib\avformat.lib                (and friends)
#   $FFMPEG_DIR\bin\avformat-*.dll              (runtime DLLs)
```

The DLLs in `$FFMPEG_DIR\bin\` need to be **on PATH or next to the
output exe at runtime**. Two options:

- **Dev**: prepend `$FFMPEG_DIR\bin` to PATH and run from a shell
  that has it set.
- **Release**: copy `*.dll` from `$FFMPEG_DIR\bin\` next to
  `target\release\vr180-render.exe`.

### 3. Build

```pwsh
cargo build --release
.\target\release\vr180-render.exe --help
```

If the build fails with `unable to find libclang`, double-check
`$env:LIBCLANG_PATH` points at the directory containing
`libclang.dll`. If linker errors mention missing `avformat-7.lib`,
re-check `$env:FFMPEG_DIR`.

### 4. Validate platform-specific code is properly gated

The `interop_macos`, `helpers::swift`, and Swift helper binaries
in `helpers/swift/` are all `#[cfg(target_os = "macos")]` and
should compile-skip on Windows. If `cargo build --release` succeeds
on Windows you're golden — the cfg gates work. CI will eventually
do this check on every commit.

Run `cargo check --workspace` after any change to
`vr180-pipeline/src/lib.rs`, `interop_macos.rs`, or anything that
imports `metal` / `core-foundation` / `objc` — those are macOS-only
crates and Windows builds will refuse to link them if a gate is
missing.

## Feature flags

`vr180-pipeline` has one feature today:

| Feature | Default | What it does |
|---|---|---|
| `no-ffmpeg` | off | Skip the `ffmpeg-next` integration (only the GPU side compiles). Useful for very early scaffolding work without an FFmpeg install. Drop once a stable FFmpeg setup is in place. |

The wgpu device requests `TEXTURE_FORMAT_16BIT_NORM` at runtime
(needed for the P010 10-bit zero-copy path). On adapters that
don't support it (older Vulkan drivers, some software adapters),
the feature is silently skipped — only the zero-copy code paths
break, the rest of the pipeline still works.

## Cross-compile

Not in scope for the current phase. CI will build natively per
platform when we get there (same as SLRStudioNeo). If you need to
cross-compile sooner, the FFmpeg side is the hard part: you need
a sysroot with libavformat / libavcodec / libavutil for the target
triple. `cargo-zigbuild` + a vendored avbuild is the path others
have taken (see Gyroflow's CI).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| **macOS**: `pkg-config error: package 'libavformat' not found` | `brew install ffmpeg pkg-config` then `brew link ffmpeg` |
| **Windows**: `error: unable to find libclang` | Set `LIBCLANG_PATH` to the directory containing `libclang.dll` (LLVM 17+ install) |
| **Windows**: linker can't find `avformat-7.lib` | Check `FFMPEG_DIR` points at the dev distribution root |
| **Windows**: runtime DLL not found | Copy `avformat-7.dll` etc. from `$FFMPEG_DIR\bin\` next to the exe, or prepend `$FFMPEG_DIR\bin` to PATH |
| **macOS**: `--zero-copy` works, output looks broken | Confirm input is P010 (10-bit HEVC, the GoPro Max default). 8-bit (NV12) GoPro footage isn't supported by the zero-copy path yet (Phase 0.6.6.5). |
| **Windows**: `--zero-copy` / `--hw-accel vt` errors out | Expected. macOS-only flags. Use `--hw-accel sw` (the default on Windows). NVENC + CUDA↔Vulkan interop coming in Phase 0.6.8 / 0.8.5. |
| Both: `cargo build` hangs on `ffmpeg-sys-next` | First-time bindgen takes 30-60 s. If it's been longer, kill it and verify libclang resolves: `clang --version`. |
