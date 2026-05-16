# Build

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust | 1.79+ | `rustup default stable` |
| FFmpeg dev libs | 7.x (compatible with `ffmpeg-next 8.1`) | See below |
| Clang / libclang | recent | needed by `ffmpeg-sys-next` bindgen |
| Swift toolchain | macOS only | for `helpers/build_swift.sh` |

### FFmpeg (the one painful setup)

`ffmpeg-next 8.1` is a thin wrapper around libav and links against
system FFmpeg dev libs. We don't ship our own libav build; we
follow Gyroflow's proven setup, exactly as SLRStudioNeo does.

**macOS:**
```sh
brew install ffmpeg pkg-config
# ffmpeg-sys-next will find headers via pkg-config — no env vars needed
```

**Windows:**
Download an avbuild dev distribution from
<https://github.com/ShiftMediaProject/FFmpeg/releases> (or
<https://www.gyan.dev/ffmpeg/builds/> dev variant). Extract somewhere
stable and set:
```cmd
set FFMPEG_DIR=C:\path\to\ffmpeg-7.x-dev
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
```

## Build the Rust crates

```sh
cargo build --release
./target/release/vr180-render --help
```

Build time on a clean cache is ~3–5 min (ffmpeg-next is the slow
crate; subsequent builds are seconds).

## Build the Swift helpers (macOS only)

```sh
./helpers/build_swift.sh
```

Outputs three executables in `helpers/bin/`:
- `mvhevc_encode`
- `vt_denoise`
- `apac_encode`

The Rust pipeline finds them via:
1. `helpers/bin/` (development)
2. Bundled in the same dir as `vr180-render.exe` (release)

## Cross-compile (later)

Not in scope for Phase 0.1. When we ship binaries, we'll build
natively per platform in CI (same as SLRStudioNeo).
