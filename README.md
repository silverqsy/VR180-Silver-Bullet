# VR180 Silver Bullet — Neo

A clean-room Rust rewrite of the [VR180 Silver Bullet](../vr180_processor/)
GoPro Max 2 VR180 mod processor.

## Why a rewrite

The Python/PyQt6 app shipped on `main` works, but its architecture
constrains us:

| Pain point in the Python app | Root cause | Neo fix |
|---|---|---|
| 3 GB Windows install | PyQt6 + Numba + cv2 + av + Python 3.11 | Single ~80 MB native binary |
| "WinError 2 / cudart in PhysX" | Bundled-binary path detection + Numba CUDA toolkit hunt | `ffmpeg-next` links libav in-process; `wgpu` needs no CUDA toolkit |
| "could not broadcast (944,1920) → (1008,1920)" | Hardcoded `5952×1920` stream dimensions in 7+ places | `EacDims` struct derived from `ffprobe` once, type-checked everywhere |
| ~8 fps export with `Decode: FFmpeg (fallback)` | subprocess pipe + Python per-frame overhead | In-process libav + zero-copy GPU = 3–10× faster |
| MLX / Numba CUDA / Numba CPU dispatch maze | Three GPU stacks, each with edge cases | `wgpu` = one path, runtime-selects Metal / Vulkan / DX12 |
| SA3D `KeyError`, scipy phantom import | Python loose typing + duck-typed metadata dicts | Rust enums + `?` propagate errors at compile time |

Architecture mirrors [Gyroflow](https://github.com/gyroflow/gyroflow):
pure-Rust headless core, in-process libav for video I/O
(`ffmpeg-next 8.1`), `wgpu` for GPU compute. The Mac-native Swift
helpers (`mvhevc_encode`, `apac_encode`, `vt_denoise`) carry over
unchanged from the Python app — they're already optimal and the
Rust pipeline spawns them as external processes.

## Workspace layout

```
crates/
├── vr180-core/      # pure Rust: gyro/VQF, EAC math, GEOC, color math,
│                    # project config, GPMF / SA3D / sv3d atom writing
├── vr180-pipeline/  # ffmpeg-next decode/encode + wgpu kernels +
│                    # Swift helper spawn glue
└── vr180-render/    # CLI binary: reads JSON config → renders → exits.
                     # The existing Python GUI on `main` will eventually
                     # shell out to this for the heavy work.

apps/                # (reserved) Tauri UI shell — comes after render
                     # CLI is at parity with the Python export pipeline.

helpers/swift/       # macOS VideoToolbox / AVFoundation helpers
                     #   mvhevc_encode  — MV-HEVC spatial video encoder
                     #   vt_denoise     — VTTemporalNoiseFilter
                     #   apac_encode    — Apple Positional Audio Codec
                     # Build via helpers/build_swift.sh
```

## Status

**Phase 0.1 — skeleton.** Workspace compiles, no functionality yet.
See [docs/ROADMAP.md](docs/ROADMAP.md) for the phased plan and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design notes.

## Build

See [docs/BUILD.md](docs/BUILD.md) for the FFmpeg / `FFMPEG_DIR`
prereqs. Once those are in place:

```sh
# Native build
cargo build --release

# CLI invocation (placeholder for now)
./target/release/vr180-render --help

# macOS Swift helpers
./helpers/build_swift.sh
```

## License

MIT.
