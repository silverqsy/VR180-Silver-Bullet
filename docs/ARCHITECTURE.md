# Architecture

## Workspace shape

```
crates/vr180-core/      pure-Rust, no I/O, no GPU
├── project            JSON project model (mirrors Python's ProcessingConfig)
├── gyro               GPMF parsing, CORI/IORI/GRAV/MNOR, VQF fusion
├── eac                cross assembly geometry, tile math, dimensions
├── geoc               KLNS / Kannala-Brandt fisheye calibration
├── color              ASC CDL lift/gamma/gain/sat, shadow/highlight/temp/tint, LUT
└── atoms              spatialmedia equivalent: sv3d/st3d/SA3D atom writers

crates/vr180-pipeline/  I/O + GPU
├── decode             ffmpeg-next hwaccel decode (VT / NVDEC)
├── encode             ffmpeg-next encode (h265 / prores) + Swift helper spawn
├── gpu                wgpu device + WGSL kernel orchestration
├── interop_macos      IOSurface ↔ MTLTexture zero-copy (Step 3 from SLRNeo)
├── interop_windows    CUDA ↔ Vulkan zero-copy (Step 3 from SLRNeo)
└── render             frame-loop glue: decode → assemble → kernel → encode

crates/vr180-render/    binary
└── main.rs            CLI: JSON config in, .mov out, NDJSON progress on stderr

apps/                   (reserved)
└── vr180-ui/          future Tauri UI shell, after render CLI is at parity
```

## GPU pipeline

Single backend: `wgpu` (Metal on macOS, Vulkan/DX12 on Windows).
Replaces the Python app's MLX / Numba CUDA / Numba CPU trichotomy.

WGSL shaders we need to port (each is currently a Numba kernel or an
MLX `@mx.compile` function in `vr180_gui.py`):

| Python kernel | WGSL shader | Notes |
|---|---|---|
| `_nb_dirs_to_cross_maps` | `cross_remap.wgsl` | direction vector → EAC cross UV |
| `apply_lut3d` (MLX) | `lut3d.wgsl` | trilinear .cube LUT, in/out RGBA16F |
| sharpen (MLX) | `sharpen.wgsl` | unsharp mask |
| `convert_360_raw_to_equirect` | `eac_to_equirect.wgsl` | EAC cross → half-equirect SBS |
| `rasterize` per-scanline RS warp | `per_scanline_rs.wgsl` | 32 row-groups, 800Hz gyro lookup |
| temp/tint/sat | `color_grade.wgsl` | combined with CDL into one pass |
| shadow/highlight (smoothstep masks) | `tonal_zones.wgsl` | pre-LUT |
| mid-detail clarity | `mid_detail.wgsl` | downsample-blur-upsample |

Frame format on the GPU is always **`Rgba16Float`** to match the
Python app's true-10-bit pipeline. We never drop to 8-bit
intermediates.

## Zero-copy decode → GPU

The Python app does `ffmpeg_subprocess → stdout bytes → np.frombuffer
→ MLX/numba → ffmpeg encode stdin` — every frame round-trips through
CPU memory at full resolution. At 8K that's hundreds of MB/sec per
frame of pure memory bandwidth tax.

Neo follows SLRStudioNeo's interop pattern:

- **macOS:** VideoToolbox decode → `IOSurface` → wrap as `MTLTexture` →
  hand to wgpu as a `wgpu::Texture` via `wgpu::hal::metal::Device::texture_from_raw`.
  No CPU memcpy on the decode side; same trick on the encode side
  feeding the `mvhevc_encode` Swift helper.
- **Windows:** NVDEC frames stay on the CUDA device via `cudarc`
  attached to FFmpeg's `AVCUDADeviceContext`; exported to wgpu as
  a Vulkan external image via the wgpu-hal Vulkan escape.

Both paths are gated behind `--hw-decode auto`; software decode is
the fallback when interop fails.

## Swift helpers

These stay as external Mach-O binaries spawned by the Rust pipeline:

- **mvhevc_encode** — `VTCompressionSessionEncodeMultiImageFrame`
  + spatial metadata baked into the format description. Stdin: raw
  BGR48LE SBS frames. Output: `.mov` with MV-HEVC stereo.
- **apac_encode** — `AVAssetWriter` + `kAudioFormatAPAC`. Accepts a
  WAV input + a video passthrough input; writes a `.mov` with APAC
  spatial audio + copied video. Encode + mux in one pass (works
  around ffmpeg's mov muxer dropping the APAC `dapa` atom).
- **vt_denoise** — `VTTemporalNoiseFilter` with `64RGBALE`
  intermediate (true 10-bit through the denoise stage).

Why not port? They use macOS-only APIs (`VideoToolbox`,
`AVAssetWriter`, `kAudioFormatAPAC`) that have no wgpu / ffmpeg-next
equivalent. The Python app already proved the helper-process pattern
works; we keep it.

## What we explicitly drop from the Python app

- **Numba.** wgpu shaders take its place.
- **PyAV.** `ffmpeg-next` (Rust binding) replaces it.
- **OpenCV.** Only used for `cvtColor` / `remap` in the Python app;
  both become wgpu kernels.
- **MLX.** wgpu Metal backend replaces it.
- **spatialmedia (Google's Python).** Port the ~200 lines of atom
  writing into `vr180-core::atoms`. Trivial.
- **scipy.** Wasn't actually used; never depend on it.
