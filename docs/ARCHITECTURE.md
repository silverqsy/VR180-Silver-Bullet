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
├── interop_macos      IOSurface ↔ MTLTexture zero-copy (Phases 0.6.5 + 0.6.6)
├── interop_windows    CUDA ↔ Vulkan zero-copy (Phase 0.6.8, planned)
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

### 10-bit end-to-end mandate

When `--bit-depth 10` (or equivalent Main10 output) is selected,
the precision invariant runs the full length of the pipeline:

```
VT decoder (P010 IOSurface, 10-bit)
  → wgpu R16Unorm/Rg16Unorm plane textures
  → nv12_to_eac_cross.wgsl (BT.709 limited-range expansion, float math)
  → Rgba16Float EAC cross texture
  → every color stage (LUT3D, CDL, tonal zones, mid-detail, sharpen)
    — all Rgba16Float in, Rgba16Float out, no 8-bit detour
  → Rgba16Float half-equirect texture
  → readback to packed RGB48LE host buffer (Vec<u16>)
  → swscale RGB48LE → yuv420p10le (libx265) or p010le (VT)
  → encoder: libx265 main10 / hevc_videotoolbox profile=main10
  → .mov / .mp4 with Main10 stream
```

Every box in that chain must be **at least 10 bits per channel**.
The current `Vec<u8>` RGB24 readback is the 8-bit fast path; a
parallel `Vec<u16>` RGB48 readback exists for 10-bit. The 8-bit
path is for fast previews and smaller files; the 10-bit path is
for shipping. **A new color shader that secretly bounces through
`Rgba8Unorm` even once is a regression** — the Python app
fought this for months ("looks the same on the laptop but the
banding shows up on the Vision Pro") and we are not relitigating
it. See [CLAUDE.md](../CLAUDE.md#key-rust-conventions) for the
hard rule.

## Zero-copy decode → GPU

The Python app does `ffmpeg_subprocess → stdout bytes → np.frombuffer
→ MLX/numba → ffmpeg encode stdin` — every frame round-trips through
CPU memory at full resolution. At 8K that's hundreds of MB/sec per
frame of pure memory bandwidth tax.

Neo's interop pattern per platform:

- **macOS:** VideoToolbox decode → `IOSurface` → wrap as `MTLTexture` →
  hand to wgpu as a `wgpu::Texture` via `wgpu::hal::metal::Device::texture_from_raw`.
  No CPU memcpy on the decode side; same trick on the encode side
  feeding the `mvhevc_encode` Swift helper.
- **Windows:** NVDEC frames stay on the CUDA device via `cudarc`
  attached to FFmpeg's `AVCUDADeviceContext`; exported to wgpu as
  a Vulkan external image via the wgpu-hal Vulkan escape.
  (Planned — Phase 0.6.8.)

Both paths are gated behind `--zero-copy`; software decode + CPU
EAC assembly is the cross-platform default fallback.

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
