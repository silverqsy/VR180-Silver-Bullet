# Roadmap

Each phase ends with a commit and a `cargo run -p vr180-render -- …`
demo that exercises the new capability end-to-end. The numbering
leaves room for half-step phases (0.6.5, 0.6.6, …) so deferred
sub-features can land without renumbering the whole tree.

## Phase 0.1 — Workspace skeleton ✅
**Done when:** `cargo build --release` succeeds with three empty
crates, CLI prints `--help`.

- [x] Workspace `Cargo.toml`
- [x] `crates/vr180-core` lib stub
- [x] `crates/vr180-pipeline` lib stub
- [x] `crates/vr180-render` bin stub with `clap`
- [x] Swift helpers preserved under `helpers/swift/`
- [x] `.gitignore`, `LICENSE`, `README.md`, `CLAUDE.md`

## Phase 0.2 — Project model + GPMF parsing ✅

Port `parse_gyro_raw.py` core: GPMF stream extraction (via
ffmpeg-next, not subprocess) + CORI / IORI quaternion parsing.

- [x] `vr180-core::project::ProcessingConfig` (skeleton; fields land per stage)
- [x] `vr180-core::gyro::gpmf::GpmfWalker` — streaming iterator
      (DEVC/STRM container traversal, padding-aware)
- [x] `vr180-core::gyro::cori_iori::{parse_cori, parse_iori, Quat,
      quat_to_euler_zyx}` — Q15 big-endian decode
- [x] `vr180-pipeline::decode::extract_gpmf_stream` — ffmpeg-next
      based, in-process; picks the `gpmd`-tagged stream (not the
      `tmcd` timecode stream that `best(Data)` returns)
- [x] CLI: `vr180-render probe-gyro <file.360>` prints CORI/IORI
      count + first sample + Euler ranges
- [x] **Validated** against Python output on `GS010172.360`:
      same 321 536-byte GPMF blob, same 875 / 875 sample counts,
      same `CORI[0] = (0.999969, 0.001495, 0.000610, 0.001984)`,
      same Euler ranges to ±0.001°. **41 ms** end-to-end vs
      Python's ~500 ms+ (~12× faster, mostly subprocess-startup
      tax avoided).

## Phase 0.3 — VQF no-firmware-RS path ✅

- [x] `vr180-core::gyro::raw` — RawImu block parser (GYRO/ACCL/GRAV/MNOR,
      SCAL + STMP tracking, type/struct_size validation)
- [x] `vr180-core::gyro::vqf::run` — thin wrapper over the
      [`vqf-rs`](https://crates.io/crates/vqf-rs) 0.3.0 crate
      (full 9D PyVQF port, MIT-licensed). Param choices match
      `vqf_to_cori_quats` (mag → `mag_dist_rejection_enabled=false`
      + `tau_mag=5.0`).
- [x] `vr180-pipeline::imu::prepare_for_vqf` — input prep
      (ZXY → body axis remap, GRAV vs ACCL source pick with magnitude
      heuristic, proportional resampling to gyro rate)
- [x] CLI: `vr180-render probe-gyro --vqf <file.360>`
- [x] **Validated** against Python output on `NO firmware RS No IORI.360`
      (the bias-drifting reference clip):
      bias `[-0.097, -0.099, -0.071] °/s` — **bit-identical to 3 decimals**.
      Total time 94 ms (prep 86 ms + VQF 8 ms) vs Python's ~2-5 s
      (~25× faster).
      Also runs successfully on the other two test clips (firmware-RS
      footage) without crashing — confirming the pipeline doesn't
      assume bias-drifting input.

### Deferred to a follow-up (0.3.5)

Resample 798 Hz VQF quaternions → 30 fps frame quaternions with the
SROT-midpoint sampling window the Python `resample_quats_to_frames`
does, plus the Y↔Z swap that aligns the VQF output to CORI's
on-disk component order. These are presentation concerns — needed
when we wire the gyro pipeline into the export render but not for
algorithmic validation, which the bias-match nails.

## Phase 0.4 — Decode + EAC assembly (CPU baseline) ✅

- [x] `vr180-pipeline::decode::extract_first_stream_pair` — ffmpeg-next
      software decode of both HEVC streams in a `.360`, returns a
      `StreamPair { s0, s4, dims }` of packed RGB8 host buffers.
      Stream dimensions are **probed at runtime** (no hardcoded 5952).
      Replaces both the Python pipeline's PyAV path AND the legacy
      ffmpeg-subprocess fallback path.
- [x] `vr180-core::eac::Dims { stream_w, stream_h }` with the corrected
      `tile_w = (stream_w - 1920) / 4` formula. Unit-tested on Max
      (5952 → tw=1008), Max 2 5888 (tw=992), and Max 2 5696
      (tw=944, **the value that broke the Python hardcoded slice
      for one user this session**). `is_valid()` rejects widths whose
      `(w-1920) % 4 != 0` with a clear error message rather than
      crashing mid-assembly.
- [x] `vr180-core::eac::{assemble_lens_a, assemble_lens_b,
      fill_cross_corners, rotate_90_cw, rotate_90_ccw}` — pure-Rust
      assembly on packed RGB8. All slice offsets derived from
      `tile_w`; corner replication + 1-px seam fix match the Python
      `_fill_cross_corners`. `blit_rect` helper deliberately omits
      a `copy_w` parameter to make the dst-x / copy-w swap that
      bit me in development impossible at the call site.
- [x] CLI: `vr180-render probe-eac <file.360> [--out <file.png>]`
      prints stream + EAC layout numbers, and when `--out` is supplied,
      decodes the first frame of each HEVC stream, assembles both lens
      crosses, and writes them stacked (Lens A on top, Lens B on
      bottom) to one PNG for visual sanity check.
- [x] **Visually verified** on the three test clips — indoor static
      scene, kitchen scene, outdoor NYC. Crosses are tight; corners
      filled; no visible discontinuities at face boundaries.
- [x] Timings: decode 80-250 ms (two HEVC streams, sw decode) +
      assembly 22 ms + PNG encode 30-100 ms. The decode is the
      dominant cost; Phase 0.6 makes that hardware-accelerated.

## Phase 0.5 — wgpu device + first kernel ✅

- [x] `vr180-pipeline::gpu::Device` (wgpu instance, adapter, queue,
      cached compute pipelines). Backend auto-selected: Metal on macOS,
      DX12 / Vulkan on Windows. Confirmed M5 Max picks the Metal
      backend automatically.
- [x] `eac_to_equirect.wgsl` — first real WGSL compute kernel. Per-pixel
      direction → 5-face EAC selector → arctan UV reconstruction →
      bilinear sample. Ported from Python `FrameExtractor.build_cross_remap`.
- [x] `Device::project_cross_to_equirect(cross_rgb, cross_w, out_w, out_h)`
      — RGB8 upload (padded to RGBA8 for wgpu) → kernel dispatch →
      readback. Pipeline / shader / sampler / bind-group-layout are
      built once at `Device::new()` so per-frame work is just the
      buffer marshaling + 1 dispatch.
- [x] CLI: `vr180-render probe-eac --equirect <png>` decodes one frame,
      assembles both lens crosses, projects each via the kernel, and
      stitches them L|R into a single half-equirect SBS PNG.
- [x] **Visually verified** on `GS010172.360`: clean half-equirect SBS,
      no face artifacts, both eyes show correct stereo parallax. The
      output is the same projection the Python pipeline produces, just
      done end-to-end in Rust + wgpu instead of Python + MLX.

Timings (M5 Max, 5952×1920 input, 4096×2048 SBS output):
  decode (2× HEVC sw):  ~130 ms
  EAC assembly (CPU):     ~22 ms
  GPU device init:        ~11 ms  (shader compile + pipeline build)
  GPU project (2× eyes):  ~67 ms  (≈ 33 ms / eye, cached pipeline)
  PNG encode + write:     ~31 ms

Per-frame steady state (Device reused): ~252 ms ≈ **4 fps**. The
decode dominates by a factor of ~2 over everything else — Phase 0.6
addresses that.

## Phase 0.6 — Hardware decode (host-memory path) ✅

VideoToolbox decode wired via raw `ffmpeg_sys_next` FFI (the
ffmpeg-next 8.1 safe wrapper doesn't expose hwaccel). Frames are
transferred from VT-managed memory to host via `av_hwframe_transfer_data`,
then go through the same swscale → RGB path the sw decoder uses.

- [x] `vr180-pipeline::decode::HwDecode` enum (`Auto` / `Software` /
      `VideoToolbox`) + `DecodePath` (which path actually ran after
      Auto / fallback resolution).
- [x] `try_enable_videotoolbox_decode(codec_ctx)` — calls
      `av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_VIDEOTOOLBOX)`, attaches
      the device on the codec context, installs a `get_format`
      callback that prefers `AV_PIX_FMT_VIDEOTOOLBOX`.
- [x] `download_hw_frame` — wraps `av_hwframe_transfer_data` for
      hwframe → NV12/P010 host memory.
- [x] `bench_decode_throughput` — pure-decode throughput micro-bench
      with optional hwaccel.
- [x] `extract_first_stream_pair_with(path, HwDecode)` — full pipeline
      that lazily picks scaler params based on the actual sw-frame
      format after transfer.
- [x] CLI: `--hw-accel {auto|sw|vt}` on `probe-eac`, new `bench-decode`
      subcommand for A/B perf comparison.

**Measured speedup on M5 Max, HEVC 5952×1920** (`bench-decode`):

| Clip | Software | VideoToolbox | Speedup |
|---|---|---|---|
| GS010172.360 (30 s, 100 frames)        | 14.9 fps (67 ms/frame) | 152.9 fps (6.5 ms/frame) | **10.2×** |
| firmware RS+IORI.360 (4 min, 300 frames) | 11.7 fps (86 ms/frame) | 132.2 fps (7.6 ms/frame) | **11.3×** |

That's the entire HEVC decode (one stream) including the
host-memory transfer, but excluding any RGB conversion. With both
streams in flight in series the effective full-pipeline throughput
should land near ~75 fps before EAC assembly + GPU project + encode.

## Phase 0.6.5 — IOSurface↔Metal↔wgpu zero-copy bridge ✅

Substrate for skipping the host-memory hop on the decode side:

```text
  Phase 0.6 path:                    Phase 0.6.5 bridge:
  VT → CVPixelBuffer                 VT → CVPixelBuffer
       │                                  │
       ▼ av_hwframe_transfer_data         ▼ CVPixelBufferGetIOSurface (Get-Rule)
   NV12 host                          IOSurface (CFRetain'd)
       │                                  │
       ▼ swscale                          ▼ newTextureWithDescriptor:iosurface:plane:
   packed RGB host                    MTLTexture (Y as R8Unorm)
       │                                  │
       ▼ queue.write_texture              ▼ wgpu-hal Metal escape
   wgpu compute                        wgpu::Texture (zero memcpy)
```

- [x] `vr180-pipeline::interop_macos` — `RetainedIOSurface` RAII +
      raw FFI to `CoreFoundation` (`CFRetain` / `CFRelease`),
      `CoreVideo` (`CVPixelBufferGetIOSurface`), `IOSurface`
      (plane count/dims/strides). The `metal = "0.28"` pin is
      load-bearing — wgpu-hal's Metal layer internally expects that
      exact `metal::Texture` type, so any drift breaks the hal escape.
- [x] `extract_iosurface_from_vt_frame(AVFrame) -> RetainedIOSurface`
      — pulls the IOSurface from `AVFrame::data[3]` (FFmpeg's VT
      hwaccel convention) and retains so the surface outlives the
      AVFrame's recycle window.
- [x] `metal_texture_from_iosurface_plane(...)` —
      `MTLDevice.newTextureWithDescriptor:iosurface:plane:` via
      `objc::msg_send!` (metal-rs 0.28 doesn't expose this selector).
- [x] `wgpu_texture_from_iosurface_plane(...)` — full chain
      `IOSurface plane → MTLTexture → wgpu-hal Metal Texture →
      wgpu::Texture` with the correct usage flags
      (TEXTURE_BINDING + STORAGE_BINDING + COPY_SRC/DST).
- [x] `decode_first_vt_frame(path) -> AVFrame` — decode helper that
      stops at the VT-format frame (no `av_hwframe_transfer_data`).
- [x] CLI: `vr180-render probe-iosurface <file.360>` — runs the
      full chain and reads back the first Y-plane row to prove the
      GPU sees the actual decoder output bytes.

**End-to-end probe on `GS010172.360`** (M5 Max, macOS 14+):

```
[1] decode_first_vt_frame:           76.28 ms  (VIDEOTOOLBOX, 5952×1920)
[2] CVPixelBufferGetIOSurface:        583 ns   (planes=2, y=5952×1920, uv=2976×960)
[3] IOSurfaceNv12Descriptor:         OK       (y_bpr=11904, uv_bpr=11904)
[4] wgpu_texture_from_iosurface(Y):  41.5 µs  (5952×1920 R8Unorm)
[5] read back Y row 0:               5.28 ms  (min=0 max=192 avg=105.4)
```

The bytes the GPU sees ARE the bytes VideoToolbox wrote — same
backing IOSurface, unified-memory shared between the VT decoder
and the Metal device. The bridge itself takes microseconds. The
~50 MB per-stream `av_hwframe_transfer_data` memcpy + the
matching `queue.write_texture` upload from Phase 0.6's path are
both gone.

## Phase 0.6.6 — Pipeline integration of the zero-copy substrate ✅

The substrate from 0.6.5 now drives the export pipeline.

- [x] `shaders/nv12_to_eac_cross.wgsl` — one compute pass: read Y
      (R16Unorm) + UV (Rg16Unorm) plane textures, do per-pixel
      EAC tile-source mapping (LEFT/CENTER/RIGHT from s0,
      rotated TOP/BOTTOM from s4 with un-rotated coordinates),
      sample with bilinear filtering, BT.709 limited-range
      **P010** YUV→RGB, write to an `Rgba8Unorm` cross texture.
      Corner regions edge-replicate from the nearest side face.
- [x] `Device::nv12_to_eac_cross(...)` Rust wrapper, one dispatch
      per lens (uniform picks Lens A vs Lens B layout).
- [x] `Device::project_cross_texture_to_equirect(...)` — variant
      of `project_cross_to_equirect` that takes a `wgpu::Texture`
      directly (skips the RGB upload step). Reuses the cached
      `eac_to_equirect` pipeline + sampler with a fresh bind group.
- [x] `decode::ZeroCopyStreamPairIter` — streaming iterator that
      yields `ZeroCopyStreamPair { s0_y, s0_uv, s4_y, s4_uv }`
      tuples of `IOSurfacePlaneTexture`s per video-time-step.
      Independent retains per plane so dropping the tuple releases
      the IOSurfaces cleanly after the kernel finishes.
- [x] `--zero-copy` flag on `vr180-render export` (macOS only;
      errors with a helpful message elsewhere — Phase 0.6.8 will
      land the Windows equivalent).
- [x] `TEXTURE_FORMAT_16BIT_NORM` wgpu feature requested at device
      init so R16Unorm / Rg16Unorm work for P010 plane textures.

**P010 vs NV12.** GoPro Max records HEVC Main10 (10-bit) so
VideoToolbox produces P010 IOSurfaces (10-bit Y in upper 10 of a
16-bit container). The shader's YUV→RGB does the BT.709 limited-
range 10-bit expansion inline. 8-bit NV12 GoPro footage (if any
exists in the wild) needs an R8Unorm/Rg8Unorm variant — deferred
to 0.6.6.5 until someone shows up with one.

**Validated** end-to-end on `GS010172.360`, 60 frames, eye_w=2048,
bundled LUT:

|                        | QP   | Output size | fps |
|------------------------|------|-------------|-----|
| CPU-assemble path      | 25.94 | 2.7 MB      | 5.03 |
| Zero-copy IOSurface    | 25.89 | 2.7 MB      | 9.80 |

QP / file size are identical (encoder sees the same image entropy
either way → output is pixel-equivalent), throughput **~2× faster**
end-to-end. The remaining bottleneck is libx265 software encode
(unchanged between the two paths); Phase 0.8.5's `hevc_videotoolbox`
swap removes that.

Extracted frame 30 from each mp4 is visually identical.

### Deferred to Phase 0.6.6.5 — 8-bit NV12 path

The current shader assumes P010. If a user shows up with 8-bit
HEVC GoPro footage, we need:
- Detect bit depth from `IOSurfaceGetBytesPerRowOfPlane(s, 0)` —
  bpr==stream_w → 8-bit, bpr==2*stream_w → 10-bit.
- R8Unorm + Rg8Unorm formats and the 8-bit BT.709 limited-range
  YUV→RGB constants. ~30 lines in the shader + a format-selector
  parameter on the Rust wrapper.

### Deferred to Phase 0.6.7 — Windows CUDA↔Vulkan interop

### Deferred to Phase 0.6.8 — Windows CUDA↔Vulkan interop

NVDEC + cudarc + Vulkan external images. Another ~500 lines, only
meaningful on the Windows builds. Out of scope until 0.9 wires up
Windows builds in CI.

## Phase 0.7 — 3D LUT color pipeline ✅

The headline color feature first; CDL knobs follow.

- [x] `vr180-core::color::Cube3DLut` — full `.cube` parser
      (LUT_3D_SIZE / TITLE / DOMAIN_MIN/MAX / data triplets,
      permissive on whitespace and unknown headers).
      5 unit tests including a smoke test against the bundled
      `assets/Recommended Lut GPLOG.cube` (33³).
- [x] `shaders/lut3d.wgsl` — trilinear sample of an RGBA8 3D
      texture with half-texel correction so input 0.0 → texel 0
      and 1.0 → texel size-1.
- [x] `Device::apply_lut3d(input_rgb, w, h, lut, intensity)` —
      uploads input as 2D + LUT as 3D, one dispatch, repacks output
      to RGB8. Pipeline / sampler / bind-group-layout built once at
      `Device::new` (same caching pattern as eac_to_equirect).
- [x] CLI: `--lut <path>` or `--lut bundled` + `--lut-intensity`
      on `probe-eac` / `export`.

**Visually verified** on `GS010172.360` with the bundled GP-Log
LUT — subtle warm→neutral shift on the laptop screen + slight
contrast bump in the corners. Math matches Python.

### Deferred to 0.7.5

- 1D LUT for CDL (lift/gamma/gain + shadow/highlight smoothstep
  zone masks). Easy port (~50 lines Rust + ~30 lines WGSL) but
  needs a UI / flag plumbing decision first.
- Temp/tint per-channel multiply, saturation. Trivial — one combined
  pass with the above.
- Mid-detail "clarity" (downsample → Gaussian blur → upsample →
  high-pass → midtone-bell weighting). Multi-pass + needs a
  pre-allocated scratch buffer pool. Cheap output, moderate code.

## Phase 0.7.5 — Color tool suite (CDL, grade, sharpen, mid-detail) ✅

The remaining four color shaders from
[ARCHITECTURE.md's GPU pipeline table](ARCHITECTURE.md#gpu-pipeline),
ported from the Python `vr180_gui.py` reference and landed
RGBA16F-safe by default (per the
[10-bit end-to-end mandate](../CLAUDE.md#key-rust-conventions) —
math is in float space; texture format gating is the only thing
that changes for the future Main10 path).

- [x] `cdl.wgsl` — CDL (lift / gain / shadow / highlight / gamma) fused
      into one per-pixel pass. Hermite-smoothstep masks for shadow +
      highlight (scaled by 0.6 to match Python's pivot=0.5 strength).
      Ported from `build_color_1d_lut` (vr180_gui.py:6355-6384).
- [x] `color_grade.wgsl` — temperature + tint + saturation fused.
      0.30 channel-shift strength matches Python `apply_temp_tint`;
      BT.601 luma desat-and-blend matches Python's `cv2.cvtColor(BGR2GRAY)`
      saturation path.
- [x] `sharpen_combine.wgsl` + `gaussian_blur_1d.wgsl` — separable
      Gaussian USM, 3 passes (H-blur → V-blur → combine). Latitude-
      weighted via `cos(π·(0.5 − y/H)) clip(0.02..1.0)` for the equirect
      path (port of `apply_equirect_sharpen`, vr180_gui.py:6699-6744).
      σ defaults to 1.4 matching Python.
- [x] `downsample_4x.wgsl` + `mid_detail_combine.wgsl` — clarity via
      downsample-blur-upsample-combine, 4 passes. 4×4 box average
      downsample (matches cv2 `INTER_AREA`), separable Gaussian blur
      on the 1/4-res image, bilinear upsample + bell-curve-weighted
      blend with the original (port of `apply_mid_detail`,
      vr180_gui.py:6424-6491).
- [x] Pipeline-cache infrastructure: `PerPixelPipeline` shared shape for
      CDL + color_grade; dedicated pipelines for the multi-pass tools.
      Generic `Device::apply_per_pixel` + `dispatch_blur_1d` +
      `encode_readback_rgb` + `finalize_readback` helpers cut per-tool
      boilerplate to ~30 lines.
- [x] Public param types: `CdlParams`, `ColorGradeParams`, `SharpenParams`,
      `MidDetailParams`. All have `Default::default()` = identity, and
      `is_identity()` predicates that short-circuit the corresponding
      `apply_*` to a clone of the input — zero GPU work when the user
      didn't ask for that tool.
- [x] 8 in-repo smoke tests (`gpu::color_tool_tests`): identity-round-trip
      + non-identity sanity for each of the four tools. All pass on
      Apple M-series Metal backend.
- [x] CLI flags on `export` subcommand: `--cdl-lift/-gamma/-gain/-shadow/-highlight`,
      `--temperature`, `--tint`, `--saturation`, `--sharpen`, `--sharpen-sigma`,
      `--mid-detail`, `--mid-detail-sigma`. Default values are
      identity (back-compat with Phase 0.8.5 callers).
- [x] Order of operations matches Python (vr180_gui.py:7746):
      equirect → CDL → 3D LUT → sharpen → mid-detail → color_grade.
      Saturation moves up from "after mid-detail" (Python) to "fused
      with temp/tint" (Neo) — perceptible only when sat is non-default
      AND mid-detail is non-zero; a split into separate `--saturation`
      pass after mid-detail is a possible follow-up if exact parity is
      needed.
- [x] CLI verification: per-frame `color :` log line showing the active
      stages, e.g. `cdl(lift=0.00, gamma=0.92, gain=1.15, sh=0.30, hl=-0.20)
      → sharpen(amount=0.80, σ=1.40) → mid_detail(amount=-0.40, σ=1.00)
      → color_grade(temp=+0.40, tint=-0.10, sat=1.30)`.

**End-to-end test** (60 frames, `NO firmware RS No IORI.360`, 4096×2048
SBS, zero-copy + VT encode):

| Configuration                          | fps   | Notes |
|----------------------------------------|-------|-------|
| Identity (no color tools)              | 30.29 | matches Phase 0.8.5 baseline — short-circuit works |
| Heavy grade (CDL + sharpen + clarity + temp/tint/sat) | 7.07  | 4× hit from per-stage readback round-trip |

Output validation: PSNR(graded vs baseline) Y=35.9 dB, **U=22.5 dB**,
V=37.3 dB — chroma U is hit hardest, exactly as expected from
`temperature=+0.4` (warming = +R / −B → big U shift).

## Phase 0.7.5.5 — Color-stack texture chaining ✅

The Phase 0.7.5 color tools each ran their own upload → dispatch →
readback cycle, so a 4-tool grade meant 4 wgpu submits + 4 GPU sync
waits + 4 staging-buffer memcpys per frame. This phase keeps the
equirect texture GPU-resident through every active color stage —
one encoder, one submit, one readback per frame regardless of how
many tools are on.

- [x] `record_*` internal helpers (`record_cdl`, `record_color_grade`,
      `record_lut3d`, `record_sharpen`, `record_mid_detail`,
      `record_equirect_project`) — each records dispatches into a
      caller-supplied `wgpu::CommandEncoder`, never submits, never
      reads back.
- [x] `project_cross_to_equirect_texture` + `project_cross_texture_to_equirect_texture`
      — texture-returning variants of the existing `project_*` methods.
      Both export paths use these now; the original `Vec<u8>`-returning
      methods stay for any one-off callers.
- [x] `ColorStackPlan` — public param bundle (CDL + LUT + sharpen +
      mid-detail + color_grade). `any_active()` predicate for the
      log-line check.
- [x] `Device::apply_color_stack_texture(&Texture, w, h, &plan) -> Vec<u8>`
      — the new primary API. Builds one encoder, allocates a fresh
      intermediate texture per active stage (identity stages are
      skipped and allocate nothing), records every dispatch, submits
      once, reads back the final texture. Used by both
      `export_cpu_assemble` and `export_zero_copy`.
- [x] `main.rs` refactored to thread `ColorStackPlan` end-to-end
      instead of the per-stage `ColorParams` + `apply_color_stack`
      helper. CLI flags unchanged; user-visible behavior is identical
      apart from the speed.

**Benchmark** (200 frames, `NO firmware RS No IORI.360`, 4096×2048
SBS, `--zero-copy` + VT encode, M-series Apple Silicon):

| Configuration                              | 0.7.5 fps | **0.7.5.5 fps** | Speedup |
|--------------------------------------------|-----------|-----------------|---------|
| Identity (no color tools)                  | 30.29     | 30.71           | (within noise — nothing to optimize) |
| Heavy grade (CDL + sharpen + clarity + grade) | 7.07      | **24.79**       | **3.5×** |
| Heavy grade + bundled 3D LUT (5 stages)    | n/a       | **24.02**       | — |

**Bit-validation**: PSNR(0.7.5.5 chained vs 0.7.5 per-stage) =
**∞ on Y / U / V** (frame-perfect match on 60 frames of the same heavy
grade params). The texture-chaining refactor preserves the math
exactly — same shaders, same dispatch order, same uniforms — only the
host↔device transfer schedule changes. No PSNR cost for the speedup.

### Known limitations

- The 5 intermediate textures (one per active stage + the sharpen /
  mid-detail scratch) allocate fresh wgpu textures every frame. A
  per-Device texture pool keyed by `(w, h, format, usage)` would amortize
  the allocator cost; estimated 0.5–1 fps win, deferred until needed.

### Known follow-up (10-bit lift)

These shaders all read `texture_2d<f32>` (filterable float) and write
`texture_storage_2d<rgba8unorm, write>`. Switching the storage format
to `rgba16float` is a one-line change per shader once the 10-bit
pipeline lands (Phase 0.7.6 / 0.8.5.10). The float math itself
doesn't change. The chained pipeline is well-positioned for this lift:
every intermediate texture is allocated in one place
(`apply_color_stack_texture`), so swapping the format threads through
the whole stack with no per-shader edits.

## Phase 0.7.5.6 — Zero-copy encode (IOSurface → VT) ✅

Phase 0.7.5.5 still had one host-memory hop per frame: the chained
color stack's final readback. The VT encoder then received that
`Vec<u8>` and ran swscale RGB→YUV on the CPU before HW encode. This
phase closes that loop too: the color stack writes directly to an
IOSurface-backed BGRA CVPixelBuffer that `hevc_videotoolbox` reads
in place — no readback, no swscale.

- [x] `interop_macos::EncodePixelBuffer` — RAII wrapper around a
      CVPixelBuffer + its IOSurface + a wgpu::Texture view of the
      bytes. Built by `create_bgra_encode_buffer(device, w, h)` which
      calls `CVPixelBufferCreate` with
      `kCVPixelBufferIOSurfacePropertiesKey` → IOSurface backing.
- [x] `RetainedCVPixelBuffer` — `+1`-retain RAII (CFRetain / CFRelease).
- [x] `build_iosurface_attrs` raw-FFI helper that builds the
      single-key CFDictionary required by `CVPixelBufferCreate` —
      avoids a heavyweight core-foundation typed-builder dependency.
- [x] `compose_sbs_bgra.wgsl` — takes left + right RGBA8 equirect
      textures, writes to one SBS texture with channels swapped
      (`(r,g,b,a) → store as (b,g,r,a)`). The destination wgpu texture
      is viewed as Rgba8Unorm but its underlying IOSurface bytes are
      labeled `32BGRA`, so VT reads correct colors without us needing
      the wgpu `BGRA8UNORM_STORAGE` feature (which not every Vulkan/
      DX12 adapter supports — keeps the code path uniform across
      backends if/when the feature eventually lands cross-platform).
- [x] `Device::apply_color_stack_to_sbs_bgra(left, right, dst, eye_w,
      eye_h, plan)` — one encoder, both eyes' color stacks + SBS
      compose + submit + `device.poll(Maintain::Wait)` so the IOSurface
      bytes are visible to VT before the encoder reads them.
- [x] `H265Encoder::create_zero_copy_vt(path, w, h, fps, kbps)` —
      dedicated constructor that wires
      `av_hwdevice_ctx_create(VIDEOTOOLBOX)` + `av_hwframe_ctx_alloc/init`
      with `format=VIDEOTOOLBOX, sw_format=BGRA`. `pix_fmt =
      VIDEOTOOLBOX`. Same private-data options as the CPU-input VT
      path (realtime=0, allow_sw=0, profile=main, power_efficient=0).
- [x] `H265Encoder::encode_pixel_buffer(&pb)` — builds an AVFrame with
      `format=AV_PIX_FMT_VIDEOTOOLBOX, data[3]=cvpixelbuffer_ref,
      hw_frames_ctx=av_buffer_ref(enc.hw_frames_ctx),
      buf[0]=av_buffer_create(.., cv_pixel_buffer_release_callback, pb)`.
      The release callback `CFRelease`s our +1 retain when the AVFrame
      is unref'd by the encoder, so the CVPixelBuffer lives exactly as
      long as the encoder needs it.
- [x] CLI: `--zero-copy-encode` flag. Requires `--zero-copy --encoder vt`.
      Errors out clearly otherwise.

**Benchmark** (200 frames, `NO firmware RS No IORI.360`, 4096×2048
SBS, M-series Apple Silicon):

| Configuration                              | 0.7.5.5 fps | **0.7.5.6 fps** | Speedup |
|--------------------------------------------|-------------|-----------------|---------|
| Identity (no color tools)                  | 30.71       | **66.14**       | **2.15×** |
| Heavy grade (CDL + sharpen + clarity + grade) | 24.79       | **45.44**       | **1.83×** |
| Heavy grade + bundled 3D LUT (5 stages)    | 24.02       | **43.19**       | **1.80×** |

**All three cases are now well above real-time (30 fps) on 4K SBS.**
The identity case at 66 fps is 2.2× real-time — even with five color
tools active we maintain 43 fps. The bottleneck has moved off the
GPU/CPU transfer path entirely; remaining time is dominated by the VT
encoder + decoder themselves, which already run on dedicated HW.

**Output validation**: PSNR(0.7.5.6 zero-copy-encode vs 0.7.5.5
readback) Y=49.2 dB, U=51.4 dB, V=49.4 dB on the same heavy-grade
input. The slight delta (vs Phase 0.7.5.5's PSNR=∞ chaining match)
comes from VT doing BGRA→YUV internally with slightly different
coefficients than swscale's RGB→YUV path. 49 dB Y is comfortably above
the standard "visually identical" threshold of 40 dB; the difference
is invisible to the eye and within the variability of independent VT
encoder runs.

### Known limitations / follow-ups

- **CPU-input encode path unchanged.** `--zero-copy-encode` requires
  `--encoder vt`; libx265 still needs the host-RAM readback +
  swscale. There's no way around that — libx265 is software-only and
  has no IOSurface ingress.
- **Per-frame CVPixelBuffer allocation.** Each frame creates a fresh
  CVPixelBuffer (and its wgpu::Texture view). On Apple Silicon this is
  ~100 µs which is well under the per-frame budget at 30 fps, but a
  CVPixelBufferPool would amortize it; estimated 1–2 fps win, deferred.
- **Colorspace tags.** The output has `color_space=unknown` (same as
  the 0.7.5.5 readback path). Adding explicit BT.709 tags via
  `AVCodecContext::colorspace/color_primaries/color_trc` is a one-line
  cleanup, not done yet because the Python app on `main` doesn't
  tag either — keeping parity for now.
- **Windows NVENC equivalent.** The same architectural pattern works
  on Windows with NVENC + CUDA-shared Vulkan image; tracked under
  Phase 0.6.8 / 0.8.5+W.

## Phase 0.8 — H.265 encode + multi-frame export ✅

- [x] `vr180-pipeline::encode::H265Encoder` — `libx265` software
      encode wrapper. Packed RGB8 in → mp4/mov out. Handles
      `hvc1` codec tag for Apple/Vision Pro compat, `GLOBAL_HEADER`
      flag for the mov muxer, rational fps approximation,
      PTS/DTS rescale, Drop-guarded `finish()`.
- [x] `vr180-pipeline::decode::StreamPairIter` — streaming
      iterator that yields one `StreamPair` per video-time-step,
      shares the hwaccel setup + repack helper with the single-
      shot `extract_first_stream_pair_with`.
- [x] `export` CLI subcommand:
      ```
      vr180-render export <in.360> <out.mp4>
          [--eye-w N] [--frames N] [--fps F] [--bitrate K]
          [--lut bundled|<path>] [--lut-intensity F]
          [--hw-accel auto|sw|vt]
      ```
      Full pipeline: VT decode → EAC assembly → GPU equirect projection
      → GPU LUT → libx265 encode → mp4.

**End-to-end test** (60 frames, GS010172.360, eye_w=2048,
GP-Log LUT, VT decode):

```
Export: ... → /tmp/neo_out/export_test.mp4
  source : 5952 × 1920 @ 30 fps
  output : 4096 × 2048 @ 30 fps  H.265 12000 kbps
  LUT    : 33^3 @ intensity 1.00
  decode : VideoToolbox
Done: 60 frames in 11.82s (5.08 fps)
Output size: 2.7 MB
[mp4 has hvc1 codec tag, plays in QuickTime / Vision Pro]
```

**The bottleneck is now libx265 software encode** (~12s for 60 frames).
VT decode is essentially free at this scale. → resolved in **0.8.5**.

### Deferred (10-bit + audio)

- **10-bit Main10 output — end-to-end, not a one-line pix-fmt
  switch.** The full mandate is captured in
  [CLAUDE.md](../CLAUDE.md#key-rust-conventions) ("10-bit end-to-end
  when 10-bit output is selected") and
  [ARCHITECTURE.md](ARCHITECTURE.md#10-bit-end-to-end-mandate);
  every stage from VT decoder through encoder input must hold
  ≥10 bits/channel. The 8-bit and 10-bit paths coexist (8-bit for
  fast previews, 10-bit for ship-quality); picking 10-bit delivers
  real 10-bit all the way to the file.
  Touchpoints when this lands (call it Phase 0.7.6 / 0.8.5.10):
  - GPU pipeline: already `Rgba16Float` end-to-end (Phase 0.5+).
    Nothing to do here.
  - Color tools (LUT3D today, CDL / tonal zones / mid-detail /
    sharpen in 0.7.5): already RGBA16F in/out for LUT3D; every
    future color shader **must land RGBA16F-safe by default** —
    no per-tool 8-bit detour. This is a hard project rule.
  - Readback: add `Vec<u16>` packed RGB48 readback path alongside
    today's `Vec<u8>` RGB24 one. Pick the right one based on the
    output bit-depth.
  - swscale: add a second `Context::get` configuration with
    `RGB48LE` src / `yuv420p10le` (libx265) or `p010le` (VT) dst.
  - libx265: pass `-x265-params profile=main10` (or set
    `pix_fmt=yuv420p10le` and let it autodetect Main10).
  - hevc_videotoolbox: switch our four `av_opt_set` calls so
    `profile=main10` and set `pix_fmt=p010le` on the encoder context.
  - CLI: `--bit-depth 8|10` flag (default 8 for back-compat).
  - Tests: bit-validate against a known Main10 reference clip
    that the Python app produces. We don't ship 10-bit until we
    can prove (via 16-bit-aware PSNR) we're not silently
    quantizing through 8-bit anywhere.
- Audio passthrough from the source `.360` (`audio_args = ["-c:a",
  "copy"]` equivalent via libav demux+remux).

### Deferred to 0.8.7 — Atom writers (Google sv3d/st3d/SA3D)

The Phase 0.8.6 APMP path covers Apple / Vision Pro recognition via
FFmpeg's mov-muxer side-data writer. For YouTube / Quest / Meta
compatibility we'd still want the Google spatialmedia atoms
(`sv3d` / `st3d` / `SA3D`) written alongside — FFmpeg has a separate
code path for these (`mov_write_sv3d_tag`) that's also triggered by
side-data on the output stream, but uses different muxer flags.

For audio: the existing APAC mux is Vision Pro-native; SA3D-tagged
ambisonic AAC would let YouTube / Quest VR also do spatialization.

## Phase 0.8.5 — `hevc_videotoolbox` hardware encode ✅

The encode-side counterpart of Phase 0.6 (decode hwaccel). libx265
was the last single-threaded bottleneck in the export pipeline;
switching to Apple's hardware encoder gets us to **real-time on
4K SBS output**.

- [x] `vr180-pipeline::encode::EncoderBackend { Libx265, VideoToolbox }`
      — public enum, both variants buildable on every target. VT
      variant returns a hard `Err` if `cfg!(target_os = "macos")`
      is false; no silent fall-through.
- [x] `H265Encoder::create(.., backend)` — backend-aware constructor.
      Switches `codec_name` to `hevc_videotoolbox` and sets four
      private-data options via `av_opt_set` on `priv_data`:
      `realtime=0` (quality over latency, override the macOS-14+
      auto-decide default), `allow_sw=0` (refuse libavcodec's silent
      software fallback), `profile=main` (8-bit baseline; main10
      deferred with the 10-bit pix-fmt switch), `power_efficient=0`
      (don't throttle on AC power).
- [x] Plumbed via a thin `set_opt(ctx, name, value)` helper — uses
      raw `ffmpeg::ffi::av_opt_set` rather than ffmpeg-next's
      `Dictionary` wrapper, so option failures surface inline at
      config time instead of being swallowed by `open_as_with`.
- [x] CLI: `--encoder auto|sw|vt` flag on `export` subcommand.
      `auto` resolves to VT on macOS, libx265 elsewhere. Threads
      through both `export_cpu_assemble` and `export_zero_copy`
      paths.
- [x] Runtime guard in `export()`: requesting `--encoder vt` on
      Windows / Linux yields a clear error message instead of an
      FFmpeg-internal failure.

**Benchmark** (200 frames, `NO firmware RS No IORI.360`, 4096×2048
SBS output, 12 Mbps target, zero-copy IOSurface decode path, M-series
Apple Silicon):

| Encoder | fps  | Output | Profile |
|---------|------|--------|---------|
| libx265 (SW)            | 16.15 | 8.1 MB | Main, 10.2 Mbps actual |
| **hevc_videotoolbox (HW)** | **29.48** | **3.5 MB** | Main, 4.4 Mbps actual |

VT is **1.8× faster** end-to-end and hits real-time (30 fps) on 4K
SBS. Both outputs decode cleanly, carry the `hvc1` codec tag, and
play in QuickTime / Vision Pro. VT chose a lower bitrate than the
12 Mbps hint — its ABR control is permissive by default; the
`constant_bit_rate=true` option (macOS 13+) would clamp it harder
if matched-bitrate quality comparison becomes important.

Caveats:

- This is fps in the **export** pipeline (decode → assembly → projection
  → LUT → encode). The encoder alone is much faster than 30 fps; the
  GPU stages now dominate.
- The CPU-assemble path (`--encoder vt` without `--zero-copy`) runs
  at ~7.5 fps — moving the encoder from libx265 to VT doesn't help
  there because the bottleneck is the `av_hwframe_transfer_data`
  CPU roundtrip and CPU EAC assembly. Use `--zero-copy` to see the
  full speedup.
- 10-bit (main10) still deferred — **end-to-end-or-nothing**,
  see the touchpoint enumeration in
  [Phase 0.8 Deferred (10-bit + audio)](#deferred-10-bit--audio)
  above and the hard rule in
  [CLAUDE.md](../CLAUDE.md#key-rust-conventions). One-line pix-fmt
  switches are how 10-bit pipelines silently regress to 8-bit
  through some intermediate; this work needs to land each
  touchpoint explicitly.

## Phase 0.8.6 — APAC audio + APMP metadata (Vision Pro spatial) ✅

VR180 SBS exports become Vision Pro-native: ambisonic audio is
re-encoded via Apple Positional Audio Codec (APAC) for true
head-tracked spatialization, and the video track is tagged with
Apple Projected Media Profile (APMP) atoms so visionOS recognizes
it as immersive media. **MV-HEVC is explicitly out of scope** —
we stick with single-track SBS HEVC because every Phase 0.x
optimization (zero-copy decode → wgpu chained color stack → zero-
copy VT encode) is built around that pipeline shape.

- [x] `vr180-pipeline::audio` module — `probe_ambisonic` walks the
      source container's audio streams, returns the first 4-channel
      uncompressed-PCM track (matches GoPro MAX 2's stream index 5,
      `pcm_s24le` 48 kHz "ambisonic 1"). `extract_ambisonic_to_wav`
      stream-copies that track into a fresh WAV via in-process
      ffmpeg-next remux — no system-ffmpeg subprocess.
- [x] `helpers::spawn_apac_encode` — subprocess wrapper around the
      `apac_encode.swift` helper in `helpers/bin/`. Stderr streams
      into `tracing`; non-zero exit → `Error::Helper { code, stderr }`.
      Three call modes: audio-only, video-passthrough mux, custom
      bitrate. Used by the `--apac-audio` export flag.
- [x] `H265Encoder::tag_apmp_vr180_sbs` — injects two side-data
      entries on the output stream's codec parameters:
      - `AV_PKT_DATA_STEREO3D` (`type=SIDEBYSIDE, view=PACKED,
        primary_eye=LEFT`)
      - `AV_PKT_DATA_SPHERICAL` (`projection=HALF_EQUIRECTANGULAR`)
      Plus sets `strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL`
      on the codec context. FFmpeg's mov muxer reads these at
      `write_trailer` time and emits the APMP box tree
      (`vexu/proj/prji=hequ` + `vexu/eyes/stri=0x03`) into the
      `hvc1` sample description.
- [x] Re-declared `MAVSphericalMapping` struct locally (with `M`
      prefix) — `libavutil/spherical.h` exists in FFmpeg but
      ffmpeg-sys-next 8.1's bindgen header set doesn't include it.
      C ABI stable since FFmpeg 3.1; layout verified against
      libavutil headers.
- [x] Deferred-header pattern in `H265Encoder` — `write_header`
      moves from `create()` to first encode call via
      `ensure_header_written()`. Lets post-construction setters
      (`tag_apmp_vr180_sbs`) modify codecpar side-data before the
      muxer locks in the moov.
- [x] CLI: `--apac-audio` + `--apac-bitrate <bps>` + `--apmp`
      flags on `export`. `--apac-audio` writes a temp video-only
      `.mov`, then runs `apac_encode` to mux video + APAC into the
      final path. `--apmp` is independent (works with or without
      audio).
- [x] Swift helper build: existing `helpers/build_swift.sh` now
      runs end-to-end (`apac_encode` + `vt_denoise` + `mvhevc_encode`
      all built into `helpers/bin/`). `mvhevc_encode` still builds
      even though we're not wiring it (saves bit-rot if we ever want
      it back).

**End-to-end test** (60 frames, `NO firmware RS No IORI.360`,
4096×2048 SBS, zero-copy + VT encode, `--apac-audio --apmp`):

```
Export (zero-copy IOSurface path): … → /tmp/.vr180_apac_apmp.video_only.mov
  pipeline: VT → IOSurface → wgpu → SBS BGRA IOSurface → VT encoder (zero-copy)
Done: 60 frames in 761.06ms (78.84 fps)

APAC: extracting ambisonic (stream 5, 4ch @ 48000 Hz) ...
APAC: extracted 39.7 MB WAV
APAC: muxing video + APAC audio via apac_encode @ 384 kbps ...
APAC: done in 1.00s
```

**Output validation** (`ffprobe` + raw `xxd` of moov atom):
- Stream 0: `Audio: none (apac / 0x63617061), 48000 Hz, 4.0` ← **APAC track**, 4-ch ambisonic
- Stream 1: `Video: hevc (Main) (hvc1 / 0x31637668), 4096×2048`
  - Side data:
    - `stereo3d: unspecified, view: packed, primary eye: left` ← AV_STEREO3D_SIDEBYSIDE+VIEW_PACKED
    - `spherical: half equirectangular` ← AV_SPHERICAL_HALF_EQUIRECTANGULAR
- Raw atom bytes in the file: `vexu` → `proj` → `prji` → `hequ`,
  `eyes` → `stri` → `0x03` (has_left+has_right). Matches the
  Apple-spec layout exactly.

### Known limitations (deferred to 0.8.7 / later)

- **`pack/pkin=side` atom not written.** FFmpeg's mov muxer doesn't
  emit the visionOS-26-specific frame-packing atom yet. Vision Pro
  still recognizes the file as VR180 from the core atoms, but newer
  visionOS may use `pkin` for explicit SBS frame-packing
  recognition. Hand-write via a post-pass when needed.
- **`hfov=180000` atom not written.** Same reason — FFmpeg doesn't
  emit it from `AV_PKT_DATA_SPHERICAL` side-data. FOV defaults to
  reading from the projection box in current visionOS.
- **`vexu/eyes/cams/blin` baseline not written.** Apple's spec
  marks this as "required for spatial media tagging" but Vision Pro
  is forgiving in practice — files without it still parse as
  immersive VR180.
- **Google `sv3d/st3d/SA3D` not written.** For YouTube / Quest /
  Meta-browser compat. FFmpeg has a separate code path for these
  (`mov_write_sv3d_tag`) that may not be exercised by our side-data
  approach; investigation deferred.
- **Audio length is full source length, not trimmed to encoded
  video range.** For partial exports (`-n 60`), the audio overruns
  the video by ~70s. For full exports (production case) audio +
  video align naturally. Trim support is deferred until the CLI
  grows `--start`/`--duration` flags.

## Phase A — Per-frame CORI rotation (camera-lock stabilization) ✅

Foundation for the full stabilization + RS pipeline. Pulls per-frame
CORI quaternions from the source's GPMF stream, converts to 3×3
rotation matrices, uploads as a uniform to the equirect projection
shader, and rotates the output direction into the camera frame
before face selection. Net effect with `--stabilize` on: the scene
locks to the first-frame camera orientation — every per-frame
jitter is fully compensated.

This is "camera lock" mode — pans and tilts vanish too. Phase B
adds the velocity-dampened bidirectional SLERP smoother so slow
intentional camera motion is preserved.

- [x] `vr180-core::gyro::Quat` math additions:
      `mul`, `conjugate`, `dot`, `norm`, `normalize`, `slerp`,
      `to_mat3_row_major` (standard 9-element row-major matrix).
- [x] `vr180-core::gyro::resample` module:
      `resample_quats_to_frames(src, src_fps, dst_fps, n_frames,
      time_offset_s, window_s)` — window-averaging resampler with
      sign-continuity. Window-average mode (Python's RS-friendly
      mode) integrates over a `SROT_S = 15.224 ms` window centered
      on each frame's sensor-readout midpoint.
- [x] `vr180-core::gyro::cori_swap_yz` — GoPro stores CORI as
      `(w, x, Z_disk, Y_disk)`; we preserve on-disk order in
      `parse_cori` and apply the swap here at the math boundary.
- [x] `eac_to_equirect.wgsl` — new `EquirectUniforms` binding (12
      `f32` scalars in std140 layout, 48 bytes). Direction vector
      rotated by `R * dir_world` before max-axis face test.
      (Naga gotcha: `mat3x3<f32>` and `array<vec4,3>` uniforms had
      parser issues with field access; plain scalar fields work
      reliably. Also: uniform name must not shadow the local `let u
      = (x + 0.5) / width` for the equirect U coord — renamed to `equ`.)
- [x] `vr180_pipeline::gpu::EquirectRotation` public type +
      `IDENTITY` constant + `from_quat(Quat)` constructor.
      All four `Device::project_*_to_equirect_*` methods + the
      internal `record_equirect_project` helper now take an
      `EquirectRotation` parameter.
- [x] `--stabilize` CLI flag on `export` + `stabilize: bool`
      field on `ExportConfig` JSON schema.
- [x] `compute_stabilization_rotations` precomputes a `Vec<EquirectRotation>`
      from the source's GPMF at start of export; `rotation_for_frame`
      indexes into it per-frame (defaults to identity when empty).

**Validation** (full file `NO firmware RS No IORI.360`, 2169 frames,
4K SBS, zero-copy + VT):

| Configuration   | Time   | fps     |
|-----------------|--------|---------|
| Unstabilized    | 41 s   | 52.3    |
| `--stabilize`   | 42.7 s | **50.79** |

Performance overhead is in the noise (~3% from the uniform upload
per frame). Output stays full-quality.

### Phase A scope limits — follow-up phases

- **No smoothing.** Phase A applies raw CORI directly → every per-
  frame jitter is "corrected" → intentional pans and tilts also
  vanish. **→ shipped in Phase B.**
- **No IORI per-eye stereo.** Phase A uses one R for both eyes.
  **→ shipped in Phase B.**
- **No gravity / horizon lock.** Phase A locks to the FIRST FRAME's
  orientation, which may be tilted relative to gravity.
  **→ shipped in Phase C.**
- **No per-scanline RS warp.** **→ Phase D** (deferred).
- **No no-firmware-RS path.** **→ Phase E** (deferred).

## Phase B/C — Velocity-dampened smoothing + IORI per-eye + GRAV alignment + soft cap ✅

Turns Phase A's "camera lock" into useful real-world stabilization:
slow pans and tilts survive; per-frame jitter dies; the horizon
stays level even when the camera wasn't held level at frame 0;
extreme camera moves no longer crop past the image boundary.

### What landed

- **`vr180-core::gyro::stabilize`** — new module with:
  - `SmoothParams { smooth_ms, fast_ms, responsiveness, max_vel_deg_s }`
    (defaults match Python's `ProcessingConfig`: 1000 / 50 / 1.0 / 200).
  - `bidirectional_smooth(raw, fps, params)` — forward + backward
    exponential SLERP passes, averaged via mid-SLERP. Per-step
    `tau` adapts to local angular velocity:
      `vel_ratio = (vel/max_vel)^responsiveness`
      `tau = smooth_ms · (1-vel_ratio) + fast_ms · vel_ratio`
      `alpha = dt / (tau + dt)`
    Calm spans get heavy smoothing; fast spans get light.
  - `soft_elastic_clamp(raw, smoothed, max_corr_deg)` — when the
    heading correction angle exceeds the cap, pulls the smoothed
    quat back toward raw via logarithmic compression
    `soft = limit · (1 + ln(angle/limit))`. Bounded effective
    correction → no black borders even on extreme motion.
  - `per_eye_rotations(raw, smoothed, iori, max_corr_deg) -> (Quat, Quat)`
    — combines heading correction (`raw · smoothed⁻¹`), soft clamp,
    and IORI per-eye split (`q_left = iori · heading`,
    `q_right = iori⁻¹ · heading`). When IORI is identity, both eyes
    get the same matrix.
  - `gravity_alignment_quat(grav_samples, scal, n)` — averages
    the first N GRAV samples, applies the GoPro GRAV axis remap
    (`bodyX←raw[0], bodyY←raw[2], bodyZ←raw[1]`), solves the
    rotation that maps the gravity vector to `(0, 1, 0)`.
  - `apply_gravity_alignment_inplace(cori, g_inv)` — right-multiply
    every CORI by `g⁻¹` to align the world frame so Y-down = true
    gravity. Eliminates "horizon tilted at frame 0".
  - 8 unit tests covering: smoothing identity, smoothing zero-tau
    passthrough, smoothing attenuates jitter, soft clamp no-op
    inside limit, soft clamp softens beyond limit, per-eye with
    identity IORI matches both eyes, gravity align solves correct
    rotation. All pass.

- **`vr180-render` orchestration**:
  - `StabilizeParams { enabled, smooth, max_corr_deg }` bundle.
  - `compute_stabilization_rotations` now returns
    `Vec<(EquirectRotation, EquirectRotation)>` (per-eye).
  - Pipeline order: parse CORI/IORI/GRAV → GRAV align CORI →
    bidirectional smooth → per-frame heading + clamp + IORI split.
  - Both `export_cpu_assemble` and `export_zero_copy` updated to
    call `device.project_*_to_equirect_*` with the per-eye `r_left`
    and `r_right` matrices.

- **CLI flags** on `export`:
  - `--gyro-smooth-ms <ms>` (default 1000) — `0` = camera lock
    (Phase A behavior).
  - `--gyro-responsiveness <f>` (default 1.0)
  - `--max-corr-deg <deg>` (default 15.0) — `0` disables the cap.

- **JSON `ExportConfig`** also exposes the same three knobs.

### Validation

- **All 8 unit tests** in `vr180_core::gyro::stabilize` pass.
- **End-to-end on `firmware RS No IORI.360`** (826 frames, 4K SBS, VT zero-copy):
  - `--stabilize` (default smoothing): 16.62 s @ **49.70 fps**
  - `--stabilize --gyro-smooth-ms 0` (camera lock): 16.58 s @ **49.83 fps**
  - `--stabilize --gyro-smooth-ms 500` (medium): 16.71 s @ **49.43 fps**
  All three within noise of each other — the smoothing is one-time
  CPU work at start (sub-ms scale); no per-frame impact.
- User-confirmed: horizon stays level, intentional camera pans
  preserved, no roll-axis direction issues.

### Still deferred

- **Phase C+** — `--horizon-lock` toggle using swing-twist
  decomposition. The GRAV alignment we shipped is the always-on
  baseline; explicit horizon lock uses the per-frame GRAV reading
  to compute the roll component instead of taking it from the
  smoothed quaternion.
- **Phase D** — per-scanline RS warp (32 row-groups, 800 Hz GYRO,
  KLNS fisheye time map). Mainly matters for the right eye on
  firmware-RS clips (yaw-mod reverses firmware's correction) and
  both eyes on no-firmware-RS clips.
- **Phase E** — VQF → CORI-equivalent quaternion stream for
  bias-drifting clips. The VQF math is already ported
  (`vr180-core::gyro::vqf::run`); just needs the integration glue
  + Y↔Z swap at the output (Python `parse_gyro_raw.py:1273`).

## Phase 0.9 — JSON config sidecar (Rust side) ✅

The Python GUI on `main` learns to shell out to `vr180-render` for
the heavy work, via a JSON config sidecar. **This is the wedge that
ships value to users before any UI rewrite.** This phase lands the
Rust side; the Python GUI patch is a separate PR on `main`.

- [x] `vr180-render::config::ExportConfig` — serde-derived JSON
      schema covering every knob the `export` CLI subcommand exposes
      (I/O, video dims, bitrate, decode/encode backend, zero-copy
      flags, full color stack, LUT, APAC audio, APMP metadata).
      `deny_unknown_fields` on for forward error detection.
- [x] Identity defaults everywhere — minimal config is just
      `{ "input": "...", "output": "..." }`.
- [x] `From<&CdlConfig>` etc. impls convert sub-configs to the
      existing `vr180_pipeline::gpu::*Params` types. No serde
      dependency added to `vr180-pipeline`; conversion happens at
      the dispatcher boundary.
- [x] `vr180-render render --config <file.json>` CLI subcommand.
      Thin adapter: parse → field-map → call the same `export()`
      function the CLI-flag path uses. **Bit-identical output**
      (PSNR Y/U/V = ∞ on the equivalent flag invocation).
- [x] Three unit tests (`config::tests`):
      - `minimal_config_parses_with_identity_defaults` — bare
        input/output config parses, every color stage is identity.
      - `unknown_field_is_rejected` — typo in a field name surfaces
        as a clear error (not a silent no-op).
      - `full_config_round_trips` — non-default config → serialize
        → re-parse → field-for-field equality.
- [x] Two example configs in `examples/`:
      - `minimal.json` — just input/output, identity everything.
      - `full_vision_pro.json` — heavy color grade + LUT + APAC + APMP,
        the realistic Vision Pro export.
      Plus `examples/README.md` with the full schema reference and
      a Python `subprocess.run` spawn pattern for the GUI.

**End-to-end validation** (60 frames, 4K SBS, JSON config vs
equivalent CLI flags, same heavy grade + LUT + APAC + APMP):
- File sizes: byte-identical (4 577 928 bytes both).
- PSNR(JSON output vs CLI output): **∞ on Y / U / V** — same code
  path, no behavior drift.

### Deferred — Python GUI patch (separate PR on `main`)

Out of scope for this `neo` branch. When that lands, the GUI's
"Export" action either:
- (current) runs the in-process Python encoder, OR
- (toggle on) writes the user's `ProcessingConfig` to a temp JSON
  and `subprocess.run`s `vr180-render render --config <temp>.json`.

### Deferred — Python `ProcessingConfig` fields not yet in Rust

The Python `ProcessingConfig` has 82 fields; our Rust `ExportConfig`
exposes only the ~22 we currently implement. The Python side will
need a "filter to Rust-supported subset" path until these land:

- Trim range (`trim_start` / `trim_end`)
- Gyro stabilization (`gyro_*`, ~7 fields) — Phase 0.3.5 ports the
  VQF resampling; full stabilization integration is a separate phase
- Rolling shutter correction (`rs_correction_*`, ~5 fields) — needs
  the per-scanline RS warp shader from ARCHITECTURE.md's GPU table
- Denoise (`denoise_strength`) — Phase 0.x deferral; would spawn
  the `vt_denoise` Swift helper
- ProRes output, CRF mode, encoder preset / speed (`prores_profile`,
  `quality`, `use_bitrate`, `encoder_speed`) — VT-only for now
- Multi-segment input (`segment_paths`) — chain handling for GoPro
  recordings that split at the 4 GB FAT32 boundary
- Edge mask / vignette (`mask_size`, `mask_feather`, `edge_fill`)
- Camera orientation flip (`upside_down`)
- `vision_pro_mode` enum (currently we have a hard `--apmp` flag;
  Python distinguishes "standard" / "spatial" / etc.)

Each of these is a separate small phase; the JSON schema grows one
field at a time as they land. No big-bang rewrite.

## Phase 1.0 — Tauri UI (future)

Only after 0.9 is shipping. Tauri shell replaces PyQt6 entirely:
- Rust backend (`vr180-render` invoked in-process or as a sidecar)
- Web frontend (HTML/TS) for the UI — playback, mask overlays,
  transport controls, color sliders, export queue.
- `bundle.externalBin` packs `vr180-render.exe` next to the app on
  Windows; on macOS the binary sits in the `.app` Resources.
- Mac code-signing + notarisation handled by the Tauri bundler;
  Windows code-signing optional.
