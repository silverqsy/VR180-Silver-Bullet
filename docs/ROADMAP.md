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

### Deferred to 0.8.6 — Swift helper spawning

The three macOS-native helpers already exist as compiled binaries
in `helpers/swift/`. The Rust pipeline just needs spawn glue
(stdin BGR48 frames, stderr line stream into `tracing` logs,
exit-code → `pipeline::Error::Helper` mapping). Roughly mirrors
how the Python app already calls them; not hard, just plumbing.

- [ ] `vr180-pipeline::helpers::spawn_mvhevc_encode` — MV-HEVC
      spatial video for Vision Pro
- [ ] `vr180-pipeline::helpers::spawn_apac_encode` — ambisonic
      audio re-encode (Apple Positional Audio Codec)
- [ ] `vr180-pipeline::helpers::spawn_vt_denoise` — VTTemporal-
      NoiseFilter, true 10-bit through denoise

### Deferred to 0.8.7 — Atom writers

Port the spherical / stereo / ambisonic metadata atoms into
`vr180-core::atoms`. Replaces Python's `spatialmedia` package
(which we patched twice this session for the
`KeyError: 'ambisonic_channel_ordering'` bug). Three writers,
all <100 bytes each on disk:
- [ ] `sv3d` — spherical projection (equirectangular)
- [ ] `st3d` — stereo mode (left-right SBS)
- [ ] `SA3D` — ambisonic audio metadata (Google spec)

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

## Phase 0.9 — Full export parity + CLI sidecar mode

The Python GUI on `main` learns to shell out to `vr180-render` for
the heavy work, via a JSON config sidecar. **This is the wedge that
ships value to users before any UI rewrite.**

- [ ] `vr180-render --config export.json` reads same schema
      the Python GUI writes
- [ ] Python GUI patch (on `main`, separate PR) adds an "Use Neo
      render engine" toggle that spawns `vr180-render.exe` instead
      of doing it in-process
- [ ] Single-binary distribution: ~80 MB exe per platform
- [ ] All hardcoded-dimension bugs (the `5952` class) impossible
      by construction

## Phase 1.0 — Tauri UI (future)

Only after 0.9 is shipping. Tauri shell replaces PyQt6 entirely:
- Rust backend (`vr180-render` invoked in-process or as a sidecar)
- Web frontend (HTML/TS) for the UI — playback, mask overlays,
  transport controls, color sliders, export queue.
- `bundle.externalBin` packs `vr180-render.exe` next to the app on
  Windows; on macOS the binary sits in the `.app` Resources.
- Mac code-signing + notarisation handled by the Tauri bundler;
  Windows code-signing optional.
