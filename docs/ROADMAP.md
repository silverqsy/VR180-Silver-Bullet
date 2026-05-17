# Roadmap

Mirrors SLRStudioNeo's phased shipping plan. Each phase ends with
a tag and a `cargo run -p vr180-render -- …` demo that exercises
the new capability end-to-end.

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

## Phase 0.6 — Hardware decode + IOSurface↔Metal interop ⏸ deferred

ffmpeg-next 8.1 does not expose hwaccel APIs at the safe-Rust level.
Doing this properly requires:

1. **Raw `ffmpeg_sys_next` FFI** to wire `av_hwdevice_ctx_create` +
   `av_hwframe_transfer_data` into the decoder context. ~200 lines of
   `unsafe` Rust + careful RAII wrappers.
2. **IOSurface ↔ MTLTexture interop** on macOS (`RetainedIOSurface`,
   `metal::Texture::new_from_iosurface`, wgpu-hal Metal escape).
   ~700 lines — SLRStudioNeo has the exact code we'd adapt.
3. **CUDA ↔ Vulkan interop** on Windows (`cudarc` driver API attached
   to `AVCUDADeviceContext`, external Vulkan images via wgpu-hal).
   Another ~500 lines.

That's a multi-session deliverable that needs its own focused pass.
The Phase 0.5 pipeline runs at ~4 fps (decode-bound, sw decode of
two HEVC 8K streams in serial). Once the interop work lands the
expected jump is to ~30 fps on M2 Max (decode drops to ~20 ms,
upload becomes zero-copy).

**Deferred** until after Phases 0.7 (color) and 0.8 (encode) so we
have a complete sw-decode→GPU→encode pipeline working end-to-end
first. Interop is a perf optimization on top of that.

## Phase 0.7 — Color pipeline + LUT + tonal zones

Port the entire pre-LUT / post-LUT color pipeline as one fused
wgpu pass. Order matters — port the validated order from the
Python `apply_export_post`.

- [ ] `tonal_zones.wgsl` (shadow/highlight smoothstep masks)
- [ ] `color_grade.wgsl` (lift / gamma / gain / sat / temp / tint)
- [ ] `lut3d.wgsl` (trilinear .cube LUT)
- [ ] `mid_detail.wgsl` (downsample-blur-upsample clarity)
- [ ] CLI: `--lut`, `--shadow`, `--highlight`, `--temp`, `--tint`,
      `--mid-detail` flags

## Phase 0.8 — Encode + Swift helper spawn

ffmpeg-next encode (h265, prores) + spawning `mvhevc_encode` /
`apac_encode` / `vt_denoise` from Rust.

- [ ] `vr180-pipeline::encode::Encoder` (h265 / prores)
- [ ] `vr180-pipeline::helpers::spawn_mvhevc_encode`
- [ ] `vr180-pipeline::helpers::spawn_apac_encode`
- [ ] `vr180-pipeline::helpers::spawn_vt_denoise`
- [ ] `vr180-core::atoms` — sv3d / st3d / SA3D writers (replaces
      Google's `spatialmedia` Python package)

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

Only after 0.9 is shipping. Tauri shell replaces PyQt6 entirely.
SLRStudioNeo's `apps/mosaic-ui/` is the template.
