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

## Phase 0.4 — Decode + EAC assembly (CPU baseline)

ffmpeg-next decode (software path first), `Rgba16Float` frame in
host memory, EAC cross assembly from s0+s4 streams. No GPU yet.

- [ ] `vr180-pipeline::decode::Decoder` (ffmpeg-next, sw decode)
- [ ] `vr180-core::eac::Dims::from_probe(&format)` — runtime sizes,
      no hardcoded 5952×1920 anywhere
- [ ] `vr180-core::eac::assemble_lens_a` / `assemble_lens_b`
- [ ] CLI: `vr180-render export --equirect --cpu <in.360> <out.png>`
      writes a single frame for sanity check

## Phase 0.5 — wgpu device + first kernel

wgpu adapter setup (Metal / Vulkan / DX12 auto-select), one WGSL
kernel running end-to-end (`cross_remap.wgsl`), CPU → GPU upload
+ GPU → CPU readback. Slow but proves the wiring.

- [ ] `vr180-pipeline::gpu::Device` (wgpu instance, adapter, queue)
- [ ] `cross_remap.wgsl` ported from Python `_nb_dirs_to_cross_maps`
- [ ] CLI: `vr180-render export --equirect <in.360> <out.png>` uses GPU
- [ ] Output pixel-matches the Phase 0.4 CPU baseline (±1 LSB)

## Phase 0.6 — Hardware decode + IOSurface↔Metal interop (macOS)

Step 3 from SLRStudioNeo's MAC-PORT.md: zero-copy VT → wgpu.

- [ ] `vr180-pipeline::interop_macos::IoSurfaceTexture`
- [ ] `--hw-decode auto` default on macOS
- [ ] CLI export of 8K throughput target ≥ 30 fps on M2 Max

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
