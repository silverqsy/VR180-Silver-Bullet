# CLAUDE.md — orientation for Claude Code sessions

Auto-loaded at session start. Read this first; deeper detail is in the
`docs/` pointers at the bottom.

## ⚡ Status: **Unreleased on `main`: X6 automatic per-frame eye-exposure matching (2026-09-27, item 15) + GoPro per-frame exposure timing from SHUT (2026-10-07, item 16) + Kandao QooCam 3 Ultra video/stills/audio (2026-10-14, item 23; Windows arms verified + matrix/gate fixes 2026-10-17, item 24) + Blackmagic RAW rebuilt on the modern path (2026-10-19, item 25).** **2.5.2 RELEASED 2026-10-10 on BOTH platforms (item 21).** Previous: 2.5.1 — macOS RELEASED 2026-09-25 (tag v2.5.1 @ 7e345ee: DMG + mac app.zip + latest.json live, feed serves 2.5.1); **Windows RELEASED 2026-09-25, RESPUN same day with the version-free product name** (setup.exe + portable zip + MERGED latest.json live on v2.5.1; the installer, shortcuts, Add/Remove entry and exe now say "VR180 Silver Bullet" — AppId and the settings dir deliberately unchanged, `[InstallDelete]` clears the old-named shortcuts on upgrade; live manifest byte-identical to local, installer minisig verified against the updater.rs pubkey; feed serves BOTH platforms). Previous: 2.5.0 RELEASED 2026-09-13, macOS respun 2026-09-14 (adds 3D display output, Snapshot, generic-SBS input toggle, Flat 3D bitrate defaults; the 2.5.0 Windows respin was never done — superseded by 2.5.1) — reframed mode, file-derived stabilization timing, Insta360 X6

VR180 Silver Bullet **2.0** is feature-complete and building/running on both
**macOS (Apple Silicon)** and **Windows (NVIDIA)**. The Windows GPU-resident
NVENC path and the macOS feature batch below were verified on both platforms
(DX12/Vulkan + NVENC on an RTX 4090; Metal on Apple Silicon). See
[CHANGELOG.md](CHANGELOG.md) for the user-facing 2.0 summary.

**2.5.0 bumps from 2.1.0** (auto-update, released) with the reframed output
mode, stabilization timing derived from the file (no IMU phase slider),
Insta360 X6 support, OSMO 360 II support and Auto align — see items 7–9
below and [CHANGELOG.md](CHANGELOG.md). **2.1.0: seamless auto-update** — see
[docs/AUTO-UPDATE.md](docs/AUTO-UPDATE.md) for the full design + release
recipe. `crates/vr180-gui/src/updater.rs`: minisign-verified updates from
GitHub Releases (`latest.json` at the stable `releases/latest/download/`
URL), toolbar badge + popover UX, whole-`.app` swap + relaunch on macOS
(E2E-verified with real signing), silent Inno install on Windows.

**Windows session TODO to catch up (2.1.0):**
1. Pull; `cargo build --release -p vr180-gui` (new deps: ureq,
   minisign-verify — pure Rust, no system libs).
2. Copy the updater signing key from the mac box to
   `C:\Users\<user>\.vr180-updater\vr180-updater.key` (private channel;
   NEVER commit) and install the `minisign` CLI (scoop or release binary).
3. Rebuild the installer with the updated `installer/windows.iss`
   (AppVer 2.5.0 + the new silent-relaunch `[Run]` entry +
   `CloseApplications`).
4. VERIFY the Windows update path (built here, not yet run on Windows):
   serve a `latest.json` locally (`VR180_UPDATE_URL` override), click
   Install & Restart → the app must exit, the installer run silently
   (per-user, no UAC), and the app relaunch. Watch for: the spawned
   `-setup.exe` inheriting a lock on itself (it runs from the staging dir
   under `%APPDATA%\VR180SilverBullet2.0\updates\` — fine), and the
   `[Run] Check: WizardSilent` entry actually firing.
5. At release: `node scripts/make-latest-json.mjs --version 2.5.0
   --notes-file <notes>` on BOTH machines (mac first, copy its
   `release-staging/latest.json` over so the entries merge), then upload
   both artifacts + `latest.json` to the `v2.5.0` release.
6. **BeyondVR Hack** (final-stage per-eye output scale, ⚙ Format window)
   landed after the seam fixes — pull + rebuild + re-upload the Windows
   artifacts so both platforms carry it (mac 2.1.0 artifacts already
   respun with it). No Windows-specific code: the stage rides the shared
   color stack (`ColorStackPlan.eye_scale`), incl. the NVENC/d3d11 arms.
7. **Stabilization timing is derived from the file — no phase constants,
   no slider** (`dji_imu::frame_sample_time_s`, 2026-09-09). Each frame's
   pose is sampled at the centre row's mid-exposure, `readout/2 +
   (video_ts − block_ts) − shutter/2` after its motion block starts: the
   readout from the file's line time (`DjiOsvImu::readout_ms`, header
   `[1.13.1]`, II `[1.11.1]`, × rows; `dji_osmo_readout_ms_for_fps` is only
   the fallback), the shutter from the per-frame exposure record
   (`exposure_ms`, `[3.2.4.1]` as a fraction of a second) and the two
   timestamps (`frame_ts_offset_ms`). Insta360 timelines are synthesized
   on the measured content time (`SampleAnchor::BlockMid`). Measured on
   the stabilized output of five clips (1/500 s → ≈9 ms, 1/120 s → ≈5.5 ms,
   1/40 s → ≈−3 ms) and consistent with DJI Studio's output; the former
   8.5 ms and 5.5 ms @ 25 fps rules were this formula on particular clips.
   The "IMU phase" slider and `Settings::dji_imu_phase_ms` are gone — **do
   not reintroduce tuned phase numbers**; a new camera or mode gets its
   timing from its metadata. Diagnostics (env): `VR180_IMU_PHASE_OFFSET_MS`,
   `VR180_DJI_EXPOSURE_COMP=0`, `VR180_DJI_RS_SHIFT_MS`. Pipeline-level,
   shared code — pull + rebuild; no Windows-specific work.
8. **Insta360 X6 `.insv` support** (`SourceKind::Insta360Insv`; the official
   X6 10-bit I-Log→Rec.709 LUT v2 is bundled as `BUILTIN_X6_LUT` and
   auto-applied on load like the DJI/GoPro builtins — every X6 clip seen so
   far is I-Log, no profile flag found in the metadata): trailer
   parser (`vr180-fisheye/src/insta360.rs`), raw-gyro → VQF → the shared
   DJI-style stab stream (`vr180-pipeline/src/insv_imu.rs`), measured X6
   lens preset, stream 0 = back lens = LEFT eye by default
   (`SourceKind::dual_stream_iter_swap`). The dual-stream iterators now
   stage packets per stream (`DualPacketPump` in `fisheye_decode.rs`) —
   the X6 interleaves its two video streams in 1 s chunks, which the old
   feed-as-you-read loops turned into dropped packets + mis-paired eyes.
   Lens calibration is read per file (the trailer's factory string, a
   UCM + radial-polynomial model, converted to KB-5 in stream coords —
   `insv_imu::stream_lens_calib`); the "Insta360 X6" preset is only a fallback. The
   lens is rendered through the shaders' unified-camera-model branch
   (`cal.xi > 0` in every `fisheye_*.wgsl`; uniform grew to 7 vec4s) with
   the principal point from the file's `window_crop_info` and the tracks
   in reverse lens order (stream 0 = calibration entry 2); readout and
   gyro offset come from the file (`rolling_shutter_time`, `gyro_timestamp`).
   Verify the UCM branch compiles/renders on D3D11/DX12 (naga → HLSL).
   Stabilization: the orientation stream is a direct gyro integration
   (`insv_imu::integrate_orientation`, VQF only supplies the bias) and the two
   eyes are timed from their own sensor's exposure record (`lens_b_timeline`,
   `dji_imu::per_eye_rotations` / `per_eye_rs_rows`) — the GPU-resident NVENC
   export path got that per-eye plumbing edited blind; check it renders.
   Windows TODO: verify the `D3d11SharedDualStreamIter` arm (edited blind
   on macOS — same pump, same shape as the VT iterator) on an `.insv` AND
   an `.osv`: frame count == trim length, no "Could not find ref with POC"
   spam, eyes in sync. The X6 gets no built-in LUT yet.
9. **Reframed output mode** (`FisheyeOutputMode::Reframe`, 2026-09-09):
   a pinhole-style side-by-side viewport of each eye instead of the VR180
   half-equirect — per-clip zoom (horizontal FOV), pan/tilt/roll, a
   "defish" blend (k-projection `r = k·tan(θ/k)`: 0 = rectilinear,
   1 = fisheye look) and a 1:1 or 16:9 per-eye aspect. Implemented as a
   second ray generator (`output_ray`, `proj_mode` uniform) inside the
   existing equirect kernels — the four `fisheye_*_to_hequirect*` shaders
   (DJI / Insta360 / BRAW) and `eac_to_equirect.wgsl` (GoPro) — driven by
   `FisheyeCalib::with_reframe` / `EacLensAdjust::with_reframe`
   (`gpu::reframe_k`, `reframe_edge_x`, `reframe_ray` are the CPU twins,
   unit-tested). The pan reuses the view adjust's global angles
   (`Settings::output_view_adjust`, tilt-up = negative pitch); stab, stereo
   offsets, per-row RS and the lens override all still apply. Export
   (`FisheyeExportProjection::Reframe { hfov_deg, defish, spatial }`)
   writes the exact viewport at a per-eye size (1080/1440/2160 lines ×
   aspect → 2:1 or 32:9 SBS) and never tags VR180 metadata (no spatial-
   video side output: Vision Pro viewers watch the VR180 export instead).
   Preview follows the aspect
   via `preview_out_dims`; in this mode dragging the preview pans the view,
   scroll / pinch zooms the FOV and double-click recenters (the alignment
   magnifier is inert), and the BeyondVR eye-scale hack is hidden and not
   applied. Preview sharpness: the zero-copy worker projects the reframed
   view from the NATIVE frame (not the 1280 working res), its output is ≥
   1920 px per eye, and a paused frame always shows the native still
   (`wants_full_res_still`). The Windows D3D11 preview path took the same
   edit blind — verify there.
10. **Windows `.360` ProRes/libx265 GPU arm + Vulkan preference + slow-path
   notes (Windows session, 2026-09-12).** `export_eac_gpu_resident` now
   takes ProRes and libx265 too: the NVENC gate became per-encoder, and the
   arm has two encode tails — NVENC keeps the CUDA/P010 ring untouched;
   ProRes/libx265 get the OSV zc arm's readback tail (format handshake →
   P210→`prores_ks_vulkan` with `warm_prores_vulkan()` before thread spawn /
   RGBA64→swscale, encode thread pipelined against decode + GPU). 8K ProRes
   422: 2.5 → 13.3 fps, arm-vs-portable mean 0.6/255 (the portable EAC loop
   is 8-bit RGB; the arm feeds 10-bit 4:2:2). Wgpu backend: eframe + the
   pipeline's own device creation now explicitly prefer Vulkan on Windows
   (probe first, DX12 fallback kept — a silent DX12 pick disabled every
   fast arm). Slow-path visibility: landing on the portable loop or a CPU
   encoder fallback logs at `warn` AND surfaces in the export bar via
   `fisheye_export::{set,append,}export_path_note`. NOT verifiable on this
   box: the DX12-fallback branch itself (Vulkan is always present here)
   and a genuine `prores_ks_vulkan` init failure — both code-reviewed only.
   FOLLOW-UP (same day): the "stalls at 100%" report — the two-pass
   temp+re-mux finalize rewrote the whole encoded video to add audio
   (longer than the encode at ProRes bitrates). The one-pass inline-audio
   mechanism (`attach_audio_passthrough`, built for the macOS VT + portable
   fisheye paths) is now wired into the Windows OSV zc d3d11 arm, the
   `.360` GPU readback tail and the portable EAC loop (stereo only —
   `one_pass_audio_eligible`); video out byte-identical, chain/trim audio
   re-verified. NVENC tails + ambisonic/APAC keep two-pass and show a
   "muxing audio…" phase note (`FinalizePhaseNote`) in the export bar.
   NEXT DAY: the NVENC tails are one-pass too — `CudaNvencEncoder` gained
   `InlineAudio` (raw-FFI twin of `AudioPassthrough`, keep in lockstep;
   audio as a `new()` param since its header isn't deferred) + the
   duration=1 / dts=pts packet fix from `drain_packets` (NVENC leaves
   duration 0 → track a frame short; the old re-mux masked it). Verified
   one-pass on .360 + OSV, single AND 2-segment chains (ffconcat playlist
   = continuous chain audio, windows cut mid-chain), 30/30 frames
   byte-identical to the two-pass reference. The macOS EAC VT arm
   (`export_eac_zerocopy_vt`) followed the next day — same gate, inline
   attach before the first frame, chain-probed duration when per_eye is
   empty; verified on a trimmed `.360` window (2–5 s: 90 frames + 3.0 s
   AAC muxed inline, no temp). Still two-pass everywhere: ambisonic / APAC.
   NEXT DAY (2026-09-15): **generic SBS sources get the Windows fast path.**
   `SbsFisheyeIter` finally honors its hw flag (d3d11va attach + hw-frame
   download — it had silently software-decoded forever, preview included),
   and the zc readback arm takes SBS: new `D3d11SharedSbsIter` (single
   stream, converts the WHOLE frame, precise-seek run-in) yields through
   `ZcFisheyeSource`/`ZcFisheyeFrame` (the arm's source is now that enum),
   and the arm splits the halves with `gpu::split_sbs_texture_16` (two DtoD
   subregion copies) — everything downstream unchanged. Single-segment
   only; the GPU-resident CUDA/NVENC arm stays dual-stream-only (SBS NVENC
   rides the readback arm's P010 feed: 5.4 → 38 fps at 4K). Verified: SBS
   NVENC/ProRes/reframe + trim + one-pass audio, cpu-vs-zc parity 1.03/255,
   and OSV arm parity byte-stable vs pre-change (same 1.233/255 / max 23).
   macOS untouched (its SBS sources ride the portable loop as before).
   SAME DAY: **playback too** — the preview dispatch gated on
   `is_dual_stream()`, so SBS previews still paid download+swscale+upload
   every frame (worst at 8K). `run_fisheye_zerocopy` now takes
   `ZcFisheyeSource` (same enum as the export arm) and `FrameHold` carries
   the imported full texture for SBS alongside the split halves (deferred
   drop must outlive the in-flight copy); `D3d11SharedSbsIter::new_with_work`
   adds the ~1280 preview downscale (native stays the export's default).
   **Contract bug found + fixed doing this:** the SBS iterator didn't
   `wait_gpu_idle()` after `share_eye_converted` like the dual-stream / EAC
   iterators do — the importer could sample memory D3D11 hadn't finished
   writing. It read as BLACK deterministically in a readback test; the
   export arm had been hiding it behind the decode→main channel hop (a real
   latent race, not just a test artifact). Costs ~2% throughput (38→37 fps).
   Verified by `examples/sbs_zc_check.rs` (gitignored): GPU-left vs CPU-left
   101/65535 while GPU-left vs CPU-RIGHT is 3122 (31× — split provably not
   swapped/offset), work-res downscale exact dims + identical mean level.
11. **VRAM gate on hardware decode (Windows, 2026-09-15).** A d3d11va DPB is
   20 surfaces and the NVIDIA driver commits **~4× the P010 payload per
   `BIND_DECODER` array slice**, so the cost is **~240 MiB per megapixel per
   stream** — 8K SBS ~7.9 GB, OSV/`.insv` ~7.0 GB (two streams), `.360`
   ~5.6 GB, 4K SBS ~2.0 GB. Model verified against measurement within 4% on
   all four. This has shipped since **v2.0.0 (2026-06-07, `4eb9ba4`)** for
   OSV and **v2.1.0 (`13098a4`)** for `.360`; generic SBS only joined on
   2026-09-15 and never shipped. **8 and 12 GB cards were already broken for
   the flagship formats**, and OSV preview+zoom (13.9 GB) fails even at 16 GB.
   The failure was NOT graceful: the pool commits LAZILY (`avcodec_open2`
   succeeds; the allocation only fails on the first frame), so FFmpeg
   silently swapped to a software pix_fmt the zero-copy importer can't use →
   `Err` mid-stream → the worker logged one `warn!` and broke → the app
   logged **"decoder thread exited cleanly"**. A VRAM failure was
   indistinguishable from EOF: frozen preview, no message. The
   constructor-keyed fallback at `decoder.rs` could never fire, because the
   constructor had already returned `Ok`.
   FIX: a pre-flight budget check inside **`decode.rs::try_enable_d3d11va_decode_n`**
   — the one function every hardware attach funnels through. It asks DXGI
   (`interop_windows::vram_headroom_mib`, `IDXGIAdapter3::QueryVideoMemoryInfo`,
   no new dependency) and declines if `need + 1 GiB reserve` doesn't fit.
   Declining up front is what makes the EXISTING ladder work: the zero-copy
   constructors all require this to succeed, so they return `Err` → callers
   take the portable path → the portable path's own attach also declines →
   software decode. Dual-stream sites pass `concurrent_streams = 2` so both
   decoders are budgeted (they commit lazily, so per-stream checks would both
   pass and then both fail). Env: `VR180_NO_HW_DECODE=1` forces software,
   `VR180_FORCE_HW_DECODE=1` bypasses, `VR180_VRAM_BUDGET_MIB=6000` fakes a
   small card for testing on a big one. Verified: simulated 8 GB card →
   decline logged, export falls to software and COMPLETES (1.6 fps);
   4K SBS / OSV / `.360` unaffected on the 4090 (fast path still engages,
   same fps). FOLLOW-UP (same day) — the three
   remaining audit items are now FIXED:
   (a) **Truncated export reported `Done`.** All three zero-copy decode
   threads (`fisheye_export.rs`: EAC gpu-resident, zc readback, OSV
   gpu-resident) now return `Result<()>` instead of `warn!`-and-break, and
   each join result is checked BEFORE finalize. A user cancel is explicitly
   NOT a failure (the partial file is kept deliberately), and
   dropped-receiver / EOF stay `Ok`. Note: FFmpeg treats a CORRUPT stream as
   early EOF (`Ok(None)`), so a damaged input still finalizes short — that is
   a different condition from the VRAM `Err` this fixes, left as-is.
   (b) **Decoder-thread panic wedged Play/Pause.** The `finished` +
   `imu_progress.abort` stores moved into a `SignalFinished` **Drop guard**
   in `app.rs`, so they run on unwind too (`finished` is the only thing that
   clears `decoder_alive`). `drain_frames` also distinguishes
   `TryRecvError::Disconnected` from `Empty` now — they were identical, so a
   dead worker looked like an idle one.
   (c) **No `on_uncaptured_error`.** Installed in `gpu::Device::from_existing`
   — the single funnel for BOTH devices (eframe's shared one and the export's
   dedicated one), so macOS gets it too. Records to `gpu::last_gpu_error()`,
   logs at ERROR, and the export bar shows a dismissable "⚠ GPU error" chip.
   `VR180_WGPU_PANIC=1` restores the panicking default for debugging.
   PROVEN both ways by `examples/gpu_error_check.rs` (gitignored): forcing a
   real validation error, the process SURVIVES and records it; with
   `VR180_WGPU_PANIC=1` the same error panics. Verified the real app records
   nothing at startup (no spurious chip).
   Regression sweep after all three: OSV/`.360`/SBS × nvenc/prores/portable
   all pass at unchanged fps, software-fallback output is pixel-identical to
   hardware (0.72/255 mean, encoder noise). SAME DAY: a **cross-GPU guard**
   went into `VulkanImportCtx::from_wgpu` (refuse the context when the wgpu
   adapter's LUID differed from DXGI adapter 0, i.e. detect a hybrid iGPU+dGPU
   box and drop it to the portable path) — **SUPERSEDED 2026-09-24 by PR #7
   (Norman3D), which is the better fix: steer the decoder onto the right
   adapter instead of detecting the mismatch.** The hazard it addressed is
   real — the zero-copy import is +0 MiB (it aliases D3D11 memory), so the
   only way `import_d3d11_handle_to_wgpu` dies is an **adapter mismatch**:
   ffmpeg's `d3d11va` device used a NULL device name = DXGI adapter 0 while
   wgpu picked its own, and on a hybrid machine the NT handle is then
   meaningless to Vulkan. Now `decode::enable_d3d11va_decode_on_adapter`
   resolves the DXGI index for `vulkan_device_luid()`, creates the d3d11va
   device on THAT adapter (index passed as the device-name string), and
   verifies the LUID off the real `ID3D11Device` before attaching — so a
   hybrid laptop KEEPS the fast path. Every zero-copy iterator (dual-stream,
   EAC pair, generic SBS) takes an `adapter_luid: Option<[u8; 8]>` from the
   `VulkanImportCtx`; `None` or an unresolvable adapter declines at
   construction so the portable ladder still works. The function ALSO carries
   the VRAM pre-flight (`concurrent_streams`) — the PR as submitted bypassed
   it, which would have silently reintroduced the item-11 first-frame death.
   `dxgi_default_adapter_luid` is gone with the guard. PROVEN on this box,
   which really is hybrid, with the PR's own `examples/windows_adapter_smoke`
   (tracked, unlike the rest of that dir): RTX 4090 AND Intel UHD 770 both
   PASS 18 eye imports/readbacks + segment seeks + missing/invalid-LUID
   rejection — the Intel was REFUSED outright under the old guard. 4090
   exports byte-identical before/after. **Still open (deliberately):** the six
   `.expect()`s in `import_d3d11_handle_to_wgpu` — with the decoder now on the
   matching adapter the realistic trigger is gone, and de-panicking them means
   threading `Option`/`Result` through ~12 call sites in hot loops (they also
   leak the `VkImage` on the allocate/bind failure paths).
12. **Matching Eyes gained an EXPOSURE trim (2026-09-15) — CROSS-PLATFORM,
   nothing Windows-specific.** `ColorStackPlan::eye_match_exposure` (stops)
   joins `eye_match_ct`/`eye_match_tint` and is applied in the same one
   place, `ColorStackPlan::for_eye`, as `cdl.gain *= 2^(±stops)` — `gain` in
   `cdl.wgsl` is a straight `x * gain` before the shadow/highlight zones and
   the [0,1] clip, so scaling it IS an exposure trim, and it lands PRE-LUT
   next to the CT/tint trim (both correct the sensor signal before the
   creative transform). Because `for_eye` is the single funnel every preview
   and export arm already calls, no new plumbing was needed. It composes with
   the global Gain slider rather than replacing it, and activates the CDL
   stage on its own (one extra dispatch; the rest of that stage is identity).
   Verified on real 16-bit textures by `examples/eye_exposure_check.rs`
   (gitignored): left/right = `v · 2^±ev` to within 1 LSB of 16-bit, the two
   eyes' geometric mean == the source, composition with Gain 1.2 exact, and
   `for_eye` still BORROWS (no clone) when every trim is 0. End-to-end on an
   OSV at 0.4 stop: 8-bit portable and 10-bit NVENC zero-copy agree to 0.02%
   (left ×1.309, right ×0.755 — short of the nominal ×1.3195 only because
   highlights clip on the brightening eye).
   ONE REAL BUG FIXED doing this: the **8-bit portable arms dropped the
   Matching Eyes trim entirely** (CT and tint too — pre-existing, not new).
   Both of them (`fisheye_export.rs`: the portable SBS loop and the portable
   EAC loop) grade the COMPOSED side-by-side buffer in one
   `apply_color_stack_rgb8` call with the un-eyed plan, while every 10-bit and
   zero-copy arm uses `for_eye`. New `apply_color_stack_rgb8_sbs` splits the
   SBS into halves, grades each with its own plan and re-joins — exact, since
   every stage in that 8-bit stack (CDL/LUT/WB/saturation) is per-pixel, so
   there is no spatial filter to straddle the halves and no seam. It short-
   circuits to the OLD single whole-SBS call when no trim is set, so the
   untrimmed 8-bit path is untouched — PROVEN byte-identical (same frame md5,
   same 4265493-byte file, before vs after).

13. **8-bit SBS was broken on the zero-copy path — FIXED (2026-09-15).**
   Reported from the mac side, reproduced and confirmed here. The d3d11va DPB
   format follows the SOURCE bit depth: 8-bit H.264/HEVC → **NV12**, 10-bit →
   **P010**. `P010Converter::convert` hardcoded `R16_UNORM`/`R16G16_UNORM`
   plane SRVs, and a plane SRV's format must match the plane's own format, so
   on NV12 `CreateShaderResourceView` returned a bare `E_INVALIDARG`
   (0x80070057). Worse, it failed at FRAME 1, not at construction — the same
   unrecoverable shape as item 11's VRAM bug: `D3d11SharedSbsIter::new`
   returned `Ok`, every caller reads that as "zero-copy is on", so there was
   no fallback left. BLAST RADIUS was exactly the generic-SBS iterator (the
   2026-09-15 addition, and the only one handed arbitrary files). The other two
   converter callers were unaffected: four GoPro `.360` files all probe
   `yuv420p10le`, and **`.insv` is 10-bit only by project scope** — the X6 is
   what this app targets, and 8-bit Insta360 footage is explicitly out of
   scope, so do not spend effort on it. (Either way the converter now takes
   NV12, so an 8-bit dual-stream source would work rather than fail.)
   FIX, in two parts: (a) `hw_plane_layout(DXGI_FORMAT)` returns the plane SRV
   formats AND the BT.709 range constants for NV12 vs P010/P016, `convert`
   reads the format off the source texture, and the constants moved into the
   cbuffer so ONE compiled shader serves both depths (the HLSL is now
   `YCBCR_TO_RGBA16_HLSL`; P010 and P016 share constants because P010 keeps
   its 10 bits in the HIGH bits, so both blacks normalize to 4096/65535).
   Output stays RGBA16, so everything downstream is depth-agnostic and 8-bit
   SBS gets the fast path rather than just a graceful fallback. (b) a
   construction-time guard — `interop_windows::hw_convert_supports_pix_fmt`,
   checked in `D3d11SharedSbsIter::new_with_work` against the stream's
   `AVCodecParameters::format` — so a format the converter can't sample
   declines UP FRONT and the portable ladder still works. Only the SBS
   iterator gets the guard: it is the one fed arbitrary files.
   VERIFIED: an 8-bit `yuv420p` SBS clip went from `** FAILED **` on frame 1
   to a full `sbs_zc_check` PASS — GPU-vs-CPU 129.5/65535 (0.20%, better than
   the 10-bit path's documented 1.03/255), split separation 31×, work-res
   downscale exact, sustained loop clean; end-to-end export zc-vs-portable
   1.234/255 max 16, matching the OSV arm's own 1.233/255. NO REGRESSION: the
   10-bit OSV zc export is **byte-identical** to before (same frame md5, same
   3868683-byte file), so the cbuffer refactor is a provable no-op for 10-bit;
   the `.360` GPU-resident arm still engages; OSV/SBS × zc/portable all pass
   with stab on. NOTE for whoever touches this next: `import_p010` has **no
   callers** — every zero-copy import is single-plane `import_rgba16` — so the
   `Features::TEXTURE_FORMAT_P010` test in the preview/export gates is
   vestigial. It is merely over-conservative (it can only decline to a working
   path, and every d3d11va-capable adapter has P010), so it was left alone
   rather than widened on shipped paths.
14. **Windows SBS converter honours the range tag (2026-09-25)** — the
   follow-up the mac side asked for with their shared-code change (`FisheyeCalib`
   gained `src_x0`/`yuv_range`, `FisheyeCalibUniforms` → 9 vec4, `ResolveDims`
   → 3 rows, `resolve_p010_planes_to_rgba16` +2 args). Those were TRANSPARENT
   to Windows — no Windows call site constructs `FisheyeCalib` by literal or
   calls the resolve, and the RGBA16 kernels are fine with the larger uniform:
   the export matrix was byte-identical before and after the pull. ONE build
   break, same class as the `VtSharedSbsIter` gate: `pix_fmt_desc` was gated
   `macos` but is called by the cross-platform `codecpar_luma_depth` (which
   `SbsFisheyeIter` now uses on every platform). Pure ffmpeg — ungated it.
   THE CHANGE: `hw_plane_layout` now returns the source bit depth, and
   `P010Converter::convert` takes `full_range` (the STREAM's
   `color_range == JPEG`, read once at `D3d11SharedSbsIter` construction — the
   same field the CPU `SbsFisheyeIter` reads) and pulls all four expansion
   constants from **`gpu::yuv_range_constants(bits, full_range)`** — one
   source of truth shared with the macOS P010 kernels, so the two platforms
   cannot drift again. (Their convention is `t·scale + off` with NEGATIVE
   offsets; the HLSL's `- off` became `+ off` to match. Limited-range values
   are numerically identical to what the HLSL had baked in.) VERIFIED:
   limited clip parity IDENTICAL to baseline (82.6/82.1 per 65535); a
   `yuvj420p|pc` full-range clip (`ffmpeg -vf scale=out_range=full`,
   `-color_range pc`) lands at **82.5/82.1** — the same parity, where a
   still-hardcoded-limited expansion would read in the THOUSANDS (≈16/255
   black offset + ≈16% gain). Smoke test both GPUs + matrix unchanged.
   **Dual-stream and EAC pass `full_range = false` explicitly**, preserving
   shipped, verified output — and here is the finding worth knowing: **GoPro
   `.360` is tagged `pc` (full range) on BOTH EAC tracks** (`ffprobe`:
   `0|yuv420p10le|pc 4|yuv420p10le|pc`), while OSV and X6 are `tv`. NEITHER
   platform honours it: the macOS EAC VT arm has no range plumbing at all,
   the Windows converter passes `false`, and the Windows portable path is
   swscale, which treats 10-bit as limited regardless of tag — which is WHY
   the Windows arm-vs-portable parity is 0.6/255. So `.360` is consistently
   limited-expanded everywhere. Do NOT “fix” one side alone: whether GoPro's
   `pc` tag is truthful (a known GoPro quirk) needs a deliberate cross-platform
   decision with both boxes comparing against a reference.

15. **Automatic per-frame eye-exposure matching for the X6 (2026-09-27) —
   SHARED code, on `main`, unreleased.** Why: the X6's two sensors auto-expose
   independently (shutter AND gain — trailer record 0x0009 is the right
   sensor's AE log, the left gain is recorded nowhere), and the difference is
   real in the image: measured after the LUT in linear light, the eyes sat up
   to ~0.9 stops apart for ~20 s on a real post-mod clip (regression of
   brightness-vs-shutter ratio: slope 0.5–0.8, r 0.87–0.99). The static
   Matching Eyes trim cannot follow that. WHAT: three new compute passes in
   `gpu.rs` + `shaders/eye_match_{hist,reduce}.wgsl`, `eye_offset_apply.wgsl`
   — per-eye 1024-bin luma histogram of the central quarter of each PROJECTED
   eye texture (one kernel for every path; workgroup-shared atomics — the
   first `atomic<u32>`/`var<workgroup>` in the codebase, core WGSL, no
   feature), a one-thread reduce (both eyes' clipped tails cut by RANK
   jointly — I-Log black 0.091 → lo 0.12, white wall 0.835 → hi 0.80 — then
   the mean shift of 5 quantiles across the remaining rank range → smoothed
   per-eye ADDITIVE code offsets in a GPU buffer, `EyeMatchState`; snap on
   |Δ| > 0.008 code, EMA α 0.3 below; min 4096 unclipped samples; reset on
   seek; `reason` tells the UI why a frame was not measured), and an apply pass as STAGE 0 of the colour stack
   (an exposure gain on log footage — additive, NOT the multiplicative CDL
   gain the static trim uses). No readback in the frame loop; the GUI readout
   uses a 3-deep async MAP_READ ring (`EyeMatchReadback`, never polls — Lesson
   #1). API: `ColorStackPlan.auto_eye_match: Option<EyeMatchMode>`,
   `Device::apply_color_stack_pair_16` (both eyes, ONE submit — the 16-bit
   stack was refactored into `record_color_stack_16`; `apply_color_stack_
   per_eye_16` is now a wrapper and bit-identical when the flag is off),
   `apply_color_stack_to_sbs_bgra_matched`, `match_eyes_8` (CPU-fallback arm),
   `create_eye_match_state`. Plumbing: `Settings.auto_eye_match/_mode`
   (serde default) → `build_color_stack`; preview funnel
   `compose_with_color_and_mode` gained an `eye_match` arg (fisheye arms pass
   `eye_match_arg(state, kind, stab_idx, enabled)`, EAC arms `None`; the single-eye
   RIGHT view now swaps AFTER grading — the per-eye trims used to land on the
   wrong eye there); export arms use the pair API with a per-stream state;
   `build_export_cfg` and `eye_match_arg` gate to `Insta360Insv`, so the
   persisted flag is inert for OSV/GoPro. UI: checkbox + reference combo
   (meet in the middle / to left / to right) + EV readout (÷ 0.067 code/stop)
   in the Colour panel's Matching Eyes block, X6 only. **Edited blind for
   Windows:** the pair-API swaps in `export_fisheye_osv_zerocopy_d3d11` and
   `export_fisheye_osv_gpu_resident` and the state creation in
   `run_fisheye_zerocopy` (pure Rust, same shape as the macOS arm; no cfg
   gates touched — `EyeMatchFrame` import is ungated). Verified: gitignored
   `examples/eye_match_check.rs` (synthetic eyes, 20 checks incl. modes,
   snap/EMA/reset, black-eye rejection, 8-bit helper — ALL PASS); real clip
   036 via `reframe_check` + `EYE_MATCH=avg|left`: the shaded segment went
   from L/R −0.44 stops (off) to +0.01 mean / ±0.06 max (on), the equal-shutter
   control segment −0.017 → −0.004, match-to-left leaves the left eye at
   +0.0000 stops vs off; `cargo test -p vr180-pipeline` 45/45. The user
   ran the GUI on real footage (preview + export work; app control for an
   automated check was declined). ADVERSARIAL REVIEW (6 finders, 1-vote
   verify) found one real defect, fixed: the valid-luma gate was on a
   linear scale while the histogram sees I-Log CODES (black 0.091, white
   wall 0.835 — measured), so clipped sky was never excluded and a per-eye
   value gate truncates the brighter eye more → shift biased toward 0. Now
   `EyeMatchTuning::insta360_ilog()` (lo 0.12 / hi 0.80, the default since
   only the X6 is enabled) and the reduce cuts both tails by RANK jointly
   (max clipped fraction over both eyes + 0.03 margin) and averages 5
   quantiles across the valid rank range — harness cases with 20–32 % of
   one eye on the white wall and I-Log-black padding recover the shift to
   ±0.0001. Also from the review: same frame index = re-render, not a seek
   (`frame_for_index`), `EyeMatchState::clear()` while the passes are off so
   re-enabling starts fresh, readback ring collected oldest→newest, the
   paused loops collect the readout each tick, the still allocates a state
   only when it can use it (the still applies the raw per-frame difference,
   the live preview its smoothing — ≤ snap apart), and the export's colour
   panel says "applied per frame" instead of "measuring" (the export stops
   the preview, which is the only readout source). SECOND PASS (3 finders):
   the tail counts are half-open (`count_bins(eye, start, end)`) — WGSL
   `select` evaluates both arms, so the old `count_range(…, lo_b - 1u)` would
   have spun 2^32 iterations for a tuning with lo = 0 (latent; the CPU side
   now clamps lo/hi too); the readout ring is generation-tagged and returns
   only values that landed in the call (no stale flash after re-enable), the
   walk stops at the first in-flight slot; the panel keys on `decoder_alive`
   (an exited worker used to leave "Measuring…"), and a frame the reduce
   cannot use (mostly sky / black) says so via `EyeMatchReadout::reason`.
   **Windows arms VERIFIED on the Windows box (2026-10-09):** real X6 clip
   (026) through BOTH blind-edited export arms via `export_smoke` +
   `VR180_EYE_MATCH=avg` — eyes sat 0.22 EV apart with matching off; with it
   on, the zc readback arm closed the gap to 0.046 EV (79%) and the
   GPU-resident NVENC arm to 0.048 EV (78%), both moving symmetrically
   toward the middle as Average specifies, and the two arms agree to
   0.1/255. No panics; OSV/SBS/.360 exports byte-identical with it off.
   The PREVIEW arm (`run_fisheye_zerocopy` state creation) is launched
   only — the app takes no file argument, so it was not driven headless.
   Known/accepted: the export does
   not publish a readout; Play stays enabled during an export.

16. **GoPro stabilization investigation + per-frame exposure timing
   (2026-10-07) — SHARED code, on `main`, unreleased.** The user rigged a
   GoPro MAX (.360, 30 fps), an Osmo 360 and an X6 (both 50 fps in set 1,
   ~30 fps in set 2) together and shot the same walk/run; the GoPro looked
   clearly worse. MEASURED (scratchpad `stab/`: resid.py centre-crop phase
   correlation, eye_rot.py whole-eye sphere fit of DIS flow, sweep_eval.py /
   timing_fit.py on camera-lock harness exports, cameras aligned by audio
   envelope): (a) on the user's soft-stab exports the residual per SECOND of
   video is the same for all three (set 1: 1.7 / 1.8 / 1.9 deg/s; set 2 the
   GoPro is ~1.4× by the centre metric) — the GoPro's per-frame residual is
   1.6× because it is 30 fps; (b) under camera LOCK on the walking window the
   GoPro leaves 0.20°/frame vs the X6's 0.056°, but that excess is
   NOT the pipeline: phase ±4/±8 ms, a 1 ms point sample instead of the
   15 ms box, VQF 6D / τ=100 s / fixed bias / plain integration, RS window
   centre and RS off all leave it at 0.20–0.22°; the raw (unstabilized)
   frames follow the pipeline's own rotation increments as well as the X6's
   do (R² 0.88 both), the residual-vs-Δω regression puts the GoPro timing
   error at 1.6 ms (the method recovers deliberate ±8 ms shifts as 7–8.5 ms),
   and the residual field is the forward-walk parallax pattern (the GoPro
   sits lower on the rig and its frames are 33 ms apart); (c) in the DARK
   (set 2 end: 1/30 s shutter) the GoPro residual improves monotonically as
   the pose is moved earlier (0.613 → 0.548 @ −8 ms → 0.542 @ −16.5 ms;
   +16.5 ms → 0.764) and the Δω-correlated component vanishes — the missing
   `−shutter/2` term. GoPro CORI on these clips is a constant (bias drift
   0.2°/s): no in-camera stabilization, so the VQF path is used. WHAT
   CHANGED: `vr180_core::gyro::parse_shut` (per-frame SHUT, type `f`),
   `resample_quats_at_times` + `gopro_frame_sample_times(fps, n, shutter,
   readout)` = `i/fps + readout/2 − shutter_i/2` (fallback to the readout
   midpoint per frame when SHUT is absent/implausible); `imu.rs`
   `vqf_cori_equivalent_stream(_multi)` sample there (`PreparedImu.shutter_s`,
   `gopro_shutter_per_frame(segments, fps)` for chains); decoder.rs RS ω
   window centred on the same instant (was frame start, a Python-parity
   choice — now DJI parity, `t_mid`). Verified: core 53 / pipeline 45 tests
   (new: SHUT parse, sample times, at-times resampler parity); dark lock
   export 0.613 → 0.542°/frame (= the −16.5 ms probe), Δω-only R² 0.060 →
   0.004; daylight 0.202 → 0.199 (unchanged). Not changed (measured not to
   matter here): VQF 9D estimator, 15 ms pose window, RS factors; the GEOC
   inter-lens extrinsics (ANGX/ANGY/ANGZ) remain unparsed — a static stereo
   alignment question, not shake. Harnesses (gitignored): eac_stab_check
   gained GOPRO_NO_STAB / GOPRO_RS_T_MS / GOPRO_RS_F; rider_export takes
   SMOOTH_MS / MAX_CORR / RESP and uses the new times.
   SECOND PASS (2026-10-08, "set 2 is all 30 fps, look again"): whole-clip
   whole-eye metric on the user's set-2 exports — slow residual identical
   for all three (ratio 0.99, r 0.97) but the GoPro carried 3.4× the
   frame-to-frame jitter (0.238° vs 0.071° rms), pitch-only, random frame
   to frame. The SHUT fix took it to 0.144°; the rest sat in two seconds
   (t = 73 and 107) that hold 66 % of the GoPro's remaining jitter energy —
   both are DROPPED FRAMES: the source video repeats a frame (consecutive
   diff ≈ 0, both tracks), the telemetry block covering it spans 29 frame
   periods (0.9676 s, 772 gyro samples), the camera's `MSKP` stream steps
   −1 at the repeat (`parse_mskp`), SHUT repeats too, and the container
   hides it (uniform 33.333 ms PTS). `build_imu_sample_times_with_skips`
   now anchors per frame (sensor frame of video frame i = i + MSKP[i]; the
   hold is placed AFTER the repeat so the preceding frame maps with slope
   1), `gopro_frame_sample_times(.., skips)` gives a repeated frame the
   instant of the frame it copies (its lock residual 2.18° → 0.00°), and
   the RS ω timeline, GRAV/ACCL/MNOR and the GUI's `GyroAngvel` all take
   the skips. Also measured and REJECTED: fine phase sweep −4..+4 ms
   (minimum at 0 on the walking window, flat on the dark one — no missing
   constant), RS readout ×0.5/×2, factor sign flip, per-axis factors, ×5
   (RS is live — ×50 moves pixels by 30 levels — but at these rates its
   effect on the residual is a few %), VQF/GRAV coupling (GRAV drifts
   slowly, spikes are single-frame), the rig blob in the GoPro's left third
   (masking it changes nothing). Still open: a ~4° single-frame spike at
   the peak of a 6 Hz ±160°/s mount oscillation (8000°/s², the clip's
   worst second, coincident with the drop) that no variant touches, and a
   residual pitch-only jitter excess of ~1.3× outside the drop seconds —
   both candidates for an under-strength RS model (readout ×2 and factors
   ×5 both nudge the burst the right way), left untuned by rule.
   `VR180_GOPRO_PHASE_MS` remains as the GoPro twin of the DJI diagnostic
   knob (default 0).
   OPEN after the 2026-10-09 pre-push review: (a) a dropped frame that is the
   FIRST frame of a telemetry block charges the shortened span to the
   previous block (`real = n + skip(f1) − skip(f0)`); which block the
   firmware stamps short is unverified — needs a clip with a boundary drop;
   (b) `gopro_shutter_per_frame`/`gopro_skips_per_frame` re-extract each
   chapter's GPMF uncached on every multi-chapter stab rebuild in the GUI
   (slow on SMB) — should take the cached blobs; (c) the headless
   `vr180-render` CLI still centres its RS window on frame start with no
   MSKP/SHUT (it has pre-existing compile errors from wgpu drift anyway).
   THIRD PASS (2026-10-08, "the mount is fine — look carefully at RS"):
   the RS term was MEASURED for the first time (scratchpad `stab/rs_probe*.py`:
   GOPRO_NO_STAB exports with RS off / on / ×2 / −1, DIS flow → Kabsch →
   9-parameter fit of the residual on the row-time and column-time RS
   bases → slope vs the gyro's Δω at the frame instants). Sign RIGHT (the
   derivation: the shader's Y↔Z swap is improper, so `r + (w×r)·Δt` with
   the raw gyro IS the physical correction — and the old memory claim
   "CORI integrates −GYRO" is false in this code's conventions: CORI
   increments vs ∫gyro slope +0.994); axis RIGHT (rows vertical, top row
   first, T_cols ≈ 0); BOTH EYES identical (L/R slopes agree to 0.1 ms;
   GEOC ANGY = 180.26°, ANGX/ANGZ < 1°: the back lens is a pure yaw of the
   front, same sensor "up"); magnitude: each unit of factor removes
   15.3 ± 0.5 ms (= the file's SROT 15.224 within 1 %). The UNCORRECTED
   readout read 17.2 ms over the full field but 15.6 ± 0.5 in the far
   field (sky/façades) and 13.7 on the near ground — walking parallax
   correlates with pitch acceleration and biases the absolute number; an
   independent skew fit on the raw EAC faces gave 17.7–18.3 (same bias).
   Three principled changes landed, none moves the lock residual
   measurably (walking 0.197→0.197, dark 0.541→0.549, burst 0.775→0.753 —
   noise): the readout now comes from the file's `SROT`
   (`vr180_pipeline::imu::gopro_readout_s`, fallback `SROT_S`; GUI seeds
   `rs_readout_ms` from it per clip, the VQF pose instant uses the same
   value); the shader's row estimate uses the camera-frame ray
   (`xr,yr,zr`) instead of the pre-rotation output ray in both EAC shaders;
   `GyroAngvel::groups_at` returns the MEAN rate over [t_ref, t_row]
   (running trapezoid integral, `from_parts` for merged chapters) so the
   per-row term is the integrated angle. CALM-SEGMENT CHECK (rotation
   without walking: 243 @ 0–8 s and 78–88 s, 244 @ 2–6 s and 96.5–99.5 s;
   the elevator segment 244 @ 112–117 s has no far field and was dropped):
   far-field uncorrected readout **15.60 ± 0.20 ms**, each unit of factor
   removes 14.81 ± 0.21, residual after the 15.224 ms correction
   **+0.78 ± 0.15 ms** (the correction nulls at an effective 16.0 ± 0.2 ms) → the true
   readout is 3–5 % above the file's SROT, an 0.08° edge-to-edge error at
   100 °/s — real but not visible, and by rule left to the file (a
   per-camera readout self-calibration from footage would be the
   file-derived way to close it). T_cols ≈ 0 (rows-only model holds).
   The 1.07 yaw scale was near-field lever-arm parallax: far-field-only
   Kabsch gives 1.01–1.02 for yaw and 0.99–1.01 for pitch/roll. By-product:
   the fitted gyro clock offset drops by one frame between the calm
   segments on either side of a dropped frame (243: −33.4 ms exactly;
   244 across the 73.2 s drop: −36 ± 3 ms) and the correction nulls on
   both far sides — the MSKP re-anchoring verified independently (244's
   second drop at 107.6 s is followed only by the unusable elevator
   segment, so it is unverified).

17. **Osmo 360 "sometimes a roll residue, X6 almost perfect" (2026-10-09) —
   investigation, diagnostic knob only, nothing user-visible changed.**
   Same rig clips as item 16 (set 2: Osmo 0017 / X6 059 / GoPro 244; audio
   offsets t_osmo = t_x6 + 3.332 = t_gopro + 1.016). MEASURED EQUAL between
   Osmo and X6 — so ruled out: per-frame roll jitter of the real exports
   (0.307 vs 0.304°/frame), the orientation inputs at the frame level
   (0.03°/frame roll after sub-frame alignment), the shared smoother's
   correction at identical settings (same to 0.1°), the table-predicted
   output motion at identical settings (0.98–1.05× per band, corr ≥ 0.993),
   inter-eye roll of the real exports (0.2–0.4° mean, all three cameras),
   static horizon tilt. MEASURED DIFFERENT: registering the real soft-stab
   exports to each other once per second (scratchpad `stab/cross_cam.py`,
   far-field DIS→Kabsch) puts the Osmo horizon within ±0.3° of the GoPro's
   except ONE Osmo-only transient at the first big turn (GoPro clock
   29–32 s, correction yaw +27°): a rise over ~2 s to a +1.5° peak held
   ~0.5 s, back to 0 (0.1 s zoom; the X6 stays <0.5° there; a second
   candidate at 62 s is ≤0.45° = noise); the X6 wanders ±1° slowly against
   both. Set 1 (0016/058/243): all three horizons agree to ±0.3° (roll std
   0.12°, p95 0.23–0.28°) for the whole clip — no Osmo event at all. Hand-eye comparison of the lock tables
   (data-audit agent): the Osmo's in-camera fused estimate shifts its tilt
   by 1–4° at each ~90° heading turn and holds it between turns (frame-
   invariant tilt-change magnitude up to 9° late in the clip); the X6's
   gyro-only integration (100 s trim) does not. In soft-stab a slow
   world-frame error cancels — actual and anchor carry it — leaving only the
   smoother-lag transient of each step (≈1–1.5° for ~τ at each turn): that
   is the roll residue. Not fixable from the file for the Osmo 360 I (no raw
   gyro; `[2.10]` is a 30 Hz accelerometer, gait-aliased; the dead
   `horizon_lock_quat`/`smooth_gravity_ema` would make it worse); the 360 II
   writes `[3.3.5]` 1 kHz gyro+gravity, where the X6 treatment would apply.
   Code: `dji_imu::apply_lock_ref_diag` + `VR180_LOCK_REF_FRAME=<k0>`
   (fisheye_export, both arms) re-anchors a camera-lock export to frame k0 —
   a lock anchored at frame 0 points wherever the camera pointed at record
   start. MEASUREMENT TRAPS (see the memory note): `acos((trace−1)/2)` on
   f32 tables floors steps <0.03° to exactly 0 (fake "frozen" output — use
   the antisymmetric part); body-frame rotation-vector "disagreement"
   between two cameras mixes heading disagreement into roll/pitch (fake
   multi-degree roll — use image registration or frame-invariant
   magnitudes); walking-scene lock residuals are parallax-dominated.

18. **Insta360 X6 DNG stills (2026-10-09) — new source kind + still export,
   SHARED code, unreleased.** The X6 `.dng` = one 15520×7760 TIFF/DNG 1.3,
   14-bit GBRG Bayer (WhiteLevel 16383, BlackLevel 0), Deflate strips,
   both fisheye circles side by side (left half = calibration entry 1 =
   LEFT eye post-mod, right half = entry 2 = RIGHT); ColorMatrix1 (D65),
   AsShotNeutral, BaselineExposure ≈ 2.6–2.7 EV per shot, OpcodeList2 = 4
   GainMap opcodes (one per CFA site, 13×34 grid over the whole frame, two
   vignetting centres, gain 1.0 → 2.65 at the rims = the lens shading);
   the factory lens strings (same 27-token format as `.insv`, 15488×7744
   frame) in EXIF UserComment and in the trailer; the trailer is the
   still-photo layout (the 1415 B before the 78 B footer ARE the
   ExtraMetadata protobuf: field 19 dims, 25 rolling shutter 37.5 ms, 27
   window_crop {src 7744² → dst 7760², offset 0}, 65 IMU ranges, 111/112
   lens strings, 210 a 3×3 orientation at capture; before it a backward
   chain `[data][u16 id swapped][u32 size]` with 0x0003 IMU (594 × 20 B,
   0.6 s around capture), 0x0009 AE log, 0x001b, 0x001c). CODE:
   `vr180_fisheye::dng` (reader: IFD walk, parallel inflate, tags, gain
   maps, lens strings, still trailer; 9 tests + an ignored sample test;
   decodes the 142 MB sample byte-identical to the Python reference in
   ~50 ms warm), `vr180_pipeline::raw_develop` (CPU: gain map, Malvar-He-
   Cutler demosaic, camera-linear RGBA64 scaled by `1/l_max` with the
   clipped support flagged in alpha and filled neutral; `develop_display`
   = WB → inv(ColorMatrix1) with Bradford to D65 → 2^(BaselineExposure+EV)
   → ACES-fitted tone → sRGB; 11 tests), `vr180_pipeline::dng_write` +
   `ljpeg` (linear DNG 1.4: IFD0 8-bit thumbnail + SubIFD LinearRaw
   3×16-bit, 256² lossless-JPEG tiles byte-identical to Adobe DNG
   Converter's own tiles, or uncompressed strips; the GainMap is baked in
   and `BaselineExposure += log2(l_max)`; 14 tests + an ignored system
   test: exiftool -validate OK, sips/ImageIO decodes, Adobe DNG Converter
   17.3 accepts sample-identically; NOTE macOS ImageIO rejects Deflate and
   multi-strip LJPEG, and any DNG < 100 KiB), `vr180_pipeline::still_export`
   (`export_still`: per half develop → `project_fisheye_to_equirect_texture_16`
   with `still_eye_calib` (= `stream_lens_calib(entry, crop, 7760, 7760)` →
   `FisheyeCalib::new_omni`) → readback → CPU SBS compose → PNG16 / JPEG /
   DNG; 0.6 s for 8192×4096 on the M5 Max), `SourceKind::Insta360Dng`
   (`is_still()`, detect = ext + TIFF header + Insta360 footer),
   GUI: still-aware load/batch-add (`dng::probe`, no video probe; fps 1,
   1 frame), settings bucket "dng", preset "Insta360 X6", no LUT, stab
   panel note, export bar "Still format" (PNG 16-bit / JPEG / DNG linear
   raw) + Native = 15520×7760, output `<stem>_SBS.<png|jpg|dng>`, the
   export worker branches on `is_still()`, command-line paths open at
   startup (`pending_open`). The preview arm (`run_still`, decoder.rs) renders
   the 2×2-binned develop through the texture-input projection and re-renders
   on settings changes. MEASURED on the sample: eye order correct (near
   objects −3° disparity, far −0.2°), far-field vertical offset +0.34° (the
   rig's relative pitch, same as video — stereo adjust handles it), Apple's
   raw pipeline renders the linear DNG with the same WB/exposure as our
   developed PNG. OPEN: the 7744→7760 padding is assumed centred (+8 px,
   the video rule; ≤0.44° common pan/tilt if wrong); gravity levelling from
   the trailer IMU/field 210 not implemented (pitch sign unverified); raw
   exposure/WB sliders not exposed (develop at 0 EV as-shot); `.insp` JPEG
   not supported; Windows readers (WIC/Lightroom) untested; the headless
   harness is `examples/dng_still_check.rs` (gitignored). **WINDOWS (2026-10-10, Windows box):** pulled + built clean first try
   (`flate2`/`image` are pure Rust). Twin harness
   `examples/dng_still_check_win.rs` (gitignored) exports the X6 concert still
   010 on the RTX 4090 Vulkan device in all five shapes — PNG16 / JPEG / DNG
   uncompressed / DNG lossless-JPEG at 2048/eye and PNG16 at the NATIVE
   7760/eye tier (15520×7760, 536 MB file) — all OK, no GPU error recorded:
   the row-band readback carries the 482 MB native eye on Windows too. The
   Windows dedicated export device, whose descriptor now requests the 16384
   texture limit, exports the OSV byte-identical to the shared device; hybrid
   smoke both GPUs, SBS parity and the export matrix all unchanged. **WIC
   (Explorer / Photos):** PNG16 decodes as Rgb48 at full size (both tiers),
   JPEG fine. Both linear DNGs OPEN under Microsoft's Raw Image Extension
   2.5.35 but it exposes ONE frame = the 256×128 IFD0 thumbnail (every
   BitmapCreateOption; no Preview/Thumbnail properties), while the camera's
   OWN X6 DNG fails outright in the same codec ("cannot decode") — so this
   is the codec stopping at IFD0, not our file: the IFD walk is textbook
   (IFD0 NewSubFileType 1, 256×128 RGB8; SubIFD[0] NewSubFileType 0,
   4096×2048 LinearRaw 16-bit, DNG 1.4 / back 1.1, LJPEG 256² tiles, Black 0
   / White 65535). Practical consequence: Explorer and Photos show a soft
   256-px preview of our DNG; a larger embedded preview (Adobe writes ~1024
   wide) would fix that cosmetically — writer's call. Lightroom CC IS
   installed on the Windows box but is GUI-only (no CLI), so it was not
   driven; the exported DNGs sit in the session scratchpad `stills/` for a
   by-hand open. **SECOND BODY / NEW FIRMWARE (2026-10-10):** four stills from the
   2026-10-07 rig test (`W:/261007 3 test/x6/IMG_…_051/052/053/057.dng`, the
   `//DATA/Database` share) — a DIFFERENT X6 (serial BXEA3ABFS89CAV vs the
   concert body's …QE5UCJ) on firmware **v1.2.35** (concert: v1.0.20) — all
   pass every shape (20/20 exports, native tier included, no GPU error).
   Nothing in the reader needed changing: TIFF/DNG tags identical to v1.0.20
   (DNG 1.3, 14-bit GBRG `[1,2,0,1]` in a 16-bit container, Deflate 256-row
   strips, WhiteLevel 16383, the same 7444-byte OpcodeList2; only
   AsShotNeutral / BaselineExposure 2.63–2.74 / capture time differ), and
   the trailer parse is structurally identical (records 0x0101 0x0003
   0x0009 0x001b 0x001c, 2 lens entries, crop, RS 37.507 ms, orientation
   3×3, 4 gain maps, calib_frame 15488×7744) except the 0x0003 IMU record
   is 396 samples on 051/052 vs 594 on 053/057 and the concert file —
   handled; the IMU is not consumed yet anyway. Output sanity: eye means
   98–140 / 255 with contrast ~64 and the two eyes within **0.04 EV** on all
   four, where the concert still's eyes sit **0.38 EV** apart (L 121 / R 157)
   — the per-sensor AE mismatch of item 15 shows in stills too, and only
   on that body/scene. WIC behaviour unchanged (PNG16 Rgb48 full-size,
   DNG thumbnail-only, the camera's own file fails). The harness gained a
   `probe` mode (third arg) that dumps the trailer fields without exporting.
   GRADE (2026-10-09, same day): PNG/JPEG bake the 16-bit colour stack
   (`apply_color_stack_pair_16` on the two eye textures, as video). The
   linear DNG carries the grade as METADATA (`still_export::grade_metadata`,
   validated with exiftool / sips / Adobe DNG Converter): temperature/tint
   → `AsShotNeutral` via n' = n ⊘ ((M⁻¹·m) ⊘ n) with m = the stack's
   encoded gains ^2.2 and M = `camera_to_srgb_matrix` (universal); CDL →
   `ProfileToneCurve` = srgb_eotf(CDL(srgb_oetf(ACES(x)))) sampled at 129
   points, pinned (0,0)/(1,1), only when the CDL is not identity (the tag
   REPLACES the developer's default curve: Adobe + Apple honour it);
   saturation → `ProfileLookTable` dims 1×2×1 (Adobe/DNG-SDK only, Apple
   ignores); exposure_ev → `BaselineExposure`; per-eye trims baked as
   M⁻¹·diag(m)·M in camera space. Not carried: 3D LUT, sharpen, mid-
   detail, eye scale, auto eye match (logged + UI note). `ProfileName`
   "VR180 Silver Bullet - graded|linear", `ProfileEmbedPolicy` 0. Tag ids:
   ProfileLookTableEncoding is 51108 (0xC7A4), NOT 50983. Brightness
   ladder at the left-lens centre of 050 (sRGB 8-bit mean): camera .insp
   JPEG 50 | Apple render of our DNG 41 | our developed PNG/JPEG 69 |
   Adobe ACR preview of our DNG 84 — our ACES develop sits between the two
   developers' defaults. FIXED the same day: PNG16 output was byte-swapped
   (the image crate takes native-endian 16-bit samples), and the full-size
   export failed on a 482 MB readback buffer (now banded, see gpu.rs). The
   harness env SAT/TEMP/TINT/LIFT/GAMMA/GAIN/SHADOW/HIGHLIGHT exercises the
   colour stack on stills.
   HIDE (2026-10-09, same day): for a still the GUI hides (not greys) every
   control that cannot act on one photo — transport row, timeline, stab,
   RS, NR, Auto align, built-in log LUT buttons; with DNG output selected
   also the 3D LUT + Detail sections, and `Settings::still_dng_mode`
   (serde-skipped, set by the GUI each frame) makes `build_color_stack`
   drop LUT / sharpen / auto eye match so preview == DNG. Stills now take
   `FisheyeExportProjection` (Fisheye → `project_fisheye_to_fisheye_texture_16`,
   Reframe → `with_reframe` on the equirect kernel, as `run_still` already
   did for the preview) and the per-eye lens Override via
   `StillLensOverride::from_export_cfg` (same formulas as the video
   resolvers; preset k fallback = explicit preset else "Insta360 X6").
   FREEZE (2026-10-09, evening): "UI stopped responding after exporting
   flat" = the export of the LOADED still called `stop_playback()` (log:
   "decoder thread exited cleanly" 11 ms after "export: spawned worker"),
   and nothing restarts a still's thread (no Play button; sliders bump a
   generation on `control = None`, which is not even logged). Not a
   main-thread block (sample: idle in `-[NSApplication run]`; reviewed by
   three independent agents). Fix: `spawn_export_job` keeps a still's
   preview alive (it holds only textures), "Restart preview" button in the
   still transport row, `stop_playback` clears `decoder_starting`,
   `maybe_reseed_fisheye_override` keeps the last seed across a same-clip
   respawn (reset in `unload_clip`), stills never "play" (Space revives
   paused), no `AudioPlayer::open` on a still, one-shot WARN when settings
   change with no decoder. Known, accepted: during a still export the
   worker's `poll(wait)` holds wgpu's fence read guard per 64 MiB band, so
   a slider drag hitches for ms — same as previewing another clip during
   a video export.
   FULL-RES PREVIEW (2026-10-09, evening): `run_still` renders two tiers —
   QUICK = `preview_eye_w` (slots 0/1) while settings are moving, FULL =
   the binned source size via `preview_out_dims(mode, aspect, src_w,
   src_h)` capped to `max_texture_dimension_2d/2` (slots 20/21), chosen
   by `STILL_SETTLE` = 120 ms since the previous render's end (a change
   within it = drag → quick, then full once the settings rest). First
   frame 7760×3880 SBS; the log line says "(full)"/"(quick)". The
   "Preview size" menu is hidden for stills. The zoom magnifier on a still
   samples `current_display` (no DetailCache for stills), so it is sharp
   now without a separate full-res path.
   NATIVE tier (same evening): `run_still` keeps the mosaic (`bayer`) and
   lazily develops the unbinned 7760² pair (`native`, slots 22/23, one
   failed attempt → binned fallback, logged). Native SOURCE for every
   resting Reframe render (output stays 3880 wide; the viewport samples a
   small lens region so source res is what shows) and native source +
   native OUTPUT (15520×7760 SBS) when `Settings::still_native_zoom` is
   set — the central panel computes `still_native_wanted` =
   zoom × on-screen eye px (physical) > binned eye px × 0.95. Needs the
   16384 texture limit: all three device descriptors (eframe's in main.rs,
   `Device::new`, the Windows dedicated export device) now request
   `adapter.limits().max_texture_dimension_2d.clamp(8192, 16384)`; the
   scratch example `tex_limit_check` confirmed 16384 + a 15520×7760
   compose on the M5 Max. Log line: "(native|full|quick, native|binned
   source)".
   SHARED X6 STEREO (same evening): `stereo_{yaw,pitch,roll}_deg` is one
   value for the "insv" + "dng" kind templates — `save_active_kind_template`
   writes it to both, `kind_template` and `select_clip` overlay it
   (`apply_x6_shared_stereo`, read via `x6_shared_stereo`: insv first,
   then dng). Per-clip INSV snapshots no longer keep their own stereo
   offset (the rig's alignment is one number); pano_* stays per clip.

19. **X6 "360° Audio" = first-order AmbiX (2026-10-10).** With the camera's
   audio mode set to 360° Audio (user did this on 2026-10-07; firmware
   v1.2.35, second body) the `.insv` carries a 4-ch AAC (channel config 4,
   no `chan`/SA3D; the LRV too). Proven against Insta360 Studio's export of
   `VID_20261008_193907_00_046`: Studio's track is SA3D (order 1, ACN,
   SN3D, map 0..3) and equals the raw decoded channels `[W, Y, Z, X]` up to
   a time-varying YAW rotation of (X, Y) (its stabilisation heading) and a
   gain (W/Z R² 0.97/0.93 per 60 s; X/Y R² 0.88–0.98 per 1 s with the fitted
   2×2 a scaled rotation, det > 0; elevations of the active-intensity DOA
   agree within 2°). Studio = gain(t)·Rot_z(θ(t))·raw, offset 26.3 ms.
   A layout downmix (swresample / most players) cancels c2+c3 (anti-phase)
   → 9.5 dB L/R imbalance = the "audio is wrong" report. Implementation:
   `audio::probe_foa_aac` (4-ch AAC), `probe_spatial_4ch` (PCM or AAC),
   `FOA_STEREO_MATRIX` / `foa_to_stereo_sample` (±45° SN3D cardioids, +X =
   camera front = front lens = rig forward), `transcode_audio_window`
   (decode → optional 4→2 mix → `aac` fltp or `pcm_s24le` → m4a/wav;
   planar frame planes via raw pointers — ffmpeg sets only linesize[0]),
   `foa_to_stereo_aac`; `extract_ambisonic_to_wav_window` decodes non-PCM.
   Export: `one_pass_audio_eligible` = Stereo-only for GoProEac AND
   Insta360Insv; `stereo_audio_source` renders the whole-track stereo temp
   (`<out>.video.tmp` → `.foa_stereo.m4a`) that the inline mux / finalize
   copies with the normal trim offsets; `finalize_fisheye_audio` routes the
   X6 to `finalize_eac_audio` (Ambisonic = passthrough + `inject_sa3d_ambix`,
   APAC = decoded WAV → helper). Verified: stereo export R² 0.999 vs the
   expected mix, lag 0; ambisonic export carries SA3D identical to Studio's;
   APAC helper accepted the WAV. GUI: `ClipInfo::foa_audio` (probe at load)
   shows the Audio section for X6; `audio_player` decodes the pair itself.
   **APAC TRAP (same day): the decoded WAV must NOT carry a positional
   channel layout.** `ChannelLayout::default(4)` = 4.0 (L R C Cs) → WAV
   channel mask 0x107 → AVFoundation reads "4.0" and the helper's
   AVAssetWriter (output layout HOA_ACN_SN3D) converts quad→HOA as pure
   SILENCE (1459 bytes of APAC for 6 s; afconvert decode = −240 dBFS; the
   user's AVP file was silent). GoPro's packet-copied WAV has mask 0 ("no
   channel layout") and works. Fix: `ambisonic_layout(n)` (ffmpeg
   AMBISONIC order via `av_channel_layout_from_string("ambisonic 1")`,
   UNSPEC fallback) → mask 0; verified the helper's output decodes with
   per-channel corr ≈ 1 vs the raw track. Check any new APAC input with
   `afinfo` ("no channel layout") and `afconvert … -f caff -d LEF32` + RMS.
   **SECOND HALF OF THE TRAP: an UNLABELLED input made the helper's
   AVAssetWriter ignore the requested HOA layout and write a
   "Quadraphonic" APAC track** (afinfo on every `.360` APAC export so far:
   "Channel layout: Quadraphonic") — audio present, but not a sound field
   for visionOS. Fix in `helpers/swift/apac_encode.swift`: the reader's
   `decodeSettings` now carry `AVChannelLayoutKey` = HOA_ACN_SN3D|4, so the
   PCM is LABELLED (not remapped — the source has no layout) and the writer
   emits "High-Order Ambisonics, ACN/SN3D" with the samples intact (corr
   0.98–0.999 vs source for both X6 and GoPro). ffmpeg's CAF muxer does NOT
   write an HOA `chan` chunk for the ambisonic order (tried), hence the
   label on the reader. Both sources verified through the real export
   path (2026-10-10). **Helper staleness guard (same day):** every Swift
   helper answers `--version` with its source sha256 (compiled as
   main.swift + generated version.swift, `-D HELPER_VERSION_EMBEDDED`);
   `helpers/build_swift.sh --check DIR` / `--install DIR`;
   `crates/vr180-gui/build.rs` rebuilds helpers/bin on macOS when a source
   changes; **`scripts/release_mac.sh` is now the release recipe** (build →
   fresh helpers installed → Info.plist bump → dylibbundler → inside-out
   Developer ID signing → verify incl. the helper check; `--notarize` adds
   notarytool/staple/update-zip/DMG; manifest + upload stay manual). A Dev
   bundle in the scratchpad keeps its own helper copies in Contents/MacOS —
   copy from helpers/bin after a rebuild (and re-sign ad hoc).
   Open: the audio is NOT rotated to follow our stabilisation (Studio does);
   `.insv` recorded in Auto/Stereo modes are untouched.

22. **X6 readout direction measured (2026-10-14).** Probe knobs in `dji_imu.rs`
   (never set in normal use): `VR180_FISHEYE_NO_STAB=1` forces identity
   stabilization poses but keeps the per-row RS term; `VR180_RS_SCALE`
   scales the signed readout window (0 off, −1 bottom-row-first, 2 doubled).
   Harness `examples/rs_probe_export.rs` (gitignored) + a DIS-flow affine
   regression against the trailer gyro on clip 058 frames 3375–3525: the
   ω-correlated residual (vertical stretch vs pitch rate) is 4.6 off, 0.95
   normal, 8.3 negated, −2.7 doubled (ms·k) — normal is right, readout
   ≈ 14.5 ms vs the file's 15.36 ms, both eyes. The raw DNG mosaic is
   upright, so the sweep is physically top→bottom of the scene.

23. **Kandao QooCam 3 Ultra support (2026-10-14).** Same coverage as the X6:
   video, stills, factory lens model, gyro stabilization + per-row RS, LUT,
   shared stereo offset. Facts and where they came from:
   - **Container**: ONE HEVC stream, both eyes stacked **top/bottom**
     (3840×7680 @ 30), 10-bit full range, tagged **HLG BT.2020** on every
     clip seen (both firmwares, kfix `HDR` 0 and 1 alike — luma statistics
     match HLG). The kfix **`HDR` flag is a SENSOR READOUT MODE** of this
     camera (the user's definition: a slower scan for more dynamic range),
     independent of the HLG encoding: it changes the readout time and the
     frame-stamp latency (see Timing), nothing about colour. Colour keys on
     the stream's transfer tag (`VideoProbe::hlg`), never on the flag.
     Proprietary `kfix` / `kvar` boxes AFTER `moov` in a key-value layout
     (`name[32] type[8] count u32 data`; `vr180-fisheye/src/kandao.rs`):
     lens record (`LENS`, a hex string of two 132-byte blocks behind a
     ONE-digit index — nibble-misaligned, scan both alignments; a flags
     byte 0xff on fw 1.1.3 / 0xfe on 1.1.6 follows the magic; the canvas
     is whatever the record says: 13888×6944 on fw 1.1.3, 3840×1920 on
     1.1.6 — every consumer scales the half to its frame; a `.kdat`
     sidecar carries the same boxes and is read when the MP4 lacks them),
     `INFO`
     (fps, IMU ranges, `V_CROP_SCALE` identity), per-frame `PTS` (µs),
     1 kHz `IMU` (`u64 t_us, i16 gyr[3], i16 acc[3]`, full scale =
     range/32768), `EXP` (`pts_ms, shutter_us`), `GAINMAP0/1` (17×13×3 f32,
     plane-major, spanning the 6944² lens half). `SourceKind::KandaoQoocam`
     is detected by the `kfix` box (before the container-layout probe);
     `EyeLayout::{SideBySide, TopBottom}` drives every single-stream split
     (CPU `SbsFisheyeIter`, `VtSharedSbsIter`, `D3d11SharedSbsIter` — the
     Windows one edited blind), the kernels sample the eye at `(src_x0,
     src_y0)` (calib uniform vec4 #9) and `split_pair_texture_16` copies
     by layout.
   - **Lens model**: QooCam Studio embeds its stitching shader as GLSL text
     in its binary (`strings` it): `θd = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸)`,
     `r = focal·θd`, `uv = (cx,cy) + r·xy/|xy|` — exactly the app's KB path
     with `k5 = p = 0`, so no new shader branch. The record is in the
     photo canvas (13888×6944, lens 0 left / lens 1 right); the video eye
     is the half scaled by 3840/6944 (`kandao_imu::stream_lens_calib`), the
     still is the canvas at scale 1. **Verified against QooCam Studio's
     VR180 export** of the sample clip (`KDOutput/…_Output.mp4`, 7680×3840
     SBS half-equirect, frame 450 ↔ the app's 15.0 s render, dense DIS flow
     + a 6-parameter fit per eye): the model and the plain-scaled focal are
     right (f −0.05 % / +0.28 %, residual 0.8 / 1.1 px median at every
     radius). Studio places the VIDEO principal point (+33.6, +11.6) /
     (+36.4, +13.8) stream px from the record's (6.5 px residual without
     it; half-field subsets agree) and uses ≈1184 px/rad for BOTH eyes
     (0.28 % above lens 1's record). **Neither is followed — the user's
     decision (2026-10-14): Studio's calibration is sometimes unreliable,
     do not match its visual output.** Video and stills both render from
     the lens record as is (`lens_pair` / `stream_lens_calib`, plain
     scaling, per-lens focals); the rig alignment is the user's Auto-align
     / stereo offset. The Studio fit stays as evidence that the model FORM
     is right. (The gyro self-calibration, −1.3 % / −30 px, was unreliable
     too — parallax/degeneracy; don't self-calibrate handheld footage.)
   - **Eye order (measured)**: top half = LEFT eye, bottom = RIGHT; DNG
     left half = LEFT (near-field parallax residual after a far-field
     affine fit on the sample clips/DNGs). `lens_pair` puts lens 0 (top)
     in `lens_b` = the left output with the swap off, and
     `dual_stream_iter_swap` passes the user toggle through (the stacked
     iterator emits left = top).
   - **Timing (measured)**: content time = `PTS + readout/2 − shutter/2 +
     latency`, PTS = the first row's READOUT start, readout and latency by
     the **kfix `HDR` flag** (`kandao_readout_ms` / `kandao_gyro_latency_ms`):
     HDR off → readout **19.4 ms**, latency 4.6 ms; HDR on (the slow
     high-DR scan) → **24.6 ms**, 8.0 ms. Measured on two cameras (sh3001
     fw 1.1.3 / bmi260 fw 1.1.6,
     `/Volumes/Silver/zoeqoo`), image motion vs gyro: HDR-off offsets
     +14…15 ms at 0.2–0.6 ms shutters, +12 at 5.7, +7.5…9.25 at 9.1–10 ms
     (= 14.5 − shutter/2, seven clips, IMUs agree to 0.7 ms); the one HDR
     clip +17.25…18 at 3.6–6.6 ms. Readout HDR-off from the app's own
     RS-scale probe (zero crossing at 0.79 × 24.4) + raw stretch
     regressions (17.8–19.3, r to −0.91); HDR-on 24.3–25.5 on five windows.
     The first reading (one HDR clip, `+shutter/2`, 24.4 ms for all) put
     indoor 10 ms-shutter clips ~9 ms off plus a 25 % RS overshoot — the
     "indoor stabilization" complaint. The file states no readout (kfix /
     kvar / INFO of both firmwares searched; EXP / ISP / FRAME_ISP
     timestamps are all `PTS/1000`). IMU axes on both units: image x ↔ −ω_y,
     image y ↔ −ω_x, roll ↔ +ω_z, gravity −1 g on IMU y when upright. `KANDAO_IMU_TO_CAM` =
     the **identity** (rows = cam x, y, z in IMU coords, the X6 convention;
     `VR180_KANDAO_BASIS` overrides for A/B) — settled empirically, not
     derived: `examples/rs_probe_export.rs` (kind-aware;
     `VR180_PROBE_MAX_CORR=0` = uncapped lock) on the clip START (a 15° cap
     saturates on a window far from frame 0 and every basis then looks
     "ineffective"), then the residual inter-frame motion of the output
     (0.5–6.5 s of Q360_20250512_160943, 512 px eye): no-stab 0.84 px,
     identity **0.08 px**, diag(−1,−1,1) 1.68, diag(1,−1,−1) 1.16,
     diag(−1,1,−1) 1.10 — a wrong sign is worse than no stabilization, as
     on the X6.
   - **Colour** (independent of the `HDR` readout flag above): the
     stream's YCbCr matrix now travels with the frame
     (`yuv_mat`, calib uniform vec4 #10; `yuv_matrix_constants`; the CPU
     path sets swscale's BT.2020 table) — BT.709 everywhere else, unchanged.
     HLG clips auto-apply the built-in `kandao_hlg_bt2020_to_rec709.cube`
     (generated by `scripts/gen_hlg_lut.py`: input ×0.77, BT.2100 inverse
     OETF, 1000 nit OOTF, BT.2408 203 nit → SDR white with a soft shoulder,
     2020→709, BT.1886); Kandao ships no LUT. A clip tagged Rec.709 gets
     none. **The ×0.77 (2026-10-17)**: the camera does not keep the
     BT.2100 headroom — measured on five clips of both units, 90 % of
     pixels below HLG 0.72–0.81 but 1 % above 0.94–1.0 and 2–4 % above
     0.9 — so with reference white at 75 % everything above HLG ~0.65 sat
     in the shoulder and read as blown; the user found the picture right
     with the pre-LUT CDL gain at 0.77, and that is baked in (full scale →
     SDR 0.977, 0.75 → 0.66, 0.5 → 0.43). Kandao Studio's own SDR export
     (percentile-matched on the park clip, `$SP/zoe/hlg/`) is close to
     code^0.7 — 0.68 → 0.77, 1.0 → 1.0, brighter mids — NOT copied; tone
     follows the user's judgement. The CDL gain is `x * gain` on the
     decoded code values BEFORE the LUT (`cdl.wgsl`), so a user gain is a
     tone change on HLG, not an exposure. The Windows d3d11 converter now
     carries the stream matrix too (item 24).
   - **Stills**: `dng::read` branches on Make = Kandao: no trailer, no
     opcodes; lens record + shading grids from the EXIF `UserComment`
     (`kandao::still_meta_from_user_comment`), grids re-expressed as one
     pitch-2 `GainMap` per CFA site per half so `raw_develop` applies them
     like the X6's; `still_eye_calib` / `still_lens_calibs` take the KB
     branch. 10-bit in a 16-bit container (black 64, white 1023), RGGB.
   - **Audio**: the 4-channel AAC is the raw microphone capsules (not
     AmbiX). Kandao's "Audio SuperSteady 360°" is **Fraunhofer upHear** — a
     parametric spatial process — so Studio's stereo and ambisonic exports
     are not linear in the capsules above ~400 Hz (per-band R² 0.92 at
     150–400 Hz, < 0.35 above) and cannot be matched by any matrix. The app
     encodes the ARRAY itself (`audio::QuadEncoder`).
     **Capsule → face, measured without Studio** (`$SP/zoe/mic/anchor5.py`,
     `hand.py`, `flowyaw.py`, park-walk clip): the capsules sit strictly on
     the four faces (user has had the camera open). FRONT/BACK: the
     loudest coherent windows (the operator, holding the camera at arm's
     length) reach ch2 first and ch0 last in every one (arrival +7.6 /
     −2.1 / −4.6 / −1.0 samples), and that lag pattern is the one cluster
     that holds through 230° of yaw → ch2 back, ch0 front. LEFT/RIGHT: the
     physical yaw was read off the image (phase correlation of the frame
     centre; r = −1.00 with gyro y, so +gyro y = turning RIGHT), and a
     world-fixed sound must keep `θ_cam − yaw` constant through the turns:
     true only with ch1 LEFT / ch3 RIGHT (world-angle spread 16° vs 48°
     for the mirror; both big turns agree). The operator then sits ~7° left
     of straight behind (right-hand grip, hand at the frame's lower right).
     **ch0 front, ch1 left, ch2 back, ch3 right.** All four in phase (peak
     sign positive 96–100 %), pair offsets within ±1.6 samples, apparent
     opposite-pair spacing ~12 samples ≈ 85 mm with body diffraction.
     TRAPS from the earlier fits: Studio's ambisonic file is standard AmbiX
     W Y Z X (ch2 = Z is the quiet one) — the [0,2,3,1] map was wrong and
     produced the "capsules on the corners" result; Studio's field is
     VIEW-locked and puts the operator at −15° (its "front" is the opposite
     face), so never anchor the layout on it; the free lag fit has a
     reflection ambiguity, so handedness needs the gyro/image anchor.
     **Encoding** (`QuadEncoder`): a spaced pair's difference is the
     pressure GRADIENT — `j·k·d·cos θ`·p, 90° off the pressure and +6
     dB/oct — so a frequency-flat matrix (the first version) gave every
     direction the same level, only a phase smear (`<W·X>` ≈ 0; the
     "intensity DOA" of that output was noise). Now: W = mean through two
     first-order high-passes at 150 Hz (`KANDAO_GRAD_HP_HZ`) and the
     all-pass `(β − z⁻¹)/(1 − βz⁻¹)` at 250 Hz; X = (c0 − c2), Y = (c1 − c3)
     through the same two HPs, two more poles at 250 Hz
     (`KANDAO_GRAD_WIND_HZ`; the all-pass on W has exactly their phase —
     without them a sub-300 Hz gust peaked 10–18 dB over W and set the
     level), a leaky integrator at 250 Hz (`KANDAO_GRAD_INTEGRATE_HZ`) and
     `c/(2π·250·0.085)` = 2.57 → SN3D `p·cos θ` from ~300 Hz to the ~2 kHz
     alias, phase-matched to W within 90° − atan(f/250) (14° at 1 kHz, never
     inverted); unit test `kandao_quad_encoder_places_a_plane_wave`. Z = 0.
     Stereo = `FOA_STEREO_MATRIX` (±45° cardioids) of that field, same
     encoder in the player (`StereoMix::Quad`, state reset on seek). Export:
     Stereo (one-pass temp), Ambisonic (`quad_to_foa_aac` → mux + SA3D),
     APAC (`quad_to_foa_wav` → helper) in `finalize_kandao_audio`; the Audio
     section shows for `quad_audio`. **Channel timing (2026-10-16)**: the
     second unit (fw 1.1.6.6, `/Volumes/Silver/zoeqoo/`) records ch0
     12–21 samples (0.25–0.45 ms) LATE in all nine of its clips (indoor,
     outdoor, Jan–Apr), by a different amount per recording; the first
     unit (fw 1.1.3.8) measures 0–2. Diagnostic: in diffuse windows ch0's
     lag against EACH of the other three is the same (+12…+15) while the
     other pairs sit at 0 — acoustics cannot do that. Uncorrected, the front
     capsule hears everything after the back one and the F–B dipole flips
     the children in front of that camera to behind it. Fix =
     `measure_quad_channel_delays` (per file, no constants: 24 s probe,
     300–4000 Hz, 2048-sample windows with all six pair peaks > 0.5 within
     ±40; per pair the 3–97 % lag-range MIDPOINT — a latency shifts the
     range, acoustics only widen it; `t_k = ¼Σ_j lag(k,j)`; applied as
     `max − t_k` delay lines in `QuadEncoder::with_delays` when ≥ 40
     windows, residual ≤ 3.5 and the largest ≥ 3), called by
     `kandao_window_prep` for exports and by the player at open (scaled to
     the device rate). Synthetic test (14-sample-late ch0 → [0,14,14,14]).
     Verified on the kids clip 084340 (operator behind, girl centre, boy at
     the right): measured [0, 8, 7, 12]; before: loud windows piled at
     −60…−30° and −180…−120°; after: loudest 5 % → 177°, the children at
     −30…0° / −60…−30°, nothing on the left; the park clip (unit 1) measures
     [2,2,2,0] → untouched. `$SP/zoe/mic/kids/{kids,offsets,offset2..5}.py`.
     The estimate is good to ~±4 samples (independent quiet-window NCC
     put the kids clip at 13–15 where the midpoint method says 8–12; the
     range midpoint is biased when one side's sources sit nearer) — that
     is ~±15° of azimuth, not a face flip; 300–4000 Hz with p3/p97 was the
     only stable variant (narrower bands lose the windows). **Channel
     response (2026-10-17)**: capsule levels differ too, the same in every
     clip of a unit (`gains.py`): unit 2 below ~500 Hz ch0 +5 dB, ch1 −8,
     ch2 −4, ch3 −5 (a low-cut on three of four), 600 Hz up ch0 +2, ch1
     −3, ch2/3 0; unit 1 a flat ±3 dB spread. Below 500 Hz the F–B
     difference is then mostly `(g0 − g2)·p` (1.07 p vs a 0.47 p
     gradient at 300 Hz). `quad_trims_from_pcm` (same probe decode, quiet
     half of the windows by mid-band total, 150–500 and 600–4000 Hz
     medians) → `QuadEncoder::with_trims`: a first-order low shelf at
     500 Hz per channel that balances each OPPOSITE PAIR about its own
     geometric mean (ch0/ch2, ch1/ch3). TRAP: matching all four to one
     mean scaled X by ×0.84 and Y by ×1.5 (mid band, unit 2) and pulled
     every direction ~20° sideways on BOTH units' clips — pair balancing
     keeps each dipole's gain and only removes the leak. `QuadCalibration
     { delays, trims }` travels in `Mix::Quad*` / `kandao_window_prep` /
     the player (`measure_quad_calibration`). Verified: kids clip loudest
     5 % → 174°, children at −30…0° (91 windows) / −60…−30° (28), nothing
     left; park 178°, front cluster −60…+30°.
     Level: the capsules sit 15–20 dB low
     and vary by that between clips, and there is no limiter, so every
     export window is decoded once through the encoder first and scaled to
     a 0.7 peak (`kandao_window_gain`, 0.5×…16× on top of the ×3; the same
     gain for Stereo / Ambisonic / APAC of a window; the live player keeps
     the fixed ×3). Verified on the park clip 0–20 s: the loudest 5 % of
     windows resolve to 175° (operator), stereo L−R +1.5 dB on those
     windows and −1.9 dB on a front-right source, W/X/Y peaks within 4 dB,
     stereo rms −33 dB (Studio's own output: −30). Limits: no per-band EQ
     beyond that; the field is not rotated with the stabilization (same as
     the X6). The player takes the mix kind from the caller
     (`AudioPlayer::open_with_mix`) — the codec alone cannot tell AmbiX
     from a mic array.
   - **GUI**: templates `"kandao"` / `"kandao_dng"`; the shared stereo
     offset is generic now (`SourceKind::stereo_sibling`); preset "Kandao
     QooCam 3 Ultra" appended LAST in `presets.rs` (index 7 = Custom is a
     hard-coded fallback — never insert before it); eye-exposure matching
     stays X6-only (I-Log tuning).

24. **QooCam on Windows (2026-10-17, Windows box).** Pulled the Kandao
   batch, built clean first try, lib tests 57 / 38 / 81 green (the pipeline
   test exe needs the FFmpeg DLL dir on PATH — `STATUS_DLL_NOT_FOUND`
   otherwise). Test material on the NAS: the 2024 QooCam 3 Ultra clips under
   `W:/Qi Siyang/20241101sandiskpro q3u/0617 q3 ultra/` are **STOCK-360
   recordings (pre-mod: the two circles face opposite ways)**, so they prove
   the Windows mechanics, not a stereo result; the only post-mod material
   here is the Jan-2025 Interval DNG set (`W:/2026 2t nvme backup/250112
   mod time/100QooCam/`, fw V1.1.0.10). Older firmwares V1.0.0.33b /
   V1.0.1.6b / V1.0.2.6 all parse (same 13888×6944 lens record, 832.6 Hz
   IMU on this unit — not 1 kHz; `imu_period_s` carries it; frame_quats ==
   frame count; the `HDR` flag is 1 on the two 2024 HLG clips and 0 on the
   60p one while every clip is tagged HLG — matches the mac reading).
   `examples/kandao_probe.rs` (gitignored) prints all of it. The stacked
   `D3d11SharedSbsIter` (edited blind) PASSES `sbs_zc_check` (now layout-
   aware, with a per-channel readout) on 3472×6944 and 3840×7680: GPU-left
   vs CPU-left 21 / 15 per 65535 against 6572 / 7751 cross-eye, work-res
   exact, sustained 108 / 96 fps. THREE things fixed here:
   (a) **the Windows export arm still declined the QooCam** — its `src_ok`
   gate keyed on `SbsFisheye` (the iterator construction below it had been
   updated, the gate had not), so every QooCam export took the portable
   loop at 1.75 fps with the note "no GPU fast path for this source type".
   Now `is_single_stream_pair()` (gate + the path-note ladder): ZERO-COPY
   ENGAGED, libx265 5.2 fps, **NVENC 32.7 fps at 3840²/eye**, cpu-vs-zc
   parity 0.775/255 max 23 (the OSV arm's class).
   (b) **the D3D11 converter had BT.709 built in** (the mac side flagged it):
   `ConvDims` now carries `(r_v, g_u, g_v, b_u)` from
   `gpu::yuv_matrix_constants`, `P010Converter::convert` and
   `share_eye_converted` take `bt2020` — the stream's `colorspace` tag read
   at `D3d11SharedSbsIter` construction next to `full_range`; the dual-
   stream and EAC iterators pass `false` (BT.709, as before). On the HLG
   clips the GPU-vs-CPU per-channel error went R 106 / G 131 / B 44 → R 28
   / G 18 / B 39 per 65535 (R and G both carry the 0.10·V term of the 709↔2020
   difference, B only 0.026·U — the signature of a matrix mismatch, now gone).
   (c) **single-lens recordings**: the camera's `Q180_…` mode writes ONE
   3840×3840 circle at 60 fps with the same `kfix` boxes; `detect` returned
   `KandaoQoocam` and the stacked split would have made two half-circle
   "eyes" (the lens calib scaled to 3840×1920, fy halved). `detect` now
   requires the stacked 1:2 frame (`kandao_frame_is_stacked`, largest video
   stream `h == 2w`; an unopenable file keeps the Kandao kind) and returns
   `Unknown` + a warn otherwise. Audio on Windows: Stereo export = 2-ch AAC
   from `quad_to_stereo_aac` (the explicit-muxer open from item 20 carries
   it — the `.m4a` temp would otherwise hit the gpl-lite `ipod` gap again),
   Ambisonic = 4-ch AAC + SA3D injected; the channel-delay probe found no
   coherent windows on these clips (nothing applied, logged). Stills: the
   2025 Interval DNG (13888×6944, GAINMAP0/1) and a 2024 6944×3472 half-res
   DNG (no gain maps) export in all five shapes incl. the native 6944² /
   3472² tier on the 4090 (`dng_still_check_win` takes the native tier from
   the file now). NO REGRESSION: a stash-build-run before/after sweep on
   the OSV (cpu / zc readback / GPU-resident NVENC) and the `.360`
   (GPU-resident NVENC) is **md5-identical** on every output — the cbuffer
   matrix is a provable no-op for BT.709 sources (`scratchpad/gen/regress.ps1`).
   GUI not driven headless; launched on a clip for the user.
   GUI (same day, on the post-mod `//SYSTEM/Silver/zoeqoo` material, fw
   V1.1.6.6, 788 Hz bmi260, lens record canvas 3840×1920 scaled to the
   3840² eye): a "load shows Click ▶ Play, Play does nothing" report could
   NOT be reproduced — command-line open, three dialog loads by the user
   and a scripted dialog load of `Q360_20260412_082220` (SetCursorPos +
   SendKeys into the native dialog; NB this display is 200 %, so a
   DPI-unaware script sees VIRTUAL coordinates = screenshot px / 2) all put
   the first frame up within 0.3 s and play at 30 fps; all 19 clips pass
   `examples/kandao_sweep.rs` (gitignored: detect → boxes/IMU → zc iterator
   3 frames → audio calibration). The report's own log was lost because the
   app truncated `vr180-gui.log` on every launch — it now renames the
   previous one to `vr180-gui.prev.log` first (`main.rs`). Also seen there:
   QooCam Studio's `KDOutput/*_Output.mp4` KEEPS the `kfix` box, so without
   the 1:2 guard a Studio export would have been split into two half-eyes.
   **THE REPORT, CAUGHT (same day, `vr180-gui.prev.log`):** load OK, Play OK,
   then a Stop→Play respawn at 50 s: the new decoder thread stalled 3 s
   (two ffmpeg opens of the SMB file while the old decoder was still winding
   down), built its iterator, and 240 ms later logged "decoder thread exited
   cleanly" — ZERO frames, no error — so the panel showed "Click ▶ Play".
   ROOT CAUSE: ffmpeg's `av_read_frame` answers `AVERROR_EOF` for a failed
   READ as well as for the end of the stream (a short/failed read on the
   `file` protocol sets the avio context's `eof_reached` + `error`), and
   ffmpeg-next's `packets().next()` folds that into `None`, which every
   iterator took as a clean EOF. FIX (pipeline, all platforms):
   `fisheye_decode::demux_eof_or_error` reads the avio `error` field at
   every EOF branch — the single-stream iterators (`SbsFisheyeIter`,
   `D3d11SharedSbsIter`, `VtSharedSbsIter`) and the EAC pair iterator now
   return `Err("demuxer read error …")` instead of `Ok(None)`; the
   dual-stream `DualPacketPump` logs it at WARN (its callers keep the EOF
   contract). FIX (GUI): `poll_load` respawns ONCE automatically when a
   decoder exits before its first frame (`decoder_retried`, cleared when a
   first frame lands; stills excluded), logging a WARN — the user never
   sees the dead "Click ▶ Play" state for a transient network read.

25. **Blackmagic RAW (PYXIS 12K + VR180 lens) brought up to the X6/QooCam
   level (2026-10-19).** The early BRAW port (Python-era: VQF nearest-sample
   hard lock, no RS, mono single-track, 8-bit only, no LUT) was rebuilt on
   the modern path. Sample material: `/Volumes/Database/Pyxis braw stab
   footage/A001_04071604_C005.braw` (12288×6480 @ 50, 56 GB) and
   `…C006.braw` (12288×5112 @ 60, 32 GB), PYXIS 12K fw 10.0, a dual-fisheye
   lens on the one sensor (two ~6360-px circles side by side), handheld.
   - **Helper (`helpers/braw/braw_helper.cpp`, SDK 6.0 — the installed SDK's
     callback interface gained `ReadAudioComplete`, which the old helper
     lacked)**: `--info` now also reports `readout_ms` (= `sensor_area_captured`
     rows × `sensor_line_time` µs; the old key `sensor_area_captured_height`
     never existed and the line time is an INTEGER variant → it read 0),
     `sensor_line_time_us`, `sensor_area_captured`, `crop_origin`,
     `shutter_type/text/s_min/s_max` (five frames sampled), `viewing_gamma /
     gamut / bmdgen`, `embedded_lut`, `pipeline` (cpu), `resolution_scales`;
     `--decode … --scale full|half|quarter|eighth` (ReadJobHints +
     `SetResolutionScale`; **the SDK's steps on this clip are 2/3, 1/3, 1/6
     — 8192 / 4096 / 2048 wide — not 1/2 1/4 1/8: take the size from the
     decode header**), `--look film|video|extended|rec709`, dual-track header
     `bit_depth` + 16-bit fix; `--meta [--frame N]` (every clip + frame key);
     `--shutters [--step N]` (per-frame `shutter_value` "7.2°" / "1/535" →
     seconds; reading a frame's metadata reads the frame: 106 s cold for all
     2894 frames of C006 — the app samples every fps/2 frames);
     `--lut <file> <out.cube>` (the embedded "Gen 5 Film to Extended Video",
     33³, stored B,G,R → written R,G,B; matches Resolve's shipped cube to
     5e-7). Decode: 12K 8-bit ~25 fps CPU (16-bit the same), the pipe +
     CPU swizzle make the export ~5.5 fps at full size.
     **GPU decode (2026-10-20, user: "用 GPU 解 BRAW")**: `--pipeline
     auto|metal|cpu` (default `auto` = the first Metal device from
     `IBlackmagicRawPipelineDeviceIterator` + `IBlackmagicRawConfiguration::
     SetFromDevice` before OpenClip; pure C++, no ObjC, no extra frameworks;
     `metal` fails instead of falling back). The processed image is then a
     MTLBuffer, read back INSIDE ProcessComplete (the only time the SDK
     guarantees it) through `IBlackmagicRawResourceManager` (from
     `IBlackmagicRawConfigurationEx::GetResourceManager`): the host pointer
     when the buffer is CPU-visible (the PYXIS clips: unified memory), else
     `CopyResource` of the WHOLE resource — a partial copy is E_INVALIDARG
     (0x80000003), which is how the URSA Cine Immersive's padded 26,116,096-
     byte track resources first failed; `BRAW_HELPER_READBACK=copy` forces
     the copy (A/B: identical bytes). The decode loop is now pipelined for
     both pipelines: read jobs carry a ticket (frame index, or index·2 +
     track) as user data, up to 3 frames in flight (≤ ~1.5 GB of frames),
     completions post by ticket (the SDK finishes in any order), buffers
     recycled; the CPU-pipeline output is byte-identical to the old helper.
     Header + `--info` gain `"pipeline"` / `"gpu_device"` ("Apple M5 Max");
     `DecodeOptions.pipeline`, `VR180_BRAW_PIPELINE=cpu|metal` for A/B, the
     GUI logs it in the "preview decode scale" line. **Measured (M5 Max,
     frames in the page cache)**: helper alone, quarter (2048-px eyes) 350–
     375 fps Metal vs 211 CPU (38 vs 81 ms CPU time per frame), half 63 vs
     62 fps wall but 80 vs 270 ms CPU, full 12K ~62 vs 35 fps; through the
     Rust iterator (pipe + BGRA→RGBA swizzle + canvas): quarter 136 vs 111
     fps, eighth 478 vs 362, half 36 vs 32, 16-bit quarter 95 vs 80 — the
     per-pixel swizzle is now the limiter (next step: ask the SDK for
     RGBAU8/U16 and memcpy). Metal pixels ≠ CPU pixels: 8-bit mean |Δ| 0.6–
     0.8 codes, max 13, a −0.54-code bias (Metal darker), no geometric shift
     (shifting by 1 px triples the difference), larger in the shadows; 16-bit
     mean 123/65535; a camera-locked export differs by the same hair (C006
     34–36 s lock residual 1.141 vs 1.136 px), and the rim-circle seed moves
     a few px with the pipeline (C006 left (2942, 3040) r 3202 CPU → (2947,
     3035) r 3189 Metal; `braw_layout` decodes with the same pipeline as the
     frames). Time
     to first frame 50–80 ms vs 20 ms (the first Metal run on a machine ~1 s:
     shader cache). The SMB NAS caps a cold quarter decode at ~72 fps. **Follow-up (same day,
     user: "RGBA 输出、eighth 和 --step 一起")**: (1) the helper asks the SDK
     for RGBAU8 / RGBAU16 (header `"format": "rgba"`; an older build's
     BGRA is still swizzled when `format` is absent) and `BrawFisheyeIter`
     copies rows (`copy_row`) with a cached black row for the canvas pad —
     the per-pixel swizzle is gone; (2) the preview picks the decode scale
     from the clip's ACTUAL step table (`BrawInfo::resolution_scales` ←
     `--info`'s `resolution_scales`, `DecodeScale::for_eye_width_in`): a
     12K clip at 50/60 fps now decodes at the SDK's "eighth" = 1024-px eyes
     (4× fewer pixels than the nominal rule's "quarter"), EXCEPT in the
     reframed output mode, which wants ≥ `REFRAME_PREVIEW_MIN_W` (1920) of
     source → "quarter"; the mode is read when the decoder opens, so
     `draw_fisheye_output_panel` respawns the decoder (current position +
     play state) when a BRAW clip crosses into/out of Reframe; (3) `--step
     N` in the helper (every Nth frame from `--start`; "frames" = emitted
     count) — the preview of a >30 fps BRAW clip asks for step 2
     (`braw_preview_step`) instead of decoding every frame and dropping
     half; `FisheyePairIter::source_decimation()` (default 1; Segmented /
     Denoising forward it) tells the loop not to thin such a source again,
     `pts_s` stays the true frame time so the stab index is unchanged.
     Export and stills keep step 1. Measured through the iterator (cached
     frames, Metal / CPU): quarter 145 / 119 fps, eighth 437 / 359, half
     51 / 38, 16-bit quarter 99, full 12K 8-bit 26 (the pipe's ~4–8 GB/s
     is the ceiling now, not the Rust side); the camera-locked export is
     byte-identical to the BGRA-path run. A 60p clip's preview now decodes
     30 fps of 1024-px eyes (or 2048-px in Reframe).
     **Live test of that build (2026-10-20)** showed the decode wait at
     0–80 µs but playback STILL render-bound: after ~0.5 s of play every
     frame's `project` took ≈ 33 ms and `compose` ≈ 17 ms — exact multiples
     of the 60 Hz refresh. Cause, confirmed in wgpu-core 29 (`present.rs`
     165: `device.fence.read()` held across `acquire_texture`; `queue.rs`
     1202: `queue_submit` takes `fence.write()`; wgpu-hal Metal `acquire_
     texture` blocks in `nextDrawable`): a UI thread that repaints FASTER
     than the display waits in `get_current_texture` almost permanently
     while holding the fence read-lock, so every `queue.submit` from the
     worker stalls until the next vsync — the old 8 ms (125 Hz) repaint
     timer while playing did exactly that (and it predates BRAW: the OSV
     path's project/compose spread was the same). Fix: `DecoderControl::
     ui_ctx` + `wake_ui()` — every preview loop calls it right after a
     successful `frame_tx.try_send`, so the window repaints once per
     delivered frame; `App::update`'s timer is now a 100 ms fallback while
     playing (50 ms paused). Mouse-driven repaints can still collide for a
     moment; merging the per-frame submits (project L + R + compose → one
     encoder) is the remaining lever if dragging ever shows it. Two other
     fixes from the same session: the audio player's resampler rejected
     every BRAW frame ("Input changed", the PCM track's EMPTY channel-
     layout mask; pre-existing — `normalized_layout` is now pub and the
     player stamps its frames), and the zoom-still path's full-size helper
     no longer decodes and holds the next 12K frames (`BrawFisheyeIter::
     new_single_frame`: lazy, `--count 1`; EOF drops the helper so a later
     seek/pull restarts).
     **2026-10-21, "卡在 click play"**: the same NAS that read 1 GB/s the
     day before delivered ~25 MB/s on cold reads (10GBase-T link up; the
     share itself was slow) — and the BRAW path did far too much I/O before
     its first frame: `load_braw_imu` ran `--shutters` (one WHOLE frame per
     sample: 160 frames ≈ 2.2 GB on C005 → ~90 s at 25 MB/s, during which
     the decoder thread logs nothing after "eye dims" and Play looks dead),
     `BrawInfo::probe` was spawned ~5× per load/play (each samples five
     frames ≈ 70 MB), and the zoom-still's constructor started an unlimited
     full-size helper that decoded 3 × 12K frames before `with_count`
     restarted it. Fixes: the live view builds its IMU on the probe's
     clip-level shutter range (`BrawShutters::default()` → `build_braw_imu`
     falls back to the mean and sets `DjiOsvImu::shutter_timing_approx`;
     content-time error ≤ ±(max−min)/2 ≈ 1.5 ms), and the export reads the
     per-frame table when the preloaded carrier is approximate
     (`resolve_export_braw_imu`); `BrawInfo::probe` is memoised per
     process by path + size + mtime (`probe_uncached` = the helper run);
     the still uses `new_single_frame` (no start until the first seek +
     pull). A 51 s gap with no log line from ANY thread during a C006 load
     looked like the SMB share stalling, not the app. At 25 MB/s the
     preview is I/O-bound (~2 fps: the SDK reads whole frames even at
     reduced scale) — nothing in the app can fix a slow share.
     **Zero-copy BRAW preview (2026-10-22, user: "直接做0拷贝")**: the helper
     has `--zerocopy N [--mono]` (`helpers/braw/braw_zc.{h,mm}`, Objective-C++
     under ARC, built as a second object + `-framework Metal -framework
     IOSurface`; the helper itself stays plain C++ because the SDK dispatch
     source may only be included in ONE translation unit): a pool of N
     slots of GLOBAL IOSurfaces (`kIOSurfaceIsGlobal`, RGBA8), one per eye,
     each already the 2:1 CANVAS (padded / cropped rows handled in the
     copy; pad rows zero-filled to opaque black once), sized up front from
     `IBlackmagicRawClipResolutions::GetClosestResolutionForScale`. A
     frame is copied into its slot inside ProcessComplete — a Metal blit
     from the SDK's MTLBuffer on the SDK's own command queue (waited), or a
     memcpy under `IOSurfaceLock` when the SDK decodes on the CPU — and
     only `F <frame> <slot>` crosses stdout; the app returns slots with
     `R <slot>` on stdin (a reader thread feeds a free list; the main loop
     never submits a read job without a slot; stdin EOF = app gone). The
     header carries `transport: iosurface`, `slots`, `eye_width/height`,
     `pad_rows`, `mono`, `surfaces: [[L,R],…]`; `--info` says `zerocopy:
     true`. App side: `vr180_braw::zc::BrawZcDecoder` (header, `F` lines,
     `ZcReleaser` on stdin), `RetainedIOSurface::lookup(id)` +
     `wgpu_texture_from_iosurface_plane(.., RGBA8Unorm, ..)` wrap every
     slot's surfaces ONCE per helper run, `FisheyePair` gained `gpu:
     Option<Arc<GpuPair>>` (textures + a `ZcSlotGuard` that releases the
     slot when the LAST Arc clone drops), `BrawZcIter` (macOS) hands such
     pairs out with the same step / seek / canvas semantics, and the
     preview loop renders them with `Device::project_fisheye_from_texture`
     (one entry point for hequirect / fisheye-out × plain / RS — the same
     kernels, no upload) and keeps an Arc clone in
     `queue.on_submitted_work_done` so the slot is only returned after the
     GPU finished reading it (callbacks fire on later submits / polls; the
     UI thread's own submits keep them flowing while paused). The preview
     opener takes the zero-copy path whenever `info.zerocopy` (and
     `VR180_BRAW_ZC` ≠ 0) and decodes at FULL size in the reframed mode
     (a zoomed viewport samples native pixels), the cap-based scale
     otherwise; the still / export keep the pipe. **Verified
     (`examples/braw_zc_check`, local C006)**: every slot surface is
     byte-identical to the pipe path's canvas (pad rows included) at
     eighth / quarter / full, the projection from the surface texture is
     byte-identical to the upload path, and the iterator runs 1141 / 328 /
     **96 fps at eighth / quarter / FULL 12K** (pipe path: 437 / 145 / 26)
     — a 60p clip's reframe preview at full 12K needs 30. Pool of 4 and a
     bounded(8) pair channel: the worker can only run ~3 pairs ahead (the
     helper waits for releases), which is fine. Global IOSurfaces are a
     deprecated-but-working route (both ends are our unsandboxed
     binaries); the mach-port hand-off is the fallback if Apple removes it.
     **In-app (same day, local C006, vsync-off build)**: half-equirect at
     eighth: render 0.8–5.8 ms per frame; reframe at FULL 12K: decode
     wait 4–33 µs, project 0.3–0.7 ms, compose 0.3 ms, render 0.8–1.2 ms,
     no render-bound skips — the per-frame cost quantised to display
     refreshes is gone too (the `AutoNoVsync` present mode works; the
     earlier `NativeOptions::vsync` attempt was inert). First build
     deadlocked at frame 0: the helper topped up its decode queue (waiting
     for a free slot) BEFORE reporting the finished frame, while the app
     returns slots only as it moves on to frames it has been told about —
     report first, then top up, and keep two slots for the app
     (`ahead = min(3, slots − 2)`, GUI pool 5).
     **Preview perf picture (same day)**: decode wait in the GUI log is ~0
     until the worker falls behind; the CPU helper's speed swung 105 → 15
     fps at the same frames with the laptop at 7 % battery + screen sharing
     (3.3× CPU time per frame), which is what the user saw as "卡"; the
     structural costs are the decode scale (`for_eye_width` uses NOMINAL
     divisors, so a 60p 12K clip gets quarter = 2048 px where the SDK's real
     eighth = 1024 px would meet the 896-px cap: 4× the pixels), decimation
     AFTER decode (BRAW is intra-only: a `--step` would halve the work), the
     swizzle (8–17 ms/frame at 2048²), per-frame output-texture allocation +
     32 MB upload on the GPU path (project 7–24 ms + compose 0.3–20 ms per
     frame at 2048²); the 2:1 canvas pad is < 1 ms/frame and +20 % upload
     on C006 only. Open: eighth via `resolution_scales`, `--step`, RGBA out,
     cached output textures.
     **SDK colour controls (2026-10-22, user: "暴露 braw helper 里自带的
     调色选项，隐去 color tab 里的重复功能")**: the Color panel shows a
     "Blackmagic RAW" section for BRAW clips (`draw_braw_color_section`,
     app.rs) instead of the Tone (CDL) and White balance sections, which
     duplicate the SDK's development and are hidden + forced neutral in
     `finish_load` for this camera (Matching Eyes and the 3D LUT stay).
     Controls = the SDK's clip attributes (`ColorScienceGen` 4/5/6 — Gen 6
     exists in SDK 6.0 —, `Gamma` / `Gamut` from the SDK's own lists: 17 /
     16 entries on the PYXIS, tone curve `Contrast` 0.002–1.998,
     `Saturation` 0–4, `Midpoint` 0–1, `Highlights` 0–1.998, `Shadows`
     0.002–2, `BlackLevel` −1–1, `WhiteLevel` 0–2, `VideoBlackLevel`,
     `HighlightRecovery`, `GamutCompressionEnable`) + frame attributes
     (`WhiteBalanceKelvin` 2000–50000, `WhiteBalanceTint` −150–150 — the
     C006 clip's own tint is 10, so "as shot" ≠ 0 —, `Exposure` ±5 stops,
     `ISO` from `GetISOList`: 125…3200). Lens shading / distortion / CA are
     read-only 0 on this camera and not shown. Data flow: `--info` gains a
     `color` block (values as `[value, min, max]`, lists; frame 0 read via
     the shutter callback's `read_frame_color`) → `BrawInfo.color:
     Option<BrawColorInfo>` → `ClipInfo.braw_color` (Arc) for the panel;
     the user's overrides are `Settings.braw_color: BrawColorParams` (all
     `Option`, `None` = as shot; persisted per kind like the rest), sliders
     show the EFFECTIVE value and mark overrides with "•", "Reset to clip"
     clears them. Decode: `DecodeOptions.color` → `--decode` flags (the old
     per-field options gamma/gamut/wb/tint/exposure/iso are gone; new flags
     `--csgen N`, `--tc-video-black 0|1`; the black-level sentinel became
     −9999 because −1 is a legal black level); the helper's argv and the
     zero-copy stdin command `P <flag> <value…>` share one parser
     (`apply_color_option`), `apply_clip_params` applies the clip-level
     fields to the clip's attributes AND to the cloned per-job overrides.
     Live path: `FisheyePairIter::apply_braw_color` (default no-op;
     Segmented/Denoising forward; `BrawFisheyeIter` restarts the helper at
     the next pull; `BrawZcIter` resolves overrides against the clip's
     values and sends only the CHANGED effective values as `P` lines —
     `BrawColorValues::changed_lines` — so a cleared override sends the
     clip's own value back; the helper drains its in-flight frames, applies,
     then the app's `S` re-decodes the frame on screen); the GUI worker
     detects `settings.braw_color` changes in its step-4 block and does
     `IterCmd::BrawColor` + a seek to the displayed frame (same gen
     bookkeeping as a seek). Stills: `DetailCache` requests carry the
     params and the cache key includes them; export: `FisheyeExportConfig.
     braw_color`. The embedded "Gen 5 Film to Extended Video" LUT expects
     the Film gamma: the panel switches it off when another gamma is chosen
     and back on for Film (`BrawColorInfo::is_film_gamma`), with a note
     under the LUT section + a "Use the clip's embedded LUT" button.
     **Verified (`examples/braw_color_check`, C006 frame 600, eighth)**:
     live `P` exposure +2 / Rec.709 gamma+gamut / tone curve + WB + ISO are
     BYTE-IDENTICAL to a fresh helper started with the same flags (both
     eyes), resetting every override is byte-identical to the as-shot
     frame, and the pipe iterator after `apply_braw_color` + seek equals the
     zero-copy frame. Trap met on the way: a regex that was meant to delete
     the old argv colour lines in `main` matched the SAME lines inside the
     new shared parser first — frame-level `P` options were "ignored" while
     argv still worked (clip-level ones did apply): always assert the
     anchor's location, not just its count. `BRAW_HELPER_DEBUG=1` prints the
     per-frame overrides the helper applies.
   - **File facts**: gyro+accel 1022 Hz, rad/s and m/s², NO timestamps
     (sample 0 = frame 0 start, rate is the clock); no lens metadata at all
     (`lens_type` etc. empty, distortion arrays zero); shutter is AUTO and
     varies within a take (C005 0.4–3.3 ms, C006 1.0–2.6 ms) → per-frame
     from `--shutters`; `crop_origin` = [24, 12] for BOTH window heights, so
     it does not place the window on the sensor (assume centred); clip is
     Blackmagic Design Film Gen 5 / BMD gamut with the display LUT embedded
     (`post_3dlut_mode` Disabled); audio PCM 24-bit stereo.
   - **Pipeline**: `probe_video` routes `.braw` to `BrawInfo` (frame = both
     eyes, fps, frame count — ffprobe pads the size and fails on URSA
     files); `SourceKind::BlackmagicRaw` now `has_frame_imu` + `is_braw`;
     `BrawFisheyeIter::new_with_options(path, info, bit_depth, scale, look,
     swap, mono, limit)` splits the frame into L|R halves (single-track = a
     VR180 lens on one sensor; dual-track = the helper's composition) and
     **fits each eye on a SQUARE canvas of its own width (the pair = a 2:1
     frame; user decision 2026-10-19, the web tool's / DeoVR's handling of
     a non-square eye): a shorter window is padded black top and bottom
     (12288×5112 → 6144² eyes, 516 rows each side), a taller one is cropped
     (12288×6480, 17:9 → 6144², 168 rows each side); `canvas_pad_rows`,
     `BrawInfo::eye_dims / frame_dims` = the canvas, `raw_eye_dims /
     raw_frame_dims` = as recorded, `canvas_pad_rows()`; the probe, the
     layout's circles (shifted by the pad), the preset's pp (`cy = Cy ×
     canvas h` exactly), the Override seed and the per-row RS table all
     speak the canvas; the canvas and the window share their centre row so
     the pose's content time (centre row of the RECORDED window) is
     unchanged, and the per-row sweep spans canvas rows × line time
     (`BrawLayout::canvas_readout_ms`: 6144 × 3 µs = 18.43 ms for both
     windows — padded rows carry no image, cropped rows are gone, the slope
     per row stays the sensor's). A mono clip stays the recorded frame.**
     Keeps 16-bit as RGBA64LE, seeks by restarting the helper at `--start`;
     the user's swap goes through the iterator; `braw_imu.rs` rewritten:
     `build_braw_imu(info, gyro, shutters, fps, n, eye_w, eye_h, circles)` →
     `DjiOsvImu` via `integrate_orientation_samples` (VQF bias), content time
     `i/fps + readout/2 − shutter(i)/2 + latency`, `HR_PER_FRAME` blocks,
     `readout_ms` from the file, `BlockMid`; export/preview/still/auto-align
     all take the common `compute_dji_stabilization` arm with per-row RS;
     `resolve_export_braw_imu`, `load_braw_imu` (cached + `seed_detected_calib`).
     **IMU basis measured** (`VR180_BRAW_BASIS` sweep of all 48 axis-aligned
     candidates, camera-locked 2 s exports, residual inter-frame motion at
     512 px): `diag(−1, 1, −1)` ≡ `diag(1, −1, 1)` (the Python app's old C)
     0.38 px vs 1.53 no-stab, 2.02 identity, every other ≥ 1.16 →
     `BRAW_IMU_TO_CAM`. Gyro latency `BRAW_GYRO_LATENCY_MS` = **+4 ms**
     (`VR180_BRAW_LATENCY_MS` sweep −12…+12 on two C006 windows: shallow
     minima +3…+5 (0.356–0.360 vs 0.380 at 0, 0.58 at −12) and +1…+5
     (0.134 flat)). Checked again on each clip's FASTEST 2 s window
     (`|ω|` 77 °/s at C006 34 s, 48 °/s at C005 7 s): no-stab 7.9 / 4.7 px,
     identity 3.8 / 1.9, this basis +4 ms 2.29 / 0.99 (latency −4 / 0 / +8:
     2.38 / 2.36 / 2.31 and 1.05 / 1.01 / 1.03). Readout stays the file's
     (rows × 3 µs): the RS-scale sweep is nearly FLAT (C006 0×…2×: 2.35 /
     2.26 / 2.29 / 2.35 / 2.33; C005: 0.98 / 0.98 / 0.99 / 1.01 / 1.00) —
     the residual on these windows is the unknown lens model (preset 210°,
     k = 0), not timing, so the line time is used as written. Also smoke-
     tested the two-track URSA Cine Immersive (`~/Downloads/A002_07241548_
     C002.braw`, 2 × 8160×7200 @ 60, 1144 Hz gyro, readout 10.8 ms, 4.17 ms
     shutter): decodes composed, stabilizes, muxes audio; its reduced-scale
     track resources are LARGER than w·h·bpp (6,537,216 for 1360×1200×4 —
     trailing padding, rows packed) so the helper's size check is `≥`.
     Probe knobs: `VR180_BRAW_DECODE_SCALE=half`
     (export decode at 2/3 for the headless probes), `VR180_PROBE_AUDIO`,
     `VR180_RS_SCALE`.
   - **Lens = manual (user decision: no in-camera calibration)**: the file
     seeds the Override UI from the IMAGE CIRCLES (`circles.rs`:
     `detect_sbs_circles` on a quarter-size decode of frame 0 — rim level from
     the eye's corners (log black ≈ 25/255), edge points per row/column, LS
     circle with outlier rejection; C005 (2944, 3044) r 3205 / (3175, 2984)
     r 3209, C006 (2942, 3040) r 3202 / (3188, 2985) r 3206 — CANVAS px
     (C005 cropped −168 rows, C006 padded +516; the two lenses sit ~60 px
     apart vertically and 160 px outward of the eye centres) + the preset's FOV (`fx = eye_w / fov_rad`, the Override UI's
     convention — the circle only locates the centre) and k; the user sets
     FOV / k / k5 / p by hand (k5/p shown for BRAW, the Camera preset combo
     offers the Blackmagic presets + Custom, the Gyroflow loader stays).
     The Override UI's "In-file" line shows the seed. Templates: "braw".
     **The user's lens calibration (2026-10-19)** is the preset
     "Blackmagic PYXIS 12K VR180 (dual fisheye)" (`BRAW_PRESET_NAME`, the
     catalog's LAST entry — Custom stays at index 7): k1..k4 =
     −0.002605415285, 0.001226067608, −0.000347504137, 0.000413900418; the
     per-eye principal points were given normalised to a 12288-wide 2:1
     frame with the 60p 12288×5112 video centred in it, origin top-left,
     **LEFT eye in the RIGHT half**: CxL 0.76546 CyL 0.50049 / CxR 0.24243
     CyR 0.49821 → stored as `stereo_pp` offsets from each eye's centre in
     units of the frame width (L +0.01546, +0.000245; R −0.00757,
     −0.000895), `left_eye_is_right_half: true` seeds `fisheye_swap_eyes`
     for fresh BRAW clips; `preset_principal_point(preset, output, swap,
     w, h)` + `preset_rescale` apply the preset's pp / FOV / k in BOTH
     resolvers and in `seed_detected_calib` (so Auto == the Override seed).
     Against our detected circles (C006 right half (3188, 2469), left
     (2942, 2524)) the user's points sit +74/+90 and +37/+21 px — the C006
     fit is the truncated-circle one. **FOV convention**: the user's
     192.709° comes from the "fisheyecenter" web tool
     (rest.s3for.me/marat/silvers_fix/fisheyecenter/indexn.html): the FOV
     spans the per-eye frame's LONGER dimension through the KB model,
     `f = (max(w,h)/2) / θd(fov/2)` (DeoVR's convention for cropped
     fisheyes). The preset keeps that number with
     `fov_convention: KbLongerDim`; `CameraPreset::app_fov_deg(eye_w,
     eye_h)` converts to the app's eye-width convention everywhere a
     preset FOV becomes a focal (`seed_lens_calib`, `preset_rescale`):
     f = 3072 / θd(96.35°) = 1788.8 px → **196.8°**. The residual sweep with
     their k + pp agrees within its shallow floor (192.7° 1.37, 195° 1.32,
     198° 1.32, 202° 1.30, 205° 1.33, 210° 1.41; our circles + their k
     1.28 at 201.4, + k = 0 1.17 at 201.4). Templates holding "Custom" / the old
     "Blackmagic Pyxis 12K" migrate once to the new preset at load
     (unless an Override is on). The single-camera "Blackmagic Pyxis 12K"
     preset (210°, mono) remains for one-circle clips.
   - **Colour**: decoded as shot (Film log) in 8-bit (preview) / 16-bit
     (10-bit + ProRes exports; the old path fed RGBA8 to the 16-bit kernels);
     the clip's embedded LUT is written to the temp dir as
     `Gen 5 Film to Extended Video (embedded).cube` by the loader
     (`LoadedMeta.braw_lut`) and auto-applied like the built-ins (the
     `" (embedded).cube"` suffix counts as a built-in for the clip-switch
     rule). `--look` renders exist in the helper for a direct video/709
     output but are not used.
   - **Audio**: `finalize_braw_audio` — helper WAV (whole clip) → the export
     window as stereo AAC (`audio::stereo_to_aac`, new) → mux. FOUND + FIXED
     on the way: a WAV's channel layout has an EMPTY mask ("2 channels"), and
     swresample then rejected every frame with "Input changed" — all audio
     decode sites normalise the layout (`normalized_layout`) and stamp the
     decoded frames. `one_pass_audio_eligible` = false for BRAW.
   - **Preview**: decode scale from `max_decode_side_for_fps` through
     `DecodeScale::for_eye_width` (12K @ 60 → quarter = 2048-px eyes);
     progressive loader arm = non-progressive like Kandao (gyro + shutters
     + circles ≈ 0.5 s).
   - **Review pass (5-angle adversarial review of 73b09ab, 29 findings
     kept, fixed in the follow-up commit)**: the circle seed now lives in
     the PIPELINE (`braw_imu::braw_layout(path, info)` — a per-process memo
     of `detect_braw_layout`: the two half circles, or ONE centred circle
     → `BrawLayout { mono, circles }`), so the GUI loader, a headless
     export (`resolve_export_braw_imu`, which also keeps a calib-only
     preloaded carrier's lens over a rebuild) and auto-align (preloaded
     carrier first) all dewarp with the same lens (the lock residual on the
     C006 window fell 0.38 → 0.17 px with the circle seed vs the eye
     centre); a mono clip (one fisheye, the "single-camera" presets) is one
     full-frame eye shown to both (`BrawFisheyeIter … mono`, app.rs
     `braw_eye_dims`); `readout_ms_for` has an explicit BRAW arm = 0 (no
     line time → per-row RS OFF, never the Osmo table; `build_braw_imu`
     warns); the user's swap goes THROUGH the iterator like the dual-stream
     kinds (both openers pass `swap_eyes`; the Swap / Upside-down toggles
     show for BRAW); the Camera combo's preset re-scales the seed's focal
     to its FOV and supplies its k in BOTH resolvers (`preset_rescale`; the
     legacy "braw" template's "Custom" = 180° → fx × 1.167); the helper's
     `--decode` forces `post_3dlut_mode` Disabled (the app owns the LUT
     stage; `--info` reports the clip's mode), `--gyro` fails instead of
     spinning on a short read; `BrawDecoder::next_frame` treats EOF before
     the announced count / a non-zero exit as `HelperFailed` with the
     helper's stderr, `start` surfaces the `ERROR:` line when no header
     arrives; the LUT cache writes a `.partial` and validates `LUT_3D_SIZE`
     n³ rows before reuse; a gyro/accel rate or count mismatch → calib only.
   - **Not done / open**: the CPU preview decodes BRAW at 8 bits before the
     LUT (the generic CPU SBS preview does the same; exports are 16-bit) —
     a 16-bit CPU preview branch is a follow-up; the shutter term's sign
     (`−shutter/2`) and the +4 ms are indistinguishable at these 1–3 ms
     shutters — pin them on long-shutter (90–180°) pans when available;
     `IBlackmagicRawPost3DLUT::WriteCubeFile` could replace the helper's
     own cube writer; Windows has no helper (the BRAW path is macOS-only by
     construction); `--look` not exposed; the URSA hood intrudes into its
     circles (the fit still landed: r ≈ 4050 on 8160-px eyes). Examples
     (gitignored): `rs_probe_export`, `audio_window_probe`, `wav_probe`.


20. **X6 360° Audio on Windows (2026-10-10, Windows box)** — the mac
   side's ambisonic work pulled, built clean first try (the new `build.rs`
   helper rebuild is gated on `CARGO_CFG_TARGET_OS`, a no-op here); export
   matrix byte-identical through the changed audio mux. The October rig clips
   (`W:/261007 3 test/x6/VID_…_046–059`) carry the 4-channel AAC track tagged
   `4.0`; the August clips are stereo only. ONE REAL BUG, Windows-only, FIXED
   (`ab894a6`): the default STEREO export of such a clip logged "X6 360°
   Audio → stereo decode failed (open output …foa_stereo.m4a: Invalid
   argument); muxing the 4-channel track as is" — the bundled gpl-lite FFmpeg
   has `mp4`/`mov`/`wav`/`matroska` muxers but NO `ipod`, which is what
   `ffmpeg::format::output(path)` resolves `.m4a` to (CLI-verified: an `.m4a`
   output fails, `.mp4` writes); Homebrew's build has it, so the mac never
   saw it. `transcode_audio_window` now opens with `output_as(path, muxer)`
   — `"mp4"` for `AacStereo`, `"wav"` for `Pcm24` — no extension guessing on
   either platform. VERIFIED on clip 059: the stereo export logs the
   AmbiX→stereo decode and carries a 2-ch 48 kHz AAC at −28.5 dBFS RMS (the
   4-ch source copy reads −28.2; peak 3 dB lower as a front-facing downmix
   should); the `Ambisonic` track copies the 4-ch stream + injects `SA3D`
   (1 box, 4 ch). Pipeline lib tests 75/75. NOTE for the mac side:
   `stereo_audio_source` decodes the WHOLE clip (`dur_s = None`, 6.08 M
   samples for a 2 s window) and the one-pass mux trims afterwards — ~7 s
   extra on a 127 s clip; passing the export window would make it
   proportional. SAME DAY, **APAC hidden on Windows** (`cbe4d16`, GUI only;
   pipeline untouched — its `#[cfg(not(macos))]` arm already falls back:
   "APAC is macOS-only — stereo fallback", verified on 059 as a 2-ch AAC at
   −28.5 dBFS, byte-identical to the stereo export): the `AF::Apac`
   `selectable_value` is `cfg(target_os = "macos")`, and a persisted/carried
   `AudioFormat::Apac` is folded to `Ambisonic` in TWO places — at picker
   draw and in `build_export_cfg`'s mapping (the picker runs only while the
   Audio section is expanded, so the mapping is what makes it
   deterministic). `AudioFormat::Apac` + `as_str` stay for serde/match
   exhaustiveness. Stale gitignored harnesses that blocked local `cargo
   test` were parked under `examples/_stale/`.
   WINDOW (mac, same day, `c3b0a8e`): `stereo_audio_source(cfg, t_in, dur)`
   now renders only `[t_in, t_in + dur)` and returns `(src, tmp, offset)` —
   the offset the mux must apply to THAT source (0 for the rendered temp,
   `t_in` for a file / playlist); all 7 sites pass it on (the 3 Windows
   arms edited blind: construction, the `(src, offset, dur)` maps, the
   cleanups' 3-tuple patterns). `transcode_audio_window` is sample-accurate:
   decoding starts `EDGE_S` = 0.1 s early (AAC overlap warm-up), the head is
   dropped from the first kept packet's pts, the tail runs 0.1 s past the
   window. Verified: X6 export audio starts at exactly t_in (lag 0 vs the
   source, corr 0.999; before: the first packet after t_in, +10.7 ms), a 6 s
   window renders in 68 ms; GoPro passthrough unchanged (offset = t_in). **WINDOWS ARMS VERIFIED (2026-10-10, Windows box):** my explicit-muxer
   open survived the edit; clip 059 through BOTH blind-edited arms (zc
   readback = Libx265, GPU-resident = NVENC), trimmed at t_in = 5 s: the
   audio starts at **+0 ms** (10 ms RMS-envelope cross-correlation vs the
   source decoded around the window; corr 0.89) on both, 2-ch 48 kHz,
   duration 2.005 s; the window is honoured — 101 376 samples decoded
   (2 s + the 0.1 s edges) instead of 6.08 M, the zc export back from
   14.2 s to 7.6 s. OSV stereo PASSTHROUGH on the zc arm with t_in = 2 s:
   `offset 2.00s` applied, start **+10 ms** = AAC packet granularity, as
   documented for passthrough. Export matrix byte-identical; untrimmed X6
   output byte-identical to the pre-window run. Harness: the measurement
   lives in scratchpad `ev/x6audio/lag.py`.

21. **2.5.2 release (2026-10-10) — macOS PUBLISHED (tag `v2.5.2` @ `fe41028`,
   marked Latest; live `releases/latest/download/latest.json` serves 2.5.2
   darwin-aarch64); **Windows RELEASED 2026-10-10** (Windows box).** macOS side cut with
   `scripts/release_mac.sh` at `8dfa2d8`: bundle 2.5.2, helpers hash-current,
   app + DMG notarized (Accepted) and stapled, `spctl` "Notarized Developer
   ID"; update zip + `.minisig` + darwin-only `latest.json`
   (`make-latest-json.mjs`); **GitHub release v2.5.2 created as a DRAFT**
   with those four assets (tag is created on publish). Apple's first
   `notarytool` call failed 403 until the Account Holder accepted the updated
   Program License Agreement; propagation was uneven for ~10 min (a 403 can
   land mid-`--wait` while the submission is already Accepted — `notarytool
   history`/`info` + `stapler staple`). Pending: publish decision (mac-only
   manifest first, as 2.5.1, or wait for Windows), Windows setup.exe + merged
   `latest.json` via `--clobber`, memory recipe updated. **WINDOWS (2026-10-10, Windows box):** the five release commits are
   version-bump-only, so the tree verified the same day (items 20/WINDOW) IS
   the 2.5.2 code — built, exe + installer identify as "VR180 Silver Bullet"
   v2.5.2. Bundle = exe + the 7 FFmpeg DLLs + README (as 2.5.1), ISCC with
   `MSYS_NO_PATHCONV=1`, portable zip in the 2.5.1 layout. The release's
   `latest.json` was pulled down FIRST (respin hazard), notes extracted from
   it, `make-latest-json.mjs --version 2.5.2` signed + merged (mac entry
   preserved byte-for-byte). Uploaded `--clobber`: setup.exe (16 381 071 B),
   its `.minisig` (mirroring the mac script's new standalone-sig upload),
   the zip (20 883 820 B) and the merged manifest. VERIFIED live: manifest
   and installer sha256 identical to local; the live installer verifies
   against the signature IN the live manifest with the updater.rs pubkey;
   darwin entry present. Feed serves 2.5.2 to BOTH platforms.
   — NOTE: the standalone `.minisig` files uploaded to v2.5.2 (mac first,
   Windows mirrored it) are REDUNDANT: the updater verifies the artifact
   against the base64 `signature` field inside `latest.json`, which IS the
   `.minisig` content; 2.5.0/2.5.1 never shipped the file. Release assets =
   DMG + app zip + setup.exe + portable zip + latest.json, nothing else

**Most recent batch (developed on macOS, then merged with the Windows EAC work):**
- **In-process noise reduction** — `VTTemporalNoiseFilter` via objc2 FFI (no
  Swift helper); GPU-resident zero-copy P010 for OSV **and** `.360`/EAC. The
  output P010 range must match the source range (full for `.360`, video for
  OSV) or you get a uniform brightness shift. See `vt_denoise.rs`.
- **Unified export/batch UX** — a single export is a batch-of-one; one queue
  runner + one `ExportOptions`; a persistent bottom export bar
  (`draw_export_bar`) with overall progress + ETA. The separate batch and
  export-options windows were deleted.
- **Chinese (中文) localization** — process-global `UI_LANG` + `tr()`/`zh()`
  table, EN/中文 toggle, bundled CJK system font (macOS `y_offset_factor`
  0.24; Windows 0.0).
- **GoPro `.360` firmware-RS auto-detect** — `decoder::detect_rs_mode` seeds
  the per-clip RS mode from the CORI signal (`cori_indicates_no_firmware`,
  the same predicate `CoriSource::Auto` uses); the toggle still overrides.
- **Windows GoPro `.360`** — zero-copy EAC preview + GPU-resident NVENC
  export (merged Windows work) + an Inno Setup installer.

> ⚠️ Building the WHOLE workspace on macOS fails on `vr180-render` (a legacy
> CLI on a stale wgpu API) — ignore it, build `-p vr180-gui`.

The macOS feature batch that the Windows session verified (kept as the
engineering record — the checklist was what to *confirm*, not to port):

1. **DJI lens model is now exact** (matched to DJI Studio's output —
   verified against its export). Auto mode loads the per-lens FACTORY
   calibration from the OSV protobuf: fx/fy, raw cx/cy (top-left, y-down,
   NO flip), k1–k4, **k5 (field 15)**, and **Brown-Conrady tangential
   p1/p2 (field 20)**. All KB shaders run the 5-coeff radial + tangential
   (zeros ⟹ byte-identical to the old 4-coeff path, so non-OSV sources are
   unchanged). Verify: load an OSV → dewarp matches DJI Studio.
2. **Fisheye SBS output is now a normalized 195° equidistant fisheye**
   (`FISHEYE_OUT_FULL_FOV_DEG` in `fisheye_export.rs`) — the projection is
   canonical (source-lens distortion removed), disk edge = 97.5° off-axis.
   Both preview + export, incl. the GPU-resident NVENC path. Was: the
   source lens's own projection.
3. **Soft-stab rewritten** — velocity-dampened Gyroflow-style smoothing
   ported from the Python app (`dji_imu.rs::smooth_quats_velocity_dampened`):
   velocity-adaptive τ with ±200 ms look-ahead, symmetric fwd/bwd passes,
   SOFT elastic max-corr limit (no snap). New **Response** slider
   (0.2–3.0). Camera-lock (`smooth_ms = 0`) keeps the hard clamp. Defaults:
   smooth 1200 ms, max corr 15°.
4. **Eye orientation**: "Swap L↔R" now actually applies live (toggling
   auto-reloads the clip — the dual-stream iterators bind eye order at
   open), and the per-eye factory calib **follows the stream** (was a bug:
   swapped view used the wrong lens's calib). New **"Upside-down mount
   (180°)"** checkbox = exact `roll+180°` in `ViewAdjust` + implicit eye
   swap (`Settings::effective_swap_eyes()`).
5. **8K export option** (export dialog → Resolution → 8192×4096): renders
   the PROJECTION at 4096²/eye (single resample from the native source —
   not a last-step upscale). Verify NVENC accepts 8192-wide @ Main10 on
   the target GPU (4090: OK).
6. **UI/perf**: View-adjustment panel above Stabilization (default-open);
   "KB parameters" collapsible (k1–k5 + p1/p2 sliders, OSV-gated); Override
   auto-reseeds from each new clip's in-file calib; camera-preset/Gyroflow
   pickers hidden for OSV; sidebar slider tracks fixed at 150 px (do NOT
   size sliders from `available_width()` — feedback loop, panel swallows
   the preview; panel capped at 420 px); zoomed paused-frame drag re-render
   throttled to 120 ms (was per pointer event on the UI thread).

> ✅ **Windows verification done** (DX12/Vulkan + NVENC, RTX 4090): all six
> confirmed — the new lens model / 195° fisheye / soft-stab flow into the
> GPU-resident CUDA path automatically via the shared `resolve_calib_pair`
> + WGSL; GPU-resident vs portable-CPU diff is encoder-noise only (mean
> ≤2/255); 8K = 8192×4096 Main10 accepted, ~17.5 fps. One real gap found
> and FIXED on Windows: **fisheye-output RS parity** — macOS p010
> RS-corrects fisheye output (`fisheye_p010_to_fisheye_rs_16`), but the
> Windows zero-copy preview + GPU-resident export had no RS-capable
> fisheye projection. Added `fisheye_to_fisheye_rs_16` (same WGSL,
> format-patched to rgba16) + `project_fisheye_rgba16_texture_to_fisheye_rs_16`
> and wired both call sites with the same `(projection, rs)` split as the
> macOS path. Nothing to verify on macOS (the macOS paths were already
> correct); listed so the divergence isn't reintroduced.

> 🔍 **Full cross-platform parity audit (Windows session, 2026-06-10).**
> OSV preview, detail-still, color stack, preview gamma, video-range tags,
> calib (k5/p1/p2/swap-follow/195°), stab + RS: **in parity** (macOS OSV
> preview is the shared CPU worker; Windows zc worker matches it). The
> Windows **readback export path now covers Fisheye too** (same
> `(projection, rs)` split, slots 30/31) so libx265 / 8-bit /
> GPU-resident-fallback fisheye exports keep RS instead of dropping to the
> portable no-RS path. Audit items, as resolved on macOS 2026-06-10:
> 1. ~~libx265 on macOS = no RS~~ / 2. ~~ProRes = no RS on both~~ —
>    **FIXED: the PORTABLE path itself now applies per-row RS** (same CPU
>    `compute_per_row_quaternions_for_frame` build + the same rgba16 RS
>    kernels via `build_eye_eq_16(rs)`, slots 30/31; 8-bit arms use the
>    byte-path `*_rs_texture` kernels). Covers macOS libx265 (8 + 10-bit),
>    ProRes on BOTH platforms, and every future fallback — the portable
>    path is now feature-complete, just slower. Bonus: the 10-bit arms no
>    longer build-and-discard two 8-bit projections per frame (the old
>    unconditional `left_tex`/`right_tex`). Verified vs the macOS p010 VT
>    reference: stab+RS libx265 vs VT mean diff = encoder noise (see
>    `examples/rs_portable_check.rs`, gitignored).
> 3. The GoPro `.360` EAC preview (legacy) ignores the view-adjust panel on
>    both platforms (per-eye bundle predates it) — consistent, just a
>    legacy-path limitation. Still open.
>
> Two MORE real bugs found + fixed while verifying (macOS, 2026-06-10):
> 4. **ProRes-VT was broken at encoder open under ffmpeg 8** —
>    `prores_videotoolbox` rejects `yuv422p10le` (its input list: yuv420p,
>    nv12, ayuv64le, uyvy422, p010le, nv16). Now fed **AYUV64LE** (16-bit
>    packed 4:4:4 — the only listed input keeping ≥10-bit + full chroma; VT
>    converts to the profile's 4:2:2 internally). `prores_ks` keeps
>    yuv422p10le. Verified: ProRes HQ 10-bit 4:2:2 output, RS-corrected
>    (1.6/255 vs the VT HEVC reference). FOLLOW-UP the same day: ProRes-VT
>    now takes the **zero-copy P010 path** (same CVPixelBuffers as VT-HEVC
>    via `create_zero_copy_vt_prores`; profile via ctx field, codec_tag
>    left 0 so the encoder sets apch/…) — ~3 fps → **~17.6 fps** @ 8K, and
>    0.7/255 vs the HEVC reference (identical projected pixels). The
>    AYUV64LE readback route remains as the non-OSV-source fallback.
> 5. **Trimmed exports were wrong on every path**: (a) the iterators'
>    `seek(trim_in)` lands on the keyframe AT/BEFORE the target with no
>    decode-forward, so exports gained up to a GOP of pre-roll (trim 2–3 s
>    → 90 frames from 0.0); (b) the audio mux copied the FULL source audio
>    at original timestamps → over-long file AND audio offset by trim_in.
>    Fixed: all 4 export loops drop pre-trim frames (`pts < t_in − dt/2`),
>    and `mux_video_with_passthrough_audio` takes `(audio_offset_s,
>    max_duration_s)` — skips, rebases to 0, and stops at the video's
>    length. Verified: trim 2..3 s → exactly 30 frames + 1.00 s audio on
>    both portable and zc paths.

Build/run quick-ref (build `-p vr180-gui`, **not** the workspace):
```pwsh
# Windows
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"   # LLVM 17+
$env:FFMPEG_DIR    = "C:\path\to\ffmpeg-8.1-dev"   # 8.1+ REQUIRED (prores_ks_vulkan)
cargo build --release -p vr180-gui
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"; .\target\release\vr180-gui.exe
```
```sh
# macOS
cargo build --release -p vr180-gui && ./target/release/vr180-gui
```
Full Windows step-by-step (toolchain, FFmpeg, gotchas):
[docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md).

> ⚠️ The Windows-only additions (`nvenc_cuda`, the `cudarc` dep, D3D11→CUDA
> interop) are `#[cfg(target_os = "windows")]` / `[target.'cfg(windows)']`
> gated — they don't build on macOS. The secondary `vr180-render` CLI still
> has a pre-existing non-exhaustive-match error; ignore it (build the GUI).
> The verification examples under `crates/vr180-pipeline/examples/` are
> gitignored local scaffolding (several are Windows/CUDA-only).

## What this is

**VR180 Silver Bullet 2.0** — a clean-room Rust rewrite of a VR180
camera-mod processor. Single self-contained native binary per platform,
no Python. The reference implementation is the Python/PyQt6 app at
`../vr180_processor/vr180_gui.py` (the `main` branch / sibling worktree)
— **we read it to port algorithms, we don't modify it.**

- **Shipped:** a native desktop GUI, `vr180-gui` (**eframe 0.34 + egui +
  egui-wgpu + wgpu 29** — note: NOT 0.28/0.20; that stale version line hid a
  display-gamma bug, see Lesson #2). This is the product.
- **Primary source format:** DJI Osmo 360 `.osv` (dual-stream fisheye).
  Also handles SBS fisheye, Blackmagic `.braw`, and GoPro Max `.360`
  (EAC) as a legacy path.
- **GPU:** `wgpu` only (Metal on macOS, DX12/Vulkan on Windows). Shaders
  are WGSL in `crates/vr180-pipeline/src/shaders/`.
- **Video I/O:** in-process libav via `ffmpeg-next 8.1` (no shelling out
  to a system `ffmpeg`).

## Build & run

| | macOS | Windows |
|---|---|---|
| Build the GUI | `cargo build --release -p vr180-gui` | same (see WINDOWS_BUILD.md for env) |
| Run | `./target/release/vr180-gui` | `.\target\release\vr180-gui.exe` (+ ffmpeg DLLs on PATH) |
| FFmpeg | `brew install ffmpeg pkg-config` (pkg-config auto-finds it) | `FFMPEG_DIR` → avbuild dev dist + `LIBCLANG_PATH` |

First build ~3-5 min (ffmpeg-sys bindgen); incremental builds are seconds.

To launch + verify after a change, the dev loop on macOS is:
`pkill -9 -f target/release/vr180-gui; cargo build --release -p vr180-gui && ./target/release/vr180-gui &`

## Cross-platform stance — the #1 rule

Code must compile and run on **both macOS and Windows**. Two platform
paths exist on purpose, behind `#[cfg(target_os = "...")]`:

- **macOS only:** `interop_macos` (IOSurface↔Metal↔wgpu zero-copy),
  `ZeroCopyDualStreamFisheyeIter` (VT P010 zero-copy decode), the P010
  IOSurface export path — gates on **VT encode (HEVC OR ProRes-VT)**;
  ProRes-VT rides the same P010 CVPixelBuffers into Apple's hardware
  ProRes engine via `create_zero_copy_vt_prores` (~17 fps @ 8K HQ, was
  ~3 fps on the readback path). Only libx265 / prores_ks ride the
  portable readback path, which since 2026-06-10 also carries per-row
  RS. The crates `metal` / `core-foundation` / `objc` / `foreign-types`
  are `[target.'cfg(target_os = "macos")'.dependencies]` — they never
  build on Windows.
- **Cross-platform + Windows fast paths:** `d3d11va` (NVDEC) decode —
  `DualStreamFisheyeIter` for stills, `D3d11SharedDualStreamIter` for the
  **zero-copy live preview** (D3D11 P010→RGBA16→Vulkan, no CPU readback);
  `hevc_nvenc` (H.265, the **Windows default**, GPU-resident — see below)
  with a `libx265` software fallback; `prores_ks` (ProRes); the whole wgpu
  compute pipeline; GUI; audio (cpal→WASAPI); file picker (rfd). Encoder
  backend auto-selected per-OS in `app.rs::commit_export`. Windows now has
  its OWN zero-copy fast paths — it's no longer "macOS fast, Windows slow."

Rules: every cfg gate matters; `pub use` from a macOS module must itself
be gated (see `vr180-pipeline/src/lib.rs`); use `Path`/`PathBuf` (never
slash string literals); `include_str!`'d shaders + `assets/` are portable.

## Architecture (where things live)

- `crates/vr180-core` — 100% portable: gyro/quaternion math, EAC dims,
  `.cube` LUT parse, GEOC calib, segments.
- `crates/vr180-pipeline` — the engine: `decode.rs`, `fisheye_decode.rs`,
  `gpu.rs` (all wgpu kernels + the color stack), `fisheye_export.rs`
  (export pipeline + the zero-copy P010 fast path), `encode.rs` (encoder
  backends), `dji_imu.rs` (OSV stabilization + rolling-shutter),
  `interop_macos.rs` (macOS-only), `panomap.rs`, `shaders/*.wgsl`.
- `crates/vr180-gui` — the app: `app.rs` (egui UI + the full-res/zoom
  "DetailCache" + export wiring), `decoder.rs` (`Settings`, the live
  decode worker, `DetailCache`, color-stack plan, persistence),
  `audio_player.rs`, `assets/` (bundled LUT).
- `crates/vr180-fisheye` — camera presets + fisheye calib (Kannala-Brandt).
- `crates/vr180-render` — legacy CLI (currently doesn't compile; ignore).

## DJI OSV lens model (exact — don't simplify)

Empirically matched to DJI Studio's output, bit-for-bit.
Per-lens calib block in the OSV protobuf (`vr180-fisheye/src/dji_osv.rs`):
`1=fx 2=fy 3=cx 4=cy 5..8=k1..k4 10=W 11=H 15=k5 20=[p1,p2]
21=mount_quat`. Projection (in every KB WGSL shader's `project_kb`):

```
θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹ + k5·θ¹¹      # 5-coeff Kannala-Brandt
u' = u + 2·p1·u·v + p2·(r²+2u²)                        # Brown-Conrady tangential
v' = v + p1·(r²+2v²) + 2·p2·u·v                        #   (r² = θ_d²)
src = (cx + fx·u',  cy − fy·v')                        # raw cx/cy, NO flip
```

k5 keeps the radial map monotonic past ~90° (the 4-coeff poly folds at
~87–90°); the tangential is small (~2–4 px at the rim) but it's what
finally matched DJI. Both resolvers (`decoder.rs::resolve_fisheye_calib_pair`
and `fisheye_export.rs::resolve_calib_pair`) must stay in lockstep — every
calib change goes in BOTH or preview ≠ export.

## Color pipeline — match the Python app

The color tools (CDL, 3D LUT, white balance, saturation, sharpen,
mid-detail) were just aligned to the Python app **exactly**. Order
(`gpu.rs::record_color_stack` / `apply_color_stack_per_eye_16`):

> **CDL → 3D LUT → sharpen → temp/tint → mid-detail → saturation**

- temp/tint is **post-LUT** (critical for log footage: it must act on the
  LUT's Rec.709 output, not the log input). White-balance and saturation
  are split passes of the one `color_grade` shader.
- CDL clips to `[0,1]` **before** gamma (matches `build_color_1d_lut`).
- Math constants match Python: temp/tint = `R×(1+0.3t)`,`G×(1−0.3·tint)`,
  `B×(1−0.3t)`; saturation = `lerp(BT.601-luma, color, sat)`;
  shadow/highlight = smoothstep masks ×0.6 about pivot 0.5.
- The DJI Osmo 360 **D-LogM→Rec.709 LUT is bundled** (`include_str!`,
  `crates/vr180-gui/assets/`), autoloaded for OSV, cached per-thread.

## 10-bit end-to-end (when 10-bit output is selected) — non-negotiable

Every stage must hold ≥10-bit precision. NOTE: intermediates are
**`Rgba16Unorm`** (not `Rgba16Float` — the old CLAUDE.md was wrong).
- macOS: VT P010 IOSurface decode → 16-bit project → 16-bit color stack →
  P010 IOSurface encode (Main10), zero-copy.
- Windows: NVDEC P010 decode → D3D11 P010→RGBA16 → 16-bit project → 16-bit
  color stack → P010 compose → **GPU-resident CUDA NVENC** (`--profile
  main10`), zero-copy (see "Windows GPU-resident export"). Fallback:
  P010/RGB48 readback → `hevc_nvenc`/`libx265`.
- The **preview** runs the same 16-bit color stack as the export, then
  composes to an `Rgba8Unorm` SBS for egui. Don't add an 8-bit color
  shortcut into the graded path.

## Windows GPU-resident H.265 export (the fast path)

On Windows, 10-bit H.265 export runs **fully GPU-resident** — no CPU
readback — at the NVENC ceiling (~36 fps @ 8K on a 4090):

> NVDEC P010 decode → D3D11 P010→RGBA16 → wgpu 16-bit project/color/compose
> → P010 plane textures → **exportable Vulkan images → CUDA external memory
> → `cuMemcpy2D` (DtoD) into NVENC's CUDA frame** → `hevc_nvenc`.

Lives in `nvenc_cuda.rs` (`#[cfg(target_os = "windows")]`), dispatched from
`fisheye_export.rs`. `cudarc` is a `[target.'cfg(windows)']` dep. The export
uses a **dedicated** wgpu device (`gpu.rs::new_dedicated_from_adapter`) so it
can't deadlock eframe's renderer (Lesson #1). Falls back to a P010/RGB48
readback `hevc_nvenc`/`libx265` path. Both `HalfEquirect` and `Fisheye`
output modes take the fast path. The `.360` arm (`export_eac_gpu_resident`)
also carries **ProRes and libx265**: same GPU front end, encode tail by
backend — NVENC via CUDA/P010, ProRes via GPU 4:2:2 readback into
`prores_ks_vulkan` (FFmpeg 8.1+; CPU `prores_ks` fallback), libx265 via
RGBA64. Every landing on a slow path warns AND sets
`fisheye_export::export_path_note` (shown in the export bar).

**Color range:** both the GPU-resident and readback paths compose
**video-range** YCbCr and tag `AVCOL_RANGE_MPEG` (Rec.709) — the
distribution standard (YouTube VR180, headsets), matching the source and the
gamma-correct preview. `compose_sbs_to_p010_textures(…, full_range: bool)`
takes a flag but **both callers pass `false`**; the brief full-range
experiment was a band-aid for the preview-gamma bug (Lesson #2) and is gone.

## Hard-won lessons — DON'T regress these

1. **Never do wgpu GPU work on a non-main thread.** The wgpu
   `Device`/`Queue`/swapchain is shared with eframe's renderer; GPU
   submits + `Maintain::Wait` from a second thread wedge Metal's drawable
   queue against the presenting main thread → permanent 0% CPU deadlock.
   `DetailCache` decodes on a background thread (CPU/VideoToolbox only —
   it returns `Vec<u8>` pairs, never touches wgpu) and does the GPU
   project/compose on the **main** thread. Keep it that way.
2. **Preview gamma (egui-wgpu 0.34 — corrected):** egui-wgpu's fragment
   shader expects display textures that are **NOT sRGB-aware** — it samples
   them as gamma/raw bytes (and on an sRGB swapchain linearizes at the very
   end so the hardware re-encode is identity). Our SBS already holds
   Rec.709-gamma bytes, so register it with a **plain `Rgba8Unorm` view** on
   every platform (`preview_view_format` in `app.rs`; used at both
   `register_native_texture` calls). An **sRGB view double-decodes** → the
   preview shows `sRGB⁻¹(v)`, too dark. Windows' non-sRGB `Bgra8Unorm`
   swapchain made it obvious; on macOS's sRGB swapchain it's wrong via a
   different path. The old "register with an sRGB view" advice was correct
   only under **egui-wgpu 0.28** and is now stale — **do not restore it.**
3. **Detail still = exact frame, decode-forward skip:** the native decoder
   seeks to a keyframe and **decodes forward** to the requested pts (else the
   still shows a keyframe up to a GOP away). The OSV dual-stream path uses
   `DualStreamFisheyeIter::decode_forward_to`: it decodes the intermediate
   GOP frames on the HW engine but runs the HW→CPU download + swscale only on
   the **target** frame (≈11 s → ≈0.6 s for a deep seek at 8K). Same
   decode-forward in the live worker's seek.
4. **OSV gyro timing:** the sensor readout (SROT) comes from the file's
   line time (`DjiOsvImu::readout_ms`: 18.301 ms ≤ 30 fps, 16.228 ms @
   50 fps, 16.06 ms on the 360 II @ 60 fps); `dji_osmo_readout_ms_for_fps`
   is the fallback for files without one. The per-frame stab pose is
   sampled at the centre row's mid-exposure (SROT/2 + timestamp gap −
   shutter/2, all from the file — item 7 under Status) and the
   rolling-shutter readout window is centred on the same point (coupled,
   as DJI does). There is no user-facing timing control. See `dji_imu.rs`.
5. **Settings persistence** lives in a per-OS config dir (macOS App
   Support / Windows `%APPDATA%` / Linux XDG) — `Settings::config_path`.
   It survives both relaunch and loading a new clip (only trim resets).

## Porting algorithms from Python

1. Read `../vr180_processor/vr180_gui.py` (the working reference).
2. Port the math exactly; bit-validate against it when feasible. The
   camera-specific constants are already validated against real footage —
   don't re-derive.
3. NOTE for the Windows session: the deep project memory the macOS dev
   used lives in that machine's `~/.claude/...` and is **not on this PC**.
   The essential facts are summarized here and in `docs/`; if you need
   more on GEOC calibration / GoPro stream conventions / VQF, read the
   Python source directly.

## Don't

- Don't run wgpu/GPU work off the main thread (see lesson #1).
- `wgpu` is the GPU answer for compute/render. The ONE sanctioned exception
  is the Windows NVENC export, which uses `cudarc` purely to feed NVENC
  (Windows-gated); don't add CUDA/OpenCL/Numba anywhere else.
- Don't shell out to a system `ffmpeg` — use `ffmpeg-next` in-process.
- Don't remove a `#[cfg(target_os = "...")]` gate without confirming the
  API exists on the other platform; the Mac-only paths are FAST paths
  with deliberate cross-platform fallbacks.
- Don't drop the graded path to 8-bit; don't re-introduce the off-thread GPU
  deadlock; don't switch the preview back to an sRGB texture view (Lesson #2)
  or the H.265 export back to full-range.

## Doc pointers

- [docs/WINDOWS_BUILD.md](docs/WINDOWS_BUILD.md) — **the Windows handoff guide (start here on Windows).**
- [docs/BUILD.md](docs/BUILD.md) — original cross-platform build notes.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate boundaries, GPU pipeline shape.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan (note: predates the GUI; treat as historical).
