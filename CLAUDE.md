# CLAUDE.md — orientation for Claude Code sessions

Auto-loaded at session start. Read this first; deeper detail is in the
`docs/` pointers at the bottom.

## ⚡ Status: **2.6.0 macOS RELEASED 2026-11-24: tag `v2.6.0` @ `a80157f`, DMG + app zip + darwin-only `latest.json` live, the feed serves 2.6.0; Windows build PENDING** (build at the tag, `make-latest-json.mjs --version 2.6.0` merges the Windows entry from the release, upload setup.exe + portable zip + `latest.json --clobber`; no bump commit needed: all four carriers are at 2.6.0). Both notarizations were accepted on the first submit; the bundle's floor is macOS 26.0, as 2.5.6's was. It ships everything on `main` since 2.5.6: ETTR recovery for X6 I-Log clips (item 30), the colour precision of the export path (31), Match QuickTime (32), Lens Space (33), X6 raw video with its development, its DNG output and the later frames (34, 39 to 44, 46, 47), several clips as one file, projects and the list kept between launches (36), the tone controls for every clip (37), Lens Space clips joined as an Apple Immersive Video file (38), a run that keeps its speed behind other windows (45). Windows has built and verified all of it (the Windows notes under items 30 to 34 and 47, items 35 and 48). The Mac's stored regression sets ran at `79cefc2` (the last bullet of item 48); nothing but notes, the README and the version changed after that. **Previous: 2.5.6 RELEASED 2026-11-13 on BOTH platforms — tag `v2.5.6` @ `d18c9fe`; macOS DMG + app zip first, Windows the same day; the feed serves 2.5.6 to both.** WINDOWS (Windows box): built at `30ac223` (= the tag + CLAUDE.md), exe file version 2.5.6; `braw_helper.exe` rebuilt by `build_braw.ps1` (CUDA blit; `d18c9fe` touched only the macOS build script); the export matrix byte-identical per stream to the 2026-11-12 baseline; bundle = exe + the seven FFmpeg DLLs the exe imports + the helper + README (now says NVENC stops at 8192 a side); the bundled exe loads its own DLLs and opens a BRAW clip on the zero-copy path with no console window. setup.exe 17 027 040 B (sha256 a0ff0176…), portable zip 21 739 462 B. `make-latest-json.mjs --version 2.5.6 --notes ""` (keeps the Mac's notes byte-identical — the release body from `gh` gains a trailing newline), darwin entry byte-identical; uploaded `--clobber` setup.exe + zip + merged `latest.json` (no .minisig); live assets sha-identical to local, the live installer verifies against the signature in the live manifest with the updater.rs pubkey, and the stable `releases/latest/download/latest.json` served the merged manifest within a minute this time. Cut found that the Swift / RAW helpers built without a deployment target take the build machine's macOS as their minimum (27.2 here; 2.5.5's APAC helper said 28.0), so they could not start on an older system: `d18c9fe` targets macOS 14 for `apac_encode` (like `mvhevc_encode`) and 26.0 for `braw_helper` (`MIN_MACOS` in `build_braw.sh`), the floor the bundled Homebrew libraries already set (their minos is 26.0, so the app as a whole needs macOS 26). 2.5.6 ships what follows: **(was UNRELEASED on `main`): Insta360 X6 7.4K / 7.4K Crop / 10K / 10K Crop recording modes + Crop mask + the few-dropped-frames rate fix (item 28) — Windows BUILT + VERIFIED 2026-11-12 (every mode on every arm, the masked shaders on Vulkan / D3D11 zero-copy / DX12; an oversize-export note fixed there); spatial video export of the reframed view, macOS only (item 29) — builds on Windows, never offered there.** **2.5.5 RELEASED 2026-11-07 on BOTH platforms — tag `v2.5.5` @ `90f11b7`, macOS and Windows live on the updater feed (item 27) — it ships everything since 2.5.2: X6 per-frame eye-exposure matching (item 15), GoPro exposure timing from SHUT (item 16), Kandao QooCam 3 Ultra (items 23/24), Blackmagic RAW rebuilt + on Windows (item 25), dropped-frame tables + pts-paced preview (item 26).** **2.5.2 RELEASED 2026-10-10 on BOTH platforms (item 21).** Previous: 2.5.1 — macOS RELEASED 2026-09-25 (tag v2.5.1 @ 7e345ee: DMG + mac app.zip + latest.json live, feed serves 2.5.1); **Windows RELEASED 2026-09-25, RESPUN same day with the version-free product name** (setup.exe + portable zip + MERGED latest.json live on v2.5.1; the installer, shortcuts, Add/Remove entry and exe now say "VR180 Silver Bullet" — AppId and the settings dir deliberately unchanged, `[InstallDelete]` clears the old-named shortcuts on upgrade; live manifest byte-identical to local, installer minisig verified against the updater.rs pubkey; feed serves BOTH platforms). Previous: 2.5.0 RELEASED 2026-09-13, macOS respun 2026-09-14 (adds 3D display output, Snapshot, generic-SBS input toggle, Flat 3D bitrate defaults; the 2.5.0 Windows respin was never done — superseded by 2.5.1) — reframed mode, file-derived stabilization timing, Insta360 X6

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
   (`FisheyeExportProjection::Reframe { hfov_deg, defish }`)
   writes the exact viewport at a per-eye size (1080/1440/2160 lines ×
   aspect → 2:1 or 32:9 SBS; since 2026-11-10 also 3840 lines for the 1:1
   view only — 3840² per eye, 7680×3840 SBS; `ReframeSize::clamped` drops a
   size the view does not offer to the largest one below it) and never
   tags VR180 metadata. Since
   2026-11-10 the same view also exports as an Apple spatial video
   (MV-HEVC, macOS) — item 29.
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
   (`export_still`: per half develop → `Device::project_eye`
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
   `FisheyeExportProjection` (Fisheye → `project_eye` with `fisheye_out`,
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
     preview loop renders them with `Device::project_eye`
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
     **Follow-up (same day, user: "tone curve 选项不生效 … 需要全分辨率导出
     prores 以及 hevc 的手段")**: (1) the SDK applies its tone-curve
     attributes ONLY under the gamma "Blackmagic Design Custom" — measured
     with `--tc-contrast 1.9` / `--tc-black-level 0.5` on C006: no change at
     all under Film, Video, Extended Video or Rec.709, mean 66→33 / 162 under
     Custom (whose default curve is also brighter than Extended Video: 95 vs
     50); highlight recovery and exposure work under every gamma. Like
     Resolve's Camera Raw panel, moving a tone-curve control now switches the
     gamma to Custom (note shown while it is not; the embedded Film LUT goes
     off with it). (2) Full-resolution export: `ExportResolution::Native`
     already exports the clip's own per-eye size (a 12K BRAW → 12288×6144
     SBS) but was labelled with the Osmo's "7680 × 3840"; the Resolution
     combo and the export bar now show the clip's real native size
     (`App::native_sbs_dims`). Encoder limits measured with ffmpeg 8 on the
     M5 Max: `hevc_videotoolbox` takes 12288×6144, 12288×6480 and even
     16384×8192 (Main / Main10); `prores_videotoolbox` refuses anything above
     8192×4096 (`-12903`, no compression session); `prores_ks` and `libx265`
     take 16K. `export_fisheye` therefore tries the VT zero-copy encoder and,
     when VT refuses the size, drops to the readback path with the software
     encoder of the same codec (`prores_ks` / `libx265`; the flags are
     `let mut`, `encoder_backend` carries the choice; hints under the
     Resolution combo). **Measured (`examples/braw_export_check`, C006, 10
     frames at native 12288×6144)**: HEVC Main10 VT zero-copy 0.22 s/frame;
     ProRes HQ → prores_ks fallback 0.59 s/frame (readback 453 MB/frame +
     CPU ProRes), both files probe at 12288×6144, and the ProRes run with
     Custom gamma + contrast 1.9 (via `cfg.braw_color`) has mean luma 44
     vs 65 as shot — the overrides reach the export. The pipe decode of a
     12K 16-bit frame is NOT the bottleneck at this size (~2 GB/s). Zsh
     trap met while probing: an unquoted `$VAR` is one word in zsh (no
     SH_WORD_SPLIT), so `helper $FLAGS` passed "--exposure 2" as a single
     argument and `size=$SZ:rate` applied the `:r` modifier — use bash for
     such loops.
     **Tone-curve semantics (same day, user: "shadows 和 highlights 是全部
     的调整范围么？效果非常轻微，shadow 数值越低暗部越亮")**: the sliders span
     the SDK's full accepted range (`GetClipAttributeRange`), the small
     effect is the controls' nature — `ToneCurveHighlights` / `Shadows` are
     Resolve's "Highlight rolloff" / "Shadow rolloff" (knee / toe SHAPE of
     the brightest ~1 % / darkest ~5 %), measured under Custom on C006:
     shadows 0.002 → p5 50→54, shadows 2 → p5 39 (lower = softer toe =
     deepest shadows lifted — the user's observation is the SDK's
     semantics), highlights 0 → p99 186→182, no change above 1; the level
     controls are strong (midpoint 0.2 / 0.8 → mean 23 / 170, black level
     ±0.2 → mean 69 / 120, white level 0.6 / 1.4 → 45 / 145, contrast
     0.5 / 1.998 → 110 / 27). Renamed in the panel with hover text. KEY
     SDK FACT: `IBlackmagicRawToneCurve::GetToneCurve(camera_type, gamma,
     gen)` (QueryInterface on the codec) returns the tone-curve parameters
     each gamma is BUILT from — PYXIS 12K Gen 5: Film = contrast 1, sat 1,
     midpoint 0.383562, hl 1, sh 1, black 0, white 1, videoBlack 0;
     Extended Video = 1.58, 1.4, 0.409008, 1, 1.25, 0, 1, 0; Custom's
     defaults = 1, 1, 0.409008, 1, 1, 0, 1, 0 (Gen 4 Extended Video = 1.4,
     1.25, 0.409, 0.6, 1.8, videoBlack 1; Video / Rec.709 answer
     E_INVALIDARG; Gen 6 has Film + Custom only). Custom seeded with a
     gamma's preset REPRODUCES that gamma (Extended Video: 180 of 7 M bytes
     differ by 1; Film: same mean) — which is why Custom's own defaults
     looked flat and bright (contrast 1 vs 1.58). The helper's `--info`
     colour block now carries `tone_presets` (per gamma, `[contrast, sat,
     midpoint, hl, sh, black, white, videoBlack]`) → `BrawColorInfo::
     tone_presets` / `tone_baseline(gamma)`; the panel shows the CURRENT
     gamma's preset as the sliders' baseline and, on the first tone edit or
     on picking Custom in the combo, makes the untouched controls explicit
     at that preset (`seed_tone_curve`) before switching — the picture
     continues from where it was and only the moved control changes.
     `EvaluateToneCurve` exists (17-point array) but its axis convention
     did not match the measured direction of the shadows parameter, so no
     curve preview was built from it. `GetToneCurveForCustomGamma` is
     mentioned in the header's comment but does not exist in SDK 6.0.
     **Colour science generations (same day, user: "选择 Gen 6 之后 behavior
     不太正常")**: the gamma / gamut LISTS, the SDK's default gamma / gamut
     and the tone presets belong to a GENERATION. Measured on C006:
     `SetClipAttribute(ColorScienceGen, 6)` drops every "Blackmagic Design"
     gamma (Film, Video, Extended Video, Custom) and the Blackmagic Design
     gamut — 13 gammas / 15 gamuts remain — and the clip's gamma / gamut
     become "DaVinci Intermediate" / "DaVinci Wide Gamut" (Gen 6 = the
     DaVinci-wide-gamut colour science; its decode of the Film clip has
     mean 48.6 vs Film's 65.8); Gen 4 and 5 share the 17 / 16 lists and
     the BMD defaults. Any Film / Custom gamma override is REJECTED under
     Gen 6 (E_INVALIDARG, `set_clip_string_attr` warns), so the old UI kept
     showing Gen 5's list and "Film", the tone curve could not switch to
     Custom, and the Gen 5 Film→Extended Video LUT stayed on top of a
     DaVinci Intermediate decode — the "abnormal behaviour". Fix: `--info`'s
     colour block now has `gens: {"4": {gamma, gamma_list, gamut,
     gamut_list, tone_presets}, "5": …, "6": …}`, read on a CLONE of the
     clip's attributes per generation (`CloneClipProcessingAttributes` +
     `SetClipAttribute(gen)` — the clone switches its own gamma like the
     clip would); `BrawColorInfo::for_gen(gen)` (view; the clip-level block
     stands in for an unknown gen), `BrawColorParams::effective_gen /
     effective_gamma / effective_gamut` — an override the generation does
     not accept is NOT pretended, the generation's own default applies;
     `resolve` → `changed_lines` sends `--csgen 6 --gamma DaVinci
     Intermediate --gamut DaVinci Wide Gamut` and a fresh helper given only
     `--csgen 6` lands on the same picture (explicit defaults reproduce each
     gen's default decode byte for byte: gen 4 / 5 / 6 checked). Panel:
     combos list the chosen generation's gammas / gamuts, an override the
     new generation rejects is dropped on switch, a note names the change,
     the tone-curve section is disabled under a generation without the
     Custom gamma, `tone_baseline(gen, gamma)`. Verified live
     (`braw_color_check` step 3b): Gen 6 on the running helper is
     byte-identical to a fresh `--csgen 6` helper, and the reset back to
     Gen 5 is byte-identical to the as-shot frame.
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
     own cube writer; `--look` not exposed; the URSA hood intrudes into its
     circles (the fit still landed: r ≈ 4050 on 8160-px eyes). Examples
     (gitignored): `rs_probe_export`, `audio_window_probe`, `wav_probe`.
   - **WINDOWS (2026-11-07, Windows box) — BRAW works on Windows.** The
     helper source is platform-neutral through `helpers/braw/braw_platform.h`:
     `SdkString` + `sdk_str` / `std_str` / `sdk_str_free` wrap CFStringRef
     (macOS, byte-for-byte the old code) or BSTR (Windows); `SdkVarType`
     (the SDK enum / VARTYPE) for `SafeArrayGetVartype`; `SdkBool` (bool /
     COM BOOL) for the out-flags of `GetClipAttributeList` & co. — the ONE
     class of compile error the port hit; `sdk_library_dir()` (mac: the
     SDK Libraries dir; Windows: `BRAW_SDK_LIB_DIR`, the SDK's
     `Win/Libraries`, the RAW Player / Speed Test runtime, the exe's own
     dir — the first holding BlackmagicRawAPI.dll); `BRAW_GPU_PIPELINE` =
     Metal on macOS, **CUDA on Windows** (device iteration, `SetFromDevice`
     and the resource-manager readback are the same SDK calls; only the
     pipeline enum differs); `platform_init_args` (Windows: binary
     stdout/stdin, `CoInitializeEx`, UTF-8 argv from `CommandLineToArgvW`);
     `--zerocopy` is refused off macOS (inert `zc_pool_*` stubs keep the
     decode loop one source) and `--info` reports `zerocopy: false` there;
     `--pipeline` takes `auto|metal|cuda|cpu` and refuses the other
     platform's name. Every helper spawn in `vr180-braw` goes through
     `helper_command` (2026-11-07): `CREATE_NO_WINDOW` on Windows — the GUI
     has no console and the helper is a console-subsystem exe, so a bare
     `Command::new(&helper)` opened a Windows Terminal window on every
     probe / gyro / LUT / decode (verified by window enumeration: WT +
     PseudoConsoleWindow without the flag, nothing with it; the export's
     `df` probe got the same flag). Never spawn the helper with a bare
     `Command::new`. The Windows SDK ships the API as an IDL:
     `helpers/braw/build_braw.ps1` runs MIDL (header + the `_i.c` IIDs,
     linked in), compiles the helper with `/FI` on the SDK's dispatch
     source (its LoadLibrary of BlackmagicRawAPI.dll) and links
     ole32/oleaut32/shell32 — needs Blackmagic RAW **6.x with the SDK
     component** (this box had only the 3.6.1 runtime from Resolve / the
     Player until the user installed it). Rust: `vr180_braw::helper_file_name()`
     = `braw_helper.exe` on Windows. VERIFIED on the user's PYXIS 12K clip
     `W:/Pyxis braw stab footage/A001_04071606_C006.braw` (34 GB over SMB,
     12288×5112 @ 60, 2894 frames): `--info` 0.5 s (gyro 49 295 @ 1022 Hz,
     readout 15.336 ms, 51.3° shutter, embedded "Gen 5 Film to Extended
     Video" LUT, pipeline cuda / RTX 4090), `--gyro` 0.2 s, `--audio` 0.2 s
     (2 ch 24-bit WAV), `--shutters`; decode CUDA vs CPU of the same 3
     quarter-scale frames 0.84/255 mean, max 14 (the two SDK pipelines'
     own rounding); throughput over SMB: CUDA full 12K **71 fps**, half
     171, quarter 381; CPU quarter 651 (the SDK's partial decode is cheap,
     CUDA pays a ~1 s start-up). The GUI opens it (circles found, LUT
     extracted, preview at eighth/2048×852 step 2 on CUDA, gyro
     integrated, first frame up; plays at the 30 fps preview cadence — ~0.6 ms
     per frame off the pipe, 2–3 ms render) and `export_smoke` 5–6 s through the
     portable loop writes 60 frames @ 60 fps, 3072×1536 yuv420p10 from the
     16-bit CUDA decode, stab engaged, audio muxed (1.79 fps at 12K — the
     readback pipe + libx265; a reduced-scale decode for sub-12K eyes and
     NVENC would be the speed-ups). The frame matches the mac's own
     `A001_04071606_C006_SBS.mp4` of the same instant.
   - **WINDOWS ZERO-COPY PREVIEW (2026-11-07, same day).** The IOSurface
     hand-off has a D3D11 twin: `helpers/braw/braw_zc_win.cpp` implements the
     same `braw_zc.h` pool with SHAREABLE D3D11 RGBA8 textures (NT handles,
     `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`), one per eye per slot on the same
     square canvas / pad rows (`row_map` verbatim), created on the adapter
     the app names (`--adapter-luid <hex>` = its Vulkan device's LUID — a
     cross-adapter handle cannot be imported) else the first hardware
     adapter. Fill, CUDA pipeline (`zc_pool_fill_gpu`, compiled when the
     CUDA toolkit headers are present — `BRAW_ZC_CUDA`, `build_braw.ps1`
     finds them via `CUDA_PATH`; `nvcuda.dll` is loaded at runtime and its
     `cuDeviceGetLuid` must equal the pool's adapter): the SDK's processed
     image is a CUDA device buffer, and `cuGraphicsD3D11RegisterResource`
     REFUSES shared textures (CUDA IPC is Linux-only), so each slot has a
     non-shared RGBA8 STAGING texture per eye registered with CUDA once; a
     fill is `cuMemcpy2DAsync` device → array (the eye's half of the frame
     straight into the canvas rows, pad rows pre-zeroed) + a CUDA event
     wait, then a D3D11 `CopyResource` staging → shared slot and an
     event-query wait, all BEFORE the "F" line (the app reads the memory
     through Vulkan the moment it sees the line). CPU pipeline / no CUDA
     headers: `UpdateSubresource` from the SDK's host buffer, or (CUDA
     decode without the blit) a resource-manager readback first
     (`BRAW_ZC_HOST_FALLBACK`; macOS keeps a failed blit a failed frame).
     Per fill at 6144² eyes (`BRAW_HELPER_DEBUG=1`): map 0.1 ms, copy
     enqueue 0.1, event wait 5.3–5.5, D3D copy + fence 0.4. The
     header gained `transport` ("iosurface" | "d3d11") and `helper_pid`;
     `surfaces` carry 64-bit tokens (IOSurface IDs / NT handle values —
     `zc_pool_handle`, `zc_transport`, `zc_owner_pid`, `zc_set_adapter_luid`
     exist on both platforms; mac side: `braw_zc.mm` gained the four
     one-liners + `<unistd.h>`, rebuild with `build_braw.sh`). App:
     `vr180_braw::zc` is cross-platform (`EXPECTED_TRANSPORT`,
     `BrawZcDecoder::start(.., adapter_luid)`), `BrawZcIter` is one source
     for both (`ZcEyeTexture` = `IOSurfacePlaneTexture` /
     `interop_windows::ImportedSharedTexture`; `new` takes the adapter and
     builds a `VulkanImportCtx` on Windows), and
     `VulkanImportCtx::import_foreign_rgba8` duplicates the helper's handle
     (`OpenProcess` + `DuplicateHandle`; `Win32_System_Threading` feature),
     validates it with `vkGetMemoryWin32HandlePropertiesKHR` (a wrong-adapter
     handle reports instead of panicking) and imports it once per slot with
     the existing `import_d3d11_handle_to_wgpu` (`Rgba8Unorm`). The GUI's
     dispatch is ungated with a fallback to the pipe on error; the reframed
     view decodes at FULL size on Windows too when the SDK runs on a GPU
     (`full_ok`: `info.gpu_device` non-empty — a CPU-pipeline box keeps the
     cap-based scale, where a host upload of 12K frames would not pay).
     VERIFIED (`examples/braw_zc_check.rs`,
     gitignored, on C006 at eighth): zc-vs-pipe **0.000/255, max 0** on both
     eyes (same canvas, pad rows black), zc-left vs pipe-right 12.5 (not
     swapped), 90 frames sustained through the 5-slot ring at **295 fps**
     with a 2-deep hold, seek to 5.0 s lands on frame 300 exactly; the GUI
     logs "zero-copy d3d11 path", plays at the 30 fps cadence with a 1 ms
     render. **FULL 12K ("fix Reframed preview sharpness", same day)**: the
     harness at `SCALE=full` (6144² eyes) is pixel-exact too (0.000/255)
     and sustains **61 fps** through the 5-slot ring with the frames in the
     page cache (hold depth 0 or 2: identical — the ring is not the
     limiter), 28 fps on a cold read of the same frames; the helper alone
     (`zc_drive.py`, R answered at once, `BRAW_HELPER_DEBUG=1` — the loop
     now reports ms/frame split into "waiting for the SDK" / "waiting for a
     slot" every 30 frames) reads 59 fps warm / 25 fps cold with the whole
     frame time in the SDK wait and the slot wait at 0.0 in both. In the
     GUI (Reframe set in the settings file, clip on the command line, Space
     posted): "zero-copy d3d11 path, decode scale Full, step 2"; cached
     frames play at EXACTLY 30 frames per second (decode wait ~30 µs,
     project 0.3 ms, compose 0.4 ms, render 1 ms); past the cached range
     the same play drops to ~20 fps with 45–83 ms decode waits — the SMB
     share delivers ~300 MB/s cold and a 60p 12K clip at step 2 needs
     30 × 11.75 MB = 352 MB/s. So the reframed BRAW preview is sharp at
     full source resolution on Windows and I/O-bound only on network
     storage; the 26 fps reading that started this was the cold read, not
     the zero-copy path.

26. **Dropped frames: per-frame tables are indexed by the container's
   sample table, not `round(pts / dt)` (2026-10-29).** Report: X6 clips
   with dropped frames lost stabilization entirely. Four sample clips
   (`/Volumes/Database/x6 firm/testfootage/VID_20261027_14*.insv`,
   `VID_20261026_235735_00_074.insv`) show what the camera does: the video
   track keeps REAL time — `stts` has 2×…10× deltas where the sensor dropped
   frames (017: 1359 frames, 52 missing in 41 holes over 28.2 s; 074: 2357
   frames, 289 missing — avg 44.5 fps for a 50 fps mode), audio continuous,
   no `ctts`, trivial `elst`. The trailer's exposure stamps (`frames` /
   `frames_b`) have ONE entry per ENCODED frame with the same holes at the
   same rows (1359 = 1359; container-vs-stamp |Δ| < 1 ms except one row at
   a drop burst where the two sensors' stamps disagree by 4 periods), so
   `build_insv_imu`'s row-i ↔ stamp-i pairing was right all along. What
   broke: libavformat's `avg_frame_rate` is `frames / duration` (48.16 fps),
   `dt = 1/fps` was too long, and every `stab_idx = round(pts / dt)` landed
   rows EARLY — 10 rows (200 ms) at row 300, before any hole — so the
   stabilizer applied another frame's pose. Fix (`crates/vr180-pipeline/src/
   frame_timeline.rs`, the first of the two suggested fixes, "index by frame
   number via the sample table"): `FrameTimeline::from_mp4/from_segments`
   reads `moov` only (stts + ctts + elst of the largest `vide` trak; chains
   rebased with the video-track durations like the iterators), `row(pts)` =
   last frame at/before pts (quarter-period slack; inside a hole = the frame
   a player keeps showing), `stab_row(tl, pts, dt, fallback)` is the one
   lookup every stab/RS/eye-match site now uses (3 preview paths, the zoom
   still via `compute_stab_for`/`render_still_from_pair`, the 4 fisheye
   export loops); `probe_video().fps` is the NOMINAL rate (dominant stts
   delta — kept bit-identical to `avg_frame_rate` when within 0.5 %, so
   constant-rate files are untouched); stab tables are sized by the track's
   frame count (`timeline_frames_or`; rows past the last frame were identity
   poses the smoother bent toward); the X6/Kandao IMU builders get that
   count as their hint. Export: `export_fill_plan` → each frame before a
   hole is encoded `1 + hold` times (and the first frame leads when the
   trim-in fell inside a hole), so the output is constant-rate at the
   nominal fps on the audio's clock (the second suggestion's export half;
   verified: 5–7 s of clip 017 = 91 source frames → 100 frames @ 50 fps,
   2.000 s, no irregular deltas, audio 2.005 s). Progress counts output
   frames (`written`); `frame_idx` stays the source-frame counter (the
   GPU-resident ring slot is chosen by it — do not choose it by message
   count). The GUI publishes the timeline in `timeline_cache` (path-keyed
   like the IMU cache; built first thing in `run_fisheye`, before the IMU
   loads and the zero-copy dispatch) and reports `timestamp_s` as the
   frame's own pts (`frame_time_s`). `ClipInfo.frame_count` = frames in
   the track. Kandao's PTS table pairs by index the same way, so a Kandao
   drop is covered too; GoPro repeats frames in the video (MSKP) and is
   untouched; BRAW/stills have no sample table (timeline `None` → the old
   arithmetic). Windows D3D11/GPU-resident loops edited blind (`EncFrame`
   now `Clone`; copies sent as extra channel messages). Diagnostics
   (gitignored examples): `insv_stamp_check` (container vs trailer rows,
   old-vs-new row error), `insv_drop_export_check` (window export + probe);
   `scratchpad/drop/mp4_stts.py` dumps a moov's timing tables.
   **Preview pacing + the NAS (same day, follow-up report "this build
   stutters on insv").** Two things, measured apart: (a) the three
   fisheye preview loops paced by their loop counter (`frame_idx ×
   preview_dt`), so on a dropped-frame clip the picture ran through the
   holes (4.7 % fast on 023, drifting off the free-running audio) and,
   with the nominal fps, no longer averaged out; now `preview_frame_t_rel`
   paces by the held frame's own pts (relative to the seek origin, clamped
   ≥ 0 — `Duration::from_secs_f64` panics on a negative): a hole holds the
   previous frame for the missing slots and the clip stays on the audio's
   clock; the no-timeline stab fallback is `round(frame_t_abs / dt)` (the
   old `round(time_offset/dt) + frame_idx` mixed source rows with preview
   slots — half a row per frame under 2× decimation). (b) The stutter
   itself is I/O: `insv_decode_pace` (gitignored example; demux + VT decode
   per `next_pair`, as the preview thread sees it) on clip 023 — local copy
   11.8 ms per pair (p99 22 ms, 58 % of real time at 50 fps, no stalls);
   the same file on `/Volumes/Database` (smbfs over Wi-Fi, en0 the only
   active link): p50 still 11.7 ms but a 0.3–1.5 s stall every ~36 groups
   (10 in 10 s, 166 % of real time) — each one trips the loop's
   decode-bound branch (wall-clock rebase → a visible freeze). `dd` from
   the share: 20–32 MB/s with 1 MiB reads, 30 MB/s with 32 KiB reads, so the
   LINK is the ceiling, not the read pattern — and 023 needs 33.6 MB/s
   (269 Mbps). No prefetch depth fixes an average deficit; play X6 50p from
   local storage (or a wired link). The pump reads and decodes
   synchronously on the pull (`DualPacketPump` → `ictx.packets()`), so a
   read-ahead thread would only smooth jitter when the average suffices.
   **WINDOWS ARMS VERIFIED (2026-11-06, Windows box):** pulled the 25
   commits, built clean (the BRAW helper is macOS-only by construction;
   `vr180-braw` compiles here with its macOS zero-copy module gated), lib
   tests 57 / 39 / 89. Regression first: OSV cpu / zc readback / GPU-
   resident NVENC and the `.360` NVENC export are byte-size IDENTICAL to
   the pre-pull baseline (6701542 / 6702715 / 9527805 / 7587651 B, cpu-vs-zc
   1.189/255 as before) — the sample-table row equals `round(pts/dt)` on a
   constant-rate file, as designed. Then the mac's own dropped-frame clip
   017 (recovered from the NAS `#recycle`; `examples/timeline_probe.rs`,
   gitignored, reads the same 1359 frames / 52 missing in 41 holes / 48.16
   fps the mac reported; 074: 2357 / 289 / 44.54): the 5–7 s window through
   the portable loop, the blind-edited zc d3d11 arm AND the GPU-resident
   NVENC arm each write **exactly 100 frames @ 50 fps, one 0.02 s delta,
   2.018 s video / 2.005 s audio**, cpu-vs-zc parity 1.263/255 — the
   cloned `EncFrame` copies and the repeated ring-slot messages land where
   the fill plan says. The Windows zero-copy PREVIEW on 017 logs the
   timeline ("52 dropped in 41 holes"), engages, and plays (Space, 12 s);
   pacing is not measurable from the log. The gitignored `export_smoke`
   needed `braw_color: Default::default()` for the new config field.
   **MAC SIDE PULLED THE WINDOWS BATCH (2026-11-07, Mac):** fast-forward
   to `e3955cd`, `vr180-gui` builds clean, lib tests 90 / 2 / 6; the macOS
   helper REBUILT from the platform-neutral source (`build_braw.sh`, one
   pre-existing `kIOSurfaceIsGlobal` deprecation warning) — `--info` on C006
   reports `pipeline metal`, `zerocopy true`, `gpu_device Apple M5 Max`;
   `braw_zc_check` (gitignored, updated to the new signatures:
   `BrawZcIter::new(adapter, device, ..)`, `BrawZcDecoder::start(.., None)`,
   `surfaces` tokens `as u32` for the IOSurface lookup) at eighth / 5.0 s:
   zc-vs-pipe IDENTICAL both eyes, projection identical, 712 fps through the
   ring, 471 fps holding two pairs, seeks OK. The OLD Oct-22 helper binary
   also still satisfied the new Rust side (the header fields are additive),
   so a stale `helpers/bin/braw_helper` is not a hard failure — but rebuild
   it after any `helpers/braw/*` change; the release script hash-checks.


28. **Insta360 X6 added recording modes (2026-11-10, Mac; UNRELEASED,
   Windows not built)** — 7.4K (3712² @ 59.94), 7.4K Crop (3712²), 10K
   (4864² @ 29.97), 10K Crop (4912²), plus the older 3808² crop / 4864²
   "9.7K" signatures. Samples: `/Volumes/Database/x6/1110/` (11 clips, fw
   v1.2.36; 004 + 009 in `drop frame/`).
   - **Identification** (`vr180_fisheye::insta360`): `X6Mode::classify`
     from the encoded eye size + the trailer's `window_crop_info.dst`
     (7680² → 7.4K, 4912×3968 → 7.4K Crop, 7912² → 10K, 4912² → 10K Crop).
     The two 3712² modes are indistinguishable without the record →
     `Unconfirmed7K4` + `X6ViewChoice` (Settings `insv_view`, Source ▸ View;
     a change evicts the cached IMU and reloads, like the eye-order
     toggles). `Unknown` (e.g. 4928²) renders through the generic rule.
   - **Geometry** = `InsvStreamGeometry { scale, origin, readout_fraction,
     cropped }`, `stream = (native − origin)·scale`: wide modes keep the
     scaled-window rule (7.4K: 3712/7680, origin 32; 10K: 4864/7912, origin
     (−80, −84) from the record's (4, 0) offset); Crop modes are native
     pixels centred on the sensor half (scale 1, origin 2016 / 1416 / 1968).
     **The window `dst` is a signature, not a decode size or a crop rect.**
     `insv_imu::stream_lens_calib` / `insv_calib_only` / `build_insv_imu`
     take the geometry / the view choice now.
   - **Timing**: nothing new — content time is the same centre-row
     mid-exposure in every mode; readout = the file's
     `rolling_shutter_time` × `readout_fraction` (7.4K Crop encodes 3712 of
     the 3968 rows read). No constants were added.
   - **Verified on footage** (scratch tools, not in the repo: gyro-predicted
     vs tracked feature motion per stream; far-field shake of stabilized
     exports): scale of every mode equals the stock mode's under the same
     field selection (±0.2 %), time offset −0.5…−0.6 ms as stock, readout
     14.0–14.3 / 14.6–15.5 / 31.2–31.7 / 21.0–21.5 ms vs the files' 14.56 /
     15.00 (16.04 × 3712/3968) / 31.65 / 21.29; stabilized shake 0.05 px per
     frame at 1024 px/eye in all four (1.2–2 px off), a V with its minimum
     at 1.0 when `VR180_RS_SCALE` sweeps 0 … 2. Side finding, ALL modes
     incl. stock: the factory lens model reads ~0.8 % small in the central
     field and ~0.2 % large at the rim against the gyro — unexplained, not
     acted on.
   - **Crop frames** (`DjiLensCalib::frame_crop` → `FisheyeCalib::
     with_frame_crop` → uniform `frame_mask`): rays that leave the frame
     render black (edge antialiased over one output pixel) and the θ clamp
     moves from the inscribed circle to the far corner, so corners keep
     their image. Full-circle sources keep replicating the rim — the user
     wants that (it hides stabilization gaps). **Crop mask** = a rounded
     rectangle on the OUTPUT's equidistant angle plane, evaluated on the
     output ray before the stab rotation (static window, same in both
     eyes), `gpu::CropMask { amount, feather, round }` → `apply_frame_crop`
     in BOTH resolvers (Settings `crop_mask`, export cfg `crop_mask`);
     skipped in the reframed view. **The reframed (flat) view does not
     black out a Crop frame either** — `frame_cover` returns 1 when
     `proj_mode > 0.5`, so the frame's edge is repeated there exactly as on
     a full-circle source (user request: flat 3D is the practical output
     for Crop clips and a repeated edge hides the stabilization border;
     measured 81/180 frames with black → 0 on a stabilized 92° view).
   - **Shaders**: all nine `fisheye_*.wgsl` now declare the SAME full
     uniform struct (`FisheyeCalibUniforms` is 12 vec4 = 192 bytes: `_p9a`
     became `frame_mask`, a new last row holds the mask) and share
     `frame_cover` / `view_mask`. Exercised on Metal under
     `VR180_WGPU_PANIC=1`: zero-copy P010 8/10-bit, the readback RGBA
     8/16-bit kernels, equirect + fisheye output. **Windows TODO: build,
     and check one Crop clip with a mask on DX12/Vulkan + the D3D11
     zero-copy path** (no `#[cfg]` line was touched; struct literals and
     call sites were audited textually).
   - **Also fixed**: `decode::nominal_or_average_fps` — ANY dropped frame
     now selects the nominal rate (a clip missing 2–8 frames sat inside the
     0.5 % tolerance and exported at the average rate: 0.27 s of A/V drift
     over a 61 s clip). The I-Log LUT follows `gamma_mode` (identity field
     22), not the frame size. A trailer that cannot be read → generic X6
     profile placed by the mode (calib-only IMU, no stab).
   - **Native-sharpness export size** (`ExportResolution::Sharp`, Crop
     clips only): the 180° canvas at the frame's centre density,
     `insv_imu::native_sharpness_side(π·f)` from the file's lenses → 6560²
     per eye = 13120 × 6560 for both Crop modes (36.4 px/° vs 18.2 on the
     8K mode). Offered in the Resolution list only when the active clip's
     geometry is `cropped` (`ClipInfo::sharp_eye` / `BatchItem::sharp_eye`,
     passed into `build_export_cfg`). A Crop clip's choice is its own field
     (`ExportOptions::crop_resolution`, read through `resolution_for(cropped)`
     everywhere): the Format window edits it while a Crop clip is open and
     `resolution` otherwise, so one list exports each sort at the size picked
     for it (it used to be one choice, and no value of it gave both their
     best).
     Mac: VideoToolbox HEVC takes it on the zero-copy path (10 fps, level
     7.0; its ceiling is 16384 × 8192), hardware ProRes does not (software
     fallback, as for a 12K BRAW). **Windows: NVENC HEVC stops at 8192
     wide — this size will fall back to libx265 there; untested.**
   - **File info shows the bit rate** (`VideoProbe::video_bitrates_bps`,
     the first two streams of the largest frame size; file size ÷ duration
     marked ≈ where the container states none).
   - Open: multi-segment `.insv`; this unit's lens A sits ~1° off the
     shared IMU basis in roll (per-unit mount tolerance — not handled).
   - **WINDOWS VERIFIED (2026-11-12, Windows box, RTX 4090, at `834094a`).**
     Built clean first try — no compile error from the blind edits (the
     12-vec4 uniform, `finalize_metadata`'s dropped argument in the three
     Windows-only functions); tests braw 6 / core
     57 / fisheye 61 / pipeline 103 / gui 14 (+ the two GPU-only zoom-still
     tests of `834094a`, run with `--ignored`: both pass on the Vulkan
     device). REGRESSION: the export matrix (OSV cpu / zc readback /
     GPU-resident NVENC, `.360` NVENC, QooCam zc, the dropped-frame X6 017
     on three arms) is byte-identical PER STREAM before and after the pull
     (`scratchpad/vhash.py`: ffprobe packet pos + size → md5 of the bytes;
     the bundled ffmpeg has no `md5` muxer), so the shared uniform struct
     and the mask functions are a no-op for full-circle sources on every
     Windows arm. MODES on the mac's samples (`W:\x6\1110`, fw v1.2.36;
     gitignored `examples/x6_mode_probe`): 001–003 7.4K, 005 + 004 7.4K
     Crop, 008 / 010 / 009 10K, 006 / 007 / 011 10K Crop — scale, origin,
     readout fraction and the 6560 native-sharpness side read as documented.
     Each mode through the portable loop, the D3D11 zero-copy readback arm
     and the GPU-resident NVENC arm (`export_smoke`, 0.5 s, stab on): all
     engage; cpu-vs-zc 1.224 (7.4K) / 0.347 (7.4K Crop) / 0.863 (10K) /
     0.591 (10K Crop) per 255; NVENC 21 fps from 3712² sources, 9.5 fps from
     4864² / 4912² (d3d11va budgets 11.6 GB for two 4912² streams — a 12 GB
     card takes the item-11 software fallback). CROP FRAME + MASK
     (`VR180_CROP_MASK=0.15,0.35,0.6` on 005 and 006): black outside the
     frame with the corners kept; the mask's bounding box is the SAME in
     both eyes (x 422..1114, y 432..1101 of 1536 on the 7.4K Crop) while the
     unmasked frame edge differs by eye with the stabilizer; portable Vulkan
     vs portable **DX12** (`VR180_BACKEND=dx12`, adapter "RTX 4090 [Dx12]")
     0.206 / 0.250 per 255, portable vs zero-copy 0.214 / 0.271, zero-copy
     vs GPU-resident 0.259 / 0.379 — naga → HLSL renders the masked kernels
     the same. Fisheye output: masked identically in both eyes; reframed
     view: 0 % black (edge repeated, mask skipped) on all three arms.
     DROPPED FRAMES: 004 (2 missing) and 009 (8 missing = 0.44 %, inside the
     old tolerance) probe at 59.940 / 29.970 and export 60 / 76 frames at
     60000/1001 / 30000/1001 with ONE pts delta on all three arms. PREVIEW:
     the app on 006 with a mask in the settings — "X6 recording mode
     Crop10K", zero-copy d3d11 preview, the Mode / Bitrate rows and the Crop
     mask sliders shown, the mask still while the picture plays under it at
     30.0 fps (window captured with `PrintWindow` flag 2, which works on the
     Vulkan swapchain even when occluded).
     **NATIVE SHARPNESS 13120 × 6560 on Windows:** the GPU-resident arm
     opens, NVENC refuses the frame (`avcodec_open2 … -40`), the readback
     arm's `hevc_nvenc` refuses it too and `open_h265_encoder` lands on
     libx265 — correct picture, Main10, ~0.5 fps. ProRes takes the size on
     the GPU (`prores_ks_vulkan` ENGAGED, 3.7 fps). Two gaps FIXED there
     (the same two also cover a 12K BRAW at its native size on Windows):
     the NVENC → libx265 fallback only logged — it now appends an export-bar
     note (`append_export_path_note`: "frame larger than the NVIDIA encoder
     takes (8192 px) — CPU H.265 encode", or "NVENC unavailable — CPU H.265
     encode"; a normal NVENC export still has no note and is byte-identical);
     and the note under the Resolution list was the Mac's text on every
     platform ("encodes in hardware" / "the Mac's hardware ProRes encoder
     refuses the frame") — `app.rs::oversize_export_hint(codec, w, h)`
     answers per platform now (Windows H.265 above 8192 a side: NVENC
     refuses, libx265 takes over; Windows ProRes: no note; ≤ 8192 wide but
     > 4096 lines: the Level 6.2 note), pinned by
     `export_size_tests::oversize_hint_follows_the_platform_encoder` on both
     platforms. TRAP: `export_smoke`'s own device has wgpu's DEFAULT limits
     (8192 px textures), so a 13120-wide run "succeeds" with an invalid
     compose texture and a 38 KB black file — set `VR180_USE_DEDICATED=1`
     (the app's export device, 16384) for anything above 8192; the app is
     not affected, but a GPU whose own limit is 8192 would be (the Sharp
     size is offered regardless — open).

29. **Spatial video export of the reframed view (2026-11-10, Mac only;
   UNRELEASED)** — `ExportCodec::Spatial` ("Spatial video (MV-HEVC)" in the
   export format's Codec list) writes `<stem>_Spatial.mov`: one MV-HEVC
   Main10 stream (left eye = layer 0 = hero) with `vexu` (eyes / stri /
   hero / cams·blin / cmfy·dadj / proj·rect) + `hfov` = the reframed
   viewport's horizontal angle.
   - **Why MV-HEVC**: a side-by-side file with the full spatial metadata is
     only "stereo" to the system. Two system tests exist (disassembled on
     macOS 27.2): AVFoundation's `AVAssetPlaybackAssistant` skips the
     multiview test as soon as `ViewPackingKind` is side-by-side and needs
     real left/right layers; Photos' `PFVideoMetadata videoTrackIsSpatial`
     wants three track characteristics, and although a `tagc` atom can
     declare the missing one, the user's headset (visionOS 27) still did not
     show such a file as spatial.
   - **Flow**: `FisheyeExportConfig::spatial: Option<SpatialVideo {
     baseline_mm, disparity }>` → `prepare_spatial` at both export entries
     (needs `Reframe` with `defish` 0 and even eye sides; pins VideoToolbox
     + 10-bit so every Mac loop takes its zero-copy P010 encode branch) →
     `open_spatial_encoder` returns `encode::ExportEncoder::MvHevc`
     (`mvhevc.rs`), which the three Mac loops feed through the same
     `encode_pixel_buffer_p010` call as the ffmpeg encoder
     (`export_fisheye_inner`, `export_fisheye_osv_zerocopy_p010`,
     `export_eac_zerocopy_vt`). `MvHevcEncoder` writes the composed
     side-by-side P010 surface's two planes (read-locked,
     `RetainedIOSurface::with_plane_bytes`) to the `mvhevc_encode` helper's
     stdin; the helper splits each frame into two pool buffers and calls
     `VTCompressionSessionEncodeMultiImageFrame`. No colour conversion and
     no intermediate file: one encode of the rendered frames. The pipe is
     the back-pressure.
   - **Audio**: never one-pass (`one_pass_audio_eligible` is false); the
     helper's video-only `.mov` goes through the usual finalize (ffmpeg
     stream-copy mux, or the APAC helper's passthrough).
     **`audio::mux_video_with_audio_stream` must carry
     `AV_DISPOSITION_MULTILAYER` to the output stream**: the MOV muxer
     writes the second view's `lhvC` box only for a stream with that flag,
     and without the box the right eye does not decode while every
     metadata check still says "spatial". The ffmpeg CLI copies
     dispositions; our mux copies only that bit.
   - **GUI**: the choice is offered when the reframed view is active on
     macOS (`spatial_video_available`: + Defish 0, not a still);
     `ExportOptions::codec_for(settings, is_still)` is the codec an item
     really exports with — a clip the choice cannot apply to runs as H.265
     and the choice itself stays put (batch of mixed clips).
     `spatial_baseline_mm` (65) and `spatial_disparity_pct` (2) are its
     two parameters.
   - **Helper**: rewritten around P010 input (was BGR48 + a pixel-transfer
     session, from the Python app, never called from Rust); exact frame
     rate as `--fps-num/--fps-den`; counts samples in = out or fails; built
     with `-target …macosx14.0` (`min_macos_for` in `build_swift.sh`) and
     availability-guarded (the field-of-view key is 14.4+). **The other two
     Swift helpers are still built for the build machine's own macOS** —
     open.
   - **Verified** (headless, the gitignored `x6_mode_export` /
     `eac_spatial_check` examples): X6 7.4K Crop, Osmo `.osv`, a finished
     VR180 SBS file (zero-copy and portable loop), GoPro `.360`; 1:1 and
     16:9; stereo / ambisonic + SA3D / APAC audio. Each output reads as
     `SpatialVideo` to `AVAssetPlaybackAssistant` and to
     `PFMetadataMovie.isSpatialMedia`, keeps `lhvC`, and VideoToolbox
     decodes both eyes of every sample. M5 Max, 7.4K Crop 59.94 fps
     stabilized: 46 fps at 2160² per eye, 32 fps at 3840×2160 per eye;
     10K Crop 29.97 fps: 20 fps at 3840² per eye. The MV-HEVC session
     itself opened at every size tried, up to 8192² per eye.
     Watched by the user in the headset's immersive view (visionOS 27,
     2026-11-11). **The export-format UI was not visually checked.**
   - **The headset's life-size limits (2026-11-11)**: in its immersive view
     visionOS sizes a spatial video from the `hfov` written, after limiting
     the view to 90° horizontally and 80° vertically with the shape kept
     (the window view does not use the field of view at all). A wider
     frame is scaled down whole, never cropped: 1:1 at 90° or at 120° is
     shown as 80° × 80°, 16:9 at 120° as 90° × 58.7°. Confirmed by the user
     on the headset with charts that differed only in the `hfov` written
     (1:1 at 60 / 80 / 120, 16:9 at 70 / 90 / 120: the last two of each set
     came out the same size). So life size ends at 80° for a 1:1 view and
     at 90° for 16:9, and the reframed view's default (90°, 1:1) is
     already shown 16 % small.
     GUI (`app.rs`): `spatial_life_size_hfov_deg(aspect)`,
     `spatial_view_shown_as(hfov, aspect)` and `SPATIAL_VIEW_SCALED_NOTE` /
     `spatial_view_scaled_note`; the spatial section of the export format
     prints an orange note (what the view is shown as, the Zoom that stays
     at life size) under "Field of view written" while the view is past a
     limit and Defish is 0. `tr` now forwards to `tr_zh` (the Chinese
     table on its own) so a test can read a translation without switching
     the process-wide UI language. Tests in `spatial_export_tests`: the
     limits, the note's placeholders in both languages, and the text the
     panel really paints (a headless `egui::Context::run_ui` of
     `export_options_ui`). Platform-neutral code, no `#[cfg]`; the section
     is never reached on Windows.
     Also read in the system's renderer and NOT acted on: in the immersive
     view the file's disparity adjustment gives way to a value derived
     from the (limited) field of view, and the baseline does not enter
     the picture's size. A lower limit of 40° is in that code too, but
     the headset check did not cover it, so there is no note for narrow
     views.
   - Open: a source whose audio is APAC (a finished VR180 export re-opened)
     fails ANY export at the audio mux (`write_header: Invalid argument`,
     ffmpeg cannot stream-copy the unknown codec) — pre-existing, not
     spatial-specific; the user ruled it an edge case, not to be fixed.
   - **VR180 (APMP) metadata no longer writes a baseline** (same day): the
     old `cams/blin` was directly under `vexu` (the spec — "QuickTime and
     ISO Base Media File Formats and Spatial and Immersive Media" v1.9.8,
     2025-06-09 — puts `cams` inside `eyes`) and `blin` was 12 bytes of
     mm × 65536 (the spec: a FullBox with a u32 in micrometres), so
     Apple's reader skipped it. The baseline is optional for projected
     media, "no default interpretation when absent", and the rendering core
     does not read it (no reference in CoreRE / RealityFoundation / AVKit),
     so the box is simply gone: `spherical_inject::apmp_vr180_atoms` =
     `vexu { eyes{stri} proj{prji hequ} pack{pkin side} }` + `hfov`, pinned
     byte for byte by a test. `FisheyeExportConfig::apmp_baseline_mm`, the
     fourth `finalize_metadata` argument (all seven call sites, three of
     them in Windows-only functions — edited without a Windows compile) and
     the slider under the Vision Pro metadata option are removed. What
     AVFoundation reads from an export is identical before and after.
   - **WINDOWS (2026-11-12):** builds as is (the three `finalize_metadata`
     call sites in Windows-only functions compile); the Spatial codec is
     never offered (`spatial_video_available` is `cfg!(macos)`) and a
     Spatial choice carried in settings runs as H.265 —
     `spatial_export_tests` passes on Windows with the non-Mac expectations.

30. **ETTR recovery for X6 I-Log clips (2026-11-14, Mac; UNRELEASED)** —
   Color ▸ "ETTR recovery (I-Log)": `Settings::ettr_stops` (0 / +1 / +2 / +3
   and a fine value, 0–4) is how far over a clip was shot; the stack takes
   that much exposure back out in scene-linear light.
   - **The curve**: `vr180_core::LogCurve` (straight toe + logarithm, f64
     `encode` / `decode` / `expose`), `LogCurve::INSTA360_ILOG` = the
     constants of Insta360's "I-Log White Paper" (June 2026, "Conversion
     Specification"). Its code is the normalised R'G'B' the decoders already
     produce: black 0.0906 (what the eye-match tuning measured as 0.091).
   - **The stage**: `gpu::LogExposure { curve, stops }` in
     `ColorStackPlan::log_exposure`, shader `log_exposure.wgsl`
     (`u' = E(D(u) · 2^stops)` per channel, the curve as a uniform; 16-bit
     variant by the usual storage-format replacement). Order in all three
     GPU stacks (`apply_color_stack_texture`, `record_color_stack`,
     `record_color_stack_16`): automatic eye offset → **log exposure** → CDL
     → LUT → … The eye offset stays first because it is measured on the
     source's own codes. The 8-bit host fallback
     (`fisheye_export::apply_color_stack_rgb8`) does it with a 256-entry
     table (`LogExposure::table8`).
   - **Which clips**: `takes_ilog(kind, x6)` in `app.rs` (an X6 video whose
     `gamma_mode` does not say otherwise), the same rule the built-in LUT
     follows. The settings as stored carry only the amount; the copy a
     renderer gets (`worker_settings`, `BatchItem::export_settings`) carries
     `Settings::ilog_source` (`#[serde(skip)]`), and `build_color_stack`
     emits the stage only with both. A new clip starts at 0
     (`add_batch_files`); "Apply settings to all" copies the amount to
     I-Log clips and zeroes it on the others; `finish_load` zeroes it on
     anything that is not I-Log.
   - **Verified**: `log_exposure_tests` (every 16-bit code against the f64
     curve at −3 / −2 / −1 / +1 stops: worst 0.5 / 65535; both 8-bit stacks
     against the table: ±1), `log_curve_tests`, `ettr_tests`; real exports
     through every arm (zero-copy 10 / 8 bit, libx265 10 / 8 bit, ProRes)
     with and without it: the recovered frame minus the curve applied to the
     untouched one averages within 0.6 of a 10-bit code.
   - **WINDOWS**: shared code only, no gates touched: pull + rebuild. The new
     shader is compiled when the device is created, so
     `cargo test -p vr180-pipeline --lib log_exposure` on the Windows GPU
     checks it on that backend; the arms to try are the same as for any
     colour stage (zero-copy D3D11, GPU-resident, portable).
   - **WINDOWS VERIFIED (2026-11-17, Windows box, RTX 4090).**
     `log_exposure_tests` pass on Vulkan and on D3D12. A real I-Log clip at
     −1 stop through the portable loop, the D3D11 zero-copy readback arm and
     the GPU-resident NVENC arm: the recovered export minus the curve
     applied to the untouched export of the same frames averages +0.23 /
     +0.24 / +0.21 of a 10-bit code (level 437 → 361, the curve says 361),
     and the three arms agree with each other as they do without the stage
     (cpu − zc 0.005 of 255 signed, either − nvenc under 0.09).

31. **Colour precision of the export path (2026-11-14, Mac; UNRELEASED).**
   Three commits, each changing export pixels by what it corrects.
   - **3D LUT** (`f880e96`): the cube is an `Rgba16Unorm` 3D texture where
     the device has the format (`Device::lut3d_texture`; 8-bit otherwise)
     and both LUT shaders interpolate the eight lattice values themselves.
     Against the cube's own values on the 16-bit stack: 2.1 codes of a
     10-bit picture with the 8-bit texture, 0.77 with a 16-bit one through
     a hardware sampler (its weights have 8 fractional bits), 0.015 now.
     +0.17 ms per 4096² eye. Test: `lut_precision_tests`.
   - **10-bit compose** (`57bf897`): `compose_sbs_p010_y` / `_uv` /
     `p210_uv` round to a whole code before the ×64 (an encoder reads the
     top ten bits: the fraction was cut off, 0.4 of a code low in Y, Cb,
     Cr); chroma coefficients to six places. Test: `ten_bit_plane_tests`
     (reads the planes `compose_sbs_to_p010_textures` writes, so it covers
     the Windows readback arm's planes too).
   - **swscale** (this item's third commit; module `swscale.rs`, read its
     header): every conversion states its matrix (the software feed
     converted RGB → Y'CbCr with **BT.601** in BT.709-tagged files; the
     `.360` software decode had the same default); 16-bit pictures cross
     on one scale (swscale's white is 65280: decode gets a contrast of
     257 / 256, the feed scales by 65280 / 65535 in `encode::load_rgb16`,
     and an 8-bit picture widens by repeating its byte); and every
     same-size Y'CbCr → RGB conversion is the exact one on a threaded
     context (`YuvToRgb`; the fast 8-bit converters are 2 of 255 dark).
     Tests: `swscale::tests`, `rgb_to_yuv_tests`, `source_matrix_tests`,
     `p010_to_rgba64_color_roundtrip` (now held to half a 10-bit code; it
     allowed ±2 %).
   - **End to end** (flat colours of known codes, lossless source, 16
     decode × encode paths): every 10-bit path and the 8-bit software one
     return the codes that went in. Software vs hardware export of the
     same real frame, in 8-bit codes: X6 8-bit −1.82 → +0.32, GoPro 10-bit
     −0.72 → +0.02. Software exports got faster (X6 10-bit ×1.8). **The
     paused native-resolution still** (magnifier, Snapshot) decodes on the
     CPU on every platform: for X6 / OSV it was 1.75 – 1.89 of 255 darker
     than the live preview of the same moment, now within 0.02.
   - **Left as they are**: 8-bit through VideoToolbox is 0.4 of a code low
     (its own BGRA conversion; an NV12 compose on the GPU would replace
     it); a conversion that also scales (the working-size CPU preview)
     keeps the fast flags; **the `.360` range is untouched**: still video
     range on every path whatever the file's `pc` tag says (item 14's rule;
     `decode::eac_rgb_converter` says so and its test pins it).
   - **WINDOWS**: shared code, no gate touched. To check there: the tests
     above on the Windows GPU; that 8-bit H.265 and the software fallbacks
     now match the 10-bit NVENC path in colour (they were BT.601); the
     CPU preview's speed (same-size conversions now run on up to 16
     swscale threads per stream; on this Mac they are faster than the
     single-threaded fast ones for 10-bit sources and slower for 8-bit
     sources, 10 ms against 2.5 for a 3840² frame).
   - **WINDOWS VERIFIED (2026-11-17, Windows box, RTX 4090).** Export
     matrix (OSV cpu / zc readback / GPU-resident NVENC, `.360` NVENC,
     QooCam zc, the dropped-frame X6 clip on three arms): every arm engages,
     frame counts and pts as before; bytes differ from the 2.5.6 baseline
     as intended. **The software arms now match NVENC**: signed difference
     from the GPU-resident NVENC export of the same frames, in 8-bit codes,
     before → after: OSV portable R +0.94 G +0.51 B +0.28 → +0.11 / +0.03 /
     +0.12, OSV zc readback (libx265) +1.40 / +0.93 / +0.69 → +0.10 / +0.03
     / +0.12, X6 portable +0.06 / +0.27 / +0.81 → +0.11 / +0.03 / +0.11, X6
     zc +0.47 / +0.68 / +1.28 → +0.11 / +0.03 / +0.11; portable and zc agree
     to 0.01. With the X6 LUT on, 8-bit H.265 against 10-bit on the same
     arm: within 0.05 (portable), 0.03 (zc), 0.13 (NVENC), level identical.
     The NVENC arms themselves moved by R +0.3 G +0.06 B +0.35 (the compose
     rounding). CPU preview speed (software decode forced, OSV): 128 ms a
     pair on 2.5.6 and on this build — the two 3840² software decodes are
     the cost, the conversion does not show.
     TWO THINGS FOUND THERE:
     (a) **The NVIDIA GPU rounds a float written to an 8-bit texel late**:
     a compute shader storing a plain ramp into `rgba8unorm` flips to the
     next code at up to 0.562 of the way there on the RTX 4090 (Vulkan and
     D3D12 alike), 0.508 on the Intel UHD 770, 0.500 on WARP (gitignored
     `examples/unorm_round_probe.rs`; Direct3D allows 0.6). So every 8-bit
     store there can sit 0.06 of a code further from its value than exact
     rounding, with a mean bias of −0.03 of a code: nothing to see, but
     `lut_precision_tests::eight_bit_lut_stage_rounds_the_cube` read 0.562
     against its 0.52 and `display_colour::tests::
     a_photograph_is_drawn_as_it_is` got one code off where the dither
     meets that rounding. Both tests keep the Mac's bound on macOS and take
     0.62 / one code elsewhere; no shader was changed.
     (b) **A full-range source deeper than 8 bits was 0.29 % bright on the
     CPU decode path — FIXED, shared code** (`swscale::set_yuv_source` takes
     the source's depth; test `a_full_range_ten_bit_source_keeps_its_scale`).
     swscale's full range is the 8-bit one shifted up: for a 10-bit source
     white is 1020, not 1023 (code 512 came out 32896 of 65535 where
     512 / 1023 is 32800; codes 1021 to 1023 clipped), while the GPU kernels
     read code / 1023. Before item 31 the missing 257 / 256 hid most of it
     (net −0.1 % by the arithmetic); with the 16-bit scale fixed it showed
     as +0.29 %. The only such source in the catalog is the QooCam 3 Ultra
     (GoPro `.360` is read as video range on purpose). Measured on a QooCam
     clip (the same lens-frame export, level 79 of 255), portable − zc
     readback: R +0.30 G +0.33 B +0.42 of 255 before, +0.004 / +0.003 /
     −0.003 after; video-range sources byte-identical. **MAC SIDE**: pull;
     the paused full-resolution still and any CPU-decoded QooCam picture
     get 0.29 % darker, onto the live preview's level.
     **MAC SIDE PULLED IT (2026-11-17, Mac):** fast-forward to `8948361`,
     `vr180-gui` builds, tests 6 / 63 / 69 / 38 / 225; the stored exports
     (65) and preview frames (169 lines) are the same as before the pull,
     none of them having a full-range source deeper than 8 bits. The fix
     itself on this machine's FFmpeg: a flat-colour 10-bit full-range SBS
     clip (eight colours) exported with the VideoToolbox zero-copy decode
     and with the software decode (`VR180_NO_HW_DECODE=1`) through the same
     hardware encoder gives the same code on every plane, on the exact
     value (luma 512 of 1023 → 502 of the 10-bit video range where the old
     scale gives 504; 940 → 869, not 871).

32. **The preview as Apple's players show the export (2026-11-14, Mac;
   UNRELEASED).** A toolbar switch, **Match QuickTime**: macOS only, off by
   default, kept in `…/VR180SilverBullet2.0/preview_colour` (a view choice
   of the app, like `lang`; not in `Settings`, which is per clip and per
   camera). Module `crates/vr180-gui/src/display_colour.rs`: read its
   header.
   - **Why**: the exports are tagged Rec.709. The system's players read
     that as `kCGColorSpaceCoreMedia709` (it is what
     `CVImageBufferCreateColorSpaceFromAttachments` returns for a decoded
     frame of one) and colour-match it to the display. The preview window
     is colour-matched too, but as **sRGB**: a `CAMetalLayer` takes
     `kCGColorSpaceSRGB` the moment its pixel format is set to an 8-bit one
     (macOS 27.2: none before, sRGB after; wgpu-hal sets the format in
     `configure` and never a colour space). Same primaries, another curve:
     mid-tones about 10 of 255 apart, QuickTime the brighter. It was always
     so, and nothing in the export is wrong: do not "fix" this by changing
     the export's tags. **The window is NOT unmanaged**: the first version
     of this work assumed it was, and its "off" removed the layer's colour
     space, which made the whole window more saturated than it had ever
     been on a wide-gamut display.
   - **The curve is ColorSync's, read from the system**: the profile's TRC
     is a `curv` gamma of 1.9609375 (502 / 256), and the parametric stages
     ColorSync builds from it are `ParamCurve3 (1.960938, 1, 0, 0.0625,
     0.05583828)`: that power above 0.0558, a straight toe of slope 1 / 16
     below. Then the destination's encoding; to sRGB there is no matrix
     between the two (same primaries), so the curve is the whole
     conversion: `display_colour::rec709_video_to_srgb`, held by a test to
     the codes CoreGraphics itself gives a grey ramp (within 0.08 of 255).
     Not the Rec.709 camera curve turned around (`kCGColorSpaceITUR_709` is
     that one, and video does not get it), not 2.4, and a bare power of
     1.961 is 2.7 codes dark at code 7.
   - **How**: with the switch on, the picture is drawn by `PreviewPainter`,
     an `egui_wgpu` paint callback with its own small shader, in place of
     egui's image widget, and a video frame goes through the curve on the
     way (`PictureEncoding::Rec709Video`; a photo source is `Srgb`,
     untouched). Both preview draws use it: the main view and the two
     halves of the 3D display. Nothing upstream changes: the frame
     textures, the zoom still, Snapshot and every export are identical with
     the switch on or off. The interface around the picture does not change
     either.
   - **The window's layer**: the conversion rests on the layer being sRGB,
     so while the switch is on `set_window_colour_matching` makes sure of
     it (`NSApp.windows` → content view → its layer → the `CAMetalLayer`
     wgpu adds as a sublayer; end of every `update`). A layer that is sRGB
     already is never written, which on macOS 27 is every layer, always.
     One that is not gets tagged and gets its own colour space back when
     the switch goes off; the app logs either.
   - **Off is the old path, untouched**: egui's image widget, no layer
     written, no painter (it is built on first use). Windows and Linux never
     run any of it: `display_colour::available()` is `cfg!(target_os =
     "macos")`, the switch is not drawn, the layer routine is a stub. The
     module compiles everywhere (plain wgpu + egui); its only gate is its
     own `platform` module pair, and `app.rs` has none.
   - **Checked against the compositor**: Core Animation's own renderer
     (`CARenderer`, in-process, into a target in the display's colour
     space) draws (a) a frame decoded from a Rec.709-tagged HEVC Main10
     file as the video surface a player's layer holds and (b) the same
     picture as an sRGB-tagged surface holding the app's matched values. 96
     flat colours (greys from 0 to 255 with the toe sampled every 2 codes,
     primaries, skin tones): worst 0.39 of 255 for the arithmetic, 0.62 for
     the 8-bit values the window holds; with the switch off the two are up
     to 11.4 apart (mean 5.2). Same result for the Studio Display's own
     profile, Display P3 and sRGB.
   - **Checked without a window**: `display_colour::tests` draw through a
     real `egui_wgpu::Renderer` into an off-screen `Bgra8Unorm` target set
     up as eframe sets up the window's pass. A video ramp comes out on the
     curve (every code of every channel within the dither, mean +0.01); a
     photo comes out code for code; and for a zoomed, off-grid rectangle on
     a 2× screen the painter's picture is identical, pixel for pixel, to
     the image widget's (egui snaps a filled rectangle to whole pixels, so
     the painter sets that viewport itself instead of taking the one egui
     cuts off at the screen's edge). The layer routine has a test on a bare
     layer tree (three Metal layers: none / sRGB / Display P3).
   - **On screen**: the user saw the preview get a little brighter with the
     switch (2026-11-14); a side-by-side with QuickTime on the same graded
     frame was still to be done when this was written.
   - **Known gaps**: a Snapshot JPEG carries no profile, so Preview reads
     it as sRGB and it looks like the preview with the switch off; the
     conversion is 8-bit in, 8-bit out with egui's dither, like the image
     path it replaces.

33. **Lens Space view and its lens-frame export (2026-11-14 for Crop
    clips, 2026-11-15 for every source and as a view of its own; Mac;
    UNRELEASED, Windows not built).** A clip can be written as a LENS FRAME
    instead of a half-equirect: each eye through a lens, the frame the
    recording's own, the lens stated in the file (APMP, projection `prim`
    with a lens collection). A Crop clip then fills its frame instead of
    sitting in a mostly black 180° canvas; a clip that holds the lens's
    whole view keeps what lies past 180° (the fisheyes of the 360 cameras
    see 200° and more). Output ▸ Format **Lens Space**, lens **Camera
    lens**: see the next block. The half-equirect export is untouched.
   - **Lens Space is a view (2026-11-15, the user's call: "这个 lens 的 APMP
     导出似乎应该取代鱼眼导出", then "要么整个换个名字吧")**. The lens frame
     began as a tick box in the Format window of the VR180 view, next to a
     Fisheye view that wrote its own disc: two fisheye outputs in two
     places, and a preview (VR180) that was not the export. Now the former
     Fisheye view IS the lens output, under the name the immersive tools
     use for footage kept in the camera's own geometry:
     - `FisheyeOutputMode::Fisheye` shows as **Lens Space (fisheye SBS)**
       (the variant keeps its name: saved settings hold it). Per clip, in
       it, `Settings::lens_space: LensSpaceLens` picks the lens the picture
       is written through: `Camera` (default; a lens frame) or
       `Equidistant` (the old disc: 195° in a square, 185° for a `.360`,
       what players and tools for fisheye video expect).
       `Settings::lens_frame_view(kind)` = that view, that lens, a clip
       with a lens (`decoder::lens_frame_possible`); everything asks it —
       `build_export_cfg`, the file name (`_Lens`), the export bar, the
       Format window (`App::lens_space_out`), the snapshot size
       (`ExportOptions::lens_frame_out_dims`), the preview.
       `ExportOptions::lens_frame` and its tick box are gone; the VR180
       view always exports VR180. A clip without a lens record (a still, a
       file that is already VR180) has no lens choice and gets the disc.
       Settings saved in the Fisheye view before this read as the camera's
       lens: such a template now exports lens frames (intended).
     - **The preview draws the export's frame.** `decoder::
       resolve_fisheye_view` (the three fisheye workers and the zoom /
       snapshot still call it in place of `resolve_fisheye_calib_pair`)
       builds the clip's lens frame with the export's own steps —
       `fisheye_export::source_lens_frame` (`LensFrame::from_sources` at
       the source's size, `with_stabilizer_margin` when stabilized,
       refused where the export refuses) and `lens_frame_calib_pair` (the
       mask from the frame's own edge, `with_frame_edge`,
       `with_lens_frame`; `resolve_calib_pair` calls the same function) —
       scaled to the working size (`LensFrame::scaled_to`, sides in pairs
       of pixels) or `fitted` to the export's exact size for a snapshot
       (`LensViewSize`). It returns the frame's size with the calibrations
       (`FisheyeView::lens_dims`), which `view_out_dims` hands the
       fisheye-output projection in place of the disc's square. A `.360`:
       `resolve_eac_lens_pair(s, geoc, out_side)` attaches
       `fisheye_export::eac_lens_frame` (a stabilized frame keeps the
       square and holds the wider field). No projection function changed:
       all of them already pick the lens-frame kernel by `out_lens`. So
       the crop mask is what-you-see (the Format window's old note about
       it is gone), the stabilizer's margin shows, and a Crop clip fills
       the preview. A lens no frame can be made of previews as the disc;
       the export says why.
     - **The disc states its own lens.** A fisheye disc is no
       half-equirect, yet with the Vision Pro metadata on it used to be
       tagged as one (`ProjectionKind` half-equirectangular). Now
       `export_fisheye` / `export_eac` call `resolve_disc_lens` for a
       `Fisheye` export without a lens frame: both VR180 flags are cleared
       (YouTube's metadata has no fisheye form: nothing is written for
       it), and where the Apple one was asked for the file states the
       disc's equidistant law in the profile's lens family
       (`LensFrame::equidistant` over the square, `picture_deg` = the
       disc's half field: stated to 98°, 93° for a `.360`; within 0.35 px
       of the disc at 3840). The disc's pixels are what they were.
     - Windows: `run_fisheye_zerocopy` got the same three substitutions as
       the macOS zero-copy worker (`resolve_fisheye_view` twice,
       `view_out_dims` once) and `run_eac_zerocopy` the third argument of
       `resolve_eac_lens_pair`, by text, uncompiled. No gate added or
       moved.
     - **Verified, Lens Space (macOS 27.2)**. Exports: the 26 + 8
       regression cases frame-identical; the 44 lens-frame / twin files of
       the every-source run frame-identical to what `408f6e2` wrote.
       Previews, through the refactor branch's headless probe (put into
       this tree and into a checkout of `408f6e2` for the run, then taken
       out again): 65 preview lines, 27 zoom stills and 21 `.360` lines
       identical before and after with the equidistant lens, so nothing
       outside the camera-lens view moved. The camera-lens view itself:
       frames of an X6 8K clip, a 10K Crop clip (with a mask), an Osmo 360
       and a `.360` looked at; the native zoom still
       against the export of the same frame (X6 8K plain; stabilized, its
       4224² export scaled to the still; 10K Crop masked + stabilized):
       every textured cell of a 5 × 5 grid at the same (−0.5, −0.5) px,
       which is what the VR180 still and its export differ by as well (the
       still decodes in software; older than this) — no scale or position
       error from the margin or the mask. Discs with the Apple flag (X6
       zero-copy, Osmo ProRes, libx265 8-bit, `.360`): parametric
       immersive to the system, its mesh within 0.0005 px of the written
       lens, the frame's edge at 97.5° / 92.5°, the picture within 0.15 px
       of the half-equirect twin in textured cells; a disc with the YouTube
       flag or none carries no projection box; VR180 exports keep theirs.
       Tests: pipeline lib 152, GUI 33 and its 3 GPU ones.
       **Not seen: the GUI itself** (the running app holds the user's
       clips; the Output panel's and the Format window's texts are tested
       headless), a disc with its lens on a headset, Windows.
   - **Every source (2026-11-15)**: the X6's full-view modes, DJI Osmo 360,
     QooCam 3 Ultra, Blackmagic RAW, a generic side-by-side file whose
     halves are fisheye images (dewarp on), and GoPro `.360`. Not a
     side-by-side file that is already VR180 (no lens: the export refuses,
     the GUI does not offer it), not a still. What it took:
     - `LensFrame::from_sources` takes any lens. Sources of the file's own
       family (unified camera model: the X6) go through the code they went
       through before, to the last bit. Any other radial law (Kannala-Brandt,
       with or without the kernels' rim extension, `source_radius`) is fitted
       by `fit_law`: for each ξ a least-squares a·m + b·m³ + c·m⁵, the ξ
       whose fit strays least at its worst sample (coarse scan 1.05 … 3.0,
       golden-section refinement, rounded to four decimals so every machine
       states the same lens). A ξ is out when its lens folds within 5° of
       the fitted reach, or when its radius does not rise with m EVERYWHERE
       (slope a + 3b·m² + 5c·m⁴ > 0 for all m): a frame has pixels past the
       fold (its corners), the kernels solve the radius for m by Newton's
       method, and only a radius that never turns back sends those pixels
       past the fold (black) instead of to whatever the iteration lands on.
       `LensFrame::reaches_its_corner` checks the same for the X6 fits (whose
       k2 can be negative, turning back far outside any frame) and the
       export refuses a frame that fails. Where no ξ qualifies over the whole
       reach (a preset's rim extension ends level at the frame's rim) the
       fit looks 2.5° less far, repeatedly.
     - Measured lenses (per-eye frame, ξ, focal, k1, k2; worst deviation
       from the source law): Osmo 360 3840² → 2.4309, 3566.92, 1.2118,
       0.2848 (3 px over 105°: its lens is nothing like equidistant, the
       density falls from 18 px/° at the centre to 11 at the rim, and it
       comes out next to the X6's); X6 8K 3840² → 2.45543 (the file's),
       3617.93, 1.1348, 0.4645, the same k for 7.4K and 10K; QooCam 3840² →
       2.4502, 4089.27, 0.1637, 0.7619; Blackmagic (preset, near
       equidistant) → 1.2366; the catch-all 180° preset → 3.0 (the range's
       end; 11 px off inside 90°, the extension's bend).
     - GoPro `.360` has no lens image: its frame is an equidistant fisheye
       of 92.5° half field over the cross's side (`LensFrame::equidistant`,
       `EAC_LENS_FRAME_HALF_FOV_DEG` = the disc kernel's own
       `FISHEYE_HALF_FOV`; ξ 1.4506, within 0.3 px, fold at 134°), at the
       cube map's own density when the frame is as large as the cross (3936
       on a MAX). `export_eac` resolves it (`resolve_eac_lens_frame`),
       `resolve_eac_lens_pair` hands it to the kernel through
       `EacLensAdjust::{out_lens, out_rim}` (uniform 80 → 96 bytes; the two
       existing kernels declare less and read none of it), and
       `gpu::eac_lens_frame_wgsl` splices the variant of
       `eac_to_fisheye.wgsl`: the shared `lens_frame_ray.wgsl` with `cal.`
       → `lens.`, and a `cover` term that blackens rays past the strip of a
       side face the cross carries (the disc kernel's "edge fill" repeats
       the last column there). No arm of `export_eac` was edited: all three
       project through `project_cross_texture_to_fisheye_texture[_16]`,
       which pick the variant by `lens.out_lens`.
     - **Black, not the repeated rim**: a full-view source repeats its rim
       past the image circle in the half-equirect and the disc (the user's
       choice there: it covers the gaps stabilization opens). A lens frame
       shows the frame's border all the time, so there it is black:
       `FisheyeCalib::with_frame_edge` (applied in `resolve_calib_pair`'s
       lens arm) = `with_frame_crop` for a lens model without the rim
       extension (θ clamp at the frame's far corner, capped 110°) and only
       `frame_mask` for a calibration that has the extension (its clamp and
       its law kept). The crop mask of that arm now applies to Crop sources
       alone, as everywhere else — a full-view clip can carry mask values
       it has no sliders for (one settings class with the Crop clips).
     - `resolve_lens_frame` reads the eye's size per source kind
       (`source_eye_dims`: the stream for dual-stream files, the split of a
       side-by-side or stacked frame, the Blackmagic layout), and no longer
       the first stream's.
     - **The stated limit**: unstabilized, the picture ends where the
       source's does, so the lens is stated to `LensFrame::picture_deg` =
       the source's θ clamp + the view adjust's tilt of the forward axis +
       2° (mounts), where that is short of the frame's corner: 113° for the
       full-view sources (clamp 110°), 95° for a `.360`. Stabilized, to the
       frame's corner or one degree short of the fold, as before (the
       picture may be anywhere in the frame): 113° X6 / Osmo / QooCam, 133°
       `.360`. Crop frames state what they stated before.
     - GUI: `lens_frame_possible(kind, settings)`; the frame's size is the
       clip's own resolution choice (`lens_frame_eye_dims(.., cropped)`: a
       full-view clip no longer reads the Crop clips' choice), never an
       enlargement (the 8K choice of a 3840² eye is 3840², of a 4864² one
       4096²); `.360` takes its cross (`native_eye` in `build_export_cfg`).
     - Verified (macOS 27.2): the 26 + 8 regression exports frame-identical;
       every earlier lens-frame case states the lens and limit it stated; 27 lens files of all source kinds and
       paths (zero-copy 10 / 8-bit, portable, libx265 10 / 8-bit, ProRes,
       stabilized, reduced copies) read by the system as parametric
       immersive, its mesh within 0.001 px of the written lens out to 95° …
       133°; pictures against the half-equirect of the same frames at
       400 Mbps: within 0.15 px for X6, Osmo, QooCam, Blackmagic and `.360`
       (cells at the hemisphere's edge excepted: the reference ends there).
       Tests: `lens_frame::tests` (15), `gpu::lens_frame_kernel_tests` (8),
       `fisheye_export::lens_frame_tests` (3), GUI `export_size_tests`.
       **Nothing wider than 180° has been seen on a headset.**
   - **Pipeline**: `FisheyeExportConfig::lens_frame: LensFrameOut` on the
     `Fisheye` projection. `export_fisheye` (the one entry of every fisheye
     path, all platforms) calls `resolve_lens_frame`: lenses only (no gyro)
     → `resolve_calib_pair` → `lens_frame::LensFrame::from_sources` → stored
     as `LensFrameOut::Resolved`, which `resolve_calib_pair` turns into
     `FisheyeCalib::with_lens_frame`; after the path returns it injects
     `spherical_inject::inject_apmp_lens`. No path function was edited, the
     Windows-only ones included.
   - **The lens** (`lens_frame.rs`): Mei-Rives, centred, ONE for both eyes —
     ξ from the source, focal / k1 / k2 a least-squares refit of the source
     lenses' mean radial law out to the frame's corner (two terms for the
     factory's four: within 1 px to the frame's edges, 3 px in the far
     corners). Frame = the source eye's own size (the fisheye view's square
     rule is skipped), or a reduced copy with the longer side 4096 for the
     8K choice. (Sources of other lens families: "Every source" above.)
   - **Kernels**: the five `fisheye_*_to_fisheye*.wgsl` files are NOT
     edited. `gpu::lens_frame_wgsl` splices `shaders/lens_frame_ray.wgsl`
     into each (anchors asserted) to make a variant, compiled on first use
     (`LensFramePipelines`, `Device::fisheye_out_*` accessors chosen by
     `FisheyeCalib::out_lens`). A branch inside the kernels was tried first:
     it changed the disc's output at rounding level (the compiler optimises
     the changed source differently) and failed the regression set. The
     uniform grew to 13 vec4 (208 B): `out_lens` + `out_rim` in a spare slot;
     the half-equirect kernels do not declare the new row.
   - **Edge policy**: in a lens frame a ray that leaves the SOURCE frame is
     black: the kernels' own `frame_cover`, as in the half-equirect (soft
     over one output pixel). It repeated the edge at first, as the reframed
     view does; on the headset the smeared band was an eyesore, so the
     variant no longer splices `frame_cover`. The view mask applies;
     `out_rim` (8 px at 4912) keeps the frame's own border black for players
     that clamp.
   - **Room for the stabilizer**: the recorded frame has no spare canvas, so
     a STABILIZED lens frame is larger than the recording by
     `lens_frame::STABILIZER_MARGIN` (5 % of the shorter side) at each
     border: `resolve_lens_frame` → `LensFrame::grown` (same lens, wider
     field), and `cfg.eye_w / eye_h` follow, so no path function knows about
     it. One figure for every clip, so the clips of a recording mode come
     out the same size (7.4K Crop 8160 × 4080, 10K Crop 10800 × 5400) and
     the GUI shows it ahead of the export (`lens_frame::stabilized_dims`).
     A shift beyond the margin is cut at the frame's edge. A reduced copy
     (the 8K choice) keeps its size: `fitted` scales the grown frame back
     into it. Stabilization off → the frame is the recorded one, byte for
     byte as before. A frame sized to each clip's own largest shift (built
     first, with a 5 % cap) gave every clip another size, which the user
     did not want. Measured on hand-held walks with soft stabilization
     (1.2 s, cap 15°): the picture leaves the recorded frame by 4 px at the
     median and 164 px (3.9°) at worst of 4912 on a 10K Crop clip; 46 px and
     343 px (8°) of 3712 on a 7.4K Crop 60 fps clip, whose frame is narrower
     (97° across) — with its 184 px margin that one is still cut at a side
     in 6 % of its frames (159 px at most).
   - **Crop mask**: on a lens frame the window is sized from the frame's
     OWN coverage (`LensFrame::half_angles`, applied in `resolve_calib_pair`
     over the one `apply_frame_crop` set from the recorded picture), so on a
     stabilized frame it can be as wide as the larger frame; the picture's
     moving border then shows inside it (the user's call). Without
     stabilization the two measures agree to 0.1°. The preview draws the
     same frame through the same function, so the mask shows where the
     export has it.
   - **File conventions, measured against the system's player mesh** (not
     what the model's textbook form suggests): the projection offset is
     stored NEGATED; the principal point counts from the view's edges; the
     player builds one mesh from the FIRST lens of the collection. Byte-exact
     tests against AVFoundation's writer.
   - **Verified** (macOS 27.2): 26 fisheye-export regression cases
     frame-identical; lens frames through zero-copy 10 / 8-bit, the portable
     loop, libx265 10 / 8-bit, ProRes, noise reduction, stabilization on and
     off — each read by the system as parametric immersive, its mesh within
     0.001 px of the written lens, its picture within 0.2 px of the same
     frames' half-equirect re-projected by an independent script. GPU tests in
     `gpu::lens_frame_kernel_tests`. Test clips of this form were viewed on
     a headset (visionOS 27): immersive, placement and scale as the
     half-equirect.
   - **Windows: untested** when this was written; verified 2026-11-17, see
     the Windows paragraph below. Shared code only (no new gates; two
     `cfg_attr(not(macos), allow(dead_code))` on accessors of the macOS-only
     P010 kernels). To check there: the spliced WGSL compiles on D3D12 /
     Vulkan, a lens-frame export through NVENC (9824 wide falls to libx265
     as the 13120 canvas does) and through the GPU-resident path.
   - **WINDOWS VERIFIED (2026-11-17, Windows box, RTX 4090).**
     **One real bug, fixed in shared code: every lens frame panicked on a
     Windows checkout.** `core.autocrlf` hands `include_str!` the kernels
     with `\r\n`, the splices' anchors of several lines are written with
     `\n` (rustc normalises its own source, not an included file), so
     `lens_frame_wgsl` / `eac_lens_frame_wgsl` asserted "anchor not found"
     the first time a lens frame was projected: the preview worker or the
     export thread died in the Lens Space view's default lens. The eight
     `lens_frame_kernel_tests` caught it. `gpu::unix_lines` now normalises
     the kernel and the spliced ray generator before the splices; nothing
     else in the crate matches a multi-line pattern against an included
     shader.
     After the fix, lens frames from an X6 8K clip (stabilized and not), a
     10K Crop clip with a crop mask, an Osmo `.osv`, a QooCam clip and a
     GoPro `.360` through the portable loop, the D3D11 zero-copy readback
     arm and the GPU-resident NVENC arm (`.360`: portable + its NVENC arm):
     all engage, the lens stated matches the Mac's figures (Osmo ξ 2.4309
     k 1.2118 / 0.2848, QooCam 2.4502 / 0.1637 / 0.7619, `.360` 1.4506 to
     133°, X6 8K to 113°, 10K Crop to 101°), the files carry the lens boxes
     (`prji prim`, `lnsc` with two `lens`), and the arms agree: portable −
     zc under 0.01 of 255 signed, either − NVENC under 0.13, mean |Δ| 0.4 to
     0.9 (encoder noise). A disc with the Apple flag states its lens (to
     98°, `.360` 93°), a plain disc carries no projection box, a VR180
     export keeps `hequ` + `hfov`. D3D12 (`VR180_BACKEND=dx12`, now also
     honoured by `Device::new` as a diagnostic so the tests can run there):
     the spliced kernels compile and export on all three kernel families;
     231 of the 232 pipeline tests pass on D3D12, the one that does not is
     `a_lens_frame_of_its_own_lens_is_the_source`: the interior is within 27
     of 65535 (0 on Vulkan) and the frame's outermost row and column are up
     to 0.5 % dim (320 of 65535), the HLSL build's arithmetic at the frame's
     own edge. Left as it is (the test runs on Vulkan by default).
     PREVIEW, in the app (window captured from the screen; `PrintWindow`
     returned a blank window this time): the X6 clip and the Osmo clip on
     the D3D11 zero-copy worker and the `.360` on the EAC zero-copy worker
     draw the lens frame with the stabilizer's margin and play at their full
     cadence; the edits made by text compile and run. No Match QuickTime
     switch on Windows.
     **NVENC stops at 8192 a side, and a stabilized lens frame of a 3840²
     source is 8448 × 4224** (`.360`: 8656 × 4328; 10K Crop 10800 × 5400,
     unstabilized 9824 × 4912). Measured from an X6 8K clip: unstabilized
     7680 × 3840 through the GPU-resident NVENC arm 30 fps; stabilized
     8448 × 4224 as H.265 falls to libx265 on the zero-copy readback arm,
     2.1 fps, with the export-bar note and the Format window's note saying
     so; as ProRes HQ it stays on the GPU (`prores_ks_vulkan`), 14.9 fps.
     **THE 8K CHOICE HOLDS THAT FRAME TO THE ENCODER, WINDOWS ONLY (same
     day, the user's call).** Where the hardware H.265 encoder has a frame
     limit (`app.rs::hw_h265_side_limit`: 8192 off macOS, none on a Mac) and
     the stabilized frame of an eye no larger than 4096 is past it, the 8K
     choice (the default) asks for `lens_frame_held_dims`: the frame with
     its margin scaled to 4096 on its longer side (3840² and the `.360`'s
     3936²: 4096², 8192 × 4096 written; 3712² fits as it is and is not
     touched). The size asked is then LARGER than the recording, which the
     pipeline never enlarges: `fisheye_export::lens_frame_request` builds
     the frame at the recording's size and holds the stabilized frame to the
     size asked (`size_lens_frame`'s `reduced_to`), and without
     stabilization writes the recorded frame; `resolve_lens_frame` and
     `resolve_eac_lens_frame` both go through it. The clip's own size stays
     8448 × 4224 (software H.265 with its note, GPU ProRes). The Format
     window lists "8192 × 4096" for such a clip and says what it is
     (`LENS_FRAME_HELD_NOTE`); the export bar shows the size written. On a
     Mac nothing changes: the 8K choice of a 3840² eye is the frame as it
     is, as before (the two tests that say so now read the platform).
     Measured: X6 8K stabilized at the 8K choice, 8192 × 4096 through the
     GPU-resident NVENC arm at 25.6 fps (the native 8448 × 4224: 2.1); lens
     stated at 4096 / 4224 of the recording's focal; against the native
     export of the same frames scaled down, mean |Δ| 0.6 of 255 with no
     shift and the same share of picture in the frame; a `.360` likewise
     (4328² held to 4096²). Exports that ask for the recording's size or
     less are byte-identical to before.
   - Not done: no MV-HEVC lens frame, no crop mask on a full-view clip, no
     automatic mask sized to the stabilizer's corrections; the sharpen
     stage still fades towards the top and bottom rows of a lens frame
     (`SharpenParams::apply_lat_weight` is on for every plan, a
     half-equirect's weight; the user has been told, their call).

34. **Insta360 X6 raw video (2026-11-16, Mac; UNRELEASED).** A raw take —
   a folder with `capture.x6b` (both lenses' 14-bit Bayer frames in pairs),
   the gyro file, the factory lens string and the sound — opens as one clip
   (`SourceKind::X6Raw`; import takes the folder, the capture file or a
   folder of takes).
   - **Reading**: `vr180-fisheye::x6b` (take description, pair scan, tile
     decoder, checked against reference pixels of three sample takes);
     `vr180-pipeline::x6_raw` (one index per take, cached on disk under the
     settings folder; frame timeline from the exposure stamps; lens and gyro
     poses; `X6RawIter` hands out pairs demosaiced on the GPU as
     camera-linear `Rgba16Unorm` eyes, at half size for the preview). Raw
     clips take the 16-bit stacks only; video export is forced to 10-bit.
   - **Motion and timing** (2026-11-16, on the first takes that carry a
     sensor record; `x6b::read_sensor_readout`, `read_accel`,
     `x6_raw::capture_timing`, `build_imu`). `clip.json.sensor_metadata`
     → `sensor-metadata.json`: per eye `line_period_ns` (3791.67 at 30 and
     60p, 4000 at 50p) and `rolling_shutter_first_to_last_row_ns`
     (14.556 / 15.356 / 14.556 ms: the camera's own video files state the
     same numbers); `accel.csv` + `accel-info.json` beside `gyro.csv`
     (m/s², gravity in it, the gyro's stamps; `status` other than `ok` =
     none, one of the seven takes has it so). RULE (rule 7: from the
     file, nothing tuned): a frame's stamp is the start of its first
     row's exposure, so the pose is sampled at the centre row's start,
     `stamp + readout / 2`, and the rows span the readout around it, top
     row first (`readout_ms` reaches the shared per-row path through the
     pose structure; `readout_ms_for` honours it). With an accelerometer
     the poses are built as the camera's own video files' are (bias from a
     6-axis pass, the integration, the 100 s trim); without one, as
     before. A take without the record: pose at the stamp, no per-row
     correction, as before.
     MEASURED (`~/vr180-internal/x6raw-imu/raw_gyro_flow.py`: features
     tracked in the binned mosaic against the gyro through the factory
     lens model, per-row times; six hand-held takes): the gyro's axes are
     the camera's own files' (the best of the 24 signed permutations, and
     a free rotation moves it 1–2°); the accelerometer shares them (its
     direction holds to 1–2° in the gyro-integrated frame through a
     minute, and reads −1 g on x at rest as the camera's files' does);
     readout 14.26–14.54 ms fitted for the record's 14.556 and 15.38 for
     15.356, top row first; centre-row content 9.9–11.4 ms after the
     stamp = half the readout + 2.7–3.6 ms. THAT REST IS NOT IN THE TAKE:
     half the shutter time plus the gyro's own lag (the same hardware's
     video files state 1.1 ms for the latter, which would make the
     shutter about 1/200 s). It is not added (no tuned phase numbers).
     What it costs, far-field shake of a locked export at 1024 px an eye,
     the hardest take: unstabilized 2.60 px a frame; pose at the stamp
     (before) 1.52; this rule 0.42; with 3 ms more 0.06. Per-row
     correction, same take with those 3 ms: top of the frame 0.54 → 0.13,
     middle 0.15 → 0.06, reversed 1.14. ASKED OF THE CAMERA SIDE: the
     shutter time of every frame (and the gyro's lag if known).
     ANSWERED the same evening (their note beside the takes): a later
     camera build writes `exposure.csv` beside the capture, header
     `request,cam0_exposure_ns,cam1_exposure_ns`, the integration time of
     each eye in ns, rows only for pairs that were written, an eye's
     value left empty when the sensor did not report it (`seal.json`
     counts such pairs in `exposure_missing_pairs`); row `r`'s
     mid-exposure = that eye's stamp + `r × line_period_ns` + half its
     integration time; zero, empty or no file = unknown, the old rule.
     Checked on their side against the sensor's own reported exposure
     (58 eye frames, then 140 on a 50p take where it ran 8.66–12.60 ms).
     IN THE APP: `x6b::read_exposures`, `build_imu` adds half the frame's
     own time to `stamp + readout / 2` (channel 0's value, the other
     eye's when that one is empty; nothing for a frame without one); the
     log says the range in use or that there is no record. RUN ON A REAL
     TAKE the same night (`1116 Raw Video/RAWVIDEO-20261116-205643`, 60p,
     3050 pairs; `clip.json` names the list in `exposure_file`, which
     `Take::exposure_file` honours; `seal.json` has `exposure_status` and
     `exposure_missing_pairs`; sound is `audio.wav`, two channels): the
     list has exactly the capture's requests, no empty value, the two
     eyes equal, 6.617–16.430 ms in 27 distinct values. WHAT THE STAMP IS,
     FROM THE NUMBERS ALONE: `stamp + integration time` ticks at the frame
     period to 0.0000 ms rms through all 32 changes of shutter (request
     809 → 810: 16.430 → 7.906 ms, the stamps 25.207 ms apart, the sums
     16.6833), so the stamp is the first row's exposure START and the
     list's values belong to their requests. THE FRAMES' TIMES
     (`TakeIndex::new`, `shutter_ns`, `shutter_s`): where every pair has a
     listed time, a pair's time is the stamp plus it (the first row's
     readout, even to 8 µs over that take); on the stamps alone that take
     read as "1 dropped in 1 hole" at request 810 and its frames wandered
     up to 9.8 ms off an even clock. Takes without a complete list keep
     the stamps. The user asked for no picture-based check on that take
     (long shutters, much motion blur), so nothing was measured from its
     pictures. STILL OPEN: the gyro's own lag
     (their words: the sensor service's stamps, filter delay unmeasured;
     the 1.1 ms of the camera's own files and this unit's fitted axes
     are not to be taken as constants for every unit) — with a take that
     has `exposure.csv` it can be measured as the fitted offset minus
     half the shutter (`raw_gyro_flow.py`). For the takes already shot
     they agree no shutter time can be recovered and no uniform 3 ms
     should be filled in.
     Diagnostics: `VR180_IMU_PHASE_OFFSET_MS`, `VR180_RS_SCALE`,
     `VR180_INSV_BASIS`, `VR180_LOCK_REF_FRAME`.
   - **Demosaic** (2026-11-16; `bayer.rs` = the specification and the CPU
     twin, `Device::demosaic_bayer` = three passes: `bayer_measure` →
     `Rg32Float`, `bayer_green` → `R32Float`, `bayer_demosaic` → the eye;
     the two work planes are kept in `RawPipelines::demosaic_work`). It
     replaced Malvar-He-Cutler (zipper along every edge, colour in it) and,
     for the preview, a 2×2 binning (colour fringes: red and blue were
     taken a site apart). It is shaped after Camera Raw's own demosaic,
     found by measuring it: the same mosaic developed there from a CFA DNG
     and from the linear DNGs of our candidates, Camera Raw's result taken
     back to camera values through the development's Jacobian (three more
     renders with one channel raised 2 %, one Newton step and a check
     render; scripts `~/vr180-internal/x6raw-acr/dem*.py`, `dem4.py` does
     the inversion, `dem7.py` is the final numpy twin). What fits: (1) the
     interpolation is done on the mosaic WHITE-BALANCED by the camera's
     neutral (`BayerLayout::neutral`; without it red and blue are two to
     four times further off); (2) green at a red / blue site = the row's
     estimate and the column's, each the two neighbours' mean + 1/8 of the
     site's own second difference (the fit is sharp at 1/8: 1/16 and 3/16
     are 40 to 90 % worse), mixed by `B² / (A² + B²)` with A, B = the sum
     over 5×5 of |neighbour − neighbour| + ½ |second difference| along the
     row / the column (window and power scanned; a decision made on the
     colour difference after interpolating fits worse); (3) red and blue =
     green + their difference from green carried bilinearly (kernels
     fitted freely to Camera Raw's result come out bilinear). Against
     Camera Raw in camera values ×1000 (level ≈ 30 / 45 / 15), three takes
     and both eyes: 0.24–0.40 / 0.19–0.26 / 0.11–0.36 against Malvar's
     0.55–0.78 / 0.33–0.53 / 0.69–0.98; developed, mean |difference| of
     255: 0.25 / 0.11 / 0.38 against 0.93 / 0.33 / 2.67; at clipped
     highlights (a frame raised three stops) 0.2–0.4. Cost: 3.1 ms an eye
     at full size, 2.5 ms at half (the half-size eye is the mean of each
     2×2 of the full result), M5 Max. Camera Raw's flat-ground response
     to single samples (`dem3.py`) says the same: green = four neighbours'
     mean + a second-difference term of 0.115–0.12 in balanced units.
     Stills (`raw_develop::mhc`) still use the old kernels. WINDOWS:
     `Rg32Float` and `R32Float` as storage textures, read back with
     `textureLoad`.
   - **Colour noise reduction** (2026-11-16; `RawSettings::color_noise`,
     0–100, default 25 = Camera Raw's; `raw_chroma.rs` = the specification
     and the CPU twin; `Device::record_raw_chroma`, first in
     `record_color_stack_16`, AHEAD of the development, on the camera's
     values: `raw_chroma_reduce` / `_smooth` (rows, columns) / `_apply`,
     two half-size `Rgba16Float` planes). Camera Raw opened with its
     defaults runs it, which is why a DNG there shows no colour in fine
     lines where its own demosaic — and ours — still has a little.
     Measured (`cnr1.py` … `cnr3.py`, `cnr_app.py`): it leaves luminance
     alone (L* amplitude 1.00–1.02 at every frequency) and takes the
     colour's fine part down (a* / b* to 0.2 / 0.33 at 0.25 cycles a
     pixel, 0.51 / 0.71 at 0.09, on a frame at 25); it is a threshold, not
     a blur (a 4 % grating in one channel goes grey at every period up to
     100 px, a 32 % one keeps 80–90 %; the share left by level × period ×
     amplitude × amount is in `data/cnr2.npz`); the luminance it keeps is
     0.195 / 0.667 / 0.138 of the white-balanced camera R / G / B (not the
     camera's CIE Y). The stage: balanced values at half size → a
     separable bilateral there (spatial sigma / 2, colour distance `tau`,
     the rows' result put back at the pixel's own luminance) → bilinear
     back, each pixel taking the smoothed colour at its OWN luminance.
     Amount → (sigma px of the export, tau): 10 → (4, 0.1), 25 → (6, 0.2),
     50 → (12, 0.3), 100 → (16, 0.5), fitted to Camera Raw's result taken
     back to camera values (`cnr3.py targets`) on take 011809 and checked
     on 021546. Against it in balanced camera values ×1000 at 25: 0.90 /
     0.33 / 0.76 with the stage, 1.77 / 0.49 / 1.51 without, floor (the
     demosaics alone) 0.75 / 0.24 / 0.45. End to end on the app's projected
     frame (`cnr_app.py`, the change each makes): strength 0.91–0.95 of
     Camera Raw's at 10–25, 0.89 at 50, 0.80–0.86 at 100; correlation
     0.88–0.95; error 32–51 % of the change (both leave a different faint
     residue). Cost +10 ms a frame at 8192×4096 (+15 at amount 100). The
     harness leaves it off unless `X6_RAW_CNR=<amount>` (the reference
     renders of the other scripts have none). NOT done: Camera Raw's reach
     at high amounts (multi-scale), its Detail / Smoothness sub-sliders
     (written as 50 / 50 in the XMP), luminance noise reduction.
   - **Development** is the first stage of `record_color_stack_16`
     (`ColorStackPlan::raw_develop`). `raw_tone.rs`'s header is the
     specification; `RawDevelop::constants_for(stats, curves)` is the one
     entry. It reproduces Adobe Camera Raw (process version 15.4), measured
     on Camera Raw 18.7 by developing charts and frames there:
     exposure → default black (keyed to the frame's darkest cell) →
     Shadows / Highlights (`gain = curve(base)`; base = a bilateral grid of
     log₂ luminance, `raw_local.rs` + `raw_reduce` / `raw_grid_build` /
     `raw_grid_blur.wgsl`; the frame's curves from networks fitted to
     Camera Raw's own on 1400 distributions; `LocalCurves` holds how the two
     sliders act on each other; gains above zero are limited toward the
     white point, `limit_gain`) → white point of the frame WITH the gain in
     (`gained_white`) → Whites → Blacks (`raw_global.rs`: tables; each a
     gain keyed to the pixel's luminance or a curve per extreme channel, by
     sign — `StageMap`) → the tone curve with Contrast in (a family of
     maps + networks; Whites above zero is `a1·v + a2·v²` from networks).
     Plain settings run `raw_develop.wgsl` (one curve); anything between
     black and tone curve runs `raw_develop_local.wgsl` (the staged one).
   - **Per frame**: `Device::raw_frame_stats` measures the frame as it is
     graded (`raw_measure.wgsl`: 126 cells along the long side) —
     `FrameStats { keys, levels, peaks }`. `Measure::Wait` for exports and
     stills, `Measure::Latest` for the preview (uses the last measurement
     that came back and asks for a redraw when it changes).
   - **One look in every view** (2026-11-16): the measurement belongs to
     the picture it is made on, so a reframed view (a part of the scene) or
     a lens frame (a black border) came out in other colours than the
     half-equirect — 8–13 of 255 with Shadows / Highlights at their ends.
     Every view that is not the half-equirect is now developed from a
     REFERENCE FRAME: the same frame's eyes drawn as half-equirects of
     `RAW_REFERENCE_SIDE` (1008 px) with the view's own rotations and
     per-row matrices (`raw_gpu::RawReference`, `Device::raw_reference`,
     `ColorStackPlan::raw_reference`); the keys and curves are read off it,
     and the local stage's cell is sized from `RawReference::long_side` =
     eyes × π × `FisheyeCalib::center_px_per_rad` of the view, so its
     neighbourhoods cover the same angle as in the half-equirect. Callers:
     `fisheye_export::plan_with_raw_reference` (the export loop's two
     10-bit arms, `raw_export`'s developed frame), `decoder.rs`
     `raw_view_reference` (live preview, still). Both resolvers are split
     for it: source calibrations (`resolve_source_calib_pair`,
     `resolve_fisheye_source_calib_pair`) and the view put on them
     (`view_calib_pair`, `fisheye_view_calib_pair`). Views against the
     half-equirect at the same scene points (`~/vr180-internal/x6raw-acr/
     viewcmp.py`, three frames, Shadows +100 / Highlights −100): disc 0.2–0.3
     of 255, lens frame 0.3, reframe 90° 0.5–0.8, 40° 0.8–1.3 — the
     resampling floor of that comparison is 0.2 / 0.2 / 0.3–0.4 / 0.6–1.0,
     mean bias under 0.25. What is left: the base grid is still built on
     the view's own picture, so at a strong zoom (20°) with both sliders at
     their ends regions differ by about half a step (a few at most); the
     cure would be a grid built on the reference and looked up through the
     view's ray. The half-equirect itself is unchanged (its eyes are the
     reference), and so is its match with Camera Raw; a DNG of another view
     is developed by Camera Raw from that picture alone.
   - **Sharpening** (2026-11-16): `RawSettings::sharpen` (`raw_sharpen.rs`:
     Amount 0–150, Radius 0.5–3, Detail 0–100, Masking 0–100; off at
     Amount 0) — Camera Raw's detail panel, measured there with charts
     and frames (scripts `~/vr180-internal/x6raw-acr/sharp*.py`, data
     `data/sharp_*.npz`). What was found, and what the stage does:
     it works on the DEVELOPED picture (its fades follow the output level
     whatever the tone settings), on `φ = Y^(1/1.8)` of the Rec.709
     luminance, and scales a pixel's linear RGB by one gain. On a HARD
     edge the change is a fixed kernel per (Amount, Radius, Detail), read
     off the response to a small step for 14 × 9 × 11 settings and held as
     five blurs (`data/raw_sharpen.f32`: `Δ = Σ cⱼ (φ − Gⱼ ∗ φ)`, σ = 0.5,
     0.8, 1.3, 2.1, 3.4; taps within 0.0015). **Low Detail = halo
     suppression** (found 2026-11-16 on edges of five softnesses over
     882 settings, `sharp8.py`, `halo_b.py`): the sharpening is the step
     kernel over a factor `h`, and of its change the part that keeps the
     pixel within the range of the original around it (3×3) is kept, the
     part beyond — the halo — is scaled by `h` (`raw_sharpen::halo_change`,
     `halo_factor`; the table holds `h` beside the coefficients). A hard
     step is all halo, so it gets the step kernel exactly; a soft edge
     gets up to `1 / h` of it. `h` = 0.34 at Detail 0 for every Amount and
     Radius, 0.5–0.75 at Detail 5–10, 1 from Detail 20–55 (later at large
     radii), and it falls again above Amount 85 at Detail 5–25; below
     Amount 10 the chart does not resolve it and Amount 10's is used. The
     range is taken over one pixel of the EXPORTED frame
     (`SharpenConstants::window` = the pixel scale rounded, 1 to 3). Then a
     soft limit, a fade by the pixel's own level (none below φ 0.028, all
     from 0.204 to 0.851, none above 0.969), limits on how far a pixel of
     each level may be darkened or lightened (44 measured levels), and
     Masking = one curve of (edge strength / threshold): the gradient's
     size spread by a blur of 1.2 + 0.5·Radius px, the threshold a table
     over Masking. GPU: `Device::record_raw_sharpen` right after the
     development in `record_color_stack_16` (`raw_sharpen_level` / `_blur`
     / `_apply.wgsl`, four `R32Float` planes, 12 to 15 passes an eye; the
     apply pass reads the level plane's 3×3 for the range); CPU twin
     `raw_sharpen::sharpen_picture`. The radius is in pixels of the
     EXPORTED frame: `ColorStackPlan::raw_pixel_scale` (the GUI sets it
     from `Settings::export_eye_w`, which `maybe_push_settings` keeps at
     `export_eye_dims().0`) scales the blurs in a preview. The settings
     go into the DNG's XMP. Against Camera Raw on three frames (the change
     each makes; `sharp_app.py`): Detail 25 error 18–34 % of the change,
     correlation 0.96–0.985; Detail 100 12–15 %, 0.99; Masking 40 / 80
     23–40 %; Detail 0: Amount 40 29–36 %, Amount 100 19–24 %, Amount 150
     16–20 % (before the halo rule 41–63 %, the app changing half as
     much); Detail 5–20 at Amount 40 21–33 %; Radius 3 at Detail 0 44–52 %.
     NOT reproduced: with any Amount above zero Camera Raw first smooths
     the finest texture (edge-preserving, scale-free: it leaves a step
     alone, takes a 1 px line down by 7 to 28 % without spreading it, acts
     mostly on local extremes; the same for every Radius up to 1.4, growing
     above; gone at Detail 100). Its size on the sample frames is 0.24/255
     rms at Detail 0 (0.66 at Radius 3), 0.16 at Detail 25, against 1.2 of
     the Amount 40 sharpening at Detail 0; running the model on Camera
     Raw's own Amount 1 picture brings Detail 0 to 18–25 % and Detail 25
     to 15–18 %, so it is most of what is left. At Amount 100+ the
     limiting at big edges is 20–40 % off. Cost: +14 ms a frame at
     4096×2048, +24 ms at 7680×3840 (Apple silicon).
   - **DNG**: `raw_export.rs` — a frame as a linear DNG with the settings
     as `crs:` XMP (written beside every Snapshot of a raw clip), and the
     export option "DNG sequence" (`ExportOptions::raw_dng_sequence`,
     folder `<stem>_DNG`). A Lens Space frame through the camera's lens is
     the lens frame the video export writes (`FrameSource::open` resolves
     it with `resolve_lens_frame`; before 2026-11-16 these frames held the
     equidistant disc).
   - **Sensor frames** (`sensor_dng.rs`, export option "DNG sequence" →
     "Camera original", `ExportOptions::raw_dng_original`, folder
     `<stem>_OriginalDNG`): the take's frames as recorded, one CFA DNG per
     channel per pair, `CAM0/<request:09>.dng` + `CAM1/…`, no GPU and
     nothing of the app's rendering. File = one little-endian IFD, 14-bit,
     `PhotometricInterpretation` 32803, CFA `[1, 2, 0, 1]`, black 64,
     white 15408 (codes above it kept), 3840×160 tiles of lossless JPEG
     (`ljpeg::encode_tile_cfa_into`: two components, half the width, so a
     sample is predicted from its own colour), `ColorMatrix1` /
     `AsShotNeutral` of `RawCamera::X6`, `UniqueCameraModel` "Insta360 X6
     Raw Video" (a name no converter has a profile for, so the file's own
     matrix is used: the condition the raw development was matched under),
     `ImageDescription` "CAM<n> request=<r> exposure_start_ns=<t>"
     (`FrameTag`). Beside the frames: `frames.jsonl` (one line a pair:
     request, both exposure starts, the record's signature, the files and
     their sizes) and the take's small files copied (everything but the
     capture and `seal.json`). One thread reads the capture front to back,
     up to 12 workers unpack / pack / write, a file appears under its name
     only when whole (`.part` + rename); a cancelled run keeps and lists
     what it finished. Measured (M5 Max): 57 pairs a second from a cached
     source (CPU-bound, 0.18 core-seconds a pair), 13 from the NAS (its
     read speed); 12.0 MB per eye frame at 30 fps, 10 at 50, 8.7 at 60;
     1.2–1.5 GB of memory. Checked: samples identical to the capture's
     (test `a_stretch_of_a_sample_take`), `exiftool -validate` OK, opens
     in ImageIO and in Camera Raw 18.7. A folder written this way is not
     yet something the app opens: `x6b::read_take` names it
     (`TakeProblem::FramesOnly`).
     WHAT THE UI SAYS OF IT (2026-11-16, after the user asked whether a
     Camera original export "still used NR" and a "native" resolution):
     the export itself never did (`spawn_export_job` routes it to
     `export_sensor_dng_sequence`: 1884 pairs in 57 s on the user's run),
     but the export bar's "NR" followed the clip's Noise Reduction
     slider and Export format still listed a Resolution. Now
     `export_runs_noise_filter` says when an export runs the filter (a
     video file; never a folder of DNG frames or a still), and Export
     format shows "3840 × 3840 per eye" in place of the list for Camera
     original.
   - **Enhanced frames** (2026-11-16; `enhanced_dng.rs`, `jxl_macos.rs`,
     `X6RawIter::with_enhanced_frames`, `FisheyeExportConfig::
     raw_enhanced_dir`, GUI `ExportOptions::raw_use_enhanced` → the
     `<stem>_OriginalDNG` folder beside the take when it has a `CAM0`).
     Camera Raw 18.7 applies Super Resolution IN PLACE (no second file):
     the DNG keeps IFD0 and its mosaic and gains three tags — `0xCD3C`
     (16-byte digest), `0xCD3D` (offset), `0xCD3E` (length) — for a block
     appended to it: a 60-byte header (5, 2, the digest, width, height,
     planes 3, sample type 3, a length, 2, 1, 0, 0), then a TIFF of its own
     (one IFD, 7680×7680, 3 × 16 bit, `Compression` 52546 = JPEG XL, 90
     tiles of 768×864, each a bare codestream tagged BT.2020 / sRGB gamma;
     32 MB for a frame: lossy). The codes are per-channel companded; the
     XMP says how (`crs:FilterList` → the `Enhance` filter → `Images` →
     `ColorVariations`: three `crs:MapPolynomial` of four fractions and
     three `crs:Black="black,white"`): camera value = `c0 + c1·x + c2·x² +
     c3·x³`, `x = (code − black) / (white − black)`, white level = 1 (in
     the sample `c1·(x + 15x³)`, each channel scaled to its own maximum).
     Checked against Camera Raw's own demosaic of that frame taken back
     to camera values: 8×8 block means within 0.03 / 0.09 / 0.28 %. Other
     settings in that XMP (the user's Shadows, sharpening …) are NOT in
     the picture: it is pre-development. `find` parses, `decode_rgba16`
     decodes (tile rows on threads) through ImageIO (`jxl_macos`: the
     provider's bytes of the decoded `CGImage`, no colour conversion;
     within 6 codes of `sips`). The iterator: eyes become the picture's
     size, both channels decode side by side, a channel with no enhanced
     picture is the capture's frame demosaiced on the CPU and enlarged
     bilinearly (`enlarged`), pairs go to the GPU as `new_camera_eye_
     texture` uploads so everything downstream (the reference frame, the
     16-bit stack) sees an ordinary GPU pair; the lens scales itself (the
     export scales the take's lens by eye size / lens canvas). Measured:
     0.41 s a pair with both eyes enhanced at 8192×4096 output (capture:
     0.07 s), 4 GB peak; render from the enhanced eye aligned to the
     capture's (shift 0, 0), mean |difference| 0.5/255 (it is sharper).
     WHICH FRAMES: the folder is listed once per export
     (`enhanced_listing`: each channel's requests and file sizes; only
     files named as `sensor_dng::frame_file` names them), and the export
     says which stretch of the take it covers (`x6_raw::EnhancedSource {
     dir, t_in, t_out }` from `FisheyeExportConfig::raw_enhanced()`; a
     single-frame export passes that frame's time and a frame later).
     Among the stretch's files the LARGEST are opened first
     (`enhanced_candidates`: a file keeps its mosaic and gains the
     picture, so an enhanced one is larger than a plain one wherever it
     sits; 8 of them, then the stretch's first / middle / last). No
     enhanced picture in the stretch → the iterator stays the plain one
     (0.07 s a pair): a folder that holds a few test frames costs an
     export nothing. (The first form probed the take's frames in order
     and gave up after 600 files: frames enhanced past the first 300
     pairs were not found.) Per frame the listing says whether a file is
     there (no stat on the share). A frame of an enhanced stretch without
     a picture: `bayer::demosaic` + `enlarged`, both on every core
     (`bayer::fill_rows`): 0.38 s a pair (single-threaded it was 1.4 s).
     The export bar's note says "frames from the enhanced DNGs, 7680 ×
     7680 per eye". Output sizes from enhanced frames, measured
     (hardware H.265 Main 10, M5 Max, ten pairs with the start-up):
     8192×4096 0.45 s a pair, 4.2 GB peak; 13120×6560 0.60 s, 5.7 GB;
     15360×7680 0.66 s, 6.6 GB; 16384×8192 0.69 s, 7.2 GB: all four
     encode and decode.
     SIZES FOR ENHANCED FRAMES (the user: both of 13120×6560 and
     16384×8192): `ExportResolution::{R13k, R16k}` (6560² / 8192² an
     eye), picked from a list of their own (`SizeList::Enhanced`,
     `ExportOptions::enhanced_resolution`, beside `Plain` and `Crop`:
     a size picked on one sort of clip never changes another's). An
     export carries BOTH sizes: `eye_w × eye_h` = the one picked without
     enhanced frames, `FisheyeExportConfig::raw_enhanced_eye` = the one
     picked for them (canvas views only: not a lens frame, not a
     reframed view), and the PIPELINE decides before anything is sized
     (`settle_raw_enhanced(t_in, t_out)`, first thing in
     `export_fisheye` and in `raw_export::FrameSource::open`; it goes by
     `x6_raw::enhanced_side`, the iterator's own search): enhanced
     frames in the stretch → the larger size; none → the folder is
     dropped, the plain size stays, and the export bar says so when a
     larger one was asked for. So a batch item, a different trim or a
     folder that changed since the list was shown cannot write an
     upscaled capture. The GUI lists the two sizes only when a lookup
     found enhanced frames for the loaded take's trim
     (`App::enhanced_frames`: `enhanced_side` on a thread, asked again
     when the take or trim changes and every 4 s while Export format is
     open; `EnhancedFrames::{Looking, Absent, Found(side)}` is also the
     line under "Use enhanced frames"). The preview's sharpening scale
     follows the size an export would write (`raw_export_eye_w`); the
     snapshot and its frame DNG stay at the plain size.
     The harness: `X6_ENHANCED=<folder>`. CFG: `#[cfg(target_os =
     "macos")] mod jxl_macos` in `lib.rs` and ONE gated pair of blocks in
     `enhanced_dng::decode_tile` (the other arm returns an error); nothing
     else is gated, tests use `cfg!()`. NOT done: Windows (needs a JPEG XL
     decoder crate), the preview and zoom view from enhanced frames,
     export sizes above the present ones for a 7680 source, Camera Raw's
     other in-place enhancements (Denoise, Raw Details: same container
     presumably, 1× size; untested).
   - **Against Camera Raw** (three frames, mean |difference| of 16-px block
     means, of 255): default 0.33; white balance 0.1–0.7; Shadows /
     Highlights ±50 0.7–1.4, ±100 1.3–2.5, both at their ends 2.9–4.6;
     Blacks 0.3–0.8; Whites below zero 0.4; Contrast ±50 0.7–1.2, ±100
     1.6–2.4; Whites +50 1.2–2.3, +100 5–16; six mixed grades 0.8–2 for
     moderate ones, 3.4–5.0 at the extremes. What limits it: the networks
     (0.03–0.15 stops on the curves), the base (Camera Raw's is not known;
     small bright things get too much of Highlights), and which reading of
     the frame's top Whites above zero goes by.
   - **Tests**: `cargo test -p vr180-pipeline --lib raw_ -- --include-ignored`
     (the GPU stages against their CPU twins, the tables and families
     against Camera Raw's numbers, the keys read off the reduced copy,
     the sharpening's kernel and limits).
   - **Stereo Auto align on a raw take** (2026-11-16, the user: "为什么raw
     视频 auto align没有了"; it never was there: `align_ok` in
     `draw_view_adjust_panel` did not list `X6Raw`, because
     `stereo_align::measure_stereo_align` projects 8-bit eyes and a raw
     take hands out 16-bit camera-linear ones). Now: the take is opened
     at half size in memory (`X6RawIter::new(.., half = true, .., None)`),
     its lens comes from `resolve_export_imu`, and each eye goes through
     `stereo_align::linear_to_picture` (scaled to the camera's neutral,
     lifted so that 99 % of the greens sit under 0.85, gamma 1/2.2,
     RGBA8) before the usual projection and patch matching. Measured
     (harness `X6_ALIGN=1`): takes 161426 and 161606 of the same camera
     give pitch +0.053 / +0.058, roll −0.206 / −0.194, yaw +0.11 / +0.05,
     residual 0.028°; a known offset on the sliders (yaw +0.30, pitch
     +0.20, roll −0.25) changes the measurement by −0.201 pitch, +0.245
     roll (yaw −0.41: the yaw is the far cluster's, as for every source);
     the measurement's own correction applied brings pitch and roll to
     0.000 / −0.004. A take full of motion blur (205643: 16 ms shutters at
     night) gives a residual of 0.58° and values not to trust: the
     residual says so.
   - **Settings read back from Camera Raw** (2026-11-16, the user: yes to
     reading them back; `raw_export::read_develop_xmp` = the inverse of
     `develop_xmp`, `enhanced_dng::read_xmp`, `x6_raw::camera_raw_settings`,
     GUI button "Read from Camera Raw" in `draw_raw_develop_section`,
     `App::raw_settings_job` / `poll_raw_settings`). The enhanced picture
     in a DNG is pre-development, so nothing done in Camera Raw's panels
     reaches the export by itself; this takes the part this development
     has controls for: `WhiteBalance` ("As Shot" = as shot, else
     `Temperature` / `Tint`), `Exposure2012`, `Contrast2012`,
     `Highlights2012`, `Shadows2012`, `Whites2012`, `Blacks2012`,
     `Sharpness` / `SharpenRadius` / `SharpenDetail` /
     `SharpenEdgeMasking`, `ColorNoiseReduction`, each held to the
     slider's range. Camera Raw 18.7 writes all of them as ATTRIBUTES of
     the packet's first `rdf:Description` (124 there in the sample; later
     descriptions belong to the Enhance filter and are not read); plain
     elements are read too. A packet without `Exposure2012` (another
     process version, or no settings) gives nothing. What else is set away
     from Camera Raw's defaults comes back as names (`not_carried`:
     Texture, Clarity, Dehaze, Vibrance, Saturation, Luminance noise
     reduction, the colour noise reduction's Detail / Smoothness, Grain,
     Tone curve (named, points, parametric), Color mixer, Color grading,
     Calibration, Vignette, Lens corrections, Geometry, Crop, Black &
     white, Masks, Healing, Profile other than Embedded) and shows under
     the button. Which file: the frame on screen's, channel 0 then 1, in
     `<take>_OriginalDNG`, else the nearest of the 12 nearest frames that
     has settings; a sidecar `.xmp` of the same name counts when the DNG
     has no packet; `eyes_differ` when channel 1 of that frame was left
     differently. The read runs on a thread; the result replaces
     `settings.raw` only if that take is still the one loaded. Checked on
     the user's sample: as shot, Shadows +100, sharpening 40 / 1.0 / 25 /
     0, colour noise 25, nothing uncarried. (Camera Raw states that
     frame's as-shot white as 3750 K, +27; this app's model says 3730.)
     FOUND ON THE WAY: the section's Temperature and Tint sliders were
     handed the as-shot values unrounded, wrote the rounded ones back and
     reported a change, so merely showing the section turned "as shot"
     into a custom white; the values are rounded before the sliders now.
   - **Noise reduction** (2026-11-16; `fisheye_decode::LinearCoding`,
     `DenoisingFisheyeIter::with_linear_source`). The filter wrapper takes
     frames IN MEMORY; a raw take hands out GPU pairs with empty buffers,
     and the 16-bit copy in `VtDenoiser::build_source` read from the empty
     `Vec`'s dangling pointer: the app died with SIGSEGV at address 1 the
     moment an export of a raw take had Noise Reduction above zero (the
     user's report). Now: `VtDenoiser::push` refuses a short frame;
     `export_fisheye` opens a raw take WITHOUT the device when the filter
     is on (`source_gpu`), so its eyes arrive in memory (CPU demosaic, on
     every core); and since those eyes are camera-linear, not
     white-balanced, they pass the filter coded: each channel scaled to
     the camera's neutral (a grey then has nothing in the half-resolution
     colour planes), then `log(1 + x / TOE) / log(1 + 1 / TOE)`, TOE =
     1/1024 (4–9 ten-bit codes per 14-bit sensor code at black, 0.7 % of
     full scale per code at white), decoded after. Measured on take
     162217, strength 0.71: mean RGB within 0.10 / 255, every brightness
     band within 0.2; grain in flat areas ×0.47; strongest 1 % of edges
     ×0.97; a sky gradient as smooth as without it; 0.31 s a pair at
     8192×4096 (0.065 without), 4.5 GB. Also runs on enhanced frames
     (7680² an eye) and in a stabilized lens frame (the export that
     crashed). Windows has no such filter (the wrapper is macOS-gated;
     the new `denoising` flag is `false` there).
     THE FILTERED FRAMES GO BACK ON THE GPU (`fisheye_decode::
     CameraEyeUploader`, same day): `plan_with_raw_reference` draws the
     reference frame a non-half-equirect view is developed from only for
     a pair that HAS GPU eyes; the filter's pairs are buffers, so a Lens
     Space (or reframed) export with the filter on was developed without
     its reference and came out unlike the preview (the user: "颜色差很
     多"; with Shadows +100 / Highlights −100 / Exposure +1.95 the mean
     level of a frame was 73 against 95, mean difference 23 / 255). With
     the eyes uploaded again after the filter the same frame differs by
     0.9 / 255, the filter's own effect. Rule: whatever hands a raw
     take's frames to the export must hand them as GPU pairs.
   - **Not done**: takes shot before the camera wrote `exposure.csv`
     have no shutter time, so their pose is sampled half a shutter (plus
     the gyro's lag) early (Motion and timing); the gyro's lag is not
     known for any take; takes of the earliest camera build have neither
     readout nor accelerometer; the
     development panel, the DNG snapshot and the DNG sequence option have
     not been seen running on a display yet.
   - **WINDOWS**: shared code; the only gate touched is the pair moved into
     `export_device()` in `app.rs`. To check there: the raw shaders compile
     on D3D12 / Vulkan (the staged one binds six storage buffers and a 3D
     `Rgba16Float` texture; the measure pass writes `Rgba32Float`; the
     sharpening writes `R32Float` storage textures and reads them with
     `textureLoad`), and the `raw_` tests pass on that GPU.
   - **WINDOWS VERIFIED (2026-11-17, Windows box, RTX 4090).** The raw
     tests pass on Vulkan and on D3D12, the GPU ones included (demosaic,
     development, local tone, sharpening and the reference frame against
     their CPU twins; demosaic 0.96 ms an eye at full size, 0.73 at half).
     A take from the NAS opens in the app (1884 pairs listed in 0.5 s),
     previews developed and plays at its 30 fps cadence (render mostly 1 to
     7 ms),
     and exports: default and graded (Shadows / Highlights at their ends,
     sharpening 40), half-equirect, lens frame and reframed view. Vulkan
     twice: byte-identical. Vulkan against D3D12, same encoder: signed
     difference under 0.01 of 255, mean |Δ| 0.6 (default) to 0.9 (tone +
     sharpening) on a dusk take, spread evenly: the demosaic's direction
     weights and the sharpening on noise, through the encoder.
     **Camera original DNGs written on Windows are byte-identical to the
     Mac's** (three takes, five of six files; the sixth is the one Camera
     Raw enhanced in place). Camera Raw's settings read back the same
     (Shadows +100, sharpening 40 / 1.0 / 25 / 0, colour noise 25).
     TWO THINGS FIXED THERE:
     (a) **Enhanced frames made a Windows export fail.** The search is
     platform-neutral and found the enhanced picture; the first frame then
     asked for the JPEG XL decoder, which only the Mac build has, and the
     export stopped with that error (with "Use enhanced frames" on, the
     default, and an enhanced frame in the stretch). New
     `enhanced_dng::decodable()` (`cfg!(macos)`); where it is false
     `x6_raw::enhanced_side` answers none and `with_enhanced_frames` leaves
     the iterator on the capture (one log line), so the export runs from
     the capture at the plain size; the Format window shows one line in
     place of the option (`EnhancedFrames::Unreadable`,
     `ENHANCED_UNREADABLE_NOTE`, no lookup thread) and never lists the two
     larger sizes. Reading Camera Raw's settings is untouched. Same take,
     same stretch: failed before, 15 frames written after.
     (b) The Windows path ladder called a raw export "CPU export path — no
     GPU fast path for this source type" in the export bar. It is the
     take's only path on every platform, with the GPU doing the demosaic,
     the projection and the grade: no note for a raw take now.
     SPEED, 8192 × 4096 from the NAS, graded, as found: NVENC 3.5 fps,
     ProRes HQ 3.9, libx265 1.5 (the Mac: 15).
     **THE PORTABLE 10-BIT LOOP FEEDS PLANES ON WINDOWS (same day).** Where
     a frame's time went (the loop now logs it every 30 frames, "export
     perf: decode / gpu / readback / encode"): readback 145 to 173 ms (the
     frame left the GPU as RGBA64, 268 MB, with a per-pixel alpha strip to
     RGB48), encode 39 ms (swscale RGB48 → P010, then NVENC; ProRes 63),
     decode 13 ms from the page cache and 33 from the NAS, GPU submits 12.
     Three changes, all in `export_fisheye_inner`'s 10-bit readback arm and
     what it calls, Windows only (`cfg!(target_os = "windows")`; the Mac's
     arms and its software fallbacks are as they were):
     (1) **Planes, not RGB48** (`PlaneFeed`): an encoder that takes P010
     (hevc_nvenc) or planar 4:2:2 (ProRes, `prores_ks_vulkan` and the CPU
     one) gets `compose_sbs_to_p010_textures` / `_p210_textures` (video
     range) read back with `read_texture_planar` and fed through
     `encode_frame_p010` / `encode_frame_yuv422p10_msb` — the Windows
     zero-copy readback arm's own compose and encoder calls. libx265, 8-bit
     and a single-eye frame keep RGB48 + swscale.
     (2) **The encoder on its own thread** (`encode::EncodeStage::on_thread`,
     fed `PlaneFrame`s with their fill count through a queue of two; it
     derefs to the `ExportEncoder` when in line, so every other call in the
     loop reads as before; `finish` joins it; a cancel closes the file as
     in line). `VR180_ENCODE_INLINE=1` keeps it in line: same file.
     (3) **A raw take is read ahead** (`X6RawIter::with_read_ahead`, on for
     the export's opener on Windows; `VR180_RAW_READ_AHEAD=0` reads in
     line): a thread with its own handle on the capture reads and unpacks
     pairs two ahead (buffers recycled), `next_pair` only uploads and
     demosaics; a seek starts it over. No GPU on that thread.
     Also `read_texture_planar` copies a plane of 8 MB or more on up to
     four threads (a 34 MB chroma plane: 7 → 3.7 ms; the zero-copy readback
     arm uses it too; same bytes).
     MEASURED, 8192 × 4096, same take: graded (tone + sharpening) NVENC
     3.5 → 10.1 fps, ProRes HQ 3.5 → 10.1; default development NVENC 4.2 →
     14.8. Per frame now, graded: decode 7 ms (upload + demosaic submit),
     gpu 13, readback 60 (about 50 of it the wait for the frame's GPU work,
     the raw grade itself), encode 0. Blackmagic RAW rides the same loop
     (quarter-scale decode, 4096 × 2048): NVENC 10.3 fps and ProRes 15.8
     where libx265 does 4.3; so does any source that falls off its
     zero-copy arm (an Osmo clip forced there, 3072 × 1536: 6.9 fps with
     NVENC, the software decode being its cost) and the D3D12 fallback.
     CHECKED: the plane feed against the GPU-resident NVENC arm on an Osmo
     clip and on the dropped-frame X6 clip, signed difference under 0.004
     of 255 (the same compose); against the zero-copy ProRes arm under
     0.002; a raw take against the RGB48 route it replaces +0.03 / 0.00 /
     +0.05 (ProRes under 0.01); against the libx265 route −0.1 / −0.03 /
     −0.1 on every source, the known distance between swscale and the GPU
     compose. The dropped-frame clip: 100 frames at 50 fps, one pts delta,
     audio in step. Raw lens frame and reframed view through the planes
     (the reference frame is drawn as before). Encoder on its thread
     against in line, and read-ahead against in-line reading: byte-identical
     files. A cancel after 40 frames leaves a 40-frame file that decodes.
     The regression matrix (every arm that does not ride this feed) is
     byte-identical. NOT measured: a raw export from the GUI itself (the
     harness runs the same `export_fisheye` on the app's dedicated device).
     **MAC SIDE**: nothing changes there; the read-ahead and the encoder
     thread are plain Rust and could be switched on for the Mac's loops
     after a check on that machine.
     **MAC FIGURES (2026-11-17):** the decode step of a raw frame (read,
     unpack, upload, the demosaic's submit) takes 11 ms here and a graded
     8192 × 4096 export through the P010 surface arm runs at 12.4 fps (81
     ms a frame), so reading ahead could bring 15 % at most, and less with
     the noise filter on, which sets the pace there: not switched on. The
     stage-time line is written only by the arm that times its stages
     (`99516c2`): through the encoder's surfaces it logged three zeros
     beside the decode time.

35. **The `refactor` branch on Windows (2026-11-17, Windows box; written
   while it was a branch only: it LANDED ON `main` the same day as
   `275cb93`, last bullet).** `origin/refactor` @ `d95dd62` (one
   projection kernel source, `project_eye`, `FramePose`, one colour-stack
   recorder, `run_fisheye_frames` / `run_eac_frames`, `lens_model`, the
   preview probe) had `main` up to `ec7de04` in it. Here `main` @ `8948361`
   (the Windows work of items 30 to 34: seven commits) was merged in.
   - **As pushed** it builds on Windows first try. Pipeline tests: 228 pass,
     3 fail, all three known from `main` before its fixes: the two cube-map
     lens-frame tests (the CRLF splice; the fisheye family no longer has
     that problem, its builds are composed line by line and come out with
     bare line feeds) and the 8-bit LUT bound (the GPU's rounding).
   - **The merge** conflicted in `fisheye_export.rs` only, where `main`'s
     portable loop had gained the plane feed, the encoder thread and the
     per-stage timing while the branch had replaced that loop with
     `run_fisheye_frames`. Ported into it: `PlaneFeed` is decided once the
     sink is open and only for `FrameSink::Readback`; the 10-bit readback
     arm composes and reads back planes or RGB48; the loop pulls its frame
     inside the `decode` timer; `EncodeStage` wraps the encoder (the macOS
     surface functions take `&mut ExportEncoder` and get it by deref).
     Everything else merged by itself (`lens_frame_request`, the raw
     read-ahead in the opener, `unix_lines`, the test bounds).
   - **Exports: 61 files byte-identical, stream for stream, between the
     `main` build and the merged branch's build** (two worktrees, the same
     harness, `ab_run.ps1` / `ab_cmp.py` in the session scratchpad): Osmo,
     X6 (dropped frames, Crop + mask), GoPro `.360`, QooCam, a raw take,
     Blackmagic RAW; the portable loop (libx265, and NVENC / ProRes through
     the plane feed), the D3D11 zero-copy readback arm, the GPU-resident
     NVENC arms, the `.360` readback tail; half-equirect, disc, lens frame
     (also the one held to 8192 × 4096), reframed; 8 and 10 bit; LUT +
     ETTR + eye match; trim; stabilization on and off.
   - **Previews: 36 probe cases print the same lines on both** (the
     branch's `VR180_PREVIEW_PROBE`, put into a throwaway worktree of
     `main` with its two hooks): X6, Osmo, QooCam, Crop and `.360` on the
     Windows zero-copy workers, the same clips on the CPU worker
     (`VR180_NO_HW_DECODE=1`), a raw take, Blackmagic RAW; half-equirect,
     Lens Space with either lens, reframed, the view modes, LUT, ETTR,
     auto eye match, crop mask, unstabilized, and the zoom still.
     ONE FIX FOR THAT, on the branch: **the probe ran the CPU preview on
     Windows.** Its device came from `Device::new()`, which does not ask for
     the P010 texture format the Windows zero-copy gate requires (the app's
     own device does, `main.rs`), so the worker edited by text was never
     reached. On Windows the probe now takes
     `Device::new_dedicated_from_adapter` on the same adapter; the Mac's
     probe device is as it was. The probe has no zoom still for a `.360`
     (both branches: "non-fisheye source").
   - Tests on the merged branch: core 63, fisheye 70, braw 6, pipeline 243
     (+ `projection_fingerprints`, which wants `VR180_FINGERPRINTS=<file>`
     and is a tool, not a check), gui 42. On D3D12 (`VR180_BACKEND=dx12`)
     pipeline 232 of 233, the one being the lens frame's outermost pixels,
     as on `main`. `windows_adapter_smoke` passes on the RTX 4090 and on
     the Intel UHD 770. The app starts, opens an X6 clip on the zero-copy
     preview and plays at its cadence.
   - A new worktree has no `helpers/bin` (ignored, built): copy
     `braw_helper.exe` from the main checkout or run `build_braw.ps1`, or
     every `.braw` probe fails.
   - As the branch's commits say, the three Windows-only arms still call
     the four `project_fisheye_rgba16_texture_*` wrappers and keep their
     own per-frame pose: moving them onto `project_eye` / `FramePose` is
     still to do, and this A/B set is the check for it.
   - **MAC SIDE (2026-11-17, Mac):** the Mac had made the same merge of
     `main` @ `8948361` within the hour and not pushed it. The two ports
     of the loop came out the same but for the spelling of the sink test
     (`!=` there, `matches!` here) and one comment, so the Mac dropped its
     own and took this head, then merged the three Mac commits of `main`
     (`99516c2`, `879331c`, `c171c06`) as `8093d7b`: the gate on the
     stage-time line lands in `run_fisheye_frames` by itself. On that
     head: it builds (the macOS-gated arms inside the shared loops were
     never compiled on Windows; this is their compile), tests braw 6,
     core 63, fisheye 69, gui 40, pipeline 235, and the Mac's stored sets
     are the same as `main`'s: 65 exports (every encoder arm of both
     loops, raw takes, lens frames, DNG frames) and 169 preview lines
     (the workers, the zoom stills). With the macOS branches switched off
     the workspace type-checks as `main` does, and the text of the
     Windows-gated items is `main`'s but for the `stab_cap_and_window`
     substitution in `run_fisheye_zerocopy`. With the A/B above the branch
     gives `main`'s pictures on both machines; what is left before it
     lands is the decision.
   - **LANDED ON `main` (2026-11-17, Mac): `275cb93`**, a merge commit, so
     that the first-parent history of `main` reads it as one step. `main`
     @ `c171c06` was already in the branch, so the merge's tree is the
     branch head's (`c31d03b`); since the head the Windows box compiled
     (`a7c1ce1`) the code differs by the stage-time gate of `99516c2` and
     the Chinese string of `879331c`. The branch's build was tried by eye
     on the Mac first (70 minutes, three clips on the zero-copy preview
     worker, no warning in its log). Built again in the main checkout and
     run there: tests braw 6, core 63, fisheye 69, gui 40, pipeline 235;
     the 65 exports and the 169 preview lines are the stored ones.
     - From here the work is on `main`: the `refactor` branch has nothing
       `main` lacks. First on the list is the Windows item above (the
       three Windows-only arms onto `project_eye` / `FramePose`).
     - Names the older items use for code that is gone:
       `record_color_stack_16` is `record_color_stack`, which records
       both depths (so do its stages, `record_cdl`, `record_lut3d` and the
       rest); the nine `fisheye_*_to_*` projection shader files are builds
       of `shaders/fisheye_project.wgsl` (`shader_compose.rs`,
       `gpu/projection.rs`); the `project_fisheye_*` entry points are
       `project_eye` / `project_eyes` (the four
       `project_fisheye_rgba16_texture_*` wrappers the Windows arms kept
       are gone too, last bullet); `timeline_frames_or` is
       `frame_timeline::frames_or`.
     - The preview probe is part of the app (`VR180_PREVIEW_PROBE=<clip>`,
       `decoder/preview_probe.rs`, taken in `main.rs` before the log
       opens): comparing previews no longer means putting it into a tree.
   - **THE THREE WINDOWS ARMS MOVED (2026-11-17, Windows box, on `main`).**
     First the pull: `main` @ `1d27309` builds, tests core 63, fisheye 69,
     braw 6, pipeline 233, gui 39 (243 and 42 with the ignored GPU ones),
     and the 61 exports of the A/B set are the stored ones, stream for
     stream. Then the item left open above:
     - The D3D11 zero-copy preview (`run_fisheye_zerocopy`) and the two
       D3D11 exports (`export_fisheye_osv_zerocopy_d3d11`,
       `export_fisheye_osv_gpu_resident`) pose a frame with
       `render::FramePose::new` and project it with `project_eyes`, as the
       Mac's zero-copy arms do. The two exports build the `EyeJob` the
       shared loop builds (calibrations from `resolve_source_calib_pair` +
       `view_calib_pair`) and share one helper,
       `fisheye_export::project_texture_eyes_16` (an imported 16-bit eye
       pair, outputs kept in slots 30 / 31; the GPU-resident arm's fisheye
       output used 32 / 33, which made no difference to a picture).
     - The rule for the per-row correction is one function,
       `fisheye_export::export_row_readout` (a stabilized source with a
       per-frame IMU, the readout from `readout_ms_for`), read by
       `run_fisheye_frames` and by both arms. The arms asked `cfg.stabilize`
       alone; the two say the same, since only the kinds with a per-frame
       IMU get one from `resolve_export_imu`.
     - `raw_gpu::raw_reference` asks `project_eye` as well, so the four
       `project_fisheye_rgba16_texture_*` wrappers are deleted. No kernel,
       no shader and no macOS-gated line was touched (one comment in
       `p010_resolve_rgba16.wgsl`).
     - CHECKED against the stored set and against the build before the
       change: the 61 exports byte-identical per stream, with the same 38
       engagements of a fast arm; the 36 preview-probe cases print the same
       lines (the zero-copy workers engaged in both); tests pipeline 243,
       gui 42, on D3D12 232 of 233 (the lens frame's outermost pixels, as
       before); the app opens an X6 clip and an Osmo clip on the zero-copy
       preview and plays them at its cadence.
     - What the Windows arms still carry of their own: the three loops
       themselves (decode thread, import, the encode tails), their
       stabilization table and trim. `run_fisheye_frames` now has the plane
       feed and the encoder thread, so the D3D11 readback arm could become
       a `FrameFeed` of it; the GPU-resident arm's ring of shared frames
       has no counterpart there. Not started.
   - **MAC SIDE PULLED IT (2026-11-17, Mac):** fast-forward to `9fd483d`:
     the arms above (`dff754a`), then the held size of a lens frame going
     by the frame as written (`9fd483d`; a Mac's encoder has no frame
     limit, `hw_h265_side_limit`, so no size changes here). Three pieces
     of the two commits are code a Mac compiles: `export_row_readout`
     inside `run_fisheye_frames`, the `project_eye` request of
     `raw_gpu::raw_reference`, the extra argument of the lens-frame size
     functions in `app.rs`. It builds; tests braw 6, core 63, fisheye 69,
     gui 40, pipeline 235; the stored sets are the same as before the
     pull, 65 exports and 169 preview lines. Among them are raw takes
     written as a lens frame, a disc and a reframed view, stabilized and
     not: those draw their reference through `raw_reference`, with
     per-row matrices and without.
     - Read against the shared definitions, the Windows arms ask what they
       asked before: `FramePose::new` is the rotation and row code they
       carried; `resolve_calib_pair` is `view_calib_pair` of
       `resolve_source_calib_pair`; `rows_for_both` is their test for both
       eyes' rows; `resolve_export_imu` answers only for the five kinds
       `has_frame_imu` names, so `export_row_readout` is on where
       `cfg.stabilize` alone was. The output cache is kept per slot and
       re-made when the size or format differs, so the GPU-resident arm's
       fisheye output sharing slots 30 / 31 costs nothing.
     - No line inside a macOS-gated item changed (two doc comments above
       such items did).
     - The `refactor` branch is deleted, on the Mac and on the remote
       (2026-11-17), with the Mac's second checkout. Every commit of it is
       in `main`: its head `c31d03b` is the second parent of `275cb93`. A
       checkout that still has the branch can drop it after
       `git fetch --prune`.

36. **Several clips as one file, projects, and the list kept between
    launches (2026-11-17, Mac; UNRELEASED).** The user-facing text is in the
    CHANGELOG. None of the code is gated by platform, and none of it has run
    on Windows.
    - **`vr180_pipeline::join`.** `join_exports(parts, out, progress,
      cancel)` puts exported files of one shape together: the video packets
      are copied into one stream (each part's time stamps moved on by the
      length of what came before, on the output's clock; the first part's
      stream parameters, so the `hvc1` tag stays), nothing is encoded again
      and every cut is a part's first keyframe. The sound of each part is
      decoded to stereo at 48 kHz, cut or filled to that part's OWN picture
      to the sample (`PartSound`: 5 ms fades at both ends, silence for a
      part without a track), written as one float WAV beside the output,
      encoded once (`stereo_to_aac`) and muxed. A part's track runs up to a
      codec frame past its picture: appended whole it puts every later part
      late (128 samples at the first cut of the test files), so each cut is
      placed from the running total of the pictures' lengths. `JoinSound`
      says what the one track is: `Stereo` (parts with any sound, or
      none), or the parts' four-channel first-order ambisonics kept channel
      for channel (decoded, cut and faded as the stereo is) and written as
      `Ambisonic` (one four-channel AAC, `audio::four_channel_to_aac`, with
      the SA3D tag) or `Apac` (macOS: `audio::four_channel_to_wav`, a 24-bit
      WAV that states no layout, through the `apac_encode` helper, which
      also writes the picture; where that fails, and off macOS, the
      ambisonic track). A part of
      another codec, frame size, frame rate (0.01 fps apart is one rate) or
      with other parameter sets, or one without a four-channel track where
      the sound is a sound field, is named in the error (`join_obstacle`).
      `planned_part` and `mismatch` tell the same BEFORE a run, from the
      export configurations: frame, frame rate, encoder with
      its depth or profile, how the file is described to a player.
      `Unjoinable`: a spatial video, a lens frame.
    - **`fisheye_export::state_export_metadata(cfg, out)`** states in a
      joined file what an export of `cfg` states in its own (the packet
      copy carries none of those boxes): the decisions of `export_fisheye`
      or `export_eac`, in their order.
    - **The run** (`app.rs`). `start_run` → `plan_join` (the checks, the
      file's name `<first clip>_Joined_<tag>.<ext>`, the parts as
      `<that name>.parts/NN_<the clip's own output name>`, the sound:
      `joined_sound` = the FIRST clip's Audio choice, with Ambisonic / APAC
      only where every clip has four-channel sound, a fact read with the
      clip's header, `four_channel_sound`; a clip without it is named and
      the run does not start) → the queue as ever, each clip written to its
      part with `AudioTrack::Stereo` or, for a sound field,
      `AudioTrack::Ambisonic` →
      `start_join` when the queue drains → `end_join` → `finish_run`. The
      join runs as the run's last `ExportJob`, so `logic()` hands over to it
      with the window covered like to any clip. One file or none: a clip
      that fails takes the waiting ones out of the queue, Stop and the
      toolbar's Cancel stop the run, Skip is not offered; the parts go once
      the joined file is whole and stay where they are, and that is said,
      when there is none. "Export all" with the box ticked is the whole
      list, whatever was exported before.
    - **The list.** `reorder.rs`: a row's handle is a drag source and the
      row the place to let go (`Move::place_of` renumbers `active_clip`,
      `batch_current` and `batch_run_indices`). Its glyph is one the
      built-in fonts hold; a test checks that, because the first choice was
      not.
    - **Projects** (`project.rs`). `Project<E>`: format `version` 1 (a later
      one is refused, in words), `FileRef` = where the file was and the same
      from the project file's folder with `/` between the parts. A file is
      written beside its place, read back, and only then moved into it: one
      this build could not open again (a number JSON has no form for) never
      replaces one it can. `ExportOptions` and its enums are serde, with
      defaults for what a file does not hold. The session is the same thing
      in `<app folder>/session.json`, compared with the list every 1.5 s and
      at close. Opening one puts an entry per clip in the list at once
      (`BatchItem::offline`: `source_kind` Unknown, never exported, kept in
      the project as read) and looks for the files on a thread
      (`ClipFacts::read`, split out of `add_batch_files`; `poll_restore`), so
      a network drive that has to wake does not hold the window.
      `GuardStage` (`session.loading`): a launch that died putting the list
      back starts empty with the session put aside, one that died loading a
      clip gets its list without a clip on screen.
    - **Checked on the Mac.** Tests: braw 6, core 63, fisheye 70, gui 56,
      pipeline 242 (21 of them new). The stored sets are as they were: 65
      exports, 169 preview lines. A joined file of five parts from three X6
      clips, written through the zero-copy loop, the portable loop and the
      noise-filter loop: the parts' parameter sets are byte for byte the
      same, its 165 frames are the parts' frames by checksum, every part's
      sound sits at a shift of 0 samples against the part's own (cross
      correlation), and its projection boxes are those of a single export.
      The same for `.360` clips as VR180 and as a disc with its lens (`cams`
      / `lnsc`) and for ProRes in `.mov`. Three X6 clips recorded in 360°
      Audio, joined as Ambisonic: every channel of every part sits in its
      own channel at a shift of 0 samples, the SA3D box is there. Joined as
      APAC: the system reads the track as four channels of ambisonics
      (ACN / SN3D) of the picture's length, and decoded by the system each
      of its channels is the ambisonic file's same channel (correlation
      0.999 to 1.000 at lag 0). In the app itself, started with a scratch home folder: a
      session of two X6 clips set to join came back at launch (trims,
      ticks, output folder, Format choices; a clip named in it that does
      not exist stayed in the list as missing), and a run wrote the two
      parts, joined them into `…_Joined_SBS.mp4` (8192 × 4096, 105 frames,
      stereo, the VR180 boxes) and took the parts' folder away. That run
      was started by a line of test code, since removed: the buttons, the
      dragging of a row and the Project menu have not been worked by hand
      yet.
    - **Windows, to look at.** (0) A joined run of X6 clips recorded in
      360° Audio with the first clip's Audio at Ambisonic: one four-channel
      AAC track with SA3D (an APAC choice carried over from a Mac is
      written as Ambisonic there, in `join.rs` itself). (1) A joined run
      through NVENC and one
      through libx265: the parts' parameter sets have to be byte for byte
      the same across clips (here the zero-copy loop, the portable loop and
      the noise-filter loop write the same ones for VideoToolbox). (2)
      `prores_ks` parts in `.mov`. (3) A project saved, its folder moved
      with the clips, opened again. (4) The list across a restart, and a
      restart with the clips' drive away.

37. **Tone controls in linear light for every clip that is not raw
    (2026-11-17, Mac; UNRELEASED).** The user-facing text is in the
    CHANGELOG. Exposure, Contrast, Highlights, Shadows, Whites and Blacks,
    the six of a raw clip's development, as ONE stage of the colour stack
    that works in linear light ahead of the CDL and the LUT. Lift / Gamma /
    Gain stay where they were (the CDL, on the codes ahead of the LUT).
    - **`vr180_pipeline::tone_linear`** (the arithmetic, no GPU).
      `ToneSettings` (the six numbers, serde), `LinearTone { settings,
      light, pre_stops, measure }` (the plan's stage), `ToneConstants` (one
      frame's tables; `apply` is the CPU twin of the shader). The stage:
      `codes → light → × exposure → Shadows / Highlights → Whites → Blacks →
      Contrast → codes`.
      - *Which light* (`ToneLight`). `Scene(LogCurve)`: the codes are
        decoded with the camera's log curve and divided by the light of
        code 1, so 1 is the brightest a code can say. Used where the curve
        is known: Insta360 I-Log. `Shown`: the light the picture SHOWS. The
        code goes through the neutral curve of what renders it after the
        stage (`Shown` = the CDL, then the LUT at its intensity, on greys:
        `ColorStackPlan::shown`, a CPU port of `cdl.wgsl` and of
        `lut3d.wgsl`'s eight-corner mix, 4097 nodes so a 17 / 33 / 65 point
        lattice's kinks fall on nodes) and is decoded as sRGB; the result
        goes back through that curve's inverse (`Shown::code_of`: a level
        stretch at either end is entered from inside). Every other clip
        takes this one. It needs no vendor formula: D-Log M has no published
        one and GP-Log's base is not confirmed, and a wrong curve would be
        worse than none. Should one be confirmed, `Settings::build_color_stack`
        is the one place that picks the light.
      - *Shadows / Highlights* are the raw development's own: the bilateral
        grid of the frame's log₂ luminance (`raw_local`), the curves from
        the frame's levels (`raw_local::local_curves`, which follow the
        frame down the scale, so the light's unit does not matter), the
        white limiter (`limit_gain`, white = light 1).
      - *Whites / Blacks / Contrast* are Camera Raw's maps of the DISPLAYED
        value (`raw_global`). A map `m` becomes a change of light through
        the rendering: a light `x` shows as `e`, so it goes to the light
        that shows as `m(e)` (`Light::through`; where the rendering is level
        at `x` the pixel keeps its ratio to that light, next to black its
        distance). The frame's levels are handed to the maps as the level
        Camera Raw's default rendering would have been fed to show the same
        (`equivalent_level`). Stored as the CHANGE of a light (`LightMap`),
        so a control at zero passes a light exactly; 1025 samples, 57 to the
        stop, made again for every frame.
      - *The tables that do not depend on the frame* (`LightTables`: light
        over the code, and the code over the sRGB encoding of the light,
        16385 samples: laid out that way the inverse of a LUT's shoulder is
        sampled evenly where it is steep) are kept from frame to frame
        (`Device::linear_tone_tables`).
      - *Under black.* A log curve's straight toe runs on under its black,
        where a clip's noise lies: such a code is light BELOW none. It is
        not cut off: it keeps its sign through the exposure and the gain, a
        stage's change runs on under zero along the line it starts with
        (`LightMap::at`), and the light goes back to a code along the toe
        (`LightTables::below`, read off the decode table's first two
        codes). Exposure alone is then `LogCurve::expose` for every code,
        as the ETTR stage is, and noise about black keeps its mean.
    - **`vr180_pipeline::tone_gpu`** (`shaders/tone_measure.wgsl`,
      `tone_reduce.wgsl`, `tone_linear.wgsl`). `Device::linear_tone_constants(eyes,
      plan)` measures the frame (the eyes as they enter the stack, two side
      by side as one frame: 126 cells along the long side, read back) and
      returns its constants; `record_linear_tone` records the stage (the
      reduce pass and the raw stage's own `grid_build` / `grid_blur` when
      Shadows or Highlights are moved, one empty grid cell and gains of zero
      otherwise). The raw module's helpers it shares are `pub(crate)` now;
      its pipelines are built when a frame first needs the stage, not when
      a clip is opened.
      `Measure::Wait` (exports, stills) waits for the frame's own levels;
      `Measure::Latest` (the live preview) takes the last ones that came
      back, queues a measurement and has the frame drawn again when it
      lands differently (`set_linear_tone_notifier`; the decoder thread sets
      it for every clip that is not raw). A frame and its enlargement for
      the zoom view are one picture at two sizes: across sizes only a change
      six times the usual tolerance counts (`ACROSS_SIZES`), or the two
      would have each other drawn again without end.
      - **The notifier fires only while the frame stands** (the worker is
        paused, or the clip is a still), for a raw clip too. Before, a raw
        clip that played bumped `settings_generation` for every frame: the
        worker resolved the clip's view again and logged "live calib
        update" 30 times a second (1277 such lines in one evening's log). A
        playing video draws its next frame with the new levels anyway.
    - **The stack** (`gpu.rs`). `ColorStackPlan.tone`; the stage is `0c`,
      after the eye offset and the log exposure and ahead of the CDL, at 8
      and at 16 bits (`record_color_stack` takes the frame's constants as a
      parameter; every entry point reads them off ALL the eyes it is given,
      so both eyes are toned alike: a test). A plan whose stage moves
      nothing (`active_tone()` is `None`) records nothing and
      `any_active()` does not count it. The app never puts the ETTR
      recovery and the stage in one plan: with a control moved the recovery
      is the stage's `pre_stops`, one decode and one encode for both.
      - Two arms graded the eyes with two `apply_color_stack_per_eye_16`
        calls: the `.360` export's 10-bit surface arm (macOS) and the
        GPU-resident `.360` arm (**Windows only: edited blind, same call as
        the Mac one, `apply_color_stack_pair_16(.., &color_plan, None)`**).
        Both are one pair call now. For a plan without the stage the result
        is the same textures as before.
      - The 8-bit path that grades on the host
        (`apply_color_stack_rgb8_sbs`) runs the stages ahead of the CDL on
        the whole frame first: the log exposure table (moved there from
        `apply_color_stack_rgb8`, byte for byte the same) and then
        `Device::apply_linear_tone_rgb8`, which uploads the eyes, tones them
        with one set of constants and reads them back (a test holds it to
        the GPU stack's bytes).
    - **GUI.** `Settings.tone` (serde, default = nothing moved; kept per
      clip like the rest of the grade). `build_color_stack`: the stage only
      when a control is moved and the clip is not raw; `Scene(I-Log)` for
      `ilog_source`, `Shown` otherwise; a linear DNG has no place for it
      (`restrict_to_dng_metadata`). Color panel: section **Tone** (not for a
      raw clip, Blackmagic RAW or DNG output), then **Lift / Gamma / Gain**.
      The CDL's own Shadow / Highlight sliders are offered only while a
      grade still holds a value in one ("earlier"), and until that section
      is reset. Blackmagic RAW keeps the six neutral at load, as it keeps
      the CDL.
    - **What it costs** (8192 × 4096, 10-bit HEVC, M-series, 100 frames):
      39 to 42 ms a frame without the stage; the same with Exposure alone
      (nothing is measured); 48 to 51 ms with a measured control. Of that
      about 12 ms is the wait for the frame's levels (it also ends the
      overlap of the GPU's work with the next frame's decode) and 0.5 to
      1 ms the tables.
    - **Known limits.** A reframed view or a lens frame is measured as it
      is drawn, not from the half-equirect (the raw development has a
      reference frame for that; this stage does not). The frame is measured
      as it enters the stack, ahead of the automatic eye offset. The
      "Matching Eyes" exposure trim stays a gain of the CDL, after the
      stage. Black outside the picture (a mask, a lens frame's border) is
      lifted by Blacks above zero, as Lift lifts it. Clips in other log
      curves work in the light their picture shows until a curve is
      confirmed. The controls are the raw development's operators, measured
      on Camera Raw with raw files: what Camera Raw does to a rendered file
      was never measured, and a log clip is rendered by its camera's LUT,
      so the numbers are not claimed to be Camera Raw's for video.
    - **Checked on the Mac.** Tests: braw 6, core 63, fisheye 70, gui 57,
      pipeline 259 (the stage's own: nine of the arithmetic, eight on the
      GPU, one of the GUI's plan). With nothing moved the stored sets are as
      they were: 65 exports, 169 preview lines. The stage's exposure on
      I-Log codes is the ETTR stage's to an eighth of a 10-bit step, every
      16-bit code (a test). On an I-Log clip shot two stops over, through
      the X6's LUT: Highlights −100 takes the brightest hundredth of the
      frame from 0.67 to 0.41 of the range and the picture keeps its edges,
      Shadows +100 the median from 0.10 to 0.19 with no halo at the window
      frame (looked at as stills). The live preview's frame, the zoom still
      and the export of one moment, stabilization off, with five settings:
      preview and still within 0.002 to 0.003 of the range (8 × 8 block
      means), and what a setting CHANGES is the same in the preview and in
      the export to 0.0014 to 0.0019 on changes of 0.03 to 0.05. Every
      export arm with the six in use, 19 runs (the X6 clip through the
      hardware encoder at 10 and 8 bits, the software one at both, ProRes,
      a reframed view, no LUT; an Osmo clip; a `.360` through its five
      arms): none failed, the 10-bit arms agree to 0.001 to 0.002, the 8-bit
      ones differ from them by what they differ by with no control moved,
      a second run gave the same bytes. Not seen in the app's window: the
      sliders, the redraw when a measurement lands, the zoom view.
    - **Windows, to look at.** (1) The GPU-resident `.360` arm compiles and
      a `.360` export with and without a moved control comes out right
      (both eyes alike). (2) The stage through the D3D11 / NVENC arms, 8 and
      10 bit. (3) The live preview's redraw when a measurement lands (the
      notifier), with the zoom view open.

38. **Lens Space clips joined as an Apple Immersive Video file (2026-11-18,
    Mac; UNRELEASED).** The user-facing text is in the CHANGELOG (the clip
    list's section). Item 36's join refused lens frames: each states its
    clip's own lens (item 33), and a file described that way has one lens.
    An Apple Immersive Video Universal file (`.aivu`) holds a VENUE, a list
    of cameras each with the mesh its picture is drawn on, and a metadata
    track that says for every frame which camera it is shown through.
    - **What an `.aivu` is, as written here.** A QuickTime movie: the video
      with the projection kind `AppleImmersiveVideo` in its sample entry;
      the venue as asset metadata (`quickTimeMetadataAIMEData`: an opaque
      blob `ImmersiveMediaSupport.VenueDescriptor` makes); one sample per
      frame on a metadata track (`quickTimeMetadataPresentationImmersiveMedia`):
      a JSON array of the commands in force, here one `setCamera`; the
      sound. **The video need not be MV-HEVC**: Apple's guide describes the
      format with MV-HEVC, but a side-by-side picture (view packing
      `SideBySide`, both eye flags) passes Apple's validator and renderer,
      HEVC and ProRes alike, and a file written this way plays on Apple
      Vision Pro (the checks below). So the parts go in packet for packet.
    - **The camera of a clip is a MESH the helper builds.** A venue's
      camera is drawn as a mesh for each eye: a piece of the unit sphere
      (x right, y up, forward −z), every vertex with the place in the eye's
      frame it shows (texture coordinates from the frame's bottom left).
      The helper builds it from the lens the part's file states
      (`pictureMesh`): 128 rings by 256 spokes (64 to each side of the
      frame, so the corners are spokes), ending along every spoke at the
      frame's rim (8 px) or the lens's limit (less 1.5 px), whichever comes
      first; triangles counter-clockwise seen from the centre, as the
      framework's own are. It goes in as a USDZ (`.usdzMesh`): Model I/O
      writes the `.usdc`, the helper zips it itself (stored, the data on a
      64-byte boundary). **The two meshes have to be named `_0_Left` and
      `_0_Right`**: with other names the framework answers `invalidData`.
      The mask is an all-white image (`no_mask`, 8640 × 4320, the eyes side
      by side), one for every camera. The mesh is flat between its
      vertices: against the lens, at 4096 px an eye, a full fisheye's
      picture is off by up to 0.26 px at its rim, 0.08 px rms.
      - **Why not a lens profile** (the first version, never committed).
        An Apple Immersive lens profile (the JSON of an `.ilpd` file; its
        model `radial2ProjectionOffsetTangential2` is item 33's Mei-Rives
        model, `distortions` = k1, k2, xi, p1, p2 with xi POSITIVE,
        `centerX` / `centerY` counting pixel centres from 0) makes the
        framework build the mesh, to 0.000 px of the app's lens, but out
        to 121.7° from the axis whatever limit is stated: past this
        model's fold and past the frame's edge, where a side-by-side
        picture holds the OTHER EYE. A mask then has to end the picture,
        and a mask of control points (a) is kept by its NAME, so cameras
        whose masks share a name all get one of them; (b) is drawn wrong
        where its outline goes over the zenith, as a full fisheye's does
        (a 112° circle came out as a band between ±68° of height); (c)
        crashes the framework, a bus error and no message, with more than
        64 control points or without every key of `maskData`
        (`isForVisionOS`, `leftMapToRight`, `maskColor`, `FOVWidth`,
        `FOVHeight`, `defaultCalibration`, `controlPointsOffsets`). With
        its own mesh the helper needs none of it.
      - **What such files hold** (three read with the framework: one from
        Apple's calibration service, one multi-camera production, one made
        with a mesh of its author's own): every camera is stored as a USDZ
        mesh, a profile's too (the venue keeps the mesh it built, named
        `_0_Left` / `_0_Right`); the mask is an all-white 8640 × 4320 image
        (`no_mask`) or 64 control points that stay under the zenith; the
        video's sample entry has `hero` = left beside the eye flags
        (written here too). All three are MV-HEVC.
    - **`helpers/swift/aivu_wrap.swift`** (macOS 26; `helpers/build_swift.sh`
      builds it LAST and as an OPTIONAL helper: where the SDK has no
      ImmersiveMediaSupport it is skipped with a note and the others still
      build; `--install`, which a release runs, requires it). `--joined
      <file> --output <file.aivu> --part <file> --frames <n> …`: reads each
      part's lens from the part's own format description (a lens with two
      focal lengths or tangential terms is refused: the mesh draws one that
      is the same all the way round, which is what the app writes), makes a
      camera `clip-N` for it (the mesh above; the venue's own reading of it
      is asked for before anything is written), copies the joined file's
      video and sound packets (the video's sample entry rebuilt: the
      Verbatim* entries and the lens collection out, the projection kind
      and the hero eye in), and writes a metadata sample per frame. A
      packet's part is known
      by COUNTING (a part's packets are together in decoding order): no
      time is matched against a cut. Metadata samples are released in
      display order once no earlier frame can still come. Apple's
      `AIVUValidator` reads the result; a refused file is removed.
      - **The writer's rule** (three deadlocks in the prototype): it takes
        no more of one track while it waits for another. Only the video is
        waited for; the sound is read two seconds ahead of it, and sound
        and metadata are handed over whenever their inputs take some; a
        track whose data has all gone in is closed at once.
    - **Rust.** `helpers::aivu_wrap_available()` (the helper is there and
      answers `--version`: it does not start before macOS 26, which is the
      test; asked once), `helpers::spawn_aivu_wrap` (macOS; progress lines,
      cancel kills it and removes the file). `join::Described::LensFrame`,
      `PlannedPart::immersive()`, `immersive_available()`;
      `planned_part_where(cfg, one_eye, immersive)` is `planned_part` with
      the system's answer handed in (what the tests use).
      `join::join_immersive(parts, output, sound, …)`: `join_parts` (the
      old `join_exports` body, which now also returns each part's packet
      count from `concat_video`) into `<output>.joined.<ext>`, then the
      helper, then that file goes; `JoinSound::Ambisonic` is taken as
      `Apac`. It exists on every platform and is an error off macOS, so
      the GUI has no platform gate.
    - **GUI.** `JoinRun.immersive` (from the first planned part);
      `joined_output_for(.., immersive)` → `<stem>_Joined_Lens.aivu`;
      `start_join` calls `join_immersive` and skips
      `state_export_metadata`; a part is now named as the clip's own export
      (`…_Lens.mp4`). `Unjoinable::LensFrame` is left for where no such
      file can be written.
    - **Checked on the Mac (macOS 27.2).** Tests: braw 6, core 63, fisheye
      70, gui 57, pipeline 260. Three clips of one X6 in three recording
      modes (8K full view, 10K Crop, 10K full view; 5 s each, exported as
      Lens Space at 8192 × 4096, 29.97 fps) through `join_immersive`:
      0.8 s; the validator passes; the system lists Apple Immersive Video
      with three cameras; every one of the 451 frames decodes to its
      part's frame (checksums) and the sound to the joined file's samples;
      the system's preview renderer (`ImmersivePreviewRenderer`, fed the
      packed frame) draws each clip through ITS lens, BOTH eyes: 0.00 px
      against the same view drawn from that lens, 10 to 127 px with another
      clip's. Views away from the axis (to the sides at 55° and 90°,
      straight up and down, behind, and diagonal ones, both eyes): the
      picture is the lens's inside its edge and NOT ONE pixel is lit
      outside it, over the zenith included (the first version, with the
      framework's mesh and a mask, showed the other eye beyond a Crop
      frame's edge). The same with ProRes parts, and with four-channel
      parts as Apple Positional Audio (`apac`, read back). Parts without a
      lens: the helper's reason comes back as the error and nothing is left
      behind. **On Apple Vision Pro (the user, 2026-11-18):** that file,
      side by side with stereo sound, plays; so does the one with Apple
      Positional Audio, and a companion with the same cameras and the
      picture as MV-HEVC. **Not checked: the run through the app's own
      buttons (the planning and the naming are tested, the function is run
      from a harness).** For whoever checks again, the
      renderer's `rotation` as `Rotation3D(eulerAngles:)` with x = pitch,
      y = yaw, order `.xyz`: a positive yaw turns the view to the RIGHT,
      and the yaw applies before the pitch.
    - **Not built.** A dip to black at a cut (the format has a `fade`
      command; a prototype's file with them passes the validator). A plain
      Lens Space join for clips that state the SAME lens (two clips of one
      recording mode do: such a file needs no venue).
    - **Windows, to look at.** Nothing is gated in the GUI:
      `immersive_available()` is false there, so a Lens Space clip in a
      joined run is still refused, with the new wording. (1) It builds
      (`join_immersive`'s other branch, `aivu_wrap_available`). (2) That
      refusal appears.

39. **A clip is rendered by ONE set of tone statistics (2026-11-18, Mac;
    UNRELEASED).** The user-facing text is in the CHANGELOG (the raw
    development's "One development for the whole clip", the tone
    controls' "The controls are the raw development's").
    - **What was wrong.** A raw clip's development (item 34) and the tone
      controls of every other clip (item 37) are derived from statistics
      of the picture (`raw_tone::FrameStats`: brightest cell, darkest
      cell, the levels of a reduced copy, the peaks). Every frame was
      measured by itself (`Measure::Wait` in an export, `Measure::Latest`
      in the preview), so the rendering followed each frame's content. On
      a take shot at one manual exposure, with Highlights at −100, a pan
      that took a tree out of the frame made the sky 39 of 255 brighter
      within seven seconds while its recorded values fell 2 %: the
      Highlights curve comes from the level histogram, which had moved.
      The white point, the default black, Contrast and Whites above zero
      go by the same statistics.
    - **What it is now.** `clip_stats.rs`. `measure` reads 16 frames
      spread evenly over the WHOLE clip (not the trimmed part: a clip's
      look does not move with its trim), draws each eye as the
      half-equirect its own lens sees (identity rotation: no
      stabilization, no view adjustment, so every output view of a clip
      gets one look), and keeps a 126 × 126 copy of each (the mean of
      8 × 8 pixels of a 1008 square). `ClipStats::raw(geometry)` and
      `ClipStats::tone(decode)` work the statistics out of the copies on
      the CPU, as the two measuring shaders do on a frame, with all the
      frames as ONE picture (`stats_from_cells`, the frames as so many
      eyes). They depend on how the picture is read (a raw clip's white
      balance; the light a code stands for, which follows the grade), so
      they are worked out when asked for and kept until that changes: no
      frame is decoded again for a slider.
    - **How a frame gets them.** `ColorStackPlan::clip_stats`. The two
      places a frame's statistics are taken (`raw_develop_constants`,
      `linear_tone_constants`) read the clip's when the plan has them and
      measure the frame only when it has none (a still photo; a preview
      that has not sampled its clip yet). `provide_clip_stats` at the top
      of `export_fisheye`, `export_eac` and `export_frame_png16` puts
      them in for an export; `for_clip` keeps what was sampled in the
      process (key: the files, their size and date, the source kind, the
      eye order, a Blackmagic RAW clip's colour settings), so the
      preview, a snapshot and the export of a clip read the same copies.
      In the app: `drive_clip_stats` samples the loaded clip on a thread
      of its own when its settings need statistics (a raw clip always;
      another once a tone control that goes by the levels is moved), on
      `export_device()` (lesson 1: no readbacks on the window's device),
      and `worker_settings` hands the result to the preview worker
      (`Settings::clip_stats`, never saved).
    - **`VR180_TONE_PER_FRAME=1`** leaves every frame of an export to its
      own statistics, for comparing; `VR180_PROBE_FRAME_STATS=1` the same
      for the preview probe, which otherwise samples as the app does.
    - **Verified on the Mac.** The take above, 36 to 43 s, Lens Space at
      1024 px, the user's settings: per frame the sky's green goes from
      130 to 170 as in the user's own export (130 to 169); with the
      clip's statistics 148 to 149, every frame "dark 0.00050 max 1.9865,
      14112 levels", three runs the same bytes, and 15 % faster per frame
      (no readback). GPU tests: the copies give what a frame gives on the
      GPU within 0.02 stops (a raw picture, one of code values), and
      a bright and a dark frame of one clip get the same constants.
    - **Not done.** The reference frame of a view that is not the
      half-equirect is still drawn for every frame, though with the
      clip's statistics only its size is used. The sampled copies are not
      kept on disk: a clip is sampled again in every run of the app.
    - **Windows, to look at.** No code is platform-specific. (1) It
      builds. (2) A raw clip and an I-Log clip with Highlights at −100:
      the log says "clip statistics: 16 frames of … sampled", the
      preview's look settles within seconds of opening, and an export
      shows no brightening over a pan. (3) The sampling runs on the
      dedicated device `export_device()` makes; the window must stay
      responsive while it runs.

40. **Automatic exposure for raw clips (2026-11-18, Mac; UNRELEASED).** The
    user-facing text is in the CHANGELOG (raw development, "Automatic
    exposure"). `RawSettings::auto_exposure`, `auto_ev`, `auto_smooth`;
    the Raw development panel has the switch under Exposure and the two
    controls while it is on.
    - **What it is.** One exposure per frame, smooth in time, and ONE
      development for the clip (`clip_stats::AutoExposure`): a frame is
      `F(camera × 2^E(t))` with `F` fixed. Nothing but the gain follows
      the frame, which is what item 39 is about.
    - **Metering.** `clip_stats::measure(.., meter = true, ..)` also reads
      a frame every `METER_INTERVAL_S` (0.5 s; at most `METER_FRAMES` =
      600 a clip): `meter_take` takes ONE eye (the left picture's channel)
      straight off the capture with `x6b::decode_record(.., Channels::One)`
      and picks the samples of a 21 × 21 half-equirect copy at places
      found once (a 512 px picture whose pixels say their own position is
      drawn through the lens; its blue channel says how much of the
      picture shows there). No demosaic, no upload, no drawing per frame:
      ~30 ms a frame from a share against ~330 ms through the frame
      iterator. `for_clip` meters a clip that `wants_meter` and was
      sampled without (the 16 statistics frames are read at the same
      moments, so the statistics do not change).
    - **What a frame asks for.** The exposure at which the mean of what
      its picture is DISPLAYED as is `AUTO_TARGET` (0.38 of the encoded
      range). The mean is over the copy, weighted cos²(latitude) ×
      cos(longitude); what a part is displayed as is read off
      `grey_response(F)`, the development's answer to a grey of each
      exposed level (its base its own level), through a histogram of the
      copy's levels. So the other controls enter the metering: that is the
      "based on the other grading" the user asked for.
    - **The loop.** `F` is derived from the clip's statistics, and those
      are the statistics of the clip AS EXPOSED (`raw_exposed`: every
      sampled frame's cells shifted by its own exposure, as a manual
      exposure shifts them; `raw_tone::Exposed` carries the headroom of
      the frame the top comes from and the largest gain, for the sensor's
      white). Exposures and `F` are iterated until no exposure moves by
      more than 0.002 stops (7 to 8 rounds from nothing, 25 to 40 ms for
      300 frames; a changed setting starts from the last solution). The
      search is a contraction because Highlights / Shadows curves follow
      the exposed levels (about 0.45 of a shift comes back with Highlights
      at −100). `auto_ev` is then a manual exposure of the clip as the
      automatic one left it (`with_whole_exposure(ev).constants_exposed`).
      Smoothing: a Gaussian over time, `auto_smooth` seconds wide, on the
      metered exposures; between two of them a straight line. A frame's
      exposure is kept within `AUTO_REACH` (3 stops) of the clip's median.
    - **Where a frame's time comes from.** `ColorStackPlan::frame_time_s`:
      `plan_with_raw_reference` sets it from the pair's `pts_s` in an
      export, `compose_view_with_color_and_mode` gets it from the live
      loop and from a still. Without a time, or for a clip not metered
      yet, the exposure that is set stands.
    - **Verified on the Mac** (same take as item 39, the user's settings).
      A clip held at one exposure (smoothness 100000) and that exposure
      set by hand give the same sky and whole-picture series; the preview
      worker and the export read +1.529 and +1.427 stops at 36.503 s and
      42.492 s alike; the fixed-exposure rendering is byte-identical to
      what it was before the feature. On the user's ten graded takes the
      automatic exposure's mean is −0.99 to +1.33 stops from the exposure
      set by hand, median +0.05: `AUTO_TARGET` was set from their median
      reading at their own exposures (0.375). Tests: 7 in `clip_stats`
      (one brightness for all frames, EV, smoothness, a steady clip =
      the same exposure by hand, the reach, the meter's reading), the raw
      panel's test.
    - **Limits.** The sensor's white is taken at the clip's LARGEST
      exposure: in a frame exposed far below that, a blown area is shown
      a little under white. One eye is metered. The copies are not kept on
      disk: a clip is metered again in every run of the app. Not seen in
      the app's window.
    - **Windows, to look at.** No code is platform-specific. (1) It
      builds. (2) Auto exposure on a raw clip: "Metering the clip… N %"
      under the controls, then the range; the Exposure control follows
      playback; the log says "… metered for the automatic exposure".

41. **Raw development: clipped channels rebuilt as Camera Raw does it
    (2026-11-18, Mac; UNRELEASED).** The user-facing text is in the
    CHANGELOG (raw development, "a highlight keeps its colour").
    - **What was wrong.** `scene()` raised a channel near its clip point
      (0.92 to 0.995 of the sensor's white) to the LARGEST white-balanced
      channel. Right for a neutral, the only case it had been measured
      on. A coloured highlight was repainted: orange with its green at
      the clip point became pure yellow (red = green, blue pushed to zero
      by the matrix), in hard-edged patches.
    - **The rule** (`raw_tone::DevelopGeometry::rebuilt`, the one WGSL
      copy `shaders/raw_scene.wgsl`): white-balanced by the white in
      force, `w(x) = clamp((x − ¼) / ¾, 0, 1)²` with `x` the channel's
      camera value (`CLIP_FROM`, `CLIP_TO`, `rebuild_weight`). The channel
      that clips first (the largest of the white's camera values; green,
      then red, on a tie) goes to `own + w · max(0, mean of the other two
      − own)`; of the other two the lower goes to `own + w · (higher −
      own)`, which is done first. Per pixel, nothing of the surroundings.
    - **One source for the shaders.** The four raw shaders that read a
      camera pixel had a copy each. They now say `//#use SCENE` and
      `raw_gpu::with_scene` composes `raw_scene.wgsl` in
      (`shader_compose`). A test parses and validates the four composed
      sources without a GPU.
    - **Verified.** Camera Raw's own values for sixteen patches are a
      test; the take's case is another (orange through the clip point of
      green: no step above 0.012, blue stays within 0.25 of green). The
      GPU stage against the CPU twin: the ignored tests pass. On the take
      (7000 K / +23, Exposure +1): pixels with red and green at 255 and
      blue under 240 in a frame, 1060 and 919 before, 0 and 0 now; with Exposure alone the wall's brightest parts are within one
      8-bit step of Camera Raw's render of the same linear DNG.
    - **Known, not changed.** (a) With Highlights below zero the
      brightest coloured parts come out more saturated than Camera Raw's
      (green 9, blue 22 of 255 under on that wall at −43): the
      Highlights gain there, not the rebuilding. (b) A view with a black
      surround (the lens frame, a fisheye view) gets a much stronger
      Shadows / Highlights from Camera Raw than from the app at the same
      numbers: its statistics count the surround, the app's are the
      clip's, read off a view without one. In the half-equirect view the
      two agree (1.5 of 255 for either slider on that frame).
    - **Windows, to look at.** No code is platform-specific. (1) It
      builds and `the_scene_is_composed_into_every_shader_that_reads_it`
      passes. (2) A raw clip opens and looks as before away from its
      brightest parts.

42. **Colour noise reduction: neighbours count by brightness too
    (2026-11-19, Mac; UNRELEASED).** `raw_chroma.rs` weighed a neighbour
    by how near its colour was and summed VALUES, so a bright neighbour
    counted by its brightness and a sky coloured the dark pixels beside
    it. Camera Raw measured with edges of light: no colour crosses a
    step of a stop or more beyond the first four pixels at any amount,
    and the mixing falls with the step as a Gaussian of about a fifth of
    a stop. `LEVEL_SIGMA = 0.3` stops, a second Gaussian in the weight
    (`raw_chroma::smooth`, `shaders/raw_chroma_smooth.wgsl`, the uniform's
    `range.y`). Tests: the model's numbers (the numpy twin's),
    `a_bright_neighbour_does_not_colour_a_dark_pixel`, the GPU stage
    against the CPU one. Against Camera Raw at amount 25 (rms of the
    white-balanced camera values × 1000, R / G / B) on two frames:
    0.90 / 0.33 / 0.76 to 0.83 / 0.31 / 0.70 and 0.59 / 0.23 / 0.49 to
    0.56 / 0.22 / 0.47. The width that fits frames best (0.3) is a little
    over what the edge charts gave at amount 100 (0.2).

43. **Shadows / Highlights: the gain is smoothed over cells of four
    pixels (2026-11-19, Mac; UNRELEASED).** For raw clips and for the tone
    controls of every other clip. The user-facing text is in the
    CHANGELOG (raw development, "leaves against a sky").
    - **What was wrong.** `gain = curve(base)` pixel by pixel, the base
      read from the grid at the pixel's own level. Fine mixtures of bright
      and dark (leaves against a sky, Highlights −100) lost their
      contrast: every bright pixel got the open sky's gain, the dark ones
      none.
    - **Camera Raw, measured** (charts through Photoshop, 2048 and 4096
      wide): mixtures of two levels at feature sizes 1 to 16 px and dark
      shares of 20 / 50 / 80 %, three contrasts, with and without a
      lens-like blur, both sliders; steps placed at every position modulo
      8. Its gain is on cells of FOUR PIXELS whatever the picture's size
      (a step on a multiple of 4 is crisp, one inside a cell is not, the
      same at 2048 and at 4096); on the side the slider moves it fades
      toward an edge over about six pixels; on the other side it is
      exactly zero; bright pixels in a fine mixture get about the share of
      bright among them times the open area's gain; dark features of one
      to three pixels take their surroundings' gain, of four and more
      none.
    - **The stage now.** (1) The reduced copy the grid is gathered from:
      texels of `GAIN_CELL` (4) pixels of the exported frame
      (`raw_local::gain_cell(pixel_scale)`: fewer in a frame drawn
      smaller, so a preview shows the export's), each the POWER mean
      (`BASE_POWER` 0.5) of its pixels' luminance, no longer the mean of
      their levels. The grid's cell is unchanged (`BASE_CELL` of the long
      side, `raw_local::base_cell`); `BASE_FALLBACK` is now a share of the
      texels a cell holds. (2) The gain of every pixel's base is written
      to a texture (`raw_gain.wgsl`, `tone_gain.wgsl`: `R32Float`), its
      means over the cells to another (`gain_cells.wgsl`: `Rgba16Float`,
      shared), and the stage (`raw_develop_local.wgsl`,
      `tone_linear.wgsl`) takes the cells' means, linear between their
      centres, held between zero and the pixel's own gain smoothed over
      a 5 × 5 binomial kernel (3 × 3 for cells of 2 or 3 pixels, none for
      1). CPU twin: `raw_local::{gain_cells, smooth_gain, limit_radius}`,
      `DevelopConstants::develop_gained`, `ToneConstants::apply_gained`.
      `LocalTone::cell` / `ToneConstants::cell` carry the cell
      (`ColorStackPlan::raw_pixel_scale` sets it for both stages).
    - **Verified.** Unit tests of the smoothing (a flat gain is kept,
      bright among dark takes its share, an edge fades on one side and
      is left alone on the other); the two GPU stages against the CPU
      twins (ignored tests, pass). Against Camera Raw, gain in stops rms:
      charts 0.120 to 0.067 (Highlights) and 0.276 to 0.142 (Shadows) for
      the strict limit, 0.078 / 0.169 for the 5 × 5 one that is in (the
      real frame wants the softer limit); three indoor frames at pixel
      resolution: Highlights the same (0.071 / 0.080 / 0.067), Shadows
      0.101 / 0.134 / 0.102 to 0.096 / 0.126 / 0.097; an outdoor one
      0.097 to 0.099 and 0.117 to 0.111. The frame that showed it (a
      half-equirect at 4096, Exposure +1.3, Highlights −100, the same
      linear DNG in Camera Raw), over the tree: mean difference 5.27 to
      4.27 of 255, the bright pixels −6.4 / −7.4 / −9.8 to +0.3 / −1.0 /
      −3.0. Cost: 120 frames at 8448×4224 in 10.5 s with Highlights −100
      against 9.5 s with no local stage at all.
    - **Not reproduced.** Camera Raw's dark features of four pixels and
      more inside a bright area get exactly no gain; here those within
      two pixels of the bright take a part of it (the limit is a smooth
      one because real pictures want it: with the strict limit the tree
      was 4.7, with none 4.05). In a mixture that is 80 % dark Camera Raw
      still pulls the bright pixels by a tenth of a stop or two, here
      next to nothing. Camera Raw darkens dark specks in a bright area by
      MORE than the area (its two finest scales look like detail added
      back in another domain). The model behind it is a pyramid; this is
      its first two levels' worth.
    - **Every developed frame with Shadows or Highlights set changes**:
      the stored regression baselines (already stale) are to be made
      again from this commit.
    - **Windows, to look at.** No code is platform-specific; three new
      compute shaders (`R32Float` and `Rgba16Float` storage, a filtered
      read of the second). (1) It builds; the two GPU tests
      (`gpu_local_tone_matches_the_cpu_one`,
      `gpu_tone_base_matches_the_cpu_one`) pass on D3D12. (2) A raw clip
      and an I-Log clip with Highlights at −100 show a tree against a sky
      with dark leaves and bright gaps, not a pale haze.

44. **Original DNG of the frame on screen; dropped frames in the file
    info and on the timeline (2026-11-19, Mac; UNRELEASED).** Two small
    GUI additions the user asked for.
    - **Original DNG** (transport row, beside Snapshot; Shift+S; raw
      video clips only): `App::request_original_dng` pauses, then a thread
      runs `sensor_dng::export_sensor_dng_frame(take, dir, name, t)` — the
      frame the timeline has on screen at `t` (`FrameTimeline::row`), its
      files written by the same `PairWriter` the sequence uses (the two
      share it; `Place::Lone` puts them side by side as
      `<name>_CAM<channel>.dng`, `sensor_dng::lone_frame_file`, and writes
      nothing else: no folder, no index, none of the take's files). Place
      and name are the snapshot's: `saved_frame_place` = `<stem>_frame<N>`
      in the export folder or beside the clip (the snapshot takes its own
      from the same function), and `unused_frame_name` /
      `unused_original_name` go on to `-2`, `-3` … while a file of the name
      is there, looked for on the writing thread. `poll_original_dng`
      reports through the snapshot status line. (Built first into a
      `<name>_OriginalDNG` folder with its index, `746fdcc`; the user
      asked the same day for the files beside the snapshots.)
    - **Dropped frames** (an X6 video file or raw take:
      `ClipInfo::x6_holes`): `FrameTimeline::holes()` → `Hole { row, at_s,
      frames }`, taken in the load worker from the timeline it already
      built for the frame count (`LoadedMeta::holes` → `ClipInfo::holes`).
      File info: `dropped_frames_label` / `dropped_frames_hover` (the
      first 24 places). Timeline: an orange mark per hole over the track's
      upper edge, as wide as the hole and 2 px at least; within 4 px of
      one the pointer shows `dropped_hole_text`.
    - **Verified.** Unit tests (the hole list, the texts, the place and
      the names, the next free name); on a sample take a frame written
      alone is byte for byte the file a one-frame sequence writes, and
      its folder holds those files and nothing else; the hole list on the
      takes and video files at hand (one take with 9 in 9 places = its own
      count of unsaved requests, the rest none; video files with up to 53
      in 29 places). NOT seen in the window: no windowed run while the
      user was at work in the app.
    - **Windows, to look at.** No code is platform-specific. (1) The
      Original DNG button appears for a raw clip only and writes
      `<clip>_frame<N>_CAM0.dng` + `…_CAM1.dng` where the snapshot's JPEG
      goes. (2) An X6 clip with dropped frames shows the orange line in
      File info and the marks on the timeline.

45. **A run keeps its speed behind other windows, and the Mac awake
    (2026-11-19, Mac; UNRELEASED).** The user-facing text is in the
    CHANGELOG (Fixes).
    - **What was wrong.** The app never told macOS it was working, so App
      Nap applied to it whenever it was not the front app or its window was
      covered: `ps -M -p <pid>` shows every thread at priority 4 then
      (31 to 47 in front). Measured on one batch, same settings: 10.3
      against 5.4 frames a second (two raw clips at 8192×4096), 1.65
      against 0.46 (a 7680² raw clip with Noise Reduction), and a 60p Lens
      Space export at about 4.5 where a night run with the app left in
      front made 11.2. The share (10 GbE, 930 MB/s on a test read), memory
      and the GPU (25 %) were not the limit. Nothing kept the machine from
      idle sleep during a run either.
    - **What it does now.** `activity::Busy` (`crates/vr180-gui/src/
      activity.rs`): `NSProcessInfo beginActivityWithOptions:reason:` with
      `NSActivityUserInitiated` (no App Nap, no idle system sleep; the
      display may sleep), ended when the value is dropped. `App::logic`
      (the one function eframe still calls while the window is covered)
      holds one in `App::busy` while `batch_running || export_job.is_some()`
      and drops it when neither is true (`tell_the_system`): from a run's
      first job to its last, the join of a one-file run and a run stopped
      by the user included. The log says when it is taken and given back.
    - **How to tell.** `pmset -g assertions` lists the app with
      `PreventUserIdleSystemSleep` named "Exporting" while a run is in
      flight; `ps -M -p <pid>` keeps its priorities with another app in
      front.
    - **Verified.** Two tests in `activity.rs` ask the system itself: the
      activity is listed under its reason with `PreventUserIdleSystemSleep`
      while the value is held, is gone once it is dropped, and of two the
      one that ends leaves the other in force. MEASURED on the first run
      with this build, another app in front the whole time: a 60p raw clip
      at 8192×4096 at 12.0 to 12.9 frames a second (133 to 143 MB/s read
      from the share), where the same clip's Lens Space export had made
      5.1 behind another app before and 11.2 with the app left in front;
      the app was listed with the system, and its threads read 20 to 47
      (4 before).
    - **Windows, to look at.** `activity::platform` is empty off macOS and
      `Busy::begin` returns `None` there, so nothing changes on Windows.
      Whether a run is slowed there when the window is minimised or behind
      another (Windows throttles background processes on some power
      plans) and whether the PC sleeps during a long batch has not been
      looked at; if it is, the Windows half belongs in that module
      (`SetThreadExecutionState` for sleep, the process power-throttling
      opt-out for speed) behind the same `Busy`.

46. **Raw takes of other frame sizes (2026-11-21, Mac; UNRELEASED).** The
    user-facing text is in the CHANGELOG (the raw video section, "Other
    frame sizes"). A paired take may hold frames of 5504×5504 or 3968×3968
    (10 bits) or 4912×4912 (14 bits) in place of 3840×3840.
    - **The capture** (`vr180_fisheye::x6b`, module header). `Layout::
      PairedOf(PairedFrame { side, coded, bits })`, told from `clip.json`'s
      `size` (`PAIRED_FRAMES`; `encoded_bayer_size` and `sensor_bits` are
      held against it, and the frame rate setting: 30 for the two larger
      frames, 50 or 60 for 3968). The record header is the paired one with
      16 tiles an eye, a zero where the 3840 layout has its threshold and
      32 payload lengths. `Profile::Frame10` (`X6D5B02`): a tile is a
      predictor stream in the split block layout and nothing else
      (`unpack_split`, as a take of one lens in its later layout).
      `Profile::Frame14` (`X6D4B01`): a tile is coded as a 30 fps 3840
      take's (`unpack_symbols`, the table that keeps every value, the tail
      of low bits) on a canvas of 4928×4928, of which the frame is the
      first 4912 columns of the first 4912 rows (`decode_tile` decodes the
      canvas tile into scratch space and copies the frame's part; the last
      tile holds 292 of its 308 rows). `Layout::coded_side`,
      `tiles_per_eye`, `bits`; `Profile::records(layout, fps)` replaces
      `fps()`; the kept pair lists name a profile by its place in
      `Profile::ALL` (the two new ones are last).
    - **The 14-bit frame's levels.** Its samples are the sensor's 10-bit
      values four bits up: on a take of it no sample of six eye frames has
      one of its low four bits set, none lies below 976, the darkest patch
      of a night scene sits at 1048 and saturated lights read 16368. So
      black is 1024 and white 16368. A take that states 64 and 15408 for
      such a frame states the levels of the 3840 layout (whose values are
      the sensor's less 960): `x6b::sensor_levels` then reads it by the
      sensor's own, `Take::stated_levels` keeps what was stated and the
      log says so once when the take is listed. A take that states any
      other pair is read as it says.
    - **Where a frame sits.** `x6b::read_sensor_windows` reads each eye's
      place from the take's sensor record: the area read
      (`analog_roi_register_xyxy_inclusive`), the binning, the kept part
      (`digital_crop_after_binning_xywh`), the size it leaves the sensor at
      (`bayer_output_wh`, smaller where the sensor scales it) and the
      frame's window in that output (`recorded_bayer_window_xywh`).
      `x6_raw::stated_frame` puts it in the calibration's frame
      (`CALIBRATION_ORIGIN` = 224: the 7680² video window lies 32 inside
      the 7744² calibration frame and at 256 on the sensor) and
      `Insta360Meta::stated_frame` → `InsvStreamGeometry::stated` hands it
      to the lens calibration in place of the recording-mode table. By
      their records: 3840 binned and 7680 of one lens → origin 32, scale
      0.5 / 1 (what they were placed at before, now from the file); 5504 →
      origin (−88, −84), scale 16/23; 4912 → 1416; 3968 → 1888, both at
      scale 1 and both a crop of the image circle (`cropped`: the frame
      covers less than the video window). The 5504 frame is 7912 columns
      of a kept part of 7920: the FIRST 7912. It was taken to hold the
      last 7912 at first (origin −80: where the camera's own video files
      place that window, 4 pixels right of the middle) and that was 8
      sensor pixels, 0.19°, off: measured 2026-11-23 by the sensor's
      fixed pattern (item 47, "On a camera's own takes"). An origin
      fitted from gyro against tracked features cannot tell: it scatters
      by 20 to 30 pixels on every mode.
    - **Rows.** `EyeWindow::first_to_last_row_ns` = the sensor rows read ×
      the record's line period, the record's own figure where it agrees.
      For the scaled 5504 frame the record has none: 7912 rows at 4000 ns =
      31.644 ms (the camera's video files of the same window state 31.65).
    - **The app.** `RawFrame` (app.rs: `BatchItem`, `ClipFacts`,
      `ClipInfo`, `LoadedMeta` carry one beside `raw_one_lens`) gives a raw
      take its `x6_geometry` and `sharp_eye`: a take whose frame is a crop
      gets the Crop mask section and the native-sharpness size of the
      Resolution list (6560 an eye), as a Crop video clip does. Nothing
      else of the GUI asks which frame a take has.
    - **DNG.** `sensor_dng::tile_rows_for`: the capture's own tile rows
      where a whole number of them is the frame (344, 248), else the
      fewest even rows that are with at most 24 tiles: 614 for the 4912
      frame. `eye_dng` writes as many tile entries as the file has tiles
      (it wrote 24 whatever the frame). Levels and bits are the take's as
      read (1024 / 16368 at 14 bits for the 4912 frame).
    - **Verified.** Unit tests: records of all three frames decode to what
      was stored, tile for tile (two tile pictures in a known order, so a
      misplaced tile would show), on one thread and many, one channel
      alone, read from a take whose records lie out of request order;
      refusals (a signature of the other depth, a frame rate the frame is
      not recorded at, header fields of another layout, bytes after a
      10-bit tile's stream, a 14-bit tail cut short or over-marked, a
      canvas or depth that is not the frame's); the levels rule; the
      sensor record's five frames; the placement of each and that 3840 /
      7680 keep theirs; a DNG of 4912 and of 3968 read back sample for
      sample with 8 and 16 tiles; the crop features. On three takes (one
      of each frame, 5504 at 30, 4912 at 30, 3968 at 60): first, middle
      and last record of each decode sample for sample as an independent
      reader decodes them (18 eye frames); each exports through the real
      pipeline and the pictures were looked at (the whole view; the crops
      with black outside the frame; neutral walls and true blacks,
      including the 4912 take read by the sensor's levels); stabilized,
      shake of a locked export at 1024 px an eye, median / 90th
      percentile px a frame: 5504 1.28 / 5.3 → 0.067 / 0.21, 4912 0.44 /
      0.89 → 0.038 / 0.079, 3968 0.30 / 0.70 → 0.050 / 0.105, and with the
      row correction scaled 0 / 0.5 / 1 / 1.5 / 2 the 5504 take's 90th
      percentile reads 0.38 / 0.28 / 0.21 / 0.24 / 0.30 (least at 1, with
      the 31.6 ms above); Camera original DNGs of each written and the
      4912 one opened by the system's raw decoder. NOT seen in the app's
      window. NOT found: pixels that read high in every frame (looked for,
      to measure the frames' places on the sensor directly: none in 120
      frames of each take).
    - **Windows, to look at.** No code is platform-specific. A take of
      each frame size should open, preview and export; a 4912 or 3968 take
      shows the Crop mask section and offers the native-sharpness size.

47. **Raw takes with a record for each eye (2026-11-22, Mac; UNRELEASED).**
    The user-facing text is in the CHANGELOG (the raw video section, "A
    record for each eye" and "The camera's white balance").
    - **The capture** (`vr180_fisheye::x6b`; the module header has the
      record header's table). `Layout::EyeRecords(EyeFrame { width, height,
      coded })`, told from `clip.json`'s `layout`
      (`paired-bayer-eye-records-v2`) and `size` (`EYE_FRAMES`: 5760×5760,
      6000×6000 coded 6016 wide, 4608×3968; a size that is none of them is
      refused by name). A record is ONE sensor's frame: a 4096-byte header
      of its own (`EyeHeader`), then stripes of 32 rows, each a `u32`
      length and a predictor stream in the split block layout
      (`unpack_split`), its prediction starting anew. Five signatures →
      `Profile::EyeScaled` / `EyeScaledNear` / `EyeNative` /
      `EyeNativeNear` / `EyeWideNear` (`Profile::eye_record`: the readout
      mode and the quantiser such a record's header must state). The
      "Near" ones store a value to within one code: their symbols are a
      level table like the 50 / 60 fps ones (`runs`: 0…84, every other
      value to 894, 896…1023; 618 levels), so a stripe is decoded like a
      tile with no tail. The 6000 frame is decoded stripe by stripe on its
      6016-wide canvas and its part copied out (`decode_tile`, as the 4912
      frame; the last stripe holds 16 of its 32 rows). `Layout::side` and
      `coded_side` are gone: `width`, `height`, `coded_width` (a frame need
      not be square).
    - **Pairs.** `scan_pairs` → `scan_eye_records` reads every header (two
      a pair) and `pair_eye_records` pairs sensor 0's frames with sensor
      1's by when their exposures started: the nearest within a
      millisecond, each frame once. The request numbers in the headers are
      not used (the two sensors' do not agree). A pair goes by sensor 0's
      frame number (`PairEntry::request`), which is what the closing
      record's window and `exposure.csv` count in. `PairEntry::other` is
      sensor 1's record and `shutter_ns` what each header says of its
      integration time (`TakeIndex::new` takes that before the take's
      list). `read_record` reads the two records one after the other and
      `decode_record` → `decode_eye_records` takes them apart, so the
      consumers (preview, export, DNG writer, clip statistics) read a pair
      as they did. A frame with no partner is in no pair; a frame number of
      the window without a pair is a missing one, and the timeline's hole
      is filled as ever. The closing record's `eye_frames` is held against
      what the walk finds, and the log says how many frames of each sensor
      were left out.
    - **Checks.** Header: the version, the frame's and the canvas's size,
      the stripe count, the quantiser, zeros where the predictor and block
      coder are named, the readout mode, sensor 0 or 1, a length that is a
      multiple of 4096 and no longer than a frame can be, stripe lengths
      that add up to the stated total. Walk: one signature through the
      capture, places in the file rising, positive stamps, every record
      inside the file, a sensor's frame numbers once each with rising
      stamps. Decode: each stripe's own length against the header's, and
      the CRC-32C of the stripes (`x6b::crc32c`: tables, eight bytes a
      step, no `unsafe`; it runs on threads of its own beside the stripe
      workers, so a pair still decodes in 10–13 ms).
    - **The kept pair list** has a second form for these takes
      (`X6BIDX02`, rows of 69 bytes: the second record's place and the two
      integration times after the 37 bytes of `X6BIDX01`). A list of one
      form is not read as the other.
    - **Where a frame sits** (`x6b::read_sensor_windows`). The scale of a
      frame the sensor scales is the scaler's own setting where the record
      carries the sensor's registers (`scaling_registers`: 16 / M with the
      scaler on), else what the rows give, as before. 5760 scaled: origin
      (136, 136), 16/22, all of the kept part. 6000 scaled: 16/21, that is
      7875 pixels each way of a kept 7880: the first ones, origin (156,
      156). A scaled frame that holds less than its kept part starts
      where the kept part starts (measured, see "On a camera's own
      takes"; the first guess, the last columns and the middle rows, put
      this frame at (161, 158.5)). The frames at the sensor's own
      pixels: (1216, 1216) for
      5760, (1096, 1096) for 6000, (1792, 2112) for 4608×3968; all three
      are crops (`cropped`). Rows: the rows read × the line period, which
      gives the two scaled frames 31.676 and 31.496 ms (their records
      state no span).
    - **White balance.** `x6b::read_white_balance`: the file `clip.json`
      names (`white_balance_file`), else `white-balance.json`; per sensor a
      status, the gains and the neutral that follows from them; any record
      whose `schema` ends in `white-balance-v1`. `x6_raw::eye_cameras` →
      each sensor's own neutral in its Camera original DNGs
      (`PairWriter::cameras`). `x6_raw::camera_of` / `camera_at` → the
      camera the take is developed as: the two sensors' neutrals met
      halfway, so both eyes are developed alike. The app carries it as
      `Settings::raw_camera` (set beside `raw_source` in the three places
      that make a worker's settings) and the Raw development section's As
      shot reads from it (`App::loaded_raw_camera`). The demosaic's own
      neutral (`BAYER.neutral`) and the noise filter's stay the reference:
      they only steer interpolation.
    - **DNG.** `Mosaic { width, height }`. `tile_rows_for` no longer takes
      the capture's tile rows where there are more than 24 of them: 240,
      250 and 248 rows for the three frames.
    - **Verified before a camera's own take was at hand.** What stood in
      for one: an independent packer and reader of the format, and whole
      takes made with them. (1) Records of the five signatures
      packed by that packer from test pictures (the quantiser's
      thresholds, every bit width, blocks of 16 beside blocks of 32,
      full-range noise): the independent reader takes them, and this
      reader gives its samples, 66 to 72 million a pair
      (`independent_records_for_one_eye_decode_to_the_same_samples`, run
      with `VR180_EYE_RECORDS`). (2) Unit tests: the signatures and
      their frames and rates, the level table against the rule value by
      value, takes whose records lie out of step with one sensor a frame
      ahead and a frame of the other never written, the pairing rule, the
      refusals (every header field, lengths that do not add up, a wrong
      checksum, a byte changed, records cut short, two records of one
      sensor, two kinds in one capture, counts that are not the closing
      record's), the kept list's second form, the sensor record's five
      frames, the white balance record and what is made of it, DNGs of the
      new sizes with a sensor's own neutral. (3) Five takes made from two
      real 3840 takes (each eye resampled into the frame's window of the
      sensor and packed by the independent packer, the stamps and the row
      period those of the content): every record passes the independent
      reader's own checks; each take opens with the pairs, the hole and
      the frames left out it was made with; the windowless preview worker
      shows it (the 4608×3968 one as 2304×1984 eyes); it exports, and the
      export's picture lies on the source take's within 0.003 px at
      1024 px an eye, both eyes, all five; a locked export shakes as the
      source's does (median px a frame 0.69 for 0.75 at 30 fps, 0.71 for
      0.72 at 60); Camera original DNGs carry the frame's size, tiles,
      levels and each sensor's neutral, and the system's raw decoder opens
      them. That stretch does not tell a row timing apart: a row period
      three times too long raised the 90th percentile from 1.84 to 2.09
      only.
    - **On a camera's own takes (2026-11-23).** Nine takes: seven of this
      layout (all five signatures) and two of the paired one. (1) Each
      opens at once: `clip.json` and the closing record have every field
      the reader goes by, the walk finds the closing record's counts, and
      the pairs are the take's own list of them row for row (9903
      pairs: number, both stamps, both records' places and lengths, both
      integration times). Most of them have holes, each sensor's own; the
      timeline holds the frame before a hole as ever. The request
      numbers in the take's lists are one above the headers' (the reader
      goes by neither). (2) An independent reader over five whole
      captures, one of each signature (148 GB,
      12506 records): every header, order and checksum right, and the
      pair it wrote out of each is this reader's, sample for sample (18
      to 36 million a sensor). (3) Row timing. Features tracked from
      frame to frame against the gyro through the lens model, the span
      from first row to last fitted, ms (fitted | the app's): 5760 scaled
      31.47 and 31.32 for the two sensors | 31.676; 6000 scaled 32.16
      (slow motion) | 31.496; 6000 at the sensor's own pixels 26.03 |
      25.996; 5760 so 24.93 | 24.956; 4608×3968 16.16 | 16.033; the
      middle row's time within 0.25 ms of the app's on all. "Rows of the
      frame × line" (24.0 ms for the scaled frames) is ruled out. In the
      app: a turning stretch exported locked with the per-row correction
      at 0 / 0.5 / 1 / 1.5 / 2 times (`VR180_RS_SCALE`), far-field shake
      in px a frame at 1024 px an eye: 5760 scaled 0.40 / 0.29 / 0.12 /
      0.26 / 0.40; 4608×3968 0.097 / 0.059 / 0.045 / 0.070 / 0.109; 5760
      at the sensor's own pixels 0.109 / 0.065 / 0.054 / 0.071 / 0.135;
      6000 scaled 0.072 / – / 0.044 / – / 0.071. (4) Where a frame sits,
      by the sensor's fixed pattern: the mean of 80 frames of a take, its
      dark corners outside the image circle (there the pattern of two
      takes of one mode correlates 0.6), one frame's pattern taken to
      sensor coordinates and laid on another's, colour plane by colour
      plane, the origin of the best fit read off. A second take of the
      same frame: (136.00, 136.00) of 136, each corner within 0.02. 6000
      scaled against 5760 scaled: (155.93, 155.94), both sensors, two
      pairs of takes, each of the four corners within 0.12 (so the scale
      is the scaler's 16/21 too): the first columns and rows, where the
      code had (161, 158.5). 5504 against 5760 scaled: (136.00, 140.00),
      where the code had 144 across. 5760 scaled against 6000 at the
      sensor's own pixels: (136.18, 136.18), a fifth of a pixel that is
      the scaler's own sample phase and is left alone; the three frames
      at the sensor's own pixels lie on each other to 0.02. The
      correction moves a 6000 scaled picture by 0.12° across and 0.08°
      down, a 5504 one by 0.19° across, both eyes alike, so nothing
      between the eyes changes; a 5760 scaled export is the same file as
      before.
    - **NOT verified.** The app's window by this session (takes of every
      frame were opened and played there by hand that day). Whether the
      white balance the camera records is the one the scene wants.
    - **Windows, to look at.** No code is platform-specific. A take of
      each frame should open, preview and export; a take that records a
      white balance shows it as As shot. The placement correction is
      shared code: a stored list of a 5504 take's exports changes.
    - **WINDOWS VERIFIED (2026-11-23, Windows box, RTX 4090).** It builds as
      pushed; tests core 63, fisheye 85, braw 6, pipeline 299 of 300 with the
      GPU ones (the one is the fingerprint tool), gui 66; on D3D12 pipeline
      298 of 300 (the lens frame's outermost pixels, as before). No code was
      changed for it.
      - **Nothing that was there moved.** The 61 exports of the A/B set are
        byte-identical per stream to the build before, with the same 38
        engagements of a fast arm, and the 36 preview-probe cases print the
        same lines. The capture reader of the commit before and this one,
        run over every take on the share (53 takes: 3840 paired at 30 / 50 /
        60, one lens, 5504, 4912, 3968; the first, middle and last pair of
        each), give the same pairs, stamps and samples. The exports of the
        5504, 4912, 3968 and one-lens takes are the stored ones (libx265 and
        ProRes byte for byte, NVENC the same decoded picture).
      - **Takes of the new layout were made here too**, none from a camera
        being at hand. `examples/eye_take_make` (gitignored) resamples each
        eye of a real 3840 take into the frame's window of the sensor (per
        Bayer site, bilinear), packs the stripes with a packer written from
        the format's description, and writes the take around them: the
        sensor record a take of that frame carries (the line period set to
        the content's) and the Mac's three irregularities (the two
        sensors' records alternating, a frame of the second sensor before
        the first pair, one of the first sensor missing). The reader they
        are held against is an independent one: two scalar unpackers
        compiled here as they are (x86-64, clang), inside a walk of the file
        that makes a verifier's checks. All five signatures: every record
        passes; the first pair's samples are the packed ones for both
        readers; every pair of every take (15 pairs of the four 30 fps
        takes, 29 of the 60 fps one) has the same samples from the app's
        reader as from the reference one (`examples/eye_take_check`); the
        two records with no partner are in no pair, and the frame number
        without a pair is the hole.
        `independent_records_for_one_eye_decode_to_the_same_samples` passes
        on those records. A pair decodes in 8 to 15 ms.
      - **Exports** (`export_smoke`, 36 runs, none failed). Each take
        through the portable loop's plane feed and NVENC, with the pairs
        read ahead (the Windows opener) and read in line: the same bytes,
        five of five. libx265, ProRes (`prores_ks_vulkan`), lens frame,
        reframed view, crop mask, a graded development, 8192 × 4096. The
        hole is filled: 16 frames from 15 pairs, 30 from 29. Vulkan against
        D3D12: signed difference 0.01 of 255, mean |Δ| 0.8. Each take's
        first frame against its source take's, unstabilized at 1536 px an
        eye: the picture within 0.02 px, both eyes, all five (mean |Δ| 1.3
        to 1.9 of 255: the resampling and the 10-bit copy).
      - **White balance.** A sample of one pair of a 3840 60 fps take with
        its white balance record and the hashes of both: the record's bytes
        and both eyes' decoded pixels have those hashes; set up as a
        take of one pair it is developed for neutral [0.6102, 1, 0.3932],
        "about 3465 K, tint +17" (against the reference neutral: R −5.1 of
        255), and its walls come out grey. A test take given that record:
        the Raw development section shows 3465 / 17 with As shot in force (a
        take without one: 3730 / 27), and each sensor's Camera original DNG
        carries the record's neutral (the reference one where there is no
        record). DNGs of the three frames: 5760² and 6000² in 24 tiles of
        240 and 250 rows, 4608 × 3968 in 16 of 248, 10 bits, 64 / 1023.
      - **In the app** (a scratch app folder). The takes open as "Insta360
        X6 raw video" with their frame and rate, and the hole in File info
        and on the timeline. The 4608 × 3968 one previews as 2304 × 1984
        eyes and plays at its 30 shown a second; the clip's statistics are
        sampled in 4 to 7 s with the window answering. The two Crop takes
        opened (5760² and 4608 × 3968) have the Crop mask section, and the
        second's Resolution list the native-sharpness size. Export all
        writes 8192 × 4096 through NVENC (30 frames, stereo sound from the
        take's four channels, 16 frames a second); Shift+S writes the
        frame's two DNGs; the kept pair list is written in its second form
        and read at the next launch ("29 pairs (kept list)").
      - **Seen, left alone.** (1) "Native" in the Resolution list of the
        4608 × 3968 take is 9216 × 3968: a half-equirect whose eye is not
        square, and past NVENC's 8192 a side (software H.265 on Windows).
        (2) A take whose `clip.json` names a sound file that is not there
        exports without sound, after "one-pass audio attach failed: open
        audio src <the take's folder>: Permission denied". (3) A Camera
        original sequence copies every file beside the capture, whatever it
        is. (4) The preview probe on a take of one pair ends "the worker
        ended before t = 0 s": the worker has shown its one frame and gone.
    - **WINDOWS, ON THE CAMERA'S OWN TAKES (2026-11-24, Windows box, RTX
      4090).** Pulled the placement correction (`c4fd70a`). It builds as
      pushed; tests core 63, fisheye 85, braw 6, pipeline 299 of 300 with the
      GPU ones (the one is the fingerprint tool), gui 66. No code was
      changed for it.
      - **What moved is the 5504 frame, by what the correction says.** The
        61 exports of the A/B set are byte-identical per stream to the build
        before, with the same 38 engagements of a fast arm, and the 36
        preview-probe cases print the same lines. Of the stored exports of
        the older raw frames, those of the 4912, 3968 and one-lens takes are
        the same bytes; the 5504 take's picture lies 1.85 to 1.9 px to the
        left at 1536 px an eye in the half-equirect and 1.2 px in its lens
        frame, the two eyes within 0.01 px of each other, nothing up or down
        (origin −88 where it was −80).
      - **The nine takes of 2026-11-23, read.** All open. Pairs, missing
        pairs and frames left out are the Mac's table's (540 pairs with 259
        missing and 132 and 147 frames without a partner on the take that
        lost a third; none missing on the first). Every 97th pair and the
        last of each take (143 pairs of the seven takes of this layout):
        the stamps are those of the take's own list, and both sensors'
        samples, stamps and frame number are the reference reader's on the
        same records (the independent reader's two scalar unpackers compiled
        here, 2026-11-23, given the records' places in the file). A pair
        decodes in 7 to 17 ms. The frames sit where the Mac measured them:
        5760 scaled (136, 136), 6000 scaled (156, 156), the three at the
        sensor's own pixels (1216, 1216), (1096, 1096) and (1792, 2112).
      - **Camera original DNGs.** Four frames the Mac's app had written
        beside the takes (5760 scaled and 6000 scaled, two takes of each),
        written again here from the same pairs: both sensors' files are the
        same bytes as the Mac's, eight of eight. A frame of each of the nine
        takes: the frame's size, its tiles (24 of 240 or 250 rows; 16 of 248
        for 4608 × 3968 and 3968²; the 3840 take's 24 of 160), its levels
        (64 / 1023 at 10 bits, the 3840 take's 64 / 15408 at 14) and the
        take's recorded neutral.
      - **Exports** (`export_smoke`, 44 runs, none failed). Each of the nine
        takes through the plane feed and NVENC over a second that has
        missing pairs: one pts step in every file and the holes filled (18
        pairs written as 30 frames on the take that lost a third, 22 as 30,
        55 as 60 at 60 fps), stereo sound from the take's four channels,
        read ahead and read in line the same bytes, nine of nine. libx265,
        ProRes (`prores_ks_vulkan`), lens frame, lens frame with a crop
        mask, reframed view, a graded development, 8192 × 4096. Vulkan
        against D3D12 on three takes: signed difference 0.01 of 255, mean
        |Δ| 0.7 to 0.8. The lens frame of a 6000² take at the recording's
        own size is 13200 × 6600 with the stabilizer's margin (the size the
        Mac writes): NVENC refuses it and libx265 takes over with the
        export-bar note, 0.5 fps; ProRes stays on the GPU. Asked for 4096 an
        eye, that lens frame is a reduced copy, 8192 × 4096 through NVENC. A
        take developed for its own white ([0.410, 1, 0.623], "about 6350 K,
        tint +32", the Mac's figure) against the reference white: R −21.6,
        B +25.3 of 255; the two takes of the paired layout record a white
        too and are developed for it.
      - **Preview and the app.** The windowless preview worker on a take of
        each signature (half-equirect with the zoom still, Lens Space, a
        reframed view, unstabilized): every case draws, no warning. In the
        app (a scratch app folder, the export folder set to one of its own):
        a 6000² scaled take opens as "Insta360 X6 raw video" 6000 × 6000 at
        29.970, with "17 in 16 places" in File info and the marks on the
        timeline; the clip's statistics are sampled in 10 s. Play shows 12.7
        frames a second: the wait for a frame is 77 to 94 ms, the share's
        reading (the take holds 895 MB a second of both eyes; the share
        gives 500 here), not the decode. Export all over two seconds: 8192 ×
        4096 through NVENC, 60 frames from 55 pairs in 4.4 s; Shift+S writes
        the frame's two DNGs; the kept pair list (its second form, 69 bytes
        a pair) is read at the next launch.
      - `examples/eye_take_make` (gitignored) starts a scaled frame where
        the kept part starts now; the test takes of 2026-11-23 were made and
        read by the rule of that day, so their agreement with their source
        takes says nothing of the placement.

48. **Items 36 to 46 on Windows (2026-11-21, Windows box, RTX 4090; on
    `main`).** Pulled `9fd483d` → `0251879` (31 commits). It builds as
    pushed; tests core 63, fisheye 78, braw 6, pipeline 297 of 298 with the
    GPU ones (the one is `projection_fingerprints`, a tool that wants its
    file), gui 63 (66 with the tests of the changes below); on D3D12
    pipeline 296 of 298 (the lens frame's outermost pixels, as before),
    with `gpu_local_tone_matches_the_cpu_one`,
    `gpu_tone_base_matches_the_cpu_one` and
    `the_scene_is_composed_into_every_shader_that_reads_it` among those
    that pass.
    - **Nothing that is not raw moved.** The 61 exports of the A/B set
      against the build before the pull: 55 byte-identical per stream, the
      six raw-take cases different (mean 2.1 to 2.7 of 255 with Shadows +100
      / Highlights −100: items 39 and 41 to 43), all 61 written, the same 38
      engagements of a fast arm. The preview probe, 36 cases: 32 print the
      same lines, the four raw ones differ.
      - The probe's three software-decode cases of two 3840² streams
        (`VR180_NO_HW_DECODE=1`) first ended "no frame for t = 0.5 s". Not a
        picture: `frame_at` asked again every 2 s, a request starts the
        worker's seek over (a frame that arrives with a request waiting is
        let go, in the one loop as in the zero-copy workers), and such a
        seek takes 4 s there. It now waits twice as long before each
        repeat; the three print the lines of the build before.
        Known from this and left alone: while the timeline is dragged a
        seek is sent with every repaint, so where a seek is slower than
        that the frame lands when the pointer is let go.
      - The comparison script read packets at `pos, size`; ffprobe prints
        `size, pos` whatever is asked. With that put right the verdicts are
        the same, those of 2026-11-17 included (the stored sets were
        compared again: 61 of 61 both times).
    - **Item 37, the tone stage** (harness `VR180_TONE="e,c,h,s,w,b"`,
      `VR180_TONE_LIGHT=ilog`). 23 exports with a control moved, none
      failed: an X6 I-Log clip through the portable loop, the D3D11
      readback arm, the GPU-resident NVENC arm and ProRes, 10 and 8 bit, a
      lens frame; an Osmo clip; a `.360` through its GPU-resident arm (the
      pair call written without a compile), the portable loop and ProRes,
      10 and 8 bit. Arms against each other with Highlights −100 / Shadows
      +100: signed difference within 0.14 of 255, what they differ by with
      nothing moved. The two eyes get one treatment: the change by level
      band is the same in both halves to 1.2 (Osmo) and 1.7 (`.360`) of
      255; 6.4 on the X6 clip in its darkest band, the same on all three
      arms. The ETTR recovery through the stage is the ETTR stage (signed
      difference under 0.005 of 255). In the app: the Tone section is
      there for a clip that is not raw, a slider dragged while paused draws
      the frame again on the D3D11 worker, and the zoom still shows the
      same look.
    - **Item 36, joined files** (harness `VR180_MODE=join`). Three X6 clips
      recorded in 360° Audio: Ambisonic gives one four-channel AAC with
      SA3D, an APAC choice the same file, Stereo a two-channel track; three
      libx265 parts (two from the readback arm, one from the portable
      loop) join; two `prores_ks` parts in `.mov` join. In every joined
      file the frames are the parts' frames, packet for packet. From the
      app's own button: two clips at 8192 × 4096 written as parts, joined
      (1200 frames, 40.04 s, the parts' folder gone), with the window in
      front and again minimised from 1.5 s in: the second clip started and
      the join ran while it was minimised. A row dragged by its handle
      changes place. The session comes back at launch; with two more clips
      named on a drive that is not there and on a host that does not
      answer the list was up at once, the first marked missing at once and
      the second after 21 s, the window answering all the while. A project
      whose folder was moved with its clips opens and finds them.
      - **The NVENC arms wrote two sets of parameter sets — FIXED.** The
        GPU-resident arm (`nvenc_cuda.rs`) asked for preset p4, the encoder
        the readback arms and the portable loop feed (`encode.rs`) for p5:
        the SPS differs in two bits and the PPS is another one, and a run
        whose clips took different arms (a camera file and a raw take, or a
        clip that fell off its fast arm) ended "encoded with different
        parameters".
        `encode::NVENC_PRESET` (p4) is the one value for both now. With it
        the three arms write the same sets byte for byte, a mixed run
        joins, and the readback arm writes the GPU-resident arm's packets
        for the same frames, byte for byte. Of the 61 exports the eight
        that reach NVENC
        without the GPU-resident arm changed (raw takes, Blackmagic RAW,
        8-bit, the portable loop with NVENC), the other 53 are
        byte-identical.
      - `project::relative_to` on Windows: a drive or a share in common is
        enough for a relative name (the rule asked for a common folder
        below the root, which two folders at a drive's root do not have),
        with a Windows twin of the naming test the Mac's is gated out of.
    - **Item 38.** `immersive_available()` is false; a joined run of Lens
      Space clips is refused before anything is exported, with the new
      wording; `join_immersive` answers "written on macOS only".
    - **Items 39 to 43, raw development.** A raw take in the app: "clip
      statistics: 16 frames … sampled in 3.8 s" on the dedicated device,
      the window answering every second of it, the frame drawn again when
      they land; with Auto exposure on, "63 metered for the automatic
      exposure", the panel's range line, the Exposure control following.
      Playback at 28 to 30 shown a second once the share has the file
      cached.
    - **Item 44.** Original DNG is in the transport row for a raw clip only
      and Shift+S writes `<take>_frame0_CAM0.dng` and `_CAM1.dng` beside
      the take; an X6 clip with dropped frames has the orange line in File
      info and the marks on the timeline.
    - **Item 45, the Windows half** (`activity.rs`). A `Busy` is a power
      request of the kind "the system is needed" (`PowerCreateRequest` /
      `PowerSetRequest`: no sleep by the idle timer, the display may still
      turn off), and while any is held the process is taken out of the
      system's power throttling (`ProcessPowerThrottling`, execution
      speed). Two tests ask the system for what it holds, and the system's
      own list says the same (`powercfg /requests` from an elevated shell,
      during a joined run of two clips started from the app's button):
      1.3 s after the app logged that it told the system, `vr180-gui.exe`
      is under SYSTEM with "Exporting"; 4 s after the run's last line it is
      gone. PowerToys Awake holds a SYSTEM request of its own on this box,
      which is why the execution state alone says nothing here. On this
      desktop on mains power a minimised
      run was as fast as one in front (30.8 against 31.5 frames a second),
      so the throttling half is for laptops on battery and is not measured.
    - **Seen in that run, not acted on.** A run leaves the preview of a
      DIFFERENT loaded clip alive (on purpose, `spawn_export_job`), and on
      Windows that preview's two d3d11va decoders keep their 7 GB: the
      export of the other X6 clip started with 15.5 GB of headroom where
      it has 22 GB with no preview open, and ran with 1.7 GB left. 33 s in,
      a zoom on the paused preview asked for the native still (decoded in
      software: the VRAM gate declined), and from that frame the
      GPU-resident export fell from 33 frames a second to 6 until the clip
      ended (1799 frames in 198 s); the next clip, whose own preview a run
      stops, made 32.9. Probably the card's memory going over its budget:
      seen once, not reproduced on purpose. By the gate's arithmetic a
      16 GB card has about 8 GB left with such a preview open, against the
      8.1 GB the export's hardware decode asks for.
    - **Item 46.** The three takes of 2026-11-21 (5504 at 30, 4912 at 30
      and 14 bits, 3968 at 60) and a take of one lens (7680²) each export:
      half-equirect, lens frame, a crop mask, NVENC, libx265 and ProRes, 11
      runs. The log states what item 46 says: the 4912 take read by the
      sensor's levels, placed at 1416 as a crop of the image circle, rows
      of 21.3 ms. The pictures were looked at (the crop with black outside
      its frame, neutral walls).
    - **For whoever drives the app by script on Windows.** A process's
      `MainWindowHandle` is winit's event-target window while the app's own
      window is minimised: find the window by its title. `APPDATA` set to a
      scratch folder for the launch keeps the user's settings and session
      out of a test.
    - **MAC SIDE PULLED IT (2026-11-24, Mac):** fast-forward `c4fd70a` →
      `0f31051`: the four code commits of this item and the Windows notes
      of items 47 and 48. It builds; tests braw 6, core 63, fisheye 85,
      gui 64, pipeline 289 (the GPU ones not asked for). By reading,
      nothing in the range runs differently on a Mac. The preset is set in
      the `hevc_nvenc` arm of `encode.rs` alone, and `build_export_cfg`
      names that encoder off macOS only. `activity.rs` gains the Windows
      module and leaves the empty one to the systems that are neither; the
      macOS module is untouched, and no other line inside a macOS-gated
      item changed. A path on a Mac has no `Component::Prefix`, so
      `relative_to` answers as before (its naming test here passes as it
      was). The probe's longer wait is taken only where a seek lasts more
      than two seconds. The stored export and preview sets were not run
      for it.
    - **THE MAC'S STORED SETS AT `79cefc2`, AND MADE AGAIN (2026-11-24,
      Mac).** Run after that. Against the stored listings: of the 65
      exports 56 have the checksums they had, the encoder arms of both
      loops and the `.360` arms all of them, and of the 169 preview lines
      149 print the same, every line of a clip that is not a raw take
      among them. The 9 and the 20 that differ are developed raw takes.
      The lists stored later differ the same way: a developed raw picture,
      or a tone control that goes by the picture's levels (17 of the tone
      stage's 19 exports; the two that are the same have exposure alone
      and no control). Those are what item 39 and the raw work after it
      changed on purpose (item 43 says the listings are to be made
      again). They are this run's now, every changed list run twice with
      the same lines.
      Two lists are new, so every kind of source the app opens has stored
      export and preview cases: Blackmagic RAW (one sensor, and two
      tracks), QooCam 3 Ultra video, an X6 and a QooCam DNG still (PNG,
      JPEG, linear DNG, a disc), raw takes of every later frame (5504, 4912,
      3968, the five with a record for each eye, the two that carry a
      white balance record) as a half-equirect, lens frames with and
      without a crop mask, a disc, a reframed view, the portable loop,
      ProRes, a second with missing pairs (30 frames written), the
      camera's own DNGs of a stretch and a developed DNG frame: 43 export
      lines and 59 preview lines, twice the same. In no list yet: joined
      runs, AIVU, the spatial video of a reframed view, clips of several
      segments, Ambisonic and APAC sound.
      Known from this and left alone (the user's call): a QooCam 3 Ultra
      frame cuts the lens's image circle, and a source that is not marked
      as a crop repeats its rim past the recorded frame, so a stabilized
      half-equirect or disc of it shows stretched bands at the edge while
      the correction is large (about 15° in the sample's first second with
      the default settings). None where the camera is steady, none
      unstabilized, none in a lens frame, which is black there.

27. **2.5.5 release (2026-11-07, Mac) — PUBLISHED: tag `v2.5.5` @ `90f11b7`
   (`gh release edit v2.5.5 --draft=false --latest`; assets = DMG + mac app
   zip + darwin-only `latest.json`, live at `releases/latest/download/latest.json`
   = 2.5.5 immediately). The release note was cut from 950 to ~400 words
   before publishing (keep them at the 2.5.1 length; the note is embedded in
   `latest.json`, so an edited body means re-running `make-latest-json.mjs`
   and re-uploading with `--clobber`). Windows released the same day — see the end of this item.** Version carriers bumped in `4bd548d` (Cargo workspace + GUI
   crate, README, `installer/windows.iss` AppVer — the Windows box needs no
   bump commit); CHANGELOG `Unreleased` closed as `2.5.5 — 2026-11-07`;
   release-note draft `~/vr180-internal/RELEASE-2.5.5.md` =
   `release-staging/RELEASE_NOTES.md` = the release body. Cut with
   `scripts/release_mac.sh --notarize`: both notarizations Accepted first
   try, app + DMG stapled, `spctl` "Notarized Developer ID", 14 dylibs,
   bundle 2.5.5, mac zip minisign-verified against the updater.rs pubkey.
   **NEW: the Mac bundle ships `braw_helper`** — script step 2b rebuilds it
   against the installed SDK, copies it to Contents/MacOS, signs it with
   `helpers/braw/braw_helper.entitlements` (`disable-library-validation`),
   and `verify_bundle` checks it is newer than `helpers/braw/*`, carries the
   entitlement and prints `Usage:` from inside the bundle. Why: the hardened
   helper loads Blackmagic's framework from `/Applications/Blackmagic
   RAW/…/Libraries` (rpath), signed by another team, and dyld refuses it
   ("different Team IDs") without the entitlement — measured on a signed
   copy; with it `--info` on C006 runs through the Metal pipeline from
   inside the bundle, and Apple notarized it. The framework is not
   redistributable: users install the "Blackmagic RAW" package (release
   note + CHANGELOG say so). Trap: `vr180-gui --version` / `--help` are not
   flags — they launch the GUI. **WINDOWS TODO (release box):** pull
   `4bd548d`+; build; put `braw_helper.exe` into the bundle folder and the
   portable zip (`windows.iss` `[Files]` now lists it with
   `skipifsourcedoesntexist`; it LoadLibrary's BlackmagicRawAPI.dll from the
   installed SDK and nvcuda.dll at runtime — no extra DLLs); ISCC →
   setup.exe; `gh release download v2.5.5 --pattern latest.json --clobber
   --dir release-staging` FIRST (merge hazard), `gh release view v2.5.5
   --json body -q .body > release-staging/RELEASE_NOTES.md`, `node
   scripts/make-latest-json.mjs --version 2.5.5 --notes-file
   release-staging/RELEASE_NOTES.md`, then `gh release upload v2.5.5
   --clobber` setup.exe + portable zip + merged `latest.json` (NOT the
   .minisig). gh addresses the draft by its tag name (view/upload/download
   all worked on the 2.5.2 and 2.5.5 drafts).
   **WINDOWS RELEASED (2026-11-07, Windows box, from `2ad13d3` + the
   no-console commit `e3955cd`):** built `vr180-gui.exe` 2.5.5 (file version
   2.5.5), bundle = exe + the SEVEN FFmpeg DLLs the exe imports (the import
   closure — `ffmpeg-8.dll` next to the exe is the CLI's own and is NOT
   shipped) + README.txt (rewritten: input list, update note, the Blackmagic
   RAW runtime requirement) + **`braw_helper.exe`** (rebuilt the same minute
   by `build_braw.ps1` with the CUDA blit; imports only system DLLs —
   static CRT — and LoadLibrary's BlackmagicRawAPI.dll / nvcuda.dll at run
   time). Smoke: the bundled exe launched from its folder loads its OWN
   DLLs (module list), opens C006 on the zero-copy d3d11 path with no
   console window. ISCC → setup.exe 16 939 579 B (sha256 cf73abe6…),
   portable zip 21 622 945 B (folder-rooted like 2.1.0). Manifest: the
   mac's `latest.json` synced first, `make-latest-json.mjs --version 2.5.5`
   signed + merged; NOTE `gh release view --json body -q .body` ends with
   a newline the mac's notes lack, so the merge was re-run with
   `--notes ""` (= keep `prev.notes`) to leave the mac's notes byte-
   identical; darwin entry byte-identical. Uploaded `--clobber`: setup.exe,
   zip, merged `latest.json` (no .minisig). VERIFIED live: the release's
   `latest.json` asset and setup.exe are sha-identical to local, the live
   installer verifies against the signature IN the live manifest with the
   updater.rs pubkey; the STABLE `releases/latest/download/latest.json`
   URL kept serving the mac's darwin-only manifest for ~6 min after the
   upload (CDN redirect cache) and then served the merged one byte-
   identically — check it twice, not once. Feed serves 2.5.5 to BOTH
   platforms.

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
