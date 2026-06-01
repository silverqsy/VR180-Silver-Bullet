//! Decoder worker thread.
//!
//! Owns a `StreamPairIter`, pulls frames sequentially, runs the
//! existing GPU pipeline (`nv12_to_eac_cross` → `project_*_to_equirect_texture`
//! → `compose_sbs_bgra`), and ships the final `wgpu::Texture` to the
//! UI thread over a bounded channel.
//!
//! Wall-clock pacing + back-pressure mean playback feels real-time:
//! - If we're behind wall-clock, **skip the GPU render** for that
//!   frame (the decoder still advances — HEVC needs sequential
//!   reference frames).
//! - If we're ahead, `thread::sleep` to pace.
//! - The frame channel is bounded (2). If the UI hasn't consumed
//!   the previous frame yet, `try_send` returns Full and we drop.
//!
//! We use the **CPU-assemble** decode path for portability (no
//! IOSurface code needed here). Decode happens in software/HW via
//! VideoToolbox on macOS; EAC assembly runs CPU-side; everything
//! from `nv12_to_eac_cross` onward stays on the GPU.

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use vr180_pipeline::gpu::{
    Device, EquirectRotation, EquirectRsParams,
};
#[allow(unused_imports)]
use vr180_pipeline::gpu::ColorStackPlan;

#[cfg(target_os = "macos")]
use vr180_pipeline::{decode::ZeroCopyStreamPairIter, gpu::Lens};
#[cfg(not(target_os = "macos"))]
use {
    vr180_core::eac::{assemble_lens_a, assemble_lens_b},
    vr180_pipeline::decode::{iter_stream_pairs, HwDecode},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub stabilize: bool,
    pub cori_source: CoriSource,
    pub smooth_ms: f32,
    pub max_corr_deg: f32,
    pub rs_correct: bool,
    pub rs_mode: RsMode,
    pub rs_readout_ms: f32,
    pub preview_eye_w: u32,
    /// Trim-in time in seconds. `None` = play from clip start.
    /// Honored by the decoder loop: when playback hits `trim_out_s`,
    /// it seeks back to `trim_in_s` (or 0 if `None`) and continues.
    pub trim_in_s: Option<f64>,
    /// Trim-out time in seconds. `None` = play to clip end.
    pub trim_out_s: Option<f64>,

    // ─── Fisheye-source settings (DJI OSV / SBS / BRAW) ──────────
    //
    // Ignored when the loaded file is GoPro EAC. The fisheye decoder
    // reads these once on settings_generation bump.

    /// Selected camera preset by name. Empty / "Auto" → pick based on
    /// file extension (`.osv` → DJI Osmo 360, `.braw` → Pyxis 12K).
    /// Any other value must match a preset in `vr180_fisheye::presets`.
    pub fisheye_preset: String,
    /// Full FOV override in degrees. `0.0` → use preset default. The
    /// fisheye decoder converts to fx via `image_w / (2 · half_fov)`.
    pub fisheye_fov_deg: f32,
    /// KB-4 distortion coefficients override. `[0,0,0,0]` → use the
    /// preset's k. Otherwise these go straight into `FisheyeCalib`.
    pub fisheye_k: [f32; 4],
    /// Principal-point offset from image center in pixels (working
    /// resolution). `(0,0)` means cx/cy = src_w/2, src_h/2.
    pub fisheye_cx_offset_px: f32,
    pub fisheye_cy_offset_px: f32,
    /// For DJI OSV / dual-stream sources: swap L↔R after decode. The
    /// Python app exposes this as `swap_eyes` at vr180_gui.py:3554.
    pub fisheye_swap_eyes: bool,
    /// Output projection. `HalfEquirect` is the standard VR180 output
    /// (left-right hemisphere). `Fisheye` stabilizes the source
    /// without un-warping (Task #10 pending).
    pub fisheye_output_mode: FisheyeOutputMode,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stabilize: false,
            cori_source: CoriSource::Auto,
            smooth_ms: 1000.0,
            max_corr_deg: 15.0,
            rs_correct: false,
            rs_mode: RsMode::Firmware,
            rs_readout_ms: 15.224,
            preview_eye_w: 768,
            trim_in_s: None,
            trim_out_s: None,
            fisheye_preset: String::new(),
            fisheye_fov_deg: 0.0,
            fisheye_k: [0.0, 0.0, 0.0, 0.0],
            fisheye_cx_offset_px: 0.0,
            fisheye_cy_offset_px: 0.0,
            fisheye_swap_eyes: false,
            fisheye_output_mode: FisheyeOutputMode::HalfEquirect,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FisheyeOutputMode {
    /// Half-equirect VR180 output (default).
    HalfEquirect,
    /// Stabilized fisheye output, no projection change.
    /// Currently falls back to HalfEquirect (Task #10 pending).
    Fisheye,
}

impl FisheyeOutputMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HalfEquirect => "half-equirect",
            Self::Fisheye      => "fisheye (stab only)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoriSource { Direct, Vqf, Auto }
impl CoriSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Vqf    => "vqf",
            Self::Auto   => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsMode { Firmware, NoFirmware }
impl RsMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Firmware   => "firmware",
            Self::NoFirmware => "no-firmware",
        }
    }
}

pub struct DecoderConfig {
    pub path: PathBuf,
    pub settings: Settings,
    pub eye_w: u32,
}

/// Commands the UI sends to the decoder.
pub enum DecoderCommand {
    Stop,
    /// Jump playback to `target_s` seconds. The decoder seeks the
    /// underlying ffmpeg context to the nearest keyframe ≤ target,
    /// flushes the decoder, then resumes decoding (and frame-drops
    /// any leading frames before `target_s` so the next output is
    /// at-or-after the requested time).
    Seek(f64),
}

/// Shared control surface between the UI and a running decoder thread.
///
/// - `paused`: when true, the decoder loop parks instead of producing
///   frames. Flipping it back to false resumes playback from wherever
///   the decoder paused — no seek, no restart from frame 0.
/// - `settings`: the live stab/RS/source knobs. The UI mutates this
///   in place every time the user nudges a slider; the decoder reads
///   it once per iteration via `settings_generation`.
/// - `settings_generation`: bumped on every settings change. The
///   decoder caches the generation it last computed for; if the
///   counter has moved, it rebuilds the per-eye bundle on the spot
///   (~ few hundred ms for a full-length clip) and keeps going.
#[derive(Default)]
pub struct DecoderControl {
    pub paused: AtomicBool,
    pub settings: RwLock<Settings>,
    pub settings_generation: AtomicU64,
}

/// One frame ready for display. `texture` is a square BGRA-bytes
/// SBS texture (left eye on the left half, right eye on the right
/// half) sitting on the shared wgpu device — egui can register it
/// without copying.
pub struct DecodedFrame {
    pub texture: Arc<wgpu::Texture>,
    pub width: u32,
    pub height: u32,
    pub frame_idx: u32,
    pub timestamp_s: f64,
}

/// Entry point. Runs to completion (or until a Stop command arrives).
///
/// macOS: uses the zero-copy IOSurface path (`ZeroCopyStreamPairIter`
/// + `nv12_to_eac_cross` shader), so decode → EAC assembly →
/// projection → SBS compose stays on the GPU end-to-end. No host
/// memory transfer per frame. This is what makes the preview hit
/// 30 fps at GoPro Max source resolution.
///
/// Non-macOS: falls back to the CPU-assemble path (decode +
/// `av_hwframe_transfer_data` + swscale + CPU `assemble_lens_*` +
/// GPU project). Real-time on smaller clips; for full-resolution
/// 5952×1920 it'll lean on the frame-skip path. Phase 0.6.8 lands
/// the Windows CUDA↔Vulkan equivalent.
pub fn start_decoder(
    pipeline: Arc<Device>,
    cfg: DecoderConfig,
    control: Arc<DecoderControl>,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    tracing::info!("decoder: starting for {}", cfg.path.display());

    // Detect source kind. GoPro EAC stays on the existing
    // zero-copy / CPU-assemble path; fisheye sources (OSV, SBS,
    // BRAW) take a new path that uses the KB projection shader.
    let kind = vr180_pipeline::source_kind::detect(&cfg.path)
        .map_err(|e| { tracing::error!("decoder: source detect failed: {e}"); e })?;
    tracing::info!("decoder: source kind = {:?} ({})", kind, kind.display());

    let probe = vr180_pipeline::decode::probe_video(&cfg.path)
        .map_err(|e| { tracing::error!("decoder: probe failed: {e}"); e })?;
    let fps = probe.fps.max(1e-3);
    let dt = 1.0 / fps as f64;
    let total_frames = (probe.duration_sec as f64 * fps as f64).round() as usize;
    tracing::info!("decoder: probed {} × {} @ {:.2} fps, {} frames total",
        probe.width, probe.height, fps, total_frames);

    let eye_w = cfg.eye_w.clamp(256, 2048);
    let eye_h = eye_w;

    // Fisheye sources skip the GoPro CORI/IORI per-eye bundle (it's a
    // GoPro-specific format). They get identity rotation for the
    // first cut; BRAW VQF integration comes with Task #8.
    if kind.is_fisheye() {
        tracing::info!("decoder: routing to fisheye path");
        return run_fisheye(pipeline, &cfg, control, kind, fps, dt, eye_w, eye_h, frame_tx, cmd_rx);
    }

    let snapshot = control.settings.read().clone();
    let per_eye = build_per_eye_frames(&cfg.path, &snapshot, fps, total_frames)
        .map_err(|e| { tracing::error!("decoder: per-eye frames build failed: {e}"); e })?;
    let cached_gen = control.settings_generation.load(Ordering::SeqCst);
    tracing::info!("decoder: per-eye bundles ready ({} entries) gen={}, eye={}×{}",
        per_eye.len(), cached_gen, eye_w, eye_h);

    #[cfg(target_os = "macos")]
    return run_zero_copy(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, cached_gen, frame_tx, cmd_rx);
    #[cfg(not(target_os = "macos"))]
    return run_cpu_assemble(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, cached_gen, frame_tx, cmd_rx);
}

/// Fisheye source decoder loop (DJI OSV, SBS fisheye, Blackmagic BRAW).
///
/// Same wall-clock pacing / pause-resume / seek / trim semantics as the
/// GoPro paths above. Stabilization is identity for this first cut —
/// Task #8 will wire in BRAW VQF; DJI OSV stab needs a follow-up that
/// parses the embedded protobuf gyro stream.
fn run_fisheye(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    kind: vr180_pipeline::SourceKind,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    use vr180_pipeline::fisheye_decode::{
        FisheyePairIter, SbsFisheyeIter, DualStreamFisheyeIter, BrawFisheyeIter,
    };
    use vr180_pipeline::decode::HwDecode;
    use vr180_pipeline::gpu::FisheyeCalib;
    use vr180_fisheye::presets;

    // === Pipelined decode ===
    //
    // The iter is created and run inside a worker thread so decode and
    // render happen in parallel. On a 3840² dual-stream HEVC source,
    // per-frame VT decode is ~20 ms — same order as the GPU
    // project+compose budget. Serial execution sums them and blows
    // the per-frame budget at 50 fps. Pipelining drops the wall-clock
    // floor to max(decode, render).
    //
    // Channels:
    //   pair_tx/rx: bounded(2), pre-decoded pairs tagged with a
    //               generation counter so seeks can discard stale
    //               in-flight frames without restarting the thread.
    //   iter_cmd:   main → worker (seek)
    //   dims_tx/rx: one-shot, worker → main with eye_dims after init.
    enum IterCmd { Seek(f64) }
    // Buffer 8 decoded pairs ahead. At 50 fps source that's ~160 ms
    // of jitter absorption — enough to ride out GC/wgpu cleanup pauses
    // and the occasional slow-decode frame (saw 21 ms outliers vs the
    // 20 ms steady state in the 50 fps trace). Latency cost: a fresh
    // seek may take ~160 ms before the first new frame surfaces, which
    // is invisible for normal playback.
    let (pair_tx, pair_rx) =
        crossbeam_channel::bounded::<(u64, vr180_pipeline::fisheye_decode::FisheyePair)>(8);
    let (iter_cmd_tx, iter_cmd_rx) = crossbeam_channel::bounded::<IterCmd>(8);
    let (dims_tx, dims_rx) = crossbeam_channel::bounded::<(u32, u32)>(1);

    let path_for_worker = cfg.path.clone();
    let kind_for_worker = kind;
    let initial_swap_eyes = control.settings.read().fisheye_swap_eyes;
    let initial_trim_in = control.settings.read().trim_in_s;

    let _decode_handle = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut iter: Box<dyn FisheyePairIter> = match kind_for_worker {
            vr180_pipeline::SourceKind::DjiOsv => {
                let swap = !initial_swap_eyes; // XOR with DJI's "swap by default"
                let cap = vr180_pipeline::fisheye_decode::max_decode_side_for_fps(fps);
                tracing::info!("decoder (fisheye/osv): fps={:.2} → max_decode_side={}", fps, cap);
                Box::new(DualStreamFisheyeIter::new_with_options(
                    &path_for_worker, HwDecode::Auto, 0, swap, cap, 8,
                )?)
            }
            vr180_pipeline::SourceKind::SbsFisheye => Box::new(
                SbsFisheyeIter::new(&path_for_worker, HwDecode::Auto, 0)?
            ),
            vr180_pipeline::SourceKind::BlackmagicRaw => {
                let info = vr180_braw::BrawInfo::probe(&path_for_worker)
                    .map_err(|e| anyhow::anyhow!("braw probe: {e}"))?;
                let opts = vr180_braw::decoder::DecodeOptions::default();
                Box::new(
                    BrawFisheyeIter::new(&path_for_worker, &info, &opts, 0)
                        .map_err(|e| anyhow::anyhow!("braw start: {e}"))?
                )
            }
            _ => anyhow::bail!("decode-worker: non-fisheye source: {kind_for_worker:?}"),
        };
        if let Some(t_in) = initial_trim_in {
            if t_in > 0.001 { let _ = iter.seek(t_in); }
        }
        let (sw, sh) = iter.eye_dims();
        if dims_tx.send((sw, sh)).is_err() { return Ok(()); }

        let mut gen: u64 = 0;
        loop {
            // Drain pending commands before each decode.
            while let Ok(cmd) = iter_cmd_rx.try_recv() {
                match cmd {
                    IterCmd::Seek(t) => {
                        let _ = iter.seek(t);
                        gen = gen.wrapping_add(1);
                    }
                }
            }
            match iter.next_pair() {
                Ok(Some(pair)) => {
                    if pair_tx.send((gen, pair)).is_err() { break; }
                }
                Ok(None) => break,        // EOS
                Err(e) => { tracing::warn!("decode-worker: {e}"); break; }
            }
        }
        Ok(())
    });

    let (src_w, src_h) = match dims_rx.recv() {
        Ok(d) => d,
        Err(_) => anyhow::bail!("decode worker failed to start"),
    };
    tracing::info!("decoder (fisheye): eye dims = {}×{} (pipelined)", src_w, src_h);
    // Main thread tracks the latest seek generation. Pairs tagged
    // with an older generation are stale (decoded before the latest
    // seek took effect) and silently discarded.
    let mut expected_gen: u64 = 0;

    // ── Wire swap_eyes from settings if this is a dual-stream source.
    {
        let s = control.settings.read();
        let _ = &s;
        // (Dual-stream iter exposes swap_eyes as a public field, but we
        //  built `decoder` as a Box<dyn> trait object — can't downcast
        //  cleanly. The Python OSV path swaps based on settings; we
        //  defer wiring until we expose swap_eyes on the trait.)
    }

    // ── DJI OSV: extract the protobuf once at startup for per-eye
    //    calibration (lens_a cx/cy differs from lens_b). Cheap and we
    //    can reuse the same blob below for stabilization.
    let dji_osv_imu: Option<vr180_fisheye::DjiOsvImu> = if matches!(
        kind, vr180_pipeline::SourceKind::DjiOsv
    ) {
        match vr180_pipeline::decode::extract_dji_meta_stream(&cfg.path) {
            Ok(blob) => match vr180_fisheye::DjiOsvImu::parse(&blob) {
                Ok(imu) => {
                    tracing::info!(
                        "decoder (fisheye/osv): protobuf parsed — lens_a.fx={:?}, lens_a.cx={:?}, lens_b.fx={:?}, lens_b.cx={:?}",
                        imu.lens_a.fx, imu.lens_a.cx, imu.lens_b.fx, imu.lens_b.cx
                    );
                    // Factory per-lens extrinsics intentionally ignored:
                    // this is a VR180-modded camera with displaced
                    // lenses, so the protobuf rotation-vector offsets
                    // don't describe our optics. Intrinsics only.
                    Some(imu)
                }
                Err(e) => {
                    tracing::warn!("dji protobuf parse failed: {e} — using preset calib for both eyes");
                    None
                }
            },
            Err(e) => {
                tracing::warn!("dji metadata extract failed: {e} — using preset calib for both eyes");
                None
            }
        }
    } else {
        None
    };

    // ── Resolve initial calibrations from settings + preset + protobuf
    let mut cached_gen = control.settings_generation.load(Ordering::SeqCst);
    let (mut calib_left, mut calib_right) = resolve_fisheye_calib_pair(
        &control.settings.read(), kind, src_w, src_h, dji_osv_imu.as_ref(),
    );
    tracing::info!(
        "decoder (fisheye): initial calib L fx={:.1}, cx={:.1}, cy={:.1} | R fx={:.1}, cx={:.1}, cy={:.1}",
        calib_left.fx, calib_left.cx, calib_left.cy,
        calib_right.fx, calib_right.cx, calib_right.cy
    );
    let _ = presets::presets(); // anchor the import for downstream use

    // ── Source-specific stabilization (Tasks #8, #15) ───────────────
    // BRAW: VQF 6D on raw gyro+accel from braw_helper --gyro.
    // DJI OSV: direct integration of camera-supplied quats from the
    //   `djmd` protobuf stream (no VQF — camera already fused).
    // SBS fisheye: identity (Insta360 / Vuze / Canon parsers are
    //   per-format and not yet implemented).
    //
    // Compute is lazy + live: the toggle is checked at each iteration
    // of the main loop; first time stabilize flips on, the gyro is
    // extracted and the per-frame rotations are cached. Toggle off
    // drops the cache. This lets the user enable stab mid-playback
    // without Stop→Play.
    let mut stab_rotations: Option<Vec<EquirectRotation>> = None;
    let mut last_stabilize_state = false;
    let mut compute_stab = |dji_osv_imu: Option<&vr180_fisheye::DjiOsvImu>,
                            control: &DecoderControl,
                            fps: f32| -> Option<Vec<EquirectRotation>> {
        let total_frames = (vr180_pipeline::decode::probe_video(&cfg.path)
            .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
            .unwrap_or(0)).max(1);
        match kind {
            vr180_pipeline::SourceKind::BlackmagicRaw => {
                match vr180_braw::BrawGyroData::extract(&cfg.path) {
                    Ok(gyro_data) => match vr180_pipeline::braw_imu::compute_braw_stabilization(
                        &gyro_data, fps, total_frames
                    ) {
                        Ok(stab) => {
                            tracing::info!(
                                "decoder (fisheye/braw): stab ready, {} frames, bias={:?}°/s",
                                stab.per_frame.len(), stab.bias_deg_s
                            );
                            Some(stab.per_frame)
                        }
                        Err(e) => { tracing::warn!("braw stab failed: {e}"); None }
                    },
                    Err(e) => { tracing::warn!("braw gyro extract failed: {e}"); None }
                }
            }
            vr180_pipeline::SourceKind::DjiOsv => {
                if let Some(osv) = dji_osv_imu {
                    // OSV is locked to pure camera-lock: smooth_ms = 0
                    // (q_target = frame_quats[0]) AND no per-frame
                    // correction cap (max_corr_deg = ∞). The Max corr
                    // slider is hidden in the GUI for OSV — there's
                    // nothing it can usefully control in this mode.
                    let max_corr_deg = f32::INFINITY;
                    let smooth_ms = 0.0_f32;
                    match vr180_pipeline::dji_imu::compute_dji_stabilization(
                        osv, total_frames, max_corr_deg, smooth_ms, fps,
                    ) {
                        Ok(stab) => {
                            tracing::info!(
                                "decoder (fisheye/osv): stab ready, {} frames, {} with HR quats",
                                stab.per_frame.len(), stab.frames_with_hr_quat
                            );
                            Some(stab.per_frame)
                        }
                        Err(e) => { tracing::warn!("dji stab failed: {e}"); None }
                    }
                } else { None }
            }
            _ => None,
        }
    };
    if control.settings.read().stabilize {
        stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control, fps);
        last_stabilize_state = true;
    }
    tracing::info!(
        "decoder (fisheye): startup stab_rotations = {} (toggle is now LIVE)",
        if stab_rotations.is_some() { "Some(per-frame matrices)" } else { "None (identity)" }
    );

    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let _ = fps;

    // Initial trim_in is applied by the worker thread BEFORE it
    // starts sending pairs (see worker spawn above). Sync the
    // main-thread `time_offset` to match.
    if let Some(t_in) = initial_trim_in {
        if t_in > 0.001 {
            tracing::info!("decoder (fisheye): initial seek to trim_in = {:.3}s (in worker)", t_in);
            time_offset = t_in;
        }
    }

    // Decimate at >30 fps even with pipelining. Tested: 50 fps native
    // preview of 3840² dual-stream HEVC is borderline on M5 Max — VT
    // decode is ~20 ms per frame (matching the 50 fps budget exactly),
    // so any sustained slowdown drains the channel buffer and we fall
    // behind permanently. Empirically, smooth playback survives ~28 s
    // at 50 fps before slipping; 25 fps preview from a 50 fps source
    // ran 60+ s without a single drop. Trades temporal resolution for
    // stable playback. Export uses full source frames separately.
    let preview_decimation: u32 = if fps > 30.5 { 2 } else { 1 };
    let preview_fps = fps / preview_decimation as f32;
    let preview_dt = 1.0_f64 / preview_fps as f64;
    if preview_decimation > 1 {
        tracing::info!(
            "decoder (fisheye): fps={:.2} → preview decimation {}× → effective {:.2} fps",
            fps, preview_decimation, preview_fps,
        );
    }

    // Time the decode by measuring from the end of the previous iter
    // to the start of this one — `pair_rx.recv()` is what's between them.
    // Because decode runs on a worker thread now, this measures channel
    // *wait* time, not raw decode time. When decode is faster than
    // render we wait ~0 µs (worker keeps the buffer full).
    let mut last_iter_end = std::time::Instant::now();
    let mut decode_us: u128 = 0;
    // Pull a pair from the worker's channel, dropping any tagged
    // with stale generations from before the latest seek.
    let recv_next_pair = |rx: &crossbeam_channel::Receiver<(u64, vr180_pipeline::fisheye_decode::FisheyePair)>,
                          expected: u64|
        -> Option<vr180_pipeline::fisheye_decode::FisheyePair>
    {
        loop {
            match rx.recv() {
                Ok((g, p)) if g == expected => return Some(p),
                Ok(_) => continue,    // stale, drop
                Err(_) => return None,
            }
        }
    };
    'main: while let Some(pair) = {
        // Pull one pair to render; drop (preview_decimation-1) more
        // to thin the rendered framerate. Decode work happens on the
        // worker — these drops do NOT save decode budget on this thread.
        let mut p = recv_next_pair(&pair_rx, expected_gen);
        for _ in 1..preview_decimation {
            match recv_next_pair(&pair_rx, expected_gen) {
                Some(next) => p = Some(next),
                None => break,
            }
        }
        p
    } {
        decode_us = last_iter_end.elapsed().as_micros();
        // ── 0. Live stabilize toggle. When the user flips the
        //       checkbox during playback, recompute on this iteration.
        //       The compute takes a few hundred ms (a brief stutter)
        //       but the user doesn't have to Stop→Play any more.
        let now_stabilize = control.settings.read().stabilize;
        if now_stabilize != last_stabilize_state {
            if now_stabilize {
                tracing::info!("decoder (fisheye): stabilize ON → computing stab");
                stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control, fps);
            } else {
                tracing::info!("decoder (fisheye): stabilize OFF → dropping stab cache");
                stab_rotations = None;
            }
            last_stabilize_state = now_stabilize;
        }

        // ── 1. Drain commands.
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            let _ = iter_cmd_tx.send(IterCmd::Seek(t));
            expected_gen = expected_gen.wrapping_add(1);
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            continue 'main;
        }

        // ── 2. Trim-out loop.
        // `frame_idx` counts loop iterations (= rendered frames after
        // decimation), so pacing uses `preview_dt` not the source `dt`.
        // `frame_t_abs` is still in source-time for trim/seek lookups.
        let frame_t_rel = frame_idx as f64 * preview_dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    let _ = iter_cmd_tx.send(IterCmd::Seek(in_t));
                    expected_gen = expected_gen.wrapping_add(1);
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    continue 'main;
                }
            }
        }

        // ── 3. Pause handling.
        if control.paused.load(Ordering::SeqCst) && !force_render_next {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            let _ = iter_cmd_tx.send(IterCmd::Seek(t.max(0.0)));
                            expected_gen = expected_gen.wrapping_add(1);
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // ── 4. Settings changed → re-resolve per-eye calibs.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snap = control.settings.read();
            let (l, r) = resolve_fisheye_calib_pair(
                &snap, kind, src_w, src_h, dji_osv_imu.as_ref(),
            );
            calib_left = l;
            calib_right = r;
            drop(snap);
            cached_gen = current_gen;
            tracing::debug!(
                "decoder (fisheye): live calib update gen={}, L.fx={:.1}, R.fx={:.1}",
                current_gen, calib_left.fx, calib_right.fx
            );
        }

        // ── 5. Wall-clock pacing + skip when behind.
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if !force_render_next && wall_t > frame_t_rel + preview_dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel)
                    - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (fisheye): behind by {:.1} ms — skipped {} (decode={}µs)",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count, decode_us);
            }
            last_iter_end = std::time::Instant::now();
            continue;
        }

        // ── 6. GPU project each eye + compose SBS.
        // Look up the stabilization rotation by the ACTUAL frame's
        // PTS, not the loop counter — seeks land on keyframes which
        // may be before the requested timestamp, and the iterator
        // can buffer frames internally. PTS is the only thing
        // guaranteed to line up with the protobuf's frame-block
        // index (one frame_block per video frame, by construction).
        let phase_t0 = std::time::Instant::now();
        let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
            (pair.pts_s / dt).round() as usize
        } else {
            // Fallback if the iterator didn't supply PTS (e.g. BRAW —
            // we synthesise PTS from frame_idx there, but be defensive).
            let abs = ((time_offset / dt).round() as i64) + frame_idx as i64;
            abs.max(0) as usize
        };
        let absolute_frame_idx = stab_idx as u32;
        let rot = stab_rotations
            .as_ref()
            .and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);

        // Per-row RS matrices for this frame. Built lazily on CPU; ~46 KB
        // for src_h=1280 so the per-frame `queue.write_buffer` is cheap.
        // When present, fused per-row stab + RS gives us DJI Studio's
        // per-slab quality (matches their `getMatrixForEisAndHorizontal`
        // x slab x scanline approach) without leaving the live pipeline.
        let rs_quats: Option<Vec<vr180_core::gyro::cori_iori::Quat>> =
            if control.settings.read().stabilize {
                dji_osv_imu.as_ref().and_then(|osv| {
                    vr180_pipeline::dji_imu::compute_per_row_quaternions_for_frame(
                        osv,
                        stab_idx,
                        // FPS-aware readout: OSMO 360 sensor mode at 50fps
                        // gives 16.23 ms vs 18.3 ms at 30fps.
                        vr180_pipeline::dji_imu::dji_osmo_readout_ms_for_fps(fps) / 1000.0,
                        src_h,
                        fps,
                    )
                })
            } else {
                None
            };
        let phase_rs_quats = phase_t0.elapsed();
        let phase_t1 = std::time::Instant::now();
        // Per-clip lens_a (factory mount) feeds into the per-row
        // basis change. Same fallback as compute_dji_stabilization.
        let lens_a_for_pack: [f32; 4] = dji_osv_imu
            .as_ref()
            .and_then(|osv| osv.lens_a.mount_quat_xyzw)
            .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
        let rs_rows_f32: Option<Vec<f32>> = rs_quats.as_ref()
            .map(|q| vr180_pipeline::dji_imu::pack_per_row_camera_matrices(q, lens_a_for_pack));
        let phase_pack = phase_t1.elapsed();

        // Diagnostic: per-axis decomposition of the stab rotation so we
        // can see if specific axes (roll, pitch, yaw) are under-
        // corrected. Roll = rotation around camera Z (optical axis);
        // pitch around X; yaw around Y. Decomposition uses Tait-Bryan
        // angles (ZYX intrinsic) computed from the row-major matrix.
        if frame_idx % 10 == 0 {
            if let Some(rq) = rs_quats.as_ref() {
                let to_deg = |q: &vr180_core::gyro::cori_iori::Quat| {
                    2.0 * q.w.abs().min(1.0).acos().to_degrees()
                };
                let intra_d = (to_deg(&rq[0]) - to_deg(&rq[rq.len() - 1])).abs();
                let tr = rot.0[0] + rot.0[4] + rot.0[8];
                let cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
                let stab_mag = cos_theta.acos().to_degrees();
                // YXZ-intrinsic Tait-Bryan for camera convention
                // (X=right, Y=up, Z=forward): yaw is around Y, pitch
                // around X, roll around Z. Indexing the row-major
                // mat3 stored as `[r00 r01 r02; r10 r11 r12; r20 r21 r22]`.
                let r00 = rot.0[0]; let r10 = rot.0[3];
                let r20 = rot.0[6]; let r21 = rot.0[7]; let r22 = rot.0[8];
                let yaw   = (-r20).atan2(r00).to_degrees();
                let pitch = r21.atan2(r22).to_degrees();
                let roll  = (-r10).asin().to_degrees();
                // Note on the previous diagnostic: it mislabeled the
                // axes (the math we feed to the shader was — and is —
                // correct, the log just printed wrong column headers).
                // Pure camera pitch (tilt) used to print as "roll"; pure
                // camera roll printed as "pitch"; pure camera yaw printed
                // negated under "yaw". Confirmed against a +130° tilt
                // motion that came through as `roll=-129.96°` in the
                // old log.
                tracing::info!(
                    "f={:>4} stab={:>6.2}° (yaw={:+6.2} pitch={:+6.2} roll={:+6.2}) intra-Δ={:.3}°",
                    frame_idx, stab_mag, yaw, pitch, roll, intra_d
                );
            }
        }

        let phase_t2 = std::time::Instant::now();
        let (left_tex, right_tex) = if let Some(rs_buf) = rs_rows_f32.as_deref() {
            let l = pipeline.project_fisheye_to_equirect_rs_texture(
                &pair.left, src_w, src_h, eye_w, eye_h, rot, calib_left, rs_buf, 0,
            )?;
            let r = pipeline.project_fisheye_to_equirect_rs_texture(
                &pair.right, src_w, src_h, eye_w, eye_h, rot, calib_right, rs_buf, 1,
            )?;
            (l, r)
        } else {
            let l = pipeline.project_fisheye_to_equirect_texture(
                &pair.left, src_w, src_h, eye_w, eye_h, rot, calib_left, 0,
            )?;
            let r = pipeline.project_fisheye_to_equirect_texture(
                &pair.right, src_w, src_h, eye_w, eye_h, rot, calib_right, 1,
            )?;
            (l, r)
        };
        let phase_project = phase_t2.elapsed();
        let phase_t3 = std::time::Instant::now();
        let sbs_tex = compose_sbs(&pipeline, &left_tex, &right_tex, eye_w, eye_h)?;
        let phase_compose = phase_t3.elapsed();
        // Per-frame phase timing. First 20 rendered frames log every
        // time so we see warm-cache numbers; after that, every 30.
        // `decode` is the time `decoder.next_pair()` blocked (i.e.,
        // VT decode + sws downscale).
        let should_log = frame_idx < 20 || frame_idx % 30 == 0;
        if should_log {
            tracing::info!(
                "perf f={:>4} decode={:>5}µs rs_quats={:>5}µs pack={:>5}µs project={:>5}µs compose={:>5}µs  render={:>5}µs  budget@{:.0}fps={:.0}µs (dec={}x)",
                frame_idx,
                decode_us,
                phase_rs_quats.as_micros(),
                phase_pack.as_micros(),
                phase_project.as_micros(),
                phase_compose.as_micros(),
                phase_t0.elapsed().as_micros(),
                preview_fps,
                1_000_000.0 / preview_fps,
                preview_decimation,
            );
        }

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: eye_w * 2,
            height: eye_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;
        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
        let _ = _check_pair_dims(&pair, src_w, src_h);
        last_iter_end = std::time::Instant::now();
    }
    Ok(())
}

fn _check_pair_dims(
    pair: &vr180_pipeline::fisheye_decode::FisheyePair,
    src_w: u32,
    src_h: u32,
) -> bool {
    // Debug-only sanity check — Python would have crashed loudly if
    // the iterator returned the wrong size; Rust just logs it.
    if pair.eye_w != src_w || pair.eye_h != src_h {
        tracing::warn!("decoder (fisheye): pair dims drift ({},{}) vs ({},{})",
            pair.eye_w, pair.eye_h, src_w, src_h);
        false
    } else {
        true
    }
}

/// Build per-eye `FisheyeCalib`s for the current settings + source +
/// working resolution. Called once at decoder start, and again on
/// every `settings_generation` bump.
///
/// Returns `(calib_left, calib_right)`. For non-DJI sources both are
/// identical. For DJI OSV, when a parsed protobuf is supplied, each
/// eye gets its own cx/cy from the per-lens protobuf entry — lens A
/// and lens B differ by tens of pixels in practice. fx and k still
/// come from the preset / settings (the protobuf's k coefficients
/// are unreliable past ~88° per the Python source).
///
/// Eye→lens mapping: DJI's stream-to-lens layout is stream 0 = Lens A
/// = right eye, stream 1 = Lens B = left eye, AND we swap_eyes at
/// the iter (see DualStreamFisheyeIter::new_with_swap). So after the
/// swap: left output = Lens B, right output = Lens A.
fn resolve_fisheye_calib_pair(
    s: &Settings,
    kind: vr180_pipeline::SourceKind,
    src_w: u32,
    src_h: u32,
    osv: Option<&vr180_fisheye::DjiOsvImu>,
) -> (vr180_pipeline::gpu::FisheyeCalib, vr180_pipeline::gpu::FisheyeCalib) {
    use vr180_fisheye::presets;

    // Pick a preset. Explicit name wins; else fall back to the
    // source-kind default.
    let preset = if !s.fisheye_preset.is_empty() && s.fisheye_preset != "Auto" {
        presets::find(&s.fisheye_preset)
    } else {
        None
    }
    .unwrap_or_else(|| {
        let auto_name = match kind {
            vr180_pipeline::SourceKind::DjiOsv         => "DJI Osmo 360",
            vr180_pipeline::SourceKind::BlackmagicRaw  => "Blackmagic Pyxis 12K",
            _                                          => "Custom",
        };
        presets::find(auto_name).unwrap_or(&presets::presets()[7])
    });

    // fx fallback when the protobuf doesn't supply fx (or for non-OSV
    // sources). FOV override always wins; otherwise prefer preset; else
    // derive from preset's default FOV.
    let fx_fallback = if s.fisheye_fov_deg > 0.0 {
        let half = (s.fisheye_fov_deg.to_radians()) * 0.5;
        (src_w as f32) / (2.0 * half)
    } else if preset.calib.fx > 0.0 && preset.calib.calib_w > 0 {
        (preset.calib.fx as f32) * (src_w as f32) / (preset.calib.calib_w as f32)
    } else {
        let half = (preset.default_fov_deg as f32).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };

    // KB distortion coefficients: settings override the preset only
    // when at least one settings slot is non-zero.
    let k_override = s.fisheye_k.iter().any(|c| c.abs() > 1e-9);
    let k = if k_override {
        s.fisheye_k
    } else {
        [preset.calib.k[0] as f32, preset.calib.k[1] as f32,
         preset.calib.k[2] as f32, preset.calib.k[3] as f32]
    };

    // Default cx/cy = image center + user offset (applied to both eyes).
    let default_cx = src_w as f32 * 0.5 + s.fisheye_cx_offset_px;
    let default_cy = src_h as f32 * 0.5 + s.fisheye_cy_offset_px;

    // For OSV: per-lens protobuf fx/fy/cx/cy + pure-KB projection
    // (no Hermite extension) so the Rust shader matches the Python
    // MLX kernel byte-for-byte. After the DJI iter's default swap:
    // left = Lens B, right = Lens A.
    match (kind, osv) {
        (vr180_pipeline::SourceKind::DjiOsv, Some(imu)) => {
            let scale_x = imu.lens_b.width.map(|w| (src_w as f32) / w).unwrap_or(1.0);
            let scale_y = imu.lens_b.height.map(|h| (src_h as f32) / h).unwrap_or(1.0);
            let cx_l = imu.lens_b.cx.map(|v| v * scale_x).unwrap_or(default_cx)
                + s.fisheye_cx_offset_px;
            let cy_l = imu.lens_b.cy.map(|v| v * scale_y).unwrap_or(default_cy)
                + s.fisheye_cy_offset_px;
            let cx_r = imu.lens_a.cx.map(|v| v * scale_x).unwrap_or(default_cx)
                + s.fisheye_cx_offset_px;
            let cy_r = imu.lens_a.cy.map(|v| v * scale_y).unwrap_or(default_cy)
                + s.fisheye_cy_offset_px;
            let fx_l = imu.lens_b.fx.map(|v| v * scale_x).unwrap_or(fx_fallback);
            let fy_l = imu.lens_b.fy.map(|v| v * scale_y).unwrap_or(fx_l);
            let fx_r = imu.lens_a.fx.map(|v| v * scale_x).unwrap_or(fx_fallback);
            let fy_r = imu.lens_a.fy.map(|v| v * scale_y).unwrap_or(fx_r);
            // Output stays at 180° VR180. Source-side theta_max comes
            // from `max_r / fx ≈ 104°` (set inside `new_pure_kb`), so
            // stab rotations can sample the lens periphery without
            // writing black at the equator.
            (
                vr180_pipeline::gpu::FisheyeCalib::new_pure_kb(
                    fx_l, fy_l, cx_l, cy_l, k, src_w as f32, src_h as f32,
                ),
                vr180_pipeline::gpu::FisheyeCalib::new_pure_kb(
                    fx_r, fy_r, cx_r, cy_r, k, src_w as f32, src_h as f32,
                ),
            )
        }
        _ => {
            // Non-OSV: keep the previous Hermite-default constructor.
            let r_max = (src_w.min(src_h) as f32) * 0.5;
            let mk = |cx, cy| vr180_pipeline::gpu::FisheyeCalib::new(
                fx_fallback, fx_fallback, cx, cy, k,
                src_w as f32, src_h as f32, r_max,
            );
            (mk(default_cx, default_cy), mk(default_cx, default_cy))
        }
    }
}

/// macOS zero-copy path. Decode → IOSurface → wgpu::Texture → GPU
/// EAC assembly → GPU equirect projection → GPU SBS compose. No CPU
/// readback, no swscale, no IPC.
#[cfg(target_os = "macos")]
fn run_zero_copy(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut per_eye: Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>,
    mut cached_gen: u64,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    let mut decoder = ZeroCopyStreamPairIter::new(&cfg.path, 0)
        .map_err(|e| { tracing::error!("decoder (zc): iter init failed: {e}"); e })?;
    let dims = decoder.dims();
    tracing::info!("decoder (zc): iter ready, EAC dims tile_w={}, cross_w={}",
        dims.tile_w(), dims.cross_w());
    if !dims.is_valid() {
        anyhow::bail!("invalid EAC layout (stream w={})", dims.stream_w);
    }

    // segment-relative frame index (resets to 0 on Seek / trim loop)
    let mut frame_idx: u32 = 0;
    // clip-absolute time at frame_idx = 0 (updated on Seek / trim loop)
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut rendered_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    // After any seek (Seek command OR trim-out loop), this flag is
    // set so the next iteration bypasses the pause check exactly
    // once and renders a fresh frame at the new position. Otherwise
    // paused + click-on-timeline would seek silently — the displayed
    // frame stays at the pre-seek timestamp until the user unpaused.
    let mut force_render_next = false;
    let _ = fps;

    let total_frames = (cfg.path.exists()
        .then(|| vr180_pipeline::decode::probe_video(&cfg.path).ok())
        .flatten().map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)) as usize;

    // Initial seek to trim_in, if set.
    {
        let s = control.settings.read();
        if let Some(t_in) = s.trim_in_s {
            if t_in > 0.001 {
                tracing::info!("decoder (zc): initial seek to trim_in = {:.3}s", t_in);
                drop(s);
                decoder.seek(t_in)?;
                time_offset = t_in;
            }
        }
    }

    'main: while let Some(pair) = decoder.next_pair(&pipeline.device)? {
        // ── 1. Drain command queue. Seek wins over Stop within
        //       a single batch; subsequent Seeks overwrite earlier ones.
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            drop(pair);
            decoder.seek(t)?;
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            tracing::debug!("decoder (zc): seek → {:.3}s", t);
            continue 'main;
        }

        // ── 2. Trim-out loop. If we've played past trim_out, jump
        //       back to trim_in (or 0). Uses absolute clip time, not
        //       segment-relative.
        let frame_t_rel = frame_idx as f64 * dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    drop(pair);
                    decoder.seek(in_t)?;
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    tracing::debug!("decoder (zc): trim loop → {:.3}s", in_t);
                    continue 'main;
                }
            }
        }

        // ── 3. Pause handling. `force_render_next` lets a freshly
        //       seeked frame be rendered before we go back to parking.
        if control.paused.load(Ordering::SeqCst) && !force_render_next {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            drop(pair);
                            decoder.seek(t)?;
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // ── 4. Settings changed → rebuild per-eye bundle.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snapshot = control.settings.read().clone();
            match build_per_eye_frames(&cfg.path, &snapshot, fps, total_frames) {
                Ok(new) => {
                    per_eye = new;
                    cached_gen = current_gen;
                    tracing::debug!("decoder (zc): per-eye bundles rebuilt @ gen {}", current_gen);
                }
                Err(e) => tracing::warn!("decoder (zc): per-eye rebuild failed: {e}"),
            }
        }

        // ── 5. Wall-clock pacing + behind-real-time skip.
        //       Bypass the skip path when force_render_next so a
        //       post-seek-while-paused frame is always rendered.
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();

        if !force_render_next && wall_t > frame_t_rel + dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (zc): behind by {:.1} ms — skipped {}",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count);
            }
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel)
                    - paused_offset);
            }
            drop(pair);
            continue;
        }

        // ── 6. GPU render.
        // We look up per-eye stab/RS by clip-absolute frame index so
        // that the rotation matrices stay tied to source time after
        // a seek (not segment-relative time).
        let absolute_frame_idx = ((time_offset / dt).round() as u32) + frame_idx;

        let cross_a = pipeline.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::A, dims,
        )?;
        let cross_b = pipeline.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::B, dims,
        )?;

        let ((rl, sl), (rr, sr)) = per_eye.get(absolute_frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));

        let left_tex = pipeline.project_cross_texture_to_equirect_texture(
            &cross_b, eye_w, eye_h, rl, sl,
        )?;
        let right_tex = pipeline.project_cross_texture_to_equirect_texture(
            &cross_a, eye_w, eye_h, rr, sr,
        )?;
        let sbs_tex = compose_sbs(&pipeline, &left_tex, &right_tex, eye_w, eye_h)?;

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: eye_w * 2,
            height: eye_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {
                rendered_count += 1;
                if rendered_count == 1 {
                    tracing::info!("decoder (zc): first frame rendered + sent");
                } else if rendered_count % 30 == 0 {
                    tracing::debug!("decoder (zc): rendered {} frames", rendered_count);
                }
            }
            Err(TrySendError::Full(_)) => { /* UI behind; drop newest */ }
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        drop(pair);
        // We rendered + shipped one frame; clear the post-seek
        // override so subsequent iterations honor pause/pacing again.
        force_render_next = false;

        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
    }
    tracing::info!("decoder (zc): finished after {} segment frames ({} skipped)",
        frame_idx, skipped_count);
    Ok(())
}

/// Non-macOS fallback — CPU EAC assembly path.
#[cfg(not(target_os = "macos"))]
fn run_cpu_assemble(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut per_eye: Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>,
    mut cached_gen: u64,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    let mut decoder = iter_stream_pairs(&cfg.path, HwDecode::Auto, 0)?;
    let dims = decoder.dims();
    if !dims.is_valid() {
        anyhow::bail!("invalid EAC layout (stream w={})", dims.stream_w);
    }
    let cross_w_px = dims.cross_w();
    let cw = cross_w_px as usize;

    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];

    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let _ = fps;
    let total_frames = (vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)) as usize;

    // Initial seek to trim_in, if set.
    {
        let s = control.settings.read();
        if let Some(t_in) = s.trim_in_s {
            if t_in > 0.001 {
                drop(s);
                decoder.seek(t_in)?;
                time_offset = t_in;
            }
        }
    }

    'main: while let Some(pair) = decoder.next_pair()? {
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            decoder.seek(t)?;
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            continue 'main;
        }

        let frame_t_rel = frame_idx as f64 * dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    decoder.seek(in_t)?;
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    continue 'main;
                }
            }
        }

        if control.paused.load(Ordering::SeqCst) && !force_render_next {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            decoder.seek(t)?;
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snapshot = control.settings.read().clone();
            match build_per_eye_frames(&cfg.path, &snapshot, fps, total_frames) {
                Ok(new) => { per_eye = new; cached_gen = current_gen; }
                Err(e) => tracing::warn!("decoder (cpu): per-eye rebuild failed: {e}"),
            }
        }

        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();

        if !force_render_next && wall_t > frame_t_rel + dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel)
                    - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (cpu): behind by {:.1} ms — skipped {}",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count);
            }
            continue;
        }

        let absolute_frame_idx = ((time_offset / dt).round() as u32) + frame_idx;

        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        let ((rl, sl), (rr, sr)) = per_eye.get(absolute_frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));

        let left_tex = pipeline.project_cross_to_equirect_texture(
            &cross_b, cross_w_px, eye_w, eye_h, rl, sl,
        )?;
        let right_tex = pipeline.project_cross_to_equirect_texture(
            &cross_a, cross_w_px, eye_w, eye_h, rr, sr,
        )?;
        let sbs_tex = compose_sbs(&pipeline, &left_tex, &right_tex, eye_w, eye_h)?;

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: eye_w * 2,
            height: eye_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;
        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
    }
    Ok(())
}

/// Build a one-shot stitch: left + right equirect → 2×W SBS, in
/// `Rgba8Unorm` format (egui-wgpu expects this format and applies
/// the swapchain's gamma curve correctly for it).
fn compose_sbs(
    pipeline: &Device,
    left: &wgpu::Texture,
    right: &wgpu::Texture,
    eye_w: u32,
    eye_h: u32,
) -> anyhow::Result<wgpu::Texture> {
    let out_w = eye_w * 2;
    let sbs = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sbs_preview"),
        size: wgpu::Extent3d { width: out_w, height: eye_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
             | wgpu::TextureUsages::TEXTURE_BINDING
             | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let mut encoder = pipeline.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sbs_compose_enc"),
    });
    // Copy left → (0, 0)
    encoder.copy_texture_to_texture(
        wgpu::ImageCopyTexture {
            texture: left, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyTexture {
            texture: &sbs, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
    );
    // Copy right → (eye_w, 0)
    encoder.copy_texture_to_texture(
        wgpu::ImageCopyTexture {
            texture: right, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyTexture {
            texture: &sbs, mip_level: 0,
            origin: wgpu::Origin3d { x: eye_w, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
    );
    pipeline.queue.submit(Some(encoder.finish()));
    Ok(sbs)
}

/// Build the per-eye rotation + RS params vec (one tuple per video
/// frame). Empty when neither stabilization nor RS is on.
fn build_per_eye_frames(
    input: &std::path::Path,
    s: &Settings,
    fps: f32,
    n_frames_video: usize,
) -> anyhow::Result<Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>> {
    use vr180_core::gyro::{
        parse_cori, parse_iori, parse_raw_imu, Quat,
        bidirectional_smooth, per_eye_rotations,
        gravity_alignment_quat, apply_gravity_alignment_inplace,
        compute_per_frame_omega, SmoothParams, SMOOTH_WINDOW_S,
    };
    use vr180_pipeline::decode::extract_gpmf_stream;

    if !s.stabilize && !s.rs_correct {
        return Ok(Vec::new());
    }

    let gpmf = extract_gpmf_stream(input)?;
    let mut cori = parse_cori(&gpmf);
    let iori = parse_iori(&gpmf);
    let raw_imu = parse_raw_imu(&gpmf);

    let mut used_vqf = false;
    if s.stabilize {
        let want_vqf = match s.cori_source {
            CoriSource::Direct => false,
            CoriSource::Vqf    => true,
            CoriSource::Auto   => cori.first()
                .map(|q| (q.x*q.x + q.y*q.y + q.z*q.z).sqrt() < 1e-3)
                .unwrap_or(false),
        };
        if want_vqf {
            cori = vr180_pipeline::imu::vqf_cori_equivalent_stream(
                input, fps, n_frames_video,
            )?;
            if let Some(ref0) = cori.first().copied() {
                let inv = ref0.conjugate();
                for q in cori.iter_mut() { *q = q.mul(inv); }
            }
            used_vqf = true;
        }
    }

    let rotations: Vec<(EquirectRotation, EquirectRotation)> =
        if s.stabilize && !cori.is_empty() {
            if !used_vqf {
                if let Some(g) = raw_imu.grav.first() {
                    let gq = gravity_alignment_quat(&g.samples, g.scal, 10);
                    apply_gravity_alignment_inplace(&mut cori, gq.conjugate());
                }
            }
            let smoothed = bidirectional_smooth(&cori, fps, &SmoothParams {
                smooth_ms: s.smooth_ms, ..Default::default()
            });
            (0..cori.len()).map(|i| {
                let (l, r) = per_eye_rotations(
                    cori[i], smoothed[i],
                    iori.get(i).copied().unwrap_or(Quat::IDENTITY),
                    s.max_corr_deg,
                );
                (EquirectRotation::from_quat(l), EquirectRotation::from_quat(r))
            }).collect()
        } else {
            Vec::new()
        };

    let rs_eyes: Vec<(EquirectRsParams, EquirectRsParams)> = if s.rs_correct {
        let geoc = vr180_core::geoc::parse_geoc(input)?
            .ok_or_else(|| anyhow::anyhow!("GEOC block missing"))?;
        let front = geoc.front.as_ref().ok_or_else(|| anyhow::anyhow!("GEOC FRNT missing"))?;
        let back  = geoc.back.as_ref().ok_or_else(|| anyhow::anyhow!("GEOC BACK missing"))?;
        let srot_s = s.rs_readout_ms / 1000.0;
        let probe = vr180_pipeline::decode::probe_video(input)?;
        let omegas = compute_per_frame_omega(
            &raw_imu.gyro, n_frames_video, fps,
            probe.duration_sec as f32, srot_s * 0.5, SMOOTH_WINDOW_S,
        );
        let (left_pf, right_pf) = match s.rs_mode {
            RsMode::NoFirmware => ((1.0_f32, 1.0, 0.0), (1.0, 1.0, 0.0)),
            RsMode::Firmware   => ((0.0_f32, 0.0, 0.0), (2.5, 2.5, 0.0)),
        };
        let mk = |om: [f32; 3], pf: (f32, f32, f32), cal: &vr180_core::geoc::LensCal| {
            let active = pf.0 != 0.0 || pf.1 != 0.0 || pf.2 != 0.0;
            EquirectRsParams {
                omega: [om[0] * pf.0, om[1] * pf.2, om[2] * pf.1],
                srot_s: if active { srot_s } else { 0.0 },
                klns: [
                    cal.klns[0] as f32, cal.klns[1] as f32, cal.klns[2] as f32,
                    cal.klns[3] as f32, cal.klns[4] as f32,
                ],
                ctry: cal.ctry,
                cal_dim: geoc.cal_dim as f32,
            }
        };
        (0..n_frames_video).map(|i| {
            (mk(omegas[i], left_pf, back), mk(omegas[i], right_pf, front))
        }).collect()
    } else {
        Vec::new()
    };

    let n = rotations.len().max(rs_eyes.len());
    Ok((0..n).map(|i| {
        let (rl, rr) = rotations.get(i).copied()
            .unwrap_or((EquirectRotation::IDENTITY, EquirectRotation::IDENTITY));
        let (sl, sr) = rs_eyes.get(i).copied()
            .unwrap_or((EquirectRsParams::DISABLED, EquirectRsParams::DISABLED));
        ((rl, sl), (rr, sr))
    }).collect())
}

