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
    let probe = vr180_pipeline::decode::probe_video(&cfg.path)
        .map_err(|e| { tracing::error!("decoder: probe failed: {e}"); e })?;
    let fps = probe.fps.max(1e-3);
    let dt = 1.0 / fps as f64;
    let total_frames = (probe.duration_sec as f64 * fps as f64).round() as usize;
    tracing::info!("decoder: probed {} × {} @ {:.2} fps, {} frames total",
        probe.width, probe.height, fps, total_frames);

    let snapshot = control.settings.read().clone();
    let per_eye = build_per_eye_frames(&cfg.path, &snapshot, fps, total_frames)
        .map_err(|e| { tracing::error!("decoder: per-eye frames build failed: {e}"); e })?;
    let cached_gen = control.settings_generation.load(Ordering::SeqCst);
    let eye_w = cfg.eye_w.clamp(256, 2048);
    let eye_h = eye_w;
    tracing::info!("decoder: per-eye bundles ready ({} entries) gen={}, eye={}×{}",
        per_eye.len(), cached_gen, eye_w, eye_h);

    #[cfg(target_os = "macos")]
    return run_zero_copy(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, cached_gen, frame_tx, cmd_rx);
    #[cfg(not(target_os = "macos"))]
    return run_cpu_assemble(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, cached_gen, frame_tx, cmd_rx);
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

    let mut frame_idx: u32 = 0;
    let mut skipped_count: u32 = 0;
    let mut rendered_count: u32 = 0;
    // Wall clock starts AFTER the first frame is ready — VT cold-start
    // can be hundreds of ms and shouldn't burn into the playback budget.
    let mut start_wall: Option<std::time::Instant> = None;
    // Total time spent paused so far. Subtracted from wall_t so the
    // "playback time" is the actual time the user has been watching.
    let mut paused_offset = std::time::Duration::ZERO;
    let _ = fps;

    let total_frames = (cfg.path.exists()
        .then(|| vr180_pipeline::decode::probe_video(&cfg.path).ok())
        .flatten().map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)) as usize;

    while let Some(pair) = decoder.next_pair(&pipeline.device)? {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd { DecoderCommand::Stop => return Ok(()) }
        }

        // Pause handling — park while the UI has paused us. We don't
        // tick frame_idx, don't decode the next pair (this `pair`
        // stays buffered for when we resume), don't ship anything.
        // We DO accumulate paused time so wall-clock pacing stays
        // honest when resumed.
        if control.paused.load(Ordering::SeqCst) {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd { DecoderCommand::Stop => return Ok(()) }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // Settings changed? Rebuild the per-eye bundle. Cheap-ish:
        // ~300 ms for a 7000-frame clip; appears as a small stutter
        // while you drag, then real-time again.
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

        // The first iteration: warm-up done, start the playback clock.
        let frame_t = frame_idx as f64 * dt;
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();

        // Behind real-time → skip GPU work for this frame. The actual
        // HEVC decode happened in `next_pair` (sequential decoding is
        // mandatory for non-key frames). Skipping the GPU stages saves
        // ~5-15 ms.
        if wall_t > frame_t + dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (zc): behind by {:.1} ms — skipped {}",
                    (wall_t - frame_t) * 1000.0, skipped_count);
            }
            // If we're WAY behind (more than 1s), the source is just
            // too heavy for the pipeline; resync wall-clock so we
            // don't permanently fast-forward.
            if wall_t > frame_t + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t)
                    - paused_offset);
            }
            // Drop the pair to release the IOSurfaces so the decoder
            // can recycle them.
            drop(pair);
            continue;
        }

        // GPU EAC assembly: NV12 plane textures → RGBA cross texture.
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

        let ((rl, sl), (rr, sr)) = per_eye.get(frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));

        // Cross texture → equirect texture per eye. Same shape as
        // the zero-copy export path.
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
            frame_idx,
            timestamp_s: frame_t,
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

        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t - now));
        }
        frame_idx += 1;
    }
    tracing::info!("decoder (zc): finished after {} frames ({} skipped)",
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
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let _ = fps;
    let total_frames = (vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)) as usize;

    while let Some(pair) = decoder.next_pair()? {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd { DecoderCommand::Stop => return Ok(()) }
        }

        if control.paused.load(Ordering::SeqCst) {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd { DecoderCommand::Stop => return Ok(()) }
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

        let frame_t = frame_idx as f64 * dt;
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();

        if wall_t > frame_t + dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if wall_t > frame_t + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t)
                    - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (cpu): behind by {:.1} ms — skipped {}",
                    (wall_t - frame_t) * 1000.0, skipped_count);
            }
            continue;
        }

        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        let ((rl, sl), (rr, sr)) = per_eye.get(frame_idx as usize).copied()
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
            frame_idx,
            timestamp_s: frame_t,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t - now));
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

