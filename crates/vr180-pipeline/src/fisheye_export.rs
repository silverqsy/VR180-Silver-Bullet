//! Fisheye source → SBS half-equirect file export.
//!
//! Mirrors the preview path in `vr180-gui::decoder::run_fisheye` but
//! writes to disk via [`crate::encode::H265Encoder`] instead of
//! shipping textures to egui. Same source dispatch (SBS / dual-stream
//! OSV / BRAW), same KB projection, same per-eye calibration, same
//! stabilization options.
//!
//! Audio mux, LUT/color stack, ProRes / MV-HEVC are intentionally not
//! in this MVP — they're queued for follow-up phases. The output is
//! a single H.265 `.mp4` / `.mov` file with no audio.

use crate::encode::{EncoderBackend, H265Encoder};
use crate::fisheye_decode::{
    BrawFisheyeIter, DualStreamFisheyeIter, FisheyePair, FisheyePairIter,
    SbsFisheyeIter,
};
use crate::gpu::{ColorStackPlan, Device, EquirectRotation, FisheyeCalib};
use crate::source_kind::SourceKind;
use crate::{Error, Result};

/// Export target projection — half-equirect (VR180) or raw fisheye SBS
/// pass-through. The fisheye option skips the equirect projection and
/// just composes the source left/right fisheye eyes into one SBS frame,
/// useful for VFX / re-grade pipelines that want unwarped source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FisheyeExportProjection {
    /// Standard VR180 — un-warp each fisheye eye through the KB calib
    /// to a half-equirect, then compose SBS.
    HalfEquirect,
    /// Normalized circular fisheye — reproject each eye into a canonical
    /// EQUIDISTANT fisheye of `FISHEYE_OUT_FULL_FOV_DEG` full FOV, with
    /// stabilization + per-eye view adjust applied and the source lens's
    /// own distortion removed. Output is a square per-eye disk → SBS is
    /// (2·side × side).
    Fisheye,
}

impl Default for FisheyeExportProjection {
    fn default() -> Self { Self::HalfEquirect }
}

/// Full FOV (degrees) of the normalized equidistant fisheye output. The
/// inscribed circle of the square output frame maps to this angle, so the
/// disk edge sits at `FISHEYE_OUT_FULL_FOV_DEG / 2` from the optical axis.
pub const FISHEYE_OUT_FULL_FOV_DEG: f32 = 195.0;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[allow(unused_imports)]
use crate::fisheye_decode::FisheyePair as _; // satisfy `use` linter when paths fall through

/// Complete export configuration. The fields mirror the GUI's
/// `Settings` plus the encoding knobs that have no equivalent in the
/// preview side.
#[derive(Debug, Clone)]
pub struct FisheyeExportConfig {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub source_kind: SourceKind,
    /// One eye's output dimensions (the SBS file is `2*eye_w × eye_h`).
    pub eye_w: u32,
    pub eye_h: u32,
    pub fps: f32,
    /// Average bitrate in kbps. Passed straight to the H.265 encoder.
    pub bitrate_kbps: u32,
    pub encoder: EncoderBackend,
    /// 8 = HEVC Main / 8-bit YUV. 10 = HEVC Main10 / 10-bit YUV
    /// (P010LE on VT, YUV420P10LE on libx265). Defaults to 10 for
    /// fisheye export so gradient regions (sky, shadows) keep
    /// gradation.
    pub bit_depth: u8,
    /// ProRes profile (0..=5: Proxy / LT / Standard / HQ / 4444 / 4444 XQ).
    /// Ignored unless `encoder` is a ProRes backend.
    pub prores_profile: i32,
    /// Inject APMP (Apple Projected Media Profile) metadata for
    /// Vision Pro VR180 playback. Default true for fisheye SBS.
    pub inject_apmp: bool,
    /// Inject Spatial Media V2 metadata (st3d + sv3d) for YouTube
    /// VR180 playback. Default true for fisheye SBS.
    pub inject_youtube_vr180: bool,
    /// Camera baseline in mm — written into APMP's `cams/blin` atom
    /// (16.16 fixed-point). Typical 60..=70 mm; 65 mm matches human IPD.
    pub apmp_baseline_mm: f32,
    /// Stabilization on/off. Calibration / preset come from below.
    pub stabilize: bool,
    /// OSV stabilization soft-cap on correction angle. `0.0` = no
    /// cap (legacy camera-lock, bit-identical to the pre-slider
    /// build). Mirrors `Settings.dji_max_corr_deg`.
    pub dji_max_corr_deg: f32,
    /// OSV stabilization smoothing window in ms. `0.0` = sharp
    /// per-frame camera-lock (legacy default). Mirrors
    /// `Settings.dji_smooth_ms`.
    pub dji_smooth_ms: f32,
    /// Soft-stab velocity→smoothing response curve (0.2–3.0, 1 = linear).
    /// Mirrors `Settings.dji_responsiveness`.
    pub dji_responsiveness: f32,
    /// Per-eye view adjustment (global pano-map + stereo offset).
    /// `ViewAdjust::IDENTITY` (all zeros) = no-op, composed AFTER
    /// stabilization in the projection step.
    pub view_adjust: crate::panomap::ViewAdjust,

    /// Color stack — CDL + 3D LUT + temp/tint/saturation. Identity by
    /// default = no-op fast path. Applied AFTER projection, BEFORE
    /// SBS compose. On the 10-bit paths the stack runs in 16-bit so
    /// the encoder still sees full 10-bit precision.
    pub color_stack: ColorStackPlan,

    /// Output projection. `HalfEquirect` = standard VR180 (default).
    /// `Fisheye` = raw fisheye pass-through SBS, no equirect projection.
    pub projection: FisheyeExportProjection,

    // ── Per-eye calibration overrides (mirror Settings) ───────────
    pub fisheye_preset: String,
    pub fisheye_override_left: bool,
    pub fisheye_override_right: bool,
    pub fisheye_fov_deg_left: f32,
    pub fisheye_fov_deg_right: f32,
    pub fisheye_k_left: [f32; 4],
    pub fisheye_k_right: [f32; 4],
    /// Per-eye manual k5 override (OSV 5th radial coeff); 0 = none.
    pub fisheye_k5_left: f32,
    pub fisheye_k5_right: f32,
    /// Per-eye manual Brown-Conrady tangential override `[p1, p2]` (OSV
    /// field 20); `[0,0]` = none. Mirrors Auto so override keeps the rim.
    pub fisheye_p_left: [f32; 2],
    pub fisheye_p_right: [f32; 2],
    /// Normalized [0,1] principal point per eye (used when override on).
    pub fisheye_cx_norm_left: f32,
    pub fisheye_cy_norm_left: f32,
    pub fisheye_cx_norm_right: f32,
    pub fisheye_cy_norm_right: f32,
    pub fisheye_swap_eyes: bool,

    // ── Trim ──────────────────────────────────────────────────────
    pub trim_in_s: Option<f64>,
    pub trim_out_s: Option<f64>,
}

/// Progress update emitted via the caller's callback after each
/// rendered frame.
#[derive(Debug, Clone, Copy)]
pub struct ExportProgress {
    /// 0-based frame index just written.
    pub frame_idx: u64,
    /// Total frames the encoder expects to write (best estimate from
    /// duration × fps, possibly with trim applied).
    pub total_frames: u64,
    /// Wall-clock encoding rate (frames per second), averaged over the
    /// whole run so far.
    pub fps_avg: f32,
}

/// Build a temp video-only path next to the final export target. The
/// encoder writes here; once it finishes we mux the source's audio
/// onto a final file at `final_out` and clean this up.
fn video_only_temp_path(final_out: &std::path::Path) -> std::path::PathBuf {
    let mut s = final_out.as_os_str().to_owned();
    s.push(".video.tmp");
    if let Some(ext) = final_out.extension() {
        s.push(".");
        s.push(ext);
    }
    std::path::PathBuf::from(s)
}

/// Open the H.265 encoder, transparently falling back from the NVIDIA
/// hardware encoder (`hevc_nvenc`) to `libx265` (CPU) if it can't be
/// created — e.g. no NVIDIA GPU, a driver mismatch, or the codec missing
/// from the FFmpeg build. The hardware path is the default on Windows
/// because libx265 is the export bottleneck at VR180 resolutions, but it
/// must degrade gracefully rather than fail the export outright.
#[allow(clippy::too_many_arguments)]
fn open_h265_encoder(
    video_tmp: &std::path::Path,
    sbs_w: u32, sbs_h: u32,
    fps: f32,
    bitrate_kbps: u32,
    backend: EncoderBackend,
    bit_depth: u8,
    prores_profile: i32,
) -> Result<H265Encoder> {
    match H265Encoder::create_with_bit_depth(
        video_tmp, sbs_w, sbs_h, fps, bitrate_kbps, backend, bit_depth, prores_profile,
    ) {
        Ok(enc) => Ok(enc),
        Err(e) if backend == EncoderBackend::HevcNvenc => {
            tracing::warn!(
                "fisheye_export: hevc_nvenc unavailable ({e}); falling back to libx265 (CPU)"
            );
            H265Encoder::create_with_bit_depth(
                video_tmp, sbs_w, sbs_h, fps, bitrate_kbps,
                EncoderBackend::Libx265, bit_depth, prores_profile,
            )
        }
        Err(e) => Err(e),
    }
}

/// Inject VR180 metadata atoms into `final_out` in place.
/// At most ONE of `inject_youtube` / `inject_apmp` can be enabled —
/// the two metadata flavours overwrite the same set of atoms and a
/// player would otherwise see conflicting signals. If both are
/// requested we prefer APMP (Vision Pro is the higher-fidelity
/// target) and log a warning.
fn finalize_metadata(
    final_out: &std::path::Path,
    inject_youtube: bool,
    inject_apmp: bool,
    apmp_baseline_mm: f32,
) -> Result<()> {
    use crate::spherical_inject::{inject_youtube_vr180, inject_apmp_vr180};
    match (inject_youtube, inject_apmp) {
        (false, false) => Ok(()),
        (true, false)  => inject_youtube_vr180(final_out),
        (false, true)  => inject_apmp_vr180(final_out, apmp_baseline_mm),
        (true,  true)  => {
            tracing::warn!(
                "fisheye_export: both YouTube and APMP metadata requested — \
                 they conflict in the same atoms; injecting APMP only"
            );
            inject_apmp_vr180(final_out, apmp_baseline_mm)
        }
    }
}

/// After the encoder closes the temp video file, mux the source's
/// audio track onto it. When the source has no audio we just rename
/// the temp into place. Caller already passed `final_out` to the GUI;
/// we don't move the file anywhere else.
fn finalize_with_audio(
    audio_src: &std::path::Path,
    video_tmp: &std::path::Path,
    final_out: &std::path::Path,
) -> Result<()> {
    match crate::audio::mux_video_with_passthrough_audio(audio_src, video_tmp, final_out) {
        Ok(0) => {
            // Source had no audio — promote the temp video to the
            // final location and we're done.
            std::fs::remove_file(final_out).ok(); // in case a stale exists
            std::fs::rename(video_tmp, final_out).map_err(|e| Error::Io(e))?;
            tracing::info!(
                "fisheye_export: no audio in source, renamed video-only output to {}",
                final_out.display()
            );
        }
        Ok(audio_packets) => {
            // Mux succeeded with audio — clean up the temp video.
            std::fs::remove_file(video_tmp).ok();
            tracing::info!(
                "fisheye_export: muxed {} audio packets from source onto final output",
                audio_packets
            );
        }
        Err(e) => {
            // Mux failed — leave the temp video so the user at least
            // has a video-only file. Surface the error.
            tracing::warn!(
                "fisheye_export: audio mux failed: {e} — temp video left at {}",
                video_tmp.display()
            );
            return Err(e);
        }
    }
    Ok(())
}

/// Encode a fisheye source to a single SBS half-equirect H.265 file.
///
/// Synchronous — runs to completion or until `cancel.load(SeqCst)`
/// flips to true. Caller is expected to drive it from a worker thread
/// and drain `progress_cb` updates from a channel.
///
/// On cancel: closes the encoder cleanly (the muxer flushes its
/// header) so the partial output file is still playable up to the
/// last written frame.
pub fn export_fisheye(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    mut progress_cb: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    tracing::info!(
        "fisheye_export: {} → {} ({}x{} @ {:.2} fps, {} kbps, {}-bit)",
        cfg.source_path.display(), cfg.output_path.display(),
        cfg.eye_w * 2, cfg.eye_h, cfg.fps, cfg.bitrate_kbps, cfg.bit_depth,
    );

    // ── Fast path: zero-copy OSV → P010 → VT (macOS, 10-bit) ──────
    // On macOS with VT encode + 10-bit + OSV source, we can stay on
    // the GPU end-to-end: VT-decoded P010 IOSurfaces are wrapped as
    // wgpu textures, fed straight into a YCbCr→RGB-baked projection
    // shader, then composed into the P010 encode IOSurface for VT.
    // No swscale, no `av_hwframe_transfer_data`, no CPU→GPU upload.
    #[cfg(target_os = "macos")]
    {
        let on_macos_vt = cfg.encoder == EncoderBackend::VideoToolbox;
        // Zero-copy P010 decode → projection supports both half-equirect
        // and the stabilized-fisheye output. BOTH 8-bit and 10-bit H.265
        // output take this path: the OSV source is always 10-bit HEVC, so
        // VT decodes P010 regardless of the chosen output depth, and only
        // the final encode differs (P010 Main10 vs BGRA Main). This means
        // 8-bit no longer pays the CPU-roundtrip decode the general path
        // uses (download → swscale → re-upload per frame), which made 8-bit
        // export slower than 10-bit.
        let can_zero_copy_decode = matches!(cfg.source_kind, SourceKind::DjiOsv)
            && on_macos_vt
            && (cfg.bit_depth == 10 || cfg.bit_depth == 8);
        if can_zero_copy_decode {
            return export_fisheye_osv_zerocopy_p010(
                pipeline, cfg, &mut progress_cb, cancel,
            );
        }
    }

    // ── Fast path: GPU-resident OSV → CUDA → NVENC (Windows, default) ────
    // The full zero-copy-encode path — keeps the frame in VRAM and feeds NVENC
    // via CUDA, no CPU readback. ~1.5× the readback path at 8K (reaches the
    // NVENC hardware ceiling, ~36 fps). Default for NVENC + 10-bit + HalfEquirect
    // + Vulkan + P010; set VR180_NO_GPU_RESIDENT to force the readback path.
    // ANY failure (setup or mid-stream) falls through to the readback path, so
    // enabling it by default is strictly safe.
    #[cfg(target_os = "windows")]
    tracing::info!(
        "fisheye_export dispatch probe: src={:?} enc={:?} bd={} proj={:?} vulkan={} p010={}",
        cfg.source_kind, cfg.encoder, cfg.bit_depth, cfg.projection,
        crate::interop_windows::is_vulkan_backend(&pipeline.device),
        pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010),
    );

    #[cfg(target_os = "windows")]
    {
        let try_gpu_resident = std::env::var_os("VR180_NO_GPU_RESIDENT").is_none()
            && std::env::var_os("VR180_EXPORT_FORCE_CPU").is_none()
            && matches!(cfg.source_kind, SourceKind::DjiOsv)
            && matches!(cfg.encoder, EncoderBackend::HevcNvenc)
            && cfg.bit_depth == 10
            && matches!(cfg.projection, FisheyeExportProjection::HalfEquirect | FisheyeExportProjection::Fisheye)
            && crate::interop_windows::is_vulkan_backend(&pipeline.device)
            && pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010);
        if try_gpu_resident {
            let swap = !cfg.fisheye_swap_eyes;
            let ctx = crate::interop_windows::VulkanImportCtx::from_wgpu(
                &pipeline.adapter, &pipeline.device,
            );
            let iter = crate::fisheye_decode::D3d11SharedDualStreamIter::new(
                &cfg.source_path, swap, u32::MAX, u32::MAX,
            );
            match (ctx, iter) {
                (Some(ctx), Ok(iter)) => {
                    tracing::info!("fisheye_export: GPU-RESIDENT NVENC(CUDA) path ENGAGED");
                    match export_fisheye_osv_gpu_resident(
                        Arc::clone(&pipeline), cfg.clone(), ctx, iter, &mut progress_cb, Arc::clone(&cancel),
                    ) {
                        Ok(()) => return Ok(()),
                        Err(e) => tracing::warn!(
                            "fisheye_export: GPU-resident path failed ({e}) — \
                             falling back to the readback path"
                        ),
                    }
                }
                (c, i) => tracing::warn!(
                    "fisheye_export: GPU-resident unavailable (vulkan_ctx={}, iter_ok={}) — \
                     falling through to readback path", c.is_some(), i.is_ok()
                ),
            }
        }
    }

    // ── Fast path: zero-copy OSV → d3d11va → Vulkan → libx265 (Windows) ──
    // GPU-resident decode: NVDEC (d3d11va) decodes the dual HEVC P010
    // streams, a D3D11 compute shader converts each eye P010→RGBA16 at
    // native res (no downscale), and the texture is imported into wgpu via
    // VK_KHR_external_memory_win32 — exactly the preview's zero-copy front
    // end. From there it's the same RS-aware KB projection + 16-bit color
    // stack + SBS compose, then RGB48 readback → libx265 (the portable
    // path's encoder tail). What we save vs the portable path: the per-frame
    // swscale P010LE→RGBA64LE + CPU upload of the 3840² dual stream.
    //
    // Scope: OSV + HEVC(libx265) + HalfEquirect + Vulkan + P010 only.
    // Anything else (ProRes, raw-fisheye pass-through, non-OSV, DX12, no
    // P010) — or the `VR180_EXPORT_FORCE_CPU` escape hatch — falls through
    // to the portable readback path below, untouched.
    #[cfg(target_os = "windows")]
    {
        let can_try = std::env::var_os("VR180_EXPORT_FORCE_CPU").is_none()
            && matches!(cfg.source_kind, SourceKind::DjiOsv)
            && matches!(cfg.encoder, EncoderBackend::Libx265 | EncoderBackend::HevcNvenc)
            && cfg.projection == FisheyeExportProjection::HalfEquirect
            && (cfg.bit_depth == 10 || cfg.bit_depth == 8)
            && crate::interop_windows::is_vulkan_backend(&pipeline.device)
            && pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010);
        if can_try {
            // OSV swap-by-default ⊕ user override (matches preview + CPU path).
            let swap = !cfg.fisheye_swap_eyes;
            let ctx = crate::interop_windows::VulkanImportCtx::from_wgpu(
                &pipeline.adapter, &pipeline.device,
            );
            // u32::MAX work dims → the iter clamps to native, so the D3D11
            // P010→RGBA16 convert is 1:1 (full-res export, no quality loss).
            let iter = crate::fisheye_decode::D3d11SharedDualStreamIter::new(
                &cfg.source_path, swap, u32::MAX, u32::MAX,
            );
            match (ctx, iter) {
                (Some(ctx), Ok(iter)) => {
                    tracing::info!(
                        "fisheye_export: ZERO-COPY d3d11va→Vulkan export path ENGAGED \
                         (native-res P010, no CPU download/swscale)"
                    );
                    return export_fisheye_osv_zerocopy_d3d11(
                        pipeline, cfg, ctx, iter, &mut progress_cb, cancel,
                    );
                }
                (c, i) => {
                    tracing::warn!(
                        "fisheye_export: zero-copy unavailable (vulkan_ctx={}, \
                         d3d11va_iter_ok={}) — falling back to CPU readback path",
                        c.is_some(), i.is_ok()
                    );
                }
            }
        }
    }

    // ── Open the right iterator ───────────────────────────────────
    // Export always passes max_decode_side=0 (no cap) so we get the
    // source's native fisheye resolution. The preview path is the
    // only consumer that caps to MAX_DECODE_SIDE for real-time fps.
    let mut decoder: Box<dyn FisheyePairIter> = match cfg.source_kind {
        SourceKind::DjiOsv => Box::new(
            DualStreamFisheyeIter::new_with_options(
                &cfg.source_path, crate::decode::HwDecode::Auto, 0,
                // OSV swap-by-default ⊕ user override (matches preview).
                !cfg.fisheye_swap_eyes,
                0, // no resolution cap — export at full native size
                // 16-bit RGBA scaler output when targeting 10-bit codec
                // so the projection input keeps P010's 10 bits of source
                // precision instead of being quantized to 8 by the
                // scaler. 8-bit codec → 8-bit scaler (the standard path).
                if cfg.bit_depth >= 10 { 16 } else { 8 },
            )?
        ),
        SourceKind::SbsFisheye => Box::new(
            SbsFisheyeIter::new(&cfg.source_path, crate::decode::HwDecode::Auto, 0)?
        ),
        SourceKind::BlackmagicRaw => {
            let info = vr180_braw::BrawInfo::probe(&cfg.source_path)
                .map_err(|e| Error::Ffmpeg(format!("braw probe: {e}")))?;
            let opts = vr180_braw::decoder::DecodeOptions::default();
            Box::new(
                BrawFisheyeIter::new(&cfg.source_path, &info, &opts, 0)
                    .map_err(|e| Error::Ffmpeg(format!("braw start: {e}")))?
            )
        }
        other => return Err(Error::Ffmpeg(format!(
            "export_fisheye called with non-fisheye source: {other:?}"
        ))),
    };
    let (src_w, src_h) = decoder.eye_dims();

    // ── DJI: extract protobuf for per-eye calib + (optional) stab ─
    let dji_osv_imu: Option<vr180_fisheye::DjiOsvImu> = if matches!(
        cfg.source_kind, SourceKind::DjiOsv
    ) {
        match crate::decode::extract_dji_meta_stream(&cfg.source_path) {
            Ok(blob) => match vr180_fisheye::DjiOsvImu::parse(&blob) {
                Ok(imu) => Some(imu),
                Err(e) => { tracing::warn!("dji protobuf parse: {e}"); None }
            },
            Err(e) => { tracing::warn!("dji meta extract: {e}"); None }
        }
    } else { None };

    // ── Resolve per-eye calibrations ─────────────────────────────
    let (calib_left, calib_right) = resolve_calib_pair(
        &cfg, src_w, src_h, dji_osv_imu.as_ref(),
    );

    // ── Stab rotations (one per source frame) ────────────────────
    let total_clip_frames = {
        let probe = crate::decode::probe_video(&cfg.source_path)
            .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
            .unwrap_or(0).max(1);
        probe
    };
    let stab_rotations: Option<Vec<EquirectRotation>> = if cfg.stabilize {
        match cfg.source_kind {
            SourceKind::DjiOsv => {
                dji_osv_imu.as_ref().and_then(|osv| {
                    // Slider=0 → legacy camera-lock (∞ cap, no
                    // smoothing). User-set values activate the cap /
                    // smoothing as in the preview.
                    let max_corr = if cfg.dji_max_corr_deg > 0.0 {
                        cfg.dji_max_corr_deg
                    } else { f32::INFINITY };
                    crate::dji_imu::compute_dji_stabilization(
                        osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps, cfg.dji_responsiveness,
                    ).ok().map(|s| s.per_frame)
                })
            }
            SourceKind::BlackmagicRaw => {
                vr180_braw::BrawGyroData::extract(&cfg.source_path)
                    .ok()
                    .and_then(|gyro_data| {
                        crate::braw_imu::compute_braw_stabilization(
                            &gyro_data, cfg.fps, total_clip_frames,
                        ).ok().map(|s| s.per_frame)
                    })
            }
            _ => None,
        }
    } else { None };
    tracing::info!(
        "fisheye_export: stab = {}",
        if stab_rotations.is_some() { "engaged" } else { "off" }
    );

    // ── Initial trim seek (clamp negative / past-EOF) ────────────
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    if t_in > 0.001 {
        decoder.seek(t_in)?;
    }
    let t_out = cfg.trim_out_s
        .map(|t| t.max(t_in + 1e-3))
        .unwrap_or(f64::INFINITY);
    let total_frames_to_write = if t_out.is_finite() {
        ((t_out - t_in) * cfg.fps as f64).round() as u64
    } else {
        (total_clip_frames as f64 - t_in * cfg.fps as f64).round().max(0.0) as u64
    };
    tracing::info!(
        "fisheye_export: writing ~{} frames (trim {:?}..{:?})",
        total_frames_to_write, cfg.trim_in_s, cfg.trim_out_s
    );

    // ── Open the encoder ─────────────────────────────────────────
    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    // Three encode paths on macOS / VT:
    //   bit_depth == 8  → BGRA IOSurface zero-copy (Main profile)
    //   bit_depth == 10 → P010 IOSurface zero-copy (Main10 profile)
    //                     — GPU writes Y + UV planes directly into a
    //                     P010 CVPixelBuffer, no swscale, no readback.
    // Non-macOS / non-VT → texture-resident readback fallback.
    // Zero-copy paths are HEVC-only; ProRes goes through the CPU
    // readback path regardless of platform.
    let on_macos_vt = cfg!(target_os = "macos")
        && cfg.encoder == EncoderBackend::VideoToolbox;
    let use_zero_copy_bgra = on_macos_vt && cfg.bit_depth == 8;
    let use_zero_copy_p010 = on_macos_vt && cfg.bit_depth == 10;
    // Write video first to a temp path next to the final output, then
    // mux the source's audio onto it at the very end. The video alone
    // would always be a complete playable file; the extra mux step is
    // what lets us preserve the OSV's stereo AAC track.
    let video_tmp = video_only_temp_path(&cfg.output_path);
    let mut encoder = if use_zero_copy_bgra {
        // macOS/VideoToolbox-only zero-copy BGRA path. `use_zero_copy_bgra`
        // is always `false` on non-macOS targets, so this arm is dead there
        // — gate the VT-only ctor and supply `unreachable!()` where the
        // value is needed (mirrors the `#[cfg]` blocks in the encode loop).
        #[cfg(target_os = "macos")]
        { H265Encoder::create_zero_copy_vt(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )? }
        #[cfg(not(target_os = "macos"))]
        { unreachable!("zero-copy BGRA encode is macOS/VideoToolbox-only") }
    } else if use_zero_copy_p010 {
        // macOS/VideoToolbox-only zero-copy P010 (Main10) path. Same gating.
        #[cfg(target_os = "macos")]
        { H265Encoder::create_zero_copy_vt_p010(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )? }
        #[cfg(not(target_os = "macos"))]
        { unreachable!("zero-copy P010 encode is macOS/VideoToolbox-only") }
    } else {
        open_h265_encoder(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
            cfg.encoder, cfg.bit_depth, cfg.prores_profile,
        )?
    };
    // Tag the file as VR180 SBS so VLC / Quest / Vision Pro pick up
    // the stereo layout. Best-effort — older ffmpeg builds may not
    // expose the side-data field.
    // APMP / YouTube metadata is now injected as a post-process atom
    // write on the final muxed file (see `finalize_with_audio` →
    // `finalize_metadata`); the ffmpeg side-data path is no longer
    // used because it can't set the equirect bounds correctly for
    // VR180 and conflicts between the two metadata flavours have to
    // be resolved by stripping anyway. Keep the codecpar header
    // unmodified here.
    if use_zero_copy_p010 {
        tracing::info!("fisheye_export: zero-copy P010 IOSurface → VT (Main10)");
    } else if use_zero_copy_bgra {
        tracing::info!("fisheye_export: zero-copy BGRA IOSurface → VT (Main)");
    } else {
        tracing::info!(
            "fisheye_export: readback + swscale path (bit_depth={}, backend={:?})",
            cfg.bit_depth, cfg.encoder
        );
    }

    // ── Encode loop ──────────────────────────────────────────────
    let dt = 1.0 / cfg.fps as f64;
    let t_start = std::time::Instant::now();
    let mut frame_idx: u64 = 0;
    let identity_plan = ColorStackPlan::default();
    let color_plan = cfg.color_stack.clone();
    let color_any = color_plan.any_active();
    if color_any {
        tracing::info!("fisheye_export: color stack active (CDL/LUT/grade)");
    }
    let projection = cfg.projection;

    while let Some(pair) = decoder.next_pair()? {
        // Cancellation check before doing any GPU work.
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("fisheye_export: cancelled at frame {}", frame_idx);
            break;
        }

        // Stop at trim_out (use the iterator's PTS rather than a counter
        // so seek-aware sources land on the right frame).
        if pair.pts_s.is_finite() && pair.pts_s >= t_out {
            tracing::info!("fisheye_export: hit trim_out @ {:.3}s", pair.pts_s);
            break;
        }

        // PTS-based stab lookup (same logic as the preview).
        let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
            (pair.pts_s / dt).round() as usize
        } else {
            frame_idx as usize
        };
        let rot = stab_rotations
            .as_ref()
            .and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);

        // Compose per-eye view adjustment (global pano-map + stereo
        // offset) AFTER stab. `is_identity()` short-circuit means
        // default-zero export is byte-identical to the pre-pano-map
        // pipeline.
        let (rot_left, rot_right) = if cfg.view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = cfg.view_adjust.per_eye_matrices();
            (
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };

        // Decide projection target. The 8-bit-path needs an Rgba8Unorm
        // per eye; we always populate `left_tex` / `right_tex` so the
        // existing 8-bit branches still work. The 10-bit branches build
        // their own Rgba16Unorm textures below. Both projections apply
        // stab + view-adjust via the same `rot_left` / `rot_right`
        // composition shaders binding.
        let (left_tex, right_tex) = match projection {
            FisheyeExportProjection::HalfEquirect => (
                pipeline.project_fisheye_to_equirect_texture(
                    &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                    rot_left, calib_left, 10,
                )?,
                pipeline.project_fisheye_to_equirect_texture(
                    &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                    rot_right, calib_right, 11,
                )?,
            ),
            FisheyeExportProjection::Fisheye => (
                pipeline.project_fisheye_to_fisheye_texture(
                    &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                    rot_left, calib_left,
                )?,
                pipeline.project_fisheye_to_fisheye_texture(
                    &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                    rot_right, calib_right,
                )?,
            ),
        };

        // 8-bit non-zero-copy fallback uses `compose_sbs_textures` then
        // does the color stack on the SBS readback below — keep the
        // per-eye `*_tex` references unchanged for that path.
        let left_post  = &left_tex;
        let right_post = &right_tex;

        if use_zero_copy_bgra {
            // 8-bit macOS zero-copy path. apply_color_stack_to_sbs_bgra
            // already supports plan ≠ identity; pass it directly so the
            // stack runs fused with the SBS compose into the IOSurface.
            #[cfg(target_os = "macos")]
            {
                let encode_pb = crate::interop_macos::create_bgra_encode_buffer(
                    &pipeline.device, sbs_w, sbs_h,
                )?;
                pipeline.apply_color_stack_to_sbs_bgra(
                    &left_tex, &right_tex, &encode_pb.wgpu_tex,
                    cfg.eye_w, cfg.eye_h, &color_plan,
                )?;
                encoder.encode_pixel_buffer(&encode_pb)?;
            }
            #[cfg(not(target_os = "macos"))]
            { let _ = identity_plan; unreachable!(); }
            let _ = (&left_tex, &right_tex, &left_post, &right_post);
        } else if use_zero_copy_p010 {
            // 10-bit macOS zero-copy: project at 16-bit, GPU-write
            // both planes of a P010 IOSurface, VT consumes directly.
            #[cfg(target_os = "macos")]
            {
                let (left_tex_16, right_tex_16) = build_eye_eq_16(
                    &pipeline, &pair, src_w, src_h,
                    cfg.eye_w, cfg.eye_h,
                    rot_left, rot_right, calib_left, calib_right,
                    projection,
                )?;
                let left_g = pipeline.apply_color_stack_per_eye_16(
                    &left_tex_16, cfg.eye_w, cfg.eye_h, &color_plan,
                )?;
                let right_g = pipeline.apply_color_stack_per_eye_16(
                    &right_tex_16, cfg.eye_w, cfg.eye_h, &color_plan,
                )?;
                let l_final = left_g .as_ref().unwrap_or(&left_tex_16);
                let r_final = right_g.as_ref().unwrap_or(&right_tex_16);
                let encode_pb = crate::interop_macos::create_p010_encode_buffer(
                    &pipeline.device, sbs_w, sbs_h,
                )?;
                pipeline.compose_sbs_to_p010(
                    l_final, r_final, &encode_pb,
                    cfg.eye_w, cfg.eye_h,
                )?;
                encoder.encode_pixel_buffer_p010(&encode_pb)?;
            }
            #[cfg(not(target_os = "macos"))]
            { let _ = identity_plan; unreachable!(); }
            let _ = (&left_tex, &right_tex, &left_post, &right_post);
        } else if cfg.bit_depth == 10 {
            // 10-bit non-macOS / non-VT fallback: texture-resident
            // 16-bit projection + RGB48 readback + libx265 Main10.
            let (left_tex_16, right_tex_16) = build_eye_eq_16(
                &*pipeline, &pair, src_w, src_h,
                cfg.eye_w, cfg.eye_h,
                rot_left, rot_right, calib_left, calib_right,
                projection,
            )?;
            let left_g = pipeline.apply_color_stack_per_eye_16(
                &left_tex_16, cfg.eye_w, cfg.eye_h, &color_plan,
            )?;
            let right_g = pipeline.apply_color_stack_per_eye_16(
                &right_tex_16, cfg.eye_w, cfg.eye_h, &color_plan,
            )?;
            let l_final = left_g .as_ref().unwrap_or(&left_tex_16);
            let r_final = right_g.as_ref().unwrap_or(&right_tex_16);
            let sbs_tex_16 = pipeline.compose_sbs_textures_16(
                l_final, r_final, cfg.eye_w, cfg.eye_h,
            )?;
            let sbs_rgb48 = pipeline.read_texture_rgb48(&sbs_tex_16, sbs_w, sbs_h)?;
            encoder.encode_frame_rgb48(&sbs_rgb48)?;
            let _ = (&left_tex, &right_tex);
        } else {
            // 8-bit non-zero-copy fallback. Color stack happens via the
            // legacy CPU roundtrip on the SBS readback below (acceptable
            // since this path is already CPU-bound).
            let sbs_tex = pipeline.compose_sbs_textures(
                left_post, right_post, cfg.eye_w, cfg.eye_h,
            )?;
            let sbs_rgb8 = pipeline.read_texture_rgb8(&sbs_tex, sbs_w, sbs_h)?;
            let graded = if color_any {
                apply_color_stack_rgb8(&*pipeline, &color_plan, sbs_rgb8, sbs_w, sbs_h)?
            } else {
                sbs_rgb8
            };
            encoder.encode_frame(&graded)?;
        }

        frame_idx += 1;
        let elapsed = t_start.elapsed().as_secs_f32().max(1e-3);
        let fps_avg = frame_idx as f32 / elapsed;
        progress_cb(ExportProgress {
            frame_idx,
            total_frames: total_frames_to_write,
            fps_avg,
        });
    }

    encoder.finish()?;
    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path)?;
    finalize_metadata(
        &cfg.output_path,
        cfg.inject_youtube_vr180,
        cfg.inject_apmp,
        cfg.apmp_baseline_mm,
    )?;
    tracing::info!(
        "fisheye_export: done, {} frames in {:.2?}",
        frame_idx, t_start.elapsed()
    );
    Ok(())
}

// ── Fast path: zero-copy OSV → P010 → VT ──────────────────────────
//
// Runs only on macOS with VT encode + 10-bit + OSV source. Pulls VT-
// decoded CVPixelBuffers from `ZeroCopyDualStreamFisheyeIter`, wraps
// their P010 IOSurface planes as wgpu textures (no swscale, no CPU
// upload), projects each eye via `project_fisheye_p010_to_equirect_texture_16`
// (YCbCr→RGB happens inline in the shader), composes into the P010
// encode IOSurface, and hands the CVPixelBuffer to VT.
//
// At native OSV resolution (3840 × 3840 × 2 streams × 10-bit) this
// eliminates ~840 MB / frame of CPU bandwidth that the regular path
// pays for the swscale P010LE→RGBA64LE + stride repack + queue.write_texture.
#[cfg(target_os = "macos")]
fn export_fisheye_osv_zerocopy_p010(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    progress_cb: &mut dyn FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    use crate::fisheye_decode::ZeroCopyDualStreamFisheyeIter;

    // Open zero-copy decoder. OSV swap convention mirrors
    // DualStreamFisheyeIter: !cfg.fisheye_swap_eyes means swap on by
    // default (Lens A == stream 0 == right eye).
    let mut decoder = ZeroCopyDualStreamFisheyeIter::new(
        &cfg.source_path,
        0, // no frame limit
        !cfg.fisheye_swap_eyes,
    )?;
    let (src_w, src_h) = decoder.eye_dims();

    // Extract DJI protobuf for per-eye calibration (and stab data).
    let dji_osv_imu = match crate::decode::extract_dji_meta_stream(&cfg.source_path) {
        Ok(blob) => match vr180_fisheye::DjiOsvImu::parse(&blob) {
            Ok(imu) => Some(imu),
            Err(e) => { tracing::warn!("dji protobuf parse: {e}"); None }
        },
        Err(e) => { tracing::warn!("dji meta extract: {e}"); None }
    };
    let (calib_left, calib_right) = resolve_calib_pair(
        &cfg, src_w, src_h, dji_osv_imu.as_ref(),
    );

    // Stab rotations (OSV is locked to camera-lock per the GUI panel).
    let total_clip_frames = crate::decode::probe_video(&cfg.source_path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0).max(1);
    let stab_rotations: Option<Vec<crate::gpu::EquirectRotation>> = if cfg.stabilize {
        dji_osv_imu.as_ref().and_then(|osv| {
            let max_corr = if cfg.dji_max_corr_deg > 0.0 {
                cfg.dji_max_corr_deg
            } else { f32::INFINITY };
            crate::dji_imu::compute_dji_stabilization(
                osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps, cfg.dji_responsiveness,
            ).ok().map(|s| s.per_frame)
        })
    } else { None };
    tracing::info!(
        "fisheye_export (zero-copy): stab = {}",
        if stab_rotations.is_some() { "engaged" } else { "off" }
    );

    // Trim seek + frame estimate.
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    if t_in > 0.001 {
        decoder.seek(t_in)?;
    }
    let t_out = cfg.trim_out_s
        .map(|t| t.max(t_in + 1e-3))
        .unwrap_or(f64::INFINITY);
    let total_frames_to_write = if t_out.is_finite() {
        ((t_out - t_in) * cfg.fps as f64).round() as u64
    } else {
        (total_clip_frames as f64 - t_in * cfg.fps as f64).round().max(0.0) as u64
    };
    tracing::info!(
        "fisheye_export (zero-copy): writing ~{} frames (trim {:?}..{:?})",
        total_frames_to_write, cfg.trim_in_s, cfg.trim_out_s
    );

    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    // Encode video to a temp path; the source's audio is muxed onto
    // the final output after `encoder.finish()` returns.
    let video_tmp = video_only_temp_path(&cfg.output_path);
    // 10-bit → P010 IOSurface encode (Main10); 8-bit → BGRA IOSurface
    // encode (Main). Both consume the same zero-copy P010 decode above.
    let ten_bit = cfg.bit_depth == 10;
    let mut encoder = if ten_bit {
        H265Encoder::create_zero_copy_vt_p010(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    } else {
        H265Encoder::create_zero_copy_vt(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    };
    // APMP / YouTube metadata is now injected as a post-process atom
    // write on the final muxed file (see `finalize_with_audio` →
    // `finalize_metadata`); the ffmpeg side-data path is no longer
    // used because it can't set the equirect bounds correctly for
    // VR180 and conflicts between the two metadata flavours have to
    // be resolved by stripping anyway. Keep the codecpar header
    // unmodified here.

    // Per-row rolling-shutter correction is on by default for OSV
    // — matches Python's behaviour when the user has the RS toggle
    // enabled in `vr180_gui.py` (default OFF in Python, but every
    // user-visible difference in OSV stab quality between Python and
    // Rust traces back to it). The DJI Osmo OQ001 readout time is
    // ~19 ms from `vr180_gui.py:500`; the per-row pass cancels the
    // intra-frame shear that per-frame stab alone can't reach.
    let rs_enabled = cfg.stabilize && dji_osv_imu.is_some();
    // FPS-aware readout: OSMO 360 sensor at 50fps uses 16.23 ms vs
    // 18.3 ms at 30fps. Wrong readout → wrong phase offset → "loose"
    // stab at fast motion.
    let readout_s = crate::dji_imu::dji_osmo_readout_ms_for_fps(cfg.fps) / 1000.0;
    tracing::info!(
        "fisheye_export (zero-copy): P010 IOSurface decode → {} encode, \
         no CPU bounce on the decode side; per-row RS = {}",
        if cfg.bit_depth == 10 { "P010 IOSurface (Main10)" } else { "BGRA IOSurface (Main, 8-bit)" },
        if rs_enabled { format!("on (readout {:.1}ms)", readout_s * 1000.0) }
        else { "off".to_string() }
    );

    let dt = 1.0 / cfg.fps as f64;
    let t_start = std::time::Instant::now();
    let mut frame_idx: u64 = 0;

    while let Some(pair) = decoder.next_pair(&pipeline.device)? {
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("fisheye_export (zero-copy): cancelled at frame {}", frame_idx);
            break;
        }
        if pair.pts_s.is_finite() && pair.pts_s >= t_out {
            tracing::info!("fisheye_export (zero-copy): hit trim_out @ {:.3}s", pair.pts_s);
            break;
        }

        let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
            (pair.pts_s / dt).round() as usize
        } else {
            frame_idx as usize
        };
        let rot = stab_rotations
            .as_ref()
            .and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(crate::gpu::EquirectRotation::IDENTITY);

        // Per-row RS matrices for this frame. Built lazily on the CPU;
        // <200 KB at 3840 rows so the per-frame `queue.write_buffer`
        // is negligible vs the 138 MB+ that the zero-copy path saved
        // on the decode side.
        let rs_rows_f32: Option<Vec<f32>> = if rs_enabled {
            dji_osv_imu.as_ref().and_then(|osv| {
                let lens_a = osv.lens_a.mount_quat_xyzw
                    .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
                crate::dji_imu::compute_per_row_quaternions_for_frame(
                    osv, stab_idx, readout_s, src_h, cfg.fps,
                )
                .map(|q| crate::dji_imu::pack_per_row_camera_matrices(&q, lens_a))
            })
        } else {
            None
        };

        // Compose per-eye view adjustment AFTER stab. is_identity()
        // short-circuit means default-zero export is byte-identical.
        let (rot_left, rot_right) = if cfg.view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = cfg.view_adjust.per_eye_matrices();
            (
                crate::gpu::EquirectRotation(
                    crate::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                crate::gpu::EquirectRotation(
                    crate::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };

        // Project Y/UV → Rgba16Unorm output, per eye. Per-row rolling-
        // shutter correction is wired for BOTH projections — the RS warp
        // operates on the source-frame direction, independent of whether
        // the output is half-equirect or fisheye.
        let (left_eq, right_eq) = match (cfg.projection, rs_rows_f32.as_deref()) {
            (FisheyeExportProjection::HalfEquirect, Some(rs_buf)) => {
                let l = pipeline.project_fisheye_p010_to_equirect_rs_texture_16(
                    &pair.left_y.texture, &pair.left_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs_buf,
                )?;
                let r = pipeline.project_fisheye_p010_to_equirect_rs_texture_16(
                    &pair.right_y.texture, &pair.right_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs_buf,
                )?;
                (l, r)
            }
            (FisheyeExportProjection::HalfEquirect, None) => {
                let l = pipeline.project_fisheye_p010_to_equirect_texture_16(
                    &pair.left_y.texture, &pair.left_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left,
                )?;
                let r = pipeline.project_fisheye_p010_to_equirect_texture_16(
                    &pair.right_y.texture, &pair.right_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right,
                )?;
                (l, r)
            }
            (FisheyeExportProjection::Fisheye, Some(rs_buf)) => {
                let l = pipeline.project_fisheye_p010_to_fisheye_rs_texture_16(
                    &pair.left_y.texture, &pair.left_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs_buf,
                )?;
                let r = pipeline.project_fisheye_p010_to_fisheye_rs_texture_16(
                    &pair.right_y.texture, &pair.right_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs_buf,
                )?;
                (l, r)
            }
            (FisheyeExportProjection::Fisheye, None) => {
                let l = pipeline.project_fisheye_p010_to_fisheye_texture_16(
                    &pair.left_y.texture, &pair.left_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left,
                )?;
                let r = pipeline.project_fisheye_p010_to_fisheye_texture_16(
                    &pair.right_y.texture, &pair.right_uv.texture,
                    src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right,
                )?;
                (l, r)
            }
        };

        // 16-bit color stack — runs only when active. Stays Rgba16Unorm
        // so VT sees full 10-bit precision when the user has added a
        // grade.
        let left_g = pipeline.apply_color_stack_per_eye_16(
            &left_eq, cfg.eye_w, cfg.eye_h, &cfg.color_stack,
        )?;
        let right_g = pipeline.apply_color_stack_per_eye_16(
            &right_eq, cfg.eye_w, cfg.eye_h, &cfg.color_stack,
        )?;
        let l_final = left_g .as_ref().unwrap_or(&left_eq);
        let r_final = right_g.as_ref().unwrap_or(&right_eq);

        // Compose the (already-graded, 16-bit) eyes into the encode
        // IOSurface and hand off to VT. 10-bit → P010 (YCbCr planes);
        // 8-bit → BGRA, composed with an IDENTITY plan (no re-grade — the
        // 16→8 downconvert happens in the compose), then Main encode.
        if ten_bit {
            let encode_pb = crate::interop_macos::create_p010_encode_buffer(
                &pipeline.device, sbs_w, sbs_h,
            )?;
            pipeline.compose_sbs_to_p010(
                l_final, r_final, &encode_pb, cfg.eye_w, cfg.eye_h,
            )?;
            encoder.encode_pixel_buffer_p010(&encode_pb)?;
        } else {
            let encode_pb = crate::interop_macos::create_bgra_encode_buffer(
                &pipeline.device, sbs_w, sbs_h,
            )?;
            pipeline.apply_color_stack_to_sbs_bgra(
                l_final, r_final, &encode_pb.wgpu_tex,
                cfg.eye_w, cfg.eye_h, &ColorStackPlan::default(),
            )?;
            encoder.encode_pixel_buffer(&encode_pb)?;
        }

        // Drop decode + compose textures so the IOSurface retains
        // release before the next frame allocates new ones.
        drop(pair);

        frame_idx += 1;
        let elapsed = t_start.elapsed().as_secs_f32().max(1e-3);
        let fps_avg = frame_idx as f32 / elapsed;
        progress_cb(ExportProgress {
            frame_idx,
            total_frames: total_frames_to_write,
            fps_avg,
        });
    }

    encoder.finish()?;
    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path)?;
    finalize_metadata(
        &cfg.output_path,
        cfg.inject_youtube_vr180,
        cfg.inject_apmp,
        cfg.apmp_baseline_mm,
    )?;
    tracing::info!(
        "fisheye_export (zero-copy): done, {} frames in {:.2?}",
        frame_idx, t_start.elapsed()
    );
    Ok(())
}

// ── Fast path: zero-copy OSV → d3d11va → Vulkan → libx265 (Windows) ────
//
// The Windows analogue of `export_fisheye_osv_zerocopy_p010`. Pulls
// GPU-resident RGBA16 eye textures from `D3d11SharedDualStreamIter`
// (NVDEC d3d11va decode + a D3D11 compute P010→RGBA16 convert at native
// res), imports each into wgpu via `VK_KHR_external_memory_win32`, runs
// the same RS-aware KB projection + 16-bit color stack + SBS compose the
// preview uses, then reads back RGB48 and hands it to libx265. The encoder
// tail is identical to the portable path — only the decode → projection
// handoff stays GPU-resident (no swscale, no per-frame CPU upload).
//
// Scope: HalfEquirect + HEVC(libx265). Errors are hard (the caller already
// committed to this path after a successful iter/ctx open); they propagate
// rather than silently restart on the CPU path mid-stream.
#[cfg(target_os = "windows")]
fn export_fisheye_osv_zerocopy_d3d11(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    ctx: crate::interop_windows::VulkanImportCtx,
    mut iter: crate::fisheye_decode::D3d11SharedDualStreamIter,
    progress_cb: &mut impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let (src_w, src_h) = iter.eye_dims(); // == native (work clamped to native)
    let (nw, nh) = iter.native_dims();
    tracing::info!(
        "fisheye_export (zero-copy): work {}x{} (native {}x{}) → {}x{} SBS",
        src_w, src_h, nw, nh, cfg.eye_w * 2, cfg.eye_h
    );

    // ── Per-eye calib + stab from the OSV protobuf (same as portable) ──
    let dji_osv_imu: Option<vr180_fisheye::DjiOsvImu> =
        match crate::decode::extract_dji_meta_stream(&cfg.source_path) {
            Ok(blob) => match vr180_fisheye::DjiOsvImu::parse(&blob) {
                Ok(imu) => Some(imu),
                Err(e) => { tracing::warn!("zc export: dji protobuf parse: {e}"); None }
            },
            Err(e) => { tracing::warn!("zc export: dji meta extract: {e}"); None }
        };
    let (calib_left, calib_right) =
        resolve_calib_pair(&cfg, src_w, src_h, dji_osv_imu.as_ref());

    let total_clip_frames = crate::decode::probe_video(&cfg.source_path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)
        .max(1);
    let stab_rotations: Option<Vec<EquirectRotation>> = if cfg.stabilize {
        dji_osv_imu.as_ref().and_then(|osv| {
            let max_corr = if cfg.dji_max_corr_deg > 0.0 {
                cfg.dji_max_corr_deg
            } else { f32::INFINITY };
            crate::dji_imu::compute_dji_stabilization(
                osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps, cfg.dji_responsiveness,
            ).ok().map(|s| s.per_frame)
        })
    } else { None };
    tracing::info!(
        "fisheye_export (zero-copy): stab = {}",
        if stab_rotations.is_some() { "engaged" } else { "off" }
    );

    // ── Trim ───────────────────────────────────────────────────────
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    if t_in > 0.001 { iter.seek(t_in)?; }
    let t_out = cfg.trim_out_s
        .map(|t| t.max(t_in + 1e-3))
        .unwrap_or(f64::INFINITY);
    let total_frames_to_write = if t_out.is_finite() {
        ((t_out - t_in) * cfg.fps as f64).round() as u64
    } else {
        (total_clip_frames as f64 - t_in * cfg.fps as f64).round().max(0.0) as u64
    };

    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    let video_tmp = video_only_temp_path(&cfg.output_path);

    let dt = 1.0 / cfg.fps as f64;
    let color_plan = cfg.color_stack.clone();
    let readout_s = crate::dji_imu::dji_osmo_readout_ms_for_fps(cfg.fps) / 1000.0;
    let lens_a_mount = dji_osv_imu.as_ref()
        .and_then(|o| o.lens_a.mount_quat_xyzw)
        .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);

    // ── 3-stage pipeline ───────────────────────────────────────────
    // The serial path summed decode + GPU + encode per frame; at native
    // res that's encode/readback-bound at ~4 fps. Split it so decode and
    // encode overlap the GPU+readback critical path on the main thread:
    //
    //   [decode thread]  D3D11/NVDEC + P010→RGBA16 convert → SharedFisheyePair
    //        │ sync_channel(3)
    //   [main thread]    import → project(+RS) → color → compose → RGB48 readback
    //        │ sync_channel(2)   (177 MB/frame at 7680×3840 → keep the bound low)
    //   [encode thread]  owns H265Encoder (NVENC, libx265 fallback) → file
    //
    // All three touch DIFFERENT GPU contexts (ffmpeg D3D11 / our dedicated
    // Vulkan device / NVENC's own session), so there's no shared queue to
    // wedge — Lesson #1 stays satisfied. Wall-clock ≈ the slowest stage
    // instead of their sum.
    use crate::fisheye_decode::SharedFisheyePair;
    // What the main thread hands the encode thread. P010 = the GPU already
    // did RGB→YUV (NVENC's native input, no swscale); Rgba64 = swscale
    // fallback (libx265 / 8-bit).
    enum EncFrame { P010 { y: Vec<u8>, uv: Vec<u8> }, Rgba64(Vec<u8>) }
    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<SharedFisheyePair>(3);
    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<EncFrame>(2);
    let (fmt_tx, fmt_rx) = std::sync::mpsc::sync_channel::<bool>(1);

    // Decode thread — D3D11 only, never touches wgpu. `iter` was already
    // trim-seeked above; just pump pairs until EOF, cancel, or the main
    // thread drops the receiver (early stop / error).
    let cancel_dec = cancel.clone();
    let mut iter = iter;
    let decode_handle = std::thread::spawn(move || {
        loop {
            if cancel_dec.load(Ordering::SeqCst) { break; }
            match iter.next_pair() {
                Ok(Some(p)) => { if pair_tx.send(p).is_err() { break; } }
                Ok(None) => break,
                Err(e) => { tracing::warn!("zc export: decode worker: {e}"); break; }
            }
        }
    });

    // Encode thread — owns the encoder so the NVENC→libx265 fallback (and
    // the codec's own threads) live here, off the GPU critical path. The
    // encoder is created HERE so a create failure cleanly drops `rgb_rx`,
    // unblocking the main thread, and surfaces via the join below.
    let enc_video_tmp = video_tmp.clone();
    let (enc_backend, enc_fps, enc_bitrate, enc_bd, enc_prores) =
        (cfg.encoder, cfg.fps, cfg.bitrate_kbps, cfg.bit_depth, cfg.prores_profile);
    let encode_handle = std::thread::spawn(move || -> Result<u64> {
        let mut encoder = open_h265_encoder(
            &enc_video_tmp, sbs_w, sbs_h, enc_fps, enc_bitrate,
            enc_backend, enc_bd, enc_prores,
        )?;
        // Report the encoder's native input so the main thread produces the
        // matching layout. On create-failure we never send → main's recv
        // errs and the pipeline winds down, surfacing the error at the join.
        let _ = fmt_tx.send(encoder.wants_p010());
        let mut n: u64 = 0;
        while let Ok(f) = frame_rx.recv() {
            match f {
                EncFrame::P010 { y, uv } => encoder.encode_frame_p010(&y, &uv)?,
                EncFrame::Rgba64(b)      => encoder.encode_frame_rgba64(&b)?,
            }
            n += 1;
        }
        encoder.finish()?;
        Ok(n)
    });
    // Block until the encoder is open and reports its input format.
    let wants_p010 = fmt_rx.recv().unwrap_or(false);
    tracing::info!(
        "fisheye_export (zero-copy): encode input = {}",
        if wants_p010 { "P010 (GPU compose → NVENC, no swscale)" } else { "RGBA64 (swscale)" }
    );

    // Main thread — GPU work. import → project → color → compose → readback.
    let t_start = std::time::Instant::now();
    let mut frame_idx: u64 = 0;
    let mut main_err: Option<Error> = None;
    // Per-stage wall-clock accountancy (which pipeline stage is the floor?).
    let (mut t_recv, mut t_gpu, mut t_send) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    // Sub-split of the GPU stage: compute (project+compose) vs readback copy
    // — tells us whether a GPU-resident encode (eliminating the readback)
    // would actually help, or whether the projection compute is the floor.
    let (mut t_gpuexec, mut t_readback) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );

    loop {
        let r0 = std::time::Instant::now();
        let sp = match pair_rx.recv() { Ok(s) => s, Err(_) => break };
        t_recv += r0.elapsed();
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("fisheye_export (zero-copy): cancelled at frame {}", frame_idx);
            break;
        }
        if sp.pts_s.is_finite() && sp.pts_s >= t_out {
            tracing::info!("fisheye_export (zero-copy): hit trim_out @ {:.3}s", sp.pts_s);
            break;
        }

        // PTS-based stab lookup (same logic as preview + portable path).
        let stab_idx = if sp.pts_s.is_finite() && sp.pts_s >= 0.0 {
            (sp.pts_s / dt).round() as usize
        } else { frame_idx as usize };
        let rot = stab_rotations
            .as_ref()
            .and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);
        let (rot_left, rot_right) = if cfg.view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = cfg.view_adjust.per_eye_matrices();
            (
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };

        // Per-row rolling-shutter matrices (only when stabilizing). Same
        // lens_a mount quat for both eyes — matches the preview.
        let rs_rows: Option<Vec<f32>> = if cfg.stabilize {
            dji_osv_imu.as_ref().and_then(|osv| {
                crate::dji_imu::compute_per_row_quaternions_for_frame(
                    osv, stab_idx, readout_s, src_h, cfg.fps,
                )
            }).map(|q| crate::dji_imu::pack_per_row_camera_matrices(&q, lens_a_mount))
        } else { None };

        // All GPU work for this frame, errors captured so we can shut the
        // pipeline down cleanly rather than unwinding past the join.
        let g0 = std::time::Instant::now();
        let readback: Result<EncFrame> = (|| {
            // Import both eyes (single-plane RGBA16 aliasing the D3D11 convert).
            let l_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.left) };
            let r_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.right) };

            // KB projection (RS variant when stabilizing). Slots 30/31 keep
            // the export's cached output textures distinct from preview 0/1.
            let (left16, right16) = if let Some(rs) = rs_rows.as_deref() {
                (
                    pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs, 31,
                    )?,
                )
            } else {
                (
                    pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, 31,
                    )?,
                )
            };

            // 16-bit color stack per eye.
            let left_g = pipeline.apply_color_stack_per_eye_16(
                &left16, cfg.eye_w, cfg.eye_h, &color_plan,
            )?;
            let right_g = pipeline.apply_color_stack_per_eye_16(
                &right16, cfg.eye_w, cfg.eye_h, &color_plan,
            )?;
            let l_final = left_g.as_ref().unwrap_or(&left16);
            let r_final = right_g.as_ref().unwrap_or(&right16);
            if wants_p010 {
                // GPU does RGB→YUV + chroma subsample; read back both P010
                // planes (≈half the RGBA64 bytes) → NVENC consumes directly,
                // no CPU swscale (the encode-stage bottleneck).
                // Video-range here (false): the readback path's encoder
                // (create_with_bit_depth) tags AVCOL_RANGE_MPEG, so the data
                // must be video-range to match. (The GPU-resident path now
                // also uses video-range — both export paths agree.)
                let (y_tex, uv_tex) = pipeline.compose_sbs_to_p010_textures(
                    l_final, r_final, cfg.eye_w, cfg.eye_h, false,
                )?;
                // Time the two readbacks separately (NO extra poll — that
                // serializes the pipeline). The Y readback's internal poll
                // drains the GPU, so the UV readback runs GPU-idle = pure copy
                // cost. That tells us what a GPU-resident encode would save.
                let ry = std::time::Instant::now();
                let y = pipeline.read_texture_planar(&y_tex, sbs_w, sbs_h, 2)?;
                t_gpuexec += ry.elapsed(); // GPU-exec + Y copy
                let ruv = std::time::Instant::now();
                let uv = pipeline.read_texture_planar(&uv_tex, sbs_w / 2, sbs_h / 2, 4)?;
                t_readback += ruv.elapsed(); // pure UV copy (GPU idle)
                Ok(EncFrame::P010 { y, uv })
            } else {
                let sbs_tex_16 = pipeline.compose_sbs_textures_16(
                    l_final, r_final, cfg.eye_w, cfg.eye_h,
                )?;
                Ok(EncFrame::Rgba64(
                    pipeline.read_texture_rgba64(&sbs_tex_16, sbs_w, sbs_h)?
                ))
            }
        })();

        // The RGB48 readback drained the GPU (map_async + Maintain::Wait), so
        // this frame's imported eye textures are no longer in flight — drop
        // the D3D11 share now (frees the import + the converted texture).
        drop(sp);
        t_gpu += g0.elapsed();

        let frame = match readback {
            Ok(r) => r,
            Err(e) => { main_err = Some(e); break; }
        };
        // Hand off to the encode thread. A send error means the encoder
        // failed to open (frame_rx dropped) — stop; the join surfaces why.
        let s0 = std::time::Instant::now();
        if frame_tx.send(frame).is_err() { break; }
        t_send += s0.elapsed();

        frame_idx += 1;
        if frame_idx % 30 == 0 {
            let n = frame_idx as u32;
            tracing::info!(
                "zc export perf: f={} avg/frame: recv_wait={:?} gpu+readback={:?} send_wait={:?} \
                 [readY(gpu+copy)={:?} readUV(pure copy)={:?}]",
                frame_idx, t_recv / n, t_gpu / n, t_send / n, t_gpuexec / n, t_readback / n,
            );
        }
        let elapsed = t_start.elapsed().as_secs_f32().max(1e-3);
        progress_cb(ExportProgress {
            frame_idx,
            total_frames: total_frames_to_write,
            fps_avg: frame_idx as f32 / elapsed,
        });
    }

    // Tear down: close the encode feed (encoder.finish runs), stop decode,
    // then join both. Order matters only in that both senders/receivers must
    // drop so the threads can exit their recv loops.
    drop(frame_tx);
    drop(pair_rx);
    let enc_result = encode_handle.join();
    let _ = decode_handle.join();

    if let Some(e) = main_err {
        return Err(e);
    }
    match enc_result {
        Ok(Ok(_n)) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(Error::Ffmpeg("zc export: encode thread panicked".into())),
    }

    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path)?;
    finalize_metadata(
        &cfg.output_path,
        cfg.inject_youtube_vr180,
        cfg.inject_apmp,
        cfg.apmp_baseline_mm,
    )?;
    tracing::info!(
        "fisheye_export (zero-copy d3d11, pipelined): done, {} frames in {:.2?}",
        frame_idx, t_start.elapsed()
    );
    Ok(())
}

// ── Fast path: GPU-resident OSV → CUDA → NVENC (Windows, opt-in) ───────
//
// The endgame of the export optimization: the composited P010 frame never
// leaves VRAM. Decode (d3d11va) → import → project → color → compose to P010
// plane textures (wgpu/Vulkan) → copy into exportable linear images shared
// with CUDA → intra-VRAM `cuMemcpy2DAsync` (DtoD) into NVENC's CUDA frame →
// `hevc_nvenc`. Eliminates the ~24 ms/frame CPU readback that caps the
// readback path at ~23 fps at 8K. Modeled on slr-studio-neo's proven
// CUDA↔Vulkan interop, adapted for wgpu 29 (linear image via `texture_from_raw`
// since wgpu-hal 29 dropped Vulkan `buffer_from_raw`).
//
// Gated behind `VR180_GPU_RESIDENT`; NVENC + 10-bit + HalfEquirect only.
// Single-threaded CUDA (the primary context is current on this thread);
// decode is inline for v1. Errors propagate (the caller already committed).
#[cfg(target_os = "windows")]
#[allow(clippy::too_many_arguments)]
fn export_fisheye_osv_gpu_resident(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    ctx: crate::interop_windows::VulkanImportCtx,
    mut iter: crate::fisheye_decode::D3d11SharedDualStreamIter,
    progress_cb: &mut impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    use crate::nvenc_cuda::{CudaNvencEncoder, SharedP010Frame};

    // Bind the device-0 primary CUDA context to THIS thread; ffmpeg's CUDA
    // hwdevice + our external-memory imports both share it.
    let _cuda = cudarc::driver::CudaDevice::new(0)
        .map_err(|e| Error::Ffmpeg(format!("cuda primary ctx: {e:?}")))?;

    let (src_w, src_h) = iter.eye_dims();
    let (nw, nh) = iter.native_dims();
    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    tracing::info!(
        "fisheye_export (GPU-resident): work {}x{} (native {}x{}) → {}x{} SBS → NVENC(CUDA)",
        src_w, src_h, nw, nh, sbs_w, sbs_h
    );

    let dji_osv_imu: Option<vr180_fisheye::DjiOsvImu> =
        match crate::decode::extract_dji_meta_stream(&cfg.source_path) {
            Ok(blob) => vr180_fisheye::DjiOsvImu::parse(&blob).ok(),
            Err(_) => None,
        };
    let (calib_left, calib_right) = resolve_calib_pair(&cfg, src_w, src_h, dji_osv_imu.as_ref());

    let total_clip_frames = crate::decode::probe_video(&cfg.source_path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)
        .max(1);
    let stab_rotations: Option<Vec<EquirectRotation>> = if cfg.stabilize {
        dji_osv_imu.as_ref().and_then(|osv| {
            let max_corr = if cfg.dji_max_corr_deg > 0.0 { cfg.dji_max_corr_deg } else { f32::INFINITY };
            crate::dji_imu::compute_dji_stabilization(
                osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps, cfg.dji_responsiveness,
            ).ok().map(|s| s.per_frame)
        })
    } else { None };

    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    if t_in > 0.001 { iter.seek(t_in)?; }
    let t_out = cfg.trim_out_s.map(|t| t.max(t_in + 1e-3)).unwrap_or(f64::INFINITY);
    let total_frames_to_write = if t_out.is_finite() {
        ((t_out - t_in) * cfg.fps as f64).round() as u64
    } else {
        (total_clip_frames as f64 - t_in * cfg.fps as f64).round().max(0.0) as u64
    };

    let video_tmp = video_only_temp_path(&cfg.output_path);
    // Ring of shared P010 frames: the main thread composes frame N+k while the
    // encode thread is still feeding NVENC frame N. RING slots + a bounded
    // channel of depth RING-2 keep main from overwriting an in-flight slot.
    const RING: usize = 4;
    let ring: Vec<SharedP010Frame> = (0..RING)
        .map(|_| SharedP010Frame::new(&ctx, &pipeline.device, sbs_w, sbs_h))
        .collect::<Result<Vec<_>>>()?;
    tracing::info!("fisheye_export (GPU-resident): {RING}-slot shared P010 ring ready");

    // Encode thread owns the NVENC(CUDA) encoder (created here so its ffmpeg
    // CUDA hwdevice + frame pool live on one thread). It receives CUDA plane
    // pointers (Copy values, valid in the shared device-0 primary context) and
    // runs the DtoD + send_frame — overlapping NVENC's ~30 ms/frame at 8K with
    // the main thread's ~4 ms GPU compose. A create failure surfaces at join.
    struct EncMsg { y_ptr: u64, y_pitch: usize, uv_ptr: u64, uv_pitch: usize }
    let (enc_tx, enc_rx) = std::sync::mpsc::sync_channel::<EncMsg>(RING - 2);
    let (efps, ebr, etmp) = (cfg.fps, cfg.bitrate_kbps, video_tmp.clone());
    let encode_handle = std::thread::spawn(move || -> Result<u64> {
        let _cuda = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| Error::Ffmpeg(format!("encode-thread cuda ctx: {e:?}")))?;
        let mut encoder = CudaNvencEncoder::new(&etmp, sbs_w, sbs_h, efps, ebr)?;
        let mut n: u64 = 0;
        while let Ok(m) = enc_rx.recv() {
            unsafe { encoder.encode_cuda_planes(m.y_ptr, m.y_pitch, m.uv_ptr, m.uv_pitch)?; }
            n += 1;
        }
        encoder.finish()?;
        Ok(n)
    });
    tracing::info!("fisheye_export (GPU-resident): NVENC(CUDA) encode thread up");

    let dt = 1.0 / cfg.fps as f64;
    let color_plan = cfg.color_stack.clone();
    let readout_s = crate::dji_imu::dji_osmo_readout_ms_for_fps(cfg.fps) / 1000.0;
    let lens_a_mount = dji_osv_imu.as_ref()
        .and_then(|o| o.lens_a.mount_quat_xyzw)
        .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);

    // Decode sub-thread (D3D11/NVDEC only — never touches CUDA/wgpu), so the
    // ~10 ms decode overlaps the GPU compose + NVENC encode on this thread.
    use crate::fisheye_decode::SharedFisheyePair;
    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<SharedFisheyePair>(3);
    let cancel_dec = cancel.clone();
    let mut iter = iter;
    let decode_handle = std::thread::spawn(move || {
        loop {
            if cancel_dec.load(Ordering::SeqCst) { break; }
            match iter.next_pair() {
                Ok(Some(p)) => { if pair_tx.send(p).is_err() { break; } }
                Ok(None) => break,
                Err(e) => { tracing::warn!("gpu-resident decode: {e}"); break; }
            }
        }
    });

    let t_start = std::time::Instant::now();
    let mut frame_idx: u64 = 0;
    let (mut t_decode, mut t_gpu, mut t_enc) = (
        std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO,
    );

    loop {
        let d0 = std::time::Instant::now();
        let sp = match pair_rx.recv() { Ok(s) => s, Err(_) => break };
        t_decode += d0.elapsed();
        if cancel.load(Ordering::SeqCst) { break; }
        if sp.pts_s.is_finite() && sp.pts_s >= t_out { break; }
        let g0 = std::time::Instant::now();

        let stab_idx = if sp.pts_s.is_finite() && sp.pts_s >= 0.0 {
            (sp.pts_s / dt).round() as usize
        } else { frame_idx as usize };
        let rot = stab_rotations.as_ref().and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);
        let (rot_left, rot_right) = if cfg.view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = cfg.view_adjust.per_eye_matrices();
            (
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                EquirectRotation(crate::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };
        let rs_rows: Option<Vec<f32>> = if cfg.stabilize {
            dji_osv_imu.as_ref().and_then(|osv| {
                crate::dji_imu::compute_per_row_quaternions_for_frame(
                    osv, stab_idx, readout_s, src_h, cfg.fps,
                )
            }).map(|q| crate::dji_imu::pack_per_row_camera_matrices(&q, lens_a_mount))
        } else { None };

        // Import + project + color (same as the readback path).
        let l_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.left) };
        let r_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.right) };
        let (left16, right16) = if cfg.projection == FisheyeExportProjection::Fisheye {
            // Normalized fisheye SBS output (slots 32/33 so the cache doesn't
            // collide with the equirect slots). RS-corrected when stabilizing —
            // same (projection, rs) split as the macOS p010 export path.
            if let Some(rs) = rs_rows.as_deref() {
                (
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs, 32)?,
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs, 33)?,
                )
            } else {
                (
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, 32)?,
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, 33)?,
                )
            }
        } else if let Some(rs) = rs_rows.as_deref() {
            (
                pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                    &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs, 30)?,
                pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                    &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs, 31)?,
            )
        } else {
            (
                pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                    &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, 30)?,
                pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                    &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, 31)?,
            )
        };
        let left_g = pipeline.apply_color_stack_per_eye_16(&left16, cfg.eye_w, cfg.eye_h, &color_plan)?;
        let right_g = pipeline.apply_color_stack_per_eye_16(&right16, cfg.eye_w, cfg.eye_h, &color_plan)?;
        let l_final = left_g.as_ref().unwrap_or(&left16);
        let r_final = right_g.as_ref().unwrap_or(&right16);

        // Diagnostic: dump the pre-encode graded RGB (the "preview" pixels,
        // before any YCbCr conversion) for one frame, so we can compare it to
        // the decoded output and isolate any color-roundtrip error.
        if std::env::var("VR180_DUMP_PREENCODE").ok().and_then(|s| s.parse::<u64>().ok()) == Some(frame_idx) {
            if let Ok(sbs16) = pipeline.compose_sbs_textures_16(l_final, r_final, cfg.eye_w, cfg.eye_h) {
                if let Ok(rgba) = pipeline.read_texture_rgba64(&sbs16, sbs_w, sbs_h) {
                    let p = cfg.output_path.with_extension("preencode.rgba64");
                    let _ = std::fs::write(&p, &rgba);
                    tracing::info!("VR180_DUMP_PREENCODE: wrote {}x{} RGBA64 → {}", sbs_w, sbs_h, p.display());
                }
            }
        }

        // Compose to P010 plane textures, then copy into this ring slot's
        // CUDA-shared linear images. submit + poll(Wait) orders the GPU
        // produce before the encode thread's CUDA DtoD reads the slot.
        let sh = &ring[frame_idx as usize % RING];
        // Video-range (limited) Rec.709 YCbCr — the distribution standard,
        // consistent with the readback/libx265 paths and the source. Tagged
        // AVCOL_RANGE_MPEG (see CudaNvencEncoder). A compliant decoder expands
        // it back to the grade the gamma-correct preview shows. (The old
        // full-range here was a band-aid for the too-dark preview, since fixed
        // in the egui display path — see app.rs `preview_view_format`.)
        let (y_opt, uv_opt) = pipeline.compose_sbs_to_p010_textures(l_final, r_final, cfg.eye_w, cfg.eye_h, false)?;
        {
            let mut enc = pipeline.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_resident_copy_to_shared"),
            });
            let copy = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, w: u32, h: u32| {
                enc.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyTextureInfo { texture: dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            };
            copy(&mut enc, &y_opt, sh.y_texture(), sbs_w, sbs_h);
            copy(&mut enc, &uv_opt, sh.uv_texture(), sbs_w / 2, sbs_h / 2);
            pipeline.queue.submit(Some(enc.finish()));
        }
        let _ = pipeline.device.poll(wgpu::PollType::wait_indefinitely());
        t_gpu += g0.elapsed();

        // Hand the (Copy) CUDA plane pointers to the encode thread. The
        // bounded channel paces us to NVENC's rate; the GPU compose above
        // overlapped this slot's predecessor's encode.
        let e0 = std::time::Instant::now();
        let (y_ptr, y_pitch) = sh.y_cuda();
        let (uv_ptr, uv_pitch) = sh.uv_cuda();
        if enc_tx.send(EncMsg { y_ptr, y_pitch, uv_ptr, uv_pitch }).is_err() {
            break; // encode thread died — surfaced at join
        }
        t_enc += e0.elapsed();

        drop(sp);
        frame_idx += 1;
        if frame_idx % 30 == 0 {
            let n = frame_idx as u32;
            tracing::info!(
                "GPU-resident export: {} frames, {:.1} fps | recv_wait={:?} gpu+poll={:?} enc_send_wait={:?}",
                frame_idx, frame_idx as f32 / t_start.elapsed().as_secs_f32().max(1e-3),
                t_decode / n, t_gpu / n, t_enc / n,
            );
        }
        progress_cb(ExportProgress {
            frame_idx,
            total_frames: total_frames_to_write,
            fps_avg: frame_idx as f32 / t_start.elapsed().as_secs_f32().max(1e-3),
        });
    }

    drop(pair_rx);
    let _ = decode_handle.join();
    drop(enc_tx); // end the encode thread's recv loop → it flushes + finishes
    let enc_result = encode_handle.join();
    drop(ring);   // free shared frames only AFTER the encoder is done reading
    match enc_result {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(Error::Ffmpeg("gpu-resident encode thread panicked".into())),
    }
    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path)?;
    finalize_metadata(&cfg.output_path, cfg.inject_youtube_vr180, cfg.inject_apmp, cfg.apmp_baseline_mm)?;
    tracing::info!(
        "fisheye_export (GPU-resident): done, {} frames in {:.2?} ({:.1} fps)",
        frame_idx, t_start.elapsed(), frame_idx as f32 / t_start.elapsed().as_secs_f32().max(1e-3)
    );
    Ok(())
}

// (Old CPU SBS compose removed — replaced by
// Device::compose_sbs_textures + read_texture_rgb8 on the GPU.)

/// Per-eye calibration resolver. Same logic as
/// `vr180-gui::decoder::resolve_fisheye_calib_pair`, lifted here so the
/// CLI / headless export doesn't pull in the GUI crate.
fn resolve_calib_pair(
    cfg: &FisheyeExportConfig,
    src_w: u32,
    src_h: u32,
    osv: Option<&vr180_fisheye::DjiOsvImu>,
) -> (FisheyeCalib, FisheyeCalib) {
    use vr180_fisheye::presets;

    let preset = if !cfg.fisheye_preset.is_empty() && cfg.fisheye_preset != "Auto" {
        presets::find(&cfg.fisheye_preset)
    } else {
        None
    }.unwrap_or_else(|| {
        let auto_name = match cfg.source_kind {
            SourceKind::DjiOsv        => "DJI Osmo 360",
            SourceKind::BlackmagicRaw => "Blackmagic Pyxis 12K",
            _                         => "Custom",
        };
        presets::find(auto_name).unwrap_or(&presets::presets()[7])
    });

    // Preset-derived auto fx (override off / protobuf missing fx).
    let fx_auto = if preset.calib.fx > 0.0 && preset.calib.calib_w > 0 {
        (preset.calib.fx as f32) * (src_w as f32) / (preset.calib.calib_w as f32)
    } else {
        let half = (preset.default_fov_deg as f32).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };
    let preset_k = [preset.calib.k[0] as f32, preset.calib.k[1] as f32,
                    preset.calib.k[2] as f32, preset.calib.k[3] as f32];
    let fx_from_fov = |fov_deg: f32| -> f32 {
        let half = fov_deg.max(1.0).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };

    // For OSV, prefer the per-lens protobuf calibration. After the DJI
    // iter's default swap: left = Lens B, right = Lens A.
    let (calib_l, calib_r) = match (cfg.source_kind, osv) {
        (SourceKind::DjiOsv, Some(imu)) => {
            let scale_x = imu.lens_b.width.map(|w| (src_w as f32) / w).unwrap_or(1.0);
            let scale_y = imu.lens_b.height.map(|h| (src_h as f32) / h).unwrap_or(1.0);
            let eye = |lens: &vr180_fisheye::DjiLensCalib,
                       ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4], km5: f32,
                       km_p: [f32; 2]|
                -> FisheyeCalib
            {
                let (fx, fy, cx, cy, k, k5) = if ov {
                    let fx = fx_from_fov(fov);
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx, fx, cxn * src_w as f32, cyn * src_h as f32, k, km5)
                } else {
                    // Auto: full per-lens factory calibration from the OSV
                    // protobuf — fx/fy (→FOV), cx/cy, the KB k1–k4 AND the 5th
                    // radial coeff k5 (field 15). Must match the GUI's
                    // resolve_fisheye_calib_pair exactly so export == preview.
                    // k is dimensionless (no scale); missing → preset / 0.
                    let cx = lens.cx.map(|v| v * scale_x).unwrap_or(src_w as f32 * 0.5);
                    let cy = lens.cy.map(|v| v * scale_y).unwrap_or(src_h as f32 * 0.5);
                    let fx = lens.fx.map(|v| v * scale_x).unwrap_or(fx_auto);
                    let fy = lens.fy.map(|v| v * scale_y).unwrap_or(fx);
                    let k = match (lens.k1, lens.k2, lens.k3, lens.k4) {
                        (Some(a), Some(b), Some(c), Some(d)) => [a, b, c, d],
                        _ => preset_k,
                    };
                    (fx, fy, cx, cy, k, lens.k5.unwrap_or(0.0))
                };
                let mut calib =
                    FisheyeCalib::new_pure_kb(fx, fy, cx, cy, k, src_w as f32, src_h as f32);
                calib.k5 = k5;
                // Tangential: Auto from file, override uses the manual [p1,p2]
                // (km_p, GUI-seeded from the file). Must match the preview path.
                calib.p1 = if ov { km_p[0] } else { lens.p1.unwrap_or(0.0) };
                calib.p2 = if ov { km_p[1] } else { lens.p2.unwrap_or(0.0) };
                calib
            };
            // Calib follows the STREAM: with the user swap on, the left
            // output carries Lens A → use Lens A's calib (matches the
            // GUI resolver so export == preview).
            let (lens_l, lens_r) = if cfg.fisheye_swap_eyes {
                (&imu.lens_a, &imu.lens_b)
            } else {
                (&imu.lens_b, &imu.lens_a)
            };
            (
                eye(lens_l, cfg.fisheye_override_left, cfg.fisheye_fov_deg_left,
                    cfg.fisheye_cx_norm_left, cfg.fisheye_cy_norm_left, cfg.fisheye_k_left,
                    cfg.fisheye_k5_left, cfg.fisheye_p_left),
                eye(lens_r, cfg.fisheye_override_right, cfg.fisheye_fov_deg_right,
                    cfg.fisheye_cx_norm_right, cfg.fisheye_cy_norm_right, cfg.fisheye_k_right,
                    cfg.fisheye_k5_right, cfg.fisheye_p_right),
            )
        }
        _ => {
            // Non-OSV: Hermite-default constructor, per-eye.
            let r_max = (src_w.min(src_h) as f32) * 0.5;
            let eye = |ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4]| -> FisheyeCalib {
                let (fx, cx, cy, k) = if ov {
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx_from_fov(fov), cxn * src_w as f32, cyn * src_h as f32, k)
                } else {
                    (fx_auto, src_w as f32 * 0.5, src_h as f32 * 0.5, preset_k)
                };
                FisheyeCalib::new(fx, fx, cx, cy, k, src_w as f32, src_h as f32, r_max)
            };
            (
                eye(cfg.fisheye_override_left, cfg.fisheye_fov_deg_left,
                    cfg.fisheye_cx_norm_left, cfg.fisheye_cy_norm_left, cfg.fisheye_k_left),
                eye(cfg.fisheye_override_right, cfg.fisheye_fov_deg_right,
                    cfg.fisheye_cx_norm_right, cfg.fisheye_cy_norm_right, cfg.fisheye_k_right),
            )
        }
    };

    // Equidistant FISHEYE output target: set the output half-FOV so the
    // shaders map the inscribed circle of the square frame to a clean
    // FISHEYE_OUT_FULL_FOV_DEG fisheye. (The equirect target keeps the
    // default 90° horizontal half-FOV.)
    if cfg.projection == FisheyeExportProjection::Fisheye {
        let hfov = (FISHEYE_OUT_FULL_FOV_DEG * 0.5).to_radians();
        (calib_l.with_output_hfov(hfov), calib_r.with_output_hfov(hfov))
    } else {
        (calib_l, calib_r)
    }
}

// (CPU compose test removed — GPU compose has no per-byte invariants
// to assert in isolation; correctness validated by an end-to-end
// export run.)

/// Build a per-eye `Rgba16Unorm` half-equirect (or fisheye pass-through)
/// for the 10-bit export paths. Honors `projection`:
///   - HalfEquirect → run KB → equirect projection on the GPU.
///   - Fisheye → upload the raw fisheye eyes as Rgba16Unorm (8-bit
///     source widened to 16-bit by replicating high byte to low byte;
///     no precision is gained, but the texture format matches the
///     P010 compose path).
fn build_eye_eq_16(
    pipeline: &Device,
    pair: &FisheyePair,
    src_w: u32, src_h: u32,
    eye_w: u32, eye_h: u32,
    rot_left: EquirectRotation,
    rot_right: EquirectRotation,
    calib_left: FisheyeCalib,
    calib_right: FisheyeCalib,
    projection: FisheyeExportProjection,
) -> Result<(wgpu::Texture, wgpu::Texture)> {
    match projection {
        FisheyeExportProjection::HalfEquirect => {
            let l = pipeline.project_fisheye_to_equirect_texture_16(
                &pair.left, src_w, src_h, eye_w, eye_h, rot_left, calib_left,
            )?;
            let r = pipeline.project_fisheye_to_equirect_texture_16(
                &pair.right, src_w, src_h, eye_w, eye_h, rot_right, calib_right,
            )?;
            Ok((l, r))
        }
        FisheyeExportProjection::Fisheye => {
            // 16-bit stabilized fisheye output. The projection helper
            // auto-widens 8-bit source bytes to 16-bit storage; 16-bit
            // sources copy verbatim.
            let l = pipeline.project_fisheye_to_fisheye_texture_16(
                &pair.left, pair.bit_depth, src_w, src_h, eye_w, eye_h,
                rot_left, calib_left,
            )?;
            let r = pipeline.project_fisheye_to_fisheye_texture_16(
                &pair.right, pair.bit_depth, src_w, src_h, eye_w, eye_h,
                rot_right, calib_right,
            )?;
            Ok((l, r))
        }
    }
}

/// Upload a packed Rgba8 buffer as an `Rgba8Unorm` texture with
/// TEXTURE_BINDING + COPY_SRC usage so it can flow downstream into the
/// compose pipeline. Used by the Fisheye pass-through projection.
fn upload_rgba8_texture(
    pipeline: &Device,
    rgba: &[u8],
    w: u32, h: u32,
    label: &str,
) -> Result<wgpu::Texture> {
    let tex = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    pipeline.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    Ok(tex)
}

/// Upload an RGB(A) buffer as an `Rgba16Unorm` texture. Accepts source
/// in 8-bit RGBA8 (widened by left-shift, low byte zero) or 16-bit
/// RGBA64LE (copied verbatim).
fn upload_rgba_as_rgba16(
    pipeline: &Device,
    src: &[u8],
    w: u32, h: u32,
    src_bit_depth: u8,
    label: &str,
) -> Result<wgpu::Texture> {
    let n_pixels = (w as usize) * (h as usize);
    let bytes_per_row = w * 8;
    let wide: std::borrow::Cow<[u8]> = if src_bit_depth >= 16 {
        std::borrow::Cow::Borrowed(src)
    } else {
        let mut out = Vec::with_capacity(n_pixels * 8);
        for px in src.chunks_exact(4) {
            out.push(0x00); out.push(px[0]);
            out.push(0x00); out.push(px[1]);
            out.push(0x00); out.push(px[2]);
            out.push(0xFF); out.push(0xFF);
        }
        std::borrow::Cow::Owned(out)
    };
    let tex = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    pipeline.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        &wide,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    Ok(tex)
}

/// Apply the color stack to an Rgba8 SBS readback buffer. Walks the
/// existing `apply_cdl` / `apply_lut3d` / `apply_color_grade` helpers on
/// the host (`Vec<u8>` in / out) — slow, only used by the 8-bit non-VT
/// fallback path.
fn apply_color_stack_rgb8(
    pipeline: &Device,
    plan: &ColorStackPlan,
    mut rgb: Vec<u8>,
    w: u32, h: u32,
) -> Result<Vec<u8>> {
    // Python order: CDL → LUT → temp/tint → saturation (this fallback has
    // no sharpen / mid-detail stage).
    if !plan.cdl.is_identity() {
        rgb = pipeline.apply_cdl(&rgb, w, h, plan.cdl)?;
    }
    if let Some((ref lut, intensity)) = plan.lut {
        rgb = pipeline.apply_lut3d(&rgb, w, h, lut, intensity)?;
    }
    // White balance (temperature / tint) — POST-LUT, matching Python.
    if plan.color_grade.has_white_balance() {
        rgb = pipeline.apply_color_grade(&rgb, w, h, plan.color_grade.white_balance_only())?;
    }
    // Saturation — last (post-LUT, post-WB).
    if plan.color_grade.has_saturation() {
        rgb = pipeline.apply_color_grade(&rgb, w, h, plan.color_grade.saturation_only())?;
    }
    Ok(rgb)
}
