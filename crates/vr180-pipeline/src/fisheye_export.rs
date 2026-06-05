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
    /// Raw fisheye pass-through — no projection. Each source fisheye
    /// eye copied straight into the SBS output. Stabilization is
    /// ignored (a rotation in 3D world space doesn't map cleanly to a
    /// fisheye pixel transform without resampling).
    Fisheye,
}

impl Default for FisheyeExportProjection {
    fn default() -> Self { Self::HalfEquirect }
}

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
                        osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps,
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
        H265Encoder::create_zero_copy_vt(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    } else if use_zero_copy_p010 {
        H265Encoder::create_zero_copy_vt_p010(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    } else {
        H265Encoder::create_with_bit_depth(
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
                osv, total_clip_frames, max_corr, cfg.dji_smooth_ms, cfg.fps,
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
    match (cfg.source_kind, osv) {
        (SourceKind::DjiOsv, Some(imu)) => {
            let scale_x = imu.lens_b.width.map(|w| (src_w as f32) / w).unwrap_or(1.0);
            let scale_y = imu.lens_b.height.map(|h| (src_h as f32) / h).unwrap_or(1.0);
            let eye = |lens: &vr180_fisheye::DjiLensCalib,
                       ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4]|
                -> FisheyeCalib
            {
                let (fx, fy, cx, cy, k) = if ov {
                    let fx = fx_from_fov(fov);
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx, fx, cxn * src_w as f32, cyn * src_h as f32, k)
                } else {
                    let cx = lens.cx.map(|v| v * scale_x).unwrap_or(src_w as f32 * 0.5);
                    let cy = lens.cy.map(|v| v * scale_y).unwrap_or(src_h as f32 * 0.5);
                    let fx = lens.fx.map(|v| v * scale_x).unwrap_or(fx_auto);
                    let fy = lens.fy.map(|v| v * scale_y).unwrap_or(fx);
                    (fx, fy, cx, cy, preset_k)
                };
                FisheyeCalib::new_pure_kb(fx, fy, cx, cy, k, src_w as f32, src_h as f32)
            };
            (
                eye(&imu.lens_b, cfg.fisheye_override_left, cfg.fisheye_fov_deg_left,
                    cfg.fisheye_cx_norm_left, cfg.fisheye_cy_norm_left, cfg.fisheye_k_left),
                eye(&imu.lens_a, cfg.fisheye_override_right, cfg.fisheye_fov_deg_right,
                    cfg.fisheye_cx_norm_right, cfg.fisheye_cy_norm_right, cfg.fisheye_k_right),
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
        wgpu::ImageCopyTexture {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::ImageDataLayout {
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
        wgpu::ImageCopyTexture {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        &wide,
        wgpu::ImageDataLayout {
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
