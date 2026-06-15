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
    SbsFisheyeIter, SegmentedFisheyeIter,
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

/// Export audio-track selection for `.360` (which carries both a
/// stereo AAC and a 4-channel ambisonic track). Mirrors the GUI's
/// `AudioFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTrack {
    /// Stereo AAC passthrough (default, universal).
    Stereo,
    /// 4-channel ambisonic passthrough + SA3D spatial-audio metadata.
    Ambisonic,
    /// Re-encode the ambisonic to Apple Positional Audio Codec (Vision
    /// Pro head-tracked spatial). macOS-only; needs the apac_encode helper.
    Apac,
}

/// Complete export configuration. The fields mirror the GUI's
/// `Settings` plus the encoding knobs that have no equivalent in the
/// preview side.
#[derive(Debug, Clone)]
pub struct FisheyeExportConfig {
    pub source_path: PathBuf,
    /// Ordered GoPro segment chain (GS01…, GS02…). For a lone clip this
    /// is `[source_path]`. EAC export decodes + gyro-aggregates them as
    /// one continuous clip; other sources ignore it.
    pub segments: Vec<PathBuf>,
    pub output_path: PathBuf,
    pub source_kind: SourceKind,
    /// Which audio track to write (`.360` stereo / ambisonic / APAC).
    pub audio_track: AudioTrack,
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
    /// Temporal noise-reduction strength (0.0 = off, 1.0 = max), applied to
    /// the source fisheye frames before projection via the in-process
    /// VideoToolbox temporal filter ([`crate::vt_denoise`]). macOS-only;
    /// ignored (treated as off) on other platforms. Non-zero forces the
    /// portable CPU decode path (the zero-copy GPU path keeps frames on the
    /// GPU, out of reach of the CPU-side filter). Mirrors
    /// `Settings.denoise_strength`.
    pub denoise_strength: f32,
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

/// Available bytes on the filesystem holding `dir`, via `df -kP`
/// (POSIX one-line-per-fs output). Best-effort — `None` on any failure.
fn fs_avail_bytes(dir: &std::path::Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-kP").arg(dir).output().ok()?;
    if !out.status.success() { return None; }
    let text = String::from_utf8_lossy(&out.stdout);
    // Columns: Filesystem 1024-blocks Used Available Capacity Mounted-on.
    let avail_kb: u64 = text.lines().last()?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kb.saturating_mul(1024))
}

/// Move `src` → `dst`, handling a cross-device boundary (local scratch →
/// network output) by copy-then-remove when a plain rename can't span it.
fn move_file(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::remove_file(dst).ok(); // clear any stale target
    if std::fs::rename(src, dst).is_ok() { return Ok(()); }
    std::fs::copy(src, dst).map_err(Error::Io)?;
    std::fs::remove_file(src).ok();
    Ok(())
}

/// `final_out` with a `.video.tmp.<ext>` suffix — the next-to-output
/// fallback location for the working video.
fn video_tmp_next_to_output(final_out: &std::path::Path) -> std::path::PathBuf {
    let mut s = final_out.as_os_str().to_owned();
    s.push(".video.tmp");
    if let Some(ext) = final_out.extension() { s.push("."); s.push(ext); }
    std::path::PathBuf::from(s)
}

/// Where the encoder writes the video-only intermediate. Prefer LOCAL
/// scratch (`temp_dir`) so the encode-write AND the finalize re-mux read of
/// the multi-GB video stay OFF a network / slow output volume (e.g. an SMB
/// NAS — that read+write was the "export keeps growing after 100%" tail);
/// the final muxed file still lands at `final_out`. Falls back to
/// next-to-output when local scratch is short on space, or when
/// `VR180_TEMP_NEXT_TO_OUTPUT` is set.
fn video_only_temp_path(final_out: &std::path::Path) -> std::path::PathBuf {
    if std::env::var_os("VR180_TEMP_NEXT_TO_OUTPUT").is_some() {
        return video_tmp_next_to_output(final_out);
    }
    // Generous floor so a large export can't fill a near-full boot disk; the
    // fallback (next-to-output) is always safe.
    const MIN_LOCAL_FREE: u64 = 96 * 1024 * 1024 * 1024; // 96 GiB
    let tmp_dir = std::env::temp_dir();
    if fs_avail_bytes(&tmp_dir).map(|a| a >= MIN_LOCAL_FREE).unwrap_or(false) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        final_out.hash(&mut h);
        let stem = final_out.file_name().and_then(|s| s.to_str()).unwrap_or("vr180_export");
        let ext = final_out.extension().and_then(|s| s.to_str()).unwrap_or("mp4");
        tracing::info!("export: working video on local scratch {}", tmp_dir.display());
        return tmp_dir.join(format!("{stem}.{:x}.video.tmp.{ext}", h.finish()));
    }
    video_tmp_next_to_output(final_out)
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
/// Audio source for an EAC export: a temp ffconcat playlist spanning
/// the segment chain (multi-segment) or the lone source file. The
/// returned temp (when `Some`) must be deleted after the mux.
fn eac_audio_source(cfg: &FisheyeExportConfig) -> (PathBuf, Option<PathBuf>) {
    let tag = cfg.output_path.file_stem().and_then(|s| s.to_str())
        .unwrap_or("eac").to_string();
    match crate::audio::build_segment_playlist(&cfg.segments, &tag) {
        Some(pl) => (pl.clone(), Some(pl)),
        None => (cfg.source_path.clone(), None),
    }
}

/// Finalize EAC audio per `cfg.audio_track`. The video temp is consumed
/// (renamed or muxed away) regardless of branch.
/// - **Stereo**: passthrough-mux the source's stereo AAC (universal).
/// - **Ambisonic**: passthrough-mux the 4-channel ambisonic track + an
///   SA3D atom so VR players head-track it.
/// - **Apac** (macOS): extract the 4-ch ambisonic to WAV and let the
///   `apac_encode` helper re-encode it to Apple Positional Audio onto
///   the video (Vision Pro head-tracked spatial). Stereo fallback if the
///   helper / ambisonic track is missing.
/// Multi-segment: the audio source is the ffconcat playlist, so the
/// `(offset, dur)` window cuts from the unified chain timeline.
fn finalize_eac_audio(
    cfg: &FisheyeExportConfig,
    video_tmp: &std::path::Path,
    offset_s: f64,
    dur_s: f64,
) -> Result<()> {
    let (audio_src, audio_tmp) = eac_audio_source(cfg);
    let drop_tmp = |t: &Option<PathBuf>| { if let Some(p) = t { std::fs::remove_file(p).ok(); } };
    let out = &cfg.output_path;

    let stereo = |audio_tmp: &Option<PathBuf>| -> Result<()> {
        finalize_with_audio(&audio_src, video_tmp, out, offset_s, dur_s)?;
        drop_tmp(audio_tmp);
        Ok(())
    };

    match cfg.audio_track {
        AudioTrack::Stereo => stereo(&audio_tmp)?,

        AudioTrack::Ambisonic => match crate::audio::probe_ambisonic(&audio_src) {
            Ok(Some(info)) => {
                let n = crate::audio::mux_video_with_audio_stream(
                    &audio_src, video_tmp, out, offset_s, Some(dur_s),
                    Some(info.stream_index))?;
                drop_tmp(&audio_tmp);
                if n > 0 {
                    std::fs::remove_file(video_tmp).ok();
                    // Tag the 4-ch track as 1st-order AmbiX so VR players
                    // head-track it (YouTube/Quest). Non-fatal on failure.
                    if let Err(e) = crate::spherical_inject::inject_sa3d_ambix(out, info.channels) {
                        tracing::warn!("SA3D inject failed (audio still present): {e}");
                    }
                } else {
                    move_file(video_tmp, out)?;
                }
            }
            _ => {
                tracing::warn!("ambisonic requested but no 4-ch track found — stereo fallback");
                stereo(&audio_tmp)?;
            }
        },

        AudioTrack::Apac => {
            #[cfg(target_os = "macos")]
            {
                let wav = video_tmp.with_extension("ambi.wav");
                match crate::audio::extract_ambisonic_to_wav_window(
                    &audio_src, &wav, offset_s, Some(dur_s)) {
                    Ok(_) => {
                        let r = crate::helpers::spawn_apac_encode(
                            &wav, Some(video_tmp), out, 256_000);
                        std::fs::remove_file(&wav).ok();
                        match r {
                            Ok(()) => { std::fs::remove_file(video_tmp).ok(); drop_tmp(&audio_tmp); }
                            Err(e) => {
                                tracing::warn!("apac_encode failed ({e}) — stereo fallback");
                                stereo(&audio_tmp)?;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("APAC: ambisonic extract failed ({e}) — stereo fallback");
                        stereo(&audio_tmp)?;
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                tracing::warn!("APAC is macOS-only — stereo fallback");
                stereo(&audio_tmp)?;
            }
        }
    }
    Ok(())
}

fn finalize_with_audio(
    audio_src: &std::path::Path,
    video_tmp: &std::path::Path,
    final_out: &std::path::Path,
    // Trim window: where the exported video starts in SOURCE time, and how
    // long it is (frames_written / fps). The mux cuts + rebases the source
    // audio to match — see `mux_video_with_passthrough_audio`.
    audio_offset_s: f64,
    video_dur_s: f64,
) -> Result<()> {
    match crate::audio::mux_video_with_passthrough_audio(
        audio_src, video_tmp, final_out,
        audio_offset_s, Some(video_dur_s),
    ) {
        Ok(0) => {
            // Source had no audio — promote the temp video to the final
            // location (cross-device move if the temp is on local scratch).
            move_file(video_tmp, final_out)?;
            tracing::info!(
                "fisheye_export: no audio in source, moved video-only output to {}",
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
            // Mux failed — salvage the video-only temp next to the output
            // (the temp may be on local scratch) so the user keeps a file.
            let salvage = video_tmp_next_to_output(final_out);
            if salvage != video_tmp { let _ = move_file(video_tmp, &salvage); }
            tracing::warn!(
                "fisheye_export: audio mux failed: {e} — video-only file left at {}",
                salvage.display()
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
/// Whether this export can mux audio in ONE pass (inline with the encode,
/// straight to the final file — no `.video.tmp` + re-mux). True for plain
/// stereo passthrough; the `.360` ambisonic + APAC formats need post-encode
/// steps (SA3D atom / APAC re-encode) so they keep the temp+finalize path.
fn one_pass_audio_eligible(cfg: &FisheyeExportConfig) -> bool {
    !matches!(cfg.source_kind, SourceKind::GoProEac)
        || matches!(cfg.audio_track, AudioTrack::Stereo)
}

/// Build a per-segment opener for the portable fisheye export decode loop
/// — native resolution (no cap), bit-depth matched to the output codec.
/// Shared by the single-file path and the multi-segment
/// [`SegmentedFisheyeIter`] so a merged recording decodes identically.
fn fisheye_export_opener(
    kind: SourceKind,
    swap_eyes: bool,
    bit_depth: u8,
) -> Box<dyn FnMut(&std::path::Path) -> Result<Box<dyn FisheyePairIter>>> {
    let bd = if bit_depth >= 10 { 16u8 } else { 8u8 };
    Box::new(move |p: &std::path::Path| -> Result<Box<dyn FisheyePairIter>> {
        Ok(match kind {
            SourceKind::DjiOsv => Box::new(DualStreamFisheyeIter::new_with_options(
                p, crate::decode::HwDecode::Auto, 0, !swap_eyes, 0, bd)?),
            SourceKind::SbsFisheye => Box::new(
                SbsFisheyeIter::new(p, crate::decode::HwDecode::Auto, 0)?),
            SourceKind::BlackmagicRaw => {
                let info = vr180_braw::BrawInfo::probe(p)
                    .map_err(|e| Error::Ffmpeg(format!("braw probe: {e}")))?;
                let opts = vr180_braw::decoder::DecodeOptions::default();
                Box::new(BrawFisheyeIter::new(p, &info, &opts, 0)
                    .map_err(|e| Error::Ffmpeg(format!("braw start: {e}")))?)
            }
            other => return Err(Error::Ffmpeg(format!(
                "export_fisheye non-fisheye source: {other:?}"))),
        })
    })
}

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
        // VT-HEVC and VT-ProRes both consume our CVPixelBuffers through the
        // same AV_PIX_FMT_VIDEOTOOLBOX hw-frames mechanism — ProRes encode
        // is Apple's hardware ProRes engine, fed the P010 IOSurface
        // directly (its supported sw-format list includes p010le).
        let on_macos_vt = matches!(
            cfg.encoder,
            EncoderBackend::VideoToolbox | EncoderBackend::ProResVideoToolbox
        );
        // Zero-copy P010 decode → projection supports both half-equirect
        // and the stabilized-fisheye output. BOTH 8-bit and 10-bit H.265
        // output take this path: the OSV source is always 10-bit HEVC, so
        // VT decodes P010 regardless of the chosen output depth, and only
        // the final encode differs (P010 Main10 vs BGRA Main / ProRes).
        // This means 8-bit no longer pays the CPU-roundtrip decode the
        // general path uses (download → swscale → re-upload per frame),
        // which made 8-bit export slower than 10-bit.
        // Single-segment only: the zero-copy iterator opens one file. A
        // merged recording (cfg.segments.len() > 1) falls through to the
        // portable loop below, which chains segments via SegmentedFisheyeIter.
        let can_zero_copy_decode = matches!(cfg.source_kind, SourceKind::DjiOsv)
            && on_macos_vt
            && (cfg.bit_depth == 10 || cfg.bit_depth == 8)
            && cfg.segments.len() <= 1;
        // NB: denoise (cfg.denoise_strength > 0) STAYS on this fast path — the
        // zero-copy export denoises the P010 IOSurfaces on the GPU via
        // DenoisingZeroCopyIter, with no CPU bounce (≈3× faster than dropping
        // to the portable readback path).
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
            && cfg.segments.len() <= 1 // merged recording → portable segmented loop
            && crate::interop_windows::is_vulkan_backend(&pipeline.device)
            && pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010)
            && cfg.denoise_strength <= 0.0; // denoise needs CPU frames → portable path
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
    // Scope: OSV + HEVC(libx265/nvenc) + either projection + Vulkan + P010.
    // Anything else (ProRes, non-OSV, DX12, no P010) — or the
    // `VR180_EXPORT_FORCE_CPU` escape hatch — falls through to the portable
    // readback path below, untouched. Covers Fisheye too (parity with the
    // macOS p010 path) so libx265 / 8-bit / GPU-resident-fallback fisheye
    // exports keep RS + the zero-copy decode instead of dropping to the
    // portable path.
    #[cfg(target_os = "windows")]
    {
        let can_try = std::env::var_os("VR180_EXPORT_FORCE_CPU").is_none()
            && matches!(cfg.source_kind, SourceKind::DjiOsv)
            && matches!(cfg.encoder, EncoderBackend::Libx265 | EncoderBackend::HevcNvenc)
            && matches!(cfg.projection, FisheyeExportProjection::HalfEquirect | FisheyeExportProjection::Fisheye)
            && (cfg.bit_depth == 10 || cfg.bit_depth == 8)
            && cfg.segments.len() <= 1 // merged recording → portable segmented loop
            && crate::interop_windows::is_vulkan_backend(&pipeline.device)
            && pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010)
            && cfg.denoise_strength <= 0.0; // denoise needs CPU frames → portable path
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
    let mut decoder: Box<dyn FisheyePairIter> = if cfg.segments.len() > 1 {
        // Merged recording: chain the segments into one continuous timeline
        // (the IMU is aggregated the same way below, so stab indexes by
        // absolute frame across the whole export).
        let durations: Vec<f64> = cfg.segments.iter()
            .map(|p| crate::decode::probe_video(p).map(|pr| pr.duration_sec).unwrap_or(0.0))
            .collect();
        tracing::info!(
            "fisheye_export: {} segments, {:.1}s total — SegmentedFisheyeIter",
            cfg.segments.len(), durations.iter().sum::<f64>());
        Box::new(SegmentedFisheyeIter::new(
            &cfg.segments, &durations,
            fisheye_export_opener(cfg.source_kind, cfg.fisheye_swap_eyes, cfg.bit_depth))?)
    } else {
        let mut open = fisheye_export_opener(cfg.source_kind, cfg.fisheye_swap_eyes, cfg.bit_depth);
        open(&cfg.source_path)?
    };

    // ── Temporal noise reduction (macOS VideoToolbox) ────────────
    // Source-domain denoise, before projection — matches the Python app.
    // Wraps the iterator so each eye's frames stream through the in-process
    // VTTemporalNoiseFilter. The zero-copy decode paths above are already
    // gated off when `denoise_strength > 0`, so we always have CPU frames
    // here. Frame count/order are preserved, so stab + audio stay aligned.
    #[cfg(target_os = "macos")]
    if cfg.denoise_strength > 0.0 {
        if crate::vt_denoise::is_supported() {
            match crate::fisheye_decode::DenoisingFisheyeIter::new(decoder, cfg.denoise_strength) {
                Ok(d) => {
                    tracing::info!(
                        "fisheye_export: temporal NR ENGAGED (strength={:.2})",
                        cfg.denoise_strength
                    );
                    decoder = Box::new(d);
                }
                Err(e) => {
                    return Err(Error::Ffmpeg(format!("denoise init failed: {e}")));
                }
            }
        } else {
            tracing::warn!(
                "fisheye_export: denoise requested but VTTemporalNoiseFilter \
                 unsupported on this machine — exporting without NR"
            );
            // `decoder` was moved into the match arm only on the Ok path; on
            // this branch it's untouched and still bound.
        }
    }

    let (src_w, src_h) = decoder.eye_dims();

    // ── DJI: extract protobuf for per-eye calib + (optional) stab ─
    let dji_osv_imu: Option<vr180_fisheye::DjiOsvImu> = if matches!(
        cfg.source_kind, SourceKind::DjiOsv
    ) {
        // Single file → parse; merged recording → parse_multi (concatenate
        // each segment's protobuf so IMU indexes by absolute frame).
        let parsed: Result<vr180_fisheye::DjiOsvImu> = if cfg.segments.len() > 1 {
            let mut blobs = Vec::with_capacity(cfg.segments.len());
            let mut acc: Result<()> = Ok(());
            for p in &cfg.segments {
                match crate::decode::extract_dji_meta_stream(p) {
                    Ok(b) => blobs.push(b),
                    Err(e) => { acc = Err(e); break; }
                }
            }
            acc.and_then(|()| vr180_fisheye::DjiOsvImu::parse_multi(&blobs)
                .map_err(|e| Error::Ffmpeg(format!("dji parse_multi: {e}"))))
        } else {
            crate::decode::extract_dji_meta_stream(&cfg.source_path)
                .and_then(|blob| vr180_fisheye::DjiOsvImu::parse(&blob)
                    .map_err(|e| Error::Ffmpeg(format!("dji parse: {e}"))))
        };
        match parsed {
            Ok(imu) => Some(imu),
            Err(e) => { tracing::warn!("dji meta: {e}"); None }
        }
    } else { None };

    // ── Resolve per-eye calibrations ─────────────────────────────
    let (calib_left, calib_right) = resolve_calib_pair(
        &cfg, src_w, src_h, dji_osv_imu.as_ref(),
    );

    // ── Stab rotations (one per source frame) ────────────────────
    let total_clip_frames = if cfg.segments.len() > 1 {
        cfg.segments.iter()
            .filter_map(|p| crate::decode::probe_video(p).ok())
            .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
            .sum::<usize>().max(1)
    } else {
        crate::decode::probe_video(&cfg.source_path)
            .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
            .unwrap_or(0).max(1)
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
    //   HEVC bit_depth == 8  → BGRA IOSurface zero-copy (Main profile)
    //   HEVC bit_depth == 10 → P010 IOSurface zero-copy (Main10 profile)
    //   ProRes-VT            → P010 IOSurface zero-copy into Apple's
    //                          hardware ProRes engine (same buffers).
    //                     — GPU writes Y + UV planes directly into a
    //                     P010 CVPixelBuffer, no swscale, no readback.
    // Non-macOS / non-VT (libx265, prores_ks) → readback fallback.
    let on_macos_vt = cfg!(target_os = "macos")
        && matches!(cfg.encoder,
            EncoderBackend::VideoToolbox | EncoderBackend::ProResVideoToolbox);
    let is_prores_vt = cfg.encoder == EncoderBackend::ProResVideoToolbox;
    let use_zero_copy_bgra = on_macos_vt && cfg.bit_depth == 8 && !is_prores_vt;
    let use_zero_copy_p010 = on_macos_vt && (cfg.bit_depth == 10 || is_prores_vt);
    // One-pass: encode STRAIGHT to the final file and mux the source's
    // audio inline (no temp + re-mux). Otherwise write a local-scratch
    // video temp and mux audio onto it afterwards (ambisonic / APAC).
    // `video_tmp` aliases the final file in the one-pass case, so the
    // encoder ctors below need no change.
    let one_pass = one_pass_audio_eligible(&cfg);
    let video_tmp = if one_pass { cfg.output_path.clone() }
                    else { video_only_temp_path(&cfg.output_path) };
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
        // macOS/VideoToolbox-only zero-copy P010 path (HEVC Main10 or
        // ProRes-VT). Same gating.
        #[cfg(target_os = "macos")]
        {
            if is_prores_vt {
                H265Encoder::create_zero_copy_vt_prores(
                    &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.prores_profile,
                )?
            } else {
                H265Encoder::create_zero_copy_vt_p010(
                    &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
                )?
            }
        }
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

    // Inline audio passthrough — must attach before the first frame.
    let one_pass_audio_tmp = if one_pass {
        let (asrc, atmp) = eac_audio_source(&cfg);
        let dur = total_frames_to_write as f64 / cfg.fps as f64;
        if let Err(e) = encoder.attach_audio_passthrough(&asrc, t_in, dur) {
            tracing::warn!("one-pass audio attach failed: {e} — video-only output");
        }
        atmp
    } else { None };

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
    // Per-row rolling-shutter for the PORTABLE path — parity with the
    // zero-copy paths (macOS p010, Windows readback / GPU-resident). The
    // portable path is what libx265-on-macOS and ProRes-on-both-platforms
    // ride, so without this those exports kept jello under stabilization
    // while every zero-copy path was RS-corrected.
    let rs_enabled = cfg.stabilize
        && matches!(cfg.source_kind, SourceKind::DjiOsv)
        && dji_osv_imu.is_some();
    let readout_s = crate::dji_imu::dji_osmo_readout_ms_for_fps(cfg.fps) / 1000.0;
    if rs_enabled {
        tracing::info!(
            "fisheye_export: per-row RS on (readout {:.1} ms)",
            readout_s * 1000.0
        );
    }

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
        // Drop pre-trim frames: the iterators' seek lands on the keyframe
        // AT/BEFORE trim_in (no decode-forward), so the first pairs can be
        // up to a GOP early — without this the export gains pre-roll the
        // user trimmed away and drifts out of sync with the trimmed audio.
        if pair.pts_s.is_finite() && pair.pts_s < t_in - 0.5 * dt {
            continue;
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

        // Per-row RS matrices for this frame — identical CPU build to the
        // zero-copy paths. None (= legacy projection, byte-identical) when
        // not stabilizing or non-OSV.
        let rs_rows: Option<Vec<f32>> = if rs_enabled {
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

        // 8-bit per-eye projection, built lazily — ONLY the arms that
        // consume it call this (BGRA zero-copy + 8-bit readback). The
        // 10-bit arms project at 16 bits inside `build_eye_eq_16`; the old
        // code built these 8-bit textures unconditionally and threw them
        // away there (two wasted full-res projections per frame). The
        // (projection, rs) four-way matches every zero-copy path, so the
        // portable path now keeps per-row RS too.
        let project_8 = |rs: Option<&[f32]>| -> Result<(wgpu::Texture, wgpu::Texture)> {
            Ok(match (projection, rs) {
                (FisheyeExportProjection::HalfEquirect, Some(rs_buf)) => (
                    pipeline.project_fisheye_to_equirect_rs_texture(
                        &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_left, calib_left, rs_buf, 10,
                    )?,
                    pipeline.project_fisheye_to_equirect_rs_texture(
                        &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_right, calib_right, rs_buf, 11,
                    )?,
                ),
                (FisheyeExportProjection::HalfEquirect, None) => (
                    pipeline.project_fisheye_to_equirect_texture(
                        &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_left, calib_left, 10,
                    )?,
                    pipeline.project_fisheye_to_equirect_texture(
                        &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_right, calib_right, 11,
                    )?,
                ),
                (FisheyeExportProjection::Fisheye, Some(rs_buf)) => (
                    pipeline.project_fisheye_to_fisheye_rs_texture(
                        &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_left, calib_left, rs_buf,
                    )?,
                    pipeline.project_fisheye_to_fisheye_rs_texture(
                        &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_right, calib_right, rs_buf,
                    )?,
                ),
                (FisheyeExportProjection::Fisheye, None) => (
                    pipeline.project_fisheye_to_fisheye_texture(
                        &pair.left, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_left, calib_left,
                    )?,
                    pipeline.project_fisheye_to_fisheye_texture(
                        &pair.right, src_w, src_h, cfg.eye_w, cfg.eye_h,
                        rot_right, calib_right,
                    )?,
                ),
            })
        };

        if use_zero_copy_bgra {
            // 8-bit macOS zero-copy path. apply_color_stack_to_sbs_bgra
            // already supports plan ≠ identity; pass it directly so the
            // stack runs fused with the SBS compose into the IOSurface.
            #[cfg(target_os = "macos")]
            {
                let (left_tex, right_tex) = project_8(rs_rows.as_deref())?;
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
        } else if use_zero_copy_p010 {
            // 10-bit macOS zero-copy: project at 16-bit, GPU-write
            // both planes of a P010 IOSurface, VT consumes directly.
            #[cfg(target_os = "macos")]
            {
                let (left_tex_16, right_tex_16) = build_eye_eq_16(
                    &pipeline, &pair, src_w, src_h,
                    cfg.eye_w, cfg.eye_h,
                    rot_left, rot_right, calib_left, calib_right,
                    projection, rs_rows.as_deref(),
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
        } else if cfg.bit_depth == 10 {
            // 10-bit non-zero-copy: texture-resident 16-bit projection +
            // RGB48 readback. This is libx265 Main10 AND ProRes (both
            // platforms) — per-row RS rides build_eye_eq_16's rs param.
            let (left_tex_16, right_tex_16) = build_eye_eq_16(
                &*pipeline, &pair, src_w, src_h,
                cfg.eye_w, cfg.eye_h,
                rot_left, rot_right, calib_left, calib_right,
                projection, rs_rows.as_deref(),
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
        } else {
            // 8-bit non-zero-copy fallback. Color stack happens via the
            // legacy CPU roundtrip on the SBS readback below (acceptable
            // since this path is already CPU-bound).
            let (left_tex, right_tex) = project_8(rs_rows.as_deref())?;
            let sbs_tex = pipeline.compose_sbs_textures(
                &left_tex, &right_tex, cfg.eye_w, cfg.eye_h,
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
    if one_pass {
        // Audio already muxed inline — drop the playlist temp, if any.
        if let Some(tmp) = one_pass_audio_tmp { std::fs::remove_file(tmp).ok(); }
    } else {
        // Multi-segment uses an ffconcat playlist so the merged recording's
        // audio is one continuous track the trim window cuts from; a lone
        // file muxes its own. (`eac_audio_source` is segment-generic.)
        let (audio_src, audio_tmp) = eac_audio_source(&cfg);
        finalize_with_audio(&audio_src, &video_tmp, &cfg.output_path,
            t_in, frame_idx as f64 * dt)?;
        if let Some(tmp) = audio_tmp { std::fs::remove_file(tmp).ok(); }
    }
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

/// Per-eye stab/RS rotation pair for one frame, as the GUI's
/// `build_per_eye_frames` produces it: `((left_rot, left_rs), (right_rot, right_rs))`.
pub type EacPerEyeFrame = (
    (EquirectRotation, crate::gpu::EquirectRsParams),
    (EquirectRotation, crate::gpu::EquirectRsParams),
);

/// GoPro `.360` (EAC) → SBS half-equirect file export.
///
/// EAC is a different projection from the fisheye sources (`export_fisheye`),
/// so it gets its own per-frame core — decode dual HEVC → assemble each
/// lens's EAC cross → project cross→equirect per eye (with the per-frame
/// stab/RS the GUI computed) → optional view-adjust → color → compose SBS
/// → encode. Everything else (encoder backends, audio passthrough, VR180
/// metadata, trim, progress) is shared with the fisheye path.
///
/// `per_eye[absolute_frame_idx]` carries the stabilization rotations +
/// rolling-shutter params; the GUI owns that (GPMF parse + Settings), so
/// it's passed in pre-computed. Empty / short → identity (no stab).
///
/// 8-bit core: GoPro Max footage is 8-bit, so the cross/project/compose run
/// at 8-bit and the color stack applies on the SBS readback (same as the
/// fisheye 8-bit arm). The encoder still honors `cfg.bit_depth` (a 10-bit
/// codec just gets the 8-bit pixels widened by swscale — no precision lost
/// vs the source).
pub fn export_eac(
    pipeline: std::sync::Arc<Device>,
    cfg: FisheyeExportConfig,
    per_eye: Vec<EacPerEyeFrame>,
    mut progress_cb: impl FnMut(ExportProgress),
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    use vr180_core::eac::{assemble_lens_a, assemble_lens_b};

    tracing::info!(
        "export_eac: {} → {} ({}x{} @ {:.2} fps, {} kbps, {}-bit, {} stab frames)",
        cfg.source_path.display(), cfg.output_path.display(),
        cfg.eye_w * 2, cfg.eye_h, cfg.fps, cfg.bitrate_kbps, cfg.bit_depth,
        per_eye.len(),
    );

    // ── Fast path: zero-copy EAC → VT (macOS) ─────────────────────
    // Same architecture as the OSV zero-copy export: VT-decoded P010
    // IOSurfaces wrapped as wgpu textures → GPU cross assembly → GPU
    // EAC→equirect projection (stab + RS + view) → compose straight
    // into the encode IOSurface (BGRA for 8-bit Main, P010 for
    // 10-bit Main10 / ProRes-VT) → VT. No swscale, no CPU assembly,
    // no readback. libx265 / prores_ks fall through to the portable
    // CPU path below (same split as OSV).
    #[cfg(target_os = "macos")]
    {
        let on_vt = matches!(
            cfg.encoder,
            EncoderBackend::VideoToolbox | EncoderBackend::ProResVideoToolbox
        );
        if on_vt {
            return export_eac_zerocopy_vt(
                pipeline, cfg, per_eye, &mut progress_cb, cancel,
            );
        }
    }

    // ── Fast path: GPU-resident EAC → CUDA → NVENC (Windows, default) ────
    // The GoPro `.360` analog of the OSV GPU-resident path: NVDEC decodes
    // both EAC HEVC streams, D3D11 converts each P010→RGBA16, wgpu assembles
    // the two lens crosses + projects + colors + composes P010, and the frame
    // is fed to NVENC over CUDA — no CPU readback, no swscale, no CPU
    // `assemble_lens_*`. Same gate shape + safe fallback as the OSV path:
    // NVENC + 10-bit + either projection + single-segment + Vulkan + P010 +
    // no denoise; ANY failure falls through to the portable path below.
    #[cfg(target_os = "windows")]
    {
        let try_gpu_resident = std::env::var_os("VR180_NO_GPU_RESIDENT").is_none()
            && std::env::var_os("VR180_EXPORT_FORCE_CPU").is_none()
            && matches!(cfg.encoder, EncoderBackend::HevcNvenc)
            && cfg.bit_depth == 10
            && matches!(cfg.projection, FisheyeExportProjection::HalfEquirect | FisheyeExportProjection::Fisheye)
            && crate::interop_windows::is_vulkan_backend(&pipeline.device)
            && pipeline.device.features().contains(wgpu::Features::TEXTURE_FORMAT_P010)
            && cfg.denoise_strength <= 0.0; // denoise is macOS-only anyway
        if try_gpu_resident {
            let ctx = crate::interop_windows::VulkanImportCtx::from_wgpu(
                &pipeline.adapter, &pipeline.device,
            );
            // Segmented iterator chains a merged recording's GS01…/GS02…/… on
            // the fast path (globalized pts → stab index stays continuous).
            let iter = crate::fisheye_decode::SegmentedD3d11SharedStreamPairIter::new(&cfg.segments);
            match (ctx, iter) {
                (Some(ctx), Ok(iter)) => {
                    tracing::info!("export_eac: GPU-RESIDENT NVENC(CUDA) path ENGAGED");
                    match export_eac_gpu_resident(
                        Arc::clone(&pipeline), cfg.clone(), per_eye.clone(),
                        ctx, iter, &mut progress_cb, Arc::clone(&cancel),
                    ) {
                        Ok(()) => return Ok(()),
                        Err(e) => tracing::warn!(
                            "export_eac: GPU-resident path failed ({e}) — \
                             falling back to the portable path"
                        ),
                    }
                }
                (c, i) => tracing::warn!(
                    "export_eac: GPU-resident unavailable (vulkan_ctx={}, iter_ok={}) — \
                     falling through to portable path", c.is_some(), i.is_ok()
                ),
            }
        }
    }

    let mut decoder = crate::decode::SegmentedStreamPairIter::new(
        &cfg.segments, crate::decode::HwDecode::Auto, 0)?;
    let dims = decoder.dims();
    if !dims.is_valid() {
        return Err(Error::Ffmpeg(format!(
            "export_eac: invalid EAC layout (stream w={})", dims.stream_w)));
    }
    let cross_w = dims.cross_w();
    let cw = cross_w as usize;

    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    let video_tmp = video_only_temp_path(&cfg.output_path);
    let mut encoder = open_h265_encoder(
        &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        cfg.encoder, cfg.bit_depth, cfg.prores_profile,
    )?;

    // Trim via PRECISE seek: the iterator's seek decodes-and-discards
    // the keyframe run-in internally, so the first delivered pair is
    // exactly `trim_in_frame` (a late trim no longer decodes the whole
    // preceding clip — a 300 s trim-in used to cost ~8 min of decode).
    let dt = 1.0 / cfg.fps as f64;
    let trim_in_frame = cfg.trim_in_s.map(|t| (t.max(0.0) * cfg.fps as f64).round() as u32).unwrap_or(0);
    let trim_out_frame = cfg.trim_out_s.map(|t| (t * cfg.fps as f64).round() as u32);
    if trim_in_frame > 0 {
        decoder.seek(trim_in_frame as f64 * dt)?;
    }

    let color_plan = cfg.color_stack.clone();
    let color_any = color_plan.any_active();
    let view = cfg.view_adjust;
    let view_active = !view.is_identity();
    let (v_l, v_r) = view.per_eye_matrices();

    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];

    let t_start = std::time::Instant::now();
    let mut frame_idx: u32 = trim_in_frame; // absolute source frame index (post-seek)
    let mut written: u64 = 0;
    let total_est = trim_out_frame
        .map(|o| (o.saturating_sub(trim_in_frame)) as u64)
        .unwrap_or(per_eye.len() as u64);

    while let Some(pair) = decoder.next_pair()? {
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("export_eac: cancelled at frame {}", frame_idx);
            break;
        }
        if let Some(out_f) = trim_out_frame {
            if frame_idx >= out_f { break; }
        }

        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        // Per-eye stab/RS for this absolute frame (identity past the end).
        let ((mut rl, sl), (mut rr, sr)) = per_eye.get(frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
            ));
        // Compose view-adjust (pano ± stereo) AFTER stab — same convention
        // as the fisheye path. Left eye = Lens B, right eye = Lens A.
        if view_active {
            rl = EquirectRotation(crate::panomap::mat3_mul_row_major(&rl.0, &v_l));
            rr = EquirectRotation(crate::panomap::mat3_mul_row_major(&rr.0, &v_r));
        }

        let (left, right) = if matches!(cfg.projection, FisheyeExportProjection::Fisheye) {
            // Upload the RGB8 crosses once, then fisheye-project both eyes.
            (pipeline.project_cross_to_fisheye_texture(
                &cross_b, cross_w, cfg.eye_w, cfg.eye_h, rl, sl)?,
             pipeline.project_cross_to_fisheye_texture(
                &cross_a, cross_w, cfg.eye_w, cfg.eye_h, rr, sr)?)
        } else {
            (pipeline.project_cross_to_equirect_texture(
                &cross_b, cross_w, cfg.eye_w, cfg.eye_h, rl, sl)?,
             pipeline.project_cross_to_equirect_texture(
                &cross_a, cross_w, cfg.eye_w, cfg.eye_h, rr, sr)?)
        };
        let sbs = pipeline.compose_sbs_textures(&left, &right, cfg.eye_w, cfg.eye_h)?;
        let rgb = pipeline.read_texture_rgb8(&sbs, sbs_w, sbs_h)?;
        let graded = if color_any {
            apply_color_stack_rgb8(&pipeline, &color_plan, rgb, sbs_w, sbs_h)?
        } else { rgb };
        encoder.encode_frame(&graded)?;

        frame_idx += 1;
        written += 1;
        let elapsed = t_start.elapsed().as_secs_f32().max(1e-3);
        progress_cb(ExportProgress {
            frame_idx: written,
            total_frames: total_est,
            fps_avg: written as f32 / elapsed,
        });
    }

    encoder.finish()?;
    // Audio: trim window in source time = (trim_in, written·dt). Passthrough
    // mux copies the source's first audio track (GoPro `.360` carries AAC);
    // multi-segment uses an ffconcat playlist so the chain's audio is one
    // continuous timeline the global (trim_in, dur) window cuts from.
    finalize_eac_audio(
        &cfg, &video_tmp,
        trim_in_frame as f64 * dt, written as f64 * dt,
    )?;
    finalize_metadata(
        &cfg.output_path, cfg.inject_youtube_vr180, cfg.inject_apmp, cfg.apmp_baseline_mm,
    )?;
    tracing::info!("export_eac: done, {} frames in {:.2?}", written, t_start.elapsed());
    Ok(())
}

/// Zero-copy EAC → VT export (macOS). The hardware-accelerated arm of
/// [`export_eac`] — mirrors `export_fisheye_osv_zerocopy_p010`:
///
/// 1. `ZeroCopyStreamPairIter` — VT decodes both HEVC streams to P010
///    IOSurfaces, wrapped as wgpu textures (no `av_hwframe_transfer`).
/// 2. `nv12_to_eac_cross` ×2 — GPU cross assembly (replaces the CPU
///    `assemble_lens_*` + RGB repack of the portable path).
/// 3. `project_cross_texture_to_equirect_texture` ×2 — stab + RS +
///    view-adjust, same shader the preview uses.
/// 4. Compose + color directly into the encode IOSurface:
///    8-bit H.265 → BGRA (color stack fused into the compose);
///    10-bit H.265 / ProRes-VT → per-eye color then P010 planes.
/// 5. `encode_pixel_buffer[_p010]` → VideoToolbox.
///
/// Trim uses the iterator's PRECISE seek (keyframe run-in handled
/// internally), so a late trim-in costs ≤ 1 GOP of decode instead of
/// the whole preceding clip.
///
/// Bit depth: 10-bit / ProRes output runs the TRUE 16-bit chain
/// (`nv12_to_eac_cross_16` → `project_..._16` → P010) so the source's
/// 10 bits survive end to end; 8-bit output uses the 8-bit chain into
/// the BGRA compose.
#[cfg(target_os = "macos")]
fn export_eac_zerocopy_vt(
    pipeline: std::sync::Arc<Device>,
    cfg: FisheyeExportConfig,
    per_eye: Vec<EacPerEyeFrame>,
    progress_cb: &mut dyn FnMut(ExportProgress),
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    use crate::gpu::Lens;

    let raw = crate::decode::SegmentedZeroCopyPairIter::new(&cfg.segments, 0)?;
    // Temporal NR, if requested, denoises the s0/s4 P010 streams on the GPU
    // before EAC cross assembly (matches the Python app's source-stream
    // denoise) — no CPU readback.
    let mut decoder = if cfg.denoise_strength > 0.0 && crate::vt_denoise::is_supported() {
        tracing::info!(
            "export_eac (zc): temporal NR ENGAGED on GPU (strength={:.2})",
            cfg.denoise_strength
        );
        crate::decode::ZcEacDecoder::Denoising(
            crate::decode::DenoisingZeroCopyEacIter::new(raw, cfg.denoise_strength)?,
        )
    } else {
        crate::decode::ZcEacDecoder::Raw(raw)
    };
    let dims = decoder.dims();
    if !dims.is_valid() {
        return Err(Error::Ffmpeg(format!(
            "export_eac (zc): invalid EAC layout (stream w={})", dims.stream_w)));
    }

    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    let dt = 1.0 / cfg.fps as f64;
    let trim_in_frame = cfg.trim_in_s.map(|t| (t.max(0.0) * cfg.fps as f64).round() as u32).unwrap_or(0);
    let trim_out_frame = cfg.trim_out_s.map(|t| (t * cfg.fps as f64).round() as u32);
    let video_tmp = video_only_temp_path(&cfg.output_path);

    let is_prores_vt = matches!(cfg.encoder, EncoderBackend::ProResVideoToolbox);
    let ten_bit = cfg.bit_depth == 10 || is_prores_vt;
    let mut encoder = if is_prores_vt {
        H265Encoder::create_zero_copy_vt_prores(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.prores_profile)?
    } else if ten_bit {
        H265Encoder::create_zero_copy_vt_p010(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps)?
    } else {
        H265Encoder::create_zero_copy_vt(
            &video_tmp, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps)?
    };
    tracing::info!(
        "export_eac (zc): VT zero-copy ENGAGED — {} ({}x{}, trim {}..{})",
        if is_prores_vt { "ProRes-VT P010" } else if ten_bit { "HEVC Main10 P010" } else { "HEVC Main BGRA" },
        sbs_w, sbs_h, trim_in_frame,
        trim_out_frame.map(|f| f.to_string()).unwrap_or_else(|| "end".into()),
    );

    // Precise seek to trim-in (run-in discarded inside the iterator).
    if trim_in_frame > 0 {
        decoder.seek(trim_in_frame as f64 * dt)?;
    }

    let color_plan = cfg.color_stack.clone();
    let view = cfg.view_adjust;
    let view_active = !view.is_identity();
    let (v_l, v_r) = view.per_eye_matrices();

    let t_start = std::time::Instant::now();
    let mut frame_idx: u32 = trim_in_frame;
    let mut written: u64 = 0;
    let total_est = trim_out_frame
        .map(|o| (o.saturating_sub(trim_in_frame)) as u64)
        .unwrap_or(per_eye.len() as u64);

    while let Some(pair) = decoder.next_pair(&pipeline.device)? {
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("export_eac (zc): cancelled at frame {}", frame_idx);
            break;
        }
        if let Some(out_f) = trim_out_frame {
            if frame_idx >= out_f { break; }
        }

        // 10-bit output → 16-bit chain end to end (P010's 10 bits
        // survive into the cross + equirect instead of an 8-bit
        // quantize); 8-bit output → 8-bit chain (cheaper, same result
        // after the BGRA compose).
        let (cross_a, cross_b) = if ten_bit {
            (pipeline.nv12_to_eac_cross_16(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::A, dims)?,
             pipeline.nv12_to_eac_cross_16(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::B, dims)?)
        } else {
            (pipeline.nv12_to_eac_cross(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::A, dims)?,
             pipeline.nv12_to_eac_cross(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::B, dims)?)
        };

        // Per-eye stab/RS for this absolute frame (identity past the end).
        let ((mut rl, sl), (mut rr, sr)) = per_eye.get(frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
            ));
        if view_active {
            rl = EquirectRotation(crate::panomap::mat3_mul_row_major(&rl.0, &v_l));
            rr = EquirectRotation(crate::panomap::mat3_mul_row_major(&rr.0, &v_r));
        }

        // Left eye = Lens B, right eye = Lens A (same as preview + CPU
        // path). Projection target follows cfg.projection — half-equirect
        // (VR180) or equidistant fisheye (parity with the OSV output).
        let fisheye_out = matches!(cfg.projection, FisheyeExportProjection::Fisheye);
        let (left, right) = match (ten_bit, fisheye_out) {
            (true, false) => (
                pipeline.project_cross_texture_to_equirect_texture_16(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
                pipeline.project_cross_texture_to_equirect_texture_16(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?),
            (false, false) => (
                pipeline.project_cross_texture_to_equirect_texture(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
                pipeline.project_cross_texture_to_equirect_texture(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?),
            (true, true) => (
                pipeline.project_cross_texture_to_fisheye_texture_16(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
                pipeline.project_cross_texture_to_fisheye_texture_16(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?),
            (false, true) => (
                pipeline.project_cross_texture_to_fisheye_texture(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
                pipeline.project_cross_texture_to_fisheye_texture(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?),
        };

        if ten_bit {
            // Per-eye color (no-op when the plan is identity), then
            // compose into the P010 encode IOSurface.
            let left_g = pipeline.apply_color_stack_per_eye_16(
                &left, cfg.eye_w, cfg.eye_h, &color_plan)?;
            let right_g = pipeline.apply_color_stack_per_eye_16(
                &right, cfg.eye_w, cfg.eye_h, &color_plan)?;
            let l_final = left_g.as_ref().unwrap_or(&left);
            let r_final = right_g.as_ref().unwrap_or(&right);
            let encode_pb = crate::interop_macos::create_p010_encode_buffer(
                &pipeline.device, sbs_w, sbs_h)?;
            pipeline.compose_sbs_to_p010(
                l_final, r_final, &encode_pb, cfg.eye_w, cfg.eye_h)?;
            encoder.encode_pixel_buffer_p010(&encode_pb)?;
        } else {
            // Color stack fused into the BGRA compose.
            let encode_pb = crate::interop_macos::create_bgra_encode_buffer(
                &pipeline.device, sbs_w, sbs_h)?;
            pipeline.apply_color_stack_to_sbs_bgra(
                &left, &right, &encode_pb.wgpu_tex,
                cfg.eye_w, cfg.eye_h, &color_plan)?;
            encoder.encode_pixel_buffer(&encode_pb)?;
        }

        drop(pair);
        frame_idx += 1;
        written += 1;
        let elapsed = t_start.elapsed().as_secs_f32().max(1e-3);
        progress_cb(ExportProgress {
            frame_idx: written,
            total_frames: total_est,
            fps_avg: written as f32 / elapsed,
        });
    }

    encoder.finish()?;
    finalize_eac_audio(
        &cfg, &video_tmp,
        trim_in_frame as f64 * dt, written as f64 * dt,
    )?;
    finalize_metadata(
        &cfg.output_path, cfg.inject_youtube_vr180, cfg.inject_apmp, cfg.apmp_baseline_mm,
    )?;
    tracing::info!("export_eac (zc): done, {} frames in {:.2?}", written, t_start.elapsed());
    Ok(())
}

/// Windows GPU-resident EAC export — the GoPro `.360` analog of
/// [`export_fisheye_osv_gpu_resident`], with the EAC per-frame core of the
/// macOS [`export_eac_zerocopy_vt`]. NVDEC decodes both EAC HEVC streams;
/// D3D11 converts each P010→RGBA16; wgpu assembles the two lens crosses
/// (`rgba16_to_eac_cross`), projects each eye (equirect or normalized
/// fisheye, with the GUI-computed stab/RS + view-adjust), runs the color
/// stack, and composes a video-range P010 SBS fed to NVENC over CUDA — no
/// CPU readback, no swscale, no CPU `assemble_lens_*`.
///
/// `per_eye[absolute_frame_idx]` carries the precomputed per-frame
/// `(rotation, RS)` (built by the GUI, same as the portable path). The
/// absolute frame index is derived from each pair's pts, so the keyframe
/// run-in after a trim seek stays aligned. Any failure returns `Err` so
/// `export_eac` falls back to the portable path.
///
/// 10-bit only (the gate requires `bit_depth == 10`): the source's 10 bits
/// survive end to end (P010 → Rgba16 cross → Rgba16 project → P010 encode),
/// matching the macOS VT 10-bit arm. 8-bit output / libx265 / multi-segment
/// take the portable path.
#[cfg(target_os = "windows")]
fn export_eac_gpu_resident(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    per_eye: Vec<EacPerEyeFrame>,
    ctx: crate::interop_windows::VulkanImportCtx,
    iter: crate::fisheye_decode::SegmentedD3d11SharedStreamPairIter,
    progress_cb: &mut impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    use crate::nvenc_cuda::{CudaNvencEncoder, SharedP010Frame};
    use crate::gpu::Lens;
    use crate::fisheye_decode::SharedEacPair;

    // Bind the device-0 primary CUDA context to THIS thread; ffmpeg's CUDA
    // hwdevice + our external-memory imports both share it.
    let _cuda = cudarc::driver::CudaDevice::new(0)
        .map_err(|e| Error::Ffmpeg(format!("cuda primary ctx: {e:?}")))?;

    let dims = iter.dims();
    let (nw, nh) = iter.native_dims();
    let sbs_w = cfg.eye_w * 2;
    let sbs_h = cfg.eye_h;
    let fisheye_out = matches!(cfg.projection, FisheyeExportProjection::Fisheye);
    tracing::info!(
        "export_eac (GPU-resident): EAC native {}x{} (cross_w={}) → {}x{} SBS → NVENC(CUDA), proj={:?}",
        nw, nh, dims.cross_w(), sbs_w, sbs_h, cfg.projection
    );

    let dt = 1.0 / cfg.fps as f64;
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    let t_out = cfg.trim_out_s.map(|t| t.max(t_in + 1e-3)).unwrap_or(f64::INFINITY);
    let trim_in_frame = (t_in * cfg.fps as f64).round() as u32;
    let total_frames_to_write = if t_out.is_finite() {
        ((t_out - t_in) * cfg.fps as f64).round() as u64
    } else {
        (per_eye.len() as f64 - t_in * cfg.fps as f64).round().max(0.0) as u64
    };

    let mut iter = iter;
    if t_in > 0.001 { iter.seek(t_in)?; }

    let video_tmp = video_only_temp_path(&cfg.output_path);
    // Ring of shared P010 frames — main composes frame N+k while the encode
    // thread feeds NVENC frame N. (Identical pacing to the OSV path.)
    const RING: usize = 4;
    let ring: Vec<SharedP010Frame> = (0..RING)
        .map(|_| SharedP010Frame::new(&ctx, &pipeline.device, sbs_w, sbs_h))
        .collect::<Result<Vec<_>>>()?;
    tracing::info!("export_eac (GPU-resident): {RING}-slot shared P010 ring ready");

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
    tracing::info!("export_eac (GPU-resident): NVENC(CUDA) encode thread up");

    let color_plan = cfg.color_stack.clone();
    let view = cfg.view_adjust;
    let view_active = !view.is_identity();
    let (v_l, v_r) = view.per_eye_matrices();

    // Decode sub-thread (D3D11/NVDEC only — never touches CUDA/wgpu), so the
    // ~10 ms dual-stream decode overlaps the GPU compose + NVENC on this thread.
    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<SharedEacPair>(3);
    let cancel_dec = cancel.clone();
    let decode_handle = std::thread::spawn(move || {
        loop {
            if cancel_dec.load(Ordering::SeqCst) { break; }
            match iter.next_pair() {
                Ok(Some(p)) => { if pair_tx.send(p).is_err() { break; } }
                Ok(None) => break,
                Err(e) => { tracing::warn!("gpu-resident EAC decode: {e}"); break; }
            }
        }
    });

    let t_start = std::time::Instant::now();
    let mut written: u64 = 0;

    loop {
        let sp = match pair_rx.recv() { Ok(s) => s, Err(_) => break };
        if cancel.load(Ordering::SeqCst) { break; }
        if sp.pts_s.is_finite() && sp.pts_s >= t_out { break; }
        // Drop the keyframe run-in (pts before trim-in).
        if sp.pts_s.is_finite() && sp.pts_s < t_in - 0.5 * dt { continue; }

        // Absolute source frame index → per-eye stab/RS lookup (identity
        // past the end / when stabilization is off and per_eye is empty).
        let abs_idx = if sp.pts_s.is_finite() && sp.pts_s >= 0.0 {
            (sp.pts_s / dt).round() as usize
        } else { trim_in_frame as usize + written as usize };
        let ((mut rl, sl), (mut rr, sr)) = per_eye.get(abs_idx).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, crate::gpu::EquirectRsParams::DISABLED),
            ));
        // View-adjust (pano ± stereo) AFTER stab — same convention as the
        // portable/macOS EAC path. Left eye = Lens B, right eye = Lens A.
        if view_active {
            rl = EquirectRotation(crate::panomap::mat3_mul_row_major(&rl.0, &v_l));
            rr = EquirectRotation(crate::panomap::mat3_mul_row_major(&rr.0, &v_r));
        }

        // Import both streams (RGBA16, native res) + assemble the two lens
        // crosses on the GPU (16-bit, so the 10 bits survive).
        let s0_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.s0) };
        let s4_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.s4) };
        let cross_a = pipeline.rgba16_to_eac_cross(&s0_tex, &s4_tex, Lens::A, dims)?;
        let cross_b = pipeline.rgba16_to_eac_cross(&s0_tex, &s4_tex, Lens::B, dims)?;

        let (left16, right16) = if fisheye_out {
            (pipeline.project_cross_texture_to_fisheye_texture_16(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
             pipeline.project_cross_texture_to_fisheye_texture_16(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?)
        } else {
            (pipeline.project_cross_texture_to_equirect_texture_16(&cross_b, cfg.eye_w, cfg.eye_h, rl, sl)?,
             pipeline.project_cross_texture_to_equirect_texture_16(&cross_a, cfg.eye_w, cfg.eye_h, rr, sr)?)
        };
        let left_g = pipeline.apply_color_stack_per_eye_16(&left16, cfg.eye_w, cfg.eye_h, &color_plan)?;
        let right_g = pipeline.apply_color_stack_per_eye_16(&right16, cfg.eye_w, cfg.eye_h, &color_plan)?;
        let l_final = left_g.as_ref().unwrap_or(&left16);
        let r_final = right_g.as_ref().unwrap_or(&right16);

        // Compose video-range Rec.709 P010 plane textures (AVCOL_RANGE_MPEG —
        // the distribution standard, consistent with the portable path), then
        // copy into this ring slot's CUDA-shared linear images.
        let sh = &ring[written as usize % RING];
        let (y_opt, uv_opt) = pipeline.compose_sbs_to_p010_textures(l_final, r_final, cfg.eye_w, cfg.eye_h, false)?;
        {
            let mut enc = pipeline.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_resident_eac_copy_to_shared"),
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

        // Hand the (Copy) CUDA plane pointers to the encode thread.
        let (y_ptr, y_pitch) = sh.y_cuda();
        let (uv_ptr, uv_pitch) = sh.uv_cuda();
        if enc_tx.send(EncMsg { y_ptr, y_pitch, uv_ptr, uv_pitch }).is_err() {
            break; // encode thread died — surfaced at join
        }

        drop(sp);
        written += 1;
        progress_cb(ExportProgress {
            frame_idx: written,
            total_frames: total_frames_to_write,
            fps_avg: written as f32 / t_start.elapsed().as_secs_f32().max(1e-3),
        });
    }

    drop(pair_rx);
    let _ = decode_handle.join();
    drop(enc_tx); // end the encode thread's recv loop → flush + finish
    let enc_result = encode_handle.join();
    drop(ring);   // free shared frames only AFTER the encoder is done reading
    match enc_result {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(Error::Ffmpeg("gpu-resident EAC encode thread panicked".into())),
    }
    // Audio + metadata: same finalize as the portable EAC path (handles the
    // GoPro ambisonic / APAC tracks + multi-segment ffconcat).
    finalize_eac_audio(&cfg, &video_tmp, trim_in_frame as f64 * dt, written as f64 * dt)?;
    finalize_metadata(
        &cfg.output_path, cfg.inject_youtube_vr180, cfg.inject_apmp, cfg.apmp_baseline_mm,
    )?;
    tracing::info!(
        "export_eac (GPU-resident): done, {} frames in {:.2?} ({:.1} fps)",
        written, t_start.elapsed(), written as f32 / t_start.elapsed().as_secs_f32().max(1e-3)
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
    use crate::fisheye_decode::{
        DenoisingZeroCopyIter, ZcDecoder, ZeroCopyDualStreamFisheyeIter,
    };

    // Open zero-copy decoder. OSV swap convention mirrors
    // DualStreamFisheyeIter: !cfg.fisheye_swap_eyes means swap on by
    // default (Lens A == stream 0 == right eye).
    let raw = ZeroCopyDualStreamFisheyeIter::new(
        &cfg.source_path,
        0, // no frame limit
        !cfg.fisheye_swap_eyes,
    )?;
    // Temporal NR, if requested, wraps the decoder and denoises the P010
    // IOSurfaces on the GPU (no CPU readback) — the whole reason this path
    // is fast. Falls back to the raw decoder when off / unsupported.
    let mut decoder = if cfg.denoise_strength > 0.0 && crate::vt_denoise::is_supported() {
        tracing::info!(
            "fisheye_export (zero-copy): temporal NR ENGAGED on GPU (strength={:.2})",
            cfg.denoise_strength
        );
        ZcDecoder::Denoising(DenoisingZeroCopyIter::new(raw, cfg.denoise_strength)?)
    } else {
        ZcDecoder::Raw(raw)
    };
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
    // One-pass: encode STRAIGHT to the final file and mux the source's
    // audio inline (no temp + re-mux). Otherwise encode to a local-scratch
    // temp the finalize muxes audio onto (ambisonic / APAC).
    let one_pass = one_pass_audio_eligible(&cfg);
    let enc_out = if one_pass { cfg.output_path.clone() }
                  else { video_only_temp_path(&cfg.output_path) };
    // ProRes-VT → P010 IOSurface straight into Apple's ProRes engine;
    // HEVC 10-bit → P010 IOSurface encode (Main10); HEVC 8-bit → BGRA
    // IOSurface encode (Main). All consume the same zero-copy P010
    // decode above. ProRes is always 10-bit here (the GUI pins it), so
    // `ten_bit` keeps the P010 encode-buffer branch in the frame loop.
    let is_prores = cfg.encoder == EncoderBackend::ProResVideoToolbox;
    let ten_bit = cfg.bit_depth == 10 || is_prores;
    let mut encoder = if is_prores {
        H265Encoder::create_zero_copy_vt_prores(
            &enc_out, sbs_w, sbs_h, cfg.fps, cfg.prores_profile,
        )?
    } else if ten_bit {
        H265Encoder::create_zero_copy_vt_p010(
            &enc_out, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    } else {
        H265Encoder::create_zero_copy_vt(
            &enc_out, sbs_w, sbs_h, cfg.fps, cfg.bitrate_kbps,
        )?
    };
    // Inline audio passthrough (must be attached before the first frame).
    let one_pass_audio_tmp = if one_pass {
        let (asrc, atmp) = eac_audio_source(&cfg);
        let dur = total_frames_to_write as f64 / cfg.fps as f64;
        if let Err(e) = encoder.attach_audio_passthrough(&asrc, t_in, dur) {
            tracing::warn!("one-pass audio attach failed: {e} — video-only output");
        }
        atmp
    } else { None };
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
        if is_prores { "P010 IOSurface → ProRes-VT (HW)" }
        else if cfg.bit_depth == 10 { "P010 IOSurface (Main10)" }
        else { "BGRA IOSurface (Main, 8-bit)" },
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
        // Drop pre-trim frames (keyframe-backward seek, no decode-forward).
        if pair.pts_s.is_finite() && pair.pts_s < t_in - 0.5 * dt {
            continue;
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
    if one_pass {
        // Audio already muxed inline — drop the playlist temp, if any.
        if let Some(t) = one_pass_audio_tmp { std::fs::remove_file(t).ok(); }
    } else {
        finalize_with_audio(&cfg.source_path, &enc_out, &cfg.output_path,
            t_in, frame_idx as f64 * dt)?;
    }
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
        // Drop pre-trim frames (keyframe-backward seek, no decode-forward).
        if sp.pts_s.is_finite() && sp.pts_s < t_in - 0.5 * dt {
            continue;
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

            // KB projection (RS variant when stabilizing; same
            // (projection, rs) split as the GPU-resident + macOS p010
            // paths). Slots 30/31 keep the export's cached output
            // textures distinct from preview 0/1.
            let (left16, right16) = match (cfg.projection, rs_rows.as_deref()) {
                (FisheyeExportProjection::HalfEquirect, Some(rs)) => (
                    pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs, 31,
                    )?,
                ),
                (FisheyeExportProjection::HalfEquirect, None) => (
                    pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, 31,
                    )?,
                ),
                (FisheyeExportProjection::Fisheye, Some(rs)) => (
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, rs, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, rs, 31,
                    )?,
                ),
                (FisheyeExportProjection::Fisheye, None) => (
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                        &l_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_left, calib_left, 30,
                    )?,
                    pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                        &r_tex, src_w, src_h, cfg.eye_w, cfg.eye_h, rot_right, calib_right, 31,
                    )?,
                ),
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

    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path,
        t_in, frame_idx as f64 * dt)?;
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
        // Drop pre-trim frames (keyframe-backward seek, no decode-forward).
        if sp.pts_s.is_finite() && sp.pts_s < t_in - 0.5 * dt { continue; }
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
    finalize_with_audio(&cfg.source_path, &video_tmp, &cfg.output_path,
        t_in, frame_idx as f64 * dt)?;
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

/// Build a per-eye `Rgba16Unorm` half-equirect (or normalized fisheye)
/// for the 10-bit export paths. Honors `projection`:
///   - HalfEquirect → run KB → equirect projection on the GPU.
///   - Fisheye → KB → normalized equidistant fisheye on the GPU (8-bit
///     source widened to 16-bit by byte shift; no precision is gained,
///     but the texture format matches the P010 compose path).
///
/// `rs` = optional per-row rolling-shutter matrices for THIS frame
/// (`pack_per_row_camera_matrices` layout). When set, each eye is
/// uploaded once as Rgba16Unorm and projected through the RS-capable
/// rgba16 kernels (the same ones the Windows readback path dispatches;
/// slots 30/31 = export, distinct from preview 0/1). `None` keeps the
/// legacy byte-path projections — byte-identical to before.
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
    rs: Option<&[f32]>,
) -> Result<(wgpu::Texture, wgpu::Texture)> {
    if let Some(rs_buf) = rs {
        let l_tex = upload_eye_rgba16(pipeline, &pair.left, pair.bit_depth, src_w, src_h);
        let r_tex = upload_eye_rgba16(pipeline, &pair.right, pair.bit_depth, src_w, src_h);
        return Ok(match projection {
            FisheyeExportProjection::HalfEquirect => (
                pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                    &l_tex, src_w, src_h, eye_w, eye_h, rot_left, calib_left, rs_buf, 30,
                )?,
                pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                    &r_tex, src_w, src_h, eye_w, eye_h, rot_right, calib_right, rs_buf, 31,
                )?,
            ),
            FisheyeExportProjection::Fisheye => (
                pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                    &l_tex, src_w, src_h, eye_w, eye_h, rot_left, calib_left, rs_buf, 30,
                )?,
                pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                    &r_tex, src_w, src_h, eye_w, eye_h, rot_right, calib_right, rs_buf, 31,
                )?,
            ),
        });
    }
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

/// Upload one decoded eye as an `Rgba16Unorm` texture for the RS-capable
/// rgba16 projection kernels. 16-bit sources (RGBA64LE) upload verbatim;
/// 8-bit RGBA widens by byte shift (low byte zero — no fake precision),
/// matching `project_fisheye_to_fisheye_texture_16`'s convention.
fn upload_eye_rgba16(
    pipeline: &Device,
    bytes: &[u8],
    bit_depth: u8,
    w: u32, h: u32,
) -> wgpu::Texture {
    let tex = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("export_eye_src_16"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let widened: std::borrow::Cow<[u8]> = if bit_depth >= 16 {
        std::borrow::Cow::Borrowed(bytes)
    } else {
        let n_pixels = (w as usize) * (h as usize);
        let mut out = Vec::with_capacity(n_pixels * 8);
        for px in bytes.chunks_exact(4) {
            out.push(0x00); out.push(px[0]);
            out.push(0x00); out.push(px[1]);
            out.push(0x00); out.push(px[2]);
            out.push(0xFF); out.push(0xFF);
        }
        std::borrow::Cow::Owned(out)
    };
    pipeline.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        &widened,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 8),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    tex
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
