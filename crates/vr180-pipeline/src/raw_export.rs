//! Frames of a raw video clip as DNG files.
//!
//! A frame is written the way the app writes a raw still: the picture as
//! the export would render it — projected, stabilized, both eyes side by
//! side — but in the camera's own linear RGB, with the camera's colour
//! matrix beside it, as a linear DNG. A raw converter then develops it
//! from the sensor's values. The clip's raw development settings travel in
//! the file as that converter's own settings (an XMP packet of `crs:`
//! properties), so it opens looking as it does here.
//!
//! [`export_frame_dng`] writes the frame at one moment; [`export_dng_sequence`]
//! writes every frame of the trim window into a folder, one file per frame,
//! named by the frame's place in the take.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::dng_write::{write_linear_dng, DngCompression, LinearDngImage, LinearDngMeta};
use crate::fisheye_decode::{FisheyePair, FisheyePairIter};
use crate::fisheye_export::{ExportProgress, FisheyeExportConfig};
use crate::gpu::{Device, EquirectRotation, FisheyeCalib};
use crate::raw_color::RawCamera;
use crate::raw_tone::RawSettings;
use crate::{Error, Result};

/// A raw converter's settings for a frame, as the XMP packet it reads them
/// from: the clip's raw development settings under Camera Raw's own
/// property names (process version 2012). Detail controls that would
/// change the pixels beyond this app's rendering are left at zero.
pub fn develop_xmp(settings: &RawSettings, camera: &RawCamera) -> String {
    let (shot_k, shot_tint) = camera.as_shot_temperature();
    let white = match (settings.temperature, settings.tint) {
        (None, None) => r#"crs:WhiteBalance="As Shot""#.to_string(),
        (k, t) => format!(
            r#"crs:WhiteBalance="Custom" crs:Temperature="{:.0}" crs:Tint="{:+.0}""#,
            k.map_or(shot_k, |v| v as f64), t.map_or(shot_tint, |v| v as f64)),
    };
    let amount = |v: f32| if v == 0.0 { "0".to_string() } else { format!("{:+.0}", v) };
    format!(
        concat!(
            "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n",
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">",
            "<rdf:Description rdf:about=\"\" xmlns:crs=\"http://ns.adobe.com/camera-raw-settings/1.0/\" ",
            "crs:ProcessVersion=\"15.4\" {white} crs:Exposure2012=\"{exposure:+.2}\" crs:Contrast2012=\"{contrast}\" ",
            "crs:Highlights2012=\"{highlights}\" crs:Shadows2012=\"{shadows}\" crs:Whites2012=\"{whites}\" crs:Blacks2012=\"{blacks}\" ",
            "crs:Sharpness=\"0\" crs:LuminanceSmoothing=\"0\" crs:ColorNoiseReduction=\"0\" crs:HasSettings=\"True\"/>",
            "</rdf:RDF></x:xmpmeta>\n<?xpacket end=\"w\"?>"),
        white = white,
        exposure = settings.exposure,
        contrast = amount(settings.contrast),
        highlights = amount(settings.highlights),
        shadows = amount(settings.shadows),
        whites = amount(settings.whites),
        blacks = amount(settings.blacks),
    )
}

/// The DNG's camera and colour metadata for a frame of `camera` developed
/// with `settings`.
fn frame_meta(camera: &RawCamera, settings: &RawSettings, mono: bool, description: String) -> LinearDngMeta {
    let m = camera.color_matrix;
    LinearDngMeta {
        make: "Arashi Vision".into(),
        model: "Insta360 X6".into(),
        unique_camera_model: "Insta360 X6".into(),
        software: format!("VR180 Silver Bullet {}", env!("CARGO_PKG_VERSION")),
        color_matrix1: [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]].map(|v| v as f32),
        calibration_illuminant1: 21,
        as_shot_neutral: camera.as_shot_neutral.map(|v| v as f32),
        baseline_exposure: camera.baseline_exposure as f32,
        baseline_noise: None,
        baseline_sharpness: None,
        profile_hue_sat_map_dims: None,
        profile_hue_sat_map_data1: None,
        profile_hue_sat_map_encoding: None,
        profile_name: Some("VR180 Silver Bullet linear".into()),
        profile_embed_policy: Some(0),
        profile_tone_curve: None,
        profile_look_table: None,
        exposure_s: None,
        f_number: None,
        iso: None,
        focal_mm: None,
        focal_35mm: None,
        datetime_original: None,
        user_comment: None,
        exposure_program: None,
        metering_mode: None,
        white_balance: None,
        exposure_compensation: None,
        shutter_speed_value: None,
        image_description: Some(if mono { description } else { format!("VR180 side-by-side (left | right). {description}") }),
        xmp: Some(develop_xmp(settings, camera)),
    }
}

/// A raw clip opened for writing frames: the decoder, the lens and motion
/// data, and what the export's own loop computes before its first frame.
struct FrameSource {
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    decoder: Box<dyn FisheyePairIter>,
    imu: Option<vr180_fisheye::DjiOsvImu>,
    calib: (FisheyeCalib, FisheyeCalib),
    stab: Option<Vec<EquirectRotation>>,
    timeline: Option<crate::frame_timeline::FrameTimeline>,
    src: (u32, u32),
    readout_s: f32,
}

impl FrameSource {
    fn open(pipeline: Arc<Device>, cfg: &FisheyeExportConfig) -> Result<Self> {
        if !cfg.source_kind.is_raw_video() {
            return Err(Error::Ffmpeg(format!("{}: not a raw video clip, its frames have no DNG", cfg.source_path.display())));
        }
        let mut open = crate::fisheye_export::fisheye_export_opener(
            cfg.source_kind, cfg.fisheye_swap_eyes, 10, cfg.braw_color.clone(), Some(pipeline.clone()));
        let decoder = open(&cfg.source_path)?;
        let src = decoder.eye_dims();
        let imu = crate::fisheye_export::resolve_export_imu(cfg);
        let calib = crate::fisheye_export::resolve_calib_pair(cfg, src.0, src.1, imu.as_ref());
        let timeline = crate::fisheye_export::export_timeline(cfg);
        let frames = timeline.as_ref().map(|t| t.len()).filter(|n| *n > 0)
            .or_else(|| crate::decode::probe_video(&cfg.source_path).ok().map(|p| (p.duration_sec * p.fps as f64).round() as usize))
            .unwrap_or(1).max(1);
        let stab = if cfg.stabilize {
            imu.as_ref().and_then(|osv| {
                let max_corr = if cfg.dji_max_corr_deg > 0.0 { cfg.dji_max_corr_deg } else { f32::INFINITY };
                crate::dji_imu::compute_dji_stabilization(osv, frames, max_corr, cfg.dji_smooth_ms, cfg.fps, cfg.dji_responsiveness)
                    .ok().map(|s| { let mut v = s.per_frame; crate::dji_imu::apply_lock_ref_diag(&mut v, cfg.dji_smooth_ms); v })
            })
        } else { None };
        let readout_s = crate::dji_imu::readout_ms_for(cfg.source_kind, cfg.fps, imu.as_ref()) / 1000.0;
        Ok(Self { pipeline, cfg: cfg.clone(), decoder, imu, calib, stab, timeline, src, readout_s })
    }

    /// The frame's row in the per-frame tables.
    fn row(&self, pair: &FisheyePair, fallback: usize) -> usize {
        crate::frame_timeline::stab_row(self.timeline.as_ref(), pair.pts_s, 1.0 / self.cfg.fps as f64, fallback)
    }

    /// The pair's two eyes as the export projects them, camera-linear.
    fn eyes(&self, pair: &FisheyePair, row: usize) -> Result<(wgpu::Texture, wgpu::Texture)> {
        let cfg = &self.cfg;
        let rot = self.stab.as_ref().and_then(|v| v.get(row).copied()).unwrap_or(EquirectRotation::IDENTITY);
        let (rl, rr) = crate::dji_imu::per_eye_rotations(rot, self.imu.as_ref(), row, cfg.fisheye_swap_eyes);
        let (rl, rr) = if cfg.view_adjust.is_identity() { (rl, rr) } else {
            let (vl, vr) = cfg.view_adjust.per_eye_matrices();
            (EquirectRotation(crate::panomap::mat3_mul_row_major(&rl.0, &vl)), EquirectRotation(crate::panomap::mat3_mul_row_major(&rr.0, &vr)))
        };
        let (rows_l, rows_r) = if cfg.stabilize && self.imu.is_some() && self.readout_s > 0.0 {
            self.imu.as_ref().map(|osv| crate::dji_imu::per_eye_rs_rows(osv, row, self.readout_s, self.src.1, cfg.fps, cfg.fisheye_swap_eyes))
                .unwrap_or((None, None))
        } else { (None, None) };
        crate::fisheye_export::build_eye_eq_16(
            &self.pipeline, pair, self.src.0, self.src.1, cfg.eye_w, cfg.eye_h,
            rl, rr, self.calib.0, self.calib.1, cfg.projection, (rows_l.as_deref(), rows_r.as_deref()))
    }

    /// Write the pair as a linear DNG.
    fn write(&self, pair: &FisheyePair, row: usize, path: &Path, compression: DngCompression) -> Result<()> {
        let cfg = &self.cfg;
        let (left, right) = self.eyes(pair, row)?;
        let mono = crate::source_kind::is_single_eye(cfg.source_kind, &cfg.source_path);
        let (w, h) = if mono { (cfg.eye_w, cfg.eye_h) } else { (cfg.eye_w * 2, cfg.eye_h) };
        let frame = if mono { left } else { self.pipeline.compose_sbs_textures_16(&left, &right, cfg.eye_w, cfg.eye_h)? };
        let bytes = self.pipeline.read_texture_rgb48(&frame, w, h)?;
        let rgb16: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        let (camera, settings) = match &cfg.color_stack.raw_develop {
            Some(stage) => (stage.camera, stage.settings),
            None => (RawCamera::X6, RawSettings::default()),
        };
        let description = format!("frame {row} at {:.4} s", pair.pts_s);
        let meta = frame_meta(&camera, &settings, mono, description);
        write_linear_dng(path, &LinearDngImage { width: w, height: h, rgb16: &rgb16, white_level: 65535 }, &meta, compression, None)
            .map_err(Error::Io)
    }
}

/// Write the frame of a raw clip at `t_s` (seconds on the clip's timeline)
/// as a linear DNG at `out`. `cfg` is the clip's export configuration: its
/// output view, size, stabilization and lens; its encoder fields are not
/// used.
pub fn export_frame_dng(pipeline: Arc<Device>, cfg: &FisheyeExportConfig, t_s: f64, out: &Path) -> Result<()> {
    let mut source = FrameSource::open(pipeline, cfg)?;
    source.decoder.seek(t_s.max(0.0))?;
    let pair = source.decoder.decode_forward_to(t_s.max(0.0), 1.0 / cfg.fps as f64)?
        .ok_or_else(|| Error::Ffmpeg("no frame at that time".into()))?;
    let row = source.row(&pair, 0);
    source.write(&pair, row, out, DngCompression::LosslessJpegTiles)
}

/// Write the frame of a raw clip at `t_s` as the app develops it (the
/// clip's whole colour stack) into a 16-bit PNG at `out`: the picture the
/// DNG of [`export_frame_dng`] should give in a raw converter.
pub fn export_frame_png16(pipeline: Arc<Device>, cfg: &FisheyeExportConfig, t_s: f64, out: &Path) -> Result<()> {
    let mut source = FrameSource::open(pipeline.clone(), cfg)?;
    source.decoder.seek(t_s.max(0.0))?;
    let pair = source.decoder.decode_forward_to(t_s.max(0.0), 1.0 / cfg.fps as f64)?
        .ok_or_else(|| Error::Ffmpeg("no frame at that time".into()))?;
    let row = source.row(&pair, 0);
    let (left, right) = source.eyes(&pair, row)?;
    let mono = crate::source_kind::is_single_eye(cfg.source_kind, &cfg.source_path);
    let (w, h) = if mono { (cfg.eye_w, cfg.eye_h) } else { (cfg.eye_w * 2, cfg.eye_h) };
    let frame = if mono {
        pipeline.apply_color_stack_per_eye_16(&left, cfg.eye_w, cfg.eye_h, &cfg.color_stack)?.unwrap_or(left)
    } else {
        let (l, r) = pipeline.apply_color_stack_pair_16(&left, &right, cfg.eye_w, cfg.eye_h, &cfg.color_stack, None)?;
        pipeline.compose_sbs_textures_16(&l.unwrap_or(left), &r.unwrap_or(right), cfg.eye_w, cfg.eye_h)?
    };
    let bytes = pipeline.read_texture_rgb48(&frame, w, h)?;
    // PNG holds 16-bit samples big-endian; `image` takes them in native order.
    let native: Vec<u8> = bytes.chunks_exact(2).flat_map(|b| u16::from_le_bytes([b[0], b[1]]).to_ne_bytes()).collect();
    image::save_buffer(out, &native, w, h, image::ColorType::Rgb16)
        .map_err(|e| Error::Ffmpeg(format!("write {}: {e}", out.display())))
}

/// The file of frame `row` in a sequence written to `dir` for the clip `stem`.
pub fn sequence_frame_path(dir: &Path, stem: &str, row: usize) -> PathBuf {
    dir.join(format!("{stem}_{row:06}.dng"))
}

/// Write every frame of a raw clip's trim window as linear DNGs into the
/// folder `cfg.output_path` (created if needed), one file per frame named
/// by the frame's place in the take; a frame the camera never saved leaves
/// its number out. Returns the number of files written.
pub fn export_dng_sequence(
    pipeline: Arc<Device>,
    cfg: FisheyeExportConfig,
    mut progress: impl FnMut(ExportProgress),
    cancel: Arc<AtomicBool>,
) -> Result<u64> {
    let mut source = FrameSource::open(pipeline, &cfg)?;
    let dir = cfg.output_path.clone();
    std::fs::create_dir_all(&dir)?;
    let stem = cfg.source_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "frame".into());
    let dt = 1.0 / cfg.fps as f64;
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    let t_out = cfg.trim_out_s.map(|t| t.max(t_in + 1e-3)).unwrap_or(f64::INFINITY);
    if t_in > 0.001 {
        source.decoder.seek(t_in)?;
    }
    let total = {
        let end = if t_out.is_finite() { t_out } else {
            crate::decode::probe_video(&cfg.source_path).map(|p| p.duration_sec).unwrap_or(t_in)
        };
        ((end - t_in) * cfg.fps as f64).round().max(0.0) as u64
    };
    let started = std::time::Instant::now();
    let mut written = 0u64;
    let mut seen = 0usize;
    while let Some(pair) = source.decoder.next_pair()? {
        if cancel.load(Ordering::SeqCst) {
            tracing::info!("raw frames: cancelled after {written} files");
            break;
        }
        if pair.pts_s >= t_out { break; }
        if pair.pts_s < t_in - 0.5 * dt { continue; }
        let row = source.row(&pair, seen);
        seen += 1;
        source.write(&pair, row, &sequence_frame_path(&dir, &stem, row), DngCompression::LosslessJpegTiles)?;
        written += 1;
        progress(ExportProgress {
            frame_idx: written,
            total_frames: total.max(written),
            fps_avg: written as f32 / started.elapsed().as_secs_f32().max(1e-3),
        });
    }
    tracing::info!("raw frames: {written} linear DNG files in {} ({:.1?})", dir.display(), started.elapsed());
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_travel_as_the_converters_own() {
        let cam = RawCamera::X6;
        let x = develop_xmp(&RawSettings::default(), &cam);
        assert!(x.contains(r#"crs:WhiteBalance="As Shot""#) && x.contains(r#"crs:Exposure2012="+0.00""#), "{x}");
        assert!(x.contains(r#"crs:Shadows2012="0""#) && x.contains(r#"crs:ProcessVersion="15.4""#));
        let s = RawSettings { temperature: Some(5200.0), exposure: -1.25, shadows: 40.0, highlights: -65.0, ..Default::default() };
        let x = develop_xmp(&s, &cam);
        assert!(x.contains(r#"crs:WhiteBalance="Custom" crs:Temperature="5200""#), "{x}");
        // The tint the user did not touch is the shot's own.
        let (_, tint) = cam.as_shot_temperature();
        assert!(x.contains(&format!(r#"crs:Tint="{:+.0}""#, tint)), "{x}");
        assert!(x.contains(r#"crs:Exposure2012="-1.25""#) && x.contains(r#"crs:Shadows2012="+40""#) && x.contains(r#"crs:Highlights2012="-65""#), "{x}");
        assert_eq!(sequence_frame_path(Path::new("/out"), "TAKE", 42), PathBuf::from("/out/TAKE_000042.dng"));
    }
}
