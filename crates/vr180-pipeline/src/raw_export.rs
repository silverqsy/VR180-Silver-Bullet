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
/// property names (process version 2012), the sharpening and the colour
/// noise reduction among them. Luminance noise reduction, which this app's
/// rendering has none of, is left at zero.
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
            "crs:Sharpness=\"{sharpness:.0}\" crs:SharpenRadius=\"{radius:+.1}\" crs:SharpenDetail=\"{detail:.0}\" crs:SharpenEdgeMasking=\"{masking:.0}\" ",
            "crs:LuminanceSmoothing=\"0\" crs:ColorNoiseReduction=\"{color_noise:.0}\" crs:ColorNoiseReductionDetail=\"50\" crs:ColorNoiseReductionSmoothness=\"50\" crs:HasSettings=\"True\"/>",
            "</rdf:RDF></x:xmpmeta>\n<?xpacket end=\"w\"?>"),
        white = white,
        exposure = settings.exposure,
        contrast = amount(settings.contrast),
        highlights = amount(settings.highlights),
        shadows = amount(settings.shadows),
        whites = amount(settings.whites),
        blacks = amount(settings.blacks),
        sharpness = if settings.sharpen.is_off() { 0.0 } else { settings.sharpen.amount.clamp(1.0, 150.0) },
        radius = settings.sharpen.radius.clamp(0.5, 3.0),
        detail = settings.sharpen.detail.clamp(0.0, 100.0),
        masking = settings.sharpen.masking.clamp(0.0, 100.0),
        color_noise = settings.color_noise.clamp(0.0, 100.0),
    )
}

/// A frame's Camera Raw settings read back ([`read_develop_xmp`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ReadDevelop {
    /// The controls this development has, at the packet's values.
    pub settings: RawSettings,
    /// What the packet sets away from Camera Raw's defaults that this
    /// development has no control for: the control's name in Camera Raw,
    /// and its value where it is a single number or name (else empty).
    pub not_carried: Vec<(&'static str, String)>,
}

/// The Camera Raw properties of an XMP packet, by name without the prefix:
/// the attributes of its first description that has any, and after them
/// what is written as plain elements.
fn crs_properties(xmp: &str) -> std::collections::HashMap<String, String> {
    let unescape = |v: &str| v.replace("&quot;", "\"").replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&");
    let mut out = std::collections::HashMap::new();
    // The opening tag of each description, to its closing bracket outside quotes.
    let mut from = 0;
    while let Some(at) = xmp[from..].find("<rdf:Description") {
        let start = from + at;
        let mut quote: Option<char> = None;
        let mut end = xmp.len();
        for (i, c) in xmp[start..].char_indices() {
            match (quote, c) {
                (Some(q), c) if c == q => quote = None,
                (None, '"') | (None, '\'') => quote = Some(c),
                (None, '>') => { end = start + i; break; }
                _ => {}
            }
        }
        let tag = &xmp[start..end];
        let mut rest = tag;
        while let Some(i) = rest.find("crs:") {
            let after = &rest[i + 4..];
            let name_len = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
            let (name, tail) = after.split_at(name_len);
            rest = tail;
            let Some(tail) = tail.strip_prefix('=') else { continue };
            let Some(q) = tail.chars().next().filter(|c| *c == '"' || *c == '\'') else { continue };
            let Some(close) = tail[1..].find(q) else { break };
            out.entry(name.to_string()).or_insert_with(|| unescape(&tail[1..1 + close]));
            rest = &tail[2 + close..];
        }
        if !out.is_empty() {
            break;
        }
        from = end;
    }
    // Plain elements (`<crs:Name>value</crs:Name>`), where the attribute is not there.
    let mut rest = xmp;
    while let Some(i) = rest.find("<crs:") {
        let after = &rest[i + 5..];
        let name_len = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
        let (name, tail) = after.split_at(name_len);
        rest = tail;
        let Some(tail) = tail.strip_prefix('>') else { continue };
        let Some(close) = tail.find('<') else { break };
        if tail[close..].starts_with(&format!("</crs:{name}>")) {
            out.entry(name.to_string()).or_insert_with(|| unescape(tail[..close].trim()));
        }
    }
    out
}

/// The points of a tone curve element (`<crs:Name><rdf:Seq><rdf:li>x, y</rdf:li>…`), if the packet has it.
fn crs_curve(xmp: &str, name: &str) -> Option<Vec<(f32, f32)>> {
    let open = format!("<crs:{name}>");
    let body = &xmp[xmp.find(&open)? + open.len()..];
    let body = &body[..body.find(&format!("</crs:{name}>"))?];
    Some(body.split("<rdf:li>").skip(1).filter_map(|li| {
        let text = &li[..li.find('<')?];
        let mut xy = text.split(',').map(|v| v.trim().parse::<f32>());
        Some((xy.next()?.ok()?, xy.next()?.ok()?))
    }).collect())
}

/// Read a frame's Camera Raw settings from its XMP packet: the inverse of
/// [`develop_xmp`] for the controls this development has (white balance,
/// the six tone controls of process version 2012, sharpening, colour noise
/// reduction), and a list of what else the packet sets away from Camera
/// Raw's defaults. `None` when the packet holds no such settings.
pub fn read_develop_xmp(xmp: &str) -> Option<ReadDevelop> {
    let p = crs_properties(xmp);
    let num = |k: &str| p.get(k).and_then(|v| v.trim().parse::<f32>().ok()).filter(|v| v.is_finite());
    let flag = |k: &str| p.get(k).is_some_and(|v| v.eq_ignore_ascii_case("true"));
    // The tone controls of process version 2012 are what this development's
    // are: a packet without them is of another kind.
    let exposure = num("Exposure2012")?;
    let mut settings = RawSettings { exposure: exposure.clamp(-5.0, 5.0), ..RawSettings::default() };
    let amount = |k: &str| num(k).map_or(0.0, |v| v.clamp(-100.0, 100.0));
    settings.contrast = amount("Contrast2012");
    settings.highlights = amount("Highlights2012");
    settings.shadows = amount("Shadows2012");
    settings.whites = amount("Whites2012");
    settings.blacks = amount("Blacks2012");
    // "As Shot" is the file's own white; anything else is stated in kelvin and tint.
    let as_shot = p.get("WhiteBalance").map_or(true, |v| v.eq_ignore_ascii_case("As Shot"));
    if !as_shot {
        settings.temperature = num("Temperature").map(|k| k.clamp(2000.0, 50000.0));
        settings.tint = num("Tint").map(|t| t.clamp(-150.0, 150.0));
    }
    settings.sharpen = crate::raw_sharpen::RawSharpen {
        amount: num("Sharpness").map_or(0.0, |v| v.clamp(0.0, 150.0)),
        radius: num("SharpenRadius").map_or(1.0, |v| v.clamp(0.5, 3.0)),
        detail: num("SharpenDetail").map_or(25.0, |v| v.clamp(0.0, 100.0)),
        masking: num("SharpenEdgeMasking").map_or(0.0, |v| v.clamp(0.0, 100.0)),
    };
    settings.color_noise = num("ColorNoiseReduction").map_or(crate::raw_tone::DEFAULT_COLOR_NOISE, |v| v.clamp(0.0, 100.0));

    // What else is set.
    let mut not_carried: Vec<(&'static str, String)> = Vec::new();
    let signed = |v: f32| if v == v.round() { format!("{:+.0}", v) } else { format!("{:+.2}", v) };
    let mut note = |name: &'static str, value: String| not_carried.push((name, value));
    for (name, key, default) in [
        ("Texture", "Texture", 0.0), ("Clarity", "Clarity2012", 0.0), ("Dehaze", "Dehaze", 0.0),
        ("Vibrance", "Vibrance", 0.0), ("Saturation", "Saturation", 0.0),
        ("Luminance noise reduction", "LuminanceSmoothing", 0.0),
        ("Color noise reduction detail", "ColorNoiseReductionDetail", 50.0),
        ("Color noise reduction smoothness", "ColorNoiseReductionSmoothness", 50.0),
        ("Grain", "GrainAmount", 0.0),
    ] {
        if let Some(v) = num(key).filter(|v| (*v - default).abs() > 1e-3) {
            note(name, if default == 0.0 { signed(v) } else { format!("{v:.0}") });
        }
    }
    // Groups of controls: named once when any of theirs is set.
    let any = |keys: &[&str], default: f32| keys.iter().any(|k| num(k).is_some_and(|v| (v - default).abs() > 1e-3));
    let any_prefixed = |prefixes: &[&str]| p.iter().any(|(k, v)| prefixes.iter().any(|pre| k.starts_with(pre)) && v.trim().parse::<f32>().is_ok_and(|v| v.abs() > 1e-3));
    let linear = |name: &str| crs_curve(xmp, name).map_or(true, |c| c.is_empty() || c == [(0.0, 0.0), (255.0, 255.0)]);
    let curve_named = p.get("ToneCurveName2012").is_some_and(|n| !n.eq_ignore_ascii_case("Linear"));
    if curve_named || !["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"].iter().all(|n| linear(n))
        || any(&["ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"], 0.0)
    {
        note("Tone curve", String::new());
    }
    if any_prefixed(&["HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"]) {
        note("Color mixer", String::new());
    }
    if any(&["SplitToningShadowSaturation", "SplitToningHighlightSaturation", "ColorGradeMidtoneSat", "ColorGradeGlobalSat",
             "ColorGradeShadowLum", "ColorGradeMidtoneLum", "ColorGradeHighlightLum", "ColorGradeGlobalLum"], 0.0) {
        note("Color grading", String::new());
    }
    if any(&["ShadowTint", "RedHue", "RedSaturation", "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation"], 0.0) {
        note("Calibration", String::new());
    }
    if any(&["PostCropVignetteAmount", "VignetteAmount"], 0.0) {
        note("Vignette", String::new());
    }
    if any(&["LensProfileEnable", "LensManualDistortionAmount", "AutoLateralCA", "DefringePurpleAmount", "DefringeGreenAmount"], 0.0) {
        note("Lens corrections", String::new());
    }
    if any(&["PerspectiveUpright", "PerspectiveVertical", "PerspectiveHorizontal", "PerspectiveRotate", "PerspectiveAspect", "PerspectiveX", "PerspectiveY"], 0.0)
        || any(&["PerspectiveScale"], 100.0)
    {
        note("Geometry", String::new());
    }
    if flag("HasCrop") {
        note("Crop", String::new());
    }
    if flag("ConvertToGrayscale") {
        note("Black & white", String::new());
    }
    if ["<crs:MaskGroupBasedCorrections", "<crs:GradientBasedCorrections", "<crs:CircularGradientBasedCorrections", "<crs:PaintBasedCorrections"]
        .iter().any(|tag| xmp.contains(tag))
    {
        note("Masks", String::new());
    }
    if xmp.contains("<crs:RetouchAreas") || xmp.contains("<crs:RetouchInfo") {
        note("Healing", String::new());
    }
    // The profile: the file's own unless another is named.
    if let Some(profile) = p.get("CameraProfile").filter(|n| !n.eq_ignore_ascii_case("Embedded") && !n.is_empty()) {
        note("Profile", profile.clone());
    } else if xmp.contains("<crs:Look") {
        note("Profile", String::new());
    }
    Some(ReadDevelop { settings, not_carried })
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
    /// The eyes' calibrations before the view, and with it.
    source_calib: (FisheyeCalib, FisheyeCalib),
    calib: (FisheyeCalib, FisheyeCalib),
    stab: Option<Vec<EquirectRotation>>,
    timeline: Option<crate::frame_timeline::FrameTimeline>,
    src: (u32, u32),
    /// The clip holds one eye: a frame is that eye, not two side by side.
    mono: bool,
}

impl FrameSource {
    /// `stretch`: the seconds of the clip the frames are taken from (what an
    /// enhanced-frames folder is looked through for); for one frame, from
    /// its time to a frame later, either of which may be the one written.
    fn open(pipeline: Arc<Device>, cfg: &FisheyeExportConfig, stretch: (Option<f64>, Option<f64>)) -> Result<Self> {
        if !cfg.source_kind.is_raw_video() {
            return Err(Error::Ffmpeg(format!("{}: not a raw video clip, its frames have no DNG", cfg.source_path.display())));
        }
        // A lens frame is fixed before the first frame, as the video export
        // fixes it: the frames are then the picture that export writes. A
        // lens no frame can be made of keeps the disc, as the preview does;
        // and a frame is no spatial video, whatever the clip's export is.
        let mut cfg = cfg.clone();
        cfg.spatial = None;
        // Enhanced frames, and the size that goes with them, where this
        // stretch has some.
        cfg.settle_raw_enhanced(stretch.0, stretch.1);
        match crate::fisheye_export::resolve_lens_frame(&mut cfg) {
            Ok(Some((frame, _))) => tracing::debug!(
                "raw frames: lens frame {}x{}, focal {:.3} px, xi {:.5}, k1 {:.6}, k2 {:.6}",
                frame.width, frame.height, frame.focal, frame.xi, frame.k1, frame.k2),
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("raw frames: no lens frame ({e}) — the frames hold the disc");
                cfg.lens_frame = crate::fisheye_export::LensFrameOut::Off;
            }
        }
        let cfg = &cfg;
        let mut open = crate::fisheye_export::fisheye_export_opener(
            cfg.source_kind, cfg.fisheye_swap_eyes, 10, cfg.braw_color.clone(), Some(pipeline.clone()),
            cfg.raw_enhanced().map(|e| crate::x6_raw::EnhancedSource { t_in: stretch.0, t_out: stretch.1, ..e }));
        let decoder = open(&cfg.source_path)?;
        let src = decoder.eye_dims();
        let imu = crate::fisheye_export::resolve_export_imu(cfg);
        let source_calib = crate::fisheye_export::resolve_source_calib_pair(cfg, src.0, src.1, imu.as_ref());
        let calib = crate::fisheye_export::view_calib_pair(cfg, source_calib);
        let timeline = crate::fisheye_export::export_timeline(cfg);
        let frames = timeline.as_ref().map(|t| t.len()).filter(|n| *n > 0)
            .or_else(|| crate::decode::probe_video(&cfg.source_path).ok().map(|p| (p.duration_sec * p.fps as f64).round() as usize))
            .unwrap_or(1).max(1);
        let stab = cfg.motion().stabilization(cfg.source_kind, imu.as_ref(), frames, cfg.fps)
            .and_then(|stab| stab.ok())
            .map(|stab| { let mut v = stab.per_frame; crate::dji_imu::apply_lock_ref_diag(&mut v, cfg.dji_smooth_ms); v });
        let mono = crate::source_kind::is_single_eye(cfg.source_kind, &cfg.source_path);
        Ok(Self { pipeline, cfg: cfg.clone(), decoder, imu, source_calib, calib, stab, timeline, src, mono })
    }

    /// The frame's row in the per-frame tables.
    fn row(&self, pair: &FisheyePair, fallback: usize) -> usize {
        crate::frame_timeline::stab_row(self.timeline.as_ref(), pair.pts_s, 1.0 / self.cfg.fps as f64, fallback)
    }

    /// The frame's pose: each eye's rotation and per-row matrices.
    fn pose(&self, row: usize) -> crate::render::FramePose {
        let cfg = &self.cfg;
        let motion = cfg.motion();
        let rot = self.stab.as_ref().and_then(|v| v.get(row).copied()).unwrap_or(EquirectRotation::IDENTITY);
        motion.pose(rot, self.imu.as_ref(), row, motion.row_readout(cfg.source_kind, self.imu.as_ref(), self.src.1, cfg.fps))
    }

    /// The frame's projection, as the export of the clip asks for it.
    fn job<'a>(&self, pose: &'a crate::render::FramePose) -> crate::render::EyeJob<'a> {
        let cfg = &self.cfg;
        crate::render::EyeJob {
            src_w: self.src.0, src_h: self.src.1, eye_w: cfg.eye_w, eye_h: cfg.eye_h, pose,
            calib_left: self.calib.0, calib_right: self.calib.1, source_calib: self.source_calib,
            projection: cfg.projection, mono: self.mono,
        }
    }

    /// The pair's two eyes as the export projects them, camera-linear.
    fn eyes(&self, pair: &FisheyePair, row: usize) -> Result<(wgpu::Texture, wgpu::Texture)> {
        let pose = self.pose(row);
        crate::fisheye_export::project_pair_16(&self.pipeline, pair, &self.job(&pose))
    }

    /// The clip's colour plan for this frame: with the frame's reference
    /// where the view is not the half-equirect.
    fn plan(&self, pair: &FisheyePair, row: usize) -> Result<std::borrow::Cow<'_, crate::gpu::ColorStackPlan>> {
        let pose = self.pose(row);
        crate::fisheye_export::plan_with_raw_reference(&self.pipeline, &self.cfg.color_stack, pair, &self.job(&pose))
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
    let mut source = FrameSource::open(pipeline, cfg, (Some(t_s), Some(t_s + 1.0 / cfg.fps.max(1.0) as f64)))?;
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
    let mut source = FrameSource::open(pipeline.clone(), cfg, (Some(t_s), Some(t_s + 1.0 / cfg.fps.max(1.0) as f64)))?;
    source.decoder.seek(t_s.max(0.0))?;
    let pair = source.decoder.decode_forward_to(t_s.max(0.0), 1.0 / cfg.fps as f64)?
        .ok_or_else(|| Error::Ffmpeg("no frame at that time".into()))?;
    let row = source.row(&pair, 0);
    let (left, right) = source.eyes(&pair, row)?;
    let plan = source.plan(&pair, row)?;
    // The frame's size is the opened source's: a lens frame sets its own.
    let cfg = &source.cfg;
    let mono = crate::source_kind::is_single_eye(cfg.source_kind, &cfg.source_path);
    let (w, h) = if mono { (cfg.eye_w, cfg.eye_h) } else { (cfg.eye_w * 2, cfg.eye_h) };
    let frame = if mono {
        pipeline.apply_color_stack_per_eye_16(&left, cfg.eye_w, cfg.eye_h, &plan)?.unwrap_or(left)
    } else {
        let (l, r) = pipeline.apply_color_stack_pair_16(&left, &right, cfg.eye_w, cfg.eye_h, &plan, None)?;
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
    let mut source = FrameSource::open(pipeline, &cfg, (cfg.trim_in_s, cfg.trim_out_s))?;
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
        // The colour noise reduction a clip starts with, and one that was moved.
        assert!(x.contains(r#"crs:LuminanceSmoothing="0" crs:ColorNoiseReduction="25" crs:ColorNoiseReductionDetail="50""#), "{x}");
        let quiet = develop_xmp(&RawSettings { color_noise: 0.0, ..Default::default() }, &cam);
        assert!(quiet.contains(r#"crs:ColorNoiseReduction="0""#), "{quiet}");
        // No sharpening: the converter's own stays off too.
        assert!(x.contains(r#"crs:Sharpness="0" crs:SharpenRadius="+1.0" crs:SharpenDetail="25" crs:SharpenEdgeMasking="0""#), "{x}");
        let sharp = RawSettings { sharpen: crate::raw_sharpen::RawSharpen { amount: 55.0, radius: 1.4, detail: 30.0, masking: 20.0 }, ..Default::default() };
        let x2 = develop_xmp(&sharp, &cam);
        assert!(x2.contains(r#"crs:Sharpness="55" crs:SharpenRadius="+1.4" crs:SharpenDetail="30" crs:SharpenEdgeMasking="20""#), "{x2}");
        let s = RawSettings { temperature: Some(5200.0), exposure: -1.25, shadows: 40.0, highlights: -65.0, ..Default::default() };
        let x = develop_xmp(&s, &cam);
        assert!(x.contains(r#"crs:WhiteBalance="Custom" crs:Temperature="5200""#), "{x}");
        // The tint the user did not touch is the shot's own.
        let (_, tint) = cam.as_shot_temperature();
        assert!(x.contains(&format!(r#"crs:Tint="{:+.0}""#, tint)), "{x}");
        assert!(x.contains(r#"crs:Exposure2012="-1.25""#) && x.contains(r#"crs:Shadows2012="+40""#) && x.contains(r#"crs:Highlights2012="-65""#), "{x}");
        assert_eq!(sequence_frame_path(Path::new("/out"), "TAKE", 42), PathBuf::from("/out/TAKE_000042.dng"));
    }

    /// What the app writes for a raw converter reads back as it was.
    #[test]
    fn settings_come_back_from_the_converters_packet() {
        let cam = RawCamera::X6;
        for s in [
            RawSettings::default(),
            RawSettings { temperature: Some(5200.0), tint: Some(-12.0), exposure: -1.25, contrast: 15.0, highlights: -65.0, shadows: 40.0, whites: 8.0, blacks: -22.0,
                sharpen: crate::raw_sharpen::RawSharpen { amount: 55.0, radius: 1.4, detail: 30.0, masking: 20.0 }, color_noise: 60.0 },
            RawSettings { exposure: 2.5, shadows: 100.0, color_noise: 0.0, ..Default::default() },
        ] {
            let back = read_develop_xmp(&develop_xmp(&s, &cam)).expect("settings in the packet");
            assert_eq!(back.settings, s);
            assert!(back.not_carried.is_empty(), "{:?}", back.not_carried);
        }
        // A packet without Camera Raw settings, and one of an older process.
        assert!(read_develop_xmp("<x:xmpmeta><rdf:RDF><rdf:Description rdf:about=\"\" dc:format=\"image/dng\"/></rdf:RDF></x:xmpmeta>").is_none());
        assert!(read_develop_xmp(r#"<rdf:Description crs:ProcessVersion="5.7" crs:Exposure="+0.50" crs:Shadows="5"/>"#).is_none());
    }

    /// A packet as Camera Raw writes it: the controls this development has
    /// are taken, held to their ranges, and everything else that is set is
    /// named.
    #[test]
    fn a_converters_packet_is_read_and_the_rest_is_named() {
        let packet = |attrs: &str, children: &str| format!(
            concat!(r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
                r#"<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Version="18.7" crs:ProcessVersion="15.4" {}>"#,
                "{}</rdf:Description></rdf:RDF></x:xmpmeta>"), attrs, children);
        let linear = "<crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>";
        // As the sample frame has it: as shot (its kelvin and tint are stated all the same), Shadows +100, the detail panel's defaults.
        let plain = r#"crs:WhiteBalance="As Shot" crs:Temperature="3750" crs:Tint="+27" crs:Exposure2012="0.00" crs:Contrast2012="0" crs:Highlights2012="0" crs:Shadows2012="+100" crs:Whites2012="0" crs:Blacks2012="0" crs:Texture="0" crs:Clarity2012="0" crs:Dehaze="0" crs:Vibrance="0" crs:Saturation="0" crs:Sharpness="40" crs:SharpenRadius="+1.0" crs:SharpenDetail="25" crs:SharpenEdgeMasking="0" crs:LuminanceSmoothing="0" crs:ColorNoiseReduction="25" crs:ColorNoiseReductionDetail="50" crs:ColorNoiseReductionSmoothness="50" crs:HueAdjustmentRed="0" crs:PerspectiveScale="100" crs:DefringePurpleHueLo="30" crs:ColorGradeBlending="50" crs:ConvertToGrayscale="False" crs:ToneCurveName2012="Linear" crs:CameraProfile="Embedded" crs:HasSettings="True" crs:HasCrop="False""#;
        let nested = r#"<crs:FilterList rdf:parseType="Resource"><crs:Filters><rdf:Seq><rdf:li><rdf:Description crs:Name="Enhance" crs:Exposure2012="+4.00" crs:Texture="+99"/></rdf:li></rdf:Seq></crs:Filters></crs:FilterList>"#;
        let r = read_develop_xmp(&packet(plain, &format!("{linear}{nested}"))).unwrap();
        let want = RawSettings { shadows: 100.0, sharpen: crate::raw_sharpen::RawSharpen { amount: 40.0, radius: 1.0, detail: 25.0, masking: 0.0 }, ..Default::default() };
        assert_eq!(r.settings, want);
        assert!(r.not_carried.is_empty(), "{:?}", r.not_carried);
        // A custom white, values past the ranges, and controls this development has not.
        let busy = r#"crs:WhiteBalance="Custom" crs:Temperature="6100" crs:Tint="-8" crs:Exposure2012="+7.50" crs:Contrast2012="+30" crs:Highlights2012="-200" crs:Shadows2012="+55" crs:Whites2012="-10" crs:Blacks2012="+12" crs:Texture="+20" crs:Clarity2012="-15" crs:Dehaze="0" crs:Vibrance="+10" crs:Saturation="0" crs:Sharpness="70" crs:SharpenRadius="+2.0" crs:SharpenDetail="10" crs:SharpenEdgeMasking="35" crs:LuminanceSmoothing="15" crs:ColorNoiseReduction="50" crs:ColorNoiseReductionDetail="50" crs:ColorNoiseReductionSmoothness="70" crs:HueAdjustmentOrange="+6" crs:ColorGradeShadowLum="-4" crs:GrainAmount="12" crs:PostCropVignetteAmount="-20" crs:LensProfileEnable="1" crs:PerspectiveScale="104" crs:RedHue="+3" crs:ConvertToGrayscale="False" crs:ToneCurveName2012="Custom" crs:CameraProfile="Adobe Vivid" crs:HasCrop="True""#;
        let curved = "<crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>128, 140</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012><crs:MaskGroupBasedCorrections><rdf:Seq/></crs:MaskGroupBasedCorrections>";
        let r = read_develop_xmp(&packet(busy, curved)).unwrap();
        let s = r.settings;
        assert_eq!((s.temperature, s.tint, s.exposure, s.contrast, s.highlights, s.shadows, s.whites, s.blacks), (Some(6100.0), Some(-8.0), 5.0, 30.0, -100.0, 55.0, -10.0, 12.0));
        assert_eq!((s.sharpen.amount, s.sharpen.radius, s.sharpen.detail, s.sharpen.masking, s.color_noise), (70.0, 2.0, 10.0, 35.0, 50.0));
        let names: Vec<&str> = r.not_carried.iter().map(|n| n.0).collect();
        for name in ["Texture", "Clarity", "Vibrance", "Luminance noise reduction", "Color noise reduction smoothness", "Grain", "Tone curve",
            "Color mixer", "Color grading", "Calibration", "Vignette", "Lens corrections", "Geometry", "Crop", "Masks", "Profile"] {
            assert!(names.contains(&name), "{name} is not named in {names:?}");
        }
        assert!(!names.contains(&"Dehaze") && !names.contains(&"Saturation") && !names.contains(&"Black & white") && !names.contains(&"Color noise reduction detail"), "{names:?}");
        let value = |name: &str| r.not_carried.iter().find(|n| n.0 == name).map(|n| n.1.as_str());
        assert_eq!((value("Texture"), value("Clarity"), value("Luminance noise reduction"), value("Color noise reduction smoothness"), value("Profile")),
            (Some("+20"), Some("-15"), Some("+15"), Some("70"), Some("Adobe Vivid")));
        // Written as elements, as some tools do: read the same way.
        let elements = "<rdf:Description rdf:about=\"\"><crs:ProcessVersion>15.4</crs:ProcessVersion><crs:WhiteBalance>Custom</crs:WhiteBalance><crs:Temperature>4800</crs:Temperature><crs:Tint>+5</crs:Tint><crs:Exposure2012>+0.35</crs:Exposure2012><crs:Shadows2012>-20</crs:Shadows2012><crs:Sharpness>25</crs:Sharpness></rdf:Description>";
        let r = read_develop_xmp(elements).unwrap();
        assert_eq!((r.settings.temperature, r.settings.tint, r.settings.exposure, r.settings.shadows, r.settings.sharpen.amount), (Some(4800.0), Some(5.0), 0.35, -20.0, 25.0));
    }
}
