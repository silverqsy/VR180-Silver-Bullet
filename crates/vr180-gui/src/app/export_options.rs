//! What an export writes: the export bar's choices, the sizes they come to, and the Format window.

use super::*;

/// Codec choice for the export pipeline. `Spatial` is Apple spatial video:
/// the two eyes of a reframed view as one MV-HEVC stream with the spatial
/// metadata. It exists only where [`spatial_video_available`] says so; any
/// other export it is selected for runs as `H265`
/// ([`ExportOptions::codec_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum ExportCodec { H265, ProRes, Spatial }

/// The note under the Resolution list for a video larger than 8192 × 4096:
/// what the encoder that really runs on this platform does with the frame.
///
/// macOS: VideoToolbox HEVC encodes it in hardware (beyond Level 6.2), the
/// hardware ProRes engine refuses it. Windows (measured on an RTX 4090 at
/// 13120 × 6560): NVENC refuses a frame above 8192 a side, so H.265 lands on
/// libx265 (~0.5 fps there); ProRes keeps its GPU encoder (`prores_ks_vulkan`,
/// ~3.7 fps), so there is nothing to warn about.
pub(super) fn oversize_export_hint(codec: ExportCodec, w: u32, h: u32) -> Option<&'static str> {
    const LEVEL: &str = "Above 8192 × 4096 the file exceeds HEVC Level 6.2 — it encodes in hardware, but most players cannot decode it.";
    if w <= 8192 && h <= 4096 {
        return None;
    }
    if cfg!(target_os = "macos") {
        return Some(match codec {
            ExportCodec::ProRes => "Above 8192 × 4096 the Mac's hardware ProRes encoder refuses the frame — the software encoder takes over (several times slower).",
            ExportCodec::H265 | ExportCodec::Spatial => LEVEL,
        });
    }
    match codec {
        ExportCodec::ProRes => None,
        ExportCodec::H265 | ExportCodec::Spatial if w > 8192 || h > 8192 => Some("Above 8192 pixels a side the NVIDIA hardware encoder (NVENC) refuses the frame — the software encoder (libx265) takes over, many times slower. The file also exceeds HEVC Level 6.2: most players cannot decode it."),
        ExportCodec::H265 | ExportCodec::Spatial => Some(LEVEL),
    }
}

/// Whether a clip with these settings can be exported as spatial video:
/// macOS (the MV-HEVC encoder is VideoToolbox's), the reframed view (a flat
/// frame with a known field of view), and Defish at 0 (a spatial video is
/// rectilinear).
pub(super) fn spatial_video_available(settings: &crate::decoder::Settings) -> bool {
    cfg!(target_os = "macos")
        // Two eye views: a single-eye source has one.
        && !settings.single_eye
        && settings.reframe_aspect_if_active().is_some()
        && settings.reframe_defish.abs() < 0.005
}

/// The widest and the tallest field of view at which Apple Vision Pro's
/// immersive view shows a spatial video at life size. Checked on a headset
/// (visionOS 27) with charts that differed only in the field of view written
/// to the file: a 1:1 frame written as 80° and one written as 120° came out
/// the same size, and so did 16:9 frames written as 90° and 120°.
pub(super) const SPATIAL_LIFE_SIZE_MAX_HFOV_DEG: f32 = 90.0;
pub(super) const SPATIAL_LIFE_SIZE_MAX_VFOV_DEG: f32 = 80.0;

/// The widest horizontal field of view a rectilinear frame of this aspect
/// keeps at life size: 90°, or less when its height reaches 80° first (a
/// 1:1 frame: 80°).
pub(super) fn spatial_life_size_hfov_deg(aspect_w_over_h: f32) -> f32 {
    let half_height = (SPATIAL_LIFE_SIZE_MAX_VFOV_DEG.to_radians() / 2.0).tan();
    let by_height = 2.0 * (half_height * aspect_w_over_h).atan().to_degrees();
    by_height.min(SPATIAL_LIFE_SIZE_MAX_HFOV_DEG)
}

/// What the headset's immersive view makes of a rectilinear spatial frame of
/// `hfov_deg` and this aspect: `None` while the frame is inside the life-size
/// limits, otherwise the `(horizontal, vertical)` field of view the whole
/// frame is scaled down to. The frame keeps its shape, so the tighter of the
/// two limits decides.
pub(super) fn spatial_view_shown_as(hfov_deg: f32, aspect_w_over_h: f32) -> Option<(f32, f32)> {
    use vr180_pipeline::gpu::reframe_vfov_deg;
    // Half of the tenth of a degree the panel prints: a frame that reads as
    // 80.0° tall is at the limit, not past it.
    const SLACK_DEG: f32 = 0.05;
    let vfov_deg = reframe_vfov_deg(hfov_deg, 0.0, aspect_w_over_h);
    if hfov_deg <= SPATIAL_LIFE_SIZE_MAX_HFOV_DEG + SLACK_DEG
        && vfov_deg <= SPATIAL_LIFE_SIZE_MAX_VFOV_DEG + SLACK_DEG
    {
        return None;
    }
    let shown_hfov = spatial_life_size_hfov_deg(aspect_w_over_h);
    Some((shown_hfov, reframe_vfov_deg(shown_hfov, 0.0, aspect_w_over_h)))
}

/// The export panel's note for a spatial video whose view is past the
/// headset's limits. Placeholders: `{max_w}` / `{max_h}` the limits, `{view}`
/// the frame's own field of view, `{shown}` what it is shown as, `{limit}`
/// the widest Zoom that stays at life size.
pub(super) const SPATIAL_VIEW_SCALED_NOTE: &str = "In its immersive view Apple Vision Pro shows a spatial video at life size only up to {max_w}° wide and {max_h}° tall. This view ({view}) is scaled down to {shown}, so everything looks smaller. For life size, set Zoom to {limit}° or less.";

/// [`SPATIAL_VIEW_SCALED_NOTE`] (or its translation) filled in for a view,
/// `None` while the view is inside the limits.
pub(super) fn spatial_view_scaled_note(template: &str, hfov_deg: f32, aspect_w_over_h: f32) -> Option<String> {
    let (shown_h, shown_v) = spatial_view_shown_as(hfov_deg, aspect_w_over_h)?;
    let vfov_deg = vr180_pipeline::gpu::reframe_vfov_deg(hfov_deg, 0.0, aspect_w_over_h);
    Some(template
        .replace("{max_w}", &format!("{SPATIAL_LIFE_SIZE_MAX_HFOV_DEG:.0}"))
        .replace("{max_h}", &format!("{SPATIAL_LIFE_SIZE_MAX_VFOV_DEG:.0}"))
        .replace("{view}", &format!("{hfov_deg:.1}° × {vfov_deg:.1}°"))
        .replace("{shown}", &format!("{shown_h:.1}° × {shown_v:.1}°"))
        .replace("{limit}", &format!("{shown_h:.0}")))
}

/// Output container for a STILL source (Insta360 X6 `.dng`). Videos
/// use `ExportCodec`; stills use this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum StillFormatChoice { Png16, Jpeg, LinearDng }

impl StillFormatChoice {
    pub(super) fn label(self) -> &'static str {
        match self {
            StillFormatChoice::Png16 => "PNG (16-bit)",
            StillFormatChoice::Jpeg => "JPEG",
            StillFormatChoice::LinearDng => "DNG (linear raw)",
        }
    }
    pub(super) fn extension(self) -> &'static str {
        match self { Self::Png16 => "png", Self::Jpeg => "jpg", Self::LinearDng => "dng" }
    }
    pub(super) fn to_pipeline(self) -> vr180_pipeline::still_export::StillFormat {
        use vr180_pipeline::still_export::StillFormat as F;
        match self {
            Self::Png16 => F::Png16,
            Self::Jpeg => F::Jpeg { quality: 92 },
            Self::LinearDng => F::LinearDng { compression: vr180_pipeline::dng_write::DngCompression::LosslessJpegTiles },
        }
    }
}

/// The Lens Space view's two lenses ([`crate::decoder::LensSpaceLens`]), as
/// the Output panel explains them.
pub(super) const LENS_SPACE_CAMERA_HOVER: &str = "Each eye is written the way the camera recorded it, through one lens that the file describes (Apple Projected Media Profile, parametric immersive projection). A fisheye clip keeps everything its lens saw, past 180° where the lens goes that far; a Crop clip fills its frame instead of sitting in a canvas that is mostly black. Stabilization, rolling shutter, eye alignment, the crop mask and colour are applied as always. Apple Vision Pro plays it; other headsets and YouTube need the VR180 format.";
pub(super) const LENS_SPACE_EQUIDISTANT_HOVER: &str = "An ideal equidistant fisheye: a disc in a square frame, the form that players and tools for fisheye video expect. With the Apple Vision Pro metadata on, the file states this lens too, and the headset shows the disc at its true angles.";
pub(super) const LENS_SPACE_CAMERA_NOTE: &str = "The recording's own frame through the camera's lens: its whole field of view, for Apple Vision Pro.";
pub(super) const LENS_SPACE_DISC_NOTE: &str = "equidistant disc, for players and tools that take fisheye video.";
/// What the Format window says of a lens frame (the Lens Space view through
/// the camera's lens).
pub(super) const LENS_FRAME_NOTE: &str = "Lens Space, camera lens: the frame is the recording's own, at the density it was recorded, so the clip's own size already holds all of its detail, and nothing the lens saw is cut off at 180°. It plays immersively on Apple Vision Pro (visionOS 26 or later) and nowhere else: for other headsets and YouTube, export the clip in the VR180 format as well. The file is named …_Lens.";
pub(super) const LENS_FRAME_STAB_NOTE: &str = "Stabilization: the frame has no spare canvas, so a stabilized frame is 5 % larger on each side than the recording, the same for every clip (an 8K copy keeps its size and scales the picture instead). Where the picture does not reach, the frame is black, as around the picture of a VR180 export; a shift beyond the margin is cut at the frame's edge. With camera lock the picture leaves the frame as the camera turns: keep that for a camera that stays put.";
pub(super) const LENS_FRAME_METADATA_NOTE: &str = "A Lens Space file through the camera's lens carries its own metadata: the lens, for Apple Vision Pro. The choice of VR180 metadata applies to the clips that export as VR180.";
pub(super) const LENS_SPACE_DISC_METADATA_NOTE: &str = "This clip exports an equidistant disc, which is not VR180: with Apple Vision Pro chosen the file states the disc's own lens instead, and for YouTube, which has no fisheye form, nothing is written.";

/// The reduced size of a lens frame (the 8K choice): the frame scaled so
/// that its longer side is 4096, its shape kept, each side a multiple of 4.
/// `None` for a frame no larger than that: a lens frame is never enlarged.
pub(super) fn lens_frame_reduced_dims(eye_w: u32, eye_h: u32) -> Option<(u32, u32)> {
    let long = eye_w.max(eye_h);
    if long <= 4096 || eye_w.min(eye_h) == 0 {
        return None;
    }
    let scale = 4096.0 / long as f64;
    let side = |v: u32| (((v as f64 * scale) / 4.0).round() as u32 * 4).max(16);
    Some((side(eye_w), side(eye_h)))
}

/// The largest frame side the platform's hardware H.265 encoder takes, where
/// an export can meet that limit: NVENC stops at 8192 (Windows; measured on
/// an RTX 4090, see [`oversize_export_hint`]). VideoToolbox on a Mac takes
/// every frame the app writes.
pub(super) fn hw_h265_side_limit() -> Option<u32> {
    if cfg!(target_os = "macos") { None } else { Some(8192) }
}

/// What the 8K choice of a STABILIZED lens frame is held to where the
/// hardware encoder has a frame limit ([`hw_h265_side_limit`]) and the
/// recorded frame with the stabilizer's margin, two of them side by side, is
/// past it (a 3840² eye: 4224², 8448 wide): that frame scaled so that its
/// longer side is half the limit, each side a multiple of 4. `None` where the
/// frame fits as it is, or nothing limits it: the 8K choice is then the
/// frame itself.
///
/// The size is larger than the recording. The pipeline reads that as the room
/// the stabilized frame may take, never as an enlargement
/// (`fisheye_export::lens_frame_request`).
pub(super) fn lens_frame_held_dims(eye_w: u32, eye_h: u32, single_eye: bool) -> Option<(u32, u32)> {
    let long = hw_h265_side_limit()? / if single_eye { 1 } else { 2 };
    if eye_w.min(eye_h) == 0 {
        return None;
    }
    let (gw, gh) = vr180_pipeline::lens_frame::stabilized_dims(eye_w, eye_h);
    if gw.max(gh) <= long {
        return None;
    }
    let scale = long as f64 / gw.max(gh) as f64;
    let side = |v: u32| (((v as f64 * scale) / 4.0).round() as u32 * 4).clamp(16, long);
    Some((side(gw), side(gh)))
}

pub(super) const LENS_FRAME_HELD_NOTE: &str = "This size is the stabilized frame scaled to fit the NVIDIA hardware encoder, which takes 8192 pixels a side: a few percent smaller than recorded, and many times faster to export as H.265 than the native size, which is encoded in software.";

/// A lens frame at one of the sizes of a raw take's enhanced frames
/// ([`ExportResolution::enhanced_side`]: the canvases the VR180 view has for
/// them): the recorded frame's shape with its longer side on `side`, each
/// side a multiple of 4. The frame is written at exactly this, the
/// stabilizer's margin inside it (`fisheye_export::lens_frame_request`).
pub(super) fn lens_frame_sized_dims(eye_w: u32, eye_h: u32, side: u32) -> (u32, u32) {
    let long = eye_w.max(eye_h);
    if long == 0 {
        return (side, side);
    }
    let scale = side as f64 / long as f64;
    let fit = |v: u32| (((v as f64 * scale) / 4.0).round() as u32 * 4).clamp(16, side);
    (fit(eye_w), fit(eye_h))
}

/// What the Format window says of a lens frame whose take has enhanced
/// frames, under [`LENS_FRAME_NOTE`].
pub(super) const LENS_FRAME_ENHANCED_NOTE: &str = "This clip's enhanced frames hold more than its own size shows: the two larger sizes keep it. At these the frame is scaled to the size picked, the stabilizer's margin inside it. A stretch without enhanced frames exports at the size picked without them.";

/// Output resolution target for the export. `Native` renders at the
/// source's per-eye dimensions — labeled by the OSV case (3840² per eye
/// → 7680×3840 SBS); a `.360` source actually renders its cross-native
/// 3936² → 7872×3936. `R8k` (the default) renders the PROJECTION at
/// 4096² per eye → 8192×4096 SBS — the projection kernel samples the
/// native-res source directly into the 4096 grid (one resample,
/// end-to-end at output res; sharpen / color / encode all run at 4096),
/// NOT a finished-frame upscale.
///
/// `Sharp` exists for a source whose frame is a CROP of the lens' view (the
/// X6 Crop modes): there the frame is native sensor pixels covering ~97° or
/// ~126°, so a 180° canvas of the frame's own size scales it down. `Sharp`
/// renders the 180° canvas at the pixel density of the frame's centre
/// (`π × the lens' centre focal`: 6560² per eye → 13120 × 6560 on the X6).
/// Offered only for such a clip, whose choice is kept apart from every
/// other clip's ([`ExportOptions::resolution_for`]).
///
/// `R13k` and `R16k` exist for a raw take read from frames a raw converter
/// enhanced (7680² an eye): such frames hold more than 8192 × 4096 shows,
/// and these two canvases (6560² and 8192² per eye) keep it. Offered only
/// where the take's enhanced frames are found, with a choice of its own.
/// A lens frame has the same two: the frame at that size
/// ([`lens_frame_sized_dims`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum ExportResolution { Native, R8k, Sharp, R13k, R16k }

impl ExportResolution {
    pub(super) fn label(self) -> &'static str {
        match self {
            ExportResolution::Native => "7680 × 3840",
            ExportResolution::R8k    => "8192 × 4096",
            ExportResolution::Sharp | ExportResolution::R13k => "13120 × 6560",
            ExportResolution::R16k   => "16384 × 8192",
        }
    }

    /// [`Self::label`] for the frame a source exports: side by side, or one
    /// eye wide for a single-eye source (the same size per eye).
    pub(super) fn frame_label(self, single_eye: bool) -> &'static str {
        match (self, single_eye) {
            (ExportResolution::Native, true) => "3840 × 3840",
            (ExportResolution::R8k, true)    => "4096 × 4096",
            (ExportResolution::Sharp | ExportResolution::R13k, true) => "6560 × 6560",
            (ExportResolution::R16k, true)   => "8192 × 8192",
            (r, false) => r.label(),
        }
    }

    /// Per-eye side of the canvases that exist for enhanced frames.
    pub(super) fn enhanced_side(self) -> Option<u32> {
        match self {
            ExportResolution::R13k => Some(6560),
            ExportResolution::R16k => Some(8192),
            _ => None,
        }
    }
}

/// Whether an export of a clip runs the noise filter, which is what the
/// export bar's "NR" says: a video file does when the clip's Noise Reduction
/// is above zero; a folder of DNG frames and a still are written without
/// it, whatever the clip's setting.
pub(super) fn export_runs_noise_filter(kind: vr180_pipeline::SourceKind, denoise_strength: f32, opts: &ExportOptions) -> bool {
    denoise_strength > 0.0 && !kind.is_still() && !(kind.is_raw_video() && opts.raw_dng_sequence)
}

/// Which list of sizes a clip picks from. Each list has a choice of its
/// own, so a size picked on one sort of clip never changes what a clip of
/// another sort exports at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SizeList {
    /// The clip's own size or 8192 × 4096: every clip but the two below.
    Plain,
    /// A clip whose frame is a crop of the lens' view: its native sharpness too.
    Crop,
    /// A raw take read from enhanced frames: the two larger canvases too.
    Enhanced,
}

impl SizeList {
    /// `cropped`: the clip has a native-sharpness size (`sharp_eye` is
    /// `Some`). `enhanced`: it is a raw take whose enhanced frames the
    /// export reads (never a cropped one).
    pub(super) fn of(cropped: bool, enhanced: bool) -> Self {
        if enhanced { Self::Enhanced } else if cropped { Self::Crop } else { Self::Plain }
    }
}

/// What is known of a raw video clip's enhanced frames (the folder of its
/// camera-original DNGs beside it), for the stretch the clip is trimmed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EnhancedFrames {
    /// The folder is being looked through.
    Looking,
    /// No enhanced frame there for this stretch (or no folder).
    Absent,
    /// Enhanced pictures of this many pixels a side per eye.
    Found(u32),
    /// This build cannot decode enhanced pictures
    /// (`enhanced_dng::decodable`): none is looked for.
    Unreadable,
}

pub(super) const ENHANCED_UNREADABLE_NOTE: &str = "Enhanced frames (a raw converter's Super Resolution, in the folder of Camera original DNGs) are read on macOS only for now: this clip exports from the capture.";

/// A take and its trim, as an enhanced-frames lookup is asked for them.
pub(super) type EnhancedKey = (PathBuf, Option<u64>, Option<u64>);

/// The loaded raw take's enhanced frames, looked up on a thread: listing a
/// folder of thousands of files on a share is not for the UI thread.
#[derive(Default)]
pub(super) struct EnhancedProbe {
    /// What the lookup in flight, or the last one, was asked for.
    pub(super) key: Option<EnhancedKey>,
    /// The last answer for the take of `key`.
    pub(super) side: Option<Option<u32>>,
    pub(super) asked_at: Option<std::time::Instant>,
    pub(super) rx: Option<crossbeam_channel::Receiver<(EnhancedKey, Option<u32>)>>,
}

/// Per-eye size of a reframed export (the side-by-side frame is twice as
/// wide): 1080 / 1440 / 2160 lines for video, and for the 1:1 view also 3840
/// (3840² per eye → a 7680×3840 side-by-side frame, or a spatial video of
/// that size); a STILL may also pick 3240 (16:9 → 5760×3240 per eye) or,
/// for the 1:1 view, 4320 (4320² per eye) — a photo renders from the native
/// lens half, so the only cost is the file. The width follows the view's
/// aspect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum ReframeSize { P1080, P1440, P2160, P3240, P3840, P4320 }

impl ReframeSize {
    pub(super) fn height(self) -> u32 {
        match self {
            Self::P1080 => 1080, Self::P1440 => 1440, Self::P2160 => 2160,
            Self::P3240 => 3240, Self::P3840 => 3840, Self::P4320 => 4320,
        }
    }
    /// The sizes offered for a source. Video stops at 2160 lines in 16:9
    /// (3840×2160 per eye) and at 3840 in the 1:1 view (16:9 at 3840 lines
    /// would be 6826×3840 per eye — a side-by-side frame no hardware
    /// encoder takes). A still adds 3240 for both aspects and 4320 for the
    /// 1:1 view only (16:9 at 4320 would be 7680×4320 per eye).
    pub(super) fn choices(is_still: bool, aspect: crate::decoder::ReframeAspect) -> &'static [ReframeSize] {
        use crate::decoder::ReframeAspect::{Square, Wide};
        match (is_still, aspect) {
            (false, Wide) => &[Self::P1080, Self::P1440, Self::P2160],
            (false, Square) => &[Self::P1080, Self::P1440, Self::P2160, Self::P3840],
            (true, Wide) => &[Self::P1080, Self::P1440, Self::P2160, Self::P3240],
            (true, Square) => &[Self::P1080, Self::P1440, Self::P2160, Self::P3240, Self::P3840, Self::P4320],
        }
    }
    /// The size clamped to what `choices` offers for the source: the size
    /// itself when offered, else the largest offered one below it — a
    /// still-only pick remembered from a photo must not size a video, and
    /// the 1:1 view's 3840 must not carry into 16:9.
    pub(super) fn clamped(self, is_still: bool, aspect: crate::decoder::ReframeAspect) -> Self {
        let allowed = Self::choices(is_still, aspect);
        allowed.iter().rev().copied()
            .find(|s| s.height() <= self.height())
            .unwrap_or(allowed[0])
    }
    pub(super) fn eye_dims(self, aspect: crate::decoder::ReframeAspect) -> (u32, u32) {
        let h = self.height();
        let w = (((h as f32 * aspect.ratio()).round() as u32) + 1) & !1;
        (w, h)
    }
    pub(super) fn label(self, aspect: crate::decoder::ReframeAspect) -> String {
        let (w, h) = self.eye_dims(aspect);
        format!("{} × {}  ({w} × {h} per eye)", 2 * w, h)
    }
    /// [`Self::label`] for the frame a source exports: a single-eye source's
    /// frame is the one eye.
    pub(super) fn frame_label(self, aspect: crate::decoder::ReframeAspect, single_eye: bool) -> String {
        if !single_eye { return self.label(aspect); }
        let (w, h) = self.eye_dims(aspect);
        format!("{w} × {h}")
    }
    /// Recommended H.265 average bitrate (Mbps) for a reframed export of
    /// this size: ~0.28 bits per pixel per frame over the side-by-side
    /// frame at 30 fps, ×1.5 at 60 fps (linear between, flat below 30),
    /// rounded to 5 and kept inside the slider's 10..=300 range. A flat
    /// rectilinear frame is ordinary video — every pixel is seen and none
    /// are stretched — so it needs a fraction of the VR180 rate:
    /// 3840×1080 → 35, 5120×1440 → 60, 7680×2160 → 140 Mbps at 30 fps.
    pub(super) fn recommended_bitrate_mbps(self, aspect: crate::decoder::ReframeAspect, fps: f32) -> u32 {
        self.recommended_bitrate_mbps_for(aspect, fps, false)
    }
    /// [`Self::recommended_bitrate_mbps`] by the pixels of the frame really
    /// written: a single-eye source's frame holds one eye, half as many.
    pub(super) fn recommended_bitrate_mbps_for(self, aspect: crate::decoder::ReframeAspect, fps: f32, single_eye: bool) -> u32 {
        let (w, h) = self.eye_dims(aspect);
        let pixels = (if single_eye { w * h } else { 2 * w * h }) as f32;
        let fps_factor = 1.0 + 0.5 * ((fps - 30.0) / 30.0).clamp(0.0, 1.0);
        let mbps = pixels * 30.0 * 0.28 * fps_factor / 1.0e6;
        (((mbps / 5.0).round() as u32) * 5).clamp(10, 300)
    }
}

impl ExportCodec {
    pub(super) fn label(self) -> &'static str {
        match self {
            ExportCodec::H265 => "H.265 (HEVC)",
            ExportCodec::ProRes => "ProRes",
            ExportCodec::Spatial => tr("Spatial video (MV-HEVC)"),
        }
    }
}

/// ProRes profiles. Numbers match ffmpeg's `-profile:v N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum ProResProfile {
    Proxy = 0,
    Lt = 1,
    Standard = 2,
    Hq = 3,
    P4444 = 4,
    P4444Xq = 5,
}

impl ProResProfile {
    pub(super) fn label(self) -> &'static str {
        match self {
            ProResProfile::Proxy    => "Proxy",
            ProResProfile::Lt       => "LT",
            ProResProfile::Standard => "Standard",
            ProResProfile::Hq       => "HQ",
            ProResProfile::P4444    => "4444",
            ProResProfile::P4444Xq  => "4444 XQ",
        }
    }
}

/// (Saved with a project and with the session: a field a file does not hold
/// takes its default.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ExportOptions {
    pub(super) codec: ExportCodec,
    /// A raw video clip exports as a folder of DNG frames, the developed
    /// view, instead of a video file (used only when the source is a raw
    /// video).
    pub(super) raw_dng_sequence: bool,
    /// Read from a file of a version in which the camera's own frames were
    /// a kind of that folder, and never written: they are no export any
    /// more ([`TakeJob::Originals`]), and a file that had picked them must
    /// not export the developed view in their place ([`Self::settled`]).
    #[serde(rename = "raw_dng_original", skip_serializing)]
    pub(super) was_dng_original: bool,
    /// A raw video clip's frames come from the folder of its camera-original
    /// DNGs where a raw converter has enhanced them (the folder beside the
    /// take, see `default_frames_dir_for`).
    pub(super) raw_use_enhanced: bool,
    /// Still-photo container (used only when the source is a still).
    pub(super) still_format: StillFormatChoice,
    /// Output resolution target (Native / 8192×4096) of every clip but a
    /// Crop one. Read it through [`ExportOptions::resolution_for`].
    pub(super) resolution: ExportResolution,
    /// The same choice for a clip whose frame is a crop of the lens' view
    /// (Native / 8192×4096 / its native sharpness). Its own, so a list that
    /// holds both sorts exports each at the size picked on a clip of its
    /// sort: "its own size" means something else on either side (the whole
    /// canvas for one, a smaller canvas than was recorded for the other).
    pub(super) crop_resolution: ExportResolution,
    /// The same choice for a raw take read from enhanced frames (Native /
    /// 8192×4096 / 13120×6560 / 16384×8192): its own for the same reason.
    pub(super) enhanced_resolution: ExportResolution,
    /// H.265 average bitrate in Mbps for VR180 output. Range 20..=1000
    /// (a native 12K export wants the top of it; the slider is logarithmic
    /// so the everyday 50–300 Mbps band keeps its resolution).
    pub(super) h265_bitrate_mbps: u32,
    /// H.265 average bitrate in Mbps for REFRAMED (Flat 3D) output. Range
    /// 10..=300, kept apart from the VR180 rate: a flat rectilinear frame
    /// needs a fraction of what a half-equirect does, and the two must not
    /// bleed into each other when the Format switches.
    pub(super) reframe_bitrate_mbps: u32,
    /// `false` → the reframed rate follows the size / aspect / frame-rate
    /// recommendation (`ReframeSize::recommended_bitrate_mbps`), re-derived
    /// per clip at export time. A manual slider change pins the value
    /// (`true`) until "Use recommended" is pressed.
    pub(super) reframe_bitrate_custom: bool,
    /// 8 or 10 (Main / Main10) for H.265.
    pub(super) h265_bit_depth: u8,
    /// H.265 encoder: hardware (NVIDIA NVENC) vs software (libx265).
    /// Hardware is the default on Windows — it's many times faster at
    /// VR180 resolutions. macOS always uses VideoToolbox regardless.
    /// Falls back to libx265 automatically if NVENC can't be opened.
    pub(super) h265_hardware: bool,
    pub(super) prores_profile: ProResProfile,
    /// Inject Apple Projected Media Profile (Vision Pro VR180).
    pub(super) inject_apmp: bool,
    /// Inject Spatial Media V2 (YouTube VR180) — st3d + sv3d boxes.
    /// MUTUALLY EXCLUSIVE with `inject_apmp` (they overwrite the same
    /// atoms in the visual sample entry) — the GUI enforces it.
    pub(super) inject_youtube: bool,
    /// Reframed exports: per-eye size (the aspect comes from the clip).
    pub(super) reframe_size: ReframeSize,
    /// Spatial video: distance between the two lenses in mm. Typical
    /// 60..=70. (VR180 metadata carries no baseline: it is optional there
    /// and the player does not use it.)
    pub(super) spatial_baseline_mm: f32,
    /// Spatial video: horizontal disparity adjustment in percent of the
    /// frame width. Positive moves the whole scene away from the viewer.
    pub(super) spatial_disparity_pct: f32,
    /// "BeyondVR Hack": at the final output stage, scale each eye of the
    /// SBS half-equirect frame about its own half-center, padding the
    /// border with black. No-op for fisheye output and in preview.
    pub(super) beyondvr_hack: bool,
    /// Per-eye scale in percent when the hack is on.
    pub(super) beyondvr_scale_pct: f32,
}

impl ExportOptions {
    /// The resolution choice that applies to a clip picking from `list`. A
    /// size its list does not hold means 8192 × 4096 there.
    pub(super) fn resolution_for(&self, list: SizeList) -> ExportResolution {
        use ExportResolution::{Native, R13k, R16k, R8k, Sharp};
        match list {
            SizeList::Plain => match self.resolution { Native => Native, _ => R8k },
            SizeList::Crop => match self.crop_resolution { r @ (Native | Sharp) => r, _ => R8k },
            SizeList::Enhanced => match self.enhanced_resolution { r @ (Native | R13k | R16k) => r, _ => R8k },
        }
    }

    /// The choice [`Self::resolution_for`] reads, for the control that edits it.
    pub(super) fn resolution_for_mut(&mut self, list: SizeList) -> &mut ExportResolution {
        match list {
            SizeList::Plain => &mut self.resolution,
            SizeList::Crop => &mut self.crop_resolution,
            SizeList::Enhanced => &mut self.enhanced_resolution,
        }
    }

    /// Per-eye size of the canvas a raw take is written on when the export
    /// reads its enhanced frames: the size picked for those. `native_eye`:
    /// the take's own eye.
    pub(super) fn enhanced_eye_dims(&self, native_eye: (u32, u32)) -> (u32, u32) {
        match self.resolution_for(SizeList::Enhanced) {
            ExportResolution::Native => native_eye,
            r => { let side = r.enhanced_side().unwrap_or(4096); (side, side) }
        }
    }

    /// Per-eye size of a lens frame for a clip whose eye is `eye_w × eye_h`:
    /// the frame as it was recorded, whatever its shape — for a Crop clip
    /// that is its native sharpness, so that choice means the same here —
    /// or its reduced copy for the 8K choice ([`lens_frame_reduced_dims`]; a
    /// frame no larger than that is never enlarged). The choice is the one
    /// of the clip's `list` ([`Self::resolution_for`]).
    ///
    /// One more meaning of the 8K choice, for a `stabilized` frame where the
    /// hardware encoder has a frame limit: the size the frame with its margin
    /// is held to ([`lens_frame_held_dims`]).
    ///
    /// A raw take read from its enhanced frames ([`SizeList::Enhanced`]) has
    /// the two larger sizes of the VR180 view as well: the frame at that
    /// size ([`lens_frame_sized_dims`]).
    pub(super) fn lens_frame_eye_dims(&self, eye_w: u32, eye_h: u32, list: SizeList, stabilized: bool, single_eye: bool) -> (u32, u32) {
        match self.resolution_for(list) {
            ExportResolution::R8k => lens_frame_reduced_dims(eye_w, eye_h)
                .or_else(|| lens_frame_held_dims(eye_w, eye_h, single_eye).filter(|_| stabilized))
                .unwrap_or((eye_w, eye_h)),
            r => match r.enhanced_side() {
                Some(side) => lens_frame_sized_dims(eye_w, eye_h, side),
                None => (eye_w, eye_h),
            },
        }
    }

    /// Per-eye size a lens-frame export is WRITTEN at:
    /// [`Self::lens_frame_eye_dims`], and for a stabilized clip at its own
    /// size the stabilizer's margin on top (`vr180_pipeline::lens_frame`; a
    /// reduced copy keeps its size, and so does a size of the enhanced
    /// frames). What the export bar shows and a snapshot renders.
    pub(super) fn lens_frame_out_dims(&self, eye_w: u32, eye_h: u32, list: SizeList, stabilized: bool, single_eye: bool) -> (u32, u32) {
        let (w, h) = self.lens_frame_eye_dims(eye_w, eye_h, list, stabilized, single_eye);
        if stabilized && (w, h) == (eye_w, eye_h) {
            vr180_pipeline::lens_frame::stabilized_dims(w, h)
        } else {
            (w, h)
        }
    }

    /// The options as read from a file: one written when the camera's own
    /// frames were a kind of DNG sequence, with that kind picked, exports a
    /// video again.
    pub(super) fn settled(mut self) -> Self {
        if std::mem::take(&mut self.was_dng_original) {
            self.raw_dng_sequence = false;
        }
        self
    }

    /// The codec an export of a clip with `settings` really uses: spatial
    /// video only where it exists ([`spatial_video_available`], and never
    /// for a still); everywhere else that choice runs as H.265. The choice
    /// itself stays put, so a batch of mixed clips exports each as it can.
    pub(super) fn codec_for(&self, settings: &crate::decoder::Settings, is_still: bool) -> ExportCodec {
        match self.codec {
            ExportCodec::Spatial if is_still || !spatial_video_available(settings) => ExportCodec::H265,
            c => c,
        }
    }

    /// The H.265 rate an export actually uses: the VR180 slider value, or
    /// in Reframed mode the reframed rate — the recommendation for this
    /// size / aspect / `fps` unless the user pinned a custom value.
    pub(super) fn effective_h265_mbps(&self, reframe: Option<crate::decoder::ReframeAspect>, fps: f32, single_eye: bool) -> u32 {
        match reframe {
            Some(_) if self.reframe_bitrate_custom => self.reframe_bitrate_mbps,
            Some(aspect) => self.reframe_size.recommended_bitrate_mbps_for(aspect, fps, single_eye),
            None => self.h265_bitrate_mbps,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            codec: ExportCodec::H265,
            raw_dng_sequence: false,
            was_dng_original: false,
            raw_use_enhanced: true,
            still_format: StillFormatChoice::Png16,
            resolution: ExportResolution::R8k,
            crop_resolution: ExportResolution::R8k,
            enhanced_resolution: ExportResolution::R8k,
            h265_bitrate_mbps: 200,
            reframe_bitrate_mbps: 35,
            reframe_bitrate_custom: false,
            h265_bit_depth: 10,
            h265_hardware: true,
            prores_profile: ProResProfile::Standard,
            // APMP is the headline target (Vision Pro). They're
            // mutually exclusive; the dialog enforces it.
            inject_apmp: true,
            inject_youtube: false,
            spatial_baseline_mm: 65.0,
            // Apple's own default: what the lenses saw at infinity lands a
            // little behind the window instead of on its surface.
            spatial_disparity_pct: 2.0,
            reframe_size: ReframeSize::P1080,
            beyondvr_hack: false,
            beyondvr_scale_pct: 85.0,
        }
    }
}

impl App {
    /// The export option that takes a raw clip's frames from the folder of
    /// its camera-original DNGs where a raw converter has enhanced them, and
    /// what that folder was found to hold.
    pub(super) fn enhanced_frames_ui(ui: &mut egui::Ui, opts: &mut ExportOptions, found: EnhancedFrames) {
        if found == EnhancedFrames::Unreadable {
            ui.label(RichText::new(tr(ENHANCED_UNREADABLE_NOTE)).small().color(Color32::GRAY));
            ui.add_space(8.0);
            return;
        }
        ui.checkbox(&mut opts.raw_use_enhanced, tr("Use enhanced frames"));
        ui.label(RichText::new(tr("When the folder of Camera original DNGs is beside the take and a raw converter has enhanced its frames in place (Camera Raw's Super Resolution), each frame is taken from there, at the enhanced size. A frame that was not enhanced comes from the capture."))
            .small().color(Color32::GRAY));
        if opts.raw_use_enhanced {
            let (text, color) = match found {
                EnhancedFrames::Found(side) => (format!("{} {side} × {side} {}", tr("Enhanced frames found:"), tr("per eye")), Color32::from_rgb(120, 190, 130)),
                EnhancedFrames::Absent => (tr("No enhanced frame found for this clip: it exports from the capture.").to_string(), Color32::GRAY),
                EnhancedFrames::Looking => (tr("Looking for enhanced frames…").to_string(), Color32::GRAY),
                EnhancedFrames::Unreadable => (tr(ENHANCED_UNREADABLE_NOTE).to_string(), Color32::GRAY),
            };
            ui.label(RichText::new(text).small().color(color));
        }
        ui.add_space(8.0);
    }

    /// The loaded raw take's enhanced frames (`None` for any other clip), as
    /// last looked up. Asks again, on a thread, when the take or its trim
    /// changed and every few seconds while the Format window is open, so
    /// frames enhanced meanwhile show up.
    pub(super) fn enhanced_frames(&mut self, ctx: &egui::Context) -> Option<EnhancedFrames> {
        // A take of one lens has none: its frames come from the capture.
        if !self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video() && !c.raw_one_lens) {
            return None;
        }
        let path = self.loaded_path.clone()?;
        if !vr180_pipeline::enhanced_dng::decodable() {
            return Some(EnhancedFrames::Unreadable);
        }
        if !self.export_opts.raw_use_enhanced {
            return Some(EnhancedFrames::Absent);
        }
        let key: EnhancedKey = (path.clone(), self.settings.trim_in_s.map(f64::to_bits), self.settings.trim_out_s.map(f64::to_bits));
        let refresh = std::time::Duration::from_secs(4);
        let probe = &mut self.enhanced_probe;
        if let Some(rx) = &probe.rx {
            match rx.try_recv() {
                Ok((asked, side)) => {
                    if asked.0 == path {
                        probe.side = Some(side);
                    }
                    probe.rx = None;
                    ctx.request_repaint();
                }
                Err(crossbeam_channel::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
                Err(crossbeam_channel::TryRecvError::Disconnected) => probe.rx = None,
            }
        }
        let other_take = probe.key.as_ref().map(|k| &k.0) != Some(&path);
        if other_take {
            probe.side = None;
        }
        let due = probe.key.as_ref() != Some(&key)
            || (self.show_export_settings && probe.asked_at.map_or(true, |t| t.elapsed() > refresh));
        if due && probe.rx.is_none() {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let asked = key.clone();
            let (t_in, t_out) = (self.settings.trim_in_s, self.settings.trim_out_s);
            let repaint = ctx.clone();
            std::thread::spawn(move || {
                let source = vr180_pipeline::x6_raw::EnhancedSource { dir: Self::originals_dir_for(&asked.0), t_in, t_out };
                let side = vr180_pipeline::x6_raw::enhanced_side(&asked.0, &source);
                let _ = tx.send((asked, side));
                repaint.request_repaint();
            });
            probe.key = Some(key);
            probe.asked_at = Some(std::time::Instant::now());
            probe.rx = Some(rx);
        }
        Some(match probe.side {
            None => EnhancedFrames::Looking,
            Some(None) => EnhancedFrames::Absent,
            Some(Some(side)) => EnhancedFrames::Found(side),
        })
    }

    /// The side of the loaded raw take's enhanced frames where an export
    /// would read them now: the last lookup's answer, the option on.
    pub(super) fn enhanced_in_use(&self) -> Option<u32> {
        let raw = self.clip.as_ref().is_some_and(|c| c.source_kind.is_raw_video());
        let same_take = self.enhanced_probe.key.as_ref().map(|k| &k.0) == self.loaded_path.as_ref();
        if raw && same_take && self.export_opts.raw_use_enhanced { self.enhanced_probe.side.flatten() } else { None }
    }

    /// The export-output controls (resolution / codec / codec-specific /
    /// VR180 metadata) shown in the export bar's Format dropdown. `id_prefix`
    /// keeps the ComboBox ids distinct.
    /// `reframe_view` = the reframed view's `(horizontal FOV °, Defish)`,
    /// read only while `reframe` is `Some` (the spatial video controls).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn export_options_ui(
        ui: &mut egui::Ui, opts: &mut ExportOptions, id_prefix: &str,
        reframe: Option<crate::decoder::ReframeAspect>,
        reframe_view: (f32, f32),
        fps: f32,
        is_still: bool,
        native_dims: Option<(u32, u32)>,
        sharp_dims: Option<(u32, u32)>,
        // The loaded clip is in the Lens Space view, and which lens that
        // writes for it: the camera's (a lens frame) or the equidistant
        // disc. `None` in every other view.
        lens_space: Option<crate::decoder::LensSpaceLens>,
        // The loaded clip is stabilized: its lens frame then carries the
        // stabilizer's margin, and the sizes shown include it.
        lens_frame_stabilized: bool,
        // The clip holds ONE eye: its frame is that eye (sizes and rates
        // follow), and nothing that needs two eyes is offered.
        single_eye: bool,
        // The clip is a raw video (it can also leave as a folder of DNG
        // frames), and what is known of its enhanced frames; `None` for any
        // other clip.
        raw_video: Option<EnhancedFrames>,
    ) {
        // Spatial video is offered for the reframed view on macOS. Selected
        // on a clip it does not exist for, the choice shows (and exports) as
        // H.265 and stays put for the clips that can use it.
        let spatial_offered = cfg!(target_os = "macos") && reframe.is_some() && !is_still && !single_eye;
        let (reframe_hfov_deg, reframe_defish) = reframe_view;
        let spatial_rectilinear = reframe_defish.abs() < 0.005;
        let codec_shown = match opts.codec {
            ExportCodec::Spatial if !spatial_offered => ExportCodec::H265,
            c => c,
        };
        if let Some(aspect) = reframe {
            // Reframed view: the exact viewport at a per-eye size.
            ui.label(RichText::new(tr(if single_eye { "Size" } else { "Per-eye size" })).strong());
            opts.reframe_size = opts.reframe_size.clamped(is_still, aspect);
            egui::ComboBox::from_id_source(format!("{id_prefix}_reframe_size"))
                .selected_text(opts.reframe_size.frame_label(aspect, single_eye))
                .width(260.0)
                .show_ui(ui, |ui| {
                    for &sz in ReframeSize::choices(is_still, aspect) {
                        ui.selectable_value(&mut opts.reframe_size, sz, sz.frame_label(aspect, single_eye));
                    }
                });
            if codec_shown != ExportCodec::Spatial {
                ui.label(RichText::new(tr("Reframed output is not VR180 — no projection metadata is written."))
                    .small().color(Color32::GRAY));
            }
        } else {
            ui.label(RichText::new(tr("Resolution")).strong());
            // Native = the clip's own size (full resolution): 7680 × 3840 for
            // an Osmo 360, 12288 × 6144 for a 12K BRAW.
            let native_label: String = match native_dims {
                Some((w, h)) => format!("{w} × {h} ({})", if is_still { tr("full") } else { tr("native") }),
                None if is_still => "native (full)".to_string(),
                None => ExportResolution::Native.frame_label(single_eye).to_string(),
            };
            // Native sharpness — only a clip whose frame is a crop of the
            // lens' view has it (the X6 Crop modes). Such a clip edits its
            // own choice, every other clip the general one: neither changes
            // what a clip of the other sort exports at.
            let sharp_label: Option<String> = sharp_dims.filter(|_| !is_still)
                .map(|(w, h)| format!("{w} × {h} ({})", tr("native sharpness")));
            let cropped = sharp_label.is_some();
            // Lens Space through the camera's lens: a lens frame, the
            // recording's own frame instead of a canvas.
            let lens_frame = lens_space == Some(crate::decoder::LensSpaceLens::Camera);
            // A stabilized lens frame is the recorded frame plus the
            // stabilizer's margin (`vr180_pipeline::lens_frame`): the size
            // it is written at.
            let grown_dims = native_dims.filter(|_| lens_frame && lens_frame_stabilized).map(|(w, h)| {
                let (ew, eh) = vr180_pipeline::lens_frame::stabilized_dims(if single_eye { w } else { w / 2 }, h);
                (if single_eye { ew } else { ew * 2 }, eh)
            });
            let native_label = match grown_dims {
                Some((w, h)) => format!("{w} × {h} ({})", tr("native + margin")),
                None => native_label,
            };
            // A raw take whose enhanced frames were found picks from the
            // list that holds the two larger sizes: canvases in the VR180
            // view, the frame itself at that size for a lens frame.
            let enhanced_sizes = matches!(raw_video, Some(EnhancedFrames::Found(_))) && opts.raw_use_enhanced && !is_still;
            let list = SizeList::of(cropped, enhanced_sizes);
            let shown = opts.resolution_for(list);
            // The 8K choice of a lens frame is its reduced copy, listed only
            // where the frame is larger than that — or, for a stabilized
            // frame past the hardware encoder's limit, the size that frame
            // is held to (`lens_frame_held_dims`).
            let lens_eye = native_dims.filter(|_| lens_frame).map(|(w, h)| (if single_eye { w } else { w / 2 }, h));
            let lens_held = lens_eye.filter(|_| lens_frame_stabilized)
                .filter(|&(ew, eh)| lens_frame_reduced_dims(ew, eh).is_none())
                .and_then(|(ew, eh)| lens_frame_held_dims(ew, eh, single_eye));
            let lens_reduced: Option<String> = lens_eye
                .and_then(|(ew, eh)| lens_frame_reduced_dims(ew, eh).or(lens_held))
                .map(|(rw, rh)| format!("{} × {rh}", if single_eye { rw } else { rw * 2 }));
            let r8k_label: String = lens_reduced.clone()
                .unwrap_or_else(|| ExportResolution::R8k.frame_label(single_eye).to_string());
            let r8k_listed = !lens_frame || lens_reduced.is_some();
            // The two sizes of a take's enhanced frames, as the frame an
            // export writes: the canvas, or a lens frame of that size
            // (the recorded frame's shape).
            let sized_dims = |size: ExportResolution| -> Option<(u32, u32)> {
                let side = size.enhanced_side()?;
                let (w, h) = match lens_eye {
                    Some((ew, eh)) => lens_frame_sized_dims(ew, eh, side),
                    None => (side, side),
                };
                Some((if single_eye { w } else { w * 2 }, h))
            };
            let sized_label = |size: ExportResolution| -> String {
                sized_dims(size).map_or_else(|| size.frame_label(single_eye).to_string(), |(w, h)| format!("{w} × {h}"))
            };
            let shown_sized = sized_label(shown);
            // In a lens frame the clip's own size IS its native sharpness:
            // that choice shows as the native size and is not listed apart.
            let sel_label: &str = match (shown, sharp_label.as_deref()) {
                (ExportResolution::Native, _) => &native_label,
                (ExportResolution::Sharp, Some(_)) if lens_frame => &native_label,
                (ExportResolution::Sharp, Some(l)) => l,
                (ExportResolution::R13k | ExportResolution::R16k, _) => &shown_sized,
                (ExportResolution::R8k, _) if !r8k_listed => &native_label,
                _ => &r8k_label,
            };
            let choice = opts.resolution_for_mut(list);
            *choice = shown;
            egui::ComboBox::from_id_source(format!("{id_prefix}_resolution"))
                .selected_text(sel_label)
                .width(260.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(choice,
                        ExportResolution::Native, native_label.clone());
                    if r8k_listed {
                        ui.selectable_value(choice, ExportResolution::R8k, r8k_label.clone());
                    }
                    if let Some(l) = sharp_label.as_ref().filter(|_| !lens_frame) {
                        ui.selectable_value(choice, ExportResolution::Sharp, l.clone());
                    }
                    if list == SizeList::Enhanced {
                        for size in [ExportResolution::R13k, ExportResolution::R16k] {
                            ui.selectable_value(choice, size, sized_label(size));
                        }
                    }
                });
            if lens_frame {
                ui.label(RichText::new(tr(LENS_FRAME_NOTE)).small().color(Color32::GRAY));
                ui.label(RichText::new(tr(LENS_FRAME_STAB_NOTE)).small().color(Color32::GRAY));
                if lens_held.is_some() && opts.resolution_for(list) == ExportResolution::R8k {
                    ui.label(RichText::new(tr(LENS_FRAME_HELD_NOTE)).small().color(Color32::GRAY));
                }
                if list == SizeList::Enhanced {
                    ui.label(RichText::new(tr(LENS_FRAME_ENHANCED_NOTE)).small().color(Color32::GRAY));
                }
            } else if sharp_label.is_some() {
                ui.label(RichText::new(tr("Native sharpness: the 180° canvas at the pixel density of the Crop frame's centre, so nothing recorded is scaled down. The clip's own size puts the same frame on a smaller canvas."))
                    .small().color(Color32::GRAY));
            } else if list == SizeList::Enhanced {
                ui.label(RichText::new(tr("This clip's enhanced frames hold more than 8192 × 4096 shows: the two larger sizes keep it. A stretch without enhanced frames exports at the size picked without them."))
                    .small().color(Color32::GRAY));
            }
            // Beyond 8192×4096 (a 12K BRAW, a Crop clip's native-sharpness
            // canvas): what the platform's encoder does with the frame —
            // see `oversize_export_hint`.
            let native_out = grown_dims.or(native_dims);
            let out_dims = match opts.resolution_for(list) {
                ExportResolution::Native => native_out,
                ExportResolution::Sharp if lens_frame => native_out,
                ExportResolution::Sharp => sharp_dims,
                r @ (ExportResolution::R13k | ExportResolution::R16k) => sized_dims(r),
                ExportResolution::R8k if !r8k_listed => native_out,
                ExportResolution::R8k => None,
            };
            if !is_still {
                if let Some(hint) = out_dims.and_then(|(w, h)| oversize_export_hint(codec_shown, w, h)) {
                    ui.label(RichText::new(tr(hint)).small().color(Color32::GRAY));
                }
            }
        }
        ui.add_space(8.0);

        if is_still {
            ui.label(RichText::new(tr("Still format")).strong());
            egui::ComboBox::from_id_source(format!("{id_prefix}_still_format"))
                .selected_text(opts.still_format.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for f in [StillFormatChoice::Png16, StillFormatChoice::Jpeg, StillFormatChoice::LinearDng] {
                        ui.selectable_value(&mut opts.still_format, f, f.label());
                    }
                });
            ui.label(RichText::new(tr("DNG keeps the camera-native linear data with the raw colour tags — grade it in a raw developer."))
                .small().color(Color32::GRAY));
            if opts.still_format == StillFormatChoice::LinearDng {
                ui.label(RichText::new(tr("Your grade travels as metadata: white balance → AsShotNeutral, lift / gamma / gain → profile tone curve, saturation → profile look table (Adobe); per-eye trims are baked. Not carried: the Tone section, 3D LUT, sharpen."))
                    .small().color(Color32::GRAY));
            }
            ui.add_space(8.0);
            return;
        }

        if let Some(found) = raw_video {
            // (The take itself, as an archive copy or as the camera's own
            // frames, is written from the Source panel: neither is a view.)
            ui.checkbox(&mut opts.raw_dng_sequence, tr("DNG sequence"));
            ui.label(RichText::new(tr("Every frame as a linear DNG in a folder, with this clip's raw development settings inside each file."))
                .small().color(Color32::GRAY));
            ui.add_space(8.0);
            Self::enhanced_frames_ui(ui, opts, found);
            if opts.raw_dng_sequence {
                return;
            }
        }

        ui.label(RichText::new(tr("Codec")).strong());
        let mut codec_pick = codec_shown;
        egui::ComboBox::from_id_source(format!("{id_prefix}_codec"))
            .selected_text(codec_shown.label())
            .width(220.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut codec_pick, ExportCodec::H265,   ExportCodec::H265.label());
                ui.selectable_value(&mut codec_pick, ExportCodec::ProRes, ExportCodec::ProRes.label());
                if spatial_offered {
                    ui.selectable_value(&mut codec_pick, ExportCodec::Spatial, ExportCodec::Spatial.label());
                }
            });
        // Only an actual pick writes: showing H.265 for a clip that cannot
        // be a spatial video must not drop the Spatial choice.
        if codec_pick != codec_shown { opts.codec = codec_pick; }
        let codec_shown = match opts.codec {
            ExportCodec::Spatial if !spatial_offered => ExportCodec::H265,
            c => c,
        };
        ui.add_space(8.0);

        match codec_shown {
            ExportCodec::H265 | ExportCodec::Spatial => {
                ui.label(RichText::new(if codec_shown == ExportCodec::Spatial {
                    tr("Bitrate (both eyes)")
                } else {
                    tr("H.265 bitrate")
                }).strong());
                if let Some(aspect) = reframe {
                    // Reframed: its own rate, seeded from the size / aspect /
                    // frame-rate recommendation until the user pins a value
                    // (dragging back onto the recommendation un-pins it).
                    let rec = opts.reframe_size.recommended_bitrate_mbps_for(aspect, fps, single_eye);
                    if !opts.reframe_bitrate_custom {
                        opts.reframe_bitrate_mbps = rec;
                    }
                    if ui.add(egui::Slider::new(&mut opts.reframe_bitrate_mbps, 10..=300)
                        .text("Mbps")).changed()
                    {
                        opts.reframe_bitrate_custom = opts.reframe_bitrate_mbps != rec;
                    }
                    let (w, h) = opts.reframe_size.eye_dims(aspect);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!(
                            "{}: {rec} Mbps · {} × {h} @ {fps:.0} fps", tr("Recommended"),
                            if single_eye { w } else { 2 * w },
                        )).small().color(Color32::GRAY));
                        if opts.reframe_bitrate_custom
                            && ui.small_button(tr("Use recommended")).clicked()
                        {
                            opts.reframe_bitrate_mbps = rec;
                            opts.reframe_bitrate_custom = false;
                        }
                    });
                } else {
                    ui.add(egui::Slider::new(&mut opts.h265_bitrate_mbps, 20..=1000)
                        .logarithmic(true)
                        .text("Mbps"));
                }
                ui.add_space(4.0);
                if codec_shown == ExportCodec::Spatial {
                    // One MV-HEVC Main10 stream; the frame's own field of
                    // view, the lens spacing and the disparity adjustment
                    // are what makes a headset show it as a spatial video.
                    ui.label(RichText::new(format!(
                        "MV-HEVC Main10 · {}: {reframe_hfov_deg:.1}°", tr("Field of view written"),
                    )).small().color(Color32::GRAY));
                    // A view past what the headset shows at life size: say
                    // what it becomes there. (With Defish above 0 the clip
                    // is not exported as spatial at all; that note is below.)
                    if let Some(note) = reframe.filter(|_| spatial_rectilinear).and_then(|aspect| {
                        spatial_view_scaled_note(tr(SPATIAL_VIEW_SCALED_NOTE), reframe_hfov_deg, aspect.ratio())
                    }) {
                        ui.label(RichText::new(note).small().color(Color32::from_rgb(230, 160, 90)));
                    }
                    ui.horizontal(|ui| {
                        ui.label(tr("Camera baseline"));
                        ui.add(egui::Slider::new(&mut opts.spatial_baseline_mm, 30.0..=120.0)
                            .suffix(" mm").fixed_decimals(1));
                    });
                    ui.horizontal(|ui| {
                        ui.label(tr("Disparity adjustment"));
                        ui.add(egui::Slider::new(&mut opts.spatial_disparity_pct, -5.0..=10.0)
                            .suffix(" %").fixed_decimals(1).step_by(0.1));
                    });
                    ui.label(RichText::new(tr("Disparity adjustment moves the whole scene away from you (positive) or towards you. 2 % keeps the far background just behind the window."))
                        .small().color(Color32::GRAY));
                    if !spatial_rectilinear {
                        ui.label(RichText::new(tr("A spatial video must be rectilinear: with Defish above 0 this clip exports as H.265 side by side instead. Set Defish to 0 in the reframed view."))
                            .small().color(Color32::from_rgb(230, 160, 90)));
                    }
                    ui.label(RichText::new(tr("Plays as a spatial video on Apple Vision Pro; other players show the left eye in 2D."))
                        .small().color(Color32::GRAY));
                } else {
                // 10-bit (Main10) is the default and stays 10-bit
                // end-to-end; 8-bit (Main) is offered for users who
                // need the wider-compatibility profile.
                ui.label(RichText::new(tr("Bit depth")).strong());
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut opts.h265_bit_depth, 10u8, tr("10-bit (Main10)"));
                    ui.selectable_value(&mut opts.h265_bit_depth, 8u8,  tr("8-bit (Main)"));
                });
                // Encoder choice — hardware vs software. macOS always
                // uses VideoToolbox, so only surface this elsewhere.
                if !cfg!(target_os = "macos") {
                    ui.add_space(4.0);
                    ui.label(RichText::new(tr("Encoder")).strong());
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut opts.h265_hardware, true,  tr("Hardware (NVENC)"));
                        ui.selectable_value(&mut opts.h265_hardware, false, tr("Software (libx265)"));
                    });
                    ui.label(RichText::new(if opts.h265_hardware {
                        "GPU HEVC — fast (NVIDIA); auto-falls back to libx265 if unavailable."
                    } else {
                        "CPU HEVC — slow at VR180 resolutions, max software quality."
                    }).weak().size(11.0));
                }
                }
            }
            ExportCodec::ProRes => {
                ui.label(RichText::new(tr("ProRes profile")).strong());
                egui::ComboBox::from_id_source(format!("{id_prefix}_prores_profile"))
                    .selected_text(opts.prores_profile.label())
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        use ProResProfile::*;
                        for p in [Proxy, Lt, Standard, Hq, P4444, P4444Xq] {
                            ui.selectable_value(&mut opts.prores_profile, p, p.label());
                        }
                    });
            }
        }

        if reframe.is_none() && lens_space == Some(crate::decoder::LensSpaceLens::Camera) {
            // A lens frame is not a half-equirect: it carries its own
            // metadata, and the VR180 choice below is for the other clips.
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(4.0);
            ui.label(RichText::new(tr("VR180 metadata")).strong());
            ui.label(RichText::new(tr(LENS_FRAME_METADATA_NOTE)).small().color(Color32::GRAY));
        } else if reframe.is_none() {
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(4.0);
        ui.label(RichText::new(tr("VR180 metadata")).strong());
        // Mutually exclusive — they overwrite the same atoms.
        // Show as radio so it's obvious only one is active.
        #[derive(PartialEq, Eq, Copy, Clone)]
        enum MetaTarget { None, Apmp, Youtube }
        let mut target = match (opts.inject_apmp, opts.inject_youtube) {
            (true, _)        => MetaTarget::Apmp,
            (false, true)    => MetaTarget::Youtube,
            _                => MetaTarget::None,
        };
        ui.radio_value(&mut target, MetaTarget::Apmp,
            tr("Apple Vision Pro (APMP — vexu + hfov)"));
        ui.radio_value(&mut target, MetaTarget::Youtube,
            tr("YouTube (Spherical V2 — st3d + sv3d)"));
        ui.radio_value(&mut target, MetaTarget::None,
            tr("None (no VR180 metadata)"));
        opts.inject_apmp    = matches!(target, MetaTarget::Apmp);
        opts.inject_youtube = matches!(target, MetaTarget::Youtube);
        ui.label(RichText::new(tr(
            "APMP and Spatial V2 conflict in the same atoms, so \
             only one can be active per file. Re-run the export \
             with the other target if you need a second copy."
        )).small().color(Color32::GRAY));
        // The equidistant disc of the Lens Space view is no half-equirect.
        if lens_space == Some(crate::decoder::LensSpaceLens::Equidistant) {
            ui.label(RichText::new(tr(LENS_SPACE_DISC_METADATA_NOTE)).small().color(Color32::GRAY));
        }
        }

        // The BeyondVR eye-scale hack is a half-equirect (VR180) fix-up of
        // the side-by-side frame — a reframed view and a single-eye frame
        // neither show nor apply it.
        if reframe.is_some() || single_eye {
            return;
        }
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(4.0);
        ui.checkbox(&mut opts.beyondvr_hack, tr("BeyondVR Hack"));
        if opts.beyondvr_hack {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(tr("Eye scale"));
                // step_by(1.0): whole percents only — a fractional value
                // (e.g. 99.7) would DISPLAY as "100 %" yet still engage
                // the scale stage and leave a thin black outline.
                ui.add(egui::Slider::new(&mut opts.beyondvr_scale_pct, 50.0..=100.0)
                    .suffix(" %").fixed_decimals(0).step_by(1.0));
                opts.beyondvr_scale_pct = opts.beyondvr_scale_pct.round();
            });
            ui.label(RichText::new(tr(
                "Scales each eye about its own center at the final \
                 output stage, padding with black. Half-equirect \
                 (VR180) output only."
            )).small().color(Color32::GRAY));
        }
    }

    /// Per-eye output size the export would use right now — the same
    /// derivation as the export config builder: source eye dims (EAC: the
    /// native cross square), 4096² for the 8K target, or the reframed
    /// per-eye size. Snapshots render the paused frame at exactly this.
    /// The side-by-side frame an export at the Native resolution produces
    /// for the loaded clip (per-eye native dims × 2 wide), for the labels —
    /// a 12K BRAW is 12288 × 6144, an Osmo 360 7680 × 3840. `None` without a
    /// clip or in the reframed mode (its own size picker).
    pub(super) fn native_sbs_dims(&self) -> Option<(u32, u32)> {
        let clip = self.clip.as_ref()?;
        if self.settings.reframe_aspect_if_active().is_some() { return None; }
        let kind = clip.source_kind;
        let (ew, eh) = if clip.fisheye_eye_w > 0 {
            (clip.fisheye_eye_w, clip.fisheye_eye_h)
        } else if kind.is_eac() {
            let cross_w = 2 * (clip.width.saturating_sub(1920) / 4) + 1920;
            (cross_w.max(512), cross_w.max(512))
        } else {
            (clip.width.max(512), clip.height.max(512))
        };
        // Two eyes side by side, or the one eye of a single-eye clip.
        let eyes = if clip.single_eye() { 1 } else { 2 };
        Some(match self.settings.fisheye_output_mode {
            // A lens frame is the recorded frame; the disc sits in a square.
            crate::decoder::FisheyeOutputMode::Fisheye if !self.settings.lens_frame_view(kind) => {
                let side = ew.min(eh);
                (side * eyes, side)
            }
            _ => (ew * eyes, eh),
        })
    }

    /// What the Lens Space view of the loaded clip writes — the camera's
    /// lens (a lens frame) or the equidistant disc, which is also what a
    /// clip without a lens record gets; `None` in every other view.
    pub(super) fn lens_space_out(&self) -> Option<crate::decoder::LensSpaceLens> {
        use crate::decoder::{FisheyeOutputMode, LensSpaceLens};
        let clip = self.clip.as_ref()?;
        (self.settings.fisheye_output_mode == FisheyeOutputMode::Fisheye).then(|| {
            if self.settings.lens_frame_view(clip.source_kind) { LensSpaceLens::Camera } else { LensSpaceLens::Equidistant }
        })
    }

    /// Width of one eye of the frame an export of the loaded raw take would
    /// write now: [`Self::export_eye_dims`], or the size picked for its
    /// enhanced frames where those were found (a canvas, or a lens frame; a
    /// reframed view has its own size).
    pub(super) fn raw_export_eye_w(&self) -> u32 {
        use crate::decoder::FisheyeOutputMode;
        let plain = self.export_eye_dims();
        let Some(clip) = self.clip.as_ref() else { return plain.0; };
        if self.enhanced_in_use().is_none() {
            return plain.0;
        }
        let own = (clip.fisheye_eye_w, clip.fisheye_eye_h);
        if self.settings.lens_frame_view(clip.source_kind) {
            return self.export_opts.lens_frame_out_dims(own.0, own.1, SizeList::Enhanced, self.settings.stabilize, clip.single_eye()).0;
        }
        match self.settings.fisheye_output_mode {
            FisheyeOutputMode::Reframe => plain.0,
            mode => {
                let (w, h) = self.export_opts.enhanced_eye_dims(own);
                if mode == FisheyeOutputMode::Fisheye { w.min(h) } else { w }
            }
        }
    }

    pub(super) fn export_eye_dims(&self) -> (u32, u32) {
        let Some(clip) = self.clip.as_ref() else { return (0, 0); };
        let kind = clip.source_kind;
        let (mut ew, mut eh) = if clip.fisheye_eye_w > 0 {
            (clip.fisheye_eye_w, clip.fisheye_eye_h)
        } else if kind.is_eac() {
            let cross_w = 2 * (clip.width.saturating_sub(1920) / 4) + 1920;
            (cross_w.max(512), cross_w.max(512))
        } else {
            (clip.width.max(512), clip.height.max(512))
        };
        // Same rule as `build_export_cfg`.
        let sharp_eye = clip.sharp_eye(self.settings.insv_view);
        // A lens frame: the frame the export writes, its margin included.
        if self.settings.lens_frame_view(kind) {
            return self.export_opts.lens_frame_out_dims(ew, eh, SizeList::of(sharp_eye.is_some(), false), self.settings.stabilize, clip.single_eye());
        }
        match (self.export_opts.resolution_for(SizeList::of(sharp_eye.is_some(), false)), sharp_eye) {
            (ExportResolution::Sharp, Some(side)) => { ew = side; eh = side; }
            (ExportResolution::Native, _) => {}
            _ => { ew = 4096; eh = 4096; }
        }
        match self.settings.fisheye_output_mode {
            crate::decoder::FisheyeOutputMode::Reframe =>
                self.export_opts.reframe_size.eye_dims(self.settings.reframe_aspect),
            crate::decoder::FisheyeOutputMode::Fisheye => { let side = ew.min(eh); (side, side) }
            crate::decoder::FisheyeOutputMode::HalfEquirect => (ew, eh),
        }
    }
}
