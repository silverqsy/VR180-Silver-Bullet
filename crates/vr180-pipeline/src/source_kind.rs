//! Source-format detection.
//!
//! Given a file path, decide whether to route the rest of the
//! pipeline through:
//! - the GoPro EAC `.360` path (`extract_first_stream_pair`,
//!   `StreamPairIter`, EAC→equirect shader)
//! - the DJI OSV dual-stream fisheye path
//!   (`DualStreamFisheyeIter`, fisheye→hequirect shader)
//! - the SBS fisheye path (`SbsFisheyeIter`, fisheye→hequirect shader)
//! - the Blackmagic BRAW path (`BrawFisheyeIter` + braw_helper)
//!
//! Detection is extension-first (cheap, matches the Python app's
//! convention at `vr180_gui.py:8317-8324`) with a fall-back probe for
//! ambiguous `.mp4` / `.mov` (where it could be either GoPro EAC or
//! SBS fisheye).

use crate::Result;
use std::path::Path;

/// What kind of camera produced this file. Routes the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// GoPro Max `.360` (and `.mp4` with a `gpmd` stream). Two HEVC
    /// streams (s0 + s4) holding the EAC cross tiles.
    GoProEac,
    /// DJI Osmo `.osv`. MP4 container with two HEVC video streams,
    /// each one a full fisheye eye.
    DjiOsv,
    /// Insta360 `.insv` (X-series, e.g. X6). Same dual-stream fisheye
    /// MP4 layout as the OSV, plus a proprietary trailer carrying the raw
    /// ~1 kHz IMU and per-frame exposure stamps (no metadata track).
    Insta360Insv,
    /// Single-stream side-by-side fisheye `.mp4` / `.mov` (Insta360,
    /// Vuze XR, QooCam, Canon RF dual-fisheye, generic dual-camera
    /// rigs muxed to one stream).
    SbsFisheye,
    /// Blackmagic RAW `.braw` (Pyxis 12K, URSA Cine Immersive, etc.).
    /// Multi-track stereo files auto-compose to SBS at the helper.
    BlackmagicRaw,
    /// Insta360 X6 `.dng` STILL: one 15520×7760 Bayer frame holding both
    /// fisheye circles side by side (left half = left eye post-mod), with
    /// the factory lens strings in the EXIF and an Insta360 trailer.
    /// Not a video: routes through `still_export` / the still preview.
    Insta360Dng,
    /// Kandao QooCam 3 Ultra `.mp4`: ONE HEVC stream holding both fisheye
    /// eyes stacked top/bottom (3840×7680 — top = lens 0 = LEFT eye
    /// post-mod), HLG BT.2020 in its HDR mode, plus the proprietary `kfix` /
    /// `kvar` boxes after `moov` (factory lens record, 1 kHz IMU, per-frame
    /// PTS + exposure). Decodes through the single-stream iterators with the
    /// stacked split.
    KandaoQoocam,
    /// Kandao QooCam 3 Ultra `.dng` STILL: one 13888×6944 RGGB frame holding
    /// both fisheye circles side by side (left half = lens 0 = LEFT eye
    /// post-mod), the lens record and shading grids in the EXIF
    /// `UserComment`. Same still pipeline as the X6's.
    KandaoDng,
    /// Unrecognised — the GUI should surface an "unknown format"
    /// message and let the user pick a preset manually.
    Unknown,
}

impl SourceKind {
    /// True if this source uses the GoPro EAC pipeline.
    pub fn is_eac(self) -> bool {
        matches!(self, Self::GoProEac)
    }

    /// True if this source uses the fisheye (KB) pipeline.
    pub fn is_fisheye(self) -> bool {
        matches!(self, Self::DjiOsv | Self::Insta360Insv | Self::SbsFisheye | Self::BlackmagicRaw | Self::KandaoQoocam)
    }

    /// ONE video stream holding both eyes stacked top/bottom (Kandao QooCam
    /// 3 Ultra). Routes through the single-stream iterators like
    /// `SbsFisheye`, split vertically: the eye is `frame_w × frame_h / 2`.
    pub fn is_stacked(self) -> bool {
        matches!(self, Self::KandaoQoocam)
    }

    /// A single video stream holding both eyes (side by side or stacked):
    /// the generic one-stream decode path, with [`Self::eye_layout`]'s split.
    pub fn is_single_stream_pair(self) -> bool {
        matches!(self, Self::SbsFisheye | Self::KandaoQoocam)
    }

    /// How a single-stream source packs its two eyes.
    pub fn eye_layout(self) -> EyeLayout {
        if self.is_stacked() { EyeLayout::TopBottom } else { EyeLayout::SideBySide }
    }

    /// Two equal-size video streams in one MP4, one full fisheye eye each
    /// (DJI OSV, Insta360 INSV). Routes through `DualStreamFisheyeIter`
    /// and the zero-copy dual-stream decoders.
    pub fn is_dual_stream(self) -> bool {
        matches!(self, Self::DjiOsv | Self::Insta360Insv)
    }

    /// `swap_eyes` flag to hand the dual-stream iterators for a given user
    /// toggle. Both VR180-modded cameras arrive with stream 0 = RIGHT eye —
    /// DJI's Lens A, and on the Insta360 X6 the back lens, which the mod
    /// turns to sit on the right facing the screen-lens direction (measured
    /// on post-mod footage; see `insv_imu`) — so the iterator swaps by
    /// default (`!user`). Every downstream consumer (calib, per-eye rotation,
    /// RS rows) keys off the raw toggle with `lens_a` = stream 0, one
    /// convention for both cameras.
    pub fn dual_stream_iter_swap(self, user_swap: bool) -> bool {
        match self {
            Self::DjiOsv | Self::Insta360Insv => !user_swap,
            _ => user_swap,
        }
    }

    /// The source carries an IMU we turn into per-frame orientations —
    /// the DJI-style stabilization controls (smooth / max-corr / IMU
    /// phase) apply. DJI reads camera quats; Insta360 fuses its raw gyro.
    pub fn has_frame_imu(self) -> bool {
        matches!(self, Self::DjiOsv | Self::Insta360Insv | Self::KandaoQoocam | Self::BlackmagicRaw)
    }

    /// Blackmagic RAW: decoded by the SDK helper, both eyes side by side
    /// (a VR180 lens on one sensor, or two tracks composed by the helper);
    /// the lens is the user's manual calibration.
    pub fn is_braw(self) -> bool {
        matches!(self, Self::BlackmagicRaw)
    }

    /// A single still photo (no timeline, no decoder, no stabilization):
    /// the Insta360 X6 or Kandao QooCam `.dng`.
    pub fn is_still(self) -> bool {
        matches!(self, Self::Insta360Dng | Self::KandaoDng)
    }

    /// The Kandao QooCam 3 Ultra, video or still.
    pub fn is_kandao(self) -> bool {
        matches!(self, Self::KandaoQoocam | Self::KandaoDng)
    }

    /// The Insta360 X6, video or still.
    pub fn is_insta360(self) -> bool {
        matches!(self, Self::Insta360Insv | Self::Insta360Dng)
    }

    /// The two kinds (video + still) one camera writes from the same pair of
    /// lenses — their stereo offset is one shared value in the GUI's
    /// per-kind templates. `None` for kinds without a sibling.
    pub fn stereo_sibling(self) -> Option<Self> {
        match self {
            Self::Insta360Insv => Some(Self::Insta360Dng),
            Self::Insta360Dng => Some(Self::Insta360Insv),
            Self::KandaoQoocam => Some(Self::KandaoDng),
            Self::KandaoDng => Some(Self::KandaoQoocam),
            _ => None,
        }
    }

    /// True if we can preview AND export this source to VR180 SBS —
    /// every fisheye source plus GoPro EAC, and the X6 still. Drives the
    /// export button, the clip list, and batch eligibility.
    pub fn is_exportable(self) -> bool {
        self.is_fisheye() || self.is_eac() || self.is_still()
    }

    /// Human-readable name (e.g. for log messages / status bars).
    pub fn display(self) -> &'static str {
        match self {
            Self::GoProEac      => "GoPro EAC (.360)",
            Self::DjiOsv        => "DJI Osmo OSV (dual-stream fisheye)",
            Self::Insta360Insv  => "Insta360 INSV (dual-stream fisheye)",
            Self::SbsFisheye    => "Side-by-side fisheye",
            Self::BlackmagicRaw => "Blackmagic RAW",
            Self::Insta360Dng   => "Insta360 X6 DNG still",
            Self::KandaoQoocam  => "Kandao QooCam 3 Ultra (stacked fisheye)",
            Self::KandaoDng     => "Kandao QooCam 3 Ultra DNG still",
            Self::Unknown       => "Unknown",
        }
    }
}

/// How one video stream packs its two fisheye eyes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EyeLayout {
    /// Left eye in the left half, right eye in the right half (`eye_w =
    /// frame_w / 2`).
    #[default]
    SideBySide,
    /// Left eye in the top half, right eye in the bottom half (`eye_h =
    /// frame_h / 2`) — the Kandao QooCam 3 Ultra's 3840×7680.
    TopBottom,
}

impl EyeLayout {
    /// Per-eye dims of a whole frame.
    pub fn eye_dims(self, frame_w: u32, frame_h: u32) -> (u32, u32) {
        match self {
            Self::SideBySide => (frame_w / 2, frame_h),
            Self::TopBottom => (frame_w, frame_h / 2),
        }
    }
    /// Origin `(x0, y0)` of eye `idx` (0 = left, 1 = right) inside the
    /// whole frame.
    pub fn eye_origin(self, idx: u32, eye_w: u32, eye_h: u32) -> (u32, u32) {
        match self {
            Self::SideBySide => (idx * eye_w, 0),
            Self::TopBottom => (0, idx * eye_h),
        }
    }
    /// The whole-frame axis that must be even for the split.
    pub fn split_axis_ok(self, frame_w: u32, frame_h: u32) -> bool {
        match self {
            Self::SideBySide => frame_w % 2 == 0,
            Self::TopBottom => frame_h % 2 == 0,
        }
    }
}

/// Detect the source kind for `path`. Cheap (no full decode):
/// extension first, container probe only when ambiguous.
///
/// Returns `Unknown` instead of erroring on unrecognised files —
/// callers should pair this with the camera preset library and let
/// the user override.
pub fn detect(path: &Path) -> Result<SourceKind> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext.as_deref() {
        Some("360")  => return Ok(SourceKind::GoProEac),
        Some("osv")  => return Ok(SourceKind::DjiOsv),
        Some("insv") => return Ok(SourceKind::Insta360Insv),
        Some("braw") => return Ok(SourceKind::BlackmagicRaw),
        Some("dng") => {
            // The Insta360 X6 still (TIFF header + the Insta360 footer
            // magic) or the Kandao QooCam still (IFD0 `Make`) — any other
            // DNG stays unsupported.
            return Ok(if vr180_fisheye::dng::is_insta360_dng(path) {
                SourceKind::Insta360Dng
            } else if vr180_fisheye::dng::is_kandao_dng(path) {
                SourceKind::KandaoDng
            } else {
                SourceKind::Unknown
            });
        }
        Some("mp4") | Some("mov") | Some("m4v") | Some("mkv") => {
            // Ambiguous: could be GoPro `.mp4` (with `gpmd` stream), DJI
            // OSV-as-`.mp4` (two video streams), a renamed Insta360 `.insv`
            // (two video streams + trailer), a Kandao QooCam (one stacked
            // stream + `kfix` box), or plain SBS fisheye. The Kandao box
            // walk is a handful of header reads, so it goes first.
            if vr180_fisheye::kandao::has_kandao_boxes(path) {
                // The same boxes ride the camera's single-lens modes (a `Q180_`
                // clip: ONE 3840x3840 circle at 60 fps), which hold one eye and
                // are no stereo pair; only the stacked 1:2 frame is a source.
                return Ok(if kandao_frame_is_stacked(path) {
                    SourceKind::KandaoQoocam
                } else {
                    tracing::warn!(
                        "{}: Kandao boxes but the frame is not a stacked 1:2 pair (a single-lens recording) - unsupported",
                        path.display());
                    SourceKind::Unknown
                });
            }
            let kind = probe_container_layout(path);
            if kind == SourceKind::DjiOsv && vr180_fisheye::insta360::has_insv_trailer(path) {
                return Ok(SourceKind::Insta360Insv);
            }
            return Ok(kind);
        }
        _ => {}
    }

    Ok(SourceKind::Unknown)
}

/// The largest video stream is `w x 2w`: the QooCam's stacked dual-fisheye
/// frame (3840x7680, 3472x6944, 2880x5760 ...). The camera's single-lens
/// recordings (`Q180_...`, 3840x3840) carry the same `kfix` boxes but one eye.
/// A file ffmpeg cannot open is left to the Kandao kind (nothing else would
/// take it either).
fn kandao_frame_is_stacked(path: &Path) -> bool {
    crate::decode::init();
    let Ok(ictx) = ffmpeg_next::format::input(path) else { return true };
    ictx.streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        // SAFETY: `AVCodecParameters` is exposed as a raw pointer; only the
        // POD width/height fields are read.
        .map(|s| unsafe { let p = &*s.parameters().as_ptr(); (p.width, p.height) })
        .max_by_key(|(w, h)| (*w as i64) * (*h as i64))
        .map_or(true, |(w, h)| w > 0 && h == 2 * w)
}

/// Open the container with ffmpeg-next and decide between GoPro /
/// dual-stream / SBS based on stream counts. Mirrors the Python
/// `_detect_and_setup_dji_file` path at `vr180_gui.py:6122`.
///
/// Heuristic:
/// 1. If a `gpmd` data stream is present → GoPro EAC.
/// 2. Else if 2+ video streams of identical dimensions exist → dual-stream.
/// 3. Else → assume SBS fisheye (single-stream split horizontally).
///
/// Errors during probe are not fatal: we return `Unknown` and let the
/// caller surface it as a UI-level "could not identify file" message.
fn probe_container_layout(path: &Path) -> SourceKind {
    crate::decode::init();
    let ictx = match ffmpeg_next::format::input(path) {
        Ok(c) => c,
        Err(_) => return SourceKind::Unknown,
    };

    // Check for gpmd data stream → GoPro family.
    for stream in ictx.streams() {
        // SAFETY: `AVCodecParameters` is exposed as a raw pointer; we
        // only read the POD `codec_tag` field.
        let tag = unsafe { (*stream.parameters().as_ptr()).codec_tag };
        const GPMD_TAG: u32 = u32::from_le_bytes(*b"gpmd");
        if tag == GPMD_TAG {
            return SourceKind::GoProEac;
        }
    }

    // Count video streams.
    let video_streams: Vec<_> = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .collect();

    if video_streams.len() >= 2 {
        // Two video streams of equal dimensions → dual fisheye.
        let p0 = video_streams[0].parameters();
        let p1 = video_streams[1].parameters();
        let (w0, h0) = unsafe {
            let pp = &*p0.as_ptr();
            (pp.width, pp.height)
        };
        let (w1, h1) = unsafe {
            let pp = &*p1.as_ptr();
            (pp.width, pp.height)
        };
        if (w0, h0) == (w1, h1) {
            return SourceKind::DjiOsv;
        }
    }

    // Default: assume SBS fisheye.
    SourceKind::SbsFisheye
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_route_correctly() {
        let cases = [
            ("clip.360", SourceKind::GoProEac),
            ("video.OSV", SourceKind::DjiOsv),
            ("VID_20260815_193658_00_030.insv", SourceKind::Insta360Insv),
            ("pyxis.BRAW", SourceKind::BlackmagicRaw),
            ("IMG_20261007_133105_00_050.dng", SourceKind::Unknown), // no trailer on a bare path
            ("Q360_20250512_160943_000001.DNG", SourceKind::Unknown), // not a file either
            ("readme.txt", SourceKind::Unknown),
        ];
        for (name, expected) in cases {
            let p = Path::new(name);
            assert_eq!(detect(p).unwrap(), expected, "{name}");
        }
    }

    #[test]
    fn eye_layout_geometry() {
        assert_eq!(EyeLayout::SideBySide.eye_dims(7680, 3840), (3840, 3840));
        assert_eq!(EyeLayout::TopBottom.eye_dims(3840, 7680), (3840, 3840));
        assert_eq!(EyeLayout::SideBySide.eye_origin(1, 3840, 3840), (3840, 0));
        assert_eq!(EyeLayout::TopBottom.eye_origin(1, 3840, 3840), (0, 3840));
        assert_eq!(EyeLayout::TopBottom.eye_origin(0, 3840, 3840), (0, 0));
        assert!(EyeLayout::TopBottom.split_axis_ok(3840, 7680));
        assert!(!EyeLayout::TopBottom.split_axis_ok(3840, 7681));
        assert_eq!(SourceKind::KandaoQoocam.eye_layout(), EyeLayout::TopBottom);
        assert_eq!(SourceKind::SbsFisheye.eye_layout(), EyeLayout::SideBySide);
        assert!(SourceKind::KandaoQoocam.is_fisheye() && SourceKind::KandaoQoocam.has_frame_imu());
        assert!(!SourceKind::KandaoQoocam.is_dual_stream());
        assert!(SourceKind::KandaoDng.is_still() && SourceKind::KandaoDng.is_exportable());
        assert_eq!(SourceKind::KandaoQoocam.stereo_sibling(), Some(SourceKind::KandaoDng));
        assert_eq!(SourceKind::GoProEac.stereo_sibling(), None);
        // The stacked iterator emits left = top, so the user swap passes through.
        assert!(!SourceKind::KandaoQoocam.dual_stream_iter_swap(false));
        assert!(SourceKind::KandaoQoocam.dual_stream_iter_swap(true));
    }
}
