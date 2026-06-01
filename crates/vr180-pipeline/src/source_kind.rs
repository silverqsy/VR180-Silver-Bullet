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
    /// Single-stream side-by-side fisheye `.mp4` / `.mov` (Insta360,
    /// Vuze XR, QooCam, Canon RF dual-fisheye, generic dual-camera
    /// rigs muxed to one stream).
    SbsFisheye,
    /// Blackmagic RAW `.braw` (Pyxis 12K, URSA Cine Immersive, etc.).
    /// Multi-track stereo files auto-compose to SBS at the helper.
    BlackmagicRaw,
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
        matches!(self, Self::DjiOsv | Self::SbsFisheye | Self::BlackmagicRaw)
    }

    /// Human-readable name (e.g. for log messages / status bars).
    pub fn display(self) -> &'static str {
        match self {
            Self::GoProEac      => "GoPro EAC (.360)",
            Self::DjiOsv        => "DJI Osmo OSV (dual-stream fisheye)",
            Self::SbsFisheye    => "Side-by-side fisheye",
            Self::BlackmagicRaw => "Blackmagic RAW",
            Self::Unknown       => "Unknown",
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
        Some("braw") => return Ok(SourceKind::BlackmagicRaw),
        Some("mp4") | Some("mov") | Some("m4v") | Some("mkv") => {
            // Ambiguous: could be GoPro `.mp4` (with `gpmd` stream), DJI
            // OSV-as-`.mp4` (two video streams), or plain SBS fisheye.
            return Ok(probe_container_layout(path));
        }
        _ => {}
    }

    Ok(SourceKind::Unknown)
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
            ("pyxis.BRAW", SourceKind::BlackmagicRaw),
            ("readme.txt", SourceKind::Unknown),
        ];
        for (name, expected) in cases {
            let p = Path::new(name);
            assert_eq!(detect(p).unwrap(), expected, "{name}");
        }
    }
}
