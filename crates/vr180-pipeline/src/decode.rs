//! ffmpeg-next decode wrapper.
//!
//! Phase 0.2 deliverable: `extract_gpmf_stream` — pull all packets
//! from the .360 file's GoPro metadata data-stream into a single
//! byte buffer. This replaces the Python pipeline's
//! `ffmpeg -i <path> -map 0:3 -c copy -f rawvideo -` subprocess
//! call: in-process libav, no fork, no stdout pipe.
//!
//! Phase 0.4 — software video decode of the two HEVC streams (s0 + s4).
//! Phase 0.6 — hardware decode (VideoToolbox / NVDEC) with
//! IOSurface ↔ Metal or CUDA ↔ Vulkan interop into wgpu.

use crate::{Error, Result};
use std::path::Path;

/// Initialize libav once per process (idempotent — `ffmpeg::init()`
/// guards itself internally). Call before the first ffmpeg-next op.
pub fn init() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `init()` returns Result but the only failure mode is "already
        // initialized," which we want to ignore.
        let _ = ffmpeg_next::init();
        // Quiet libav's per-frame stderr chatter unless the user opts in
        // via RUST_LOG / FFREPORT.
        ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    });
}

/// FourCC `'gpmd'` as a little-endian `u32` — the codec tag GoPro
/// uses for the GPMF telemetry data stream in `.360` / `.mp4` files.
/// Stored as bytes `g`,`p`,`m`,`d` on disk; the libav `AVCodecParameters::codec_tag`
/// field is a u32 in host (LE) byte order on Apple Silicon / x86.
const GPMD_TAG: u32 = u32::from_le_bytes(*b"gpmd");

/// Pull the entire GPMF (GoPro Metadata Format) data stream from a
/// .360 file into memory. Equivalent to the Python pipeline's
/// `ffmpeg -i <path> -map 0:3 -c copy -f rawvideo -` subprocess.
///
/// Stream selection: we look specifically for the data stream whose
/// codec tag is `gpmd` (GoPro Metadata). A `.360` file typically
/// has two data streams — index 2 is `tmcd` (timecode, ~4 bytes total)
/// and index 3 is `gpmd` (GPMF telemetry, hundreds of KB). Picking
/// "best Data" returns the timecode, which is wrong.
pub fn extract_gpmf_stream(path: &Path) -> Result<Vec<u8>> {
    init();

    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = find_gpmd_stream(&ictx)
        .ok_or_else(|| Error::Ffmpeg(
            format!("no `gpmd` data stream in {path:?}")
        ))?;

    // Pre-size: GPMF telemetry is roughly 5-15 KB/s of footage. For a
    // 30-second clip that's ~300 KB; for an hour ~50 MB. Either way
    // it's fine to materialize in memory.
    let mut out = Vec::with_capacity(512 * 1024);
    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_idx {
            if let Some(data) = packet.data() {
                out.extend_from_slice(data);
            }
        }
    }

    if out.is_empty() {
        return Err(Error::Ffmpeg(format!(
            "GPMF data stream {} of {path:?} was empty", stream_idx
        )));
    }
    Ok(out)
}

/// Find the stream index whose codec tag is `gpmd`. Returns the first
/// match in stream-index order. Returns `None` if no such stream exists.
fn find_gpmd_stream(ictx: &ffmpeg_next::format::context::Input) -> Option<usize> {
    for stream in ictx.streams() {
        // SAFETY: `AVCodecParameters` is exposed as a raw pointer; we
        // only read POD fields (`codec_tag`).
        let tag = unsafe { (*stream.parameters().as_ptr()).codec_tag };
        if tag == GPMD_TAG {
            return Some(stream.index());
        }
    }
    None
}

/// Inspect basic video info for a .360 (or any container): fps,
/// duration, and the dimensions of the largest video stream. Used by
/// `probe-gyro` to convert sample counts → seconds in the printout.
#[derive(Debug, Clone, Copy)]
pub struct VideoProbe {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub duration_sec: f64,
}

pub fn probe_video(path: &Path) -> Result<VideoProbe> {
    init();
    let ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
    let stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?;
    let params = stream.parameters();
    let avg = stream.avg_frame_rate();
    let fps = if avg.denominator() != 0 {
        avg.numerator() as f32 / avg.denominator() as f32
    } else {
        0.0
    };
    let tb = stream.time_base();
    let duration_sec = if tb.denominator() != 0 && stream.duration() != ffmpeg_next::ffi::AV_NOPTS_VALUE {
        stream.duration() as f64 * tb.numerator() as f64 / tb.denominator() as f64
    } else {
        ictx.duration() as f64 / ffmpeg_next::ffi::AV_TIME_BASE as f64
    };
    // SAFETY: AVCodecParameters is exposed by ffmpeg-next via as_ptr;
    // we read width/height which are POD fields.
    let raw = unsafe { &*params.as_ptr() };
    Ok(VideoProbe {
        width: raw.width as u32,
        height: raw.height as u32,
        fps,
        duration_sec,
    })
}
