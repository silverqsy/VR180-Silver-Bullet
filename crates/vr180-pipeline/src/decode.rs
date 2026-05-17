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

/// One decoded frame from each of the two HEVC streams in a .360 file,
/// converted to packed RGB8 host memory. The dimensions are probed from
/// the streams themselves — no hardcoded `5952×1920`.
#[derive(Debug)]
pub struct StreamPair {
    /// Packed RGB8 bytes, `stream_h × stream_w × 3`, from the first
    /// HEVC stream (`s0` in the Python code).
    pub s0: Vec<u8>,
    /// Same shape, from the second HEVC stream (`s4`).
    pub s4: Vec<u8>,
    /// Dimensions probed from the streams. Caller asserts the two
    /// streams agree on size — they always do on real GoPro footage.
    pub dims: vr180_core::eac::Dims,
}

/// Extract the first decoded video frame from each of the two HEVC
/// streams. Single-frame, software decode, no hwaccel — Phase 0.4
/// CPU baseline. Replaces the Python pipeline's PyAV-or-subprocess
/// dual-path with one in-process libav call.
pub fn extract_first_stream_pair(path: &Path) -> Result<StreamPair> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    // Find both HEVC video streams. GoPro .360 always has exactly two
    // (s0 + s4 in Python notation).
    let video_indices: Vec<usize> = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .map(|s| s.index())
        .collect();
    if video_indices.len() < 2 {
        return Err(Error::Ffmpeg(format!(
            "expected 2 video streams in .360 file, found {}",
            video_indices.len()
        )));
    }

    // Set up a decoder per stream. We hold these as Vec<_> so the
    // borrow rules play nicely with the packet-read loop below.
    let mut decoders: Vec<ffmpeg_next::codec::decoder::Video> = video_indices
        .iter()
        .map(|&idx| {
            let stream = ictx.stream(idx).unwrap();
            let codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;

    // Both streams must agree on dimensions on real GoPro footage.
    let (w0, h0) = (decoders[0].width(), decoders[0].height());
    let (w1, h1) = (decoders[1].width(), decoders[1].height());
    if (w0, h0) != (w1, h1) {
        return Err(Error::Ffmpeg(format!(
            "video streams disagree on dimensions: {w0}x{h0} vs {w1}x{h1}"
        )));
    }
    let dims = vr180_core::eac::Dims::new(w0, h0);
    if !dims.is_valid() {
        return Err(Error::Ffmpeg(format!(
            "stream width {w0} not a valid EAC layout (need (w-1920) % 4 == 0)"
        )));
    }

    // One scaler per stream: YUV (whatever pixel format the decoder
    // produces, typically yuv420p / p010le) → RGB24.
    let mut scalers: Vec<Option<ffmpeg_next::software::scaling::Context>> = vec![None, None];

    // Receive loop: keep reading packets until we have one frame from
    // each stream. ffmpeg's HEVC decoder may need a few packets of
    // buffering before it can emit a frame, hence the loop.
    let mut frames: [Option<Vec<u8>>; 2] = [None, None];
    'outer: for (stream, packet) in ictx.packets() {
        let pos = match video_indices.iter().position(|&i| i == stream.index()) {
            Some(p) if frames[p].is_none() => p,
            _ => continue,
        };
        let dec = &mut decoders[pos];
        if dec.send_packet(&packet).is_err() { continue; }
        let mut decoded = ffmpeg_next::frame::Video::empty();
        while dec.receive_frame(&mut decoded).is_ok() {
            // Lazily build the scaler now that we know the actual pixel format.
            let scaler = match &mut scalers[pos] {
                Some(s) => s,
                None => {
                    let s = ffmpeg_next::software::scaling::Context::get(
                        decoded.format(),
                        decoded.width(),
                        decoded.height(),
                        ffmpeg_next::format::Pixel::RGB24,
                        decoded.width(),
                        decoded.height(),
                        ffmpeg_next::software::scaling::Flags::BILINEAR,
                    ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?;
                    scalers[pos] = Some(s);
                    scalers[pos].as_mut().unwrap()
                }
            };
            let mut rgb = ffmpeg_next::frame::Video::empty();
            scaler.run(&decoded, &mut rgb)
                .map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
            // swscale gives us a frame whose row stride may exceed
            // width*3 (padded to 16/32 for SIMD). Repack to tightly
            // packed RGB8 for downstream simplicity.
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            let src = rgb.data(0);
            let src_stride = rgb.stride(0);
            let mut out = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                let row = &src[y * src_stride..y * src_stride + w * 3];
                out.extend_from_slice(row);
            }
            frames[pos] = Some(out);
            if frames.iter().all(|f| f.is_some()) {
                break 'outer;
            }
            break; // got a frame for this stream; move to next packet
        }
    }

    let (Some(s0), Some(s4)) = (frames[0].take(), frames[1].take()) else {
        return Err(Error::Ffmpeg(
            "EOF reached before both streams produced a decoded frame".into()
        ));
    };
    Ok(StreamPair { s0, s4, dims })
}
