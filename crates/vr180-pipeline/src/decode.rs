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
    /// Which decode path actually produced this frame (after `Auto`
    /// resolution + any silent fallback).
    pub decode_path: DecodePath,
}

/// User-facing hwaccel selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwDecode {
    /// Pick the platform default: VideoToolbox on macOS, software
    /// elsewhere. Silently falls back to software if hwaccel setup fails.
    Auto,
    /// Force software decode (CPU only).
    Software,
    /// Force VideoToolbox (macOS only). Returns an error if hwaccel
    /// setup fails — use [`HwDecode::Auto`] for graceful fallback.
    VideoToolbox,
}

/// Which decode path actually ran (after fallback resolution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    Software,
    VideoToolbox,
}

impl std::fmt::Display for DecodePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodePath::Software => f.write_str("software"),
            DecodePath::VideoToolbox => f.write_str("VideoToolbox"),
        }
    }
}

/// Extract the first decoded video frame from each of the two HEVC
/// streams. Defaults to platform-best hwaccel via [`HwDecode::Auto`].
pub fn extract_first_stream_pair(path: &Path) -> Result<StreamPair> {
    extract_first_stream_pair_with(path, HwDecode::Auto)
}

/// Same as [`extract_first_stream_pair`] but with explicit hwaccel
/// control. On macOS, `Auto` and `VideoToolbox` both wire VT decode
/// via `av_hwdevice_ctx_create` + a custom `get_format` callback that
/// returns `AV_PIX_FMT_VIDEOTOOLBOX`. Decoded hwframes are downloaded
/// to host memory via `av_hwframe_transfer_data` (typically NV12 for
/// 8-bit, P010 for 10-bit), then swscale converts to RGB24.
///
/// **Phase 0.6 scope:** decode path is hardware-accelerated end-to-end,
/// but frames still go through host memory before reaching the GPU
/// compositor. The zero-copy IOSurface↔Metal interop fast path
/// (skipping `av_hwframe_transfer_data` + the wgpu upload) is deferred
/// to Phase 0.6.5 — needs ~700 lines of objc/IOSurface bridging that's
/// its own focused work.
pub fn extract_first_stream_pair_with(path: &Path, hw: HwDecode) -> Result<StreamPair> {
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

    // Set up a decoder per stream + optionally attach the VT hwaccel.
    // We hold decoders as Vec<_> so borrow rules play nicely with the
    // packet-read loop below.
    let mut decoders: Vec<ffmpeg_next::codec::decoder::Video> = Vec::with_capacity(2);
    let mut hw_active_per_stream = [false, false];
    for (i, &idx) in video_indices.iter().take(2).enumerate() {
        let stream = ictx.stream(idx).unwrap();
        let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

        // Attempt to attach VT hwaccel BEFORE turning the Context into a
        // decoder — once the decoder is built we'd need its raw codec
        // context pointer to install `hw_device_ctx` / `get_format`.
        #[cfg(target_os = "macos")]
        {
            let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
            if want_vt {
                if try_enable_videotoolbox_decode(&mut codec_ctx) {
                    hw_active_per_stream[i] = true;
                } else if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg(format!(
                        "stream {idx}: --hw-accel videotoolbox requested but \
                         av_hwdevice_ctx_create failed (no VT support on this system)"
                    )));
                }
                // else: Auto + setup failed → silently fall back to sw
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (i, hw); // mark used so cfg gating doesn't warn
            if matches!(hw, HwDecode::VideoToolbox) {
                return Err(Error::Ffmpeg(
                    "--hw-accel videotoolbox is macOS-only".into()
                ));
            }
        }

        let decoder = codec_ctx.decoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;
        decoders.push(decoder);
    }

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

    // Lazy scaler per stream: built once we see the actual sw-frame
    // pixel format (which differs depending on hwaccel: NV12/P010 after
    // VT transfer, yuv420p/p010le from sw decode).
    let mut scalers: Vec<Option<ffmpeg_next::software::scaling::Context>> = vec![None, None];

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
            // Hardware frame? Download to host memory via av_hwframe_transfer_data.
            // sw_frame ends up NV12 (8-bit) or P010 (10-bit) — swscale handles both.
            let mut sw_storage = ffmpeg_next::frame::Video::empty();
            let frame_ref: &ffmpeg_next::frame::Video = if hw_active_per_stream[pos]
                && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
            {
                download_hw_frame(&decoded, &mut sw_storage)?;
                &sw_storage
            } else {
                &decoded
            };

            let scaler = match &mut scalers[pos] {
                Some(s) => s,
                None => {
                    let s = ffmpeg_next::software::scaling::Context::get(
                        frame_ref.format(),
                        frame_ref.width(),
                        frame_ref.height(),
                        ffmpeg_next::format::Pixel::RGB24,
                        frame_ref.width(),
                        frame_ref.height(),
                        ffmpeg_next::software::scaling::Flags::BILINEAR,
                    ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?;
                    scalers[pos] = Some(s);
                    scalers[pos].as_mut().unwrap()
                }
            };
            let mut rgb = ffmpeg_next::frame::Video::empty();
            scaler.run(frame_ref, &mut rgb)
                .map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
            // swscale row stride may exceed width*3 (SIMD padding).
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
            break;
        }
    }

    let (Some(s0), Some(s4)) = (frames[0].take(), frames[1].take()) else {
        return Err(Error::Ffmpeg(
            "EOF reached before both streams produced a decoded frame".into()
        ));
    };
    let decode_path = if hw_active_per_stream.iter().all(|&a| a) {
        DecodePath::VideoToolbox
    } else {
        DecodePath::Software
    };
    Ok(StreamPair { s0, s4, dims, decode_path })
}

/// Decode the first video frame using VideoToolbox hwaccel and return
/// it **without** running `av_hwframe_transfer_data`. The frame's
/// `format == AV_PIX_FMT_VIDEOTOOLBOX` and `data[3]` is the live
/// CVPixelBuffer the IOSurface lives inside.
///
/// Used by Phase 0.6.5's IOSurface demo (`probe-iosurface`) and will be
/// the entry point for the Phase 0.6.6 zero-copy compute path.
#[cfg(target_os = "macos")]
pub fn decode_first_vt_frame(path: &Path) -> Result<ffmpeg_next::frame::Video> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .next()
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?
        .index();

    let stream = ictx.stream(stream_idx).unwrap();
    let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

    if !try_enable_videotoolbox_decode(&mut codec_ctx) {
        return Err(Error::Ffmpeg(
            "VideoToolbox hwaccel setup failed — IOSurface path needs VT".into()
        ));
    }
    let mut decoder = codec_ctx.decoder().video()
        .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx { continue; }
        if decoder.send_packet(&packet).is_err() { continue; }
        let mut decoded = ffmpeg_next::frame::Video::empty();
        if decoder.receive_frame(&mut decoded).is_ok() {
            if decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX {
                return Ok(decoded);
            } else {
                return Err(Error::Ffmpeg(format!(
                    "expected VIDEOTOOLBOX pixel format, decoder produced {:?}",
                    decoded.format()
                )));
            }
        }
    }
    Err(Error::Ffmpeg("EOF before first frame decoded".into()))
}

// ─── VideoToolbox hwaccel plumbing (macOS only) ────────────────────────

/// Wire VideoToolbox hardware decode onto a codec context.
///
/// Returns `true` if the hwaccel was successfully attached. Mirrors
/// `SLRStudioNeo::try_enable_videotoolbox_decode`. The codec context
/// takes ownership of the `AVBufferRef` and will free it; we deliberately
/// don't hold an independent clone here since this Phase 0.6 pipeline
/// only decodes one frame per stream and exits.
#[cfg(target_os = "macos")]
fn try_enable_videotoolbox_decode(
    dec_ctx: &mut ffmpeg_next::codec::context::Context,
) -> bool {
    use ffmpeg_next::ffi::*;
    let mut hw_device: *mut AVBufferRef = std::ptr::null_mut();
    let ret = unsafe {
        av_hwdevice_ctx_create(
            &mut hw_device,
            AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            std::ptr::null(),     // device name — VT picks the system device
            std::ptr::null_mut(), // options
            0,
        )
    };
    if ret < 0 || hw_device.is_null() {
        tracing::debug!("av_hwdevice_ctx_create VIDEOTOOLBOX returned {ret}");
        return false;
    }
    unsafe {
        let raw = dec_ctx.as_mut_ptr();
        (*raw).hw_device_ctx = hw_device;
        (*raw).get_format = Some(videotoolbox_get_format);
    }
    true
}

/// FFmpeg `get_format` callback: prefer `AV_PIX_FMT_VIDEOTOOLBOX` from
/// the offered list, fall through to the first software format otherwise
/// so a codec VT can't accelerate still decodes (just on the CPU).
#[cfg(target_os = "macos")]
unsafe extern "C" fn videotoolbox_get_format(
    _ctx: *mut ffmpeg_next::ffi::AVCodecContext,
    pix_fmts: *const ffmpeg_next::ffi::AVPixelFormat,
) -> ffmpeg_next::ffi::AVPixelFormat {
    use ffmpeg_next::ffi::AVPixelFormat::*;
    let mut p = pix_fmts;
    while unsafe { *p } != AV_PIX_FMT_NONE {
        if unsafe { *p } == AV_PIX_FMT_VIDEOTOOLBOX {
            return AV_PIX_FMT_VIDEOTOOLBOX;
        }
        p = unsafe { p.add(1) };
    }
    unsafe { *pix_fmts }
}

/// Download a hardware-resident frame (VideoToolbox CVPixelBuffer) to
/// host system memory via `av_hwframe_transfer_data`. The destination
/// frame's pixel format is determined by FFmpeg — typically `NV12` for
/// 8-bit HEVC/H.264 and `P010` for 10-bit Main10.
fn download_hw_frame(
    hw_frame: &ffmpeg_next::frame::Video,
    sw_frame: &mut ffmpeg_next::frame::Video,
) -> Result<()> {
    let ret = unsafe {
        ffmpeg_next::ffi::av_hwframe_transfer_data(
            sw_frame.as_mut_ptr(),
            hw_frame.as_ptr(),
            0,
        )
    };
    if ret < 0 {
        return Err(Error::Ffmpeg(format!("av_hwframe_transfer_data: {ret}")));
    }
    Ok(())
}

/// Per-stream decode timing for a multi-frame benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct BenchResult {
    pub frames: u32,
    pub total: std::time::Duration,
    pub decode_path: DecodePath,
}

impl BenchResult {
    pub fn fps(&self) -> f32 {
        if self.total.is_zero() { 0.0 }
        else { self.frames as f32 / self.total.as_secs_f32() }
    }
}

/// Open a `.360`, return a streaming iterator that yields one
/// [`StreamPair`] per video-time-step (one frame from each of the two
/// HEVC streams, packed RGB8). Used by Phase 0.8's multi-frame `export`.
///
/// All the codec / hwaccel setup lives in [`StreamPairIter::new`]; the
/// iterator itself is just decoder.send_packet → receive_frame → swscale
/// → repack RGB8. Stops at EOF or after `n_frames` if non-zero.
pub fn iter_stream_pairs(path: &Path, hw: HwDecode, n_frames: u32) -> Result<StreamPairIter> {
    StreamPairIter::new(path, hw, n_frames)
}

pub struct StreamPairIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: Vec<usize>,
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    scalers: Vec<Option<ffmpeg_next::software::scaling::Context>>,
    hw_active: [bool; 2],
    dims: vr180_core::eac::Dims,
    decode_path: DecodePath,
    frame_limit: u32,
    frames_yielded: u32,
}

impl StreamPairIter {
    pub fn new(path: &Path, hw: HwDecode, frame_limit: u32) -> Result<Self> {
        init();
        let mut ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

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
        let mut decoders = Vec::with_capacity(2);
        let mut hw_active = [false, false];
        for (i, &idx) in video_indices.iter().take(2).enumerate() {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            #[cfg(target_os = "macos")]
            {
                let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
                if want_vt {
                    if try_enable_videotoolbox_decode(&mut codec_ctx) {
                        hw_active[i] = true;
                    } else if matches!(hw, HwDecode::VideoToolbox) {
                        return Err(Error::Ffmpeg("VT requested but unavailable".into()));
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (i, hw);
                if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg("VT is macOS-only".into()));
                }
            }
            decoders.push(codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?);
        }

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
                "stream width {w0} not a valid EAC layout"
            )));
        }
        let decode_path = if hw_active.iter().all(|&a| a) {
            DecodePath::VideoToolbox
        } else {
            DecodePath::Software
        };
        Ok(Self {
            ictx, video_indices, decoders,
            scalers: vec![None, None],
            hw_active, dims, decode_path,
            frame_limit,
            frames_yielded: 0,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.dims }
    pub fn decode_path(&self) -> DecodePath { self.decode_path }

    /// Pull the next `StreamPair` (one frame from each video stream).
    /// Returns `Ok(None)` at EOF or after the frame limit is hit.
    pub fn next_pair(&mut self) -> Result<Option<StreamPair>> {
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut frames: [Option<Vec<u8>>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut sw_storage = ffmpeg_next::frame::Video::empty();

        // Try receiving leftover-buffered frames first (the HEVC decoder
        // often has prepared frames from previous packets).
        for pos in 0..2 {
            if frames[pos].is_some() { continue; }
            let dec = &mut self.decoders[pos];
            if dec.receive_frame(&mut decoded).is_ok() {
                frames[pos] = Some(repack_to_rgb8(
                    &mut decoded, &mut sw_storage,
                    self.hw_active[pos], &mut self.scalers[pos],
                )?);
            }
        }

        // Then drive the packet loop until both slots are filled or EOF.
        if frames.iter().any(|f| f.is_none()) {
            loop {
                let res = self.ictx.packets().next();
                let (stream, packet) = match res {
                    Some(x) => x,
                    None => break,
                };
                let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                    Some(p) if frames[p].is_none() => p,
                    _ => continue,
                };
                let dec = &mut self.decoders[pos];
                if dec.send_packet(&packet).is_err() { continue; }
                while dec.receive_frame(&mut decoded).is_ok() {
                    if frames[pos].is_some() { break; }
                    frames[pos] = Some(repack_to_rgb8(
                        &mut decoded, &mut sw_storage,
                        self.hw_active[pos], &mut self.scalers[pos],
                    )?);
                    if frames.iter().all(|f| f.is_some()) { break; }
                }
                if frames.iter().all(|f| f.is_some()) { break; }
            }
        }

        let (Some(s0), Some(s4)) = (frames[0].take(), frames[1].take()) else {
            return Ok(None);  // EOF before both streams had a frame
        };
        self.frames_yielded += 1;
        Ok(Some(StreamPair { s0, s4, dims: self.dims, decode_path: self.decode_path }))
    }
}

/// Shared frame → packed-RGB8 conversion used by both single-frame and
/// streaming code paths.
fn repack_to_rgb8(
    decoded: &mut ffmpeg_next::frame::Video,
    sw_storage: &mut ffmpeg_next::frame::Video,
    hw_active: bool,
    scaler: &mut Option<ffmpeg_next::software::scaling::Context>,
) -> Result<Vec<u8>> {
    let frame_ref: &ffmpeg_next::frame::Video = if hw_active
        && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
    {
        download_hw_frame(decoded, sw_storage)?;
        sw_storage
    } else {
        decoded
    };
    let s = match scaler {
        Some(s) => s,
        None => {
            *scaler = Some(ffmpeg_next::software::scaling::Context::get(
                frame_ref.format(),
                frame_ref.width(), frame_ref.height(),
                ffmpeg_next::format::Pixel::RGB24,
                frame_ref.width(), frame_ref.height(),
                ffmpeg_next::software::scaling::Flags::BILINEAR,
            ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?);
            scaler.as_mut().unwrap()
        }
    };
    let mut rgb = ffmpeg_next::frame::Video::empty();
    s.run(frame_ref, &mut rgb).map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let src = rgb.data(0);
    let src_stride = rgb.stride(0);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &src[y * src_stride..y * src_stride + w * 3];
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// Multi-frame decode benchmark: decode `n_frames` from the FIRST video
/// stream as fast as possible, **no swscale RGB conversion**, returning
/// per-stream timing. This measures the steady-state decode throughput
/// where the single-frame cold-start cost has been amortized away —
/// which is where any hwaccel speedup actually shows up.
///
/// Doesn't assemble crosses; doesn't run the GPU pipeline. Pure decode
/// throughput test. Useful for the Phase 0.6 sw-vs-VT comparison.
pub fn bench_decode_throughput(
    path: &Path,
    n_frames: u32,
    hw: HwDecode,
) -> Result<BenchResult> {
    init();
    let mut ictx = ffmpeg_next::format::input(path)
        .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;

    let stream_idx = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
        .next()
        .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?
        .index();

    let stream = ictx.stream(stream_idx).unwrap();
    let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;

    let mut hw_active = false;
    #[cfg(target_os = "macos")]
    {
        let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
        if want_vt {
            if try_enable_videotoolbox_decode(&mut codec_ctx) {
                hw_active = true;
            } else if matches!(hw, HwDecode::VideoToolbox) {
                return Err(Error::Ffmpeg(
                    "VT hwaccel requested but unavailable on this system".into()
                ));
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = hw;
        if matches!(hw, HwDecode::VideoToolbox) {
            return Err(Error::Ffmpeg("VT is macOS-only".into()));
        }
    }

    let mut decoder = codec_ctx.decoder().video()
        .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;

    let mut decoded = ffmpeg_next::frame::Video::empty();
    let mut sw_storage = ffmpeg_next::frame::Video::empty();
    let mut count: u32 = 0;
    let t_start = std::time::Instant::now();

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx { continue; }
        if decoder.send_packet(&packet).is_err() { continue; }
        while decoder.receive_frame(&mut decoded).is_ok() {
            // For hwframes, do the transfer so we measure the full
            // "frames usable in host memory" throughput (matching real
            // pipeline usage). Skip if you want raw decode throughput.
            if hw_active
                && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
            {
                download_hw_frame(&decoded, &mut sw_storage)?;
            }
            count += 1;
            if count >= n_frames { break 'outer; }
        }
    }
    let total = t_start.elapsed();
    let decode_path = if hw_active { DecodePath::VideoToolbox } else { DecodePath::Software };
    Ok(BenchResult { frames: count, total, decode_path })
}
