//! H.265 encode via `ffmpeg-next`. Two backends:
//!
//! - **`Libx265`** — software, cross-platform, slow but quality-controllable
//!   via a bitrate target.
//! - **`VideoToolbox`** — macOS-only hardware encode (`hevc_videotoolbox`).
//!   Phase 0.8.5 deliverable. 3-5× faster than libx265 on Apple Silicon at
//!   matched bitrate; refuses to fall back to software (`allow_sw=0`) so a
//!   missing HW encoder surfaces as a hard error instead of a silent perf
//!   cliff.
//!
//! Both backends produce a `.mov` / `.mp4` with `hvc1` codec tag (Apple /
//! Vision Pro compat) and an 8-bit yuv420p stream.
//!
//! Deferred to later phases:
//! - 10-bit Main10 output (would require `p010le` input plumbing).
//! - The Swift helpers (`mvhevc_encode` for MV-HEVC stereo, `apac_encode`
//!   for ambisonic audio, `vt_denoise`). Binaries exist in `helpers/swift/`;
//!   just need spawn glue (Phase 0.8.6).
//! - Audio passthrough from the source `.360` (Phase 0.8.6 too).
//! - sv3d / st3d / SA3D atom writers (Phase 0.8.7).

use crate::{Error, Result};
use ffmpeg_next as ffmpeg;
use std::ffi::CString;
use std::path::Path;

/// Which encoder backend `H265Encoder` should use.
///
/// `VideoToolbox` is selectable on every target so the CLI can give a
/// uniform error ("vt is macOS-only") on Windows / Linux rather than
/// not compiling the variant — but `create` will refuse to open it
/// outside macOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    /// `libx265` software encoder. Cross-platform, quality-tunable.
    Libx265,
    /// `hevc_videotoolbox` Apple hardware encoder. macOS only.
    VideoToolbox,
}

impl EncoderBackend {
    fn codec_name(self) -> &'static str {
        match self {
            EncoderBackend::Libx265      => "libx265",
            EncoderBackend::VideoToolbox => "hevc_videotoolbox",
        }
    }
}

/// One-shot H.265 encoder for `Rgba8`-style packed RGB8 frames.
/// Backend is libx265 (cross-platform software) or hevc_videotoolbox
/// (macOS hardware) — see `EncoderBackend`.
///
/// Usage:
/// ```ignore
/// let mut enc = H265Encoder::create(
///     out_path, w, h, 30.0, 8000,
///     EncoderBackend::VideoToolbox,
/// )?;
/// for frame in frames {
///     enc.encode_frame(&frame)?;
/// }
/// enc.finish()?;
/// ```
pub struct H265Encoder {
    octx: ffmpeg::format::context::Output,
    encoder: ffmpeg::codec::encoder::Video,
    scaler: ffmpeg::software::scaling::Context,
    stream_index: usize,
    time_base: ffmpeg::Rational,
    frame_count: i64,
    w: u32,
    h: u32,
    finished: bool,
}

impl H265Encoder {
    /// Create a new encoder that writes a fresh `.mov` / `.mp4` at `path`.
    /// Container is picked from the file extension; HEVC-in-MP4 needs
    /// the `hvc1` codec tag (set automatically).
    ///
    /// `bitrate_kbps` is the target average bitrate, honored by both
    /// backends (libx265 maps it to its 1-pass ABR mode; VT maps it to
    /// the `bit_rate` AVCodecContext field).
    ///
    /// `backend` picks the codec. `VideoToolbox` is macOS-only and will
    /// return an `Err` on other platforms.
    pub fn create(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
        backend: EncoderBackend,
    ) -> Result<Self> {
        crate::decode::init(); // shared one-time ffmpeg_init

        if backend == EncoderBackend::VideoToolbox && !cfg!(target_os = "macos") {
            return Err(Error::Ffmpeg(
                "hevc_videotoolbox is macOS-only. Use EncoderBackend::Libx265 \
                 on Windows / Linux (NVENC + Vulkan interop arrive in Phase 0.8.5+W).".into()
            ));
        }

        let mut octx = ffmpeg::format::output(&path)
            .map_err(|e| Error::Ffmpeg(format!("output ctx {path:?}: {e}")))?;

        let codec_name = backend.codec_name();
        let codec = ffmpeg::codec::encoder::find_by_name(codec_name)
            .ok_or_else(|| Error::Ffmpeg(format!(
                "{codec_name} encoder not available in this FFmpeg build"
            )))?;

        // Add the output stream up front so we can finalize its params
        // after the encoder is opened. We grab the index here and drop
        // the borrow before any mut work on octx.
        let stream_index = {
            let stream = octx.add_stream(codec)
                .map_err(|e| Error::Ffmpeg(format!("add_stream: {e}")))?;
            stream.index()
        };

        // Pick a sensible AVRational fps (29.97 ≈ 30000/1001, etc.).
        let (num, den) = approx_rational(fps);
        let time_base = ffmpeg::Rational(den, num);

        let mut enc_ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
        unsafe {
            let raw = enc_ctx.as_mut_ptr();
            (*raw).width = w as i32;
            (*raw).height = h as i32;
            (*raw).pix_fmt = ffmpeg::format::Pixel::YUV420P.into();
            (*raw).time_base = time_base.into();
            (*raw).framerate = ffmpeg::Rational(num, den).into();
            (*raw).bit_rate = (bitrate_kbps as i64) * 1000;
            (*raw).gop_size = 60; // ~2s at 30 fps
            // MP4 needs hvc1 codec tag (vs hev1) for QuickTime / Vision Pro.
            (*raw).codec_tag = u32::from_le_bytes(*b"hvc1");
            // Global header flag for mp4 / mov muxers.
            if octx.format().flags()
                .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER)
            {
                (*raw).flags |= ffmpeg::ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }

            // Backend-specific private-data tuning. Set via av_opt_set on
            // priv_data so we don't depend on ffmpeg-next's `Dictionary`
            // wrapper (whose API drifts between releases).
            match backend {
                EncoderBackend::VideoToolbox => {
                    // realtime=0: prioritize quality over latency. The default
                    // is "auto", which on macOS 14+ VT decides per-frame and
                    // usually picks fast → quality regression. Pin to 0.
                    set_opt(raw, "realtime",      "0")?;
                    // allow_sw=0: refuse the libavcodec→libx265 silent fallback
                    // if VT init fails for any reason. We want a hard error so
                    // the user knows the HW path is broken.
                    set_opt(raw, "allow_sw",      "0")?;
                    // profile=main: 8-bit, baseline of Apple decoders. main10
                    // would require p010le input; deferred until we wire the
                    // 10-bit RGB16Float→P010 swscale path.
                    set_opt(raw, "profile",       "main")?;
                    // power_efficient=0: don't let VT throttle to save battery
                    // when we're plugged in and want max throughput.
                    set_opt(raw, "power_efficient", "0")?;
                }
                EncoderBackend::Libx265 => {
                    // libx265 defaults are fine — ABR at the configured
                    // bit_rate, medium preset. Could expose --preset
                    // and --crf knobs later if we want a quality dial.
                }
            }
        }

        let encoder = enc_ctx.encoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video encoder context: {e}")))?
            .open_as(codec)
            .map_err(|e| Error::Ffmpeg(format!("open {codec_name}: {e}")))?;

        // Hand the configured codec parameters to the output stream so
        // the muxer writes the correct sample description.
        {
            let mut stream = octx.stream_mut(stream_index)
                .ok_or_else(|| Error::Ffmpeg("output stream vanished".into()))?;
            stream.set_parameters(&encoder);
            stream.set_time_base(time_base);
        }

        // swscale: packed RGB24 → YUV420P
        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24,
            w, h,
            ffmpeg::format::Pixel::YUV420P,
            w, h,
            ffmpeg::software::scaling::Flags::BICUBIC,
        ).map_err(|e| Error::Ffmpeg(format!("rgb→yuv scaler: {e}")))?;

        octx.write_header()
            .map_err(|e| Error::Ffmpeg(format!("write_header: {e}")))?;

        Ok(Self {
            octx, encoder, scaler, stream_index, time_base,
            frame_count: 0, w, h, finished: false,
        })
    }

    /// Encode one packed RGB8 frame. Length must equal `w * h * 3`.
    pub fn encode_frame(&mut self, rgb8: &[u8]) -> Result<()> {
        let want = (self.w as usize) * (self.h as usize) * 3;
        if rgb8.len() != want {
            return Err(Error::Ffmpeg(format!(
                "encode_frame: expected {want} bytes, got {}", rgb8.len()
            )));
        }

        // Source frame: packed RGB24. We have to fill its single plane
        // honoring linesize alignment (ffmpeg-next typically gives us
        // an aligned linesize ≥ w*3). Grab the stride before borrowing
        // the data buffer mutably.
        let mut rgb_frame = ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB24, self.w, self.h
        );
        let stride = rgb_frame.stride(0);
        {
            let dst = rgb_frame.data_mut(0);
            let row_bytes = (self.w as usize) * 3;
            for y in 0..self.h as usize {
                let dst_off = y * stride;
                let src_off = y * row_bytes;
                dst[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&rgb8[src_off..src_off + row_bytes]);
            }
        }

        // Convert to YUV420P.
        let mut yuv_frame = ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::YUV420P, self.w, self.h
        );
        self.scaler.run(&rgb_frame, &mut yuv_frame)
            .map_err(|e| Error::Ffmpeg(format!("scaler run: {e}")))?;
        yuv_frame.set_pts(Some(self.frame_count));
        self.frame_count += 1;

        // Send + drain.
        self.encoder.send_frame(&yuv_frame)
            .map_err(|e| Error::Ffmpeg(format!("send_frame: {e}")))?;
        self.drain_packets()?;
        Ok(())
    }

    fn drain_packets(&mut self) -> Result<()> {
        use ffmpeg::util::error::EAGAIN;
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.stream_index);
                    let stream_tb = self.octx.stream(self.stream_index).unwrap().time_base();
                    packet.rescale_ts(self.time_base, stream_tb);
                    packet.write_interleaved(&mut self.octx)
                        .map_err(|e| Error::Ffmpeg(format!("write packet: {e}")))?;
                }
                Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => break,
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(Error::Ffmpeg(format!("receive_packet: {e}"))),
            }
        }
        Ok(())
    }

    /// Flush the encoder, write the trailer, and close the file.
    /// Must be called or the output `.mp4` will be unplayable.
    pub fn finish(mut self) -> Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<()> {
        if self.finished { return Ok(()); }
        self.encoder.send_eof()
            .map_err(|e| Error::Ffmpeg(format!("send_eof: {e}")))?;
        self.drain_packets()?;
        self.octx.write_trailer()
            .map_err(|e| Error::Ffmpeg(format!("write_trailer: {e}")))?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for H265Encoder {
    fn drop(&mut self) {
        // Best-effort flush if user forgot to call .finish() — keeps
        // the output file at least partially playable.
        let _ = self.finish_inner();
    }
}

/// Set a string-valued option on an AVCodecContext's encoder private data
/// (e.g. `realtime` / `allow_sw` / `profile` on `hevc_videotoolbox`).
///
/// We go through `av_opt_set` rather than ffmpeg-next's `Dictionary`
/// wrapper because we already hold the raw `AVCodecContext*` and want
/// errors surfaced inline at config time, not silently swallowed by
/// `open_as_with`.
///
/// # Safety
/// Caller must pass a valid `AVCodecContext*` that has not been opened
/// yet. The option name must be one the encoder actually recognizes;
/// passing an unknown name returns `AVERROR_OPTION_NOT_FOUND`.
unsafe fn set_opt(
    ctx: *mut ffmpeg::ffi::AVCodecContext,
    name: &str, value: &str,
) -> Result<()> {
    let name_c  = CString::new(name).unwrap();
    let value_c = CString::new(value).unwrap();
    // search_flags=AV_OPT_SEARCH_CHILDREN so we hit the encoder's priv_data
    // options (which live one level below the AVCodecContext options).
    let ret = ffmpeg::ffi::av_opt_set(
        (*ctx).priv_data,
        name_c.as_ptr(),
        value_c.as_ptr(),
        0,
    );
    if ret < 0 {
        return Err(Error::Ffmpeg(format!(
            "av_opt_set({name}={value}) failed: ret={ret}"
        )));
    }
    Ok(())
}

/// Approximate an f32 fps as a rational (num/den).
/// 29.97 → 30000/1001, 59.94 → 60000/1001, integer fps → fps/1.
fn approx_rational(fps: f32) -> (i32, i32) {
    let candidates = [
        (24000, 1001), (24, 1),
        (25, 1),
        (30000, 1001), (30, 1),
        (50, 1),
        (60000, 1001), (60, 1),
    ];
    for (n, d) in candidates {
        let approx = n as f32 / d as f32;
        if (approx - fps).abs() < 0.01 { return (n, d); }
    }
    // Fallback: scale by 1000 (lossy on weird fps but always positive).
    ((fps * 1000.0).round() as i32, 1000)
}
