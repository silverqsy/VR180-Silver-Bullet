//! H.265 software encode via `ffmpeg-next` + `libx265`.
//!
//! Phase 0.8 deliverable: a minimal write-side wrapper that takes
//! packed RGB8 frames and produces a playable `.mov` / `.mp4`.
//!
//! Deferred to 0.8.5:
//! - VideoToolbox HEVC hardware encode (`hevc_videotoolbox`) for the
//!   ~5-8× speedup vs libx265 on Apple Silicon.
//! - The Swift helpers (`mvhevc_encode` for spatial video, `apac_encode`
//!   for ambisonic audio, `vt_denoise` for temporal denoise). The
//!   binaries already exist in `helpers/swift/`; just need spawn glue.
//! - 10-bit Main10 output (currently 8-bit yuv420p).
//! - Audio passthrough from the source `.360`.
//! - sv3d / st3d / SA3D atom writers (replaces Python `spatialmedia`).

use crate::{Error, Result};
use ffmpeg_next as ffmpeg;
use std::path::Path;

/// One-shot H.265 software encoder for `Rgba8`-style packed RGB8 frames.
///
/// Usage:
/// ```ignore
/// let mut enc = H265Encoder::create(out_path, w, h, 30.0, 8000)?;
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
    /// `bitrate_kbps` is hint-only — libx265 also respects `--crf`-style
    /// quality control via `x265-params`; we use a single bitrate target
    /// for now since Phase 0.8 is just proving the wiring.
    pub fn create(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        crate::decode::init(); // shared one-time ffmpeg_init

        let mut octx = ffmpeg::format::output(&path)
            .map_err(|e| Error::Ffmpeg(format!("output ctx {path:?}: {e}")))?;

        let codec = ffmpeg::codec::encoder::find_by_name("libx265")
            .ok_or_else(|| Error::Ffmpeg(
                "libx265 encoder not available in this FFmpeg build".into()
            ))?;

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
        }

        let encoder = enc_ctx.encoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video encoder context: {e}")))?
            .open_as(codec)
            .map_err(|e| Error::Ffmpeg(format!("open libx265: {e}")))?;

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
