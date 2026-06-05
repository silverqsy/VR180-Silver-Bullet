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
    /// `prores_videotoolbox` Apple hardware encoder. macOS only.
    /// Profile passed via the encoder config's `prores_profile` field.
    ProResVideoToolbox,
    /// `prores_ks` Kostya Shishkov's software encoder. Cross-platform.
    ProResKs,
}

impl EncoderBackend {
    pub fn codec_name(self) -> &'static str {
        match self {
            EncoderBackend::Libx265             => "libx265",
            EncoderBackend::VideoToolbox        => "hevc_videotoolbox",
            EncoderBackend::ProResVideoToolbox  => "prores_videotoolbox",
            EncoderBackend::ProResKs            => "prores_ks",
        }
    }

    pub fn is_prores(self) -> bool {
        matches!(self, EncoderBackend::ProResVideoToolbox | EncoderBackend::ProResKs)
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
///
/// Phase 0.7.5.6: when `encode_path` is `EncodePath::ZeroCopyVt`, the
/// `scaler` field is unused and `encode_pixel_buffer` is the entry
/// point (instead of `encode_frame`). The encoder context has
/// `hw_device_ctx` + `hw_frames_ctx` set so it accepts AVFrames with
/// `format = AV_PIX_FMT_VIDEOTOOLBOX` and `data[3] = CVPixelBufferRef`.
pub struct H265Encoder {
    octx: ffmpeg::format::context::Output,
    encoder: ffmpeg::codec::encoder::Video,
    /// Used only by the CPU-input path (`encode_frame`). On the
    /// zero-copy-VT path, we feed AVFrames with format=VIDEOTOOLBOX
    /// directly and skip swscale entirely; a 1×1 placeholder scaler
    /// fills the field so the struct shape stays uniform.
    scaler: ffmpeg::software::scaling::Context,
    stream_index: usize,
    time_base: ffmpeg::Rational,
    frame_count: i64,
    w: u32,
    h: u32,
    finished: bool,
    encode_path: EncodePath,
    /// Deferred to first encode call so post-construction setters
    /// (`tag_apmp_vr180_sbs`) can modify the stream's codecpar
    /// side-data before the muxer writes the moov header.
    header_written: bool,
    /// Target pixel format the scaler writes (one of YUV420P /
    /// YUV420P10LE / P010LE depending on backend × bit_depth). Used
    /// by `encode_frame` to allocate the destination frame.
    enc_pix_fmt: ffmpeg::format::Pixel,
    /// Lazy RGB48LE → enc_pix_fmt scaler. Initialised on first call
    /// to `encode_frame_rgb48`. Independent of the primary RGB24
    /// scaler so 10-bit and 8-bit input frames can both be handled
    /// without rebuilding swscale every frame.
    scaler_rgb48: Option<ffmpeg::software::scaling::Context>,
}

/// Which `encode_*` entry point this encoder was configured for.
/// `CpuInput` is the legacy path (`encode_frame(&[u8])`); `ZeroCopyVt`
/// is the Phase 0.7.5.6 path (`encode_pixel_buffer(&EncodePixelBuffer)`).
/// Mixing them is a bug — methods of the wrong kind return `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodePath {
    CpuInput,
    #[cfg(target_os = "macos")]
    ZeroCopyVt,
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
    /// Legacy 8-bit Main-profile entry point. New callers should use
    /// [`H265Encoder::create_with_bit_depth`] and pick 8 or 10.
    pub fn create(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
        backend: EncoderBackend,
    ) -> Result<Self> {
        Self::create_with_bit_depth(path, w, h, fps, bitrate_kbps, backend, 8, -1)
    }

    /// `bit_depth = 8` → Main profile, YUV420P input to the encoder.
    /// `bit_depth = 10` → Main10 profile, YUV420P10LE (libx265) /
    /// P010LE (VideoToolbox). Caller still passes packed RGB8 frames
    /// to `encode_frame` — libswscale handles the 8 → 10 bit
    /// expansion. True 10-bit shader output is a follow-up; for now
    /// the codec gets a wider quantization grid (better gradation in
    /// shadows / sky) at the cost of larger files.
    /// `prores_profile` (0..=5) is the ProRes profile when `backend.is_prores()`.
    /// Pass `-1` (or any negative value) for HEVC.
    pub fn create_with_bit_depth(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
        backend: EncoderBackend,
        bit_depth: u8,
        prores_profile: i32,
    ) -> Result<Self> {
        crate::decode::init(); // shared one-time ffmpeg_init

        if backend == EncoderBackend::VideoToolbox && !cfg!(target_os = "macos") {
            return Err(Error::Ffmpeg(
                "hevc_videotoolbox is macOS-only. Use EncoderBackend::Libx265 \
                 on Windows / Linux (NVENC + Vulkan interop arrive in Phase 0.8.5+W).".into()
            ));
        }
        if bit_depth != 8 && bit_depth != 10 {
            return Err(Error::Ffmpeg(format!(
                "H265Encoder: bit_depth must be 8 or 10, got {bit_depth}"
            )));
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

        // Pick encoder pixel format + corresponding swscale target by
        // bit_depth. VT prefers P010LE for hardware-native 10-bit
        // (semi-planar); libx265 wants YUV420P10LE (fully planar).
        let enc_pix_fmt = match (bit_depth, backend) {
            (10, EncoderBackend::VideoToolbox)        => ffmpeg::format::Pixel::P010LE,
            (10, EncoderBackend::Libx265)             => ffmpeg::format::Pixel::YUV420P10LE,
            (_,  EncoderBackend::ProResVideoToolbox)
                | (_, EncoderBackend::ProResKs)       => ffmpeg::format::Pixel::YUV422P10LE,
            _                                         => ffmpeg::format::Pixel::YUV420P,
        };

        let mut enc_ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
        unsafe {
            let raw = enc_ctx.as_mut_ptr();
            (*raw).width = w as i32;
            (*raw).height = h as i32;
            (*raw).pix_fmt = enc_pix_fmt.into();
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
                    // profile = main / main10 depending on bit_depth.
                    let prof = if bit_depth == 10 { "main10" } else { "main" };
                    set_opt(raw, "profile", prof)?;
                    // power_efficient=0: don't let VT throttle to save battery
                    // when we're plugged in and want max throughput.
                    set_opt(raw, "power_efficient", "0")?;
                }
                EncoderBackend::Libx265 => {
                    // libx265 defaults are fine — ABR at the configured
                    // bit_rate, medium preset. Could expose --preset
                    // and --crf knobs later if we want a quality dial.
                }
                EncoderBackend::ProResVideoToolbox | EncoderBackend::ProResKs => {
                    // ProRes ignores bit_rate (the profile picks the
                    // target rate). Codec tag is per-profile and the
                    // muxer derives it from the profile — leave the
                    // hvc1 we set above out by overwriting with 0 so
                    // the muxer picks the right tag (apch / apcn /
                    // apco / apcs / ap4h / ap4x).
                    (*raw).codec_tag = 0;
                    (*raw).bit_rate = 0;
                    // Profile is read from the encoder context's
                    // `profile` field directly.
                    if prores_profile >= 0 {
                        (*raw).profile = prores_profile;
                    }
                    if matches!(backend, EncoderBackend::ProResKs) {
                        // prores_ks (libavcodec): pin the QuickTime
                        // "Apple" vendor tag so QuickTime / FCP recognize
                        // the file as native ProRes (not "Other").
                        set_opt(raw, "vendor", "apl0")?;
                    }
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

        // swscale: packed RGB24 → encoder's pix_fmt (handles 8 → 10
        // bit expansion internally for the P010LE / YUV420P10LE
        // targets).
        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24,
            w, h,
            enc_pix_fmt,
            w, h,
            ffmpeg::software::scaling::Flags::BICUBIC,
        ).map_err(|e| Error::Ffmpeg(format!("rgb→yuv scaler: {e}")))?;

        Ok(Self {
            octx, encoder, scaler, stream_index, time_base,
            frame_count: 0, w, h, finished: false,
            encode_path: EncodePath::CpuInput,
            header_written: false,
            enc_pix_fmt,
            scaler_rgb48: None,
        })
    }

    /// Write the moov header if it hasn't been written yet. Called
    /// before the first encode dispatch on every path. Header writes
    /// are deferred from `create()` to here so post-construction
    /// setters (like `tag_apmp_vr180_sbs`) can modify the stream's
    /// codecpar side-data before the muxer locks in the moov.
    fn ensure_header_written(&mut self) -> Result<()> {
        if self.header_written { return Ok(()); }
        self.octx.write_header()
            .map_err(|e| Error::Ffmpeg(format!("write_header: {e}")))?;
        self.header_written = true;
        Ok(())
    }

    /// Encode one packed RGB8 frame. Length must equal `w * h * 3`.
    pub fn encode_frame(&mut self, rgb8: &[u8]) -> Result<()> {
        self.ensure_header_written()?;
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

        // Convert to the encoder's pixel format (YUV420P for 8-bit,
        // YUV420P10LE / P010LE for 10-bit).
        let mut yuv_frame = ffmpeg::frame::Video::new(
            self.enc_pix_fmt, self.w, self.h
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

    /// Encode one packed RGB48LE frame (`w * h * 6` bytes, little-endian
    /// 16-bit per channel, alpha already stripped). Only meaningful for
    /// Main10 (bit_depth == 10) encoders — for 8-bit encoders we'd be
    /// scaling down to 8-bit at the swscale step anyway.
    ///
    /// Uses a lazily-initialised second swscale context (RGB48LE →
    /// `enc_pix_fmt`) since the primary scaler is set up for RGB24.
    pub fn encode_frame_rgb48(&mut self, rgb48le: &[u8]) -> Result<()> {
        self.ensure_header_written()?;
        let want = (self.w as usize) * (self.h as usize) * 6;
        if rgb48le.len() != want {
            return Err(Error::Ffmpeg(format!(
                "encode_frame_rgb48: expected {want} bytes, got {}", rgb48le.len()
            )));
        }
        // Lazy second scaler for RGB48LE input.
        if self.scaler_rgb48.is_none() {
            self.scaler_rgb48 = Some(
                ffmpeg::software::scaling::Context::get(
                    ffmpeg::format::Pixel::RGB48LE,
                    self.w, self.h,
                    self.enc_pix_fmt,
                    self.w, self.h,
                    ffmpeg::software::scaling::Flags::BICUBIC,
                ).map_err(|e| Error::Ffmpeg(format!("rgb48→yuv scaler: {e}")))?
            );
        }

        let mut rgb_frame = ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB48LE, self.w, self.h
        );
        let stride = rgb_frame.stride(0);
        {
            let dst = rgb_frame.data_mut(0);
            let row_bytes = (self.w as usize) * 6;
            for y in 0..self.h as usize {
                let dst_off = y * stride;
                let src_off = y * row_bytes;
                dst[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&rgb48le[src_off..src_off + row_bytes]);
            }
        }

        let mut yuv_frame = ffmpeg::frame::Video::new(
            self.enc_pix_fmt, self.w, self.h
        );
        self.scaler_rgb48.as_mut().unwrap().run(&rgb_frame, &mut yuv_frame)
            .map_err(|e| Error::Ffmpeg(format!("rgb48 scaler run: {e}")))?;
        yuv_frame.set_pts(Some(self.frame_count));
        self.frame_count += 1;

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

    /// Encode an IOSurface-backed BGRA pixel buffer directly via the
    /// hardware encoder — Phase 0.7.5.6 zero-copy path. The encoder
    /// must have been created with `create_zero_copy_vt`. The
    /// underlying CVPixelBuffer is `CFRetain`'d for the encoder's
    /// lifetime needs; an AVBufferRef with a `cv_release_cb`
    /// callback releases the retain when the AVFrame is freed.
    ///
    /// Caller is responsible for ensuring all wgpu writes to the
    /// IOSurface have completed before this call (e.g. via
    /// `device.poll(Maintain::Wait)`). Without that the encoder may
    /// see partially-written pixels.
    #[cfg(target_os = "macos")]
    pub fn encode_pixel_buffer(
        &mut self,
        pb: &crate::interop_macos::EncodePixelBuffer,
    ) -> Result<()> {
        self.encode_cv_pixel_buffer_raw(pb.pixel_buffer.as_raw() as *mut std::ffi::c_void)
    }

    /// Same as [`Self::encode_pixel_buffer`] but takes the 10-bit P010
    /// IOSurface variant. The encoder must have been created with
    /// `create_zero_copy_vt_p010`.
    #[cfg(target_os = "macos")]
    pub fn encode_pixel_buffer_p010(
        &mut self,
        pb: &crate::interop_macos::EncodePixelBufferP010,
    ) -> Result<()> {
        self.encode_cv_pixel_buffer_raw(pb.pixel_buffer.as_raw() as *mut std::ffi::c_void)
    }

    /// Internal: feed a raw CVPixelBufferRef (whether BGRA or P010) to
    /// the zero-copy VT encoder. The encoder must have been created
    /// with `create_zero_copy_vt` (or `_p010` variant).
    #[cfg(target_os = "macos")]
    fn encode_cv_pixel_buffer_raw(
        &mut self,
        pb_raw: *mut std::ffi::c_void,
    ) -> Result<()> {
        if self.encode_path != EncodePath::ZeroCopyVt {
            return Err(Error::Ffmpeg(
                "encode_pixel_buffer requires create_zero_copy_vt".into()
            ));
        }
        self.ensure_header_written()?;
        use ffmpeg::ffi::*;
        unsafe {
            let frame = av_frame_alloc();
            if frame.is_null() {
                return Err(Error::Ffmpeg("av_frame_alloc failed".into()));
            }
            (*frame).format = AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX as i32;
            (*frame).width  = self.w as i32;
            (*frame).height = self.h as i32;
            (*frame).pts    = self.frame_count;

            let enc_raw = self.encoder.as_mut_ptr();
            let hwf_ref = (*enc_raw).hw_frames_ctx;
            if hwf_ref.is_null() {
                av_frame_free(&mut (frame as *mut _));
                return Err(Error::Ffmpeg(
                    "encoder has no hw_frames_ctx (zero-copy VT init failed)".into()
                ));
            }
            (*frame).hw_frames_ctx = av_buffer_ref(hwf_ref);

            cf_retain(pb_raw as *const _);
            (*frame).data[3] = pb_raw as *mut u8;

            let buf = av_buffer_create(
                std::ptr::null_mut(), 0,
                Some(cv_pixel_buffer_release_callback),
                pb_raw as *mut std::ffi::c_void,
                AV_BUFFER_FLAG_READONLY as i32,
            );
            if buf.is_null() {
                cf_release(pb_raw as *const _);
                av_buffer_unref(&mut (*frame).hw_frames_ctx);
                av_frame_free(&mut (frame as *mut _));
                return Err(Error::Ffmpeg("av_buffer_create failed".into()));
            }
            (*frame).buf[0] = buf;

            self.frame_count += 1;

            let ret = avcodec_send_frame(enc_raw, frame);
            if ret < 0 {
                av_frame_free(&mut (frame as *mut _));
                return Err(Error::Ffmpeg(format!("avcodec_send_frame: {ret}")));
            }
            av_frame_free(&mut (frame as *mut _));
        }
        self.drain_packets()?;
        Ok(())
    }

    /// Flush the encoder, write the trailer, and close the file.
    /// Must be called or the output `.mp4` will be unplayable.
    pub fn finish(mut self) -> Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<()> {
        if self.finished { return Ok(()); }
        self.ensure_header_written()?;
        self.encoder.send_eof()
            .map_err(|e| Error::Ffmpeg(format!("send_eof: {e}")))?;
        self.drain_packets()?;
        self.octx.write_trailer()
            .map_err(|e| Error::Ffmpeg(format!("write_trailer: {e}")))?;
        self.finished = true;
        Ok(())
    }
}

// ===== Phase 0.8.6: AVSphericalMapping (re-declared) =====
//
// `AVSphericalMapping` is defined in `libavutil/spherical.h` but
// ffmpeg-sys-next 8.1's bindgen header set doesn't include that file,
// so the type isn't available via `ffmpeg::ffi`. The C ABI is stable
// (the struct hasn't changed since FFmpeg 3.1 added it), so we
// re-declare it locally with the `M` prefix to avoid name clashes if
// ffmpeg-sys-next ever ships its own version. Field ordering and
// types must match `libavutil/spherical.h` exactly — verified
// against FFmpeg 7.x.
//
// `AVSphericalProjection` enum values match the C enum:
//   EQUIRECTANGULAR      = 0
//   CUBEMAP              = 1
//   EQUIRECTANGULAR_TILE = 2
//   HALF_EQUIRECTANGULAR = 3   ← what APMP "hequ" maps from
//   FISHEYE              = 4

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MAVSphericalMapping {
    /// AVSphericalProjection (C enum, sized as `c_int` = i32).
    projection:   i32,
    /// Rotation in 16.16 fixed-point degrees (per AVSphericalMapping
    /// docs). 0 = identity orientation, which is what we want — gyro
    /// stabilization has already aligned the frame upstream.
    yaw:          i32,
    pitch:        i32,
    roll:         i32,
    /// 0.32 fixed-point fractional bounds (0 = no crop / full image).
    bound_left:   u32,
    bound_top:    u32,
    bound_right:  u32,
    bound_bottom: u32,
    /// Padding-frame count for cubemaps; 0 for equi / half-equi.
    padding:      u32,
}

const M_AV_SPHERICAL_HALF_EQUIRECTANGULAR: i32 = 3;

// ===== Phase 0.8.6: APMP atom side-data injection =====
//
// FFmpeg's mov muxer writes the Apple Projected Media Profile atom
// tree (vexu/proj/prji/eyes/stri/pack/pkin + hfov) when:
//   (a) `strict_std_compliance ≤ FF_COMPLIANCE_UNOFFICIAL` (= -1), AND
//   (b) the output stream's `codecpar->coded_side_data` carries
//       both `AV_PKT_DATA_STEREO3D` and `AV_PKT_DATA_SPHERICAL`
//       entries with the right field values.
//
// We piggy-back on FFmpeg's writer rather than hand-rolling the box
// tree, because FFmpeg's writer is already validated against
// AVFoundation. The exact box layout is in
// `libavformat/movenc.c::mov_write_vexu_tag` and friends; the spec
// (Apple's QuickTime + ISOBMFF Spatial Media Extensions, v1.9.8 Beta)
// matches FFmpeg's output bit-for-bit.
//
// We side-step the AV_FRAME_DATA path (per-frame side-data) and set
// the side-data on the output stream's codecpar once at setup time —
// the muxer reads it when finalizing the moov atom.

impl H265Encoder {
    /// Tag the video track for VR180 SBS recognition by visionOS /
    /// Vision Pro. Adds APMP atoms (`vexu/proj/prji=hequ`,
    /// `vexu/eyes/stri=0x03`, `vexu/pack/pkin=side`, `hfov=180°`) to
    /// the `hvc1` sample description, plus Google sv3d/st3d for
    /// YouTube / Quest compat.
    ///
    /// Implementation: set side-data on the output stream's
    /// `codecpar->coded_side_data` via `av_packet_side_data_new`.
    /// FFmpeg's mov muxer reads from there when writing the moov.
    ///
    /// MUST be called before the first `encode_*` (the header write
    /// is deferred to first encode for exactly this reason).
    ///
    /// Requires FFmpeg ≥ 7.0 for the `av_packet_side_data_new` API
    /// on the codecpar array. ffmpeg-next 8.1's bundled `ffmpeg-sys-next`
    /// 8.1 binds against libavformat ≥ 62, which has this. Older
    /// FFmpegs (≤ 6.x) would compile but the atoms wouldn't show up
    /// (the muxer ignores side-data it doesn't recognize).
    pub fn tag_apmp_vr180_sbs(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::Ffmpeg(
                "tag_apmp_vr180_sbs must be called before the first encode()".into()
            ));
        }
        use ffmpeg::ffi::*;

        // 1. Allow unofficial extensions on the codec (needed by the
        //    muxer's APMP write path). Set strict_std_compliance =
        //    FF_COMPLIANCE_UNOFFICIAL on the encoder context.
        const FF_COMPLIANCE_UNOFFICIAL: i32 = -1;
        unsafe {
            let raw = self.encoder.as_mut_ptr();
            (*raw).strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;
        }

        // 2. Get the output stream's codec parameters. The mov muxer
        //    reads side-data from here at write_trailer time.
        let codecpar = unsafe {
            let mut stream = self.octx.stream_mut(self.stream_index)
                .ok_or_else(|| Error::Ffmpeg("output stream missing".into()))?;
            // ffmpeg-next's Stream::parameters() returns a wrapper;
            // we need the raw AVCodecParameters* to use the
            // av_packet_side_data_new API.
            (*stream.as_mut_ptr()).codecpar
        };
        if codecpar.is_null() {
            return Err(Error::Ffmpeg("stream has no codecpar".into()));
        }

        // 3. AV_PKT_DATA_STEREO3D — frame-packed side-by-side, view=packed.
        unsafe {
            // AVStereo3D layout (libavutil/stereo3d.h). We zero-init
            // and set the fields the muxer needs. Don't use
            // av_stereo3d_alloc() — that returns a heap pointer we'd
            // have to free; av_packet_side_data_new gives us a buffer
            // of the right size to memcpy into.
            let mut stereo: AVStereo3D = std::mem::zeroed();
            stereo.type_ = AVStereo3DType::AV_STEREO3D_SIDEBYSIDE;
            stereo.view  = AVStereo3DView::AV_STEREO3D_VIEW_PACKED;
            // Optional: primary_eye=LEFT, baseline=0 (we don't know
            // the GoPro Max IPD precisely; AVFoundation accepts 0).
            stereo.primary_eye = AVStereo3DPrimaryEye::AV_PRIMARY_EYE_LEFT;

            let entry = av_packet_side_data_new(
                &mut (*codecpar).coded_side_data,
                &mut (*codecpar).nb_coded_side_data,
                AVPacketSideDataType::AV_PKT_DATA_STEREO3D,
                std::mem::size_of::<AVStereo3D>(),
                0,
            );
            if entry.is_null() {
                return Err(Error::Ffmpeg(
                    "av_packet_side_data_new STEREO3D returned NULL".into()
                ));
            }
            std::ptr::copy_nonoverlapping(
                &stereo as *const _ as *const u8,
                (*entry).data,
                std::mem::size_of::<AVStereo3D>(),
            );
        }

        // 4. AV_PKT_DATA_SPHERICAL — half-equirectangular (VR180).
        // `AVSphericalMapping` is in libavutil/spherical.h but not
        // exposed by ffmpeg-sys-next 8.1's bindgen header set. The
        // C ABI is stable, so we re-declare it locally — see the
        // module note above (`MAVSphericalMapping` etc.).
        unsafe {
            let mut sph = MAVSphericalMapping {
                projection: M_AV_SPHERICAL_HALF_EQUIRECTANGULAR,
                yaw: 0, pitch: 0, roll: 0,
                bound_left: 0, bound_top: 0, bound_right: 0, bound_bottom: 0,
                padding: 0,
            };
            // Half-equirect for VR180. FFmpeg movenc.c maps this to
            // prji.projection_kind = 'hequ' (the APMP code).

            let entry = av_packet_side_data_new(
                &mut (*codecpar).coded_side_data,
                &mut (*codecpar).nb_coded_side_data,
                AVPacketSideDataType::AV_PKT_DATA_SPHERICAL,
                std::mem::size_of::<MAVSphericalMapping>(),
                0,
            );
            if entry.is_null() {
                return Err(Error::Ffmpeg(
                    "av_packet_side_data_new SPHERICAL returned NULL".into()
                ));
            }
            std::ptr::copy_nonoverlapping(
                &mut sph as *mut _ as *const u8,
                (*entry).data,
                std::mem::size_of::<MAVSphericalMapping>(),
            );
        }

        tracing::info!("APMP VR180 SBS tagging applied to stream {}", self.stream_index);
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

// ===== Phase 0.7.5.6: zero-copy VT encode constructor + FFI =====

#[cfg(target_os = "macos")]
impl H265Encoder {
    /// Create an `hevc_videotoolbox` encoder that accepts AVFrames
    /// with `format = AV_PIX_FMT_VIDEOTOOLBOX` and `data[3] =
    /// CVPixelBufferRef`. Use `encode_pixel_buffer` to feed it.
    ///
    /// Internally:
    /// - Creates a VideoToolbox hardware device context
    ///   (`av_hwdevice_ctx_create`).
    /// - Creates a hwframes context (`av_hwframe_ctx_alloc` +
    ///   `av_hwframe_ctx_init`) with `format=VIDEOTOOLBOX` /
    ///   `sw_format=BGRA` matching the IOSurfaces we'll produce.
    /// - Configures the codec with `pix_fmt=VIDEOTOOLBOX`,
    ///   `hw_device_ctx`, `hw_frames_ctx`, plus the same VT private
    ///   options as the CPU-input VT path (realtime=0, allow_sw=0,
    ///   profile=main, power_efficient=0).
    pub fn create_zero_copy_vt(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        Self::create_zero_copy_vt_with_format(path, w, h, fps, bitrate_kbps, 8)
    }

    /// P010-input zero-copy: same as `create_zero_copy_vt` but the
    /// VT encoder is configured with `sw_format = P010LE` and
    /// `profile = main10` so the IOSurface VT reads is a 10-bit
    /// semi-planar YUV buffer (Y plane + UV plane). Pair with
    /// [`crate::interop_macos::create_p010_encode_buffer`] +
    /// `encode_pixel_buffer_p010`.
    pub fn create_zero_copy_vt_p010(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        Self::create_zero_copy_vt_with_format(path, w, h, fps, bitrate_kbps, 10)
    }

    fn create_zero_copy_vt_with_format(
        path: &Path,
        w: u32, h: u32,
        fps: f32,
        bitrate_kbps: u32,
        bit_depth: u8,
    ) -> Result<Self> {
        crate::decode::init();
        use ffmpeg::ffi::*;

        let mut octx = ffmpeg::format::output(&path)
            .map_err(|e| Error::Ffmpeg(format!("output ctx {path:?}: {e}")))?;

        let codec = ffmpeg::codec::encoder::find_by_name("hevc_videotoolbox")
            .ok_or_else(|| Error::Ffmpeg(
                "hevc_videotoolbox encoder not available in this FFmpeg build".into()
            ))?;

        let stream_index = {
            let stream = octx.add_stream(codec)
                .map_err(|e| Error::Ffmpeg(format!("add_stream: {e}")))?;
            stream.index()
        };

        let (num, den) = approx_rational(fps);
        let time_base = ffmpeg::Rational(den, num);

        let mut enc_ctx = ffmpeg::codec::context::Context::new_with_codec(codec);

        // hw_device_ctx — VideoToolbox device.
        let hw_device: *mut AVBufferRef = unsafe {
            let mut d: *mut AVBufferRef = std::ptr::null_mut();
            let ret = av_hwdevice_ctx_create(
                &mut d,
                AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                std::ptr::null(), std::ptr::null_mut(), 0,
            );
            if ret < 0 || d.is_null() {
                return Err(Error::Ffmpeg(format!(
                    "av_hwdevice_ctx_create VIDEOTOOLBOX (encode) failed: ret={ret}"
                )));
            }
            d
        };

        // hw_frames_ctx — pool descriptor (we don't actually use the
        // pool since we bring our own CVPixelBuffers, but the encoder
        // requires this to be set so it knows the input format).
        let hw_frames: *mut AVBufferRef = unsafe {
            let f = av_hwframe_ctx_alloc(hw_device);
            if f.is_null() {
                av_buffer_unref(&mut (hw_device as *mut _));
                return Err(Error::Ffmpeg("av_hwframe_ctx_alloc failed".into()));
            }
            let frames_ctx = (*f).data as *mut AVHWFramesContext;
            (*frames_ctx).format    = AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX;
            (*frames_ctx).sw_format = if bit_depth == 10 {
                AVPixelFormat::AV_PIX_FMT_P010LE
            } else {
                AVPixelFormat::AV_PIX_FMT_BGRA
            };
            (*frames_ctx).width     = w as i32;
            (*frames_ctx).height    = h as i32;
            // initial_pool_size=0 → no pre-allocation; we feed our own
            // CVPixelBuffers per frame.
            (*frames_ctx).initial_pool_size = 0;
            let ret = av_hwframe_ctx_init(f);
            if ret < 0 {
                av_buffer_unref(&mut (f as *mut _));
                av_buffer_unref(&mut (hw_device as *mut _));
                return Err(Error::Ffmpeg(format!(
                    "av_hwframe_ctx_init failed: ret={ret}"
                )));
            }
            f
        };

        unsafe {
            let raw = enc_ctx.as_mut_ptr();
            (*raw).width = w as i32;
            (*raw).height = h as i32;
            (*raw).pix_fmt = AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX;
            (*raw).time_base = time_base.into();
            (*raw).framerate = ffmpeg::Rational(num, den).into();
            (*raw).bit_rate = (bitrate_kbps as i64) * 1000;
            (*raw).gop_size = 60;
            (*raw).codec_tag = u32::from_le_bytes(*b"hvc1");
            if octx.format().flags()
                .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER)
            {
                (*raw).flags |= AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }
            (*raw).hw_device_ctx = hw_device;
            (*raw).hw_frames_ctx = hw_frames;

            // Colorimetry tags — these flow into the HEVC bitstream's
            // VUI (video usability info) and the mp4's colr atom so
            // players know how to inverse-transform our YCbCr back to
            // RGB. Critical on the P010 zero-copy path: we hand VT
            // pre-computed video-range Rec.709 YCbCr from our shader,
            // so the output MUST be tagged that way or players will
            // guess wrong (full-range interpretation → lifted blacks;
            // Rec.601 inverse on Rec.709 data → off saturation).
            (*raw).color_range     = AVColorRange::AVCOL_RANGE_MPEG;       // video range
            (*raw).color_primaries = AVColorPrimaries::AVCOL_PRI_BT709;
            (*raw).color_trc       = AVColorTransferCharacteristic::AVCOL_TRC_BT709;
            (*raw).colorspace      = AVColorSpace::AVCOL_SPC_BT709;

            // Same VT private-data tuning as the CPU-input path.
            set_opt(raw, "realtime",        "0")?;
            set_opt(raw, "allow_sw",        "0")?;
            set_opt(raw, "profile", if bit_depth == 10 { "main10" } else { "main" })?;
            set_opt(raw, "power_efficient", "0")?;
        }

        let encoder = enc_ctx.encoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video encoder context: {e}")))?
            .open_as(codec)
            .map_err(|e| Error::Ffmpeg(format!("open hevc_videotoolbox (zero-copy): {e}")))?;

        {
            let mut stream = octx.stream_mut(stream_index)
                .ok_or_else(|| Error::Ffmpeg("output stream vanished".into()))?;
            stream.set_parameters(&encoder);
            stream.set_time_base(time_base);
        }

        // Placeholder scaler — never used on this path. We still need
        // *some* Context::get to satisfy the field type; 1×1 RGB→YUV
        // is the cheapest possible.
        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24, 1, 1,
            ffmpeg::format::Pixel::YUV420P, 1, 1,
            ffmpeg::software::scaling::Flags::BICUBIC,
        ).map_err(|e| Error::Ffmpeg(format!("placeholder scaler: {e}")))?;

        Ok(Self {
            octx, encoder, scaler, stream_index, time_base,
            frame_count: 0, w, h, finished: false,
            encode_path: EncodePath::ZeroCopyVt,
            header_written: false,
            // Zero-copy VT path doesn't use the scaler — the encoder's
            // pix_fmt is AV_PIX_FMT_VIDEOTOOLBOX, the swscale field is
            // a 1×1 placeholder. Set this to YUV420P so any incidental
            // use of `enc_pix_fmt` does something sensible.
            enc_pix_fmt: ffmpeg::format::Pixel::YUV420P,
            scaler_rgb48: None,
        })
    }
}

/// CFRetain stub linked into encode.rs without pulling in the full
/// interop_macos module's FFI declarations. CoreFoundation is already
/// linked by other modules; the symbol resolves at link time.
#[cfg(target_os = "macos")]
unsafe fn cf_retain(p: *const std::ffi::c_void) {
    extern "C" {
        fn CFRetain(cf: *const std::ffi::c_void) -> *const std::ffi::c_void;
    }
    let _ = CFRetain(p);
}

#[cfg(target_os = "macos")]
unsafe fn cf_release(p: *const std::ffi::c_void) {
    extern "C" {
        fn CFRelease(cf: *const std::ffi::c_void);
    }
    CFRelease(p);
}

/// `av_buffer_create` callback: release the CVPixelBuffer retain we
/// took in `encode_pixel_buffer`. Called when the AVBufferRef's
/// refcount drops to zero (typically when the encoder is done with
/// the frame).
///
/// `_data` is the FFmpeg-side buffer (null because we passed null to
/// av_buffer_create); `opaque` is the CVPixelBufferRef we stashed.
#[cfg(target_os = "macos")]
unsafe extern "C" fn cv_pixel_buffer_release_callback(
    opaque: *mut std::ffi::c_void,
    _data: *mut u8,
) {
    if !opaque.is_null() {
        cf_release(opaque as *const _);
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
