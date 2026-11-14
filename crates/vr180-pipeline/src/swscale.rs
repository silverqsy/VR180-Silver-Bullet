//! swscale as this crate uses it for colour: Y'CbCr ↔ R'G'B' between a
//! decoder or an encoder and the GPU.
//!
//! Three things swscale does on its own are wrong for that, each measured
//! on the build the app ships:
//!
//! - **The matrix.** A context converts with BT.601 unless it is told
//!   otherwise through `sws_setColorspaceDetails`; `Context::run` and
//!   `sws_scale_frame` read the matrix from the context, never from the
//!   frames' tags. Every stream here is Rec.709 (BT.2020 for an HLG clip).
//! - **The 16-bit scale.** swscale's 16-bit RGB is its 8-bit RGB shifted
//!   up: white is 255 << 8 = 65280. The GPU reads the same words as
//!   unorm16, white = 65535. Decoding, a contrast of 257 / 256 puts white
//!   on 65535; encoding, the picture is scaled by 65280 / 65535 on its way
//!   into the frame (`encode::load_rgb16`).
//! - **The 8-bit output.** The fast converters to 8-bit RGB (the table
//!   ones: anything from a 10-bit source, anything into RGB24, anything
//!   with a width off a multiple of 16) come out about 2 of 255 dark on
//!   every level above black. `BICUBIC | FULL_CHR_H_INT | ACCURATE_RND`
//!   is exact for every source and layout, and slow on one thread (50 ms
//!   for a 3840² frame against 2 to 23): hence the threaded context.

use crate::{Error, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg::format::Pixel;
use std::ffi::CString;

/// `SWS_BICUBIC | SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND`: the one flag set
/// whose Y'CbCr → RGB conversion is exact for every source and output.
/// (The #defines are not re-exported by ffmpeg-sys-next.)
pub(crate) const SWS_EXACT: i64 = 4 | 0x2000 | 0x40000;

/// A slice-threaded swscale context. `sws_getContext()` (what ffmpeg-next's
/// `scaling::Context::get` wraps) has no threads parameter: libswscale
/// threading is only reachable by building the context through AVOptions
/// before init, which is what the ffmpeg CLI does. Slice threading splits
/// the frame into row bands; the per-pixel arithmetic is the same.
pub(crate) struct ThreadedScaler {
    ctx: *mut ffmpeg::ffi::SwsContext,
}

// A context is only ever used from the one thread that owns the decoder or
// the encoder it belongs to; Send is needed because those move across threads.
unsafe impl Send for ThreadedScaler {}

impl ThreadedScaler {
    /// `flags`: `SWS_*` bits (4 = BICUBIC). `None` when swscale refuses the
    /// conversion or the options: the caller falls back to a plain context.
    pub(crate) fn build(
        src: Pixel, src_w: u32, src_h: u32,
        dst: Pixel, dst_w: u32, dst_h: u32,
        flags: i64,
    ) -> Option<Self> {
        unsafe {
            use ffmpeg::ffi::*;
            let ctx = sws_alloc_context();
            if ctx.is_null() {
                return None;
            }
            let obj = ctx as *mut std::ffi::c_void;
            let set = |name: &str, val: i64| -> bool {
                let cname = CString::new(name).unwrap();
                av_opt_set_int(obj, cname.as_ptr(), val, 0) >= 0
            };
            let threads = std::thread::available_parallelism()
                .map(|n| n.get().min(16) as i64)
                .unwrap_or(1);
            let src_fmt: AVPixelFormat = src.into();
            let dst_fmt: AVPixelFormat = dst.into();
            let ok = set("srcw", src_w as i64)
                && set("srch", src_h as i64)
                && set("src_format", src_fmt as i64)
                && set("dstw", dst_w as i64)
                && set("dsth", dst_h as i64)
                && set("dst_format", dst_fmt as i64)
                && set("sws_flags", flags)
                && set("threads", threads);
            if !ok || sws_init_context(ctx, std::ptr::null_mut(), std::ptr::null_mut()) < 0 {
                sws_freeContext(ctx);
                tracing::warn!("threaded swscale init failed ({src:?} → {dst:?}) — falling back to single-threaded");
                return None;
            }
            tracing::info!("threaded swscale: {threads} threads for {src:?} {src_w}×{src_h} → {dst:?} {dst_w}×{dst_h}");
            Some(Self { ctx })
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffmpeg::ffi::SwsContext {
        self.ctx
    }

    /// Convert `src` into `dst` (allocated here when it is empty). Returns
    /// false on failure.
    pub(crate) fn run(&mut self, src: &ffmpeg::frame::Video, dst: &mut ffmpeg::frame::Video) -> bool {
        unsafe {
            // MUST be sws_scale_frame(): the legacy sws_scale() entry point
            // detects a threads>1 context and silently converts on
            // slice_ctx[0], single-threaded. Only sws_scale_frame()
            // dispatches slices across the thread pool.
            ffmpeg::ffi::sws_scale_frame(self.ctx, dst.as_mut_ptr(), src.as_ptr()) >= 0
        }
    }
}

impl Drop for ThreadedScaler {
    fn drop(&mut self) {
        unsafe { ffmpeg::ffi::sws_freeContext(self.ctx); }
    }
}

/// Tell a Y'CbCr → RGB context its source: BT.2020 (NCL) for a stream
/// tagged so, Rec.709 otherwise (the same choice the GPU kernels make
/// through `yuv_matrix_constants`); video range unless `full_range`. The
/// RGB out is full range, and for a 16-bit output on the GPU's scale
/// (white = 65535, see the module notes).
///
/// # Safety
/// `ctx` is a live, initialised swscale context.
pub(crate) unsafe fn set_yuv_source(
    ctx: *mut ffmpeg::ffi::SwsContext,
    is_16bit: bool,
    full_range: bool,
    bt2020: bool,
) {
    use ffmpeg::ffi::*;
    // sws_getCoefficients returns the Y'CbCr matrix coefficients for a
    // colourspace constant: SWS_CS_ITU709 = 1, SWS_CS_BT2020 = 9.
    const SWS_CS_ITU709: std::os::raw::c_int = 1;
    const SWS_CS_BT2020: std::os::raw::c_int = 9;
    let src_coeffs = sws_getCoefficients(if bt2020 { SWS_CS_BT2020 } else { SWS_CS_ITU709 });
    let dst_coeffs = sws_getCoefficients(SWS_CS_ITU709);
    // The contrast term is a gain on the whole conversion: 257 / 256 of it
    // puts a 16-bit white on 65535 (255 × 257) where swscale's own scale
    // leaves it on 65280.
    let contrast = if is_16bit { (1 << 16) * 257 / 256 } else { 1 << 16 };
    // (ctx, inverse table, source range, table, destination range,
    //  brightness, contrast, saturation); ranges: 0 = video, 1 = full.
    let _ = sws_setColorspaceDetails(
        ctx,
        src_coeffs, if full_range { 1 } else { 0 },
        dst_coeffs, 1,
        0, contrast, 1 << 16,
    );
}

/// Give an RGB → Y'CbCr context the matrix this app's streams are tagged
/// with: full-range R'G'B' in, Rec.709 at video range out. Left alone it
/// converts with BT.601, and a player that believes the stream's BT.709 tag
/// shows every saturated colour off (a 75 % red 4 % of the range too bright
/// in luma; greys are the same under both). Covers a threaded context too:
/// swscale hands the call on to each of its slice contexts.
///
/// # Safety
/// `ctx` is a live, initialised swscale context.
pub(crate) unsafe fn set_rgb_to_rec709(ctx: *mut ffmpeg::ffi::SwsContext) {
    const SWS_CS_ITU709: std::os::raw::c_int = 1;
    let coeffs = ffmpeg::ffi::sws_getCoefficients(SWS_CS_ITU709);
    let ret = ffmpeg::ffi::sws_setColorspaceDetails(ctx, coeffs, 1, coeffs, 0, 0, 1 << 16, 1 << 16);
    if ret < 0 {
        tracing::warn!("swscale refused the Rec.709 matrix ({ret}): the RGB → YUV conversion stays BT.601");
    }
}

/// A decoded frame's conversion to packed RGB for the GPU.
pub(crate) enum YuvToRgb {
    /// The exact conversion, slice-threaded: every same-size conversion.
    Threaded(ThreadedScaler),
    /// A plain context: a conversion that also scales (the working-size
    /// preview, where speed is the point), or the fallback when the
    /// threaded one cannot be built.
    Plain(ffmpeg::software::scaling::Context),
}

impl YuvToRgb {
    /// `dst` = `RGBA`, `RGB24` or `RGBA64LE`. A same-size conversion is the
    /// exact one; a scaling one keeps `scaled_flags` (see [`Self::Plain`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        src: Pixel, src_w: u32, src_h: u32,
        dst: Pixel, dst_w: u32, dst_h: u32,
        scaled_flags: ffmpeg::software::scaling::Flags,
        full_range: bool,
        bt2020: bool,
    ) -> Result<Self> {
        let is_16bit = dst == Pixel::RGBA64LE;
        let same_size = (src_w, src_h) == (dst_w, dst_h);
        if same_size {
            if let Some(mut t) = ThreadedScaler::build(src, src_w, src_h, dst, dst_w, dst_h, SWS_EXACT) {
                unsafe { set_yuv_source(t.as_mut_ptr(), is_16bit, full_range, bt2020); }
                return Ok(Self::Threaded(t));
            }
        }
        let flags = if same_size {
            ffmpeg::software::scaling::Flags::BICUBIC
                | ffmpeg::software::scaling::Flags::FULL_CHR_H_INT
                | ffmpeg::software::scaling::Flags::ACCURATE_RND
        } else {
            scaled_flags
        };
        let mut plain = ffmpeg::software::scaling::Context::get(src, src_w, src_h, dst, dst_w, dst_h, flags)
            .map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?;
        unsafe { set_yuv_source(plain.as_mut_ptr(), is_16bit, full_range, bt2020); }
        Ok(Self::Plain(plain))
    }

    pub(crate) fn run(&mut self, src: &ffmpeg::frame::Video, dst: &mut ffmpeg::frame::Video) -> Result<()> {
        match self {
            Self::Threaded(t) => {
                if t.run(src, dst) { Ok(()) } else { Err(Error::Ffmpeg("scale: swscale refused the frame".into())) }
            }
            Self::Plain(p) => p.run(src, dst).map_err(|e| Error::Ffmpeg(format!("scale: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat frame of one video-range colour, given in 8-bit codes, in any
    /// of the layouts a decoder hands over.
    fn flat(fmt: Pixel, w: u32, h: u32, y: u16, cb: u16, cr: u16) -> ffmpeg::frame::Video {
        let mut f = ffmpeg::frame::Video::new(fmt, w, h);
        let wide = matches!(fmt, Pixel::YUV420P10LE | Pixel::P010LE);
        let shift = match fmt { Pixel::YUV420P10LE => 2, Pixel::P010LE => 8, _ => 0 };
        let semi = matches!(fmt, Pixel::NV12 | Pixel::P010LE);
        for plane in 0..f.planes() {
            let d = f.data_mut(plane);
            let put = |d: &mut [u8], i: usize, v: u16| {
                let v = v << shift;
                if wide { d[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes()); } else { d[i] = v as u8; }
            };
            let n = d.len() / if wide { 2 } else { 1 };
            for i in 0..n {
                let v = match (plane, semi) {
                    (0, _) => y,
                    (1, true) => if i % 2 == 0 { cb } else { cr },
                    (1, false) => cb,
                    _ => cr,
                };
                put(d, i, v);
            }
        }
        f
    }

    /// R'G'B' (0..1) of a video-range triple in 8-bit codes under Rec.709.
    fn rec709(y: f64, cb: f64, cr: f64) -> [f64; 3] {
        let (yn, pb, pr) = ((y - 16.0) / 219.0, (cb - 128.0) / 224.0, (cr - 128.0) / 224.0);
        let r = yn + 1.5748 * pr;
        let b = yn + 1.8556 * pb;
        [r, (yn - 0.2126 * r - 0.0722 * b) / 0.7152, b]
    }

    /// Every same-size conversion a decoder asks for is exact: each source
    /// layout into each packed RGB, at a width swscale has a fast converter
    /// for (a multiple of 16) and at one it has not.
    #[test]
    fn same_size_conversions_are_exact() {
        crate::decode::init();
        // (Y, Cb, Cr): black, white, greys, a red, a green, a sky blue.
        let colours = [(16u16, 128u16, 128u16), (235, 128, 128), (40, 128, 128), (126, 128, 128), (90, 100, 200), (150, 70, 60), (200, 150, 110)];
        for src in [Pixel::YUV420P, Pixel::NV12, Pixel::YUV420P10LE, Pixel::P010LE] {
            for (dst, bytes, scale, bound) in [(Pixel::RGBA, 1usize, 255.0, 0.6), (Pixel::RGB24, 1, 255.0, 0.6), (Pixel::RGBA64LE, 2, 65535.0, 40.0)] {
                for (w, h) in [(64u32, 32u32), (40, 24)] {
                    let mut conv = YuvToRgb::new(src, w, h, dst, w, h, ffmpeg::software::scaling::Flags::FAST_BILINEAR, false, false).expect("converter");
                    assert!(matches!(conv, YuvToRgb::Threaded(_)), "{src:?} → {dst:?}: the threaded context was not built");
                    let mut worst = 0.0f64;
                    for (y, cb, cr) in colours {
                        let frame = flat(src, w, h, y, cb, cr);
                        let mut out = ffmpeg::frame::Video::empty();
                        conv.run(&frame, &mut out).expect("run");
                        let want = rec709(y as f64, cb as f64, cr as f64);
                        let px = if dst == Pixel::RGB24 { 3 } else { 4 } * bytes;
                        let o = (h as usize / 2) * out.stride(0) + (w as usize / 2) * px;
                        let d = out.data(0);
                        for ch in 0..3 {
                            let v = if bytes == 2 { u16::from_le_bytes([d[o + ch * 2], d[o + ch * 2 + 1]]) as f64 } else { d[o + ch] as f64 };
                            worst = worst.max((v - want[ch].clamp(0.0, 1.0) * scale).abs());
                        }
                    }
                    assert!(worst <= bound, "{src:?} → {dst:?} at {w}×{h}: {worst:.2} from Rec.709 (of {scale})");
                }
            }
        }
    }

    /// A full-range source is read as one, and a BT.2020 source with its
    /// own matrix.
    #[test]
    fn range_and_matrix_follow_the_source() {
        crate::decode::init();
        let (w, h) = (32u32, 16u32);
        // Full range: Y 128 of 255 is a 50.2 % grey.
        let mut conv = YuvToRgb::new(Pixel::YUV420P, w, h, Pixel::RGBA, w, h, ffmpeg::software::scaling::Flags::FAST_BILINEAR, true, false).expect("converter");
        let mut out = ffmpeg::frame::Video::empty();
        conv.run(&flat(Pixel::YUV420P, w, h, 128, 128, 128), &mut out).expect("run");
        let o = 8 * out.stride(0) + 16 * 4;
        assert!((out.data(0)[o] as i32 - 128).abs() <= 1, "full range grey came out {}", out.data(0)[o]);
        // BT.2020: R' = Y' + 1.4746 Pr.
        let mut conv = YuvToRgb::new(Pixel::YUV420P, w, h, Pixel::RGBA, w, h, ffmpeg::software::scaling::Flags::FAST_BILINEAR, false, true).expect("converter");
        let mut out = ffmpeg::frame::Video::empty();
        conv.run(&flat(Pixel::YUV420P, w, h, 126, 128, 180), &mut out).expect("run");
        let want = ((126.0 - 16.0) / 219.0 + 1.4746 * (180.0 - 128.0) / 224.0) * 255.0;
        let got = out.data(0)[8 * out.stride(0) + 16 * 4] as f64;
        assert!((got - want).abs() <= 1.0, "BT.2020 red came out {got}, exact {want:.1}");
    }
}
