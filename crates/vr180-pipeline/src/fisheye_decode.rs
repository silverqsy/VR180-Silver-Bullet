//! Fisheye-source decoders — SBS `.mp4`, dual-stream OSV, BRAW.
//!
//! Counterpart to `decode.rs`'s GoPro EAC-pair iterators. Each
//! concrete iterator emits [`FisheyePair`]s — one frame of left-eye
//! and right-eye fisheye image, packed RGBA8. The downstream
//! pipeline doesn't care which kind of camera produced the pair.
//!
//! ## Layouts
//!
//! - [`SbsFisheyeIter`] — one HEVC video stream, each frame split
//!   horizontally into L/R halves. Matches the Python "sbs" path at
//!   `vr180_gui.py:3563-3631`.
//! - [`DualStreamFisheyeIter`] — two video streams in one container
//!   (DJI OSV, custom dual-camera rigs). Each frame is one full eye.
//!   Matches the Python "dual_stream" path at
//!   `vr180_gui.py:3551-3560`.
//! - [`BrawFisheyeIter`] — Blackmagic RAW via the `braw_helper`
//!   subprocess. The helper auto-composites multi-track stereo (URSA
//!   Cine Immersive, Pyxis 12K) into SBS BGRA, so this is essentially
//!   "SBS but the bytes come from a pipe instead of ffmpeg". Matches
//!   the Python "braw" path at `vr180_gui.py:3541-3550`.
//!
//! ## Working colour
//!
//! Output is always RGBA8 (4 bytes/pixel). For BRAW we tell
//! `braw_helper` to emit BGRA8 (the BMD SDK's default), and swizzle
//! B↔R on the Rust side. For ffmpeg paths we use the existing
//! BILINEAR scaler set to RGBA.

use crate::decode::{init as ffmpeg_init, HwDecode, DecodePath};
use crate::{Error, Result};
use std::path::Path;

/// One pair of fisheye eyes (left + right) ready for the GPU.
///
/// Both buffers contain packed pixel data; `bit_depth` tells the
/// consumer how to interpret. Default 8 (RGBA8, 4 bytes/pixel — what
/// the preview path uses). 16 = RGBA64LE (8 bytes/pixel, little-endian
/// 16-bit per channel) — used by the true-10-bit export path so
/// projection input is genuinely 10-bit-precise after the source
/// HEVC's P010 → RGB conversion.
#[derive(Debug, Clone)]
pub struct FisheyePair {
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub eye_w: u32,
    pub eye_h: u32,
    /// `8` (RGBA8) or `16` (RGBA64LE).
    pub bit_depth: u8,
    /// Presentation timestamp in seconds, or `0.0` if unknown.
    pub pts_s: f64,
}

/// Common iterator interface for the three fisheye sources. Hand-
/// rolled instead of `Iterator` because `next()` borrows `self` and
/// returns a `Result` that needs to bubble out cleanly.
pub trait FisheyePairIter {
    /// Pull the next L/R fisheye pair. Returns `Ok(None)` at EOF.
    fn next_pair(&mut self) -> Result<Option<FisheyePair>>;

    /// Seek to `target_s` (clip-relative seconds). Subsequent
    /// `next_pair` calls return frames at or after this timestamp.
    /// Implementations call into ffmpeg `seek` + decoder `flush`, or
    /// restart the BRAW helper at a new `--start` index.
    fn seek(&mut self, target_s: f64) -> Result<()>;

    /// Eye dimensions (one fisheye, not the SBS-composed frame).
    fn eye_dims(&self) -> (u32, u32);
}

// ── SBS fisheye (single ffmpeg stream, split horizontally) ─────────

/// SBS fisheye `.mp4` iterator. One ffmpeg video stream, each frame
/// split into left/right halves. Same code path as the Python
/// `_decode_sbs_thread` at `vr180_gui.py:3606-3631`.
pub struct SbsFisheyeIter {
    ictx: ffmpeg_next::format::context::Input,
    video_idx: usize,
    decoder: ffmpeg_next::codec::decoder::Video,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
    eye_w: u32,
    eye_h: u32,
    frame_limit: u32,
    frames_yielded: u32,
    /// Frames-per-second from the stream metadata. Used to convert
    /// PTS ticks to seconds when the packet stream lacks pts info.
    time_base_s: f64,
}

impl std::fmt::Debug for SbsFisheyeIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SbsFisheyeIter")
            .field("video_idx", &self.video_idx)
            .field("eye_w", &self.eye_w)
            .field("eye_h", &self.eye_h)
            .field("frames_yielded", &self.frames_yielded)
            .finish_non_exhaustive()
    }
}

impl SbsFisheyeIter {
    /// Open an SBS fisheye file. `hw` controls VideoToolbox decode
    /// (Auto = use VT on macOS if available, fall back to software).
    pub fn new(path: &Path, _hw: HwDecode, frame_limit: u32) -> Result<Self> {
        ffmpeg_init();
        let ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
        let video = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .max_by_key(|s| {
                // Largest stream wins — protects against random data
                // streams (timecode, etc.).
                let p = s.parameters();
                unsafe {
                    let codec_pars = &*p.as_ptr();
                    (codec_pars.width as u64) * (codec_pars.height as u64)
                }
            })
            .ok_or_else(|| Error::Ffmpeg("no video stream".into()))?;
        let video_idx = video.index();
        let time_base = video.time_base();
        let time_base_s = time_base.numerator() as f64 / time_base.denominator() as f64;

        let codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(video.parameters())
            .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
        let decoder = codec_ctx.decoder().video()
            .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;

        let frame_w = decoder.width();
        let frame_h = decoder.height();
        if frame_w % 2 != 0 {
            return Err(Error::Ffmpeg(format!(
                "SBS frame width {frame_w} must be even for L/R split"
            )));
        }
        let eye_w = frame_w / 2;
        let eye_h = frame_h;

        Ok(Self {
            ictx, video_idx, decoder, scaler: None,
            eye_w, eye_h,
            frame_limit, frames_yielded: 0,
            time_base_s,
        })
    }
}

impl FisheyePairIter for SbsFisheyeIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut sw_storage = ffmpeg_next::frame::Video::empty();

        // Try buffered output first.
        loop {
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                let pts_ticks = decoded.pts().unwrap_or(0);
                let pts_s = pts_ticks as f64 * self.time_base_s;
                let (left, right) = self.repack_split(&mut decoded, &mut sw_storage)?;
                self.frames_yielded += 1;
                return Ok(Some(FisheyePair {
                    left, right,
                    eye_w: self.eye_w, eye_h: self.eye_h,
                    bit_depth: 8,
                    pts_s,
                }));
            }

            // Need more packets.
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() != self.video_idx { continue; }
                    let _ = self.decoder.send_packet(&packet);
                }
                None => {
                    // Drain.
                    let _ = self.decoder.send_eof();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        let pts_ticks = decoded.pts().unwrap_or(0);
                        let pts_s = pts_ticks as f64 * self.time_base_s;
                        let (left, right) = self.repack_split(&mut decoded, &mut sw_storage)?;
                        self.frames_yielded += 1;
                        return Ok(Some(FisheyePair {
                            left, right,
                            eye_w: self.eye_w, eye_h: self.eye_h,
                            bit_depth: 8,
                            pts_s,
                        }));
                    }
                    return Ok(None);
                }
            }
        }
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        let ts = (target_s.max(0.0) * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek {target_s:.3}s: {e}")))?;
        self.decoder.flush();
        self.frames_yielded = 0;
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        (self.eye_w, self.eye_h)
    }
}

impl SbsFisheyeIter {
    /// Scale to RGBA8 and split the frame down the middle. Returns
    /// (left_rgba, right_rgba), each `eye_w × eye_h × 4`.
    fn repack_split(
        &mut self,
        decoded: &mut ffmpeg_next::frame::Video,
        _sw_storage: &mut ffmpeg_next::frame::Video,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let frame_w = decoded.width();
        let frame_h = decoded.height();
        let scaler = match &mut self.scaler {
            Some(s) => s,
            None => {
                self.scaler = Some(
                    ffmpeg_next::software::scaling::Context::get(
                        decoded.format(),
                        frame_w, frame_h,
                        ffmpeg_next::format::Pixel::RGBA,
                        frame_w, frame_h,
                        ffmpeg_next::software::scaling::Flags::FAST_BILINEAR,
                    ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?
                );
                self.scaler.as_mut().unwrap()
            }
        };

        let mut rgba_frame = ffmpeg_next::frame::Video::empty();
        scaler.run(decoded, &mut rgba_frame)
            .map_err(|e| Error::Ffmpeg(format!("scale: {e}")))?;

        // Repack with stride → contiguous, then split L/R.
        let stride = rgba_frame.stride(0);
        let data = rgba_frame.data(0);
        let eye_w = self.eye_w as usize;
        let eye_h = self.eye_h as usize;
        let mut left = Vec::with_capacity(eye_w * eye_h * 4);
        let mut right = Vec::with_capacity(eye_w * eye_h * 4);
        for y in 0..eye_h {
            let row_start = y * stride;
            // Each row is 2*eye_w pixels = 8*eye_w bytes.
            let left_start = row_start;
            let left_end = row_start + eye_w * 4;
            let right_end = row_start + 2 * eye_w * 4;
            left.extend_from_slice(&data[left_start..left_end]);
            right.extend_from_slice(&data[left_end..right_end]);
        }
        Ok((left, right))
    }
}

// ── Dual-stream OSV (two video streams in one container) ───────────

/// DJI OSV (and other dual-camera) iterator. Two video streams,
/// decoded in lockstep, one per eye. Eyes are NOT split — each
/// stream is one full fisheye image.
///
/// Mirrors `_decode_osv_thread` at `vr180_gui.py:3551-3560`. Uses
/// VideoToolbox decode on macOS (the source streams are 10-bit HEVC
/// Main10 at 3840×3840 — software decode is way too slow for
/// real-time preview). The downloaded NV12/P010 frames are
/// scaled-down to at most `MAX_DECODE_SIDE` per axis to keep the
/// CPU→GPU upload bounded.
pub struct DualStreamFisheyeIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: [usize; 2],
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    scalers: Vec<Option<ffmpeg_next::software::scaling::Context>>,
    /// Per-stream: true if VideoToolbox decode is active for this
    /// stream (i.e. we need to download HW frames before scaling).
    hw_active: [bool; 2],
    /// Eye dimensions AFTER scaler downsampling — what
    /// `next_pair` yields to the consumer.
    eye_w: u32,
    eye_h: u32,
    /// Native per-stream dimensions (decoder.width()/height()).
    /// Used by the scaler setup; the consumer doesn't see these.
    native_w: u32,
    native_h: u32,
    /// Output bit depth (8 = RGBA8, 16 = RGBA64LE). The scaler is
    /// configured for this on first frame.
    output_bit_depth: u8,
    frame_limit: u32,
    frames_yielded: u32,
    time_base_s: f64,
    /// If true, output left = stream[1], right = stream[0]. Default
    /// false: left = stream[0]. The Python OSV path swaps based on
    /// `cfg.swap_eyes` (`vr180_gui.py:3554`).
    pub swap_eyes: bool,
}

/// Per-axis cap on the working fisheye resolution after CPU
/// downsampling. 3840-pixel-square OSV / Pyxis sources are decoded
/// at full resolution by VT (zero CPU cost) but THEN the scaler
/// downsamples to this cap before the CPU→GPU upload — 1280² × RGBA
/// = ~6.25 MiB per eye, vs 3840² × RGBA = ~60 MiB per eye.
///
/// 1280 is still 1.67× the default preview eye width (768) so
/// projection quality stays good. Bumping the preview eye width
/// past 1280 will start losing detail; the eventual fix is to make
/// this scale with preview_eye_w. Lower bound is "real-time decode
/// throughput on M-series" — measured ~30 fps at 1280, ~23 fps at
/// 1920 on M5 Max for 3840×3840 10-bit HEVC × 2 streams via VT.
const MAX_DECODE_SIDE: u32 = 1280;

/// FPS-aware working-resolution cap. The 1280 baseline was tuned at
/// 30 fps — VT throughput at 3840×3840 dual-stream HEVC peaks around
/// 30 fps at that working res. At higher fps the per-frame decode
/// budget shrinks (20 ms at 50 fps vs 33 ms at 30 fps), so we step
/// down the working res to keep decode within budget.
///
/// Roughly `floor(1280 · √(30 / fps))`, snapped to a multiple of 32
/// for shader-friendly dispatch sizes.
pub fn max_decode_side_for_fps(fps: f32) -> u32 {
    if fps <= 30.5 {
        return MAX_DECODE_SIDE;
    }
    let scaled = (MAX_DECODE_SIDE as f32) * (30.0 / fps).sqrt();
    let side = scaled as u32;
    let snapped = side & !31u32; // round down to multiple of 32
    snapped.max(384)
}

fn clamp_decode_dim(w: u32, h: u32) -> (u32, u32) {
    clamp_decode_dim_to(w, h, MAX_DECODE_SIDE)
}

fn clamp_decode_dim_to(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let s = (max_side as f32 / w.max(h) as f32).min(1.0);
    if s >= 1.0 { (w, h) }
    else {
        // Even dims for any downstream that prefers them.
        let cw = (((w as f32) * s) as u32 + 1) & !1u32;
        let ch = (((h as f32) * s) as u32 + 1) & !1u32;
        (cw.max(2), ch.max(2))
    }
}

impl std::fmt::Debug for DualStreamFisheyeIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DualStreamFisheyeIter")
            .field("video_indices", &self.video_indices)
            .field("eye_w", &self.eye_w)
            .field("eye_h", &self.eye_h)
            .field("swap_eyes", &self.swap_eyes)
            .field("frames_yielded", &self.frames_yielded)
            .finish_non_exhaustive()
    }
}

impl DualStreamFisheyeIter {
    /// Open a dual-stream container with the default eye order
    /// (left = stream 0). For DJI OSV use `new_with_swap(path, _, _, true)` —
    /// the OSV convention is the reverse (stream 0 = Lens A = right
    /// eye, stream 1 = Lens B = left eye; see `vr180_gui.py:6149`).
    pub fn new(path: &Path, hw: HwDecode, frame_limit: u32) -> Result<Self> {
        Self::new_with_options(path, hw, frame_limit, false, MAX_DECODE_SIDE, 8)
    }

    pub fn new_with_swap(path: &Path, hw: HwDecode, frame_limit: u32, swap_eyes: bool) -> Result<Self> {
        Self::new_with_options(path, hw, frame_limit, swap_eyes, MAX_DECODE_SIDE, 8)
    }

    /// Full options constructor.
    /// - `max_decode_side = 0` → no clamp (full native resolution).
    /// - `output_bit_depth` = 8 (RGBA8) or 16 (RGBA64LE). Export uses
    ///   16 to preserve 10-bit P010 source precision through the
    ///   scaler instead of quantizing to 8.
    pub fn new_with_options(
        path: &Path,
        hw: HwDecode,
        frame_limit: u32,
        swap_eyes: bool,
        max_decode_side: u32,
        output_bit_depth: u8,
    ) -> Result<Self> {
        if output_bit_depth != 8 && output_bit_depth != 16 {
            return Err(Error::Ffmpeg(format!(
                "DualStreamFisheyeIter: bit_depth must be 8 or 16, got {output_bit_depth}"
            )));
        }
        ffmpeg_init();
        let ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
        let video_indices: Vec<usize> = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .map(|s| s.index())
            .take(2)
            .collect();
        if video_indices.len() < 2 {
            return Err(Error::Ffmpeg(format!(
                "expected 2 video streams (OSV / dual-camera), found {}",
                video_indices.len()
            )));
        }
        let video_indices = [video_indices[0], video_indices[1]];
        let mut decoders = Vec::with_capacity(2);
        let mut hw_active = [false, false];
        for (i, &idx) in video_indices.iter().enumerate() {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            // VideoToolbox decode is critical at OSV resolution (3840×3840
            // 10-bit HEVC × 2 streams). Software decode is roughly an
            // order of magnitude too slow for real-time preview.
            #[cfg(target_os = "macos")]
            {
                let want_vt = matches!(hw, HwDecode::Auto | HwDecode::VideoToolbox);
                if want_vt {
                    if crate::decode::try_enable_videotoolbox_decode(&mut codec_ctx) {
                        hw_active[i] = true;
                    } else if matches!(hw, HwDecode::VideoToolbox) {
                        return Err(Error::Ffmpeg(
                            "VideoToolbox requested but unavailable".into()
                        ));
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (i, hw);
                if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg(
                        "VideoToolbox is macOS-only".into()
                    ));
                }
            }
            let dec = codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?;
            decoders.push(dec);
        }
        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "OSV streams disagree on dims: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let (eye_w, eye_h) = if max_decode_side == 0 {
            (w0, h0)
        } else {
            clamp_decode_dim_to(w0, h0, max_decode_side)
        };
        tracing::info!(
            "DualStreamFisheyeIter: native {}x{}, VT={}/{}, working {}x{} (cap={})",
            w0, h0, hw_active[0], hw_active[1], eye_w, eye_h,
            if max_decode_side == 0 { "off".into() } else { max_decode_side.to_string() }
        );

        let time_base = ictx.stream(video_indices[0]).unwrap().time_base();
        let time_base_s = time_base.numerator() as f64 / time_base.denominator() as f64;

        Ok(Self {
            ictx, video_indices, decoders,
            scalers: vec![None, None],
            hw_active,
            eye_w, eye_h,
            native_w: w0, native_h: h0,
            output_bit_depth,
            frame_limit, frames_yielded: 0,
            time_base_s,
            swap_eyes,
        })
    }
}

impl FisheyePairIter for DualStreamFisheyeIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut frames: [Option<(Vec<u8>, i64)>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut sw_storage = ffmpeg_next::frame::Video::empty();

        for pos in 0..2 {
            if frames[pos].is_some() { continue; }
            let dec = &mut self.decoders[pos];
            if dec.receive_frame(&mut decoded).is_ok() {
                let pts = decoded.pts().unwrap_or(0);
                let rgba = self.scale_one(pos, &mut decoded, &mut sw_storage)?;
                frames[pos] = Some((rgba, pts));
            }
        }

        if frames.iter().any(|f| f.is_none()) {
            loop {
                let res = self.ictx.packets().next();
                let (stream, packet) = match res {
                    Some(x) => x,
                    None => break,
                };
                let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                    Some(p) => p,
                    None => continue,
                };
                let dec = &mut self.decoders[pos];
                if dec.send_packet(&packet).is_err() { continue; }
                if frames[pos].is_none()
                    && dec.receive_frame(&mut decoded).is_ok()
                {
                    let pts = decoded.pts().unwrap_or(0);
                    let rgba = self.scale_one(pos, &mut decoded, &mut sw_storage)?;
                    frames[pos] = Some((rgba, pts));
                }
                if frames.iter().all(|f| f.is_some()) { break; }
            }
        }

        let (Some((f0, pts0)), Some((f1, _pts1))) = (frames[0].take(), frames[1].take()) else {
            return Ok(None);
        };
        let pts_s = pts0 as f64 * self.time_base_s;
        let (left, right) = if self.swap_eyes { (f1, f0) } else { (f0, f1) };
        self.frames_yielded += 1;
        Ok(Some(FisheyePair {
            left, right,
            eye_w: self.eye_w, eye_h: self.eye_h,
            bit_depth: self.output_bit_depth,
            pts_s,
        }))
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        let ts = (target_s.max(0.0) * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek {target_s:.3}s: {e}")))?;
        for d in &mut self.decoders {
            d.flush();
        }
        self.frames_yielded = 0;
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        (self.eye_w, self.eye_h)
    }
}

impl DualStreamFisheyeIter {
    /// Download HW frame (if VT-active), build/reuse the lazy scaler
    /// (sized at self.eye_w × self.eye_h — the clamped working
    /// dims), and pack the result as RGBA8. Shared between the
    /// receive-buffered branch and the packet-driven branch in
    /// `next_pair`.
    fn scale_one(
        &mut self,
        pos: usize,
        decoded: &mut ffmpeg_next::frame::Video,
        sw_storage: &mut ffmpeg_next::frame::Video,
    ) -> Result<Vec<u8>> {
        let src_ref: &mut ffmpeg_next::frame::Video = if self.hw_active[pos]
            && decoded.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX
        {
            crate::decode::download_hw_frame(decoded, sw_storage)?;
            sw_storage
        } else {
            decoded
        };

        // Explicitly tag the source AVFrame with Rec.709 video range
        // before swscale picks it up. VT-decoded frames often arrive
        // with colorspace/color_range UNSPECIFIED, in which case
        // swscale's defaults kick in (typically Rec.601 for the YCbCr
        // matrix and full-range for chroma scaling) — both wrong for
        // HD OSV footage and very visible in the 16-bit code path
        // (the 8-bit one is more forgiving). Setting the tags on the
        // SOURCE frame tells swscale exactly how to invert the YCbCr.
        unsafe {
            let raw = src_ref.as_mut_ptr();
            use ffmpeg_next::ffi::*;
            (*raw).colorspace      = AVColorSpace::AVCOL_SPC_BT709;
            (*raw).color_range     = AVColorRange::AVCOL_RANGE_MPEG;
            (*raw).color_primaries = AVColorPrimaries::AVCOL_PRI_BT709;
            (*raw).color_trc       = AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        }

        let is_16bit = self.output_bit_depth == 16;
        let target_pix_fmt = if is_16bit {
            ffmpeg_next::format::Pixel::RGBA64LE
        } else {
            ffmpeg_next::format::Pixel::RGBA
        };
        // For 16-bit color, use BICUBIC — swscale's RGB48/RGBA64 code
        // path is less mature than the 8-bit one and tends to need
        // the full color-conversion accuracy enabled. FAST_BILINEAR
        // skips some intermediate precision steps for speed.
        let scaler_flags = if is_16bit {
            ffmpeg_next::software::scaling::Flags::BICUBIC
                | ffmpeg_next::software::scaling::Flags::FULL_CHR_H_INT
        } else {
            ffmpeg_next::software::scaling::Flags::FAST_BILINEAR
        };
        let scaler = match &mut self.scalers[pos] {
            Some(s) => s,
            None => {
                self.scalers[pos] = Some(
                    ffmpeg_next::software::scaling::Context::get(
                        src_ref.format(),
                        src_ref.width(), src_ref.height(),
                        target_pix_fmt,
                        self.eye_w, self.eye_h,
                        scaler_flags,
                    ).map_err(|e| Error::Ffmpeg(format!("scaler: {e}")))?
                );
                // Tell swscale the exact source / dest colorimetry
                // via sws_setColorspaceDetails. The 16-bit RGB
                // output path otherwise picks Rec.601 for input and
                // arbitrary range for output.
                let s = self.scalers[pos].as_mut().unwrap();
                set_sws_colorspace_details(s, is_16bit);
                s
            }
        };
        let mut rgba_frame = ffmpeg_next::frame::Video::empty();
        scaler.run(src_ref, &mut rgba_frame)
            .map_err(|e| Error::Ffmpeg(format!("scale: {e}")))?;
        let bpp = if is_16bit { 8 } else { 4 };
        Ok(extract_packed_rgba_n(&rgba_frame, bpp))
    }
}

// ── Zero-copy dual-stream (macOS only) ─────────────────────────────
//
// Mirrors the GoPro `ZeroCopyStreamPairIter` in decode.rs, but for the
// fisheye dual-stream layout (OSV / dual-camera). Instead of yielding
// (RGBA8 left, RGBA8 right) Vec<u8>s — which requires
// `av_hwframe_transfer_data` → swscale P010LE → RGBA64LE → CPU→GPU
// upload (~840 MB/frame at native OSV res) — this iterator wraps each
// VideoToolbox-decoded CVPixelBuffer's two planes (Y at full res, UV
// at half res) as wgpu textures via the IOSurface bridge. Downstream
// consumers (export's `project_fisheye_p010_to_equirect_texture_16`)
// read the planes directly and do YCbCr→RGB inline in the projection
// shader.
//
// Used by `fisheye_export.rs` when running 10-bit OSV export on macOS
// with VideoToolbox encode active. Preview still uses
// `DualStreamFisheyeIter` because the egui texture path expects RGBA.

/// One zero-copy fisheye pair — four `IOSurfacePlaneTexture`s
/// (Y + UV for each eye). Hold onto the returned struct while the
/// projection dispatch is in flight; dropping it releases the
/// IOSurface retains.
#[cfg(target_os = "macos")]
pub struct ZeroCopyFisheyePair {
    pub left_y:  crate::interop_macos::IOSurfacePlaneTexture,
    pub left_uv: crate::interop_macos::IOSurfacePlaneTexture,
    pub right_y: crate::interop_macos::IOSurfacePlaneTexture,
    pub right_uv: crate::interop_macos::IOSurfacePlaneTexture,
    /// Native fisheye dims (matches the Y plane). Calibration MUST be
    /// resolved against these.
    pub eye_w: u32,
    pub eye_h: u32,
    /// Presentation timestamp in seconds, `0.0` if unknown.
    pub pts_s: f64,
}

#[cfg(target_os = "macos")]
pub struct ZeroCopyDualStreamFisheyeIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: [usize; 2],
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    /// Native per-stream dimensions (decoder.width()/height()) — also
    /// the Y plane dims of the IOSurface we surface to the consumer.
    eye_w: u32,
    eye_h: u32,
    frame_limit: u32,
    frames_yielded: u32,
    time_base_s: f64,
    /// If true, output left = stream[1], right = stream[0] (matches
    /// DJI OSV convention: stream 0 = Lens A = right eye after the
    /// yaw-mod-equivalent ordering in the protobuf).
    swap_eyes: bool,
}

#[cfg(target_os = "macos")]
impl ZeroCopyDualStreamFisheyeIter {
    pub fn new(path: &Path, frame_limit: u32, swap_eyes: bool) -> Result<Self> {
        ffmpeg_init();
        let ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
        let video_indices: Vec<usize> = ictx
            .streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .map(|s| s.index())
            .take(2)
            .collect();
        if video_indices.len() < 2 {
            return Err(Error::Ffmpeg(format!(
                "expected 2 video streams (OSV / dual-camera), found {}",
                video_indices.len()
            )));
        }
        let video_indices = [video_indices[0], video_indices[1]];
        let mut decoders = Vec::with_capacity(2);
        for &idx in video_indices.iter() {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            if !crate::decode::try_enable_videotoolbox_decode(&mut codec_ctx) {
                return Err(Error::Ffmpeg(format!(
                    "zero-copy fisheye path requires VideoToolbox hwaccel \
                     — VT setup failed on stream {idx}"
                )));
            }
            decoders.push(codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?);
        }
        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "OSV zero-copy streams disagree on dims: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let time_base = ictx.stream(video_indices[0]).unwrap().time_base();
        let time_base_s = time_base.numerator() as f64 / time_base.denominator() as f64;
        tracing::info!(
            "ZeroCopyDualStreamFisheyeIter: native {}x{}, swap_eyes={}",
            w0, h0, swap_eyes
        );
        Ok(Self {
            ictx, video_indices, decoders,
            eye_w: w0, eye_h: h0,
            frame_limit, frames_yielded: 0,
            time_base_s,
            swap_eyes,
        })
    }

    pub fn eye_dims(&self) -> (u32, u32) { (self.eye_w, self.eye_h) }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let ts = (target_s.max(0.0) * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek {target_s:.3}s: {e}")))?;
        for d in &mut self.decoders {
            d.flush();
        }
        self.frames_yielded = 0;
        Ok(())
    }

    /// Pull the next zero-copy fisheye pair. Wraps each stream's
    /// VT-decoded CVPixelBuffer Y and UV planes as wgpu textures
    /// backed by the same IOSurface. Returns `Ok(None)` at EOF /
    /// frame-limit.
    pub fn next_pair(
        &mut self,
        wgpu_device: &wgpu::Device,
    ) -> Result<Option<ZeroCopyFisheyePair>> {
        use crate::interop_macos::{
            extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
            IOSurfaceNv12Descriptor, RetainedIOSurface,
        };
        if self.frame_limit > 0 && self.frames_yielded >= self.frame_limit {
            return Ok(None);
        }
        let mut frames: [Option<(ffmpeg_next::frame::Video, i64)>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();

        // Drain any pre-buffered frames first.
        for pos in 0..2 {
            if frames[pos].is_some() { continue; }
            if self.decoders[pos].receive_frame(&mut decoded).is_ok() {
                let pts = decoded.pts().unwrap_or(0);
                frames[pos] = Some((
                    std::mem::replace(&mut decoded, ffmpeg_next::frame::Video::empty()),
                    pts,
                ));
            }
        }

        if frames.iter().any(|f| f.is_none()) {
            loop {
                let (stream, packet) = match self.ictx.packets().next() {
                    Some(x) => x,
                    None => break,
                };
                let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                    Some(p) => p,
                    None => continue,
                };
                // Always feed packet to the right decoder, even if we
                // already have its frame (otherwise that decoder falls
                // behind and pairings desync).
                if self.decoders[pos].send_packet(&packet).is_err() { continue; }
                if frames[pos].is_none()
                    && self.decoders[pos].receive_frame(&mut decoded).is_ok()
                {
                    let pts = decoded.pts().unwrap_or(0);
                    frames[pos] = Some((
                        std::mem::replace(&mut decoded, ffmpeg_next::frame::Video::empty()),
                        pts,
                    ));
                }
                if frames.iter().all(|f| f.is_some()) { break; }
            }
        }

        let (Some((f0, pts0)), Some((f1, _pts1))) = (frames[0].take(), frames[1].take()) else {
            return Ok(None);
        };
        let pts_s = pts0 as f64 * self.time_base_s;

        // Extract IOSurfaces + wrap planes for both eyes. Each plane
        // wrapper holds its own retain on the IOSurface so the original
        // descriptor can be dropped safely.
        let surf0 = extract_iosurface_from_vt_frame(&f0)?;
        let desc0 = IOSurfaceNv12Descriptor::new(surf0)?;
        let s0_y_surf  = unsafe { RetainedIOSurface::retain(desc0.surface.as_raw()) };
        let s0_uv_surf = unsafe { RetainedIOSurface::retain(desc0.surface.as_raw()) };
        let s0_y = wgpu_texture_from_iosurface_plane(
            wgpu_device, s0_y_surf, 0,
            metal::MTLPixelFormat::R16Unorm,  wgpu::TextureFormat::R16Unorm,
            desc0.width, desc0.height, "osv_s0_y",
        )?;
        let s0_uv = wgpu_texture_from_iosurface_plane(
            wgpu_device, s0_uv_surf, 1,
            metal::MTLPixelFormat::RG16Unorm, wgpu::TextureFormat::Rg16Unorm,
            desc0.width / 2, desc0.height / 2, "osv_s0_uv",
        )?;
        drop(desc0);

        let surf1 = extract_iosurface_from_vt_frame(&f1)?;
        let desc1 = IOSurfaceNv12Descriptor::new(surf1)?;
        let s1_y_surf  = unsafe { RetainedIOSurface::retain(desc1.surface.as_raw()) };
        let s1_uv_surf = unsafe { RetainedIOSurface::retain(desc1.surface.as_raw()) };
        let s1_y = wgpu_texture_from_iosurface_plane(
            wgpu_device, s1_y_surf, 0,
            metal::MTLPixelFormat::R16Unorm,  wgpu::TextureFormat::R16Unorm,
            desc1.width, desc1.height, "osv_s1_y",
        )?;
        let s1_uv = wgpu_texture_from_iosurface_plane(
            wgpu_device, s1_uv_surf, 1,
            metal::MTLPixelFormat::RG16Unorm, wgpu::TextureFormat::Rg16Unorm,
            desc1.width / 2, desc1.height / 2, "osv_s1_uv",
        )?;
        drop(desc1);

        let (left_y, left_uv, right_y, right_uv) = if self.swap_eyes {
            (s1_y, s1_uv, s0_y, s0_uv)
        } else {
            (s0_y, s0_uv, s1_y, s1_uv)
        };

        self.frames_yielded += 1;
        Ok(Some(ZeroCopyFisheyePair {
            left_y, left_uv, right_y, right_uv,
            eye_w: self.eye_w, eye_h: self.eye_h,
            pts_s,
        }))
    }
}

/// Apply Rec.709 source / full-range RGB destination colorspace via
/// `sws_setColorspaceDetails`. ffmpeg-next 8.1 doesn't wrap this so
/// we go through raw FFI.
fn set_sws_colorspace_details(
    scaler: &mut ffmpeg_next::software::scaling::Context,
    is_16bit: bool,
) {
    use ffmpeg_next::ffi::*;
    unsafe {
        // sws_getCoefficients returns a [4]int32 with the Y'CbCr
        // matrix coefficients for a given colorspace constant.
        // SWS_CS_ITU709 (= 1) → Rec.709 coefficients.
        const SWS_CS_ITU709: std::os::raw::c_int = 1;
        let src_coeffs = sws_getCoefficients(SWS_CS_ITU709);
        let dst_coeffs = sws_getCoefficients(SWS_CS_ITU709);
        // sws_setColorspaceDetails(ctx, inv_table, src_range, table,
        //                          dst_range, brightness, contrast,
        //                          saturation). Ranges are 0=mpeg
        //                          (video), 1=jpeg (full). Source is
        //                          video range (matches OSV's HEVC
        //                          encode); destination is full
        //                          range for RGB output (RGBA64LE).
        let src_range = 0;          // video range Y/UV in
        let dst_range = if is_16bit { 1 } else { 1 }; // full range RGB out
        let _ = sws_setColorspaceDetails(
            scaler.as_mut_ptr(),
            src_coeffs, src_range,
            dst_coeffs, dst_range,
            0,                       // brightness offset (0)
            1 << 16,                 // contrast = 1.0
            1 << 16,                 // saturation = 1.0
        );
    }
}

// Helper: pull packed RGBA bytes out of an ffmpeg Video frame, honoring stride.
fn extract_packed_rgba(frame: &ffmpeg_next::frame::Video) -> Vec<u8> {
    extract_packed_rgba_n(frame, 4)
}

/// Generalised stride-aware row repack. `bytes_per_pixel` is 4 for
/// RGBA8 / 8 for RGBA64LE.
fn extract_packed_rgba_n(frame: &ffmpeg_next::frame::Video, bytes_per_pixel: usize) -> Vec<u8> {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let stride = frame.stride(0);
    let data = frame.data(0);
    let row_bytes = w * bytes_per_pixel;
    let mut out = Vec::with_capacity(row_bytes * h);
    for y in 0..h {
        let row_start = y * stride;
        out.extend_from_slice(&data[row_start..row_start + row_bytes]);
    }
    out
}

// ── BRAW (braw_helper subprocess pipe) ──────────────────────────────

/// Blackmagic RAW iterator. Spawns `braw_helper --decode` and reads
/// BGRA frames off stdout. Multi-track stereo files (URSA Cine
/// Immersive, Pyxis 12K) emit SBS BGRA directly from the helper, so
/// the split logic is the same as the SBS-fisheye path.
///
/// Mirrors `_decode_braw_thread` at `vr180_gui.py:3633-3663`.
pub struct BrawFisheyeIter {
    decoder: vr180_braw::BrawDecoder,
    frame_buf: Vec<u8>,
    eye_w: u32,
    eye_h: u32,
    /// 1.0 / fps — derived from `BrawInfo::frame_rate` so PTS counts
    /// can be turned into seconds.
    dt_s: f64,
    frame_idx: u64,
    frame_limit: u32,
    /// True when the helper composed a side-by-side image (multi-
    /// track stereo). False = single-camera, no split.
    sbs_composed: bool,
}

impl std::fmt::Debug for BrawFisheyeIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrawFisheyeIter")
            .field("eye_w", &self.eye_w)
            .field("eye_h", &self.eye_h)
            .field("dt_s", &self.dt_s)
            .field("frame_idx", &self.frame_idx)
            .field("sbs_composed", &self.sbs_composed)
            .finish_non_exhaustive()
    }
}

impl BrawFisheyeIter {
    /// Open a BRAW file and start decoding. `info` is the result of a
    /// prior `BrawInfo::probe(path)`; pass it in so callers don't pay
    /// the probe twice. `opts` are the colour / exposure knobs
    /// forwarded to braw_helper.
    pub fn new(
        path: &Path,
        info: &vr180_braw::BrawInfo,
        opts: &vr180_braw::decoder::DecodeOptions,
        frame_limit: u32,
    ) -> Result<Self> {
        let decoder = vr180_braw::BrawDecoder::start(path, opts)
            .map_err(|e| Error::Ffmpeg(format!("braw decode start: {e}")))?;
        let header = decoder.header().clone();
        // Multi-track files get composed SBS at the helper level —
        // header.width is then track0_width + track1_width and
        // dual_stream is true. Single-track files emit one eye and
        // we treat them as monocular (left == right).
        let (eye_w, eye_h, sbs_composed) = if header.dual_stream {
            (header.width / 2, header.height, true)
        } else {
            (header.width, header.height, false)
        };
        let dt_s = if info.frame_rate > 0.0 {
            1.0 / info.frame_rate
        } else {
            1.0 / 24.0
        };
        Ok(Self {
            decoder,
            frame_buf: Vec::new(),
            eye_w, eye_h, dt_s,
            frame_idx: 0,
            frame_limit,
            sbs_composed,
        })
    }
}

impl FisheyePairIter for BrawFisheyeIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        if self.frame_limit > 0 && self.frame_idx >= self.frame_limit as u64 {
            return Ok(None);
        }
        let got = self.decoder
            .next_frame(&mut self.frame_buf)
            .map_err(|e| Error::Ffmpeg(format!("braw next_frame: {e}")))?;
        if !got {
            return Ok(None);
        }
        let pts_s = self.frame_idx as f64 * self.dt_s;
        self.frame_idx += 1;

        // BRAW is BGRA. Swizzle to RGBA on the fly while splitting.
        let header = self.decoder.header();
        let bytes_per_pixel = header.bytes_per_pixel(); // 4 or 8
        let frame_w = header.width as usize;
        let frame_h = header.height as usize;
        let mut left = Vec::with_capacity((self.eye_w * self.eye_h * 4) as usize);
        let mut right = Vec::with_capacity((self.eye_w * self.eye_h * 4) as usize);

        if self.sbs_composed {
            let eye_w = self.eye_w as usize;
            for y in 0..frame_h {
                let row_start = y * frame_w * bytes_per_pixel;
                bgra_to_rgba8_row(&self.frame_buf[row_start..row_start + eye_w * bytes_per_pixel],
                                  bytes_per_pixel, &mut left);
                let right_row = row_start + eye_w * bytes_per_pixel;
                bgra_to_rgba8_row(&self.frame_buf[right_row..right_row + eye_w * bytes_per_pixel],
                                  bytes_per_pixel, &mut right);
            }
        } else {
            // Single-eye source: duplicate into both halves. The
            // downstream pipeline treats one eye as the canonical
            // input; the other is a no-op for monocular fisheye.
            for y in 0..frame_h {
                let row_start = y * frame_w * bytes_per_pixel;
                bgra_to_rgba8_row(&self.frame_buf[row_start..row_start + frame_w * bytes_per_pixel],
                                  bytes_per_pixel, &mut left);
            }
            right = left.clone();
        }

        Ok(Some(FisheyePair {
            left, right,
            eye_w: self.eye_w, eye_h: self.eye_h,
            bit_depth: 8,
            pts_s,
        }))
    }

    fn seek(&mut self, _target_s: f64) -> Result<()> {
        // The braw_helper subprocess can't seek mid-stream. The
        // caller has to restart by tearing down and re-opening with
        // a fresh `--start <N>`. Returning Err here would crash the
        // GUI's seek-while-paused flow; instead, we ignore the seek
        // silently and let the caller observe the frame index stays
        // monotonic. The GUI's BRAW handling will need its own
        // restart logic.
        tracing::warn!(
            "BrawFisheyeIter::seek not supported in-process — caller \
             must restart decoder with new --start"
        );
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        (self.eye_w, self.eye_h)
    }
}

/// Copy one row of BGRA[8|16] into a u8 RGBA accumulator. For 16-bit
/// input, the high byte of each channel is taken (linear truncation —
/// matches what the Python kernel's uint8 fallback does, and the GPU
/// shader operates in uint8 anyway).
fn bgra_to_rgba8_row(src: &[u8], bytes_per_pixel: usize, dst: &mut Vec<u8>) {
    match bytes_per_pixel {
        4 => {
            // BGRA8 → RGBA8 swizzle.
            for px in src.chunks_exact(4) {
                dst.push(px[2]); // R from B-slot? No — BGRA means B=0, G=1, R=2, A=3.
                dst.push(px[1]); // G
                dst.push(px[0]); // B
                dst.push(px[3]); // A
            }
        }
        8 => {
            // BGRA16-LE → RGBA8: take the high byte of each channel.
            for px in src.chunks_exact(8) {
                dst.push(px[5]); // R high
                dst.push(px[3]); // G high
                dst.push(px[1]); // B high
                dst.push(px[7]); // A high
            }
        }
        _ => {
            tracing::error!("unexpected BRAW bpp={bytes_per_pixel}");
        }
    }
}

/// Suppress "unused" warnings on the public symbol — every consumer
/// uses the trait by-value or by-reference.
#[allow(dead_code)]
fn _force_link(_: &dyn FisheyePairIter) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a P010LE AVFrame filled with a single Y/U/V triple
    /// (10-bit video range values, e.g. Y=940 for white). Stores the
    /// 10-bit value in the top 10 bits of each 16-bit sample, which
    /// is the P010 convention.
    fn make_p010_frame(w: u32, h: u32, y10: u16, cb10: u16, cr10: u16)
        -> ffmpeg_next::frame::Video
    {
        let mut f = ffmpeg_next::frame::Video::new(
            ffmpeg_next::format::Pixel::P010LE, w, h,
        );
        // P010 stores the 10-bit value in the top 10 bits of each
        // 16-bit word — i.e. value_in_word = real_value << 6.
        let y_word  = (y10  as u32) << 6;
        let cb_word = (cb10 as u32) << 6;
        let cr_word = (cr10 as u32) << 6;

        // Y plane: w*h, 2 bytes each (LE).
        let y_stride = f.stride(0);
        let y_data = f.data_mut(0);
        for row in 0..h as usize {
            let off = row * y_stride;
            for col in 0..w as usize {
                let p = off + col * 2;
                y_data[p    ] = (y_word & 0xFF) as u8;
                y_data[p + 1] = ((y_word >> 8) & 0xFF) as u8;
            }
        }
        // UV plane: (w/2)*(h/2) pixels, interleaved Cb Cr (4 bytes per UV pixel).
        let uv_stride = f.stride(1);
        let uv_data = f.data_mut(1);
        for row in 0..(h / 2) as usize {
            let off = row * uv_stride;
            for col in 0..(w / 2) as usize {
                let p = off + col * 4;
                uv_data[p    ] = (cb_word & 0xFF) as u8;
                uv_data[p + 1] = ((cb_word >> 8) & 0xFF) as u8;
                uv_data[p + 2] = (cr_word & 0xFF) as u8;
                uv_data[p + 3] = ((cr_word >> 8) & 0xFF) as u8;
            }
        }
        f
    }

    /// Convert a single P010 (Y, Cb, Cr) triple through swscale into
    /// RGBA64LE using the same configuration that
    /// DualStreamFisheyeIter::scale_one applies (Rec.709 source +
    /// full-range destination + colorspace details). Returns the
    /// 16-bit RGBA at pixel (0,0).
    fn swscale_p010_to_rgba64(
        y10: u16, cb10: u16, cr10: u16,
    ) -> (u16, u16, u16, u16) {
        let w = 4;
        let h = 4;
        crate::decode::init();
        let mut src = make_p010_frame(w, h, y10, cb10, cr10);
        unsafe {
            let raw = src.as_mut_ptr();
            use ffmpeg_next::ffi::*;
            (*raw).colorspace      = AVColorSpace::AVCOL_SPC_BT709;
            (*raw).color_range     = AVColorRange::AVCOL_RANGE_MPEG;
            (*raw).color_primaries = AVColorPrimaries::AVCOL_PRI_BT709;
            (*raw).color_trc       = AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        }
        let mut sws = ffmpeg_next::software::scaling::Context::get(
            ffmpeg_next::format::Pixel::P010LE, w, h,
            ffmpeg_next::format::Pixel::RGBA64LE, w, h,
            ffmpeg_next::software::scaling::Flags::BICUBIC
                | ffmpeg_next::software::scaling::Flags::FULL_CHR_H_INT,
        ).expect("sws ctx");
        set_sws_colorspace_details(&mut sws, true);
        let mut dst = ffmpeg_next::frame::Video::empty();
        sws.run(&src, &mut dst).expect("sws run");
        // Pixel (0, 0) = first 8 bytes (RGBA, 2 bytes per channel LE).
        let d = dst.data(0);
        let r = u16::from_le_bytes([d[0], d[1]]);
        let g = u16::from_le_bytes([d[2], d[3]]);
        let b = u16::from_le_bytes([d[4], d[5]]);
        let a = u16::from_le_bytes([d[6], d[7]]);
        (r, g, b, a)
    }

    /// Color round-trip for the P010LE → RGBA64LE conversion. Each
    /// test color is fed in as 10-bit video-range Y/Cb/Cr, the
    /// resulting RGBA64LE is compared against the expected
    /// full-range 16-bit RGB. Tolerance is ±2% of 65535 — swscale's
    /// matrix has some rounding plus we're starting from quantized
    /// 10-bit data, so exact equality isn't expected.
    #[test]
    fn p010_to_rgba64_color_roundtrip() {
        // (description, y10, cb10, cr10, expected_r, expected_g, expected_b)
        // Computed from Rec.709 ITU-R BT.709 formulas, video-range Y in
        // [64, 940], video-range Cb/Cr centred at 512 spanning [64, 960].
        let cases: &[(&str, u16, u16, u16, u16, u16, u16)] = &[
            // Pure black: Y=64, Cb=Cr=512 → RGB(0,0,0)
            ("black", 64,  512, 512,  0,     0,     0),
            // Pure white: Y=940, Cb=Cr=512 → RGB(65535,65535,65535)
            ("white", 940, 512, 512,  65535, 65535, 65535),
            // Mid-gray: Y=502 (≈midpoint), Cb=Cr=512 → ~32768
            ("gray",  502, 512, 512,  32768, 32768, 32768),
            // Pure red: Y=250, Cb=410, Cr=960 → RGB(65535, 0, 0)
            ("red",   250, 410, 960,  65535, 0,     0),
            // Pure green: Y=691, Cb=167, Cr=105 → RGB(0, 65535, 0)
            ("green", 691, 167, 105,  0,     65535, 0),
            // Pure blue: Y=127, Cb=960, Cr=471 → RGB(0, 0, 65535)
            ("blue",  127, 960, 471,  0,     0,     65535),
        ];
        let tolerance = (65535.0 * 0.02) as i32;   // ±2%
        for (name, y, cb, cr, er, eg, eb) in cases {
            let (r, g, b, _a) = swscale_p010_to_rgba64(*y, *cb, *cr);
            let dr = (r as i32 - *er as i32).abs();
            let dg = (g as i32 - *eg as i32).abs();
            let db = (b as i32 - *eb as i32).abs();
            eprintln!(
                "{name}: Y/Cb/Cr=({y},{cb},{cr}) → R/G/B=({r},{g},{b}) \
                 expected ({er},{eg},{eb}) Δ=({dr},{dg},{db})"
            );
            assert!(
                dr <= tolerance && dg <= tolerance && db <= tolerance,
                "{name}: out=({r},{g},{b}) exp=({er},{eg},{eb}) Δ=({dr},{dg},{db}) tol=±{tolerance}"
            );
        }
    }

    /// Decode-throughput micro-benchmark against a real .osv file (if
    /// present). Times pulling N frames through DualStreamFisheyeIter
    /// with VT decode + scaler downsample, reports fps.
    ///
    /// Useful for spot-checking that VT is actually engaged (sw decode
    /// of 3840×3840 10-bit HEVC × 2 streams runs at single-digit fps).
    #[test]
    fn bench_osv_decode_throughput() {
        let candidates = [
            "/Volumes/Silver/250826OSV Swap/CAM_20250811172419_0044_D.OSV",
            "/Volumes/Silver/develop/vr180_fisheye_converter/CAM_20260225224810_0003_D.osv",
        ];
        for path_str in candidates {
            let path = std::path::Path::new(path_str);
            if !path.exists() { continue; }
            let mut it = match DualStreamFisheyeIter::new_with_swap(
                path, crate::decode::HwDecode::Auto, 60, true,
            ) {
                Ok(it) => it,
                Err(e) => { eprintln!("skip bench {path_str}: {e}"); continue; }
            };
            let (w, h) = it.eye_dims();
            eprintln!("VT engaged: {:?}", it.hw_active);
            let t = std::time::Instant::now();
            let mut n = 0u32;
            while let Ok(Some(_pair)) = it.next_pair() { n += 1; }
            let elapsed = t.elapsed();
            let fps = n as f64 / elapsed.as_secs_f64();
            eprintln!(
                "BENCH {} → {} frames @ {}x{} in {:.2?} = {:.1} fps",
                path_str, n, w, h, elapsed, fps
            );
            // 30 fps real-time threshold for the typical 29.97 fps OSV.
            if n >= 30 {
                assert!(fps >= 20.0, "decode too slow for real-time: {fps:.1} fps");
            }
            break;
        }
    }

    #[test]
    fn bgra8_to_rgba8_swizzle() {
        // One BGRA8 pixel: B=10, G=20, R=30, A=255 → after swizzle
        // should land as R=30, G=20, B=10, A=255 in dst.
        let src = vec![10u8, 20, 30, 255];
        let mut dst = Vec::new();
        bgra_to_rgba8_row(&src, 4, &mut dst);
        assert_eq!(dst, vec![30, 20, 10, 255]);
    }

    #[test]
    fn bgra16le_to_rgba8_high_byte() {
        // BGRA16 LE: B = 0x0102 = bytes [02, 01], G = 0x0304 = [04, 03],
        // R = 0x0506 = [06, 05], A = 0x0708 = [08, 07].
        // High byte: B=01, G=03, R=05, A=07.
        // Swizzled to RGBA: R, G, B, A = 05, 03, 01, 07.
        let src = vec![0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07];
        let mut dst = Vec::new();
        bgra_to_rgba8_row(&src, 8, &mut dst);
        assert_eq!(dst, vec![0x05, 0x03, 0x01, 0x07]);
    }
}
