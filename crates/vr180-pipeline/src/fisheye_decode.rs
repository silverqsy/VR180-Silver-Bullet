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

    /// Decode-forward to the first frame at/after `target_pts_s` (within half
    /// a frame `dt`), returning ONLY that frame. Call after `seek`.
    ///
    /// The default scans via `next_pair`, so every intermediate GOP frame
    /// still pays the full HW→CPU download + swscale. Iterators where that
    /// conversion dominates the seek cost (the dual-stream OSV detail-still
    /// path) override this to decode the intermediates on the hardware engine
    /// but convert only the target frame. Returns the last decoded frame if
    /// the target is past EOF, or `Ok(None)` if nothing decodes.
    fn decode_forward_to(&mut self, target_pts_s: f64, dt: f64)
        -> Result<Option<FisheyePair>>
    {
        let cutoff = target_pts_s - dt * 0.5;
        let mut latest: Option<FisheyePair> = None;
        for _ in 0..4000 {
            match self.next_pair()? {
                Some(p) => {
                    let reached = p.pts_s >= cutoff;
                    latest = Some(p);
                    if reached { break; }
                }
                None => break,
            }
        }
        Ok(latest)
    }
}

// ── Segmented fisheye (chain several files of one recording) ───────

/// Chains several fisheye files (DJI OSV / SBS / BRAW segments of one
/// recording) into a single continuous `FisheyePairIter`, mirroring the
/// EAC [`crate::decode::SegmentedStreamPairIter`]. The per-segment opener
/// is supplied by the caller (it varies by source kind). Yielded `pts_s`
/// is rebased onto the GLOBAL clip timeline so stabilization-by-pts and
/// trim stay aligned across segment boundaries.
pub struct SegmentedFisheyeIter {
    seg_paths: Vec<std::path::PathBuf>,
    /// Cumulative clip-time start of each segment (seconds).
    seg_start_s: Vec<f64>,
    total_dur_s: f64,
    cur_idx: usize,
    cur: Box<dyn FisheyePairIter>,
    opener: Box<dyn FnMut(&Path) -> Result<Box<dyn FisheyePairIter>>>,
}

impl SegmentedFisheyeIter {
    /// `opener` opens one segment into a `FisheyePairIter` (source-kind
    /// specific — e.g. a `DualStreamFisheyeIter` for OSV). `seg_durations_s`
    /// are the per-segment durations (probed by the caller) used to rebase
    /// timestamps and to map a global seek to (segment, local time).
    pub fn new(
        segments: &[std::path::PathBuf],
        seg_durations_s: &[f64],
        mut opener: Box<dyn FnMut(&Path) -> Result<Box<dyn FisheyePairIter>>>,
    ) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for i in 0..segments.len() {
            seg_start_s.push(acc);
            acc += seg_durations_s.get(i).copied().unwrap_or(0.0);
        }
        let cur = opener(&segments[0])?;
        Ok(Self {
            seg_paths: segments.to_vec(),
            seg_start_s,
            total_dur_s: acc,
            cur_idx: 0,
            cur,
            opener,
        })
    }

    /// Total clip duration across all segments (seconds).
    pub fn total_duration_s(&self) -> f64 { self.total_dur_s }
}

impl FisheyePairIter for SegmentedFisheyeIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        loop {
            if let Some(mut p) = self.cur.next_pair()? {
                p.pts_s += self.seg_start_s[self.cur_idx]; // local → global
                return Ok(Some(p));
            }
            // Current segment exhausted — advance to the next, if any.
            if self.cur_idx + 1 >= self.seg_paths.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            let path = self.seg_paths[self.cur_idx].clone();
            self.cur = (self.opener)(&path)?;
        }
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        let t = target_s.clamp(0.0, self.total_dur_s);
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            let path = self.seg_paths[idx].clone();
            self.cur = (self.opener)(&path)?;
            self.cur_idx = idx;
        }
        self.cur.seek(t - self.seg_start_s[idx])
    }

    fn eye_dims(&self) -> (u32, u32) { self.cur.eye_dims() }
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
    /// Temporary decode-stage profiling: cumulative seconds spent in the
    /// HW-frame download and in swscale, summed across both eyes.
    dbg_download_s: f64,
    dbg_scale_s: f64,
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
                if matches!(hw, HwDecode::VideoToolbox) {
                    return Err(Error::Ffmpeg(
                        "VideoToolbox is macOS-only".into()
                    ));
                }
                // Windows: offload HEVC decode to the GPU video engine via
                // D3D11VA (NVDEC on NVIDIA, equivalent on Intel/AMD). `Auto`
                // falls back to software if the hwaccel can't attach.
                // Decoded surfaces arrive as AV_PIX_FMT_D3D11 and are
                // downloaded to host memory in `scale_one`.
                #[cfg(target_os = "windows")]
                if matches!(hw, HwDecode::Auto)
                    && crate::decode::try_enable_d3d11va_decode(&mut codec_ctx)
                {
                    hw_active[i] = true;
                    tracing::info!(
                        "DualStreamFisheyeIter: D3D11VA hw decode active on stream {i}"
                    );
                }
                #[cfg(not(target_os = "windows"))]
                { let _ = i; }
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
            "DualStreamFisheyeIter: native {}x{}, hw={}/{}, working {}x{} (cap={})",
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
            dbg_download_s: 0.0,
            dbg_scale_s: 0.0,
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

    fn decode_forward_to(&mut self, target_pts_s: f64, dt: f64)
        -> Result<Option<FisheyePair>>
    {
        // Decode-forward on the HW video engine, running the HW→CPU download +
        // swscale ONLY for the target frame. Intermediate GOP frames are still
        // decoded (the decoder needs them as references) but their surfaces are
        // dropped without the ~200 ms/frame conversion — the dominant cost when
        // seeking deep into a GOP for a single still (≈11 s → ≈2 s at 8K).
        let cutoff = target_pts_s - dt * 0.5;
        let mut result: [Option<(Vec<u8>, i64)>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut sw_storage = ffmpeg_next::frame::Video::empty();
        let mut eof = false;
        for _ in 0..8000 {
            for pos in 0..2 {
                if result[pos].is_some() { continue; }
                // Drain everything this decoder has buffered, skipping the
                // download+swscale until we hit the target frame.
                while self.decoders[pos].receive_frame(&mut decoded).is_ok() {
                    let pts = decoded.pts().unwrap_or(0);
                    if (pts as f64 * self.time_base_s) >= cutoff {
                        let rgba = self.scale_one(pos, &mut decoded, &mut sw_storage)?;
                        result[pos] = Some((rgba, pts));
                        break;
                    }
                    // else: intermediate frame — decoded for reference, dropped.
                }
            }
            if result.iter().all(|r| r.is_some()) { break; }
            if eof { break; }
            // Feed one more packet to the stream that needs it. (Bind first so
            // the ictx borrow ends before we touch the decoders, as next_pair.)
            let res = self.ictx.packets().next();
            match res {
                Some((stream, packet)) => {
                    if let Some(pos) =
                        self.video_indices.iter().position(|&i| i == stream.index())
                    {
                        let _ = self.decoders[pos].send_packet(&packet);
                    }
                }
                None => eof = true, // no more packets — drain once more, then stop
            }
        }
        let (Some((f0, pts0)), Some((f1, _))) = (result[0].take(), result[1].take())
        else {
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
        // Hardware-resident frame? Download to host memory. Covers both
        // the macOS VideoToolbox surface and the Windows D3D11VA surface
        // (NV12 for 8-bit, P010 for 10-bit Main10) — swscale handles both.
        let src_ref: &mut ffmpeg_next::frame::Video = if self.hw_active[pos]
            && matches!(
                decoded.format(),
                ffmpeg_next::format::Pixel::VIDEOTOOLBOX
                    | ffmpeg_next::format::Pixel::D3D11
            )
        {
            let t_dl = std::time::Instant::now();
            crate::decode::download_hw_frame(decoded, sw_storage)?;
            self.dbg_download_s += t_dl.elapsed().as_secs_f64();
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
        let t_scale = std::time::Instant::now();
        let run_res = scaler.run(src_ref, &mut rgba_frame);
        let scale_elapsed = t_scale.elapsed().as_secs_f64();
        run_res.map_err(|e| Error::Ffmpeg(format!("scale: {e}")))?;
        self.dbg_scale_s += scale_elapsed;
        let bpp = if is_16bit { 8 } else { 4 };
        Ok(extract_packed_rgba_n(&rgba_frame, bpp))
    }

    /// Temporary profiling accessor: cumulative (download, swscale)
    /// seconds since open, summed across both eyes.
    pub fn debug_timing_s(&self) -> (f64, f64) {
        (self.dbg_download_s, self.dbg_scale_s)
    }
}

// ── Windows zero-copy dual-stream (d3d11va, GPU-resident) ──────────
//
// The Windows analogue of the macOS `ZeroCopyDualStreamFisheyeIter`
// below. Both OSV streams decode via `d3d11va` (NVDEC on NVIDIA) and
// stay GPU-resident as `AV_PIX_FMT_D3D11` P010 textures. Instead of the
// `DualStreamFisheyeIter` download + swscale → RGBA `Vec<u8>` (the 54%
// download + 29% swscale the profiler flagged), each eye's decoder DPB
// slice is `CopySubresourceRegion`'d into a fresh shareable texture and
// exported as an NT handle. The consumer imports those handles into
// wgpu's Vulkan device (zero-copy memory alias) and projects the P010
// planes directly via `project_fisheye_p010_planar_to_equirect_texture_16`.
//
// Used by the GUI preview's `run_fisheye_zerocopy` path. Export still has
// its own path; this is preview-only for now.

/// One GPU-resident fisheye pair — each eye is a D3D11 P010 texture shared
/// out as an NT handle (see [`crate::interop_windows::D3d11SharedTexture`]).
/// Hold this alive until the projection GPU work that imports + reads it has
/// been submitted; dropping it closes the NT handles and releases the D3D11
/// textures.
#[cfg(target_os = "windows")]
pub struct SharedFisheyePair {
    pub left: crate::interop_windows::D3d11SharedTexture,
    pub right: crate::interop_windows::D3d11SharedTexture,
    /// Native fisheye dims (the full-res P010 — calibration resolves against
    /// these, NOT a downscaled working res).
    pub eye_w: u32,
    pub eye_h: u32,
    /// Presentation timestamp in seconds, `0.0` if unknown.
    pub pts_s: f64,
}

// SAFETY: the COM interfaces inside `D3d11SharedTexture` are windows-rs agile
// wrappers (Send + Sync) and the NT handle is a process-global value valid on
// any thread. We move the pair from the decode worker thread to the consuming
// (project) thread and there only (a) import via the NT handle on the Vulkan
// side and (b) Release/CloseHandle on drop — we never issue D3D11 immediate-
// context calls on the consuming thread. Moving it across threads is sound.
#[cfg(target_os = "windows")]
unsafe impl Send for SharedFisheyePair {}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for SharedFisheyePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedFisheyePair")
            .field("eye_w", &self.eye_w)
            .field("eye_h", &self.eye_h)
            .field("pts_s", &self.pts_s)
            .finish_non_exhaustive()
    }
}

/// Dual-stream OSV iterator that keeps frames GPU-resident via `d3d11va` and
/// yields [`SharedFisheyePair`]s of single-plane **RGBA16** textures. Each
/// eye's P010 is converted to RGBA16 (+ box downscale to the working res) on
/// the D3D11 side ([`crate::interop_windows::P010Converter`]) — the multi-plane
/// P010 imports into Vulkan with a broken chroma-plane offset, so we hand the
/// importer a single-plane RGBA16 instead. No CPU download, no swscale.
#[cfg(target_os = "windows")]
pub struct D3d11SharedDualStreamIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: [usize; 2],
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    /// Native decoded fisheye dims.
    native_w: u32,
    native_h: u32,
    /// Working (downscaled) dims the converter outputs / we yield.
    work_w: u32,
    work_h: u32,
    /// One converter per stream (each stream decodes on its own d3d11va
    /// device), built lazily on the first frame.
    converters: [Option<crate::interop_windows::P010Converter>; 2],
    frames_yielded: u32,
    time_base_s: f64,
    /// If true, output left = stream[1], right = stream[0] (DJI OSV
    /// convention; see `DualStreamFisheyeIter`).
    swap_eyes: bool,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for D3d11SharedDualStreamIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d11SharedDualStreamIter")
            .field("video_indices", &self.video_indices)
            .field("native", &(self.native_w, self.native_h))
            .field("work", &(self.work_w, self.work_h))
            .field("swap_eyes", &self.swap_eyes)
            .field("frames_yielded", &self.frames_yielded)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "windows")]
impl D3d11SharedDualStreamIter {
    /// Open a dual-stream OSV and enable `d3d11va` HEVC decode on both streams.
    /// `work_w`/`work_h` is the preview working resolution the P010 is converted
    /// + downscaled to (clamped to never upscale past native). Returns an error
    /// (so the caller can fall back to the CPU path) if there aren't two video
    /// streams or `d3d11va` can't attach to either stream.
    pub fn new(path: &Path, swap_eyes: bool, work_w: u32, work_h: u32) -> Result<Self> {
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
            let mut codec_ctx =
                ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                    .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            if !crate::decode::try_enable_d3d11va_decode(&mut codec_ctx) {
                return Err(Error::Ffmpeg(format!(
                    "zero-copy OSV path requires d3d11va hwaccel — setup failed on stream {idx}"
                )));
            }
            decoders.push(
                codec_ctx.decoder().video()
                    .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?,
            );
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
        let work_w = work_w.clamp(2, w0);
        let work_h = work_h.clamp(2, h0);
        tracing::info!(
            "D3d11SharedDualStreamIter: native {}x{} → work {}x{} (D3D11 P010→RGBA16 + downscale), swap_eyes={}",
            w0, h0, work_w, work_h, swap_eyes
        );
        Ok(Self {
            ictx, video_indices, decoders,
            native_w: w0, native_h: h0,
            work_w, work_h,
            converters: [None, None],
            frames_yielded: 0,
            time_base_s,
            swap_eyes,
        })
    }

    /// Working (downscaled) dims — what `next_pair` yields.
    pub fn eye_dims(&self) -> (u32, u32) { (self.work_w, self.work_h) }
    /// Native decoded dims (before downscale).
    pub fn native_dims(&self) -> (u32, u32) { (self.native_w, self.native_h) }

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

    /// Pull the next GPU-resident pair. Decodes one frame from each stream
    /// (holding both simultaneously), shares each eye's DPB slice into a fresh
    /// NT-handle texture, then releases the source frames. Returns `Ok(None)`
    /// at EOF.
    pub fn next_pair(&mut self) -> Result<Option<SharedFisheyePair>> {
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
                // Always feed the packet to its decoder, even if that eye's
                // frame is already in hand, so the decoders stay in lockstep.
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

        // Convert each eye P010 → shareable RGBA16 (D3D11-side YCbCr→RGB +
        // downscale, then NT-handle share). After this the source frames can
        // drop — their pixels live in our converted copies.
        let (work_w, work_h) = (self.work_w, self.work_h);
        let s0 = unsafe {
            crate::interop_windows::share_eye_converted(&f0, &mut self.converters[0], work_w, work_h)
        }
        .ok_or_else(|| Error::Ffmpeg("zero-copy: convert eye0 failed".into()))?;
        let s1 = unsafe {
            crate::interop_windows::share_eye_converted(&f1, &mut self.converters[1], work_w, work_h)
        }
        .ok_or_else(|| Error::Ffmpeg("zero-copy: convert eye1 failed".into()))?;
        // Both eyes' decode+convert were submitted (on their two separate
        // d3d11va devices) WITHOUT an intervening fence, so the GPU runs them
        // concurrently. Now fence both before handing the pair off — the
        // importer must see completed pixels (the chroma fix depends on it).
        unsafe { s0.wait_gpu_idle(); s1.wait_gpu_idle(); }
        drop(f0);
        drop(f1);

        let (left, right) = if self.swap_eyes { (s1, s0) } else { (s0, s1) };
        self.frames_yielded += 1;
        Ok(Some(SharedFisheyePair {
            left, right,
            eye_w: self.work_w, eye_h: self.work_h,
            pts_s,
        }))
    }
}

/// One VideoToolbox zero-copy OSV pair: each eye's Y + UV planes wrapped as
/// wgpu textures that ALIAS the decoded P010 IOSurface (no host download, no
/// swscale). The consumer box-downscales to the preview working res with
/// `Device::resolve_p010_planes_to_rgba16` and projects from there — same
/// anti-aliased working res as the CPU path, but kept on the GPU. Hold the
/// pair alive until the resolve pass that reads it has been submitted (and
/// given the GPU a frame to finish); dropping it releases the IOSurface
/// retains. macOS analogue of [`SharedFisheyePair`].
#[cfg(target_os = "macos")]
pub struct VtSharedFisheyePair {
    pub left_y:   crate::interop_macos::IOSurfacePlaneTexture,
    pub left_uv:  crate::interop_macos::IOSurfacePlaneTexture,
    pub right_y:  crate::interop_macos::IOSurfacePlaneTexture,
    pub right_uv: crate::interop_macos::IOSurfacePlaneTexture,
    /// Native fisheye (Y-plane) dims — the resolve downscales from these.
    pub native_w: u32,
    pub native_h: u32,
    /// Presentation timestamp in seconds, `0.0` if unknown.
    pub pts_s: f64,
}

/// Dual-stream OSV iterator that keeps frames GPU-resident via VideoToolbox
/// and yields [`VtSharedFisheyePair`]s of P010 plane textures. macOS analogue
/// of [`D3d11SharedDualStreamIter`]; the EAC sibling is
/// [`crate::decode::ZeroCopyStreamPairIter`] (same IOSurface-plane recipe,
/// different output topology). Single-segment only — merged OSV recordings
/// stay on the CPU path for now.
#[cfg(target_os = "macos")]
pub struct VtSharedDualStreamIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: [usize; 2],
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    native_w: u32,
    native_h: u32,
    /// When true output left = stream[1], right = stream[0] (DJI OSV
    /// convention; the caller XORs in the user swap, same as
    /// [`DualStreamFisheyeIter`]).
    swap_eyes: bool,
    time_base_s: f64,
    /// Nominal frame duration (1/fps) for the precise-seek run-in window.
    dt_s: f64,
    frames_yielded: u32,
    /// Precise-seek state — `next_pair` decodes-and-discards the keyframe→target
    /// run-in so the first pair returned is the exact requested frame (matches
    /// [`crate::decode::ZeroCopyStreamPairIter::seek`]).
    skip_until_s: Option<f64>,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for VtSharedDualStreamIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VtSharedDualStreamIter")
            .field("native", &(self.native_w, self.native_h))
            .field("swap_eyes", &self.swap_eyes)
            .field("frames_yielded", &self.frames_yielded)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl VtSharedDualStreamIter {
    /// Open a dual-stream OSV and force VideoToolbox decode on both streams.
    /// Returns an error (so the caller can fall back to the CPU path) if there
    /// aren't two matching video streams or VT can't attach to either.
    pub fn new(path: &Path, swap_eyes: bool) -> Result<Self> {
        ffmpeg_init();
        let ictx = ffmpeg_next::format::input(path)
            .map_err(|e| Error::Ffmpeg(format!("open {path:?}: {e}")))?;
        let video_indices: Vec<usize> = ictx.streams()
            .filter(|s| s.parameters().medium() == ffmpeg_next::media::Type::Video)
            .map(|s| s.index())
            .take(2)
            .collect();
        if video_indices.len() < 2 {
            return Err(Error::Ffmpeg(format!(
                "expected 2 video streams (OSV), found {}", video_indices.len()
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
                    "VT zero-copy OSV path requires VideoToolbox — setup failed on stream {idx}"
                )));
            }
            decoders.push(codec_ctx.decoder().video()
                .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?);
        }
        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "OSV VT streams disagree on dims: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let s0 = ictx.stream(video_indices[0]).unwrap();
        let tb = s0.time_base();
        let time_base_s = tb.numerator() as f64 / tb.denominator().max(1) as f64;
        let fr = s0.avg_frame_rate();
        let dt_s = if fr.numerator() > 0 {
            fr.denominator() as f64 / fr.numerator() as f64
        } else { 1.0 / 30.0 };
        Ok(Self {
            ictx, video_indices, decoders,
            native_w: w0, native_h: h0,
            swap_eyes, time_base_s, dt_s,
            frames_yielded: 0, skip_until_s: None,
        })
    }

    /// Native fisheye dims — the consumer downscales to the working res itself.
    pub fn eye_dims(&self) -> (u32, u32) { (self.native_w, self.native_h) }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let target_s = target_s.max(0.0);
        let ts = (target_s * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek {target_s:.3}s: {e}")))?;
        for d in &mut self.decoders { d.flush(); }
        self.frames_yielded = 0;
        self.skip_until_s = Some(target_s);
        Ok(())
    }

    /// Wrap a VT-decoded P010 frame's IOSurface as (Y: `R16Unorm`,
    /// UV: `Rg16Unorm`) plane textures aliasing the surface — the proven
    /// `ZeroCopyStreamPairIter` recipe (10-bit GoPro/OSV both produce P010).
    fn wrap_p010_planes(
        device: &wgpu::Device,
        frame: &ffmpeg_next::frame::Video,
    ) -> Result<(crate::interop_macos::IOSurfacePlaneTexture,
                 crate::interop_macos::IOSurfacePlaneTexture)> {
        use crate::interop_macos::{
            extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
            IOSurfaceNv12Descriptor, RetainedIOSurface,
        };
        let surf = extract_iosurface_from_vt_frame(frame)?;
        let desc = IOSurfaceNv12Descriptor::new(surf)?;
        let y_surf  = unsafe { RetainedIOSurface::retain(desc.surface.as_raw()) };
        let uv_surf = unsafe { RetainedIOSurface::retain(desc.surface.as_raw()) };
        let y = wgpu_texture_from_iosurface_plane(
            device, y_surf, 0,
            metal::MTLPixelFormat::R16Unorm, wgpu::TextureFormat::R16Unorm,
            desc.width, desc.height, "osv_vt_y")?;
        let uv = wgpu_texture_from_iosurface_plane(
            device, uv_surf, 1,
            metal::MTLPixelFormat::RG16Unorm, wgpu::TextureFormat::Rg16Unorm,
            desc.width / 2, desc.height / 2, "osv_vt_uv")?;
        drop(desc);
        Ok((y, uv))
    }

    /// Decode the next (s0, s1) frame pair, wrap each eye's P010 planes, apply
    /// the eye swap, and return them. Honours the precise-seek run-in discard.
    pub fn next_pair(&mut self, device: &wgpu::Device)
        -> Result<Option<VtSharedFisheyePair>>
    {
        let mut frames: [Option<ffmpeg_next::frame::Video>; 2] = [None, None];
        let mut decoded = ffmpeg_next::frame::Video::empty();

        let (f0, f1) = 'fill: loop {
            for pos in 0..2 {
                if frames[pos].is_some() { continue; }
                if self.decoders[pos].receive_frame(&mut decoded).is_ok() {
                    frames[pos] = Some(std::mem::replace(
                        &mut decoded, ffmpeg_next::frame::Video::empty()));
                }
            }
            if frames.iter().any(|f| f.is_none()) {
                loop {
                    let (stream, packet) = match self.ictx.packets().next() {
                        Some(x) => x, None => break,
                    };
                    let pos = match self.video_indices.iter().position(|&i| i == stream.index()) {
                        Some(p) => p, None => continue,
                    };
                    if self.decoders[pos].send_packet(&packet).is_err() { continue; }
                    if frames[pos].is_none()
                        && self.decoders[pos].receive_frame(&mut decoded).is_ok()
                    {
                        frames[pos] = Some(std::mem::replace(
                            &mut decoded, ffmpeg_next::frame::Video::empty()));
                    }
                    if frames.iter().all(|f| f.is_some()) { break; }
                }
            }
            let (Some(f0), Some(f1)) = (frames[0].take(), frames[1].take()) else {
                return Ok(None);
            };
            // Precise-seek run-in: discard pairs before the target frame.
            if let Some(target) = self.skip_until_s {
                let t = f0.pts().unwrap_or(0) as f64 * self.time_base_s;
                if t < target - 0.5 * self.dt_s { continue 'fill; }
                self.skip_until_s = None;
            }
            break 'fill (f0, f1);
        };

        let pts_s = f0.pts().unwrap_or(0) as f64 * self.time_base_s;
        let (s0_y, s0_uv) = Self::wrap_p010_planes(device, &f0)?;
        let (s1_y, s1_uv) = Self::wrap_p010_planes(device, &f1)?;
        self.frames_yielded += 1;
        // Eye mapping mirrors DualStreamFisheyeIter: swap → left = stream[1].
        let (left_y, left_uv, right_y, right_uv) = if self.swap_eyes {
            (s1_y, s1_uv, s0_y, s0_uv)
        } else {
            (s0_y, s0_uv, s1_y, s1_uv)
        };
        Ok(Some(VtSharedFisheyePair {
            left_y, left_uv, right_y, right_uv,
            native_w: self.native_w, native_h: self.native_h,
            pts_s,
        }))
    }
}

/// One GPU-resident EAC stream pair — the two GoPro `.360` HEVC streams
/// (s0, s4), each decoded P010 by `d3d11va`/NVDEC and converted to a
/// single-plane Rgba16Unorm D3D11 texture shared out as an NT handle. The
/// EAC cross assembly (`Device::rgba*_to_eac_cross`) reads BOTH streams to
/// build each lens's cross, so — unlike the OSV [`SharedFisheyePair`] —
/// these stay in STREAM order (s0/s4), not eye order; the lens→eye mapping
/// happens in the cross shader. Hold alive until the import + GPU read has
/// been submitted; dropping closes the NT handles.
#[cfg(target_os = "windows")]
pub struct SharedEacPair {
    pub s0: crate::interop_windows::D3d11SharedTexture,
    pub s4: crate::interop_windows::D3d11SharedTexture,
    /// EAC layout (stream/tile/cross dims), derived from the s0 stream.
    pub dims: vr180_core::eac::Dims,
    /// Presentation timestamp in seconds, `0.0` if unknown.
    pub pts_s: f64,
}

// SAFETY: identical reasoning to `SharedFisheyePair` — the COM interfaces are
// agile and the NT handles are process-global; we only import (Vulkan side)
// and CloseHandle on drop on the consuming thread, never D3D11 immediate-
// context calls. Moving across the decode→project thread boundary is sound.
#[cfg(target_os = "windows")]
unsafe impl Send for SharedEacPair {}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for SharedEacPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEacPair")
            .field("dims", &self.dims)
            .field("pts_s", &self.pts_s)
            .finish_non_exhaustive()
    }
}

/// Windows zero-copy EAC decoder — the GoPro `.360` analog of
/// [`D3d11SharedDualStreamIter`]. Decodes both EAC HEVC streams with
/// `d3d11va` (NVDEC), converts each P010 frame to single-plane Rgba16Unorm
/// at NATIVE stream resolution (the cross assembly indexes exact stream
/// pixels, so there is NO downscale), shares each as an NT-handle texture,
/// and yields [`SharedEacPair`]s. No CPU download, no swscale, no CPU
/// `assemble_lens_*`. Returns an error from `new` (so the caller can fall
/// back to the CPU `StreamPairIter` path) if there aren't two video streams,
/// they disagree on dims, the width isn't a valid EAC layout, or `d3d11va`
/// can't attach.
#[cfg(target_os = "windows")]
pub struct D3d11SharedStreamPairIter {
    ictx: ffmpeg_next::format::context::Input,
    video_indices: [usize; 2],
    decoders: Vec<ffmpeg_next::codec::decoder::Video>,
    dims: vr180_core::eac::Dims,
    /// One converter per stream (each stream decodes on its own d3d11va
    /// device), built lazily on the first frame.
    converters: [Option<crate::interop_windows::P010Converter>; 2],
    frames_yielded: u32,
    time_base_s: f64,
    /// Nominal frame duration (1 / avg_frame_rate), for the precise-seek
    /// run-in window.
    dt_s: f64,
    /// PRECISE-seek state — mirrors `ZeroCopyStreamPairIter::skip_until_s`.
    /// After `seek(t)` the container lands on the keyframe ≤ t; `next_pair`
    /// decodes-and-discards the keyframe→target run-in (WITHOUT the expensive
    /// P010→RGBA convert) so the first pair RETURNED is the exact target frame.
    /// Without this, the consumer's stab/RS index `round(t/dt)+n` is off by up
    /// to a GOP after every scrub and the stabilization visibly fights the
    /// content (the preview shakes).
    skip_until_s: Option<f64>,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for D3d11SharedStreamPairIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d11SharedStreamPairIter")
            .field("video_indices", &self.video_indices)
            .field("dims", &self.dims)
            .field("frames_yielded", &self.frames_yielded)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "windows")]
impl D3d11SharedStreamPairIter {
    /// Open a GoPro `.360` and enable `d3d11va` HEVC decode on both EAC
    /// streams. The two video streams (s0, s4) are the first two video
    /// tracks — same selection as `StreamPairIter` / the macOS
    /// `ZeroCopyStreamPairIter`.
    pub fn new(path: &Path) -> Result<Self> {
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
                "expected 2 video streams (.360 EAC), found {}",
                video_indices.len()
            )));
        }
        let video_indices = [video_indices[0], video_indices[1]];
        let mut decoders = Vec::with_capacity(2);
        for &idx in video_indices.iter() {
            let stream = ictx.stream(idx).unwrap();
            let mut codec_ctx =
                ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
                    .map_err(|e| Error::Ffmpeg(format!("codec ctx: {e}")))?;
            if !crate::decode::try_enable_d3d11va_decode(&mut codec_ctx) {
                return Err(Error::Ffmpeg(format!(
                    "zero-copy EAC path requires d3d11va hwaccel — setup failed on stream {idx}"
                )));
            }
            decoders.push(
                codec_ctx.decoder().video()
                    .map_err(|e| Error::Ffmpeg(format!("video decoder: {e}")))?,
            );
        }
        let (w0, h0) = (decoders[0].width(), decoders[0].height());
        let (w1, h1) = (decoders[1].width(), decoders[1].height());
        if (w0, h0) != (w1, h1) {
            return Err(Error::Ffmpeg(format!(
                "EAC zero-copy streams disagree on dims: {w0}x{h0} vs {w1}x{h1}"
            )));
        }
        let dims = vr180_core::eac::Dims::new(w0, h0);
        if !dims.is_valid() {
            return Err(Error::Ffmpeg(format!(
                "EAC zero-copy: stream width {w0} not a valid EAC layout"
            )));
        }
        let s0_stream = ictx.stream(video_indices[0]).unwrap();
        let time_base = s0_stream.time_base();
        let time_base_s = time_base.numerator() as f64 / time_base.denominator().max(1) as f64;
        let fr = s0_stream.avg_frame_rate();
        let dt_s = if fr.numerator() > 0 {
            fr.denominator() as f64 / fr.numerator() as f64
        } else {
            1.0 / 30.0
        };
        tracing::info!(
            "D3d11SharedStreamPairIter: EAC streams {:?}, native {}x{}, cross_w={} \
             (D3D11 P010→RGBA16, no downscale)",
            video_indices, w0, h0, dims.cross_w()
        );
        Ok(Self {
            ictx, video_indices, decoders, dims,
            converters: [None, None],
            frames_yielded: 0,
            time_base_s,
            dt_s,
            skip_until_s: None,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.dims }
    pub fn native_dims(&self) -> (u32, u32) { (self.dims.stream_w, self.dims.stream_h) }

    /// PRECISE seek: the next `next_pair` returns the frame AT `target_s`
    /// (±½ frame). `av_seek_frame` lands on the keyframe ≤ target; arming
    /// `skip_until_s` makes `next_pair` decode-and-discard the keyframe→target
    /// run-in before returning, so the consumer's frame-index-keyed stab/RS
    /// stays aligned after a scrub (otherwise the preview shakes — see the
    /// `skip_until_s` field doc and `ZeroCopyStreamPairIter::seek`).
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let target_s = target_s.max(0.0);
        let ts = (target_s * 1_000_000.0) as i64;
        self.ictx.seek(ts, ..ts)
            .map_err(|e| Error::Ffmpeg(format!("seek {target_s:.3}s: {e}")))?;
        for d in &mut self.decoders { d.flush(); }
        self.frames_yielded = 0;
        self.skip_until_s = Some(target_s);
        Ok(())
    }

    /// Pull the next GPU-resident `(s0, s4)` pair. Decodes one frame from
    /// each stream (held simultaneously), converts + shares each at native
    /// res, then releases the source frames. Returns `Ok(None)` at EOF.
    pub fn next_pair(&mut self) -> Result<Option<SharedEacPair>> {
        // Outer loop so a precise-seek run-in pair can be decoded+dropped and
        // the next pair pulled, all without converting/sharing the discards.
        loop {
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
                    // Feed every packet to its decoder even if that stream's
                    // frame is already in hand, so the decoders stay in lockstep.
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

            let (Some((f0, pts0)), Some((f4, _pts4))) = (frames[0].take(), frames[1].take()) else {
                return Ok(None);
            };
            let pts_s = pts0 as f64 * self.time_base_s;

            // Precise-seek run-in: drop pairs before the target WITHOUT the
            // expensive P010→RGBA convert — only the target frame is converted.
            // (Same convention as `ZeroCopyStreamPairIter`; keeps the consumer's
            // stab/RS frame index aligned after a scrub so the preview doesn't
            // shake.)
            if let Some(target) = self.skip_until_s {
                if pts_s < target - 0.5 * self.dt_s {
                    drop(f0);
                    drop(f4);
                    continue;
                }
                self.skip_until_s = None;
            }

            // Convert each stream P010 → shareable single-plane Rgba16Unorm at
            // NATIVE res (D3D11-side BT.709 YCbCr→RGB, no downscale), then NT
            // share. After this the source frames can drop.
            let (nw, nh) = (self.dims.stream_w, self.dims.stream_h);
            let s0 = unsafe {
                crate::interop_windows::share_eye_converted(&f0, &mut self.converters[0], nw, nh)
            }
            .ok_or_else(|| Error::Ffmpeg("zero-copy EAC: convert s0 failed".into()))?;
            let s4 = unsafe {
                crate::interop_windows::share_eye_converted(&f4, &mut self.converters[1], nw, nh)
            }
            .ok_or_else(|| Error::Ffmpeg("zero-copy EAC: convert s4 failed".into()))?;
            // Both stream converts were submitted on their own d3d11va devices
            // with no intervening fence (concurrent on the GPU). Fence both
            // before handing off — the importer must see completed pixels.
            unsafe { s0.wait_gpu_idle(); s4.wait_gpu_idle(); }
            drop(f0);
            drop(f4);

            self.frames_yielded += 1;
            return Ok(Some(SharedEacPair { s0, s4, dims: self.dims, pts_s }));
        }
    }
}

/// Windows zero-copy counterpart to [`crate::decode::SegmentedStreamPairIter`]
/// — chains GoPro `.360` segments (a long recording GoPro splits into
/// GS01…/GS02…/…) through [`D3d11SharedStreamPairIter`], so a merged clip
/// previews on the fast NVDEC→D3D11→Vulkan path instead of falling back to the
/// CPU-assemble path. Single-element passthrough. Yielded pts are GLOBALIZED to
/// clip time (segment start + local pts) so the consumer's frame-index-keyed
/// stab/RS stays continuous across segment boundaries.
#[cfg(target_os = "windows")]
pub struct SegmentedD3d11SharedStreamPairIter {
    segments: Vec<std::path::PathBuf>,
    seg_start_s: Vec<f64>,
    total_dur_s: f64,
    cur_idx: usize,
    cur: D3d11SharedStreamPairIter,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for SegmentedD3d11SharedStreamPairIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentedD3d11SharedStreamPairIter")
            .field("segments", &self.segments.len())
            .field("cur_idx", &self.cur_idx)
            .field("total_dur_s", &self.total_dur_s)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "windows")]
impl SegmentedD3d11SharedStreamPairIter {
    pub fn new(segments: &[std::path::PathBuf]) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for seg in segments {
            seg_start_s.push(acc);
            acc += crate::decode::probe_video(seg).map(|p| p.duration_sec).unwrap_or(0.0);
        }
        let cur = D3d11SharedStreamPairIter::new(&segments[0])?;
        tracing::info!(
            "SegmentedD3d11SharedStreamPairIter: {} segment(s), {:.1}s total",
            segments.len(), acc,
        );
        Ok(Self {
            segments: segments.to_vec(),
            seg_start_s, total_dur_s: acc, cur_idx: 0, cur,
        })
    }

    pub fn dims(&self) -> vr180_core::eac::Dims { self.cur.dims() }
    pub fn native_dims(&self) -> (u32, u32) { self.cur.native_dims() }
    pub fn total_duration_s(&self) -> f64 { self.total_dur_s }

    /// PRECISE seek to a GLOBAL clip time: map to the owning segment + local
    /// time, (re)open that segment, and precise-seek within it.
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        let t = target_s.clamp(0.0, self.total_dur_s);
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            self.cur = D3d11SharedStreamPairIter::new(&self.segments[idx])?;
            self.cur_idx = idx;
        }
        self.cur.seek(t - self.seg_start_s[idx])
    }

    pub fn next_pair(&mut self) -> Result<Option<SharedEacPair>> {
        loop {
            if let Some(mut p) = self.cur.next_pair()? {
                // Segment-local → global clip time.
                p.pts_s += self.seg_start_s[self.cur_idx];
                return Ok(Some(p));
            }
            if self.cur_idx + 1 >= self.segments.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            self.cur = D3d11SharedStreamPairIter::new(&self.segments[self.cur_idx])?;
        }
    }
}

/// Segment-chaining wrapper around [`D3d11SharedDualStreamIter`] — the OSV
/// analog of [`SegmentedD3d11SharedStreamPairIter`]. A merged (auto-merge)
/// DJI recording's `.osv` parts play as ONE continuous zero-copy stream:
/// yielded pts are rebased onto the global clip timeline (so stab-by-pts,
/// trim and the per-frame IMU index — `parse_multi` concatenates quats
/// across segments — stay aligned across seams), EOF rolls over to the next
/// segment transparently, and seek maps a global time to the owning
/// segment. One entry behaves exactly like the plain iterator. Before this,
/// merged recordings were gated OFF the Windows GPU-resident/zero-copy
/// export arms (2 fps portable loop vs ~35 fps) and the zero-copy preview
/// froze past segment 0.
#[cfg(target_os = "windows")]
pub struct SegmentedD3d11SharedDualStreamIter {
    segments: Vec<std::path::PathBuf>,
    seg_start_s: Vec<f64>,
    total_dur_s: f64,
    cur_idx: usize,
    cur: D3d11SharedDualStreamIter,
    // Re-open parameters for the next segment (same convert/downscale).
    swap_eyes: bool,
    work_w: u32,
    work_h: u32,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for SegmentedD3d11SharedDualStreamIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentedD3d11SharedDualStreamIter")
            .field("segments", &self.segments.len())
            .field("cur_idx", &self.cur_idx)
            .field("total_dur_s", &self.total_dur_s)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "windows")]
impl SegmentedD3d11SharedDualStreamIter {
    pub fn new(
        segments: &[std::path::PathBuf],
        swap_eyes: bool,
        work_w: u32,
        work_h: u32,
    ) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for seg in segments {
            seg_start_s.push(acc);
            acc += crate::decode::probe_video(seg).map(|p| p.duration_sec).unwrap_or(0.0);
        }
        let cur = D3d11SharedDualStreamIter::new(&segments[0], swap_eyes, work_w, work_h)?;
        tracing::info!(
            "SegmentedD3d11SharedDualStreamIter: {} segment(s), {:.1}s total",
            segments.len(), acc,
        );
        Ok(Self {
            segments: segments.to_vec(),
            seg_start_s, total_dur_s: acc, cur_idx: 0, cur,
            swap_eyes, work_w, work_h,
        })
    }

    pub fn eye_dims(&self) -> (u32, u32) { self.cur.eye_dims() }
    pub fn native_dims(&self) -> (u32, u32) { self.cur.native_dims() }
    pub fn total_duration_s(&self) -> f64 { self.total_dur_s }

    fn open_segment(&self, idx: usize) -> Result<D3d11SharedDualStreamIter> {
        D3d11SharedDualStreamIter::new(
            &self.segments[idx], self.swap_eyes, self.work_w, self.work_h)
    }

    /// PRECISE seek to a GLOBAL clip time: map to the owning segment + local
    /// time, (re)open that segment, and precise-seek within it.
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        // If every per-segment probe failed (total 0), don't clamp the
        // target to 0 — pass it through like the plain iterator would.
        let t = if self.total_dur_s > 0.0 {
            target_s.clamp(0.0, self.total_dur_s)
        } else {
            target_s.max(0.0)
        };
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            self.cur = self.open_segment(idx)?;
            self.cur_idx = idx;
        }
        self.cur.seek(t - self.seg_start_s[idx])
    }

    pub fn next_pair(&mut self) -> Result<Option<SharedFisheyePair>> {
        loop {
            if let Some(mut p) = self.cur.next_pair()? {
                // Segment-local → global clip time.
                p.pts_s += self.seg_start_s[self.cur_idx];
                return Ok(Some(p));
            }
            if self.cur_idx + 1 >= self.segments.len() {
                return Ok(None);
            }
            self.cur_idx += 1;
            let expect = self.cur.native_dims();
            self.cur = self.open_segment(self.cur_idx)?;
            // Downstream textures/calib are sized off segment 0 — a chain
            // with mismatched native dims would render garbage. Real DJI
            // chains never differ; warn loudly if one somehow does (same
            // stance as the macOS zero-copy iter's reopen).
            if self.cur.native_dims() != expect {
                tracing::warn!(
                    "segmented OSV: segment {} native dims {:?} != {:?} — \
                     output may be wrong",
                    self.cur_idx, self.cur.native_dims(), expect);
            }
        }
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
    /// Segment chain (a merged recording's OSV parts). One entry for a lone
    /// file. The decoder transparently rolls over to the next segment at EOF,
    /// so a consumer (incl. the temporal denoiser) sees ONE continuous stream.
    seg_paths: Vec<std::path::PathBuf>,
    /// Cumulative clip-time start of each segment (s); yielded pts are rebased
    /// onto this GLOBAL timeline so stab-by-pts + trim stay aligned across seams.
    seg_start_s: Vec<f64>,
    cur_idx: usize,
}

/// Open one OSV/dual-camera file's two VT-hwaccel video decoders. Shared by the
/// single-file and segmented constructors and by the segment roll-over.
#[cfg(target_os = "macos")]
fn open_dual_stream_vt(path: &Path) -> Result<(
    ffmpeg_next::format::context::Input,
    [usize; 2],
    Vec<ffmpeg_next::codec::decoder::Video>,
    u32, u32, f64,
)> {
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
    Ok((ictx, video_indices, decoders, w0, h0, time_base_s))
}

#[cfg(target_os = "macos")]
impl ZeroCopyDualStreamFisheyeIter {
    pub fn new(path: &Path, frame_limit: u32, swap_eyes: bool) -> Result<Self> {
        let (ictx, video_indices, decoders, eye_w, eye_h, time_base_s) = open_dual_stream_vt(path)?;
        tracing::info!(
            "ZeroCopyDualStreamFisheyeIter: native {}x{}, swap_eyes={}",
            eye_w, eye_h, swap_eyes
        );
        Ok(Self {
            ictx, video_indices, decoders,
            eye_w, eye_h,
            frame_limit, frames_yielded: 0,
            time_base_s,
            swap_eyes,
            seg_paths: vec![path.to_path_buf()],
            seg_start_s: vec![0.0],
            cur_idx: 0,
        })
    }

    /// Chain several OSV segments of one recording into one continuous zero-copy
    /// stream. `seg_durations_s` are per-segment durations (used to rebase pts +
    /// map a global seek to a segment). Yielded pts are GLOBAL (chain-relative).
    pub fn new_segmented(
        segments: &[std::path::PathBuf],
        seg_durations_s: &[f64],
        frame_limit: u32,
        swap_eyes: bool,
    ) -> Result<Self> {
        assert!(!segments.is_empty(), "segments must be non-empty");
        let (ictx, video_indices, decoders, eye_w, eye_h, time_base_s) =
            open_dual_stream_vt(&segments[0])?;
        let mut seg_start_s = Vec::with_capacity(segments.len());
        let mut acc = 0.0_f64;
        for i in 0..segments.len() {
            seg_start_s.push(acc);
            acc += seg_durations_s.get(i).copied().unwrap_or(0.0);
        }
        tracing::info!(
            "ZeroCopyDualStreamFisheyeIter: {} segments ({:.1}s), native {}x{}, swap_eyes={}",
            segments.len(), acc, eye_w, eye_h, swap_eyes
        );
        Ok(Self {
            ictx, video_indices, decoders,
            eye_w, eye_h,
            frame_limit, frames_yielded: 0,
            time_base_s,
            swap_eyes,
            seg_paths: segments.to_vec(),
            seg_start_s,
            cur_idx: 0,
        })
    }

    /// Re-open the decoder onto `path` (a segment roll-over or a cross-segment
    /// seek). Keeps the recording's eye dims (warns if a segment disagrees).
    fn reopen(&mut self, path: &Path) -> Result<()> {
        let (ictx, video_indices, decoders, w, h, time_base_s) = open_dual_stream_vt(path)?;
        if (w, h) != (self.eye_w, self.eye_h) {
            tracing::warn!(
                "ZeroCopy segment {path:?} dims {w}x{h} != first {}x{} — keeping first",
                self.eye_w, self.eye_h
            );
        }
        self.ictx = ictx;
        self.video_indices = video_indices;
        self.decoders = decoders;
        self.time_base_s = time_base_s;
        Ok(())
    }

    pub fn eye_dims(&self) -> (u32, u32) { (self.eye_w, self.eye_h) }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        // Map the global target to (segment, local time) across the chain.
        let t = target_s.max(0.0);
        let idx = self.seg_start_s.iter().rposition(|&s| s <= t).unwrap_or(0);
        if idx != self.cur_idx {
            let path = self.seg_paths[idx].clone();
            self.reopen(&path)?;
            self.cur_idx = idx;
        }
        let local = (t - self.seg_start_s.get(idx).copied().unwrap_or(0.0)).max(0.0);
        let ts = (local * 1_000_000.0) as i64;
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
    /// Decode the next paired frame from both streams. Shared by
    /// [`next_pair`](Self::next_pair) (texture wrapping, the clean path) and
    /// [`next_p010_pair`](Self::next_p010_pair) (raw `CVPixelBuffer`s for the
    /// zero-copy denoise path). Returns the two VT frames + clip-time pts.
    /// Decode the next pair, rolling over to the next segment at EOF and
    /// rebasing the pts onto the global chain timeline.
    fn decode_frame_pair(
        &mut self,
    ) -> Result<Option<([ffmpeg_next::frame::Video; 2], f64)>> {
        loop {
            match self.decode_one_segment()? {
                Some(([f0, f1], local_pts)) => {
                    let pts_s = local_pts
                        + self.seg_start_s.get(self.cur_idx).copied().unwrap_or(0.0);
                    return Ok(Some(([f0, f1], pts_s)));
                }
                None => {
                    if self.cur_idx + 1 >= self.seg_paths.len() {
                        return Ok(None);
                    }
                    self.cur_idx += 1;
                    let path = self.seg_paths[self.cur_idx].clone();
                    tracing::info!(
                        "ZeroCopyDualStreamFisheyeIter: → segment {}/{}",
                        self.cur_idx + 1, self.seg_paths.len()
                    );
                    self.reopen(&path)?;
                }
            }
        }
    }

    /// Decode one pair from the CURRENT segment (local pts). `None` at this
    /// segment's EOF — [`decode_frame_pair`] then advances the chain.
    fn decode_one_segment(
        &mut self,
    ) -> Result<Option<([ffmpeg_next::frame::Video; 2], f64)>> {
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
        self.frames_yielded += 1;
        Ok(Some(([f0, f1], pts_s)))
    }

    /// Pull the next pair as raw VT-decoded P010 `CVPixelBuffer`s (retained),
    /// for the zero-copy denoise path. Same swap convention as
    /// [`next_pair`](Self::next_pair). The buffers feed
    /// `VTPixelTransferSession`; the caller may drop them once denoised.
    #[allow(clippy::type_complexity)]
    pub fn next_p010_pair(
        &mut self,
    ) -> Result<Option<(
        crate::interop_macos::RetainedCVPixelBuffer,
        crate::interop_macos::RetainedCVPixelBuffer,
        f64,
    )>> {
        use crate::interop_macos::extract_cvpixelbuffer_from_vt_frame;
        let ([f0, f1], pts_s) = match self.decode_frame_pair()? {
            Some(x) => x,
            None => return Ok(None),
        };
        let pb0 = extract_cvpixelbuffer_from_vt_frame(&f0)?;
        let pb1 = extract_cvpixelbuffer_from_vt_frame(&f1)?;
        let (left, right) = if self.swap_eyes { (pb1, pb0) } else { (pb0, pb1) };
        Ok(Some((left, right, pts_s)))
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
        let ([f0, f1], pts_s) = match self.decode_frame_pair()? {
            Some(x) => x,
            None => return Ok(None),
        };

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

        Ok(Some(ZeroCopyFisheyePair {
            left_y, left_uv, right_y, right_uv,
            eye_w: self.eye_w, eye_h: self.eye_h,
            pts_s,
        }))
    }
}

/// Repackage a pair of denoised P010 [`EncodePixelBufferP010`]s as a
/// [`ZeroCopyFisheyePair`] — splitting each into its Y + UV plane textures.
/// The plane textures retain the IOSurface, so the source CVPixelBuffer can
/// drop while the pixels stay alive for the projection dispatch.
#[cfg(target_os = "macos")]
fn epb_pair_to_zerocopy(
    left: crate::interop_macos::EncodePixelBufferP010,
    right: crate::interop_macos::EncodePixelBufferP010,
    eye_w: u32,
    eye_h: u32,
    pts_s: f64,
) -> ZeroCopyFisheyePair {
    use crate::interop_macos::{EncodePixelBufferP010, IOSurfacePlaneTexture};
    fn split(epb: EncodePixelBufferP010) -> (IOSurfacePlaneTexture, IOSurfacePlaneTexture) {
        let (w, h) = (epb.width, epb.height);
        let y = IOSurfacePlaneTexture {
            surface: epb.iosurface.clone(),
            texture: epb.y_tex,
            width: w,
            height: h,
            format: wgpu::TextureFormat::R16Unorm,
        };
        let uv = IOSurfacePlaneTexture {
            surface: epb.iosurface,
            texture: epb.uv_tex,
            width: w / 2,
            height: h / 2,
            format: wgpu::TextureFormat::Rg16Unorm,
        };
        (y, uv)
    }
    let (left_y, left_uv) = split(left);
    let (right_y, right_uv) = split(right);
    ZeroCopyFisheyePair { left_y, left_uv, right_y, right_uv, eye_w, eye_h, pts_s }
}

/// Wraps [`ZeroCopyDualStreamFisheyeIter`] and runs each eye's P010 frames
/// through an in-process [`P010Denoiser`](crate::vt_denoise::P010Denoiser)
/// — temporal noise reduction with NO CPU readback, so the zero-copy OSV
/// export keeps its speed. Yields denoised [`ZeroCopyFisheyePair`]s with the
/// same shape as the raw iterator, so the export loop is unchanged. Output
/// count/order preserved 1:1 (delayed by the filter look-ahead, drained at
/// EOF); pts mapped FIFO.
#[cfg(target_os = "macos")]
pub struct DenoisingZeroCopyIter {
    inner: ZeroCopyDualStreamFisheyeIter,
    left: crate::vt_denoise::P010Denoiser,
    right: crate::vt_denoise::P010Denoiser,
    eye_w: u32,
    eye_h: u32,
    strength: f32,
    pending_pts: std::collections::VecDeque<f64>,
    ready: std::collections::VecDeque<ZeroCopyFisheyePair>,
    inner_done: bool,
    /// Don't denoise frames before this pts (the trim-in, set by `seek`): pre-trim
    /// frames are still DECODED (P-frame dependency) but skipping their denoise
    /// avoids burning GPU on frames the export's trim discards — which is what
    /// made a deep trim_in take a long time to produce the first output frame.
    /// `NEG_INFINITY` = denoise everything.
    skip_until_pts: f64,
    n_skipped: u64,
}

#[cfg(target_os = "macos")]
impl DenoisingZeroCopyIter {
    pub fn new(inner: ZeroCopyDualStreamFisheyeIter, strength: f32) -> Result<Self> {
        let (eye_w, eye_h) = inner.eye_dims();
        let left = crate::vt_denoise::P010Denoiser::new(eye_w, eye_h, strength)?;
        let right = crate::vt_denoise::P010Denoiser::new(eye_w, eye_h, strength)?;
        Ok(Self {
            inner,
            left,
            right,
            eye_w,
            eye_h,
            strength,
            pending_pts: std::collections::VecDeque::new(),
            ready: std::collections::VecDeque::new(),
            inner_done: false,
            skip_until_pts: f64::NEG_INFINITY,
            n_skipped: 0,
        })
    }

    pub fn eye_dims(&self) -> (u32, u32) {
        (self.eye_w, self.eye_h)
    }

    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        self.inner.seek(target_s)?;
        // Temporal window is meaningless across a seek — rebuild the
        // denoisers so the next frame restarts the filter cleanly.
        self.left = crate::vt_denoise::P010Denoiser::new(self.eye_w, self.eye_h, self.strength)?;
        self.right = crate::vt_denoise::P010Denoiser::new(self.eye_w, self.eye_h, self.strength)?;
        self.pending_pts.clear();
        self.ready.clear();
        self.inner_done = false;
        // The seek target is the trim-in: skip denoising everything before it
        // (decode-only) so a deep trim doesn't denoise thousands of dropped frames.
        self.skip_until_pts = target_s;
        self.n_skipped = 0;
        Ok(())
    }

    pub fn next_pair(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Option<ZeroCopyFisheyePair>> {
        loop {
            if let Some(p) = self.ready.pop_front() {
                return Ok(Some(p));
            }
            if self.inner_done {
                return Ok(None);
            }
            match self.inner.next_p010_pair()? {
                Some((l_src, r_src, pts)) => {
                    // Pre-trim frames: decode is required (P-frame chain) but skip
                    // the denoise — the export's trim discards them anyway, and the
                    // filter restarts at the trim point regardless.
                    if pts < self.skip_until_pts {
                        drop(l_src);
                        drop(r_src);
                        self.n_skipped += 1;
                        continue;
                    }
                    if self.n_skipped > 0 {
                        tracing::info!(
                            "DenoisingZeroCopyIter: skipped denoise on {} pre-trim frames (decode-only)",
                            self.n_skipped
                        );
                        self.n_skipped = 0;
                    }
                    self.pending_pts.push_back(pts);
                    let louts = self.left.push(device, l_src.as_raw())?;
                    let routs = self.right.push(device, r_src.as_raw())?;
                    // Source P010 buffers are now converted into the windows.
                    drop(l_src);
                    drop(r_src);
                    for (lo, ro) in louts.into_iter().zip(routs.into_iter()) {
                        let pts = self.pending_pts.pop_front().unwrap_or(0.0);
                        self.ready.push_back(epb_pair_to_zerocopy(
                            lo, ro, self.eye_w, self.eye_h, pts,
                        ));
                    }
                }
                None => {
                    self.inner_done = true;
                    let louts = self.left.finish(device)?;
                    let routs = self.right.finish(device)?;
                    for (lo, ro) in louts.into_iter().zip(routs.into_iter()) {
                        let pts = self.pending_pts.pop_front().unwrap_or(0.0);
                        self.ready.push_back(epb_pair_to_zerocopy(
                            lo, ro, self.eye_w, self.eye_h, pts,
                        ));
                    }
                }
            }
        }
    }
}

/// Either the raw zero-copy decoder or the denoising wrapper — lets the
/// export loop call `next_pair`/`seek`/`eye_dims` uniformly.
#[cfg(target_os = "macos")]
pub enum ZcDecoder {
    Raw(ZeroCopyDualStreamFisheyeIter),
    Denoising(DenoisingZeroCopyIter),
}

#[cfg(target_os = "macos")]
impl ZcDecoder {
    pub fn eye_dims(&self) -> (u32, u32) {
        match self {
            Self::Raw(d) => d.eye_dims(),
            Self::Denoising(d) => d.eye_dims(),
        }
    }
    pub fn seek(&mut self, target_s: f64) -> Result<()> {
        match self {
            Self::Raw(d) => d.seek(target_s),
            Self::Denoising(d) => d.seek(target_s),
        }
    }
    pub fn next_pair(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Option<ZeroCopyFisheyePair>> {
        match self {
            Self::Raw(d) => d.next_pair(device),
            Self::Denoising(d) => d.next_pair(device),
        }
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

// ── Temporal noise reduction wrapper (macOS VideoToolbox) ─────────
//
// Wraps any [`FisheyePairIter`] and runs each eye's frames through an
// in-process `VTTemporalNoiseFilter` (see [`crate::vt_denoise`]) — the
// same algorithm the Python app reaches through its Swift helper, matched
// here without a subprocess. Source-domain denoise (before projection),
// mirroring the Python pipeline. Output order and frame count are
// preserved 1:1 (delayed by the filter's look-ahead, drained at EOF), so
// stabilization-by-index and audio sync downstream stay aligned.
#[cfg(target_os = "macos")]
pub struct DenoisingFisheyeIter {
    inner: Box<dyn FisheyePairIter>,
    strength: f32,
    eye_w: u32,
    eye_h: u32,
    bit_depth: u8,
    // Lazily created on the first pair (we need its bit depth).
    left: Option<crate::vt_denoise::VtDenoiser>,
    right: Option<crate::vt_denoise::VtDenoiser>,
    pending_pts: std::collections::VecDeque<f64>,
    ready: std::collections::VecDeque<FisheyePair>,
    /// First pair, kept only as the passthrough fallback for the
    /// degenerate single-frame clip (no temporal reference possible).
    first_pair: Option<FisheyePair>,
    pushed: usize,
    emitted: usize,
    inner_done: bool,
}

#[cfg(target_os = "macos")]
impl DenoisingFisheyeIter {
    /// Wrap `inner`, denoising at `strength` (0.0–1.0). Cheap — the
    /// VideoToolbox sessions are built lazily on the first frame.
    pub fn new(inner: Box<dyn FisheyePairIter>, strength: f32) -> Result<Self> {
        let (eye_w, eye_h) = inner.eye_dims();
        Ok(Self {
            inner,
            strength,
            eye_w,
            eye_h,
            bit_depth: 8,
            left: None,
            right: None,
            pending_pts: std::collections::VecDeque::new(),
            ready: std::collections::VecDeque::new(),
            first_pair: None,
            pushed: 0,
            emitted: 0,
            inner_done: false,
        })
    }

    fn ensure_denoisers(&mut self, bit_depth: u8) -> Result<()> {
        if self.left.is_none() {
            self.bit_depth = bit_depth;
            self.left = Some(crate::vt_denoise::VtDenoiser::new(
                self.eye_w, self.eye_h, bit_depth, self.strength,
            )?);
            self.right = Some(crate::vt_denoise::VtDenoiser::new(
                self.eye_w, self.eye_h, bit_depth, self.strength,
            )?);
        }
        Ok(())
    }

    fn emit(&mut self, lefts: Vec<Vec<u8>>, rights: Vec<Vec<u8>>) {
        for (l, r) in lefts.into_iter().zip(rights.into_iter()) {
            let pts = self.pending_pts.pop_front().unwrap_or(0.0);
            self.emitted += 1;
            self.ready.push_back(FisheyePair {
                left: l,
                right: r,
                eye_w: self.eye_w,
                eye_h: self.eye_h,
                bit_depth: self.bit_depth,
                pts_s: pts,
            });
        }
    }
}

#[cfg(target_os = "macos")]
impl FisheyePairIter for DenoisingFisheyeIter {
    fn next_pair(&mut self) -> Result<Option<FisheyePair>> {
        loop {
            if let Some(p) = self.ready.pop_front() {
                return Ok(Some(p));
            }
            if self.inner_done {
                return Ok(None);
            }
            match self.inner.next_pair()? {
                Some(pair) => {
                    self.ensure_denoisers(pair.bit_depth)?;
                    if self.pushed == 0 {
                        self.first_pair = Some(pair.clone());
                    }
                    self.pushed += 1;
                    self.pending_pts.push_back(pair.pts_s);
                    let lefts = self.left.as_mut().unwrap().push(&pair.left)?;
                    let rights = self.right.as_mut().unwrap().push(&pair.right)?;
                    self.emit(lefts, rights);
                }
                None => {
                    self.inner_done = true;
                    if let (Some(l), Some(r)) = (self.left.as_mut(), self.right.as_mut()) {
                        let lefts = l.finish()?;
                        let rights = r.finish()?;
                        self.emit(lefts, rights);
                    }
                    // Single-frame clip: nothing could be denoised — pass the
                    // one frame through untouched so the count is preserved.
                    if self.emitted == 0 {
                        if let Some(fp) = self.first_pair.take() {
                            self.ready.push_back(fp);
                        }
                    }
                }
            }
        }
    }

    fn seek(&mut self, target_s: f64) -> Result<()> {
        self.inner.seek(target_s)?;
        // The temporal window is meaningless across a seek — reset so the
        // next frame restarts the filter (fresh discontinuity).
        self.left = None;
        self.right = None;
        self.pending_pts.clear();
        self.ready.clear();
        self.first_pair = None;
        self.pushed = 0;
        self.emitted = 0;
        self.inner_done = false;
        Ok(())
    }

    fn eye_dims(&self) -> (u32, u32) {
        (self.eye_w, self.eye_h)
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
