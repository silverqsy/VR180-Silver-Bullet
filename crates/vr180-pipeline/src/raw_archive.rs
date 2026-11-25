//! An archive copy of a raw take: the take's folder again, with a capture a
//! fraction of the size, which the app opens like the take itself.
//!
//! The camera writes a gigabyte a second and packs its samples with what a
//! processor can do at that pace. A computer has the time to do better, and
//! a copy that need not hold every sample exactly can be far smaller still:
//!
//! 1. **The codes.** A sample's value is turned into one of 1024 codes
//!    ([`Curve`]). The codes lie where the noise needs them: two values the
//!    sensor's noise cannot tell apart share a code, two it can are several
//!    codes apart, at every level alike. The noise is measured on the take
//!    ([`noise_of`]), once: between frames that follow each other, on the
//!    flat parts of the picture. A take whose values come from a coarse
//!    table gets a few codes for every level of the table.
//! 2. **The pictures.** Each of a frame's four colour planes is a picture of
//!    its own, half the frame's width and height, and is stored as one
//!    intra-coded HEVC picture of 10 bits ([`x6b::ARCHIVE_HEVC_PLANES`]):
//!    the graphics hardware writes and reads those, on both systems. Every
//!    picture stands alone, so a frame is read without the ones before it.
//! 3. **The size.** The encoder works at one quality for the whole take,
//!    found before the copy starts as the one at which a few of the take's
//!    frames come to the bits a pixel asked for ([`archive_take`]).
//!
//! The container is [`vr180_fisheye::x6b`]'s (a record a pair, the files
//! beside the capture copied as they are); this module makes the pictures
//! and reads them back.

use std::collections::VecDeque;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ffmpeg_next as ffmpeg;
use vr180_fisheye::x6b::{self, ArchiveHeader, Channels, Layout, PairEntry, Profile, ARCHIVE_PICTURES};

use crate::x6_raw;
use crate::{Error, Result};

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Ffmpeg(format!("archive copy: {msg}"))
}

// ── The codes ──────────────────────────────────────────────────────────

/// Codes a sample is stored in.
const CODES: usize = x6b::ARCHIVE_CODES;
/// Codes left free at each end: a picture read back may stray past the
/// last value's code.
const MARGIN: f64 = 32.0;
/// Codes two neighbouring values are apart at most.
const APART_MOST: f64 = 8.0;
/// Codes the white level is above the value below it: a sample the sensor
/// could not go beyond comes back as that level and no other, and one below
/// it does not come back as it. (A picture read back is a code or two off
/// on a flat part, and more beside an edge.)
const WHITE_APART: f64 = 40.0;

/// How a take's sample values become codes and codes values.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    /// The code of every value from 0 up to the largest a sample can have.
    code: Vec<u16>,
    /// The value each code stands for.
    value: Vec<u16>,
}

impl Curve {
    /// The codes for a take whose samples can have `values` (rising), of
    /// which those from `lowest` up to `white`, the level of a saturated
    /// sample, each need a place (smaller ones share `lowest`'s, larger
    /// ones `white`'s). `noise` = `(a, b)`: a sample of value `v` has a
    /// noise of variance `a · max(v − black, 0) + b`.
    ///
    /// Two neighbouring values are as many codes apart as they are noise
    /// deviations apart, times one factor for the whole curve, and never
    /// more than [`APART_MOST`]; the factor is the largest at which all the
    /// values fit. The white level alone stands [`WHITE_APART`] codes off.
    pub fn new(values: &[u16], lowest: u16, black: u16, white: u16, noise: (f64, f64)) -> Self {
        let mut kept: Vec<u16> = values.iter().copied().filter(|v| *v >= lowest && *v < white).collect();
        kept.push(white);
        let sigma = |v: u16| (noise.0 * (v as f64 - black as f64).max(0.0) + noise.1).max(1e-9).sqrt();
        // Every step but the last, which is the one up to white.
        let apart: Vec<f64> = kept[..kept.len() - 1].windows(2).map(|w| (w[1] - w[0]) as f64 / sigma(w[0])).collect();
        let to_white = if kept.len() > 1 { WHITE_APART } else { 0.0 };
        let room = CODES as f64 - 1.0 - 2.0 * MARGIN - to_white;
        let total = |g: f64| apart.iter().map(|&x| (g * x).min(APART_MOST)).sum::<f64>();
        let factor = if apart.len() as f64 * APART_MOST <= room {
            f64::INFINITY
        } else {
            // The sum rises with the factor: halve the interval that holds the answer.
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            while total(hi) < room {
                hi *= 2.0;
            }
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if total(mid) <= room { lo = mid } else { hi = mid }
            }
            lo
        };
        let mut place = Vec::with_capacity(kept.len());
        let mut at = MARGIN;
        place.push(at);
        for &x in &apart {
            at += if factor.is_finite() { (factor * x).min(APART_MOST) } else { APART_MOST };
            place.push(at);
        }
        if kept.len() > 1 {
            place.push(at + to_white);
        }
        // A value's code: its place, rounded; a value below the lowest kept
        // one has that one's, a value between two kept ones the lower one's.
        let top = kept.last().copied().unwrap_or(0) as usize;
        let mut code = vec![0u16; top + 1];
        let mut k = 0usize;
        for (v, c) in code.iter_mut().enumerate() {
            while k + 1 < kept.len() && kept[k + 1] as usize <= v {
                k += 1;
            }
            *c = place[k].round() as u16;
        }
        // A code's value: the kept value whose place is nearest.
        let mut value = vec![0u16; CODES];
        let mut k = 0usize;
        for (c, v) in value.iter_mut().enumerate() {
            while k + 1 < kept.len() && (place[k + 1] - c as f64).abs() <= (place[k] - c as f64).abs() {
                k += 1;
            }
            *v = kept[k];
        }
        Self { code, value }
    }

    /// The code of `value` (a value past the largest has the largest's).
    pub fn code_of(&self, value: u16) -> u16 {
        self.code[(value as usize).min(self.code.len() - 1)]
    }

    /// The value each code stands for.
    pub fn values(&self) -> &[u16] {
        &self.value
    }
}

/// The noise of a take's samples, `(a, b)` of variance `a · (v − black) + b`,
/// measured on `frames`: pairs of frames that follow each other, each the
/// take's channels one after the other (`width × height` samples a channel).
///
/// On the flat parts of the picture two frames a thirtieth of a second apart
/// differ by their noise alone. A sample is taken where its same-colour
/// neighbours differ least in both frames (the flattest three in ten of each
/// level); the differences are sorted by level, the variance of each level
/// is half that of its differences, and a line is laid through the levels.
/// A level whose differences are far above the line (something moved there)
/// is left out of a second fit.
pub fn noise_of(frames: &[(&[u16], &[u16])], width: usize, height: usize, black: u16, white: u16) -> (f64, f64) {
    let range = (white.saturating_sub(black)).max(1) as f64;
    let unit = range / 512.0;
    // What a take whose noise cannot be measured is given: that of a 10-bit
    // frame of this sensor in daylight, on the take's own scale.
    let scale = range / 959.0;
    let unmeasured = (0.0256 * scale, 0.268 * scale * scale);
    const BINS: usize = 12;
    let mut bins: Vec<Vec<(f32, f32, f32)>> = vec![Vec::new(); BINS];
    let per = width * height;
    for (first, second) in frames {
        for (a, b) in first.chunks_exact(per).zip(second.chunks_exact(per)) {
            // Every third site of every third row: all four colours come by.
            for y in (2..height.saturating_sub(2)).step_by(3) {
                let row = y * width;
                for x in (2..width.saturating_sub(2)).step_by(3) {
                    let i = row + x;
                    let (c0, c1) = (a[i] as f64, b[i] as f64);
                    if c0.max(c1) >= 0.95 * white as f64 {
                        continue;
                    }
                    let flat = |f: &[u16]| (f[i + 2] as f64 - f[i - 2] as f64).abs() + (f[i + 2 * width] as f64 - f[i - 2 * width] as f64).abs();
                    let level = 0.5 * (c0 + c1) - black as f64;
                    let bin = if level < unit { 0 } else { ((level / unit).log2() as usize + 1).min(BINS - 1) };
                    bins[bin].push((level as f32, (flat(a) + flat(b)) as f32, (c0 - c1) as f32));
                }
            }
        }
    }
    // (level, variance, weight) of each level that has enough samples.
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    for bin in &mut bins {
        if bin.len() < 2000 {
            continue;
        }
        let cut = bin.len() * 3 / 10;
        bin.select_nth_unstable_by(cut, |p, q| p.1.total_cmp(&q.1));
        let flat = &bin[..cut];
        let mut kept: Vec<f64> = flat.iter().map(|p| p.2 as f64).collect();
        for _ in 0..3 {
            let n = kept.len().max(1) as f64;
            let mean = kept.iter().sum::<f64>() / n;
            let sd = (kept.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n).sqrt();
            let inside: Vec<f64> = kept.iter().copied().filter(|d| (d - mean).abs() <= 3.0 * sd.max(1e-6)).collect();
            if inside.len() == kept.len() || inside.len() < 100 {
                break;
            }
            kept = inside;
        }
        let n = kept.len() as f64;
        let mean = kept.iter().sum::<f64>() / n;
        let variance = kept.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n / 2.0;
        let level = flat.iter().map(|p| p.0 as f64).sum::<f64>() / flat.len() as f64;
        points.push((level.max(0.0), variance, n));
    }
    if points.len() < 2 {
        return unmeasured;
    }
    let fit = |points: &[(f64, f64, f64)]| -> (f64, f64) {
        // Weighted by the count over the variance squared: a level's variance is known to a share of itself.
        let w = |p: &(f64, f64, f64)| p.2 / (p.1 * p.1).max(1e-12);
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for p in points {
            let k = w(p);
            sw += k; sx += k * p.0; sy += k * p.1; sxx += k * p.0 * p.0; sxy += k * p.0 * p.1;
        }
        let den = sw * sxx - sx * sx;
        if den.abs() < 1e-12 {
            return (0.0, sy / sw.max(1e-12));
        }
        let a = (sw * sxy - sx * sy) / den;
        if a <= 0.0 {
            return (0.0, sy / sw.max(1e-12));
        }
        (a, ((sy - a * sx) / sw).max(0.0))
    };
    let first = fit(&points);
    let still: Vec<(f64, f64, f64)> = points.iter().copied().filter(|p| p.1 <= 2.5 * (first.0 * p.0 + first.1).max(1e-12)).collect();
    let (a, b) = if still.len() >= 2 { fit(&still) } else { first };
    // A noise of nothing at black would put every dark value codes apart for no reason the picture has.
    (a, b.max(1e-4 * scale * scale))
}

// ── The pictures ───────────────────────────────────────────────────────

/// Row and column a colour plane starts at in the mosaic.
fn plane_origin(plane: usize) -> (usize, usize) {
    (plane / 2, plane % 2)
}

/// Colour plane `plane` of a mosaic of `width × height`, as codes.
fn plane_codes(mosaic: &[u16], width: usize, height: usize, plane: usize, curve: &Curve, out: &mut Vec<u16>) {
    let (dy, dx) = plane_origin(plane);
    out.clear();
    out.reserve(width / 2 * (height / 2));
    for y in (dy..height).step_by(2) {
        let row = &mosaic[y * width..(y + 1) * width];
        out.extend(row[dx..].iter().step_by(2).map(|&v| curve.code_of(v)));
    }
}

/// A picture read back: its frame in memory, and how far up in their
/// sixteen bits its ten-bit codes sit.
struct Picture {
    frame: ffmpeg::frame::Video,
    shift: u32,
}

impl Picture {
    /// Row `r` of the picture's codes, as the bytes of `width` words.
    fn row(&self, r: usize, width: usize) -> &[u8] {
        let stride = self.frame.stride(0);
        &self.frame.data(0)[r * stride..r * stride + 2 * width]
    }
}

/// The four planes' pictures back into a channel's mosaic as values.
fn put_planes(planes: &[Picture], width: usize, height: usize, values: &[u16], mosaic: &mut [u16]) {
    let (pw, top) = (width / 2, values.len() - 1);
    let rows = height / 2;
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8).min(rows.max(1));
    let band = rows.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (b, part) in mosaic.chunks_mut(band * 2 * width).enumerate() {
            scope.spawn(move || {
                for (r, two) in part.chunks_mut(2 * width).enumerate() {
                    let at = b * band + r;
                    // A mosaic row takes its even samples from one plane and its odd ones from the next.
                    for (line, pair) in two.chunks_mut(width).zip(planes.chunks(2)) {
                        let (even, odd) = (&pair[0], &pair[1]);
                        for ((px, e), o) in line.chunks_exact_mut(2).zip(even.row(at, pw).chunks_exact(2)).zip(odd.row(at, pw).chunks_exact(2)) {
                            px[0] = values[((u16::from_le_bytes([e[0], e[1]]) >> even.shift) as usize).min(top)];
                            px[1] = values[((u16::from_le_bytes([o[0], o[1]]) >> odd.shift) as usize).min(top)];
                        }
                    }
                }
            });
        }
    });
}

/// The encoders a picture may be written by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureEncoder {
    /// Apple's hardware encoder.
    VideoToolbox,
    /// NVIDIA's hardware encoder.
    Nvenc,
    /// The software encoder: any system, far slower.
    X265,
}

impl PictureEncoder {
    /// The encoders in the order they are tried: the graphics hardware's,
    /// then the software one.
    pub const ALL: [Self; 3] = [Self::VideoToolbox, Self::Nvenc, Self::X265];

    pub fn name(self) -> &'static str {
        match self {
            Self::VideoToolbox => "hevc_videotoolbox",
            Self::Nvenc => "hevc_nvenc",
            Self::X265 => "libx265",
        }
    }

    /// The encoder's own quality setting at `quality` (0 = smallest, 1 =
    /// finest), as the number it is given.
    fn setting(self, quality: f64) -> f64 {
        let q = quality.clamp(0.0, 1.0);
        match self {
            Self::VideoToolbox => 30.0 + 70.0 * q,
            Self::Nvenc => (45.0 - 35.0 * q).round(),
            Self::X265 => 34.0 - 30.0 * q,
        }
    }
}

unsafe fn set_opt(ctx: *mut ffmpeg::ffi::AVCodecContext, name: &str, value: &str) -> Result<()> {
    let (name_c, value_c) = (CString::new(name).map_err(err)?, CString::new(value).map_err(err)?);
    let ret = ffmpeg::ffi::av_opt_set((*ctx).priv_data, name_c.as_ptr(), value_c.as_ptr(), 0);
    if ret < 0 {
        return Err(err(format!("the encoder does not take {name} = {value} ({ret})")));
    }
    Ok(())
}

/// One encoder at work on pictures of one size: codes in, access units out
/// in the order the pictures went in.
struct Encoder {
    kind: PictureEncoder,
    ctx: ffmpeg::encoder::Video,
    width: usize,
    height: usize,
    sent: i64,
    /// The parameter sets, as Annex B, where the encoder keeps them out of
    /// its pictures.
    sets: Vec<u8>,
    /// The access units that have come out, each with the number of its
    /// picture, in the order the pictures went in.
    ready: VecDeque<(i64, Vec<u8>)>,
}

impl Encoder {
    fn open(kind: PictureEncoder, width: usize, height: usize, quality: f64) -> Result<Self> {
        crate::decode::init();
        let codec = ffmpeg::codec::encoder::find_by_name(kind.name()).ok_or_else(|| err(format!("no {} in this build", kind.name())))?;
        let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
        let setting = kind.setting(quality);
        unsafe {
            let raw = ctx.as_mut_ptr();
            (*raw).width = width as i32;
            (*raw).height = height as i32;
            (*raw).pix_fmt = match kind {
                PictureEncoder::X265 => ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_YUV420P10LE,
                _ => ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_P010LE,
            };
            (*raw).time_base = ffmpeg::Rational(1, 30).into();
            (*raw).framerate = ffmpeg::Rational(30, 1).into();
            // Every picture stands alone.
            (*raw).gop_size = 1;
            (*raw).max_b_frames = 0;
            (*raw).color_range = ffmpeg::ffi::AVColorRange::AVCOL_RANGE_JPEG;
            match kind {
                PictureEncoder::VideoToolbox => {
                    (*raw).flags |= ffmpeg::ffi::AV_CODEC_FLAG_QSCALE as i32;
                    (*raw).global_quality = (setting * ffmpeg::ffi::FF_QP2LAMBDA as f64).round() as i32;
                    set_opt(raw, "realtime", "0")?;
                    set_opt(raw, "allow_sw", "0")?;
                    set_opt(raw, "profile", "main10")?;
                    set_opt(raw, "power_efficient", "0")?;
                }
                PictureEncoder::Nvenc => {
                    set_opt(raw, "preset", "p6")?;
                    set_opt(raw, "tune", "hq")?;
                    set_opt(raw, "profile", "main10")?;
                    set_opt(raw, "rc", "constqp")?;
                    set_opt(raw, "qp", &format!("{}", setting as i32))?;
                    set_opt(raw, "forced-idr", "1")?;
                }
                PictureEncoder::X265 => {
                    set_opt(raw, "preset", "fast")?;
                    set_opt(raw, "crf", &format!("{setting:.2}"))?;
                    set_opt(raw, "x265-params", "keyint=1:min-keyint=1:repeat-headers=1:range=full:log-level=error")?;
                    (*raw).thread_count = 0;
                }
            }
        }
        let ctx = ctx.encoder().video().map_err(|e| err(format!("{}: {e}", kind.name())))?
            .open_as(codec).map_err(|e| err(format!("{} does not open for {width} x {height}: {e}", kind.name())))?;
        let sets = unsafe {
            let raw = ctx.as_ptr();
            if (*raw).extradata.is_null() || (*raw).extradata_size <= 0 {
                Vec::new()
            } else {
                parameter_sets(std::slice::from_raw_parts((*raw).extradata, (*raw).extradata_size as usize))
            }
        };
        Ok(Self { kind, ctx, width, height, sent: 0, sets, ready: VecDeque::new() })
    }

    /// A picture's codes (`width × height`) to the encoder.
    fn send(&mut self, codes: &[u16]) -> Result<()> {
        let (w, h) = (self.width, self.height);
        if codes.len() != w * h {
            return Err(err("a picture is not the encoder's size"));
        }
        let planar = self.kind == PictureEncoder::X265;
        let format = if planar { ffmpeg::format::Pixel::YUV420P10LE } else { ffmpeg::format::Pixel::P010LE };
        let mut frame = ffmpeg::frame::Video::new(format, w as u32, h as u32);
        // P010 holds its ten bits at the top of sixteen.
        let shift = if planar { 0 } else { 6 };
        let stride = frame.stride(0);
        for (row, line) in codes.chunks_exact(w).zip(frame.data_mut(0).chunks_mut(stride)) {
            for (v, px) in row.iter().zip(line.chunks_exact_mut(2)) {
                px.copy_from_slice(&(v << shift).to_le_bytes());
            }
        }
        // No colour: both chroma samples at the middle.
        let grey = (512u16 << shift).to_le_bytes();
        for plane in 1..frame.planes() {
            for px in frame.data_mut(plane).chunks_exact_mut(2) {
                px.copy_from_slice(&grey);
            }
        }
        frame.set_pts(Some(self.sent));
        unsafe {
            (*frame.as_mut_ptr()).color_range = ffmpeg::ffi::AVColorRange::AVCOL_RANGE_JPEG;
            (*frame.as_mut_ptr()).pict_type = ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_I;
        }
        self.sent += 1;
        // An encoder that has pictures waiting takes no more until they are fetched.
        loop {
            match self.ctx.send_frame(&frame) {
                Ok(()) => break,
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {
                    let before = self.ready.len();
                    self.receive()?;
                    if self.ready.len() == before {
                        std::thread::yield_now();
                    }
                }
                Err(e) => return Err(err(format!("{}: {e}", self.kind.name()))),
            }
        }
        self.receive()
    }

    /// Fetch the access units that are ready into [`Self::ready`].
    fn receive(&mut self) -> Result<()> {
        use ffmpeg::util::error::EAGAIN;
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.ctx.receive_packet(&mut packet) {
                Ok(()) => {
                    let number = packet.pts().ok_or_else(|| err("the encoder handed a picture without its number"))?;
                    let unit = annex_b(packet.data().unwrap_or(&[]), &self.sets)?;
                    self.ready.push_back((number, unit));
                }
                Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => return Ok(()),
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) => return Err(err(format!("{}: {e}", self.kind.name()))),
            }
        }
    }

    /// No picture follows: the ones still inside come out.
    fn finish(&mut self) -> Result<()> {
        self.ctx.send_eof().map_err(|e| err(format!("{}: {e}", self.kind.name())))?;
        self.receive()
    }
}

/// The parameter sets of an encoder's configuration record, as Annex B: the
/// record itself where it already is Annex B, else the NAL units of an
/// `hvcC` record.
fn parameter_sets(extra: &[u8]) -> Vec<u8> {
    if extra.starts_with(&[0, 0, 0, 1]) || extra.starts_with(&[0, 0, 1]) {
        return extra.to_vec();
    }
    let mut out = Vec::new();
    let Some(&arrays) = extra.get(22) else { return out };
    let mut at = 23usize;
    for _ in 0..arrays {
        let Some(count) = extra.get(at + 1..at + 3).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize) else { return out };
        at += 3;
        for _ in 0..count {
            let Some(len) = extra.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize) else { return out };
            let Some(unit) = extra.get(at + 2..at + 2 + len) else { return out };
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(unit);
            at += 2 + len;
        }
    }
    out
}

/// The type of each NAL unit of an Annex B access unit, with where its
/// bytes start.
fn nal_units(unit: &[u8]) -> Vec<(u8, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < unit.len() {
        if unit[i] == 0 && unit[i + 1] == 0 && unit[i + 2] == 1 {
            out.push(((unit[i + 3] >> 1) & 0x3f, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// A picture as an encoder hands it, as one Annex B access unit that
/// carries its parameter sets: start codes for length words where the
/// encoder wrote those, and `sets` in front where the picture has none.
fn annex_b(packet: &[u8], sets: &[u8]) -> Result<Vec<u8>> {
    if packet.len() < 5 {
        return Err(err("the encoder handed an empty picture"));
    }
    let mut unit = if packet.starts_with(&[0, 0, 0, 1]) || packet.starts_with(&[0, 0, 1]) {
        packet.to_vec()
    } else {
        let mut out = Vec::with_capacity(packet.len() + 16);
        let mut at = 0usize;
        while at + 4 <= packet.len() {
            let len = u32::from_be_bytes([packet[at], packet[at + 1], packet[at + 2], packet[at + 3]]) as usize;
            let Some(nal) = packet.get(at + 4..at + 4 + len) else { return Err(err("the encoder handed a picture that does not add up")) };
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
            at += 4 + len;
        }
        out
    };
    // 32, 33, 34: the video, sequence and picture parameter sets.
    let types: Vec<u8> = nal_units(&unit).into_iter().map(|u| u.0).collect();
    if ![32u8, 33, 34].iter().all(|t| types.contains(t)) {
        if sets.is_empty() {
            return Err(err("the encoder handed a picture without its parameter sets"));
        }
        let mut whole = sets.to_vec();
        whole.append(&mut unit);
        unit = whole;
    }
    Ok(unit)
}

/// The decoders of the graphics hardware a picture is offered to, and the
/// pixel format each hands its pictures in.
const HARDWARE: [(ffmpeg::ffi::AVHWDeviceType, ffmpeg::ffi::AVPixelFormat); 3] = [
    (ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX, ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX),
    (ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA, ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_D3D11),
    (ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA, ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA),
];

/// The decoder asks which of the formats it offers to use: a hardware one
/// this module set up, else the last, which is the one in memory.
unsafe extern "C" fn pick_format(_ctx: *mut ffmpeg::ffi::AVCodecContext, offered: *const ffmpeg::ffi::AVPixelFormat) -> ffmpeg::ffi::AVPixelFormat {
    let mut p = offered;
    let mut last = *offered;
    while *p != ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
        if HARDWARE.iter().any(|(_, format)| *format == *p) {
            return *p;
        }
        last = *p;
        p = p.add(1);
    }
    last
}

/// One decoder: an access unit in, its picture out.
struct Decoder {
    ctx: ffmpeg::decoder::Video,
    hardware: bool,
}

impl Decoder {
    /// A decoder on the graphics hardware where there is one to be had
    /// (`hardware`), else in software.
    fn open(hardware: bool) -> Result<Self> {
        crate::decode::init();
        let codec = ffmpeg::codec::decoder::find(ffmpeg::codec::Id::HEVC).ok_or_else(|| err("no HEVC decoder in this build"))?;
        let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
        let mut attached = false;
        unsafe {
            let raw = ctx.as_mut_ptr();
            (*raw).flags |= ffmpeg::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
            if hardware {
                for (kind, _) in HARDWARE {
                    let mut device: *mut ffmpeg::ffi::AVBufferRef = std::ptr::null_mut();
                    if ffmpeg::ffi::av_hwdevice_ctx_create(&mut device, kind, std::ptr::null(), std::ptr::null_mut(), 0) >= 0 && !device.is_null() {
                        (*raw).hw_device_ctx = device;
                        (*raw).get_format = Some(pick_format);
                        attached = true;
                        break;
                    }
                }
            }
            if !attached {
                // In software a picture's rows decode side by side; no picture waits for the next.
                (*raw).thread_type = ffmpeg::ffi::FF_THREAD_SLICE as i32;
                (*raw).thread_count = 0;
            }
        }
        let ctx = ctx.decoder().video().map_err(|e| err(format!("the HEVC decoder does not open: {e}")))?;
        Ok(Self { ctx, hardware: attached })
    }

    /// The picture `unit` holds (`width × height`). `spare`: a frame in
    /// memory to fetch a hardware decoder's picture into.
    fn decode(&mut self, unit: &[u8], width: usize, height: usize, spare: Option<ffmpeg::frame::Video>) -> Result<Picture> {
        use ffmpeg::util::error::EAGAIN;
        let packet = ffmpeg::Packet::copy(unit);
        self.ctx.send_packet(&packet).map_err(|e| err(format!("a picture does not decode: {e}")))?;
        let mut frame = ffmpeg::frame::Video::empty();
        match self.ctx.receive_frame(&mut frame) {
            Ok(()) => {}
            Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => {
                // The decoder holds the picture back for one that may follow: none does.
                self.ctx.send_eof().map_err(|e| err(format!("a picture does not decode: {e}")))?;
                let got = self.ctx.receive_frame(&mut frame);
                self.ctx.flush();
                got.map_err(|e| err(format!("a picture does not decode: {e}")))?;
            }
            Err(e) => {
                self.ctx.flush();
                return Err(err(format!("a picture does not decode: {e}")));
            }
        }
        let frame = if HARDWARE.iter().any(|(_, format)| ffmpeg::format::Pixel::from(*format) == frame.format()) {
            let fits = |f: &ffmpeg::frame::Video| f.format() == ffmpeg::format::Pixel::P010LE && (f.width() as usize, f.height() as usize) == (width, height);
            let mut in_memory = spare.filter(fits).unwrap_or_else(ffmpeg::frame::Video::empty);
            crate::decode::download_hw_frame(&frame, &mut in_memory)?;
            in_memory
        } else {
            frame
        };
        if (frame.width() as usize, frame.height() as usize) != (width, height) {
            return Err(err(format!("a picture is {} x {}, not the {width} x {height} of its frame", frame.width(), frame.height())));
        }
        let shift = match frame.format() {
            ffmpeg::format::Pixel::P010LE => 6,
            ffmpeg::format::Pixel::YUV420P10LE => 0,
            other => return Err(err(format!("a picture decodes as {other:?}, not as 10 bits"))),
        };
        Ok(Picture { frame, shift })
    }
}

/// Decoders kept between frames: opening one on the graphics hardware
/// takes as long as several pictures.
fn decoder_pool() -> &'static Mutex<Vec<Decoder>> {
    static POOL: std::sync::OnceLock<Mutex<Vec<Decoder>>> = std::sync::OnceLock::new();
    POOL.get_or_init(Default::default)
}

/// Frames in memory kept between frames, for a hardware decoder's pictures.
fn frame_pool() -> &'static Mutex<Vec<ffmpeg::frame::Video>> {
    static POOL: std::sync::OnceLock<Mutex<Vec<ffmpeg::frame::Video>>> = std::sync::OnceLock::new();
    POOL.get_or_init(Default::default)
}

/// Decoders at work side by side at most: a pair's eight pictures.
///
/// On the graphics hardware the video engine takes one picture after the
/// other and its callers only keep it busy (a pair of 5760 x 5760: 75 ms
/// through one decoder, 44 through two, 30 through four, 28 through eight).
/// Software decoders working beside the hardware ones were tried and are
/// not used: a picture a processor core takes on is done long after the
/// engine would have had it, and the pair waits for it.
const DECODERS: usize = 2 * ARCHIVE_PICTURES;

/// No decoder attaches to the graphics hardware here (found out once).
static NO_HARDWARE: AtomicBool = AtomicBool::new(false);

/// Decode `units` (pictures of `width × height`) side by side, in their
/// order.
fn decode_pictures(units: &[&[u8]], width: usize, height: usize) -> Result<Vec<Picture>> {
    let want = units.len().min(DECODERS).max(1);
    let hardware = !NO_HARDWARE.load(Ordering::Relaxed) && std::env::var_os("VR180_ARCHIVE_SOFTWARE").is_none();
    let mut decoders: Vec<Decoder> = {
        let mut pool = decoder_pool().lock().map_err(|_| err("the decoders are gone"))?;
        // (A decoder that fell back to software stays in use: the pool is of whatever works here.)
        let keep = pool.len().saturating_sub(want);
        pool.split_off(keep)
    };
    while decoders.len() < want {
        let decoder = Decoder::open(hardware && !NO_HARDWARE.load(Ordering::Relaxed))?;
        if hardware && !decoder.hardware {
            NO_HARDWARE.store(true, Ordering::Relaxed);
        }
        decoders.push(decoder);
    }
    let mut spares: Vec<ffmpeg::frame::Video> = {
        let mut pool = frame_pool().lock().map_err(|_| err("the frames are gone"))?;
        let keep = pool.len().saturating_sub(units.len());
        pool.split_off(keep)
    };
    let mut pictures: Vec<Option<Picture>> = (0..units.len()).map(|_| None).collect();
    let failed: Mutex<Option<Error>> = Mutex::new(None);
    let jobs: Mutex<Vec<(usize, &mut Option<Picture>, Option<ffmpeg::frame::Video>)>> =
        Mutex::new(pictures.iter_mut().enumerate().rev().map(|(i, slot)| (i, slot, spares.pop())).collect());
    std::thread::scope(|scope| {
        for decoder in decoders.iter_mut() {
            let (jobs, failed) = (&jobs, &failed);
            scope.spawn(move || loop {
                let job = jobs.lock().ok().and_then(|mut j| j.pop());
                let Some((i, slot, spare)) = job else { break };
                let mut result = decoder.decode(units[i], width, height, spare);
                // A hardware decoder that will not take the picture: once more in software, and from then on.
                if result.is_err() && decoder.hardware {
                    if let Ok(software) = Decoder::open(false) {
                        *decoder = software;
                        result = decoder.decode(units[i], width, height, None);
                    }
                }
                match result {
                    Ok(picture) => *slot = Some(picture),
                    Err(e) => {
                        if let Ok(mut first) = failed.lock() { first.get_or_insert(e); }
                        break;
                    }
                }
            });
        }
    });
    if let Ok(mut pool) = decoder_pool().lock() {
        pool.append(&mut decoders);
        pool.truncate(2 * DECODERS);
    }
    if let Some(e) = failed.into_inner().ok().flatten() {
        return Err(e);
    }
    pictures.into_iter().map(|p| p.ok_or_else(|| err("a picture was not decoded"))).collect()
}

/// Hand the frames of `pictures` back for the next ones (those in memory
/// of this module's own; a software decoder's go back to it by themselves).
fn recycle(pictures: Vec<Picture>) {
    if let Ok(mut pool) = frame_pool().lock() {
        pool.extend(pictures.into_iter().filter(|p| p.shift == 6).map(|p| p.frame));
        pool.truncate(4 * ARCHIVE_PICTURES);
    }
}

/// Decode the archive record `record` holds into `out`, the layout's
/// samples: channel 0's frame, then channel 1's (a take of one lens: its one
/// channel). A channel that is not asked for is left as it is.
pub fn decode_record(record: &[u8], layout: Layout, channels: Channels, out: &mut [u16]) -> Result<PairEntry> {
    if out.len() != layout.samples() {
        return Err(err("output buffer is not the record's size"));
    }
    let (header, pictures) = x6b::archive_pictures(record, layout).map_err(err)?;
    let (width, height) = (layout.width(), layout.height());
    let wanted: Vec<usize> = (0..header.channels).filter(|c| !matches!(channels, Channels::One(only) if header.channels > 1 && only != *c)).collect();
    let units: Vec<&[u8]> = wanted.iter().flat_map(|c| pictures[c * ARCHIVE_PICTURES..(c + 1) * ARCHIVE_PICTURES].iter().copied()).collect();
    let planes = decode_pictures(&units, width / 2, height / 2)?;
    for (k, channel) in wanted.iter().enumerate() {
        let eye = &mut out[channel * layout.eye_samples()..(channel + 1) * layout.eye_samples()];
        put_planes(&planes[k * ARCHIVE_PICTURES..(k + 1) * ARCHIVE_PICTURES], width, height, &header.values, eye);
    }
    recycle(planes);
    let one = header.channels == 1;
    Ok(PairEntry {
        request: header.request, offset: 0, len: header.len, profile: Profile::Archive,
        exposure_ns: if one { [header.exposure_ns[0]; 2] } else { header.exposure_ns },
        other: None,
        shutter_ns: if one { [header.shutter_ns[0]; 2] } else { header.shutter_ns },
    })
}

// ── The copy ───────────────────────────────────────────────────────────

/// What an archive copy is to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchiveOptions {
    /// Bits a pixel to aim at ([`BITS_LEAST`] to [`BITS_MOST`]).
    pub bits_per_pixel: f64,
    /// The encoder to write the pictures with; `None`: the first of
    /// [`PictureEncoder::ALL`] that opens.
    pub encoder: Option<PictureEncoder>,
}

/// The fewest and the most bits a pixel a lossy copy may be asked to hold:
/// below the first the shadows of a frame turn to flat patches when they
/// are lifted, above the second a copy that keeps every sample is smaller.
pub const BITS_LEAST: f64 = 0.5;
pub const BITS_MOST: f64 = 1.0;

impl Default for ArchiveOptions {
    fn default() -> Self {
        Self { bits_per_pixel: BITS_MOST, encoder: None }
    }
}

/// What an archive copy came to.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveReport {
    /// Pairs in the copy (frames, of a take of one lens).
    pub pairs: usize,
    /// Length of those pairs' records in the take, and of the copy's.
    pub source_bytes: u64,
    pub bytes: u64,
    /// Bits a pixel of the copy, and of those pairs in the take.
    pub bits_per_pixel: f64,
    pub source_bits_per_pixel: f64,
    pub encoder: &'static str,
    /// The encoder's own quality setting.
    pub setting: f64,
    /// The noise the codes were laid out by.
    pub noise: (f64, f64),
    /// How far the copy's samples are from the take's on the frames looked
    /// at: root mean square, in sample values and in codes (two values the
    /// noise cannot tell apart are less than a code apart, two levels of a
    /// table a few codes).
    pub error_rms: f64,
    pub error_in_codes: f64,
    pub seconds: f64,
}

/// Frames looked at before the copy starts: for the noise, and for the
/// encoder's setting.
const LOOKED_AT: usize = 3;

/// The pictures of one channel of a pair, coded: its four planes through
/// `encoder`. Returns nothing; the pictures come out of the encoder later.
fn send_channel(encoder: &mut Encoder, mosaic: &[u16], width: usize, height: usize, curve: &Curve, codes: &mut Vec<u16>) -> Result<()> {
    for plane in 0..ARCHIVE_PICTURES {
        plane_codes(mosaic, width, height, plane, curve, codes);
        encoder.send(codes)?;
    }
    Ok(())
}

/// Bits a pixel `frames` (each the take's channels one after the other)
/// come to through `kind` at `quality`.
fn bits_at(kind: PictureEncoder, quality: f64, frames: &[Vec<u16>], layout: Layout, curve: &Curve) -> Result<f64> {
    let (width, height, per) = (layout.width(), layout.height(), layout.eye_samples());
    let mut encoder = Encoder::open(kind, width / 2, height / 2, quality)?;
    let mut codes = Vec::new();
    for frame in frames {
        for mosaic in frame.chunks_exact(per) {
            send_channel(&mut encoder, mosaic, width, height, curve, &mut codes)?;
        }
    }
    encoder.finish()?;
    let bytes: usize = encoder.ready.iter().map(|(_, unit)| unit.len()).sum();
    Ok(bytes as f64 * 8.0 / (frames.len() * layout.samples()) as f64)
}

/// The quality (0 to 1) at which `frames` come nearest to `target` bits a
/// pixel through `kind`, and the bits they come to there.
fn quality_for(kind: PictureEncoder, target: f64, frames: &[Vec<u16>], layout: Layout, curve: &Curve) -> Result<(f64, f64)> {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut best = (0.5, f64::INFINITY);
    for _ in 0..9 {
        let mid = 0.5 * (lo + hi);
        let bits = bits_at(kind, mid, frames, layout, curve)?;
        if (bits / target).ln().abs() < (best.1 / target).ln().abs() {
            best = (mid, bits);
        }
        if (bits / target - 1.0).abs() < 0.02 {
            break;
        }
        if bits > target { hi = mid } else { lo = mid }
    }
    Ok(best)
}

/// Write the frames `frames` of the take `source` names as an archive copy
/// into the folder `dir` (made here; it must not hold a capture yet).
///
/// `progress` hears how far the copy has come (0 to 1); `cancel` set stops
/// it, and what was written of the capture is taken away again. The files
/// beside the capture are copied as they are; `clip.json` gains its
/// `archive` record and the closing record is written last, so a copy that
/// did not finish does not open.
pub fn archive_take(
    source: &Path, dir: &Path, frames: std::ops::Range<usize>, options: &ArchiveOptions,
    mut progress: impl FnMut(f32), cancel: &AtomicBool,
) -> Result<ArchiveReport> {
    let started = std::time::Instant::now();
    let index = x6_raw::open(source)?;
    let take = &index.take;
    let layout = take.layout;
    let frames = frames.start.min(index.frames())..frames.end.min(index.frames());
    if frames.is_empty() {
        return Err(err("no frames to copy"));
    }
    if index.pairs[frames.start].profile == Profile::Archive {
        return Err(err(format!("{} is an archive copy already: a copy is made from the camera's take", take.dir.display())));
    }
    let target = options.bits_per_pixel.clamp(BITS_LEAST, BITS_MOST);
    let capture = dir.join(x6b::CAPTURE_FILE);
    if capture.exists() {
        return Err(err(format!("{} holds a capture already", dir.display())));
    }
    if dir.canonicalize().ok().is_some_and(|d| take.dir.canonicalize().ok() == Some(d)) {
        return Err(err("the copy cannot be written into the take's own folder"));
    }
    let (width, height, channels, per) = (layout.width(), layout.height(), layout.channels(), layout.eye_samples());
    let mut file = File::open(take.capture_path())?;
    let mut record = Vec::new();
    let mut decode = |i: usize, out: &mut Vec<u16>| -> Result<()> {
        out.resize(layout.samples(), 0);
        x6_raw::decode_pair(&mut file, &index.pairs[i], take.format(), Channels::Both, &mut record, out, 0)
    };

    // The frames looked at: spread over the stretch, each with the one after it.
    let count = frames.len();
    let places: Vec<usize> = (0..LOOKED_AT.min(count)).map(|k| frames.start + (2 * k + 1) * count / (2 * LOOKED_AT.min(count))).collect();
    let mut looked: Vec<Vec<u16>> = Vec::new();
    let mut after: Vec<Vec<u16>> = Vec::new();
    for &i in &places {
        let mut frame = Vec::new();
        decode(i, &mut frame)?;
        looked.push(frame);
        if i + 1 < index.frames() {
            let mut next = Vec::new();
            decode(i + 1, &mut next)?;
            after.push(next);
        }
    }
    let pairs_of: Vec<(&[u16], &[u16])> = looked.iter().zip(&after).map(|(a, b)| (&a[..], &b[..])).collect();
    let noise = noise_of(&pairs_of, width, height, take.black_level, take.white_level);
    // The smallest value the frames hold, and a little below it.
    let lowest = looked.iter().flat_map(|f| f.iter().step_by(7)).copied().min().unwrap_or(0);
    let values = index.pairs[frames.start].profile.sample_values();
    let below = values.partition_point(|v| *v < lowest).saturating_sub(4);
    let curve = Curve::new(&values, values.get(below).copied().unwrap_or(0), take.black_level, take.white_level, noise);
    drop(after);

    // The encoder, and its setting for the bits asked for.
    let mut opened = None;
    let mut why = String::new();
    for kind in options.encoder.map_or(PictureEncoder::ALL.to_vec(), |k| vec![k]) {
        match quality_for(kind, target, &looked, layout, &curve) {
            Ok((quality, bits)) => {
                tracing::info!("archive copy: {} at {:.2} gives {bits:.3} bits a pixel on {} frames ({target:.2} asked for)", kind.name(), kind.setting(quality), looked.len());
                opened = Some((kind, quality));
                break;
            }
            Err(e) => why = format!("{why}{}: {e}; ", kind.name()),
        }
    }
    let Some((kind, quality)) = opened else { return Err(err(format!("no encoder writes the pictures ({why})"))) };

    std::fs::create_dir_all(dir)?;
    let written = (|| -> Result<(u64, u64)> {
        let mut out = std::io::BufWriter::with_capacity(1 << 22, File::create(&capture)?);
        let mut encoders: Vec<Encoder> = (0..channels).map(|_| Encoder::open(kind, width / 2, height / 2, quality)).collect::<Result<_>>()?;
        let mut codes: Vec<Vec<u16>> = vec![Vec::new(); channels];
        let (mut bytes, mut source_bytes) = (0u64, 0u64);
        let mut next = frames.start;
        // The records whose pictures have all come out of the encoders, written in order.
        let mut write_ready = |encoders: &mut Vec<Encoder>, next: &mut usize, bytes: &mut u64| -> Result<()> {
            while encoders.iter().all(|e| e.ready.len() >= ARCHIVE_PICTURES) {
                let entry = &index.pairs[*next];
                let mut pictures: Vec<Vec<u8>> = Vec::with_capacity(channels * ARCHIVE_PICTURES);
                for (channel, encoder) in encoders.iter_mut().enumerate() {
                    for plane in 0..ARCHIVE_PICTURES {
                        let (number, unit) = encoder.ready.pop_front().expect("counted");
                        if number != ((*next - frames.start) * ARCHIVE_PICTURES + plane) as i64 {
                            return Err(err(format!("the encoder of channel {channel} handed picture {number} out of its turn")));
                        }
                        pictures.push(unit);
                    }
                }
                let shutter = index.shutter_ns[*next];
                let header = ArchiveHeader {
                    coding: x6b::ARCHIVE_HEVC_PLANES, channels, width: width as u32, height: height as u32, request: entry.request,
                    exposure_ns: entry.exposure_ns, shutter_ns: [shutter[0].unwrap_or(0), shutter[1].unwrap_or(0)],
                    picture_len: [0; 2 * ARCHIVE_PICTURES], crc: 0,
                    source_profile: Profile::ALL.iter().position(|p| *p == entry.profile).unwrap_or(0) as u32,
                    values: curve.values().to_vec(), len: 0,
                };
                let refs: Vec<&[u8]> = pictures.iter().map(|p| &p[..]).collect();
                let record = x6b::archive_record(header, &refs).map_err(err)?;
                out.write_all(&record)?;
                *bytes += record.len() as u64;
                *next += 1;
            }
            Ok(())
        };
        let mut frame = Vec::new();
        for i in frames.clone() {
            if cancel.load(Ordering::Relaxed) {
                return Err(err("stopped"));
            }
            decode(i, &mut frame)?;
            let entry = &index.pairs[i];
            source_bytes += entry.len + entry.other.map_or(0, |(_, len)| len);
            // The channels' pictures are made side by side.
            let sent: Vec<Result<()>> = std::thread::scope(|scope| {
                let handles: Vec<_> = encoders.iter_mut().zip(codes.iter_mut()).zip(frame.chunks_exact(per))
                    .map(|((encoder, codes), mosaic)| {
                        let curve = &curve;
                        scope.spawn(move || send_channel(encoder, mosaic, width, height, curve, codes))
                    }).collect();
                handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(err("an encoder stopped")))).collect()
            });
            for result in sent {
                result?;
            }
            write_ready(&mut encoders, &mut next, &mut bytes)?;
            progress((i + 1 - frames.start) as f32 / count as f32 * 0.98);
        }
        for encoder in encoders.iter_mut() {
            encoder.finish()?;
        }
        write_ready(&mut encoders, &mut next, &mut bytes)?;
        if next != frames.end {
            return Err(err(format!("the encoders handed the pictures of {} frames of {count}", next - frames.start)));
        }
        out.flush()?;
        Ok((bytes, source_bytes))
    })();
    let (bytes, source_bytes) = match written {
        Ok(sizes) => sizes,
        Err(e) => {
            // Nothing of a copy that did not finish stays: its capture, and the folder where that leaves it empty.
            let _ = std::fs::remove_file(&capture);
            let _ = std::fs::remove_dir(dir);
            return Err(e);
        }
    };

    // The files beside the capture, as they are.
    for entry in std::fs::read_dir(&take.dir)?.flatten() {
        let name = entry.file_name();
        let own = [x6b::CAPTURE_FILE, x6b::CLIP_FILE, x6b::SEAL_FILE].iter().any(|f| name.to_str().is_some_and(|n| n.eq_ignore_ascii_case(f)));
        if own || name.to_string_lossy().starts_with('.') || !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        std::fs::copy(entry.path(), dir.join(&name))?;
    }

    // How far the copy is from the take, on the frames looked at.
    let bits = bytes as f64 * 8.0 / (count * layout.samples()) as f64;
    let (mut squares, mut in_codes, mut samples) = (0.0f64, 0.0f64, 0usize);
    {
        let mut copy = File::open(&capture)?;
        let (mut buf, mut back) = (Vec::new(), vec![0u16; layout.samples()]);
        let mut at = 0u64;
        let mut head = vec![0u8; x6b::BLOCK as usize];
        use std::io::{Read, Seek, SeekFrom};
        for i in frames.clone() {
            copy.seek(SeekFrom::Start(at))?;
            copy.read_exact(&mut head)?;
            let len = ArchiveHeader::parse(&head, layout, at).map_err(err)?.len;
            if let Some(k) = places.iter().position(|&p| p == i) {
                let entry = PairEntry { request: index.pairs[i].request, offset: at, len, profile: Profile::Archive, exposure_ns: index.pairs[i].exposure_ns, other: None, shutter_ns: [0; 2] };
                x6b::read_record(&mut copy, &entry, &mut buf).map_err(err)?;
                decode_record(&buf, layout, Channels::Both, &mut back)?;
                for (a, b) in looked[k].iter().zip(&back).step_by(5) {
                    let d = *a as f64 - *b as f64;
                    squares += d * d;
                    let c = curve.code_of(*a) as f64 - curve.code_of(*b) as f64;
                    in_codes += c * c;
                    samples += 1;
                }
            }
            at += len;
        }
    }
    let (error_rms, error_in_codes) = ((squares / samples.max(1) as f64).sqrt(), (in_codes / samples.max(1) as f64).sqrt());

    // What the folder says of itself, and the closing record last.
    let mut clip: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(x6b::CLIP_FILE))?).map_err(err)?;
    let source_bits = source_bytes as f64 * 8.0 / (count * layout.samples()) as f64;
    if let Some(object) = clip.as_object_mut() {
        object.insert("archive".into(), serde_json::json!({
            "version": 1,
            "kind": "lossy",
            "target_bits_per_pixel": target,
            "bits_per_pixel": (bits * 1000.0).round() / 1000.0,
            "source": take.dir.file_name().map(|n| n.to_string_lossy().into_owned()),
            "source_bits_per_pixel": (source_bits * 1000.0).round() / 1000.0,
            "pairs": count,
            "pairs_of_source": index.frames(),
            "coding": "hevc-planes",
            "encoder": kind.name(),
            "encoder_setting": kind.setting(quality),
            "noise": [noise.0, noise.1],
            "error_rms": (error_rms * 1000.0).round() / 1000.0,
            "error_in_codes": (error_in_codes * 1000.0).round() / 1000.0,
            "made_by": format!("VR180 Silver Bullet {}", env!("CARGO_PKG_VERSION")),
        }));
    }
    std::fs::write(dir.join(x6b::CLIP_FILE), serde_json::to_vec_pretty(&clip).map_err(err)?)?;
    let mut seal: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(x6b::SEAL_FILE))?).map_err(err)?;
    if let Some(object) = seal.as_object_mut() {
        object.insert("bytes".into(), bytes.into());
        object.remove("eye_frames");
        object.remove("eye_hardware_gaps");
        if take.window_numbers_records() {
            object.insert("request_window".into(), serde_json::json!([index.pairs[frames.start].request, index.pairs[frames.end - 1].request]));
            object.insert("pairs".into(), count.into());
            object.insert("received_pairs".into(), count.into());
        } else {
            object.insert("frames".into(), count.into());
            object.remove("received_pairs");
        }
    }
    std::fs::write(dir.join(x6b::SEAL_FILE), serde_json::to_vec_pretty(&seal).map_err(err)?)?;
    progress(1.0);
    let report = ArchiveReport {
        pairs: count, source_bytes, bytes, bits_per_pixel: bits, source_bits_per_pixel: source_bits,
        encoder: kind.name(), setting: kind.setting(quality), noise, error_rms, error_in_codes, seconds: started.elapsed().as_secs_f64(),
    };
    tracing::info!(
        "archive copy of {}: {count} frames, {:.2} GB to {:.2} GB ({:.2} to {:.2} bits a pixel) through {} at {:.2}, {:.2} values rms from the take ({:.2} codes), {:.1} s",
        take.dir.display(), source_bytes as f64 / 1e9, bytes as f64 / 1e9, source_bits, bits, kind.name(), kind.setting(quality), error_rms, error_in_codes, report.seconds);
    Ok(report)
}

/// [`archive_take`] as the exporter runs it: the frames between `t_in` and
/// `t_out` (seconds from the first frame; either end open when `None`, the
/// way the other exports pick them), at `bits_per_pixel`, with the
/// exporter's progress.
pub fn export_archive(
    source: &Path, dir: &Path, t_in: Option<f64>, t_out: Option<f64>, bits_per_pixel: f64,
    mut progress: impl FnMut(crate::fisheye_export::ExportProgress), cancel: std::sync::Arc<AtomicBool>,
) -> Result<ArchiveReport> {
    let index = x6_raw::open(source)?;
    let frames = crate::sensor_dng::frames_in(&index, t_in, t_out);
    let total = frames.len() as u64;
    let started = std::time::Instant::now();
    archive_take(source, dir, frames, &ArchiveOptions { bits_per_pixel, encoder: None }, |done| {
        let frame = ((done as f64 * total as f64).round() as u64).min(total);
        progress(crate::fisheye_export::ExportProgress {
            frame_idx: frame, total_frames: total, fps_avg: frame as f32 / started.elapsed().as_secs_f32().max(1e-3),
        });
    }, &cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value of a 10-bit frame from just below black: the dark
    /// values, whose noise is small, lie codes apart, the bright ones share
    /// codes, and no value comes back further off than the noise at it.
    #[test]
    fn the_codes_follow_the_noise() {
        let values: Vec<u16> = (0..1024).collect();
        let noise = (0.0256, 0.268);
        let curve = Curve::new(&values, 60, 64, 1023, noise);
        let codes: Vec<u16> = values.iter().map(|&v| curve.code_of(v)).collect();
        assert!(codes.windows(2).all(|w| w[0] <= w[1]));
        assert!(codes[..=60].iter().all(|c| *c == codes[60]) && codes[60] == 32 && codes[1023] <= 1023 - 32);
        // White stands off: a code a dozen from it is still white, and no value below it is.
        assert_eq!(codes[1023] - codes[1022], WHITE_APART as u16);
        assert!((codes[1023] - 19..=1023).all(|c| curve.values()[c as usize] == 1023));
        assert!((0..=codes[1023] - 21).all(|c| curve.values()[c as usize] < 1023));
        // At black the noise is half a code value: a value is several codes from the next.
        assert!((4..=6).contains(&(codes[65] - codes[64])), "{}", codes[65] - codes[64]);
        // Near white it is five values: several values to a code.
        assert!(codes[1000] - codes[990] <= 6);
        assert!(curve.values().windows(2).all(|w| w[0] <= w[1]) && curve.values().len() == CODES);
        for &v in &values[60..] {
            let back = curve.values()[curve.code_of(v) as usize];
            let sigma = (noise.0 * (v as f64 - 64.0).max(0.0) + noise.1).sqrt();
            assert!((back as f64 - v as f64).abs() <= 0.5 * sigma + 0.5, "{v} came back as {back}");
        }
        // A code no value has stands for the value nearest to it.
        assert_eq!(curve.values()[(codes[64] + 1) as usize], 64);
        assert_eq!(curve.values()[0], 60);
    }

    /// A take whose values are the levels of a table: every level keeps a
    /// code of its own, a few codes from the next.
    #[test]
    fn a_table_of_levels_keeps_every_level() {
        let levels = Profile::Raised60.sample_values();
        let levels: Vec<u16> = levels.into_iter().filter(|v| *v < 13472).collect();
        let curve = Curve::new(&levels, 64, 64, 13440, (0.5, 6.0));
        let codes: Vec<u16> = levels.iter().filter(|v| **v >= 64).map(|&v| curve.code_of(v)).collect();
        assert!(codes.windows(2).all(|w| w[1] > w[0] + 1), "levels share a code or sit side by side");
        for &v in levels.iter().filter(|v| **v >= 64) {
            assert_eq!(curve.values()[curve.code_of(v) as usize], v);
        }
        // Nothing to tell the noise by: every level the most codes from the next, as far as they fit.
        let even = Curve::new(&levels[..41], 0, 0, levels[40], (0.0, 1e-9));
        assert!((0..39).all(|i| even.code_of(levels[i + 1]) - even.code_of(levels[i]) == APART_MOST as u16));
        assert_eq!(even.code_of(levels[40]) - even.code_of(levels[39]), WHITE_APART as u16);
    }

    /// Frames of a known noise: the fit finds it.
    #[test]
    fn the_noise_of_made_up_frames_is_found() {
        let (w, h, black, white) = (480usize, 360usize, 64u16, 1023u16);
        let (a, b) = (0.03f64, 0.3f64);
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut uniform = move || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut frame = |seed_level: &dyn Fn(usize, usize) -> f64| -> Vec<u16> {
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let level = seed_level(x, y);
                    let sd = (a * level + b).sqrt();
                    // Twelve uniform draws: near enough to a normal one.
                    let n: f64 = (0..12).map(|_| uniform()).sum::<f64>() - 6.0;
                    out.push((black as f64 + level + sd * n).round().clamp(0.0, white as f64) as u16);
                }
            }
            out
        };
        // Flat bands of rising brightness.
        let level = |_x: usize, y: usize| 2.0 * 1.6f64.powi((y * 12 / h) as i32);
        let (first, second) = (frame(&level), frame(&level));
        let (fa, fb) = noise_of(&[(&first, &second)], w, h, black, white);
        assert!((fa / a - 1.0).abs() < 0.2 && (fb - b).abs() < 0.15, "{fa} {fb}");
    }

    #[test]
    fn a_picture_gets_start_codes_and_its_parameter_sets() {
        // An hvcC record with one array of each set: 23 bytes, then the arrays.
        let mut record = vec![1u8; 22];
        record.push(3);
        for (kind, body) in [(32u8, [0x40u8, 1, 9]), (33, [0x42, 1, 8]), (34, [0x44, 1, 7])] {
            record.extend_from_slice(&[kind, 0, 1, 0, 3]);
            record.extend_from_slice(&body);
        }
        let sets = parameter_sets(&record);
        assert_eq!(sets, [0, 0, 0, 1, 0x40, 1, 9, 0, 0, 0, 1, 0x42, 1, 8, 0, 0, 0, 1, 0x44, 1, 7]);
        // A picture with a length word (an IDR slice, type 19), no sets of its own.
        let packet = [0u8, 0, 0, 5, 0x26, 1, 0xaa, 0xbb, 0xcc];
        let unit = annex_b(&packet, &sets).unwrap();
        assert_eq!(nal_units(&unit).iter().map(|u| u.0).collect::<Vec<_>>(), [32, 33, 34, 19]);
        assert!(unit.ends_with(&[0, 0, 0, 1, 0x26, 1, 0xaa, 0xbb, 0xcc]));
        // One that has them is left as it is.
        assert_eq!(annex_b(&unit, &[]).unwrap(), unit);
        assert!(annex_b(&packet, &[]).is_err());
    }

    /// A take named by `VR180_ARCHIVE_TAKE` (skipped without one): its first
    /// frames copied, and the copy opened as a take. The same frames with
    /// the same numbers, stamps and integration times; samples within the
    /// noise of the take's; saturated samples saturated and no others; no
    /// copy of a copy, and none over another.
    #[test]
    fn a_take_is_copied_and_the_copy_opens_as_a_take() {
        let Some(take) = std::env::var_os("VR180_ARCHIVE_TAKE").map(std::path::PathBuf::from) else { return };
        let dir = std::env::temp_dir().join(format!("vr180-archive-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let index = x6_raw::open(&take).unwrap();
        let frames = 0..index.frames().min(8);
        let never = AtomicBool::new(false);
        let mut heard = Vec::new();
        let report = archive_take(&take, &dir, frames.clone(), &ArchiveOptions::default(), |p| heard.push(p), &never).unwrap();
        println!("{report:?}");
        assert_eq!(report.pairs, frames.len());
        assert!((0.75..=1.15).contains(&report.bits_per_pixel) && report.bits_per_pixel < report.source_bits_per_pixel, "{report:?}");
        assert!(report.error_in_codes < 5.0, "{report:?}");
        assert!(heard.windows(2).all(|w| w[0] <= w[1]) && heard.last() == Some(&1.0));
        let copy = x6_raw::open(&dir).unwrap();
        assert_eq!((copy.frames(), copy.take.layout, copy.take.fps_setting), (frames.len(), index.take.layout, index.take.fps_setting));
        for (a, b) in copy.pairs.iter().zip(&index.pairs[frames.clone()]) {
            assert_eq!((a.request, a.exposure_ns, a.profile, a.other), (b.request, b.exposure_ns, Profile::Archive, None));
        }
        assert_eq!(copy.shutter_ns, index.shutter_ns[frames.clone()]);
        assert!(frames.clone().all(|i| (copy.pts_s(i) - index.pts_s(i)).abs() < 1e-9));
        assert!(x6_raw::archive_note(&dir).is_some_and(|n| !n.exact && n.target_bits_per_pixel == Some(1.0)));
        assert!(x6_raw::archive_note(&take).is_none());
        // A frame of the copy against the take's.
        let layout = index.take.layout;
        let (mut record, mut theirs, mut ours) = (Vec::new(), vec![0u16; layout.samples()], vec![0u16; layout.samples()]);
        let i = frames.len() / 2;
        x6_raw::decode_pair(&mut File::open(index.take.capture_path()).unwrap(), &index.pairs[i], index.take.format(), Channels::Both, &mut record, &mut theirs, 0).unwrap();
        x6_raw::decode_pair(&mut File::open(copy.take.capture_path()).unwrap(), &copy.pairs[i], copy.take.format(), Channels::Both, &mut record, &mut ours, 0).unwrap();
        let white = index.take.white_level;
        let (mut squares, mut whites, mut kept, mut made) = (0.0f64, 0usize, 0usize, 0usize);
        for (a, b) in theirs.iter().zip(&ours) {
            squares += (*a as f64 - *b as f64).powi(2);
            whites += (*a >= white) as usize;
            kept += (*a >= white && *b >= white) as usize;
            made += (*a < white && *b >= white) as usize;
        }
        let rms = (squares / theirs.len() as f64).sqrt();
        println!("frame {i}: {rms:.3} values rms from the take's; {whites} saturated samples, {kept} of them in the copy, {made} more there");
        assert!(rms < 0.004 * (white - index.take.black_level) as f64, "{rms}");
        // (A sample at the very edge of a saturated patch may come back below it: a handful in a frame.)
        assert!(whites - kept <= whites / 2000 + 2 && made <= whites / 2000 + 2);
        // One channel alone is the same channel.
        let mut one = vec![0u16; layout.samples()];
        if layout.channels() == 2 {
            x6_raw::decode_pair(&mut File::open(copy.take.capture_path()).unwrap(), &copy.pairs[i], copy.take.format(), Channels::One(1), &mut record, &mut one, 0).unwrap();
            assert!(one[layout.eye_samples()..] == ours[layout.eye_samples()..] && one[..layout.eye_samples()].iter().all(|v| *v == 0));
        }
        assert!(archive_take(&dir, &dir.with_extension("again"), 0..2, &ArchiveOptions::default(), |_| {}, &never).unwrap_err().to_string().contains("already"));
        assert!(archive_take(&take, &dir, 0..2, &ArchiveOptions::default(), |_| {}, &never).unwrap_err().to_string().contains("holds a capture"));
        // Stopped: nothing of a capture is left behind.
        let stopped = dir.with_extension("stopped");
        let _ = std::fs::remove_dir_all(&stopped);
        assert!(archive_take(&take, &stopped, frames.clone(), &ArchiveOptions::default(), |_| {}, &AtomicBool::new(true)).is_err());
        assert!(!stopped.join(x6b::CAPTURE_FILE).exists() && !stopped.join(x6b::SEAL_FILE).exists());
        let _ = std::fs::remove_dir_all(&stopped);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A picture through an encoder and back: every code within a few of
    /// what went in (skipped where no encoder opens).
    #[test]
    fn a_picture_comes_back_through_the_encoder() {
        let (w, h) = (256usize, 128usize);
        let codes: Vec<u16> = (0..w * h).map(|i| (100 + (i % w) * 2 + i / w) as u16).collect();
        let Some(mut encoder) = PictureEncoder::ALL.iter().find_map(|k| Encoder::open(*k, w, h, 0.9).ok()) else { return };
        for _ in 0..3 {
            encoder.send(&codes).unwrap();
        }
        encoder.finish().unwrap();
        let out = std::mem::take(&mut encoder.ready);
        assert_eq!(out.iter().map(|(n, _)| *n).collect::<Vec<_>>(), [0, 1, 2]);
        for (_, unit) in &out {
            let types: Vec<u8> = nal_units(unit).iter().map(|u| u.0).collect();
            assert!([32u8, 33, 34].iter().all(|t| types.contains(t)), "{types:?}");
        }
        let units: Vec<&[u8]> = out.iter().map(|(_, u)| &u[..]).collect();
        let back = decode_pictures(&units, w, h).unwrap();
        for picture in &back {
            let got: Vec<u16> = (0..h).flat_map(|r| picture.row(r, w).chunks_exact(2).map(|px| u16::from_le_bytes([px[0], px[1]]) >> picture.shift).collect::<Vec<u16>>()).collect();
            let worst = got.iter().zip(&codes).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
            assert!(got.len() == w * h && worst <= 6, "{worst}");
        }
        // The four planes of a frame back into its mosaic.
        let values: Vec<u16> = (0..CODES as u16).collect();
        let mut mosaic = vec![0u16; 4 * w * h];
        let four: Vec<&[u8]> = (0..4).map(|_| units[0]).collect();
        put_planes(&decode_pictures(&four, w, h).unwrap(), 2 * w, 2 * h, &values, &mut mosaic);
        for (y, x) in [(0usize, 0usize), (1, 1), (0, 2 * w - 1), (2 * h - 1, 0), (77, 130)] {
            assert!((mosaic[y * 2 * w + x] as i32 - codes[(y / 2) * w + x / 2] as i32).abs() <= 6, "{y} {x}");
        }
    }
}
