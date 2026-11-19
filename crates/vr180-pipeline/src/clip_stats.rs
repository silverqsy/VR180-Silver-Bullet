//! One set of tone statistics for a whole clip.
//!
//! A raw clip's development ([`crate::raw_tone`]) and the tone controls of
//! every other clip ([`crate::tone_linear`]) are derived from statistics of
//! the picture: its brightest and its darkest level and how its levels are
//! spread ([`FrameStats`]). Read off each frame by itself they follow that
//! frame: when the picture's content changes (a pan lets more sky in, a tree
//! leaves the frame) the rendering of everything else in it changes too, and
//! a clip shot at one exposure brightens and darkens by itself. Nothing in
//! the recording does that: the controls would.
//!
//! So a clip is rendered by ONE set of statistics: those of frames spread
//! over it, taken together as one picture (the frames side by side, the way
//! a frame's two eyes already are one picture). [`measure`] draws
//! [`SAMPLES`] frames the way a frame is measured (each eye's half-sphere as
//! its lens sees it: no stabilization, no view adjustment, so the view that
//! is drawn changes nothing) and keeps a small copy of each
//! ([`ClipStats`]).
//!
//! What the statistics are depends on how the picture is read: on a raw
//! clip's white balance, and for any other clip on the light its codes stand
//! for, which follows the grade. They are therefore worked out from the
//! copies when they are asked for ([`ClipStats::raw`], [`ClipStats::tone`]):
//! a change of white balance or of grade needs no frame decoded again, and
//! the preview, the export and a still of one clip all read the same copies
//! ([`for_clip`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::fisheye_decode::{FisheyePairIter, SegmentedFisheyeIter};
use crate::fisheye_export::FisheyeExportConfig;
use crate::gpu::{Device, EquirectRotation, EyePicture, EyeProjection};
use crate::raw_gpu::{stats_from_cells, KEY_CELLS, RAW_REFERENCE_SIDE};
use crate::raw_tone::{DevelopGeometry, FrameStats, BASE_FLOOR};
use crate::{Error, Result, SourceKind};

/// Frames of a clip its statistics are read from, spread evenly over it.
pub const SAMPLES: usize = 16;

/// The side, in pixels, of the copy kept of each sampled eye: two pixels to
/// a cell of the measurement when a frame has two eyes, one when it has one.
pub const SIDE: usize = 126;

/// Pixels of a drawn eye along one side of a pixel of its copy.
const REDUCE: usize = RAW_REFERENCE_SIDE as usize / SIDE;

/// What the copies hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Values {
    /// A raw clip's camera-linear RGB (the sensor's white = 1).
    Camera,
    /// Code values, 0..1.
    Codes,
}

/// A clip's sampled frames, and the statistics they give.
pub struct ClipStats {
    values: Values,
    /// Eyes of a frame: 1 or 2.
    eyes: usize,
    /// Every eye of every sampled frame, [`SIDE`] × [`SIDE`] RGB, one after
    /// the other.
    pictures: Vec<f32>,
    /// What was asked for last, and under which reading of the pictures.
    raw: Mutex<Option<([u32; 12], FrameStats)>>,
    tone: Mutex<Option<((u64, usize), FrameStats)>>,
}

impl std::fmt::Debug for ClipStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipStats").field("values", &self.values).field("eyes", &self.eyes).field("frames", &self.frames()).finish()
    }
}

/// The picture a cell's level is the level of: Rec.709's weights for a
/// picture of code values, as its tone stage counts luminance.
const CODE_LUMA: [f32; 3] = crate::tone_linear::LUMA;

impl ClipStats {
    /// `pictures`: every eye of every sampled frame, [`SIDE`] × [`SIDE`]
    /// RGB each, `eyes` to a frame.
    pub fn new(values: Values, eyes: usize, pictures: Vec<f32>) -> Self {
        let eyes = eyes.clamp(1, 2);
        assert_eq!(pictures.len() % (SIDE * SIDE * 3 * eyes), 0, "whole frames of {eyes} eye(s)");
        Self { values, eyes, pictures, raw: Mutex::new(None), tone: Mutex::new(None) }
    }

    /// Frames sampled.
    pub fn frames(&self) -> usize {
        self.pictures.len() / (SIDE * SIDE * 3 * self.eyes)
    }

    pub fn values(&self) -> Values {
        self.values
    }

    /// The statistics of all the pictures as one: `pixel` gives what a
    /// pixel of a copy adds to its cell (the cell's value, and its level in
    /// log₂). The cells and what is made of them are a frame's
    /// ([`stats_from_cells`]), with more of them.
    fn pooled(&self, pixel: impl Fn([f32; 3]) -> ([f32; 3], f32)) -> FrameStats {
        // A frame is its eyes side by side: the cells of two eyes are half
        // as many along a side as those of one.
        let cells = KEY_CELLS as usize / self.eyes;
        let px = SIDE / cells;
        let mut bytes = Vec::with_capacity(self.pictures.len() / 3 / (px * px) * 16);
        for eye in self.pictures.chunks_exact(SIDE * SIDE * 3) {
            for cy in 0..cells {
                for cx in 0..cells {
                    let (mut sum, mut level) = ([0f32; 3], 0f32);
                    for y in cy * px..(cy + 1) * px {
                        for x in cx * px..(cx + 1) * px {
                            let i = (y * SIDE + x) * 3;
                            let (rgb, l) = pixel([eye[i], eye[i + 1], eye[i + 2]]);
                            for k in 0..3 {
                                sum[k] += rgb[k];
                            }
                            level += l;
                        }
                    }
                    let n = (px * px) as f32;
                    for v in [sum[0] / n, sum[1] / n, sum[2] / n, level / n] {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        stats_from_cells(&bytes, cells, cells).unwrap_or_else(FrameStats::unknown)
    }

    /// The statistics a raw clip is developed by, under the white balance
    /// of `geometry`: what `raw_measure.wgsl` reads off a frame, off the
    /// copies.
    pub fn raw(&self, geometry: &DevelopGeometry) -> FrameStats {
        let mut key = [0u32; 12];
        for (k, v) in key.iter_mut().zip(geometry.neutral.iter().chain(geometry.camera_to_work.iter().flatten())) {
            *k = v.to_bits();
        }
        if let Ok(kept) = self.raw.lock() {
            if let Some((k, stats)) = &*kept {
                if *k == key {
                    return stats.clone();
                }
            }
        }
        let luma = crate::raw_color::prophoto_luma().map(|v| v as f32);
        let stats = self.pooled(|camera| {
            let rgb = geometry.scene(camera);
            let lit = rgb.map(|v| v.max(0.0));
            // A channel above 1 is a blown highlight rebuilt: for the
            // levels it counts as white.
            let over = lit[0].max(lit[1]).max(lit[2]).max(1.0);
            (rgb, ((lit[0] * luma[0] + lit[1] * luma[1] + lit[2] * luma[2]) / over).max(BASE_FLOOR).log2())
        });
        if let Ok(mut kept) = self.raw.lock() {
            *kept = Some((key, stats.clone()));
        }
        stats
    }

    /// The statistics the tone controls of a clip that is not raw go by, in
    /// the light `decode` gives a code (a tone stage's table: equally
    /// spaced codes from 0 to 1): what `tone_measure.wgsl` reads off a
    /// frame, off the copies.
    pub fn tone(&self, decode: &[f32]) -> FrameStats {
        if decode.len() < 2 {
            return FrameStats::unknown();
        }
        // FNV-1a over the table's bits.
        let hash = decode.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3));
        let key = (hash, decode.len());
        if let Ok(kept) = self.tone.lock() {
            if let Some((k, stats)) = &*kept {
                if *k == key {
                    return stats.clone();
                }
            }
        }
        let last = (decode.len() - 1) as f32;
        let light = |code: f32| -> f32 {
            let y = code.clamp(0.0, 1.0) * last;
            let i = y.floor().min(last - 1.0);
            let (f, k) = (y - i, i as usize);
            decode[k] * (1.0 - f) + decode[k + 1] * f
        };
        let stats = self.pooled(|codes| {
            let lit = codes.map(|c| light(c).max(0.0));
            (lit, (lit[0] * CODE_LUMA[0] + lit[1] * CODE_LUMA[1] + lit[2] * CODE_LUMA[2]).max(BASE_FLOOR).log2())
        });
        if let Ok(mut kept) = self.tone.lock() {
            *kept = Some((key, stats.clone()));
        }
        stats
    }
}

/// A drawn eye (`RAW_REFERENCE_SIDE` square, `channels` values of `bytes`
/// bytes a pixel, the first three its RGB) as its copy: the mean of every
/// [`REDUCE`] × [`REDUCE`] pixels, 0..1.
fn reduce(drawn: &[u8], channels: usize, bytes: usize) -> Vec<f32> {
    let side = RAW_REFERENCE_SIDE as usize;
    let px = channels * bytes;
    assert_eq!(drawn.len(), side * side * px, "a drawn eye");
    let scale = if bytes == 2 { 1.0 / 65535.0 } else { 1.0 / 255.0 } / (REDUCE * REDUCE) as f64;
    let mut out = Vec::with_capacity(SIDE * SIDE * 3);
    for cy in 0..SIDE {
        for cx in 0..SIDE {
            let mut sum = [0u64; 3];
            for y in cy * REDUCE..(cy + 1) * REDUCE {
                let row = &drawn[(y * side + cx * REDUCE) * px..(y * side + (cx + 1) * REDUCE) * px];
                for p in row.chunks_exact(px) {
                    for (k, s) in sum.iter_mut().enumerate() {
                        *s += if bytes == 2 { u16::from_le_bytes([p[k * 2], p[k * 2 + 1]]) as u64 } else { p[k] as u64 };
                    }
                }
            }
            out.extend(sum.map(|s| (s as f64 * scale) as f32));
        }
    }
    out
}

/// The decoder a fisheye-family clip's frames are sampled through: what an
/// export opens, at a size and depth that suffice for a copy this small.
fn open_frames(cfg: &FisheyeExportConfig) -> Result<Box<dyn FisheyePairIter>> {
    let kind = cfg.source_kind;
    if kind.is_raw_video() {
        // A raw take: its eyes at half size, in memory, camera-linear.
        return Ok(Box::new(crate::x6_raw::X6RawIter::new(
            &cfg.source_path, kind.dual_stream_iter_swap(cfg.fisheye_swap_eyes), true, 1, None)?));
    }
    let mut open: Box<dyn FnMut(&Path) -> Result<Box<dyn FisheyePairIter>>> = if kind == SourceKind::BlackmagicRaw {
        // A quarter of the frame each way: a full decode of a 12K frame
        // takes long, and the copy is 126 pixels wide.
        let (swap, color) = (cfg.fisheye_swap_eyes, cfg.braw_color.clone());
        Box::new(move |p: &Path| -> Result<Box<dyn FisheyePairIter>> {
            let info = vr180_braw::BrawInfo::probe(p).map_err(|e| Error::Ffmpeg(format!("braw probe: {e}")))?;
            let layout = crate::braw_imu::braw_layout(p, &info);
            Ok(Box::new(crate::fisheye_decode::BrawFisheyeIter::new_with_options(
                p, &info, 8, vr180_braw::DecodeScale::Quarter, vr180_braw::DecodeLook::Film, swap, layout.mono, 1, 0, &color)?))
        })
    } else {
        crate::fisheye_export::fisheye_export_opener(kind, cfg.fisheye_swap_eyes, 8, cfg.braw_color.clone(), None, None)
    };
    if cfg.segments.len() > 1 {
        let durations: Vec<f64> = cfg.segments.iter()
            .map(|p| crate::decode::probe_video_duration_via_moov(p).ok()
                .or_else(|| crate::decode::probe_video(p).ok().map(|pr| pr.duration_sec))
                .unwrap_or(0.0))
            .collect();
        Ok(Box::new(SegmentedFisheyeIter::new(&cfg.segments, &durations, open)?))
    } else {
        open(&cfg.source_path)
    }
}

/// The files a clip's pictures are in.
fn files(cfg: &FisheyeExportConfig) -> Vec<PathBuf> {
    if cfg.segments.is_empty() { vec![cfg.source_path.clone()] } else { cfg.segments.clone() }
}

/// Sample the clip of `cfg` (an export's configuration of it: source,
/// lens, eye order; its view, its stabilization and its trim are not used)
/// and keep the copies its statistics are read from. The whole clip is
/// sampled, whatever part of it is trimmed for an export: its look does not
/// move with the trim.
pub fn measure(pipeline: &Device, cfg: &FisheyeExportConfig, cancel: &AtomicBool) -> Result<ClipStats> {
    let started = std::time::Instant::now();
    let files = files(cfg);
    let total: f64 = files.iter().map(|p| crate::decode::probe_video(p).map(|pr| pr.duration_sec).unwrap_or(0.0)).sum();
    let times: Vec<f64> = (0..SAMPLES).map(|k| total * (k as f64 + 0.5) / SAMPLES as f64).collect();
    let side = RAW_REFERENCE_SIDE;
    let cancelled = || -> Result<()> {
        if cancel.load(Ordering::SeqCst) { Err(Error::Ffmpeg("clip statistics: cancelled".into())) } else { Ok(()) }
    };
    let mut pictures: Vec<f32> = Vec::new();
    let (mut eyes, mut frames) = (2usize, 0usize);
    let values = if cfg.source_kind.is_raw_video() { Values::Camera } else { Values::Codes };
    if cfg.source_kind.is_eac() {
        // A `.360` file: its two streams, a cube map per lens, each drawn
        // as a half-equirect (left = lens B, as an export has it).
        use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
        let mut decoder = crate::decode::SegmentedStreamPairIter::new(&files, crate::decode::HwDecode::Auto, 0)?;
        let dims = decoder.dims();
        if !dims.is_valid() {
            return Err(Error::Ffmpeg("clip statistics: invalid EAC layout".into()));
        }
        let cross_w = dims.cross_w();
        let cw = cross_w as usize;
        let (lens_l, lens_r) = crate::fisheye_export::resolve_eac_lens_pair(cfg);
        let rs = crate::gpu::EquirectRsParams::DISABLED;
        let (mut cross_a, mut cross_b) = (vec![0u8; cw * cw * 3], vec![0u8; cw * cw * 3]);
        for &t in &times {
            cancelled()?;
            if decoder.seek(t).is_err() { continue; }
            let Some(pair) = decoder.next_pair()? else { continue };
            assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
            assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);
            for (cross, lens) in [(&cross_b, &lens_l), (&cross_a, &lens_r)] {
                let drawn = pipeline.project_cross_to_equirect_texture(cross, cross_w, side, side, EquirectRotation::IDENTITY, rs, lens)?;
                pictures.extend(reduce(&pipeline.read_texture_rgb8(&drawn, side, side)?, 3, 1));
            }
            frames += 1;
        }
    } else {
        let mut decoder = open_frames(cfg)?;
        // The clip's own lens record, without its motion data.
        let lens_cfg = FisheyeExportConfig { stabilize: false, ..cfg.clone() };
        let carrier = crate::stereo_align::lens_carrier(&lens_cfg);
        let mut calib = None;
        for &t in &times {
            cancelled()?;
            if decoder.seek(t).is_err() { continue; }
            let Some(pair) = decoder.next_pair()? else { continue };
            let (calib_l, calib_r) = *calib.get_or_insert_with(|| {
                crate::fisheye_export::resolve_source_calib_pair(cfg, pair.eye_w, pair.eye_h, carrier.as_ref())
            });
            let mono = pair.single_eye();
            let mut draw = |bytes: &[u8], calib| -> Result<()> {
                let drawn = pipeline.project_eye(&EyeProjection::new(
                    EyePicture::Bytes16 { bytes, bit_depth: pair.bit_depth, w: pair.eye_w, h: pair.eye_h },
                    side, side, EquirectRotation::IDENTITY, calib).wide())?;
                pictures.extend(reduce(&pipeline.read_texture_rgba64(&drawn, side, side)?, 4, 2));
                Ok(())
            };
            draw(&pair.left, calib_l)?;
            if !mono {
                draw(&pair.right, calib_r)?;
            }
            eyes = if mono { 1 } else { 2 };
            frames += 1;
        }
    }
    if frames == 0 {
        return Err(Error::Ffmpeg(format!("clip statistics: no frame of {} could be read", cfg.source_path.display())));
    }
    tracing::info!(
        "clip statistics: {frames} frames of {} ({} eye(s) each) sampled in {:.2?}",
        cfg.source_path.display(), eyes, started.elapsed());
    Ok(ClipStats::new(values, eyes, pictures))
}

/// What tells one clip's pictures from another's: its files as they are on
/// disk, and what decides how they decode.
#[derive(Debug, Clone, PartialEq)]
struct ClipKey {
    files: Vec<PathBuf>,
    kind: SourceKind,
    swap_eyes: bool,
    /// A decoder's own colour settings (Blackmagic RAW).
    decode: String,
    /// Size and time of change of the first file (of a raw take: of its
    /// capture).
    stamp: Option<(u64, std::time::SystemTime)>,
}

fn key_of(cfg: &FisheyeExportConfig) -> ClipKey {
    let files = files(cfg);
    let stamped = match files.first() {
        Some(first) if cfg.source_kind.is_raw_video() => crate::x6_raw::take_dir(first)
            .map_or_else(|| first.clone(), |dir| dir.join(vr180_fisheye::x6b::CAPTURE_FILE)),
        Some(first) => first.clone(),
        None => cfg.source_path.clone(),
    };
    let stamp = std::fs::metadata(&stamped).ok().and_then(|m| Some((m.len(), m.modified().ok()?)));
    ClipKey {
        files,
        kind: cfg.source_kind,
        swap_eyes: cfg.fisheye_swap_eyes,
        decode: if cfg.source_kind == SourceKind::BlackmagicRaw { format!("{:?}", cfg.braw_color) } else { String::new() },
        stamp,
    }
}

/// The clips sampled in this run of the program, the last asked for first.
fn kept() -> &'static Mutex<Vec<(ClipKey, Arc<ClipStats>)>> {
    static KEPT: OnceLock<Mutex<Vec<(ClipKey, Arc<ClipStats>)>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(Vec::new()))
}

/// Clips whose copies are kept (a few megabytes each).
const KEPT_CLIPS: usize = 64;

/// The clip's statistics where it has been sampled already.
pub fn cached(cfg: &FisheyeExportConfig) -> Option<Arc<ClipStats>> {
    let key = key_of(cfg);
    let mut kept = kept().lock().ok()?;
    let at = kept.iter().position(|(k, _)| *k == key)?;
    let entry = kept.remove(at);
    let stats = entry.1.clone();
    kept.insert(0, entry);
    Some(stats)
}

/// The clip's statistics: sampled now ([`measure`]) unless they were before.
/// The preview of a clip, its export and its stills come through here, so
/// they all read the same copies.
pub fn for_clip(pipeline: &Device, cfg: &FisheyeExportConfig, cancel: &AtomicBool) -> Result<Arc<ClipStats>> {
    if let Some(stats) = cached(cfg) {
        return Ok(stats);
    }
    let key = key_of(cfg);
    let stats = Arc::new(measure(pipeline, cfg, cancel)?);
    if let Ok(mut kept) = kept().lock() {
        kept.retain(|(k, _)| *k != key);
        kept.insert(0, (key, stats.clone()));
        kept.truncate(KEPT_CLIPS);
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_color::RawCamera;
    use crate::raw_tone::{Measure, RawDevelop, RawSettings};

    /// `frames` frames of two eyes, every eye one flat value.
    fn flat(values: Values, eyes: usize, levels: &[f32]) -> ClipStats {
        let mut pictures = Vec::new();
        for l in levels {
            pictures.extend(std::iter::repeat(*l).take(SIDE * SIDE * 3));
        }
        ClipStats::new(values, eyes, pictures)
    }

    fn stage() -> RawDevelop {
        RawDevelop { camera: RawCamera::X6, settings: RawSettings::default(), measure: Measure::Wait }
    }

    #[test]
    fn the_frames_are_one_picture() {
        // Four eyes (two frames of two) at 0.02, 0.04, 0.08 and 0.16 of the
        // sensor's white: the clip's brightest and darkest are the pictures'
        // brightest and darkest, and its levels are all of theirs.
        let clip = flat(Values::Camera, 2, &[0.02, 0.04, 0.08, 0.16]);
        assert_eq!(clip.frames(), 2);
        let g = stage().geometry();
        let stats = clip.raw(&g);
        let scene = |v: f32| g.scene([v; 3]);
        let top = scene(0.16).iter().cloned().fold(0f32, f32::max);
        assert!((stats.keys.max - top).abs() < 1e-6 * top, "{} vs {top}", stats.keys.max);
        let luma = crate::raw_color::prophoto_luma().map(|v| v as f32);
        let y = |v: f32| { let s = scene(v); s[0] * luma[0] + s[1] * luma[1] + s[2] * luma[2] };
        assert!((stats.keys.dark - y(0.02)).abs() < 1e-6, "{} vs {}", stats.keys.dark, y(0.02));
        // 21 × 21 blocks an eye, four eyes.
        assert_eq!(stats.levels.len(), 4 * 21 * 21);
        assert!((stats.levels[0] - y(0.02).log2()).abs() < 1e-4 && (stats.levels[stats.levels.len() - 1] - y(0.16).log2()).abs() < 1e-4);
        // A quarter of the levels belongs to each eye.
        let quarter = stats.levels.len() / 4;
        assert!((stats.levels[quarter - 1] - y(0.02).log2()).abs() < 1e-4 && (stats.levels[quarter] - y(0.04).log2()).abs() < 1e-4);
    }

    #[test]
    fn one_eye_has_the_cells_of_a_frame_of_one() {
        let clip = flat(Values::Camera, 1, &[0.05, 0.1]);
        assert_eq!(clip.frames(), 2);
        // 126 cells a side, 42 blocks a side, two frames.
        assert_eq!(clip.raw(&stage().geometry()).levels.len(), 2 * 42 * 42);
    }

    #[test]
    fn the_white_balance_is_read_from_the_same_pictures() {
        // A grey of the as-shot white, and the same pictures under another
        // white: the statistics move, the pictures do not.
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let mut pictures = Vec::new();
        for _ in 0..2 * SIDE * SIDE {
            pictures.extend(n.map(|v| v * 0.2));
        }
        let clip = ClipStats::new(Values::Camera, 2, pictures);
        let as_shot = clip.raw(&stage().geometry());
        let warm = RawDevelop { settings: RawSettings { temperature: Some(9000.0), tint: Some(0.0), ..Default::default() }, ..stage() };
        let warmed = clip.raw(&warm.geometry());
        assert!((as_shot.keys.max - warmed.keys.max).abs() > 1e-3, "{} {}", as_shot.keys.max, warmed.keys.max);
        // Asked again, the first reading is worked out again, the same.
        assert_eq!(clip.raw(&stage().geometry()), as_shot);
    }

    #[test]
    fn codes_are_read_through_the_table_they_are_given() {
        let clip = flat(Values::Codes, 2, &[0.25, 0.5]);
        let linear: Vec<f32> = (0..=64).map(|i| i as f32 / 64.0).collect();
        let stats = clip.tone(&linear);
        assert!((stats.keys.max - 0.5).abs() < 1e-6 && (stats.keys.dark - 0.25).abs() < 1e-6, "{:?}", stats.keys);
        let squared: Vec<f32> = (0..=64).map(|i| (i as f32 / 64.0).powi(2)).collect();
        let stats = clip.tone(&squared);
        assert!((stats.keys.max - 0.25).abs() < 1e-6 && (stats.keys.dark - 0.0625).abs() < 1e-6, "{:?}", stats.keys);
        assert!((stats.levels[0] + 4.0).abs() < 1e-4 && (stats.levels[stats.levels.len() - 1] + 2.0).abs() < 1e-4);
        assert!(clip.tone(&[]).levels.is_empty());
    }

    /// A 16-bit eye of the size a clip's eyes are drawn at.
    fn drawn_eye(dev: &Device, value_at: &dyn Fn(usize, usize) -> [f32; 3]) -> wgpu::Texture {
        let side = RAW_REFERENCE_SIDE;
        let px: Vec<u16> = (0..(side * side) as usize).flat_map(|i| {
            let v = value_at(i % side as usize, i / side as usize);
            [(v[0] * 65535.0) as u16, (v[1] * 65535.0) as u16, (v[2] * 65535.0) as u16, u16::MAX]
        }).collect();
        let t = dev.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test_drawn_eye"),
            size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&px),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(side * 8), rows_per_image: Some(side) },
            wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 });
        t
    }

    /// How far two sets of statistics of one picture are apart: the keys
    /// (in stops) and the levels at their ranks (in stops).
    fn apart(a: &FrameStats, b: &FrameStats) -> (f32, f32) {
        assert_eq!(a.levels.len(), b.levels.len());
        let stops = |x: f32, y: f32| (x.max(1e-9).log2() - y.max(1e-9).log2()).abs();
        let keys = stops(a.keys.max, b.keys.max).max(stops(a.keys.dark, b.keys.dark));
        let levels = a.levels.iter().zip(b.levels.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        (keys, levels)
    }

    /// The copies of a frame, read on the CPU, give what the frame itself
    /// gives on the GPU: a clip that does not change is rendered as each
    /// of its frames would be.
    #[test]
    #[ignore = "needs a GPU"]
    fn the_copies_measure_as_the_frame_does() {
        let dev = Device::new().expect("device");
        let side = RAW_REFERENCE_SIDE as usize;
        // A picture that is smooth at the scale of a copy's pixel: levels
        // over eight stops across, brighter and darker bands down.
        let level = |x: usize, y: usize| 0.002 * 2f32.powf(8.0 * x as f32 / side as f32) * (1.0 + 0.6 * (y as f32 / 90.0).sin());
        // Raw: greys of the camera's white, the other eye a little warmer.
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let left = drawn_eye(&dev, &|x, y| n.map(|c| (c * level(x, y)).min(1.0)));
        let right = drawn_eye(&dev, &|x, y| [n[0] * 1.1, n[1], n[2] * 0.9].map(|c| (c * level(y, x)).min(1.0)));
        let copies = |a: &wgpu::Texture, b: &wgpu::Texture| -> Vec<f32> {
            let mut out = reduce(&dev.read_texture_rgba64(a, side as u32, side as u32).unwrap(), 4, 2);
            out.extend(reduce(&dev.read_texture_rgba64(b, side as u32, side as u32).unwrap(), 4, 2));
            out
        };
        let on_gpu = dev.raw_frame_stats(&[&left, &right], &stage());
        let clip = ClipStats::new(Values::Camera, 2, copies(&left, &right));
        let (keys, levels) = apart(&clip.raw(&stage().geometry()), &on_gpu);
        assert!(keys < 0.02 && levels < 0.02, "raw: keys {keys} stops, levels {levels} stops apart");
        // One eye alone.
        let on_gpu = dev.raw_frame_stats(&[&left], &stage());
        let clip = ClipStats::new(Values::Camera, 1, reduce(&dev.read_texture_rgba64(&left, side as u32, side as u32).unwrap(), 4, 2));
        let (keys, levels) = apart(&clip.raw(&stage().geometry()), &on_gpu);
        assert!(keys < 0.02 && levels < 0.02, "raw, one eye: keys {keys} stops, levels {levels} stops apart");
        // Codes, through a curve.
        let decode: Vec<f32> = (0..crate::tone_linear::CODE_SAMPLES).map(|i| (i as f32 / (crate::tone_linear::CODE_SAMPLES - 1) as f32).powf(2.2)).collect();
        let code = |x: usize, y: usize| (0.08 + 0.9 * x as f32 / side as f32) * (0.8 + 0.2 * (y as f32 / 90.0).sin());
        let left = drawn_eye(&dev, &|x, y| [code(x, y); 3]);
        let right = drawn_eye(&dev, &|x, y| [code(y, x), code(y, x) * 0.9, code(y, x) * 0.8]);
        let on_gpu = dev.linear_tone_stats(&[&left, &right], &decode, Measure::Wait);
        let clip = ClipStats::new(Values::Codes, 2, copies(&left, &right));
        let (keys, levels) = apart(&clip.tone(&decode), &on_gpu);
        assert!(keys < 0.02 && levels < 0.02, "codes: keys {keys} stops, levels {levels} stops apart");
    }

    /// With its clip's statistics in the plan a frame is rendered by them
    /// whatever it holds: a bright frame and a dark one get the same
    /// constants. Without, each gets its own.
    #[test]
    #[ignore = "needs a GPU"]
    fn the_frames_of_a_clip_are_rendered_alike() {
        use crate::gpu::ColorStackPlan;
        use crate::tone_linear::{LinearTone, ToneLight, ToneSettings};
        let dev = Device::new().expect("device");
        let side = RAW_REFERENCE_SIDE as usize;
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let grey = |top: f32| move |x: usize, _y: usize| -> f32 { top * 2f32.powf(-6.0 * (1.0 - x as f32 / side as f32)) };
        let (bright, dark) = (grey(0.8), grey(0.05));
        // A raw clip: Highlights pulled down, which goes by the levels.
        let (a, b) = (drawn_eye(&dev, &|x, y| n.map(|c| c * bright(x, y))), drawn_eye(&dev, &|x, y| n.map(|c| c * dark(x, y))));
        let read = |t: &wgpu::Texture| reduce(&dev.read_texture_rgba64(t, side as u32, side as u32).unwrap(), 4, 2);
        let clip = ClipStats::new(Values::Camera, 1, [read(&a), read(&b)].concat());
        let develop = RawDevelop { settings: RawSettings { highlights: -100.0, ..Default::default() }, ..stage() };
        let own = (dev.raw_develop_constants(&[&a], None, &develop, 1.0, None), dev.raw_develop_constants(&[&b], None, &develop, 1.0, None));
        assert_ne!(own.0, own.1, "each frame by itself: its own constants");
        let held = (dev.raw_develop_constants(&[&a], None, &develop, 1.0, Some(&clip)), dev.raw_develop_constants(&[&b], None, &develop, 1.0, Some(&clip)));
        assert_eq!(held.0, held.1, "the clip's statistics: one set of constants");
        // A clip of code values: the same of its tone controls.
        let (a, b) = (drawn_eye(&dev, &|x, y| [bright(x, y); 3]), drawn_eye(&dev, &|x, y| [dark(x, y); 3]));
        let clip = Arc::new(ClipStats::new(Values::Codes, 1, [read(&a), read(&b)].concat()));
        let tone = LinearTone { settings: ToneSettings { highlights: -100.0, ..Default::default() }, light: ToneLight::Shown, pre_stops: 0.0, measure: Measure::Wait };
        let mut plan = ColorStackPlan { tone: Some(tone), ..Default::default() };
        assert!(plan.wants_clip_stats());
        let own = (dev.linear_tone_constants(&[&a], &plan), dev.linear_tone_constants(&[&b], &plan));
        assert!(own.0.is_some() && own.0 != own.1, "each frame by itself: its own constants");
        plan.clip_stats = Some(clip);
        let held = (dev.linear_tone_constants(&[&a], &plan), dev.linear_tone_constants(&[&b], &plan));
        assert!(held.0.is_some() && held.0 == held.1, "the clip's statistics: one set of constants");
        // A plan nothing in which goes by the levels wants none.
        let exposure_only = ColorStackPlan { tone: Some(LinearTone { settings: ToneSettings { exposure: 1.0, ..Default::default() }, ..tone }), ..Default::default() };
        assert!(!exposure_only.wants_clip_stats() && !ColorStackPlan::default().wants_clip_stats());
    }

    #[test]
    fn a_drawn_eye_is_reduced_by_its_means() {
        let side = RAW_REFERENCE_SIDE as usize;
        // 16-bit, four channels: the value rises with the column.
        let mut drawn = Vec::with_capacity(side * side * 8);
        for _y in 0..side {
            for x in 0..side {
                let v = (x * 60) as u16;
                for c in [v, v / 2, 0, 65535] {
                    drawn.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        let copy = reduce(&drawn, 4, 2);
        assert_eq!(copy.len(), SIDE * SIDE * 3);
        // Pixel 3 of a row: the mean of columns 24..32.
        let want = (24..32).map(|x| (x * 60) as f64).sum::<f64>() / 8.0 / 65535.0;
        assert!((copy[3 * 3] as f64 - want).abs() < 1e-7 && copy[3 * 3 + 2] == 0.0);
        // 8-bit, three channels.
        let drawn = vec![51u8; side * side * 3];
        assert!(reduce(&drawn, 3, 1).iter().all(|v| (*v - 0.2).abs() < 1e-6));
    }
}
