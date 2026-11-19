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
use crate::raw_tone::{DevelopConstants, DevelopGeometry, Exposed, FrameStats, RawDevelop, BASE_FLOOR};
use crate::{Error, Result, SourceKind};

/// Frames of a clip its statistics are read from, spread evenly over it.
pub const SAMPLES: usize = 16;

/// The side, in pixels, of the copy kept of each sampled eye: two pixels to
/// a cell of the measurement when a frame has two eyes, one when it has one.
pub const SIDE: usize = 126;

/// What the copies hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Values {
    /// A raw clip's camera-linear RGB (the sensor's white = 1).
    Camera,
    /// Code values, 0..1.
    Codes,
}

/// The side, in pixels, of the copy an automatic exposure keeps of each eye
/// of a frame it meters ([`Meter`]).
pub const METER_SIDE: usize = 21;

/// The side an eye is drawn at for that copy.
const METER_DRAWN: usize = 252;

/// Seconds between two metered frames of a clip, and the most a clip has:
/// a longer clip is metered at wider intervals.
pub const METER_INTERVAL_S: f64 = 0.5;
pub const METER_FRAMES: usize = 600;

/// Small copies of a clip's frames at short intervals: what an automatic
/// exposure reads the picture's brightness from as the clip goes on.
struct Meter {
    /// The clip time of each metered frame, seconds, rising.
    times: Vec<f64>,
    /// Eyes of a metered frame that were copied (one is enough to meter by).
    eyes: usize,
    /// Every copied eye of every metered frame, [`METER_SIDE`] ×
    /// [`METER_SIDE`] camera RGB, one after the other.
    copies: Vec<f32>,
}

/// The cells of a raw clip's sampled pictures under one white balance, at
/// the exposure they were recorded with: what its statistics are put
/// together from, whatever exposure each frame is then given.
struct RawCells {
    /// Four values a cell (its mean scene RGB, its level in log₂), the
    /// pictures one after the other.
    cells: Vec<f32>,
    /// Cells along a side of a picture.
    side: usize,
    /// Of each frame by itself: the top of its levels and its darkest
    /// cell's luminance.
    tops: Vec<f32>,
    darks: Vec<f32>,
}

/// A clip's sampled frames, and the statistics they give.
pub struct ClipStats {
    values: Values,
    /// Eyes of a frame: 1 or 2.
    eyes: usize,
    /// Every eye of every sampled frame, [`SIDE`] × [`SIDE`] RGB, one after
    /// the other.
    pictures: Vec<f32>,
    /// The clip time of each sampled frame, seconds (empty when not known).
    frame_times: Vec<f64>,
    /// A raw clip metered for an automatic exposure.
    meter: Option<Meter>,
    /// What was asked for last, and under which reading of the pictures.
    raw: Mutex<Option<([u32; 12], Arc<RawCells>, FrameStats)>>,
    tone: Mutex<Option<((u64, usize), FrameStats)>>,
    auto: Mutex<Option<([u32; 19], Arc<AutoExposure>)>>,
}

impl std::fmt::Debug for ClipStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipStats").field("values", &self.values).field("eyes", &self.eyes).field("frames", &self.frames())
            .field("metered", &self.meter.as_ref().map(|m| m.times.len())).finish()
    }
}

/// The picture a cell's level is the level of: Rec.709's weights for a
/// picture of code values, as its tone stage counts luminance.
const CODE_LUMA: [f32; 3] = crate::tone_linear::LUMA;

fn geometry_key(geometry: &DevelopGeometry) -> [u32; 12] {
    let mut key = [0u32; 12];
    for (k, v) in key.iter_mut().zip(geometry.neutral.iter().chain(geometry.camera_to_work.iter().flatten())) {
        *k = v.to_bits();
    }
    key
}

impl ClipStats {
    /// `pictures`: every eye of every sampled frame, [`SIDE`] × [`SIDE`]
    /// RGB each, `eyes` to a frame.
    pub fn new(values: Values, eyes: usize, pictures: Vec<f32>) -> Self {
        let eyes = eyes.clamp(1, 2);
        assert_eq!(pictures.len() % (SIDE * SIDE * 3 * eyes), 0, "whole frames of {eyes} eye(s)");
        Self {
            values, eyes, pictures, frame_times: Vec::new(), meter: None,
            raw: Mutex::new(None), tone: Mutex::new(None), auto: Mutex::new(None),
        }
    }

    /// These frames with the clip time of each, and the clip metered at
    /// `metered` times: `eyes` eyes of every such frame as [`METER_SIDE`] ×
    /// [`METER_SIDE`] camera RGB in `copies`.
    pub fn with_meter(mut self, frame_times: Vec<f64>, metered: Vec<f64>, eyes: usize, copies: Vec<f32>) -> Self {
        assert_eq!(frame_times.len(), self.frames(), "a time for each sampled frame");
        assert_eq!(copies.len(), metered.len() * eyes * METER_SIDE * METER_SIDE * 3, "a copy of each metered eye");
        self.frame_times = frame_times;
        self.meter = (!metered.is_empty() && eyes > 0).then_some(Meter { times: metered, eyes, copies });
        self
    }

    /// Frames sampled.
    pub fn frames(&self) -> usize {
        self.pictures.len() / (SIDE * SIDE * 3 * self.eyes)
    }

    pub fn values(&self) -> Values {
        self.values
    }

    /// The clip was metered for an automatic exposure.
    pub fn has_meter(&self) -> bool {
        self.meter.is_some()
    }

    /// The cells of all the pictures: `pixel` gives what a pixel of a copy
    /// adds to its cell (the cell's value, and its level in log₂). Four
    /// values a cell; also the cells along a side of a picture.
    fn cells(&self, pixel: impl Fn([f32; 3]) -> ([f32; 3], f32)) -> (Vec<f32>, usize) {
        // A frame is its eyes side by side: the cells of two eyes are half
        // as many along a side as those of one.
        let side = KEY_CELLS as usize / self.eyes;
        let px = SIDE / side;
        let mut cells = Vec::with_capacity(self.pictures.len() / 3 / (px * px) * 4);
        for eye in self.pictures.chunks_exact(SIDE * SIDE * 3) {
            for cy in 0..side {
                for cx in 0..side {
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
                    cells.extend([sum[0] / n, sum[1] / n, sum[2] / n, level / n]);
                }
            }
        }
        (cells, side)
    }

    /// The statistics of cells as one picture: a frame's
    /// ([`stats_from_cells`]), with more of them.
    fn stats_of(cells: &[f32], side: usize) -> FrameStats {
        stats_from_cells(bytemuck::cast_slice(cells), side, side).unwrap_or_else(FrameStats::unknown)
    }

    /// A raw clip's cells under the white balance of `geometry`, and its
    /// statistics as it was recorded.
    fn raw_cells(&self, geometry: &DevelopGeometry) -> (Arc<RawCells>, FrameStats) {
        let key = geometry_key(geometry);
        if let Ok(kept) = self.raw.lock() {
            if let Some((k, cells, stats)) = &*kept {
                if *k == key {
                    return (cells.clone(), stats.clone());
                }
            }
        }
        let luma = crate::raw_color::prophoto_luma().map(|v| v as f32);
        let (cells, side) = self.cells(|camera| {
            let rgb = geometry.scene(camera);
            let lit = rgb.map(|v| v.max(0.0));
            // A channel above 1 is a blown highlight rebuilt: for the
            // levels it counts as white.
            let over = lit[0].max(lit[1]).max(lit[2]).max(1.0);
            (rgb, ((lit[0] * luma[0] + lit[1] * luma[1] + lit[2] * luma[2]) / over).max(BASE_FLOOR).log2())
        });
        let stats = Self::stats_of(&cells, side);
        // Each frame by itself: the top of its levels, its darkest cell.
        let per_frame = side * side * 4 * self.eyes;
        let (tops, darks) = cells.chunks_exact(per_frame.max(1)).map(|frame| {
            let own = Self::stats_of(frame, side);
            (own.levels.last().copied().unwrap_or(0.0), own.keys.dark)
        }).unzip();
        let cells = Arc::new(RawCells { cells, side, tops, darks });
        if let Ok(mut kept) = self.raw.lock() {
            *kept = Some((key, cells.clone(), stats.clone()));
        }
        (cells, stats)
    }

    /// The statistics a raw clip is developed by, under the white balance
    /// of `geometry`: what `raw_measure.wgsl` reads off a frame, off the
    /// copies.
    pub fn raw(&self, geometry: &DevelopGeometry) -> FrameStats {
        self.raw_cells(geometry).1
    }

    /// The statistics of a raw clip whose frames have each been given an
    /// exposure of their own (`stops`, one per sampled frame): every frame's
    /// cells that much brighter, as an exposure moves a frame's statistics,
    /// and all of them as one picture. With it what those statistics no
    /// longer tell ([`Exposed`]).
    fn raw_exposed(cells: &RawCells, eyes: usize, stops: &[f32]) -> (FrameStats, Exposed) {
        let per_frame = cells.side * cells.side * 4 * eyes;
        let mut moved = cells.cells.clone();
        for (frame, e) in moved.chunks_exact_mut(per_frame.max(1)).zip(stops) {
            let gain = 2f32.powf(*e);
            for cell in frame.chunks_exact_mut(4) {
                for v in &mut cell[..3] {
                    *v *= gain;
                }
                cell[3] += *e;
            }
        }
        let stats = Self::stats_of(&moved, cells.side);
        // The frame the top of the exposed levels comes from: how far that
        // top was below the sensor's clipping when it was recorded.
        let top = (0..stops.len().min(cells.tops.len()))
            .max_by(|a, b| (cells.tops[*a] + stops[*a]).partial_cmp(&(cells.tops[*b] + stops[*b])).unwrap_or(std::cmp::Ordering::Equal));
        let headroom = top.map_or(0.0, |k| {
            let shot_key = ((stats.keys.dark / 2f32.powf(stops[k])) as f64).max(2f64.powi(-30)).log2();
            -(crate::raw_tone::default_black(2f64.powf(cells.tops[k] as f64), shot_key).max(1e-9).log2()) as f32
        });
        let sensor_gain = stops.iter().fold(f32::MIN, |a, e| a.max(*e));
        (stats, Exposed { headroom, sensor_gain: 2f64.powf(if stops.is_empty() { 0.0 } else { sensor_gain as f64 }) })
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
        let (cells, side) = self.cells(|codes| {
            let lit = codes.map(|c| light(c).max(0.0));
            (lit, (lit[0] * CODE_LUMA[0] + lit[1] * CODE_LUMA[1] + lit[2] * CODE_LUMA[2]).max(BASE_FLOOR).log2())
        });
        let stats = Self::stats_of(&cells, side);
        if let Ok(mut kept) = self.tone.lock() {
            *kept = Some((key, stats.clone()));
        }
        stats
    }
}

// ── Automatic exposure ─────────────────────────────────────────────────
//
// A raw clip shot at one exposure shows what the light did: the picture
// darkens in the shade and brightens in the sun. An automatic exposure
// gives every frame the exposure its own picture asks for, as a camera
// does while it records, with two differences: it knows the whole clip, so
// it can follow the light smoothly from both sides instead of chasing it,
// and it meters the picture as the clip's own development renders it.
//
// The clip is metered at short intervals ([`Meter`]). A metered frame asks
// for the exposure at which its developed picture has a mean brightness of
// [`AUTO_TARGET`]: the mean, weighted toward where the lens looks, of what
// each part of the picture is displayed as. The development that decides
// what a part is displayed as is the clip's, with every other control as
// the user set it: pull the highlights down and the frames are exposed up
// to keep the picture where it was.
//
// That development is itself derived from the clip's statistics, and those
// are the statistics of the clip AS EXPOSED: a frame the exposure brightens
// counts as that much brighter (as it would if the camera had exposed it
// so). The two settle together in a few rounds. What comes out is one
// development for the whole clip, a fixed rendering of exposed values, and
// one exposure per frame that changes smoothly: nothing else follows the
// frame.

/// What a metered frame's developed picture is brought to: the mean of its
/// displayed (encoded) values, 0..1. (The middle of what ten daylight
/// takes read at the exposures their owner had set by eye, each with its
/// own Highlights and Shadows: they read 0.21 to 0.54, half of them 0.32
/// to 0.45.)
pub const AUTO_TARGET: f32 = 0.38;

/// The exposures an automatic exposure stays within, stops.
pub const AUTO_STOPS: (f32, f32) = (-4.0, 7.0);

/// How far from the clip's middle exposure a frame's may go, stops: a
/// frame the lens saw nothing in (covered, or set down) asks for
/// everything there is, and would be lifted into its noise.
pub const AUTO_REACH: f32 = 3.0;

/// The exposures and the development that meters them have settled when a
/// round moves no exposure by more than this, stops; and the most rounds
/// they are given.
const AUTO_SETTLED: f32 = 0.002;
const AUTO_ROUNDS: usize = 60;

/// A metered frame's levels: bins of log₂ luminance.
const LEVEL_LO: f32 = -20.0;
const LEVEL_PER_STOP: f32 = 8.0;
const LEVEL_BINS: usize = 208;

/// What a grey of an exposed value is displayed as, over log₂ of the value.
const SHOWN_LO: f32 = -34.0;
const SHOWN_SAMPLES: usize = 497;

/// A raw clip's automatic exposure under one set of development settings:
/// the exposure of every moment of the clip, and the one development all
/// its frames get.
#[derive(Debug)]
pub struct AutoExposure {
    /// The clip times the exposure is known at, seconds, rising.
    times: Vec<f64>,
    /// The whole exposure at each, in stops, as the picture asks for it
    /// and smoothed: without the user's own stops on top.
    stops: Vec<f32>,
    /// The user's own stops on top (`RawSettings::auto_ev`).
    ev: f32,
    /// The development of every frame; its gain is the frame's own.
    constants: DevelopConstants,
}

/// The value at `t` of samples `v` taken at the rising times `times`: the
/// straight line between the two around it, the first or the last outside.
fn at_time(times: &[f64], v: &[f32], t: f64) -> f32 {
    let Some(last) = times.len().checked_sub(1) else { return 0.0 };
    if t <= times[0] {
        return v[0];
    }
    if t >= times[last] {
        return v[last];
    }
    let hi = times.partition_point(|x| *x <= t).min(last);
    let (a, b) = (times[hi - 1], times[hi]);
    let f = if b > a { ((t - a) / (b - a)) as f32 } else { 0.0 };
    v[hi - 1] * (1.0 - f) + v[hi] * f
}

/// Samples `v` at the times `times`, each replaced by the mean of those
/// around it under a bell `seconds` wide (its standard deviation). 0, or
/// less: as they are.
fn smoothed(times: &[f64], v: &[f32], seconds: f32) -> Vec<f32> {
    if !(seconds > 0.0) || v.len() < 2 {
        return v.to_vec();
    }
    let scale = -0.5 / (seconds as f64 * seconds as f64);
    (0..v.len()).map(|i| {
        let (mut sum, mut weight) = (0f64, 0f64);
        for j in 0..v.len() {
            let d = times[j] - times[i];
            let w = (d * d * scale).exp();
            sum += w * v[j] as f64;
            weight += w;
        }
        (sum / weight) as f32
    }).collect()
}

/// How much each pixel of a metered eye counts: a half-equirect's pixel by
/// the part of the sphere it covers, and by how near the lens's axis it
/// looks (the middle of the picture counts most, the rim not at all).
fn meter_weights() -> Vec<f32> {
    let m = METER_SIDE as f32;
    (0..METER_SIDE * METER_SIDE).map(|i| {
        let (x, y) = ((i % METER_SIDE) as f32, (i / METER_SIDE) as f32);
        let lon = ((x + 0.5) / m - 0.5) * std::f32::consts::PI;
        let lat = ((y + 0.5) / m - 0.5) * std::f32::consts::PI;
        lat.cos() * lat.cos() * lon.cos()
    }).collect()
}

/// What a grey is displayed as (encoded, 0..1) by the development `k`,
/// over log₂ of its exposed value: [`SHOWN_SAMPLES`] samples from
/// [`SHOWN_LO`], [`LEVEL_PER_STOP`] to the stop. A grey's base is its own
/// level.
fn grey_response(k: &DevelopConstants) -> Vec<f32> {
    let unit = DevelopConstants { gain: 1.0, ..k.clone() };
    let n = unit.geometry.neutral;
    (0..SHOWN_SAMPLES).map(|i| {
        let v = 2f32.powf(SHOWN_LO + i as f32 / LEVEL_PER_STOP);
        unit.develop_local(n.map(|c| c * v), v.max(BASE_FLOOR).log2())[1]
    }).collect()
}

/// The mean of what a metered frame's picture is displayed as at an
/// exposure of `stops`: `levels` its weighted levels (bins from
/// [`LEVEL_LO`], summing to 1), `shown` a [`grey_response`].
fn mean_shown(levels: &[f32], shown: &[f32], stops: f32) -> f32 {
    let last = (shown.len() - 1) as f32;
    levels.iter().enumerate().filter(|(_, w)| **w > 0.0).map(|(b, w)| {
        let level = LEVEL_LO + (b as f32 + 0.5) / LEVEL_PER_STOP + stops;
        let at = ((level - SHOWN_LO) * LEVEL_PER_STOP).clamp(0.0, last);
        let i = (at.floor() as usize).min(shown.len() - 2);
        let f = at - i as f32;
        w * (shown[i] * (1.0 - f) + shown[i + 1] * f)
    }).sum()
}

/// The exposure at which a metered frame's picture is displayed at
/// `target` on the mean, within [`AUTO_STOPS`].
fn exposure_for(levels: &[f32], shown: &[f32], target: f32) -> f32 {
    let (mut lo, mut hi) = AUTO_STOPS;
    if mean_shown(levels, shown, lo) >= target {
        return lo;
    }
    if mean_shown(levels, shown, hi) <= target {
        return hi;
    }
    for _ in 0..28 {
        let mid = 0.5 * (lo + hi);
        if mean_shown(levels, shown, mid) < target { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

impl AutoExposure {
    /// The whole exposure of the frame at clip time `t`, stops: what its
    /// picture asks for and the user's own stops.
    pub fn stops_at(&self, t: f64) -> f32 {
        at_time(&self.times, &self.stops, t) + self.ev
    }

    /// The development of the frame at clip time `t`: the clip's, at the
    /// frame's exposure.
    pub fn constants_at(&self, t: f64) -> DevelopConstants {
        DevelopConstants { gain: 2f32.powf(self.stops_at(t)), ..self.constants.clone() }
    }

    /// The least and the most of the exposures over the clip, stops.
    pub fn range(&self) -> (f32, f32) {
        let lo = self.stops.iter().cloned().fold(f32::MAX, f32::min);
        let hi = self.stops.iter().cloned().fold(f32::MIN, f32::max);
        (lo + self.ev, hi + self.ev)
    }
}

impl ClipStats {
    /// What the meter reads of the clip as `stage` develops it with the
    /// exposure that is SET (no automatic exposure): for every metered
    /// frame its clip time and the mean its picture is displayed at, 0..1
    /// (what an automatic exposure brings to [`AUTO_TARGET`]). `None` for a
    /// clip that was not metered.
    pub fn metered_brightness(&self, stage: &RawDevelop) -> Option<Vec<(f64, f32)>> {
        let meter = self.meter.as_ref()?;
        let geometry = stage.geometry();
        let k = stage.constants_for(&self.raw(&geometry), None);
        let shown = grey_response(&k);
        let stops = k.gain.log2();
        Some(meter.times.iter().zip(self.meter_levels(meter, &geometry)).map(|(t, levels)| (*t, mean_shown(&levels, &shown, stops))).collect())
    }

    /// Every metered frame's levels under the white balance of `geometry`:
    /// its pixels' luminance, weighted ([`meter_weights`]), in bins from
    /// [`LEVEL_LO`] that sum to 1.
    fn meter_levels(&self, meter: &Meter, geometry: &DevelopGeometry) -> Vec<Vec<f32>> {
        let luma = crate::raw_color::prophoto_luma().map(|v| v as f32);
        let weights = meter_weights();
        let per_eye = METER_SIDE * METER_SIDE * 3;
        meter.copies.chunks_exact(per_eye * meter.eyes).map(|frame| {
            let mut bins = vec![0f32; LEVEL_BINS];
            let mut total = 0f32;
            for eye in frame.chunks_exact(per_eye) {
                for (px, w) in eye.chunks_exact(3).zip(&weights) {
                    let rgb = geometry.scene([px[0], px[1], px[2]]);
                    let y = (rgb[0].max(0.0) * luma[0] + rgb[1].max(0.0) * luma[1] + rgb[2].max(0.0) * luma[2]).max(BASE_FLOOR);
                    let b = (((y.log2() - LEVEL_LO) * LEVEL_PER_STOP).floor().max(0.0) as usize).min(LEVEL_BINS - 1);
                    bins[b] += w;
                    total += w;
                }
            }
            bins.iter_mut().for_each(|v| *v /= total.max(1e-12));
            bins
        }).collect()
    }

    /// The clip's automatic exposure under `stage`'s settings (its white
    /// balance and every tone control but Exposure; `auto_ev`,
    /// `auto_smooth`). `None` for a clip that was not metered.
    pub fn auto_exposure(&self, stage: &RawDevelop) -> Option<Arc<AutoExposure>> {
        let meter = self.meter.as_ref()?;
        if self.values != Values::Camera || self.frame_times.len() != self.frames() || meter.times.is_empty() {
            return None;
        }
        let geometry = stage.geometry();
        let s = &stage.settings;
        let mut key = [0u32; 19];
        key[..12].copy_from_slice(&geometry_key(&geometry));
        for (k, v) in key[12..].iter_mut().zip([s.contrast, s.highlights, s.shadows, s.whites, s.blacks, s.auto_ev, s.auto_smooth]) {
            *k = v.to_bits();
        }
        // What was found last, under other settings: where the search
        // starts from (a slider moved a little moves the exposures a little).
        let mut before: Option<Arc<AutoExposure>> = None;
        if let Ok(kept) = self.auto.lock() {
            if let Some((k, auto)) = &*kept {
                if *k == key {
                    return Some(auto.clone());
                }
                before = Some(auto.clone());
            }
        }
        let started = std::time::Instant::now();
        let (cells, _) = self.raw_cells(&geometry);
        let levels = self.meter_levels(meter, &geometry);
        // The exposures and the development that meters them, settling.
        let unit = stage.with_whole_exposure(0.0);
        let mut stops = match before.filter(|b| b.stops.len() == meter.times.len()) {
            Some(b) => b.stops.clone(),
            None => vec![0f32; meter.times.len()],
        };
        let mut moved = f32::MAX;
        let mut round = 0;
        let constants = loop {
            let at_frames: Vec<f32> = self.frame_times.iter().map(|t| at_time(&meter.times, &stops, *t)).collect();
            let (stats, exposed) = Self::raw_exposed(&cells, self.eyes, &at_frames);
            if moved < AUTO_SETTLED || round == AUTO_ROUNDS {
                // The user's own stops are an exposure of the clip as the
                // automatic one has left it.
                break stage.with_whole_exposure(s.auto_ev).constants_exposed(&stats, None, Some(exposed));
            }
            let shown = grey_response(&unit.constants_exposed(&stats, None, Some(exposed)));
            let mut asked: Vec<f32> = levels.iter().map(|l| exposure_for(l, &shown, AUTO_TARGET)).collect();
            let mut sorted = asked.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let middle = sorted[sorted.len() / 2];
            asked.iter_mut().for_each(|e| *e = e.clamp(middle - AUTO_REACH, middle + AUTO_REACH));
            let next = smoothed(&meter.times, &asked, s.auto_smooth);
            moved = next.iter().zip(&stops).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            stops = next;
            round += 1;
        };
        let auto = Arc::new(AutoExposure { times: meter.times.clone(), stops, ev: s.auto_ev, constants });
        let (lo, hi) = auto.range();
        tracing::debug!(
            "automatic exposure: {} metered frames, {lo:+.2} to {hi:+.2} stops ({round} rounds, the last moved it by {moved:.4}), in {:.1?}",
            meter.times.len(), started.elapsed());
        if let Ok(mut kept) = self.auto.lock() {
            *kept = Some((key, auto.clone()));
        }
        Some(auto)
    }
}

/// A drawn eye (`side` square, `channels` values of `bytes` bytes a pixel,
/// the first three its RGB) as a copy `copy` square: the mean of the pixels
/// each of the copy's stands for, 0..1.
fn reduce_to(drawn: &[u8], side: usize, copy: usize, channels: usize, bytes: usize) -> Vec<f32> {
    let px = channels * bytes;
    assert_eq!(drawn.len(), side * side * px, "a drawn eye");
    assert_eq!(side % copy, 0, "whole pixels to a pixel of the copy");
    let reduce = side / copy;
    let scale = if bytes == 2 { 1.0 / 65535.0 } else { 1.0 / 255.0 } / (reduce * reduce) as f64;
    let mut out = Vec::with_capacity(copy * copy * 3);
    for cy in 0..copy {
        for cx in 0..copy {
            let mut sum = [0u64; 3];
            for y in cy * reduce..(cy + 1) * reduce {
                let row = &drawn[(y * side + cx * reduce) * px..(y * side + (cx + 1) * reduce) * px];
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

/// A drawn eye of [`RAW_REFERENCE_SIDE`] as its [`SIDE`] copy.
fn reduce(drawn: &[u8], channels: usize, bytes: usize) -> Vec<f32> {
    reduce_to(drawn, RAW_REFERENCE_SIDE as usize, SIDE, channels, bytes)
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

/// How far a sampling has come, for whoever shows it: frames read, and
/// frames to read in all.
#[derive(Debug, Default)]
pub struct Progress {
    pub done: std::sync::atomic::AtomicUsize,
    pub total: std::sync::atomic::AtomicUsize,
}

impl Progress {
    /// The part done, 0..1.
    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { (self.done.load(Ordering::Relaxed) as f32 / total as f32).min(1.0) }
    }
}

/// Sample the clip of `cfg` (an export's configuration of it: source,
/// lens, eye order; its view, its stabilization and its trim are not used)
/// and keep the copies its statistics are read from. The whole clip is
/// sampled, whatever part of it is trimmed for an export: its look does not
/// move with the trim. `meter`: also meter a raw clip at short intervals,
/// for an automatic exposure ([`Meter`]).
pub fn measure(pipeline: &Device, cfg: &FisheyeExportConfig, cancel: &AtomicBool, meter: bool, progress: Option<&Progress>) -> Result<ClipStats> {
    let started = std::time::Instant::now();
    let files = files(cfg);
    let total: f64 = files.iter().map(|p| crate::decode::probe_video(p).map(|pr| pr.duration_sec).unwrap_or(0.0)).sum();
    let times: Vec<f64> = (0..SAMPLES).map(|k| total * (k as f64 + 0.5) / SAMPLES as f64).collect();
    // A raw clip metered: a frame every `METER_INTERVAL_S`, at most
    // `METER_FRAMES` of them.
    let meter = meter && cfg.source_kind.is_raw_video();
    let metered: Vec<f64> = if meter {
        let n = ((total / METER_INTERVAL_S).ceil() as usize).clamp(SAMPLES, METER_FRAMES);
        (0..n).map(|k| total * (k as f64 + 0.5) / n as f64).collect()
    } else {
        Vec::new()
    };
    if let Some(p) = progress {
        p.total.store(times.len() + metered.len(), Ordering::Relaxed);
        p.done.store(0, Ordering::Relaxed);
    }
    let step = || {
        if let Some(p) = progress {
            p.done.fetch_add(1, Ordering::Relaxed);
        }
    };
    let (mut frame_times, mut meter_times, mut copies): (Vec<f64>, Vec<f64>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
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
            frame_times.push(t);
            frames += 1;
            step();
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
            frame_times.push(if pair.pts_s > 0.0 { pair.pts_s } else { t });
            frames += 1;
            step();
        }
        // The metered frames, read straight off the take's capture.
        if let (true, Some((calib_l, _))) = (frames > 0 && !metered.is_empty(), calib) {
            drop(decoder);
            // The lens at the size of the picture the places are found with.
            let small = crate::fisheye_export::resolve_source_calib_pair(cfg, PLACES_SIDE as u32, PLACES_SIDE as u32, carrier.as_ref()).0;
            let _ = calib_l;
            (meter_times, copies) = meter_take(pipeline, cfg, &metered, small, &cancelled, &step)?;
        }
    }
    if frames == 0 {
        return Err(Error::Ffmpeg(format!("clip statistics: no frame of {} could be read", cfg.source_path.display())));
    }
    // Metered frames in the order of their times, each once.
    let per_frame = METER_SIDE * METER_SIDE * 3;
    let mut order: Vec<usize> = (0..meter_times.len()).collect();
    order.sort_by(|a, b| meter_times[*a].partial_cmp(&meter_times[*b]).unwrap_or(std::cmp::Ordering::Equal));
    order.dedup_by(|a, b| meter_times[*a] == meter_times[*b]);
    let (meter_times, copies): (Vec<f64>, Vec<f32>) = (
        order.iter().map(|i| meter_times[*i]).collect(),
        order.iter().flat_map(|i| copies[i * per_frame..(i + 1) * per_frame].iter().copied()).collect(),
    );
    tracing::info!(
        "clip statistics: {frames} frames of {} ({} eye(s) each) sampled{} in {:.2?}",
        cfg.source_path.display(), eyes,
        if meter { format!(", {} metered for the automatic exposure", meter_times.len()) } else { String::new() },
        started.elapsed());
    Ok(ClipStats::new(values, eyes, pictures).with_meter(frame_times, meter_times, 1, copies))
}

/// The side of the picture a metered eye's places are found with
/// ([`meter_take`]).
const PLACES_SIDE: usize = 512;

/// Meter a raw take at `times`: ONE eye of each frame (the left picture's
/// channel; the two see the same scene), read and unpacked straight from
/// the capture, and reduced to a [`METER_SIDE`] copy of the half-equirect
/// its lens sees. No frame is demosaiced, uploaded or drawn: where each
/// pixel of the drawn eye lies on the sensor is found once, by drawing a
/// picture whose pixels say their own place through the lens `lens` (the
/// source lens at [`PLACES_SIDE`]), and every frame's samples are then
/// picked at those places. Returns the frames' clip times and their copies.
fn meter_take(
    pipeline: &Device, cfg: &FisheyeExportConfig, times: &[f64], lens: crate::gpu::FisheyeCalib,
    cancelled: &dyn Fn() -> Result<()>, step: &dyn Fn(),
) -> Result<(Vec<f64>, Vec<f32>)> {
    use vr180_fisheye::x6b;
    let index = crate::x6_raw::open(&cfg.source_path)?;
    let take = &index.take;
    let (w, h) = (take.width as usize, take.height as usize);
    // The channel the left picture is made of, and where it lies among a
    // record's samples.
    let (channel, at) = match take.layout {
        x6b::Layout::Paired => {
            let c = if cfg.source_kind.dual_stream_iter_swap(cfg.fisheye_swap_eyes) { 1 } else { 0 };
            (c, c * w * h)
        }
        x6b::Layout::Single { channel } => (channel, 0),
    };
    // The places: a picture of PLACES_SIDE whose red and green are a
    // pixel's own x and y and whose blue is 1, drawn as the eye is.
    let n = PLACES_SIDE;
    let mut picture = Vec::with_capacity(n * n * 8);
    for y in 0..n {
        for x in 0..n {
            for v in [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, 1.0, 1.0] {
                picture.extend_from_slice(&((v * 65535.0).round() as u16).to_le_bytes());
            }
        }
    }
    let drawn_side = METER_DRAWN as u32;
    let drawn = pipeline.project_eye(&EyeProjection::new(
        EyePicture::Bytes16 { bytes: &picture, bit_depth: 16, w: n as u32, h: n as u32 },
        drawn_side, drawn_side, EquirectRotation::IDENTITY, lens).wide())?;
    let read = pipeline.read_texture_rgba64(&drawn, drawn_side, drawn_side)?;
    // Per pixel of the drawn eye: the sensor sample it starts at (the
    // even column and row of its 2 × 2 group), and how much of the picture
    // shows there (a pixel the lens does not reach shows none).
    let places: Vec<(usize, f32)> = read.chunks_exact(8).map(|px| {
        let v = |k: usize| u16::from_le_bytes([px[2 * k], px[2 * k + 1]]) as f32 / 65535.0;
        let shows = v(2);
        if shows < 1e-3 {
            return (0, 0.0);
        }
        let x = ((v(0) / shows * w as f32) as usize).min(w - 1) & !1;
        let y = ((v(1) / shows * h as f32) as usize).min(h - 1) & !1;
        (y.min(h - 2) * w + x.min(w - 2), shows)
    }).collect();
    // The colour of each sample of a 2 × 2 group, and the take's levels.
    let cfa = take.cfa;
    let (black, scale) = (take.black_level as f32, 1.0 / (take.white_level as f32 - take.black_level as f32).max(1.0));
    let greens = cfa.iter().filter(|c| **c == 1).count().max(1) as f32;
    let reduce = METER_DRAWN / METER_SIDE;
    let mut file = std::fs::File::open(take.capture_path())?;
    let (mut record, mut samples) = (Vec::new(), vec![0u16; take.layout.samples()]);
    let (mut found, mut copies) = (Vec::with_capacity(times.len()), Vec::with_capacity(times.len() * METER_SIDE * METER_SIDE * 3));
    for &t in times {
        cancelled()?;
        step();
        let i = index.frame_at(t);
        let Some(entry) = index.pairs.get(i) else { continue };
        if x6b::read_record(&mut file, entry, &mut record).is_err() { continue; }
        if x6b::decode_record(&record, take.format(), x6b::Channels::One(channel), &mut samples, 0).is_err() { continue; }
        let eye = &samples[at..at + w * h];
        let mut copy = vec![0f32; METER_SIDE * METER_SIDE * 3];
        for (p, (start, shows)) in places.iter().enumerate() {
            if *shows <= 0.0 {
                continue;
            }
            let (dx, dy) = (p % METER_DRAWN, p / METER_DRAWN);
            let cell = ((dy / reduce) * METER_SIDE + dx / reduce) * 3;
            let group = [eye[*start], eye[*start + 1], eye[*start + w], eye[*start + w + 1]];
            let mut rgb = [0f32; 3];
            for (v, colour) in group.iter().zip(cfa) {
                rgb[colour.min(2) as usize] += (*v as f32 - black) * scale;
            }
            rgb[1] /= greens;
            for k in 0..3 {
                copy[cell + k] += rgb[k] * shows;
            }
        }
        let per_cell = (reduce * reduce) as f32;
        copies.extend(copy.iter().map(|v| v / per_cell));
        found.push(index.pts_s(i));
    }
    Ok((found, copies))
}

/// An export of `cfg` meters its clip: a raw clip developed with an
/// automatic exposure.
pub fn wants_meter(cfg: &FisheyeExportConfig) -> bool {
    cfg.source_kind.is_raw_video() && cfg.color_stack.raw_develop.as_ref().is_some_and(|s| s.settings.auto_exposure)
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

/// The clip's statistics where it has been sampled already (metered or
/// not: [`ClipStats::has_meter`]).
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
    for_clip_shown(pipeline, cfg, cancel, None)
}

/// [`for_clip`], saying how far the sampling has come. A clip that
/// [`wants_meter`] and was sampled without one is sampled again, metered:
/// its statistics come out the same, the copies they are read from are
/// read at the same moments.
pub fn for_clip_shown(pipeline: &Device, cfg: &FisheyeExportConfig, cancel: &AtomicBool, progress: Option<&Progress>) -> Result<Arc<ClipStats>> {
    let meter = wants_meter(cfg);
    if let Some(stats) = cached(cfg).filter(|s| !meter || s.has_meter()) {
        return Ok(stats);
    }
    let key = key_of(cfg);
    let stats = Arc::new(measure(pipeline, cfg, cancel, meter, progress)?);
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

    /// A raw clip of `brightness.len()` moments one second apart, each a
    /// picture of the camera's white whose levels run over six stops below
    /// `brightness` (the same picture, exposed differently): sampled and
    /// metered at every one of them.
    fn metered(brightness: &[f32]) -> ClipStats {
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        let level = |x: usize, side: usize| 2f32.powf(-6.0 * (1.0 - (x as f32 + 0.5) / side as f32));
        let (mut pictures, mut copies) = (Vec::new(), Vec::new());
        for b in brightness {
            for i in 0..SIDE * SIDE {
                pictures.extend(n.map(|c| c * b * level(i % SIDE, SIDE)));
            }
            for i in 0..METER_SIDE * METER_SIDE {
                copies.extend(n.map(|c| c * b * level(i % METER_SIDE, METER_SIDE)));
            }
        }
        let times: Vec<f64> = (0..brightness.len()).map(|k| k as f64).collect();
        ClipStats::new(Values::Camera, 1, pictures).with_meter(times.clone(), times, 1, copies)
    }

    fn auto_stage(settings: RawSettings) -> RawDevelop {
        RawDevelop { settings: RawSettings { auto_exposure: true, ..settings }, ..stage() }
    }

    #[test]
    fn samples_are_read_between_and_smoothed() {
        let times = [0.0, 1.0, 2.0, 4.0];
        let v = [0.0f32, 1.0, 0.0, 2.0];
        assert_eq!((at_time(&times, &v, -1.0), at_time(&times, &v, 0.5), at_time(&times, &v, 3.0), at_time(&times, &v, 9.0)), (0.0, 0.5, 1.0, 2.0));
        assert_eq!(at_time(&[], &[], 1.0), 0.0);
        // No width: as they are. A wide one: all the same, their mean.
        assert_eq!(smoothed(&times, &v, 0.0), v);
        let wide = smoothed(&times, &v, 1000.0);
        assert!(wide.iter().all(|x| (x - 0.75).abs() < 1e-4), "{wide:?}");
        // A step keeps its ends and its sum, and rises all the way.
        let step: Vec<f32> = (0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect();
        let t: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let soft = smoothed(&t, &step, 2.0);
        assert!(soft[0] < 0.01 && soft[39] > 0.99 && (soft[19] + soft[20] - 1.0).abs() < 1e-4, "{soft:?}");
        assert!(soft.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn a_clip_that_was_not_metered_has_no_automatic_exposure() {
        let clip = flat(Values::Camera, 2, &[0.02, 0.04, 0.08, 0.16]);
        assert!(!clip.has_meter() && clip.auto_exposure(&auto_stage(RawSettings::default())).is_none());
        assert!(clip.metered_brightness(&stage()).is_none());
    }

    /// Every frame is brought to the same brightness: a frame of twice the
    /// light gets a stop less, and the picture each then shows reads the
    /// target on the meter.
    #[test]
    fn frames_are_exposed_to_one_brightness() {
        let light = [0.05f32, 0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2];
        let clip = metered(&light);
        let follow = RawSettings { auto_smooth: 0.0, highlights: -60.0, shadows: 30.0, ..Default::default() };
        let auto = clip.auto_exposure(&auto_stage(follow)).expect("a metered clip");
        let at = |k: usize| auto.stops_at(k as f64);
        assert!((at(0) - at(3) - 1.0).abs() < 0.01 && (at(3) - at(4) - 1.0).abs() < 0.01, "{} {} {}", at(0), at(3), at(4));
        assert!((at(0) - at(1)).abs() < 1e-4 && (at(5) - at(7)).abs() < 1e-4);
        // Between two metered moments: between their exposures.
        assert!((auto.stops_at(3.5) - 0.5 * (at(3) + at(4))).abs() < 1e-4);
        // One development for all of them, at each one's own exposure.
        let (a, b) = (auto.constants_at(0.0), auto.constants_at(6.0));
        assert!((a.gain.log2() - at(0)).abs() < 1e-4 && (b.gain.log2() - at(6)).abs() < 1e-4);
        assert_eq!(DevelopConstants { gain: 1.0, ..a.clone() }, DevelopConstants { gain: 1.0, ..b });
        // What each frame's picture then reads on the meter: the target.
        let g = auto_stage(follow).geometry();
        let shown = grey_response(&a);
        let meter = clip.meter.as_ref().unwrap();
        for (k, levels) in clip.meter_levels(meter, &g).iter().enumerate() {
            let read = mean_shown(levels, &shown, at(k));
            assert!((read - AUTO_TARGET).abs() < 0.004, "frame {k}: {read}");
        }
        // The user's stops go on top, and the picture is that much brighter.
        let brighter = clip.auto_exposure(&auto_stage(RawSettings { auto_ev: 0.75, ..follow })).unwrap();
        assert!((brighter.stops_at(2.0) - at(2) - 0.75).abs() < 1e-3 && (brighter.stops_at(6.0) - at(6) - 0.75).abs() < 1e-3);
        let (lo, hi) = brighter.range();
        assert!((lo - at(7) - 0.75).abs() < 1e-3 && (hi - at(0) - 0.75).abs() < 1e-3, "{lo} {hi}");
    }

    /// The exposure follows the light as slowly as it is told to: at once,
    /// over a few seconds, or not at all (one exposure for the clip).
    #[test]
    fn the_exposure_follows_as_slowly_as_asked() {
        let light: Vec<f32> = (0..24).map(|k| if k < 12 { 0.05 } else { 0.2 }).collect();
        let clip = metered(&light);
        let with = |smooth: f32| clip.auto_exposure(&auto_stage(RawSettings { auto_smooth: smooth, ..Default::default() })).unwrap();
        let (sharp, soft, held) = (with(0.0), with(2.0), with(10_000.0));
        // At once: two stops between the two halves, all of it between two frames.
        assert!((sharp.stops_at(11.0) - sharp.stops_at(12.0) - 2.0).abs() < 0.02);
        // Softly: the same two stops from end to end, never more than a
        // part of them from one second to the next, never rising.
        let s: Vec<f32> = (0..24).map(|k| soft.stops_at(k as f64)).collect();
        assert!((s[0] - s[23] - 2.0).abs() < 0.05, "{s:?}");
        assert!(s.windows(2).all(|w| w[0] >= w[1] - 1e-4 && w[0] - w[1] < 0.5), "{s:?}");
        // Held: one exposure, between the two.
        let (lo, hi) = held.range();
        assert!(hi - lo < 0.01 && lo < sharp.stops_at(0.0) && hi > sharp.stops_at(23.0), "{lo} {hi}");
    }

    /// A clip that does not change, exposed automatically, is developed as
    /// the same exposure set by hand develops it: the automatic exposure
    /// is an exposure, nothing else.
    #[test]
    fn a_steady_clip_is_developed_as_by_hand() {
        let clip = metered(&[0.1; 6]);
        for settings in [RawSettings::default(), RawSettings { highlights: -100.0, shadows: 40.0, contrast: 20.0, blacks: 10.0, ..Default::default() }] {
            let auto = clip.auto_exposure(&auto_stage(settings)).unwrap();
            let stops = auto.stops_at(2.0);
            let hand = stage().with_whole_exposure(stops);
            let hand = RawDevelop { settings: RawSettings { exposure: hand.settings.exposure, ..settings }, ..stage() };
            let (a, h) = (auto.constants_at(2.0), hand.constants_for(&clip.raw(&hand.geometry()), None));
            assert!((a.gain - h.gain).abs() < 1e-4 * h.gain && (a.white_point - h.white_point).abs() < 1e-3 * h.white_point, "{} {}", a.white_point, h.white_point);
            assert!((a.dark_key - h.dark_key).abs() < 1e-3);
            let apart = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            assert!(apart(&a.curve, &h.curve) < 2e-3, "curve {}", apart(&a.curve, &h.curve));
            match (&a.local, &h.local) {
                (Some(x), Some(y)) => assert!(apart(&x.gain, &y.gain) < 5e-3 && (x.white - y.white).abs() < 2e-3 * y.white, "gain {}", apart(&x.gain, &y.gain)),
                (None, None) => {}
                _ => panic!("one has a local stage, the other none"),
            }
        }
    }

    /// A frame the lens saw nothing in asks for every stop there is: it
    /// gets no more than a few beyond the clip's middle.
    #[test]
    fn a_dark_frame_is_not_lifted_without_end() {
        let mut light = vec![0.1f32; 12];
        light[5] = 1e-5;
        let auto = metered(&light).auto_exposure(&auto_stage(RawSettings { auto_smooth: 0.0, ..Default::default() })).unwrap();
        let (usual, dark) = (auto.stops_at(0.0), auto.stops_at(5.0));
        assert!((dark - usual - AUTO_REACH).abs() < 0.01, "{usual} {dark}");
    }

    /// The meter's reading of a clip at the exposure that is set rises with
    /// that exposure and passes the target where the automatic one is.
    #[test]
    fn the_meter_reads_the_set_exposure() {
        let clip = metered(&[0.1; 4]);
        let auto = clip.auto_exposure(&auto_stage(RawSettings::default())).unwrap().stops_at(0.0);
        let read = |stops: f32| clip.metered_brightness(&stage().with_whole_exposure(stops)).unwrap()[0].1;
        assert!(read(auto - 1.0) < read(auto) && read(auto) < read(auto + 1.0));
        assert!((read(auto) - AUTO_TARGET).abs() < 0.004, "{}", read(auto));
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
        let own = (dev.raw_develop_constants(&[&a], None, &develop, 1.0, None, None), dev.raw_develop_constants(&[&b], None, &develop, 1.0, None, None));
        assert_ne!(own.0, own.1, "each frame by itself: its own constants");
        let held = (dev.raw_develop_constants(&[&a], None, &develop, 1.0, Some(&clip), None), dev.raw_develop_constants(&[&b], None, &develop, 1.0, Some(&clip), None));
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
