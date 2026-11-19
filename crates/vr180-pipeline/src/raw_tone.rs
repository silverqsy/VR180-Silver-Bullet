//! Tone of raw development: the user's settings and what they turn into
//! for one frame — the constants the GPU stage (`shaders/raw_develop.wgsl`)
//! and its CPU twin ([`DevelopConstants::develop`]) run on.
//!
//! The rendering follows Adobe Camera Raw's (process version 2012 and
//! later), so a frame developed here and the same frame's DNG opened there
//! look alike. Its default — every control at zero — was measured on Camera
//! Raw 18.7 with synthetic charts and is, on the frame's scene-linear RGB
//! (linear ProPhoto, a neutral grey = equal channels, the level at which
//! the first sensor channel clips = 1):
//!
//! 1. **Highlights are kept past the first clipped channel.** A clipped
//!    channel is rebuilt from the ones that still read, so a neutral goes
//!    on rising to the level at which its LAST channel clips: the sensor's
//!    white `W`.
//! 2. **Default black.** Shadows are pulled down by an amount that follows
//!    the frame's darkest level (a table measured over that level): from a
//!    quarter of a per cent at the very bottom to the plain DNG black point
//!    of 0.005 once the darkest level is above 2⁻⁷.
//! 3. **White point.** A shoulder above 0.25 maps `W*` to 1 (a table
//!    measured over `W*`), where `W*` is the sensor's white, but at most
//!    twice the frame's own maximum and at least 1. At `W* = 1` it does
//!    nothing.
//! 4. **The default tone curve** of the DNG reference rendering (measured
//!    node by node; it is the DNG SDK's default curve).
//!
//! All four are curves of one variable, and they are applied the way the
//! DNG reference does it — the largest and the smallest channel through the
//! curve, the middle one keeping its place between them — so they compose
//! into ONE curve per frame, sampled over log₂ of the scene value. The
//! frame's maximum and darkest level ([`ToneKeys`]) are measured on a
//! reduced copy of the frame as it is graded (`raw_measure.wgsl`).
//!
//! **A clip has one set of statistics.** Measured on every frame by itself,
//! all of the above follows the frame: a pan that lets more sky in changes
//! how everything else is rendered, and a clip shot at one exposure
//! brightens and darkens. So a clip's frames are all developed by the
//! statistics of the clip, read once off frames spread over it
//! ([`crate::clip_stats`]); "the frame" below is then the clip taken as one
//! picture. A still is its own one picture and is measured by itself.
//!
//! Then: ProPhoto → Rec.709 primaries, clipped to 0..1, sRGB encoding.
//!
//! **Shadows and Highlights** are a gain between the default black and the
//! white point (measured with exposure series: the gain is on the signal
//! the default black has been taken off, ahead of the white point's
//! shoulder), the same for the three channels, and it is local:
//! `gain = curve(base)`, where `base` is the pixel's level smoothed across
//! what surrounds it without crossing strong edges (a bilateral grid of
//! log₂ luminance: cells [`BASE_CELL`] of the frame's long side by
//! [`BASE_RANGE`] stops), so texture keeps its contrast and an object is
//! lifted as a whole. The curves belong to the frame: Camera Raw derives
//! them from a reduced copy of the picture ([`FrameStats::levels`]), and
//! [`local_curves`] reproduces what it derives ([`LocalCurves`] has the
//! rules by which the two sliders combine). Two more things follow the
//! gain:
//!
//! - the **white point** is that of the frame with the gain in (twice its
//!   brightest value, within the same bounds), so pulling the highlights
//!   down relaxes the shoulder and raising them tightens it;
//! - a gain above zero is **limited toward the white point**: with `r`
//!   stops between the pixel and the white point, a gain up to `r/2` is
//!   taken whole and a larger gain `G` becomes `r − r²/(4G)`, so nothing is
//!   pushed through white ([`limit_gain`]).

use crate::raw_color::{apply, prophoto_luma, prophoto_to_rec709, xy_from_temperature, RawCamera, Xy, M3};
pub use crate::raw_local::local_curves;

/// The raw development settings of a clip: the white balance and the tone
/// controls of a raw converter's basic panel. `Default` develops the frame
/// as shot with nothing moved.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RawSettings {
    /// White balance as a colour temperature in kelvin; `None` = as shot.
    pub temperature: Option<f32>,
    /// Tint beside that temperature (positive = magenta); `None` = as shot.
    pub tint: Option<f32>,
    /// Exposure in stops.
    pub exposure: f32,
    /// −100..100.
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    /// Sharpening of the developed picture (the detail panel's four
    /// controls): see [`crate::raw_sharpen`].
    pub sharpen: crate::raw_sharpen::RawSharpen,
    /// Colour noise reduction, 0..100 (a raw converter's default is 25):
    /// see [`crate::raw_chroma`].
    pub color_noise: f32,
    /// The exposure is set by the clip itself, frame by frame, in place of
    /// `exposure`: each frame gets the exposure at which its picture, as
    /// the other settings here develop it, comes out at the brightness a
    /// normally exposed picture has ([`crate::clip_stats::AutoExposure`]).
    /// A clip only: a still keeps `exposure`.
    pub auto_exposure: bool,
    /// Stops added to what the automatic exposure finds: brighter or
    /// darker than its own target.
    pub auto_ev: f32,
    /// How slowly the automatic exposure follows the picture, in seconds
    /// (the width of the stretch of the clip each frame's exposure is
    /// averaged over): 0 follows the picture as it changes, a large value
    /// gives the whole clip one exposure.
    pub auto_smooth: f32,
}

/// The colour noise reduction a clip starts with: a raw converter's default.
pub const DEFAULT_COLOR_NOISE: f32 = 25.0;

/// How slowly an automatic exposure follows the picture when nothing else
/// is set, in seconds.
pub const DEFAULT_AUTO_SMOOTH: f32 = 2.0;

impl Default for RawSettings {
    fn default() -> Self {
        Self {
            temperature: None, tint: None, exposure: 0.0, contrast: 0.0, highlights: 0.0, shadows: 0.0, whites: 0.0, blacks: 0.0,
            sharpen: crate::raw_sharpen::RawSharpen::default(),
            color_noise: DEFAULT_COLOR_NOISE,
            auto_exposure: false, auto_ev: 0.0, auto_smooth: DEFAULT_AUTO_SMOOTH,
        }
    }
}

/// What the rendering reads off the frame itself, on a reduced copy some
/// 126 cells along the frame's longer side, in scene values at exposure 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneKeys {
    /// The brightest channel of any cell.
    pub max: f32,
    /// The luminance of the darkest cell: the frame's darkest level.
    /// (Measured on frames: the default black follows the darkest cell, not
    /// a low percentile of the cells — half a stop to a stop lower.)
    pub dark: f32,
}

impl ToneKeys {
    /// A frame nothing is known about: bright enough for the sensor's white
    /// to be the white point, with the gentlest default black.
    pub const UNKNOWN: ToneKeys = ToneKeys { max: f32::MAX, dark: 0.0 };

    /// The keys of a reduced copy: `cells` are scene RGB means (linear
    /// ProPhoto), any order. `None` when there are none.
    pub fn from_cells(cells: &[[f32; 3]]) -> Option<Self> {
        if cells.is_empty() {
            return None;
        }
        let luma = prophoto_luma().map(|v| v as f32);
        let max = cells.iter().flat_map(|c| c.iter().copied()).fold(0f32, f32::max);
        let dark = cells.iter().map(|c| (c[0] * luma[0] + c[1] * luma[1] + c[2] * luma[2]).max(0.0)).fold(f32::MAX, f32::min);
        Some(Self { max, dark })
    }
}

/// What a frame's rendering is derived from: its tone keys and the levels
/// of its reduced copy.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    pub keys: ToneKeys,
    /// log₂ luminance of the reduced copy's cells (each [`STAT_CELL`] of the
    /// frame's long side, averaged in log), at exposure 0, sorted rising;
    /// a pixel with a channel above 1 (a blown highlight, rebuilt) counts
    /// with that channel at 1. Empty when nothing was measured.
    pub levels: std::sync::Arc<[f32]>,
    /// The brightest value at each level: for sample `i` of a local curve
    /// ([`LOCAL_SAMPLES`] of them), the brightest channel of the key cells
    /// whose own level is nearest to that sample's, 0 where there is none;
    /// scene values at exposure 0. What the white point of the frame with
    /// Shadows / Highlights in is found from. Empty when nothing was
    /// measured.
    pub peaks: std::sync::Arc<[f32]>,
}

impl FrameStats {
    /// A frame nothing is known about.
    pub fn unknown() -> Self {
        Self { keys: ToneKeys::UNKNOWN, levels: Vec::new().into(), peaks: Vec::new().into() }
    }
}

/// A cell of the reduced copy the tone curves are derived from, as a
/// fraction of the frame's long side.
pub const STAT_CELL: f32 = 1.0 / 42.0;
/// A cell of the base's grid, as a fraction of the frame's long side.
pub const BASE_CELL: f32 = 0.006;
/// A cell of the base's grid along level, in stops.
pub const BASE_RANGE: f32 = 1.2;
/// The grid's first level (log₂ of the scene value) and its cells along level.
pub const BASE_LO: f32 = -15.0;
pub const BASE_LEVELS: u32 = 18;
/// The darkest luminance the base distinguishes.
pub const BASE_FLOOR: f32 = 6.103_515_6e-5;
/// Weight of the plain local average where the grid holds nothing at the
/// pixel's own level (a speck unlike everything around it takes the gain
/// of its surroundings).
pub const BASE_FALLBACK: f32 = 0.5;

/// log₂ of the base level of the first sample of a local curve, samples
/// per stop, and the sample count (levels −16 … +4).
pub const LOCAL_LO: f32 = -16.0;
pub const LOCAL_PER_STOP: f32 = 16.0;
pub const LOCAL_SAMPLES: usize = 321;

/// The frame's Shadows / Highlights curves: log₂ gain at full slider over
/// the base level ([`LOCAL_SAMPLES`] samples from [`LOCAL_LO`]), and what the
/// two sliders do to each other.
///
/// Measured on Camera Raw with every pair of settings (charts whose levels
/// are spread like a frame's):
///
/// - Either slider alone is linear in the slider, in stops. Shadows −100 is
///   the negative of +100. Highlights +100 has its own curve near the
///   frame's top (the −100 one levels off there, the +100 one does not);
///   it is kept here as it is before the white limiter ([`limit_gain`]).
/// - The two curves share a pivot: the level at which Shadows +100 and
///   Highlights −100 cancel.
/// - **Both the same way** (both up, or both down): the sum of the two,
///   plus one constant over the whole range ([`Self::shift`], about twice
///   the gain at the pivot): the picture as a whole goes further.
/// - **In opposition** (one up, one down): the sum, plus a tilt about the
///   pivot — two straight lines, from nothing at the pivot to
///   [`Self::tilt_top`] at the frame's top and [`Self::tilt_bottom`] at its
///   black end: each slider alone holds its far end still, together they
///   let go of it.
/// - Away from the corners these additions scale with the smaller of the
///   two amounts, bent toward the larger (see [`Self::gain`]).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalCurves {
    /// Shadows +100 (−100 is its negative).
    pub shadows: Vec<f32>,
    /// Highlights −100.
    pub highlights: Vec<f32>,
    /// Highlights +100, before the white limiter.
    pub highlights_up: Vec<f32>,
    /// What Shadows +100 with Highlights −100 add to their sum at the
    /// frame's top and at its black end, in stops.
    pub tilt_top: f32,
    pub tilt_bottom: f32,
    /// What Shadows +100 with Highlights +100 add to their sum, in stops.
    pub shift: f32,
    /// The frame's own range: the levels (log₂) of its top and of its black
    /// end.
    pub top: f32,
    pub bottom: f32,
}

impl LocalCurves {
    /// No gain anywhere.
    pub fn flat() -> Self {
        let z = || vec![0.0; LOCAL_SAMPLES];
        Self { shadows: z(), highlights: z(), highlights_up: z(), tilt_top: 0.0, tilt_bottom: 0.0, shift: 0.0, top: 0.0, bottom: 0.0 }
    }

    /// The level (log₂) at which Shadows +100 and Highlights −100 cancel;
    /// the middle of the frame's range when they never do.
    pub fn pivot(&self) -> f32 {
        let sum = |i: usize| self.shadows[i] + self.highlights[i];
        let crossings: Vec<f32> = (0..LOCAL_SAMPLES - 1).filter(|&i| sum(i) > 0.0 && sum(i + 1) <= 0.0)
            .map(|i| LOCAL_LO + (i as f32 + sum(i) / (sum(i) - sum(i + 1))) / LOCAL_PER_STOP).collect();
        crossings.get(crossings.len() / 2).copied().unwrap_or(0.5 * (self.top + self.bottom))
    }

    /// The gain curve of the settings, before the white limiter: `shadows`,
    /// `highlights` in −100..100.
    ///
    /// With `s`, `h` the two amounts (0..1) and `m` the smaller of them:
    /// the same way, the shift comes in as `m · (m / larger)^¼`; in
    /// opposition, the tilt comes in above the pivot as
    /// `min(s, √(s·h))` and below it as `min(h, √(s·h))` — the far end a
    /// slider holds still is let go as soon as the other one moves at all.
    /// (Measured on a 9 × 9 grid of settings: within 0.02–0.04 stops; the
    /// four quadrants mirror each other.)
    pub fn gain(&self, shadows: f32, highlights: f32) -> Vec<f32> {
        let (s, h) = (shadows / 100.0, highlights / 100.0);
        let (sa, ha) = (s.abs(), h.abs());
        let single = if h < 0.0 { &self.highlights } else { &self.highlights_up };
        let plain = |i: usize| s * self.shadows[i] + ha * single[i];
        if sa == 0.0 || ha == 0.0 {
            return (0..LOCAL_SAMPLES).map(plain).collect();
        }
        let sign = if s < 0.0 { -1.0 } else { 1.0 };
        let (least, most) = (sa.min(ha), sa.max(ha));
        if (s > 0.0) == (h > 0.0) {
            let add = sign * self.shift * least * (least / most).powf(0.25);
            return (0..LOCAL_SAMPLES).map(|i| plain(i) + add).collect();
        }
        let pivot = self.pivot();
        let mean = (sa * ha).sqrt();
        let (above, below) = (sa.min(mean), ha.min(mean));
        (0..LOCAL_SAMPLES).map(|i| {
            let level = (LOCAL_LO + i as f32 / LOCAL_PER_STOP).clamp(self.bottom, self.top);
            let tilt = if level >= pivot {
                above * self.tilt_top * (level - pivot) / (self.top - pivot).max(1e-3)
            } else {
                below * self.tilt_bottom * (pivot - level) / (pivot - self.bottom).max(1e-3)
            };
            plain(i) + sign * tilt
        }).collect()
    }
}

/// A gain of `gain` stops where `room` stops are left below the white point:
/// taken whole up to half the room, beyond that `room − room² / (4·gain)`,
/// which approaches the room and never passes it. Gains of zero or less are
/// not touched. (Measured on Camera Raw: Highlights +100 and both sliders up
/// on charts reaching white, exact to a hundredth of a stop.)
pub fn limit_gain(gain: f32, room: f32) -> f32 {
    let room = room.max(0.0);
    if gain <= 0.5 * room { gain } else { room - room * room / (4.0 * gain) }
}

/// How a frame that has no clip's statistics to go by is measured (a clip's
/// frames are all rendered by the clip's: [`crate::clip_stats`], through
/// [`crate::gpu::ColorStackPlan::clip_stats`]; then nothing is measured).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Measure {
    /// Measure this frame and wait for the result (a still, which is its
    /// own one picture).
    #[default]
    Wait,
    /// Use the last measurement that has come back and queue one of this
    /// frame (the live preview of a clip that has not been sampled yet: it
    /// must not wait on the GPU).
    Latest,
}

/// What the statistics of frames that were each given an exposure of their
/// own no longer tell ([`RawDevelop::constants_exposed`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposed {
    /// How far the top of the levels was below the sensor's clipping when
    /// it was recorded, in stops ([`crate::raw_local::local_curves`]).
    pub headroom: f32,
    /// The largest of the exposures given, as a factor: where the sensor's
    /// white lies among the exposed values.
    pub sensor_gain: f64,
}

/// The raw development stage of a colour stack: a camera and the settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawDevelop {
    pub camera: RawCamera,
    pub settings: RawSettings,
    pub measure: Measure,
}

impl RawDevelop {
    /// The white the frame is developed for.
    pub fn white(&self) -> Xy {
        let (k, t) = self.camera.as_shot_temperature();
        match (self.settings.temperature, self.settings.tint) {
            (None, None) => self.camera.neutral_to_xy(self.camera.as_shot_neutral),
            (kelvin, tint) => xy_from_temperature(kelvin.map_or(k, |v| v as f64), tint.map_or(t, |v| v as f64)),
        }
    }

    /// Exposure in stops, the camera's own offset included.
    fn exposure(&self) -> f64 {
        self.settings.exposure as f64 + self.camera.baseline_exposure
    }

    /// This stage with its exposure set so that the whole of it, the
    /// camera's own offset included, is `stops`.
    pub fn with_whole_exposure(&self, stops: f32) -> Self {
        Self { settings: RawSettings { exposure: stops - self.camera.baseline_exposure as f32, ..self.settings }, ..*self }
    }

    /// What the Exposure control would read for a whole exposure of `stops`.
    pub fn exposure_control(&self, stops: f32) -> f32 {
        stops - self.camera.baseline_exposure as f32
    }

    /// What does not depend on the frame: the camera's white and matrices.
    pub fn geometry(&self) -> DevelopGeometry {
        let white = self.white();
        DevelopGeometry {
            neutral: self.camera.camera_white(white).map(|v| v as f32),
            camera_to_work: to_f32(&self.camera.camera_to_prophoto(white)),
            work_to_out: to_f32(&prophoto_to_rec709()),
        }
    }

    /// The settings move Shadows or Highlights: the frame needs its base.
    pub fn is_local(&self) -> bool {
        self.settings.shadows != 0.0 || self.settings.highlights != 0.0
    }

    /// The settings move Whites or Blacks: stages of their own after the
    /// white point.
    pub fn is_keyed(&self) -> bool {
        self.settings.whites != 0.0 || self.settings.blacks != 0.0
    }

    /// Everything the stage needs for one frame with the keys `keys` and
    /// nothing else known of it (Shadows / Highlights then have no curves
    /// to go by: see [`Self::constants_for`]).
    pub fn constants(&self, keys: ToneKeys) -> DevelopConstants {
        self.constants_for(&FrameStats { keys, levels: Vec::new().into(), peaks: Vec::new().into() }, None)
    }

    /// Everything the stage needs for one frame: `stats` measured on it;
    /// `curves` replaces the Shadows / Highlights curves derived from them.
    pub fn constants_for(&self, stats: &FrameStats, curves: Option<&LocalCurves>) -> DevelopConstants {
        self.constants_exposed(stats, curves, None)
    }

    /// [`Self::constants_for`] for statistics that are not as the sensor
    /// recorded them: those of a clip whose frames have each been given an
    /// exposure of their own already (`exposed` says what the statistics
    /// alone no longer tell), which this stage's exposure is then added to.
    pub fn constants_exposed(&self, stats: &FrameStats, curves: Option<&LocalCurves>, exposed: Option<Exposed>) -> DevelopConstants {
        let geometry = self.geometry();
        let gain = 2f64.powf(self.exposure());
        let sensor_white = self.sensor_white(gain * exposed.map_or(1.0, |e| e.sensor_gain));
        let plain_white = white_point_of(stats.keys.max as f64 * gain, sensor_white);
        let dark_key = (stats.keys.dark as f64 * gain).max(2f64.powi(-30)).log2();
        // Shadows / Highlights: the gain over the base level, and the white
        // point of the frame with it in.
        let log_gain: Option<Vec<f32>> = self.is_local().then(|| {
            let derived;
            let curves = match curves {
                Some(c) => c,
                None => {
                    let shift = gain.log2() as f32;
                    let levels: Vec<f32> = stats.levels.iter().map(|l| l + shift).collect();
                    // How far the frame's top is below the sensor's
                    // clipping, as shot: exposure moves the curves with the
                    // frame, it does not change them (measured with
                    // exposure series).
                    let headroom = exposed.map(|e| e.headroom).unwrap_or_else(|| stats.levels.last().map_or(0.0, |top| {
                        let shot_key = (stats.keys.dark as f64).max(2f64.powi(-30)).log2();
                        -(default_black(2f64.powf(*top as f64), shot_key).max(1e-9).log2()) as f32
                    }));
                    derived = local_curves(&levels, headroom);
                    &derived
                }
            };
            curves.gain(self.settings.shadows, self.settings.highlights)
        });
        let white = match &log_gain {
            Some(g) => gained_white(stats, g, gain, dark_key, plain_white, sensor_white),
            None => plain_white,
        };
        // Contrast: the tone curve the fed value goes through.
        let tone = self.tone_table(&stats.levels);
        let tone_at = |fed: f64| match &tone {
            Some(t) => lerp_table(t, fed.clamp(0.0, 1.0) * (t.len() - 1) as f64),
            None => default_curve(fed),
        };
        let hi = plain_white.log2().max(0.0);
        let curve: Vec<f32> = (0..CURVE_SAMPLES).map(|i| {
            let x = 2f64.powf(CURVE_LO + (hi - CURVE_LO) * i as f64 / (CURVE_SAMPLES - 1) as f64);
            tone_at(default_white(default_black(x, dark_key), plain_white)) as f32
        }).collect();
        let floor = tone_at(0.0) as f32;
        // The staged development: whatever sits between the default black
        // and the tone curve.
        let staged = (log_gain.is_some() || self.is_keyed()).then(|| {
            let log_gain = log_gain.unwrap_or_else(|| vec![0.0; LOCAL_SAMPLES]);
            let level = |i: usize, lo: f64, hi: f64| 2f64.powf(lo + (hi - lo) * i as f64 / (CURVE_SAMPLES - 1) as f64);
            let black_hi = sensor_white.log2().max(0.0);
            let black: Vec<f32> = (0..CURVE_SAMPLES).map(|i| default_black(level(i, CURVE_LO, black_hi), dark_key) as f32).collect();
            let white_hi = white.log2().max(0.0);
            let fed: Vec<f32> = (0..CURVE_SAMPLES).map(|i| default_white(level(i, OUT_LO, white_hi), white).clamp(0.0, 1.0) as f32).collect();
            // Whites, then Blacks, as maps of the fed value. The frame's
            // darkest level as Blacks meet it: exposed, and with the
            // Shadows / Highlights gain of that level.
            let dark = dark_key as f32 + sample_curve(&log_gain, (dark_key as f32 - LOCAL_LO) * LOCAL_PER_STOP);
            let fed_map = |map: Option<Vec<f32>>| -> (Vec<f32>, f32) {
                let at = |v: f64| -> f64 {
                    match &map {
                        Some(m) => default_curve_inverse(srgb_decode(crate::raw_global::apply(m, srgb_encode(default_curve(v) as f32))) as f64),
                        None => v,
                    }
                };
                ((0..CURVE_SAMPLES).map(|i| at(level(i, OUT_LO, 0.0)) as f32).collect(), at(0.0) as f32)
            };
            // Whites above zero go by the frame as they meet it: its levels
            // and its top with exposure and the gain in, the top as the
            // white point hands it on.
            let whites_up = || -> Option<(f32, f32)> {
                let shift = gain.log2() as f32;
                let gained = |l: f32| l + shift + sample_curve(&log_gain, (l + shift - LOCAL_LO) * LOCAL_PER_STOP);
                let mut levels: Vec<f32> = stats.levels.iter().map(|l| gained(*l)).collect();
                levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let top = stats.levels.last()?;
                let lit = default_black(2f64.powf((*top + shift) as f64), dark_key) * 2f64.powf((gained(*top) - *top - shift) as f64);
                let fed_top = default_white(lit, white).clamp(1e-9, 1.0).log2() as f32;
                crate::raw_global::whites_up(self.settings.whites, &levels, fed_top)
            };
            let (whites, whites_floor) = match whites_up() {
                Some((a1, a2)) => ((0..CURVE_SAMPLES).map(|i| {
                    let v = level(i, OUT_LO, 0.0);
                    (a1 as f64 * v + a2 as f64 * v * v).min(1.0) as f32
                }).collect(), 0.0),
                None => fed_map(crate::raw_global::whites_map(self.settings.whites)),
            };
            let (blacks, blacks_floor) = fed_map(crate::raw_global::blacks_map(self.settings.blacks, dark));
            let tone: Vec<f32> = (0..CURVE_SAMPLES).map(|i| tone_at(level(i, OUT_LO, 0.0)) as f32).collect();
            LocalTone {
                gain: log_gain.into(),
                black: black.into(),
                black_hi: black_hi as f32,
                fed: fed.into(),
                fed_hi: white_hi as f32,
                whites: StageMap { map: whites.into(), floor: whites_floor, keyed: self.settings.whites < 0.0 },
                blacks: StageMap { map: blacks.into(), floor: blacks_floor, keyed: self.settings.blacks > 0.0 },
                tone: tone.into(),
                white: white as f32,
                floor,
                long_side: 0,
            }
        });
        DevelopConstants { geometry, gain: gain as f32, curve: curve.into(), curve_hi: hi as f32, white_point: plain_white as f32, dark_key: dark_key as f32, floor, local: staged, sharpen: None, chroma: None }
    }

    /// The frame's tone curve with Contrast in: output light over the value
    /// the default curve is fed, [`TONE_SAMPLES`] equally spaced samples
    /// over 0..1. `None`: the default curve. `levels`: the frame's reduced
    /// copy as shot (exposure and Shadows / Highlights do not change the
    /// curve: measured).
    fn tone_table(&self, levels: &[f32]) -> Option<Vec<f32>> {
        let map = crate::raw_global::contrast_map(self.settings.contrast, levels)?;
        Some((0..TONE_SAMPLES).map(|i| {
            let e = srgb_encode(default_curve(i as f64 / (TONE_SAMPLES - 1) as f64) as f32);
            srgb_decode(crate::raw_global::apply(&map, e))
        }).collect())
    }

    /// The sensor's white at an exposure gain: where a neutral's last
    /// channel clips.
    fn sensor_white(&self, gain: f64) -> f64 {
        let lowest = self.geometry().neutral.iter().cloned().fold(f32::MAX, f32::min).max(1e-3) as f64;
        gain / lowest
    }
}

/// The white point of a frame whose brightest value is `frame_max`: twice
/// that, at least 1, at most the sensor's white.
fn white_point_of(frame_max: f64, sensor_white: f64) -> f64 {
    sensor_white.min(2.0 * frame_max).max(1.0)
}

/// The white point of the frame with the Shadows / Highlights gain
/// `log_gain` in. The frame's brightest value then depends on the gain at
/// its level, and for gains above zero on the limiter, which depends on the
/// white point: found by starting from the white point of the gain taken
/// whole (an upper bound) and letting the two settle.
fn gained_white(stats: &FrameStats, log_gain: &[f32], exposure_gain: f64, dark_key: f64, plain: f64, sensor_white: f64) -> f64 {
    if stats.peaks.is_empty() {
        return plain;
    }
    let shift = exposure_gain.log2();
    // Per level the frame has: its brightest value, the black-subtracted
    // luminance of that level, the gain there.
    let cells: Vec<(f64, f64, f64)> = stats.peaks.iter().enumerate().filter(|(_, p)| **p > 0.0).map(|(i, p)| {
        let level = LOCAL_LO as f64 + i as f64 / LOCAL_PER_STOP as f64 + shift;
        let g = sample_curve(log_gain, (level as f32 - LOCAL_LO) * LOCAL_PER_STOP) as f64;
        (*p as f64 * exposure_gain, default_black(2f64.powf(level), dark_key).max(1e-12), g)
    }).collect();
    let white_of = |limited_to: Option<f64>| -> f64 {
        let top = cells.iter().map(|(peak, lit, g)| {
            let g = match limited_to {
                Some(w) => limit_gain(*g as f32, (w / lit).log2() as f32) as f64,
                None => *g,
            };
            peak * 2f64.powf(g)
        }).fold(0.0, f64::max);
        white_point_of(top, sensor_white)
    };
    let mut white = white_of(None);
    for _ in 0..8 {
        white = white_of(Some(white));
    }
    white
}

fn to_f32(m: &M3) -> [[f32; 3]; 3] {
    m.map(|r| r.map(|v| v as f32))
}

/// The camera's white and the two matrices of a development.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DevelopGeometry {
    /// Camera values of the white, largest channel = 1.
    pub neutral: [f32; 3],
    pub camera_to_work: [[f32; 3]; 3],
    pub work_to_out: [[f32; 3]; 3],
}

impl DevelopGeometry {
    /// Scene RGB (linear ProPhoto, exposure 0) of a camera pixel, its
    /// clipped channels rebuilt: white-balanced, every channel reads what a
    /// neutral of that level would; a channel near its clip point is raised
    /// to what the brightest channel says.
    pub fn scene(&self, camera: [f32; 3]) -> [f32; 3] {
        let wb = [0, 1, 2].map(|i| camera[i].max(0.0) / self.neutral[i]);
        let est = wb[0].max(wb[1]).max(wb[2]);
        let rebuilt = [0, 1, 2].map(|i| {
            let w = smoothstep(CLIP_FROM, CLIP_TO, camera[i]);
            (wb[i] + w * (est - wb[i])) * self.neutral[i]
        });
        let m = self.camera_to_work.map(|r| r.map(|x| x as f64));
        apply(&m, rebuilt.map(|x| x as f64)).map(|x| x as f32)
    }
}

/// A camera channel counts as clipped from here (of the sensor's white)…
pub const CLIP_FROM: f32 = 0.92;
/// …to here, where it is taken as fully clipped.
pub const CLIP_TO: f32 = 0.995;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── The measured default rendering ─────────────────────────────────────

fn floats(bytes: &'static [u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

struct Tables {
    /// 1025 nodes over 0..1.
    curve: Vec<f32>,
    /// `BLACK_KEYS` rows × `BLACK_COLS`: what the default black takes off
    /// at log₂ x = −14 … 0.
    black: Vec<f32>,
    /// `WHITE_ROWS` × `WHITE_COLS`: the white point's output at
    /// `x = 0.25 · (W / 0.25)^t`, `t` = 0 … 1, for `W` = 2^(row / 8).
    white: Vec<f32>,
}

const BLACK_KEYS: usize = 15;
const BLACK_COLS: usize = 449;
const BLACK_KEY_LO: f64 = -14.0;
const BLACK_KEY_HI: f64 = -7.0;
const WHITE_ROWS: usize = 57;
const WHITE_COLS: usize = 257;

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let t = Tables {
            curve: floats(include_bytes!("data/raw_default_curve.f32")),
            black: floats(include_bytes!("data/raw_default_black.f32")),
            white: floats(include_bytes!("data/raw_white_point.f32")),
        };
        assert_eq!((t.curve.len(), t.black.len(), t.white.len()), (1025, BLACK_KEYS * BLACK_COLS, WHITE_ROWS * WHITE_COLS));
        t
    })
}

fn lerp_table(row: &[f32], at: f64) -> f64 {
    let last = row.len() - 1;
    let y = at.clamp(0.0, last as f64);
    let i = (y.floor() as usize).min(last - 1);
    let f = y - i as f64;
    row[i] as f64 * (1.0 - f) + row[i + 1] as f64 * f
}

/// The default tone curve: scene 0..1 → output light 0..1.
pub fn default_curve(x: f64) -> f64 {
    lerp_table(&tables().curve, x.clamp(0.0, 1.0) * 1024.0)
}

/// The value the default tone curve turns into the output light `y`.
pub fn default_curve_inverse(y: f64) -> f64 {
    if y <= 0.0 { return 0.0; }
    if y >= 1.0 { return 1.0; }
    let (mut lo, mut hi) = (0f64, 1f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if default_curve(mid) < y { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// The default black of a frame whose darkest level is `2^dark_key`.
pub fn default_black(x: f64, dark_key: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return x; }
    let t = tables();
    let k = (dark_key.clamp(BLACK_KEY_LO, BLACK_KEY_HI) - BLACK_KEY_LO) * 2.0;
    let i = (k.floor() as usize).min(BLACK_KEYS - 2);
    let f = k - i as f64;
    let u = x.log2();
    let off = |row: usize| -> f64 {
        let r = &t.black[row * BLACK_COLS..(row + 1) * BLACK_COLS];
        // Below the table's first column the amount shrinks with the level.
        if u < -14.0 { r[0] as f64 * x / 2f64.powi(-14) } else { lerp_table(r, (u + 14.0) * 32.0) }
    };
    (x - (off(i) * (1.0 - f) + off(i + 1) * f)).max(0.0)
}

/// The default white point: a shoulder that takes `white` to 1 and leaves
/// everything below 0.25 alone.
pub fn default_white(x: f64, white: f64) -> f64 {
    if x <= 0.25 { return x; }
    let white = white.max(1.0);
    if white <= 1.0 + 1e-9 { return x.min(1.0); }
    let t = tables();
    let s = (white.log2() * 8.0).clamp(0.0, (WHITE_ROWS - 1) as f64);
    let i = (s.floor() as usize).min(WHITE_ROWS - 2);
    let f = s - i as f64;
    let pos = ((x / 0.25).ln() / (white / 0.25).ln()).clamp(0.0, 1.0) * (WHITE_COLS - 1) as f64;
    let row = |r: usize| lerp_table(&t.white[r * WHITE_COLS..(r + 1) * WHITE_COLS], pos);
    row(i) * (1.0 - f) + row(i + 1) * f
}

/// The whole default rendering of a scene value: default black, white
/// point, tone curve → output light.
pub fn default_render(x: f64, white_point: f64, dark_key: f64) -> f64 {
    default_curve(default_white(default_black(x, dark_key), white_point).clamp(0.0, 1.0))
}

/// Samples of the per-frame curve.
pub const CURVE_SAMPLES: usize = 4096;
/// Samples of a frame's own tone curve over the value it is fed (0..1).
pub const TONE_SAMPLES: usize = 2049;
/// log₂ of the scene value of the curve's first sample; below it the
/// output shrinks with the value.
pub const CURVE_LO: f64 = -16.0;

/// A sampled curve shared with the GPU.
pub type CurveSamples = std::sync::Arc<[f32]>;

/// The per-frame constants of the stage.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopConstants {
    pub geometry: DevelopGeometry,
    /// Linear gain of the exposure.
    pub gain: f32,
    /// Output light over log₂(scene) from [`CURVE_LO`] to `curve_hi`,
    /// [`CURVE_SAMPLES`] equally spaced samples.
    pub curve: CurveSamples,
    pub curve_hi: f32,
    /// The white point and the dark key the curve was built for.
    pub white_point: f32,
    pub dark_key: f32,
    /// Output light of a scene value of zero (above zero when Blacks lift).
    pub floor: f32,
    /// Shadows / Highlights, when the settings move them.
    pub local: Option<LocalTone>,
    /// The sharpening of the developed picture, when it is on. Not set by
    /// [`RawDevelop::constants_for`], which does not know the size the
    /// frame is drawn at: see [`crate::gpu::Device::raw_develop_constants`].
    pub sharpen: Option<crate::raw_sharpen::SharpenConstants>,
    /// The colour noise reduction ahead of the development, when it is on.
    /// Set beside `sharpen`, for the same reason.
    pub chroma: Option<crate::raw_chroma::ChromaConstants>,
}

/// log₂ of the first sample of the staged development's tables over a
/// gained or fed value; below it they run straight to their value at zero.
pub const OUT_LO: f64 = -18.0;

/// Whites or Blacks for one frame: what a fed value becomes.
///
/// Measured on frames against Camera Raw, the two sliders act on colour in
/// two ways. Where they take contrast out — Blacks above zero, Whites below
/// — they are a gain keyed to the pixel's own luminance, the same for its
/// three channels (0.3–0.5/255; put through the tone curve instead, lifted
/// shadows lose their colour: 2/255). Where they add contrast — Blacks
/// below zero, Whites above — they are a curve the largest and the smallest
/// channel go through, like the tone curve (0.6–0.8/255; as a gain: 3–5).
#[derive(Debug, Clone, PartialEq)]
pub struct StageMap {
    /// What a fed value becomes, over its log₂ from [`OUT_LO`] to 0.
    pub map: CurveSamples,
    /// What a fed value of zero becomes.
    pub floor: f32,
    /// A gain keyed to the pixel's luminance (else a curve per extreme).
    pub keyed: bool,
}

impl StageMap {
    pub fn at(&self, v: f32) -> f32 {
        staged_sample(&self.map, 0.0, self.floor, v)
    }

    /// The largest and the smallest fed channel of a pixel through the
    /// stage; `between` = where its luminance lies between them (0..1).
    pub fn extremes(&self, hi: f32, lo: f32, between: f32) -> (f32, f32) {
        if !self.keyed {
            return (self.at(hi), self.at(lo));
        }
        // The luminance goes where the map sends it, the channels keep
        // their ratios (next to black: their differences).
        let y = lo + (hi - lo) * between;
        let to = self.at(y);
        let over = y.max(2f32.powf(OUT_LO as f32));
        let (scale, add) = (to / over, to * (1.0 - y / over));
        (scale * hi + add, scale * lo + add)
    }
}

/// The staged development of one frame — what sits between the default
/// black and the tone curve: the Shadows / Highlights gain, the white point
/// of the gained frame, then Whites and Blacks.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalTone {
    /// log₂ gain over the base level, before the white limiter:
    /// [`LOCAL_SAMPLES`] samples from [`LOCAL_LO`], [`LOCAL_PER_STOP`] per
    /// stop. All zero without Shadows / Highlights.
    pub gain: CurveSamples,
    /// The black-subtracted value over log₂(scene): [`CURVE_SAMPLES`]
    /// equally spaced samples from [`CURVE_LO`] to `black_hi` (the sensor's
    /// white).
    pub black: CurveSamples,
    pub black_hi: f32,
    /// The value the tone curve is fed over log₂ of the gained value, from
    /// [`OUT_LO`] to `fed_hi` (the white point): the white point's shoulder.
    pub fed: CurveSamples,
    pub fed_hi: f32,
    /// Whites, then Blacks.
    pub whites: StageMap,
    pub blacks: StageMap,
    /// Output light over log₂ of the fed value, from [`OUT_LO`] to 0: the
    /// tone curve (Contrast in).
    pub tone: CurveSamples,
    /// The white point of the frame with the gain in.
    pub white: f32,
    /// Output light of a fed value of zero.
    pub floor: f32,
    /// The long side, in pixels, of the frame the eyes belong to (two eyes
    /// side by side count as one frame); 0 = the eye's own.
    pub long_side: u32,
}

/// A table over log₂ of a value from [`OUT_LO`] to `hi`, at `v`; below the
/// first sample straight to `zero` at zero.
fn staged_sample(table: &[f32], hi: f32, zero: f32, v: f32) -> f32 {
    if v <= 0.0 { return zero; }
    let lo = OUT_LO as f32;
    let u = v.log2();
    if u <= lo { return zero + (table[0] - zero) * v / 2f32.powf(lo); }
    sample_curve(table, (u - lo) / (hi - lo) * (table.len() - 1) as f32)
}

impl LocalTone {
    /// The gain (log₂) at a base level, before the white limiter.
    pub fn gain_at(&self, base: f32) -> f32 {
        sample_curve(&self.gain, (base - LOCAL_LO) * LOCAL_PER_STOP)
    }

    /// The black-subtracted value of the scene value `x`.
    pub fn black_of(&self, x: f32) -> f32 {
        if x <= 0.0 { return 0.0; }
        let lo = CURVE_LO as f32;
        let u = x.log2();
        if u <= lo { return self.black[0] * x / 2f32.powf(lo); }
        sample_curve(&self.black, (u - lo) / (self.black_hi - lo) * (self.black.len() - 1) as f32)
    }

    /// The value the tone curve is fed for the gained value `v`.
    pub fn fed_of(&self, v: f32) -> f32 {
        staged_sample(&self.fed, self.fed_hi, 0.0, v)
    }

    /// Output light of the fed value `v`.
    pub fn tone_of(&self, v: f32) -> f32 {
        staged_sample(&self.tone, 0.0, self.floor, v)
    }

    /// The gain (linear) of a pixel whose base level is `base` and whose own
    /// luminance (scene value) is `luminance`: the curve's, limited by the
    /// room the pixel has below the white point.
    pub fn gain_of(&self, base: f32, luminance: f32) -> f32 {
        let lit = self.black_of(luminance).max(1e-12);
        2f32.powf(limit_gain(self.gain_at(base), self.white.log2() - lit.log2()))
    }
}

fn sample_curve(curve: &[f32], at: f32) -> f32 {
    let last = (curve.len() - 1) as f32;
    let y = at.clamp(0.0, last);
    let i = (y.floor() as usize).min(curve.len() - 2);
    let f = y - i as f32;
    curve[i] * (1.0 - f) + curve[i + 1] * f
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_decode(e: f32) -> f32 {
    if e <= 0.040_45 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) }
}

impl DevelopConstants {
    /// The frame's curve at the scene value `x`.
    pub fn tone(&self, x: f32) -> f32 {
        if x <= 0.0 { return self.floor; }
        let lo = CURVE_LO as f32;
        let u = x.log2();
        if u <= lo {
            return self.floor + (self.curve[0] - self.floor) * x / 2f32.powf(lo);
        }
        let last = (self.curve.len() - 1) as f32;
        let y = ((u - lo) / (self.curve_hi - lo) * last).min(last);
        let i = (y.floor() as usize).min(self.curve.len() - 2);
        let f = y - i as f32;
        self.curve[i] * (1.0 - f) + self.curve[i + 1] * f
    }

    /// The luminance the base is taken from: of the scene value at this
    /// frame's exposure, not below [`BASE_FLOOR`].
    pub fn base_luminance(&self, camera: [f32; 3]) -> f32 {
        let luma = prophoto_luma().map(|v| v as f32);
        let scene = self.geometry.scene(camera).map(|v| (v * self.gain).max(0.0));
        (scene[0] * luma[0] + scene[1] * luma[1] + scene[2] * luma[2]).max(BASE_FLOOR)
    }

    /// Develop one pixel whose base level (log₂) is `base`: the default
    /// black, the Shadows / Highlights gain of that level (limited toward
    /// the white point), the white point of the gained frame, Whites, then
    /// Blacks, the tone curve. The GPU stage computes the same.
    pub fn develop_local(&self, camera: [f32; 3], base: f32) -> [f32; 3] {
        let Some(local) = &self.local else { return self.develop(camera) };
        let scene = self.geometry.scene(camera).map(|v| (v * self.gain).max(0.0));
        let luma = prophoto_luma().map(|v| v as f32);
        let luminance = scene[0] * luma[0] + scene[1] * luma[1] + scene[2] * luma[2];
        let g = local.gain_of(base, luminance);
        let hi = scene.iter().cloned().fold(f32::MIN, f32::max);
        let lo = scene.iter().cloned().fold(f32::MAX, f32::min);
        let (fhi, flo) = (local.fed_of(g * local.black_of(hi)), local.fed_of(g * local.black_of(lo)));
        // Where the pixel's luminance lies between its extremes: kept by
        // every stage from here on.
        let between = if hi > lo { (luminance - lo) / (hi - lo) } else { 0.0 };
        let (whi, wlo) = local.whites.extremes(fhi, flo, between);
        let (bhi, blo) = local.blacks.extremes(whi, wlo, between);
        let (thi, tlo) = (local.tone_of(bhi), local.tone_of(blo));
        let toned = scene.map(|v| if hi > lo { tlo + (thi - tlo) * (v - lo) / (hi - lo) } else { thi });
        let m = self.geometry.work_to_out.map(|r| r.map(|x| x as f64));
        apply(&m, toned.map(|x| x as f64)).map(|v| srgb_encode((v as f32).clamp(0.0, 1.0)))
    }

    /// Develop one pixel on the CPU: camera-linear RGB in, sRGB-encoded
    /// Rec.709 out. The GPU stage computes the same. With Shadows /
    /// Highlights the pixel's own level stands in for its base (see
    /// [`Self::develop_local`] for the picture's).
    pub fn develop(&self, camera: [f32; 3]) -> [f32; 3] {
        if self.local.is_some() {
            return self.develop_local(camera, self.base_luminance(camera).log2());
        }
        let scene = self.geometry.scene(camera).map(|v| (v * self.gain).max(0.0));
        // Hue-keeping tone: the extremes through the curve, the middle
        // channel at its old place between them.
        let hi = scene.iter().cloned().fold(f32::MIN, f32::max);
        let lo = scene.iter().cloned().fold(f32::MAX, f32::min);
        let (thi, tlo) = (self.tone(hi), self.tone(lo));
        let toned = scene.map(|v| if hi > lo { tlo + (thi - tlo) * (v - lo) / (hi - lo) } else { thi });
        let m = self.geometry.work_to_out.map(|r| r.map(|x| x as f64));
        apply(&m, toned.map(|x| x as f64)).map(|v| srgb_encode((v as f32).clamp(0.0, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x6(settings: RawSettings) -> RawDevelop {
        RawDevelop { camera: RawCamera::X6, settings, measure: Measure::Wait }
    }

    #[test]
    fn the_default_rendering_is_the_measured_one() {
        // The default curve at nodes of the public table it reproduces.
        for (node, want) in [(8, 0.00623), (16, 0.01485), (64, 0.10433), (128, 0.25961), (256, 0.52069), (1024, 1.0)] {
            let got = default_curve(node as f64 / 1024.0);
            assert!((got - want).abs() < 4e-4, "node {node}: {got}");
        }
        // The composed default against the measuring model's own numbers
        // (scene value, white point, dark key → output light).
        let reference = [
            (0.18, 1.0, -14.0, 0.381005),
            (0.004, 1.0, -14.0, 0.002629),
            (0.004, 1.0, -9.0, 0.001791),
            (0.004, 1.0, -5.0, 0.000000),
            (0.02, 1.0, -5.0, 0.014140),
            (0.5, 2.0, -14.0, 0.773778),
            (1.0, 2.49, -11.3, 0.927975),
            (2.0, 4.0, -14.0, 0.983006),
            (3.0, 9.96, -8.25, 0.990551),
            (0.02, 2.49, -10.5, 0.018183),
            (0.0001, 1.0, -14.0, 0.000031),
        ];
        for (x, white, key, want) in reference {
            let got = default_render(x, white, key);
            assert!((got - want).abs() < 2e-5, "x {x} white {white} key {key}: {got} vs {want}");
        }
        // A bright frame's default black is the plain black point of 0.005.
        for x in [0.01, 0.1, 0.5] {
            assert!((default_black(x, -6.0) - (x - 0.005) / 0.995).abs() < 1e-6);
        }
        // The white point: nothing below a quarter, the white itself at 1, rising.
        for white in [1.0, 1.3, 2.49, 4.0, 9.0, 60.0, 500.0] {
            assert_eq!(default_white(0.2, white), 0.2);
            assert!((default_white(white, white) - 1.0).abs() < 1e-6, "{white}");
            let mut last = 0.0;
            for i in 0..=200 {
                let y = default_white(white * i as f64 / 200.0, white);
                assert!(y >= last - 1e-9 && y <= 1.0 + 1e-9, "white {white} at {i}: {y} after {last}");
                last = y;
            }
        }
    }

    #[test]
    fn keys_of_a_reduced_copy() {
        assert!(ToneKeys::from_cells(&[]).is_none());
        // 2000 cells from 0.001 to 0.2, one with a bright red channel.
        let mut cells: Vec<[f32; 3]> = (0..2000).map(|i| [0.001 + 0.0001 * i as f32; 3]).collect();
        cells[77] = [1.4, 0.3, 0.2];
        let k = ToneKeys::from_cells(&cells).unwrap();
        assert_eq!(k.max, 1.4);
        // The darkest cell.
        assert!((k.dark - 0.001).abs() < 1e-6, "{}", k.dark);
    }

    #[test]
    fn a_neutral_stays_neutral_through_development() {
        let dev = x6(RawSettings::default());
        // A frame with highlights up to the sensor's white.
        let k = dev.constants(ToneKeys { max: 2.49, dark: 0.0 });
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        // Greys of the as-shot white, from deep shadow up through the level
        // where the first channel clips to the one where the last one does.
        let mut last = -1.0;
        for level in [0.001f32, 0.01, 0.05, 0.18, 0.5, 0.9, 1.0, 1.3, 1.7, 2.2, 2.49] {
            let camera = n.map(|v| (v * level).min(1.0));
            let out = k.develop(camera);
            assert!((out[0] - out[1]).abs() < 6e-3 && (out[2] - out[1]).abs() < 6e-3, "level {level}: {out:?}");
            assert!(out[1] > last, "not rising at {level}: {} after {last}", out[1]);
            last = out[1];
        }
        // The sensor's clip point (every channel at 1) is white.
        let white = k.develop([1.0, 1.0, 1.0]);
        assert!(white.iter().all(|v| (v - 1.0).abs() < 2e-3), "{white:?}");
        // The keys in use: the sensor's white is the white point.
        assert!((k.white_point - 1.0 / 0.4013671875).abs() < 1e-3 && k.dark_key < -20.0, "{k:?}");
        // A dim frame has no shoulder at all.
        let dim = dev.constants(ToneKeys { max: 0.2, dark: 0.001 });
        assert_eq!(dim.white_point, 1.0);
        assert!((dim.dark_key - 0.001f32.log2()).abs() < 1e-4);
        // One stop more exposure is twice the light before the curve, and
        // the sensor's white moves up with it.
        let brighter = x6(RawSettings { exposure: 1.0, ..Default::default() }).constants(ToneKeys { max: 0.1, dark: 0.0 });
        let plain = dev.constants(ToneKeys { max: 0.2, dark: 0.0 });
        let a = brighter.develop(n.map(|v| v * 0.05));
        let b = plain.develop(n.map(|v| v * 0.1));
        assert!((a[1] - b[1]).abs() < 1e-4, "{a:?} {b:?}");
        // A warmer white balance setting makes the same pixel bluer.
        let warm = x6(RawSettings { temperature: Some(9000.0), ..Default::default() }).constants(ToneKeys::UNKNOWN);
        let cool = x6(RawSettings { temperature: Some(3000.0), ..Default::default() }).constants(ToneKeys::UNKNOWN);
        let px = n.map(|v| v * 0.2);
        let (w, c) = (warm.develop(px), cool.develop(px));
        assert!(w[0] > w[2] && c[2] > c[0], "9000 K {w:?}, 3000 K {c:?}");
    }

    #[test]
    fn the_sampled_curve_follows_the_rendering() {
        let k = x6(RawSettings::default()).constants(ToneKeys { max: 1.0, dark: 0.002 });
        for x in [2e-5f32, 3e-4, 0.004, 0.05, 0.18, 0.7, 1.0, 1.9, 2.4] {
            let want = default_render(x as f64, k.white_point as f64, k.dark_key as f64) as f32;
            assert!((k.tone(x) - want).abs() < 2e-4, "x {x}: {} vs {want}", k.tone(x));
        }
        assert_eq!(k.tone(0.0), 0.0);
        assert!((k.tone(50.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_gain_is_limited_toward_white() {
        // Taken whole up to half the room, then bent; never past the room.
        assert_eq!(limit_gain(0.4, 1.0), 0.4);
        assert_eq!(limit_gain(0.5, 1.0), 0.5);
        assert!((limit_gain(1.0, 1.0) - 0.75).abs() < 1e-6 && (limit_gain(2.0, 1.0) - 0.875).abs() < 1e-6);
        assert!((limit_gain(100.0, 1.3) - 1.3).abs() < 0.01 && limit_gain(100.0, 1.3) < 1.3);
        // Nothing is left at white; gains at or below zero pass.
        assert_eq!(limit_gain(1.0, 0.0), 0.0);
        assert_eq!(limit_gain(1.0, -0.5), 0.0);
        assert_eq!((limit_gain(-2.0, 0.3), limit_gain(0.0, 0.0)), (-2.0, 0.0));
    }

    /// A frame whose levels are spread evenly from 2⁻⁹ to `top` stops, grey.
    fn spread_frame(top: f32) -> FrameStats {
        let levels: Vec<f32> = (0..441).map(|i| -9.0 + (top + 9.0) * i as f32 / 440.0).collect();
        let mut peaks = vec![0f32; LOCAL_SAMPLES];
        for l in &levels {
            peaks[((l - LOCAL_LO) * LOCAL_PER_STOP).round() as usize] = 2f32.powf(*l);
        }
        FrameStats { keys: ToneKeys { max: 2f32.powf(top), dark: 2f32.powf(-9.0) }, levels: levels.into(), peaks: peaks.into() }
    }

    #[test]
    fn the_white_point_follows_the_gained_frame() {
        let stats = spread_frame(0.0);
        let stage = |exposure: f32, highlights: f32, shadows: f32| x6(RawSettings { exposure, highlights, shadows, ..Default::default() });
        // Highlights curves by hand: a stop down (or up) from two stops below the top.
        let level = |i: usize| LOCAL_LO + i as f32 / LOCAL_PER_STOP;
        let ramp = |sign: f32, top: f32| -> Vec<f32> { (0..LOCAL_SAMPLES).map(|i| sign * ((level(i).min(top) - (top - 2.0)) / 2.0).clamp(0.0, 1.0)).collect() };
        let curves = |top: f32| LocalCurves { shadows: vec![0.0; LOCAL_SAMPLES], highlights: ramp(-1.0, top), highlights_up: ramp(1.0, top), tilt_top: 0.0, tilt_bottom: 0.0, shift: 0.0, top, bottom: top - 9.0 };
        // Three stops of exposure: the frame's top is at 8, the white point
        // at twice that (the sensor's white is further up, at 19.9).
        let plain = stage(3.0, 0.0, 0.0).constants(stats.keys);
        assert!((plain.white_point - 16.0).abs() < 0.01, "{}", plain.white_point);
        // Highlights −100 takes the top down a stop: twice the gained top.
        let down = stage(3.0, -100.0, 0.0).constants_for(&stats, Some(&curves(3.0)));
        let local = down.local.as_ref().unwrap();
        assert!((local.white - 8.0).abs() < 0.05 && (local.fed_hi - 3.0).abs() < 0.01, "{}", local.white);
        // The stage: black, gain, that white point, the curve.
        for x in [0.02f32, 0.3, 2.0, 7.9] {
            let base = x.log2();
            let g = 2f64.powf(local.gain_at(base) as f64);
            let want = default_curve(default_white(g * default_black(x as f64, down.dark_key as f64), local.white as f64).clamp(0.0, 1.0)) as f32;
            let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
            let got = down.develop_local(n.map(|v| v * x / down.gain), base);
            // A grey: every channel the tone of its level (sRGB-encoded).
            let enc = if want <= 0.003_130_8 { 12.92 * want } else { 1.055 * want.powf(1.0 / 2.4) - 0.055 };
            assert!((got[1] - enc).abs() < 3e-3, "x {x}: {} vs {enc}", got[1]);
        }
        // Highlights +100 at exposure 0: the top would go a stop past the
        // sensor's first clipped channel; twice the gained top is past the
        // sensor's white (2.49), which is then the white point, and the
        // gain is limited to the room below it.
        let up = stage(0.0, 100.0, 0.0).constants_for(&stats, Some(&curves(0.0)));
        let local = up.local.as_ref().unwrap();
        let sensor = 1.0 / 0.4013671875f32;
        assert!((local.white - sensor).abs() < 0.01, "{}", local.white);
        // At the frame's top: one stop wanted, 1.32 stops of room.
        let top_gain = local.gain_of(0.0, 1.0).log2();
        assert!((top_gain - limit_gain(1.0, sensor.log2())).abs() < 0.01 && top_gain < 0.9, "{top_gain}");
        // The curves the frame gets by itself are of its own range.
        let own = stage(0.0, 0.0, 60.0).constants_for(&stats, None);
        let g = &own.local.as_ref().unwrap().gain;
        assert!(g[((-7.0 - LOCAL_LO) * LOCAL_PER_STOP) as usize] > 0.8 && g[((-0.5 - LOCAL_LO) * LOCAL_PER_STOP) as usize].abs() < 0.2);
        // Nothing measured: the plain white point.
        let blind = stage(3.0, -100.0, 0.0).constants_for(&FrameStats::unknown(), Some(&curves(3.0)));
        assert_eq!(blind.local.as_ref().unwrap().white, blind.white_point);
    }

    #[test]
    fn whites_and_blacks_after_the_white_point() {
        let stats = spread_frame(0.0);
        let with = |settings: RawSettings| x6(settings).constants_for(&stats, None);
        let plain = with(RawSettings::default());
        assert!(plain.local.is_none() && plain.floor == 0.0);
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        // A grey of scene value `x` developed: its green.
        let grey = |k: &DevelopConstants, x: f32| k.develop_local(n.map(|v| v * x), x.max(1e-9).log2())[1];
        // Blacks +100 on a dark frame: black itself is lifted, the shadows
        // with it, white not at all.
        let up = with(RawSettings { blacks: 100.0, ..Default::default() });
        let local = up.local.as_ref().unwrap();
        assert!(local.gain.iter().all(|g| *g == 0.0) && local.blacks.keyed && local.blacks.floor > 1e-4 && local.blacks.at(0.0) == local.blacks.floor);
        assert!(grey(&up, 0.0) > 0.01 && grey(&up, 0.02) > grey(&plain, 0.02) + 0.05 && grey(&up, 0.18) > grey(&plain, 0.18));
        assert!((grey(&up, up.white_point) - 1.0).abs() < 3e-3);
        // Blacks −100: the deep shadows are black, the rest darker.
        let down = with(RawSettings { blacks: -100.0, ..Default::default() });
        assert!(!down.local.as_ref().unwrap().blacks.keyed);
        assert!(grey(&down, 0.004) < 1e-6 && grey(&down, 0.4) < grey(&plain, 0.4) && grey(&down, 0.4) > 0.5 * grey(&plain, 0.4), "{} {} {}", grey(&down, 0.004), grey(&down, 0.4), grey(&plain, 0.4));
        // Whites −100: the upper tones come down, the shadows stay.
        let dim = with(RawSettings { whites: -100.0, ..Default::default() });
        assert!(dim.local.as_ref().unwrap().whites.keyed);
        assert!((grey(&dim, 0.01) - grey(&plain, 0.01)).abs() < 2e-3 && grey(&dim, 0.5) < grey(&plain, 0.5) - 0.03);
        // Together: Whites, then Blacks, each a map of the encoded output.
        let both = with(RawSettings { whites: -100.0, blacks: -100.0, ..Default::default() });
        let (w, b) = (crate::raw_global::whites_map(-100.0).unwrap(), crate::raw_global::blacks_map(-100.0, -9.0).unwrap());
        for x in [0.05f32, 0.2, 0.6] {
            let want = crate::raw_global::apply(&b, crate::raw_global::apply(&w, grey(&plain, x)));
            assert!((grey(&both, x) - want).abs() < 3e-3, "x {x}: {} vs {want}", grey(&both, x));
        }
        // A keyed stage keeps a colour's channel ratios: twice the light in
        // a channel stays twice; a curve stage does not.
        let keyed = &dim.local.as_ref().unwrap().whites;
        let (khi, klo) = keyed.extremes(0.4, 0.2, 0.5);
        assert!(khi < 0.4 && (khi / klo - 2.0).abs() < 1e-4, "{khi} {klo}");
        let curve = &down.local.as_ref().unwrap().blacks;
        let (chi, clo) = curve.extremes(0.4, 0.2, 0.5);
        assert!(chi / clo > 2.05 && (chi, clo) == (curve.at(0.4), curve.at(0.2)), "{chi} {clo}");
        // With Shadows the gain lifts the frame's darkest level, and Blacks
        // go by the lifted one.
        let lifted = with(RawSettings { blacks: 100.0, shadows: 50.0, ..Default::default() });
        assert!(lifted.local.as_ref().unwrap().blacks.floor != up.local.as_ref().unwrap().blacks.floor);
    }

    #[test]
    fn contrast_and_whites_above_zero() {
        let stats = spread_frame(0.0);
        let with = |settings: RawSettings| x6(settings).constants_for(&stats, None);
        let plain = with(RawSettings::default());
        // Contrast alone is a tone curve of the frame's own: no stage.
        let hard = with(RawSettings { contrast: 100.0, ..Default::default() });
        assert!(hard.local.is_none() && hard.curve != plain.curve);
        // Shadows darker, highlights brighter, the two ends where they were.
        assert!(hard.tone(0.01) < 0.7 * plain.tone(0.01) && hard.tone(0.7) > plain.tone(0.7) + 0.02, "{} {} / {} {}", hard.tone(0.01), plain.tone(0.01), hard.tone(0.7), plain.tone(0.7));
        assert!(hard.tone(0.0) == 0.0 && (hard.tone(hard.white_point) - 1.0).abs() < 1e-3);
        let soft = with(RawSettings { contrast: -100.0, ..Default::default() });
        assert!(soft.tone(0.01) > 1.3 * plain.tone(0.01) && soft.tone(0.7) < plain.tone(0.7));
        // With Shadows the stage's tone curve carries it.
        let both = with(RawSettings { contrast: 100.0, shadows: 30.0, ..Default::default() });
        let local = both.local.as_ref().unwrap();
        assert!((local.tone_of(0.05) - default_curve(0.05) as f32).abs() > 0.01);
        // Whites above zero: a curve of the fed value, brighter everywhere
        // and most at the top; nothing passes white.
        let bright = with(RawSettings { whites: 60.0, ..Default::default() });
        let w = &bright.local.as_ref().unwrap().whites;
        assert!(!w.keyed && w.floor == 0.0 && w.at(0.02) > 0.02 && w.at(0.02) < 0.03 && w.at(0.4) > 0.5 && w.at(0.99) <= 1.0, "{} {}", w.at(0.02), w.at(0.4));
        // A frame whose top is lower gets more of it.
        let low = x6(RawSettings { whites: 60.0, ..Default::default() }).constants_for(&spread_frame(-3.0), None);
        assert!(low.local.as_ref().unwrap().whites.at(0.05) > w.at(0.05) * 1.2);
        // Nothing known of the frame: Whites above zero do nothing.
        let blind = x6(RawSettings { whites: 60.0, ..Default::default() }).constants(ToneKeys { max: 1.0, dark: 0.002 });
        assert!((blind.local.as_ref().unwrap().whites.at(0.3) - 0.3).abs() < 1e-4);
    }

    #[test]
    fn settings_read_from_older_files() {
        let s: RawSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, RawSettings::default());
        // A file from before the colour noise reduction gets a raw converter's default.
        assert_eq!(s.color_noise, 25.0);
        let s: RawSettings = serde_json::from_str(r#"{"color_noise":0.0}"#).unwrap();
        assert_eq!(s.color_noise, 0.0);
        let s: RawSettings = serde_json::from_str(r#"{"exposure":1.5,"temperature":5200.0}"#).unwrap();
        assert_eq!((s.exposure, s.temperature, s.tint), (1.5, Some(5200.0), None));
    }
}
