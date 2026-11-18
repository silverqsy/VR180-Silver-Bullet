//! Tone controls in linear light for a source that is not raw: Exposure,
//! Contrast, Highlights, Shadows, Whites and Blacks, the controls a raw
//! clip has in its development ([`crate::raw_tone`]), for every clip.
//!
//! A video clip arrives as code values (a log curve's, or a display
//! gamma's) and is shown through what follows in the colour stack: the CDL
//! and, for log footage, a 3D LUT that also rolls the highlights off. A tone
//! control applied to the codes, or to the picture after the LUT, works on
//! whatever those curves left. Here every control is a change of exposure in
//! LINEAR LIGHT, made ahead of that rendering:
//!
//! ```text
//! codes → light → × exposure → Shadows / Highlights (a gain that follows
//!   the picture's base) → Whites → Blacks → Contrast → codes
//! ```
//!
//! so a highlight the LUT pressed against white comes back from what the
//! file holds when it is pulled down, and the LUT then renders it where it
//! has room.
//!
//! **Which light.** Where the camera's log curve is known
//! ([`ToneLight::Scene`]) the codes are decoded with it: the light is the
//! scene's, a stop is a stop of exposure, and what lies beyond the LUT's
//! shoulder is all there. For any other clip ([`ToneLight::Shown`]) the
//! light is the one the picture SHOWS: a code is taken through the
//! rendering's own neutral curve ([`Shown`]) and decoded as sRGB, and the
//! result is put back through that curve's inverse. A neutral pixel is then
//! changed exactly as its displayed light says, whatever the LUT.
//!
//! **Shadows and Highlights** are the raw development's: a gain in stops
//! read off a curve at the pixel's base level, the base being the frame's
//! log₂ luminance smoothed without crossing strong edges
//! ([`crate::raw_local`]), the curves derived from the frame's own levels.
//! A gain above zero is limited toward white as there.
//!
//! **Whites, Blacks and Contrast** are, as in a raw converter, curves of
//! the picture as it is displayed ([`crate::raw_global`]). They become
//! changes of light through the rendering's neutral curve: a value `x`
//! shows as `e`, the control wants `m(e)` shown, so `x` goes to the light
//! that shows as `m(e)`. Where the rendering is flat (above a LUT's clip
//! point, below its black) there is no such light to be told apart, and the
//! pixel keeps its ratio to the clip point (its distance from black) instead.
//!
//! With every control at zero the stage is not run at all.

use crate::raw_tone::{
    default_curve, default_curve_inverse, limit_gain, CurveSamples, FrameStats, LocalCurves, Measure,
    LOCAL_LO, LOCAL_PER_STOP, LOCAL_SAMPLES, OUT_LO,
};
use vr180_core::LogCurve;

/// The tone controls of a clip. `Default`: nothing moved.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ToneSettings {
    /// Exposure in stops.
    pub exposure: f32,
    /// −100..100.
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
}

impl ToneSettings {
    /// Nothing is moved.
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// The settings move Shadows or Highlights: the frame needs its base.
    pub fn is_local(&self) -> bool {
        self.shadows != 0.0 || self.highlights != 0.0
    }

    /// The settings need the frame measured: everything but Exposure and
    /// Whites below zero follows the frame's own levels.
    pub fn needs_stats(&self) -> bool {
        self.is_local() || self.contrast != 0.0 || self.blacks != 0.0 || self.whites > 0.0
    }
}

/// The light the stage works in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneLight {
    /// The scene's: the codes are the camera's log curve.
    Scene(LogCurve),
    /// The light the picture shows: the code through the rendering's
    /// neutral curve, decoded as sRGB.
    Shown,
}

/// The stage of a colour stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearTone {
    pub settings: ToneSettings,
    pub light: ToneLight,
    /// Exposure the stage makes besides the user's, in stops: what a clip
    /// shot bright on purpose is brought back by (negative), in place of
    /// the stack's own stage for it.
    pub pre_stops: f32,
    pub measure: Measure,
}

impl LinearTone {
    /// The stage changes nothing.
    pub fn is_identity(&self) -> bool {
        self.settings.is_identity() && self.pre_stops == 0.0
    }
}

/// Nodes of a [`Shown`] curve and of a decode table: a multiple of the
/// lattices a LUT comes in (17, 33, 65 points), so their kinks fall on nodes.
pub const CODE_SAMPLES: usize = 4097;

/// The neutral curve of what renders the picture after the stage: the
/// displayed (encoded) value of a grey of each code, [`CODE_SAMPLES`]
/// equally spaced codes from 0 to 1, never falling.
#[derive(Debug, Clone, PartialEq)]
pub struct Shown {
    values: Vec<f32>,
    /// The last node that shows what the darkest code shows, and the first
    /// that shows what the brightest does: where a level stretch at either
    /// end is entered from inside.
    dark_end: usize,
    bright_start: usize,
}

impl Shown {
    /// A picture shown as its codes are.
    pub fn identity() -> Self {
        Self::of(|code| code)
    }

    /// The curve of `render`, a grey's code to what is displayed. A
    /// rendering that falls somewhere (a creative LUT may) is held level
    /// there.
    pub fn of(render: impl Fn(f32) -> f32) -> Self {
        let mut top = 0f32;
        let values: Vec<f32> = (0..CODE_SAMPLES).map(|i| {
            top = top.max(render(i as f32 / (CODE_SAMPLES - 1) as f32).clamp(0.0, 1.0));
            top
        }).collect();
        let last = CODE_SAMPLES - 1;
        let dark_end = values.iter().rposition(|x| *x <= values[0]).unwrap_or(0);
        let bright_start = values.iter().position(|x| *x >= values[last]).unwrap_or(last);
        Self { values, dark_end, bright_start }
    }

    /// What a grey of `code` is displayed as.
    pub fn at(&self, code: f64) -> f64 {
        let last = (CODE_SAMPLES - 1) as f64;
        let y = code.clamp(0.0, 1.0) * last;
        let i = (y.floor() as usize).min(CODE_SAMPLES - 2);
        let f = y - i as f64;
        self.values[i] as f64 * (1.0 - f) + self.values[i + 1] as f64 * f
    }

    /// The code displayed as `e`. Below what the darkest code shows: the
    /// last code that still shows that; above what the brightest shows: the
    /// first that shows it (so a level stretch is entered from the side the
    /// picture comes from).
    pub fn code_of(&self, e: f64) -> f64 {
        let v = &self.values;
        let last = CODE_SAMPLES - 1;
        let step = 1.0 / last as f64;
        if e <= v[0] as f64 {
            return self.dark_end as f64 * step;
        }
        if e >= v[last] as f64 {
            return self.bright_start as f64 * step;
        }
        // The first node at or above `e`: the value is reached between it
        // and the node before, which is below.
        let hi = v.partition_point(|x| (*x as f64) < e);
        let (a, b) = (v[hi - 1] as f64, v[hi] as f64);
        ((hi - 1) as f64 + (e - a) / (b - a)) * step
    }
}

fn srgb_encode(x: f64) -> f64 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_decode(e: f64) -> f64 {
    if e <= 0.040_45 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) }
}

/// The stage's light and the rendering together: how a code becomes light
/// (1 = the brightest a code can say), how light becomes a code again, and
/// what a light is shown as.
struct Light<'a> {
    light: ToneLight,
    shown: &'a Shown,
    /// Scene light of code 1.
    white: f64,
}

impl<'a> Light<'a> {
    fn new(light: ToneLight, shown: &'a Shown) -> Self {
        let white = match light {
            ToneLight::Scene(curve) => curve.decode(1.0).max(1e-6),
            ToneLight::Shown => 1.0,
        };
        Self { light, shown, white }
    }

    /// Code → light.
    fn decode(&self, code: f64) -> f64 {
        match self.light {
            ToneLight::Scene(curve) => curve.decode(code) / self.white,
            ToneLight::Shown => srgb_decode(self.shown.at(code)),
        }
    }

    /// Light → code.
    fn encode(&self, x: f64) -> f64 {
        match self.light {
            ToneLight::Scene(curve) => curve.encode(x.max(0.0) * self.white).clamp(0.0, 1.0),
            ToneLight::Shown => self.shown.code_of(srgb_encode(x.clamp(0.0, 1.0))),
        }
    }

    /// What a light is displayed as.
    fn show(&self, x: f64) -> f64 {
        match self.light {
            ToneLight::Scene(_) => self.shown.at(self.encode(x)),
            ToneLight::Shown => srgb_encode(x.clamp(0.0, 1.0)),
        }
    }

    /// The light displayed as `e`.
    fn light_of(&self, e: f64) -> f64 {
        match self.light {
            ToneLight::Scene(_) => self.decode(self.shown.code_of(e)).max(0.0),
            ToneLight::Shown => srgb_decode(e.clamp(0.0, 1.0)),
        }
    }

    /// A map `m` of the displayed value as a change of light: `x` goes to
    /// the light that shows as `m` of what `x` shows as. Where the
    /// rendering is level at `x`, the light that shows the same is not `x`
    /// itself: the pixel then keeps its ratio to that light, and next to
    /// black its distance from it.
    fn through(&self, x: f64, m: impl Fn(f64) -> f64) -> f64 {
        let e = self.show(x);
        let (from, to) = (self.light_of(e), self.light_of(m(e).clamp(0.0, 1.0)));
        if from > 1e-9 { to + (x - from) * (to / from) } else { to + (x - from) }
    }

    /// The level (log₂) Camera Raw's default rendering would have been fed
    /// to show a light as it is shown here: what its curves of the frame's
    /// levels are read at.
    fn equivalent_level(&self, x: f64) -> f32 {
        default_curve_inverse(srgb_decode(self.show(x))).max(2f64.powi(-30)).log2() as f32
    }
}

/// Whites, Blacks or Contrast for one frame, as what a light becomes.
///
/// Kept as the CHANGE of the light (what it becomes less what it is), so
/// that a control at zero, and every light a control leaves where it is,
/// passes exactly: a table of the lights themselves would return each one
/// a few parts in a million off, between its samples.
///
/// As in the raw development ([`crate::raw_tone::StageMap`]), a control
/// acts on colour in one of two ways: a gain keyed to the pixel's luminance
/// that keeps its channels' ratios, or a curve its largest and smallest
/// channel go through.
#[derive(Debug, Clone, PartialEq)]
pub struct LightMap {
    /// The change of a light, over its log₂ from [`OUT_LO`] to 0
    /// ([`MAP_SAMPLES`] samples).
    pub change: CurveSamples,
    /// The change of no light: what black is lifted to.
    pub floor: f32,
    /// A gain keyed to the pixel's luminance (else a curve per extreme).
    pub keyed: bool,
}

impl LightMap {
    /// What the light `v` becomes. Under the first sample the change runs
    /// in a straight line to what no light becomes, and on along that line
    /// for a light below none (a code under the curve's black: noise), so
    /// what lies as far under black as over it stays so.
    pub fn at(&self, v: f32) -> f32 {
        let lo = OUT_LO as f32;
        let first = 2f32.powf(lo);
        let change = if v <= first {
            self.floor + (self.change[0] - self.floor) * v / first
        } else {
            sample_curve(&self.change, (v.log2() - lo) / -lo * (self.change.len() - 1) as f32)
        };
        v + change
    }

    /// The largest and the smallest channel of a pixel through the stage;
    /// `between` = where its luminance lies between them (0..1).
    pub fn extremes(&self, hi: f32, lo: f32, between: f32) -> (f32, f32) {
        if !self.keyed {
            return (self.at(hi), self.at(lo));
        }
        // The luminance goes where the map sends it, the channels keep
        // their ratios to it (next to black: their differences from it,
        // and below it those differences as they are).
        let y = lo + (hi - lo) * between;
        let to = self.at(y);
        if y <= 0.0 {
            return (hi + (to - y), lo + (to - y));
        }
        let over = y.max(2f32.powf(OUT_LO as f32));
        let (scale, add) = (to / over, to * (1.0 - y / over));
        (scale * hi + add, scale * lo + add)
    }
}

/// The per-frame constants of the stage: the tables the GPU stage
/// (`shaders/tone_linear.wgsl`) and its CPU twin ([`Self::apply`]) run on.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneConstants {
    /// Light over the code: [`CODE_SAMPLES`] equally spaced codes.
    pub decode: CurveSamples,
    /// The exposure, as a factor of light.
    pub gain: f32,
    /// Shadows / Highlights: log₂ gain over the base level, before the
    /// white limiter ([`LOCAL_SAMPLES`] samples from [`LOCAL_LO`],
    /// [`LOCAL_PER_STOP`] per stop). `None`: neither is moved, and the
    /// frame needs no base.
    pub local: Option<CurveSamples>,
    /// Whites, then Blacks, then Contrast.
    pub whites: LightMap,
    pub blacks: LightMap,
    pub contrast: LightMap,
    /// The code of a light, over the light's sRGB encoding from 0 to 1
    /// ([`ENCODE_SAMPLES`] equally spaced samples).
    pub encode: CurveSamples,
    /// Codes to a unit of light below none ([`LightTables::below`]).
    pub below: f32,
    /// The long side, in pixels, of the frame the eyes belong to (two eyes
    /// side by side count as one frame); 0 = the eye's own.
    pub long_side: u32,
}

/// The weights of a light's luminance (Rec.709's: what the picture is
/// shown in).
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn sample_curve(curve: &[f32], at: f32) -> f32 {
    let last = (curve.len() - 1) as f32;
    let y = at.clamp(0.0, last);
    let i = (y.floor() as usize).min(curve.len() - 2);
    let f = y - i as f32;
    curve[i] * (1.0 - f) + curve[i + 1] * f
}

/// A sampled map of the displayed value as a function of it.
fn table_map(map: &Option<Vec<f32>>) -> Option<impl Fn(f64) -> f64 + '_> {
    map.as_ref().map(|m| move |e: f64| crate::raw_global::apply(m, e as f32) as f64)
}

/// Samples of the table that turns light back into codes. It is laid out
/// over the sRGB encoding of the light, not over the light or its
/// logarithm: the code of a light through the inverse of a LUT's shoulder
/// is steep and bent just below white, and that is where an encoding made
/// for display spends its values evenly. A pixel the stage leaves alone
/// then comes out as the code it went in as, to a few parts in a million.
pub const ENCODE_SAMPLES: usize = 16385;

/// Samples of a [`LightMap`]: 57 to the stop. The maps are made again for
/// every frame, and what they hold bends slowly (a LUT's lattice puts a
/// kink in it every tenth of a stop or so at the closest).
pub const MAP_SAMPLES: usize = 1025;

/// The light of sample `i` of a table of `n` samples over log₂ from
/// [`OUT_LO`] to 0.
fn table_light(i: usize, n: usize) -> f64 {
    2f64.powf(OUT_LO * (1.0 - i as f64 / (n - 1) as f64))
}

/// The two tables of a stage that do not depend on the frame: light over
/// the code and the code over the light, for one light and one rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct LightTables {
    /// Light over the code: [`CODE_SAMPLES`] equally spaced codes.
    pub decode: CurveSamples,
    /// The code of a light, over the light's sRGB encoding from 0 to 1
    /// ([`ENCODE_SAMPLES`] equally spaced samples).
    pub encode: CurveSamples,
    /// Codes to a unit of light below none. A log curve's straight toe runs
    /// on under its black, where a clip's noise lies: such a code is light
    /// below none, and goes back to a code along the same line, so the
    /// noise about black keeps its mean. 0 where no code says less than no
    /// light.
    pub below: f32,
}

impl LightTables {
    pub fn of(light: ToneLight, shown: &Shown) -> Self {
        let light = Light::new(light, shown);
        let decode: CurveSamples = (0..CODE_SAMPLES).map(|i| light.decode(i as f64 / (CODE_SAMPLES - 1) as f64) as f32).collect();
        // The toe's slope, off its first two codes.
        let step = (decode[1] - decode[0]) * (CODE_SAMPLES - 1) as f32;
        let below = if decode[0] < 0.0 && step > 0.0 { 1.0 / step } else { 0.0 };
        Self {
            decode,
            encode: (0..ENCODE_SAMPLES).map(|i| light.encode(srgb_decode(i as f64 / (ENCODE_SAMPLES - 1) as f64)) as f32).collect(),
            below,
        }
    }
}

impl LinearTone {
    /// Everything the stage needs for one frame, the two tables of its
    /// light worked out here: see [`Self::constants_with`].
    pub fn constants_for(&self, stats: &FrameStats, shown: &Shown, curves: Option<&LocalCurves>) -> ToneConstants {
        self.constants_with(&LightTables::of(self.light, shown), stats, shown, curves)
    }

    /// Everything the stage needs for one frame: `tables` of this stage's
    /// light and `shown` ([`LightTables::of`]; they are the same for every
    /// frame of a clip while its grade stands), `stats` measured on the
    /// frame in the stage's light before the exposure (see
    /// [`crate::gpu::Device::linear_tone_constants`]; `FrameStats::unknown`
    /// where the settings need none), `shown` the rendering that follows,
    /// `curves` in place of the Shadows / Highlights curves derived from
    /// the frame.
    pub fn constants_with(&self, tables: &LightTables, stats: &FrameStats, shown: &Shown, curves: Option<&LocalCurves>) -> ToneConstants {
        let s = &self.settings;
        let light = Light::new(self.light, shown);
        let gain = 2f64.powf((s.exposure + self.pre_stops) as f64);
        let shift = gain.log2() as f32;
        // Shadows / Highlights: the gain over the base level. The frame's
        // top was that far below white when it was recorded.
        let local: Option<Vec<f32>> = s.is_local().then(|| {
            let derived;
            let curves = match curves {
                Some(c) => c,
                None => {
                    let levels: Vec<f32> = stats.levels.iter().map(|l| l + shift).collect();
                    let headroom = stats.levels.last().map_or(0.0, |top| (-top).max(0.0));
                    derived = crate::raw_local::local_curves(&levels, headroom);
                    &derived
                }
            };
            curves.gain(s.shadows, s.highlights)
        });
        // A level of the frame (log₂, as measured) as the stages after
        // Shadows / Highlights meet it: exposed, with the gain of its own
        // level as its base would have it.
        let gained = |level: f32| -> f64 {
            let exposed = level + shift;
            let stops = local.as_ref().map_or(0.0, |g| limit_gain(sample_curve(g, (exposed - LOCAL_LO) * LOCAL_PER_STOP), -exposed));
            2f64.powf((exposed + stops) as f64)
        };
        let sorted = |mut v: Vec<f32>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v };
        // The maps of the displayed value, each for the frame as it is
        // shown: Contrast by the spread of its levels as recorded, Blacks
        // by its darkest level and Whites above zero by its levels and its
        // top as they come to them.
        let whites_map: Option<Box<dyn Fn(f64) -> f64>> = if s.whites > 0.0 {
            let levels = sorted(stats.levels.iter().map(|l| light.equivalent_level(gained(*l))).collect());
            levels.last().copied().and_then(|top| crate::raw_global::whites_up(s.whites, &levels, top)).map(|(a1, a2)| {
                Box::new(move |e: f64| {
                    let fed = default_curve_inverse(srgb_decode(e));
                    srgb_encode(default_curve((a1 as f64 * fed + a2 as f64 * fed * fed).min(1.0)))
                }) as Box<dyn Fn(f64) -> f64>
            })
        } else {
            crate::raw_global::whites_map(s.whites).map(|m| Box::new(move |e: f64| crate::raw_global::apply(&m, e as f32) as f64) as Box<dyn Fn(f64) -> f64>)
        };
        let dark = light.equivalent_level(gained((stats.keys.dark as f64).max(2f64.powi(-30)).log2() as f32));
        let blacks_map = crate::raw_global::blacks_map(s.blacks, dark);
        let contrast_map = {
            let levels = sorted(stats.levels.iter().map(|l| light.equivalent_level(2f64.powf(*l as f64))).collect());
            crate::raw_global::contrast_map(s.contrast, &levels)
        };
        // A map as what a light becomes, sampled over the light.
        let stage = |m: Option<&dyn Fn(f64) -> f64>, keyed: bool| -> LightMap {
            let change = |x: f64| -> f64 {
                match m {
                    Some(m) => light.through(x, m).clamp(0.0, 1.0) - x,
                    None => 0.0,
                }
            };
            LightMap {
                change: (0..MAP_SAMPLES).map(|i| change(table_light(i, MAP_SAMPLES)) as f32).collect(),
                floor: change(0.0) as f32,
                keyed,
            }
        };
        let (blacks_fn, contrast_fn) = (table_map(&blacks_map), table_map(&contrast_map));
        ToneConstants {
            decode: tables.decode.clone(),
            gain: gain as f32,
            local: local.map(Into::into),
            // Where they take contrast out (Whites below zero, Blacks
            // above) the two are a gain keyed to the pixel's luminance;
            // where they add it, and Contrast, a curve the largest and the
            // smallest channel go through: as in the raw development.
            whites: stage(whites_map.as_deref(), s.whites < 0.0),
            blacks: stage(blacks_fn.as_ref().map(|f| f as &dyn Fn(f64) -> f64), s.blacks > 0.0),
            contrast: stage(contrast_fn.as_ref().map(|f| f as &dyn Fn(f64) -> f64), false),
            encode: tables.encode.clone(),
            below: tables.below,
            long_side: 0,
        }
    }
}

impl ToneConstants {
    /// Light of a code.
    pub fn light(&self, code: f32) -> f32 {
        sample_curve(&self.decode, code.clamp(0.0, 1.0) * (self.decode.len() - 1) as f32)
    }

    /// Code of a light.
    pub fn code(&self, x: f32) -> f32 {
        if x < 0.0 {
            return (self.encode[0] + x * self.below).max(0.0);
        }
        let q = srgb_encode(x.min(1.0) as f64) as f32;
        sample_curve(&self.encode, q * (self.encode.len() - 1) as f32)
    }

    /// The luminance the base is taken from: of the pixel's light at the
    /// frame's exposure (a channel below no light counts as none).
    pub fn base_luminance(&self, codes: [f32; 3]) -> f32 {
        let lit = codes.map(|c| (self.light(c) * self.gain).max(0.0));
        (lit[0] * LUMA[0] + lit[1] * LUMA[1] + lit[2] * LUMA[2]).max(crate::raw_tone::BASE_FLOOR)
    }

    /// One pixel through the stage: `codes` in, codes out; `base` is its
    /// base level (log₂; its own level where the frame has none). The GPU
    /// stage computes the same.
    pub fn apply(&self, codes: [f32; 3], base: f32) -> [f32; 3] {
        // A channel may be below no light: a code under the curve's black.
        let lit = codes.map(|c| self.light(c) * self.gain);
        let luminance = lit[0] * LUMA[0] + lit[1] * LUMA[1] + lit[2] * LUMA[2];
        // Shadows / Highlights: the gain of the base, limited by the room
        // the pixel has below white.
        let g = match &self.local {
            Some(gains) => {
                let room = (-luminance.max(1e-12).log2()).max(0.0);
                2f32.powf(limit_gain(sample_curve(gains, (base - LOCAL_LO) * LOCAL_PER_STOP), room))
            }
            None => 1.0,
        };
        let hi = lit.iter().cloned().fold(f32::MIN, f32::max);
        let lo = lit.iter().cloned().fold(f32::MAX, f32::min);
        let between = if hi > lo { (luminance - lo) / (hi - lo) } else { 0.0 };
        let (whi, wlo) = self.whites.extremes((g * hi).min(1.0), (g * lo).min(1.0), between);
        let (bhi, blo) = self.blacks.extremes(whi, wlo, between);
        let (chi, clo) = self.contrast.extremes(bhi, blo, between);
        // The middle channel keeps its place between the two, in light.
        lit.map(|v| self.code(if hi > lo { clo + (chi - clo) * (v - lo) / (hi - lo) } else { chi }))
    }
}

/// Samples of a gain curve that moves nothing (for a frame whose settings
/// have no Shadows / Highlights the GPU stage still reads one).
pub fn flat_gains() -> Vec<f32> {
    vec![0.0; LOCAL_SAMPLES]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_tone::ToneKeys;

    const ILOG: LogCurve = LogCurve::INSTA360_ILOG;

    fn tone(settings: ToneSettings, light: ToneLight) -> LinearTone {
        LinearTone { settings, light, pre_stops: 0.0, measure: Measure::Wait }
    }

    /// A frame whose reduced copy spreads evenly over `stops` below `top`
    /// (log₂ of the light, 0 = white).
    fn frame(top: f32, stops: f32) -> FrameStats {
        let levels: Vec<f32> = (0..1764).map(|i| top - stops * (1.0 - i as f32 / 1763.0)).collect();
        FrameStats {
            keys: ToneKeys { max: 2f32.powf(top), dark: 2f32.powf(top - stops) },
            levels: levels.into(),
            peaks: Vec::new().into(),
        }
    }

    /// A rendering like a log clip's LUT: nothing below a black code, a
    /// steep middle, a long shoulder that presses the top stops together
    /// and still rises a little at white (a tenth of a code per code, as
    /// the cameras' own LUTs do).
    fn log_lut() -> Shown {
        Shown::of(|code| {
            let t = ((code - 0.09) / 0.91).clamp(0.0, 1.0);
            1.0 - 0.9 * (1.0 - t).powf(3.2) - 0.1 * (1.0 - t)
        })
    }

    /// The neutral axis of a camera maker's own log → Rec.709 LUT (the one
    /// the app applies to I-Log clips), every sixteenth of the code range:
    /// black up to code 0.09, and the top quarter of the codes within a
    /// tenth of white.
    fn camera_lut() -> Shown {
        const AXIS: [f32; 17] = [
            0.0010, 0.0010, 0.0291, 0.0733, 0.1309, 0.2094, 0.3080, 0.4216, 0.5402,
            0.6526, 0.7531, 0.8389, 0.9066, 0.9528, 0.9791, 0.9918, 0.9994,
        ];
        Shown::of(|code| {
            let y = code.clamp(0.0, 1.0) * 16.0;
            let i = (y.floor() as usize).min(15);
            AXIS[i] + (AXIS[i + 1] - AXIS[i]) * (y - i as f32)
        })
    }

    fn grey(k: &ToneConstants, code: f32) -> f32 {
        let base = k.base_luminance([code; 3]).log2();
        k.apply([code; 3], base)[1]
    }

    #[test]
    fn the_curve_of_a_rendering_and_back() {
        let s = log_lut();
        // Level below the black code and at the top: entered from inside.
        assert!(s.at(0.05).abs() < 1e-7);
        assert!((s.code_of(0.0) - 0.09).abs() < 3e-4, "{}", s.code_of(0.0));
        assert!((s.code_of(1.0) - 1.0).abs() < 1e-9);
        // A stretch that is level at the top is entered at its first code.
        let clipped = Shown::of(|c| (c / 0.8).min(1.0));
        assert!((clipped.code_of(1.0) - 0.8).abs() < 3e-4 && clipped.at(0.9) == 1.0);
        // In between, the code a value is shown at shows that value.
        for i in 1..200 {
            let e = i as f64 / 200.0;
            assert!((s.at(s.code_of(e)) - e).abs() < 1e-6, "e = {e}");
        }
        // A rendering that falls is held level.
        let dip = Shown::of(|c| if c > 0.6 && c < 0.7 { 0.3 } else { c });
        assert!(dip.at(0.65) >= dip.at(0.6) - 1e-9);
        // The identity is itself both ways.
        let id = Shown::identity();
        assert!((id.at(0.3) - 0.3).abs() < 1e-7 && (id.code_of(0.3) - 0.3).abs() < 1e-7);
    }

    #[test]
    fn a_stage_that_moves_nothing_returns_the_codes() {
        // No control moved, in both lights and through a LUT: every code
        // comes back as it went in. Where the codes go through the inverse
        // of a LUT's shoulder (steep, and bent) to within a thirtieth of a
        // 10-bit step; elsewhere to a few parts in a million.
        for (light, shown, within) in [
            (ToneLight::Scene(ILOG), log_lut(), 3e-6),
            (ToneLight::Shown, Shown::identity(), 3e-6),
            (ToneLight::Shown, log_lut(), 3e-5),
        ] {
            let k = tone(ToneSettings::default(), light).constants_for(&FrameStats::unknown(), &shown, None);
            for i in 0..=200 {
                let code = 0.1 + 0.85 * i as f32 / 200.0;
                assert!((grey(&k, code) - code).abs() < within, "{light:?} code {code}: {}", grey(&k, code));
            }
            // In the scene's light the codes under the curve's black too
            // (the clip's noise there): its toe runs on below it.
            if let ToneLight::Scene(_) = light {
                for code in [0.0f32, 0.02, 0.05, 0.085, 0.0906, 0.095] {
                    assert!((grey(&k, code) - code).abs() < within, "{light:?} code {code}: {}", grey(&k, code));
                }
                let out = k.apply([0.07, 0.11, 0.05], k.base_luminance([0.07, 0.11, 0.05]).log2());
                for (a, b) in out.iter().zip([0.07, 0.11, 0.05]) {
                    assert!((a - b).abs() < within, "{light:?}: {out:?}");
                }
            }
            // A colour too.
            let out = k.apply([0.62, 0.41, 0.35], k.base_luminance([0.62, 0.41, 0.35]).log2());
            for (a, b) in out.iter().zip([0.62, 0.41, 0.35]) {
                assert!((a - b).abs() < within, "{light:?}: {out:?}");
            }
        }
    }

    #[test]
    fn exposure_is_a_factor_of_light() {
        // In the scene's light it is the log curve's own exposure change.
        for stops in [-2.0f32, -1.0, 0.5, 1.5] {
            let k = tone(ToneSettings { exposure: stops, ..Default::default() }, ToneLight::Scene(ILOG))
                .constants_for(&FrameStats::unknown(), &log_lut(), None);
            // Under the curve's black as well: noise there keeps its mean.
            for code in [0.03f32, 0.07, 0.12, 0.2, 0.35, 0.422, 0.6, 0.75] {
                let want = ILOG.expose(code as f64, stops as f64).clamp(0.0, 1.0) as f32;
                assert!((grey(&k, code) - want).abs() < 6e-5, "{stops} stops at {code}: {} vs {want}", grey(&k, code));
            }
        }
        // In the light the picture shows, what is displayed moves by the
        // factor, whatever curve renders it: through the LUT and without.
        for shown in [log_lut(), Shown::identity()] {
            for stops in [-1.0f32, 1.0] {
                let k = tone(ToneSettings { exposure: stops, ..Default::default() }, ToneLight::Shown)
                    .constants_for(&FrameStats::unknown(), &shown, None);
                for code in [0.2f32, 0.3, 0.45, 0.6] {
                    let (before, after) = (srgb_decode(shown.at(code as f64)), srgb_decode(shown.at(grey(&k, code) as f64)));
                    if before * 2f64.powf(stops as f64) < 0.98 {
                        assert!((after / before - 2f64.powf(stops as f64)).abs() < 4e-3, "{stops} stops at {code}: ×{}", after / before);
                    }
                }
            }
        }
        // The exposure a clip was shot over is taken out by the same stage.
        let k = LinearTone { pre_stops: -2.0, ..tone(ToneSettings::default(), ToneLight::Scene(ILOG)) }
            .constants_for(&FrameStats::unknown(), &Shown::identity(), None);
        assert!((grey(&k, ILOG.encode(0.72) as f32) - ILOG.encode(0.18) as f32).abs() < 6e-5);
    }

    #[test]
    fn highlights_pressed_together_by_the_lut_come_apart_again() {
        // Two highlights a stop apart in the scene, both on the shoulder of
        // the camera's LUT: shown nearly alike.
        let shown = camera_lut();
        let (a, b) = (ILOG.encode(8.0) as f32, ILOG.encode(16.0) as f32);
        let apart = |ca: f32, cb: f32| shown.at(cb as f64) - shown.at(ca as f64);
        let before = apart(a, b);
        assert!(before < 0.02, "the two already show {before} apart");
        // The frame: nine stops, up to a little under white.
        let stats = frame(-0.3, 9.0);
        let k = tone(ToneSettings { highlights: -100.0, ..Default::default() }, ToneLight::Scene(ILOG))
            .constants_for(&stats, &shown, None);
        let (a2, b2) = (grey(&k, a), grey(&k, b));
        // Pulled down they are still about a stop apart in light (their
        // bases here are their own levels, so the curve's slope takes a
        // little of it), and the LUT now shows the difference.
        assert!(a2 < a && b2 < b && b2 > a2, "{a} → {a2}, {b} → {b2}");
        let stops = ((ILOG.decode(b2 as f64) / ILOG.decode(a2 as f64)).log2(), (ILOG.decode(b as f64) / ILOG.decode(b2 as f64)).log2());
        assert!(stops.0 > 0.6 && stops.0 <= 1.0 + 1e-3, "{} stops apart after", stops.0);
        let after = apart(a2, b2);
        assert!(after > 2.0 * before, "shown {before} apart before, {after} after ({} stops down)", stops.1);
        // The same pull made on the picture after the LUT has only what the
        // LUT left: the two shown values, each scaled in light.
        let pull = |e: f64| srgb_encode(srgb_decode(e) * 2f64.powf(-stops.1));
        let late = pull(shown.at(b as f64)) - pull(shown.at(a as f64));
        assert!(after > 2.0 * late, "{after} apart here, {late} when graded after the LUT");
    }

    #[test]
    fn the_global_controls_move_what_is_shown_as_their_maps_say() {
        let stats = frame(-0.5, 10.0);
        for (light, shown) in [(ToneLight::Shown, Shown::identity()), (ToneLight::Shown, log_lut()), (ToneLight::Scene(ILOG), log_lut())] {
            // Blacks below zero: darker shadows, white kept.
            let k = tone(ToneSettings { blacks: -60.0, ..Default::default() }, light).constants_for(&stats, &shown, None);
            let dark = shown.code_of(0.12) as f32;
            assert!(shown.at(grey(&k, dark) as f64) < 0.115, "{light:?}: {}", shown.at(grey(&k, dark) as f64));
            let bright = shown.code_of(0.9) as f32;
            assert!((shown.at(grey(&k, bright) as f64) - 0.9).abs() < 0.03, "{light:?}");
            // Whites below zero: the top comes down, the shadows stay.
            let k = tone(ToneSettings { whites: -80.0, ..Default::default() }, light).constants_for(&stats, &shown, None);
            assert!(shown.at(grey(&k, bright) as f64) < 0.88, "{light:?}: {}", shown.at(grey(&k, bright) as f64));
            assert!((shown.at(grey(&k, dark) as f64) - 0.12).abs() < 0.02, "{light:?}");
            // Contrast above zero: shadows darker, highlights brighter.
            let k = tone(ToneSettings { contrast: 70.0, ..Default::default() }, light).constants_for(&stats, &shown, None);
            let (lo, hi) = (shown.code_of(0.2) as f32, shown.code_of(0.8) as f32);
            assert!(shown.at(grey(&k, lo) as f64) < 0.2 && shown.at(grey(&k, hi) as f64) > 0.8, "{light:?}");
            // A grey ramp stays in order under all of them at once.
            let k = tone(ToneSettings { exposure: 0.4, contrast: 40.0, highlights: -70.0, shadows: 60.0, whites: 30.0, blacks: -25.0 }, light)
                .constants_for(&stats, &shown, None);
            let mut last = -1.0;
            for i in 0..=400 {
                let code = 0.1 + 0.9 * i as f32 / 400.0;
                let out = grey(&k, code);
                assert!(out >= last - 2e-5, "{light:?}: {out} after {last} at {code}");
                last = out;
            }
        }
    }

    #[test]
    fn a_map_of_the_shown_value_is_met_exactly_by_a_neutral() {
        // Whites below zero is the same map for every frame: a grey's
        // displayed value must land on the map's.
        let m = crate::raw_global::whites_map(-50.0).unwrap();
        for (light, shown) in [(ToneLight::Shown, Shown::identity()), (ToneLight::Shown, log_lut()), (ToneLight::Scene(ILOG), log_lut())] {
            let k = tone(ToneSettings { whites: -50.0, ..Default::default() }, light).constants_for(&FrameStats::unknown(), &shown, None);
            for e in [0.15f64, 0.3, 0.5, 0.7, 0.9] {
                let code = shown.code_of(e) as f32;
                let got = shown.at(grey(&k, code) as f64);
                let want = crate::raw_global::apply(&m, e as f32) as f64;
                assert!((got - want).abs() < 3e-3, "{light:?} at {e}: shown {got}, the map says {want}");
            }
        }
    }

    #[test]
    fn a_keyed_control_keeps_a_colours_ratios_in_light() {
        // Whites below zero is a gain keyed to the luminance.
        let k = tone(ToneSettings { whites: -80.0, ..Default::default() }, ToneLight::Shown)
            .constants_for(&FrameStats::unknown(), &Shown::identity(), None);
        let codes = [0.85f32, 0.6, 0.45];
        let out = k.apply(codes, k.base_luminance(codes).log2());
        let (before, after) = (codes.map(|c| srgb_decode(c as f64)), out.map(|c| srgb_decode(c as f64)));
        assert!(after[0] < before[0]);
        for ch in 1..3 {
            assert!((after[ch] / after[0] - before[ch] / before[0]).abs() < 2e-3, "{before:?} → {after:?}");
        }
    }

    #[test]
    fn a_code_under_black_goes_as_far_under_what_black_becomes_as_one_over_it_goes_over() {
        let stats = frame(-0.5, 10.0);
        let black = ILOG.encode(0.0) as f32;
        for settings in [
            ToneSettings { blacks: 40.0, ..Default::default() },
            ToneSettings { blacks: -40.0, contrast: 30.0, ..Default::default() },
            ToneSettings { whites: -60.0, ..Default::default() },
            ToneSettings { contrast: -40.0, ..Default::default() },
            ToneSettings { shadows: 80.0, highlights: -50.0, ..Default::default() },
        ] {
            let k = tone(settings, ToneLight::Scene(ILOG)).constants_for(&stats, &log_lut(), None);
            let at_black = grey(&k, black);
            // Noise a little under black and as much over it: nothing is
            // cut off at black, the two come out alike on either side of
            // what black becomes (both ON it where the control presses the
            // darkest tones together). Shadows and Highlights are a gain of
            // the pixel's base, the same for both here.
            let base = k.base_luminance([black; 3]).log2();
            let d = 2e-4f32;
            let (under, over) = (k.apply([black - d; 3], base)[1], k.apply([black + d; 3], base)[1]);
            let (down, up) = (at_black - under, over - at_black);
            assert!(down >= -1e-6 && up >= -1e-6, "{settings:?}: {under} {at_black} {over}");
            assert!((down - up).abs() <= 0.1 * down.max(up) + 2e-6, "{settings:?}: {down} under, {up} over");
        }
        // A control that hardly reaches the darkest tones (Highlights: a
        // few thousandths of a stop down there) leaves the noise under
        // black where it is, to a third of a 10-bit step.
        let k = tone(ToneSettings { highlights: -80.0, ..Default::default() }, ToneLight::Scene(ILOG)).constants_for(&stats, &log_lut(), None);
        let far_down = (stats.levels[0] - 6.0).min(-17.0);
        for code in [0.03f32, 0.06, 0.085] {
            assert!((k.apply([code; 3], far_down)[1] - code).abs() < 3e-4, "{code}: {}", k.apply([code; 3], far_down)[1]);
        }
        // The light the picture shows has nothing under black: its darkest
        // code is no light.
        let k = tone(ToneSettings { exposure: 1.0, ..Default::default() }, ToneLight::Shown).constants_for(&stats, &Shown::identity(), None);
        assert_eq!(k.below, 0.0);
        assert!(grey(&k, 0.0).abs() < 1e-6);
    }

    #[test]
    fn what_needs_the_frame() {
        let none = ToneSettings::default();
        assert!(none.is_identity() && !none.needs_stats());
        assert!(!ToneSettings { exposure: 1.0, ..none }.needs_stats());
        assert!(!ToneSettings { whites: -30.0, ..none }.needs_stats());
        for s in [
            ToneSettings { whites: 30.0, ..none }, ToneSettings { blacks: 5.0, ..none }, ToneSettings { contrast: -5.0, ..none },
            ToneSettings { shadows: 5.0, ..none }, ToneSettings { highlights: -5.0, ..none },
        ] {
            assert!(s.needs_stats() && !s.is_identity(), "{s:?}");
        }
        // The exposure taken out for a clip shot bright counts as a change.
        assert!(!LinearTone { pre_stops: -1.0, ..tone(none, ToneLight::Scene(ILOG)) }.is_identity());
    }
}
