//! Tone of raw development: the user's settings and what they turn into
//! for one frame — the constants the GPU stage (`shaders/raw_develop.wgsl`)
//! and its CPU twin ([`DevelopConstants::develop`]) run on.
//!
//! The order, on the camera's linear RGB (sensor white = 1, see
//! [`crate::raw_color`]):
//!
//! 1. clip to the camera white, then camera → linear ProPhoto, clipped to
//!    0..1;
//! 2. exposure: a gain of `2^stops` up, with everything that leaves 0..1
//!    clipped; a negative exposure is taken off in the tone curve instead,
//!    with a shoulder that keeps white at white;
//! 3. the tone curve, applied so that hue is kept: the largest and the
//!    smallest channel go through the curve and the middle one keeps its
//!    place between them;
//! 4. ProPhoto → Rec.709 primaries, clipped to 0..1, sRGB encoding.

use crate::raw_color::{apply, prophoto_to_rec709, xy_from_temperature, RawCamera, Xy, M3};

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
}

impl Default for RawSettings {
    fn default() -> Self {
        Self { temperature: None, tint: None, exposure: 0.0, contrast: 0.0, highlights: 0.0, shadows: 0.0, whites: 0.0, blacks: 0.0 }
    }
}

/// The raw development stage of a colour stack: a camera and the settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawDevelop {
    pub camera: RawCamera,
    pub settings: RawSettings,
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

    /// Everything the stage needs for one frame.
    pub fn constants(&self) -> DevelopConstants {
        let white = self.white();
        let exposure = self.settings.exposure as f64 + self.camera.baseline_exposure;
        DevelopConstants {
            camera_white: self.camera.camera_white(white).map(|v| v as f32),
            camera_to_work: to_f32(&self.camera.camera_to_prophoto(white)),
            work_to_out: to_f32(&prophoto_to_rec709()),
            ramp: ExposureRamp::new(1.0 / 2f64.powf(exposure.max(0.0)), 0.0),
            shoulder: ExposureShoulder::new(exposure),
            curve: default_curve(),
        }
    }
}

fn to_f32(m: &M3) -> [[f32; 3]; 3] {
    m.map(|r| r.map(|v| v as f32))
}

/// A straight line from `black` to `white` (→ 0..1), with a short parabola
/// easing into the black point instead of a corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureRamp {
    pub slope: f32,
    pub black: f32,
    pub radius: f32,
    pub q_scale: f32,
}

impl ExposureRamp {
    pub fn new(white: f64, black: f64) -> Self {
        let black = black.min(0.99 * white);
        let slope = 1.0 / (white - black);
        // The ease is at most half the black point wide and a sixteenth of
        // white high.
        let radius = (0.5 * black).min((1.0 / 16.0) / slope);
        let q_scale = if radius > 0.0 { slope / (4.0 * radius) } else { 0.0 };
        Self { slope: slope as f32, black: black as f32, radius: radius as f32, q_scale: q_scale as f32 }
    }

    pub fn eval(&self, x: f32) -> f32 {
        if x <= self.black - self.radius {
            0.0
        } else if x >= self.black + self.radius {
            ((x - self.black) * self.slope).min(1.0)
        } else {
            let y = x - (self.black - self.radius);
            self.q_scale * y * y
        }
    }
}

/// Darkening for a negative exposure, applied in front of the tone curve:
/// a plain gain below a quarter of white, a parabola above that still
/// takes white to white.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureShoulder {
    pub slope: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl ExposureShoulder {
    pub fn new(exposure: f64) -> Self {
        if exposure >= 0.0 {
            return Self { slope: 1.0, a: 0.0, b: 1.0, c: 0.0 };
        }
        let slope = 2f64.powf(exposure);
        let a = 16.0 / 9.0 * (1.0 - slope);
        let b = slope - 0.5 * a;
        let c = 1.0 - a - b;
        Self { slope: slope as f32, a: a as f32, b: b as f32, c: c as f32 }
    }

    pub fn eval(&self, x: f32) -> f32 {
        if x <= 0.25 { x * self.slope } else { (self.a * x + self.b) * x + self.c }
    }
}

/// A tone curve as equally spaced samples over 0..1 (linear light in,
/// linear light out), read with linear interpolation.
pub type CurveSamples = std::sync::Arc<[f32]>;

/// The curve a frame is developed with when nothing else is chosen. A
/// filmic stand-in (the stills' ACES fit, normalised to take white to
/// white) until the raw converter's own default curve is in.
pub fn default_curve() -> CurveSamples {
    static CURVE: std::sync::OnceLock<CurveSamples> = std::sync::OnceLock::new();
    CURVE.get_or_init(|| {
        const N: usize = 4097;
        let aces = |x: f64| (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        let top = aces(1.0);
        (0..N).map(|i| (aces(i as f64 / (N - 1) as f64) / top) as f32).collect::<Vec<f32>>().into()
    }).clone()
}

/// Read a sampled curve at `x` (0..1).
pub fn curve_at(curve: &[f32], x: f32) -> f32 {
    let last = curve.len() - 1;
    let y = x.clamp(0.0, 1.0) * last as f32;
    let i = (y as usize).min(last - 1);
    let f = y - i as f32;
    curve[i] * (1.0 - f) + curve[i + 1] * f
}

/// The per-frame constants of the stage.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopConstants {
    pub camera_white: [f32; 3],
    pub camera_to_work: [[f32; 3]; 3],
    pub work_to_out: [[f32; 3]; 3],
    pub ramp: ExposureRamp,
    pub shoulder: ExposureShoulder,
    pub curve: CurveSamples,
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

impl DevelopConstants {
    fn tone(&self, x: f32) -> f32 {
        curve_at(&self.curve, self.shoulder.eval(x))
    }

    /// Develop one pixel on the CPU: camera-linear RGB in, sRGB-encoded
    /// Rec.709 out. The GPU stage computes the same.
    pub fn develop(&self, camera: [f32; 3]) -> [f32; 3] {
        let m = |m: &[[f32; 3]; 3], v: [f32; 3]| {
            let v64 = apply(&m.map(|r| r.map(|x| x as f64)), v.map(|x| x as f64));
            v64.map(|x| x as f32)
        };
        let clipped = [0, 1, 2].map(|i| camera[i].clamp(0.0, self.camera_white[i]));
        let work = m(&self.camera_to_work, clipped).map(|v| v.clamp(0.0, 1.0));
        let lit = work.map(|v| self.ramp.eval(v));
        // Hue-keeping tone: the extremes through the curve, the middle
        // channel at its old place between them.
        let hi = lit.iter().cloned().fold(f32::MIN, f32::max);
        let lo = lit.iter().cloned().fold(f32::MAX, f32::min);
        let (thi, tlo) = (self.tone(hi), self.tone(lo));
        let toned = lit.map(|v| if hi > lo { tlo + (thi - tlo) * (v - lo) / (hi - lo) } else { thi });
        m(&self.work_to_out, toned).map(|v| srgb_encode(v.clamp(0.0, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_ramp_and_shoulder() {
        // No black point: a plain gain, clipped at white.
        let up = ExposureRamp::new(0.5, 0.0);
        assert!((up.eval(0.2) - 0.4).abs() < 1e-6 && up.eval(0.6) == 1.0 && up.eval(0.0) == 0.0);
        // With one, the ease joins the line where it starts.
        let r = ExposureRamp::new(1.0, 0.005);
        assert_eq!(r.eval(0.0), 0.0);
        let edge = r.black + r.radius;
        assert!((r.eval(edge - 1e-6) - r.eval(edge + 1e-6)).abs() < 1e-4);
        assert!((r.eval(1.0) - 1.0).abs() < 1e-6);
        // A negative exposure keeps white at white and is the gain below a quarter.
        let s = ExposureShoulder::new(-1.0);
        assert!((s.eval(1.0) - 1.0).abs() < 1e-6 && (s.eval(0.2) - 0.1).abs() < 1e-6);
        assert!((s.eval(0.25 + 1e-6) - s.eval(0.25)).abs() < 1e-5);
        let none = ExposureShoulder::new(0.5);
        assert_eq!((none.eval(0.3), none.eval(0.9)), (0.3, 0.9));
    }

    #[test]
    fn a_neutral_stays_neutral_through_development() {
        let dev = RawDevelop { camera: RawCamera::X6, settings: RawSettings::default() };
        let k = dev.constants();
        let n = RawCamera::X6.as_shot_neutral.map(|v| v as f32);
        // Greys of the as-shot white, from deep shadow to the clip point.
        let mut last = -1.0;
        for level in [0.001f32, 0.01, 0.05, 0.18, 0.5, 1.0, 1.7] {
            let out = k.develop(n.map(|v| v * level));
            assert!((out[0] - out[1]).abs() < 2e-4 && (out[2] - out[1]).abs() < 2e-4, "level {level}: {out:?}");
            assert!(out[1] >= last, "not rising at {level}");
            last = out[1];
        }
        // The sensor's clip point (every channel at 1) is white, not a colour.
        let white = k.develop([1.0, 1.0, 1.0]);
        assert!(white.iter().all(|v| (v - 1.0).abs() < 1e-4), "{white:?}");
        // One stop more exposure is twice the light before the curve.
        let brighter = RawDevelop { camera: RawCamera::X6, settings: RawSettings { exposure: 1.0, ..Default::default() } }.constants();
        let a = brighter.develop(n.map(|v| v * 0.05));
        let b = k.develop(n.map(|v| v * 0.1));
        assert!((a[1] - b[1]).abs() < 1e-5, "{a:?} {b:?}");
        // A warmer white balance setting makes the same pixel bluer.
        let warm = RawDevelop { camera: RawCamera::X6, settings: RawSettings { temperature: Some(9000.0), ..Default::default() } }.constants();
        let cool = RawDevelop { camera: RawCamera::X6, settings: RawSettings { temperature: Some(3000.0), ..Default::default() } }.constants();
        let px = n.map(|v| v * 0.2);
        let (w, c) = (warm.develop(px), cool.develop(px));
        assert!(w[0] > w[2] && c[2] > c[0], "9000 K {w:?}, 3000 K {c:?}");
    }

    #[test]
    fn settings_read_from_older_files() {
        let s: RawSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, RawSettings::default());
        let s: RawSettings = serde_json::from_str(r#"{"exposure":1.5,"temperature":5200.0}"#).unwrap();
        assert_eq!((s.exposure, s.temperature, s.tint), (1.5, Some(5200.0), None));
    }
}
