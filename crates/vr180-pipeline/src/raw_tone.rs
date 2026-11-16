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
//! 2. **White point.** A shoulder above 0.25 maps `W*` to 1 (a table
//!    measured over `W*`), where `W*` is the sensor's white, but at most
//!    twice the frame's own maximum and at least 1. At `W* = 1` it does
//!    nothing.
//! 3. **Default black.** Shadows are pulled down by an amount that follows
//!    the frame's darkest level (a table measured over that level): from a
//!    quarter of a per cent at the very bottom to the plain DNG black point
//!    of 0.005 once the darkest level is above 2⁻⁷.
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
//! Then: ProPhoto → Rec.709 primaries, clipped to 0..1, sRGB encoding.

use crate::raw_color::{apply, prophoto_luma, prophoto_to_rec709, xy_from_temperature, RawCamera, Xy, M3};

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

/// What the rendering reads off the frame itself, on a reduced copy 128
/// cells along the frame's longer side, in scene values at exposure 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneKeys {
    /// The brightest channel of any cell.
    pub max: f32,
    /// The 0.1th percentile of the cells' luminance: the frame's darkest level.
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
        let mut y: Vec<f32> = cells.iter().map(|c| (c[0] * luma[0] + c[1] * luma[1] + c[2] * luma[2]).max(0.0)).collect();
        y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Linear between the two cells the rank falls between.
        let rank = 0.001 * (y.len() - 1) as f32;
        let (i, f) = (rank.floor() as usize, rank.fract());
        let dark = y[i] * (1.0 - f) + y[(i + 1).min(y.len() - 1)] * f;
        Some(Self { max, dark })
    }
}

/// When the frame's keys are measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Measure {
    /// Measure this frame and wait for the result (an export, a still).
    #[default]
    Wait,
    /// Use the last measurement that has come back and queue one of this
    /// frame (the live preview, which must not wait on the GPU).
    Latest,
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

    /// What does not depend on the frame: the camera's white and matrices.
    pub fn geometry(&self) -> DevelopGeometry {
        let white = self.white();
        DevelopGeometry {
            neutral: self.camera.camera_white(white).map(|v| v as f32),
            camera_to_work: to_f32(&self.camera.camera_to_prophoto(white)),
            work_to_out: to_f32(&prophoto_to_rec709()),
        }
    }

    /// Everything the stage needs for one frame with the keys `keys`.
    pub fn constants(&self, keys: ToneKeys) -> DevelopConstants {
        let geometry = self.geometry();
        let gain = 2f64.powf(self.exposure());
        // The sensor's white: where a neutral's last channel clips.
        let lowest = geometry.neutral.iter().cloned().fold(f32::MAX, f32::min).max(1e-3) as f64;
        let sensor_white = gain / lowest;
        let frame_max = keys.max as f64 * gain;
        let white_point = sensor_white.min(2.0 * frame_max).max(1.0);
        let dark_key = (keys.dark as f64 * gain).max(2f64.powi(-30)).log2();
        let hi = white_point.log2().max(0.0);
        let curve: Vec<f32> = (0..CURVE_SAMPLES).map(|i| {
            let x = 2f64.powf(CURVE_LO + (hi - CURVE_LO) * i as f64 / (CURVE_SAMPLES - 1) as f64);
            default_render(x, white_point, dark_key) as f32
        }).collect();
        DevelopConstants { geometry, gain: gain as f32, curve: curve.into(), curve_hi: hi as f32, white_point: white_point as f32, dark_key: dark_key as f32 }
    }
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

/// The whole default rendering of a scene value: white point, default
/// black, tone curve → output light.
pub fn default_render(x: f64, white_point: f64, dark_key: f64) -> f64 {
    default_curve(default_black(default_white(x, white_point), dark_key).clamp(0.0, 1.0))
}

/// Samples of the per-frame curve.
pub const CURVE_SAMPLES: usize = 4096;
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
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

impl DevelopConstants {
    /// The frame's curve at the scene value `x`.
    pub fn tone(&self, x: f32) -> f32 {
        if x <= 0.0 { return 0.0; }
        let lo = CURVE_LO as f32;
        let u = x.log2();
        if u <= lo {
            return self.curve[0] * x / 2f32.powf(lo);
        }
        let last = (self.curve.len() - 1) as f32;
        let y = ((u - lo) / (self.curve_hi - lo) * last).min(last);
        let i = (y.floor() as usize).min(self.curve.len() - 2);
        let f = y - i as f32;
        self.curve[i] * (1.0 - f) + self.curve[i + 1] * f
    }

    /// Develop one pixel on the CPU: camera-linear RGB in, sRGB-encoded
    /// Rec.709 out. The GPU stage computes the same.
    pub fn develop(&self, camera: [f32; 3]) -> [f32; 3] {
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
            (0.5, 2.0, -14.0, 0.773082),
            (1.0, 2.49, -11.3, 0.927377),
            (2.0, 4.0, -14.0, 0.982925),
            (3.0, 9.96, -8.25, 0.990487),
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
        // The 0.1th percentile: two cells up from the darkest.
        assert!((k.dark - 0.0012).abs() < 2e-5, "{}", k.dark);
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
    fn settings_read_from_older_files() {
        let s: RawSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(s, RawSettings::default());
        let s: RawSettings = serde_json::from_str(r#"{"exposure":1.5,"temperature":5200.0}"#).unwrap();
        assert_eq!((s.exposure, s.temperature, s.tint), (1.5, Some(5200.0), None));
    }
}
