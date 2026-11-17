//! Colour science of raw development: white balance, the camera matrix and
//! the working space, after the public DNG rendering model (the DNG
//! specification's "Mapping Camera Color Space to CIE XYZ Space" and the
//! reference rendering of the DNG SDK), so that a frame developed here and
//! the same frame opened from its DNG in a raw converter start from the
//! same colours.
//!
//! The model, for a camera described by one `ColorMatrix` (XYZ → camera
//! under its calibration illuminant):
//!
//! 1. The white point is an `xy` chromaticity. It comes from the frame's
//!    neutral (`neutral_to_xy`) or from a temperature and a tint
//!    (`xy_from_temperature`, the correlated colour temperature along the
//!    Planckian locus by Robertson's isotherms and a signed distance from
//!    it).
//! 2. The camera's response to that white, scaled so its largest channel is
//!    1, is the CAMERA WHITE: every camera value is clipped there first, so
//!    a saturated sensor reads as white, not as a colour.
//! 3. Camera → working space is the inverse of `ColorMatrix · adapt(D50 →
//!    white)`, scaled so the camera white lands on the working white. The
//!    working space is linear ProPhoto RGB (ROMM primaries, D50).
//! 4. Working → output is a fixed matrix (ProPhoto D50 → Rec.709 / sRGB
//!    primaries D65, Bradford adapted).
//!
//! Everything here is `f64` on the CPU: it runs once per settings change
//! and hands the GPU a few constants.

/// 3×3 matrix, row-major.
pub type M3 = [[f64; 3]; 3];

/// CIE 1931 chromaticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xy {
    pub x: f64,
    pub y: f64,
}

impl Xy {
    pub const D50: Xy = Xy { x: 0.3457, y: 0.3585 };
    pub const D65: Xy = Xy { x: 0.3127, y: 0.3290 };

    /// XYZ of this chromaticity at Y = 1.
    pub fn to_xyz(self) -> [f64; 3] {
        let y = self.y.max(1e-9);
        [self.x / y, 1.0, (1.0 - self.x - self.y) / y]
    }

    pub fn from_xyz(xyz: [f64; 3]) -> Xy {
        let sum = xyz[0] + xyz[1] + xyz[2];
        if sum.abs() < 1e-12 { return Xy::D50; }
        Xy { x: xyz[0] / sum, y: xyz[1] / sum }
    }
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn apply(m: &M3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Inverse of a 3×3 matrix; `None` when it is singular.
pub fn invert(m: &M3) -> Option<M3> {
    let c = |i: usize, j: usize| {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (p, q) = ((j + 1) % 3, (j + 2) % 3);
        m[a][p] * m[b][q] - m[a][q] * m[b][p]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    if det.abs() < 1e-14 { return None; }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[j][i] = c(i, j) / det;
        }
    }
    Some(out)
}

fn scale(m: &M3, s: f64) -> M3 {
    m.map(|r| r.map(|v| v * s))
}

const BRADFORD: M3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Linear Bradford adaptation: the matrix taking XYZ under white `from` to
/// XYZ under white `to`.
pub fn adapt(from: Xy, to: Xy) -> M3 {
    let inv = invert(&BRADFORD).expect("Bradford is regular");
    let a = apply(&BRADFORD, from.to_xyz());
    let b = apply(&BRADFORD, to.to_xyz());
    let ratio = [0, 1, 2].map(|i| {
        let (a, b) = (a[i].max(0.0), b[i].max(0.0));
        if a > 0.0 { (b / a).clamp(0.1, 10.0) } else { 10.0 }
    });
    let diag: M3 = [[ratio[0], 0.0, 0.0], [0.0, ratio[1], 0.0], [0.0, 0.0, ratio[2]]];
    mul(&inv, &mul(&diag, &BRADFORD))
}

/// Scale the rows of an RGB → XYZ(D50) matrix so the space's white (1, 1,
/// 1) lands exactly on the D50 white the model adapts to (published
/// matrices are rounded).
fn white_exact(m: &M3) -> M3 {
    let w = apply(m, [1.0, 1.0, 1.0]);
    let d50 = Xy::D50.to_xyz();
    [0, 1, 2].map(|i| m[i].map(|v| v * d50[i] / w[i]))
}

/// Linear ProPhoto RGB (ROMM primaries, D50) → XYZ under D50.
pub fn prophoto_to_xyz() -> M3 {
    white_exact(&[
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8249],
    ])
}

/// Linear Rec.709 / sRGB primaries → XYZ under D50 (the primaries adapted
/// from their own D65 white).
pub fn rec709_to_xyz() -> M3 {
    white_exact(&[
        [0.4361, 0.3851, 0.1431],
        [0.2225, 0.7169, 0.0606],
        [0.0139, 0.0971, 0.7141],
    ])
}

/// Linear ProPhoto → linear Rec.709 primaries (white stays white).
pub fn prophoto_to_rec709() -> M3 {
    mul(&invert(&rec709_to_xyz()).expect("regular"), &prophoto_to_xyz())
}

/// Luminance of linear ProPhoto RGB (the Y row of [`prophoto_to_xyz`]).
pub fn prophoto_luma() -> [f64; 3] {
    prophoto_to_xyz()[1]
}

// ── Temperature and tint ───────────────────────────────────────────────

/// One tint unit is this distance in CIE 1960 `uv`, away from green.
const TINT_SCALE: f64 = -3000.0;

/// Robertson's isotherms: reciprocal temperature (mired), the Planckian
/// locus point in CIE 1960 `uv`, the isotherm's slope (Wyszecki & Stiles,
/// "Color Science", 2nd ed., p. 228).
const ISOTHERMS: [(f64, f64, f64, f64); 31] = [
    (0.0, 0.18006, 0.26352, -0.24341),
    (10.0, 0.18066, 0.26589, -0.25479),
    (20.0, 0.18133, 0.26846, -0.26876),
    (30.0, 0.18208, 0.27119, -0.28539),
    (40.0, 0.18293, 0.27407, -0.30470),
    (50.0, 0.18388, 0.27709, -0.32675),
    (60.0, 0.18494, 0.28021, -0.35156),
    (70.0, 0.18611, 0.28342, -0.37915),
    (80.0, 0.18740, 0.28668, -0.40955),
    (90.0, 0.18880, 0.28997, -0.44278),
    (100.0, 0.19032, 0.29326, -0.47888),
    (125.0, 0.19462, 0.30141, -0.58204),
    (150.0, 0.19962, 0.30921, -0.70471),
    (175.0, 0.20525, 0.31647, -0.84901),
    (200.0, 0.21142, 0.32312, -1.0182),
    (225.0, 0.21807, 0.32909, -1.2168),
    (250.0, 0.22511, 0.33439, -1.4512),
    (275.0, 0.23247, 0.33904, -1.7298),
    (300.0, 0.24010, 0.34308, -2.0637),
    (325.0, 0.24792, 0.34655, -2.4681),
    (350.0, 0.25591, 0.34951, -2.9641),
    (375.0, 0.26400, 0.35200, -3.5814),
    (400.0, 0.27218, 0.35407, -4.3633),
    (425.0, 0.28039, 0.35577, -5.3762),
    (450.0, 0.28863, 0.35714, -6.7262),
    (475.0, 0.29685, 0.35823, -8.5955),
    (500.0, 0.30505, 0.35907, -11.324),
    (525.0, 0.31320, 0.35968, -15.628),
    (550.0, 0.32129, 0.36011, -23.325),
    (575.0, 0.32931, 0.36038, -40.770),
    (600.0, 0.33724, 0.36051, -116.45),
];

fn unit(slope: f64) -> (f64, f64) {
    let len = (1.0 + slope * slope).sqrt();
    (1.0 / len, slope / len)
}

/// Chromaticity of a white given as a colour temperature in kelvin and a
/// tint (positive = toward magenta), the raw converters' two white balance
/// numbers.
pub fn xy_from_temperature(kelvin: f64, tint: f64) -> Xy {
    let r = 1.0e6 / kelvin.max(1.0);
    let offset = tint / TINT_SCALE;
    let last = ISOTHERMS.len() - 2;
    for i in 0..=last {
        let (a, b) = (ISOTHERMS[i], ISOTHERMS[i + 1]);
        if r < b.0 || i == last {
            let f = (b.0 - r) / (b.0 - a.0);
            let mut u = a.1 * f + b.1 * (1.0 - f);
            let mut v = a.2 * f + b.2 * (1.0 - f);
            let (u1, v1) = unit(a.3);
            let (u2, v2) = unit(b.3);
            let (mut du, mut dv) = (u1 * f + u2 * (1.0 - f), v1 * f + v2 * (1.0 - f));
            let len = (du * du + dv * dv).sqrt();
            du /= len;
            dv /= len;
            u += du * offset;
            v += dv * offset;
            let d = u - 4.0 * v + 2.0;
            return Xy { x: 1.5 * u / d, y: v / d };
        }
    }
    Xy::D50
}

/// The temperature (kelvin) and tint of a white's chromaticity: the inverse
/// of [`xy_from_temperature`].
pub fn temperature_from_xy(xy: Xy) -> (f64, f64) {
    let d = 1.5 - xy.x + 6.0 * xy.y;
    let (u, v) = (2.0 * xy.x / d, 3.0 * xy.y / d);
    let (mut last_dt, mut last_du, mut last_dv) = (0.0, 0.0, 0.0);
    let n = ISOTHERMS.len() - 1;
    for i in 1..=n {
        let (du, dv) = unit(ISOTHERMS[i].3);
        let (uu, vv) = (u - ISOTHERMS[i].1, v - ISOTHERMS[i].2);
        let mut dt = -uu * dv + vv * du;
        if dt <= 0.0 || i == n {
            if dt > 0.0 { dt = 0.0; }
            dt = -dt;
            let f = if i == 1 { 0.0 } else { dt / (last_dt + dt) };
            let (a, b) = (ISOTHERMS[i - 1], ISOTHERMS[i]);
            let kelvin = 1.0e6 / (a.0 * f + b.0 * (1.0 - f));
            let (uu, vv) = (u - (a.1 * f + b.1 * (1.0 - f)), v - (a.2 * f + b.2 * (1.0 - f)));
            let (mut du, mut dv) = (du * (1.0 - f) + last_du * f, dv * (1.0 - f) + last_dv * f);
            let len = (du * du + dv * dv).sqrt();
            du /= len;
            dv /= len;
            return (kelvin, (uu * du + vv * dv) * TINT_SCALE);
        }
        last_dt = dt;
        last_du = du;
        last_dv = dv;
    }
    (5000.0, 0.0)
}

// ── The camera ─────────────────────────────────────────────────────────

/// What a raw source says about its colours: one colour matrix (XYZ →
/// camera, as a DNG's `ColorMatrix1`), the neutral the frames were shot
/// with (camera values of a grey, green = 1; a DNG's `AsShotNeutral`), and
/// the exposure offset of the camera's own rendering in stops (a DNG's
/// `BaselineExposure`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawCamera {
    pub color_matrix: M3,
    pub as_shot_neutral: [f64; 3],
    pub baseline_exposure: f64,
}

impl RawCamera {
    /// The Insta360 X6's sensor as its own DNG stills describe it
    /// (calibration illuminant D65). A raw video take records no white
    /// balance of its own: the neutral is the camera side's reference grey.
    pub const X6: RawCamera = RawCamera {
        color_matrix: [
            [0.71875, -0.1484375, -0.09375],
            [-0.609375, 1.453125, 0.109375],
            [-0.15625, 0.296875, 0.4765625],
        ],
        as_shot_neutral: [0.5654296875, 1.0, 0.4013671875],
        baseline_exposure: 0.0,
    };

    /// Chromaticity of the white a camera neutral stands for.
    pub fn neutral_to_xy(&self, neutral: [f64; 3]) -> Xy {
        match invert(&self.color_matrix) {
            Some(inv) => Xy::from_xyz(apply(&inv, neutral)),
            None => Xy::D50,
        }
    }

    /// The as-shot white as a temperature (kelvin) and a tint.
    pub fn as_shot_temperature(&self) -> (f64, f64) {
        temperature_from_xy(self.neutral_to_xy(self.as_shot_neutral))
    }

    /// Camera values of the white `xy`, largest channel = 1: the level each
    /// channel is clipped at.
    pub fn camera_white(&self, white: Xy) -> [f64; 3] {
        let n = apply(&self.color_matrix, white.to_xyz());
        let max = n.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
        n.map(|v| (v / max).clamp(0.001, 1.0))
    }

    /// Camera → linear ProPhoto (D50) for a scene lit by the white `xy`:
    /// the camera white maps to the working white (1, 1, 1).
    pub fn camera_to_prophoto(&self, white: Xy) -> M3 {
        // Working white (D50 XYZ) → camera, scaled so the largest camera
        // channel of that white is 1.
        let mut pcs_to_camera = mul(&self.color_matrix, &adapt(Xy::D50, white));
        let w = apply(&pcs_to_camera, Xy::D50.to_xyz());
        let max = w.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
        pcs_to_camera = scale(&pcs_to_camera, 1.0 / max);
        let camera_to_pcs = invert(&pcs_to_camera).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let xyz_to_prophoto = invert(&prophoto_to_xyz()).expect("regular");
        mul(&xyz_to_prophoto, &camera_to_pcs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol }

    #[test]
    fn temperature_and_tint_round_trip() {
        for &(k, tint) in &[(2850.0, 0.0), (4000.0, 12.0), (5000.0, -8.0), (6500.0, 10.0), (9000.0, 30.0), (20000.0, -20.0)] {
            let xy = xy_from_temperature(k, tint);
            let (k2, t2) = temperature_from_xy(xy);
            assert!(close(k2, k, k * 2e-3) && close(t2, tint, 0.05), "{k} K {tint}: {k2} {t2}");
        }
        // Daylight whites sit a little to the green side of the locus.
        let (k, tint) = temperature_from_xy(Xy::D65);
        assert!(close(k, 6504.0, 20.0) && tint > 5.0 && tint < 12.0, "D65 = {k} K, tint {tint}");
        let (k, tint) = temperature_from_xy(Xy::D50);
        assert!(close(k, 5003.0, 20.0) && tint > 3.0 && tint < 12.0, "D50 = {k} K, tint {tint}");
        // The locus itself: the 6500 K black body.
        let p = xy_from_temperature(6500.0, 0.0);
        assert!(close(p.x, 0.3135, 1e-3) && close(p.y, 0.3237, 1e-3), "{p:?}");
    }

    #[test]
    fn matrices_invert_and_adapt() {
        let m = RawCamera::X6.color_matrix;
        let id = mul(&m, &invert(&m).unwrap());
        for i in 0..3 { for j in 0..3 { assert!(close(id[i][j], if i == j { 1.0 } else { 0.0 }, 1e-12)); } }
        // Adapting a white to itself changes nothing; D50 → D65 moves the white.
        let same = adapt(Xy::D65, Xy::D65);
        for i in 0..3 { for j in 0..3 { assert!(close(same[i][j], if i == j { 1.0 } else { 0.0 }, 1e-12)); } }
        let w = apply(&adapt(Xy::D50, Xy::D65), Xy::D50.to_xyz());
        let d65 = Xy::D65.to_xyz();
        for i in 0..3 { assert!(close(w[i], d65[i], 1e-9), "{w:?}"); }
        // ProPhoto white is Rec.709 white, and a saturated ProPhoto colour
        // is outside Rec.709.
        let white = apply(&prophoto_to_rec709(), [1.0, 1.0, 1.0]);
        for v in white { assert!(close(v, 1.0, 1e-12), "{white:?}"); }
        let green = apply(&prophoto_to_rec709(), [0.0, 1.0, 0.0]);
        assert!(green[0] < 0.0 && green[1] > 1.0, "{green:?}");
        assert!(close(prophoto_luma().iter().sum::<f64>(), 1.0, 1e-12));
    }

    #[test]
    fn the_camera_white_develops_to_white() {
        let cam = RawCamera::X6;
        for white in [cam.neutral_to_xy(cam.as_shot_neutral), Xy::D65, xy_from_temperature(3200.0, 5.0)] {
            let cw = cam.camera_white(white);
            assert!(close(cw.iter().cloned().fold(0.0, f64::max), 1.0, 1e-12));
            let rgb = apply(&cam.camera_to_prophoto(white), cw);
            for v in rgb { assert!(close(v, 1.0, 1e-9), "{white:?}: {rgb:?}"); }
        }
        // The as-shot neutral is its own camera white.
        let xy = cam.neutral_to_xy(cam.as_shot_neutral);
        let cw = cam.camera_white(xy);
        for i in 0..3 { assert!(close(cw[i], cam.as_shot_neutral[i], 1e-9), "{cw:?}"); }
        let (k, tint) = cam.as_shot_temperature();
        assert!(k > 3000.0 && k < 9000.0 && tint.abs() < 60.0, "reference neutral = {k} K, tint {tint}");
    }
}
