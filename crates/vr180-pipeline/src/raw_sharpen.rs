//! Sharpening of a raw clip's developed picture: the four controls of a raw
//! converter's detail panel — Amount, Radius, Detail, Masking — rendered as
//! Adobe Camera Raw renders them.
//!
//! It was measured there (Camera Raw 18.7, process version 15.4) on charts
//! and on frames of sample takes, by developing the same picture with and
//! without each setting:
//!
//! - It acts on the DEVELOPED picture, after every tone control, and on its
//!   luminance alone: the linear RGB of a pixel is scaled by one factor
//!   (the three channels' ratios agree to 3 parts in 10 000 on real frames).
//!   The luminance is worked on as `φ = Y^(1/1.8)`.
//! - On a hard edge the change is a fixed kernel for each Amount / Radius /
//!   Detail. It was read off the response to a small step for 14 Amounts ×
//!   9 Radii × 11 Details and is held here as a sum of five blurs:
//!   `Δ = Σ cⱼ (φ − Gⱼ ∗ φ)`, the `cⱼ` in the table (taps within 0.0015
//!   of the measured kernels). Between the measured settings the
//!   coefficients are interpolated.
//! - Low Detail holds halos back, and that is all it does to an edge: the
//!   sharpening is a stronger kernel (the step kernel over a factor `h`),
//!   and of its change the part that stays within the range of the
//!   original picture around the pixel is kept whole, while the part that
//!   leaves that range — the halo — is scaled by `h` ([`halo_change`]). A
//!   hard step is all halo, so it gets the step kernel; a soft edge, whose
//!   steepening stays between its two sides, gets up to `1 / h` of it.
//!   `h` is a third at Detail 0 for every Amount and Radius, a half to
//!   three quarters around Detail 5 to 10, and reaches one between Detail
//!   20 and 55 (later for a large Radius); it was fitted to edges of five
//!   softnesses at every setting ([`halo_factor`]).
//! - A large change is limited softly ([`soft_limit`]), a pixel near black
//!   or white keeps a share of it only ([`fade`]), and no pixel is
//!   darkened or lightened further than its own level allows (the limits
//!   of [`SharpenConstants::down`] / [`SharpenConstants::lift`], measured
//!   beside big edges at 44 levels).
//! - Masking keeps the sharpening to edges: the strength of the edge a
//!   pixel belongs to (the gradient's size, spread by a blur that widens
//!   with Radius) against a threshold that grows with Masking, through one
//!   curve ([`MaskConstants`]).
//!
//! What it does not do: with any Amount above zero Camera Raw also smooths
//! fine texture a little before it sharpens (an edge-preserving smoothing
//! that leaves steps alone and takes isolated extremes down; strongest at
//! Detail 0 and at large radii, gone by Detail 100). That is not here, so
//! at low Detail the finest texture and noise come out a little crisper
//! than there.
//!
//! The radius is in pixels of the frame an export writes. A preview drawn
//! smaller scales it ([`RawSharpen::constants`]'s `pixel_scale`), so it
//! shows the export's picture reduced, not a stronger sharpening.

use std::sync::OnceLock;

const TABLE: &[u8] = include_bytes!("data/raw_sharpen.f32");

/// The blurs the kernel is a sum of.
pub const BLURS: usize = 5;
/// Samples of the limit tables, over φ from 0 to 1.
pub const LIMIT_SAMPLES: usize = 129;
/// Samples of the mask curve, over log₂ of (edge strength / threshold).
pub const MASK_SAMPLES: usize = 97;
/// log₂ of the first and the last sample of the mask curve.
pub const MASK_LO: f32 = -4.0;
pub const MASK_HI: f32 = 8.0;
/// The widest blur a pass runs, taps to each side.
pub const MAX_RADIUS: usize = 31;
/// The widest neighbourhood the halo rule looks at, pixels to each side.
pub const MAX_WINDOW: u32 = 3;

/// The sharpening settings of a raw clip. `Default` is off.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RawSharpen {
    /// 0..150; 0 = off.
    pub amount: f32,
    /// 0.5..3.0, pixels of the exported frame.
    pub radius: f32,
    /// 0..100.
    pub detail: f32,
    /// 0..100.
    pub masking: f32,
}

impl Default for RawSharpen {
    fn default() -> Self {
        Self { amount: 0.0, radius: 1.0, detail: 25.0, masking: 0.0 }
    }
}

impl RawSharpen {
    /// Nothing is sharpened (the stage is skipped).
    pub fn is_off(&self) -> bool {
        self.amount < 0.5
    }
}

struct Tables {
    amounts: Vec<f32>,
    radii: Vec<f32>,
    details: Vec<f32>,
    sigmas: [f32; BLURS],
    /// `[amount][radius][detail][blur]`: the kernel of a hard step.
    coeff: Vec<f32>,
    /// `[amount][radius][detail]`: the share kept of a change that leaves
    /// the range of the original around the pixel.
    halo: Vec<f32>,
    levels: Vec<f32>,
    down: Vec<f32>,
    lift: Vec<f32>,
    mask_at: Vec<f32>,
    mask_threshold: Vec<f32>,
    mask_u: Vec<f32>,
    mask_g: Vec<f32>,
    mask_radii: Vec<f32>,
    mask_gain: Vec<f32>,
    gamma: f32,
    fade: [f32; 4],
    q: f32,
    soft: [f32; 4],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let v: Vec<f32> = TABLE.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let n: Vec<usize> = v[..8].iter().map(|x| *x as usize).collect();
        assert_eq!(n[3], BLURS, "sharpening table: blurs");
        let mut at = 8;
        let mut take = |count: usize| { let s = v[at..at + count].to_vec(); at += count; s };
        let (amounts, radii, details) = (take(n[0]), take(n[1]), take(n[2]));
        let sigmas: [f32; BLURS] = take(BLURS).try_into().expect("sigmas");
        let coeff = take(n[0] * n[1] * n[2] * BLURS);
        let halo = take(n[0] * n[1] * n[2]);
        let (levels, down, lift) = (take(n[4]), take(n[4]), take(n[4]));
        let (mask_at, mask_threshold) = (take(n[5]), take(n[5]));
        let (mask_u, mask_g) = (take(n[6]), take(n[6]));
        let (mask_radii, mask_gain) = (take(n[7]), take(n[7]));
        let c = take(10);
        assert_eq!(at, v.len(), "sharpening table: length");
        Tables {
            amounts, radii, details, sigmas, coeff, halo, levels, down, lift, mask_at, mask_threshold, mask_u, mask_g, mask_radii, mask_gain,
            gamma: c[0], fade: [c[1], c[2], c[3], c[4]], q: c[5], soft: [c[6], c[7], c[8], c[9]],
        }
    })
}

/// Where `v` falls on a rising grid: the node below and the share of the
/// way to the next (held at the ends).
fn locate(grid: &[f32], v: f32) -> (usize, f32) {
    let v = v.clamp(grid[0], grid[grid.len() - 1]);
    let i = grid.partition_point(|g| *g <= v).saturating_sub(1).min(grid.len() - 2);
    (i, (v - grid[i]) / (grid[i + 1] - grid[i]))
}

/// A table over a rising grid at `v`, linear between its nodes, held past
/// its ends.
fn interp(grid: &[f32], values: &[f32], v: f32) -> f32 {
    let (i, f) = locate(grid, v);
    values[i] * (1.0 - f) + values[i + 1] * f
}

/// The eight settings around an Amount, a Radius and a Detail: each one's
/// place in the tables (`(amount · radii + radius) · details + detail`)
/// and its weight (linear between the measured settings, held outside).
fn corners(amount: f32, radius: f32, detail: f32) -> [(usize, f32); 8] {
    let t = tables();
    let ((ia, fa), (ir, fr), (id, fd)) = (locate(&t.amounts, amount), locate(&t.radii, radius), locate(&t.details, detail));
    let (nr, nd) = (t.radii.len(), t.details.len());
    let mut out = [(0usize, 0f32); 8];
    let mut n = 0;
    for (da, wa) in [(0, 1.0 - fa), (1, fa)] {
        for (dr, wr) in [(0, 1.0 - fr), (1, fr)] {
            for (dd, wd) in [(0, 1.0 - fd), (1, fd)] {
                out[n] = (((ia + da) * nr + ir + dr) * nd + id + dd, wa * wr * wd);
                n += 1;
            }
        }
    }
    out
}

/// The blur coefficients of the kernel a hard step is sharpened with, for
/// an Amount, a Radius and a Detail.
pub fn coefficients(amount: f32, radius: f32, detail: f32) -> [f32; BLURS] {
    let t = tables();
    let mut out = [0f32; BLURS];
    for (at, w) in corners(amount, radius, detail) {
        for (j, o) in out.iter_mut().enumerate() {
            *o += w * t.coeff[at * BLURS + j];
        }
    }
    out
}

/// The share kept of a change that leaves the range of the original around
/// the pixel (see [`halo_change`]), for an Amount, a Radius and a Detail:
/// 1 where halos are not held back.
pub fn halo_factor(amount: f32, radius: f32, detail: f32) -> f32 {
    let t = tables();
    corners(amount, radius, detail).iter().map(|(at, w)| w * t.halo[*at]).sum()
}

/// One blur as its taps from the centre outward (`[0]` is the centre; the
/// kernel is symmetric and sums to one): the Gaussian of `sigma` sampled
/// at whole pixels out to three sigmas. Below half a pixel it is three
/// taps whose spread shrinks with the square of sigma — what the sampled
/// Gaussian of half a pixel continues into, so a preview's scaled blur
/// never jumps to nothing.
pub fn blur_taps(sigma: f32) -> Vec<f32> {
    if sigma < 0.5 {
        let side = SIDE_AT_HALF * (sigma / 0.5) * (sigma / 0.5);
        return vec![1.0 - 2.0 * side, side];
    }
    let radius = ((3.0 * sigma).ceil() as usize).min(MAX_RADIUS);
    let taps: Vec<f32> = (0..=radius).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let sum = taps[0] + 2.0 * taps[1..].iter().sum::<f32>();
    taps.iter().map(|t| t / sum).collect()
}

/// The side tap of the sampled Gaussian of half a pixel.
const SIDE_AT_HALF: f32 = 0.106_450_7;

/// Masking for one frame: the edge strength's blur and scale, the
/// threshold, the curve.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskConstants {
    /// The blur the gradient's size is spread by, pixels.
    pub sigma: f32,
    /// What the spread gradient is multiplied by: the pixels beside a step
    /// of height `s` then read `s` at Radius 1.
    pub gain: f32,
    /// The edge strength at which half the sharpening is kept.
    pub threshold: f32,
    /// The share kept over log₂(strength / threshold), [`MASK_SAMPLES`]
    /// from [`MASK_LO`] to [`MASK_HI`].
    pub curve: Vec<f32>,
}

/// Everything the stage needs for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SharpenConstants {
    /// The luminance is worked on as `Y^(1/gamma)`.
    pub gamma: f32,
    /// The blurs, pixels of the frame drawn.
    pub sigmas: [f32; BLURS],
    /// The whole kernel: the step kernel over `halo`.
    pub coeff: [f32; BLURS],
    /// The share kept of a change that leaves the range of the original
    /// within `window` pixels (1 = all of it).
    pub halo: f32,
    /// Pixels of the frame drawn, to each side.
    pub window: u32,
    /// The soft limit's scale and the mix of its two shapes.
    pub soft: f32,
    pub q: f32,
    /// The fades: none of the change at or below `[0]`, all of it from
    /// `[1]` up to `[2]`, none at or above `[3]`.
    pub fade: [f32; 4],
    /// The largest darkening and lightening of a pixel by its own level:
    /// [`LIMIT_SAMPLES`] over φ from 0 to 1.
    pub down: Vec<f32>,
    pub lift: Vec<f32>,
    pub mask: Option<MaskConstants>,
}

impl RawSharpen {
    /// The constants of these settings for a frame drawn at `pixel_scale`
    /// pixels to one pixel of the exported frame (1 for the export itself).
    /// `None` when nothing is sharpened.
    pub fn constants(&self, pixel_scale: f32) -> Option<SharpenConstants> {
        if self.is_off() {
            return None;
        }
        let t = tables();
        let scale = if pixel_scale.is_finite() && pixel_scale > 0.0 { pixel_scale } else { 1.0 };
        let (amount, radius, detail) = (self.amount.clamp(1.0, 150.0), self.radius.clamp(0.5, 3.0), self.detail.clamp(0.0, 100.0));
        let resample = |values: &[f32]| -> Vec<f32> {
            (0..LIMIT_SAMPLES).map(|i| interp(&t.levels, values, i as f32 / (LIMIT_SAMPLES - 1) as f32)).collect()
        };
        let masking = self.masking.clamp(0.0, 100.0);
        let mask = (masking > 0.0).then(|| {
            let sigma = (1.2 + 0.5 * radius) * scale;
            let taps = blur_taps(sigma);
            // What a unit step's two gradient samples leave beside the edge.
            let beside = 0.5 * (taps[0] + taps.get(1).copied().unwrap_or(0.0));
            let first = t.mask_at[0];
            let threshold = if masking >= first {
                let logs: Vec<f32> = t.mask_threshold.iter().map(|v| v.ln()).collect();
                interp(&t.mask_at, &logs, masking).exp()
            } else {
                t.mask_threshold[0] * masking / first
            };
            let logs: Vec<f32> = t.mask_u.iter().map(|u| u.log2()).collect();
            let curve = (0..MASK_SAMPLES).map(|i| {
                let l = MASK_LO + (MASK_HI - MASK_LO) * i as f32 / (MASK_SAMPLES - 1) as f32;
                if l <= logs[0] {
                    // Below the curve's first point the share falls with the square.
                    t.mask_g[0] * 4f32.powf(l - logs[0])
                } else {
                    interp(&logs, &t.mask_g, l)
                }
            }).collect();
            MaskConstants { sigma, gain: interp(&t.mask_radii, &t.mask_gain, radius) / beside, threshold, curve }
        });
        let halo = halo_factor(amount, radius, detail).clamp(0.05, 1.0);
        Some(SharpenConstants {
            gamma: t.gamma,
            sigmas: t.sigmas.map(|s| s * scale),
            coeff: coefficients(amount, radius, detail).map(|c| c / halo),
            halo,
            // One pixel of the exported frame; a frame drawn larger takes
            // as many of its own.
            window: (scale.round() as u32).clamp(1, MAX_WINDOW),
            soft: interp(&[0.0, 25.0, 50.0, 100.0], &t.soft, detail),
            q: t.q,
            fade: t.fade,
            down: resample(&t.down),
            lift: resample(&t.lift),
            mask,
        })
    }
}

/// The share of the sharpening a pixel of level `phi` keeps: less toward
/// black and toward white.
pub fn fade(phi: f32, k: &SharpenConstants) -> f32 {
    let [d0, d1, b0, b1] = k.fade;
    ((phi - d0) / (d1 - d0)).clamp(0.0, 1.0) * ((b1 - phi) / (b1 - b0)).clamp(0.0, 1.0)
}

/// The soft limit: a change `d` becomes `soft · shape(|d| / soft)`, which
/// is `d` for a small one and never passes `soft`.
pub fn soft_limit(d: f32, k: &SharpenConstants) -> f32 {
    let x = d.abs() / k.soft;
    d.signum() * k.soft * ((1.0 - k.q) * (1.0 - (-x).exp()) + k.q * x / (1.0 + x))
}

/// A limit table at `phi`.
fn table_at(table: &[f32], phi: f32) -> f32 {
    let x = phi.clamp(0.0, 1.0) * (table.len() - 1) as f32;
    let i = (x.floor() as usize).min(table.len() - 2);
    let f = x - i as f32;
    table[i] * (1.0 - f) + table[i + 1] * f
}

/// The share of the sharpening Masking keeps where the edge strength is
/// `strength`.
pub fn mask_share(strength: f32, m: &MaskConstants) -> f32 {
    let l = (strength.max(1e-12) / m.threshold).log2();
    let x = ((l - MASK_LO) / (MASK_HI - MASK_LO)).clamp(0.0, 1.0) * (MASK_SAMPLES - 1) as f32;
    let i = (x.floor() as usize).min(MASK_SAMPLES - 2);
    let f = x - i as f32;
    m.curve[i] * (1.0 - f) + m.curve[i + 1] * f
}

/// The kernel's change `delta` of a pixel of level `phi` with its halo
/// held back: the part that keeps the sharpened level between `lo` and `hi`
/// (the smallest and the largest level of the original around the pixel)
/// is kept, the part beyond is scaled by `halo`.
pub fn halo_change(phi: f32, delta: f32, lo: f32, hi: f32, halo: f32) -> f32 {
    let sharp = phi + delta;
    let held = sharp.clamp(lo, hi);
    (held - phi) + halo * (sharp - held)
}

/// What a pixel of level `phi` becomes, given the change `delta` (the
/// kernel's, its halo held back) and the share Masking keeps: the soft
/// limit, the fade, the mask, the limits of its own level.
pub fn sharpened_level(phi: f32, delta: f32, mask: f32, k: &SharpenConstants) -> f32 {
    let d = fade(phi, k) * mask * soft_limit(delta, k);
    (phi + d.clamp(-table_at(&k.down, phi), table_at(&k.lift, phi))).clamp(0.0, 1.0)
}

fn srgb_decode(e: f32) -> f32 {
    if e <= 0.040_45 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) }
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

/// Rec.709 luminance weights (the developed picture's primaries).
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// The darkest luminance the level is taken of.
pub const LUMA_FLOOR: f32 = 1e-6;

/// One plane blurred along both axes with the taps of [`blur_taps`]
/// (the picture's edge pixels repeat past its border).
fn blur_plane(plane: &[f32], w: usize, h: usize, taps: &[f32]) -> Vec<f32> {
    let r = taps.len() - 1;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = taps[0] * plane[y * w + x];
            for i in 1..=r {
                s += taps[i] * (plane[y * w + x.saturating_sub(i)] + plane[y * w + (x + i).min(w - 1)]);
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = taps[0] * tmp[y * w + x];
            for i in 1..=r {
                s += taps[i] * (tmp[y.saturating_sub(i) * w + x] + tmp[(y + i).min(h - 1) * w + x]);
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// The stage on a whole picture, on the CPU: `rgb` is the developed
/// picture (sRGB-encoded, `w × h`). The GPU stage's twin.
pub fn sharpen_picture(rgb: &[[f32; 3]], w: usize, h: usize, k: &SharpenConstants) -> Vec<[f32; 3]> {
    let linear: Vec<[f32; 3]> = rgb.iter().map(|p| p.map(srgb_decode)).collect();
    let phi: Vec<f32> = linear.iter().map(|p| (p[0] * LUMA[0] + p[1] * LUMA[1] + p[2] * LUMA[2]).max(LUMA_FLOOR).powf(1.0 / k.gamma)).collect();
    let mut delta = vec![0f32; w * h];
    for (sigma, c) in k.sigmas.iter().zip(&k.coeff) {
        let blurred = blur_plane(&phi, w, h, &blur_taps(*sigma));
        for ((d, p), b) in delta.iter_mut().zip(&phi).zip(&blurred) {
            *d += c * (p - b);
        }
    }
    // The range of the original around each pixel: rows, then columns.
    let r = k.window as usize;
    let range = |plane: &[f32], pick: fn(f32, f32) -> f32| -> Vec<f32> {
        let mut rows = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                rows[y * w + x] = (x.saturating_sub(r)..=(x + r).min(w - 1)).map(|i| plane[y * w + i]).fold(plane[y * w + x], pick);
            }
        }
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = (y.saturating_sub(r)..=(y + r).min(h - 1)).map(|i| rows[i * w + x]).fold(rows[y * w + x], pick);
            }
        }
        out
    };
    let (lo, hi) = (range(&phi, f32::min), range(&phi, f32::max));
    let strength = k.mask.as_ref().map(|m| {
        let mut size = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let gx = if x > 0 && x + 1 < w { 0.5 * (phi[y * w + x + 1] - phi[y * w + x - 1]) } else { 0.0 };
                let gy = if y > 0 && y + 1 < h { 0.5 * (phi[(y + 1) * w + x] - phi[(y - 1) * w + x]) } else { 0.0 };
                size[y * w + x] = (gx * gx + gy * gy).sqrt();
            }
        }
        blur_plane(&size, w, h, &blur_taps(m.sigma))
    });
    (0..w * h).map(|i| {
        let mask = match (&k.mask, &strength) {
            (Some(m), Some(s)) => mask_share(m.gain * s[i], m),
            _ => 1.0,
        };
        let out = sharpened_level(phi[i], halo_change(phi[i], delta[i], lo[i], hi[i], k.halo), mask, k);
        let gain = (out / phi[i]).powf(k.gamma);
        linear[i].map(|c| srgb_encode((c * gain).clamp(0.0, 1.0)))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of `phi` levels as a grey picture one row high... the stage
    /// works on pictures, so the row is repeated.
    fn grey_rows(levels: &[f32], gamma: f32, rows: usize) -> Vec<[f32; 3]> {
        let row: Vec<[f32; 3]> = levels.iter().map(|p| [srgb_encode(p.powf(gamma)); 3]).collect();
        (0..rows).flat_map(|_| row.clone()).collect()
    }

    fn levels_of(picture: &[[f32; 3]], w: usize, row: usize, gamma: f32) -> Vec<f32> {
        picture[row * w..(row + 1) * w].iter().map(|p| srgb_decode(p[1]).powf(1.0 / gamma)).collect()
    }

    #[test]
    fn the_table_at_and_between_its_settings() {
        // At a measured setting, and between settings (numbers of the
        // fit's own interpolation).
        let near = |a: [f32; BLURS], b: [f32; BLURS]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 2e-4 * (1.0 + y.abs()));
        assert!(near(coefficients(40.0, 1.0, 25.0), [-0.97132, 1.90921, 0.10352, -0.07381, 0.01971]), "{:?}", coefficients(40.0, 1.0, 25.0));
        assert!(near(coefficients(100.0, 2.0, 55.0), [6.60594, -5.35279, 10.906, -3.29471, 0.37018]), "{:?}", coefficients(100.0, 2.0, 55.0));
        assert!(near(coefficients(47.0, 1.1, 30.0), [-1.60164, 2.83213, 0.03987, -0.09157, 0.02694]), "{:?}", coefficients(47.0, 1.1, 30.0));
        assert!(near(coefficients(40.0, 1.0, 5.0), [-1.01699, 1.08285, 0.03357, -0.00296, 0.00034]), "{:?}", coefficients(40.0, 1.0, 5.0));
        // Held outside the measured range.
        assert_eq!(coefficients(400.0, 9.0, 300.0), coefficients(150.0, 3.0, 100.0));
        // The halo factor: a third at Detail 0 whatever the Amount, rising
        // with Detail, one where halos are not held back.
        for (a, r, d, want) in [(40.0f32, 1.0f32, 0.0f32, 0.3375f32), (40.0, 1.0, 10.0, 0.6625), (70.0, 2.0, 15.0, 0.7125), (47.0, 1.1, 7.0, 0.55225),
                                (5.0, 1.0, 0.0, 0.3375), (100.0, 1.0, 25.0, 0.6875), (40.0, 1.0, 25.0, 1.0), (40.0, 1.0, 55.0, 1.0), (150.0, 3.0, 100.0, 1.0)] {
            assert!((halo_factor(a, r, d) - want).abs() < 2e-4, "halo({a}, {r}, {d}) = {}", halo_factor(a, r, d));
        }
        // The constants carry the whole kernel: the step kernel over the factor.
        let k = RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }.constants(1.0).expect("on");
        assert!((k.halo - 0.3375).abs() < 2e-4 && k.window == 1);
        for (whole, step) in k.coeff.iter().zip(coefficients(40.0, 1.0, 0.0)) {
            assert!((whole * k.halo - step).abs() < 1e-5 * (1.0 + step.abs()));
        }
        // Off below half an Amount; on from there.
        assert!(RawSharpen::default().constants(1.0).is_none());
        assert!(RawSharpen { amount: 0.4, ..Default::default() }.constants(1.0).is_none());
        assert!(RawSharpen { amount: 1.0, ..Default::default() }.constants(1.0).is_some());
    }

    #[test]
    fn a_blur_is_its_taps() {
        // Half a pixel: the sampled Gaussian, and what the narrower blurs
        // continue from.
        let half = blur_taps(0.5);
        assert_eq!(half.len(), 3);
        assert!((half[1] - SIDE_AT_HALF).abs() < 1e-6 && half[2] < 1e-3, "{half:?}");
        let narrow = blur_taps(0.25);
        assert_eq!(narrow.len(), 2);
        assert!((narrow[1] - SIDE_AT_HALF / 4.0).abs() < 1e-7 && (narrow[0] + 2.0 * narrow[1] - 1.0).abs() < 1e-6);
        for sigma in [0.1f32, 0.5, 0.8, 1.3, 2.1, 3.4, 6.8] {
            let t = blur_taps(sigma);
            assert!((t[0] + 2.0 * t[1..].iter().sum::<f32>() - 1.0).abs() < 1e-5, "sigma {sigma}");
            assert_eq!(t.len() - 1, if sigma < 0.5 { 1 } else { (3.0 * sigma).ceil() as usize });
        }
    }

    /// The kernel an edge is sharpened with, as Camera Raw's own: the
    /// response to a line at Amount 40, Radius 1, Detail 25 was measured
    /// there as 1.7791, −0.3459, −0.0445, 0.0009.
    #[test]
    fn the_kernel_measured_on_camera_raw() {
        let c = coefficients(40.0, 1.0, 25.0);
        let t = tables();
        let mut kernel = [0f32; 6];
        kernel[0] = 1.0 + c.iter().sum::<f32>();
        for (cj, sigma) in c.iter().zip(&t.sigmas) {
            for (i, tap) in blur_taps(*sigma).iter().enumerate().take(6) {
                kernel[i] -= cj * tap;
            }
        }
        for (got, want) in kernel.iter().zip([1.7791f32, -0.3459, -0.0445, 0.0009]) {
            assert!((got - want).abs() < 2e-3, "{kernel:?}");
        }
    }

    /// Steps through the whole stage, against the numbers of the model
    /// the tables were fitted with.
    #[test]
    fn steps_come_out_as_the_model_says() {
        let step = |lo: f32, hi: f32| -> Vec<f32> { (0..32).map(|i| if i >= 16 { hi } else { lo }).collect() };
        let run = |s: RawSharpen, levels: &[f32]| -> Vec<f32> {
            let k = s.constants(1.0).expect("on");
            let out = sharpen_picture(&grey_rows(levels, k.gamma, 3), levels.len(), 3, &k);
            levels_of(&out, levels.len(), 1, k.gamma)
        };
        // A small step in the middle tones: the kernel alone.
        let got = run(RawSharpen { amount: 40.0, radius: 1.0, detail: 25.0, masking: 0.0 }, &step(0.40, 0.45));
        for (g, w) in got[13..19].iter().zip([0.400029f32, 0.397832, 0.381983, 0.468017, 0.452168, 0.449971]) {
            assert!((g - w).abs() < 2e-4, "{:?}", &got[13..19]);
        }
        // A big one at a strong setting: the soft limit and the limits of
        // each pixel's own level.
        let got = run(RawSharpen { amount: 100.0, radius: 1.0, detail: 100.0, masking: 0.0 }, &step(0.30, 0.80));
        for (g, w) in got[13..19].iter().zip([0.378839f32, 0.468796, 0.195526, 0.924766, 0.631204, 0.721161]) {
            assert!((g - w).abs() < 2e-3, "{:?}", &got[13..19]);
        }
        // Flat ground stays as it is, whatever the setting.
        let flat = run(RawSharpen { amount: 150.0, radius: 3.0, detail: 100.0, masking: 0.0 }, &[0.5; 32]);
        assert!(flat.iter().all(|v| (v - 0.5).abs() < 1e-5));
        let flat = run(RawSharpen { amount: 150.0, radius: 3.0, detail: 0.0, masking: 0.0 }, &[0.5; 32]);
        assert!(flat.iter().all(|v| (v - 0.5).abs() < 1e-5));
    }

    /// Low Detail: a hard step is sharpened as the step kernel says, a soft
    /// edge more, its steepening kept and only its halo held back. Against
    /// the model's numbers.
    #[test]
    fn low_detail_holds_the_halo_back_and_nothing_else() {
        let run = |s: RawSharpen, levels: &[f32]| -> Vec<f32> {
            let k = s.constants(1.0).expect("on");
            let out = sharpen_picture(&grey_rows(levels, k.gamma, 5), levels.len(), 5, &k);
            levels_of(&out, levels.len(), 2, k.gamma)
        };
        // 0.40 to 0.45 over four pixels.
        let ramp: Vec<f32> = (0..32).map(|i| 0.40 + 0.01 * (i as f32 - 13.0).clamp(0.0, 5.0)).collect();
        let got = run(RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }, &ramp);
        let want = [0.39999f32, 0.399981, 0.399769, 0.398829, 0.409339, 0.41997, 0.43003, 0.440661, 0.451171, 0.450231, 0.450019, 0.45001];
        for (g, w) in got[10..22].iter().zip(want) {
            assert!((g - w).abs() < 2e-4, "{:?}", &got[10..22]);
        }
        let got = run(RawSharpen { amount: 100.0, radius: 2.0, detail: 10.0, masking: 0.0 }, &ramp);
        let want = [0.399968f32, 0.399609, 0.397538, 0.391747, 0.405273, 0.419286, 0.430714, 0.444727, 0.458253, 0.452462, 0.450391, 0.450032];
        for (g, w) in got[10..22].iter().zip(want) {
            assert!((g - w).abs() < 3e-4, "{:?}", &got[10..22]);
        }
        // The ramp's inside is steepened by more than the step kernel alone
        // would (which is what it got before the rule): 1 / h of it.
        let k = RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }.constants(1.0).expect("on");
        let mut plain = k.clone();
        plain.coeff = coefficients(40.0, 1.0, 0.0);
        plain.halo = 1.0;
        let out = sharpen_picture(&grey_rows(&ramp, k.gamma, 5), 32, 5, &plain);
        let old = levels_of(&out, 32, 2, k.gamma);
        let got = run(RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }, &ramp);
        let (inside_new, inside_old) = (ramp[14] - got[14], ramp[14] - old[14]);
        assert!(inside_old > 1e-4 && (inside_new / inside_old - 1.0 / k.halo).abs() < 0.05, "{inside_new} {inside_old}");
        // A hard step is all halo: the same picture with the rule and without.
        let step: Vec<f32> = (0..32).map(|i| if i >= 16 { 0.45 } else { 0.40 }).collect();
        let with = run(RawSharpen { amount: 40.0, radius: 1.0, detail: 0.0, masking: 0.0 }, &step);
        let out = sharpen_picture(&grey_rows(&step, k.gamma, 5), 32, 5, &plain);
        let without = levels_of(&out, 32, 2, k.gamma);
        for (a, b) in with.iter().zip(&without) {
            assert!((a - b).abs() < 2e-5, "{with:?} {without:?}");
        }
        for (g, w) in with[13..19].iter().zip([0.39995f32, 0.398933, 0.395355, 0.454645, 0.451067, 0.45005]) {
            assert!((g - w).abs() < 2e-4, "{:?}", &with[13..19]);
        }
        // The rule itself.
        assert!((halo_change(0.5, 0.02, 0.4, 0.6, 0.3) - 0.02).abs() < 1e-7);
        assert!((halo_change(0.5, 0.2, 0.4, 0.6, 0.3) - (0.1 + 0.3 * 0.1)).abs() < 1e-6);
        assert!((halo_change(0.5, -0.3, 0.45, 0.6, 0.5) - (-0.05 - 0.5 * 0.25)).abs() < 1e-6);
        assert!((halo_change(0.5, 0.2, 0.5, 0.5, 1.0) - 0.2).abs() < 1e-7);
    }

    #[test]
    fn the_change_fades_toward_black_and_white() {
        let k = RawSharpen { amount: 40.0, ..Default::default() }.constants(1.0).expect("on");
        assert_eq!(fade(0.02, &k), 0.0);
        assert!((fade(0.116, &k) - 0.5).abs() < 0.01, "{}", fade(0.116, &k));
        assert_eq!(fade(0.5, &k), 1.0);
        assert!((fade(0.91, &k) - 0.5).abs() < 0.01, "{}", fade(0.91, &k));
        assert_eq!(fade(0.98, &k), 0.0);
        // The soft limit: a small change whole, a huge one held at the scale.
        assert!((soft_limit(1e-4, &k) - 1e-4).abs() < 1e-6);
        assert!(soft_limit(5.0, &k) < k.soft && soft_limit(5.0, &k) > 0.9 * k.soft);
        assert_eq!(soft_limit(-0.07, &k), -soft_limit(0.07, &k));
        // No darkening of a pixel in the deep shadows, a lightening only
        // by its fade; in the middle both, within the measured limits.
        assert_eq!(sharpened_level(0.08, -0.5, 1.0, &k), 0.08);
        assert!(sharpened_level(0.08, 0.5, 1.0, &k) > 0.08);
        let (down, up) = (0.5 - sharpened_level(0.5, -5.0, 1.0, &k), sharpened_level(0.5, 5.0, 1.0, &k) - 0.5);
        assert!((0.15..0.21).contains(&down) && (0.15..0.21).contains(&up), "{down} {up}");
    }

    /// Masking: at a step of 0.5 in φ, Radius 1, Masking 60, the model the
    /// tables came from keeps 0.958 beside the edge, 0.937 one pixel on
    /// and 0.858 two on.
    #[test]
    fn masking_keeps_the_sharpening_to_edges() {
        let k = RawSharpen { amount: 100.0, radius: 1.0, detail: 100.0, masking: 60.0 }.constants(1.0).expect("on");
        let m = k.mask.as_ref().expect("a mask");
        // The curve: half at the threshold, nothing far below, all far above.
        assert!((mask_share(m.threshold, m) - 0.5).abs() < 0.01);
        assert!(mask_share(m.threshold * 0.02, m) < 0.01 && mask_share(m.threshold * 100.0, m) > 0.99);
        // The edge strength of a step, pixel by pixel.
        let levels: Vec<f32> = (0..32).map(|i| if i >= 16 { 0.80 } else { 0.30 }).collect();
        let (w, h) = (32usize, 9usize);
        let phi: Vec<f32> = (0..h).flat_map(|_| levels.clone()).collect();
        let mut size = vec![0f32; w * h];
        for y in 0..h {
            for x in 1..w - 1 {
                size[y * w + x] = 0.5 * (phi[y * w + x + 1] - phi[y * w + x - 1]).abs();
            }
        }
        let strength = blur_plane(&size, w, h, &blur_taps(m.sigma));
        let shares: Vec<f32> = (13..19).map(|x| mask_share(m.gain * strength[4 * w + x], m)).collect();
        for (g, want) in shares.iter().zip([0.8577f32, 0.93712, 0.95766, 0.95766, 0.93712, 0.8577]) {
            assert!((g - want).abs() < 4e-3, "{shares:?}");
        }
        // Masking 0 is no mask; more Masking, a higher threshold.
        assert!(RawSharpen { amount: 40.0, ..Default::default() }.constants(1.0).expect("on").mask.is_none());
        let more = RawSharpen { amount: 100.0, masking: 90.0, ..Default::default() }.constants(1.0).expect("on").mask.expect("a mask");
        assert!(more.threshold > 2.0 * m.threshold);
    }

    /// A preview drawn smaller than the export scales the blurs, so it
    /// shows the export's picture reduced.
    #[test]
    fn a_smaller_frame_scales_the_blurs() {
        let s = RawSharpen { amount: 60.0, radius: 2.0, detail: 40.0, masking: 30.0 };
        let (full, half) = (s.constants(1.0).expect("on"), s.constants(0.5).expect("on"));
        assert_eq!(full.coeff, half.coeff);
        for (a, b) in full.sigmas.iter().zip(&half.sigmas) {
            assert!((b - 0.5 * a).abs() < 1e-6);
        }
        let (mf, mh) = (full.mask.expect("a mask"), half.mask.expect("a mask"));
        assert!((mh.sigma - 0.5 * mf.sigma).abs() < 1e-6 && mh.threshold == mf.threshold);
        // A scale that makes no sense is the export's own.
        assert_eq!(s.constants(f32::NAN).expect("on").sigmas, full.sigmas);
        // The halo rule's neighbourhood is one pixel of the export: one of
        // a smaller frame's too, as many of a larger one's.
        assert_eq!((full.window, half.window), (1, 1));
        assert_eq!((s.constants(2.0).expect("on").window, s.constants(2.6).expect("on").window, s.constants(40.0).expect("on").window), (2, 3, MAX_WINDOW));
    }

    /// Colour: the three channels of a pixel are scaled alike.
    #[test]
    fn a_pixel_keeps_its_colour() {
        let k = RawSharpen { amount: 80.0, ..Default::default() }.constants(1.0).expect("on");
        let (w, h) = (16usize, 5usize);
        let picture: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w >= 8 { [0.72, 0.45, 0.30] } else { [0.36, 0.30, 0.42] }).collect();
        let out = sharpen_picture(&picture, w, h, &k);
        for (a, b) in picture.iter().zip(&out) {
            let (la, lb) = (a.map(srgb_decode), b.map(srgb_decode));
            let gain = lb[1] / la[1];
            assert!((lb[0] / la[0] - gain).abs() < 1e-3 && (lb[2] / la[2] - gain).abs() < 1e-3);
        }
        // And the pixels beside the edge did change.
        assert!((out[2 * w + 8][1] - picture[2 * w + 8][1]).abs() > 0.01);
    }
}
