//! Colour noise reduction of a raw clip: the picture's colour smoothed, its
//! luminance kept — a raw converter's "Color" noise reduction, after Adobe
//! Camera Raw's.
//!
//! Measured there (Camera Raw 18.7, on charts and on frames of sample
//! takes): the control leaves the picture's luminance alone and takes its
//! colour's fine part away — at the default 25, to a fifth to a third of
//! its amplitude at a quarter of a cycle a pixel, about a half to three
//! quarters at a tenth. It is a threshold, not a blur: a small colour
//! variation goes at every scale, a large one (a real colour edge) stays.
//! A grating in one camera channel comes out, at 100, as a change of grey
//! of 0.195 / 0.667 / 0.138 of it for red / green / blue: the luminance it
//! keeps ([`LUMA`], on the white-balanced camera values).
//!
//! What is done here, in white-balanced camera values `w` (camera values
//! over the camera's neutral) with `L = LUMA · w`, ahead of the
//! development:
//!
//! 1. the picture at half size (the mean of each 2×2);
//! 2. there, along the rows and then along the columns, every pixel takes
//!    the weighted mean of its neighbours' `w`: a Gaussian of half the
//!    radius, times a Gaussian (of `tau`) of how far the neighbour's colour
//!    `w / L` is from the pixel's own — so a colour edge is not smeared —
//!    times a Gaussian (of [`LEVEL_SIGMA`] stops) of how far its luminance
//!    is from the pixel's own — so no colour crosses an edge of light —
//!    and the mean is put back at the pixel's own luminance;
//! 3. back at full size (bilinear), every pixel takes that smoothed colour
//!    at its OWN luminance: `w' = s · L / (LUMA · s)`.
//!
//! The radius and `tau` follow the amount ([`settings`]); they were fitted
//! to Camera Raw's result, taken back to camera values, on frames of a
//! sample take and checked on another. Up to the default the result is
//! close to its floor (what the two demosaics differ by): at 25 the
//! remaining difference is 0.9 / 0.33 / 0.76 thousandths of the white
//! level against 1.77 / 0.49 / 1.51 without the stage and a floor of
//! 0.75 / 0.24 / 0.45. Camera Raw reaches further at high amounts — it
//! flattens faint colour over tens of pixels — than one pass at half size
//! does: at 100 a third to a half of its change is still missing.
//!
//! **The edge of light.** The mean is of values, so a bright neighbour
//! weighs by its brightness: without the luminance term, leaves against
//! the sky (five stops under it, and not far from it in colour as the
//! sensor sees them) took the sky's colour over the whole radius, and with
//! the highlights pulled down a tree stood in a pale blue haze. Measured on
//! charts (two colours side by side, the darker one 0 to 6 stops under the
//! other): Camera Raw carries no colour across a step of a stop or more
//! beyond the first four pixels, at any amount, and what it carries across
//! a smaller step falls as a Gaussian of about 0.2 stops of the step. On
//! the sample takes' frames a term of 0.3 to 0.5 stops also brings the
//! stage nearer Camera Raw's result at 10 to 50 (0.83 / 0.31 / 0.70 at 25
//! where it was 0.90 / 0.33 / 0.76) and leaves 100 about where it was.

/// The luminance the stage keeps: weights of the white-balanced camera
/// values.
pub const LUMA: [f32; 3] = [0.195, 0.667, 0.138];
/// The darkest luminance a colour is taken of.
pub const FLOOR: f32 = 1e-4;
/// The widest smoothing a pass runs, taps to each side.
pub const MAX_RADIUS: usize = 32;
/// How near in luminance a neighbour has to be to count, stops (a
/// Gaussian's sigma): colour is smoothed within a level of light, not
/// across an edge of it.
pub const LEVEL_SIGMA: f32 = 0.3;
/// Amount → the smoothing's radius (a Gaussian's sigma, pixels of the
/// exported frame) and the colour distance `tau` it weighs neighbours by.
const TABLE: [(f32, f32, f32); 5] = [(0.0, 0.0, 0.0), (10.0, 4.0, 0.1), (25.0, 6.0, 0.2), (50.0, 12.0, 0.3), (100.0, 16.0, 0.5)];

/// The radius (sigma, pixels of the exported frame) and `tau` of an amount
/// (0..100), linear between the fitted amounts.
pub fn settings(amount: f32) -> (f32, f32) {
    let a = amount.clamp(0.0, 100.0);
    let i = TABLE.partition_point(|t| t.0 <= a).clamp(1, TABLE.len() - 1);
    let (lo, hi) = (TABLE[i - 1], TABLE[i]);
    let f = (a - lo.0) / (hi.0 - lo.0);
    (lo.1 + f * (hi.1 - lo.1), lo.2 + f * (hi.2 - lo.2))
}

/// Everything the stage needs for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaConstants {
    /// The smoothing's sigma in pixels of the HALF-size picture.
    pub sigma: f32,
    pub tau: f32,
    /// The camera's neutral the values are balanced by.
    pub neutral: [f32; 3],
}

/// The constants of an amount for a frame drawn at `pixel_scale` pixels to
/// one pixel of the exported frame (1 for the export itself). `None` when
/// the amount is zero.
pub fn constants(amount: f32, pixel_scale: f32, neutral: [f32; 3]) -> Option<ChromaConstants> {
    let (sigma, tau) = settings(amount);
    if !(sigma > 0.0 && tau > 0.0) {
        return None;
    }
    let scale = if pixel_scale.is_finite() && pixel_scale > 0.0 { pixel_scale } else { 1.0 };
    Some(ChromaConstants { sigma: 0.5 * sigma * scale, tau, neutral: neutral.map(|n| n.max(1e-3)) })
}

impl ChromaConstants {
    /// The spatial weights from the centre outward (`[0]` is the centre):
    /// the Gaussian of `sigma` at whole pixels out to three sigmas. They
    /// are not normalised: a pass divides by what its weights sum to.
    pub fn taps(&self) -> Vec<f32> {
        let radius = ((3.0 * self.sigma).ceil() as usize).clamp(1, MAX_RADIUS);
        (0..=radius).map(|i| (-((i * i) as f32) / (2.0 * self.sigma * self.sigma)).exp()).collect()
    }
}

fn luminance(w: [f32; 3]) -> f32 {
    (LUMA[0] * w[0] + LUMA[1] * w[1] + LUMA[2] * w[2]).max(FLOOR)
}

/// One pass along an axis of a `w × h` picture of balanced values: the
/// weighted sum of each pixel's neighbours (the border repeated), put back
/// at the pixel's own luminance when `own_level`.
fn smooth(src: &[[f32; 3]], w: usize, h: usize, taps: &[f32], tau: f32, vertical: bool, own_level: bool) -> Vec<[f32; 3]> {
    let range = 1.0 / (2.0 * tau * tau);
    let level_range = 1.0 / (2.0 * LEVEL_SIGMA * LEVEL_SIGMA);
    let r = taps.len() as isize - 1;
    (0..w * h).map(|i| {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        let c = src[i];
        let lc = luminance(c);
        let qc = c.map(|v| v / lc);
        let mut sum = c.map(|v| taps[0] * v);
        for d in (-r..=r).filter(|d| *d != 0) {
            let (nx, ny) = if vertical { (x, (y + d).clamp(0, h as isize - 1)) } else { ((x + d).clamp(0, w as isize - 1), y) };
            let n = src[ny as usize * w + nx as usize];
            let ln = luminance(n);
            let dist: f32 = (0..3).map(|k| { let e = n[k] / ln - qc[k]; e * e }).sum();
            let step = (ln / lc).log2();
            let wt = taps[d.unsigned_abs()] * (-dist * range - step * step * level_range).exp();
            for k in 0..3 {
                sum[k] += wt * n[k];
            }
        }
        if own_level {
            let g = lc / luminance(sum);
            sum.map(|v| v * g)
        } else {
            sum
        }
    }).collect()
}

/// The stage on a whole picture, on the CPU: `cam` is the camera-linear
/// picture (`w × h`, white level at 1). The GPU stage's twin.
pub fn denoise_picture(cam: &[[f32; 3]], w: usize, h: usize, k: &ChromaConstants) -> Vec<[f32; 3]> {
    let balanced: Vec<[f32; 3]> = cam.iter().map(|p| [p[0] / k.neutral[0], p[1] / k.neutral[1], p[2] / k.neutral[2]]).collect();
    // Half size: the mean of each 2×2 (the last row or column repeated on an odd side).
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    let small: Vec<[f32; 3]> = (0..hw * hh).map(|i| {
        let (x, y) = (2 * (i % hw), 2 * (i / hw));
        let at = |dx: usize, dy: usize| balanced[(y + dy).min(h - 1) * w + (x + dx).min(w - 1)];
        let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
        [0, 1, 2].map(|j| 0.25 * (a[j] + b[j] + c[j] + d[j]))
    }).collect();
    let taps = k.taps();
    let rows = smooth(&small, hw, hh, &taps, k.tau, false, true);
    let both = smooth(&rows, hw, hh, &taps, k.tau, true, false);
    // Back at full size, at each pixel's own luminance.
    (0..w * h).map(|i| {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        let (px, py) = ((x - 0.5) * 0.5, (y - 0.5) * 0.5);
        let (fx, fy) = (px - px.floor(), py - py.floor());
        let ix = |d: f32| (px.floor() + d).clamp(0.0, hw as f32 - 1.0) as usize;
        let iy = |d: f32| (py.floor() + d).clamp(0.0, hh as f32 - 1.0) as usize;
        let s = |a: usize, b: usize| both[b * hw + a];
        let (p00, p10, p01, p11) = (s(ix(0.0), iy(0.0)), s(ix(1.0), iy(0.0)), s(ix(0.0), iy(1.0)), s(ix(1.0), iy(1.0)));
        let up = [0, 1, 2].map(|j| (p00[j] * (1.0 - fx) + p10[j] * fx) * (1.0 - fy) + (p01[j] * (1.0 - fx) + p11[j] * fx) * fy);
        let g = luminance(balanced[i]) / luminance(up);
        [0, 1, 2].map(|j| (up[j] * g * k.neutral[j]).clamp(0.0, 1.0))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEUTRAL: [f32; 3] = [0.565_429_7, 1.0, 0.401_367_2];

    /// The picture of the model's own test (the script the amounts were
    /// fitted with).
    fn model_picture(w: usize, h: usize) -> Vec<[f32; 3]> {
        (0..w * h).map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let lum = 1.0 + 0.15 * (x * 1.7 + y * 2.3).sin();
            let r = (0.10 + 0.01 * x + if y > 5.0 { 0.05 } else { 0.0 }) * lum * (1.0 + 0.2 * (x * 2.9 + y).cos());
            let g = (0.20 + 0.012 * x) * lum;
            let b = (0.07 + 0.004 * y + if x > 9.0 { 0.06 } else { 0.0 }) * lum * (1.0 + 0.2 * (x * 0.7 - y * 3.1).sin());
            [r as f32, g as f32, b as f32]
        }).collect()
    }

    #[test]
    fn the_amounts_settings() {
        let near = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-6;
        assert!(near(settings(25.0), (6.0, 0.2)) && near(settings(40.0), (9.6, 0.26)) && near(settings(5.0), (2.0, 0.05)));
        assert!(near(settings(100.0), (16.0, 0.5)) && near(settings(400.0), (16.0, 0.5)) && near(settings(-3.0), (0.0, 0.0)));
        // Off at zero; the radius is the half-size picture's, and follows the frame's scale.
        assert!(constants(0.0, 1.0, NEUTRAL).is_none());
        let k = constants(25.0, 1.0, NEUTRAL).expect("on");
        assert!((k.sigma - 3.0).abs() < 1e-6 && (k.tau - 0.2).abs() < 1e-6);
        assert!((constants(25.0, 0.5, NEUTRAL).unwrap().sigma - 1.5).abs() < 1e-6);
        assert_eq!(constants(25.0, f32::NAN, NEUTRAL).unwrap().sigma, k.sigma);
        assert_eq!(k.taps().len(), 10);
        assert_eq!(constants(100.0, 8.0, NEUTRAL).unwrap().taps().len(), MAX_RADIUS + 1);
    }

    /// Against the numbers of the model the amounts were fitted with.
    #[test]
    fn the_models_numbers() {
        let (w, h) = (16usize, 12usize);
        let cam = model_picture(w, h);
        let k = constants(25.0, 1.0, NEUTRAL).unwrap();
        let out = denoise_picture(&cam, w, h, &k);
        for ((x, y), before, after) in [
            ((0usize, 0usize), [0.12f32, 0.2, 0.07], [0.107707f32, 0.20496, 0.072709]),
            ((4, 3), [0.144825, 0.281702, 0.089136], [0.161636, 0.271879, 0.091328]),
            ((9, 5), [0.25988, 0.353985, 0.098826], [0.238559, 0.35023, 0.127497]),
            ((10, 6), [0.187067, 0.29227, 0.163802], [0.215394, 0.298392, 0.123512]),
            ((15, 11), [0.293222, 0.409037, 0.224728], [0.317617, 0.41593, 0.186888]),
            ((2, 8), [0.186413, 0.230384, 0.125614], [0.177668, 0.241852, 0.112138]),
        ] {
            let (got_in, got) = (cam[y * w + x], out[y * w + x]);
            assert!(got_in.iter().zip(&before).all(|(a, b)| (a - b).abs() < 2e-6), "({x}, {y}) in: {got_in:?}");
            assert!(got.iter().zip(&after).all(|(a, b)| (a - b).abs() < 2e-5), "({x}, {y}): {got:?}, want {after:?}");
        }
        let sum: f64 = out.iter().flat_map(|p| p.iter()).map(|v| *v as f64).sum();
        assert!((sum - 116.172062).abs() < 2e-3, "{sum}");
    }

    #[test]
    fn luminance_stays_and_colour_noise_goes() {
        let (w, h) = (64usize, 48usize);
        let mut s = 7u32;
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s % 10_000) as f32 / 10_000.0 - 0.5 };
        // A grey slope with noise of its own in each channel.
        let cam: Vec<[f32; 3]> = (0..w * h).map(|i| {
            let v = 0.1 + 0.3 * (i % w) as f32 / w as f32;
            [v * NEUTRAL[0] * (1.0 + 0.2 * next()), v * (1.0 + 0.2 * next()), v * NEUTRAL[2] * (1.0 + 0.2 * next())]
        }).collect();
        let k = constants(25.0, 1.0, NEUTRAL).unwrap();
        let out = denoise_picture(&cam, w, h, &k);
        let lum = |p: &[f32; 3]| luminance([p[0] / NEUTRAL[0], p[1], p[2] / NEUTRAL[2]]);
        let colour = |p: &[f32; 3]| { let l = lum(p); (p[0] / NEUTRAL[0] / l - 1.0).abs() + (p[2] / NEUTRAL[2] / l - 1.0).abs() };
        let (mut before, mut after) = (0f32, 0f32);
        for (a, b) in cam.iter().zip(&out) {
            assert!((lum(a) - lum(b)).abs() < 1e-5 * (1.0 + lum(a)), "luminance moved: {a:?} {b:?}");
            before += colour(a);
            after += colour(b);
        }
        assert!(after < 0.3 * before, "colour noise {before} -> {after}");
        // Plain grey is left as it is.
        let grey = vec![[0.3 * NEUTRAL[0], 0.3, 0.3 * NEUTRAL[2]]; w * h];
        for p in denoise_picture(&grey, w, h, &k) {
            assert!((p[0] - 0.3 * NEUTRAL[0]).abs() < 1e-6 && (p[1] - 0.3).abs() < 1e-6 && (p[2] - 0.3 * NEUTRAL[2]).abs() < 1e-6);
        }
    }

    /// No colour crosses an edge of light: leaves beside a sky five stops
    /// brighter, the two not far apart in colour as the sensor sees them
    /// (a take's own values). A mean of values lets the bright side weigh
    /// by its brightness: the leaves took the sky's colour over the whole
    /// radius. They keep their own beyond the first pixels of the edge, as
    /// they do in Camera Raw; the same two colours at one level of light
    /// are still smoothed into each other.
    #[test]
    fn a_bright_neighbour_does_not_colour_a_dark_pixel() {
        let (w, h) = (96usize, 16usize);
        let (sky, leaf) = ([0.217f32, 0.761, 0.597], [0.005f32, 0.016, 0.010]);
        let k = constants(25.0, 1.0, NEUTRAL).unwrap();
        let ratios = |p: [f32; 3]| [p[0] / p[1], p[2] / p[1]];
        let cam: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w < 48 { sky } else { leaf }).collect();
        let out = denoise_picture(&cam, w, h, &k);
        for y in 0..h {
            for x in (0..44).chain(52..w) {
                let (want, got) = (ratios(cam[y * w + x]), ratios(out[y * w + x]));
                assert!((got[0] - want[0]).abs() < 0.004 && (got[1] - want[1]).abs() < 0.004, "({x}, {y}): {got:?} for {want:?}");
            }
        }
        // At one level of light the same two colours do go toward each
        // other beside their border (a tenth of their distance and more):
        // it is the edge of light that kept them apart, not their colours.
        let level = |p: [f32; 3], to: f32| { let l = luminance([p[0] / NEUTRAL[0], p[1], p[2] / NEUTRAL[2]]); p.map(|v| v * to / l) };
        let (a, b) = (level(sky, 0.2), level(leaf, 0.2));
        let cam: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w < 48 { a } else { b }).collect();
        let out = denoise_picture(&cam, w, h, &k);
        let (left, right) = (ratios(out[8 * w + 46])[1], ratios(out[8 * w + 49])[1]);
        let (own_left, own_right) = (ratios(a)[1], ratios(b)[1]);
        let apart = own_left - own_right;
        assert!(apart > 0.1 && own_left - left > 0.1 * apart && right - own_right > 0.1 * apart, "{left} {right} of {own_left} {own_right}");
    }

    /// A real colour edge is not smeared: red beside blue, far apart in
    /// colour, keep their colours up to the pixels at the edge.
    #[test]
    fn a_colour_edge_stays() {
        let (w, h) = (48usize, 16usize);
        let (red, blue) = ([0.30f32, 0.10, 0.02], [0.03f32, 0.10, 0.20]);
        let cam: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w < 24 { red } else { blue }).collect();
        let out = denoise_picture(&cam, w, h, &constants(25.0, 1.0, NEUTRAL).unwrap());
        for y in 0..h {
            for x in (0..22).chain(26..w) {
                let (want, got) = (cam[y * w + x], out[y * w + x]);
                assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 0.01 * (a + 0.05)), "({x}, {y}): {got:?}");
            }
        }
    }
}
