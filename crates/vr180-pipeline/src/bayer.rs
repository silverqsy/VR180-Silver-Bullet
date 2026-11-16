//! Demosaic of a Bayer mosaic: the CPU twin of the GPU passes
//! (`bayer_measure` / `bayer_green` / `bayer_demosaic.wgsl`,
//! [`crate::gpu::Device::demosaic_bayer`]).
//!
//! The method is shaped after Adobe Camera Raw's own demosaic, which was
//! measured: the same mosaics were developed there, its result taken back
//! to camera values, and candidate forms compared with it (three takes,
//! both eyes). What fits it:
//!
//! - It works on the mosaic WHITE-BALANCED: red and blue samples divided
//!   by the camera's neutral, so a grey edge carries no colour difference
//!   whatever its brightness.
//! - Green at a red or blue site is interpolated along the row or along
//!   the column — the two green neighbours' mean plus an EIGHTH of the
//!   site's own second difference — and the two estimates are mixed by how
//!   much the mosaic changes along each direction around the site (the
//!   slope across a site plus half its own bend, summed over 5×5): the
//!   row's share is `B² / (A² + B²)`.
//! - Red and blue are their difference from green, carried from their own
//!   sites to the others bilinearly, and added to green. A site keeps its
//!   own sample.
//!
//! Against Camera Raw's result in camera values this is two to five times
//! closer than the Malvar-He-Cutler kernels it replaces (root mean square
//! over three takes, of a level around 0.03: 0.0003 / 0.0002 / 0.0002 for
//! red / green / blue, from 0.0007 / 0.0005 / 0.0009), and its picture
//! shows the same edges: no zipper along them, no colour in them beyond
//! what Camera Raw's own shows before its noise reduction.

use crate::raw_gpu::BayerLayout;

/// The weight of a site's own second difference in its green estimate.
pub const CURVATURE: f32 = 0.125;
/// The measures are summed over a square of this many sites to each side.
pub const WINDOW: i32 = 2;
/// What keeps the mix defined on flat ground.
pub const FLOOR: f32 = 1e-5;

/// A coordinate past the border, mirrored (which keeps a site's colour).
#[inline]
fn mirror(i: i32, n: i32) -> usize {
    let k = if i < 0 { -i } else { i };
    let k = if k >= n { 2 * (n - 1) - k } else { k };
    k.clamp(0, n - 1) as usize
}

/// The demosaiced picture of a `w × h` mosaic: camera-linear RGB with the
/// sensor's white level at 1, row-major. `half` gives the picture at half
/// size, each pixel the mean of the 2×2 full-size pixels it covers.
///
/// # Panics
/// On a pattern that is no Bayer 2×2 or a mosaic of another size.
pub fn demosaic(mosaic: &[u16], w: usize, h: usize, layout: &BayerLayout, half: bool) -> Vec<[f32; 3]> {
    assert_eq!(mosaic.len(), w * h, "mosaic size");
    let (red_x, red_y) = layout.red_site().expect("a Bayer pattern");
    let scale = 1.0 / (layout.white - layout.black).max(1.0);
    let (wi, hi) = (w as i32, h as i32);
    let red_col = |x: i32| (x as u32 & 1) == red_x;
    let red_row = |y: i32| (y as u32 & 1) == red_y;
    let (gain_r, gain_b) = (1.0 / layout.neutral[0], 1.0 / layout.neutral[2]);
    // The white-balanced mosaic.
    let balanced: Vec<f32> = (0..w * h).map(|i| {
        let (x, y) = ((i % w) as i32, (i / w) as i32);
        let lin = (mosaic[i] as f32 - layout.black) * scale;
        if red_col(x) == red_row(y) { lin * if red_row(y) { gain_r } else { gain_b } } else { lin }
    }).collect();
    let bal = |x: i32, y: i32| balanced[mirror(y, hi) * w + mirror(x, wi)];
    // How much it changes along the row and along the column, at every site.
    let mut along = vec![[0f32; 2]; w * h];
    for y in 0..hi {
        for x in 0..wi {
            let c = bal(x, y);
            along[y as usize * w + x as usize] = [
                (bal(x - 1, y) - bal(x + 1, y)).abs() + 0.5 * (2.0 * c - bal(x - 2, y) - bal(x + 2, y)).abs(),
                (bal(x, y - 1) - bal(x, y + 1)).abs() + 0.5 * (2.0 * c - bal(x, y - 2) - bal(x, y + 2)).abs(),
            ];
        }
    }
    // Green.
    let cells = ((2 * WINDOW + 1) * (2 * WINDOW + 1)) as f32;
    let mut green = vec![0f32; w * h];
    for y in 0..hi {
        for x in 0..wi {
            let c = bal(x, y);
            green[y as usize * w + x as usize] = if red_col(x) != red_row(y) { c } else {
                let mut sum = [0f32; 2];
                for dy in -WINDOW..=WINDOW {
                    for dx in -WINDOW..=WINDOW {
                        let m = along[mirror(y + dy, hi) * w + mirror(x + dx, wi)];
                        sum[0] += m[0];
                        sum[1] += m[1];
                    }
                }
                let (a, b) = (sum[0] / cells + FLOOR, sum[1] / cells + FLOOR);
                let share = b * b / (a * a + b * b);
                let row = 0.5 * (bal(x - 1, y) + bal(x + 1, y)) + CURVATURE * (2.0 * c - bal(x - 2, y) - bal(x + 2, y));
                let col = 0.5 * (bal(x, y - 1) + bal(x, y + 1)) + CURVATURE * (2.0 * c - bal(x, y - 2) - bal(x, y + 2));
                share * row + (1.0 - share) * col
            };
        }
    }
    let grn = |x: i32, y: i32| green[mirror(y, hi) * w + mirror(x, wi)];
    let dif = |x: i32, y: i32| bal(x, y) - grn(x, y);
    // Red and blue.
    let full = |x: i32, y: i32| -> [f32; 3] {
        let g = grn(x, y);
        let (r, b) = if red_col(x) == red_row(y) {
            let diag = 0.25 * (dif(x - 1, y - 1) + dif(x + 1, y - 1) + dif(x - 1, y + 1) + dif(x + 1, y + 1));
            if red_row(y) { (bal(x, y), g + diag) } else { (g + diag, bal(x, y)) }
        } else {
            let beside = 0.5 * (dif(x - 1, y) + dif(x + 1, y));
            let above = 0.5 * (dif(x, y - 1) + dif(x, y + 1));
            if red_row(y) { (g + beside, g + above) } else { (g + above, g + beside) }
        };
        [(r * layout.neutral[0]).clamp(0.0, 1.0), g.clamp(0.0, 1.0), (b * layout.neutral[2]).clamp(0.0, 1.0)]
    };
    if !half {
        return (0..w * h).map(|i| full((i % w) as i32, (i / w) as i32)).collect();
    }
    let (ow, oh) = (w / 2, h / 2);
    (0..ow * oh).map(|i| {
        let (x, y) = (2 * (i % ow) as i32, 2 * (i / ow) as i32);
        let (p, q, r, s) = (full(x, y), full(x + 1, y), full(x, y + 1), full(x + 1, y + 1));
        [0, 1, 2].map(|c| 0.25 * (p[c] + q[c] + r[c] + s[c]))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const X6: BayerLayout = crate::x6_raw::BAYER;

    /// The mosaic of the model's own test: a slope, a bright bar and some
    /// texture, in a GBRG mosaic.
    fn model_mosaic(w: usize, h: usize) -> Vec<u16> {
        (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            let v = 1500 + 90 * x + 40 * y + if (6..9).contains(&y) { 2500 } else { 0 } + ((x * 7919 + y * 104_729 + x * y * 31) % 257) * 3;
            let site = if x & 1 == 0 && y & 1 == 1 { 0.62 } else if x & 1 == 1 && y & 1 == 0 { 0.45 } else { 1.0 };
            (64.0 + v as f64 * site).round() as u16
        }).collect()
    }

    /// The numbers of the model the method was fitted with (the script that
    /// compared it with Camera Raw's result).
    #[test]
    fn the_models_numbers() {
        let (w, h) = (24usize, 16usize);
        let m = model_mosaic(w, h);
        let out = demosaic(&m, w, h, &X6, false);
        for ((x, y), want) in [
            ((7usize, 5usize), [0.106571f32, 0.17414, 0.084221]), ((7, 6), [0.203073, 0.330792, 0.160779]), ((8, 6), [0.21711, 0.339872, 0.164849]),
            ((8, 7), [0.226994, 0.348493, 0.160807]), ((9, 8), [0.214857, 0.354017, 0.15159]), ((10, 9), [0.128389, 0.209031, 0.095181]),
            ((0, 0), [0.066575, 0.097758, 0.054262]), ((23, 15), [0.188102, 0.317323, 0.147827]), ((2, 3), [0.099778, 0.142673, 0.066098]),
            ((13, 12), [0.141134, 0.223096, 0.102776]),
        ] {
            let got = out[y * w + x];
            // The model works in double precision: where the two directions are close the mix differs in the fifth decimal.
            assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 5e-5), "({x}, {y}): {got:?}, want {want:?}");
        }
        let sum: f64 = out.iter().flat_map(|p| p.iter()).map(|v| *v as f64).sum();
        assert!((sum - 189.54541).abs() < 5e-3, "{sum}");
        // Half size: the mean of each 2×2.
        let half = demosaic(&m, w, h, &X6, true);
        assert_eq!(half.len(), w * h / 4);
        for ((x, y), want) in [((3usize, 3usize), [0.203137f32, 0.330479, 0.158289]), ((0, 0), [0.076775, 0.11717, 0.059186]), ((11, 7), [0.177955, 0.2912, 0.137122])] {
            let got = half[y * (w / 2) + x];
            assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 5e-5), "half ({x}, {y}): {got:?}, want {want:?}");
            let mean = [0, 1, 2].map(|c| 0.25 * (out[2 * y * w + 2 * x][c] + out[2 * y * w + 2 * x + 1][c] + out[(2 * y + 1) * w + 2 * x][c] + out[(2 * y + 1) * w + 2 * x + 1][c]));
            assert!(got.iter().zip(&mean).all(|(g, m)| (g - m).abs() < 1e-6));
        }
    }

    /// A camera value as a code of the X6's mosaic.
    fn code(v: f32) -> u16 {
        (X6.black + v * (X6.white - X6.black)).round() as u16
    }

    /// The camera values a grey of level `v` has at site (x, y).
    fn grey_site(v: f32, x: usize, y: usize) -> f32 {
        let (rx, ry) = X6.red_site().unwrap();
        let (rc, rr) = ((x as u32 & 1) == rx, (y as u32 & 1) == ry);
        if rc == rr { v * if rr { X6.neutral[0] } else { X6.neutral[2] } } else { v }
    }

    #[test]
    fn a_site_keeps_its_sample_and_grey_stays_grey() {
        let (w, h) = (32usize, 24usize);
        // Flat grey: every pixel is the grey's camera values.
        let flat: Vec<u16> = (0..w * h).map(|i| code(grey_site(0.2, i % w, i / w))).collect();
        for p in demosaic(&flat, w, h, &X6, false) {
            assert!((p[0] - 0.2 * X6.neutral[0]).abs() < 1e-4 && (p[1] - 0.2).abs() < 1e-4 && (p[2] - 0.2 * X6.neutral[2]).abs() < 1e-4, "{p:?}");
        }
        // Any mosaic: the colour a site measured comes out as measured.
        let m = model_mosaic(w, h);
        let out = demosaic(&m, w, h, &X6, false);
        let (rx, ry) = X6.red_site().unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = (m[y * w + x] as f32 - X6.black) / (X6.white - X6.black);
                let (rc, rr) = ((x as u32 & 1) == rx, (y as u32 & 1) == ry);
                let channel = if rc != rr { 1 } else if rr { 0 } else { 2 };
                assert!((out[y * w + x][channel] - v).abs() < 2e-6, "({x}, {y})");
            }
        }
    }

    /// A grey edge, along the rows or along the columns, hard or one site
    /// wide: no colour beside it and no alternation along it (what the
    /// kernels this replaced left there).
    #[test]
    fn a_grey_edge_has_no_colour_and_no_zipper() {
        let (w, h) = (40usize, 40usize);
        for vertical in [false, true] {
            for line in [false, true] {
                let level = |x: usize, y: usize| {
                    let t = if vertical { x } else { y };
                    if line { if t == 20 { 0.6 } else { 0.15 } } else if t >= 20 { 0.6 } else { 0.15 }
                };
                let m: Vec<u16> = (0..w * h).map(|i| code(grey_site(level(i % w, i / w), i % w, i / w))).collect();
                let out = demosaic(&m, w, h, &X6, false);
                let mut worst = 0f32;
                for y in 6..h - 6 {
                    for x in 6..w - 6 {
                        let p = out[y * w + x];
                        let v = level(x, y);
                        // Grey: red and blue in the neutral's ratio to green; and the level itself.
                        worst = worst.max((p[0] / X6.neutral[0] - p[1]).abs()).max((p[2] / X6.neutral[2] - p[1]).abs()).max((p[1] - v).abs());
                    }
                }
                assert!(worst < 2e-3, "vertical {vertical} line {line}: {worst}");
            }
        }
    }

    #[test]
    fn the_cameras_neutral_is_the_raw_cameras() {
        let n = crate::raw_color::RawCamera::X6.as_shot_neutral;
        assert_eq!(X6.neutral, [n[0] as f32, n[1] as f32, n[2] as f32]);
        assert_eq!(X6.red_site(), Some((0, 1)));
    }
}
