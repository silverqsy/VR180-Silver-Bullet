//! The local part of raw development (Shadows / Highlights) on the CPU:
//! the base — a bilateral grid of the frame's log₂ luminance, the twin of
//! `shaders/raw_reduce.wgsl`, `raw_grid_build.wgsl`, `raw_grid_blur.wgsl`
//! and the lookup in `raw_gain.wgsl` —, the smoothing of the gain that base
//! gives (`gain_cells.wgsl` and the read in `raw_develop_local.wgsl`), and
//! the frame's curves.
//!
//! See [`crate::raw_tone`] for what the stage does with them.

use crate::raw_tone::{
    LocalCurves, BASE_CELL, BASE_FALLBACK, BASE_FLOOR, BASE_LEVELS, BASE_LO, BASE_POWER, BASE_RANGE, GAIN_CELL, LOCAL_LO, LOCAL_PER_STOP,
    LOCAL_SAMPLES,
};

/// Pixels of an eye per cell of the smoothed gain — and per texel of the
/// reduced copy the grid is gathered from — for a picture drawn at
/// `pixel_scale` pixels to one pixel of the exported frame (1 for the export
/// itself): [`GAIN_CELL`] pixels of the exported frame.
pub fn gain_cell(pixel_scale: f32) -> u32 {
    let scale = if pixel_scale.is_finite() && pixel_scale > 0.0 { pixel_scale } else { 1.0 };
    ((GAIN_CELL * scale).round() as u32).clamp(1, 16)
}

/// The grid's cell, in texels of a reduced copy of `n` pixels per texel,
/// for a frame whose long side is `long` pixels: [`BASE_CELL`] of the long
/// side, one texel at least.
pub fn base_cell(long: u32, n: u32) -> f32 {
    (BASE_CELL * long as f32 / n.max(1) as f32).max(1.0)
}

/// The weight of the plain local average in a grid whose cells are `cell`
/// texels wide ([`BASE_FALLBACK`] of the texels a cell holds).
pub fn base_fallback(cell: f32) -> f32 {
    BASE_FALLBACK * cell * cell
}

/// The grid's size for a reduced copy of `rw × rh` texels and cells of
/// `cell` texels.
pub fn grid_dims(rw: u32, rh: u32, cell: f32) -> (u32, u32, u32) {
    ((rw as f32 / cell).ceil() as u32 + 2, (rh as f32 / cell).ceil() as u32 + 2, BASE_LEVELS)
}

/// The reduced copy: log₂ of the power mean ([`BASE_POWER`]) of the
/// luminance over every `n × n` pixels (the last row / column of blocks
/// repeats the edge pixels). A mean of light, not of levels: a block of
/// bright and dark counts nearer its bright part, as a mixture too fine to
/// be seen apart does.
pub fn reduce_levels(luminance: &[f32], w: usize, h: usize, n: usize) -> (Vec<f32>, usize, usize) {
    let (rw, rh) = (w.div_ceil(n), h.div_ceil(n));
    let mut out = vec![0f32; rw * rh];
    for ry in 0..rh {
        for rx in 0..rw {
            let mut sum = 0f32;
            for j in 0..n {
                let y = (ry * n + j).min(h - 1);
                for i in 0..n {
                    let x = (rx * n + i).min(w - 1);
                    sum += luminance[y * w + x].max(BASE_FLOOR).powf(BASE_POWER);
                }
            }
            out[ry * rw + rx] = (sum / (n * n) as f32).log2() / BASE_POWER;
        }
    }
    (out, rw, rh)
}

/// The means of a picture's gain (`w × h`, stops) over cells of `c × c`
/// pixels (the last row / column of cells repeats the edge pixels).
pub fn gain_cells(gain: &[f32], w: usize, h: usize, c: usize) -> (Vec<f32>, usize, usize) {
    let (cw, ch) = (w.div_ceil(c), h.div_ceil(c));
    let mut out = vec![0f32; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let mut sum = 0f32;
            for j in 0..c {
                let y = (cy * c + j).min(h - 1);
                for i in 0..c {
                    sum += gain[y * w + (cx * c + i).min(w - 1)];
                }
            }
            out[cy * cw + cx] = sum / (c * c) as f32;
        }
    }
    (out, cw, ch)
}

/// Pixels to either side over which a pixel's own gain is smoothed for the
/// limit of [`smooth_gain`], for cells of `c` pixels: the binomial kernel
/// of `2r + 1` pixels a side.
pub fn limit_radius(c: usize) -> usize {
    (c / 2).min(2)
}

/// The binomial weight of the pixel `d` away, of `2r + 1` (`r` ≤ 2).
pub fn limit_weight(d: isize, r: usize) -> f32 {
    match (r, d.unsigned_abs()) {
        (2, 0) => 6.0 / 16.0,
        (2, 1) => 4.0 / 16.0,
        (2, 2) => 1.0 / 16.0,
        (1, 0) => 0.5,
        (1, 1) => 0.25,
        (0, 0) => 1.0,
        _ => 0.0,
    }
}

/// The gain (stops) the picture gives the pixel `(x, y)`: the mean of the
/// gain over the cells ([`gain_cells`]: `cells`, `cw × ch`, of `c` pixels),
/// linear between the cells' centres, held between zero and the pixel's own
/// gain lightly smoothed ([`limit_radius`]). So a pixel takes what the few
/// pixels round it take together — a bright one among dark ones the share
/// of bright among them — but never more than itself (and what touches it)
/// would, and never the other way: beside an edge the side the gain moves
/// fades toward it, the other side is left alone.
#[allow(clippy::too_many_arguments)]
pub fn smooth_gain(gain: &[f32], w: usize, h: usize, cells: &[f32], cw: usize, ch: usize, c: usize, x: usize, y: usize) -> f32 {
    let between = |v: usize, n: usize| {
        let f = ((v as f32 + 0.5) / c as f32 - 0.5).clamp(0.0, n as f32 - 1.0);
        let i = (f.floor() as usize).min(n.saturating_sub(2));
        (i, (i + 1).min(n - 1), f - i as f32)
    };
    let ((x0, x1, tx), (y0, y1, ty)) = (between(x, cw), between(y, ch));
    let at = |cx: usize, cy: usize| cells[cy * cw + cx];
    let smooth = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
    let r = limit_radius(c) as isize;
    let mut own = 0f32;
    for dy in -r..=r {
        let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
        for dx in -r..=r {
            let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
            own += limit_weight(dx, r as usize) * limit_weight(dy, r as usize) * gain[yy * w + xx];
        }
    }
    smooth.clamp(own.min(0.0), own.max(0.0))
}

/// A frame's bilateral grid: per vertex `(Σ w·(level − lo), Σ w, column mean − lo)`.
#[derive(Debug, Clone)]
pub struct BaseGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell in texels of the reduced copy.
    pub cell: f32,
    pub data: Vec<[f32; 3]>,
}

impl BaseGrid {
    fn at(&self, x: isize, y: isize, z: isize) -> [f32; 3] {
        let c = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
        self.data[(c(z, self.nz) * self.ny + c(y, self.ny)) * self.nx + c(x, self.nx)]
    }

    /// Gather the reduced copy (`rw × rh` log₂ levels) into a grid of
    /// cells of `cell` texels, then smooth it.
    pub fn build(reduced: &[f32], rw: usize, rh: usize, cell: f32) -> Self {
        let (nx, ny, nz) = grid_dims(rw as u32, rh as u32, cell);
        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
        let mut g = Self { nx, ny, nz, cell, data: vec![[0.0; 3]; nx * ny * nz] };
        for y in 0..rh {
            let fy = (y as f32 + 0.5) / cell + 0.5;
            let iy = fy.floor() as usize;
            let ty = fy - iy as f32;
            for x in 0..rw {
                let fx = (x as f32 + 0.5) / cell + 0.5;
                let ix = fx.floor() as usize;
                let tx = fx - ix as f32;
                let level = reduced[y * rw + x];
                let fz = (level - BASE_LO) / BASE_RANGE + 0.5;
                let iz = (fz.floor() as isize).clamp(0, nz as isize - 2) as usize;
                let tz = (fz - iz as f32).clamp(0.0, 1.0);
                for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
                    for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
                        for (dz, wz) in [(0, 1.0 - tz), (1, tz)] {
                            let w = wy * wx * wz;
                            let v = &mut g.data[((iz + dz) * ny + iy + dy) * nx + ix + dx];
                            v[0] += w * (level - BASE_LO);
                            v[1] += w;
                        }
                    }
                }
            }
        }
        g.blur();
        g
    }

    /// [1 2 1] along the three axes (edges repeated), then every column's
    /// plain average in the third component.
    fn blur(&mut self) {
        let mut out = vec![[0f32; 3]; self.data.len()];
        for z in 0..self.nz as isize {
            for y in 0..self.ny as isize {
                for x in 0..self.nx as isize {
                    let mut sum = [0f32; 2];
                    for dz in -1..=1isize {
                        for dy in -1..=1isize {
                            for dx in -1..=1isize {
                                let w = ((2 - dx.abs()) * (2 - dy.abs()) * (2 - dz.abs())) as f32;
                                let v = self.at(x + dx, y + dy, z + dz);
                                sum[0] += w * v[0];
                                sum[1] += w * v[1];
                            }
                        }
                    }
                    out[(z as usize * self.ny + y as usize) * self.nx + x as usize] = [sum[0] / 64.0, sum[1] / 64.0, 0.0];
                }
            }
        }
        for y in 0..self.ny {
            for x in 0..self.nx {
                let (mut a, mut b) = (0f32, 0f32);
                for z in 0..self.nz {
                    let v = out[(z * self.ny + y) * self.nx + x];
                    a += v[0];
                    b += v[1];
                }
                let mean = a / b.max(1e-9);
                for z in 0..self.nz {
                    out[(z * self.ny + y) * self.nx + x][2] = mean;
                }
            }
        }
        self.data = out;
    }

    /// The base level (log₂) of the pixel at `(x, y)` of an eye whose
    /// reduced copy has `n` pixels per texel and whose own level is `level`.
    pub fn base(&self, x: f32, y: f32, n: f32, level: f32) -> f32 {
        let cell_px = self.cell * n;
        let f = [
            ((x + 0.5) / cell_px + 0.5).clamp(0.0, self.nx as f32 - 1.0),
            ((y + 0.5) / cell_px + 0.5).clamp(0.0, self.ny as f32 - 1.0),
            ((level - BASE_LO) / BASE_RANGE + 0.5).clamp(0.0, self.nz as f32 - 1.0),
        ];
        let i = [0, 1, 2].map(|k| (f[k].floor() as usize).min([self.nx, self.ny, self.nz][k] - 2));
        let t = [0, 1, 2].map(|k| f[k] - i[k] as f32);
        let mut v = [0f32; 3];
        for (dz, wz) in [(0, 1.0 - t[2]), (1, t[2])] {
            for (dy, wy) in [(0, 1.0 - t[1]), (1, t[1])] {
                for (dx, wx) in [(0, 1.0 - t[0]), (1, t[0])] {
                    let s = self.data[((i[2] + dz) * self.ny + i[1] + dy) * self.nx + i[0] + dx];
                    let w = wx * wy * wz;
                    v[0] += w * s[0];
                    v[1] += w * s[1];
                    v[2] += w * s[2];
                }
            }
        }
        let fallback = base_fallback(self.cell);
        (v[0] + fallback * v[2]) / (v[1] + fallback) + BASE_LO
    }
}

/// The level of sample `i` of a local curve.
pub fn local_level(i: usize) -> f32 {
    LOCAL_LO + i as f32 / LOCAL_PER_STOP
}

fn power_mean(levels: &[f32], p: f64) -> f64 {
    let sum: f64 = levels.iter().map(|l| 2f64.powf(p * *l as f64)).sum();
    (sum / levels.len() as f64).log2() / p
}

/// The level at rank `q` (0..1) of sorted levels, linear between ranks.
fn quantile(sorted: &[f32], q: f64) -> f64 {
    let at = q * (sorted.len() - 1) as f64;
    let i = (at.floor() as usize).min(sorted.len() - 1);
    let f = at - i as f64;
    sorted[i] as f64 * (1.0 - f) + sorted[(i + 1).min(sorted.len() - 1)] as f64 * f
}

/// Ranks of the reduced copy's levels the curves are derived from.
const RANKS: [f64; 17] = [0.0, 0.002, 0.01, 0.03, 0.07, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.93, 0.97, 0.99, 0.998, 1.0];
/// The curves are derived at `anchor + CURVE_FIRST + i · CURVE_STEP` stops.
const CURVE_FIRST: f32 = -12.0;
const CURVE_STEP: f32 = 0.25;
const CURVE_POINTS: usize = 73;

/// The most headroom the curves tell apart, in stops.
pub const HEADROOM_MAX: f32 = 4.0;

/// What the curves are derived from: the levels at [`RANKS`] and three
/// means of the copy, all relative to the anchor (the power mean, p = 0.6,
/// of the copy's luminance), then `headroom`; and the anchor itself.
pub(crate) fn curve_inputs(sorted: &[f32], headroom: f32) -> (Vec<f32>, f32) {
    let anchor = power_mean(sorted, 0.6);
    let mean = sorted.iter().map(|l| *l as f64).sum::<f64>() / sorted.len() as f64;
    let mut x: Vec<f32> = RANKS.iter().map(|q| (quantile(sorted, *q) - anchor) as f32).collect();
    x.extend([power_mean(sorted, 1.0) - anchor, power_mean(sorted, 0.3) - anchor, mean - anchor].map(|v| v as f32));
    x.push(headroom.clamp(0.0, HEADROOM_MAX));
    (x, anchor as f32)
}

/// One fitted network: inputs standardised, `tanh` layers, a linear last one.
pub(crate) struct CurveNet {
    mean: Vec<f32>,
    scale: Vec<f32>,
    /// Per layer: weights (`inputs × outputs`, an input's row after the
    /// other), biases.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl CurveNet {
    /// Networks as they are stored from `values[*at]` on: their count; per
    /// network the layer count, the sizes, the inputs' means and scales,
    /// then every layer's weights and biases.
    pub(crate) fn read(values: &[f32], at: &mut usize) -> Vec<CurveNet> {
        let mut take = |n: usize| { let s = values[*at..*at + n].to_vec(); *at += n; s };
        let count = take(1)[0] as usize;
        (0..count).map(|_| {
            let layers = take(1)[0] as usize;
            let sizes: Vec<usize> = take(layers + 1).iter().map(|s| *s as usize).collect();
            let (mean, scale) = (take(sizes[0]), take(sizes[0]));
            let layers = sizes.windows(2).map(|w| (take(w[0] * w[1]), take(w[1]))).collect();
            CurveNet { mean, scale, layers }
        }).collect()
    }

    /// The mean of several networks' outputs for the inputs `x`.
    pub(crate) fn mean_of(nets: &[CurveNet], x: &[f32]) -> Vec<f32> {
        let mut y: Vec<f32> = Vec::new();
        for net in nets {
            let out = net.eval(x);
            if y.is_empty() {
                y = vec![0.0; out.len()];
            }
            for (acc, v) in y.iter_mut().zip(&out) {
                *acc += v / nets.len() as f32;
            }
        }
        y
    }

    pub(crate) fn eval(&self, x: &[f32]) -> Vec<f32> {
        let mut h: Vec<f32> = x.iter().zip(&self.mean).zip(&self.scale).map(|((v, m), s)| (v - m) / s).collect();
        let last = self.layers.len() - 1;
        for (n, (w, b)) in self.layers.iter().enumerate() {
            let mut z = b.clone();
            for (i, hi) in h.iter().enumerate() {
                let row = &w[i * b.len()..(i + 1) * b.len()];
                for (zj, wij) in z.iter_mut().zip(row) {
                    *zj += hi * wij;
                }
            }
            if n < last {
                z.iter_mut().for_each(|v| *v = v.tanh());
            }
            h = z;
        }
        h
    }
}

/// The networks fitted to Camera Raw's curves (`data/raw_local_curves.f32`:
/// their count; per network the layer count, the sizes, the inputs' means
/// and scales, then every layer's weights and biases). Their outputs are
/// averaged.
fn curve_nets() -> &'static [CurveNet] {
    static NETS: std::sync::OnceLock<Vec<CurveNet>> = std::sync::OnceLock::new();
    NETS.get_or_init(|| {
        let v: Vec<f32> = include_bytes!("data/raw_local_curves.f32").chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        CurveNet::read(&v, &mut 0)
    })
}

/// The Shadows / Highlights curves of a frame whose reduced copy has the
/// log₂ levels `levels` (sorted rising, exposure applied) and whose top
/// level sat `headroom` stops below the sensor's clipping when shot.
///
/// Camera Raw's curves have one pivot for both sliders. Shadows: a gain at
/// the pivot that falls linearly to zero at the frame's top and, below the
/// pivot, rises to a plateau near the frame's black; Highlights mirrors it
/// about the pivot. Where the pivot lies and how strong the branches are
/// follows the copy's whole distribution, by a rule this does not know:
/// networks fitted to the curves Camera Raw gave 1400 distributions (real
/// frames' and synthetic ones) stand in for it. They give the three single
/// curves and the three numbers by which the sliders act on each other
/// ([`LocalCurves`]). The curves follow the frame down the scale unchanged
/// (a darker exposure of the same frame has the same curves lower), but
/// for what happens next to white: hence the headroom.
pub fn local_curves(levels: &[f32], headroom: f32) -> LocalCurves {
    if levels.len() < 8 {
        return LocalCurves::flat();
    }
    let (top, bottom) = (levels[levels.len() - 1], levels[0]);
    if top - bottom < 0.05 {
        return LocalCurves::flat();
    }
    let (x, anchor) = curve_inputs(levels, headroom);
    let y = CurveNet::mean_of(curve_nets(), &x);
    assert_eq!(y.len(), 3 * CURVE_POINTS + 3, "raw_local_curves.f32: three curves and three numbers a network");
    // A curve over the frame's own range; outside it it stays level.
    let curve = |n: usize| -> Vec<f32> {
        let points = &y[n * CURVE_POINTS..(n + 1) * CURVE_POINTS];
        (0..LOCAL_SAMPLES).map(|i| {
            let pos = ((local_level(i).clamp(bottom, top) - anchor - CURVE_FIRST) / CURVE_STEP).clamp(0.0, (CURVE_POINTS - 1) as f32);
            let k = (pos.floor() as usize).min(CURVE_POINTS - 2);
            let f = pos - k as f32;
            points[k] * (1.0 - f) + points[k + 1] * f
        }).collect()
    };
    LocalCurves {
        shadows: curve(0),
        highlights: curve(1),
        highlights_up: curve(2),
        tilt_top: y[3 * CURVE_POINTS],
        tilt_bottom: y[3 * CURVE_POINTS + 1],
        shift: y[3 * CURVE_POINTS + 2],
        top,
        bottom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_keeps_a_flat_field_and_an_edge() {
        // Two flat halves three stops apart: each keeps its own level right
        // up to the border (the grid separates them by level).
        let (w, h) = (96usize, 64usize);
        let lum: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.01 } else { 0.08 }).collect();
        let (red, rw, rh) = reduce_levels(&lum, w, h, 2);
        let g = BaseGrid::build(&red, rw, rh, 6.0);
        for (x, want) in [(4.0, 0.01f32), (44.0, 0.01), (47.0, 0.01), (48.0, 0.08), (52.0, 0.08), (90.0, 0.08)] {
            let b = g.base(x, 30.0, 2.0, want.log2());
            // A tenth of a stop of the other side leaks in beside the border.
            let near = (x - 48.0f32).abs() < 12.0;
            assert!((b - want.log2()).abs() < if near { 0.15 } else { 0.02 }, "x {x}: {b} vs {}", want.log2());
        }
        // A speck unlike everything around it takes the level of its place.
        let b = g.base(20.0, 30.0, 2.0, 1.0f32.log2());
        assert!((b - 0.01f32.log2()).abs() < 0.02, "{b}");
    }

    #[test]
    fn texture_is_smoothed_into_its_region() {
        // A fine pattern half a stop peak to peak: the base is its mean
        // (of light to the power the reduced copy is taken at).
        let (w, h) = (128usize, 128usize);
        let lum: Vec<f32> = (0..w * h).map(|i| if (i % w + i / w) % 2 == 0 { 0.02 } else { 0.02 * 1.4142 }).collect();
        let (red, rw, rh) = reduce_levels(&lum, w, h, 4);
        let g = BaseGrid::build(&red, rw, rh, 6.0);
        let mean = ((0.02f32.powf(BASE_POWER) + (0.02f32 * 1.4142).powf(BASE_POWER)) / 2.0).log2() / BASE_POWER;
        for level in [0.02f32.log2(), (0.02f32 * 1.4142).log2()] {
            assert!((g.base(60.0, 60.0, 4.0, level) - mean).abs() < 0.01);
        }
    }

    #[test]
    fn a_block_of_bright_and_dark_counts_nearer_its_bright_part() {
        // Half at 2⁻¹, half at 2⁻⁵: the mean of the square roots, squared.
        let lum: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 0.5 } else { 0.03125 }).collect();
        let (red, rw, rh) = reduce_levels(&lum, 8, 8, 4);
        assert_eq!((rw, rh), (2, 2));
        let want = (((0.5f32).sqrt() + (0.03125f32).sqrt()) / 2.0).powi(2).log2();
        assert!(red.iter().all(|l| (l - want).abs() < 1e-4), "{red:?} vs {want}");
        // 0.64 stops above the mean of the two levels (−3).
        assert!(want > -3.0 + 0.6 && want < -3.0 + 0.7, "{want}");
    }

    #[test]
    fn layout_follows_the_frame() {
        // Four pixels of the exported frame, whatever its size; fewer in a
        // frame drawn smaller.
        assert_eq!((gain_cell(1.0), gain_cell(0.5), gain_cell(0.3), gain_cell(0.1), gain_cell(2.0)), (4, 2, 1, 1, 8));
        assert_eq!((gain_cell(0.0), gain_cell(f32::NAN)), (4, 4));
        // The grid's cell: a fraction of the long side, a texel at least.
        assert!((base_cell(4096, 4) - 6.144).abs() < 1e-3 && (base_cell(8192, 4) - 12.288).abs() < 1e-3);
        assert_eq!(base_cell(256, 4), 1.0);
        assert!((base_fallback(6.144) - 0.502).abs() < 1e-3);
        assert_eq!(grid_dims(1024, 512, 6.0), (173, 88, BASE_LEVELS));
        assert_eq!((limit_radius(4), limit_radius(3), limit_radius(2), limit_radius(1), limit_radius(8)), (2, 1, 1, 0, 2));
        for r in 0..=2usize {
            let sum: f32 = (-2..=2).map(|d| limit_weight(d, r)).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    /// The smoothed gain of every pixel of a picture's gain.
    fn smoothed(gain: &[f32], w: usize, h: usize, c: usize) -> Vec<f32> {
        let (cells, cw, ch) = gain_cells(gain, w, h, c);
        (0..w * h).map(|i| smooth_gain(gain, w, h, &cells, cw, ch, c, i % w, i / w)).collect()
    }

    #[test]
    fn a_gain_that_is_the_same_all_round_is_kept() {
        let (w, h) = (37usize, 22usize);
        for value in [-1.3f32, 0.0, 0.8] {
            let out = smoothed(&vec![value; w * h], w, h, 4);
            assert!(out.iter().all(|g| (g - value).abs() < 1e-6));
        }
        // Cells of one pixel: every pixel is its own cell.
        let gain: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 13) as f32 * 0.2 - 1.2).collect();
        assert_eq!(smoothed(&gain, w, h, 1), gain);
    }

    #[test]
    fn bright_among_dark_takes_the_share_of_bright() {
        // One pixel in four pulled down a stop, the others not at all (a
        // sky seen through leaves): the bright ones take a quarter of it,
        // the dark ones none.
        let (w, h) = (64usize, 64usize);
        let gain: Vec<f32> = (0..w * h).map(|i| if (i % w) % 2 == 0 && (i / w) % 2 == 0 { -1.0 } else { 0.0 }).collect();
        let out = smoothed(&gain, w, h, 4);
        for y in 8..56 {
            for x in 8..56 {
                let g = out[y * w + x];
                if gain[y * w + x] < 0.0 {
                    assert!((g + 0.25).abs() < 1e-5, "({x}, {y}): {g}");
                } else {
                    // Never more than itself and what touches it would take.
                    assert!((-0.25..=0.0).contains(&g), "({x}, {y}): {g}");
                }
            }
        }
        // The same pattern lifted: the mirror.
        let lifted: Vec<f32> = gain.iter().map(|g| -g).collect();
        let up = smoothed(&lifted, w, h, 4);
        assert!(up.iter().zip(&out).all(|(a, b)| (a + b).abs() < 1e-6));
    }

    #[test]
    fn beside_an_edge_one_side_fades_and_the_other_is_left_alone() {
        // Left of x = 50 a stop down, right of it nothing; the border is
        // inside a cell (50 = 12 · 4 + 2).
        let (w, h) = (96usize, 32usize);
        let gain: Vec<f32> = (0..w * h).map(|i| if i % w < 50 { -1.0 } else { 0.0 }).collect();
        let out = smoothed(&gain, w, h, 4);
        let row = |x: usize| out[16 * w + x];
        // Far from the border: as it was.
        assert!((row(20) + 1.0).abs() < 1e-6 && row(80) == 0.0);
        // The side the gain moves fades toward the border, monotonically…
        assert!(row(49) > -0.8 && row(49) < -0.3, "{}", row(49));
        assert!((40..50).all(|x| row(x + 1) >= row(x) - 1e-6));
        assert!((row(42) + 1.0).abs() < 0.02);
        // …the other side takes a little beside it and nothing beyond what
        // touches it.
        assert!(row(50) <= 0.0 && row(50) > -0.35, "{}", row(50));
        assert!((53..96).all(|x| row(x) == 0.0));
        // No pixel is moved the other way or by more than it was to be.
        assert!(out.iter().zip(&gain).all(|(o, g)| *o <= 0.0 && *o >= g.min(-0.3125)));
    }

    #[test]
    fn curves_of_a_frame() {
        // A flat frame has none.
        let c = local_curves(&[-3.0; 100], 3.0);
        assert_eq!(c, LocalCurves::flat());
        // Levels spread evenly over nine stops (Camera Raw on such a chart,
        // as gains on the black-subtracted signal: pivot 3.2 stops below the
        // top with ±0.16 there; Shadows +1.5 at −5, +2.7 at −7, next to
        // nothing at the top; Highlights −1.0 at −2, −1.8 at the top,
        // nothing at the black end; +100 the mirror of −100 before the
        // limiter; together the same way +0.31 everywhere; in opposition
        // +0.24 at the top, −0.20 at the black end).
        let levels: Vec<f32> = (0..900).map(|i| -9.0 + 0.01 * i as f32).collect();
        let c = local_curves(&levels, 0.0);
        let at = |curve: &[f32], l: f32| curve[((l - LOCAL_LO) * LOCAL_PER_STOP).round() as usize];
        for (l, want, tol) in [(-0.25, 0.03, 0.12), (-3.0, 0.14, 0.15), (-5.0, 1.47, 0.3), (-7.0, 2.7, 0.5)] {
            assert!((at(&c.shadows, l) - want).abs() < tol, "shadows at {l}: {} vs {want}", at(&c.shadows, l));
        }
        for (l, want, tol) in [(-8.0, -0.05, 0.12), (-4.0, -0.14, 0.15), (-2.0, -1.0, 0.3), (-0.25, -1.8, 0.4)] {
            assert!((at(&c.highlights, l) - want).abs() < tol, "highlights at {l}: {} vs {want}", at(&c.highlights, l));
        }
        for (l, want, tol) in [(-8.0, 0.05, 0.12), (-2.0, 1.0, 0.3), (-0.5, 1.8, 0.5)] {
            assert!((at(&c.highlights_up, l) - want).abs() < tol, "highlights up at {l}: {} vs {want}", at(&c.highlights_up, l));
        }
        assert!((c.pivot() + 3.2).abs() < 0.5, "pivot {}", c.pivot());
        assert!((c.shift - 0.31).abs() < 0.15 && (c.tilt_top - 0.24).abs() < 0.15 && (c.tilt_bottom + 0.25).abs() < 0.2, "{} {} {}", c.shift, c.tilt_top, c.tilt_bottom);
        assert!((c.top + 0.01).abs() < 1e-4 && c.bottom == -9.0);
        // Level outside the frame's own range.
        assert_eq!(at(&c.shadows, -12.0), at(&c.shadows, -9.0));
        assert_eq!(at(&c.highlights, 2.0), at(&c.highlights, -0.01));
        // The same frame two stops darker with as much room above it: the
        // same curves two stops lower.
        let darker: Vec<f32> = levels.iter().map(|l| l - 2.0).collect();
        let d = local_curves(&darker, 0.0);
        for l in [-8.0f32, -6.0, -4.0, -2.0, -1.0] {
            assert!((at(&d.shadows, l - 2.0) - at(&c.shadows, l)).abs() < 0.02 && (at(&d.highlights, l - 2.0) - at(&c.highlights, l)).abs() < 0.02);
        }
        assert!((d.shift - c.shift).abs() < 1e-4 && (d.pivot() - c.pivot() + 2.0).abs() < 0.02);
        assert_eq!(c.gain(50.0, 0.0)[100], 0.5 * c.shadows[100]);
        assert_eq!(c.gain(0.0, -100.0)[200], c.highlights[200]);
        assert_eq!(c.gain(0.0, 100.0)[200], c.highlights_up[200]);
    }

    #[test]
    fn the_two_sliders_together() {
        // A frame from −16 to 0 with, `t` stops above its bottom, Shadows
        // 1 − t/4 and Highlights −t/8: they cancel at t = 8/3.
        let level = |i: usize| LOCAL_LO + i as f32 / LOCAL_PER_STOP;
        let t = |i: usize| level(i).clamp(-16.0, 0.0) + 16.0;
        let c = LocalCurves {
            shadows: (0..LOCAL_SAMPLES).map(|i| 1.0 - t(i) / 4.0).collect(),
            highlights: (0..LOCAL_SAMPLES).map(|i| -t(i) / 8.0).collect(),
            highlights_up: (0..LOCAL_SAMPLES).map(|i| t(i) / 4.0).collect(),
            tilt_top: 0.6, tilt_bottom: -0.3, shift: 0.5, top: 0.0, bottom: -16.0,
        };
        let pivot = -16.0 + 8.0 / 3.0;
        assert!((c.pivot() - pivot).abs() < 1e-3, "{}", c.pivot());
        let near = |a: f32, b: f32| (a - b).abs() < 2e-4;
        let (top, low) = (256usize, 0usize);          // level 0 and level −16
        // Either slider alone, and Shadows below zero.
        assert!(near(c.gain(100.0, 0.0)[low], 1.0) && near(c.gain(-40.0, 0.0)[low], -0.4));
        assert!(near(c.gain(0.0, -100.0)[top], -2.0) && near(c.gain(0.0, 50.0)[top], 2.0));
        // The same way: the sum and the shift; its part off the diagonal.
        assert!(near(c.gain(100.0, 100.0)[top], -3.0 + 4.0 + 0.5) && near(c.gain(100.0, 100.0)[low], 1.0 + 0.5));
        assert!(near(c.gain(50.0, 50.0)[low], 0.5 + 0.25));
        assert!(near(c.gain(100.0, 25.0)[low], 1.0 + 0.5 * 0.25 * 0.25f32.powf(0.25)));
        // Both down: the mirror, with Highlights −100's own curve.
        assert!(near(c.gain(-100.0, -100.0)[top], 3.0 - 2.0 - 0.5));
        // In opposition: the tilt, whole at the two ends, none at the pivot.
        assert!(near(c.gain(100.0, -100.0)[top], -3.0 - 2.0 + 0.6) && near(c.gain(100.0, -100.0)[low], 1.0 - 0.3));
        let at_pivot = ((pivot - LOCAL_LO) * LOCAL_PER_STOP).round() as usize;
        assert!(c.gain(100.0, -100.0)[at_pivot].abs() < 0.03);
        // Off the diagonal: above the pivot by Shadows' amount at most,
        // below it by Highlights'.
        assert!(near(c.gain(100.0, -25.0)[top], -3.0 - 0.5 + 0.5 * 0.6) && near(c.gain(100.0, -25.0)[low], 1.0 - 0.25 * 0.3));
        assert!(near(c.gain(25.0, -100.0)[top], -0.75 - 2.0 + 0.25 * 0.6) && near(c.gain(25.0, -100.0)[low], 0.25 - 0.5 * 0.3));
        // Shadows down with Highlights up: the mirror.
        assert!(near(c.gain(-100.0, 100.0)[low], -1.0 + 0.3) && near(c.gain(-100.0, 100.0)[top], 3.0 + 4.0 - 0.6));
        // Level outside the frame's range.
        assert_eq!(c.gain(100.0, -100.0)[300], c.gain(100.0, -100.0)[top]);
    }
}
