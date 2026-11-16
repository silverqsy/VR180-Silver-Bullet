//! The local part of raw development (Shadows / Highlights) on the CPU:
//! the base — a bilateral grid of the frame's log₂ luminance, the twin of
//! `shaders/raw_reduce.wgsl`, `raw_grid_build.wgsl`, `raw_grid_blur.wgsl`
//! and the lookup in `raw_develop_local.wgsl` — and the frame's curves.
//!
//! See [`crate::raw_tone`] for what the stage does with them.

use crate::raw_tone::{
    LocalCurves, BASE_FALLBACK, BASE_FLOOR, BASE_LEVELS, BASE_LO, BASE_RANGE, LOCAL_LO, LOCAL_PER_STOP, LOCAL_SAMPLES,
};

/// Pixels of an eye per texel of the reduced copy the grid is built from,
/// and the grid's cell in those texels, for a frame whose long side is
/// `long` pixels: the cell is [`crate::raw_tone::BASE_CELL`] of the long
/// side, about six texels of the copy.
pub fn base_layout(long: u32) -> (u32, f32) {
    let cell_px = (crate::raw_tone::BASE_CELL * long as f32).max(1.0);
    let n = ((cell_px / 6.0).round() as u32).clamp(1, 16);
    (n, cell_px / n as f32)
}

/// The grid's size for a reduced copy of `rw × rh` texels and cells of
/// `cell` texels.
pub fn grid_dims(rw: u32, rh: u32, cell: f32) -> (u32, u32, u32) {
    ((rw as f32 / cell).ceil() as u32 + 2, (rh as f32 / cell).ceil() as u32 + 2, BASE_LEVELS)
}

/// The reduced copy: the mean of log₂ luminance over every `n × n` pixels
/// (the last row / column of blocks repeats the edge pixels).
pub fn reduce_log(luminance: &[f32], w: usize, h: usize, n: usize) -> (Vec<f32>, usize, usize) {
    let (rw, rh) = (w.div_ceil(n), h.div_ceil(n));
    let mut out = vec![0f32; rw * rh];
    for ry in 0..rh {
        for rx in 0..rw {
            let mut sum = 0f32;
            for j in 0..n {
                let y = (ry * n + j).min(h - 1);
                for i in 0..n {
                    let x = (rx * n + i).min(w - 1);
                    sum += luminance[y * w + x].max(BASE_FLOOR).log2();
                }
            }
            out[ry * rw + rx] = sum / (n * n) as f32;
        }
    }
    (out, rw, rh)
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
        (v[0] + BASE_FALLBACK * v[2]) / (v[1] + BASE_FALLBACK) + BASE_LO
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

/// What the curves are derived from: the levels at [`RANKS`] and three
/// means of the copy, all relative to the anchor (the power mean, p = 0.6,
/// of the copy's luminance), and the anchor itself.
fn curve_inputs(sorted: &[f32]) -> (Vec<f32>, f32) {
    let anchor = power_mean(sorted, 0.6);
    let mean = sorted.iter().map(|l| *l as f64).sum::<f64>() / sorted.len() as f64;
    let mut x: Vec<f32> = RANKS.iter().map(|q| (quantile(sorted, *q) - anchor) as f32).collect();
    x.extend([power_mean(sorted, 1.0) - anchor, power_mean(sorted, 0.3) - anchor, mean - anchor].map(|v| v as f32));
    (x, anchor as f32)
}

/// One fitted network: inputs standardised, `tanh` layers, a linear last one.
struct CurveNet {
    mean: Vec<f32>,
    scale: Vec<f32>,
    /// Per layer: weights (`inputs × outputs`, an input's row after the
    /// other), biases.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl CurveNet {
    fn eval(&self, x: &[f32]) -> Vec<f32> {
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
        let mut at = 0usize;
        let mut take = |n: usize| { let s = v[at..at + n].to_vec(); at += n; s };
        let count = take(1)[0] as usize;
        (0..count).map(|_| {
            let layers = take(1)[0] as usize;
            let sizes: Vec<usize> = take(layers + 1).iter().map(|s| *s as usize).collect();
            let (mean, scale) = (take(sizes[0]), take(sizes[0]));
            let layers = sizes.windows(2).map(|w| (take(w[0] * w[1]), take(w[1]))).collect();
            CurveNet { mean, scale, layers }
        }).collect()
    })
}

/// The Shadows / Highlights curves of a frame whose reduced copy has the
/// log₂ levels `levels` (sorted rising, exposure applied).
///
/// Camera Raw's curves have one pivot for both sliders. Shadows: a gain at
/// the pivot that falls linearly to zero at the frame's top and, below the
/// pivot, rises to a plateau near the frame's black; Highlights mirrors it
/// about the pivot. Where the pivot lies and how strong the branches are
/// follows the copy's whole distribution, by a rule this does not know:
/// networks fitted to the curves Camera Raw gave some 1600 distributions
/// (real frames' and synthetic ones) stand in for it.
pub fn local_curves(levels: &[f32]) -> LocalCurves {
    let flat = || LocalCurves { shadows: vec![0.0; LOCAL_SAMPLES], highlights: vec![0.0; LOCAL_SAMPLES], highlights_up: vec![0.0; LOCAL_SAMPLES] };
    if levels.len() < 8 {
        return flat();
    }
    let (top, bottom) = (levels[levels.len() - 1], levels[0]);
    if top - bottom < 0.05 {
        return flat();
    }
    let (x, anchor) = curve_inputs(levels);
    let nets = curve_nets();
    let mut y = vec![0f32; 3 * CURVE_POINTS];
    for net in nets {
        let out = net.eval(&x);
        for (acc, v) in y.iter_mut().zip(&out) {
            *acc += v / nets.len() as f32;
        }
        // A network without the third curve: Highlights up as the mirror of down.
        if out.len() < 3 * CURVE_POINTS {
            for i in 0..CURVE_POINTS {
                y[2 * CURVE_POINTS + i] -= out[CURVE_POINTS + i] / nets.len() as f32;
            }
        }
    }
    // The curves over the frame's own range; outside it they stay level.
    let at = |curve: &[f32], level: f32| -> f32 {
        let pos = ((level.clamp(bottom, top) - anchor - CURVE_FIRST) / CURVE_STEP).clamp(0.0, (CURVE_POINTS - 1) as f32);
        let i = (pos.floor() as usize).min(CURVE_POINTS - 2);
        let f = pos - i as f32;
        curve[i] * (1.0 - f) + curve[i + 1] * f
    };
    let mut c = flat();
    for i in 0..LOCAL_SAMPLES {
        let l = local_level(i);
        c.shadows[i] = at(&y[..CURVE_POINTS], l);
        c.highlights[i] = at(&y[CURVE_POINTS..2 * CURVE_POINTS], l);
        c.highlights_up[i] = at(&y[2 * CURVE_POINTS..], l);
    }
    c
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
        let (red, rw, rh) = reduce_log(&lum, w, h, 2);
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
        // A fine pattern half a stop peak to peak: the base is its mean.
        let (w, h) = (128usize, 128usize);
        let lum: Vec<f32> = (0..w * h).map(|i| if (i % w + i / w) % 2 == 0 { 0.02 } else { 0.02 * 1.4142 }).collect();
        let (red, rw, rh) = reduce_log(&lum, w, h, 4);
        let g = BaseGrid::build(&red, rw, rh, 6.0);
        let mean = (0.02f32.log2() + (0.02f32 * 1.4142).log2()) / 2.0;
        for level in [0.02f32.log2(), (0.02f32 * 1.4142).log2()] {
            assert!((g.base(60.0, 60.0, 4.0, level) - mean).abs() < 0.01);
        }
    }

    #[test]
    fn layout_follows_the_frame() {
        assert_eq!(base_layout(4096).0, 4);
        assert!((base_layout(4096).1 - 6.144).abs() < 1e-3);
        assert_eq!(base_layout(1024).0, 1);
        assert_eq!(base_layout(16384).0, 16);
        assert_eq!(grid_dims(1024, 512, 6.0), (173, 88, BASE_LEVELS));
    }

    #[test]
    fn curves_of_a_frame() {
        // A flat frame has none.
        let c = local_curves(&[-3.0; 100]);
        assert!(c.shadows.iter().chain(&c.highlights).all(|v| *v == 0.0));
        // Levels spread evenly over nine stops (Camera Raw on such a chart,
        // as gains on the black-subtracted signal: pivot 3.3 stops below the
        // top; Shadows +0.13 there, +1.5 at −5, +2.7 at −7, next to nothing
        // at the top; Highlights −0.14 at the pivot, −1.0 at −2, −1.7 at
        // the top, nothing at the black end).
        let levels: Vec<f32> = (0..900).map(|i| -9.0 + 0.01 * i as f32).collect();
        let c = local_curves(&levels);
        let at = |curve: &[f32], l: f32| curve[((l - LOCAL_LO) * LOCAL_PER_STOP).round() as usize];
        for (l, want, tol) in [(-0.25, 0.03, 0.12), (-3.0, 0.13, 0.15), (-5.0, 1.47, 0.3), (-7.0, 2.7, 0.5)] {
            assert!((at(&c.shadows, l) - want).abs() < tol, "shadows at {l}: {} vs {want}", at(&c.shadows, l));
        }
        for (l, want, tol) in [(-8.0, -0.05, 0.12), (-4.0, -0.13, 0.15), (-2.0, -1.0, 0.3), (-0.25, -1.7, 0.4)] {
            assert!((at(&c.highlights, l) - want).abs() < tol, "highlights at {l}: {} vs {want}", at(&c.highlights, l));
        }
        // Level outside the frame's own range.
        assert_eq!(at(&c.shadows, -12.0), at(&c.shadows, -9.0));
        assert_eq!(at(&c.highlights, 2.0), at(&c.highlights, 0.0));
        // The same frame two stops darker: the same curves two stops lower.
        let darker: Vec<f32> = levels.iter().map(|l| l - 2.0).collect();
        let d = local_curves(&darker);
        for l in [-8.0f32, -6.0, -4.0, -2.0, -1.0] {
            assert!((at(&d.shadows, l - 2.0) - at(&c.shadows, l)).abs() < 0.02 && (at(&d.highlights, l - 2.0) - at(&c.highlights, l)).abs() < 0.02);
        }
        assert_eq!(c.gain(50.0, 0.0)[100], 0.5 * c.shadows[100]);
        assert_eq!(c.gain(0.0, -100.0)[200], c.highlights[200]);
        assert_eq!(c.gain(0.0, 100.0)[200], c.highlights_up[200]);
    }
}
