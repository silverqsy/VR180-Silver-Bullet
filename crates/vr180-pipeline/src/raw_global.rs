//! The global part of raw development: Whites, Blacks and Contrast.
//!
//! Measured on Camera Raw (charts, and three frames against the charts
//! that have their levels): each of the three is ONE curve of the default
//! rendering's output, the same for every pixel of the frame. Set together
//! they are applied one after the other — Whites, then Blacks, then
//! Contrast — each exactly as it is alone (to 0.03/255). They come after
//! exposure and after Shadows / Highlights.
//!
//! The curves here are maps of the sRGB-encoded default output (0..1) onto
//! the encoded output, sampled at [`NODES`] equally spaced values:
//!
//! - **Whites below zero**: the same curve for every picture (a table over
//!   the slider).
//! - **Blacks**: belongs to the picture only through its darkest level — a
//!   bright picture's black is clipped, or lifted, from where the picture's
//!   own black is (a table over the slider and that level).
//! - **Contrast** belongs to the frame by where its weight lies: over 300
//!   distributions its maps at every slider value are the median maps plus
//!   three shapes (0.2/255 left over), and small networks give a frame's
//!   three numbers from its reduced copy — the same copy the Shadows /
//!   Highlights curves come from.
//! - **Whites above zero** is no map of the output but, on the value the
//!   tone curve is fed, `v' = a1·v + a2·v²` (to 0.2–0.8/255 on every chart),
//!   with the two numbers following mostly how far the frame's top is from
//!   white: networks again ([`whites_up`]).

use crate::raw_local::{curve_inputs, CurveNet};

/// Samples of a map: the encoded default output `k / (NODES − 1)`.
pub const NODES: usize = 129;

fn floats(bytes: &'static [u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Maps measured at slider values `amounts` (rising, zero not among them)
/// and, for Blacks, darkest levels `darks`: `values[amount][dark][node]`.
struct Table {
    amounts: Vec<f32>,
    darks: Vec<f32>,
    values: Vec<f32>,
}

impl Table {
    fn map(&self, amount: usize, dark: usize) -> &[f32] {
        let at = (amount * self.darks.len().max(1) + dark) * NODES;
        &self.values[at..at + NODES]
    }

    /// The map of slider value `amount` (index into `amounts`) for a darkest
    /// level `dark`: between the two levels measured around it.
    fn at_dark(&self, amount: usize, dark: f32) -> Vec<f32> {
        if self.darks.len() < 2 {
            return self.map(amount, 0).to_vec();
        }
        let last = self.darks.len() - 1;
        let step = (self.darks[last] - self.darks[0]) / last as f32;
        let pos = ((dark - self.darks[0]) / step).clamp(0.0, last as f32);
        let i = (pos.floor() as usize).min(last - 1);
        let f = pos - i as f32;
        self.map(amount, i).iter().zip(self.map(amount, i + 1)).map(|(a, b)| a * (1.0 - f) + b * f).collect()
    }

    /// The map of the slider at `amount` (−100..100): between the two
    /// measured values around it, zero being the identity. `None` at zero
    /// and on the side of zero this table has nothing for.
    fn at(&self, amount: f32, dark: f32) -> Option<Vec<f32>> {
        if amount == 0.0 {
            return None;
        }
        let identity = || (0..NODES).map(|k| k as f32 / (NODES - 1) as f32).collect::<Vec<f32>>();
        // The measured values on this side of zero, nearest zero first.
        let mut side: Vec<usize> = (0..self.amounts.len()).filter(|i| (self.amounts[*i] > 0.0) == (amount > 0.0)).collect();
        if side.is_empty() {
            return None;
        }
        side.sort_by(|a, b| self.amounts[*a].abs().partial_cmp(&self.amounts[*b].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let want = amount.abs().min(self.amounts[*side.last().unwrap()].abs());
        let (mut below, mut below_at) = (identity(), 0.0f32);
        for i in side {
            let here = self.amounts[i].abs();
            let above = self.at_dark(i, dark);
            if want <= here {
                let f = (want - below_at) / (here - below_at);
                return Some(below.iter().zip(&above).map(|(a, b)| a * (1.0 - f) + b * f).collect());
            }
            (below, below_at) = (above, here);
        }
        Some(below)
    }
}

fn blacks() -> &'static Table {
    static T: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        // Counts of slider values, darkest levels and nodes; the slider
        // values; the levels; the maps.
        let v = floats(include_bytes!("data/raw_blacks.f32"));
        let (na, nd, nn) = (v[0] as usize, v[1] as usize, v[2] as usize);
        assert_eq!((nn, v.len()), (NODES, 3 + na + nd + na * nd * NODES));
        Table { amounts: v[3..3 + na].to_vec(), darks: v[3 + na..3 + na + nd].to_vec(), values: v[3 + na + nd..].to_vec() }
    })
}

fn whites_down() -> &'static Table {
    static T: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let v = floats(include_bytes!("data/raw_whites_down.f32"));
        let (na, nn) = (v[0] as usize, v[1] as usize);
        assert_eq!((nn, v.len()), (NODES, 2 + na + na * NODES));
        Table { amounts: v[2..2 + na].to_vec(), darks: Vec::new(), values: v[2 + na..].to_vec() }
    })
}

/// The map of Blacks at `amount` (−100..100) for a frame whose darkest
/// level is `2^dark` (the scene value, exposure and the Shadows /
/// Highlights gain of that level in). `None` at zero.
pub fn blacks_map(amount: f32, dark: f32) -> Option<Vec<f32>> {
    blacks().at(amount, dark)
}

/// The map of Whites at `amount` below zero. `None` at zero and above
/// (see [`whites_up`]).
pub fn whites_map(amount: f32) -> Option<Vec<f32>> {
    whites_down().at(amount, 0.0)
}

/// The inputs the networks take of a frame: the shape of its reduced copy's
/// levels (sorted rising). `None` when there is too little to go by.
fn shape_inputs(levels: &[f32]) -> Option<Vec<f32>> {
    if levels.len() < 8 || levels[levels.len() - 1] - levels[0] < 0.05 {
        return None;
    }
    let (mut x, _) = curve_inputs(levels, 0.0);
    x.truncate(20);
    Some(x)
}

struct ContrastFamily {
    amounts: Vec<f32>,
    nodes: usize,
    /// `amounts × nodes`: the median map less the identity.
    median: Vec<f32>,
    /// Per shape `amounts × nodes`.
    shapes: Vec<Vec<f32>>,
    /// A frame's numbers on the shapes: the networks' outputs times `scale`
    /// plus `mean`.
    mean: Vec<f32>,
    scale: Vec<f32>,
    nets: Vec<CurveNet>,
}

fn contrast_family() -> &'static ContrastFamily {
    static F: std::sync::OnceLock<ContrastFamily> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        let v = floats(include_bytes!("data/raw_contrast.f32"));
        let na = v[0] as usize;
        let amounts = v[1..1 + na].to_vec();
        let (nodes, rank) = (v[1 + na] as usize, v[2 + na] as usize);
        let mut at = 3 + na;
        let mut take = |n: usize| { let s = v[at..at + n].to_vec(); at += n; s };
        let median = take(na * nodes);
        let shapes = (0..rank).map(|_| take(na * nodes)).collect();
        let (mean, scale) = (take(rank), take(rank));
        let nets = CurveNet::read(&v, &mut at);
        assert_eq!(at, v.len(), "raw_contrast.f32");
        ContrastFamily { amounts, nodes, median, shapes, mean, scale, nets }
    })
}

/// The map of Contrast at `amount` (−100..100) for a frame whose reduced
/// copy has the levels `levels` (sorted rising; as shot or exposed, only
/// their shape counts). `None` at zero. A frame nothing is known of gets
/// the median maps.
pub fn contrast_map(amount: f32, levels: &[f32]) -> Option<Vec<f32>> {
    if amount == 0.0 {
        return None;
    }
    let f = contrast_family();
    let numbers: Vec<f32> = match shape_inputs(levels) {
        Some(x) => CurveNet::mean_of(&f.nets, &x).iter().zip(&f.scale).zip(&f.mean).map(|((y, s), m)| y * s + m).collect(),
        None => vec![0.0; f.mean.len()],
    };
    // The frame's maps at the measured slider values, on the tables' nodes:
    // rising, within 0..1, white kept.
    let last = (f.nodes - 1) as f32;
    let mut values = Vec::with_capacity(f.amounts.len() * NODES);
    for a in 0..f.amounts.len() {
        let coarse: Vec<f32> = (0..f.nodes).map(|k| {
            let i = a * f.nodes + k;
            k as f32 / last + f.median[i] + numbers.iter().zip(&f.shapes).map(|(n, s)| n * s[i]).sum::<f32>()
        }).collect();
        let mut top = 0f32;
        for k in 0..NODES {
            let e = k as f32 / (NODES - 1) as f32;
            top = top.max(apply(&coarse, e).clamp(0.0, 1.0));
            values.push(if k == NODES - 1 { 1.0 } else { top });
        }
    }
    Table { amounts: f.amounts.clone(), darks: Vec::new(), values }.at(amount, 0.0)
}

struct WhitesUp {
    amounts: Vec<f32>,
    /// Per slider value log₂ a1, log₂ a2: the networks' outputs times
    /// `scale` plus `mean`.
    mean: Vec<f32>,
    scale: Vec<f32>,
    nets: Vec<CurveNet>,
}

fn whites_up_nets() -> &'static WhitesUp {
    static W: std::sync::OnceLock<WhitesUp> = std::sync::OnceLock::new();
    W.get_or_init(|| {
        let v = floats(include_bytes!("data/raw_whites_up.f32"));
        let na = v[0] as usize;
        let mut at = 1 + na;
        let mut take = |n: usize| { let s = v[at..at + n].to_vec(); at += n; s };
        let (mean, scale) = (take(2 * na), take(2 * na));
        let nets = CurveNet::read(&v, &mut at);
        assert_eq!(at, v.len(), "raw_whites_up.f32");
        WhitesUp { amounts: v[1..1 + na].to_vec(), mean, scale, nets }
    })
}

/// What the frame's top reads lower on a frame than on the charts the
/// networks were fitted to, in stops (nineteen frames: 0.27 ± 0.11).
pub const WHITES_TOP_OFFSET: f32 = 0.27;

/// Whites at `amount` above zero: `(a1, a2)` of `v' = a1·v + a2·v²` on the
/// value the tone curve is fed (the result not above 1). `levels`: the
/// reduced copy's levels as Whites meet the frame (exposed, Shadows /
/// Highlights in; sorted rising); `top`: log₂ of the fed value of the
/// frame's top. `None` at zero or below, or with nothing known of the frame.
pub fn whites_up(amount: f32, levels: &[f32], top: f32) -> Option<(f32, f32)> {
    if amount <= 0.0 {
        return None;
    }
    let w = whites_up_nets();
    let mut x = shape_inputs(levels)?;
    x.push((top + WHITES_TOP_OFFSET).clamp(-8.0, 0.0));
    let y: Vec<f32> = CurveNet::mean_of(&w.nets, &x).iter().zip(&w.scale).zip(&w.mean).map(|((y, s), m)| y * s + m).collect();
    let amount = amount.min(w.amounts[w.amounts.len() - 1]);
    // Below the first measured value: log₂ a1 with the slider, a2 with its square.
    let first = w.amounts[0];
    if amount <= first {
        let t = amount / first;
        return Some((2f32.powf(y[0] * t), 2f32.powf(y[1]) * t * t));
    }
    // Between two measured values: both in log₂.
    let i = (1..w.amounts.len()).find(|i| amount <= w.amounts[*i]).unwrap_or(w.amounts.len() - 1);
    let t = (amount - w.amounts[i - 1]) / (w.amounts[i] - w.amounts[i - 1]);
    let between = |k: usize| 2f32.powf(y[2 * (i - 1) + k] * (1.0 - t) + y[2 * i + k] * t);
    Some((between(0), between(1)))
}

/// A map at the encoded default output `e`.
pub fn apply(map: &[f32], e: f32) -> f32 {
    let last = (map.len() - 1) as f32;
    let y = (e * last).clamp(0.0, last);
    let i = (y.floor() as usize).min(map.len() - 2);
    let f = y - i as f32;
    map[i] * (1.0 - f) + map[i + 1] * f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(map: &[f32], out_of_255: f32) -> f32 {
        apply(map, out_of_255 / 255.0) * 255.0
    }

    #[test]
    fn blacks_follow_the_frames_darkest_level() {
        assert!(blacks_map(0.0, -10.0).is_none());
        // A dark frame (Camera Raw on a chart from 2⁻¹⁰ up): −100 clips
        // what the default puts below 48 and takes 96 to 52; +100 lifts.
        let down = blacks_map(-100.0, -10.0).unwrap();
        assert!(at(&down, 32.0) < 0.2 && (at(&down, 96.0) - 51.7).abs() < 1.0 && (at(&down, 191.0) - 178.3).abs() < 1.0, "{} {}", at(&down, 96.0), at(&down, 191.0));
        let up = blacks_map(100.0, -10.0).unwrap();
        assert!((at(&up, 32.0) - 53.0).abs() < 1.0 && (at(&up, 128.0) - 147.4).abs() < 1.0, "{} {}", at(&up, 32.0), at(&up, 128.0));
        // White stays white, the maps rise.
        for m in [&down, &up] {
            assert!((m[NODES - 1] - 1.0).abs() < 2e-3 && m.windows(2).all(|w| w[1] >= w[0] - 1e-6));
        }
        // A frame whose black is high: −100 clips far further up.
        let bright = blacks_map(-100.0, -3.0).unwrap();
        assert!(at(&bright, 128.0) < 0.5 && (at(&bright, 191.0) - 28.6).abs() < 2.0, "{}", at(&bright, 191.0));
        // Half the slider is the map measured there, not half the change.
        let half = blacks_map(-50.0, -10.0).unwrap();
        assert!((at(&half, 96.0) - 83.7).abs() < 1.0 && (at(&half, 32.0) - 11.8).abs() < 1.0);
        // Between measured values and between levels: in between.
        let a = at(&blacks_map(-60.0, -10.0).unwrap(), 96.0);
        assert!(a < at(&half, 96.0) && a > at(&blacks_map(-75.0, -10.0).unwrap(), 96.0));
        let b = at(&blacks_map(-100.0, -4.2).unwrap(), 191.0);
        assert!(b < at(&blacks_map(-100.0, -5.0).unwrap(), 191.0) && b > at(&blacks_map(-100.0, -4.0).unwrap(), 191.0) - 1e-3);
        // Toward zero the map goes to the identity; past the last value it stays.
        assert!((at(&blacks_map(-1.0, -10.0).unwrap(), 96.0) - 96.0).abs() < 0.5);
        assert_eq!(blacks_map(-140.0, -10.0), blacks_map(-100.0, -10.0));
        assert_eq!(blacks_map(-100.0, -30.0), blacks_map(-100.0, -13.0));
    }

    /// Levels spread evenly from `lo` to `hi`.
    fn spread(lo: f32, hi: f32) -> Vec<f32> {
        (0..441).map(|i| lo + (hi - lo) * i as f32 / 440.0).collect()
    }

    #[test]
    fn contrast_belongs_to_the_frame() {
        let levels = spread(-9.0, 0.0);
        assert!(contrast_map(0.0, &levels).is_none());
        // Camera Raw on a chart spread evenly over nine stops, +100:
        // 32 → 9, 64 → 41, 128 → 137, 192 → 219; −100: 32 → 54, 128 → 121,
        // 192 → 168; what is left alone lies near 113.
        let up = contrast_map(100.0, &levels).unwrap();
        for (e, want, tol) in [(32.0, 9.4, 4.0), (64.0, 40.7, 6.0), (128.0, 137.5, 8.0), (192.0, 218.6, 6.0)] {
            assert!((at(&up, e) - want).abs() < tol, "+100 at {e}: {} vs {want}", at(&up, e));
        }
        let down = contrast_map(-100.0, &levels).unwrap();
        for (e, want, tol) in [(32.0, 54.2, 5.0), (128.0, 121.4, 8.0), (192.0, 167.9, 6.0)] {
            assert!((at(&down, e) - want).abs() < tol, "-100 at {e}: {} vs {want}", at(&down, e));
        }
        for m in [&up, &down] {
            assert!(m[0] >= 0.0 && (m[NODES - 1] - 1.0).abs() < 1e-6 && m.windows(2).all(|w| w[1] >= w[0]));
        }
        // The same levels three stops down: the same maps (only the shape counts).
        let lower: Vec<f32> = levels.iter().map(|l| l - 3.0).collect();
        let same = contrast_map(100.0, &lower).unwrap();
        assert!(up.iter().zip(&same).all(|(a, b)| (a - b).abs() < 1e-4));
        // A real frame (a room by lamplight; its reduced copy's levels at
        // every twentieth of the ranks): what is left alone lies higher,
        // near 150 — Camera Raw takes 128 to 111 on it, 96 to 68.
        let ranks = [-10.40, -8.03, -7.28, -6.84, -6.50, -6.21, -5.83, -5.53, -5.23, -5.04, -4.86, -4.63, -4.51, -4.42, -4.27, -4.06, -3.76, -3.51, -3.35, -3.16, -3.01f32];
        let room: Vec<f32> = (0..=400).map(|i| { let (k, f) = (i / 20, (i % 20) as f32 / 20.0); ranks[k] * (1.0 - f) + ranks[(k + 1).min(20)] * f }).collect();
        let lamp = contrast_map(100.0, &room).unwrap();
        assert!((at(&lamp, 128.0) - 111.4).abs() < 8.0 && (at(&lamp, 96.0) - 67.8).abs() < 6.0, "{} {}", at(&lamp, 128.0), at(&lamp, 96.0));
        // Half the slider: the map measured there; nothing known: the median maps.
        let half = contrast_map(50.0, &levels).unwrap();
        assert!(at(&half, 64.0) > at(&up, 64.0) + 5.0 && at(&half, 64.0) < 64.0);
        assert!(contrast_map(100.0, &[]).is_some());
    }

    #[test]
    fn whites_above_zero_follow_the_frames_top() {
        let levels = spread(-9.0, 0.0);
        assert!(whites_up(0.0, &levels, 0.0).is_none() && whites_up(-30.0, &levels, 0.0).is_none() && whites_up(50.0, &[], 0.0).is_none());
        // A frame reaching white: +100 is about v + 7·v² (Camera Raw on such
        // a chart: a fed value of 1/8 goes to 0.24).
        let (a1, a2) = whites_up(100.0, &levels, -0.15 - WHITES_TOP_OFFSET).unwrap();
        let to = a1 * 0.125 + a2 * 0.125 * 0.125;
        assert!((to - 0.243).abs() < 0.07 && a1 > 0.9 && a1 < 1.3, "{a1} {a2} {to}");
        // A frame whose top is three stops down: far more.
        let (b1, b2) = whites_up(100.0, &spread(-10.0, -3.0), -3.0 - WHITES_TOP_OFFSET).unwrap();
        assert!(b2 > 20.0 * a2 && b1 > a1, "{b1} {b2}");
        // Less slider, less of both; toward zero nothing.
        let (c1, c2) = whites_up(50.0, &levels, -0.15 - WHITES_TOP_OFFSET).unwrap();
        assert!(c2 < 0.4 * a2 && c1 > 0.95 && c1 < 1.1, "{c1} {c2}");
        let (d1, d2) = whites_up(1.0, &levels, -0.15 - WHITES_TOP_OFFSET).unwrap();
        assert!((d1 - 1.0).abs() < 2e-3 && d2 < 1e-3);
    }

    #[test]
    fn whites_below_zero_are_one_curve() {
        assert!(whites_map(0.0).is_none() && whites_map(40.0).is_none());
        let w = whites_map(-100.0).unwrap();
        // Camera Raw: 128 → 118.6, 192 → 171.5, 224 → 197.1; white stays.
        assert!((at(&w, 128.0) - 118.9).abs() < 1.0 && (at(&w, 191.25) - 171.5).abs() < 1.0 && (at(&w, 223.1) - 197.1).abs() < 1.0, "{} {}", at(&w, 128.0), at(&w, 191.25));
        assert!((w[NODES - 1] - 1.0).abs() < 2e-3 && (at(&w, 16.0) - 15.9).abs() < 0.3);
        let half = whites_map(-50.0).unwrap();
        assert!((at(&half, 191.25) - 178.7).abs() < 1.0);
    }
}
