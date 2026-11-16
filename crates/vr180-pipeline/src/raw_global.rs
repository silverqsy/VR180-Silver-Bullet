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
//! - **Whites above zero** and **Contrast** belong to the frame (how far
//!   its top is from white; where its weight lies): not here yet.

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

/// The map of Whites at `amount` (−100..100). `None` at zero, and above
/// zero (the frame's own curve: not here yet).
pub fn whites_map(amount: f32) -> Option<Vec<f32>> {
    whites_down().at(amount, 0.0)
}

/// The map of Contrast at `amount`: the frame's own curve, not here yet.
pub fn contrast_map(_amount: f32) -> Option<Vec<f32>> {
    None
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
