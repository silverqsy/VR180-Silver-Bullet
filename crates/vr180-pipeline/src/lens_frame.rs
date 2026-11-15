//! Lens frame: an export whose frame is the image of a lens, with the lens
//! stated in the file.
//!
//! A source whose frame is a crop of its lens's view (the Insta360 X6 Crop
//! modes) covers ~97° or ~126°. Written as VR180, it sits in the middle of a
//! 180° canvas that is half to two thirds black, and the canvas has to be far
//! larger than the recording to keep its detail. A lens frame writes the eye
//! the way the camera recorded it instead: through a lens with the camera
//! lens's own radial law, so the frame is full and the pixel density is the
//! recording's everywhere. Apple's Projected Media Profile can state such a
//! lens (its parametric immersive projection, see
//! [`crate::spherical_inject::inject_apmp_lens`]), and an Apple Vision Pro
//! then shows every pixel at its own angle.
//!
//! The lens is not the camera's, though. A player builds one surface for
//! both views, from the first lens in the file, and the profile holds nine
//! numbers where a factory calibration holds more. So both eyes are rendered
//! into ONE lens ([`LensFrame`]): the radially symmetric part of the source
//! lenses, centred in the frame. Everything that is particular to an eye —
//! its off-centre axis, its tangential and prism terms, its mount, the
//! stabilization — is resolved by the projection, exactly as for a VR180
//! frame, and the file's lens is exact by construction.

use crate::gpu::FisheyeCalib;
use crate::spherical_inject::ApmpLens;

/// The lens of a lens frame: Mei and Rives' model, centred. A ray lands on
/// the plane point `m = (x, y) / (z + xi)` and reaches the frame at
/// `focal · m · (1 + k1·|m|² + k2·|m|⁴)` from its centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensFrame {
    /// One eye's frame, pixels.
    pub width: u32,
    pub height: u32,
    pub xi: f32,
    /// Focal length in pixels of the frame.
    pub focal: f32,
    pub k1: f32,
    pub k2: f32,
    /// Pixels at the frame's border that stay black: what a player repeats
    /// past the frame's edge is then black, not the picture's last column.
    pub rim_px: f32,
}

/// How far off the axis the fit of [`LensFrame::from_sources`] looks, at
/// most: short of where this lens family folds back (114° for the X6).
const FIT_LIMIT_DEG: f64 = 105.0;

impl LensFrame {
    /// The lens frame of `width × height` pixels for a source seen through
    /// `sources` (one calibration per eye, in the source's own pixels): their
    /// mean radial law, refitted with two radial terms over the source frame
    /// out to its corner, at the scale that keeps the smaller side of the
    /// source frame on the smaller side of this one.
    ///
    /// `None` unless every source is a unified-camera-model lens: that is
    /// the family the file's lens belongs to, and the only one for which the
    /// refit is a refit and not a change of projection.
    pub fn from_sources(sources: &[&FisheyeCalib], width: u32, height: u32) -> Option<Self> {
        let first = sources.first()?;
        if width == 0 || height == 0 || sources.iter().any(|c| !(c.xi > 0.0) || c.src_proj > 0.5) {
            return None;
        }
        let n = sources.len() as f64;
        let xi = sources.iter().map(|c| c.xi as f64).sum::<f64>() / n;
        // Image radius (source pixels) of the plane radius `m`, averaged.
        let radius = |m: f64| -> f64 {
            sources.iter().map(|c| {
                let m2 = m * m;
                let d = 1.0 + m2 * (c.k[0] as f64 + m2 * (c.k[1] as f64 + m2 * (c.k[2] as f64
                    + m2 * (c.k[3] as f64 + m2 * c.k5 as f64))));
                0.5 * (c.fx as f64 + c.fy as f64) * m * d
            }).sum::<f64>() / n
        };
        let (src_w, src_h) = (first.src_w as f64, first.src_h as f64);
        if !(src_w > 0.0 && src_h > 0.0) {
            return None;
        }
        // The plane radius the frame's corner reaches, capped by the limit.
        let corner = 0.5 * (src_w * src_w + src_h * src_h).sqrt();
        let m_cap = {
            let t = FIT_LIMIT_DEG.to_radians();
            t.sin() / (xi + t.cos())
        };
        let m_max = if radius(m_cap) <= corner {
            m_cap
        } else {
            let (mut lo, mut hi) = (0.0_f64, m_cap);
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                if radius(mid) < corner { lo = mid } else { hi = mid }
            }
            0.5 * (lo + hi)
        };
        // Least squares for a·m + b·m³ + c·m⁵ over (0, m_max].
        const SAMPLES: usize = 512;
        let mut ata = [[0.0_f64; 3]; 3];
        let mut atb = [0.0_f64; 3];
        for i in 1..=SAMPLES {
            let m = m_max * i as f64 / SAMPLES as f64;
            let basis = [m, m * m * m, m * m * m * m * m];
            let r = radius(m);
            for (row, bi) in basis.iter().enumerate() {
                atb[row] += bi * r;
                for (col, bj) in basis.iter().enumerate() {
                    ata[row][col] += bi * bj;
                }
            }
        }
        let [a, b, c] = solve3(ata, atb)?;
        if !(a > 0.0) || !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return None;
        }
        let side = width.min(height) as f64;
        let scale = side / src_w.min(src_h);
        Some(Self {
            width,
            height,
            xi: xi as f32,
            focal: (a * scale) as f32,
            k1: (b / a) as f32,
            k2: (c / a) as f32,
            rim_px: (side / 614.0).round().max(4.0) as f32,
        })
    }

    /// The ray of a point of the frame — `(px, py)` in pixels from its left
    /// and top edges — as a unit vector (x right, y up, z forward). `None`
    /// past the model's fold. CPU twin of the fisheye-output shaders'
    /// `out_ray` (which also blackens the rim).
    pub fn ray(&self, px: f64, py: f64) -> Option<[f64; 3]> {
        let (xi, k1, k2) = (self.xi as f64, self.k1 as f64, self.k2 as f64);
        let dx = px - self.width as f64 * 0.5;
        let dy = self.height as f64 * 0.5 - py;
        let r = (dx * dx + dy * dy).sqrt();
        let rd = r / self.focal as f64;
        let mut m = rd;
        for _ in 0..32 {
            let m2 = m * m;
            m -= (m * (1.0 + m2 * (k1 + k2 * m2)) - rd) / (1.0 + m2 * (3.0 * k1 + 5.0 * k2 * m2));
        }
        let mm = m * m;
        let disc = 1.0 + (1.0 - xi * xi) * mm;
        if disc < 0.0 || !m.is_finite() {
            return None;
        }
        let g = (xi + disc.sqrt()) / (mm + 1.0);
        let (ux, uy) = if r > 1e-12 { (dx / r, dy / r) } else { (0.0, 0.0) };
        Some([g * m * ux, g * m * uy, g - xi])
    }

    /// Where the unit ray `dir` (x right, y up, z forward) lands, in pixels
    /// from the frame's left and top edges: the lens as the file states it.
    pub fn project(&self, dir: [f64; 3]) -> [f64; 2] {
        let (xi, k1, k2, f) = (self.xi as f64, self.k1 as f64, self.k2 as f64, self.focal as f64);
        let den = dir[2] + xi;
        let (x, y) = (dir[0] / den, -dir[1] / den);
        let r2 = x * x + y * y;
        let d = 1.0 + r2 * (k1 + k2 * r2);
        [self.width as f64 * 0.5 + f * x * d, self.height as f64 * 0.5 + f * y * d]
    }

    /// Angle off the axis, degrees, where this lens family stops (the plane
    /// point runs away): 180° for ξ ≤ 1.
    fn fold_deg(&self) -> f64 {
        let xi = self.xi as f64;
        if xi > 1.0 { (-1.0 / xi).acos().to_degrees() } else { 180.0 }
    }

    /// How far off the axis the frame reaches, degrees: its corner.
    pub fn reach_deg(&self) -> f32 {
        match self.ray(0.0, 0.0) {
            Some(d) => d[2].clamp(-1.0, 1.0).acos().to_degrees() as f32,
            None => self.fold_deg() as f32,
        }
    }

    /// The lens as the file states it: the frame's own numbers, stated out
    /// to the next whole degree past its corner.
    pub fn apmp(&self) -> ApmpLens {
        let limit = ((self.reach_deg() as f64 + 0.5).ceil()).min(self.fold_deg() - 1.0);
        ApmpLens {
            width: self.width,
            height: self.height,
            xi: self.xi,
            focal: self.focal,
            k1: self.k1,
            k2: self.k2,
            limit_deg: limit as f32,
        }
    }
}

/// Solve a 3 × 3 linear system (Gaussian elimination with pivoting).
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            for k in col..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0_f64; 3];
    for row in (0..3).rev() {
        let mut v = b[row];
        for k in row + 1..3 {
            v -= a[row][k] * x[k];
        }
        x[row] = v / a[row][row];
    }
    Some(x)
}

impl FisheyeCalib {
    /// Lens-frame output: the fisheye-output kernels write the image of
    /// `frame`'s lens over the whole frame instead of their equidistant
    /// disc. The calibration keeps describing the SOURCE; `frame` must be
    /// the size of the texture projected into.
    pub fn with_lens_frame(mut self, frame: &LensFrame) -> Self {
        self.out_lens = [frame.xi, frame.focal, frame.k1, frame.k2];
        self.out_rim = frame.rim_px;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two lenses of an X6 as its file states them, placed in a 10K Crop
    /// frame (4912 × 4912 native pixels).
    fn crop_10k() -> (FisheyeCalib, FisheyeCalib) {
        let left = FisheyeCalib::new_omni(
            7222.06, 7222.06, 2443.7, 2464.16, 2.45543,
            [1.28201866, -0.80810684, 1.14986908, 12.02910519, 0.0],
            [8.391e-4, 8.153e-4, -9.025e-3, 3.787e-3], [-1.04e-3, 3.74e-3, -1.97e-3, -3.64e-2],
            4912.0, 4912.0,
        );
        let right = FisheyeCalib::new_omni(
            7200.68, 7200.68, 2463.4, 2461.74, 2.45543,
            [1.27704287, -0.69810301, -0.66506159, 19.39754486, 0.0],
            [4.581e-4, 2.088e-3, 3.922e-3, 1.784e-3], [2.61e-4, -1.21e-2, -2.06e-3, -1.90e-2],
            4912.0, 4912.0,
        );
        (left, right)
    }

    /// The frame's lens follows the source lenses' radial law to within a
    /// pixel out to the frame's edges and three in its far corners (two
    /// radial terms standing in for the factory's four), so rendering into
    /// it resamples the recording at very nearly its own density.
    #[test]
    fn the_lens_keeps_the_source_radial_law() {
        let (l, r) = crop_10k();
        let frame = LensFrame::from_sources(&[&l, &r], 4912, 4912).expect("a lens");
        assert_eq!((frame.width, frame.height), (4912, 4912));
        assert!((frame.xi - 2.45543).abs() < 1e-6);
        // Within half a percent of the factory focal lengths' mean.
        assert!((frame.focal / 7211.37 - 1.0).abs() < 0.005, "focal {}", frame.focal);
        let mean_radius = |theta: f64| 0.5 * (l.radius_pure(theta as f32) as f64 + r.radius_pure(theta as f32) as f64);
        let (mut worst_to_edge, mut worst) = (0.0_f64, 0.0_f64);
        for i in 0..=176 {
            let theta = (i as f64 * 0.5).to_radians();
            let p = frame.project([theta.sin(), 0.0, theta.cos()]);
            let off = (p[0] - 2456.0 - mean_radius(theta)).abs();
            worst = worst.max(off);
            // The edge midpoints of this frame sit 63° off the axis.
            if i <= 126 {
                worst_to_edge = worst_to_edge.max(off);
            }
        }
        assert!(worst_to_edge < 1.0, "radial law off by {worst_to_edge} px inside the edges");
        assert!(worst < 3.0, "radial law off by {worst} px in the corners");
        // The corner of a 10K Crop frame is 88° off the axis.
        assert!((frame.reach_deg() - 88.3).abs() < 0.5, "reach {}", frame.reach_deg());
        assert_eq!(frame.rim_px, 8.0);
    }

    /// A frame of another size is the same lens at that scale.
    #[test]
    fn a_scaled_frame_is_the_same_lens() {
        let (l, r) = crop_10k();
        let full = LensFrame::from_sources(&[&l, &r], 4912, 4912).unwrap();
        let small = LensFrame::from_sources(&[&l, &r], 4096, 4096).unwrap();
        assert!((small.focal / full.focal - 4096.0 / 4912.0).abs() < 1e-6);
        assert_eq!((small.k1, small.k2, small.xi), (full.k1, full.k2, full.xi));
        assert!((small.reach_deg() - full.reach_deg()).abs() < 1e-3);
        // One eye alone gives a lens too (a source that holds one).
        let one = LensFrame::from_sources(&[&l], 4912, 4912).unwrap();
        assert!((one.focal / full.focal - 1.0).abs() < 0.003);
    }

    /// Pixel → ray → pixel is the identity, the centre looks straight ahead,
    /// and up in the frame is up in the world.
    #[test]
    fn rays_and_pixels_agree() {
        let (l, r) = crop_10k();
        let frame = LensFrame::from_sources(&[&l, &r], 4912, 4912).unwrap();
        let centre = frame.ray(2456.0, 2456.0).unwrap();
        assert!(centre[0].abs() < 1e-12 && centre[1].abs() < 1e-12 && (centre[2] - 1.0).abs() < 1e-12);
        let up = frame.ray(2456.0, 1000.0).unwrap();
        assert!(up[1] > 0.0 && up[0].abs() < 1e-12);
        let right = frame.ray(4000.0, 2456.0).unwrap();
        assert!(right[0] > 0.0 && right[1].abs() < 1e-12);
        for &(px, py) in &[(0.5, 0.5), (4911.5, 0.5), (100.0, 4800.0), (2456.0, 2456.0), (3000.25, 1200.75)] {
            let d = frame.ray(px, py).expect("inside the model");
            assert!((d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - 1.0).abs() < 1e-12);
            let p = frame.project(d);
            assert!((p[0] - px).abs() < 1e-6 && (p[1] - py).abs() < 1e-6, "({px}, {py}) → {p:?}");
        }
    }

    /// What goes into the file: the frame's own lens, stated one degree past
    /// its corner.
    #[test]
    fn the_file_states_the_frame_lens() {
        let (l, r) = crop_10k();
        let frame = LensFrame::from_sources(&[&l, &r], 4912, 4912).unwrap();
        let apmp = frame.apmp();
        assert_eq!((apmp.width, apmp.height, apmp.xi, apmp.focal, apmp.k1, apmp.k2),
                   (4912, 4912, frame.xi, frame.focal, frame.k1, frame.k2));
        assert_eq!(apmp.limit_deg, 89.0);
        // The calibration that drives the kernels carries the same numbers.
        let c = l.with_lens_frame(&frame);
        assert_eq!(c.out_lens, [frame.xi, frame.focal, frame.k1, frame.k2]);
        assert_eq!(c.out_rim, frame.rim_px);
    }

    /// Only a unified-camera-model source has such a lens.
    #[test]
    fn other_lens_families_have_no_lens_frame() {
        let kb = FisheyeCalib::new_pure_kb(1044.0, 1044.0, 1920.0, 1920.0, [0.08, -0.02, 0.001, 0.0], 3840.0, 3840.0);
        assert!(LensFrame::from_sources(&[&kb], 3840, 3840).is_none());
        let (l, _) = crop_10k();
        assert!(LensFrame::from_sources(&[&l, &kb], 4912, 4912).is_none());
        assert!(LensFrame::from_sources(&[], 4912, 4912).is_none());
        assert!(LensFrame::from_sources(&[&l.with_equirect_input()], 4912, 4912).is_none());
    }
}
