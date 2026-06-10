//! Global pano-map and per-eye stereo offset math.
//!
//! Ports the Python `_build_ypr_matrix` at `vr180_gui.py:12515-12523`
//! and the per-eye composition at `vr180_gui.py:12789-12804`. Both
//! sides (preview and export) compose
//!
//!     R_final_eye = R_stab · R_view_eye
//!
//! where `R_view_eye` is built from `(global ± stereo)` Tait-Bryan
//! angles. Convention is `R = R_y(yaw) · R_x(pitch) · R_z(roll)` —
//! same intrinsic order the Python uses.

/// Build a row-major 3×3 rotation matrix from yaw/pitch/roll degrees.
///
/// `R = R_y(yaw) · R_x(pitch) · R_z(roll)`. Identity when all three
/// inputs are 0 — the caller relies on that to skip the multiply in
/// the hot path when the sliders are at their defaults.
pub fn ypr_to_mat3_row_major(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> [f32; 9] {
    let y = yaw_deg.to_radians();
    let p = pitch_deg.to_radians();
    let r = roll_deg.to_radians();
    let (cy, sy) = (y.cos(), y.sin());
    let (cp, sp) = (p.cos(), p.sin());
    let (cr, sr) = (r.cos(), r.sin());
    [
        cy * cr + sy * sp * sr,    -cy * sr + sy * sp * cr,    sy * cp,
        cp * sr,                    cp * cr,                   -sp,
        -sy * cr + cy * sp * sr,    sy * sr + cy * sp * cr,    cy * cp,
    ]
}

/// Row-major 3×3 matrix product: `out = a · b`.
pub fn mat3_mul_row_major(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = s;
        }
    }
    out
}

/// Per-eye view-adjustment angles. `pano_*` are the global offsets
/// (shared between eyes); `stereo_*` flip sign between left and
/// right (right = global + stereo, left = global − stereo — same
/// convention as the Python app at `vr180_gui.py:12433-12438`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewAdjust {
    pub pano_yaw_deg: f32,
    pub pano_pitch_deg: f32,
    pub pano_roll_deg: f32,
    pub stereo_yaw_deg: f32,
    pub stereo_pitch_deg: f32,
    pub stereo_roll_deg: f32,
    /// Camera mounted upside-down: rotate the output 180° in-plane.
    /// Because the YPR convention is `R = Ry·Rx·Rz(roll)` with roll
    /// innermost, this is EXACTLY `roll += 180°` on both eyes — the
    /// flip applies to the output ray first, so pano/stereo sliders
    /// and stabilization still act in unflipped world space. The
    /// caller must ALSO swap the L/R eye assignment (an upside-down
    /// rig mirrors the eye positions); see `Settings::effective_swap_eyes`.
    pub upside_down: bool,
}

impl ViewAdjust {
    pub const IDENTITY: Self = Self {
        pano_yaw_deg: 0.0,
        pano_pitch_deg: 0.0,
        pano_roll_deg: 0.0,
        stereo_yaw_deg: 0.0,
        stereo_pitch_deg: 0.0,
        stereo_roll_deg: 0.0,
        upside_down: false,
    };

    /// Fast path: when every slider is at 0 the view matrix is
    /// identity and callers can skip the matrix multiply entirely.
    pub fn is_identity(&self) -> bool {
        self.pano_yaw_deg == 0.0
            && self.pano_pitch_deg == 0.0
            && self.pano_roll_deg == 0.0
            && self.stereo_yaw_deg == 0.0
            && self.stereo_pitch_deg == 0.0
            && self.stereo_roll_deg == 0.0
            && !self.upside_down
    }

    /// Build `(R_view_left, R_view_right)` row-major 3×3 matrices.
    pub fn per_eye_matrices(&self) -> ([f32; 9], [f32; 9]) {
        let flip = if self.upside_down { 180.0 } else { 0.0 };
        let left = ypr_to_mat3_row_major(
            self.pano_yaw_deg - self.stereo_yaw_deg,
            self.pano_pitch_deg - self.stereo_pitch_deg,
            self.pano_roll_deg - self.stereo_roll_deg + flip,
        );
        let right = ypr_to_mat3_row_major(
            self.pano_yaw_deg + self.stereo_yaw_deg,
            self.pano_pitch_deg + self.stereo_pitch_deg,
            self.pano_roll_deg + self.stereo_roll_deg + flip,
        );
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }

    #[test]
    fn ypr_zero_is_identity() {
        let m = ypr_to_mat3_row_major(0.0, 0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(m[i * 3 + j], expected),
                    "[{i},{j}] got {}, expected {expected}", m[i * 3 + j]);
            }
        }
    }

    #[test]
    fn mat3_mul_identity_left_is_noop() {
        let id = ypr_to_mat3_row_major(0.0, 0.0, 0.0);
        let m = ypr_to_mat3_row_major(13.0, -7.0, 5.0);
        let out = mat3_mul_row_major(&id, &m);
        for i in 0..9 { assert!(approx_eq(out[i], m[i])); }
    }

    #[test]
    fn yaw_90_rotates_z_to_x() {
        // R_y(90°) · [0, 0, 1] should give [1, 0, 0].
        let m = ypr_to_mat3_row_major(90.0, 0.0, 0.0);
        let v = [0.0_f32, 0.0, 1.0];
        let out = [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ];
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 0.0));
        assert!(approx_eq(out[2], 0.0));
    }

    #[test]
    fn view_adjust_identity_short_circuit() {
        let v = ViewAdjust::IDENTITY;
        assert!(v.is_identity());
        let (l, r) = v.per_eye_matrices();
        let id = ypr_to_mat3_row_major(0.0, 0.0, 0.0);
        for i in 0..9 {
            assert!(approx_eq(l[i], id[i]));
            assert!(approx_eq(r[i], id[i]));
        }
    }

    #[test]
    fn stereo_offset_splits_left_right() {
        let v = ViewAdjust {
            pano_yaw_deg: 5.0,
            stereo_yaw_deg: 2.0,
            ..ViewAdjust::IDENTITY
        };
        let (l, r) = v.per_eye_matrices();
        // Left should be yaw=3, right should be yaw=7. Compare to
        // direct builds.
        let l_expected = ypr_to_mat3_row_major(3.0, 0.0, 0.0);
        let r_expected = ypr_to_mat3_row_major(7.0, 0.0, 0.0);
        for i in 0..9 {
            assert!(approx_eq(l[i], l_expected[i]));
            assert!(approx_eq(r[i], r_expected[i]));
        }
    }
}
