//! Global pano-map and per-eye stereo offset math.
//!
//! Ports the Python `_build_ypr_matrix` at `vr180_gui.py:12515-12523`
//! and the per-eye composition at `vr180_gui.py:12789-12804`. Both
//! sides (preview and export) compose
//!
//! ```text
//! R_final_eye = R_stab · R_view_eye
//! ```
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
    /// flip applies to the output ray first, so stabilization still acts
    /// in unflipped world space. The caller must ALSO swap the L/R eye
    /// assignment (an upside-down rig mirrors the eye positions); see
    /// `Settings::effective_swap_eyes`.
    ///
    /// The angles keep their meaning under it ([`Self::per_eye_matrices`]):
    /// the global ones act as the viewer sees them, and the stereo ones stay
    /// with their lens, so one alignment of a rig serves both ways it can be
    /// mounted.
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

    /// The adjustment for a source that holds ONE eye: there is no second
    /// eye to offset it against, so the stereo angles do not apply.
    pub fn single_eye(mut self) -> Self {
        self.stereo_yaw_deg = 0.0;
        self.stereo_pitch_deg = 0.0;
        self.stereo_roll_deg = 0.0;
        self
    }

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
    ///
    /// Upside down, each angle keeps what it means to the user:
    ///
    /// - The global angles act as the viewer sees them. The half turn is
    ///   innermost, so a yaw or a pitch written before it acts in the
    ///   camera's frame, which the viewer sees turned: it goes in with its
    ///   sign reversed, and the matrix is the upright one followed by the
    ///   half turn (`Rz(180°) · R`). A roll about the view axis is the same
    ///   in both frames.
    /// - The stereo angles are a rig's alignment: each lens's own small
    ///   rotation. The eye swap gives each OUTPUT eye the other lens, so the
    ///   offsets change sides with it: the left eye gets what the right eye
    ///   has upright (`R · Rz(180°)` of that eye). Measured on footage shot
    ///   upside down: with the offsets left on their output eye the two
    ///   eyes were twice the rig's own error apart (0.4° of roll), with
    ///   them on their lens as near as upright (0.03°).
    pub fn per_eye_matrices(&self) -> ([f32; 9], [f32; 9]) {
        let (flip, seen) = if self.upside_down { (180.0, -1.0) } else { (0.0, 1.0) };
        let left = ypr_to_mat3_row_major(
            seen * (self.pano_yaw_deg - self.stereo_yaw_deg),
            seen * (self.pano_pitch_deg - self.stereo_pitch_deg),
            self.pano_roll_deg - seen * self.stereo_roll_deg + flip,
        );
        let right = ypr_to_mat3_row_major(
            seen * (self.pano_yaw_deg + self.stereo_yaw_deg),
            seen * (self.pano_pitch_deg + self.stereo_pitch_deg),
            self.pano_roll_deg + seen * self.stereo_roll_deg + flip,
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

    /// Upside down, the global angles act as the viewer sees them: the view
    /// is the upright one with the same angles, followed by the half turn.
    #[test]
    fn upside_down_the_global_angles_act_as_seen() {
        let half_turn = ypr_to_mat3_row_major(0.0, 0.0, 180.0);
        for (yaw, pitch, roll) in [(12.0, 0.0, 0.0), (0.0, -7.0, 0.0), (0.0, 0.0, 5.0), (13.0, -7.0, 5.0)] {
            let upright = ViewAdjust { pano_yaw_deg: yaw, pano_pitch_deg: pitch, pano_roll_deg: roll, ..ViewAdjust::IDENTITY };
            let turned = ViewAdjust { upside_down: true, ..upright };
            let (l, r) = turned.per_eye_matrices();
            let expected = mat3_mul_row_major(&half_turn, &upright.per_eye_matrices().0);
            for i in 0..9 {
                assert!(approx_eq(l[i], expected[i]) && approx_eq(r[i], expected[i]), "yaw {yaw} pitch {pitch} roll {roll} [{i}]: {} {} against {}", l[i], r[i], expected[i]);
            }
        }
        // A yaw to the right is a yaw to the right: the ray the centre of the picture looks along moves the same way
        // in the scene, whichever way up the camera is. (Upright, the centre ray goes to +x; upside down the scene's
        // right is the camera's -x.)
        let centre = |m: &[f32; 9]| [m[2], m[5], m[8]];
        let upright = ViewAdjust { pano_yaw_deg: 10.0, ..ViewAdjust::IDENTITY };
        let turned = ViewAdjust { upside_down: true, ..upright };
        let (u, t) = (centre(&upright.per_eye_matrices().0), centre(&turned.per_eye_matrices().0));
        assert!(u[0] > 0.1 && approx_eq(t[0], -u[0]) && approx_eq(t[1], -u[1]) && approx_eq(t[2], u[2]), "{u:?} {t:?}");
    }

    /// Upside down, the stereo offsets stay with their lens: the left eye,
    /// which then shows the lens the right eye shows upright, gets that
    /// eye's offsets, ahead of the half turn.
    #[test]
    fn upside_down_the_stereo_offsets_stay_with_their_lens() {
        let half_turn = ypr_to_mat3_row_major(0.0, 0.0, 180.0);
        let upright = ViewAdjust { stereo_yaw_deg: -0.07, stereo_pitch_deg: 0.08, stereo_roll_deg: -0.23, ..ViewAdjust::IDENTITY };
        let turned = ViewAdjust { upside_down: true, ..upright };
        let (l_up, r_up) = upright.per_eye_matrices();
        let (l, r) = turned.per_eye_matrices();
        let (l_expected, r_expected) = (mat3_mul_row_major(&r_up, &half_turn), mat3_mul_row_major(&l_up, &half_turn));
        for i in 0..9 {
            assert!(approx_eq(l[i], l_expected[i]) && approx_eq(r[i], r_expected[i]), "[{i}]: {} {} against {} {}", l[i], r[i], l_expected[i], r_expected[i]);
        }
        // Nothing but the half turn where nothing is set, and no identity then.
        let bare = ViewAdjust { upside_down: true, ..ViewAdjust::IDENTITY };
        assert!(!bare.is_identity());
        let (l, r) = bare.per_eye_matrices();
        for i in 0..9 {
            assert!(approx_eq(l[i], half_turn[i]) && approx_eq(r[i], half_turn[i]));
        }
    }
}
