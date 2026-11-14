//! What every arm does to one frame, in one place.
//!
//! The exports, the previews and the stills pose the two eyes, project
//! them, grade and compose them the same way; they differ in where a frame
//! comes from and where it goes. This module is the common part, growing
//! one step at a time out of the arms that each carried a copy of it.

use crate::gpu::EquirectRotation;
use crate::panomap::ViewAdjust;
use vr180_fisheye::DjiOsvImu;

/// The per-row rolling-shutter correction of a clip: what turns the IMU's
/// high-rate samples into one small rotation per source row.
#[derive(Debug, Clone, Copy)]
pub struct RowReadout {
    /// Sensor readout time of the frame, seconds.
    pub readout_s: f32,
    /// Rows of the picture being projected (the working height, not the
    /// native one, where the two differ).
    pub src_h: u32,
    pub fps: f32,
}

/// How the two eyes of one frame are posed: the rotation each eye's
/// projection takes, and each eye's per-row matrices when it has them.
#[derive(Debug, Clone)]
pub struct FramePose {
    pub rot_left: EquirectRotation,
    pub rot_right: EquirectRotation,
    pub rows_left: Option<Vec<f32>>,
    pub rows_right: Option<Vec<f32>>,
}

impl FramePose {
    /// `rot` is the frame's stabilization rotation (identity when off),
    /// `stab_idx` its row in the IMU's per-frame tables, `swapped` the eye
    /// swap. Each eye gets its lens's own rotation where the source's two
    /// lenses expose at different times ([`crate::dji_imu::per_eye_rotations`]),
    /// then the view adjustment (pano-map + stereo offset) on top; a view
    /// adjustment of zero leaves the rotation exactly as it was. `rows`
    /// asks for the per-row matrices; without an IMU there are none.
    pub fn new(
        rot: EquirectRotation,
        imu: Option<&DjiOsvImu>,
        stab_idx: usize,
        swapped: bool,
        view: &ViewAdjust,
        rows: Option<RowReadout>,
    ) -> Self {
        let (left, right) = crate::dji_imu::per_eye_rotations(rot, imu, stab_idx, swapped);
        let (rot_left, rot_right) = if view.is_identity() {
            (left, right)
        } else {
            let (v_l, v_r) = view.per_eye_matrices();
            (
                EquirectRotation(crate::panomap::mat3_mul_row_major(&left.0, &v_l)),
                EquirectRotation(crate::panomap::mat3_mul_row_major(&right.0, &v_r)),
            )
        };
        let (rows_left, rows_right) = match (rows, imu) {
            (Some(r), Some(imu)) => crate::dji_imu::per_eye_rs_rows(
                imu, stab_idx, r.readout_s, r.src_h, r.fps, swapped),
            _ => (None, None),
        };
        Self { rot_left, rot_right, rows_left, rows_right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_view_adjustment_and_no_imu_is_the_stab_rotation_on_both_eyes() {
        let rot = EquirectRotation([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let pose = FramePose::new(rot, None, 3, false, &ViewAdjust::IDENTITY,
            Some(RowReadout { readout_s: 0.015, src_h: 16, fps: 30.0 }));
        assert_eq!(pose.rot_left.0, rot.0);
        assert_eq!(pose.rot_right.0, rot.0);
        assert!(pose.rows_left.is_none() && pose.rows_right.is_none());
    }

    #[test]
    fn the_view_adjustment_goes_on_top_of_each_eye() {
        let rot = EquirectRotation([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let view = ViewAdjust { pano_yaw_deg: 10.0, stereo_yaw_deg: 2.0, ..ViewAdjust::IDENTITY };
        let pose = FramePose::new(rot, None, 0, false, &view, None);
        let (v_l, v_r) = view.per_eye_matrices();
        assert_eq!(pose.rot_left.0, crate::panomap::mat3_mul_row_major(&rot.0, &v_l));
        assert_eq!(pose.rot_right.0, crate::panomap::mat3_mul_row_major(&rot.0, &v_r));
        assert_ne!(pose.rot_left.0, pose.rot_right.0);
    }
}
