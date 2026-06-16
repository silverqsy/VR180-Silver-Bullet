//! BRAW gyro → per-frame stabilization rotations.
//!
//! Bridges `vr180-braw::BrawGyroData` (raw IMU samples from the
//! `braw_helper --gyro` subprocess) and `vr180-pipeline::gpu::EquirectRotation`
//! (the 3×3 rotation matrix the fisheye projection shader consumes).
//!
//! Algorithm matches `vr180_gui.py:1032-1200` (`BrawGyroStabilizer`):
//!
//! 1. Run VQF 6D (no magnetometer — BRAW gyro/accel only) to get a
//!    per-sample orientation quaternion.
//! 2. Sample one quat per video frame at `t_mid = (fi + 0.5) · dt`
//!    by nearest-index lookup.
//! 3. Reference is `q_ref = quat_at_frame_0`.
//! 4. Per-frame correction: `q_corr = q⁻¹ ⊗ q_ref` — left-multiply
//!    the actual inverse. Matches Python `BrawGyroStabilizer` and
//!    `DjiGyroStabilizer` at `vr180_gui.py:713,1109-1200`. The
//!    composition order is part of the empirically-calibrated pair
//!    `[formula × C_IMU_TO_CAM]`; changing one without the other
//!    breaks stabilization on real footage (verified — the
//!    theoretically-equivalent formula `q ⊗ q_ref⁻¹` looks correct
//!    on paper but produces no visible stab on real OSV clips).
//! 5. Apply IMU→camera basis transform
//!    `C = diag(+1, -1, +1)` (negate-Y only — matches
//!    `vr180_gui.py:1006`, empirically verified for Pyxis / URSA Cine
//!    Immersive).
//!
//! Rolling shutter: BRAW disables RS (`vr180_gui.py:3800-3801`). One
//! rotation per video frame, not per-row.

use crate::Result;
use crate::gpu::EquirectRotation;
use vr180_core::gyro::cori_iori::Quat;

/// Per-frame stabilization rotations + diagnostics.
#[derive(Debug)]
pub struct BrawStabResult {
    /// One rotation per video frame, length == `n_frames` requested.
    pub per_frame: Vec<EquirectRotation>,
    /// VQF's final bias estimate in deg/s (printed for diagnostics).
    pub bias_deg_s: [f32; 3],
    /// Number of gyro samples consumed (sanity check).
    pub gyro_sample_count: usize,
}

/// Compute per-frame stabilization rotations from a `BrawGyroData`.
///
/// `n_frames` is the number of video frames to produce (typically
/// equal to `BrawInfo::frame_count`).
///
/// Returns `Ok(BrawStabResult)` on success. If the gyro stream is
/// empty or shorter than what one frame needs, returns `n_frames`
/// identity rotations — callers get a clean fallback rather than an
/// error in the middle of the playback hot path.
pub fn compute_braw_stabilization(
    gyro_data: &vr180_braw::BrawGyroData,
    fps: f32,
    n_frames: usize,
) -> Result<BrawStabResult> {
    let n_samples = gyro_data.len();
    let identity_run = || BrawStabResult {
        per_frame: vec![EquirectRotation::IDENTITY; n_frames],
        bias_deg_s: [0.0; 3],
        gyro_sample_count: n_samples,
    };

    if n_samples < 8 || fps <= 0.0 || n_frames == 0 {
        tracing::warn!(
            "braw_imu: gyro={} fps={} n_frames={} — falling back to identity",
            n_samples, fps, n_frames
        );
        return Ok(identity_run());
    }

    // De-interleave the flat gyro/accel arrays into [[f32;3]; N].
    let mut gyro_xyz: Vec<[f32; 3]> = Vec::with_capacity(n_samples);
    let mut acc_xyz:  Vec<[f32; 3]> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let go = i * 3;
        gyro_xyz.push([
            gyro_data.gyro[go + 0],
            gyro_data.gyro[go + 1],
            gyro_data.gyro[go + 2],
        ]);
        acc_xyz.push([
            gyro_data.accel[go + 0],
            gyro_data.accel[go + 1],
            gyro_data.accel[go + 2],
        ]);
    }

    let gyr_ts = gyro_data.sample_period_s() as f32;
    if !gyr_ts.is_finite() || gyr_ts <= 0.0 {
        tracing::warn!("braw_imu: invalid gyr_ts {gyr_ts} — falling back to identity");
        return Ok(identity_run());
    }

    let t_vqf = std::time::Instant::now();
    let vqf = vr180_core::gyro::vqf::run(&gyro_xyz, &acc_xyz, None, gyr_ts);
    tracing::info!(
        "braw_imu: VQF 6D ran in {:.1?}, bias = {:?}°/s",
        t_vqf.elapsed(), vqf.bias_deg_s()
    );

    // Sample one quaternion per video frame at t_mid = (fi + 0.5) * dt.
    let dt = 1.0_f32 / fps;
    let mut frame_quats: Vec<Quat> = Vec::with_capacity(n_frames);
    for fi in 0..n_frames {
        let t_mid = (fi as f32 + 0.5) * dt;
        let mut idx = (t_mid / gyr_ts).round() as usize;
        if idx >= n_samples {
            idx = n_samples - 1;
        }
        frame_quats.push(vqf.quats[idx]);
    }

    // Reference = first frame's quat.
    let q_ref = frame_quats[0];

    // Build per-frame correction matrices.
    // R_imu = q_frame⁻¹ ⊗ q_ref, as a 3×3 matrix.
    // R_cam = C @ R_imu @ Cᵀ, where C = diag(+1, -1, +1).
    //
    // Composition matches Python BrawGyroStabilizer at
    // vr180_gui.py:1109-1200 (same `Q_inv * Q_ref` order as the OSV
    // path). The combined `formula × C_IMU_TO_CAM` pair is what's
    // empirically calibrated — see compute_dji_stabilization comment
    // for the full discussion of why we don't change the composition
    // independently of C.
    let mut per_frame = Vec::with_capacity(n_frames);
    for &q in &frame_quats {
        let q_corr = q.conjugate().mul(q_ref).normalize();
        let r = q_corr.to_mat3_row_major();
        // Apply C·R·Cᵀ. With C = diag(1,-1,1):
        //   C·R has rows 0 and 2 unchanged, row 1 negated.
        //   (C·R)·Cᵀ has columns 0 and 2 unchanged, column 1 negated.
        // Net result: r[0,1], r[1,0], r[1,2], r[2,1] all negate.
        let r_cam = [
             r[0],       -r[1],        r[2],
            -r[3],        r[4],       -r[5],
             r[6],       -r[7],        r[8],
        ];
        per_frame.push(EquirectRotation(r_cam));
    }

    Ok(BrawStabResult {
        per_frame,
        bias_deg_s: vqf.bias_deg_s(),
        gyro_sample_count: n_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vr180_braw::{BrawGyroData, BrawGyroHeader};

    /// Stationary IMU (gyro=0, accel=gravity) should produce
    /// near-identity per-frame rotations across all frames.
    #[test]
    fn stationary_gyro_produces_near_identity_rotations() {
        // Fake 800 Hz, 5 seconds → 4000 samples.
        let n = 4000;
        let mut gyro = vec![0f32; n * 3];
        let mut accel = vec![0f32; n * 3];
        for i in 0..n {
            accel[i * 3 + 2] = 9.81; // +Z = gravity
            let _ = gyro[i * 3]; // silence unused-warn
        }
        let header = BrawGyroHeader {
            gyro_sample_count: n as u64,
            gyro_sample_rate: 800.0,
            accel_sample_count: n as u64,
            accel_sample_rate: 800.0,
            sample_count: n as u64,
            frame_rate: 24.0,
        };
        let data = BrawGyroData { header, gyro, accel };
        let stab = compute_braw_stabilization(&data, 24.0, 120)
            .expect("compute_braw_stabilization");
        assert_eq!(stab.per_frame.len(), 120);
        // Every frame's rotation should be near identity.
        for (i, rot) in stab.per_frame.iter().enumerate() {
            let m = rot.0;
            // Diagonal close to 1, off-diagonal close to 0. The
            // very first few frames will be drifty (VQF inclination
            // correction not converged yet), so skip them.
            if i < 10 { continue; }
            let trace = m[0] + m[4] + m[8];
            assert!(
                (trace - 3.0).abs() < 0.05,
                "frame {i} trace = {trace:.6} (expected ~3.0)"
            );
        }
    }

    /// Empty gyro returns identity rotations cleanly (no panic).
    #[test]
    fn empty_gyro_returns_identity() {
        let header = BrawGyroHeader {
            gyro_sample_count: 0,
            gyro_sample_rate: 800.0,
            accel_sample_count: 0,
            accel_sample_rate: 800.0,
            sample_count: 0,
            frame_rate: 24.0,
        };
        let data = BrawGyroData { header, gyro: vec![], accel: vec![] };
        let stab = compute_braw_stabilization(&data, 24.0, 60)
            .expect("identity fallback");
        assert_eq!(stab.per_frame.len(), 60);
        for rot in &stab.per_frame {
            assert_eq!(rot.0, EquirectRotation::IDENTITY.0);
        }
    }
}
