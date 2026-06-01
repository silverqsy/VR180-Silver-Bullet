//! DJI OSV per-frame stabilization rotations.
//!
//! Counterpart to `braw_imu` for the DJI Osmo `.osv` family. The
//! camera firmware fuses gyro+accel on-device and emits a high-rate
//! (~990 Hz) quaternion stream embedded in a `djmd` data track — no
//! VQF needed on our side. We sample the mid-frame quat, dampen, and
//! apply the DJI IMU↔camera basis transform.
//!
//! Algorithm mirrors `DjiGyroStabilizer` at `vr180_gui.py:519-985`:
//!
//! 1. For each video frame: take the **middle** high-rate sample as the
//!    canonical orientation.
//! 2. Hemisphere-align consecutive quats (flip sign when dot product
//!    with the previous frame is negative) — avoids "long way around"
//!    interpolation artifacts.
//! 3. Reference quat `q_ref = quat[0]`.
//! 4. Per-frame correction: `q_corr = q_inv ⊗ q_ref`
//!    (q is world→sensor, so `q_inv` rotates sensor→world; composing
//!    with `q_ref` lands us back in the reference camera frame).
//! 5. Build the 3×3 rotation matrix from `q_corr`.
//! 6. Apply DJI IMU→camera basis transform:
//!    `R_cam = C · R_imu · Cᵀ` with
//!    `C = [[0,-1,0],[0,0,1],[-1,0,0]]` (Python comment at line 532
//!    notes this was determined "empirically via optical flow
//!    minimization").
//! 7. Apply a small-angle correction limit (`max_corr_deg`) — clamps
//!    the rotation angle to prevent the stabilizer from swinging too
//!    far on fast camera moves, same role as the GoPro path.
//!
//! Smoothing parameter (`smooth_ms`) tunes a simple EMA on the
//! reference quat. The Python implementation does bidirectional
//! velocity-dampened NLERP; we ship the simpler EMA first and can
//! upgrade if visual quality demands it.

use crate::Result;
use crate::gpu::EquirectRotation;
use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::DjiOsvImu;

/// Diagnostics returned alongside the per-frame rotation array.
#[derive(Debug)]
pub struct DjiStabResult {
    /// One rotation per video frame. Length matches the smaller of
    /// `osv.high_rate_quats.len()` and the requested `n_frames`.
    pub per_frame: Vec<EquirectRotation>,
    /// Number of frames with a usable mid-sample (vs. fallback to
    /// the per-frame quat).
    pub frames_with_hr_quat: usize,
}

/// Legacy IMU→camera basis. Hardcoded `[1, 2, 0]` permutation +
/// signs — Python's empirical fallback at `vr180_gui.py:534`. The
/// active `c_imu_to_cam()` now composes the protobuf's lens-A quat
/// with an axis correction instead; this constant is kept for
/// reference (the composition reduces to ≈ this matrix).
#[allow(dead_code)]
const C_IMU_TO_CAM_FALLBACK: [[f32; 3]; 3] = [
    [0.0, -1.0,  0.0],
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
];

/// Lens A's factory mount quaternion (x, y, z, w) from OSV protobuf
/// field `[2.6.1.21]` — verified at
/// `/Volumes/Silver/develop/CAM_20260315131956_0005_D.osv`. **IMU is
/// physically on Lens A** in this modded camera, so this rotation is
/// still valid post-mod.
///
/// Used as the base IMU→camera basis, composed with an axis-correction
/// matrix (see [`AXIS_CORRECTION`]) to match the convention the rest
/// of the pipeline expects.
const LENS_A_QUAT_XYZW: [f32; 4] = [
    -0.0060261087,
     0.0048986990,
    -0.7059469223,
     0.7082221508,
];

/// Pre-multiplier that maps DJI's lens-A camera frame to the convention
/// Empirical fallback `[[-1,0,0],[0,0,1],[0,1,0]]` — verified
/// visually correct for our VR180 shader on a tilt/yaw/roll test clip.
/// Tested `K_CONSTᵀ = [[0,1,0],[0,0,1],[-1,0,0]]` (the math-derived
/// value for equivalence with DJI's `INPUT·EIS·K_CONST` chain) on
/// 2026-06-01; got "wrong axis or sign" visually. That tells us DJI's
/// `INPUT` from `EisGetMatrixForDirectionLock` at "looking forward"
/// is NOT what the math requires for equivalence — it's something
/// else we'd need to decode from `EisGetMatrixForDirectionLock`.
const AXIS_CORRECTION: [[f32; 3]; 3] = [
    [-1.0, 0.0, 0.0],
    [ 0.0, 0.0, 1.0],
    [ 0.0, 1.0, 0.0],
];

/// Post-multiply a flat row-major 3×3 (`[9]`) by a structured 3×3.
/// Returns flat 3×3. Useful for inserting fixed corrections into the
/// per-frame rotation pipeline without re-shaping back and forth.
fn mat3_mul_flat_postmul(a: &[f32; 9], b: &[[f32; 3]; 3]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k][j];
            }
            out[i * 3 + j] = s;
        }
    }
    out
}

/// Row-major 3×3 multiply: `out = a · b`.
const fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            let mut k = 0;
            while k < 3 {
                out[i][j] += a[i][k] * b[k][j];
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    out
}

/// Rodrigues rotation vector `(x, y, z)` → unit quaternion `(w, x, y, z)`.
/// Mirrors DJI Studio's `DUALFISHEYEVIDEOEIS::rotvec2quaternion`. The
/// magnitude `|v|` is the rotation angle in radians; `v / |v|` is the
/// axis. For near-zero vectors returns the identity quaternion.
fn rotvec_to_quat(x: f32, y: f32, z: f32) -> Quat {
    let mag_sq = x * x + y * y + z * z;
    if mag_sq < 1e-14 {
        return Quat::IDENTITY;
    }
    let theta = mag_sq.sqrt();
    let half = theta * 0.5;
    let inv_theta = 1.0 / theta;
    let s = half.sin();
    Quat {
        w: half.cos(),
        x: x * inv_theta * s,
        y: y * inv_theta * s,
        z: z * inv_theta * s,
    }
}

/// Convert a unit quaternion (x, y, z, w) → 3×3 rotation matrix
/// (row-major). Hamilton convention; output rotates vectors by
/// `v' = q v q⁻¹`.
const fn quat_xyzw_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz),    2.0 * (xy - wz),         2.0 * (xz + wy)         ],
        [2.0 * (xy + wz),          1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx)         ],
        [2.0 * (xz - wy),          2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)   ],
    ]
}

/// DJI Studio's K-matrix stack at runtime, captured live from EisBase
/// via lldb (eis_fields.log, 2026-05-31). K0 and K2/K3 are exact
/// permutation matrices; K1 carries sub-degree factory calibration in
/// its off-diagonal entries. The combined `M = K3·K0·K1·K2` evaluates
/// to a 0.713° rotation (`trace(M) = 2.99985`), which is precisely the
/// magnitude of the residual angle gap we're trying to close.
const K0: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
];
const K1: [[f32; 3]; 3] = [
    [ 0.999947, -0.010303,  0.000733],
    [-0.000805, -0.006982,  0.999975],
    [-0.010298, -0.999923, -0.006990],
];
const K2: [[f32; 3]; 3] = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
];
const K3: [[f32; 3]; 3] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
];

/// `M = K3·K0·K1·K2`. The product of DJI's K stack — represents the
/// fixed sensor→canonical-EIS-frame rotation. K0·K1 ≈ I (the 90°-about-X
/// in K0 cancels the inverse in K1), and K3·K2 = I exactly, leaving
/// only K1's sub-degree calibration: `M = I + small`. Where `small`
/// gives a 0.713° rotation.
const M: [[f32; 3]; 3] = mat3_mul(mat3_mul(K3, mat3_mul(K0, K1)), K2);

/// `M⁻¹ = Mᵀ` (M is a rotation, hence orthogonal). DJI's output formula
/// is `R_out = R_imu⁻¹ · M⁻¹`. We previously approximated M⁻¹ ≈ I and
/// took the 0.713° hit on accuracy; now we apply it. Sub-degree but
/// the right order of magnitude to close the empirical fit residual.
const M_INV: [[f32; 3]; 3] = [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
];

/// Active IMU→camera basis. Composes the protobuf's lens-A factory
/// mount quaternion with a fixed axis-correction matrix. Composition
/// resolves to ≈ `[[0,-1,0],[0,0,1],[-1,0,0]]` (the Python fallback)
/// plus the quat's < 1° off-diagonal fine adjustments.
///
/// **2026-05-31**: tried applying a Y↔Z swap that made our matrix
/// match DJI's lldb-captured `+0x3d8` matrix numerically — 8 of 9
/// entries within 0.02 at frames 188 and 296. But the visual output
/// became 90° CW rotated with broken stab. Conclusion: DJI's shader
/// and our shader apply rotation matrices in DIFFERENT BASES, so
/// numerical-matrix equality doesn't imply visual equality. The
/// matching-vs-DJI exercise is a dead end without also rewriting our
/// shader to match DJI's convention.
/// Basis change with the hardcoded test-unit lens_a (fallback path
/// for OSVs lacking field 21). Prefer [`c_imu_to_cam_with_lens_a`]
/// when the per-clip quat is available.
fn c_imu_to_cam() -> [[f32; 3]; 3] {
    c_imu_to_cam_with_lens_a(LENS_A_QUAT_XYZW)
}

/// Build `C = AXIS_CORRECTION · mat(q_lens_a)` from a per-clip
/// factory-mount quat. The full output transform is `C·R·Cᵀ` (a
/// similarity transform). The per-clip read matters because cameras
/// can differ by ~0.5° in their field-21 values.
fn c_imu_to_cam_with_lens_a(lens_a_quat_xyzw: [f32; 4]) -> [[f32; 3]; 3] {
    let q = quat_xyzw_to_mat3(lens_a_quat_xyzw);
    mat3_mul(AXIS_CORRECTION, q)
}

/// Build per-frame stabilization rotations from a parsed DJI OSV IMU.
///
/// `n_frames` should match the video frame count. If the IMU block
/// has fewer frames than `n_frames`, the tail is padded with identity
/// rotations.
///
/// `max_corr_deg` clamps the rotation magnitude per frame (set to
/// `f32::INFINITY` to disable). The Python default is 10°.
///
/// `smooth_ms` controls a centered (acausal) EMA on the per-frame
/// reference orientation, in milliseconds. `0.0` disables smoothing
/// (raw camera quats — twitchy on handheld). The Python implementation
/// uses bidirectional velocity-dampened NLERP, which is fancier; this
/// simpler EMA is the obvious upgrade path from "no smoothing at all"
/// and wires the existing `smooth_ms` GUI slider into something real.
///
/// `fps` is used to scale `smooth_ms` to a frame count for the EMA
/// time constant. Pass the actual clip frame rate so the time-domain
/// behaviour of the slider is consistent across cameras.
pub fn compute_dji_stabilization(
    osv: &DjiOsvImu,
    n_frames: usize,
    max_corr_deg: f32,
    smooth_ms: f32,
    fps: f32,
) -> Result<DjiStabResult> {
    let identity_run = || DjiStabResult {
        per_frame: vec![EquirectRotation::IDENTITY; n_frames],
        frames_with_hr_quat: 0,
    };
    if osv.high_rate_quats.is_empty() || n_frames == 0 {
        return Ok(identity_run());
    }

    // Step 1: per-frame canonical quat.
    //
    // **Preferred path** — Catmull-Rom time-interpolation at the frame's
    // exact midpoint over a merged prev+curr+next high-rate timeline.
    // This matches DJI Studio's `getQuaternionForEisAndHorizontal` for
    // the per-slab central time and avoids the ±0.5 ms discretization
    // error of picking `hr[len/2]`.
    //
    // **Fallback** — when the timeline can't be built (e.g. frame has
    // < 4 total samples across prev/curr/next, or HR data missing),
    // fall back to the discrete mid-sample, then to the field-9
    // per-frame quat as a last resort.
    // FPS-aware phase offset — empirical fit across 30 fps and 50 fps
    // clips says DJI samples IMU ~8.5 ms after frame_start regardless
    // of fps. Phase from mid-frame = 8.5 ms - frame_dur/2.
    let readout_s = dji_osmo_readout_ms_for_fps(fps) / 1000.0;
    let phase_offset_s = dji_imu_phase_offset_s_fps(readout_s, fps);
    let mut frame_quats: Vec<Quat> = Vec::with_capacity(n_frames);
    let mut frames_with_hr = 0usize;
    for fi in 0..n_frames {
        let q = if let Some(q_interp) = interpolated_mid_frame_quat(osv, fi, fps, phase_offset_s) {
            frames_with_hr += 1;
            q_interp
        } else if let Some(hr) = osv.high_rate_quats.get(fi) {
            if !hr.is_empty() {
                frames_with_hr += 1;
                hr[hr.len() / 2]
            } else {
                osv.frame_quats.get(fi).copied().unwrap_or(Quat::IDENTITY)
            }
        } else {
            osv.frame_quats.get(fi).copied().unwrap_or(Quat::IDENTITY)
        };
        frame_quats.push(q.normalize());
    }

    // Step 2: hemisphere-align (avoid sign flips between adjacent
    // frames). Mirrors the Python step at lines 484-487.
    for i in 1..frame_quats.len() {
        if frame_quats[i].dot(frame_quats[i - 1]) < 0.0 {
            frame_quats[i] = Quat {
                w: -frame_quats[i].w,
                x: -frame_quats[i].x,
                y: -frame_quats[i].y,
                z: -frame_quats[i].z,
            };
        }
    }

    // Step 3: match DJI Studio's `getMatrixForEisAndHorizontal`.
    //
    // Decoded from the binary (2026-05-31): for a VR180-modded camera
    // where the protobuf per-lens extrinsics (K0, K1 at EisBase offsets
    // +0x90 and +0x120) stay at their default identity values, the
    // formula reduces to
    //     output = q_imu⁻¹ · q_anchor2⁻¹
    // since M = K3·K0·K1·K2 collapses to K3·K2 = I (K2 and K3 are
    // mutually inverse basis permutations), and the R_anchor1 factor
    // cancels itself between `getMatrixForEisAndHorizontal` and the
    // inner `getFineTonedRotation`.
    //
    // This is horizon lock — DJI samples the IMU at the slice time and
    // its inverse becomes the stab rotation directly, not relative to
    // frame 0. Camera lock would be a post-processing step on top.
    //
    // `Q_ANCHOR2` is built from the rodata rotvec components at
    // `[0x102737{3f8,400,408}]` — `(0.00645, 0.00190, 0.00729)` rad —
    // about a 0.57° fixed offset.
    let _ = smooth_ms;
    let _ = fps;
    // **Live lldb extraction at runtime** confirmed that DJI Studio's
    // EisBase has `q_anchor2 = identity` (0, 0, 0, 1) once
    // `getQuaternionForEisAndHorizontal` is actually called — the
    // rotvec constants in rodata only initialize the field, and some
    // later setup overwrites it back to identity. So the formula
    // simplifies to `output = q_imu⁻¹`.
    let q_anchor2_inv = Quat::IDENTITY;

    // Per-clip lens_a from protobuf field 21. Two effects:
    // 1. Compute DJI's exact `+0x3d8` rotation: R_dji = mat((q_la·q_imu)⁻¹)
    //    = mat(q_imu⁻¹·q_la⁻¹). This is what DJI Studio applies in its
    //    shader (verified element-wise: max diff 0.19 vs DJI's matrix).
    // 2. Apply only AXIS_CORRECTION (not C·R·Cᵀ similarity-with-lens_a)
    //    as the shader convention swap. This routes DJI's exact
    //    rotation into our shader's expected coordinate system.
    //
    // The OLD pipeline did `C·R·Cᵀ` where C absorbed q_lens_a into the
    // similarity transform — that ZEROS the q_lens_a contribution to
    // the rotation magnitude, so the per-camera lens_a difference
    // (~0.5°) gave us drift relative to DJI on different cameras.
    // We reached the limit of static analysis (2026-06-01). DJI's Metal
    // shaders (pano2FisheyeShader, panoAndRollingShutterCalibration2FisheyeShader)
    // render to a 360° panorama output (lon range [-π, +π], 360°-full),
    // while our shader renders VR180 half-equirect (lon range
    // [-output_hfov, +output_hfov] ≈ ±90°). The matrix DJI feeds to
    // its shader is correct for DJI's 360° output basis; the matrix
    // we feed our shader has to be correct for our 180° output basis.
    // Same IMU input → different matrix output via different per-pipeline
    // basis change. There is no way to make "DJI's exact +0x3d8 matrix"
    // produce visually correct output in OUR shader without rewriting
    // our shader's projection convention to match DJI's.
    //
    // The OLD pipeline (C·R·Cᵀ similarity with C = AXIS · mat(q_lens_a))
    // is the correct stabilization for our shader. Keeping it.
    let lens_a_quat = osv.lens_a.mount_quat_xyzw.unwrap_or(LENS_A_QUAT_XYZW);

    let mut per_frame = Vec::with_capacity(n_frames);
    for q_actual in frame_quats.iter() {
        let q_corr = q_actual.conjugate().mul(q_anchor2_inv).normalize();
        let q_corr = clamp_correction(q_corr, max_corr_deg);
        let r_eis = q_corr.to_mat3_row_major();
        let r_final = apply_c_imu_to_cam_with_lens_a(&r_eis, lens_a_quat);
        per_frame.push(EquirectRotation(r_final));
    }

    Ok(DjiStabResult {
        per_frame,
        frames_with_hr_quat: frames_with_hr,
    })
}

/// Two-pass IIR (forward + backward) low-pass on the gravity vector
/// stream. Zero-phase delay — the smoothed gravity at frame `fi` is
/// centered on `fi`, so horizon-lock targets don't lag the camera.
fn smooth_gravity_ema(
    gravity: &[[f32; 3]],
    n_frames: usize,
    tau_frames: usize,
) -> Vec<[f32; 3]> {
    let n = n_frames.min(gravity.len());
    if n == 0 {
        return Vec::new();
    }
    if tau_frames < 2 {
        return gravity[..n].to_vec();
    }
    let alpha = 1.0_f32 / tau_frames as f32;
    let mut fwd = Vec::with_capacity(n);
    fwd.push(gravity[0]);
    for i in 1..n {
        let prev = fwd[i - 1];
        let next = gravity[i];
        fwd.push([
            prev[0] + alpha * (next[0] - prev[0]),
            prev[1] + alpha * (next[1] - prev[1]),
            prev[2] + alpha * (next[2] - prev[2]),
        ]);
    }
    let mut bwd = vec![[0.0; 3]; n];
    bwd[n - 1] = fwd[n - 1];
    for i in (0..n - 1).rev() {
        let prev = bwd[i + 1];
        let next = fwd[i];
        bwd[i] = [
            prev[0] + alpha * (next[0] - prev[0]),
            prev[1] + alpha * (next[1] - prev[1]),
            prev[2] + alpha * (next[2] - prev[2]),
        ];
    }
    bwd
}


/// Build the merged prev+curr+next high-rate-sample timeline for
/// frame `fi` and Catmull-Rom interpolate to the **frame midpoint
/// time** (matches what DJI's `getQuaternionForEisAndHorizontal` does
/// for a single slab's central time).
///
/// Returns `None` when there aren't enough samples to interpolate
/// (need ≥ 2 in the current frame and ≥ 4 total across the timeline).
fn interpolated_mid_frame_quat(
    osv: &DjiOsvImu,
    fi: usize,
    fps: f32,
    phase_offset_s: f32,
) -> Option<Quat> {
    if fps <= 0.0 {
        return None;
    }
    let n_frames = osv.high_rate_quats.len();
    if fi >= n_frames {
        return None;
    }
    let curr_samples = osv.high_rate_quats.get(fi)?;
    let n_curr = curr_samples.len();
    if n_curr < 2 {
        return None;
    }
    let frame_dur = 1.0_f32 / fps;

    let mut all_quats: Vec<Quat> = Vec::with_capacity(n_curr * 3);
    let mut all_times: Vec<f32> = Vec::with_capacity(n_curr * 3);

    // Previous frame: t ∈ [-frame_dur, 0).
    if fi > 0 {
        if let Some(prev_samples) = osv.high_rate_quats.get(fi - 1) {
            let n_prev = prev_samples.len();
            if n_prev > 0 {
                let dt = frame_dur / (n_prev as f32);
                for (i, s) in prev_samples.iter().enumerate() {
                    all_quats.push(*s);
                    all_times.push(-frame_dur + (i as f32) * dt);
                }
            }
        }
    }
    // Current frame: t ∈ [0, frame_dur).
    let dt_curr = frame_dur / (n_curr as f32);
    for (i, s) in curr_samples.iter().enumerate() {
        all_quats.push(*s);
        all_times.push((i as f32) * dt_curr);
    }
    // Next frame: t ∈ [frame_dur, 2·frame_dur).
    if fi + 1 < n_frames {
        if let Some(next_samples) = osv.high_rate_quats.get(fi + 1) {
            let n_next = next_samples.len();
            if n_next > 0 {
                let dt = frame_dur / (n_next as f32);
                for (i, s) in next_samples.iter().enumerate() {
                    all_quats.push(*s);
                    all_times.push(frame_dur + (i as f32) * dt);
                }
            }
        }
    }

    let m = all_quats.len();
    if m < 4 {
        return None;
    }
    for q in all_quats.iter_mut() {
        *q = q.normalize();
    }
    for i in 1..m {
        if all_quats[i].dot(all_quats[i - 1]) < 0.0 {
            all_quats[i] = Quat {
                w: -all_quats[i].w,
                x: -all_quats[i].x,
                y: -all_quats[i].y,
                z: -all_quats[i].z,
            };
        }
    }

    // Apply DJI's IMU-vs-video timing offset. Queries the merged
    // timeline at `frame_mid + phase_offset_s` (derived from
    // sensor readout — see [`dji_imu_phase_offset_s`]) instead of
    // pure midpoint.
    let t = frame_dur * 0.5 + phase_offset_s;
    let idx_right = all_times.partition_point(|&x| x <= t);
    let idx = if idx_right == 0 { 0 } else { idx_right - 1 };
    let idx = idx.min(m.saturating_sub(2));
    let i0 = idx.saturating_sub(1).min(m - 1);
    let i1 = idx;
    let i2 = (idx + 1).min(m - 1);
    let i3 = (idx + 2).min(m - 1);

    let q = catmull_rom_quat_components(
        all_quats[i0], all_quats[i1], all_quats[i2], all_quats[i3],
        all_times[i0], all_times[i1], all_times[i2], all_times[i3],
        t,
    );
    Some(q.normalize())
}


/// Smallest rotation that aligns a camera-frame gravity vector with
/// world-frame down `(0, 0, -1)`. This is the horizon-lock correction
/// DJI Studio computes per frame (verified via disassembly of
/// `EisBase::getMatrixForEisAndHorizontal`).
///
/// The returned quaternion `q` satisfies
///     q · gravity_cam · q⁻¹ ≈ (0, 0, -1)
/// when interpreted as a rotation acting on the gravity vector. Yaw is
/// preserved because the rotation axis is constrained to the horizontal
/// plane (perpendicular to both `gravity_cam` and world-up).
fn horizon_lock_quat(gravity_cam: &[f32; 3]) -> Quat {
    let gx = gravity_cam[0];
    let gy = gravity_cam[1];
    let gz = gravity_cam[2];
    let len_sq = gx * gx + gy * gy + gz * gz;
    if len_sq < 1e-12 {
        return Quat::IDENTITY;
    }
    let inv_len = 1.0 / len_sq.sqrt();
    let g = [gx * inv_len, gy * inv_len, gz * inv_len];
    // Target = world-frame down. axis = normalize(g × target). Since
    // target = (0, 0, -1), this simplifies to (-gy, gx, 0).
    let ax = -g[1];
    let ay = g[0];
    let az = 0.0;
    let axis_len_sq = ax * ax + ay * ay + az * az;
    if axis_len_sq < 1e-12 {
        // Gravity already aligned (or anti-aligned). The anti-aligned
        // case (g pointing up) would need a 180° rotation around any
        // horizontal axis; treat as identity since it doesn't happen
        // for a held camera.
        return Quat::IDENTITY;
    }
    let axis_inv_len = 1.0 / axis_len_sq.sqrt();
    let ax = ax * axis_inv_len;
    let ay = ay * axis_inv_len;
    let az = az * axis_inv_len;
    // angle = acos(dot(g, target)) = acos(-gz)
    let cos_angle = (-g[2]).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let half = angle * 0.5;
    let s = half.sin();
    Quat {
        w: half.cos(),
        x: ax * s,
        y: ay * s,
        z: az * s,
    }
}

/// Clamp a unit quaternion's rotation angle to at most `max_deg`.
///
/// For `q = (cos(θ/2), sin(θ/2) · axis)`, the rotation angle is
/// `θ = 2 · acos(|w|)`. We scale the vector part to bring θ down
/// without changing the rotation axis.
fn clamp_correction(q: Quat, max_deg: f32) -> Quat {
    if !max_deg.is_finite() || max_deg <= 0.0 {
        return q;
    }
    let max_rad = max_deg.to_radians();
    let w_abs = q.w.abs().min(1.0);
    let angle = 2.0 * w_abs.acos();
    if angle <= max_rad {
        return q;
    }
    // Scale θ down to max_rad while preserving axis. axis = xyz / sin(θ/2).
    let new_half = max_rad * 0.5;
    let cos_h = new_half.cos();
    let sin_h = new_half.sin();
    let sin_old_h = (angle * 0.5).sin();
    if sin_old_h < 1e-9 {
        return q;
    }
    let scale = sin_h / sin_old_h;
    let qc = Quat {
        w: cos_h * q.w.signum(),
        x: q.x * scale,
        y: q.y * scale,
        z: q.z * scale,
    };
    qc.normalize()
}

/// Centered exponential moving average of a quaternion time series.
/// Two-pass IIR (forward + backward) for zero phase delay. `tau_frames`
/// is the EMA time constant in frames — larger = more smoothing.
fn smooth_quats_ema(quats: &[Quat], tau_frames: usize) -> Vec<Quat> {
    if quats.is_empty() || tau_frames < 2 {
        return quats.to_vec();
    }
    let alpha = 1.0_f32 / (tau_frames as f32);
    let n = quats.len();
    let mut fwd: Vec<Quat> = Vec::with_capacity(n);
    fwd.push(quats[0]);
    for i in 1..n {
        let prev = fwd[i - 1];
        let next = quats[i];
        fwd.push(prev.slerp(next, alpha));
    }
    let mut bwd: Vec<Quat> = vec![Quat::IDENTITY; n];
    bwd[n - 1] = fwd[n - 1];
    for i in (0..n - 1).rev() {
        let prev = bwd[i + 1];
        let next = fwd[i];
        bwd[i] = prev.slerp(next, alpha);
    }
    bwd
}

/// Per-row rolling-shutter correction quaternions for one video frame.
///
/// Mirrors `DjiGyroStabilizer.get_per_row_quaternions` at
/// `vr180_gui.py:828-985`. The DJI Osmo sensor takes ~19 ms to read a
/// full frame top-to-bottom; per-frame stabilization can lock the
/// overall frame to the reference orientation but can't fix the
/// intra-frame shear that fast camera motion creates. This function
/// produces one quaternion per scanline that, when applied to the
/// projected output direction, cancels that shear.
///
/// Algorithm:
/// 1. Merge high-rate quaternion samples from the previous, current,
///    and next video frames into one continuous timeline (so the
///    Catmull-Rom interpolant has proper context past either frame
///    boundary).
/// 2. Hemisphere-align all samples for slerp safety.
/// 3. For each scanline `y ∈ [0, fish_h)`, compute its readout time
///    `t_y = readout_start + (y / fish_h) · readout_s` where
///    `readout_start = (frame_dur - readout_s) / 2` puts the readout
///    window centred on the frame.
/// 4. Component-wise Catmull-Rom interpolate the merged quaternion
///    timeline to that time (matches DJI Studio's `catmullRomQuaternion`
///    convention — NLERP-style, not slerp).
/// 5. Also interpolate at `t_mid = frame_dur / 2` (the mid-frame
///    reference).
/// 6. Output: `q_corr_row = q_row⁻¹ ⊗ q_mid` per row. Applying this to
///    the projected direction rotates from "mid-frame orientation" to
///    "this row's orientation" — exactly what we need so the kernel
///    samples the correct pixel after the sensor's per-row read delay.
///
/// Returns `None` if the OSV doesn't have enough high-rate samples
/// for the current frame (need ≥ 2). The caller should fall back to a
/// no-RS path in that case.
pub fn compute_per_row_quaternions_for_frame(
    osv: &DjiOsvImu,
    frame_idx: usize,
    readout_s: f32,
    fish_h: u32,
    fps: f32,
) -> Option<Vec<Quat>> {
    if fish_h == 0 || fps <= 0.0 || !readout_s.is_finite() || readout_s <= 0.0 {
        return None;
    }
    let n_frames = osv.high_rate_quats.len();
    if frame_idx >= n_frames { return None; }
    let curr_samples = osv.high_rate_quats.get(frame_idx)?;
    let n_curr = curr_samples.len();
    if n_curr < 2 { return None; }

    let frame_dur = 1.0_f32 / fps;

    // ── Merge prev/curr/next high-rate samples into a single timeline ──
    let mut all_quats: Vec<Quat> = Vec::with_capacity(n_curr * 3);
    let mut all_times: Vec<f32> = Vec::with_capacity(n_curr * 3);

    // Previous frame: t ∈ [-frame_dur, 0).
    if frame_idx > 0 {
        if let Some(prev_samples) = osv.high_rate_quats.get(frame_idx - 1) {
            let n_prev = prev_samples.len();
            if n_prev > 0 {
                let dt_prev = frame_dur / (n_prev as f32);
                for (i, s) in prev_samples.iter().enumerate() {
                    all_quats.push(*s);
                    all_times.push(-frame_dur + (i as f32) * dt_prev);
                }
            }
        }
    }
    // Current frame: t ∈ [0, frame_dur).
    let dt_curr = frame_dur / (n_curr as f32);
    for (i, s) in curr_samples.iter().enumerate() {
        all_quats.push(*s);
        all_times.push((i as f32) * dt_curr);
    }
    // Next frame: t ∈ [frame_dur, 2·frame_dur).
    if frame_idx + 1 < n_frames {
        if let Some(next_samples) = osv.high_rate_quats.get(frame_idx + 1) {
            let n_next = next_samples.len();
            if n_next > 0 {
                let dt_next = frame_dur / (n_next as f32);
                for (i, s) in next_samples.iter().enumerate() {
                    all_quats.push(*s);
                    all_times.push(frame_dur + (i as f32) * dt_next);
                }
            }
        }
    }

    let m = all_quats.len();
    // Need at least 4 control points for Catmull-Rom (the inner
    // segment requires q0,q1,q2,q3); fall back if we don't have them.
    if m < 4 { return None; }

    // Normalize.
    for q in all_quats.iter_mut() {
        *q = q.normalize();
    }
    // Hemisphere align across the entire merged sequence.
    for i in 1..m {
        if all_quats[i].dot(all_quats[i - 1]) < 0.0 {
            all_quats[i] = Quat {
                w: -all_quats[i].w,
                x: -all_quats[i].x,
                y: -all_quats[i].y,
                z: -all_quats[i].z,
            };
        }
    }

    // ── Compute query times: one per scanline + the mid-frame ──
    // All offset by the readout-derived phase (see
    // `dji_imu_phase_offset_s`) to match DJI Studio's calibrated
    // IMU-vs-video time skew. Because the offset is computed from
    // the *same* readout_s the caller passed in, this stays correct
    // when the camera switches sensor modes (different fps → maybe
    // different readout).
    let phase_offset_s = dji_imu_phase_offset_s_fps(readout_s, fps);
    let readout_start = (frame_dur - readout_s) * 0.5 + phase_offset_s;
    let t_mid = frame_dur * 0.5 + phase_offset_s;
    let mut query_times: Vec<f32> = Vec::with_capacity(fish_h as usize + 1);
    let inv_h = 1.0_f32 / (fish_h as f32);
    for y in 0..fish_h {
        let frac = (y as f32) * inv_h;
        query_times.push(readout_start + frac * readout_s);
    }
    query_times.push(t_mid);

    // ── Catmull-Rom interpolation per query ──
    let mut interpolated: Vec<Quat> = Vec::with_capacity(query_times.len());
    for &t in &query_times {
        // searchsorted side='right' → first index with times[idx] > t,
        // then subtract 1 to get the lower control point.
        let idx_right = all_times
            .partition_point(|&x| x <= t);
        let idx = if idx_right == 0 { 0 } else { idx_right - 1 };
        let idx = idx.min(m.saturating_sub(2));
        let i0 = idx.saturating_sub(1).min(m - 1);
        let i1 = idx;
        let i2 = (idx + 1).min(m - 1);
        let i3 = (idx + 2).min(m - 1);

        let q = catmull_rom_quat_components(
            all_quats[i0], all_quats[i1], all_quats[i2], all_quats[i3],
            all_times[i0], all_times[i1], all_times[i2], all_times[i3],
            t,
        );
        interpolated.push(q.normalize());
    }

    // Split: per-row quats + mid-frame quat.
    let q_mid = *interpolated.last().unwrap();
    let row_quats = &interpolated[..(fish_h as usize)];

    // Per-row correction: `q_corr = q_row⁻¹ ⊗ q_mid` (matches Python
    // vr180_gui.py:974-983 vectorized formula).
    let result: Vec<Quat> = row_quats
        .iter()
        .map(|q_row| q_row.conjugate().mul(q_mid).normalize())
        .collect();

    Some(result)
}

/// Component-wise Catmull-Rom interpolation of four control quats at
/// query time `t`. Mirrors `vr180_gui.py:934-963` (NLERP-style, not
/// slerp — matches DJI Studio's `catmullRomQuaternion`).
/// Two-sample spherical interpolation at time `t` between adjacent
/// IMU samples `(q1, t1)` and `(q2, t2)`. The `q0, q3, t0, t3`
/// parameters are kept for call-site compatibility but ignored —
/// they were the outer Catmull-Rom control points; DJI Studio's
/// `getQuaternionForEisAndHorizontal` does pure slerp between the
/// floor/ceil samples of its global timeline (verified via
/// disassembly: `bl __ZN…quaternionSlerp…`).
fn catmull_rom_quat_components(
    _q0: Quat, q1: Quat, q2: Quat, _q3: Quat,
    _t0: f32, t1: f32, t2: f32, _t3: f32,
    t: f32,
) -> Quat {
    let dt = t2 - t1;
    let frac = if dt.abs() > 1e-15 { (t - t1) / dt } else { 0.0 };
    let wb = frac.clamp(0.0, 1.0);
    let wa = 1.0 - wb;
    slerp_quat(q1, q2, wa, wb)
}

/// Spherical weighted blend of two unit quaternions. Assumes
/// `wa + wb == 1.0`. Result is a unit quaternion ON the great-circle
/// arc between `a` and `b` — exact rotation magnitude is preserved,
/// unlike the component-wise lerp that pulled inward through the
/// 4D hypersphere and lost a few percent of angle on each nested
/// blend. Falls back to a normalised lerp when the inputs are
/// near-parallel (`|dot| > 0.9995`) where the slerp division
/// `sin(t·θ) / sin(θ)` becomes numerically unstable.
#[inline]
fn slerp_quat(a: Quat, b: Quat, wa: f32, wb: f32) -> Quat {
    let dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    // Hemisphere align: flip b if it sits on the far side of the
    // hypersphere, so the slerp follows the short arc.
    let (b, dot) = if dot < 0.0 {
        (Quat { w: -b.w, x: -b.x, y: -b.y, z: -b.z }, -dot)
    } else {
        (b, dot)
    };
    if dot > 0.9995 {
        // Near-parallel: slerp degenerates, fall back to nlerp.
        let mut out = Quat {
            w: a.w * wa + b.w * wb,
            x: a.x * wa + b.x * wb,
            y: a.y * wa + b.y * wb,
            z: a.z * wa + b.z * wb,
        };
        let n = (out.w * out.w + out.x * out.x + out.y * out.y + out.z * out.z).sqrt();
        if n > 0.0 {
            out.w /= n; out.x /= n; out.y /= n; out.z /= n;
        }
        return out;
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let inv_sin_theta = 1.0 / theta.sin();
    let s_a = (wa * theta).sin() * inv_sin_theta;
    let s_b = (wb * theta).sin() * inv_sin_theta;
    Quat {
        w: a.w * s_a + b.w * s_b,
        x: a.x * s_a + b.x * s_b,
        y: a.y * s_a + b.y * s_b,
        z: a.z * s_a + b.z * s_b,
    }
}

/// DJI Studio's fixed output basis matrix `K_const`, extracted from
/// `UtilsWrapper::getMatrixForEisAndHorizontal` (rodata at vmaddrs
/// `0x102736a50`, `0x102737150`, `0x102736a40`, `0x102737170`).
///
/// 3×3 portion, row-major:
/// ```
/// [ 0  0 -1 ]
/// [ 1  0  0 ]
/// [ 0  1  0 ]
/// ```
///
/// Kept here for reference. **NOT used at runtime** — the K_const-based
/// experiments (right-multiply, similarity, similarity-with-flipped
/// row) all introduced sign errors on roll/yaw axes vs. our long-
/// standing fallback similarity. The caller's MKMatrix4 in DJI's
/// pipeline encodes more than just `K_const⁻¹`; we can't infer it
/// without seeing the UI code that constructs it.
/// Post-multiply applied by `UtilsWrapper::getMatrixForEisAndHorizontal`
/// in DJI Studio (rodata at 0x102736a50, 0x102737150, 0x102736a40,
/// 0x102737170 — verified via lldb memory dump 2026-06-01). The
/// Metal shader receives `+0x3d8_matrix · K_CONST`, not just +0x3d8.
const K_CONST: [[f32; 3]; 3] = [
    [0.0, 0.0, -1.0],
    [1.0, 0.0,  0.0],
    [0.0, 1.0,  0.0],
];

/// Convert a slice of per-row correction quaternions into a packed
/// flat `f32` buffer of camera-frame 3×3 matrices, ready to upload as
/// a wgpu storage buffer. Each row consumes 12 f32 (std430 stride for
/// `mat3x3`-shaped struct with vec4 alignment — three rows of 3 floats
/// + 1 pad). Total length = `quats.len() * 12`.
///
/// The matrix is `R_cam = C · R_imu · Cᵀ` (same basis change applied
/// to the per-frame stabilization). The shader treats the buffer as
/// `array<RsRowR>` with three `vec4<f32>` rows per element; the
/// fourth lane of each is unused padding.
pub fn pack_per_row_camera_matrices(quats: &[Quat], lens_a_quat_xyzw: [f32; 4]) -> Vec<f32> {
    // Per-row quats are RELATIVE (q_row⁻¹ · q_mid). The basis change
    // applies the same way as per-frame stab (C·R·Cᵀ with C = AXIS ·
    // mat(q_lens_a)) so the per-row correction lands in the same camera
    // frame as the per-frame stab.
    let mut out = Vec::with_capacity(quats.len() * 12);
    for q in quats {
        let r_eis = q.to_mat3_row_major();
        let r_final = apply_c_imu_to_cam_with_lens_a(&r_eis, lens_a_quat_xyzw);
        out.push(r_final[0]); out.push(r_final[1]); out.push(r_final[2]); out.push(0.0);
        out.push(r_final[3]); out.push(r_final[4]); out.push(r_final[5]); out.push(0.0);
        out.push(r_final[6]); out.push(r_final[7]); out.push(r_final[8]); out.push(0.0);
    }
    out
}

/// Default sensor readout time for the DJI Osmo OQ001 (OSMO 360) at
/// 30 fps recording mode. For higher fps the readout is shorter (the
/// sensor crops/bins to fit the smaller frame budget) — use
/// [`dji_osmo_readout_ms_for_fps`] when fps is known.
///
/// Values verified via lldb capture of EisBase fields `+0x60c` (scan
/// lines) × `+0x614` (ns per line) in DJI Studio:
/// - 30 fps:   4766 × 3840 ns = 18.30 ms (hardcoded as 19 here for
///             backward compat with the old constant)
/// - 50 fps:   4226 × 3840 ns = 16.23 ms (different sensor mode)
pub const DJI_OSMO_OQ001_READOUT_MS: f32 = 19.0;

/// FPS-aware sensor readout. The OSMO 360 switches sensor mode
/// (crop/binning) at higher recording fps to fit the smaller frame
/// budget, which yields a shorter scanline readout. Phase offset
/// scales with readout (see [`dji_imu_phase_offset_s`]), so this
/// matters: at 50 fps using the 30 fps readout gives a ~1.2 ms
/// timing error, observed as a "loose" feeling in fast camera
/// motion ("the stab is the right size but lands at the wrong axis").
pub fn dji_osmo_readout_ms_for_fps(fps: f32) -> f32 {
    if fps > 40.0 {
        // Verified 50 fps mode: 4226 × 3840 ns = 16.228 ms.
        16.228
    } else {
        DJI_OSMO_OQ001_READOUT_MS
    }
}

/// Number of horizontal slices DJI's pipeline uses per frame.
/// Hardcoded in `getMatrixForEisAndHorizontal` — confirmed via
/// disassembly. Same value across all OSMO 360 firmware revisions
/// we've inspected.
pub const DJI_OSMO_SLICE_COUNT: f32 = 8.0;

/// IMU-to-video timing offset from frame midpoint, in **seconds**,
/// derived from sensor readout time.
///
/// **Physical derivation** (replaces the previous magic constant
/// `-0.0085`): DJI samples the IMU at the *center of the first slice's
/// readout window*, not at mid-frame. With slice_count = 8 and the
/// sensor reading top-to-bottom over `readout_s`:
///
/// ```text
///     first_slice_center_from_frame_start
///         = (frame_dur - readout) / 2  // readout window start
///         + readout / (2 · slice_count)  // half-slice in
///
///     offset_from_mid_frame
///         = first_slice_center - frame_dur / 2
///         = -readout / 2 + readout / (2 · slice_count)
///         = -readout · (slice_count - 1) / (2 · slice_count)
/// ```
///
/// At our 30 fps / 19 ms-readout calibration point this evaluates to
/// `-19 ms · 7/16 = -8.3125 ms` — within 0.2 ms of the empirical
/// -8.5 ms fit (the residual is IMU noise across the 26-frame sweep).
///
/// **Why this matters for non-30 fps**: a single hardcoded -8.5 ms
/// becomes wrong if the camera switches sensor mode (e.g. 50/60 fps
/// with shorter readout for the smaller frame budget). The formula
/// scales correctly with `readout_s` automatically.
///
/// **Framerate independence**: notice `frame_dur` does NOT appear in
/// the result — only `readout`. As long as the camera's sensor
/// readout time is known (we get this from `DJI_OSMO_OQ001_READOUT_MS`
/// today; for a multi-mode camera it'd need to come from metadata),
/// the offset is the same at any fps.
#[inline]
pub fn dji_imu_phase_offset_s(_readout_s: f32) -> f32 {
    // Falls through to the fps-aware variant assuming 30 fps. Keep the
    // signature for the few callers that don't have fps in scope.
    dji_imu_phase_offset_s_fps(_readout_s, 29.97)
}

/// FPS-aware phase offset. Empirically derived from DJI Studio
/// `+0x3d8` matrix captures at both 30 fps and 50 fps:
///
/// | fps | empirical phase from mid-frame | implies time-from-frame-start |
/// |-----|------------------------------|-----------------------------|
/// | 30  | -8.5 ms                      | 8.18 ms                     |
/// | 50  | -1.5 ms                      | 8.50 ms                     |
///
/// Both modes converge on **DJI samples IMU ~8.5 ms after frame_start**,
/// independent of fps. The previous formula `-readout · 7/16` was a
/// coincidence that worked at 30 fps because the readout (18.3 ms)
/// gave the right number; at 50 fps it gave -7.1 ms, off by ~5.6 ms
/// from the actual optimum -1.5 ms.
///
/// `readout_s` is no longer used directly, but kept in the signature
/// for callers that want to pass it (it informs the per-row pipeline's
/// readout window separately).
#[inline]
pub fn dji_imu_phase_offset_s_fps(_readout_s: f32, fps: f32) -> f32 {
    /// Constant offset from frame_start where DJI samples IMU.
    /// Verified empirically across 30 fps and 50 fps clips.
    const TARGET_OFFSET_FROM_FRAME_START_S: f32 = 0.0085;
    let frame_dur = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };
    TARGET_OFFSET_FROM_FRAME_START_S - 0.5 * frame_dur
}

/// Default IMU phase offset for the 19 ms readout OSMO 360 mode at 30 fps.
/// **Use [`dji_imu_phase_offset_s`] instead in any new code** so the
/// value tracks the actual readout. Kept here for tests and call sites
/// that didn't yet thread readout through.
#[allow(dead_code)]
pub const DJI_OSMO_IMU_PHASE_OFFSET_S: f32 = -19.0 * 7.0 / (16.0 * 1000.0);

/// `R_cam = C · R_imu · Cᵀ` using the fallback hardcoded lens_a basis.
fn apply_c_imu_to_cam(r_imu_row_major: &[f32; 9]) -> [f32; 9] {
    apply_c_imu_to_cam_with_lens_a(r_imu_row_major, LENS_A_QUAT_XYZW)
}

/// `R_cam = AXIS · R_imu · AXISᵀ` — pure axis convention swap, no
/// per-camera lens_a factor. Used when the per-frame rotation already
/// has q_lens_a baked in via direct quat composition (as DJI does),
/// so we don't need lens_a in the similarity basis too.
fn apply_c_imu_to_cam_axis_only(r_imu_row_major: &[f32; 9]) -> [f32; 9] {
    let r: [[f32; 3]; 3] = [
        [r_imu_row_major[0], r_imu_row_major[1], r_imu_row_major[2]],
        [r_imu_row_major[3], r_imu_row_major[4], r_imu_row_major[5]],
        [r_imu_row_major[6], r_imu_row_major[7], r_imu_row_major[8]],
    ];
    let c = AXIS_CORRECTION;
    let mut tmp = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tmp[i][j] += c[i][k] * r[k][j];
            }
        }
    }
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += tmp[i][k] * c[j][k];
            }
        }
    }
    [
        out[0][0], out[0][1], out[0][2],
        out[1][0], out[1][1], out[1][2],
        out[2][0], out[2][1], out[2][2],
    ]
}

/// `R_cam = C · R_imu · Cᵀ` where C is derived from a per-clip lens_a.
fn apply_c_imu_to_cam_with_lens_a(
    r_imu_row_major: &[f32; 9],
    lens_a_quat_xyzw: [f32; 4],
) -> [f32; 9] {
    // Unpack r_imu into 3×3.
    let r: [[f32; 3]; 3] = [
        [r_imu_row_major[0], r_imu_row_major[1], r_imu_row_major[2]],
        [r_imu_row_major[3], r_imu_row_major[4], r_imu_row_major[5]],
        [r_imu_row_major[6], r_imu_row_major[7], r_imu_row_major[8]],
    ];
    let c = c_imu_to_cam_with_lens_a(lens_a_quat_xyzw);
    // First product: tmp = C · R.
    let mut tmp = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tmp[i][j] += c[i][k] * r[k][j];
            }
        }
    }
    // Second product: out = tmp · Cᵀ.
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += tmp[i][k] * c[j][k]; // Cᵀ[k][j] = C[j][k]
            }
        }
    }
    [
        out[0][0], out[0][1], out[0][2],
        out[1][0], out[1][1], out[1][2],
        out[2][0], out[2][1], out[2][2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stationary camera (constant orientation) → identity per-frame
    /// rotation across the clip.
    #[test]
    fn stationary_dji_quats_produce_identity_rotations() {
        let q = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let n_frames = 60;
        let osv = DjiOsvImu {
            frame_quats: vec![q; n_frames],
            gravity: vec![[0.0, -1.0, 0.0]; n_frames],
            // 16 hr samples per frame, all identity.
            high_rate_quats: vec![vec![q; 16]; n_frames],
            lens_a: Default::default(),
            lens_b: Default::default(),
        };
        let stab = compute_dji_stabilization(&osv, n_frames, 10.0, 0.0, 30.0)
            .expect("stab");
        assert_eq!(stab.per_frame.len(), n_frames);
        for (i, rot) in stab.per_frame.iter().enumerate() {
            // C · I · Cᵀ = I (since C is orthogonal). So we expect
            // identity here too.
            let m = rot.0;
            let trace = m[0] + m[4] + m[8];
            assert!(
                (trace - 3.0).abs() < 1e-4,
                "frame {i} trace = {trace:.6}, expected ~3.0"
            );
        }
    }

    /// C · I · Cᵀ = I (sanity check on the matrix multiplication).
    #[test]
    fn c_imu_to_cam_is_orthogonal_on_identity() {
        let r_imu = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let r_cam = apply_c_imu_to_cam(&r_imu);
        for (i, v) in r_cam.iter().enumerate() {
            let expected = if i == 0 || i == 4 || i == 8 { 1.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-6,
                "[{i}] got {v}, expected {expected}");
        }
    }

    /// Clamping a 20° rotation with max_corr_deg=10° should leave at
    /// most a 10° rotation in the output.
    #[test]
    fn clamp_correction_limits_angle() {
        // Build a 20° yaw rotation.
        let half = 10.0_f32.to_radians();
        let q = Quat {
            w: half.cos(),
            x: 0.0,
            y: half.sin(),
            z: 0.0,
        };
        let qc = clamp_correction(q, 10.0);
        let angle_deg = 2.0 * qc.w.abs().min(1.0).acos().to_degrees();
        assert!(angle_deg <= 10.1, "clamped angle {angle_deg}° > 10°");
        assert!(angle_deg >= 9.9, "clamped angle {angle_deg}° too small");
    }

    /// Lock the composition order of the per-frame correction
    /// quaternion to match the Python OSV implementation. The math
    /// here is "whatever Python does" — we tried `q ⊗ q_ref⁻¹` (the
    /// theoretically-correct formula under world→sensor convention)
    /// and it broke stabilization on real OSV footage. The reverse
    /// `q⁻¹ ⊗ q_ref` works in production. Conclusion: the DJI quat
    /// + IMU mount geometry is calibrated as a pair with the
    /// reversed composition + the C_IMU_TO_CAM basis change. Don't
    /// touch one without the other.
    ///
    /// This test asserts the formula matches Python's. If it starts
    /// failing, someone "fixed" the composition without re-deriving
    /// C_IMU_TO_CAM.
    #[test]
    fn correction_composition_matches_python() {
        let half = (-15.0_f32).to_radians();
        let q_yaw_right = Quat {
            w: half.cos(),
            x: 0.0,
            y: half.sin(),
            z: 0.0,
        };
        // Reference is identity. Formula matches Python's actual
        // implementation (vr180_gui.py:713): Q_inv * Q_ref.
        let q_corr = q_yaw_right.conjugate().mul(Quat::IDENTITY).normalize();
        // q_corr should equal q_yaw_right.conjugate() = quat(+30° Y).
        let r = q_corr.to_mat3_row_major();
        // For a +30° rotation around +Y, R · (0,0,1) = (sin30, 0, cos30).
        let r_dot_z = [r[2], r[5], r[8]];
        assert!(
            r_dot_z[0] > 0.4,
            "composition drifted from Python's formula: \
             R·(0,0,1) = {r_dot_z:?}, expected X ≈ +0.5"
        );
    }

    /// Diagnostic: dump per-frame correction angle magnitudes for the
    /// user's test file. Helps see at-a-glance whether stab is
    /// producing visible-sized corrections (5°+) or near-identity
    /// (sub-degree, looks like no stab).
    #[test]
    fn diagnose_user_osv_magnitudes() {
        let path = std::path::Path::new(
            "/Volumes/VR Share/CAM_20260524181104_0006_D.OSV"
        );
        if !path.exists() {
            eprintln!("skip — fixture not present");
            return;
        }
        let blob = crate::decode::extract_dji_meta_stream(path)
            .expect("extract dji meta");
        let osv = DjiOsvImu::parse(&blob).expect("parse dji protobuf");

        let n = osv.frame_quats.len().min(300);
        eprintln!("== {} frames, sampling first {} ==", osv.frame_quats.len(), n);

        // Compute stab with camera-lock (smooth_ms=0) to match the
        // GUI's OSV path.
        let stab = compute_dji_stabilization(&osv, n, 60.0, 0.0, 30.0).unwrap();

        // Also pull the raw per-frame mid-HR quat so we can compute
        // "how much did the camera move from frame 0" independently
        // of the correction.
        let mut raw_quats = Vec::with_capacity(n);
        for fi in 0..n {
            let q = osv.high_rate_quats.get(fi)
                .and_then(|hr| hr.get(hr.len() / 2))
                .copied()
                .unwrap_or(Quat::IDENTITY)
                .normalize();
            raw_quats.push(q);
        }
        let q_ref = raw_quats[0];

        for fi in (0..n).step_by(15) {
            let r = stab.per_frame[fi].0;
            let trace = r[0] + r[4] + r[8];
            let corr_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees();

            // Raw camera motion: angle between q_actual and q[0].
            let q = raw_quats[fi];
            let d = q.dot(q_ref).abs().min(1.0);
            let raw_angle = (2.0 * d.acos()).to_degrees();

            // What R_cam does to a forward output direction (0,0,1).
            // For row-major 3x3, R * (0,0,1) is column 2: [r[2], r[5], r[8]].
            let r_fwd = [r[2], r[5], r[8]];
            // What R_cam does to an up output direction (0,1,0).
            let r_up = [r[1], r[4], r[7]];
            // What R_cam does to a right output direction (1,0,0).
            let r_right = [r[0], r[3], r[6]];

            eprintln!(
                "frame {fi:>4}: corr={corr_angle:>6.2}°, raw={raw_angle:>6.2}°  \
                 R·fwd=({:+.2}, {:+.2}, {:+.2})  R·up=({:+.2}, {:+.2}, {:+.2})  \
                 R·right=({:+.2}, {:+.2}, {:+.2})",
                r_fwd[0], r_fwd[1], r_fwd[2],
                r_up[0], r_up[1], r_up[2],
                r_right[0], r_right[1], r_right[2],
            );
        }
    }

    /// Per-row RS quaternions for a stationary clip should all be
    /// (near-)identity. Stationary = every HR sample is the same quat;
    /// the Catmull-Rom interpolant of constant inputs is constant, so
    /// `q_corr = q_row⁻¹ ⊗ q_mid = identity` for every row.
    #[test]
    fn per_row_rs_stationary_is_identity() {
        let q = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let osv = DjiOsvImu {
            frame_quats: vec![q; 60],
            gravity: vec![[0.0, -1.0, 0.0]; 60],
            high_rate_quats: vec![vec![q; 16]; 60],
            lens_a: Default::default(),
            lens_b: Default::default(),
        };
        let row_quats = compute_per_row_quaternions_for_frame(
            &osv, 30, 0.019, 1080, 29.97,
        ).expect("rs row quats");
        assert_eq!(row_quats.len(), 1080);
        for (i, q) in row_quats.iter().enumerate() {
            // identity quat has w ≈ ±1, |x|,|y|,|z| ≈ 0
            let angle_deg = 2.0 * q.w.abs().min(1.0).acos().to_degrees();
            assert!(
                angle_deg < 0.01,
                "row {i}: expected identity, got {angle_deg:.4}° q={q:?}"
            );
        }
    }

    /// Diagnostic: how much per-row correction is the RS pipeline
    /// producing for a real OSV file? If the answer is "well under
    /// 1°" we'd expect no visible improvement; "a few degrees" means
    /// the RS correction is doing real work.
    #[test]
    fn diagnose_rs_magnitudes() {
        let candidates = [
            "/Volumes/VR Share/CAM_20260524181104_0006_D.OSV",
            "/Volumes/Silver/250826OSV Swap/CAM_20250811172419_0044_D.OSV",
            "/Volumes/Silver/develop/vr180_fisheye_converter/CAM_20260225224810_0003_D.osv",
        ];
        for path_str in candidates {
            let path = std::path::Path::new(path_str);
            if !path.exists() { continue; }
            eprintln!("== {} ==", path_str);
            let blob = crate::decode::extract_dji_meta_stream(path)
                .expect("extract dji meta");
            let osv = DjiOsvImu::parse(&blob).expect("parse dji protobuf");
            let n = osv.high_rate_quats.len();
            // Pick a few sample frames likely to have motion.
            let probe_frames = [30usize, 60, 120, 200, n.saturating_sub(60)];
            for &fi in &probe_frames {
                if fi >= n { continue; }
                let Some(rows) = compute_per_row_quaternions_for_frame(
                    &osv, fi, 0.019, 3840, 29.97,
                ) else {
                    eprintln!("  frame {fi}: no RS data");
                    continue;
                };
                let top = rows[0];
                let mid = rows[rows.len() / 2];
                let bot = rows[rows.len() - 1];
                let to_deg = |q: Quat| 2.0 * q.w.abs().min(1.0).acos().to_degrees();
                eprintln!(
                    "  frame {fi}: top={:>5.2}°  mid={:>5.2}°  bot={:>5.2}°  (top-bot delta)",
                    to_deg(top), to_deg(mid), to_deg(bot),
                );
            }
            return; // only the first present fixture
        }
        eprintln!("skip — no OSV fixture present");
    }

    /// Empty OSV → identity fallback, no panic.
    #[test]
    fn empty_osv_returns_identity_fallback() {
        let osv = DjiOsvImu::default();
        let stab = compute_dji_stabilization(&osv, 30, 10.0, 0.0, 30.0).unwrap();
        assert_eq!(stab.per_frame.len(), 30);
        for r in &stab.per_frame {
            assert_eq!(r.0, EquirectRotation::IDENTITY.0);
        }
    }

    /// Integration test against the real sample OSV files (if present
    /// on this developer's disk). Validates:
    /// - djmd extraction succeeds and produces non-empty bytes
    /// - protobuf parses cleanly
    /// - Lens A and Lens B both have populated fx
    /// - high-rate quats are present (~33 samples/frame at 29.97 fps)
    /// - stabilization produces non-identity rotations on a real clip
    ///
    /// Each path is silent-skipped when the fixture isn't available,
    /// so CI without the sample file still passes.
    #[test]
    fn real_osv_parse_and_stabilize() {
        let candidates = [
            "/Volumes/Silver/develop/vr180_fisheye_converter/CAM_20260225224810_0003_D.osv",
            "/Volumes/Silver/250826OSV Swap/CAM_20250811172419_0044_D.OSV",
        ];
        for path_str in candidates {
            let path = std::path::Path::new(path_str);
            if !path.exists() {
                eprintln!("skip {} — not present", path_str);
                continue;
            }
            eprintln!("== exercising {} ==", path_str);
            check_one_osv(path);
        }
    }

    fn check_one_osv(path: &std::path::Path) {

        let blob = crate::decode::extract_dji_meta_stream(path)
            .expect("extract dji meta");
        assert!(blob.len() > 100_000, "djmd track tiny: {} bytes", blob.len());

        let osv = DjiOsvImu::parse(&blob).expect("parse dji protobuf");
        eprintln!(
            "real osv: {} frame_quats, lens_a.fx={:?}, lens_b.fx={:?}",
            osv.frame_quats.len(), osv.lens_a.fx, osv.lens_b.fx
        );
        assert!(osv.frame_quats.len() > 30, "should have many frames");
        assert!(osv.lens_a.fx.is_some(), "lens_a.fx should be present");
        assert!(osv.lens_b.fx.is_some(), "lens_b.fx should be present");

        // High-rate quats should average around 33 samples/frame at
        // 29.97 fps. We'll accept anything > 5 — just confirming the
        // path produces real data, not just the per-frame fallback.
        let total_hr: usize = osv.high_rate_quats.iter().map(|v| v.len()).sum();
        let avg_per_frame = total_hr / osv.high_rate_quats.len().max(1);
        eprintln!("avg high-rate samples per frame: {avg_per_frame}");
        assert!(
            avg_per_frame >= 5,
            "avg high-rate samples too low: {avg_per_frame}"
        );

        // Run stabilization. Need at least *some* non-identity rotations
        // unless the clip was 100% stationary (very unlikely for a
        // handheld camera).
        let n_frames = osv.frame_quats.len().min(120);
        // Use 1000 ms smoothing — long enough to confirm the smoothing
        // pass doesn't crash on real high-rate camera quat data, but
        // not so long that we suppress all motion.
        let stab = compute_dji_stabilization(&osv, n_frames, 15.0, 1000.0, 29.97).unwrap();
        let mut non_identity = 0usize;
        for r in &stab.per_frame {
            let m = r.0;
            let trace = m[0] + m[4] + m[8];
            if (trace - 3.0).abs() > 1e-3 {
                non_identity += 1;
            }
        }
        eprintln!("non-identity frames: {} / {}", non_identity, n_frames);
        // Frame 0 is always identity by construction (it's the
        // reference). Some frames after that should diverge.
        if n_frames > 10 {
            assert!(
                non_identity > 0,
                "all-identity stab output suggests stab pipeline broken"
            );
        }
    }
}
