//! Per-frame rotation matrix computation for stabilization.
//!
//! Phase A applied raw CORI directly ("camera lock"). This module
//! adds the pieces that turn camera lock into useful real-world
//! stabilization:
//!
//! - **Bidirectional velocity-dampened SLERP smoothing** (Phase B
//!   core). Forward + backward exponential SLERP passes, averaged.
//!   `tau` adapts from `smooth_ms` (calm) toward `fast_ms` (fast
//!   motion >`max_vel_deg_s`) so slow pans survive but jitter dies.
//!   `q_heading = q_raw · q_smooth⁻¹` is the per-frame correction.
//! - **Gravity alignment** (Phase C). First N GRAV samples → unit
//!   vector → quaternion rotating it to `(0, 1, 0)` (the GoPro
//!   convention's gravity-down axis). Right-multiply every CORI
//!   quat by `g⁻¹` so the world frame has Y-down = true gravity.
//! - **Soft elastic max-corr clamp** (Phase C). When the heading
//!   correction angle exceeds `max_corr_deg`, the smoothed quat is
//!   logarithmically softened back toward raw. Stops the
//!   stabilization from cropping past the image boundary on extreme
//!   camera moves (which would otherwise produce black borders).
//! - **Per-eye IORI split** (Phase B). GoPro firmware-stabilized
//!   files store the in-camera rotation in IORI; left pixels get
//!   `+IORI`, right pixels get `-IORI`. We compensate by applying
//!   per-eye `q_left = q_iori · q_heading`, `q_right = q_iori⁻¹ ·
//!   q_heading`. When IORI = identity (our test files), both eyes
//!   end up with the same matrix and this is a no-op.
//! - **Horizon lock** (Phase C+). Swing-twist decomposition of the
//!   smoothed quat around the camera forward axis (+Z) drops the
//!   Z-twist before computing the heading. Net result: the roll
//!   component of the rendered output is always taken from raw,
//!   so the horizon stays level even during slow camera pans
//!   that the smoother would otherwise carry along.
//!
//! Math ports `smooth()` in `vr180_gui.py:4434-4721` (the heart of
//! `GyroStabilizer`). Coordinate-frame conventions follow the Python
//! reference verbatim — see Phase A's `compute_stabilization_rotations`
//! commit message for the file-byte-order quirk on CORI.

use super::cori_iori::Quat;

/// User-tunable smoothing parameters. Defaults match the Python
/// `ProcessingConfig` defaults.
#[derive(Debug, Clone, Copy)]
pub struct SmoothParams {
    /// Time constant for calm periods, in ms. Default 1000 ms.
    /// Higher = more stable but laggier on intentional pans.
    /// `0` = camera lock (no smoothing — every raw frame is the smoothed value).
    pub smooth_ms: f32,
    /// Time constant for fast motion (>`max_vel_deg_s`), in ms.
    /// Default 50 ms — short enough that fast pans/whips don't get
    /// over-stabilized into a delayed blur.
    pub fast_ms: f32,
    /// Power curve exponent for the velocity → smoothing mapping.
    /// `1.0` = linear blend; `<1` = anticipatory (follows fast
    /// motion early); `>1` = laggy (holds longer, then catches up).
    pub responsiveness: f32,
    /// Velocity threshold (deg/sec) where `tau` reaches `fast_ms`.
    /// Default 200°/s — empirical "this is fast motion" cutoff.
    pub max_vel_deg_s: f32,
}

impl Default for SmoothParams {
    fn default() -> Self {
        Self {
            smooth_ms: 1000.0,
            fast_ms: 50.0,
            responsiveness: 1.0,
            max_vel_deg_s: 200.0,
        }
    }
}

/// Bidirectional velocity-dampened SLERP smoother.
/// Runs a forward exponential SLERP pass and a backward pass; the
/// final smoothed quaternion at each frame is the midpoint SLERP of
/// the two. The bidirectional structure makes the smoothing
/// non-causal (preserves shape on both sides of fast motion), and the
/// per-step `tau` adapts to the local angular velocity so calm spans
/// get heavy smoothing and fast moves get light smoothing.
pub fn bidirectional_smooth(raw: &[Quat], fps: f32, params: &SmoothParams) -> Vec<Quat> {
    let n = raw.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return raw.to_vec(); }
    // `smooth_ms = 0` means camera-lock mode (no smoothing at all).
    if params.smooth_ms <= 0.0 {
        return raw.to_vec();
    }
    let dt = 1.0 / fps.max(1e-6);

    // Closure: per-step alpha from local angular velocity.
    let alpha_for_velocity = |vel_deg_s: f32| -> f32 {
        let vel_norm = (vel_deg_s / params.max_vel_deg_s).clamp(0.0, 1.0);
        let vel_ratio = vel_norm.powf(params.responsiveness);
        let tau_ms = params.smooth_ms * (1.0 - vel_ratio)
            + params.fast_ms * vel_ratio;
        let tau_s = (tau_ms / 1000.0).max(1e-6);
        // First-order exponential lowpass: alpha = dt / (tau + dt).
        // High `tau` → small alpha → heavy smoothing.
        dt / (tau_s + dt)
    };

    // Forward pass.
    let mut fwd = Vec::with_capacity(n);
    fwd.push(raw[0]);
    for i in 1..n {
        let vel = quat_angular_velocity_deg_s(raw[i - 1], raw[i], dt);
        let alpha = alpha_for_velocity(vel);
        fwd.push(fwd[i - 1].slerp(raw[i], alpha));
    }

    // Backward pass (mirror image).
    let mut bwd = vec![Quat::IDENTITY; n];
    bwd[n - 1] = raw[n - 1];
    for i in (0..n - 1).rev() {
        let vel = quat_angular_velocity_deg_s(raw[i + 1], raw[i], dt);
        let alpha = alpha_for_velocity(vel);
        bwd[i] = bwd[i + 1].slerp(raw[i], alpha);
    }

    // Average forward + backward by SLERP(fwd, bwd, 0.5).
    (0..n).map(|i| fwd[i].slerp(bwd[i], 0.5)).collect()
}

/// Angular velocity between two unit quats, in degrees/second.
/// Uses the relative rotation `b · a⁻¹`; its rotation angle is
/// `2·acos(|w|)`. The `abs(w)` handles the q/-q double-cover
/// ambiguity (short-arc).
fn quat_angular_velocity_deg_s(a: Quat, b: Quat, dt: f32) -> f32 {
    let rel = b.mul(a.conjugate());
    let angle_rad = 2.0 * rel.w.abs().clamp(0.0, 1.0).acos();
    angle_rad.to_degrees() / dt.max(1e-6)
}

/// Soft elastic clamp on the heading correction angle. When the
/// raw-vs-smoothed delta exceeds `max_corr_deg`, the smoothed quat
/// is pulled back toward raw via logarithmic compression:
///
///   soft_angle = max_corr_deg · (1 + ln(angle / max_corr_deg))
///   t          = soft_angle / angle      (∈ (0, 1])
///   smoothed'  = SLERP(raw, smoothed, t)
///
/// At `angle = max_corr_deg`, this is a no-op (soft = limit, t = 1).
/// At `angle = e · max_corr_deg`, soft = 2 · max_corr_deg, t = 2/e
/// ≈ 0.74 — the smoothing is partially relaxed. As `angle → ∞`, the
/// log keeps growing but slowly; the effective correction stays
/// bounded ~`limit · ln(angle/limit)`. Prevents black borders.
pub fn soft_elastic_clamp(raw: Quat, smoothed: Quat, max_corr_deg: f32) -> Quat {
    if max_corr_deg <= 0.0 { return smoothed; }
    let correction = raw.mul(smoothed.conjugate());
    let angle_rad = 2.0 * correction.w.abs().clamp(0.0, 1.0).acos();
    let angle_deg = angle_rad.to_degrees();
    if angle_deg <= max_corr_deg { return smoothed; }
    let soft_angle = max_corr_deg * (1.0 + (angle_deg / max_corr_deg).ln());
    let t = (soft_angle / angle_deg).clamp(0.0, 1.0);
    raw.slerp(smoothed, t)
}

/// Per-frame, per-eye rotation matrices. Combines:
/// 1. Heading correction: `q_heading = q_raw · q_smooth⁻¹`
/// 2. Optional max-corr clamp (set `max_corr_deg = 0` to disable)
/// 3. IORI per-eye split:
///     - `q_left  = q_iori · q_heading`
///     - `q_right = q_iori⁻¹ · q_heading`
///    When IORI = identity, both eyes get `q_heading`.
///
/// Returns `(q_left, q_right)` quaternions — caller converts to
/// rotation matrices for the GPU.
///
/// Equivalent to [`per_eye_rotations_horizon_lock`] with
/// `horizon_lock = false`. Kept as a thin wrapper for back-compat
/// with code that doesn't care about horizon lock yet.
pub fn per_eye_rotations(
    raw: Quat,
    smoothed: Quat,
    iori: Quat,
    max_corr_deg: f32,
) -> (Quat, Quat) {
    per_eye_rotations_horizon_lock(raw, smoothed, iori, max_corr_deg, false)
}

/// Like [`per_eye_rotations`] but with an explicit horizon-lock toggle.
///
/// When `horizon_lock = true`, the smoothed quat is decomposed via
/// swing-twist around the camera forward axis (+Z) and only the
/// **swing** (pitch + yaw) survives. The Z-**twist** (roll) is
/// dropped from `smoothed`, so:
///
/// ```text
///   q_heading = q_raw · swing_z(smoothed_clamped)⁻¹
/// ```
///
/// carries the full Z-twist of `q_raw`. Applied via the equirect
/// shader's `R` uniform, that Z-twist un-rolls the output. Effect:
/// the rendered horizon stays level regardless of how the camera
/// was tilted around its optical axis (provided GRAV alignment is
/// also on, which anchors world-up to true gravity at clip start).
///
/// Note: rolling around the **view direction** never crops a VR180
/// half-equirect (it's symmetric under Z-rotation by definition),
/// so the soft-clamp does not need to fight horizon-lock. We keep
/// the clamp active on the **swing** part to still bound pitch+yaw
/// over-correction on extreme moves.
pub fn per_eye_rotations_horizon_lock(
    raw: Quat,
    smoothed: Quat,
    iori: Quat,
    max_corr_deg: f32,
    horizon_lock: bool,
) -> (Quat, Quat) {
    // Clamp first (against the original smoothed). The clamp is about
    // the heading angle a user perceives — we measure it before
    // touching the roll component.
    let smoothed_clamped = soft_elastic_clamp(raw, smoothed, max_corr_deg);
    // Horizon-lock pass: drop the Z-twist of the smoothed quat so the
    // resulting heading captures `raw`'s roll verbatim.
    let smoothed_final = if horizon_lock {
        let (swing, _twist) = swing_twist(smoothed_clamped, [0.0, 0.0, 1.0]);
        swing
    } else {
        smoothed_clamped
    };
    let q_heading = raw.mul(smoothed_final.conjugate());
    let q_left  = iori.mul(q_heading);
    let q_right = iori.conjugate().mul(q_heading);
    (q_left, q_right)
}

/// Swing-twist decomposition of a unit quaternion `q` around the
/// unit `axis`. Returns `(swing, twist)` such that
/// `q = swing · twist` (twist applied first, then swing) where
/// **twist** is a rotation purely around `axis` and **swing** is
/// a rotation perpendicular to `axis`.
///
/// Algorithm (standard):
/// 1. Project the quat's `(x, y, z)` component onto `axis`.
/// 2. `twist_raw = (w, axis · projection)`.
/// 3. Normalize → `twist`.
/// 4. `swing = q · twist⁻¹`.
///
/// Degenerate case: when `q`'s vector part is purely perpendicular
/// to `axis`, `twist_raw` has zero vector part and could fail to
/// normalize. We fall back to identity twist (all rotation is
/// swing) in that case.
pub fn swing_twist(q: Quat, axis: [f32; 3]) -> (Quat, Quat) {
    let p_dot_a = q.x * axis[0] + q.y * axis[1] + q.z * axis[2];
    let twist_raw = Quat {
        w: q.w,
        x: axis[0] * p_dot_a,
        y: axis[1] * p_dot_a,
        z: axis[2] * p_dot_a,
    };
    let n = twist_raw.norm();
    let twist = if n < 1e-6 {
        Quat::IDENTITY
    } else {
        Quat {
            w: twist_raw.w / n,
            x: twist_raw.x / n,
            y: twist_raw.y / n,
            z: twist_raw.z / n,
        }
    };
    let swing = q.mul(twist.conjugate());
    (swing, twist)
}

/// Compute the gravity-alignment quaternion from the first `n` GRAV
/// samples. Output: a unit quaternion `g` such that right-multiplying
/// every CORI by `g⁻¹` aligns the camera's world frame so that
/// gravity points in `(0, 1, 0)` (GoPro's convention).
///
/// GRAV axis remap (per the Python `parse_gyro_raw.py` convention):
/// bodyX ← raw[0], bodyY ← raw[2], bodyZ ← raw[1]. Different from
/// GYRO's `(1, 2, 0)` map — this is GoPro-specific and worth
/// double-checking against `gravity_alignment.md` for new firmware.
pub fn gravity_alignment_quat(
    grav_samples: &[[f32; 3]],
    grav_scal: f32,
    n: usize,
) -> Quat {
    let n = n.min(grav_samples.len());
    if n == 0 { return Quat::IDENTITY; }

    // Average + axis remap.
    let inv_scal = 1.0 / grav_scal.max(1e-6);
    let mut acc = [0.0_f32; 3];
    for i in 0..n {
        let raw = grav_samples[i];
        // GRAV order: bodyX = raw[0], bodyY = raw[2], bodyZ = raw[1].
        acc[0] += raw[0] * inv_scal;
        acc[1] += raw[2] * inv_scal;
        acc[2] += raw[1] * inv_scal;
    }
    let inv_n = 1.0 / n as f32;
    let g = [acc[0] * inv_n, acc[1] * inv_n, acc[2] * inv_n];
    let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if mag < 1e-6 { return Quat::IDENTITY; }
    let gn = [g[0] / mag, g[1] / mag, g[2] / mag];

    // Quat rotating `gn` to (0, 1, 0):
    //   axis = gn × (0, 1, 0) = (gn.z, 0, -gn.x)   (right-hand rule)
    //   cos_angle = gn · (0, 1, 0) = gn.y
    let cos_a = gn[1].clamp(-1.0, 1.0);
    let angle = cos_a.acos();
    if angle < 1e-6 {
        // Already aligned with up axis.
        return Quat::IDENTITY;
    }
    let axis_raw = [gn[2], 0.0, -gn[0]];
    let axis_mag = (axis_raw[0] * axis_raw[0]
        + axis_raw[1] * axis_raw[1]
        + axis_raw[2] * axis_raw[2]).sqrt();
    if axis_mag < 1e-6 {
        // Gravity is anti-parallel to up (camera mounted upside-down).
        // Rotate 180° around any horizontal axis.
        return Quat { w: 0.0, x: 1.0, y: 0.0, z: 0.0 };
    }
    let an = [axis_raw[0] / axis_mag, axis_raw[1] / axis_mag, axis_raw[2] / axis_mag];
    let half = angle * 0.5;
    let s = half.sin();
    Quat {
        w: half.cos(),
        x: s * an[0],
        y: s * an[1],
        z: s * an[2],
    }
}

/// Right-multiply each CORI quat by `g_inv` in place. Equivalent to
/// rotating the world frame so gravity points along `+Y`.
pub fn apply_gravity_alignment_inplace(cori: &mut [Quat], g_inv: Quat) {
    for q in cori.iter_mut() {
        *q = q.mul(g_inv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(w: f32, x: f32, y: f32, z: f32) -> Quat { Quat { w, x, y, z } }

    #[test]
    fn smooth_identity_input_returns_identity() {
        let raw = vec![Quat::IDENTITY; 10];
        let out = bidirectional_smooth(&raw, 30.0, &SmoothParams::default());
        for q in out {
            assert!((q.w - 1.0).abs() < 1e-4);
            assert!(q.x.abs() < 1e-4);
            assert!(q.y.abs() < 1e-4);
            assert!(q.z.abs() < 1e-4);
        }
    }

    #[test]
    fn smooth_zero_smooth_ms_is_passthrough() {
        let raw: Vec<Quat> = (0..10)
            .map(|i| q(((i as f32) * 0.1f32).cos(), (i as f32 * 0.1).sin(), 0.0, 0.0))
            .collect();
        let params = SmoothParams { smooth_ms: 0.0, ..Default::default() };
        let out = bidirectional_smooth(&raw, 30.0, &params);
        assert_eq!(out.len(), raw.len());
        for (a, b) in raw.iter().zip(out.iter()) {
            assert!((a.w - b.w).abs() < 1e-5);
            assert!((a.x - b.x).abs() < 1e-5);
        }
    }

    #[test]
    fn smooth_attenuates_high_freq_jitter() {
        // Build a "true motion" (slow ramp on x axis) + per-frame
        // high-freq jitter. Smoother should preserve the slow ramp
        // and kill the jitter.
        let n = 100;
        let mut raw = Vec::with_capacity(n);
        for i in 0..n {
            let slow_x = (i as f32) * 0.001;
            let jitter = if i % 2 == 0 { 0.05 } else { -0.05 };
            let x = slow_x + jitter;
            raw.push(q((1.0_f32 - x*x).sqrt(), x, 0.0, 0.0).normalize());
        }
        let out = bidirectional_smooth(&raw, 30.0, &SmoothParams::default());
        // Mid-output should track the slow ramp (~0.05) much more
        // closely than the raw jitter (raw[50].x = 0.05 + ±0.05).
        let raw_jitter = raw[50].x;
        let smooth_x = out[50].x;
        // The smoothed value's x should be within 0.02 of the slow
        // ramp at frame 50 (= 0.05), regardless of the jitter sign.
        assert!((smooth_x - 0.05).abs() < 0.02,
            "smoothed x at frame 50 = {}, raw = {}", smooth_x, raw_jitter);
    }

    #[test]
    fn soft_clamp_within_limit_is_noop() {
        let raw = q(0.9998, 0.02, 0.0, 0.0);  // ~2.3° rotation
        let smoothed = Quat::IDENTITY;
        let r = soft_elastic_clamp(raw, smoothed, 15.0);
        assert!((r.w - smoothed.w).abs() < 1e-5);
    }

    #[test]
    fn soft_clamp_beyond_limit_softens() {
        // Build a raw quat that's ~30° from smoothed (identity).
        // After soft clamp with limit 15°, output should be partially
        // pulled back toward raw — angle(raw, output) < angle(raw, smoothed) = 30°.
        let half = 15.0_f32.to_radians();  // 30°/2
        let raw = q(half.cos(), half.sin(), 0.0, 0.0);
        let smoothed = Quat::IDENTITY;
        let r = soft_elastic_clamp(raw, smoothed, 15.0);

        let corr_after = raw.mul(r.conjugate());
        let angle_after = 2.0 * corr_after.w.abs().clamp(0.0, 1.0).acos().to_degrees();
        assert!(angle_after < 30.0,
            "soft clamp should reduce 30° correction; got {}", angle_after);
        assert!(angle_after > 15.0,
            "soft clamp shouldn't fully cap at limit; got {}", angle_after);
    }

    #[test]
    fn per_eye_with_identity_iori_gives_same_matrix() {
        let raw = q(0.99, 0.1, 0.0, 0.0).normalize();
        let smoothed = q(0.999, 0.04, 0.0, 0.0).normalize();
        let (l, r) = per_eye_rotations(raw, smoothed, Quat::IDENTITY, 0.0);
        // With IORI = identity, left = right = q_heading.
        assert!((l.w - r.w).abs() < 1e-5);
        assert!((l.x - r.x).abs() < 1e-5);
        assert!((l.y - r.y).abs() < 1e-5);
        assert!((l.z - r.z).abs() < 1e-5);
    }

    #[test]
    fn gravity_align_vertical_returns_identity() {
        // Camera level → body-frame grav = (0, 1, 0) → q_g = identity.
        // The GRAV axis remap (bodyX=raw[0], bodyY=raw[2], bodyZ=raw[1])
        // means we have to supply raw [0, 0, 1] to land at body (0, 1, 0).
        let samples = vec![[0.0, 0.0, 1.0]; 10];
        let g = gravity_alignment_quat(&samples, 1.0, 10);
        assert!((g.w - 1.0).abs() < 1e-4, "g.w = {}", g.w);
        assert!(g.x.abs() < 1e-4);
        assert!(g.y.abs() < 1e-4);
        assert!(g.z.abs() < 1e-4);
    }

    #[test]
    fn swing_twist_roundtrip_recovers_q() {
        // Pick a quat with both a Z-twist and an off-axis swing.
        let half_z = 20.0_f32.to_radians() * 0.5;
        let twist_only = q(half_z.cos(), 0.0, 0.0, half_z.sin());
        let half_x = 30.0_f32.to_radians() * 0.5;
        let swing_only = q(half_x.cos(), half_x.sin(), 0.0, 0.0);
        // swing applied after twist: q = swing · twist
        let qfull = swing_only.mul(twist_only);
        let (s, t) = swing_twist(qfull, [0.0, 0.0, 1.0]);
        // s · t should equal qfull bit-for-bit (modulo numerical error).
        let recon = s.mul(t);
        assert!((recon.w - qfull.w).abs() < 1e-5, "w: {} vs {}", recon.w, qfull.w);
        assert!((recon.x - qfull.x).abs() < 1e-5);
        assert!((recon.y - qfull.y).abs() < 1e-5);
        assert!((recon.z - qfull.z).abs() < 1e-5);
        // The twist should be pure-Z (x, y components zero).
        assert!(t.x.abs() < 1e-4, "twist x = {}", t.x);
        assert!(t.y.abs() < 1e-4, "twist y = {}", t.y);
        // The swing should have no Z-twist component.
        let (_, t2) = swing_twist(s, [0.0, 0.0, 1.0]);
        assert!((t2.w.abs() - 1.0).abs() < 1e-4, "swing has residual twist (w={})", t2.w);
    }

    #[test]
    fn swing_twist_pure_z_rotation_has_identity_swing() {
        // q = pure rotation around Z by 30°.
        let half = 15.0_f32.to_radians();
        let qz = q(half.cos(), 0.0, 0.0, half.sin());
        let (swing, twist) = swing_twist(qz, [0.0, 0.0, 1.0]);
        // Swing should be identity.
        assert!((swing.w - 1.0).abs() < 1e-4);
        assert!(swing.x.abs() < 1e-4);
        assert!(swing.y.abs() < 1e-4);
        assert!(swing.z.abs() < 1e-4);
        // Twist should equal qz.
        assert!((twist.w - qz.w).abs() < 1e-4);
        assert!((twist.z - qz.z).abs() < 1e-4);
    }

    #[test]
    fn horizon_lock_zeroes_smoothed_roll() {
        // raw = identity (no jitter).  smoothed = pure 20° roll around Z.
        // Without horizon-lock: q_heading = raw · smoothed⁻¹ = smoothed⁻¹
        //   → carries the 20° un-roll.
        // With horizon-lock: smoothed's Z-twist is dropped, so
        //   smoothed_for_heading = identity, q_heading = identity
        //   → NO un-rolling happens; the rendered output keeps
        //   raw's roll, which (since raw = identity) is zero. So
        //   output horizon stays level WITHOUT any correction.
        // Either way the rendered horizon ends up level, but the
        // mechanism differs. Check the right one fires.
        let half = 10.0_f32.to_radians();
        let smoothed = q(half.cos(), 0.0, 0.0, half.sin()); // 20° around Z
        let raw = Quat::IDENTITY;

        // Without horizon-lock: q_heading has roll.
        let (l_off, _) = per_eye_rotations_horizon_lock(
            raw, smoothed, Quat::IDENTITY, 0.0, false,
        );
        assert!(l_off.z.abs() > 0.05, "expected z-twist in heading without lock; got {}", l_off.z);

        // With horizon-lock: q_heading is identity.
        let (l_on, r_on) = per_eye_rotations_horizon_lock(
            raw, smoothed, Quat::IDENTITY, 0.0, true,
        );
        assert!((l_on.w - 1.0).abs() < 1e-3, "expected identity heading w with lock; got {}", l_on.w);
        assert!(l_on.x.abs() < 1e-3);
        assert!(l_on.y.abs() < 1e-3);
        assert!(l_on.z.abs() < 1e-3);
        // Both eyes still match (IORI identity).
        assert!((l_on.w - r_on.w).abs() < 1e-5);
    }

    #[test]
    fn horizon_lock_preserves_smoothed_yaw_pan() {
        // raw = 10° yaw, smoothed = 10° yaw (steady pan).
        // Either way q_heading should be ~identity (yaw matches),
        // because swing-twist around Z leaves Y-rotation in the swing.
        let half = 5.0_f32.to_radians();
        let qy = q(half.cos(), 0.0, half.sin(), 0.0);
        let (l, _) = per_eye_rotations_horizon_lock(
            qy, qy, Quat::IDENTITY, 0.0, true,
        );
        assert!((l.w - 1.0).abs() < 1e-3, "yaw pan should produce identity heading; got w={}", l.w);
        assert!(l.x.abs() < 1e-3);
        assert!(l.y.abs() < 1e-3);
        assert!(l.z.abs() < 1e-3);
    }

    #[test]
    fn gravity_align_tilted_solves_correct_rotation() {
        // Camera tilted 30° toward one side: body gravity vector is
        // (sin30, cos30, 0). After the GRAV axis remap (bodyX=raw[0],
        // bodyY=raw[2], bodyZ=raw[1]), to land at body (sin30, cos30, 0)
        // we supply raw = (sin30, 0, cos30): raw[0] → bodyX = sin30,
        // raw[2] → bodyY = cos30, raw[1] → bodyZ = 0. q_g should
        // rotate this body vector onto (0, 1, 0), i.e. a 30° rotation.
        // Quat angle = 30°, so |w| = cos(15°).
        let s30 = 30.0_f32.to_radians().sin();
        let c30 = 30.0_f32.to_radians().cos();
        let samples = vec![[s30, 0.0, c30]; 10];
        let g = gravity_alignment_quat(&samples, 1.0, 10);
        let expected_w = 15.0_f32.to_radians().cos();
        assert!((g.w - expected_w).abs() < 1e-3,
            "g.w = {}, expected {}", g.w, expected_w);
    }
}
