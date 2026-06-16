//! Thin wrapper over the [`vqf_rs`] crate matching the Python
//! `vqf_to_cori_quats` call site:
//!
//! ```text
//! vqf = PyVQF(gyro_dt, magDistRejectionEnabled=False, tauMag=5.0)  # if mag
//! vqf.updateBatch(gyro, acc, mag)
//! quat9D = result['quat9D']
//! bias = vqf.getBiasEstimate()[0]  # rad/s
//! ```
//!
//! Use [`run`] for the batch path. The wrapper preserves the per-sample
//! quaternion output (one [`Quat`] per gyro sample, in `quat9D` mode when
//! `mag` is `Some`, `quat6D` otherwise) so a downstream resampler can
//! pick / average to video frame rate as the Python pipeline does.

use super::cori_iori::Quat;

/// Batch VQF result.
#[derive(Debug, Clone)]
pub struct VqfRun {
    /// Per-sample orientation quaternions (one per gyro sample).
    ///
    /// 9D when [`run`] was called with `mag = Some(_)`, otherwise 6D.
    pub quats: Vec<Quat>,
    /// Final gyro bias estimate, **in rad/s** (PyVQF convention).
    pub bias_rad_s: [f32; 3],
    /// Standard deviation of the bias estimate (rad/s).
    pub bias_sigma: f32,
}

impl VqfRun {
    /// Convenience: bias in degrees/second (what the Python printout uses).
    pub fn bias_deg_s(&self) -> [f32; 3] {
        const F: f32 = 180.0 / std::f32::consts::PI;
        [self.bias_rad_s[0] * F, self.bias_rad_s[1] * F, self.bias_rad_s[2] * F]
    }
}

/// Run the VQF filter end-to-end over pre-prepared sample arrays.
///
/// - `gyro` and `acc` must be the same length (`acc` resampled to gyro
///   rate by the caller — VQF is stepped once per gyro sample).
/// - `mag` is optional; when supplied, must match `gyro`'s length.
/// - `gyr_ts` is the gyro sampling interval in seconds (e.g. `1.0 / 798.6`
///   for 798.6 Hz).
///
/// Parameter choices mirror the Python pipeline: when `mag` is present we
/// disable magnetic disturbance rejection (`mag_dist_rejection_enabled = false`)
/// and set `tau_mag = 5.0 s` (Python sets `tauMag=5.0`). The other defaults
/// match PyVQF / vqf-rs defaults.
pub fn run(
    gyro: &[[f32; 3]],
    acc: &[[f32; 3]],
    mag: Option<&[[f32; 3]]>,
    gyr_ts: f32,
) -> VqfRun {
    assert_eq!(gyro.len(), acc.len(), "gyro and acc length mismatch");
    if let Some(m) = mag {
        assert_eq!(m.len(), gyro.len(), "mag length must match gyro");
    }

    // Build params. vqf_rs::Params field names mirror PyVQFParams.
    let mut params = vqf_rs::Params::default();
    if mag.is_some() {
        params.mag_dist_rejection_enabled = false;
        params.tau_mag = 5.0;
    }

    // acc_ts / mag_ts == gyr_ts because the caller resampled both to gyro rate.
    let mut filter = vqf_rs::VQF::new(gyr_ts, None, None, Some(params));

    let mut quats = Vec::with_capacity(gyro.len());
    for i in 0..gyro.len() {
        filter.update(gyro[i], acc[i], mag.map(|m| m[i]));
        let q = if mag.is_some() { filter.quat_9d() } else { filter.quat_6d() };
        // `vqf_rs::Quaternion` is a tuple struct (Float, Float, Float, Float)
        // in (w, x, y, z) order — same component order our `Quat` uses.
        quats.push(Quat { w: q.0, x: q.1, y: q.2, z: q.3 });
    }

    let (bias_arr, sigma) = filter.bias_estimate();
    VqfRun {
        quats,
        bias_rad_s: bias_arr,
        bias_sigma: sigma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stationary IMU: gyro = 0, acc = +g down, no mag.
    /// Bias estimate should drift toward zero, orientation should stay
    /// close to identity (the accelerometer-only inclination correction
    /// aligns to gravity = down).
    #[test]
    fn stationary_imu_produces_near_identity_orientation_and_zero_bias() {
        let n = 1000;
        let gyro = vec![[0.0_f32; 3]; n];
        let acc = vec![[0.0, 0.0, 9.81]; n];
        let result = run(&gyro, &acc, None, 1.0 / 800.0);

        // Bias should be tiny (<= 0.05°/s in any axis).
        for (i, &b) in result.bias_deg_s().iter().enumerate() {
            assert!(b.abs() < 0.05, "axis {i} bias {b:.4}°/s is too large for stationary input");
        }
        // Last orientation should be close to identity. We check |sin(angle/2)|
        // = sqrt(x²+y²+z²); a few degrees tilt is fine (the inclination
        // correction takes a few hundred samples to settle from the default
        // identity start).
        let q = *result.quats.last().unwrap();
        let xyz_norm = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        assert!(xyz_norm < 0.1, "stationary orientation drifted: |xyz|={xyz_norm:.4}");
    }
}
