//! Insta360 raw IMU → the per-frame orientation stream the shared
//! (DJI-style) stabilizer consumes.
//!
//! The `.insv` trailer gives a ~1 kHz gyro + accel stream and, per video
//! frame, the end-of-exposure timestamp on the same clock. This module
//! runs VQF 6D over the gyro to get a per-sample orientation and
//! resamples it into the [`DjiOsvImu`] shape so `compute_dji_stabilization`
//! (camera-lock / soft-stab / max-corr / IMU-phase) and the per-row
//! rolling-shutter path apply unchanged:
//!
//! - `frame_quats[i]`     = orientation at frame *i*'s content time;
//! - `high_rate_quats[i]` = [`HR_PER_FRAME`] samples spread uniformly over
//!   the frame period centred on the content time — sample *k* sits at
//!   `t_content − frame_dur/2 + k·frame_dur/HR`. That is exactly the
//!   timeline the DJI sampler reconstructs, so with the IMU phase at
//!   mid-frame (the default seeded for `.insv`) the stab sample lands on
//!   the content time, and the phase slider becomes a ±sync trim.
//!
//! Content time = `t_end_of_exposure − exposure/2 + gyro offset`: the
//! mid-exposure instant of the centre rows plus the file's `gyro_timestamp`
//! (1.1 ms on the X6; [`INSV_X6_SYNC_MS`] when absent — the measured value).
//! The rolling-shutter readout is the file's `rolling_shutter_time`
//! ([`INSV_X6_READOUT_MS`] when absent).
//!
//! Lens model: **per-file factory calibration** from the trailer's
//! calibration string — a Unified Camera Model (ξ) + even radial polynomial
//! + tangential + thin-prism terms (`vr180_fisheye::insta360::InsvLensCalib`,
//! matched to Insta360 Studio's output). [`stream_lens_calib`] expresses it
//! in stream coordinates: the video is the sensor window from the file's
//! `window_crop_info` (X6: the 7744² half minus a 32 px border, i.e. 7680²)
//! binned 2×2 to 3840², so `fx` halves and the principal point is
//! `(c − 32)/2` — per lens, NOT the frame centre (entry 2 sits ~24 px left
//! of it). The exact model rides along as `DjiLensCalib::omni` for the
//! renderers; `fx/k1..k5` hold a KB-5 fit of its radial curve (0.02 px
//! inside 95°) for the override UI and CPU consumers. Without a usable
//! string the measured X6 profile in [`x6_stream_lens_calib`] is the
//! fallback. Track order: the X6 stores its tracks reversed (track 0 =
//! calibration entry 2), per the file's `stream_type`/track-order fields —
//! verified by matching features across the two streams' overlap through
//! the factory extrinsics.

use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::insta360::{Insta360Meta, InsvLensCalib, InsvWindowCrop};
use vr180_fisheye::{DjiLensCalib, DjiOsvImu, OmniLensModel};

/// IMU→camera basis for the X6's stream-0 (back) lens — rows are the
/// camera axes (x right, y up, z optical) expressed in IMU coordinates.
/// Measured on four walking clips by fitting the rotation between
/// consecutive frames (exact factory lens model, far-field features) to the
/// gyro's rotation over the same interval: a signed axis permutation plus a
/// 0.6° tilt about the camera's x axis, consistent per clip to 0.3° and
/// with the lens extrinsics in the calibration string. (An earlier value
/// carried a 4.5° tilt, an artefact of the lens model used at the time; it
/// left a residual proportional to the motion.) Re-orthonormalised at use.
pub const INSV_X6_IMU_TO_CAM: [[f32; 3]; 3] = [
    [0.0035, 0.0044, -1.0000],
    [1.0000, 0.0083, 0.0036],
    [0.0084, -1.0000, -0.0044],
];

/// Fallback centre-row content time relative to `t_end_of_exposure −
/// exposure/2` (ms), used when the file has no `gyro_timestamp`. Best-fit
/// sync of image motion vs gyro (R² plateau 1.0–2.5 ms); the X6 files
/// state 1.1 ms.
pub const INSV_X6_SYNC_MS: f32 = 1.5;

/// Fallback sensor readout (rolling-shutter) time, ms, for the 2×2-binned
/// 3840² video modes, used when the file has no `rolling_shutter_time`
/// (the X6 states 15.36 ms at 50 fps and 14.56 ms at 30 fps; a
/// predict-then-measure sweep on the walking clip agreed on 9–18 ms).
pub const INSV_X6_READOUT_MS: f32 = 15.0;

/// Inclination trim time constant for the orientation stream, s. Insta360
/// Studio's orientation is gyro integration with a fixed bias and a batch
/// gravity alignment (no per-sample accelerometer correction — that is what
/// wobbles at walking cadence); a 100 s trim reproduces it to 0.01° rms
/// while still bounding tilt drift over long takes.
pub const INSV_TAU_ACC_S: f64 = 100.0;
/// Low-pass on the earth-frame accelerometer that feeds the trim, s.
const INSV_TRIM_LP_S: f64 = 2.0;

/// Insta360 Studio's per-frame orientation lags the raw gyro integration by
/// this much (matched on a walking clip, ±0.1 ms), so the content time is
/// sampled that much earlier.
pub const INSV_STUDIO_SAMPLE_LAG_MS: f64 = 0.63;

/// X6 video sensor window when the file omits `window_crop_info`.
const X6_DEFAULT_CROP: InsvWindowCrop = InsvWindowCrop {
    src_w: 7744, src_h: 7744, dst_w: 7680, dst_h: 7680, offset_x: 0, offset_y: 0,
};

/// Fallback stream-space (3840²) focal length, px/rad — KB-4 fit of a
/// typical X6 factory entry (paraxial 1044 px/rad); used only when the
/// trailer has no usable calibration string.
pub const INSV_X6_FX: f32 = 1044.389;
/// Fallback stream-space KB-4 radial coefficients (θ³, θ⁵, θ⁷, θ⁹ terms).
pub const INSV_X6_K: [f32; 4] = [0.087926, -0.023232, 0.001227, -0.000388];
/// The stream side the constants above were measured at.
pub const INSV_X6_CALIB_SIDE: f32 = 3840.0;

/// Synthetic high-rate samples per video frame.
pub const HR_PER_FRAME: usize = 16;

/// Gram-Schmidt re-orthonormalisation of the (rounded) measured basis so
/// the similarity transform downstream is an exact rotation.
pub fn insv_x6_imu_to_cam() -> [[f32; 3]; 3] {
    // Diagnostic override: `VR180_INSV_BASIS="r00,r01,…,r22"` (row-major)
    // lets the basis be A/B-tested headlessly without a rebuild.
    let m = std::env::var("VR180_INSV_BASIS").ok()
        .and_then(|v| {
            let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (f.len() == 9).then(|| [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]])
        })
        .unwrap_or(INSV_X6_IMU_TO_CAM);
    let norm = |v: [f32; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let x = norm(m[0]);
    let dy = dot(m[1], x);
    let y = norm([m[1][0] - dy * x[0], m[1][1] - dy * x[1], m[1][2] - dy * x[2]]);
    // z = x × y keeps the frame right-handed.
    let z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    [x, y, z]
}

/// The measured X6 lens model expressed for a `stream_w × stream_h`
/// working frame (both lenses share it; the two factory entries differ by
/// 0.15 % in focal, below what the measurement resolves). Principal point
/// = frame centre: the firmware re-centres each lens when cropping the
/// 3840² streams (rim fit lands within 2 px of centre on both streams).
pub fn x6_stream_lens_calib(stream_w: u32, stream_h: u32) -> DjiLensCalib {
    let s = stream_w as f32 / INSV_X6_CALIB_SIDE;
    DjiLensCalib {
        fx: Some(INSV_X6_FX * s),
        fy: Some(INSV_X6_FX * s),
        cx: Some(stream_w as f32 * 0.5),
        cy: Some(stream_h as f32 * 0.5),
        k1: Some(INSV_X6_K[0]),
        k2: Some(INSV_X6_K[1]),
        k3: Some(INSV_X6_K[2]),
        k4: Some(INSV_X6_K[3]),
        k5: Some(0.0),
        p1: Some(0.0),
        p2: Some(0.0),
        width: Some(stream_w as f32),
        height: Some(stream_h as f32),
        ..Default::default()
    }
}

/// Stream pixels per native calibration pixel: the recorded window
/// (`dst`) is scaled to the stream size (X6: 7680 → 3840 = ½, the 2×2 binning).
pub fn stream_scale(crop: Option<&InsvWindowCrop>, stream_w: u32) -> f64 {
    let c = crop.copied().unwrap_or(X6_DEFAULT_CROP);
    stream_w as f64 / c.dst_w.max(1) as f64
}

/// Principal point of a factory entry in stream coordinates: the native
/// centre, relative to its sensor half, shifted by the window origin —
/// centred `(src − dst)/2` plus the stored offset — then scaled. Matches
/// Insta360 Studio's conversion.
pub fn stream_principal_point(entry: &InsvLensCalib, crop: Option<&InsvWindowCrop>, stream_w: u32, stream_h: u32) -> (f32, f32) {
    let c = crop.copied().unwrap_or(X6_DEFAULT_CROP);
    let sx = stream_w as f64 / c.dst_w.max(1) as f64;
    let sy = stream_h as f64 / c.dst_h.max(1) as f64;
    let ox = entry.half_origin_x() as f64 + (c.src_w as f64 - c.dst_w as f64) * 0.5 + c.offset_x as f64;
    let oy = (c.src_h as f64 - c.dst_h as f64) * 0.5 + c.offset_y as f64;
    (((entry.cx as f64 - ox) * sx) as f32, ((entry.cy as f64 - oy) * sy) as f32)
}

/// Per-file lens calibration for one video stream, in stream coordinates.
///
/// - Frame: the file's sensor window (`crop`, X6 default 7744² → 7680²
///   centred) scaled to the stream (3840² = 2×2-binned → exactly ½), so the
///   principal point is `(c − 32)/2` per lens and `fx` halves.
/// - `omni`: the exact factory model (UCM + radial + tangential + prism),
///   which the renderers use.
/// - `fx/k1..k5`: a weighted linear least-squares KB-5 fit of the radial
///   curve `r(θ)` on θ ∈ [0, 106°] (full weight inside 96°) for the override
///   UI seeds and CPU consumers (0.02 px inside 95°).
pub fn stream_lens_calib(entry: &InsvLensCalib, crop: Option<&InsvWindowCrop>, stream_w: u32, stream_h: u32) -> DjiLensCalib {
    let bin = stream_scale(crop, stream_w);
    let (cx, cy) = stream_principal_point(entry, crop, stream_w, stream_h);
    let (f, k) = fit_kb5(|th| entry.project_radius(th) * bin);
    let omni = OmniLensModel {
        xi: entry.xi,
        fx: (entry.fx as f64 * bin) as f32,
        fy: (entry.fy as f64 * bin) as f32,
        radial: entry.radial5(),
        tangential: entry.tangential(),
        prism: entry.prism(),
    };
    DjiLensCalib {
        fx: Some(f as f32),
        fy: Some(f as f32),
        cx: Some(cx),
        cy: Some(cy),
        k1: Some(k[0] as f32),
        k2: Some(k[1] as f32),
        k3: Some(k[2] as f32),
        k4: Some(k[3] as f32),
        k5: Some(k[4] as f32),
        p1: Some(0.0),
        p2: Some(0.0),
        width: Some(stream_w as f32),
        height: Some(stream_h as f32),
        omni: Some(omni),
        ..Default::default()
    }
}

/// Weighted linear least squares of the KB-5 odd polynomial to a radius
/// curve `r(θ)` (px), θ ∈ [0, 106°]. Returns `(f, [k1..k5])`. Solved in the
/// normalised variable `t = θ / θ_max` for conditioning, then rescaled.
fn fit_kb5(r_of_theta: impl Fn(f64) -> f64) -> (f64, [f64; 5]) {
    const N: usize = 6;
    let th_max = 106f64.to_radians();
    let mut ata = [[0f64; N]; N];
    let mut atb = [0f64; N];
    let steps = 424; // 0.25° grid
    for i in 0..=steps {
        let th = th_max * i as f64 / steps as f64;
        let r = r_of_theta(th);
        let w = if th < 96f64.to_radians() { 1.0 } else { 0.3 };
        let t = th / th_max;
        let mut phi = [0f64; N];
        let mut p = t;
        for j in 0..N { phi[j] = p; p *= t * t; }
        for a in 0..N {
            for b in 0..N { ata[a][b] += w * phi[a] * phi[b]; }
            atb[a] += w * phi[a] * r;
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut m = ata;
    let mut v = atb;
    for col in 0..N {
        let piv = (col..N).max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap()).unwrap();
        m.swap(col, piv); v.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-300 { continue; }
        for row in col + 1..N {
            let fac = m[row][col] / d;
            for c2 in col..N { m[row][c2] -= fac * m[col][c2]; }
            v[row] -= fac * v[col];
        }
    }
    let mut a = [0f64; N];
    for row in (0..N).rev() {
        let mut sum = v[row];
        for c2 in row + 1..N { sum -= m[row][c2] * a[c2]; }
        a[row] = if m[row][row].abs() > 1e-300 { sum / m[row][row] } else { 0.0 };
    }
    // r = Σ a_j t^(2j+1) = Σ a_j θ^(2j+1) / θ_max^(2j+1)  ⇒  f = a_0/θ_max, k_j = (a_j/a_0)/θ_max^(2j)
    let f = a[0] / th_max;
    let mut k = [0f64; 5];
    for j in 1..N { k[j - 1] = (a[j] / a[0]) / th_max.powi(2 * j as i32); }
    (f, k)
}

/// Calibration-only IMU (no quats) — enough to dewarp and to seed the
/// Override UI; stabilization stays off until the full build lands.
pub fn insv_calib_only(meta: &Insta360Meta, stream_w: u32, stream_h: u32) -> DjiOsvImu {
    // Stream 0 (back lens = left eye by default = `lens_b`) carries
    // calibration entry 2 on the X6 — its tracks are stored in reverse lens
    // order (file flag; default when the file doesn't say) — and stream 1
    // (screen lens = `lens_a`) entry 1.
    let reversed = meta.track_order_reversed.unwrap_or(true);
    let (i0, i1) = if reversed { (1, 0) } else { (0, 1) };
    let crop = meta.window_crop.as_ref();
    let (lens_b, lens_a) = if meta.lenses.len() >= 2 {
        (stream_lens_calib(&meta.lenses[i0], crop, stream_w, stream_h),
         stream_lens_calib(&meta.lenses[i1], crop, stream_w, stream_h))
    } else {
        tracing::warn!("insv: no per-lens calibration string — using the generic X6 profile");
        (x6_stream_lens_calib(stream_w, stream_h), x6_stream_lens_calib(stream_w, stream_h))
    };
    DjiOsvImu {
        lens_a,
        lens_b,
        camera_model: Some(if meta.model.is_empty() { "Insta360".to_string() } else { meta.model.clone() }),
        imu_to_cam: Some(insv_x6_imu_to_cam()),
        readout_ms: meta.readout_ms.map(|v| v as f32).filter(|v| *v > 0.0),
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
struct Q64 { w: f64, x: f64, y: f64, z: f64 }

impl Q64 {
    const IDENTITY: Q64 = Q64 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
    fn mul(self, b: Q64) -> Q64 {
        Q64 {
            w: self.w * b.w - self.x * b.x - self.y * b.y - self.z * b.z,
            x: self.w * b.x + self.x * b.w + self.y * b.z - self.z * b.y,
            y: self.w * b.y - self.x * b.z + self.y * b.w + self.z * b.x,
            z: self.w * b.z + self.x * b.y - self.y * b.x + self.z * b.w,
        }
    }
    fn normalize(self) -> Q64 {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt().max(1e-300);
        Q64 { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }
    fn from_rotvec(v: [f64; 3]) -> Q64 {
        let a = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if a < 1e-15 { return Q64::IDENTITY; }
        let (s, c) = ((a * 0.5).sin() / a, (a * 0.5).cos());
        Q64 { w: c, x: v[0] * s, y: v[1] * s, z: v[2] * s }
    }
    /// Rotate a body vector into the earth frame (`q v q*`).
    fn rotate(self, v: [f64; 3]) -> [f64; 3] {
        let p = Q64 { w: 0.0, x: v[0], y: v[1], z: v[2] };
        let r = self.mul(p).mul(Q64 { w: self.w, x: -self.x, y: -self.y, z: -self.z });
        [r.x, r.y, r.z]
    }
    fn to_quat(self) -> Quat {
        Quat { w: self.w as f32, x: self.x as f32, y: self.y as f32, z: self.z as f32 }
    }
}

/// Small earth-frame rotation taking unit vector `from` toward `+z`, scaled by
/// `gain` (0..1 of the full correction).
fn tilt_toward_up(from: [f64; 3], gain: f64) -> Q64 {
    let n = (from[0] * from[0] + from[1] * from[1] + from[2] * from[2]).sqrt();
    if n < 1e-9 { return Q64::IDENTITY; }
    let f = [from[0] / n, from[1] / n, from[2] / n];
    // axis = f × z = (f.y, −f.x, 0); angle = acos(f.z)
    let axis = [f[1], -f[0], 0.0];
    let s = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
    if s < 1e-12 { return Q64::IDENTITY; }
    let ang = f[2].clamp(-1.0, 1.0).acos() * gain;
    Q64::from_rotvec([axis[0] / s * ang, axis[1] / s * ang, 0.0])
}

/// Orientation stream matched to Insta360 Studio's: trapezoid integration of
/// the bias-corrected gyro on the sample timestamps, a batch gravity
/// alignment (the whole-clip mean of the earth-frame accelerometer → `+z`, so
/// walking accelerations cancel instead of wobbling the horizon), and a slow
/// inclination trim (`tau_s`) that keeps long takes level. Body → earth
/// quaternions in the same convention as the VQF output.
pub fn integrate_orientation(meta: &Insta360Meta, bias_rad_s: [f32; 3], tau_s: f64) -> Vec<Quat> {
    const DEG: f64 = std::f64::consts::PI / 180.0;
    let n = meta.imu.len();
    let t: Vec<f64> = meta.imu.iter().map(|s| s.t_us as f64 * 1e-6).collect();
    let w: Vec<[f64; 3]> = meta.imu.iter()
        .map(|s| [s.gyr_dps[0] as f64 * DEG - bias_rad_s[0] as f64,
                  s.gyr_dps[1] as f64 * DEG - bias_rad_s[1] as f64,
                  s.gyr_dps[2] as f64 * DEG - bias_rad_s[2] as f64])
        .collect();
    let step = |k: usize| -> Q64 {
        let dt = (t[k + 1] - t[k]).clamp(0.0, 0.05);
        Q64::from_rotvec([(w[k][0] + w[k + 1][0]) * 0.5 * dt,
                          (w[k][1] + w[k + 1][1]) * 0.5 * dt,
                          (w[k][2] + w[k + 1][2]) * 0.5 * dt])
    };
    // Pass 1: gyro only from identity; mean earth-frame accelerometer.
    let mut q = Q64::IDENTITY;
    let mut g_sum = [0f64; 3];
    for k in 0..n {
        let a = meta.imu[k].acc_g;
        let ae = q.rotate([a[0] as f64, a[1] as f64, a[2] as f64]);
        g_sum[0] += ae[0]; g_sum[1] += ae[1]; g_sum[2] += ae[2];
        if k + 1 < n { q = q.mul(step(k)).normalize(); }
    }
    let align = tilt_toward_up(g_sum, 1.0);
    // Pass 2: same integration from the aligned start, with the slow trim.
    let mut out = Vec::with_capacity(n);
    let mut q = align;
    let mut lp = [0.0, 0.0, 1.0];
    for k in 0..n {
        out.push(q.to_quat());
        if k + 1 >= n { break; }
        let dt = (t[k + 1] - t[k]).clamp(0.0, 0.05);
        q = q.mul(step(k)).normalize();
        if tau_s > 0.0 {
            let a = meta.imu[k + 1].acc_g;
            let ae = q.rotate([a[0] as f64, a[1] as f64, a[2] as f64]);
            let kf = (dt / INSV_TRIM_LP_S).min(1.0);
            for i in 0..3 { lp[i] += (ae[i] - lp[i]) * kf; }
            q = tilt_toward_up(lp, (dt / tau_s).min(1.0)).mul(q).normalize();
        }
    }
    out
}

/// Orientation lookup by time over the VQF output (slerp between the two
/// bracketing samples; clamps outside the stream).
struct QuatTimeline<'a> {
    t_s: Vec<f64>,
    quats: &'a [Quat],
}

impl<'a> QuatTimeline<'a> {
    fn at(&self, t: f64) -> Quat {
        let n = self.t_s.len();
        if n == 0 {
            return Quat::IDENTITY;
        }
        if t <= self.t_s[0] {
            return self.quats[0];
        }
        if t >= self.t_s[n - 1] {
            return self.quats[n - 1];
        }
        let hi = self.t_s.partition_point(|&x| x <= t).min(n - 1);
        let lo = hi - 1;
        let span = self.t_s[hi] - self.t_s[lo];
        let f = if span > 0.0 { ((t - self.t_s[lo]) / span) as f32 } else { 0.0 };
        self.quats[lo].slerp(self.quats[hi], f.clamp(0.0, 1.0))
    }
}

/// Build the stabilizer's per-frame orientation stream from an `.insv`
/// trailer. `n_frames_hint` is the video's frame count (the stamp table
/// normally matches it; extra frames extrapolate at `1/fps`). Returns a
/// calibration-only structure (empty `frame_quats`) when the trailer has
/// no usable IMU or frame stamps.
pub fn build_insv_imu(
    meta: &Insta360Meta,
    fps: f32,
    n_frames_hint: usize,
    stream_w: u32,
    stream_h: u32,
) -> DjiOsvImu {
    let mut out = insv_calib_only(meta, stream_w, stream_h);
    // The synthetic blocks below are centred on each frame's content time.
    out.sample_anchor = vr180_fisheye::SampleAnchor::BlockMid;
    let n_imu = meta.imu.len();
    if n_imu < 16 || meta.frames.is_empty() || fps <= 0.0 {
        tracing::warn!(
            "insv_imu: no usable motion data (imu={} frames={} fps={}) — calib only",
            n_imu, meta.frames.len(), fps
        );
        return out;
    }
    let dt = meta.imu_period_s() as f32;
    if !dt.is_finite() || dt <= 0.0 {
        tracing::warn!("insv_imu: bad IMU period {dt} — calib only");
        return out;
    }

    // Orientation stream — matched to Insta360 Studio's output. Studio
    // integrates the gyro; its orientation carries none of the walking-cadence
    // wobble a per-sample accelerometer correction adds (0.07° rms ≈ 1.3 px
    // at 3840 — visible as residual shake). VQF runs once, for its whole-clip
    // gyro-bias estimate; the stream itself is `integrate_orientation`.
    const DEG: f32 = std::f32::consts::PI / 180.0;
    const G: f32 = 9.80665;
    let gyro: Vec<[f32; 3]> = meta.imu.iter()
        .map(|s| [s.gyr_dps[0] * DEG, s.gyr_dps[1] * DEG, s.gyr_dps[2] * DEG])
        .collect();
    let acc: Vec<[f32; 3]> = meta.imu.iter()
        .map(|s| [s.acc_g[0] * G, s.acc_g[1] * G, s.acc_g[2] * G])
        .collect();
    let t_vqf = std::time::Instant::now();
    let pass1 = vr180_core::gyro::vqf::run(&gyro, &acc, None, dt);
    let quats = integrate_orientation(meta, pass1.bias_rad_s, INSV_TAU_ACC_S);
    tracing::info!(
        "insv_imu: gyro integration over {} samples @ {:.1} Hz in {:.1?}, bias = {:?} °/s (trim tau {} s)",
        n_imu, 1.0 / dt, t_vqf.elapsed(), pass1.bias_deg_s(), INSV_TAU_ACC_S
    );
    let timeline = QuatTimeline {
        t_s: meta.imu.iter().map(|s| s.t_us as f64 * 1e-6).collect(),
        quats: &quats,
    };

    // Per-frame content times: centre rows' mid-exposure + the file's gyro
    // offset − Studio's sampling lag. Each sensor has its own exposure
    // record, so lens B (the second sensor) gets its own timeline.
    let sync_s = (meta.gyro_offset_ms.unwrap_or(INSV_X6_SYNC_MS as f64) - INSV_STUDIO_SAMPLE_LAG_MS) * 1e-3;
    let frame_dur = 1.0 / fps as f64;
    let n_frames = n_frames_hint.max(meta.frames.len());
    if meta.frames.len() != n_frames_hint && n_frames_hint > 0 {
        tracing::info!(
            "insv_imu: {} frame stamps vs {} video frames — extrapolating the tail at 1/fps",
            meta.frames.len(), n_frames_hint
        );
    }
    let build = |stamps: &[vr180_fisheye::insta360::InsvFrameStamp], out: &mut DjiOsvImu| {
        let last = stamps.len() - 1;
        let content_time = |i: usize| -> f64 {
            if i < stamps.len() {
                let f = &stamps[i];
                f.t_us as f64 * 1e-6 - f.exposure_s * 0.5 + sync_s
            } else {
                let f = &stamps[last];
                f.t_us as f64 * 1e-6 - f.exposure_s * 0.5 + sync_s + (i - last) as f64 * frame_dur
            }
        };
        out.frame_quats.clear();
        out.high_rate_quats.clear();
        out.gravity.clear();
        out.frame_quats.reserve(n_frames);
        out.high_rate_quats.reserve(n_frames);
        out.gravity.reserve(n_frames);
        for i in 0..n_frames {
            let tc = content_time(i);
            out.frame_quats.push(timeline.at(tc));
            let start = tc - frame_dur * 0.5;
            let hr: Vec<Quat> = (0..HR_PER_FRAME)
                .map(|k| timeline.at(start + frame_dur * (k as f64) / (HR_PER_FRAME as f64)))
                .collect();
            out.high_rate_quats.push(hr);
            out.gravity.push([0.0, -1.0, 0.0]);
        }
        content_time(0)
    };
    let t0 = build(&meta.frames, &mut out);
    if !meta.frames_b.is_empty() {
        let mut b = DjiOsvImu {
            lens_a: out.lens_a.clone(),
            lens_b: out.lens_b.clone(),
            camera_model: out.camera_model.clone(),
            imu_to_cam: out.imu_to_cam,
            readout_ms: out.readout_ms,
            sample_anchor: vr180_fisheye::SampleAnchor::BlockMid,
            ..Default::default()
        };
        let t0b = build(&meta.frames_b, &mut b);
        let n_diff = meta.frames.iter().zip(meta.frames_b.iter())
            .filter(|(a, c)| ((a.t_us as f64 - a.exposure_s * 0.5e6) - (c.t_us as f64 - c.exposure_s * 0.5e6)).abs() > 500.0)
            .count();
        tracing::info!(
            "insv_imu: lens B timeline from the second sensor's stamps ({} frames, {} differ from lens A by > 0.5 ms; frame 0 {:+.2} ms)",
            meta.frames_b.len(), n_diff, (t0b - t0) * 1e3
        );
        out.lens_b_timeline = Some(Box::new(b));
    }
    tracing::info!(
        "insv_imu: {} frames ({} stamped), content t0 = {:.3} s, imu span {:.3}..{:.3} s",
        n_frames, meta.frames.len(), t0,
        timeline.t_s[0], timeline.t_s[timeline.t_s.len() - 1]
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_a_proper_rotation() {
        let c = insv_x6_imu_to_cam();
        for i in 0..3 {
            for j in 0..3 {
                let d: f32 = (0..3).map(|k| c[i][k] * c[j][k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-5, "row {i}·row {j} = {d}");
            }
        }
        let det = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
            - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
            + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
        assert!((det - 1.0).abs() < 1e-5, "det = {det}");
        // Stays close to the measured values.
        assert!((c[1][0] - 1.0).abs() < 0.01 && c[1][1].abs() < 0.02, "{c:?}");
        assert!((c[0][2] + 1.0).abs() < 0.01);
    }

    #[test]
    fn stream_calib_scales_with_working_res() {
        let c = x6_stream_lens_calib(1280, 1280);
        assert!((c.fx.unwrap() - INSV_X6_FX / 3.0).abs() < 1e-3);
        assert_eq!(c.cx, Some(640.0));
        assert_eq!(c.width, Some(1280.0));
    }

    /// Real-file smoke test — skipped when the fixture isn't mounted.
    #[test]
    fn real_x6_clip_builds_quats() {
        let p = std::path::Path::new("/Volumes/Database/x6/concert/VID_20260815_193658_00_030.insv");
        if !p.exists() { return; }
        let meta = vr180_fisheye::insta360::read_insta360_meta(p).expect("meta");
        let imu = build_insv_imu(&meta, 50.0, 2161, 3840, 3840);
        assert_eq!(imu.frame_quats.len(), 2161);
        assert_eq!(imu.high_rate_quats[100].len(), HR_PER_FRAME);
        // Mid sample of the HR block == the frame quat (content time).
        let a = imu.frame_quats[100];
        let b = imu.high_rate_quats[100][HR_PER_FRAME / 2];
        assert!(a.dot(b).abs() > 0.999_99, "mid HR sample must match frame quat");
        assert!(imu.imu_to_cam.is_some());
        // Per-file factory calibration in stream coords (entry 1 → stream 0 → lens_b).
        assert!((imu.readout_ms.unwrap() - 15.36).abs() < 1e-4);
        let b = imu.lens_b_timeline.as_deref().expect("second sensor timeline");
        assert_eq!(b.frame_quats.len(), 2161);
        // Frame 0: lens B mid-exposure is 4.84 ms earlier → a small but non-zero delta.
        let d = imu.frame_quats[0].conjugate().mul(b.frame_quats[0]).normalize();
        let ang = 2.0 * d.w.abs().min(1.0).acos().to_degrees();
        assert!(ang > 0.05 && ang < 2.0, "lens B delta at frame 0 = {ang}°");
        // Reversed track order: stream 0 (lens_b) = entry 2, stream 1 = entry 1.
        let b = &imu.lens_b;
        assert!((b.fx.unwrap() - 1045.66).abs() < 0.3, "fx {:?}", b.fx);
        assert!((b.cx.unwrap() - 1896.55).abs() < 0.01 && (b.cy.unwrap() - 1922.985).abs() < 0.01, "pp {:?} {:?}", b.cx, b.cy);
        let bo = b.omni.unwrap();
        assert!((bo.fx - 7227.07 * 0.5).abs() < 1e-3 && (bo.xi - 2.45543).abs() < 1e-5);
        let a = &imu.lens_a;
        assert!((a.fx.unwrap() - 1044.14).abs() < 0.3, "fx {:?}", a.fx);
        assert!((a.cx.unwrap() - 1911.73).abs() < 0.01 && (a.cy.unwrap() - 1926.265).abs() < 0.01, "pp {:?} {:?}", a.cx, a.cy);
        assert!((a.k1.unwrap() - 0.0895).abs() < 0.001);
        assert_eq!(a.omni.unwrap().tangential, [-0.00032840, 0.00113587, 0.00412989, 0.00653467]);
    }

    #[test]
    fn kb5_fit_reproduces_factory_curve() {
        let s = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193";
        let entry = &vr180_fisheye::insta360::parse_calib_string(s).unwrap()[0];
        let c = stream_lens_calib(entry, None, 3840, 3840);
        let (f, k) = (c.fx.unwrap() as f64, [c.k1.unwrap() as f64, c.k2.unwrap() as f64, c.k3.unwrap() as f64, c.k4.unwrap() as f64, c.k5.unwrap() as f64]);
        let mut worst = 0f64;
        for i in 0..=1025 {
            let th = 102.5f64.to_radians() * i as f64 / 1025.0;
            let t2 = th * th;
            let r_kb = f * th * (1.0 + k[0]*t2 + k[1]*t2*t2 + k[2]*t2*t2*t2 + k[3]*t2*t2*t2*t2 + k[4]*t2*t2*t2*t2*t2);
            let r_true = entry.project_radius(th) * 0.5;
            worst = worst.max((r_kb - r_true).abs());
        }
        assert!(worst < 0.1, "KB5 max error {worst} px");
    }
}
