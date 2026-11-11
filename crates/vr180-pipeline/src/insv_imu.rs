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
//! in stream coordinates through the clip's
//! [`InsvStreamGeometry`](vr180_fisheye::InsvStreamGeometry): the stock
//! video is the sensor window from the file's `window_crop_info` (the 7744²
//! half minus a 32 px border, i.e. 7680²) binned 2×2 to 3840², so `fx`
//! halves and the principal point is `(c − 32)/2` — per lens, NOT the frame
//! centre (entry 2 sits ~24 px left of it). The 7.4K and 10K modes scale
//! their window the same way; the Crop modes are native pixels cut from the
//! middle of the sensor (scale 1, the frame holds part of the image circle,
//! and the 7.4K Crop spans 3712 of the 3968 rows the sensor reads — its
//! readout is the file's value times that share). The exact model rides
//! along as `DjiLensCalib::omni` for the
//! renderers; `fx/k1..k5` hold a KB-5 fit of its radial curve (0.02 px
//! inside 95°) for the override UI and CPU consumers. Without a usable
//! string the measured X6 profile in [`x6_stream_lens_calib`] is the
//! fallback. Track order: the X6 stores its tracks reversed (track 0 =
//! calibration entry 2), per the file's `stream_type`/track-order fields —
//! verified by matching features across the two streams' overlap through
//! the factory extrinsics.

use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::insta360::{Insta360Meta, InsvLensCalib, InsvStreamGeometry, X6ViewChoice};
use vr180_fisheye::{DjiLensCalib, DjiOsvImu, OmniLensModel};

/// IMU→camera basis for the VR180-modded X6 — rows are the camera axes
/// (x right, y up, z optical) expressed in IMU coordinates. Both lenses
/// share it: after the mod they face the same way, and the two sensors'
/// timelines differ only in timing (`lens_b_timeline`).
///
/// This is the STOCK back-lens measurement yawed 180° about the camera's
/// vertical axis (x and z rows negated). The mount was measured on four
/// pre-mod walking clips against the back lens (stream 0), by fitting the
/// rotation between consecutive frames (exact factory lens model,
/// far-field features) to the gyro's rotation over the same interval: a
/// signed axis permutation plus a 0.6° tilt about the camera's x axis,
/// consistent per clip to 0.3° and with the lens extrinsics in the
/// calibration string — `[[0.0035, 0.0044, -1], [1, 0.0083, 0.0036],
/// [0.0084, -1, -0.0044]]`. The VR180 mod turns that lens through 180°
/// about the body's vertical axis to sit beside the screen lens facing the
/// screen-lens direction, so the frame both lenses now share is the stock
/// one with x and z reversed. Verified on post-mod footage
/// (VID_20260921_195239, 5–8 s, camera-lock): the stock basis left 1.73×
/// the un-stabilised frame-to-frame motion — the correction applied
/// backwards, the "twice as shaky" report — and the yawed basis 0.00×. The
/// 0.6° tilt is about x and survives the yaw unchanged. (An earlier value
/// carried a 4.5° tilt, an artefact of the lens model used at the time.)
/// Re-orthonormalised at use.
pub const INSV_X6_IMU_TO_CAM: [[f32; 3]; 3] = [
    [-0.0035, -0.0044, 1.0000],
    [1.0000, 0.0083, 0.0036],
    [-0.0084, 1.0000, 0.0044],
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
/// working frame holding the stock window (both lenses share it; the two
/// factory entries differ by 0.15 % in focal, below what the measurement
/// resolves). Principal point = frame centre: the firmware re-centres each
/// lens when cropping the 3840² streams (rim fit lands within 2 px of
/// centre on both streams).
pub fn x6_stream_lens_calib(stream_w: u32, stream_h: u32) -> DjiLensCalib {
    x6_profile_lens_calib(stream_w as f64 / (2.0 * INSV_X6_CALIB_SIDE as f64), false, stream_w, stream_h)
}

/// The same fallback profile for a stream at `scale` stream pixels per
/// native pixel (the profile was measured at ½, the 3840² stream); `cropped`
/// marks a frame that holds only part of the image circle.
pub fn x6_profile_lens_calib(scale: f64, cropped: bool, stream_w: u32, stream_h: u32) -> DjiLensCalib {
    let s = (scale * 2.0) as f32;
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
        frame_crop: cropped,
        ..Default::default()
    }
}

/// Side (px) of a per-eye 180° half-equirect canvas that keeps a lens'
/// centre sharpness: such a canvas has `W/π` pixels per radian and the lens
/// `paraxial_px_per_rad` there, so `W = π·f`, rounded up to a multiple of 16.
pub fn native_sharpness_side(paraxial_px_per_rad: f64) -> u32 {
    ((std::f64::consts::PI * paraxial_px_per_rad / 16.0).ceil().max(1.0) as u32) * 16
}

/// Mean centre scale of a clip's two lenses in NATIVE pixels per radian —
/// the factory entries' `fx / (1 + ξ)` (≈ 2087 on the X6), or the generic
/// profile's when the file carries no calibration string.
pub fn x6_paraxial_native(meta: &Insta360Meta) -> f64 {
    if meta.lenses.is_empty() {
        INSV_X6_FX as f64 * 2.0 // the profile was measured on the ½-scale 3840² stream
    } else {
        meta.lenses.iter().map(|l| l.paraxial_focal()).sum::<f64>() / meta.lenses.len() as f64
    }
}

/// Per-eye side of the 180° canvas at a Crop clip's native sharpness: its
/// frame is native sensor pixels covering part of the lens (126° on the 10K
/// Crop), so the 4912² frame put on a 180° canvas of its own size would be
/// scaled down by a quarter. `None` unless the clip's frame is a crop of the
/// image circle — a full-view frame already spans more than 180°.
pub fn crop_native_side(meta: &Insta360Meta, stream_w: u32, stream_h: u32, view: X6ViewChoice) -> Option<u32> {
    let geom = meta.stream_geometry(stream_w, stream_h, view);
    geom.cropped.then(|| native_sharpness_side(x6_paraxial_native(meta) * geom.scale_x))
}

/// Per-file lens calibration for one video stream, in stream coordinates.
///
/// - Frame: `geom` places the stream in the calibration's native frame —
///   the stock 3840² stream is the 7680² window binned 2×2 (principal point
///   `(c − 32)/2` per lens, `fx` halved); a Crop mode is native pixels
///   (`fx` unchanged, principal point `c − origin`) and sets `frame_crop`.
/// - `omni`: the exact factory model (UCM + radial + tangential + prism),
///   which the renderers use.
/// - `fx/k1..k5`: a weighted linear least-squares KB-5 fit of the radial
///   curve `r(θ)` on θ ∈ [0, 106°] (full weight inside 96°) for the override
///   UI seeds and CPU consumers (0.02 px inside 95° at the 3840² scale).
pub fn stream_lens_calib(entry: &InsvLensCalib, geom: &InsvStreamGeometry, stream_w: u32, stream_h: u32) -> DjiLensCalib {
    let (cx, cy) = geom.principal_point(entry);
    let (f, k) = fit_kb5(|th| entry.project_radius(th) * geom.scale_x);
    let omni = OmniLensModel {
        xi: entry.xi,
        fx: (entry.fx as f64 * geom.scale_x) as f32,
        fy: (entry.fy as f64 * geom.scale_y) as f32,
        radial: entry.radial5(),
        tangential: entry.tangential(),
        prism: entry.prism(),
    };
    DjiLensCalib {
        fx: Some(f as f32),
        fy: Some((f * geom.scale_y / geom.scale_x.max(1e-12)) as f32),
        cx: Some(cx as f32),
        cy: Some(cy as f32),
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
        frame_crop: geom.cropped,
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
pub fn insv_calib_only(meta: &Insta360Meta, stream_w: u32, stream_h: u32, view: X6ViewChoice) -> DjiOsvImu {
    // Stream 0 (back lens = RIGHT eye by default = `lens_a`, the DJI
    // convention — see `SourceKind::dual_stream_iter_swap`) carries
    // calibration entry 2 on the X6 — its tracks are stored in reverse lens
    // order (file flag; default when the file doesn't say) — and stream 1
    // (screen lens = `lens_b`) entry 1.
    let reversed = meta.track_order_reversed.unwrap_or(true);
    let (i0, i1) = if reversed { (1, 0) } else { (0, 1) };
    // Where the stream sits in the calibration frame — per recording mode
    // (the file's window record + the encoded size; see `X6Mode`).
    let geom = meta.stream_geometry(stream_w, stream_h, view);
    tracing::info!(
        "insv: {}x{} window {:?} → mode {:?}{} — scale {:.6}, origin ({:.1}, {:.1}), rows span {:.4} of the readout{}",
        stream_w, stream_h, meta.window_crop.map(|w| (w.dst_w, w.dst_h, w.offset_x, w.offset_y)), geom.mode,
        if geom.detected != geom.mode { format!(" (file: {:?}, view {:?})", geom.detected, view) } else { String::new() },
        geom.scale_x, geom.origin_x, geom.origin_y, geom.readout_fraction,
        if geom.cropped { ", frame = crop of the image circle" } else { "" },
    );
    let (lens_a, lens_b) = if meta.lenses.len() >= 2 {
        (stream_lens_calib(&meta.lenses[i0], &geom, stream_w, stream_h),
         stream_lens_calib(&meta.lenses[i1], &geom, stream_w, stream_h))
    } else {
        tracing::warn!("insv: no per-lens calibration string — using the generic X6 profile");
        let c = x6_profile_lens_calib(geom.scale_x, geom.cropped, stream_w, stream_h);
        (c.clone(), c)
    };
    DjiOsvImu {
        lens_a,
        lens_b,
        camera_model: Some(if meta.model.is_empty() { "Insta360".to_string() } else { meta.model.clone() }),
        imu_to_cam: Some(insv_x6_imu_to_cam()),
        // The file's readout covers every row the sensor read; a frame that
        // is the centred part of them spans that share of it.
        readout_ms: meta.readout_ms.map(|v| (v * geom.readout_fraction) as f32).filter(|v| *v > 0.0),
        insv_view: view,
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
    integrate_orientation_samples(&meta.imu, bias_rad_s, tau_s)
}

/// [`integrate_orientation`] over raw samples — shared with every other
/// raw-gyro source (the Kandao QooCam hands its 1 kHz stream in as
/// [`InsvImuSample`]s).
pub fn integrate_orientation_samples(imu: &[vr180_fisheye::insta360::InsvImuSample], bias_rad_s: [f32; 3], tau_s: f64) -> Vec<Quat> {
    const DEG: f64 = std::f64::consts::PI / 180.0;
    let n = imu.len();
    let t: Vec<f64> = imu.iter().map(|s| s.t_us as f64 * 1e-6).collect();
    let w: Vec<[f64; 3]> = imu.iter()
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
        let a = imu[k].acc_g;
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
            let a = imu[k + 1].acc_g;
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
pub(crate) struct QuatTimeline<'a> {
    pub(crate) t_s: Vec<f64>,
    pub(crate) quats: &'a [Quat],
}

impl<'a> QuatTimeline<'a> {
    pub(crate) fn at(&self, t: f64) -> Quat {
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
    view: X6ViewChoice,
) -> DjiOsvImu {
    let mut out = insv_calib_only(meta, stream_w, stream_h, view);
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
    // Stream 0 = `lens_a` = calibration entry 2 = the SECOND sensor's stamps
    // (record 0x000c), so those drive the main timeline; the first sensor's
    // (entry 1, stream 1) become lens B's. A file without the second record
    // has one stamp table serving both eyes and no lens-B timeline.
    let (stamps_a, stamps_b): (&[vr180_fisheye::insta360::InsvFrameStamp],
                               Option<&[vr180_fisheye::insta360::InsvFrameStamp]>) =
        if meta.frames_b.is_empty() { (&meta.frames, None) }
        else { (&meta.frames_b, Some(&meta.frames)) };
    let n_frames = n_frames_hint.max(stamps_a.len());
    if stamps_a.len() != n_frames_hint && n_frames_hint > 0 {
        tracing::info!(
            "insv_imu: {} frame stamps vs {} video frames — extrapolating the tail at 1/fps",
            stamps_a.len(), n_frames_hint
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
    let t0 = build(stamps_a, &mut out);
    if let Some(sb) = stamps_b {
        let mut b = DjiOsvImu {
            lens_a: out.lens_a.clone(),
            lens_b: out.lens_b.clone(),
            camera_model: out.camera_model.clone(),
            imu_to_cam: out.imu_to_cam,
            readout_ms: out.readout_ms,
            insv_view: out.insv_view,
            sample_anchor: vr180_fisheye::SampleAnchor::BlockMid,
            ..Default::default()
        };
        let t0b = build(sb, &mut b);
        let n_diff = stamps_a.iter().zip(sb.iter())
            .filter(|(a, c)| ((a.t_us as f64 - a.exposure_s * 0.5e6) - (c.t_us as f64 - c.exposure_s * 0.5e6)).abs() > 500.0)
            .count();
        tracing::info!(
            "insv_imu: lens B timeline from the first sensor's stamps (entry 1, stream 1; {} frames, {} differ from lens A by > 0.5 ms; frame 0 {:+.2} ms)",
            sb.len(), n_diff, (t0b - t0) * 1e3
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
        // Stays close to the measured values. The y row is the stock
        // measurement; the x and z rows are its 180° yaw (the VR180 mod), so
        // x_cam·z_imu and z_cam·y_imu are +1 where the stock lens had −1.
        assert!((c[1][0] - 1.0).abs() < 0.01 && c[1][1].abs() < 0.02, "{c:?}");
        assert!((c[0][2] - 1.0).abs() < 0.01, "x row not yawed: {:?}", c[0]);
        assert!((c[2][1] - 1.0).abs() < 0.01, "z row not yawed: {:?}", c[2]);
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
        let imu = build_insv_imu(&meta, 50.0, 2161, 3840, 3840, X6ViewChoice::Auto);
        assert_eq!(imu.frame_quats.len(), 2161);
        assert_eq!(imu.high_rate_quats[100].len(), HR_PER_FRAME);
        // Mid sample of the HR block == the frame quat (content time).
        let a = imu.frame_quats[100];
        let b = imu.high_rate_quats[100][HR_PER_FRAME / 2];
        assert!(a.dot(b).abs() > 0.999_99, "mid HR sample must match frame quat");
        assert!(imu.imu_to_cam.is_some());
        // Per-file factory calibration in stream coords (entry 2 → stream 0 → lens_a).
        assert!((imu.readout_ms.unwrap() - 15.36).abs() < 1e-4);
        let b = imu.lens_b_timeline.as_deref().expect("second sensor timeline");
        assert_eq!(b.frame_quats.len(), 2161);
        // Frame 0: lens B (entry 1) mid-exposure is 4.84 ms later than lens A's
        // → a small but non-zero delta.
        let d = imu.frame_quats[0].conjugate().mul(b.frame_quats[0]).normalize();
        let ang = 2.0 * d.w.abs().min(1.0).acos().to_degrees();
        assert!(ang > 0.05 && ang < 2.0, "lens B delta at frame 0 = {ang}°");
        // Reversed track order: stream 0 (lens_a) = entry 2, stream 1 (lens_b) = entry 1.
        let a = &imu.lens_a;
        assert!((a.fx.unwrap() - 1045.66).abs() < 0.3, "fx {:?}", a.fx);
        assert!((a.cx.unwrap() - 1896.55).abs() < 0.01 && (a.cy.unwrap() - 1922.985).abs() < 0.01, "pp {:?} {:?}", a.cx, a.cy);
        let ao = a.omni.unwrap();
        assert!((ao.fx - 7227.07 * 0.5).abs() < 1e-3 && (ao.xi - 2.45543).abs() < 1e-5);
        let b = &imu.lens_b;
        assert!((b.fx.unwrap() - 1044.14).abs() < 0.3, "fx {:?}", b.fx);
        assert!((b.cx.unwrap() - 1911.73).abs() < 0.01 && (b.cy.unwrap() - 1926.265).abs() < 0.01, "pp {:?} {:?}", b.cx, b.cy);
        assert!((b.k1.unwrap() - 0.0895).abs() < 0.001);
        assert_eq!(b.omni.unwrap().tangential, [-0.00032840, 0.00113587, 0.00412989, 0.00653467]);
    }

    /// The four new X6 modes on the real samples (skipped when not mounted):
    /// each stream's calibration is the factory entry moved into the mode's
    /// frame, and the readout is the file's value over the rows encoded.
    #[test]
    fn real_x6_modes_calibrate() {
        let dir = std::path::Path::new("/Volumes/Database/x6/1110");
        // (file, eye side, scale, origin x, origin y, readout ms, cropped)
        let cases: [(&str, u32, f64, f64, f64, f64, bool); 4] = [
            ("VID_20261110_144736_00_003.insv", 3712, 3712.0 / 7680.0, 32.0, 32.0, 14.56, false),
            ("drop frame/VID_20261110_144829_00_004.insv", 3712, 1.0, 2016.0, 2016.0, 16.037333 * 3712.0 / 3968.0, true),
            ("VID_20261110_145518_00_008.insv", 4864, 4864.0 / 7912.0, -80.0, -84.0, 31.648, false),
            ("VID_20261110_145135_00_006.insv", 4912, 1.0, 1416.0, 1416.0, 21.285333, true),
        ];
        for (name, side, scale, ox, oy, readout, cropped) in cases {
            let p = dir.join(name);
            if !p.exists() { continue; }
            let meta = vr180_fisheye::insta360::read_insta360_identity(&p).expect("identity");
            let imu = insv_calib_only(&meta, side, side, X6ViewChoice::Auto);
            // Reversed track order: lens_a = stream 0 = entry 2, lens_b = entry 1.
            for (lens, entry) in [(&imu.lens_a, &meta.lenses[1]), (&imu.lens_b, &meta.lenses[0])] {
                let o = lens.omni.expect("factory model");
                assert!((o.fx as f64 - entry.fx as f64 * scale).abs() < 2e-3, "{name}: fx {}", o.fx);
                let cx = (entry.cx as f64 - entry.half_origin_x() as f64 - ox) * scale;
                let cy = (entry.cy as f64 - oy) * scale;
                assert!((lens.cx.unwrap() as f64 - cx).abs() < 2e-3 && (lens.cy.unwrap() as f64 - cy).abs() < 2e-3,
                    "{name}: pp {:?} {:?} want ({cx}, {cy})", lens.cx, lens.cy);
                // Every mode is centred on the sensor: the optical axis
                // lands within 2 % of the frame centre.
                let half = side as f32 * 0.5;
                assert!((lens.cx.unwrap() - half).abs() < 0.02 * side as f32 && (lens.cy.unwrap() - half).abs() < 0.02 * side as f32,
                    "{name}: pp {:?} {:?}", lens.cx, lens.cy);
                assert_eq!(lens.frame_crop, cropped, "{name}");
                assert_eq!((lens.width, lens.height), (Some(side as f32), Some(side as f32)));
                // KB seed for the override UI follows the same scale.
                assert!((lens.fx.unwrap() as f64 / (entry.paraxial_focal() * scale) - 1.0).abs() < 0.01, "{name}: kb fx {:?}", lens.fx);
            }
            assert!((imu.readout_ms.unwrap() as f64 - readout).abs() < 1e-3, "{name}: readout {:?}", imu.readout_ms);
        }
    }

    /// A factory 8K clip still takes the original path: the 7680² window
    /// binned 2×2 — `(c − 32)/2`, `fx/2`, the file's readout, no crop flag.
    /// Skipped when not mounted.
    #[test]
    fn real_x6_stock_clip_is_unchanged() {
        let p = std::path::Path::new("/Volumes/Database/x6/1106/VID_20261106_203704_00_015.insv");
        if !p.exists() { return; }
        let meta = vr180_fisheye::insta360::read_insta360_identity(p).expect("identity");
        let g = meta.stream_geometry(3840, 3840, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.scale_x, g.origin_x, g.origin_y, g.readout_fraction, g.cropped),
                   (vr180_fisheye::X6Mode::Stock, 0.5, 32.0, 32.0, 1.0, false));
        // The view choice is ignored by a clip the file identifies.
        assert_eq!(meta.stream_geometry(3840, 3840, X6ViewChoice::Crop), g);
        let imu = insv_calib_only(&meta, 3840, 3840, X6ViewChoice::Auto);
        let (a, e2) = (&imu.lens_a, &meta.lenses[1]);
        assert!((a.cx.unwrap() - (e2.cx - 7744.0 - 32.0) / 2.0).abs() < 2e-3 && (a.cy.unwrap() - (e2.cy - 32.0) / 2.0).abs() < 2e-3);
        assert!((a.omni.unwrap().fx - e2.fx / 2.0).abs() < 2e-3 && !a.frame_crop);
        assert!((imu.readout_ms.unwrap() - 14.56).abs() < 1e-4);
    }

    #[test]
    fn native_sharpness_canvas() {
        // π·f rounded up to a multiple of 16.
        assert_eq!(native_sharpness_side(2087.0), 6560);
        assert_eq!(native_sharpness_side(16.0 / std::f64::consts::PI), 16);
        assert_eq!(native_sharpness_side(16.01 / std::f64::consts::PI), 32);
        let lenses = vr180_fisheye::insta360::parse_calib_string(
            "2_2.455430_7222.060_7222.060_3859.700_3880.160_0.268_0.029_89.599_0.000000_0.000000_0.000000_1.28201866_-0.80810684_1.14986908_12.02910519_0.00000000_0.00083911_0.00081532_-0.00902508_0.00378748_-0.00103918_-0.00197197_0.00374385_-0.03644488_15488_7744_193_2.455430_7200.680_7200.680_11623.400_3877.740_0.077_-0.047_89.595_-0.000117_0.000083_-0.032567_1.27704287_-0.69810301_-0.66506159_19.39754486_0.00000000_0.00045809_0.00208752_0.00392174_0.00178436_0.00026127_-0.00205885_-0.01210092_-0.01896361_15488_7744_193_394240").unwrap();
        let win = |dw, dh| vr180_fisheye::InsvWindowCrop { src_w: 7744, src_h: 7744, dst_w: dw, dst_h: dh, offset_x: 0, offset_y: 0 };
        let mut meta = Insta360Meta { lenses, ..Default::default() };
        assert!((x6_paraxial_native(&meta) - 2086.97).abs() < 0.05);
        // Both Crop modes are native pixels: the same density, the same canvas.
        meta.window_crop = Some(win(4912, 4912));
        assert_eq!(crop_native_side(&meta, 4912, 4912, X6ViewChoice::Auto), Some(6560));
        meta.window_crop = Some(win(4912, 3968));
        assert_eq!(crop_native_side(&meta, 3712, 3712, X6ViewChoice::Auto), Some(6560));
        // A full-view frame has no such size …
        meta.window_crop = Some(win(7680, 7680));
        assert_eq!(crop_native_side(&meta, 3712, 3712, X6ViewChoice::Auto), None);
        assert_eq!(crop_native_side(&meta, 3840, 3840, X6ViewChoice::Auto), None);
        // … and an unconfirmed 3712² clip gets it only once the user says Crop.
        meta.window_crop = None;
        assert_eq!(crop_native_side(&meta, 3712, 3712, X6ViewChoice::Auto), None);
        assert_eq!(crop_native_side(&meta, 3712, 3712, X6ViewChoice::Crop), Some(6560));
        // No calibration string: the generic profile's centre scale (2088.8 px/rad).
        let bare = Insta360Meta::default();
        assert!((x6_paraxial_native(&bare) - 2088.78).abs() < 0.05);
        assert_eq!(crop_native_side(&bare, 4912, 4912, X6ViewChoice::Auto), Some(6576));
    }

    #[test]
    fn kb5_fit_reproduces_factory_curve() {
        let s = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193";
        let entry = &vr180_fisheye::insta360::parse_calib_string(s).unwrap()[0];
        let c = stream_lens_calib(entry, &InsvStreamGeometry::scaled_window(&vr180_fisheye::insta360::X6_DEFAULT_WINDOW, 3840, 3840), 3840, 3840);
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
