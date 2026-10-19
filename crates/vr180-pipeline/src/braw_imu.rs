//! Blackmagic RAW: the clip's gyro / accelerometer stream + metadata → the
//! stabilizer's [`DjiOsvImu`], the way `kandao_imu` does it for the QooCam.
//!
//! ## What the file gives and what it does not
//!
//! A BRAW clip carries a raw gyro + accelerometer stream (1022 Hz on the
//! PYXIS 12K; `IBlackmagicRawClipGyroscopeMotion`, rad/s and m/s², no
//! timestamps — sample 0 is the start of frame 0 and the rate is the
//! clock), the sensor's line time (`sensor_line_time`, µs; readout =
//! rows × line time: 19.44 ms for the 12288×6480 window, 15.34 ms for
//! 12288×5112), and a per-frame shutter (`shutter_value`, "7.2°" or
//! "1/535" — the camera's auto exposure moves it within a take, 0.4–3.3 ms
//! on the sample clip, so it is sampled per frame rather than read once).
//! It carries NO lens calibration (every `lens_*` key is empty on a
//! third-party dual-fisheye lens): the lens is the user's manual entry in
//! the Override UI, seeded from the image circles the app finds in a
//! decoded frame ([`crate::circles`]) and the camera preset's FOV / k.
//!
//! ## Motion
//!
//! The raw stream is fused exactly as the X6's and the QooCam's
//! (`integrate_orientation_samples`: trapezoid gyro integration with a
//! whole-clip VQF bias estimate, batch gravity alignment, the slow
//! inclination trim). Per-frame content time = the centre row's
//! mid-exposure: `i / fps + readout / 2 − shutter(i) / 2 + latency`
//! ([`BRAW_GYRO_LATENCY_MS`]; `VR180_BRAW_LATENCY_MS` overrides it for the
//! probe). The IMU → camera basis ([`BRAW_IMU_TO_CAM`], rows = camera
//! x right / y up / z in IMU coordinates; `VR180_BRAW_BASIS` overrides) is
//! settled the way the others were: uncapped camera-lock exports of a
//! clip window under each candidate, the one that leaves the least
//! residual motion wins.

use vr180_core::gyro::cori_iori::Quat;
use vr180_fisheye::insta360::InsvImuSample;
use vr180_fisheye::{DjiLensCalib, DjiOsvImu};

use crate::insv_imu::{integrate_orientation_samples, HR_PER_FRAME, INSV_TAU_ACC_S};

/// IMU→camera basis for a Blackmagic camera with a forward-facing VR180
/// lens — rows are the camera axes (x right, y up, z) in IMU coordinates,
/// the X6 / QooCam convention: a 180° turn about y. Measured on the PYXIS
/// 12K sample (2026-10-19): camera-locked exports of a 2 s handheld window
/// under all 48 axis-aligned candidates (`VR180_BRAW_BASIS`), residual
/// inter-frame motion of the output at 512 px — this one 0.38 px against
/// 1.53 with no stabilization and 2.02 for the identity; every other
/// candidate ≥ 1.16. `diag(1, −1, 1)` scores identically (a basis and its
/// negative conjugate the same way), and is the Python app's old
/// `C_IMU_TO_CAM`; the proper rotation is kept.
pub const BRAW_IMU_TO_CAM: [[f32; 3]; 3] = [
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
];

/// Gyro delay relative to `i / fps + readout/2 − shutter/2`, ms — the
/// residual once the sensor's own timing is accounted for. Measured on
/// the PYXIS 12K sample (camera-locked 2 s exports, residual inter-frame
/// motion at 512 px, `VR180_BRAW_LATENCY_MS` swept −12…+12): a shallow
/// minimum at +3…+5 ms on one window (0.356–0.360 vs 0.380 at 0 and 0.58
/// at −12) and +1…+5 on another (0.134 flat), so +4. The readout itself
/// (the file's 3 µs line time) could not be checked the same way: the
/// RS-scale sweep is flat on this handheld footage (0.354–0.360 for
/// 0×…2×) — no fast pans in it.
pub const BRAW_GYRO_LATENCY_MS: f32 = 4.0;

/// The camera-preset name the BRAW kind resolves to.
pub const BRAW_PRESET_NAME: &str = "Blackmagic Pyxis 12K";

/// `VR180_BRAW_BASIS="r00,r01,…,r22"` (row-major) overrides the basis for
/// headless A/B tests. Rows are normalised but NOT re-orthogonalised, so a
/// reflected candidate (det −1) is tested as given.
pub fn braw_imu_to_cam() -> [[f32; 3]; 3] {
    let m = std::env::var("VR180_BRAW_BASIS").ok()
        .and_then(|v| {
            let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (f.len() == 9).then(|| [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]])
        })
        .unwrap_or(BRAW_IMU_TO_CAM);
    let norm = |v: [f32; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / n, v[1] / n, v[2] / n]
    };
    [norm(m[0]), norm(m[1]), norm(m[2])]
}

/// Frames between shutter samples: every half second — the camera's auto
/// exposure ramps slowly, and reading a frame's metadata means reading the
/// frame (~10 ms each in one helper process).
pub fn shutter_sample_step(fps: f64) -> u64 {
    ((fps * 0.5).round() as u64).max(1)
}

pub fn braw_gyro_latency_ms() -> f32 {
    std::env::var("VR180_BRAW_LATENCY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(BRAW_GYRO_LATENCY_MS)
}

/// A detected image circle of one eye, in that eye's pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeCircle { pub cx: f32, pub cy: f32, pub r: f32 }

/// How a single-track BRAW frame holds its eyes, and where the image
/// circles sit — read off a quarter-size decode of the first frame, once
/// per clip ([`braw_layout`]). A dual-fisheye VR180 lens puts two circles
/// side by side; a single fisheye (the "single-camera" presets) puts one
/// in the middle, and that frame is then one eye shown to both.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BrawLayout {
    /// One circle for the whole frame: the frame is ONE eye (duplicated).
    pub mono: bool,
    /// `[left, right]` circles in native eye pixels (`mono`: the one circle
    /// twice), when found.
    pub circles: Option<[EyeCircle; 2]>,
}

impl BrawLayout {
    /// Size of one eye for this layout.
    pub fn eye_dims(&self, info: &vr180_braw::BrawInfo) -> (u32, u32) {
        if self.mono { info.frame_dims() } else { info.eye_dims() }
    }
}

/// Detect the layout + circles of a clip's first frame (a quarter-size
/// decode: ~10 px on a 6144-px eye). Two circles that each fill their half
/// → side by side; one circle centred on the frame → mono; nothing found →
/// side by side with no seed (the VR180 default).
pub fn detect_braw_layout(path: &std::path::Path, info: &vr180_braw::BrawInfo) -> BrawLayout {
    let opts = vr180_braw::decoder::DecodeOptions {
        scale: vr180_braw::DecodeScale::Quarter, count: Some(1), ..Default::default()
    };
    let mut dec = match vr180_braw::BrawDecoder::start(path, &opts) {
        Ok(d) => d,
        Err(e) => { tracing::warn!("braw layout: decode start failed: {e}"); return BrawLayout::default(); }
    };
    let mut buf = Vec::new();
    match dec.next_frame(&mut buf) {
        Ok(true) => {}
        other => { tracing::warn!("braw layout: no frame ({other:?})"); return BrawLayout::default(); }
    }
    let h = dec.header().clone();
    let (fw, _) = info.frame_dims();
    let scale = fw as f32 / h.width.max(1) as f32;
    let bpp = h.bytes_per_pixel();
    // BGRA in, but the detector only takes max(R, G, B) — order-free.
    let sbs = crate::circles::detect_sbs_circles(&buf, h.width, h.height, bpp, scale);
    // A single centred circle wider than a half (a mono fisheye, or a
    // two-track clip is never mono) — only for single-track frames.
    let whole = if info.is_dual_track() { None } else {
        crate::circles::detect_eye_circle(&buf, h.width, h.height, bpp, 0, h.width)
            .filter(|c| (c.cx - h.width as f32 * 0.5).abs() < h.width as f32 * 0.06
                && (c.cy - h.height as f32 * 0.5).abs() < h.height as f32 * 0.1
                && c.r > h.width as f32 * 0.3)
            .map(|c| EyeCircle { cx: c.cx * scale, cy: c.cy * scale, r: c.r * scale })
    };
    match (sbs, whole) {
        (Some([l, r]), _) => {
            tracing::info!("braw layout: side by side — left ({:.0}, {:.0}) r {:.0} | right ({:.0}, {:.0}) r {:.0} (native eye px)", l.cx, l.cy, l.r, r.cx, r.cy, r.r);
            BrawLayout { mono: false, circles: Some([l, r]) }
        }
        (None, Some(c)) => {
            tracing::info!("braw layout: ONE centred circle ({:.0}, {:.0}) r {:.0} — a single fisheye, the frame is one eye", c.cx, c.cy, c.r);
            BrawLayout { mono: true, circles: Some([c, c]) }
        }
        (None, None) => {
            tracing::info!("braw layout: no circles found — side by side, seeding the eye centres");
            BrawLayout::default()
        }
    }
}

/// [`detect_braw_layout`] memoised per clip path for the process: the
/// loader, the export, the still and auto-align all read one answer.
pub fn braw_layout(path: &std::path::Path, info: &vr180_braw::BrawInfo) -> BrawLayout {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<std::path::PathBuf, BrawLayout>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(l) = cache.lock().unwrap().get(path) { return *l; }
    let l = detect_braw_layout(path, info);
    cache.lock().unwrap().insert(path.to_path_buf(), l);
    l
}

/// The focal factor and `k` a selected camera preset applies on top of the
/// circle seed (which is built with [`BRAW_PRESET_NAME`]'s FOV / k):
/// `fx × (seed_half_fov / preset_half_fov)` and the preset's `k` when it is
/// another preset; `(1, None)` for the default (and for presets without a
/// FOV). Both calibration resolvers apply it identically.
pub fn preset_rescale(preset: &vr180_fisheye::presets::CameraPreset) -> (f32, Option<[f32; 4]>) {
    if preset.name == BRAW_PRESET_NAME || preset.default_fov_deg <= 0.0 { return (1.0, None); }
    let seed_fov = vr180_fisheye::presets::find(BRAW_PRESET_NAME).map(|p| p.default_fov_deg as f32).unwrap_or(190.0);
    let sx = (seed_fov.max(1.0) * 0.5) / ((preset.default_fov_deg as f32).max(1.0) * 0.5);
    (sx, Some([preset.calib.k[0] as f32, preset.calib.k[1] as f32, preset.calib.k[2] as f32, preset.calib.k[3] as f32]))
}

/// The seed lens for one eye: principal point at the detected circle (or
/// the eye centre), focal from the circle radius and the preset's FOV (an
/// equidistant lens: `r = f · θ`), the preset's `k`; `k5 = p = 0`. The user
/// refines all of it in the Override UI — a Blackmagic file has no lens
/// record of its own.
pub fn seed_lens_calib(eye_w: u32, eye_h: u32, circle: Option<EyeCircle>) -> DjiLensCalib {
    let preset = vr180_fisheye::presets::find(BRAW_PRESET_NAME);
    let fov_deg = preset.map(|p| p.default_fov_deg as f32).unwrap_or(190.0);
    let k = preset
        .map(|p| [p.calib.k[0] as f32, p.calib.k[1] as f32, p.calib.k[2] as f32, p.calib.k[3] as f32])
        .unwrap_or([0.0; 4]);
    let half_fov = fov_deg.max(1.0).to_radians() * 0.5;
    let (cx, cy, fx) = match circle {
        Some(c) if c.r > 1.0 => (c.cx, c.cy, c.r / half_fov),
        _ => (eye_w as f32 * 0.5, eye_h as f32 * 0.5, (eye_w.min(eye_h) as f32 * 0.5) / half_fov),
    };
    DjiLensCalib {
        fx: Some(fx), fy: Some(fx), cx: Some(cx), cy: Some(cy),
        k1: Some(k[0]), k2: Some(k[1]), k3: Some(k[2]), k4: Some(k[3]),
        k5: Some(0.0), p1: Some(0.0), p2: Some(0.0),
        width: Some(eye_w as f32), height: Some(eye_h as f32),
        omni: None,
        ..Default::default()
    }
}

/// `(lens_a, lens_b)` for the stabilizer: `lens_b` = the LEFT eye (the left
/// half / track 0, the resolvers' default with the swap off), `lens_a` =
/// the RIGHT. `circles` = `[left, right]` when detected (`eye_w × eye_h` =
/// the layout's eye, the whole frame for a mono clip).
pub fn lens_pair(eye_w: u32, eye_h: u32, circles: Option<[EyeCircle; 2]>) -> (DjiLensCalib, DjiLensCalib) {
    let (l, r) = match circles { Some([l, r]) => (Some(l), Some(r)), None => (None, None) };
    (seed_lens_calib(eye_w, eye_h, r), seed_lens_calib(eye_w, eye_h, l))
}

/// Calibration-only IMU (no quats): enough to dewarp and seed the Override
/// UI; stabilization stays identity.
pub fn braw_calib_only(info: &vr180_braw::BrawInfo, eye_w: u32, eye_h: u32, circles: Option<[EyeCircle; 2]>) -> DjiOsvImu {
    let (lens_a, lens_b) = lens_pair(eye_w, eye_h, circles);
    DjiOsvImu {
        lens_a,
        lens_b,
        camera_model: Some(if info.camera_model.is_empty() { "Blackmagic RAW".to_string() } else { info.camera_model.clone() }),
        imu_to_cam: Some(braw_imu_to_cam()),
        readout_ms: info.readout_s().map(|s| (s * 1e3) as f32),
        sample_anchor: vr180_fisheye::SampleAnchor::BlockMid,
        ..Default::default()
    }
}

/// The stabilizer's per-frame orientation stream from the clip's gyro.
/// `n_frames_hint` is the video's frame count (`BrawInfo::frame_count`);
/// `shutters` the sampled per-frame shutter (a hold lookup; empty → the
/// `--info` sample, else 0). Returns the calibration-only structure when
/// the gyro is unusable.
pub fn build_braw_imu(
    info: &vr180_braw::BrawInfo,
    gyro: &vr180_braw::BrawGyroData,
    shutters: &vr180_braw::BrawShutters,
    fps: f32,
    n_frames_hint: usize,
    eye_w: u32,
    eye_h: u32,
    circles: Option<[EyeCircle; 2]>,
) -> DjiOsvImu {
    let mut out = braw_calib_only(info, eye_w, eye_h, circles);
    let n_imu = gyro.len();
    let rate = gyro.header.gyro_sample_rate;
    if n_imu < 16 || rate <= 0.0 || fps <= 0.0 {
        tracing::warn!("braw_imu: no usable motion data (samples={} rate={} fps={}) — calib only", n_imu, rate, fps);
        return out;
    }
    // The helper pairs gyro and accelerometer samples by INDEX; a clip whose
    // two sensors run at different rates would put the accelerometer on a
    // stretched clock inside the fusion — refuse rather than fuse wrongly.
    let acc_rate = gyro.header.accel_sample_rate;
    if (acc_rate - rate).abs() > 0.5 || gyro.header.gyro_sample_count != gyro.header.accel_sample_count {
        tracing::warn!("braw_imu: gyro {rate} Hz × {} / accel {acc_rate} Hz × {} — not index-paired, calib only",
            gyro.header.gyro_sample_count, gyro.header.accel_sample_count);
        return out;
    }
    if info.readout_s().is_none() {
        tracing::warn!("braw_imu: the clip states no sensor_line_time — content time at the first row, per-row RS off");
    }
    let dt = (1.0 / rate) as f32;
    // The SDK stream: rad/s and m/s², sample k at k / rate from the clip start.
    const DEG: f32 = 180.0 / std::f32::consts::PI;
    const G: f32 = 9.80665;
    let samples: Vec<InsvImuSample> = (0..n_imu).map(|k| InsvImuSample {
        t_us: (k as f64 * 1e6 / rate).round() as u64,
        acc_g: [gyro.accel[k * 3] / G, gyro.accel[k * 3 + 1] / G, gyro.accel[k * 3 + 2] / G],
        gyr_dps: [gyro.gyro[k * 3] * DEG, gyro.gyro[k * 3 + 1] * DEG, gyro.gyro[k * 3 + 2] * DEG],
    }).collect();
    let gyr: Vec<[f32; 3]> = (0..n_imu).map(|k| [gyro.gyro[k * 3], gyro.gyro[k * 3 + 1], gyro.gyro[k * 3 + 2]]).collect();
    let acc: Vec<[f32; 3]> = (0..n_imu).map(|k| [gyro.accel[k * 3], gyro.accel[k * 3 + 1], gyro.accel[k * 3 + 2]]).collect();
    let t0 = std::time::Instant::now();
    let pass1 = vr180_core::gyro::vqf::run(&gyr, &acc, None, dt);
    let quats = integrate_orientation_samples(&samples, pass1.bias_rad_s, INSV_TAU_ACC_S);
    tracing::info!(
        "braw_imu: gyro integration over {} samples @ {:.1} Hz in {:.1?}, bias = {:?} °/s",
        n_imu, rate, t0.elapsed(), pass1.bias_deg_s()
    );
    let timeline = crate::insv_imu::QuatTimeline {
        t_s: samples.iter().map(|s| s.t_us as f64 * 1e-6).collect(),
        quats: &quats,
    };
    let readout_s = info.readout_s().unwrap_or(0.0);
    let latency_s = braw_gyro_latency_ms() as f64 * 1e-3;
    let frame_dur = 1.0 / fps as f64;
    let fallback_shutter = if info.shutter_s_min > 0.0 { 0.5 * (info.shutter_s_min + info.shutter_s_max) } else { 0.0 };
    let shutter = |i: usize| -> f64 { shutters.at(i as u64).unwrap_or(fallback_shutter) };
    // Centre row mid-exposure: frame i's first row starts reading out at
    // i / fps, the centre row readout/2 later, and its exposure ended
    // shutter/2 before its own readout.
    let content_time = |i: usize| -> f64 { i as f64 * frame_dur + readout_s * 0.5 - shutter(i) * 0.5 + latency_s };
    let n_frames = n_frames_hint.max(1);
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
    out.exposure_ms = (0..n_frames).map(|i| (shutter(i) * 1e3) as f32).collect();
    tracing::info!(
        "braw_imu: {} frames @ {:.3} fps, readout {:.2} ms, shutter {:.2}…{:.2} ms ({} sampled), latency {:.1} ms, imu span {:.3}..{:.3} s, basis {:?}",
        n_frames, fps, readout_s * 1e3, info.shutter_s_min * 1e3, info.shutter_s_max * 1e3, shutters.samples.len(),
        latency_s * 1e3, timeline.t_s[0], timeline.t_s[timeline.t_s.len() - 1], out.imu_to_cam
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> vr180_braw::BrawInfo {
        serde_json::from_str(r#"{"width":12288,"height":6480,"frame_count":100,"frame_rate":50.0,"duration":2.0,
            "camera_model":"Blackmagic PYXIS 12K","gyro_sample_rate":1022.0,"readout_ms":19.44,"sensor_line_time_us":3.0,
            "shutter_s_min":0.0004,"shutter_s_max":0.0033}"#).unwrap()
    }

    #[test]
    fn seed_lens_from_circle_and_preset() {
        let c = EyeCircle { cx: 2910.0, cy: 3247.0, r: 3183.0 };
        let l = seed_lens_calib(6144, 6480, Some(c));
        assert_eq!(l.cx, Some(2910.0));
        assert_eq!(l.cy, Some(3247.0));
        let fov = vr180_fisheye::presets::find(BRAW_PRESET_NAME).map(|p| p.default_fov_deg as f32).unwrap_or(190.0);
        assert!((l.fx.unwrap() - 3183.0 / (fov.to_radians() * 0.5)).abs() < 1e-2);
        let (a, b) = lens_pair(6144, 6480, Some([c, EyeCircle { cx: 3170.0, cy: 3187.0, r: 3178.0 }]));
        assert_eq!(b.cx, Some(2910.0));   // lens_b = left
        assert_eq!(a.cx, Some(3170.0));
        let n = seed_lens_calib(6144, 6480, None);
        assert_eq!(n.cx, Some(3072.0));
    }

    #[test]
    fn stream_from_gyro_and_shutters() {
        let rate = 1022.0f64;
        let n = (rate * 2.5) as usize;
        let header: vr180_braw::BrawGyroHeader = serde_json::from_str(&format!(
            r#"{{"gyro_sample_count":{n},"gyro_sample_rate":{rate},"accel_sample_count":{n},"accel_sample_rate":{rate},"sample_count":{n},"frame_rate":50.0}}"#)).unwrap();
        let mut accel = vec![0f32; n * 3];
        for k in 0..n { accel[k * 3 + 1] = -9.80665; }
        let gyro = vr180_braw::BrawGyroData { header, gyro: vec![0f32; n * 3], accel };
        let sh = vr180_braw::BrawShutters { samples: vec![(0, 0.001), (50, 0.002)] };
        let imu = build_braw_imu(&info(), &gyro, &sh, 50.0, 100, 6144, 6480, None);
        assert_eq!(imu.frame_quats.len(), 100);
        assert_eq!(imu.high_rate_quats[0].len(), HR_PER_FRAME);
        assert!((imu.readout_ms.unwrap() - 19.44).abs() < 1e-3);
        assert!((imu.exposure_ms[10] - 1.0).abs() < 1e-4 && (imu.exposure_ms[60] - 2.0).abs() < 1e-4);
        assert_eq!(imu.sample_anchor, vr180_fisheye::SampleAnchor::BlockMid);
        let q0 = imu.frame_quats[0]; let q9 = imu.frame_quats[99];
        assert!((q0.w - q9.w).abs() < 1e-4 && (q0.x - q9.x).abs() < 1e-4);
        let calib = braw_calib_only(&info(), 6144, 6480, None);
        assert!(calib.frame_quats.is_empty());
        assert_eq!(calib.imu_to_cam, Some(BRAW_IMU_TO_CAM));
    }
}
