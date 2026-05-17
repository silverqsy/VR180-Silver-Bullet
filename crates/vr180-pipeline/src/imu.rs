//! High-level IMU input prep — file path → arrays VQF can consume.
//!
//! Pipeline (matches `parse_gyro_raw.py::vqf_to_cori_quats`):
//!
//! 1. Extract the GPMF data stream via [`crate::decode::extract_gpmf_stream`].
//! 2. Parse raw GYRO/ACCL/GRAV/MNOR blocks via `vr180_core::gyro::raw`.
//! 3. Flatten gyro samples, applying the ORIN="ZXY" → body-frame axis
//!    remap (`col0 = raw[1], col1 = raw[2], col2 = raw[0]`).
//! 4. Pick accelerometer source: `GRAV × 9.81` if the gravity vector
//!    has plausible magnitude, otherwise raw `ACCL`.
//! 5. Resample acc (and mag, if MNOR present) to gyro rate by
//!    nearest-neighbor proportional indexing — same algorithm Python uses.
//!
//! Output is consumed by `vr180_core::gyro::vqf::run`.

use crate::{Error, Result};
use std::path::Path;
use vr180_core::gyro::raw::{parse_raw_imu, RawImu, ImuBlock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccSource { Grav, Raw, None }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagSource { Mnor, None }

/// Sample arrays ready to be fed to `vr180_core::gyro::vqf::run`.
#[derive(Debug)]
pub struct PreparedImu {
    /// Per-sample 3-axis angular velocity, **rad/s, body frame**.
    pub gyro_body: Vec<[f32; 3]>,
    /// Per-sample 3-axis acceleration, **m/s², body frame**, resampled to gyro rate.
    pub acc_body: Vec<[f32; 3]>,
    /// Optional per-sample 3-axis magnetic-north vector, resampled to gyro rate.
    pub mag_body: Option<Vec<[f32; 3]>>,
    /// Gyro sample interval in seconds (e.g. `~0.00125` for 800 Hz).
    pub gyr_ts: f32,
    pub acc_source: AccSource,
    pub mag_source: MagSource,
    /// Raw counts pre-resample for the printout (acc, mag).
    pub n_acc_input: usize,
    pub n_mag_input: usize,
}

/// Build [`PreparedImu`] from a .360 file. Equivalent to the input-prep
/// half of `vqf_to_cori_quats` in Python.
pub fn prepare_for_vqf(path: &Path) -> Result<PreparedImu> {
    use crate::decode::{extract_gpmf_stream, probe_video};
    let gpmf = extract_gpmf_stream(path)?;
    let probe = probe_video(path)?;
    let raw = parse_raw_imu(&gpmf);
    if raw.gyro.is_empty() {
        return Err(Error::Ffmpeg("no GYRO blocks in GPMF stream".into()));
    }

    // 3a. Flatten gyro with ZXY → body-frame axis remap.
    // GoPro ORIN="ZXY": raw[0]=Z, raw[1]=X, raw[2]=Y.
    // Body frame: bodyX←raw[1], bodyY←raw[2], bodyZ←raw[0].
    let mut gyro_body: Vec<[f32; 3]> = Vec::with_capacity(raw.total(|r| &r.gyro));
    for blk in &raw.gyro {
        for s in &blk.samples {
            gyro_body.push([s[1], s[2], s[0]]);
        }
    }

    let duration = probe.duration_sec as f32;
    let gyr_ts = duration / gyro_body.len() as f32;

    // 4. Pick acc source: GRAV × 9.81 (already filtered by GoPro firmware,
    //    cleaner than raw ACCL) if gravity magnitude is plausible.
    let (acc_body, acc_source, n_acc_input) =
        pick_acc_source(&raw, gyro_body.len());

    // 5. Pick mag source: MNOR (firmware-calibrated magnetic north).
    let (mag_body, mag_source, n_mag_input) =
        pick_mag_source(&raw, gyro_body.len());

    Ok(PreparedImu {
        gyro_body,
        acc_body,
        mag_body,
        gyr_ts,
        acc_source,
        mag_source,
        n_acc_input,
        n_mag_input,
    })
}

/// Resample a (N×3) sequence of samples to `target_len` by
/// proportional nearest-neighbor indexing.
///
/// Equivalent to Python's
/// `idx = min(int(t / src_dt), len(src) - 1)` where
/// `t = i * target_dt` — algebraically `idx = (i * src_len) / target_len`.
fn resample_proportional(src: &[[f32; 3]], target_len: usize) -> Vec<[f32; 3]> {
    if src.is_empty() {
        return vec![[0.0; 3]; target_len];
    }
    let src_len = src.len();
    (0..target_len)
        .map(|i| src[(i * src_len / target_len).min(src_len - 1)])
        .collect()
}

fn flatten_with_remap(blocks: &[ImuBlock], remap: [usize; 3]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(blocks.iter().map(|b| b.samples.len()).sum());
    for b in blocks {
        for s in &b.samples {
            out.push([s[remap[0]], s[remap[1]], s[remap[2]]]);
        }
    }
    out
}

fn pick_acc_source(raw: &RawImu, gyro_len: usize)
    -> (Vec<[f32; 3]>, AccSource, usize)
{
    // GRAV uses axis map (0, 2, 1) per project memory; magnitude in g
    // (multiply by 9.81 to get m/s²).
    let grav_body = flatten_with_remap(&raw.grav, [0, 2, 1]);
    if !grav_body.is_empty() {
        let n = grav_body.len().min(30);
        let mut sum = [0.0_f32; 3];
        for s in &grav_body[..n] {
            for j in 0..3 { sum[j] += s[j]; }
        }
        let mean = [sum[0] / n as f32, sum[1] / n as f32, sum[2] / n as f32];
        let mag = (mean[0]*mean[0] + mean[1]*mean[1] + mean[2]*mean[2]).sqrt();
        if mag > 0.1 {
            let mut scaled: Vec<[f32; 3]> = grav_body.iter()
                .map(|s| [s[0] * 9.81, s[1] * 9.81, s[2] * 9.81])
                .collect();
            let n_input = scaled.len();
            scaled = resample_proportional(&scaled, gyro_len);
            return (scaled, AccSource::Grav, n_input);
        }
    }
    // Fallback: raw ACCL with same axis map as gyro (1, 2, 0).
    let accl_body = flatten_with_remap(&raw.accl, [1, 2, 0]);
    if !accl_body.is_empty() {
        let n_input = accl_body.len();
        let resampled = resample_proportional(&accl_body, gyro_len);
        return (resampled, AccSource::Raw, n_input);
    }
    (vec![[0.0; 3]; gyro_len], AccSource::None, 0)
}

fn pick_mag_source(raw: &RawImu, gyro_len: usize)
    -> (Option<Vec<[f32; 3]>>, MagSource, usize)
{
    // MNOR — Python doesn't define an explicit axis map, so we pass it
    // through verbatim (raw on-disk axes). If validation reveals a remap
    // is needed we'll add one here.
    if raw.mnor.is_empty() {
        return (None, MagSource::None, 0);
    }
    let mnor_body: Vec<[f32; 3]> = raw.mnor.iter()
        .flat_map(|b| b.samples.iter().copied())
        .collect();
    let n_input = mnor_body.len();
    let resampled = resample_proportional(&mnor_body, gyro_len);
    (Some(resampled), MagSource::Mnor, n_input)
}
