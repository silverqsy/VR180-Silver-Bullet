//! Raw IMU block extraction from a GPMF byte stream.
//!
//! Where [`super::cori_iori`] handles the cooked CORI/IORI quaternions
//! GoPro firmware emits, this module handles the **raw** sensor blocks:
//!
//! - **GYRO** — 3-axis angular velocity (rad/s after SCAL division), ~800 Hz
//! - **ACCL** — 3-axis linear acceleration (m/s² after SCAL), ~200 Hz
//! - **GRAV** — 3-axis gravity vector (g, post-firmware-fusion), ~30 Hz
//! - **MNOR** — 3-axis magnetic-north reference (firmware-calibrated), ~60 Hz
//!
//! Each block carries an STMP (uint32/uint64 microsecond timestamp) and is
//! preceded by a SCAL entry whose value(s) divide the raw int16 samples to
//! recover physical units. The parser tracks `current_scal` and `current_stmp`
//! across STRM boundaries, matching the reference implementation in
//! `parse_gyro_raw.py::parse_gyro_accl_full`.
//!
//! Output shape mirrors the Python `gyro_blocks` / `accl_blocks` / etc. lists
//! so we can validate by aggregating sample counts + STMP-ordering against
//! the Python reference.

use super::gpmf::{FourCC, GpmfWalker};

/// One block of N×3 sensor samples.
#[derive(Debug, Clone)]
pub struct ImuBlock {
    /// Sample axes in **raw on-disk order** — no axis remap applied here.
    /// The remap from sensor frame to body frame is the consumer's job
    /// (see `vr180-pipeline::imu`).
    pub samples: Vec<[f32; 3]>,
    /// SCAL divisor that was in effect for this block (1.0 if absent).
    pub scal: f32,
    /// Stream timestamp in microseconds, if present.
    pub stmp_us: Option<u64>,
}

/// Aggregated raw IMU output of one GPMF stream walk.
#[derive(Debug, Clone, Default)]
pub struct RawImu {
    pub gyro: Vec<ImuBlock>,
    pub accl: Vec<ImuBlock>,
    pub grav: Vec<ImuBlock>,
    pub mnor: Vec<ImuBlock>,
}

impl RawImu {
    /// Total sample count across all blocks for one sensor type.
    pub fn total<F: Fn(&Self) -> &Vec<ImuBlock>>(&self, pick: F) -> usize {
        pick(self).iter().map(|b| b.samples.len()).sum()
    }
}

const GYRO: FourCC = FourCC::new(b"GYRO");
const ACCL: FourCC = FourCC::new(b"ACCL");
const GRAV: FourCC = FourCC::new(b"GRAV");
const MNOR: FourCC = FourCC::new(b"MNOR");
const STMP: FourCC = FourCC::new(b"STMP");
const SCAL: FourCC = FourCC::new(b"SCAL");
const STRM: FourCC = FourCC::new(b"STRM");

/// Walk the GPMF stream and extract every raw IMU block found, with its
/// SCAL divisor and STMP timestamp resolved from the surrounding STRM
/// context.
///
/// Mirrors `parse_gyro_raw.py::parse_gyro_accl_full` — same SCAL/STMP
/// reset semantics on STRM entry, same `type='s' && struct_size=6` gate
/// for accepting an entry as a real sensor block.
pub fn parse_raw_imu(gpmf: &[u8]) -> RawImu {
    let mut out = RawImu::default();
    let mut current_scal: f32 = 1.0;
    let mut current_stmp: Option<u64> = None;

    for entry in GpmfWalker::new(gpmf) {
        // STRM resets per-stream state (Python `parse_gyro_accl_full` line 182).
        if entry.fourcc == STRM {
            current_scal = 1.0;
            current_stmp = None;
            continue;
        }

        // STMP — uint64 ('J' / 8 bytes) or uint32 ('L' / 4 bytes).
        if entry.fourcc == STMP {
            current_stmp = parse_stmp(entry.type_char, entry.struct_size, entry.payload);
            continue;
        }

        // SCAL — divisor for the next data block. Multiple types possible;
        // we use the first element only (Python's `vals[0] if len(vals) == 1`
        // semantics — multi-element SCAL not seen in real Max footage).
        if entry.fourcc == SCAL {
            if let Some(v) = parse_scal_first(entry.type_char, entry.struct_size, entry.payload) {
                current_scal = if v == 0.0 { 1.0 } else { v };
            }
            continue;
        }

        // GYRO/ACCL/GRAV/MNOR — all int16 (type 's'), struct_size 6 (3 × i16).
        let target = match entry.fourcc {
            f if f == GYRO => Some(&mut out.gyro),
            f if f == ACCL => Some(&mut out.accl),
            f if f == GRAV => Some(&mut out.grav),
            f if f == MNOR => Some(&mut out.mnor),
            _ => None,
        };
        if let Some(target) = target {
            if entry.type_char != b's' || entry.struct_size != 6 {
                continue;
            }
            let n = entry.repeat as usize;
            let want = n * 6;
            if entry.payload.len() < want { continue; }
            let mut samples = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 6;
                let x = i16::from_be_bytes([entry.payload[off],   entry.payload[off+1]]);
                let y = i16::from_be_bytes([entry.payload[off+2], entry.payload[off+3]]);
                let z = i16::from_be_bytes([entry.payload[off+4], entry.payload[off+5]]);
                samples.push([
                    x as f32 / current_scal,
                    y as f32 / current_scal,
                    z as f32 / current_scal,
                ]);
            }
            target.push(ImuBlock {
                samples,
                scal: current_scal,
                stmp_us: current_stmp,
            });
        }
    }
    out
}

fn parse_stmp(type_char: u8, struct_size: u8, payload: &[u8]) -> Option<u64> {
    match (type_char, struct_size) {
        (b'J', 8) if payload.len() >= 8 => Some(u64::from_be_bytes(payload[..8].try_into().ok()?)),
        (b'L', 4) if payload.len() >= 4 => Some(u32::from_be_bytes(payload[..4].try_into().ok()?) as u64),
        _ => None,
    }
}

fn parse_scal_first(type_char: u8, struct_size: u8, payload: &[u8]) -> Option<f32> {
    match (type_char, struct_size) {
        (b's', 2) if payload.len() >= 2 => Some(i16::from_be_bytes([payload[0], payload[1]]) as f32),
        (b'S', 2) if payload.len() >= 2 => Some(u16::from_be_bytes([payload[0], payload[1]]) as f32),
        (b'l', 4) if payload.len() >= 4 => Some(i32::from_be_bytes(payload[..4].try_into().ok()?) as f32),
        (b'L', 4) if payload.len() >= 4 => Some(u32::from_be_bytes(payload[..4].try_into().ok()?) as f32),
        (b'f', 4) if payload.len() >= 4 => Some(f32::from_be_bytes(payload[..4].try_into().ok()?)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        std::fs::read("tests/fixtures/GS010172.gpmf").expect("fixture missing")
    }

    /// On the GS010172 fixture (30s clip), expect:
    ///   * 30 GYRO blocks totaling ~24 000 samples (~800 Hz × 30 s)
    ///   * 30 ACCL blocks
    ///   * 30 GRAV blocks
    ///   * 30 MNOR blocks
    /// All blocks should have non-zero SCAL and a valid STMP.
    #[test]
    fn raw_imu_block_counts_match_python() {
        let imu = parse_raw_imu(&fixture());
        assert_eq!(imu.gyro.len(), 30, "GYRO block count");
        assert_eq!(imu.accl.len(), 30, "ACCL block count");
        assert_eq!(imu.grav.len(), 30, "GRAV block count");
        assert_eq!(imu.mnor.len(), 30, "MNOR block count");

        let gyro_total = imu.total(|r| &r.gyro);
        assert!(gyro_total > 20_000 && gyro_total < 30_000,
            "GYRO sample total {gyro_total} out of expected ~24k range");

        for blk in &imu.gyro {
            assert!(blk.scal != 0.0, "zero SCAL on a gyro block");
            assert!(blk.stmp_us.is_some(), "missing STMP on a gyro block");
        }
    }

    #[test]
    fn gyro_samples_have_realistic_magnitudes() {
        let imu = parse_raw_imu(&fixture());
        // GYRO is post-SCAL (in rad/s). For normal handheld footage the
        // magnitude should be in [0, ~30] rad/s (very fast shakes ~1700°/s).
        // If we forgot SCAL, magnitudes would be in the tens of thousands.
        let mut max_mag: f32 = 0.0;
        for blk in &imu.gyro {
            for s in &blk.samples {
                let m = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
                if m > max_mag { max_mag = m; }
            }
        }
        assert!(max_mag < 50.0, "max gyro magnitude {max_mag} rad/s — SCAL probably not applied");
    }
}
