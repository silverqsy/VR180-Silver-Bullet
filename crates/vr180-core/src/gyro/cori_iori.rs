//! CORI / IORI quaternion extraction from a GPMF byte stream.
//!
//! - **CORI** (Camera Orientation) — physical camera orientation
//!   (raw motion). One quaternion sample per video frame.
//! - **IORI** (Image Orientation) — rotation the GoPro firmware
//!   applied during in-camera stabilization. For raw / uncorrected
//!   footage this is identity at every frame.
//!
//! GPMF storage convention:
//! - FourCC: `CORI` or `IORI`
//! - type_char: `'s'` (signed int16)
//! - struct_size: 8 (4 × int16 = quaternion)
//! - Component order in file: **W, X, Y, Z** (standard order)
//! - Encoding: **Q15 fixed-point** — divide each int16 by 32768.0
//!   to recover float in `[-1, 1]`
//! - Endianness: big-endian
//!
//! The Python project memory notes "CORI in file stores (w, x, Z, Y) —
//! y/z slots swapped from standard"; that swap happens in the
//! axis-remap step that consumes these samples, NOT here. This module
//! preserves the on-disk order verbatim so callers can re-derive any
//! axis convention they need.

use super::gpmf::{GpmfWalker, FourCC};

/// A unit quaternion in `(w, x, y, z)` order (as stored in the file).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// Decode a 4-int16 big-endian Q15 quaternion from raw bytes.
    /// Panics if `bytes.len() < 8` — callers must validate.
    #[inline]
    fn from_q15_be(bytes: &[u8; 8]) -> Self {
        let w = i16::from_be_bytes([bytes[0], bytes[1]]);
        let x = i16::from_be_bytes([bytes[2], bytes[3]]);
        let y = i16::from_be_bytes([bytes[4], bytes[5]]);
        let z = i16::from_be_bytes([bytes[6], bytes[7]]);
        const INV: f32 = 1.0 / 32768.0;
        Quat {
            w: w as f32 * INV,
            x: x as f32 * INV,
            y: y as f32 * INV,
            z: z as f32 * INV,
        }
    }

    /// Quaternion multiplication. `a * b` represents the composition
    /// "rotate by `b` first, then by `a`" when used to rotate a vector
    /// via `q.rotate(v)`.
    #[inline]
    pub fn mul(self, b: Quat) -> Quat {
        Quat {
            w: self.w * b.w - self.x * b.x - self.y * b.y - self.z * b.z,
            x: self.w * b.x + self.x * b.w + self.y * b.z - self.z * b.y,
            y: self.w * b.y - self.x * b.z + self.y * b.w + self.z * b.x,
            z: self.w * b.z + self.x * b.y - self.y * b.x + self.z * b.w,
        }
    }

    /// Conjugate — for a unit quaternion, this equals the inverse
    /// (i.e. the rotation in the opposite direction).
    #[inline]
    pub fn conjugate(self) -> Quat {
        Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// L2 norm (typically very close to 1 for a unit quaternion).
    #[inline]
    pub fn norm(self) -> f32 {
        (self.w * self.w + self.x * self.x
            + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Re-normalize. Useful after a sequence of mul / slerp ops
    /// where accumulated f32 error has drifted the norm off 1.
    #[inline]
    pub fn normalize(self) -> Quat {
        let n = self.norm();
        if n < 1e-12 { return Quat::IDENTITY; }
        Quat { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    /// Dot product. Used for slerp and quaternion-distance metrics.
    #[inline]
    pub fn dot(self, b: Quat) -> f32 {
        self.w * b.w + self.x * b.x + self.y * b.y + self.z * b.z
    }

    /// Spherical linear interpolation. `t ∈ [0, 1]` blends from
    /// `self` to `b`. Uses the short-arc rule (flips `b` if dot < 0)
    /// to avoid the "wrong way around" interpolation when the two
    /// quaternions are on opposite hemispheres.
    pub fn slerp(self, mut b: Quat, t: f32) -> Quat {
        let mut d = self.dot(b);
        if d < 0.0 {
            // Take the short arc by negating b (same rotation, opposite hemi).
            b = Quat { w: -b.w, x: -b.x, y: -b.y, z: -b.z };
            d = -d;
        }
        // Numerical safety: when the quats are nearly parallel, fall
        // back to lerp (avoids div-by-zero in sin(θ)).
        if d > 0.9995 {
            let r = Quat {
                w: self.w + t * (b.w - self.w),
                x: self.x + t * (b.x - self.x),
                y: self.y + t * (b.y - self.y),
                z: self.z + t * (b.z - self.z),
            };
            return r.normalize();
        }
        let theta = d.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let s1 = ((1.0 - t) * theta).sin() / sin_theta;
        let s2 = (t * theta).sin() / sin_theta;
        Quat {
            w: s1 * self.w + s2 * b.w,
            x: s1 * self.x + s2 * b.x,
            y: s1 * self.y + s2 * b.y,
            z: s1 * self.z + s2 * b.z,
        }
    }

    /// Convert to a 3×3 rotation matrix (row-major).
    /// Standard derivation; result is the matrix that, applied to a
    /// column vector `v`, gives the rotated vector `R·v`.
    ///
    /// Returns `[r00, r01, r02, r10, r11, r12, r20, r21, r22]` —
    /// row-major. For wgpu uniform upload, the caller pads each row
    /// to 4 floats (std140 mat3 layout).
    pub fn to_mat3_row_major(self) -> [f32; 9] {
        let Quat { w, x, y, z } = self;
        let xx = x * x; let yy = y * y; let zz = z * z;
        let xy = x * y; let xz = x * z; let yz = y * z;
        let wx = w * x; let wy = w * y; let wz = w * z;
        [
            1.0 - 2.0 * (yy + zz),   2.0 * (xy - wz),         2.0 * (xz + wy),
            2.0 * (xy + wz),         1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx),
            2.0 * (xz - wy),         2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy),
        ]
    }
}

const CORI: FourCC = FourCC::new(b"CORI");
const IORI: FourCC = FourCC::new(b"IORI");

/// Extract all CORI samples from a GPMF byte stream.
///
/// The walker yields one [`GpmfEntry`] per CORI record; each record
/// contains `repeat` quaternion samples (typically the per-frame
/// quaternions for a 1-second chunk of footage).
pub fn parse_cori(gpmf: &[u8]) -> Vec<Quat> {
    parse_quaternion(gpmf, CORI)
}

/// Extract all IORI samples from a GPMF byte stream. See [`parse_cori`].
pub fn parse_iori(gpmf: &[u8]) -> Vec<Quat> {
    parse_quaternion(gpmf, IORI)
}

fn parse_quaternion(gpmf: &[u8], want: FourCC) -> Vec<Quat> {
    let mut out = Vec::new();
    for entry in GpmfWalker::new(gpmf) {
        if entry.fourcc != want { continue; }
        // Defense against false positives (e.g. "CORI" appearing inside
        // an unrelated payload): require type='s', struct_size=8.
        if entry.type_char != b's' || entry.struct_size != 8 { continue; }
        let want_bytes = entry.repeat as usize * 8;
        if entry.payload.len() < want_bytes { continue; }
        for i in 0..entry.repeat as usize {
            let off = i * 8;
            let chunk: &[u8; 8] = entry.payload[off..off + 8].try_into().unwrap();
            out.push(Quat::from_q15_be(chunk));
        }
    }
    out
}

/// Convert a quaternion to intrinsic ZYX Euler angles `(roll, pitch, yaw)`
/// in **degrees**. Matches the convention used by the Python reference's
/// CORI Euler printouts; the math is the standard aerospace formulation.
pub fn quat_to_euler_zyx(q: Quat) -> (f32, f32, f32) {
    // roll (x-axis)
    let sr = 2.0 * (q.w * q.x + q.y * q.z);
    let cr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    let roll = sr.atan2(cr);
    // pitch (y-axis) — clamp to avoid asin domain errors at the pole
    let sp = (2.0 * (q.w * q.y - q.z * q.x)).clamp(-1.0, 1.0);
    let pitch = sp.asin();
    // yaw (z-axis)
    let sy = 2.0 * (q.w * q.z + q.x * q.y);
    let cy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    let yaw = sy.atan2(cy);
    (roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        std::fs::read("tests/fixtures/GS010172.gpmf").expect("fixture missing")
    }

    /// Reference values come from running the Python parser on the same
    /// fixture (see the dump in the Phase 0.2 commit message).
    #[test]
    fn cori_count_matches_python() {
        assert_eq!(parse_cori(&fixture()).len(), 875);
    }

    #[test]
    fn iori_count_matches_python() {
        assert_eq!(parse_iori(&fixture()).len(), 875);
    }

    #[test]
    fn cori_first_sample_matches_python() {
        let q = parse_cori(&fixture())[0];
        // Python: (0.999969..., 0.001495..., 0.000610..., 0.001984...)
        // f32 precision: compare to ~1e-6.
        let want = Quat {
            w: 0.999_969_5,
            x: 0.001_495_36,
            y: 0.000_610_35,
            z: 0.001_983_64,
        };
        for (g, w) in [(q.w, want.w), (q.x, want.x), (q.y, want.y), (q.z, want.z)] {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}, diff {}", g - w);
        }
    }

    #[test]
    fn iori_first_sample_is_identity() {
        // Python: IORI[0] = (1.0, 0.0, 0.0, 0.0) — exact (no stabilization rotation).
        let q = parse_iori(&fixture())[0];
        assert!((q.w - 1.0).abs() < 1e-4, "w={}", q.w);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn cori_euler_range_matches_python() {
        // Python reported ranges for this clip:
        //   roll  [-22.046°, 159.214°]
        //   pitch [-70.929°,  30.750°]
        //   yaw   [-64.449°,  28.713°]
        let cori = parse_cori(&fixture());
        let (mut rmin, mut rmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut pmin, mut pmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for q in cori {
            let (r, p, y) = quat_to_euler_zyx(q);
            rmin = rmin.min(r); rmax = rmax.max(r);
            pmin = pmin.min(p); pmax = pmax.max(p);
            ymin = ymin.min(y); ymax = ymax.max(y);
        }
        // Tolerance: 0.01° (f32 trig accumulation noise).
        assert!((rmin - -22.046).abs() < 0.01, "roll min: {rmin}");
        assert!((rmax - 159.214).abs() < 0.01, "roll max: {rmax}");
        assert!((pmin - -70.929).abs() < 0.01, "pitch min: {pmin}");
        assert!((pmax -  30.750).abs() < 0.01, "pitch max: {pmax}");
        assert!((ymin - -64.449).abs() < 0.01, "yaw min: {ymin}");
        assert!((ymax -  28.713).abs() < 0.01, "yaw max: {ymax}");
    }
}
