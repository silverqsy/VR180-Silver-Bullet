//! DJI Osmo `.osv` IMU + calibration extraction.
//!
//! Port of the Python `_parse_dji_imu_data` family at
//! `vr180_gui.py:218-516`. The `.osv` container is a standard MP4
//! with five tracks: two HEVC fisheye streams, one AAC audio track,
//! and two `djmd` data tracks. The first `djmd` track (handler_name
//! `"CAM meta"`) carries a hand-rolled protobuf blob with the camera's
//! per-frame orientation, gravity vector, lens calibration, and the
//! high-rate (~990 Hz) quaternion stream we use for stabilization.
//!
//! This crate parses **only the protobuf bytes** — the MP4 atom walk
//! that gets those bytes lives in `vr180-pipeline::decode::extract_dji_meta_stream`
//! since it needs ffmpeg-next.
//!
//! ## Structure (reverse-engineered from the Python parser)
//!
//! ```text
//! top-level {
//!     field 1 (bytes)        = header (unused)
//!     field 2 (bytes)        = video_meta {
//!         field 6 (bytes)    = lens_calib_container {
//!             field 1 (bytes) = lens_A calib   // map<fn, float>: 1..15
//!             field 2 (bytes) = lens_B calib
//!         }
//!     }
//!     field 3 (bytes)        = frame_block (repeated, one per video frame) {
//!         field 2 (bytes)    = orientation {
//!             field 9 (bytes, ≥16 B) = packed floats: per-frame quat (w,x,y,z)
//!             field 10 (bytes)       = packed floats: gravity (gx,gy,gz)
//!         }
//!         field 3 (bytes)    = imu_container {
//!             field 2 (bytes) = lens_arrays {
//!                 field 1 (bytes) = lens_A_samples {
//!                     field 3 (bytes, ≥16 B) = per-sample quat (REPEATED ~33 ×)
//!                 }
//!                 field 2 (bytes) = lens_B_samples (ignored)
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! Quaternion convention is **world → sensor** (i.e. applying the
//! quaternion rotates a world vector into the sensor frame). The Python
//! parser comment at line 649 is explicit about this — get it wrong and
//! stabilization runs the wrong direction.

use crate::{Error, Result};
use vr180_core::gyro::cori_iori::Quat;

/// Per-lens calibration entry from the OSV protobuf.
///
/// Field map (verified against DJI Studio's runtime `Intrinsic` via lldb
/// capture during an export — corrects the old `vr180_gui.py:301-307` guess):
/// `1=fx 2=fy 3=cx 4=cy  5=k1 6=k2 7=k3 8=k4  10=width 11=height
///  15=k5  20=[p1,p2](tangential)  21=mount_quat(x,y,z,w)`.
/// f12/13/14 (yaw/nominal-fov/pitch) are stored but DJI's renderer ignores
/// them; the mount comes from the f21 quaternion. f15 is the 5th radial
/// KB coefficient (NOT roll_offset as previously assumed).
#[derive(Debug, Clone, Default)]
pub struct DjiLensCalib {
    pub fx: Option<f32>,
    pub fy: Option<f32>,
    pub cx: Option<f32>,
    pub cy: Option<f32>,
    pub k1: Option<f32>,
    pub k2: Option<f32>,
    pub k3: Option<f32>,
    pub k4: Option<f32>,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub yaw_offset: Option<f32>,
    pub half_fov: Option<f32>,
    pub pitch_offset: Option<f32>,
    /// 5th Kannala-Brandt radial coefficient — protobuf **field 15**.
    /// DJI's lens model is a 5-coefficient odd-power KB:
    /// `θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹ + k5·θ¹¹`. The k5 term is
    /// what keeps the projection monotonic past ~90° out to the full
    /// ~105° lens FOV. Verified by lldb-capturing DJI Studio's runtime
    /// `Intrinsic` during export (it read field 15 as the 5th radial
    /// coeff — this field was previously mislabeled "roll_offset").
    pub k5: Option<f32>,
    /// Brown-Conrady tangential distortion `(p1, p2)` — protobuf **field
    /// 20** (2×f32). Tiny (~1e-4) but part of DJI's exact model:
    /// `u' = u + 2·p1·u·v + p2·(r²+2u²)`, `v' = v + p1·(r²+2v²) + 2·p2·u·v`.
    pub p1: Option<f32>,
    pub p2: Option<f32>,
    /// Factory IMU-mount quaternion `(x, y, z, w)` — at protobuf field
    /// 21. Differs per camera unit by up to ~0.5°. We previously
    /// hardcoded the value from one test clip, which left a small
    /// rotation error on other cameras; reading per-clip removes that.
    /// (This quat — NOT fields 12/13/14 — is the orientation DJI's
    /// renderer actually uses; f12/13/14 are unused metadata.)
    pub mount_quat_xyzw: Option<[f32; 4]>,
}

/// Extracted IMU + calibration block from a DJI OSV file.
#[derive(Debug, Clone, Default)]
pub struct DjiOsvImu {
    /// Per-frame "reference" quaternion (field 2/sub-9). Less accurate
    /// than the high-rate stream (Python uses it as a fallback only).
    pub frame_quats: Vec<Quat>,
    /// Per-frame gravity vector in world frame (field 2/sub-10).
    pub gravity: Vec<[f32; 3]>,
    /// Per-frame high-rate quaternion stream (field 3 → lens_A path).
    /// Typically ~33 samples per video frame → ~990 Hz at 29.97 fps.
    pub high_rate_quats: Vec<Vec<Quat>>,
    /// Lens-A calibration (right eye → stream 0).
    pub lens_a: DjiLensCalib,
    /// Lens-B calibration (left eye → stream 1).
    pub lens_b: DjiLensCalib,
}

impl DjiOsvImu {
    /// Parse the protobuf bytes pulled out of the `djmd` track.
    /// Errors only on malformed varints / out-of-bounds length-delimited
    /// reads; unknown wire types are silently skipped (matching the
    /// Python parser's tolerance at `vr180_gui.py:269-290`).
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let mut out = DjiOsvImu::default();
        let top = walk_fields(blob)?;

        for f in &top {
            match f.field_num {
                // video_meta — calibration container.
                2 if matches!(f.wire, WireType::LengthDelimited) => {
                    parse_video_meta(f.bytes(blob), &mut out)?;
                }
                // frame_block — repeated, one per video frame.
                3 if matches!(f.wire, WireType::LengthDelimited) => {
                    parse_frame_block(f.bytes(blob), &mut out)?;
                }
                _ => {}
            }
        }

        Ok(out)
    }
}

// ── Internal protobuf walker ────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireType {
    Varint,
    Fixed64,
    LengthDelimited,
    Fixed32,
    Unknown,
}

impl WireType {
    fn from_u8(b: u8) -> Self {
        match b & 0b111 {
            0 => Self::Varint,
            1 => Self::Fixed64,
            2 => Self::LengthDelimited,
            5 => Self::Fixed32,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProtoField {
    field_num: u32,
    wire: WireType,
    /// Byte range in the parent slice (start..end) for LengthDelimited.
    /// For Varint / Fixed32 / Fixed64 it's a tight slice of the value bytes
    /// (so callers can extract directly without re-parsing).
    start: usize,
    end: usize,
    /// Decoded varint value (only set for Varint fields).
    varint: u64,
}

impl ProtoField {
    fn bytes<'a>(&self, parent: &'a [u8]) -> &'a [u8] {
        &parent[self.start..self.end]
    }
    fn as_f32(&self, parent: &[u8]) -> f32 {
        debug_assert!(matches!(self.wire, WireType::Fixed32));
        f32::from_le_bytes(parent[self.start..self.start + 4].try_into().unwrap())
    }
    fn as_f64(&self, parent: &[u8]) -> f64 {
        debug_assert!(matches!(self.wire, WireType::Fixed64));
        f64::from_le_bytes(parent[self.start..self.start + 8].try_into().unwrap())
    }
}

fn read_varint(buf: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut shift: u32 = 0;
    let mut value: u64 = 0;
    loop {
        if *cursor >= buf.len() {
            return Err(Error::GyroflowJson("dji protobuf: varint truncated".into()));
        }
        let b = buf[*cursor];
        *cursor += 1;
        value |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::GyroflowJson("dji protobuf: varint too long".into()));
        }
    }
}

/// Walk one protobuf message, returning the field metadata.
/// Doesn't recurse into LengthDelimited fields — caller decides whether
/// to parse them as sub-messages or treat them as raw bytes.
fn walk_fields(buf: &[u8]) -> Result<Vec<ProtoField>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < buf.len() {
        let tag = read_varint(buf, &mut cursor)? as u32;
        let field_num = tag >> 3;
        let wire = WireType::from_u8(tag as u8);
        match wire {
            WireType::Varint => {
                let start = cursor;
                let v = read_varint(buf, &mut cursor)?;
                let end = cursor;
                out.push(ProtoField {
                    field_num, wire, start, end, varint: v,
                });
            }
            WireType::Fixed32 => {
                if cursor + 4 > buf.len() {
                    return Err(Error::GyroflowJson("dji protobuf: fixed32 truncated".into()));
                }
                out.push(ProtoField {
                    field_num, wire, start: cursor, end: cursor + 4, varint: 0,
                });
                cursor += 4;
            }
            WireType::Fixed64 => {
                if cursor + 8 > buf.len() {
                    return Err(Error::GyroflowJson("dji protobuf: fixed64 truncated".into()));
                }
                out.push(ProtoField {
                    field_num, wire, start: cursor, end: cursor + 8, varint: 0,
                });
                cursor += 8;
            }
            WireType::LengthDelimited => {
                let len = read_varint(buf, &mut cursor)? as usize;
                if cursor + len > buf.len() {
                    return Err(Error::GyroflowJson(format!(
                        "dji protobuf: ld field len={} oob (cursor={}, buf={})",
                        len, cursor, buf.len()
                    )));
                }
                out.push(ProtoField {
                    field_num, wire, start: cursor, end: cursor + len, varint: 0,
                });
                cursor += len;
            }
            WireType::Unknown => {
                // Skip groups / deprecated wire types silently — matches
                // the permissive Python parser at lines 269-290.
                tracing::trace!("dji protobuf: skipping unknown wire type at cursor={cursor}");
                break;
            }
        }
    }
    Ok(out)
}

/// Read all `Fixed32` values inside a length-delimited sub-message,
/// in field order. This is how DJI packs floats — they appear as
/// individual wire-type-5 fields with sequential field numbers.
/// The Python equivalent is `_parse_floats_from_sub` at line 269.
fn extract_floats_from(buf: &[u8]) -> Vec<f32> {
    let fields = match walk_fields(buf) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        match f.wire {
            WireType::Fixed32 => out.push(f.as_f32(buf)),
            WireType::Fixed64 => out.push(f.as_f64(buf) as f32),
            // Coerce a varint into a float — matches Python's lossy
            // handling at line 287.
            WireType::Varint  => out.push(f.varint as f32),
            _ => {}
        }
    }
    out
}

// ── Section-specific parsers ────────────────────────────────────────

fn parse_video_meta(buf: &[u8], out: &mut DjiOsvImu) -> Result<()> {
    let fields = walk_fields(buf)?;
    for f in &fields {
        // Calibration container.
        if f.field_num == 6 && matches!(f.wire, WireType::LengthDelimited) {
            let container = f.bytes(buf);
            let inner = walk_fields(container)?;
            // Lens-A is field 1, Lens-B is field 2.
            for lens in inner {
                if !matches!(lens.wire, WireType::LengthDelimited) { continue; }
                let calib = parse_lens_calib(lens.bytes(container))?;
                match lens.field_num {
                    1 => out.lens_a = calib,
                    2 => out.lens_b = calib,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn parse_lens_calib(buf: &[u8]) -> Result<DjiLensCalib> {
    let mut out = DjiLensCalib::default();
    let fields = walk_fields(buf)?;
    for f in &fields {
        // Field 21 is length-delimited: 4×float32 = 16 bytes for the
        // factory mount quaternion (x, y, z, w).
        if f.field_num == 21 && matches!(f.wire, WireType::LengthDelimited) {
            let bytes = f.bytes(buf);
            if bytes.len() >= 16 {
                let x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
                let y = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let z = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
                let w = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
                out.mount_quat_xyzw = Some([x, y, z, w]);
            }
            continue;
        }
        // Field 20 is length-delimited: 2×float32 = 8 bytes for the
        // Brown-Conrady tangential distortion (p1, p2). DJI feeds these
        // into the projection (verified in the runtime Intrinsic).
        if f.field_num == 20 && matches!(f.wire, WireType::LengthDelimited) {
            let bytes = f.bytes(buf);
            if bytes.len() >= 8 {
                out.p1 = Some(f32::from_le_bytes(bytes[0..4].try_into().unwrap()));
                out.p2 = Some(f32::from_le_bytes(bytes[4..8].try_into().unwrap()));
            }
            continue;
        }
        let v: Option<f32> = match f.wire {
            WireType::Fixed32 => Some(f.as_f32(buf)),
            WireType::Fixed64 => Some(f.as_f64(buf) as f32),
            WireType::Varint  => Some(f.varint as f32),
            _ => None,
        };
        let Some(v) = v else { continue };
        match f.field_num {
            1  => out.fx = Some(v),
            2  => out.fy = Some(v),
            3  => out.cx = Some(v),
            4  => out.cy = Some(v),
            5  => out.k1 = Some(v),
            6  => out.k2 = Some(v),
            7  => out.k3 = Some(v),
            8  => out.k4 = Some(v),
            10 => out.width  = Some(v),
            11 => out.height = Some(v),
            12 => out.yaw_offset   = Some(v),  // unused metadata
            13 => out.half_fov     = Some(v),  // unused metadata (nominal FOV°)
            14 => out.pitch_offset = Some(v),  // unused metadata
            15 => out.k5 = Some(v),            // 5th KB radial coeff (NOT roll!)
            _ => {}
        }
    }
    Ok(out)
}

fn parse_frame_block(buf: &[u8], out: &mut DjiOsvImu) -> Result<()> {
    let fields = walk_fields(buf)?;
    let mut frame_quat = Quat::IDENTITY;
    let mut gravity = [0.0_f32, -1.0, 0.0]; // Python fallback at line 455
    let mut hr_quats: Vec<Quat> = Vec::new();

    for f in &fields {
        match f.field_num {
            // Orientation sub-message: per-frame quat + gravity.
            2 if matches!(f.wire, WireType::LengthDelimited) => {
                let orient = f.bytes(buf);
                let sub = walk_fields(orient)?;
                for s in &sub {
                    if !matches!(s.wire, WireType::LengthDelimited) { continue; }
                    let bytes = s.bytes(orient);
                    let floats = extract_floats_from(bytes);
                    if s.field_num == 9 && floats.len() >= 4 {
                        // Stream is (w, x, y, z). The Python parser
                        // uses this order and produces stable working
                        // stabilization. DJI's INTERNAL struct uses
                        // (x, y, z, w) field order (verified via
                        // disassembly of `quaternionToRotation`), but
                        // their parser shuffles the stream bytes into
                        // that struct explicitly — the bytes on the
                        // wire still arrive in (w, x, y, z) order.
                        frame_quat = Quat {
                            w: floats[0], x: floats[1],
                            y: floats[2], z: floats[3],
                        };
                    } else if s.field_num == 10 && floats.len() >= 3 {
                        gravity = [floats[0], floats[1], floats[2]];
                    }
                }
            }
            // IMU container: high-rate per-frame quaternion stream.
            3 if matches!(f.wire, WireType::LengthDelimited) => {
                let imu = f.bytes(buf);
                let imu_fields = walk_fields(imu)?;
                for ifield in &imu_fields {
                    if ifield.field_num != 2
                        || !matches!(ifield.wire, WireType::LengthDelimited) { continue; }
                    let lens_arrays = ifield.bytes(imu);
                    let arr_fields = walk_fields(lens_arrays)?;
                    for arr in &arr_fields {
                        // Only the Lens-A array (field 1) is used.
                        if arr.field_num != 1
                            || !matches!(arr.wire, WireType::LengthDelimited) { continue; }
                        let lens_payload = arr.bytes(lens_arrays);
                        // Inside the lens payload: repeated sub-message
                        // field 3 holding one quat each.
                        let lens_fields = walk_fields(lens_payload)?;
                        for lf in &lens_fields {
                            if lf.field_num != 3
                                || !matches!(lf.wire, WireType::LengthDelimited) { continue; }
                            let q_bytes = lf.bytes(lens_payload);
                            let floats = extract_floats_from(q_bytes);
                            if floats.len() >= 4 {
                                // Stream is (w, x, y, z) — see frame-quat
                                // parser above for the disassembly note.
                                hr_quats.push(Quat {
                                    w: floats[0], x: floats[1],
                                    y: floats[2], z: floats[3],
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Normalize the frame quat — DJI sends non-unit quats occasionally.
    let frame_quat = frame_quat.normalize();
    let hr_quats: Vec<Quat> = hr_quats.into_iter().map(|q| q.normalize()).collect();

    out.frame_quats.push(frame_quat);
    out.gravity.push(gravity);
    out.high_rate_quats.push(hr_quats);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-frame DJI protobuf with a known quat / gravity /
    /// high-rate quat, then parse it back. Smoke test the wire layout.
    #[test]
    fn synthetic_single_frame_roundtrip() {
        // Build a payload by hand. Easier to assert on the layout this
        // way than to set up a real .osv mock.
        let mut blob = Vec::new();

        // We'll construct one frame_block (top field 3, LD).
        //   inside it: orientation (field 2, LD) with
        //     field 9 LD with 4 fixed32 floats: (1.0, 0.0, 0.0, 0.0)
        //     field 10 LD with 3 fixed32 floats: (0.0, -1.0, 0.0)
        //   inside it: imu_container (field 3, LD) with
        //     field 2 LD with
        //       field 1 LD with
        //         repeated field 3 LD with 4 fixed32 floats

        fn write_tag(buf: &mut Vec<u8>, field_num: u32, wire: u8) {
            let tag = (field_num << 3) | wire as u32;
            // Varint encode.
            let mut v = tag as u64;
            while v >= 0x80 { buf.push((v as u8) | 0x80); v >>= 7; }
            buf.push(v as u8);
        }
        fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
            while v >= 0x80 { buf.push((v as u8) | 0x80); v >>= 7; }
            buf.push(v as u8);
        }
        fn write_ld(buf: &mut Vec<u8>, field_num: u32, inner: &[u8]) {
            write_tag(buf, field_num, 2);
            write_varint(buf, inner.len() as u64);
            buf.extend_from_slice(inner);
        }
        fn write_f32(buf: &mut Vec<u8>, field_num: u32, v: f32) {
            write_tag(buf, field_num, 5);
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // packed floats in their own sub-message.
        let mut frame_quat_inner = Vec::new();
        for (i, v) in [1.0_f32, 0.0, 0.0, 0.0].iter().enumerate() {
            write_f32(&mut frame_quat_inner, (i + 1) as u32, *v);
        }
        let mut gravity_inner = Vec::new();
        for (i, v) in [0.0_f32, -1.0, 0.0].iter().enumerate() {
            write_f32(&mut gravity_inner, (i + 1) as u32, *v);
        }

        let mut orientation = Vec::new();
        write_ld(&mut orientation, 9, &frame_quat_inner);
        write_ld(&mut orientation, 10, &gravity_inner);

        let mut hr_quat_inner = Vec::new();
        for (i, v) in [0.99_f32, 0.01, 0.02, 0.03].iter().enumerate() {
            write_f32(&mut hr_quat_inner, (i + 1) as u32, *v);
        }
        let mut lens_a_array = Vec::new();
        write_ld(&mut lens_a_array, 3, &hr_quat_inner);

        let mut lens_arrays = Vec::new();
        write_ld(&mut lens_arrays, 1, &lens_a_array);

        let mut imu_container = Vec::new();
        write_ld(&mut imu_container, 2, &lens_arrays);

        let mut frame_block = Vec::new();
        write_ld(&mut frame_block, 2, &orientation);
        write_ld(&mut frame_block, 3, &imu_container);

        write_ld(&mut blob, 3, &frame_block);

        let imu = DjiOsvImu::parse(&blob).expect("parse");
        assert_eq!(imu.frame_quats.len(), 1);
        let q = imu.frame_quats[0];
        assert!((q.w - 1.0).abs() < 1e-6);
        assert_eq!(imu.gravity[0], [0.0, -1.0, 0.0]);
        assert_eq!(imu.high_rate_quats.len(), 1);
        assert_eq!(imu.high_rate_quats[0].len(), 1);
        let hr = imu.high_rate_quats[0][0];
        // After normalize, w should be very close to 0.99 (input was
        // already near-unit, so normalize is ~identity).
        assert!((hr.w - 0.99).abs() < 0.01);
    }

    #[test]
    fn unknown_wire_type_skips_cleanly() {
        // Wire type 3 (start group, deprecated). Parser should stop
        // gracefully without erroring.
        let blob = vec![0b00001011]; // field 1, wire 3
        let imu = DjiOsvImu::parse(&blob).expect("parse should succeed");
        assert!(imu.frame_quats.is_empty());
    }
}
