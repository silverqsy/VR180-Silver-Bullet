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
//! ## Structure (derived from the Python parser)
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
/// Field map (verified against DJI Studio's output — corrects the old
/// `vr180_gui.py:301-307` guess):
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
    /// ~105° lens FOV. Verified against DJI Studio's output (field 15 is
    /// the 5th radial coeff — this field was previously mislabeled
    /// "roll_offset").
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
    /// Exact unified-camera-model description of the same lens, when the
    /// source provides one (Insta360 factory calibration). `fx`/`cx`/`cy`/`k`
    /// above then hold a Kannala-Brandt fit of its radial curve for the
    /// override UI and CPU consumers; the renderers prefer this model.
    pub omni: Option<OmniLensModel>,
}

/// Unified camera model (Mei/Geyer `ξ` + even radial polynomial + tangential
/// + thin prism), in the same pixel frame as the owning [`DjiLensCalib`]'s
/// `cx`/`cy`. Projection of a camera-frame ray `(X, Y, Z)` (x right, y DOWN,
/// z optical): `(x, y) = (X, Y) / (ξ·|d| + Z)`, `r² = x²+y²`,
/// `D = 1 + k1 r² + k2 r⁴ + k3 r⁶ + k4 r⁸ + k5 r¹⁰`,
/// `x' = x·D + (r²+2x²)(A + C r²) + 2xy(B + E r²) + s1 r² + s2 r⁴`,
/// `y' = y·D + (r²+2y²)(B + E r²) + 2xy(A + C r²) + s3 r² + s4 r⁴`,
/// `u = cx + fx·x'`, `v = cy + fy·y'`. Matched to Insta360 Studio's output.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OmniLensModel {
    pub xi: f32,
    pub fx: f32,
    pub fy: f32,
    /// `[k1, k2, k3, k4, k5]`.
    pub radial: [f32; 5],
    /// `[A, B, C, E]`.
    pub tangential: [f32; 4],
    /// `[s1, s2, s3, s4]` — x gets `s1 r² + s2 r⁴`, y gets `s3 r² + s4 r⁴`.
    pub prism: [f32; 4],
}

/// Where on its high-rate block a frame's stabilization pose is sampled.
/// Follows from how the block was recorded — never a tuned number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleAnchor {
    /// The block starts at the sensor's frame event (DJI files): the pose is
    /// the centre row's mid-exposure, `readout/2 + (video_ts − block_ts) −
    /// shutter/2` after the block's first sample (`exposure_ms`,
    /// `frame_ts_offset_ms`).
    #[default]
    ReadoutMid,
    /// The block was synthesized centred on the frame's measured content
    /// time (Insta360): the pose is the block's midpoint.
    BlockMid,
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
    /// Camera model string from the video-meta header (`[2.2.1.4]`) —
    /// "Osmo OQ001" = OSMO 360, "Osmo OQ002" = OSMO 360 II. Used for
    /// per-model defaults (the II's streams arrive eye-swapped relative
    /// to the I after the VR180 mod).
    pub camera_model: Option<String>,
    /// Explicit IMU→camera basis (rows = camera x/y/z axes expressed in IMU
    /// coordinates). `None` for DJI files — the basis is then derived from
    /// the lens-A mount quaternion. Set by sources that synthesize this
    /// structure from a raw gyro (Insta360 `.insv`), whose sensor mount is
    /// measured rather than read from the file.
    pub imu_to_cam: Option<[[f32; 3]; 3]>,
    /// Sensor readout (rolling-shutter sweep) time in ms when the file states
    /// it (Insta360 `rolling_shutter_time`). `None` → the per-source default.
    pub readout_ms: Option<f32>,
    /// The same motion stream resampled at lens B's own per-frame content
    /// times (`frame_quats` / `high_rate_quats` / `gravity` only), for
    /// cameras whose two sensors expose independently (Insta360: the
    /// second sensor's exposure record). `None` when both lenses share the
    /// frame timing (DJI). Consumers stabilize the eye that shows lens B
    /// from this timeline and everything else from `self`.
    pub lens_b_timeline: Option<Box<DjiOsvImu>>,
    /// Per-frame exposure (shutter) time in ms, from the frame's exposure
    /// record (`[2.4.1]`: a `{numerator, denominator}` fraction of a second —
    /// `[1, 520]` = 1/520 s on the OSMO 360, `[100000, 79298304]` on the
    /// 360 II). `NaN` when the file doesn't carry it; empty for synthesized
    /// sources. Indexed like `frame_quats`.
    pub exposure_ms: Vec<f32>,
    /// Per-frame offset (ms) of the video frame's timestamp (`[1.2]`) from
    /// its IMU block's timestamp (`[3.2.1.1]`, the time of the block's first
    /// high-rate sample): the sensor's frame event lands ~0.5–0.6 ms after
    /// the block starts. `NaN` when unavailable. Indexed like `frame_quats`.
    pub frame_ts_offset_ms: Vec<f32>,
    /// Sensor scan-line time (ns) from the clip header, when stated —
    /// `[1.13.1]` on the OSMO 360, `[1.11.1]` on the 360 II. `readout_ms` is
    /// derived from it (× the frame's rows).
    pub line_time_ns: Option<f32>,
    /// How the per-frame pose is placed on `high_rate_quats` — see
    /// [`SampleAnchor`].
    pub sample_anchor: SampleAnchor,
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
                // Clip header — sensor line time.
                1 if matches!(f.wire, WireType::LengthDelimited) => {
                    parse_clip_header(f.bytes(blob), &mut out)?;
                }
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

        // Rolling-shutter readout = line time × rows. 4766 ns × 3840 =
        // 18.301 ms (≤ 30 fps modes) and 4226 ns × 3840 = 16.228 ms (50 fps)
        // — both matched to DJI Studio's output; the 360 II states 4183 ns
        // (16.06 ms at 60 fps). Lets the stabilizer time every mode from the
        // file instead of a per-fps table.
        if out.readout_ms.is_none() {
            if let Some(ns) = out.line_time_ns {
                let rows = out.lens_a.height.filter(|h| *h > 0.0).unwrap_or(3840.0);
                let readout_ms = ns * rows / 1.0e6;
                if (8.0..=40.0).contains(&readout_ms) {
                    out.readout_ms = Some(readout_ms);
                }
            }
        }

        Ok(out)
    }

    /// Parse and concatenate the IMU of several OSV files that form ONE
    /// continuous recording (sequential `_0023`, `_0024`, … segments),
    /// end to end. The per-frame streams (`frame_quats` / `gravity` /
    /// `high_rate_quats`) are appended in segment order so they index by
    /// ABSOLUTE frame number across the whole merged timeline — exactly
    /// what `compute_dji_stabilization` expects when given the total frame
    /// count. Calibration is taken from the first segment (it's identical
    /// across a recording). `blobs` are the raw `djmd` payloads, one per
    /// segment, already extracted by the caller.
    ///
    /// NOTE: the camera writes orientation continuously across a split
    /// recording, so a straight concatenation keeps the timeline coherent.
    /// (If a future camera referenced each segment to its own start, this
    /// is where a boundary re-base would go.)
    pub fn parse_multi(blobs: &[Vec<u8>]) -> Result<Self> {
        let mut out = DjiOsvImu::default();
        for (i, blob) in blobs.iter().enumerate() {
            let seg = Self::parse(blob)?;
            if i == 0 {
                out.lens_a = seg.lens_a;
                out.lens_b = seg.lens_b;
                out.camera_model = seg.camera_model;
                out.line_time_ns = seg.line_time_ns;
                out.readout_ms = seg.readout_ms;
            }
            out.frame_quats.extend(seg.frame_quats);
            out.gravity.extend(seg.gravity);
            out.high_rate_quats.extend(seg.high_rate_quats);
            out.exposure_ms.extend(seg.exposure_ms);
            out.frame_ts_offset_ms.extend(seg.frame_ts_offset_ms);
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

/// Clip header (top-level field 1): the sensor's scan-line time in ns —
/// sub-message 13 on the OSMO 360 (its sub-message 11 holds the frame rate
/// as a float and fails the varint check), sub-message 11 on the 360 II.
fn parse_clip_header(buf: &[u8], out: &mut DjiOsvImu) -> Result<()> {
    let fields = walk_fields(buf)?;
    for f in &fields {
        if !(f.field_num == 13 || f.field_num == 11)
            || !matches!(f.wire, WireType::LengthDelimited)
        {
            continue;
        }
        let msg = f.bytes(buf);
        let sub = walk_fields(msg)?;
        if let Some(v) = sub.iter().find(|s| s.field_num == 1 && matches!(s.wire, WireType::Varint)) {
            // Field 13 wins when both decode.
            if f.field_num == 13 || out.line_time_ns.is_none() {
                let ns = v.varint as f32;
                if (1000.0..=20000.0).contains(&ns) {
                    out.line_time_ns = Some(ns);
                }
            }
        }
    }
    Ok(())
}

fn parse_video_meta(buf: &[u8], out: &mut DjiOsvImu) -> Result<()> {
    let fields = walk_fields(buf)?;
    for f in &fields {
        // Calibration container. OSMO 360 (`dvtm` v1) stores it in field
        // 6; OSMO 360 II (`dvtm_OQ102.proto`) moved it to field 5 with
        // the SAME dewarp-parameter field numbering inside (verified against
        // a real OQ002 clip — fx/fy/cx/cy/k1..k4/dims/f12±180° all line
        // up; f21 mount quat is present but zeroed on the II).
        if (f.field_num == 6 || f.field_num == 5)
            && matches!(f.wire, WireType::LengthDelimited)
        {
            let container = f.bytes(buf);
            let inner = walk_fields(container)?;
            // Lens-A is field 1, Lens-B is field 2 — SAME on both the OSMO
            // 360 (container field 6) and the II (container field 5).
            // Confirmed against DJI Studio's output on BOTH cameras: the
            // per-sensor mesh loads entry 2 for the forward (0°) sensor and
            // entry 1 for the backward (180°) sensor on BOTH generations,
            // i.e. the entry↔sensor↔role mapping is identical — so the II
            // uses the v1 assignment. (A rig-pitch-immune disparity test
            // agrees: v1-style pairing residual-STD 5.9 px vs reversed 7.2 px.)
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
        // Device block (field 2) → sub 1 → sub 4 = camera model string
        // ("Osmo OQ001" / "Osmo OQ002").
        if f.field_num == 2 && matches!(f.wire, WireType::LengthDelimited) {
            let dev = f.bytes(buf);
            if let Ok(sub) = walk_fields(dev) {
                for s in &sub {
                    if s.field_num != 1 || !matches!(s.wire, WireType::LengthDelimited) { continue; }
                    let inner = s.bytes(dev);
                    if let Ok(inner_fields) = walk_fields(inner) {
                        for i in &inner_fields {
                            if i.field_num == 4 && matches!(i.wire, WireType::LengthDelimited) {
                                if let Ok(name) = std::str::from_utf8(i.bytes(inner)) {
                                    out.camera_model = Some(name.to_string());
                                }
                            }
                        }
                    }
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
                // OSMO 360 II writes this field but ZEROED — a zero quat
                // would corrupt the stabilization basis change. Treat
                // degenerate values as absent (falls back to the
                // hardcoded LENS_A quat downstream).
                if (x * x + y * y + z * z + w * w).sqrt() > 0.5 {
                    out.mount_quat_xyzw = Some([x, y, z, w]);
                }
            }
            continue;
        }
        // Field 20 is length-delimited: 2×float32 = 8 bytes for the
        // Brown-Conrady tangential distortion (p1, p2). DJI feeds these
        // into the projection (verified against DJI Studio's output).
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
    let mut video_ts: Option<u64> = None;
    let mut block_ts: Option<u64> = None;
    let mut exposure_ms = f32::NAN;

    for f in &fields {
        match f.field_num {
            // Frame header: sub-field 2 = the frame's video timestamp (µs).
            1 if matches!(f.wire, WireType::LengthDelimited) => {
                for s in walk_fields(f.bytes(buf))? {
                    if s.field_num == 2 && matches!(s.wire, WireType::Varint) {
                        video_ts = Some(s.varint);
                    }
                }
            }
            // Orientation sub-message: per-frame quat + gravity (+ exposure).
            2 if matches!(f.wire, WireType::LengthDelimited) => {
                let orient = f.bytes(buf);
                let sub = walk_fields(orient)?;
                for s in &sub {
                    if !matches!(s.wire, WireType::LengthDelimited) { continue; }
                    let bytes = s.bytes(orient);
                    if s.field_num == 4 {
                        // Exposure record — the shutter time as a fraction
                        // of a second (see `DjiOsvImu::exposure_ms`).
                        if let Some(e) = parse_exposure_ms(bytes) {
                            exposure_ms = e;
                        }
                        continue;
                    }
                    let floats = extract_floats_from(bytes);
                    if s.field_num == 9 && floats.len() >= 4 {
                        // Stream is (w, x, y, z). The Python parser
                        // uses this order and produces stable working
                        // stabilization. DJI's internal struct uses
                        // (x, y, z, w) field order, but their parser
                        // shuffles the stream bytes into that struct
                        // explicitly — the bytes on the wire still
                        // arrive in (w, x, y, z) order.
                        frame_quat = Quat {
                            w: floats[0], x: floats[1],
                            y: floats[2], z: floats[3],
                        };
                    } else if s.field_num == 10 && floats.len() >= 3 {
                        gravity = [floats[0], floats[1], floats[2]];
                    } else if s.field_num == 22 && floats.len() >= 4 {
                        // OSMO 360 II schema: per-frame quat moved from
                        // field 9 to field 22 (same w,x,y,z order —
                        // verified: matches the HR sample stream).
                        frame_quat = Quat {
                            w: floats[0], x: floats[1],
                            y: floats[2], z: floats[3],
                        };
                    } else if s.field_num == 23 && floats.len() >= 3 {
                        // OSMO 360 II: gravity moved from field 10 to 23
                        // (last 3 floats = x, y, z).
                        let n = floats.len();
                        gravity = [floats[n - 3], floats[n - 2], floats[n - 1]];
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
                            // Sub-field 1 = the block's timestamp (µs, 32-bit
                            // counter) = the time of the first sample below.
                            if lf.field_num == 1 && matches!(lf.wire, WireType::Varint) {
                                block_ts = Some(lf.varint);
                                continue;
                            }
                            if lf.field_num != 3
                                || !matches!(lf.wire, WireType::LengthDelimited) { continue; }
                            let q_bytes = lf.bytes(lens_payload);
                            let floats = extract_floats_from(q_bytes);
                            if floats.len() >= 4 {
                                // Stream is (w, x, y, z) — see frame-quat
                                // parser above for the ordering note.
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

    // Video-timestamp − block-timestamp, modulo 2^32 (the block counter is
    // 32-bit; the video timestamp is 64-bit).
    let frame_ts_offset_ms = match (video_ts, block_ts) {
        (Some(v), Some(b)) => (v.wrapping_sub(b) as u32 as i32) as f32 / 1000.0,
        _ => f32::NAN,
    };

    out.frame_quats.push(frame_quat);
    out.gravity.push(gravity);
    out.high_rate_quats.push(hr_quats);
    out.exposure_ms.push(exposure_ms);
    out.frame_ts_offset_ms.push(frame_ts_offset_ms);
    Ok(())
}

/// Decode a frame's exposure record (`[2.4.1]`): varints `{numerator,
/// denominator}` (packed, or as repeated fields) of the shutter time in
/// seconds → milliseconds. `None` when absent or nonsensical.
fn parse_exposure_ms(record: &[u8]) -> Option<f32> {
    let inner = walk_fields(record).ok()?;
    let mut vals: Vec<u64> = Vec::new();
    for f in &inner {
        if f.field_num != 1 { continue; }
        match f.wire {
            WireType::Varint => vals.push(f.varint),
            WireType::LengthDelimited => {
                let b = f.bytes(record);
                let mut cursor = 0usize;
                while cursor < b.len() {
                    match read_varint(b, &mut cursor) {
                        Ok(v) => vals.push(v),
                        Err(_) => break,
                    }
                }
            }
            _ => {}
        }
    }
    if vals.len() < 2 || vals[0] == 0 || vals[1] == 0 { return None; }
    let ms = vals[0] as f64 / vals[1] as f64 * 1000.0;
    (ms.is_finite() && ms > 0.0 && ms < 1000.0).then_some(ms as f32)
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
    fn frame_exposure_and_timestamp_offset_parse() {
        fn varint(buf: &mut Vec<u8>, mut v: u64) {
            loop {
                let b = (v & 0x7F) as u8;
                v >>= 7;
                if v == 0 { buf.push(b); break; } else { buf.push(b | 0x80); }
            }
        }
        fn ld(buf: &mut Vec<u8>, field_num: u32, inner: &[u8]) {
            varint(buf, ((field_num << 3) | 2) as u64);
            varint(buf, inner.len() as u64);
            buf.extend_from_slice(inner);
        }
        fn vi(buf: &mut Vec<u8>, field_num: u32, v: u64) {
            varint(buf, (field_num << 3) as u64);
            varint(buf, v);
        }
        fn f32s(buf: &mut Vec<u8>, vals: &[f32]) {
            for (i, v) in vals.iter().enumerate() {
                varint(buf, (((i as u32 + 1) << 3) | 5) as u64);
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Frame header: [1.2] = 64-bit video timestamp.
        let mut hdr = Vec::new();
        vi(&mut hdr, 1, 200);
        vi(&mut hdr, 2, 5_438_584_595);
        // Orientation: [2.4.1] = packed {1, 520} (1/520 s), [2.9] = quat.
        let mut packed = Vec::new();
        varint(&mut packed, 1);
        varint(&mut packed, 520);
        let mut expo = Vec::new();
        ld(&mut expo, 1, &packed);
        let mut quat = Vec::new();
        f32s(&mut quat, &[1.0, 0.0, 0.0, 0.0]);
        let mut orient = Vec::new();
        ld(&mut orient, 4, &expo);
        ld(&mut orient, 9, &quat);
        // IMU: [3.2.1.1] = 32-bit block timestamp, [3.2.1.3] = samples.
        let mut lens_payload = Vec::new();
        vi(&mut lens_payload, 1, 1_143_616_793);
        vi(&mut lens_payload, 2, 5169);
        ld(&mut lens_payload, 3, &quat);
        ld(&mut lens_payload, 3, &quat);
        let mut lens_arrays = Vec::new();
        ld(&mut lens_arrays, 1, &lens_payload);
        let mut imu = Vec::new();
        ld(&mut imu, 2, &lens_arrays);
        let mut frame = Vec::new();
        ld(&mut frame, 1, &hdr);
        ld(&mut frame, 2, &orient);
        ld(&mut frame, 3, &imu);
        let mut blob = Vec::new();
        ld(&mut blob, 3, &frame);
        // A second frame without an exposure record → NaN, but still aligned.
        let mut frame2 = Vec::new();
        ld(&mut frame2, 2, &{ let mut o = Vec::new(); ld(&mut o, 9, &quat); o });
        ld(&mut frame2, 3, &imu);
        ld(&mut blob, 3, &frame2);

        let parsed = DjiOsvImu::parse(&blob).expect("parse");
        assert_eq!(parsed.frame_quats.len(), 2);
        assert_eq!(parsed.exposure_ms.len(), 2);
        assert_eq!(parsed.frame_ts_offset_ms.len(), 2);
        assert!((parsed.exposure_ms[0] - 1000.0 / 520.0).abs() < 1e-3);
        // 5_438_584_595 − 1_143_616_793 = 2^32 + 506 → 0.506 ms modulo 2^32.
        assert!((parsed.frame_ts_offset_ms[0] - 0.506).abs() < 1e-4);
        assert!(parsed.exposure_ms[1].is_nan());
        assert!(parsed.frame_ts_offset_ms[1].is_nan());
        assert_eq!(parsed.high_rate_quats[0].len(), 2);
        // OSMO 360 II encoding: {100000, 79298304}.
        let mut packed2 = Vec::new();
        varint(&mut packed2, 100_000);
        varint(&mut packed2, 79_298_304);
        let mut expo2 = Vec::new();
        ld(&mut expo2, 1, &packed2);
        assert!((parse_exposure_ms(&expo2).unwrap() - 1.2611).abs() < 1e-3);
        // Unpacked repeated varints decode the same way.
        let mut expo3 = Vec::new();
        vi(&mut expo3, 1, 1);
        vi(&mut expo3, 1, 120);
        assert!((parse_exposure_ms(&expo3).unwrap() - 8.3333).abs() < 1e-3);
    }

    #[test]
    fn readout_follows_the_header_line_time() {
        fn varint(buf: &mut Vec<u8>, mut v: u64) {
            loop {
                let b = (v & 0x7F) as u8;
                v >>= 7;
                if v == 0 { buf.push(b); break; } else { buf.push(b | 0x80); }
            }
        }
        fn ld(buf: &mut Vec<u8>, field_num: u32, inner: &[u8]) {
            varint(buf, ((field_num << 3) | 2) as u64);
            varint(buf, inner.len() as u64);
            buf.extend_from_slice(inner);
        }
        fn vi(buf: &mut Vec<u8>, field_num: u32, v: u64) {
            varint(buf, (field_num << 3) as u64);
            varint(buf, v);
        }
        // OSMO 360: [1.11.1] = fps as f32 (ignored), [1.13.1] = 4766 ns.
        let mut fps = Vec::new();
        varint(&mut fps, (1 << 3) | 5);
        fps.extend_from_slice(&24.9988_f32.to_le_bytes());
        let mut lt = Vec::new();
        vi(&mut lt, 1, 4766);
        let mut hdr = Vec::new();
        ld(&mut hdr, 11, &fps);
        ld(&mut hdr, 13, &lt);
        let mut blob = Vec::new();
        ld(&mut blob, 1, &hdr);
        let parsed = DjiOsvImu::parse(&blob).expect("parse");
        assert_eq!(parsed.line_time_ns, Some(4766.0));
        assert!((parsed.readout_ms.unwrap() - 18.301).abs() < 1e-3);
        // 360 II: [1.11.1] = 4183 ns.
        let mut lt2 = Vec::new();
        vi(&mut lt2, 1, 4183);
        let mut hdr2 = Vec::new();
        ld(&mut hdr2, 11, &lt2);
        let mut blob2 = Vec::new();
        ld(&mut blob2, 1, &hdr2);
        let parsed2 = DjiOsvImu::parse(&blob2).expect("parse");
        assert!((parsed2.readout_ms.unwrap() - 16.063).abs() < 1e-2);
        // No header → no readout (callers fall back to the per-mode table).
        assert!(DjiOsvImu::parse(&[]).expect("parse").readout_ms.is_none());
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
