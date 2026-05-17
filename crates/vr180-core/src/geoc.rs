//! GEOC lens calibration parsing (KLNS Kannala-Brandt coefficients,
//! CTRX/CTRY principal point offsets, CALW/CALH sensor dimensions).
//!
//! GoPro stores GEOC in the file tail as a separate atom block.
//! The Python implementation in `vr180_gui.py::parse_geoc` reads
//! the last 1 MiB of the file and walks GPMF-style records.
//!
//! Confirmed stream→lens mapping (from project memory):
//!   - s0 → FRNT lens → use `FRNT.KLNS` (right eye after yaw mod)
//!   - s4 → BACK lens → use `BACK.KLNS` (left eye after yaw mod)
//!
//! ## Format quirks
//!
//! GEOC is encoded GPMF-style (fourcc / type / struct_size / count) but
//! lives outside the main GPMF metadata stream — it's an atom near the
//! end of the file, near the `udta` / `meta` block. We mirror the
//! Python code's approach: read the last 1 MiB, find `GEOC`, walk the
//! ~2 KB that follows.
//!
//! Records of interest:
//! - `DVID` ('c'): "BACK", "FRNT", "USRM", "HLMT" — selects which
//!   lens's records the following fourcc's belong to.
//! - `KLNS` ('d', struct_size=40, count=1): 5×f64 Kannala-Brandt
//!   polynomial coefficients `[c0, c1, c2, c3, c4]` defining the
//!   fisheye radius as a function of polar angle θ:
//!     `r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹`
//! - `CTRX`, `CTRY` ('d'): principal point offset from sensor center
//!   (pixels). For GoPro Max, typically a few px from center.
//! - `CALW`, `CALH` ('L', 'd'): calibration sensor dimensions
//!   (4216 for GoPro Max).

use std::path::Path;

/// Per-lens calibration.
#[derive(Debug, Clone)]
pub struct LensCal {
    /// Kannala-Brandt polynomial coefficients: `r = c0·θ + c1·θ³ + c2·θ⁵ + c3·θ⁷ + c4·θ⁹`.
    pub klns: [f64; 5],
    /// Principal point X offset from sensor center, pixels.
    pub ctrx: f32,
    /// Principal point Y offset from sensor center, pixels.
    pub ctry: f32,
}

/// Parsed GEOC for one .360 file.
#[derive(Debug, Clone)]
pub struct Geoc {
    /// Sensor calibration dimension (square; 4216 for GoPro Max).
    pub cal_dim: u32,
    /// FRNT lens (= s0 stream, = right eye after yaw mod).
    pub front: Option<LensCal>,
    /// BACK lens (= s4 stream, = left eye after yaw mod).
    pub back: Option<LensCal>,
}

impl Geoc {
    /// Pick the calibration for an eye, mapping right→FRNT, left→BACK.
    pub fn for_eye(&self, eye: Eye) -> Option<&LensCal> {
        match eye {
            Eye::Right => self.front.as_ref(),
            Eye::Left  => self.back.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye { Left, Right }

/// Parse GEOC from the file tail of a `.360`.
///
/// Returns `Ok(None)` if no `GEOC` marker is present (not a calibrated
/// GoPro file or a non-`.360` container that happens to share the
/// extension). Returns `Ok(Some(_))` with at least one lens populated
/// on success.
pub fn parse_geoc(path: &Path) -> std::io::Result<Option<Geoc>> {
    use std::io::{Read, Seek, SeekFrom};
    const TAIL_SIZE: u64 = 1024 * 1024;

    let mut f = std::fs::File::open(path)?;
    let file_size = f.metadata()?.len();
    let read_size = TAIL_SIZE.min(file_size);
    let start = file_size - read_size;
    f.seek(SeekFrom::Start(start))?;
    let mut tail = vec![0u8; read_size as usize];
    f.read_exact(&mut tail)?;

    Ok(parse_geoc_bytes(&tail))
}

/// Locate the SROT (Sensor Read-Out Time) value in raw bytes.
///
/// SROT can live in either the GPMF telemetry stream (between samples
/// from a given stream) or in the file tail's geometry-calibration
/// block (alongside GEOC). Both encodings are GPMF-style records:
///
/// - `'J'` / 8 bytes: uint64 microseconds → ms (÷ 1000)
/// - `'L'` / 4 bytes: uint32 microseconds → ms (÷ 1000)
/// - `'f'` / 4 bytes: float32 ms (or μs if `> 1000`)
///
/// Returns the SROT in **milliseconds** if found; `None` otherwise.
/// Caller divides by 1000 for seconds.
///
/// Mirrors `vr180_gui.py::parse_srot` (lines 3939-3978).
pub fn find_srot_ms(data: &[u8]) -> Option<f32> {
    let mut idx = 0usize;
    while idx + 8 <= data.len() {
        let Some(found) = find_subslice(&data[idx..], b"SROT") else { return None; };
        let pos = idx + found;
        if pos + 8 > data.len() {
            return None;
        }
        let type_char = data[pos + 4];
        let struct_size = data[pos + 5] as usize;
        let count = u16::from_be_bytes([data[pos + 6], data[pos + 7]]) as usize;

        if count < 1 || pos + 8 + struct_size > data.len() {
            idx = pos + 1;
            continue;
        }
        let payload = &data[pos + 8..pos + 8 + struct_size];

        match (type_char, struct_size) {
            (b'J', 8) => {
                let micros = u64::from_be_bytes(payload.try_into().ok()?);
                return Some(micros as f32 / 1000.0);  // μs → ms
            }
            (b'L', 4) => {
                let micros = u32::from_be_bytes(payload.try_into().ok()?);
                return Some(micros as f32 / 1000.0);
            }
            (b'f', 4) => {
                let v = f32::from_be_bytes(payload.try_into().ok()?);
                // Heuristic from Python: > 1000 implies μs were
                // mis-typed as ms, divide; otherwise the value is
                // already ms.
                return Some(if v > 1000.0 { v / 1000.0 } else { v });
            }
            _ => {
                // Some other field happens to spell `SROT`; keep
                // searching past this position.
                idx = pos + 1;
                continue;
            }
        }
    }
    None
}

/// Look up the SROT for one .360 file. Tries (in order):
/// 1. The optional GPMF bytes (cheap — caller usually already has them).
/// 2. The last 512 KiB of the file (where GEOC lives).
///
/// Returns SROT in **seconds**. `None` if not found in either place;
/// caller should fall back to a sensible default (15.224 ms = the
/// GoPro Max constant).
pub fn lookup_srot_s(
    path: &Path,
    gpmf: Option<&[u8]>,
) -> std::io::Result<Option<f32>> {
    if let Some(g) = gpmf {
        if let Some(ms) = find_srot_ms(g) {
            return Ok(Some(ms / 1000.0));
        }
    }
    use std::io::{Read, Seek, SeekFrom};
    const TAIL_SIZE: u64 = 512 * 1024;
    let mut f = std::fs::File::open(path)?;
    let file_size = f.metadata()?.len();
    let read_size = TAIL_SIZE.min(file_size);
    let start = file_size - read_size;
    f.seek(SeekFrom::Start(start))?;
    let mut tail = vec![0u8; read_size as usize];
    f.read_exact(&mut tail)?;
    Ok(find_srot_ms(&tail).map(|ms| ms / 1000.0))
}

/// Parse GEOC from raw tail bytes. Returns `None` if no `GEOC` marker
/// is found. Public to make unit testing easy without touching disk.
pub fn parse_geoc_bytes(tail: &[u8]) -> Option<Geoc> {
    let idx = find_subslice(tail, b"GEOC")?;
    if idx < 8 { return None; }

    let mut front = LensCalBuilder::default();
    let mut back  = LensCalBuilder::default();
    let mut cal_dim: u32 = 4216;  // GoPro Max default if not found
    let mut current = Section::Global;

    let mut pos = idx - 8;
    let end = (idx + 2000).min(tail.len());
    while pos + 8 <= end {
        let fourcc = &tail[pos..pos + 4];
        let type_char = tail[pos + 4];
        let struct_size = tail[pos + 5] as usize;
        let cnt = u16::from_be_bytes([tail[pos + 6], tail[pos + 7]]) as usize;
        let payload_size = struct_size * cnt;
        let padded = (payload_size + 3) & !3;

        // `DEVC` is a container — skip its 8-byte header without
        // advancing past its children (same trick the Python parser uses).
        if fourcc == b"DEVC" {
            pos += 8;
            continue;
        }

        if pos + 8 + payload_size > tail.len() {
            break;
        }
        let payload = &tail[pos + 8..pos + 8 + payload_size];

        // Section selector: DVID payload picks which lens the
        // following records belong to. "USRM" / "HLMT" → return
        // to the "outside any lens" state. The payload type can be
        // either `'F'` (4-byte fourcc, the in-the-wild encoding for
        // GoPro `.360` files) or `'c'` (general ASCII string, the
        // synthetic-test encoding) — handle both.
        if fourcc == b"DVID" && (type_char == b'F' || type_char == b'c') {
            let s: String = payload.iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char).collect();
            current = match s.as_str() {
                "FRNT" => Section::Front,
                "BACK" => Section::Back,
                _      => Section::Global,
            };
        }
        // CALW / CALH (sensor calibration dimension).
        else if (fourcc == b"CALW" || fourcc == b"CALH") && cnt == 1 {
            if let Some(v) = read_scalar(type_char, struct_size, payload) {
                cal_dim = v.round() as u32;
            }
        }
        // KLNS: 5×f64 polynomial coefficients.
        else if fourcc == b"KLNS" && struct_size == 40 && cnt == 1
            && payload.len() >= 40
        {
            let mut k = [0.0_f64; 5];
            for i in 0..5 {
                k[i] = f64::from_be_bytes(payload[i*8..(i+1)*8].try_into().unwrap());
            }
            match current {
                Section::Front  => back_or_front(&mut front, |b| b.klns = Some(k)),
                Section::Back   => back_or_front(&mut back,  |b| b.klns = Some(k)),
                Section::Global => {}
            }
        }
        // CTRX / CTRY: principal point offsets.
        else if (fourcc == b"CTRX" || fourcc == b"CTRY") && cnt == 1 {
            if let Some(v) = read_scalar(type_char, struct_size, payload) {
                let v = v as f32;
                let key = fourcc;
                match current {
                    Section::Front => back_or_front(&mut front, |b| {
                        if key == b"CTRX" { b.ctrx = Some(v); }
                        else              { b.ctry = Some(v); }
                    }),
                    Section::Back => back_or_front(&mut back, |b| {
                        if key == b"CTRX" { b.ctrx = Some(v); }
                        else              { b.ctry = Some(v); }
                    }),
                    Section::Global => {}
                }
            }
        }

        pos += 8 + padded;
    }

    let front = front.finish();
    let back  = back.finish();
    if front.is_none() && back.is_none() {
        return None;
    }
    Some(Geoc { cal_dim, front, back })
}

#[derive(Debug, Default)]
struct LensCalBuilder {
    klns: Option<[f64; 5]>,
    ctrx: Option<f32>,
    ctry: Option<f32>,
}

impl LensCalBuilder {
    fn finish(self) -> Option<LensCal> {
        Some(LensCal {
            klns: self.klns?,
            ctrx: self.ctrx.unwrap_or(0.0),
            ctry: self.ctry.unwrap_or(0.0),
        })
    }
}

fn back_or_front<F: FnOnce(&mut LensCalBuilder)>(b: &mut LensCalBuilder, f: F) {
    f(b);
}

#[derive(Debug, Clone, Copy)]
enum Section { Global, Front, Back }

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Parse a single scalar GPMF payload of the given type. Returns the
/// value as `f64` so any numeric type can flow through one path.
fn read_scalar(type_char: u8, struct_size: usize, payload: &[u8]) -> Option<f64> {
    match (type_char, struct_size) {
        (b'd', 8) if payload.len() >= 8 => Some(f64::from_be_bytes(payload[..8].try_into().ok()?)),
        (b'f', 4) if payload.len() >= 4 => Some(f32::from_be_bytes(payload[..4].try_into().ok()?) as f64),
        (b'L', 4) if payload.len() >= 4 => Some(u32::from_be_bytes(payload[..4].try_into().ok()?) as f64),
        (b'l', 4) if payload.len() >= 4 => Some(i32::from_be_bytes(payload[..4].try_into().ok()?) as f64),
        (b'S', 2) if payload.len() >= 2 => Some(u16::from_be_bytes([payload[0], payload[1]]) as f64),
        (b's', 2) if payload.len() >= 2 => Some(i16::from_be_bytes([payload[0], payload[1]]) as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a small GEOC blob: global header, FRNT section with
    /// KLNS + CTRX + CTRY, BACK section with same. Parse and check.
    #[test]
    fn parse_synthetic_geoc() {
        let mut buf: Vec<u8> = Vec::new();
        // 8-byte fake leading data so `pos = idx - 8` doesn't underflow.
        buf.extend_from_slice(&[0u8; 8]);
        // GEOC header (we treat the marker itself as just a tag).
        emit_record(&mut buf, b"GEOC", b'c', 1, 1, b"\x00");
        // Global: CALW = 4216.
        emit_record(&mut buf, b"CALW", b'L', 4, 1, &4216_u32.to_be_bytes());
        // FRNT section.
        emit_record(&mut buf, b"DVID", b'c', 4, 1, b"FRNT");
        let mut klns_payload = Vec::new();
        for v in &[1.1_f64, 2.2, 3.3, 4.4, 5.5] {
            klns_payload.extend_from_slice(&v.to_be_bytes());
        }
        emit_record(&mut buf, b"KLNS", b'd', 40, 1, &klns_payload);
        emit_record(&mut buf, b"CTRX", b'd', 8, 1, &(-2.5_f64).to_be_bytes());
        emit_record(&mut buf, b"CTRY", b'd', 8, 1, &(1.25_f64).to_be_bytes());
        // BACK section.
        emit_record(&mut buf, b"DVID", b'c', 4, 1, b"BACK");
        let mut klns_back = Vec::new();
        for v in &[10.0_f64, 20.0, 30.0, 40.0, 50.0] {
            klns_back.extend_from_slice(&v.to_be_bytes());
        }
        emit_record(&mut buf, b"KLNS", b'd', 40, 1, &klns_back);

        let g = parse_geoc_bytes(&buf).expect("parse failed");
        assert_eq!(g.cal_dim, 4216);
        let f = g.front.as_ref().expect("FRNT missing");
        assert!((f.klns[0] - 1.1).abs() < 1e-9);
        assert!((f.klns[4] - 5.5).abs() < 1e-9);
        assert!((f.ctrx - (-2.5)).abs() < 1e-5);
        assert!((f.ctry - 1.25).abs() < 1e-5);
        let b = g.back.as_ref().expect("BACK missing");
        assert!((b.klns[2] - 30.0).abs() < 1e-9);
        // BACK has no CTRX/CTRY → defaults to 0.
        assert_eq!(b.ctrx, 0.0);
    }

    #[test]
    fn no_geoc_returns_none() {
        let buf = vec![0u8; 4096];
        assert!(parse_geoc_bytes(&buf).is_none());
    }

    #[test]
    fn find_srot_uint64_microseconds() {
        // Build a record: "SROT" + 'J' (uint64) + struct_size=8 +
        // count=1 + payload = 15224 μs (15.224 ms).
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0u8; 16]);  // some prefix junk
        buf.extend_from_slice(b"SROT");
        buf.push(b'J');
        buf.push(8);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&(15224u64).to_be_bytes());
        let ms = find_srot_ms(&buf).expect("should find SROT");
        assert!((ms - 15.224).abs() < 1e-3, "got {} ms", ms);
    }

    #[test]
    fn find_srot_float_ms_direct() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"SROT");
        buf.push(b'f');
        buf.push(4);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&(12.5_f32).to_be_bytes());
        let ms = find_srot_ms(&buf).expect("should find SROT");
        assert!((ms - 12.5).abs() < 1e-4, "got {} ms", ms);
    }

    #[test]
    fn find_srot_returns_none_when_absent() {
        let buf = vec![0u8; 4096];
        assert!(find_srot_ms(&buf).is_none());
    }

    #[test]
    fn find_srot_skips_invalid_match() {
        // "SROT" appearing in junk bytes (not as a real GPMF record).
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"SROT");
        buf.push(b'Z');   // unknown type
        buf.push(99);
        buf.extend_from_slice(&[0u8, 0u8]);
        buf.extend_from_slice(&[0u8; 100]);
        // Then a real SROT entry.
        buf.extend_from_slice(b"SROT");
        buf.push(b'L');
        buf.push(4);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&(20000u32).to_be_bytes()); // 20 ms
        let ms = find_srot_ms(&buf).expect("should find real SROT past junk");
        assert!((ms - 20.0).abs() < 1e-4);
    }

    fn emit_record(
        out: &mut Vec<u8>,
        fourcc: &[u8; 4],
        type_char: u8,
        struct_size: u8,
        count: u16,
        payload: &[u8],
    ) {
        out.extend_from_slice(fourcc);
        out.push(type_char);
        out.push(struct_size);
        out.extend_from_slice(&count.to_be_bytes());
        out.extend_from_slice(payload);
        // Pad to 4 bytes.
        let pad = (4 - (payload.len() % 4)) % 4;
        for _ in 0..pad { out.push(0); }
    }
}
