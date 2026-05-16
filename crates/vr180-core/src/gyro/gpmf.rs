//! GPMF (GoPro Metadata Format) byte-stream walker.
//!
//! The .360 telemetry stream (track index typically 0:3) is a stream of
//! length-prefixed records. Each record has an 8-byte header:
//!
//! ```text
//!  offset 0..4   FourCC (4 ASCII bytes)         e.g. "CORI", "DEVC", "STMP"
//!  offset 4      type char (1 byte)             e.g. 's' int16, 'L' uint32
//!  offset 5      struct_size (1 byte)           bytes per element
//!  offset 6..8   repeat count (BE u16)          element count
//!  offset 8..    payload (struct_size × repeat) padded to 4-byte boundary
//! ```
//!
//! Two FourCCs (`DEVC`, `STRM`) are CONTAINERS — the parser only consumes
//! their 8-byte header and steps INTO their children. Everything else is
//! a leaf: advance by `8 + padded(struct_size × repeat)`.
//!
//! This walker is a streaming iterator that yields one [`GpmfEntry`] per
//! record (leaf or container). The caller decides what to do with them
//! — extract quaternions, gyro samples, scale factors, etc. — by matching
//! on the FourCC. Pure-Rust, no allocations beyond the input slice.
//!
//! See the [GPMF spec on GitHub](https://github.com/gopro/gpmf-parser)
//! for the full type-char list. The Python reference is
//! `parse_gyro_raw.py::parse_gpmf_entries`.

/// 4-byte FourCC identifier.
///
/// We keep this as raw bytes (not a `&str`) so non-ASCII garbage in
/// malformed streams doesn't cause UTF-8 errors mid-parse; the parser
/// skips records whose FourCC isn't valid ASCII printable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const fn new(s: &[u8; 4]) -> Self {
        FourCC(*s)
    }

    pub fn as_str(&self) -> &str {
        // We only construct FourCC after validating bytes are printable
        // ASCII, so this never panics in practice.
        std::str::from_utf8(&self.0).unwrap_or("???")
    }

    /// True if all four bytes are printable ASCII (`0x20..=0x7e`).
    fn is_valid(b: [u8; 4]) -> bool {
        b.iter().all(|&c| (0x20..=0x7e).contains(&c))
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Container FourCCs — these are stepped INTO rather than skipped over.
const DEVC: FourCC = FourCC::new(b"DEVC");
const STRM: FourCC = FourCC::new(b"STRM");

/// A single GPMF record (header + borrowed payload).
#[derive(Debug, Clone, Copy)]
pub struct GpmfEntry<'a> {
    pub fourcc: FourCC,
    /// Type char (e.g. `b's'` for int16, `b'L'` for uint32, `b'f'` for f32).
    pub type_char: u8,
    /// Bytes per element.
    pub struct_size: u8,
    /// Number of elements.
    pub repeat: u16,
    /// Raw payload slice (`struct_size × repeat` bytes).
    pub payload: &'a [u8],
    /// True if this is a container record (DEVC / STRM). The payload of
    /// a container is the concatenation of its child records.
    pub is_container: bool,
}

impl<'a> GpmfEntry<'a> {
    pub fn payload_size(&self) -> usize {
        self.struct_size as usize * self.repeat as usize
    }
}

/// Streaming iterator over GPMF records in a byte slice.
///
/// Yields one entry per record. The walker handles DEVC/STRM container
/// traversal automatically — when it encounters a container, the next
/// `next()` call yields the container's first child, not the next
/// sibling. This matches how the Python reference walks the stream.
#[derive(Debug, Clone)]
pub struct GpmfWalker<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GpmfWalker<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Length of the GPMF blob being walked.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> Iterator for GpmfWalker<'a> {
    type Item = GpmfEntry<'a>;

    fn next(&mut self) -> Option<GpmfEntry<'a>> {
        while self.pos + 8 <= self.data.len() {
            let p = self.pos;
            let fourcc_bytes: [u8; 4] = self.data[p..p + 4].try_into().ok()?;

            // Skip records whose FourCC isn't printable ASCII. The Python
            // reference advances by 1 byte in this case (lib-paranoid resync).
            if !FourCC::is_valid(fourcc_bytes) {
                self.pos += 1;
                continue;
            }

            let fourcc = FourCC(fourcc_bytes);
            let type_char = self.data[p + 4];
            let struct_size = self.data[p + 5];
            let repeat = u16::from_be_bytes([self.data[p + 6], self.data[p + 7]]);
            let payload_size = struct_size as usize * repeat as usize;
            let padded = (payload_size + 3) & !3;
            let payload_end = (p + 8 + payload_size).min(self.data.len());
            let payload = &self.data[p + 8..payload_end];

            let is_container = fourcc == DEVC || fourcc == STRM;
            if is_container {
                // Step INTO the container — its body is the next entries.
                self.pos = p + 8;
            } else {
                self.pos = p + 8 + padded;
            }

            return Some(GpmfEntry {
                fourcc,
                type_char,
                struct_size,
                repeat,
                payload,
                is_container,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the bundled GPMF fixture (~314 KiB, extracted from a real
    /// GoPro Max .360 clip via `ffmpeg -map 0:3 -c copy -f rawvideo`).
    fn fixture() -> Vec<u8> {
        std::fs::read("tests/fixtures/GS010172.gpmf")
            .expect("fixture missing — see crates/vr180-core/tests/fixtures/README")
    }

    #[test]
    fn first_record_is_devc() {
        let data = fixture();
        let first = GpmfWalker::new(&data).next().expect("at least one entry");
        assert_eq!(first.fourcc.as_str(), "DEVC");
        assert!(first.is_container);
    }

    #[test]
    fn walker_finds_expected_containers_and_leaves() {
        // Expected structural counts (one record per ~1s of footage, 30s clip):
        //   30 DEVC, 30 CORI, 30 IORI, 30 GYRO, 30 ACCL, 30 GRAV, 30 MNOR.
        //
        // Note: `grep -aoc "MNOR"` on the raw file returns 60 — that's
        // false-positive byte hits INSIDE payloads (e.g. the literal "MNOR"
        // sequence appearing inside a SCAL or GYRO sample payload). The
        // walker's count is the correct structural one because it only
        // emits entries at record boundaries; that's also why the Python
        // parser validates `type_char` and `struct_size` before treating an
        // entry as a real MNOR record.
        let data = fixture();
        let mut counts = std::collections::HashMap::<[u8; 4], usize>::new();
        for entry in GpmfWalker::new(&data) {
            *counts.entry(entry.fourcc.0).or_insert(0) += 1;
        }
        let n = |s: &[u8; 4]| counts.get(s).copied().unwrap_or(0);
        assert_eq!(n(b"DEVC"), 30, "DEVC count");
        assert_eq!(n(b"CORI"), 30, "CORI count");
        assert_eq!(n(b"IORI"), 30, "IORI count");
        assert_eq!(n(b"GYRO"), 30, "GYRO count");
        assert_eq!(n(b"MNOR"), 30, "MNOR count");
    }
}
