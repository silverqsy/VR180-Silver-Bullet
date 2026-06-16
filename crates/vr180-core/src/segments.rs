//! Multi-segment chain detection for GoPro recordings.
//!
//! GoPro cameras split long recordings into ~4 GB chapter files at the
//! FAT32 boundary. Each segment is named `G{PFX}{cc}{IIII}.{ext}` where:
//!
//! - `PFX` ∈ `S`, `H`, `X`, `L`, `P` — recording profile / type
//! - `cc` — chapter number, `01..99`
//! - `IIII` — recording ID (shared across all chapters of one capture)
//! - `ext` — `360`, `MP4`, `MOV`, `LRV`, ... (case-insensitive)
//!
//! Naming patterns we care about:
//! - `.360`: `GS01XXXX.360`, `GH01XXXX.360`
//! - `.MP4`: `GX01XXXX.MP4`, `GH01XXXX.MP4`, `GP01XXXX.MP4`
//! - `.MOV`: same prefixes
//! - `.LRV`: `GL01XXXX.LRV` (low-res preview)
//!
//! Given any one segment's path, [`detect_segments`] returns the full
//! sorted list of segment paths for the same recording. A standalone
//! file (or any non-GoPro path) returns `vec![path]`.
//!
//! Port of `vr180_gui.py::detect_gopro_segments` (lines 3496-3530).

use std::path::{Path, PathBuf};

/// Detect all chapter segments of a GoPro recording from any one
/// segment's path. Returns the sorted list of all existing segments
/// (chapter 01, 02, 03, ...). For a non-GoPro-formatted filename or
/// a standalone file, returns `vec![path.to_path_buf()]`.
///
/// Implementation:
/// 1. Match the filename against `G[SHXLP]{2-digit chapter}{4-digit
///    recording id}.{ext}` (case-insensitive).
/// 2. If no match → return `[path]`.
/// 3. Else iterate chapter 1..99, probe each candidate path. Stop
///    iterating after 5 consecutive missing chapters past the input
///    chapter (a recording is rarely more than ~50 chapters in
///    practice but the 99 upper bound is the format limit).
/// 4. Sort ascending by chapter and return.
pub fn detect_segments(path: &Path) -> Vec<PathBuf> {
    let path = path.to_path_buf();
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return vec![path];
    };
    let Some(parsed) = parse_segment_name(name) else {
        return vec![path];
    };

    let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let mut found: Vec<(u32, PathBuf)> = Vec::new();
    let mut missing_streak = 0u32;
    for chapter in 1..=99u32 {
        let candidate = parent.join(format!(
            "{}{:02}{}.{}", parsed.prefix, chapter, parsed.rec_id, parsed.ext_original,
        ));
        if candidate.is_file() {
            found.push((chapter, candidate));
            missing_streak = 0;
        } else {
            // Stop early if we've passed the input chapter by 5 and
            // haven't seen anything in a while (recordings can have
            // gaps if intermediate chapters were deleted, but the
            // input chapter is by definition present).
            if chapter > parsed.chapter {
                missing_streak += 1;
                if missing_streak > 5 {
                    break;
                }
            }
        }
    }

    if found.is_empty() {
        // Defensive: the input path matched the pattern but doesn't
        // exist on disk (or has a permission error). Return as-is
        // so downstream code surfaces the open error with a useful path.
        return vec![path];
    }
    // The list is already sorted because we iterated 1..=99.
    found.into_iter().map(|(_, p)| p).collect()
}

#[derive(Debug, Clone)]
struct ParsedName {
    /// `"GS"`, `"GH"`, ... — first two chars of the file name, preserved case.
    prefix: String,
    /// Chapter number from the filename (1-99).
    chapter: u32,
    /// 4-digit recording ID as a string (preserves leading zeros).
    rec_id: String,
    /// Extension preserved exactly as written (case + value).
    ext_original: String,
}

/// Parse a GoPro filename. Returns `None` if the name doesn't match
/// the `G{PFX}{cc}{IIII}.{ext}` pattern.
fn parse_segment_name(name: &str) -> Option<ParsedName> {
    let bytes = name.as_bytes();
    // Minimum length: G + 1 prefix + 2 chapter + 4 rec_id + 1 dot + 1 ext = 10
    if bytes.len() < 10 { return None; }
    // First char must be 'G' or 'g'.
    if !bytes[0].eq_ignore_ascii_case(&b'G') { return None; }
    // Second char must be one of S/H/X/L/P (case-insensitive).
    let pfx_b = bytes[1].to_ascii_uppercase();
    if !matches!(pfx_b, b'S' | b'H' | b'X' | b'L' | b'P') { return None; }
    // Chapter: bytes 2..4 are digits.
    if !bytes[2].is_ascii_digit() || !bytes[3].is_ascii_digit() { return None; }
    let chapter: u32 = (bytes[2] - b'0') as u32 * 10 + (bytes[3] - b'0') as u32;
    if chapter == 0 { return None; }   // chapter must be at least 1
    // rec_id: bytes 4..8 are digits.
    for i in 4..8 {
        if !bytes[i].is_ascii_digit() { return None; }
    }
    // The dot at position 8.
    if bytes[8] != b'.' { return None; }
    // Extension is the rest (at least one char).
    if bytes.len() < 10 { return None; }

    Some(ParsedName {
        prefix: format!("{}{}", bytes[0] as char, bytes[1] as char),
        chapter,
        rec_id: std::str::from_utf8(&bytes[4..8]).ok()?.to_string(),
        ext_original: std::str::from_utf8(&bytes[9..]).ok()?.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_gopro_names() {
        let p = parse_segment_name("GS010172.360").unwrap();
        assert_eq!(p.prefix, "GS");
        assert_eq!(p.chapter, 1);
        assert_eq!(p.rec_id, "0172");
        assert_eq!(p.ext_original, "360");

        let p = parse_segment_name("GH020172.360").unwrap();
        assert_eq!(p.chapter, 2);

        let p = parse_segment_name("GX150042.MP4").unwrap();
        assert_eq!(p.prefix, "GX");
        assert_eq!(p.chapter, 15);
        assert_eq!(p.ext_original, "MP4");

        let p = parse_segment_name("GL010172.LRV").unwrap();
        assert_eq!(p.prefix, "GL");
    }

    #[test]
    fn rejects_non_gopro_names() {
        assert!(parse_segment_name("foo.mp4").is_none());
        assert!(parse_segment_name("GZ010172.360").is_none(),
            "Z is not a GoPro recording-type prefix");
        assert!(parse_segment_name("GS001A0172.360").is_none(),
            "chapter must be 2 digits");
        assert!(parse_segment_name("GS00ABCD.360").is_none(),
            "rec_id must be digits");
        assert!(parse_segment_name("GS010172_extra.360").is_none(),
            "no underscore allowed in stem");
        assert!(parse_segment_name("").is_none());
        // chapter 00 → reject (chapters start at 01).
        assert!(parse_segment_name("GS000172.360").is_none());
    }

    #[test]
    fn detect_returns_path_for_non_gopro_input() {
        let tmp = std::env::temp_dir().join("not-a-gopro.mov");
        std::fs::write(&tmp, b"").unwrap();
        let r = detect_segments(&tmp);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], tmp);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn detect_finds_chain_when_segments_exist() {
        // Build a fake 3-segment chain in temp.
        let dir = std::env::temp_dir().join("vr180_segment_test_abcd1234");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for chapter in [1, 2, 3] {
            let name = format!("GS{:02}9999.360", chapter);
            std::fs::write(dir.join(&name), b"").unwrap();
        }
        // Start from chapter 2 → should find all 3.
        let chain = detect_segments(&dir.join("GS029999.360"));
        assert_eq!(chain.len(), 3);
        assert!(chain[0].ends_with("GS019999.360"));
        assert!(chain[1].ends_with("GS029999.360"));
        assert!(chain[2].ends_with("GS039999.360"));
        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detect_stops_after_5_consecutive_missing() {
        let dir = std::env::temp_dir().join("vr180_seg_gap_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // chapter 01 only.
        std::fs::write(dir.join("GS018888.360"), b"").unwrap();
        let chain = detect_segments(&dir.join("GS018888.360"));
        assert_eq!(chain.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
