//! Color pipeline math.
//!
//! Phase 0.7 deliverable: `.cube` 3D LUT parser (the headline feature
//! — every user-facing color grade in this app starts with a LUT).
//!
//! Deferred to 0.7.5:
//!   - 1D CDL LUT builder (lift/gamma/gain/shadow/highlight + the
//!     smoothstep tonal-zone masks)
//!   - temp/tint per-channel multiply
//!   - mid-detail clarity (Gaussian-blur-of-downsample-upsample,
//!     multi-pass, weighted by midtone bell curve)
//!
//! Order (from Python `apply_export_post`), for when those land:
//!   1. 1D LUT (lift/gamma/gain + shadow/highlight zone masks)
//!   2. 3D LUT (this phase) ← creative grade
//!   3. temp / tint
//!   4. mid-detail clarity
//!   5. saturation

use std::path::Path;

/// A 3D color lookup table loaded from a `.cube` file.
///
/// Stored as a flat `Vec<f32>` in slowest-to-fastest axis order
/// **R → G → B**, with **B varying fastest** — which matches the
/// `.cube` file spec (per-line is one R,G,B triplet, and the file
/// lists triplets in B-then-G-then-R nested-loop order, i.e. for
/// each R: for each G: for each B: write R,G,B). That same memory
/// layout uploads cleanly to a wgpu `TextureDimension::D3` texture
/// where the texel at coordinate `(b, g, r)` (texture coords are
/// always XYZ = innermost-to-outermost) is what we sample.
///
/// Each entry is one f32 RGB triplet; total length = `size³ × 3`.
#[derive(Debug, Clone)]
pub struct Cube3DLut {
    /// Per-axis cube length. Typical: 16, 32, 33, 64.
    pub size: u32,
    /// `size³ × 3` floats. See struct doc for layout.
    pub data: Vec<f32>,
    pub title: Option<String>,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
}

impl Cube3DLut {
    /// Parse a `.cube` file from disk. Permissive: ignores comments,
    /// blank lines, and unknown header keywords. Recognises
    /// `LUT_3D_SIZE`, `TITLE`, `DOMAIN_MIN`, `DOMAIN_MAX`. Returns an
    /// error if no `LUT_3D_SIZE` line is found and the data length
    /// isn't a perfect cube.
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_str(&text).map_err(|e| std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("parse {}: {e}", path.display())
        ))
    }

    /// Parse a `.cube` file from a string buffer.
    pub fn from_str(text: &str) -> Result<Self, String> {
        let mut size: Option<u32> = None;
        let mut title: Option<String> = None;
        let mut domain_min = [0.0_f32; 3];
        let mut domain_max = [1.0_f32; 3];
        let mut data = Vec::<f32>::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                size = rest.trim().parse::<u32>().ok();
                continue;
            }
            if let Some(rest) = line.strip_prefix("TITLE") {
                title = Some(rest.trim().trim_matches('"').to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
                let parts: Vec<f32> = rest.split_whitespace()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if parts.len() == 3 { domain_min = [parts[0], parts[1], parts[2]]; }
                continue;
            }
            if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
                let parts: Vec<f32> = rest.split_whitespace()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if parts.len() == 3 { domain_max = [parts[0], parts[1], parts[2]]; }
                continue;
            }
            // Data row. Skip silently on parse failure (some .cube files
            // have stray header keywords we don't recognise).
            let parts: Vec<f32> = line.split_whitespace()
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();
            if parts.len() >= 3 {
                data.push(parts[0]);
                data.push(parts[1]);
                data.push(parts[2]);
            }
        }

        let triplet_count = data.len() / 3;
        let resolved_size = match size {
            Some(n) => n,
            None => {
                // Derive from cube root of triplet count.
                let n = (triplet_count as f64).cbrt().round() as u32;
                if (n as usize).pow(3) != triplet_count {
                    return Err(format!(
                        "no LUT_3D_SIZE and {triplet_count} triplets is not a perfect cube"
                    ));
                }
                n
            }
        };
        let want = (resolved_size as usize).pow(3) * 3;
        if data.len() != want {
            return Err(format!(
                "LUT_3D_SIZE={resolved_size} expects {want} floats, got {}", data.len()
            ));
        }
        Ok(Cube3DLut {
            size: resolved_size,
            data,
            title,
            domain_min,
            domain_max,
        })
    }

    /// Repack this LUT for a wgpu 3D texture upload.
    ///
    /// wgpu storage / sampled 3D textures take `Rgba8Unorm` (no RGB-
    /// only 3-byte format), so we widen each f32 triplet to RGBA u8
    /// by saturating cast to `[0, 255]`. Alpha is 255.
    ///
    /// Returns `(rgba_bytes, size)` ready for `Queue::write_texture`.
    pub fn to_rgba8_for_upload(&self) -> Vec<u8> {
        let n = (self.size as usize).pow(3);
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let r = (self.data[i * 3]    .clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (self.data[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (self.data[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            out.push(r); out.push(g); out.push(b); out.push(255);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity LUT_3D_SIZE=2: each corner is its own input coord.
    #[test]
    fn parses_identity_2x2x2() {
        let s = r#"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"#;
        let lut = Cube3DLut::from_str(s).unwrap();
        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 8 * 3);
        // 1st triplet = (0,0,0), last = (1,1,1)
        assert_eq!(&lut.data[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&lut.data[21..24], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn skips_comments_blank_lines_and_title() {
        let s = "# comment\nTITLE \"hello\"\nLUT_3D_SIZE 2\n\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = Cube3DLut::from_str(s).unwrap();
        assert_eq!(lut.size, 2);
        assert_eq!(lut.title.as_deref(), Some("hello"));
    }

    #[test]
    fn errors_on_truncated_data() {
        let s = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n";
        let err = Cube3DLut::from_str(s).unwrap_err();
        assert!(err.contains("expects 24 floats, got 6"), "err: {err}");
    }

    #[test]
    fn rgba8_upload_buffer_size_matches_cube() {
        let s = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = Cube3DLut::from_str(s).unwrap();
        let bytes = lut.to_rgba8_for_upload();
        assert_eq!(bytes.len(), 8 * 4);
    }

    /// Smoke test against the actual bundled LUT.
    #[test]
    fn parses_bundled_gplog_lut() {
        let path = std::path::Path::new("../../assets/Recommended Lut GPLOG.cube");
        if !path.exists() {
            // CI / out-of-tree run: skip.
            eprintln!("skipping bundled-LUT test, file not found");
            return;
        }
        let lut = Cube3DLut::from_file(path).expect("parse bundled LUT");
        // GoPro GP-Log LUTs are typically 33-cubed.
        assert!([16, 17, 32, 33, 64].contains(&lut.size),
            "bundled LUT size {} is unusual", lut.size);
        assert_eq!(lut.data.len(), (lut.size as usize).pow(3) * 3);
    }
}
