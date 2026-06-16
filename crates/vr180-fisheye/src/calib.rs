//! Camera calibration: `FisheyeCalibration` + Gyroflow lens-profile loader.
//!
//! `FisheyeCalibration` is the parameter set that defines a Kannala-
//! Brandt fisheye lens at one working resolution. It's what the GPU
//! shader, the GUI sliders, and the preset library all speak.
//!
//! Gyroflow lens profiles are JSON files like `bmci 4096.json` or
//! `cine immersive 8192 7200.json` — community-maintained calibrations
//! for popular cameras. They use the same KB-4 model we do.

use crate::projection::{kb_inverse, THETA_TRANS_DEFAULT};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Per-eye fisheye lens calibration.
///
/// This is the parameter set the WGSL shader consumes (one struct per
/// eye, packed into a uniform buffer). Mirrors the Metal kernel's
/// per-eye calib slice at `vr180_gui.py:1350-1400`.
///
/// Field naming follows the Gyroflow / OpenCV `cv::fisheye` convention,
/// NOT the GoPro GEOC convention (which folds fx into the polynomial
/// coefficients as `c0..c5`). The conversion is:
///
/// ```text
/// GEOC c0  =  fx          (linear term)
/// GEOC c1  =  fx · k1     (cubic term in θ)
/// GEOC c2  =  fx · k2     (quintic)
/// …
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FisheyeCalibration {
    /// Focal length in pixels (x axis).
    pub fx: f64,
    /// Focal length in pixels (y axis). Usually `== fx` for square
    /// sensors; differs for anamorphic.
    pub fy: f64,
    /// Principal point X in pixels (from image origin top-left).
    pub cx: f64,
    /// Principal point Y in pixels.
    pub cy: f64,
    /// KB-4 distortion coefficients (θ², θ⁴, θ⁶, θ⁸ terms).
    pub k: [f64; 4],
    /// Native resolution this calibration was performed at. Set to
    /// (0, 0) if unknown (in which case downstream code recomputes fx
    /// from the working footage resolution and a separate FOV value).
    pub calib_w: u32,
    pub calib_h: u32,
}

impl FisheyeCalibration {
    /// Build a calibration from "raw" parameters (no preset).
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, k: [f64; 4], calib_w: u32, calib_h: u32) -> Self {
        Self { fx, fy, cx, cy, k, calib_w, calib_h }
    }

    /// Hardcoded DJI Osmo 360 OSV defaults (matches the Python app at
    /// `vr180_gui.py:2857-2871`). cx/cy are zero — they get filled
    /// from the per-clip DJI protobuf when an `.osv` is loaded, or
    /// from "center of frame" otherwise.
    pub fn dji_osmo_360() -> Self {
        Self {
            fx: 0.0, // recomputed from FOV at working width
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            k: [
                0.063_054_046_599,
                0.003_034_146_878,
                -0.004_623_015_478,
                -0.000_516_517_650,
            ],
            calib_w: 0,
            calib_h: 0,
        }
    }

    /// Derive `fx` (and `fy = fx`) from a target full FOV and image
    /// width. Mirrors `_default_calib` at `vr180_gui.py:3313-3320`:
    ///
    /// ```text
    /// fx = image_width / (2 · half_fov_rad)
    /// ```
    ///
    /// For asymmetric H/V FOV (anamorphic), call this once with the
    /// horizontal pair and once with the vertical pair, then mix.
    pub fn fx_from_fov(image_width: u32, full_fov_rad: f64) -> f64 {
        let half = full_fov_rad * 0.5;
        if half <= 0.0 {
            0.0
        } else {
            image_width as f64 / (2.0 * half)
        }
    }

    /// Compute the maximum half-FOV that this calibration covers
    /// before hitting the image-circle boundary at radius `r_max_px`.
    /// Inverse-Newton-Raphson on KB. Used when loading Gyroflow JSONs
    /// (no FOV field, derive from cx/cy + r_edge).
    ///
    /// Returns FULL FOV in radians (caller can convert to degrees).
    pub fn full_fov_from_rim(&self, r_max_px: f64) -> f64 {
        let half = kb_inverse(r_max_px, self.fx, self.k, THETA_TRANS_DEFAULT, 80, 1e-12);
        2.0 * half
    }

    /// Returns true if the calibration has been populated with real
    /// values (not just the all-zero default).
    pub fn is_populated(&self) -> bool {
        self.fx > 0.0 || self.k.iter().any(|c| c.abs() > 1e-12)
    }
}

impl Default for FisheyeCalibration {
    fn default() -> Self {
        Self::dji_osmo_360()
    }
}

// ---- Gyroflow lens profile JSON ---------------------------------

/// Gyroflow lens profile (`.json`).
///
/// Schema reference: <https://github.com/gyroflow/lens_profiles>
///
/// We only consume the fields the Python loader at
/// `vr180_gui.py:7807-7886` uses: `fisheye_params.camera_matrix` and
/// `fisheye_params.distortion_coeffs`. The rest of the file (calib
/// dimension, native dimension, model name, etc.) is parsed loosely.
#[derive(Debug, Clone, Deserialize)]
pub struct GyroflowLensProfile {
    /// Free-form camera name, e.g. "Blackmagic URSA Cine Immersive".
    #[serde(default)]
    pub name: Option<String>,
    /// "BMD URSA Cine Immersive 7.4mm" style lens id.
    #[serde(default)]
    pub lens_model: Option<String>,
    /// Resolution this calibration was performed at.
    #[serde(default)]
    pub calib_dimension: Option<CalibDimension>,
    /// Native sensor resolution (may differ from calib).
    #[serde(default)]
    pub orig_dimension: Option<CalibDimension>,
    /// Output dimension (square crop, etc.). Optional.
    #[serde(default)]
    pub output_dimension: Option<CalibDimension>,
    /// "fisheye" / "rectilinear" / null. We only handle fisheye —
    /// rectilinear lens profiles would need a different distortion
    /// model. Default null means "assume fisheye" (matches Python).
    #[serde(default)]
    pub distortion_model: Option<String>,
    /// Per-camera fisheye-params block. Contains the actual numbers
    /// we feed the shader.
    pub fisheye_params: FisheyeParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CalibDimension {
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FisheyeParams {
    /// 3x3 camera matrix, row-major:
    /// ```text
    /// [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    /// ```
    pub camera_matrix: Vec<Vec<f64>>,
    /// KB-4 distortion coefficients `[k1, k2, k3, k4]`.
    pub distortion_coeffs: Vec<f64>,
}

impl GyroflowLensProfile {
    /// Load and parse a `.json` lens profile from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)?;
        let prof: Self = serde_json::from_str(&s)
            .map_err(|e| Error::GyroflowJson(format!("{}: {e}", path.display())))?;
        Ok(prof)
    }

    /// Convert to our `FisheyeCalibration`. Validates that
    /// `camera_matrix` is the expected 3x3 shape and that
    /// `distortion_coeffs` has 4 entries.
    pub fn to_calibration(&self) -> Result<FisheyeCalibration> {
        let m = &self.fisheye_params.camera_matrix;
        if m.len() != 3 || m.iter().any(|row| row.len() != 3) {
            return Err(Error::GyroflowJson(format!(
                "camera_matrix must be 3x3, got {} rows × {:?} cols",
                m.len(),
                m.iter().map(|r| r.len()).collect::<Vec<_>>()
            )));
        }
        let k = &self.fisheye_params.distortion_coeffs;
        if k.len() != 4 {
            return Err(Error::GyroflowJson(format!(
                "distortion_coeffs must have 4 entries (KB-4), got {}",
                k.len()
            )));
        }
        let (calib_w, calib_h) = self
            .calib_dimension
            .as_ref()
            .map(|d| (d.w, d.h))
            .unwrap_or((0, 0));
        Ok(FisheyeCalibration {
            fx: m[0][0],
            fy: m[1][1],
            cx: m[0][2],
            cy: m[1][2],
            k: [k[0], k[1], k[2], k[3]],
            calib_w,
            calib_h,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test against the vendored bmci 4096.json (if present).
    /// CI environments may not have the file — the test silently
    /// passes in that case.
    #[test]
    fn parse_bmci_4096() {
        let path = Path::new("/Volumes/Silver/develop/vr180_fisheye_converter/bmci 4096.json");
        if !path.exists() {
            eprintln!("skip parse_bmci_4096 — fixture not present");
            return;
        }
        let prof = GyroflowLensProfile::load(path).expect("load bmci 4096.json");
        let cal = prof.to_calibration().expect("convert to calibration");
        // Sanity: fx > 0, image is reasonably big, KB k1 is non-zero.
        assert!(cal.fx > 0.0);
        assert!(cal.calib_w >= 1000);
        assert!(cal.k[0].abs() > 1e-6);
    }

    #[test]
    fn fx_from_fov_round_trip() {
        let fx = FisheyeCalibration::fx_from_fov(4096, 207.68_f64.to_radians());
        // Half-FOV ≈ 1.813 rad → fx ≈ 4096 / 3.625 ≈ 1129.0
        assert!((fx - 1129.97).abs() < 1.0, "fx={fx}");
    }
}
