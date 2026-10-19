//! Curated camera-preset catalog.
//!
//! Named presets for popular fisheye cameras. Each entry holds:
//! - Human-readable name (UI dropdown).
//! - Default `FisheyeCalibration` (KB k1..k4 + native calibration size).
//! - Native sensor resolution (the "target" of the calibration; UI uses
//!   it to scale fx/cx/cy when the working footage resolution differs).
//! - Container/codec hint (`.osv` / `.braw` / `.mp4` etc.) so we can
//!   auto-select a preset on file open.
//!
//! The data lives inline here rather than as external JSON files so the
//! library is hermetic — no missing-fixture failures, no shipping
//! resources alongside the binary. Users can still load arbitrary
//! Gyroflow `.json` profiles through the GUI's "Load Lens Profile"
//! button.
//!
//! Values sourced from:
//! - DJI Osmo 360: `vr180_gui.py:2857-2871` (vendor calibration, used
//!   in the Python app's default config).
//! - Blackmagic URSA Cine Immersive 8192×7200: `cine immersive 8192
//!   7200.json` (Gyroflow community lens profile).
//! - Blackmagic Cine Immersive 4096: `bmci 4096.json`.
//! - Other entries: best-effort starting points; users will need to
//!   refine k1..k4 from a Gyroflow profile or a self-calibration.

use crate::calib::FisheyeCalibration;

/// A preset row in the camera catalog.
#[derive(Debug, Clone)]
pub struct CameraPreset {
    /// UI label, e.g. "DJI Osmo 360".
    pub name: &'static str,
    /// Optional submanufacturer/lens, e.g. "Blackmagic 210° Fisheye 7.4mm".
    pub lens: Option<&'static str>,
    /// Default calibration. `calib_w`/`calib_h` are the native size
    /// the constants are valid at.
    pub calib: FisheyeCalibration,
    /// Default full FOV in degrees. Used by the FOV slider as the
    /// starting value. The actual rendering uses cx/cy and the KB
    /// inverse — FOV is mostly a UX hint.
    pub default_fov_deg: f64,
    /// File-extension hints — letters only, no leading dot.
    /// E.g. `&["osv"]` or `&["braw"]`. SBS fisheye gets `&["mp4", "mov"]`.
    pub file_hints: &'static [&'static str],
    /// Input layout this preset expects.
    pub input_layout: InputLayout,
    /// A stereo lens's per-eye principal points, `[left eye, right eye]`,
    /// each an OFFSET from its eye's centre as a fraction of the FULL
    /// frame width (both eyes) — so one entry serves every sensor window
    /// height the camera offers. `None` = the eye centre / the file's
    /// calibration / the detected image circle.
    pub stereo_pp: Option<[[f64; 2]; 2]>,
    /// The physical LEFT eye sits in the RIGHT half of the frame (the lens
    /// crosses over): fresh clips of this preset seed the eye swap on.
    pub left_eye_is_right_half: bool,
    /// What `default_fov_deg` means — see [`FovConvention`].
    pub fov_convention: FovConvention,
}

/// How a preset's `default_fov_deg` relates to the focal length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FovConvention {
    /// The app's own: the angle the EYE WIDTH spans under an ideal
    /// equidistant lens, `fx = eye_w / fov_rad`, the KB polynomial on top
    /// (the Override slider, `fx_from_fov`).
    EyeWidth,
    /// The "fisheyecenter" web tool's (rest.s3for.me/marat/silvers_fix/
    /// fisheyecenter): the FOV spans the per-eye frame's LONGER dimension
    /// through the KB model, `f = (max(eye_w, eye_h) / 2) / θd(fov / 2)`
    /// with `θd = θ·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)` — a cropped fisheye
    /// keeps its FOV along the uncropped axis (DeoVR's convention).
    KbLongerDim,
}

impl CameraPreset {
    /// `default_fov_deg` expressed in the app's [`FovConvention::EyeWidth`]
    /// for an eye of `eye_w × eye_h`, so the resolvers can build `fx = eye_w
    /// / fov_rad` whatever convention the preset was entered in.
    pub fn app_fov_deg(&self, eye_w: u32, eye_h: u32) -> f64 {
        match self.fov_convention {
            FovConvention::EyeWidth => self.default_fov_deg,
            FovConvention::KbLongerDim => {
                let half = self.default_fov_deg.max(1.0).to_radians() * 0.5;
                let t2 = half * half;
                let k = self.calib.k;
                let theta_d = half * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
                if theta_d <= 1e-9 || eye_w == 0 { return self.default_fov_deg; }
                let f = (eye_w.max(eye_h) as f64 * 0.5) / theta_d;
                (eye_w as f64 / f).to_degrees()
            }
        }
    }
}

/// Which decoder path this preset routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputLayout {
    /// Single-stream side-by-side fisheye (`.mp4` / `.mov`).
    /// Split horizontally into left/right halves.
    SideBySide,
    /// Two-stream dual fisheye (DJI `.osv`, custom dual-cam rigs).
    /// `-map 0:0` + `-map 0:1` parallel decoders.
    DualStream,
    /// Single-stream fisheye with the eyes stacked top/bottom (Kandao
    /// QooCam 3 Ultra). Split vertically into top/bottom halves.
    Stacked,
    /// Blackmagic RAW (`.braw`). Wrap braw_helper subprocess.
    /// Multi-track files (URSA Cine Immersive stereo, Pyxis 12K) are
    /// emitted as a SBS BGRA stream by the helper.
    BlackmagicRaw,
}

/// The dual-fisheye PYXIS preset's FOV as the user calibrated it in the
/// "fisheyecenter" web tool ([`FovConvention::KbLongerDim`]): 192.709108254°
/// across the eye's longer dimension through the KB model, which for the
/// 6144-wide eyes is `f = 3072 / θd(96.35°) = 1788.8 px` → **196.8°** in
/// the app's eye-width convention ([`CameraPreset::app_fov_deg`]). The
/// camera-locked residual sweep on the sample clip (fastest 2 s window,
/// their k + principal points) agrees within its shallow floor: 192.7°
/// 1.37, 195° 1.32, 198° 1.32, 202° 1.30, 205° 1.33, 210° 1.41 px.
pub const BRAW_DUAL_FISHEYE_FOV_DEG: f64 = 192.709108254;

/// Built-in preset catalog. Static — no allocation, no I/O.
pub fn presets() -> &'static [CameraPreset] {
    &PRESETS
}

/// Look up a preset by case-insensitive name.
pub fn find(name: &str) -> Option<&'static CameraPreset> {
    PRESETS.iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

/// Pick the best-guess preset for a given file extension (lowercase,
/// no leading dot). Returns `None` if no preset claims the extension.
pub fn for_extension(ext: &str) -> Option<&'static CameraPreset> {
    let ext_lower = ext.to_ascii_lowercase();
    PRESETS.iter().find(|p| p.file_hints.iter().any(|h| *h == ext_lower))
}

// ---- Catalog ------------------------------------------------------

static PRESETS: [CameraPreset; 11] = [
    // ============================================================
    // DJI Osmo 360
    // ============================================================
    // Default OSV calibration from vr180_gui.py:2864-2868.
    // Two physical lenses, ~207.68° FOV each. cx/cy filled from the
    // per-clip protobuf at runtime (lens_a / lens_b blocks). The
    // protobuf calibration is the canonical source — preset values
    // are just the no-protobuf fallback.
    CameraPreset {
        name: "DJI Osmo 360",
        lens: Some("2× DJI 207.68° fisheye"),
        calib: FisheyeCalibration {
            fx: 0.0, // recomputed per-clip from FOV + working width
            fy: 0.0,
            cx: 0.0, // overridden from .osv protobuf
            cy: 0.0,
            k: [
                0.063_054_046_599,
                0.003_034_146_878,
                -0.004_623_015_478,
                -0.000_516_517_650,
            ],
            calib_w: 0,
            calib_h: 0,
        },
        default_fov_deg: 207.68,
        file_hints: &["osv"],
        input_layout: InputLayout::DualStream,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Blackmagic Pyxis 12K (single-camera fisheye)
    // ============================================================
    // Pyxis 12K with Blackmagic 7.4mm 210° fisheye. SBS shoots happen
    // by physically pairing two Pyxis bodies — wgpu sees one .braw
    // per body.
    CameraPreset {
        name: "Blackmagic Pyxis 12K",
        lens: Some("Blackmagic 210° fisheye 7.4mm"),
        calib: FisheyeCalibration {
            // From the bmci 4096.json lens profile (community).
            fx: 1834.0,
            fy: 1834.0,
            cx: 2048.0,
            cy: 2048.0,
            k: [0.0, 0.0, 0.0, 0.0], // refine via Gyroflow JSON load
            calib_w: 4096,
            calib_h: 4096,
        },
        default_fov_deg: 210.0,
        file_hints: &["braw"],
        input_layout: InputLayout::BlackmagicRaw,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Blackmagic URSA Cine Immersive
    // ============================================================
    // Multi-video stereo BRAW container (track 0 = left, track 1 =
    // right at 8192×7200 each). Community Gyroflow profile available.
    CameraPreset {
        name: "Blackmagic URSA Cine Immersive",
        lens: Some("Blackmagic 210° fisheye 7.4mm (stereo)"),
        calib: FisheyeCalibration {
            fx: 3668.0,
            fy: 3668.0,
            cx: 4080.0, // ≈ 8160/2
            cy: 3600.0, // ≈ 7200/2
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 8160,
            calib_h: 7200,
        },
        default_fov_deg: 210.0,
        file_hints: &["braw"],
        input_layout: InputLayout::BlackmagicRaw,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Insta360 EVO
    // ============================================================
    // Folding consumer 3D 180° camera. Outputs SBS 5760×2880 H.264.
    CameraPreset {
        name: "Insta360 EVO",
        lens: Some("2× fisheye, ~200° each"),
        calib: FisheyeCalibration {
            fx: 1440.0,
            fy: 1440.0,
            cx: 1440.0,
            cy: 1440.0,
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 2880,
            calib_h: 2880,
        },
        default_fov_deg: 200.0,
        file_hints: &["mp4", "mov"],
        input_layout: InputLayout::SideBySide,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Vuze XR
    // ============================================================
    // Folding 3D 180° camera by HumanEyes. SBS 5760×2880 30fps H.264.
    CameraPreset {
        name: "Vuze XR",
        lens: Some("2× fisheye, ~180° each"),
        calib: FisheyeCalibration {
            fx: 1440.0,
            fy: 1440.0,
            cx: 1440.0,
            cy: 1440.0,
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 2880,
            calib_h: 2880,
        },
        default_fov_deg: 180.0,
        file_hints: &["mp4", "mov"],
        input_layout: InputLayout::SideBySide,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // QooCam EGO
    // ============================================================
    // 3D 4K side-by-side camera by Kandao. SBS H.264.
    CameraPreset {
        name: "Kandao QooCam EGO",
        lens: Some("2× fisheye, ~140° each"),
        calib: FisheyeCalibration {
            fx: 960.0,
            fy: 960.0,
            cx: 960.0,
            cy: 960.0,
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 1920,
            calib_h: 1920,
        },
        default_fov_deg: 140.0,
        file_hints: &["mp4", "mov"],
        input_layout: InputLayout::SideBySide,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Canon RF 5.2mm dual fisheye
    // ============================================================
    // Native VR180 stereo lens for EOS R5 / R5C. SBS 8192×4096 H.265
    // or RAW. Lens delivers ~190° FOV per eye.
    CameraPreset {
        name: "Canon RF 5.2mm Dual Fisheye",
        lens: Some("Canon RF 5.2mm f/2.8 L Dual Fisheye"),
        calib: FisheyeCalibration {
            fx: 2050.0,
            fy: 2050.0,
            cx: 2048.0,
            cy: 2048.0,
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 4096,
            calib_h: 4096,
        },
        default_fov_deg: 190.0,
        file_hints: &["mp4", "mov", "cr3"],
        input_layout: InputLayout::SideBySide,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Custom / manual KB
    // ============================================================
    // Catch-all fallback for unrecognised cameras. UI shows the
    // sliders fully editable; no auto-fill.
    CameraPreset {
        name: "Custom",
        lens: None,
        calib: FisheyeCalibration {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            k: [0.0, 0.0, 0.0, 0.0],
            calib_w: 0,
            calib_h: 0,
        },
        default_fov_deg: 180.0,
        file_hints: &[],
        input_layout: InputLayout::SideBySide,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Insta360 X6 (dual-stream .insv, VR180-modded)
    // ============================================================
    // FALLBACK only — `.insv` files carry a per-lens factory calibration
    // (UCM ξ + radial polynomial, decoded against the camera's own stitched
    // previews) that the pipeline converts per file; see
    // vr180-pipeline::insv_imu::stream_lens_calib. These constants are that
    // conversion for a typical unit: 3840² stream = 2×2-binned sensor,
    // principal point at the frame centre, rim ≈ 102° → ≈204° FOV.
    CameraPreset {
        name: "Insta360 X6",
        lens: Some("2× Insta360 ~204° fisheye"),
        calib: FisheyeCalibration {
            fx: 1044.389,
            fy: 1044.389,
            cx: 1920.0,
            cy: 1920.0,
            k: [0.087926, -0.023232, 0.001227, -0.000388],
            calib_w: 3840,
            calib_h: 3840,
        },
        default_fov_deg: 204.5,
        file_hints: &["insv"],
        input_layout: InputLayout::DualStream,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Kandao QooCam 3 Ultra (stacked single-stream .mp4, VR180-modded)
    // ============================================================
    // FALLBACK only — the files carry a per-lens factory Kannala-Brandt
    // record (focal, principal point, k1..k4 in the 13888×6944 photo
    // canvas) that `vr180-pipeline::kandao_imu::stream_lens_calib` scales to
    // the eye size per file. These are the sample camera's values at the
    // 3840² video eye: focal 2142.13 px/rad × 3840/6944, principal point
    // at the centre, rim ≈ 106° → ≈212° FOV.
    CameraPreset {
        name: "Kandao QooCam 3 Ultra",
        lens: Some("2× Kandao ~210° fisheye"),
        calib: FisheyeCalibration {
            fx: 1184.59,
            fy: 1184.59,
            cx: 1920.0,
            cy: 1920.0,
            k: [-0.011315, 0.001868, -0.003552, 0.000244],
            calib_w: 3840,
            calib_h: 3840,
        },
        default_fov_deg: 212.0,
        file_hints: &[],
        input_layout: InputLayout::Stacked,
        stereo_pp: None,
        left_eye_is_right_half: false,
        fov_convention: FovConvention::EyeWidth,
    },

    // ============================================================
    // Blackmagic PYXIS 12K with a dual-fisheye VR180 lens (one sensor)
    // ============================================================
    // The user's calibration for the lens this camera is commonly used
    // with (2026-10-19): full FOV 192.709108254°, KB k1..k4 below, per-eye
    // principal points given normalised to a 12288-wide 2:1 frame with the
    // 60p 12288×5112 video centred in it, origin top-left, and the LEFT eye
    // in the RIGHT half: CxL 0.76546, CyL 0.50049, CxR 0.24243, CyR 0.49821.
    // Stored as offsets from each eye's centre in units of the frame width
    // (dx = Cx − 0.75 / 0.25, dy = (Cy − 0.5) / 2), so the 50p 12288×6480
    // window shares them. The FOV is in the web tool's convention
    // (`FovConvention::KbLongerDim`); `app_fov_deg` converts it.
    CameraPreset {
        name: "Blackmagic PYXIS 12K VR180 (dual fisheye)",
        lens: Some("Dual fisheye on one sensor, 192.7°"),
        calib: FisheyeCalibration {
            fx: 0.0, // from the FOV at the working size
            fy: 0.0,
            cx: 3072.0,
            cy: 2556.0,
            k: [-0.002605415285, 0.001226067608, -0.000347504137, 0.000413900418],
            calib_w: 6144,
            calib_h: 5112,
        },
        default_fov_deg: BRAW_DUAL_FISHEYE_FOV_DEG,
        file_hints: &["braw"],
        input_layout: InputLayout::BlackmagicRaw,
        stereo_pp: Some([
            [0.76546 - 0.75, (0.50049 - 0.5) / 2.0],   // left eye (right half)
            [0.24243 - 0.25, (0.49821 - 0.5) / 2.0],   // right eye (left half)
        ]),
        left_eye_is_right_half: true,
        fov_convention: FovConvention::KbLongerDim,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_unique_names() {
        let mut seen = std::collections::HashSet::new();
        for p in presets() {
            assert!(seen.insert(p.name), "duplicate preset name: {}", p.name);
        }
    }

    #[test]
    fn osv_resolves_to_dji() {
        let p = for_extension("osv").expect("osv → preset");
        assert_eq!(p.name, "DJI Osmo 360");
        assert_eq!(p.input_layout, InputLayout::DualStream);
    }

    #[test]
    fn braw_resolves_to_pyxis() {
        // First .braw-claiming preset wins (Pyxis 12K).
        let p = for_extension("braw").expect("braw → preset");
        assert_eq!(p.input_layout, InputLayout::BlackmagicRaw);
    }

    #[test]
    fn mp4_resolves_to_sbs() {
        let p = for_extension("mp4").expect("mp4 → preset");
        assert_eq!(p.input_layout, InputLayout::SideBySide);
    }

    #[test]
    fn find_by_name_case_insensitive() {
        assert!(find("dji osmo 360").is_some());
        assert!(find("DJI OSMO 360").is_some());
        assert!(find("nonsense").is_none());
    }
}

#[cfg(test)]
mod braw_preset_tests {
    use super::*;

    #[test]
    fn pyxis_dual_fisheye_fov_converts_through_the_kb_model() {
        let p = find("Blackmagic PYXIS 12K VR180 (dual fisheye)").expect("preset");
        assert_eq!(p.fov_convention, FovConvention::KbLongerDim);
        // 6144×5112 eyes: f = 3072 / θd(96.35°) = 1788.8 px → 6144 / f rad = 196.8°.
        let fov = p.app_fov_deg(6144, 5112);
        assert!((fov - 196.79).abs() < 0.1, "{fov}");
        // Square eyes (the tool's 2:1 timeline): the same focal, the same answer.
        assert!((p.app_fov_deg(6144, 6144) - fov).abs() < 1e-9);
        // The Custom slot must stay at index 7 (callers index it).
        assert_eq!(presets()[7].name, "Custom");
        // An eye-width preset passes its value through.
        let osmo = find("DJI Osmo 360").unwrap();
        assert_eq!(osmo.app_fov_deg(3840, 3840), osmo.default_fov_deg);
    }
}
