//! Each eye's lens model for a clip, decided in one place.
//!
//! The preview asks from the sidebar's settings and an export from its
//! configuration, and both must get the same answer for the same clip: the
//! lens the file describes (per-lens factory calibration, scaled to the
//! size being projected), or a preset, or the user's manual override, then
//! what the output asks of it (the fisheye disk's field of view, a reframed
//! viewport). Each had its own copy of this; they now build a
//! [`LensRequest`] and call [`resolve_lens_pair`].

use crate::gpu::{CropMask, FisheyeCalib};
use crate::source_kind::SourceKind;

/// One eye's manual lens fields: the calibration panel's Override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeLensOverride {
    /// Off: the eye takes the file's own calibration (or the preset).
    pub on: bool,
    pub fov_deg: f32,
    /// Principal point as a fraction of the eye's frame.
    pub cx_norm: f32,
    pub cy_norm: f32,
    /// Radial coefficients k1…k4, and the fifth.
    pub k: [f32; 4],
    pub k5: f32,
    /// Tangential p1, p2.
    pub p: [f32; 2],
}

/// What the projection that uses the lens model renders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LensOutput {
    /// 180° × 180° per eye (VR180).
    HalfEquirect,
    /// The equidistant fisheye disk.
    Fisheye,
    /// A flat viewport `hfov_deg` wide, `aspect` = width / height, blended
    /// toward a fisheye by `defish`.
    Reframe { hfov_deg: f32, defish: f32, aspect: f32 },
}

/// Everything that decides a clip's two lens models besides the frame size
/// and the file's own calibration.
#[derive(Debug, Clone)]
pub struct LensRequest<'a> {
    pub kind: SourceKind,
    /// The camera preset by name; empty or "Auto" follows the source kind.
    pub preset: &'a str,
    /// The user's eye swap: the left output then carries lens A.
    pub swap_eyes: bool,
    pub left: EyeLensOverride,
    pub right: EyeLensOverride,
    /// The mask of a frame that is a crop of the image circle.
    pub crop_mask: CropMask,
    /// A generic side-by-side clip: its halves are fisheye pictures to
    /// dewarp (off: they are half-equirect already).
    pub sbs_dewarp_fisheye: bool,
    pub output: LensOutput,
}

/// The (left, right) lens models for eyes of `src_w × src_h`, the size
/// being projected. `osv` carries the file's per-lens calibration when the
/// source has one.
pub fn resolve_lens_pair(
    req: &LensRequest,
    src_w: u32,
    src_h: u32,
    osv: Option<&vr180_fisheye::DjiOsvImu>,
) -> (FisheyeCalib, FisheyeCalib) {
    use vr180_fisheye::presets;

    let preset = if !req.preset.is_empty() && req.preset != "Auto" {
        presets::find(&req.preset)
    } else {
        None
    }.unwrap_or_else(|| {
        let auto_name = match req.kind {
            SourceKind::DjiOsv        => "DJI Osmo 360",
            SourceKind::Insta360Insv
            | SourceKind::Insta360Dng => "Insta360 X6",
            SourceKind::KandaoQoocam
            | SourceKind::KandaoDng   => crate::kandao_imu::KANDAO_PRESET_NAME,
            SourceKind::BlackmagicRaw => crate::braw_imu::BRAW_PRESET_NAME,
            _                         => "Custom",
        };
        presets::find(auto_name).unwrap_or(&presets::presets()[7])
    });

    // Preset-derived auto fx (override off / protobuf missing fx).
    let fx_auto = if preset.calib.fx > 0.0 && preset.calib.calib_w > 0 {
        (preset.calib.fx as f32) * (src_w as f32) / (preset.calib.calib_w as f32)
    } else {
        let half = (preset.default_fov_deg as f32).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };
    let preset_k = [preset.calib.k[0] as f32, preset.calib.k[1] as f32,
                    preset.calib.k[2] as f32, preset.calib.k[3] as f32];
    let fx_from_fov = |fov_deg: f32| -> f32 {
        let half = fov_deg.max(1.0).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };

    // For OSV, prefer the per-lens protobuf calibration. After the DJI
    // iter's default swap: left = Lens B, right = Lens A.
    let (calib_l, calib_r) = match (req.kind, osv) {
        (SourceKind::DjiOsv
         | SourceKind::Insta360Insv
         | SourceKind::Insta360Dng
         | SourceKind::KandaoQoocam
         | SourceKind::KandaoDng
         | SourceKind::BlackmagicRaw, Some(imu)) => {
            let scale_x = imu.lens_b.width.map(|w| (src_w as f32) / w).unwrap_or(1.0);
            let scale_y = imu.lens_b.height.map(|h| (src_h as f32) / h).unwrap_or(1.0);
            let eye = |output: usize, lens: &vr180_fisheye::DjiLensCalib,
                       ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4], km5: f32,
                       km_p: [f32; 2]|
                -> FisheyeCalib
            {
                // Auto + an exact factory model (Insta360): use it as is —
                // the KB fields only seed the override UI. Override drops
                // to KB (the manual fields describe a KB lens).
                if !ov {
                    if let Some(o) = lens.omni {
                        let cx = lens.cx.map(|v| v * scale_x).unwrap_or(src_w as f32 * 0.5);
                        let cy = lens.cy.map(|v| v * scale_y).unwrap_or(src_h as f32 * 0.5);
                        return FisheyeCalib::new_omni(
                            o.fx * scale_x, o.fy * scale_y, cx, cy, o.xi,
                            o.radial, o.tangential, o.prism, src_w as f32, src_h as f32,
                        );
                    }
                }
                let (fx, fy, cx, cy, k, k5) = if ov {
                    let fx = fx_from_fov(fov);
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx, fx, cxn * src_w as f32, cyn * src_h as f32, k, km5)
                } else {
                    // Auto: full per-lens factory calibration from the OSV
                    // protobuf — fx/fy (→FOV), cx/cy, the KB k1–k4 AND the 5th
                    // radial coeff k5 (field 15). Must match the GUI's
                    // resolve_fisheye_calib_pair exactly so export == preview.
                    // k is dimensionless (no scale); missing → preset / 0.
                    let cx = lens.cx.map(|v| v * scale_x).unwrap_or(src_w as f32 * 0.5);
                    let cy = lens.cy.map(|v| v * scale_y).unwrap_or(src_h as f32 * 0.5);
                    let fx = lens.fx.map(|v| v * scale_x).unwrap_or(fx_auto);
                    let fy = lens.fy.map(|v| v * scale_y).unwrap_or(fx);
                    let k = match (lens.k1, lens.k2, lens.k3, lens.k4) {
                        (Some(a), Some(b), Some(c), Some(d)) => [a, b, c, d],
                        _ => preset_k,
                    };
                    // Blackmagic: the seed is the image circle + the DEFAULT
                    // preset's FOV / k; the Camera combo's choice re-scales the
                    // focal to its FOV and takes its k (same in the GUI resolver).
                    let (fx, fy, cx, cy, k) = if req.kind == SourceKind::BlackmagicRaw {
                        let (sx, sk) = crate::braw_imu::preset_rescale(preset, src_w, src_h);
                        let (cx, cy) = crate::braw_imu::preset_principal_point(preset, output, req.swap_eyes, src_w as f32, src_h as f32)
                            .unwrap_or((cx, cy));
                        (fx * sx, fy * sx, cx, cy, sk.unwrap_or(k))
                    } else { (fx, fy, cx, cy, k) };
                    (fx, fy, cx, cy, k, lens.k5.unwrap_or(0.0))
                };
                let mut calib =
                    FisheyeCalib::new_pure_kb(fx, fy, cx, cy, k, src_w as f32, src_h as f32);
                calib.k5 = k5;
                // Tangential: Auto from file, override uses the manual [p1,p2]
                // (km_p, GUI-seeded from the file). Must match the preview path.
                calib.p1 = if ov { km_p[0] } else { lens.p1.unwrap_or(0.0) };
                calib.p2 = if ov { km_p[1] } else { lens.p2.unwrap_or(0.0) };
                calib
            };
            // Calib follows the STREAM: with the user swap on, the left
            // output carries Lens A → use Lens A's calib (matches the
            // GUI resolver so export == preview).
            let (lens_l, lens_r) = if req.swap_eyes {
                (&imu.lens_a, &imu.lens_b)
            } else {
                (&imu.lens_b, &imu.lens_a)
            };
            let pair = (
                eye(0, lens_l, req.left.on, req.left.fov_deg,
                    req.left.cx_norm, req.left.cy_norm, req.left.k,
                    req.left.k5, req.left.p),
                eye(1, lens_r, req.right.on, req.right.fov_deg,
                    req.right.cx_norm, req.right.cy_norm, req.right.k,
                    req.right.k5, req.right.p),
            );
            // A frame that is a crop of the image circle (the X6 Crop
            // modes): black outside the frame, corners kept, plus the
            // user's mask window. Must match the GUI resolver.
            if lens_l.frame_crop || lens_r.frame_crop {
                crate::gpu::apply_frame_crop(pair.0, pair.1, req.crop_mask)
            } else {
                pair
            }
        }
        _ => {
            // Non-OSV: Hermite-default constructor, per-eye.
            let r_max = (src_w.min(src_h) as f32) * 0.5;
            let eye = |ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4]| -> FisheyeCalib {
                let (fx, cx, cy, k) = if ov {
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx_from_fov(fov), cxn * src_w as f32, cyn * src_h as f32, k)
                } else {
                    (fx_auto, src_w as f32 * 0.5, src_h as f32 * 0.5, preset_k)
                };
                FisheyeCalib::new(fx, fx, cx, cy, k, src_w as f32, src_h as f32, r_max)
            };
            (
                eye(req.left.on, req.left.fov_deg,
                    req.left.cx_norm, req.left.cy_norm, req.left.k),
                eye(req.right.on, req.right.fov_deg,
                    req.right.cx_norm, req.right.cy_norm, req.right.k),
            )
        }
    };

    // Generic SBS with the fisheye dewarp off: the halves are already
    // half-equirect — bypass the lens model (identity resample for the
    // default output, a correct (lon, lat) lookup for the reframed view).
    let (calib_l, calib_r) = if req.kind == SourceKind::SbsFisheye && !req.sbs_dewarp_fisheye {
        (calib_l.with_equirect_input(), calib_r.with_equirect_input())
    } else {
        (calib_l, calib_r)
    };

    // Equidistant FISHEYE output target: set the output half-FOV so the
    // shaders map the inscribed circle of the square frame to a clean
    // FISHEYE_OUT_FULL_FOV_DEG fisheye. (The equirect target keeps the
    // default 90° horizontal half-FOV.)
    match req.output {
        LensOutput::Fisheye => {
            let hfov = (crate::fisheye_export::FISHEYE_OUT_FULL_FOV_DEG * 0.5).to_radians();
            (calib_l.with_output_hfov(hfov), calib_r.with_output_hfov(hfov))
        }
        // Reframed view: a flat viewport of this width and shape.
        LensOutput::Reframe { hfov_deg, defish, aspect } => (
            calib_l.with_reframe(hfov_deg, defish, aspect),
            calib_r.with_reframe(hfov_deg, defish, aspect),
        ),
        LensOutput::HalfEquirect => (calib_l, calib_r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(on: bool, fov_deg: f32, cx_norm: f32) -> EyeLensOverride {
        EyeLensOverride { on, fov_deg, cx_norm, cy_norm: 0.5, k: [0.0; 4], k5: 0.0, p: [0.0; 2] }
    }

    fn request(left: EyeLensOverride, right: EyeLensOverride) -> LensRequest<'static> {
        LensRequest {
            kind: SourceKind::SbsFisheye,
            preset: "",
            swap_eyes: false,
            left,
            right,
            crop_mask: CropMask::default(),
            sbs_dewarp_fisheye: true,
            output: LensOutput::HalfEquirect,
        }
    }

    #[test]
    fn each_eye_keeps_its_own_override() {
        // Left overridden to 200° with its centre a tenth left of the
        // middle; right left on Auto (the preset's 180°).
        let (l, r) = resolve_lens_pair(&request(eye(true, 200.0, 0.4), eye(false, 0.0, 0.5)), 2000, 2000, None);
        // 200° across 2000 px, in px per radian; Auto: 180° across them.
        assert!((l.fx - 2000.0 / 200f32.to_radians()).abs() < 1e-2, "{}", l.fx);
        assert!((r.fx - 2000.0 / std::f32::consts::PI).abs() < 1e-2, "{}", r.fx);
        assert!((l.cx - 800.0).abs() < 1e-3 && (l.cy - 1000.0).abs() < 1e-3);
        assert!((r.cx - 1000.0).abs() < 1e-3 && (r.cy - 1000.0).abs() < 1e-3);
        assert_ne!(l.fx, r.fx);
        // And the other way round.
        let (l2, r2) = resolve_lens_pair(&request(eye(false, 0.0, 0.5), eye(true, 200.0, 0.4)), 2000, 2000, None);
        assert_eq!((l2.fx, l2.cx), (r.fx, r.cx));
        assert_eq!((r2.fx, r2.cx), (l.fx, l.cx));
    }

    #[test]
    fn the_output_view_does_not_move_the_lens() {
        let base = request(eye(true, 190.0, 0.5), eye(true, 190.0, 0.5));
        let plain = resolve_lens_pair(&base, 1920, 1920, None);
        for output in [LensOutput::Fisheye, LensOutput::Reframe { hfov_deg: 90.0, defish: 0.0, aspect: 16.0 / 9.0 }] {
            let shaped = resolve_lens_pair(&LensRequest { output, ..base.clone() }, 1920, 1920, None);
            assert_eq!((shaped.0.fx, shaped.0.cx, shaped.0.cy, shaped.0.k), (plain.0.fx, plain.0.cx, plain.0.cy, plain.0.k));
            assert_ne!(format!("{:?}", shaped.0), format!("{:?}", plain.0), "{output:?} leaves a mark on the model");
        }
    }
}
