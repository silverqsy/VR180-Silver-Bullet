//! Still-photo export: one Insta360 X6 `.dng` → a VR180 side-by-side still.
//!
//! The still is one 15520×7760 Bayer frame holding both fisheye circles
//! (left half = calibration entry 1 = LEFT eye post-mod, right half = entry 2
//! = RIGHT eye). Each half is developed on the CPU
//! ([`raw_develop::develop_linear`]: gain map, demosaic, camera-linear
//! RGBA64), projected through the same UCM fisheye kernels the video path
//! uses (16-bit, [`Device::project_fisheye_to_equirect_texture_16`]), read
//! back, and composed side by side on the CPU (a native 15520-wide SBS
//! texture would exceed wgpu's default 8192 limit).
//!
//! Two kinds of output:
//! - **Developed** (PNG 16-bit / JPEG): [`raw_develop::develop_display`]
//!   runs before projection (white balance, matrix, exposure, tone, OETF).
//! - **Linear DNG**: the camera-native linear data is projected as is and
//!   written by [`dng_write`] with the source's colour tags, so a raw
//!   developer renders the VR180 still with the camera's own profile and
//!   full raw latitude. The gain map is baked in before reprojection and
//!   therefore dropped from the output; the headroom it adds is carried by
//!   `BaselineExposure + log2(l_max)` with the data scaled by `1/l_max`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::dng_write::{self, DngCompression, LinearDngImage, LinearDngMeta, LookTable};
use crate::fisheye_export::{FisheyeExportProjection, FISHEYE_OUT_FULL_FOV_DEG};
use crate::gpu::{ColorStackPlan, Device, EquirectRotation, FisheyeCalib};
use crate::insv_imu;
use crate::panomap::{mat3_mul_row_major, ViewAdjust};
use crate::raw_develop::{self, DevelopParams, DngColorLike, GainMapLike, Half, LinearEye, RawParams, WhiteBalance};
use crate::{Error, Result};
use vr180_fisheye::dng::{self as x6dng, X6Still};
use vr180_fisheye::insta360::InsvWindowCrop;

/// Output container for a still export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StillFormat {
    /// 16-bit RGB PNG, display-referred (Rec.709 / sRGB).
    Png16,
    /// 8-bit JPEG, display-referred. `quality` 1..=100.
    Jpeg { quality: u8 },
    /// Linear DNG (camera-native linear RGB, raw developer renders it).
    LinearDng { compression: DngCompression },
}

impl StillFormat {
    pub fn extension(self) -> &'static str {
        match self {
            StillFormat::Png16 => "png",
            StillFormat::Jpeg { .. } => "jpg",
            StillFormat::LinearDng { .. } => "dng",
        }
    }
    pub fn is_linear_dng(self) -> bool { matches!(self, StillFormat::LinearDng { .. }) }
}

/// Everything a still export needs.
#[derive(Debug, Clone)]
pub struct StillExportConfig {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub format: StillFormat,
    /// Output size per eye (half-equirect, square). The SBS file is
    /// `2·eye_w × eye_h`.
    pub eye_w: u32,
    pub eye_h: u32,
    /// Pano / stereo view adjust (same semantics as the video export).
    pub view_adjust: ViewAdjust,
    /// Swap the two halves (user toggle). The halves are already in L|R
    /// order, so this is a plain swap — unlike the dual-stream video path.
    pub swap_eyes: bool,
    /// Developed-output parameters (ignored for the linear DNG).
    pub develop: DevelopParams,
    /// The app's colour stack (CDL / LUT / temperature-tint-saturation /
    /// sharpen …). Baked into PNG / JPEG after projection, exactly as the
    /// video export does; for the linear DNG it is NOT baked (see the
    /// metadata path).
    pub color_stack: ColorStackPlan,
    /// Output projection — half-equirect VR180, the normalized
    /// `FISHEYE_OUT_FULL_FOV_DEG` fisheye, or a reframed viewport — with
    /// the per-eye calibration shaped exactly as the video export shapes it.
    /// `eye_w × eye_h` is the per-eye frame of that projection (square for
    /// the fisheye, the viewport for a reframe).
    pub projection: FisheyeExportProjection,
    /// Per-OUTPUT-eye manual lens Override (`[left, right]`; `None` = the
    /// DNG's factory model). Mirrors the preview's resolver so export ==
    /// preview when the user has edited an eye's FOV / centre / distortion.
    pub lens_override: [Option<StillLensOverride>; 2],
}

/// One eye's manual lens Override for a still: a Kannala-Brandt lens
/// described by the sidebar's fields, in place of the DNG's factory UCM
/// model (the same substitution the video preview / export make).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StillLensOverride {
    /// Per-eye field of view (degrees) → `fx = w / fov`.
    pub fov_deg: f32,
    /// Principal point as a fraction of the eye frame.
    pub cx_norm: f32,
    pub cy_norm: f32,
    /// KB k1–k4 (the caller substitutes the camera preset's when the
    /// manual ones are all zero, as the preview resolver does).
    pub k: [f32; 4],
    pub k5: f32,
    /// Brown-Conrady tangential `[p1, p2]`.
    pub p: [f32; 2],
}

impl StillLensOverride {
    /// The `[left, right]` output-eye Overrides from a video-style export
    /// config (the GUI fills the same per-eye fields for every source kind).
    /// Manual k1–k4 that are all zero take the camera preset's — the explicit
    /// preset name wins, else the X6 preset — exactly as the video resolvers.
    pub fn from_export_cfg(cfg: &crate::fisheye_export::FisheyeExportConfig) -> [Option<Self>; 2] {
        use vr180_fisheye::presets;
        let auto_name = if cfg.source_kind.is_kandao() { crate::kandao_imu::KANDAO_PRESET_NAME } else { "Insta360 X6" };
        let preset = if !cfg.fisheye_preset.is_empty() && cfg.fisheye_preset != "Auto" {
            presets::find(&cfg.fisheye_preset)
        } else {
            None
        }
        .or_else(|| presets::find(auto_name))
        .unwrap_or(&presets::presets()[7]);
        let preset_k = [preset.calib.k[0] as f32, preset.calib.k[1] as f32,
                        preset.calib.k[2] as f32, preset.calib.k[3] as f32];
        let mk = |on: bool, fov: f32, cxn: f32, cyn: f32, k: [f32; 4], k5: f32, p: [f32; 2]| {
            on.then(|| Self {
                fov_deg: fov, cx_norm: cxn, cy_norm: cyn,
                k: if k.iter().any(|c| c.abs() > 1e-9) { k } else { preset_k },
                k5, p,
            })
        };
        [
            mk(cfg.fisheye_override_left, cfg.fisheye_fov_deg_left,
               cfg.fisheye_cx_norm_left, cfg.fisheye_cy_norm_left,
               cfg.fisheye_k_left, cfg.fisheye_k5_left, cfg.fisheye_p_left),
            mk(cfg.fisheye_override_right, cfg.fisheye_fov_deg_right,
               cfg.fisheye_cx_norm_right, cfg.fisheye_cy_norm_right,
               cfg.fisheye_k_right, cfg.fisheye_k5_right, cfg.fisheye_p_right),
        ]
    }
}

/// Progress callback: `(stage, fraction 0..1)`.
pub type StillProgress<'a> = &'a (dyn Fn(&str, f32) + Sync);

/// The per-eye calibration for the still in its own 7760² half-frame.
///
/// The lens strings describe a 15488×7744 frame; the still is two 7760²
/// halves with the 7744 window placed at offset 0 (trailer field 27:
/// `src 7744² → dst 7760², offset 0`). `stream_lens_calib` applies the same
/// `(src − dst)/2 + offset` rule that was verified against Insta360 Studio
/// for video, at scale 1 (no binning for stills).
pub fn still_eye_calib(still: &X6Still, half: Half) -> Result<FisheyeCalib> {
    let half_w = still.width / 2;
    let half_h = still.height;
    // Kandao QooCam: the DNG is the lens record's canvas — the
    // Kannala-Brandt entry of this half at scale 1 (`k5 = p = 0`).
    if let Some(k) = &still.kandao {
        let idx = match half { Half::Left => 0u8, Half::Right => 1u8 };
        let lens = k.lenses.iter().find(|l| l.index == idx)
            .ok_or_else(|| Error::Ffmpeg(format!("QooCam DNG: no lens record for lens {idx}")))?;
        let c = crate::kandao_imu::stream_lens_calib(lens, half_w, half_h);
        let k4 = [c.k1.unwrap_or(0.0), c.k2.unwrap_or(0.0), c.k3.unwrap_or(0.0), c.k4.unwrap_or(0.0)];
        return Ok(FisheyeCalib::new_pure_kb(
            c.fx.unwrap_or(1.0), c.fy.unwrap_or(1.0),
            c.cx.unwrap_or(half_w as f32 * 0.5), c.cy.unwrap_or(half_h as f32 * 0.5),
            k4, half_w as f32, half_h as f32,
        ));
    }
    let entry = still
        .lenses
        .get(match half { Half::Left => 0, Half::Right => 1 })
        .ok_or_else(|| Error::Ffmpeg("DNG: fewer than two lens calibrations".into()))?;
    let crop = still.crop.unwrap_or(InsvWindowCrop {
        src_w: still.calib_frame.1, src_h: still.calib_frame.1,
        dst_w: half_w, dst_h: half_h, offset_x: 0, offset_y: 0,
    });
    let lens = insv_imu::stream_lens_calib(
        entry, &vr180_fisheye::InsvStreamGeometry::scaled_window(&crop, half_w, half_h), half_w, half_h);
    let omni = lens.omni.ok_or_else(|| Error::Ffmpeg("DNG: lens calibration has no UCM model".into()))?;
    let cx = lens.cx.unwrap_or(half_w as f32 * 0.5);
    let cy = lens.cy.unwrap_or(half_h as f32 * 0.5);
    Ok(FisheyeCalib::new_omni(
        omni.fx, omni.fy, cx, cy, omni.xi, omni.radial, omni.tangential, omni.prism,
        half_w as f32, half_h as f32,
    ))
}

/// The calibration the projection kernels get for one OUTPUT eye: the
/// factory model (or the user's Override), then shaped for the output
/// projection exactly as `fisheye_export::resolve_fisheye_calib` shapes it
/// for video — the fisheye target's output half-FOV, the reframed
/// viewport's ray generator. `eye_w × eye_h` is the output eye frame.
pub fn still_output_calib(
    still: &X6Still,
    half: Half,
    ov: Option<&StillLensOverride>,
    projection: FisheyeExportProjection,
    eye_w: u32,
    eye_h: u32,
) -> Result<FisheyeCalib> {
    let calib = match ov {
        None => still_eye_calib(still, half)?,
        Some(o) => {
            let (w, h) = ((still.width / 2) as f32, still.height as f32);
            // Same formulas as the video resolvers' Override branch.
            let half_fov = o.fov_deg.max(1.0).to_radians() * 0.5;
            let fx = w / (2.0 * half_fov);
            let mut c = FisheyeCalib::new_pure_kb(fx, fx, o.cx_norm * w, o.cy_norm * h, o.k, w, h);
            c.k5 = o.k5;
            c.p1 = o.p[0];
            c.p2 = o.p[1];
            c
        }
    };
    Ok(match projection {
        FisheyeExportProjection::HalfEquirect => calib,
        FisheyeExportProjection::Fisheye =>
            calib.with_output_hfov((FISHEYE_OUT_FULL_FOV_DEG * 0.5).to_radians()),
        FisheyeExportProjection::Reframe { hfov_deg, defish } =>
            calib.with_reframe(hfov_deg, defish, eye_w as f32 / eye_h.max(1) as f32),
    })
}

/// Per-eye output rotations for a still: no stabilization (identity),
/// only the pano / stereo view adjust, exactly as the preview's
/// `render_still_from_pair` composes them.
pub fn still_eye_rotations(view_adjust: &ViewAdjust) -> (EquirectRotation, EquirectRotation) {
    if view_adjust.is_identity() {
        return (EquirectRotation::IDENTITY, EquirectRotation::IDENTITY);
    }
    let (vl, vr) = view_adjust.per_eye_matrices();
    (
        EquirectRotation(mat3_mul_row_major(&EquirectRotation::IDENTITY.0, &vl)),
        EquirectRotation(mat3_mul_row_major(&EquirectRotation::IDENTITY.0, &vr)),
    )
}

/// One projected eye as an `Rgba16Unorm` texture, `eye_w × eye_h`: the
/// equirect kernel for VR180 and the reframed viewport (the calib carries
/// the viewport), the fisheye kernel for the normalized fisheye output.
#[allow(clippy::too_many_arguments)]
fn project_eye(
    device: &Device,
    rgba64: &[u16],
    src_w: u32, src_h: u32,
    eye_w: u32, eye_h: u32,
    rot: EquirectRotation,
    calib: FisheyeCalib,
    projection: FisheyeExportProjection,
) -> Result<wgpu::Texture> {
    let bytes: &[u8] = bytemuck::cast_slice(rgba64);
    if projection == FisheyeExportProjection::Fisheye {
        device.project_fisheye_to_fisheye_texture_16(bytes, 16, src_w, src_h, eye_w, eye_h, rot, calib)
    } else {
        device.project_fisheye_to_equirect_texture_16(bytes, src_w, src_h, eye_w, eye_h, rot, calib)
    }
}

/// Compose two RGBA64LE eyes side by side into interleaved RGB16 (no alpha).
fn compose_rgb16(left: &[u8], right: &[u8], eye_w: u32, eye_h: u32) -> Vec<u16> {
    let (w, h) = (eye_w as usize, eye_h as usize);
    let mut out = vec![0u16; w * 2 * h * 3];
    let l: &[u16] = bytemuck::cast_slice(left);
    let r: &[u16] = bytemuck::cast_slice(right);
    for y in 0..h {
        let row = &mut out[y * w * 2 * 3..(y + 1) * w * 2 * 3];
        for (dst, src) in [(0usize, l), (w, r)] {
            let srow = &src[y * w * 4..(y + 1) * w * 4];
            for x in 0..w {
                let o = (dst + x) * 3;
                row[o] = srow[x * 4];
                row[o + 1] = srow[x * 4 + 1];
                row[o + 2] = srow[x * 4 + 2];
            }
        }
    }
    out
}

// ── The grade as DNG metadata ─────────────────────────────────────────────
//
// The app's colour stack acts on display-encoded (sRGB OETF, ACES-tone-
// mapped) values. A linear DNG cannot carry those stages as pixels without
// baking, but three of them map onto profile metadata that raw developers
// honour (Adobe Camera Raw / Lightroom, Apple ImageIO; validated 2026-10-09
// with exiftool, sips and Adobe DNG Converter):
//
// - white balance (temperature / tint)  → `AsShotNeutral` (universal)
// - the CDL tone (lift / gamma / gain / shadow / highlight) → `ProfileToneCurve`
//   = the app's ACES tone composed with the CDL, as a linear→linear curve
//   (Adobe + Apple; it REPLACES the developer's default curve, so it is
//   written only when the CDL is not identity)
// - saturation → `ProfileLookTable` (Adobe / DNG-SDK developers; Apple ignores)
// - the raw exposure offset → `BaselineExposure`
//
// Per-eye "Matching Eyes" trims cannot be metadata in one DNG (one profile
// for both eyes), so they are baked into the camera-linear data as exact
// per-channel gains. Not carried: the 3D LUT, sharpen, mid-detail, eye
// scale, automatic eye matching (reported in the log).

/// The app's grade translated for the linear DNG.
#[derive(Debug, Clone)]
pub struct GradeMeta {
    pub as_shot_neutral: [f32; 3],
    pub exposure_delta_ev: f32,
    pub tone_curve: Option<Vec<[f32; 2]>>,
    pub look_table: Option<LookTable>,
    /// Anything that changed vs the camera's own profile.
    pub graded: bool,
    /// Stages that cannot travel as metadata (and were not applied).
    pub dropped: Vec<&'static str>,
}

/// CPU twin of `cdl_16.wgsl` (one channel; the stack applies it per
/// channel with scalar knobs).
pub fn cdl_apply(x: f32, cdl: &crate::gpu::CdlParams) -> f32 {
    let hermite = |t: f32| { let t = t.clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
    let mut x = x + cdl.lift * (1.0 - x);
    x *= cdl.gain;
    x += cdl.shadow * hermite(1.0 - 2.0 * x) * 0.6;
    x += cdl.highlight * hermite(2.0 * x - 1.0) * 0.6;
    x = x.clamp(0.0, 1.0);
    x.powf(1.0 / cdl.gamma.max(1e-4))
}

fn srgb_eotf(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn inv3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 { return None; }
    let d = 1.0 / det;
    Some([
        [(a[1][1] * a[2][2] - a[1][2] * a[2][1]) * d, (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * d, (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * d],
        [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) * d, (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * d, (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * d],
        [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) * d, (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * d, (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * d],
    ])
}

fn mul3v(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
     m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
     m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]
}

/// The stack's white-balance stage as linear-sRGB white multipliers: the
/// shader scales the ENCODED channels by `k = (1+0.3T, 1−0.3t, 1−0.3T)`,
/// which is `k^2.2` on linear values to first order.
fn wb_linear_gains(temperature: f32, tint: f32) -> [f32; 3] {
    let k = [1.0 + 0.30 * temperature, 1.0 - 0.30 * tint, 1.0 - 0.30 * temperature];
    [k[0].max(1e-3).powf(2.2), k[1].max(1e-3).powf(2.2), k[2].max(1e-3).powf(2.2)]
}

/// Translate the app's grade into DNG metadata (see the module notes above).
pub fn grade_metadata(plan: &ColorStackPlan, develop: &DevelopParams, color: &DngColorLike) -> GradeMeta {
    let n = color.as_shot_neutral;
    let mut out = GradeMeta {
        as_shot_neutral: n, exposure_delta_ev: develop.exposure_ev, tone_curve: None, look_table: None,
        graded: develop.exposure_ev != 0.0, dropped: Vec::new(),
    };
    // White balance → AsShotNeutral. `M` = camera_to_srgb_matrix maps the
    // UNBALANCED as-shot neutral n to white (M·n = 1). A developer renders a
    // camera pixel p as B·(p ⊘ n') with B = M·diag(n) (B·1 = 1). For the
    // scene neutral p = n to come out as the stack's white `m`:
    // n ⊘ n' = B⁻¹·m = (M⁻¹·m) ⊘ n  ⇒  n' = n ⊘ ((M⁻¹·m) ⊘ n).
    let (t, ti) = (plan.color_grade.temperature, plan.color_grade.tint);
    if t != 0.0 || ti != 0.0 {
        let m = wb_linear_gains(t, ti);
        let cam_to_srgb = raw_develop::camera_to_srgb_matrix(color, WhiteBalance::AsShot);
        if let Some(inv) = inv3(&cam_to_srgb) {
            let u = mul3v(&inv, m);
            let v = [u[0] / n[0].max(1e-6), u[1] / n[1].max(1e-6), u[2] / n[2].max(1e-6)];
            let mut np = [n[0] / v[0].max(1e-6), n[1] / v[1].max(1e-6), n[2] / v[2].max(1e-6)];
            let g = np[1].max(1e-6);
            np = [np[0] / g, 1.0, np[2] / g];
            out.as_shot_neutral = np;
            out.graded = true;
        }
    }
    // Tone → ProfileToneCurve: the app's ACES tone composed with the CDL,
    // sampled as linear→linear. Only when the CDL is not identity, so an
    // ungraded DNG keeps the developer's own default tone.
    if !plan.cdl.is_identity() {
        const N: usize = 129;
        let mut pts: Vec<[f32; 2]> = (0..N).map(|i| {
            let x = i as f32 / (N - 1) as f32;
            let y = srgb_eotf(cdl_apply(raw_develop::srgb_oetf(raw_develop::tone_aces(x)), &plan.cdl));
            [x, y]
        }).collect();
        // Spec pins: first (0,0), last (1,1); keep it monotone.
        pts[0] = [0.0, 0.0];
        pts[N - 1] = [1.0, 1.0];
        for i in 1..N { if pts[i][1] < pts[i - 1][1] { pts[i][1] = pts[i - 1][1]; } }
        out.tone_curve = Some(pts);
        out.graded = true;
    }
    // Saturation → ProfileLookTable (sat scale, hue 0, val 1).
    let sat = plan.color_grade.saturation * develop.saturation;
    if (sat - 1.0).abs() > 1e-4 {
        let s = sat.max(0.0);
        out.look_table = Some(LookTable { dims: [1, 2, 1], data: vec![0.0, s, 1.0, 0.0, s, 1.0], encoding: 0 });
        out.graded = true;
    }
    if plan.lut.is_some() { out.dropped.push("3D LUT"); }
    if !plan.sharpen.is_identity() { out.dropped.push("sharpen"); }
    if !plan.mid_detail.is_identity() { out.dropped.push("mid-detail"); }
    if (plan.eye_scale - 1.0).abs() > 1e-6 { out.dropped.push("eye scale"); }
    if plan.auto_eye_match.is_some() { out.dropped.push("automatic eye matching"); }
    out
}

/// Bake the per-eye "Matching Eyes" trims (CT / tint / exposure, applied
/// `+` to the left eye and `−` to the right) into camera-linear RGBA64, as
/// the camera-space matrix `M⁻¹·diag(m)·M` of the stack's linear gains.
pub fn bake_eye_trims(rgba: &mut [u16], plan: &ColorStackPlan, color: &DngColorLike, is_left: bool) -> bool {
    if plan.eye_match_ct == 0.0 && plan.eye_match_tint == 0.0 && plan.eye_match_exposure == 0.0 {
        return false;
    }
    let sgn = if is_left { 1.0 } else { -1.0 };
    let mut m = wb_linear_gains(sgn * plan.eye_match_ct, sgn * plan.eye_match_tint);
    let e = 2.0_f32.powf(sgn * plan.eye_match_exposure).powf(2.2);
    for c in m.iter_mut() { *c *= e; }
    let cam_to_srgb = raw_develop::camera_to_srgb_matrix(color, WhiteBalance::AsShot);
    let Some(inv) = inv3(&cam_to_srgb) else { return false; };
    // A = inv · diag(m) · M
    let mut a = [[0.0_f32; 3]; 3];
    for i in 0..3 { for j in 0..3 {
        a[i][j] = (0..3).map(|k| inv[i][k] * m[k] * cam_to_srgb[k][j]).sum();
    } }
    for px in rgba.chunks_exact_mut(4) {
        let v = [px[0] as f32, px[1] as f32, px[2] as f32];
        let o = mul3v(&a, v);
        px[0] = o[0].round().clamp(0.0, 65535.0) as u16;
        px[1] = o[1].round().clamp(0.0, 65535.0) as u16;
        px[2] = o[2].round().clamp(0.0, 65535.0) as u16;
    }
    true
}

fn linear_dng_meta(still: &X6Still, l_max: f32, grade: Option<&GradeMeta>) -> LinearDngMeta {
    let c = &still.color;
    let e = &still.exif;
    LinearDngMeta {
        make: if e.make.is_empty() { "Arashi Vision".into() } else { e.make.clone() },
        model: if e.model.is_empty() { "Insta360 X6".into() } else { e.model.clone() },
        unique_camera_model: if e.unique_camera_model.is_empty() { "Insta360 X6".into() } else { e.unique_camera_model.clone() },
        software: format!("VR180 Silver Bullet {}", env!("CARGO_PKG_VERSION")),
        color_matrix1: c.color_matrix1,
        calibration_illuminant1: c.calibration_illuminant1,
        as_shot_neutral: grade.map(|g| g.as_shot_neutral).unwrap_or(c.as_shot_neutral),
        // The gain map is baked into the data at 1/l_max scale: tell the
        // developer the scene is l_max brighter than the codes say — plus
        // the app's raw exposure offset, as metadata.
        baseline_exposure: c.baseline_exposure + l_max.max(1.0).log2() + grade.map(|g| g.exposure_delta_ev).unwrap_or(0.0),
        baseline_noise: c.baseline_noise,
        baseline_sharpness: c.baseline_sharpness,
        profile_hue_sat_map_dims: c.profile_hue_sat_map_dims,
        profile_hue_sat_map_data1: c.profile_hue_sat_map_data1.clone(),
        profile_hue_sat_map_encoding: c.profile_hue_sat_map_encoding,
        profile_name: Some(if grade.map_or(false, |g| g.graded) { "VR180 Silver Bullet - graded".into() } else { "VR180 Silver Bullet linear".into() }),
        profile_embed_policy: Some(0),
        profile_tone_curve: grade.and_then(|g| g.tone_curve.clone()),
        profile_look_table: grade.and_then(|g| g.look_table.clone()),
        exposure_s: e.exposure_s,
        f_number: e.f_number,
        iso: e.iso,
        focal_mm: e.focal_mm,
        focal_35mm: e.focal_35mm,
        datetime_original: e.datetime_original.clone(),
        user_comment: e.user_comment.clone(),
        exposure_program: e.exposure_program,
        metering_mode: e.metering_mode,
        white_balance: e.white_balance,
        exposure_compensation: e.exposure_compensation,
        shutter_speed_value: e.shutter_speed_value,
        image_description: Some("VR180 side-by-side (left | right), half-equirect per eye".into()),
    }
}

/// Export one still. Blocking; call from a worker thread.
pub fn export_still(device: Arc<Device>, cfg: &StillExportConfig, progress: StillProgress) -> Result<()> {
    progress("Reading DNG", 0.0);
    let still = x6dng::read(&cfg.source_path).map_err(|e| Error::Ffmpeg(format!("DNG read: {e}")))?;
    let kandao_lenses = still.kandao.as_ref().map_or(0, |k| k.lenses.len());
    if still.lenses.len() < 2 && kandao_lenses < 2 {
        return Err(Error::Ffmpeg("DNG: no factory lens calibration found".into()));
    }
    let half_w = still.width / 2;
    let half_h = still.height;
    let params = RawParams::from(&still);
    let gain_maps: Vec<GainMapLike> = raw_develop::gain_maps_from(&still.gain_maps);
    let color_like = DngColorLike::from(&still.color);
    let (rot_l, rot_r) = still_eye_rotations(&cfg.view_adjust);
    let linear = cfg.format.is_linear_dng();
    let grade = if linear {
        let g = grade_metadata(&cfg.color_stack, &cfg.develop, &color_like);
        if g.graded || !g.dropped.is_empty() {
            tracing::info!(
                "still export (DNG): grade as metadata — neutral {:?}, exposure {:+.2} EV, tone curve {}, saturation table {}; not carried: {}",
                g.as_shot_neutral, g.exposure_delta_ev, g.tone_curve.is_some(), g.look_table.is_some(),
                if g.dropped.is_empty() { "nothing".to_string() } else { g.dropped.join(", ") },
            );
        }
        Some(g)
    } else { None };

    // Eye order: left half → left eye unless swapped.
    let order = if cfg.swap_eyes { [Half::Right, Half::Left] } else { [Half::Left, Half::Right] };
    let mut eye_tex: Vec<wgpu::Texture> = Vec::with_capacity(2);
    let mut l_max = 1.0_f32;
    for (i, half) in order.iter().enumerate() {
        let label = if i == 0 { "left" } else { "right" };
        progress(&format!("Developing {label} eye"), 0.1 + 0.3 * i as f32);
        let eye: LinearEye = raw_develop::develop_linear(&still.bayer, &params, &gain_maps, *half, false);
        l_max = l_max.max(eye.l_max);
        let rgba: Vec<u16> = if linear {
            let mut v = eye.rgba.clone();
            // Per-eye trims cannot be metadata in one DNG: bake them exactly.
            if bake_eye_trims(&mut v, &cfg.color_stack, &color_like, i == 0) {
                tracing::info!("still export (DNG): per-eye trims baked into the {label} eye");
            }
            v
        } else {
            raw_develop::develop_display(&eye, &color_like, &cfg.develop)
        };
        drop(eye);
        progress(&format!("Projecting {label} eye"), 0.25 + 0.3 * i as f32);
        let calib = still_output_calib(&still, *half, cfg.lens_override[i].as_ref(), cfg.projection, cfg.eye_w, cfg.eye_h)?;
        let rot = if i == 0 { rot_l } else { rot_r };
        eye_tex.push(project_eye(&device, &rgba, half_w, half_h, cfg.eye_w, cfg.eye_h, rot, calib, cfg.projection)?);
    }
    // The colour stack: baked into the developed outputs (same 16-bit GPU
    // stages as the video export, per eye, after projection). The linear
    // DNG keeps camera-native data — its grade travels as metadata.
    if !linear && cfg.color_stack.any_active() {
        progress("Colour stack", 0.8);
        let (l, r) = device.apply_color_stack_pair_16(&eye_tex[0], &eye_tex[1], cfg.eye_w, cfg.eye_h, &cfg.color_stack, None)?;
        if let Some(t) = l { eye_tex[0] = t; }
        if let Some(t) = r { eye_tex[1] = t; }
    }
    progress("Reading back", 0.85);
    let left = device.read_texture_rgba64(&eye_tex[0], cfg.eye_w, cfg.eye_h)?;
    let right = device.read_texture_rgba64(&eye_tex[1], cfg.eye_w, cfg.eye_h)?;
    drop(eye_tex);
    progress("Writing", 0.9);
    let rgb16 = compose_rgb16(&left, &right, cfg.eye_w, cfg.eye_h);
    drop(left); drop(right);
    let (w, h) = (cfg.eye_w * 2, cfg.eye_h);
    write_output(&cfg.output_path, cfg.format, w, h, &rgb16, &still, l_max, grade.as_ref())?;
    progress("Done", 1.0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_output(path: &Path, format: StillFormat, w: u32, h: u32, rgb16: &[u16], still: &X6Still, l_max: f32, grade: Option<&GradeMeta>) -> Result<()> {
    match format {
        StillFormat::Png16 => {
            // The image crate takes 16-bit samples in NATIVE byte order (it
            // swaps for the PNG stream itself); big-endian bytes here came out
            // byte-swapped on little-endian machines.
            let bytes: &[u8] = bytemuck::cast_slice(rgb16);
            image::save_buffer(path, bytes, w, h, image::ExtendedColorType::Rgb16)
                .map_err(|e| Error::Ffmpeg(format!("PNG write: {e}")))?;
        }
        StillFormat::Jpeg { quality } => {
            let rgb8: Vec<u8> = rgb16.iter().map(|v| (v >> 8) as u8).collect();
            let file = std::fs::File::create(path)?;
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), quality.clamp(1, 100));
            use image::ImageEncoder;
            enc.write_image(&rgb8, w, h, image::ExtendedColorType::Rgb8)
                .map_err(|e| Error::Ffmpeg(format!("JPEG write: {e}")))?;
        }
        StillFormat::LinearDng { compression } => {
            let meta = linear_dng_meta(still, l_max, grade);
            let img = LinearDngImage { width: w, height: h, rgb16, white_level: 65535 };
            dng_write::write_linear_dng(path, &img, &meta, compression, None)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{CdlParams, ColorGradeParams};

    fn x6_color() -> DngColorLike {
        DngColorLike {
            color_matrix1: [0.71875, -0.1484375, -0.09375, -0.609375, 1.453125, 0.109375, -0.15625, 0.296875, 0.4765625],
            as_shot_neutral: [0.4599609375, 1.0, 0.537109375],
            baseline_exposure: 2.6,
        }
    }

    #[test]
    fn ungraded_stack_leaves_the_camera_profile_alone() {
        let g = grade_metadata(&ColorStackPlan::default(), &DevelopParams::default(), &x6_color());
        assert!(!g.graded);
        assert_eq!(g.as_shot_neutral, x6_color().as_shot_neutral);
        assert!(g.tone_curve.is_none() && g.look_table.is_none() && g.dropped.is_empty());
    }

    #[test]
    fn warmer_temperature_lowers_the_red_neutral() {
        // +temperature = warmer render: the developer divides by the neutral,
        // so the red neutral must drop relative to green (and blue rise).
        let mut plan = ColorStackPlan::default();
        plan.color_grade = ColorGradeParams { temperature: 0.5, tint: 0.0, saturation: 1.0 };
        let g = grade_metadata(&plan, &DevelopParams::default(), &x6_color());
        let n = x6_color().as_shot_neutral;
        assert!(g.graded);
        assert_eq!(g.as_shot_neutral[1], 1.0);
        assert!(g.as_shot_neutral[0] / g.as_shot_neutral[1] < n[0] / n[1], "{:?}", g.as_shot_neutral);
        assert!(g.as_shot_neutral[2] / g.as_shot_neutral[1] > n[2] / n[1], "{:?}", g.as_shot_neutral);
    }

    #[test]
    fn cdl_becomes_a_pinned_monotone_tone_curve() {
        let mut plan = ColorStackPlan::default();
        plan.cdl = CdlParams { lift: 0.05, gamma: 1.2, gain: 1.1, shadow: 0.1, highlight: -0.1 };
        let g = grade_metadata(&plan, &DevelopParams::default(), &x6_color());
        let tc = g.tone_curve.expect("tone curve");
        assert_eq!(tc[0], [0.0, 0.0]);
        assert_eq!(*tc.last().unwrap(), [1.0, 1.0]);
        assert!(tc.windows(2).all(|w| w[1][0] > w[0][0] && w[1][1] >= w[0][1]));
        // Mid-grey: brighter than linear (ACES + gain + gamma 1.2 lift it).
        let mid = tc[64];
        assert!((mid[0] - 0.5).abs() < 1e-6 && mid[1] > 0.5, "{mid:?}");
    }

    #[test]
    fn saturation_becomes_a_look_table_and_lut_is_reported_dropped() {
        let mut plan = ColorStackPlan::default();
        plan.color_grade = ColorGradeParams { temperature: 0.0, tint: 0.0, saturation: 1.4 };
        plan.lut = Some((vr180_core::color::Cube3DLut {
            size: 2, data: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            title: None, domain_min: [0.0; 3], domain_max: [1.0; 3],
        }, 1.0));
        let g = grade_metadata(&plan, &DevelopParams::default(), &x6_color());
        let lt = g.look_table.expect("look table");
        assert_eq!(lt.dims, [1, 2, 1]);
        assert_eq!(lt.data, vec![0.0, 1.4, 1.0, 0.0, 1.4, 1.0]);
        assert!(g.dropped.contains(&"3D LUT"));
    }

    #[test]
    fn eye_trims_bake_oppositely_and_zero_is_a_no_op() {
        let color = x6_color();
        let mut plan = ColorStackPlan::default();
        let px: Vec<u16> = vec![20000, 30000, 25000, 0, 1000, 2000, 1500, 65535];
        let mut a = px.clone();
        assert!(!bake_eye_trims(&mut a, &plan, &color, true));
        assert_eq!(a, px);
        plan.eye_match_exposure = 0.2;
        let mut l = px.clone(); let mut r = px.clone();
        assert!(bake_eye_trims(&mut l, &plan, &color, true));
        assert!(bake_eye_trims(&mut r, &plan, &color, false));
        assert!(l[1] > px[1] && r[1] < px[1], "left {} right {} src {}", l[1], r[1], px[1]);
        assert_eq!(l[3], 0); assert_eq!(l[7], 65535); // alpha untouched
    }
}
