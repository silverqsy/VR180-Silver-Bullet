//! Automatic stereo extrinsic alignment ("Auto align").
//!
//! Renders both eyes through the exact export dewarp (current calib +
//! current ViewAdjust, stab off) at several timestamps, measures the
//! inter-eye disparity field with normalized cross-correlation over a
//! patch grid, and fits the residual relative rotation between the eyes:
//!
//! - vertical disparity `dy(λ) = pitch + roll·λ` — always solvable: no
//!   scene geometry produces vertical disparity, so it is pure error.
//! - horizontal disparity `dx = yaw + depth(patch)` — yaw is only
//!   solvable from far content (depth → 0 at infinity), so it is gated
//!   on a tight far-cluster and returned as `Option`.
//!
//! Because the render includes the CURRENT ViewAdjust, the result is the
//! correction to ADD to the stereo offset sliders (which apply ± per eye,
//! hence the /2). Re-running after applying should converge to ~0.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::fisheye_decode::{FisheyePairIter, SegmentedFisheyeIter};
use crate::fisheye_export::FisheyeExportConfig;
use crate::gpu::{Device, EquirectRotation};
use crate::{Error, Result};

/// Correction to ADD to the stereo offset sliders.
#[derive(Debug, Clone, Copy)]
pub struct StereoAlignResult {
    pub d_stereo_pitch_deg: f32,
    pub d_stereo_roll_deg: f32,
    /// `None` when the scene had no trustworthy far content.
    pub d_stereo_yaw_deg: Option<f32>,
    /// RMS vertical disparity after the fit, in degrees — doubles as a
    /// per-eye INTRINSIC health metric: with good per-lens calibration
    /// this lands well under ~0.1°; a wrong lens model leaves a spatially
    /// structured residual the rigid-rotation fit cannot absorb.
    pub residual_deg: f32,
    pub patches: usize,
    pub frames: usize,
}

const EYE: u32 = 1024;           // render size per eye for measurement
const PATCH_HALF: usize = 48;    // 96×96 patches
const SEARCH_X: i32 = 30;        // ±px horizontal search (depth disparity)
const SEARCH_Y: i32 = 22;        // ±px vertical search
const MIN_STD: f32 = 3.0;        // texture gate (0-255 luma units)
const MIN_NCC: f32 = 0.40;       // correlation confidence gate
// Sign of slider-vs-measured-disparity, fixed by the synthetic
// closure test in examples/stereo_align_check.rs (apply a known slider
// offset → the solver must return its negation).
const SIGN_PITCH: f32 = 1.0;
const SIGN_ROLL: f32 = -1.0;
const SIGN_YAW: f32 = 1.0;

struct PatchObs {
    lam: f32,   // longitude of patch center (rad, 0 at eye center)
    dx: f32,    // horizontal disparity (rad, R relative to L)
    dy: f32,    // vertical disparity (rad)
}

/// Measure the stereo alignment correction for `cfg` (built exactly like
/// an export config from the current GUI state). Ignores `cfg.eye_w/h`
/// (renders at its own measurement size), stabilization, and encode
/// fields. Fisheye-family sources only (OSV / SBS / BRAW).
pub fn measure_stereo_align(
    pipeline: Arc<Device>,
    cfg: &FisheyeExportConfig,
    cancel: Arc<AtomicBool>,
) -> Result<StereoAlignResult> {
    let total_dur: f64 = cfg.segments.iter()
        .map(|p| crate::decode::probe_video(p).map(|pr| pr.duration_sec).unwrap_or(0.0))
        .sum();
    let t_in = cfg.trim_in_s.unwrap_or(0.0).max(0.0);
    let t_out = cfg.trim_out_s.unwrap_or(total_dur).min(total_dur.max(0.1));
    let span = (t_out - t_in).max(0.1);
    let sample_ts: Vec<f64> = [0.12, 0.30, 0.50, 0.70, 0.88]
        .iter().map(|f| t_in + span * f).collect();

    // View rotation: current ViewAdjust only (no stab — inter-eye
    // disparity is unaffected by the common stabilization rotation).
    let (rot_l, rot_r) = if cfg.view_adjust.is_identity() {
        (EquirectRotation::IDENTITY, EquirectRotation::IDENTITY)
    } else {
        let (v_l, v_r) = cfg.view_adjust.per_eye_matrices();
        (EquirectRotation(v_l), EquirectRotation(v_r))
    };

    let mut obs: Vec<PatchObs> = Vec::new();
    let mut frames_used = 0usize;

    if cfg.source_kind.is_eac() {
        // ── GoPro .360: dual-stream → EAC cross per eye → equirect,
        //    mirroring the portable export_eac loop (left = Lens B). ──
        use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
        let mut decoder = crate::decode::SegmentedStreamPairIter::new(
            &cfg.segments, crate::decode::HwDecode::Auto, 0)?;
        let dims = decoder.dims();
        if !dims.is_valid() {
            return Err(Error::Ffmpeg("auto-align: invalid EAC layout".into()));
        }
        let cross_w = dims.cross_w();
        let cw = cross_w as usize;
        let (lens_l, lens_r) = crate::fisheye_export::resolve_eac_lens_pair(cfg);
        let rs = crate::gpu::EquirectRsParams::DISABLED;
        let mut cross_a = vec![0u8; cw * cw * 3];
        let mut cross_b = vec![0u8; cw * cw * 3];
        for &ts in &sample_ts {
            if cancel.load(Ordering::SeqCst) {
                return Err(Error::Ffmpeg("auto-align cancelled".into()));
            }
            if decoder.seek(ts).is_err() { continue; }
            let Some(pair) = decoder.next_pair()? else { continue };
            assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
            assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);
            let tex_l = pipeline.project_cross_to_equirect_texture(
                &cross_b, cross_w, EYE, EYE, rot_l, rs, &lens_l)?;
            let tex_r = pipeline.project_cross_to_equirect_texture(
                &cross_a, cross_w, EYE, EYE, rot_r, rs, &lens_r)?;
            let gray_l = to_gray(&pipeline.read_texture_rgb8(&tex_l, EYE, EYE)?);
            let gray_r = to_gray(&pipeline.read_texture_rgb8(&tex_r, EYE, EYE)?);
            let n_before = obs.len();
            collect_patches(&gray_l, &gray_r, &mut obs);
            if obs.len() > n_before {
                frames_used += 1;
            }
        }
    } else {
        // ── fisheye family: same opener as the portable export, 8-bit ──
        let mut open = crate::fisheye_export::fisheye_export_opener(
            cfg.source_kind, cfg.fisheye_swap_eyes, 8);
        let mut decoder: Box<dyn FisheyePairIter> = if cfg.segments.len() > 1 {
            let durations: Vec<f64> = cfg.segments.iter()
                .map(|p| crate::decode::probe_video_duration_via_moov(p).ok()
                    .or_else(|| crate::decode::probe_video(p).ok().map(|pr| pr.duration_sec))
                    .unwrap_or(0.0))
                .collect();
            Box::new(SegmentedFisheyeIter::new(&cfg.segments, &durations, open)?)
        } else {
            open(&cfg.source_path)?
        };

        // per-lens calib (same resolver as export/preview)
        let osv = if cfg.source_kind == crate::SourceKind::DjiOsv {
            crate::decode::extract_dji_calib_blob(&cfg.source_path).ok()
                .and_then(|b| vr180_fisheye::DjiOsvImu::parse(&b).ok())
        } else {
            None
        };
        let mut calib_pair = None;

        for &ts in &sample_ts {
            if cancel.load(Ordering::SeqCst) {
                return Err(Error::Ffmpeg("auto-align cancelled".into()));
            }
            if decoder.seek(ts).is_err() { continue; }
            let Some(pair) = decoder.next_pair()? else { continue };
            let (src_w, src_h) = (pair.eye_w, pair.eye_h);
            let (calib_l, calib_r) = *calib_pair.get_or_insert_with(|| {
                crate::fisheye_export::resolve_calib_pair(cfg, src_w, src_h, osv.as_ref())
            });

            let tex_l = pipeline.project_fisheye_to_equirect_texture(
                &pair.left, src_w, src_h, EYE, EYE, rot_l, calib_l, 10)?;
            let tex_r = pipeline.project_fisheye_to_equirect_texture(
                &pair.right, src_w, src_h, EYE, EYE, rot_r, calib_r, 11)?;
            let gray_l = to_gray(&pipeline.read_texture_rgb8(&tex_l, EYE, EYE)?);
            let gray_r = to_gray(&pipeline.read_texture_rgb8(&tex_r, EYE, EYE)?);

            let n_before = obs.len();
            collect_patches(&gray_l, &gray_r, &mut obs);
            if obs.len() > n_before {
                frames_used += 1;
            }
        }
    }

    if obs.len() < 12 || frames_used == 0 {
        return Err(Error::Ffmpeg(format!(
            "auto-align: not enough usable patches ({} across {} frames) — \
             needs textured, unblurred content", obs.len(), frames_used)));
    }

    // ── robust fit of the exact small-rotation model on equirect:
    //      dφ(λ) = R·sin λ − P·cos λ      (φ-independent)
    // where P = relative pitch, R = relative roll between the eyes. We
    // fit dy = a·cos λ + b·sin λ (IRLS with hard outlier rejection) and
    // map (a, b) → (pitch, roll) with the SIGN_* constants (fixed by the
    // synthetic closure test in examples/stereo_align_check.rs).
    let mut keep: Vec<bool> = vec![true; obs.len()];
    let (mut a, mut b) = (0.0f32, 0.0f32);
    let model = |a: f32, b: f32, o: &PatchObs| a * o.lam.cos() + b * o.lam.sin();
    for _round in 0..3 {
        let (mut scc, mut scs, mut sss, mut syc, mut sys, mut n) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (o, k) in obs.iter().zip(&keep) {
            if !*k { continue; }
            let (c, s2, y) = (o.lam.cos() as f64, o.lam.sin() as f64, o.dy as f64);
            scc += c * c; scs += c * s2; sss += s2 * s2;
            syc += y * c; sys += y * s2;
            n += 1.0;
        }
        if n < 8.0 { break; }
        let det = scc * sss - scs * scs;
        if det.abs() < 1e-9 { break; }
        a = ((sss * syc - scs * sys) / det) as f32;
        b = ((scc * sys - scs * syc) / det) as f32;
        // residual-based rejection
        let mut resid: Vec<f32> = obs.iter().zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(o, _)| (o.dy - model(a, b, o)).abs())
            .collect();
        resid.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mad = resid[resid.len() / 2].max(1e-5);
        for (o, k) in obs.iter().zip(keep.iter_mut()) {
            if (o.dy - model(a, b, o)).abs() > 3.5 * mad {
                *k = false;
            }
        }
    }
    let kept: Vec<&PatchObs> = obs.iter().zip(&keep)
        .filter(|(_, k)| **k).map(|(o, _)| o).collect();
    let rms = (kept.iter()
        .map(|o| { let r = o.dy - model(a, b, o); (r * r) as f64 })
        .sum::<f64>() / kept.len().max(1) as f64)
        .sqrt() as f32;

    // ── yaw from the far cluster of dx (gated) ──
    let mut dxs: Vec<f32> = kept.iter().map(|o| o.dx).collect();
    dxs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let d_yaw = solve_yaw(&dxs);

    Ok(StereoAlignResult {
        d_stereo_pitch_deg: SIGN_PITCH * 0.5 * a.to_degrees(),
        d_stereo_roll_deg:  SIGN_ROLL  * 0.5 * b.to_degrees(),
        d_stereo_yaw_deg:   d_yaw.map(|y| SIGN_YAW * 0.5 * y.to_degrees()),
        residual_deg: rms.to_degrees(),
        patches: kept.len(),
        frames: frames_used,
    })
}

/// Far content sits at zero disparity; near content spreads to ONE side
/// of the dx distribution. The far end is whichever tail is more tightly
/// clustered. Gated: the cluster must be tight and populated, else None.
fn solve_yaw(sorted_dx: &[f32]) -> Option<f32> {
    if sorted_dx.len() < 16 {
        return None;
    }
    let n = sorted_dx.len();
    let px = std::f32::consts::PI / EYE as f32; // 1 measurement pixel, rad
    let q = |f: f32| sorted_dx[((n - 1) as f32 * f) as usize];
    // The far end is whichever tail is more tightly clustered (depth
    // disparity spreads near content toward one side only; far content
    // piles up at its zero).
    let lo_spread = (q(0.15) - q(0.02)).abs();
    let hi_spread = (q(0.98) - q(0.85)).abs();
    let far_anchor = if lo_spread < hi_spread { q(0.06) } else { q(0.94) };
    // Cluster = members within ~2.2 px of the anchor (NCC subpixel noise
    // is ~0.3 px; depth between ~10 m and infinity spans under 1 px).
    let members: Vec<f32> = sorted_dx.iter().copied()
        .filter(|d| (d - far_anchor).abs() < 2.2 * px)
        .collect();
    if members.len() < (n / 5).max(10) {
        return None;
    }
    // The cluster itself must be dense (a smeared mid-depth tail is not
    // "far"): inner-quartile spread under ~1.3 px.
    let m = members.len();
    let iqr = (members[m * 3 / 4] - members[m / 4]).abs();
    if iqr > 1.3 * px {
        return None;
    }
    Some(members[m / 2])
}

fn to_gray(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect()
}

fn collect_patches(gray_l: &[f32], gray_r: &[f32], out: &mut Vec<PatchObs>) {
    let n = EYE as usize;
    let centers: Vec<usize> = (0..6).map(|i| 192 + i * 128).collect();
    for &py in &centers {
        for &px in &centers {
            if let Some((dx, dy)) = ncc_disparity(gray_l, gray_r, n, px, py) {
                let lam = (px as f32 / n as f32 - 0.5) * std::f32::consts::PI;
                out.push(PatchObs {
                    lam,
                    dx: dx * std::f32::consts::PI / n as f32,
                    dy: dy * std::f32::consts::PI / n as f32,
                });
            }
        }
    }
}

/// NCC of the left patch against the right image over the search window;
/// subpixel via parabolic refinement. Returns (dx, dy) = position of the
/// matching content in R relative to L, in pixels.
fn ncc_disparity(l: &[f32], r: &[f32], n: usize, cx: usize, cy: usize) -> Option<(f32, f32)> {
    let h = PATCH_HALF;
    let (x0, y0) = (cx - h, cy - h);
    let side = 2 * h;
    // left patch stats
    let mut lm = 0.0f32;
    for y in 0..side { for x in 0..side { lm += l[(y0 + y) * n + x0 + x]; } }
    lm /= (side * side) as f32;
    let mut lv = 0.0f32;
    let mut lp = vec![0.0f32; side * side];
    for y in 0..side {
        for x in 0..side {
            let v = l[(y0 + y) * n + x0 + x] - lm;
            lp[y * side + x] = v;
            lv += v * v;
        }
    }
    let lnorm = lv.sqrt();
    if lnorm / side as f32 <= MIN_STD { return None; }

    let mut best = (0i32, 0i32, -2.0f32);
    let mut scores = std::collections::HashMap::new();
    let mut eval = |sx: i32, sy: i32, r: &[f32]| -> f32 {
        let bx = x0 as i32 + sx; let by = y0 as i32 + sy;
        if bx < 0 || by < 0
            || bx as usize + side >= n || by as usize + side >= n {
            return -2.0;
        }
        let (bx, by) = (bx as usize, by as usize);
        let mut rm = 0.0f32;
        for y in 0..side { for x in 0..side { rm += r[(by + y) * n + bx + x]; } }
        rm /= (side * side) as f32;
        let (mut num, mut rv) = (0.0f32, 0.0f32);
        for y in 0..side {
            for x in 0..side {
                let v = r[(by + y) * n + bx + x] - rm;
                num += v * lp[y * side + x];
                rv += v * v;
            }
        }
        let d = lnorm * rv.sqrt();
        if d < 1e-3 { -2.0 } else { num / d }
    };
    // coarse (step 2) then fine pass around the coarse peak
    for sy in (-SEARCH_Y..=SEARCH_Y).step_by(2) {
        for sx in (-SEARCH_X..=SEARCH_X).step_by(2) {
            let s = eval(sx, sy, r);
            scores.insert((sx, sy), s);
            if s > best.2 { best = (sx, sy, s); }
        }
    }
    for sy in (best.1 - 2)..=(best.1 + 2) {
        for sx in (best.0 - 2)..=(best.0 + 2) {
            let s = *scores.entry((sx, sy)).or_insert_with(|| eval(sx, sy, r));
            if s > best.2 { best = (sx, sy, s); }
        }
    }
    let (bx, by, bs) = best;
    if bs < MIN_NCC
        || bx.abs() >= SEARCH_X - 1 || by.abs() >= SEARCH_Y - 1 {
        return None;
    }
    // parabolic subpixel on each axis
    let g = |sx: i32, sy: i32, sc: &mut std::collections::HashMap<(i32, i32), f32>| {
        *sc.entry((sx, sy)).or_insert_with(|| eval(sx, sy, r))
    };
    let (xm, xc, xp) = (g(bx - 1, by, &mut scores), bs, g(bx + 1, by, &mut scores));
    let (ym, yc, yp) = (g(bx, by - 1, &mut scores), bs, g(bx, by + 1, &mut scores));
    let sub = |m: f32, c: f32, p: f32| -> f32 {
        let d = m - 2.0 * c + p;
        if d.abs() < 1e-6 { 0.0 } else { (0.5 * (m - p) / d).clamp(-1.0, 1.0) }
    };
    Some((bx as f32 + sub(xm, xc, xp), by as f32 + sub(ym, yc, yp)))
}
