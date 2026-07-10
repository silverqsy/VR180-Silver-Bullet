//! Decoder worker thread.
//!
//! Owns a `StreamPairIter`, pulls frames sequentially, runs the
//! existing GPU pipeline (`nv12_to_eac_cross` → `project_*_to_equirect_texture`
//! → `compose_sbs_bgra`), and ships the final `wgpu::Texture` to the
//! UI thread over a bounded channel.
//!
//! Wall-clock pacing + back-pressure mean playback feels real-time:
//! - If we're behind wall-clock, **skip the GPU render** for that
//!   frame (the decoder still advances — HEVC needs sequential
//!   reference frames).
//! - If we're ahead, `thread::sleep` to pace.
//! - The frame channel is bounded (2). If the UI hasn't consumed
//!   the previous frame yet, `try_send` returns Full and we drop.
//!
//! We use the **CPU-assemble** decode path for portability (no
//! IOSurface code needed here). Decode happens in software/HW via
//! VideoToolbox on macOS; EAC assembly runs CPU-side; everything
//! from `nv12_to_eac_cross` onward stays on the GPU.

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use vr180_pipeline::gpu::{
    Device, EquirectRotation, EquirectRsParams,
};
#[allow(unused_imports)]
use vr180_pipeline::gpu::ColorStackPlan;

#[cfg(target_os = "macos")]
use vr180_pipeline::{decode::ZeroCopyStreamPairIter, gpu::Lens};
// `run_cpu_assemble` (the cross-platform CPU EAC fallback) is compiled on
// every platform, so the symbols it references must be in scope everywhere —
// not only on non-macOS (macOS just never reaches it at runtime).
use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
use vr180_pipeline::decode::HwDecode;
#[cfg(not(target_os = "macos"))]
#[allow(unused_imports)]
use vr180_pipeline::decode::iter_stream_pairs;

/// Serde default + pre-file-load seed for the (non-persisted) IMU-phase
/// slider: SROT/2 at 30 fps (9.15 ms). Refreshed to the actual clip fps in
/// `App::load_file` once a file is open.
fn default_dji_imu_phase_ms() -> f32 {
    vr180_pipeline::dji_imu::dji_imu_phase_default_ms_for_fps(30.0)
}

/// PER-CLIP GoPro RS defaults. `rs_mode` + `rs_readout_ms` are `#[serde(skip)]`
/// and re-seeded to these on every clip load (see `app.rs`), so a prior clip's
/// tweak never carries over — same lifecycle as the IMU phase. (`rs_correct`
/// stays a persisted preference.) 15.224 ms is the GoPro Max firmware SROT.
pub(crate) fn default_rs_mode() -> RsMode { RsMode::Firmware }
pub(crate) fn default_rs_readout_ms() -> f32 { 15.224 }

/// Firmware-RS detector from the GoPro CORI stream — the signal that tells a
/// firmware-stabilised clip from a no-firmware one.
///
/// A no-firmware clip's CORI is raw gyro integration starting from an
/// assumed-identity orientation, so `cori[0]` ≈ exact identity (the real
/// physical tilt lives in GRAV, not CORI) and only drifts later from gyro
/// bias. A firmware-RS clip's `cori[0]` already carries a real initial
/// orientation because the in-camera EIS/RS pipeline ran. This is the SAME
/// signal `CoriSource::Auto` keys on — verified on the test set: no-firmware
/// `cori[0]` xyz≈6e-5 vs firmware ≥2.5e-3 (threshold 5e-4).
///
/// Returns `true` when the firmware pipeline was OFF. Empty CORI counts as
/// "off" to mirror the `CoriSource::Auto` routing; callers that want a
/// no-signal fallback guard on `cori.len()` first (see [`detect_rs_mode`]).
pub(crate) fn cori_indicates_no_firmware(cori: &[vr180_core::gyro::Quat]) -> bool {
    let head = &cori[..cori.len().min(10)];
    let cori_is_zero = head.iter().all(|q|
        q.w.abs() < 0.01 && q.x.abs() < 0.01
            && q.y.abs() < 0.01 && q.z.abs() < 0.01);
    let cori_is_bias_drift = !cori_is_zero
        && cori.len() > 30
        && {
            let q0 = cori[0];
            q0.x.abs().max(q0.y.abs()).max(q0.z.abs()) < 0.0005
        };
    cori_is_zero || cori_is_bias_drift
}

/// Auto-seed the per-clip RS mode from the CORI stream. Needs a real CORI
/// stream (≥30 samples) to trust the signal; otherwise keeps the firmware
/// default (matches prior behaviour for files without usable gyro). The UI
/// toggle still overrides this seed.
pub(crate) fn detect_rs_mode(cori: &[vr180_core::gyro::Quat]) -> RsMode {
    if cori.len() < 30 {
        return default_rs_mode();
    }
    let mode = if cori_indicates_no_firmware(cori) {
        RsMode::NoFirmware
    } else {
        RsMode::Firmware
    };
    let q0 = cori[0];
    tracing::info!(
        "rs_mode auto-detect: cori[0] xyz_max={:.2e} over {} samples → {}",
        q0.x.abs().max(q0.y.abs()).max(q0.z.abs()), cori.len(), mode.as_str()
    );
    mode
}

/// Path-based convenience for the batch-add path (no CORI parsed yet). Only
/// GoPro `.360` (EAC) has the firmware-RS distinction; every other source
/// keeps the firmware default.
pub(crate) fn detect_rs_mode_for_path(
    path: &std::path::Path,
    kind: vr180_pipeline::SourceKind,
) -> RsMode {
    if kind != vr180_pipeline::SourceKind::GoProEac {
        return default_rs_mode();
    }
    match extract_gpmf_cached(path) {
        Ok(gpmf) => detect_rs_mode(&vr180_core::gyro::parse_cori(&gpmf)),
        Err(_) => default_rs_mode(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub stabilize: bool,
    pub cori_source: CoriSource,
    pub smooth_ms: f32,
    pub max_corr_deg: f32,
    /// GoPro soft-stab velocity→smoothing response curve (Python's
    /// "Responsiveness" slider, 0.2–3.0). < 1 = follows motion early,
    /// 1 = linear, > 1 = holds longer then catches up. Mirrors the
    /// OSV panel's Response slider.
    pub gyro_responsiveness: f32,
    /// DJI OSV stabilization smoothing window. `0.0` = sharp
    /// per-frame camera-lock (legacy default — bit-identical to the
    /// pre-slider build). > 0 lowpasses the per-frame correction
    /// quats over that window for a softer lock.
    pub dji_smooth_ms: f32,
    /// DJI OSV stabilization soft-cap on correction angle. `0.0` =
    /// no cap (legacy default — unlimited correction). > 0 caps the
    /// per-frame rotation magnitude in degrees.
    pub dji_max_corr_deg: f32,
    /// Soft-stab velocity→smoothing response curve (Python "Response"
    /// slider, 0.2–3.0). < 1 = follows motion early (anticipatory),
    /// 1 = linear, > 1 = holds longer then catches up (cinematic lag).
    /// Only used when `dji_smooth_ms > 0`.
    pub dji_responsiveness: f32,
    /// IMU sample offset, ms after frame_start. Defaults to **SROT/2** (the
    /// readout-window midpoint; 9.15 ms @30 fps, 8.11 ms @50 fps) and is
    /// **refreshed on every file load** to the new clip's fps. A live
    /// A/B-test knob for stabilization timing vs DJI Studio; feeds BOTH the
    /// per-frame stab sample and the rolling-shutter window (coupled, as DJI
    /// does). NOT persisted (`#[serde(skip)]`) — each session/clip starts at
    /// the SROT/2 default regardless of prior tweaks.
    #[serde(skip, default = "default_dji_imu_phase_ms")]
    pub dji_imu_phase_ms: f32,
    /// Global pano-map adjustment angles (degrees). Shared between
    /// eyes. Applied as `R_view = R_y(yaw) · R_x(pitch) · R_z(roll)`
    /// composed AFTER stabilization. All-zero default = identity =
    /// bit-identical to the no-pano-map pipeline.
    pub pano_yaw_deg: f32,
    pub pano_pitch_deg: f32,
    pub pano_roll_deg: f32,
    /// Per-eye stereo offset angles (degrees). Sign-flips between
    /// eyes: right = pano + stereo, left = pano − stereo. Same
    /// convention as the Python app at `vr180_gui.py:12433-12438`.
    pub stereo_yaw_deg: f32,
    pub stereo_pitch_deg: f32,
    pub stereo_roll_deg: f32,
    pub rs_correct: bool,
    /// RS mode + readout are PER-CLIP, not persisted: each clip load resets them
    /// to the GoPro firmware default (`default_rs_*`) so a prior clip's tweak
    /// never carries over. `rs_correct` (the on/off) stays a persisted preference.
    #[serde(skip, default = "default_rs_mode")]
    pub rs_mode: RsMode,
    #[serde(skip, default = "default_rs_readout_ms")]
    pub rs_readout_ms: f32,
    pub preview_eye_w: u32,
    /// Trim-in time in seconds. `None` = play from clip start.
    /// Honored by the decoder loop: when playback hits `trim_out_s`,
    /// it seeks back to `trim_in_s` (or 0 if `None`) and continues.
    pub trim_in_s: Option<f64>,
    /// Trim-out time in seconds. `None` = play to clip end.
    pub trim_out_s: Option<f64>,

    // ─── Per-eye lens settings (fisheye sources + GoPro EAC) ─────
    //
    // For fisheye sources (OSV / SBS / BRAW) the Override fields drive
    // the KB dewarp directly. For GoPro EAC (.360) the SAME fields
    // drive the ".360 Lens calibration" re-dewarp warp (seeded from the
    // file's GEOC factory model; see `resolve_eac_lens_pair`) — values
    // never collide across kinds because settings persist per source
    // kind. Decoders re-read on settings_generation bump.

    /// Selected camera preset by name. Empty / "Auto" → pick based on
    /// file extension (`.osv` → DJI Osmo 360, `.braw` → Pyxis 12K).
    /// Any other value must match a preset in `vr180_fisheye::presets`.
    pub fisheye_preset: String,
    /// Per-eye manual-override master switch. When OFF, that eye uses the
    /// in-file (OSV protobuf) / preset calibration and ALL the manual
    /// fields below are ignored. When ON, the eye is fully described by
    /// the manual fov / cx / cy / k below.
    pub fisheye_override_left: bool,
    pub fisheye_override_right: bool,
    /// One-time sanitation marker for the ".360 Lens calibration" feature.
    /// Settings written BEFORE the feature carried DEAD Override fields in
    /// the "eac" kind bucket (typically stale values inherited from the
    /// legacy single-settings migration) — left alone they would engage the
    /// new EAC re-dewarp warp with garbage. Files lacking this field
    /// deserialize `false` (field-level serde default), which makes the
    /// loaders clear the EAC Override flags ONCE and set this true; fresh
    /// settings are born `true` (nothing to sanitize), so a deliberate
    /// .360 Override set after this build persists normally.
    #[serde(default)]
    pub eac_lens_sanitized: bool,
    /// Per-eye manual full FOV in degrees (used only when override is on).
    /// Converted to fx via `image_w / (2 · half_fov)`.
    pub fisheye_fov_deg_left: f32,
    pub fisheye_fov_deg_right: f32,
    /// Per-eye manual KB-4 distortion coefficients (override on).
    pub fisheye_k_left: [f32; 4],
    pub fisheye_k_right: [f32; 4],
    /// Per-eye manual override for the 5th KB radial coefficient (k5).
    /// OSV-only (DJI's 5-coeff model); 0 = none. In Auto mode k5 is loaded
    /// from the file regardless of this. Serde-defaulted so old settings
    /// files load fine.
    pub fisheye_k5_left: f32,
    pub fisheye_k5_right: f32,
    /// Per-eye manual override for the Brown-Conrady tangential coeffs
    /// `[p1, p2]` (OSV field 20). OSV-only; `[0,0]` = none. In Auto mode
    /// these are loaded from the file regardless. Serde-defaulted so old
    /// settings files load fine.
    pub fisheye_p_left: [f32; 2],
    pub fisheye_p_right: [f32; 2],
    /// Per-eye principal point, NORMALIZED to [0,1] of the frame
    /// (`cx_norm = cx_px / image_w`). Stored normalized so the same value
    /// is correct at the capped preview resolution and the full export
    /// resolution; the UI displays/edits it as absolute native pixels.
    /// Used only when override is on. `0.5` = image center.
    pub fisheye_cx_norm_left: f32,
    pub fisheye_cy_norm_left: f32,
    pub fisheye_cx_norm_right: f32,
    pub fisheye_cy_norm_right: f32,
    /// For DJI OSV / dual-stream sources: swap L↔R after decode. The
    /// Python app exposes this as `swap_eyes` at vr180_gui.py:3554.
    pub fisheye_swap_eyes: bool,
    /// Camera mounted upside-down. Implies BOTH a 180° in-plane output
    /// rotation (via `ViewAdjust::upside_down`) AND an L↔R eye swap
    /// (the rig flip mirrors the eye positions) — see
    /// [`Settings::effective_swap_eyes`].
    pub camera_upside_down: bool,
    /// Output projection. `HalfEquirect` is the standard VR180 output
    /// (left-right hemisphere). `Fisheye` skips the projection entirely
    /// and writes the raw fisheye eyes as SBS — useful for VFX or
    /// re-grade pipelines.
    pub fisheye_output_mode: FisheyeOutputMode,
    /// Export audio track choice for `.360` (stereo / ambisonic / APAC).
    pub audio_format: AudioFormat,

    // ─── Color grade ─────────────────────────────────────────────
    // All defaults are identity (no change). Mirrors
    // `_default_processing_config` in `vr180_gui.py`.

    /// Lift (Resolve "Offset"). Raises the black point; positive
    /// values lift shadows.
    pub lift: f32,
    /// Gamma (Resolve "Pow"). Reciprocal applied to mids. 1.0 = neutral.
    pub gamma: f32,
    /// Gain (Resolve "Slope"). Multiplier on highlights. 1.0 = neutral.
    pub gain: f32,
    /// Shadow smooth-rolloff zone adjust [-1..+1].
    pub shadow: f32,
    /// Highlight smooth-rolloff zone adjust [-1..+1].
    pub highlight: f32,
    /// Temperature [-1..+1]. Positive = warmer, negative = cooler.
    pub temperature: f32,
    /// Tint [-1..+1]. Positive = magenta, negative = green.
    pub tint: f32,
    /// Saturation. 1.0 = neutral, 0.0 = grayscale, 2.0 = double.
    pub saturation: f32,
    /// "Matching Eyes" inter-eye white-balance trim [-1..+1]. Applied
    /// OPPOSITELY to the two eyes (left `+`, right `−`) to correct an
    /// inter-lens color discrepancy without shifting overall color. 0 = off.
    pub eye_match_ct: f32,
    /// "Matching Eyes" inter-eye tint trim [-1..+1] (opposite per eye).
    pub eye_match_tint: f32,
    /// Optional 3D LUT file path. Empty string = no LUT.
    /// Equirect-aware unsharp-mask amount (0 = off, 0.5 subtle, 1 mod,
    /// 2 strong). Ported from the Python app; applies to every source.
    pub sharpen_amount: f32,
    /// Unsharp-mask radius / Gaussian sigma (px). Only used when
    /// `sharpen_amount > 0`.
    pub sharpen_radius: f32,
    /// Temporal noise-reduction strength (0 = off, 1 = max). macOS-only
    /// (VideoToolbox `VTTemporalNoiseFilter`), applied to the source fisheye
    /// frames before projection on export — the in-process equivalent of the
    /// Python app's "VT noise reduction". Ignored where unsupported.
    pub denoise_strength: f32,
    pub lut_path: String,
    /// LUT blend strength [0..1].
    pub lut_intensity: f32,

    /// Preview composer mode — SBS / anaglyph / 50% overlay / single eye.
    pub preview_mode: PreviewMode,
    /// In Single-eye mode, which eye to show: false = left, true = right.
    /// The "switch eye" toggle flips this; the viewpoint (zoom/pan) is kept.
    pub preview_eye_right: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewMode { Sbs, Anaglyph, Overlay50, SingleEye }

impl PreviewMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PreviewMode::Sbs       => "SBS",
            PreviewMode::Anaglyph  => "Anaglyph (red/cyan)",
            PreviewMode::Overlay50 => "50% overlay",
            PreviewMode::SingleEye => "Single eye",
        }
    }
    pub fn to_pipeline(self) -> vr180_pipeline::gpu::PreviewMode {
        match self {
            PreviewMode::Sbs       => vr180_pipeline::gpu::PreviewMode::Sbs,
            PreviewMode::Anaglyph  => vr180_pipeline::gpu::PreviewMode::Anaglyph,
            PreviewMode::Overlay50 => vr180_pipeline::gpu::PreviewMode::Overlay50,
            PreviewMode::SingleEye => vr180_pipeline::gpu::PreviewMode::SingleEye,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            stabilize: true,
            cori_source: CoriSource::Auto,
            // Defaults match across both paths: 1000 ms soft-stab +
            // 15° soft cap. Drop smooth_ms to 0 for sharp camera-lock
            // to frame 0 instead.
            smooth_ms: 1000.0,
            max_corr_deg: 15.0,
            gyro_responsiveness: 1.0,
            sharpen_amount: 0.0,
            sharpen_radius: 1.5,
            denoise_strength: 0.0,
            dji_smooth_ms: 1200.0,
            dji_max_corr_deg: 15.0,
            dji_responsiveness: 1.0,
            dji_imu_phase_ms: default_dji_imu_phase_ms(),
            pano_yaw_deg: 0.0,
            pano_pitch_deg: 0.0,
            pano_roll_deg: 0.0,
            stereo_yaw_deg: 0.0,
            stereo_pitch_deg: 0.0,
            stereo_roll_deg: 0.0,
            rs_correct: true,
            rs_mode: default_rs_mode(),
            rs_readout_ms: default_rs_readout_ms(),
            preview_eye_w: 768,
            trim_in_s: None,
            trim_out_s: None,
            fisheye_preset: String::new(),
            fisheye_override_left: false,
            fisheye_override_right: false,
            // Fresh settings never carry stale pre-feature EAC Override
            // values — born sanitized. (Old persisted files deserialize
            // `false` here via the field-level serde default.)
            eac_lens_sanitized: true,
            fisheye_fov_deg_left: 0.0,
            fisheye_fov_deg_right: 0.0,
            fisheye_k_left: [0.0, 0.0, 0.0, 0.0],
            fisheye_k_right: [0.0, 0.0, 0.0, 0.0],
            fisheye_k5_left: 0.0,
            fisheye_k5_right: 0.0,
            fisheye_p_left: [0.0, 0.0],
            fisheye_p_right: [0.0, 0.0],
            fisheye_cx_norm_left: 0.5,
            fisheye_cy_norm_left: 0.5,
            fisheye_cx_norm_right: 0.5,
            fisheye_cy_norm_right: 0.5,
            fisheye_swap_eyes: false,
            camera_upside_down: false,
            fisheye_output_mode: FisheyeOutputMode::HalfEquirect,
            audio_format: AudioFormat::Stereo,
            lift: 0.0,
            gamma: 1.0,
            gain: 1.0,
            shadow: 0.0,
            highlight: 0.0,
            temperature: 0.0,
            tint: 0.0,
            saturation: 1.0,
            eye_match_ct: 0.0,
            eye_match_tint: 0.0,
            lut_path: String::new(),
            lut_intensity: 1.0,
            preview_mode: PreviewMode::Sbs,
            preview_eye_right: false,
        }
    }
}

/// The DJI Osmo 360 "D-Log M → Rec.709" conversion LUT, embedded in the
/// binary so it's always available regardless of any DaVinci install.
pub const BUILTIN_OSMO_LUT: &str =
    include_str!("../assets/dji_osmo360_dlogm_to_rec709.cube");
/// Sentinel `lut_path` value that means "use the embedded Osmo 360 LUT".
/// (Chosen so it can never collide with a real filesystem path.)
pub const BUILTIN_OSMO_LUT_PATH: &str = "<builtin:osmo360-dlogm-rec709>";
/// Friendly name shown in the LUT picker for the builtin.
pub const BUILTIN_OSMO_LUT_NAME: &str = "DJI Osmo 360 D-LogM→709 (built-in)";

/// The GoPro "Recommended Lut GPLOG" (GP-Log → Rec.709), embedded —
/// byte-identical to the Python app's bundled `Recommended Lut
/// GPLOG.cube`, which it auto-applies to `.360` clips.
pub const BUILTIN_GPLOG_LUT: &str =
    include_str!("../assets/gopro_gplog_recommended.cube");
/// Sentinel `lut_path` value for the embedded GoPro GP-Log LUT.
pub const BUILTIN_GPLOG_LUT_PATH: &str = "<builtin:gopro-gplog-rec709>";
/// Friendly name shown in the LUT picker for the GoPro builtin.
pub const BUILTIN_GPLOG_LUT_NAME: &str = "GoPro GP-Log→709 (built-in)";

thread_local! {
    /// Per-thread cache of the most-recently-loaded parsed LUT, keyed by
    /// `lut_path`. `build_color_stack` is called once per preview frame,
    /// so without this the (≈1 MB) `.cube` would be re-read and re-parsed
    /// every frame. We keep just one entry — the LUT path changes rarely.
    static LUT_CACHE: std::cell::RefCell<Option<(String, vr180_core::Cube3DLut)>>
        = std::cell::RefCell::new(None);
}

/// Load + parse the LUT for `lut_path`, caching the parsed result per
/// thread. Resolves the builtin sentinel to the embedded `.cube`.
pub(crate) fn load_lut_cached(lut_path: &str) -> Option<vr180_core::Cube3DLut> {
    LUT_CACHE.with(|cache| {
        if let Some((k, lut)) = cache.borrow().as_ref() {
            if k == lut_path { return Some(lut.clone()); }
        }
        let parsed = if lut_path == BUILTIN_OSMO_LUT_PATH {
            match vr180_core::Cube3DLut::from_str(BUILTIN_OSMO_LUT) {
                Ok(l) => Some(l),
                Err(e) => { tracing::warn!("builtin Osmo LUT parse failed: {e}"); None }
            }
        } else if lut_path == BUILTIN_GPLOG_LUT_PATH {
            match vr180_core::Cube3DLut::from_str(BUILTIN_GPLOG_LUT) {
                Ok(l) => Some(l),
                Err(e) => { tracing::warn!("builtin GP-Log LUT parse failed: {e}"); None }
            }
        } else {
            let p = std::path::Path::new(lut_path);
            if p.exists() {
                match vr180_core::Cube3DLut::from_file(p) {
                    Ok(l) => Some(l),
                    Err(e) => { tracing::warn!("LUT load failed ({lut_path}): {e}"); None }
                }
            } else { None }
        };
        if let Some(l) = &parsed {
            *cache.borrow_mut() = Some((lut_path.to_string(), l.clone()));
        }
        parsed
    })
}

impl Settings {
    /// The L↔R eye swap that should actually be applied: the manual
    /// "Swap L↔R" toggle XOR the upside-down mount (flipping the rig
    /// 180° mirrors the physical eye positions, so upside-down implies
    /// a swap on top of whatever the user chose).
    pub fn effective_swap_eyes(&self) -> bool {
        self.fisheye_swap_eyes ^ self.camera_upside_down
    }

    /// Build a `ColorStackPlan` from the slider state. Returns
    /// identity (`ColorStackPlan::default()`) when no color knob is
    /// active and the LUT path is empty — the pipeline `any_active()`
    /// check then short-circuits the whole stage chain.
    pub fn build_color_stack(&self) -> vr180_pipeline::gpu::ColorStackPlan {
        let mut plan = vr180_pipeline::gpu::ColorStackPlan::default();
        plan.cdl = vr180_pipeline::gpu::CdlParams {
            lift: self.lift,
            gamma: self.gamma.max(0.0001),
            gain: self.gain,
            shadow: self.shadow,
            highlight: self.highlight,
        };
        plan.color_grade = vr180_pipeline::gpu::ColorGradeParams {
            temperature: self.temperature,
            tint: self.tint,
            saturation: self.saturation,
        };
        // "Matching Eyes" inter-eye trim — applied oppositely per eye by
        // `ColorStackPlan::for_eye` at each per-eye color-stack apply site.
        plan.eye_match_ct = self.eye_match_ct;
        plan.eye_match_tint = self.eye_match_tint;
        if !self.lut_path.is_empty() {
            if let Some(lut) = load_lut_cached(&self.lut_path) {
                plan.lut = Some((lut, self.lut_intensity.clamp(0.0, 1.0)));
            }
        }
        plan.sharpen = vr180_pipeline::gpu::SharpenParams {
            amount: self.sharpen_amount.max(0.0),
            sigma: self.sharpen_radius.max(0.1),
            apply_lat_weight: true, // equirect output (both EAC + OSV)
        };
        plan
    }

    /// Path to the persisted settings JSON in the per-user config dir,
    /// resolved per-OS so persistence works on macOS, Windows, and Linux:
    ///   • macOS:   `~/Library/Application Support/VR180SilverBullet2.0/`
    ///   • Windows: `%APPDATA%\VR180SilverBullet2.0\`
    ///   • Linux:   `$XDG_CONFIG_HOME` or `~/.config` + `/VR180SilverBullet2.0/`
    pub fn config_path() -> Option<PathBuf> {
        let dir = if cfg!(target_os = "macos") {
            let home = std::env::var_os("HOME")?;
            PathBuf::from(home).join("Library/Application Support/VR180SilverBullet2.0")
        } else if cfg!(target_os = "windows") {
            // %APPDATA% = C:\Users\<user>\AppData\Roaming
            let appdata = std::env::var_os("APPDATA")?;
            PathBuf::from(appdata).join("VR180SilverBullet2.0")
        } else {
            // Linux / other: XDG_CONFIG_HOME, falling back to ~/.config.
            let base = std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
            base.join("VR180SilverBullet2.0")
        };
        Some(dir.join("settings.json"))
    }

    /// Load persisted settings, falling back to defaults if there's no file
    /// yet or anything fails to parse. `#[serde(default)]` on the struct
    /// means a config written by an OLDER build (missing newer fields) still
    /// loads — each missing field takes its `Default` value.
    pub fn load_persisted() -> Self {
        let Some(path) = Self::config_path() else { return Self::default() };
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Settings>(&text) {
                Ok(s) => {
                    tracing::info!("settings: restored from {}", path.display());
                    s
                }
                Err(e) => {
                    tracing::warn!("settings: parse {} failed ({e}); using defaults",
                        path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(), // no file yet → defaults
        }
    }

    /// Write the current settings to disk (best-effort; logs on failure).
    pub fn save_persisted(&self) {
        let Some(path) = Self::config_path() else { return };
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!("settings: mkdir {} failed: {e}", dir.display());
                return;
            }
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("settings: write {} failed: {e}", path.display());
            },
            Err(e) => tracing::warn!("settings: serialize failed: {e}"),
        }
    }

    /// Path for the PER-KIND settings map (sibling of `settings.json`).
    fn kind_map_config_path() -> Option<std::path::PathBuf> {
        Self::config_path().map(|p| p.with_file_name("settings_by_kind.json"))
    }

    /// Load the per-source-kind settings map (`"osv"` / `"eac"` / `"sbs"` /
    /// `"braw"` → Settings). Lets each camera type keep its OWN setup,
    /// independent of the others. Missing → migrate by seeding every kind
    /// from the legacy single `settings.json` so existing tuning carries.
    pub fn load_kind_map() -> std::collections::HashMap<String, Settings> {
        let mut map = 'load: {
            if let Some(path) = Self::kind_map_config_path() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    if let Ok(m) = serde_json::from_str::<std::collections::HashMap<String, Settings>>(&text) {
                        tracing::info!("settings: per-kind map restored from {}", path.display());
                        break 'load m;
                    }
                }
            }
            let legacy = Self::load_persisted();
            ["osv", "eac", "sbs", "braw"].into_iter()
                .map(|k| (k.to_string(), legacy.clone()))
                .collect()
        };
        // One-time: settings written before the ".360 Lens calibration"
        // feature carried DEAD Override fields in the "eac" bucket (stale
        // values from the legacy migration) — clear them so the new EAC
        // re-dewarp warp stays a no-op until the user actually enables it.
        if let Some(eac) = map.get_mut("eac") {
            if !eac.eac_lens_sanitized {
                if eac.fisheye_override_left || eac.fisheye_override_right {
                    tracing::info!(
                        "settings: clearing stale pre-feature .360 lens Override \
                         (was L={} R={})",
                        eac.fisheye_override_left, eac.fisheye_override_right);
                }
                eac.fisheye_override_left = false;
                eac.fisheye_override_right = false;
                eac.eac_lens_sanitized = true;
            }
        }
        map
    }

    /// Persist the per-kind settings map (best-effort).
    pub fn save_kind_map(map: &std::collections::HashMap<String, Settings>) {
        let Some(path) = Self::kind_map_config_path() else { return };
        if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
        if let Ok(json) = serde_json::to_string_pretty(map) {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("settings: write per-kind {} failed: {e}", path.display());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FisheyeOutputMode {
    /// Half-equirect VR180 output (default).
    HalfEquirect,
    /// Normalized circular-fisheye SBS output: an EQUIDISTANT fisheye
    /// at the source lens's captured FOV (195° for the DJI/OSV lens,
    /// 185° for the GoPro Max `.360`). Full reprojection — stab, panomap,
    /// stereo offset, per-row RS all apply; the source lens's own
    /// distortion is removed.
    Fisheye,
}

impl FisheyeOutputMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HalfEquirect => "Half-equirect (VR180)",
            Self::Fisheye      => "Fisheye SBS (equidist.)",
        }
    }
}

/// Audio track to write on export, for sources that carry both a stereo
/// AAC track and a 4-channel ambisonic track (GoPro MAX `.360`). Mirrors
/// the Python app's audio-format choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// Stereo AAC passthrough — max compatibility (every player). Default.
    Stereo,
    /// 4-channel 1st-order ambisonic (AmbiX W,Y,Z,X) passthrough + SA3D
    /// metadata, so VR-aware players (YouTube VR, Quest Browser) head-track
    /// the audio. (Vision Pro ignores SA3D → use APAC for it.)
    Ambisonic,
    /// Re-encode the 4ch ambisonic to Apple Positional Audio Codec for
    /// true head-tracked spatial audio on Vision Pro. macOS-only (needs
    /// the `apac_encode` helper).
    Apac,
}
impl AudioFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stereo    => "Stereo AAC",
            Self::Ambisonic => "Ambisonic 4ch (+SA3D)",
            Self::Apac      => "APAC (Vision Pro spatial)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoriSource { Direct, Vqf, Auto }
impl CoriSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Vqf    => "vqf",
            Self::Auto   => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsMode { Firmware, NoFirmware }
impl RsMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Firmware   => "firmware",
            Self::NoFirmware => "no-firmware",
        }
    }
}

pub struct DecoderConfig {
    pub path: PathBuf,
    /// Ordered GoPro segment chain (GS01…, GS02…, …). Single-element
    /// `[path]` for a lone clip. Played + gyro-aggregated as one.
    pub segments: Vec<PathBuf>,
    pub settings: Settings,
    pub eye_w: u32,
}

/// Total decoded-frame count across a segment chain: `Σ round(durᵢ·fps)`.
/// Single element → the lone clip's frame count.
pub(crate) fn segments_total_frames(segments: &[PathBuf]) -> usize {
    segments.iter()
        .filter_map(|s| vr180_pipeline::decode::probe_video(s).ok())
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .sum()
}

/// Commands the UI sends to the decoder.
pub enum DecoderCommand {
    Stop,
    /// Jump playback to `target_s` seconds. The decoder seeks the
    /// underlying ffmpeg context to the nearest keyframe ≤ target,
    /// flushes the decoder, then resumes decoding (and frame-drops
    /// any leading frames before `target_s` so the next output is
    /// at-or-after the requested time).
    Seek(f64),
}

/// Shared control surface between the UI and a running decoder thread.
///
/// - `paused`: when true, the decoder loop parks instead of producing
///   frames. Flipping it back to false resumes playback from wherever
///   the decoder paused — no seek, no restart from frame 0.
/// - `settings`: the live stab/RS/source knobs. The UI mutates this
///   in place every time the user nudges a slider; the decoder reads
///   it once per iteration via `settings_generation`.
/// - `settings_generation`: bumped on every settings change. The
///   decoder caches the generation it last computed for; if the
///   counter has moved, it rebuilds the per-eye bundle on the spot
///   (~ few hundred ms for a full-length clip) and keeps going.
#[derive(Default)]
pub struct DecoderControl {
    pub paused: AtomicBool,
    pub settings: RwLock<Settings>,
    pub settings_generation: AtomicU64,
    /// Per-eye in-file calibration the decoder resolved from the source
    /// (OSV protobuf). `None` until the decoder has parsed it (or for
    /// sources without an in-file calibration). The UI reads this to
    /// display the actual calibration and to seed the per-eye Override
    /// fields, instead of guessing image-center.
    pub detected_calib: parking_lot::Mutex<Option<DetectedLensCalib>>,
    /// Set by the UI while paused + zoomed on an EAC (.360) clip: asks
    /// the decoder to ALSO emit a native-resolution still of the current
    /// frame on `detail_tx` for the zoom magnifier. The live preview
    /// stays at the capped working size; this is the full-detail copy.
    /// Fisheye sources use `DetailCache` instead and ignore this.
    pub want_detail: AtomicBool,
    /// Sender for the native-res zoom stills (see `want_detail`). Installed
    /// by the UI at decoder spawn; `None` for paths that never produce them.
    pub detail_tx: parking_lot::Mutex<Option<Sender<DecodedFrame>>>,
    /// Set true by the spawning thread once `start_decoder` returns (clean
    /// EOS, error, or Stop). The UI watches this so that a decoder which dies
    /// *before* delivering its first frame — e.g. a transient failure on a
    /// cold network read — doesn't wedge playback: `decoder_starting` would
    /// otherwise stay set forever and silently gate every Play/seek respawn.
    pub finished: AtomicBool,
    /// Progressive IMU state for DJI OSV. The decoder loads the lens calib FAST
    /// (first metadata sample) so the preview dewarps immediately, then streams
    /// the heavy per-frame quaternions (needed for stabilization) in the
    /// background. `false` while those quats are still loading; flips `true`
    /// once they're in (or immediately for non-OSV / already-cached clips). The
    /// UI grays the Stabilization panel + shows a spinner while this is `false`.
    pub imu_ready: AtomicBool,
    /// Progress of the background per-frame IMU (quaternion) read, for the UI's
    /// "Loading stabilization data… NN%". Chunk `done`/`total`; 0% until the
    /// sample table is parsed. `Arc` so the background reader thread updates it
    /// while the UI polls it. See [`load_dji_imu_progressive`].
    pub imu_progress: std::sync::Arc<vr180_pipeline::decode::ReadProgress>,
    /// True while the background per-frame quat read is in flight (drives the
    /// loading spinner + a double-spawn guard). The read runs ONLY when
    /// stabilization is on; canceling stab aborts it via `imu_progress.cancel`.
    pub stab_loading: AtomicBool,
}

/// One eye's in-file calibration, in the same units the UI's per-eye
/// Override controls use: FOV in degrees (in this app's equidistant
/// `fov ↔ fx` convention, so feeding it back reproduces the same fx),
/// principal point NORMALIZED to [0,1], and KB k1–k4.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EyeCalibSeed {
    pub fov_deg: f32,
    pub cx_norm: f32,
    pub cy_norm: f32,
    pub k: [f32; 4],
    /// 5th KB radial coefficient from the file (OSV field 15); 0 if absent.
    pub k5: f32,
    /// Brown-Conrady tangential coeffs `[p1, p2]` from the file (field 20);
    /// `[0,0]` if absent.
    pub p: [f32; 2],
}

/// The pair of in-file per-eye calibrations (after the L=lens_b,
/// R=lens_a eye mapping).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DetectedLensCalib {
    pub left: EyeCalibSeed,
    pub right: EyeCalibSeed,
}

/// Resolve the DJI OSV per-lens factory IMU/calibration for a fisheye decoder
/// AND publish it everywhere it's needed, so every decode path stays in sync.
/// In one call it:
///   1. reuses the path-keyed cache, else extracts+parses (single file, or
///      `parse_multi` concatenating each segment's `djmd` protobuf so the IMU
///      indexes by ABSOLUTE frame across a merged recording);
///   2. PUBLISHES to the shared cache so the main-thread zoom-still
///      (`DetailCache`) resolves the SAME factory calib instead of the centered
///      preset — the publish each new decode path used to forget (zoom-shift
///      bug);
///   3. seeds `control.detected_calib` so the Override UI shows/seeds the
///      in-file per-lens values.
/// Returns `None` for non-OSV sources or on parse failure (caller falls back to
/// the preset calib). `src_w`/`src_h` only backstop the seed's normalized dims
/// when the file omits per-lens width/height.
///
/// EVERY fisheye decode path must call this — that is the point: centralizing
/// the publish makes it impossible for a future path to drop it.
fn load_or_publish_dji_imu(
    cfg: &DecoderConfig,
    kind: vr180_pipeline::SourceKind,
    control: &DecoderControl,
    src_w: u32,
    src_h: u32,
) -> Option<vr180_fisheye::DjiOsvImu> {
    if !matches!(kind, vr180_pipeline::SourceKind::DjiOsv) {
        return None;
    }
    // 1. Get the IMU.
    let parsed: vr180_pipeline::Result<vr180_fisheye::DjiOsvImu> = if cfg.segments.len() > 1 {
        let mut blobs = Vec::with_capacity(cfg.segments.len());
        for p in &cfg.segments {
            match vr180_pipeline::decode::extract_dji_meta_stream(p) {
                Ok(b) => blobs.push(b),
                Err(e) => {
                    tracing::warn!("dji metadata failed: {e} — using preset calib for both eyes");
                    return None;
                }
            }
        }
        vr180_fisheye::DjiOsvImu::parse_multi(&blobs)
            .map_err(|e| vr180_pipeline::Error::Ffmpeg(format!("dji parse_multi: {e}")))
    } else if let Some(arc) = cached_dji_imu(&cfg.path).filter(|a| !a.frame_quats.is_empty()) {
        tracing::info!("decoder (fisheye/osv): reusing cached DJI metadata for {}", cfg.path.display());
        Ok((*arc).clone())
    } else {
        vr180_pipeline::decode::extract_dji_meta_stream(&cfg.path)
            .and_then(|blob| vr180_fisheye::DjiOsvImu::parse(&blob)
                .map_err(|e| vr180_pipeline::Error::Ffmpeg(format!("dji parse: {e}"))))
    };
    let imu = match parsed {
        Ok(imu) => imu,
        Err(e) => {
            tracing::warn!("dji metadata failed: {e} — using preset calib for both eyes");
            return None;
        }
    };
    tracing::info!(
        "decoder (fisheye/osv): protobuf parsed — lens_a.fx={:?}, lens_b.fx={:?}, {} frame_quats ({} segs)",
        imu.lens_a.fx, imu.lens_b.fx, imu.frame_quats.len(), cfg.segments.len()
    );
    // 2. Publish to the shared cache (the main-thread zoom-still reads it).
    cache_dji_imu(&cfg.path, std::sync::Arc::new(imu.clone()));
    // 3. Seed the Override UI's per-eye fields from the in-file calib.
    seed_detected_calib(control, &imu, src_w, src_h);
    Some(imu)
}

/// Seed `control.detected_calib` (the Override UI's per-eye defaults) from a
/// parsed OSV IMU, following the user swap so each output eye shows the lens
/// actually on it. Works on a calib-only (no-quats) partial too — only
/// `lens_a`/`lens_b` are read.
fn seed_detected_calib(
    control: &DecoderControl, imu: &vr180_fisheye::DjiOsvImu, src_w: u32, src_h: u32,
) {
    let seed = |lens: &vr180_fisheye::DjiLensCalib| -> EyeCalibSeed {
        let w = lens.width.unwrap_or(src_w as f32).max(1.0);
        let h = lens.height.unwrap_or(src_h as f32).max(1.0);
        // fx = w/(2·half) ⟹ fov = w/fx radians, so feeding it back reproduces fx.
        let fov_deg = lens.fx.map(|fx| (w / fx.max(1.0)).to_degrees()).unwrap_or(180.0);
        EyeCalibSeed {
            fov_deg,
            cx_norm: lens.cx.map(|v| v / w).unwrap_or(0.5),
            cy_norm: lens.cy.map(|v| v / h).unwrap_or(0.5),
            k: [lens.k1.unwrap_or(0.0), lens.k2.unwrap_or(0.0),
                lens.k3.unwrap_or(0.0), lens.k4.unwrap_or(0.0)],
            k5: lens.k5.unwrap_or(0.0),
            p: [lens.p1.unwrap_or(0.0), lens.p2.unwrap_or(0.0)],
        }
    };
    let swapped = control.settings.read().effective_swap_eyes();
    let (sl, sr) = if swapped { (&imu.lens_a, &imu.lens_b) } else { (&imu.lens_b, &imu.lens_a) };
    *control.detected_calib.lock() = Some(DetectedLensCalib { left: seed(sl), right: seed(sr) });
}

/// Progressive DJI OSV IMU load: get the lens calibration FAST (first metadata
/// sample → the preview dewarps in ~1 s) and stream the heavy per-frame
/// quaternions (needed for stabilization) in the BACKGROUND, so a huge clip's
/// minutes-long metadata read doesn't block the first frame. Returns a
/// calib-only partial (empty `frame_quats`) immediately; the decode loop calls
/// [`try_upgrade_dji_imu`] each iteration and swaps in the full IMU once the
/// background thread caches it. Sets `control.imu_ready` accordingly (`false`
/// while quats load; `true` for non-OSV / already-cached / fallback). The UI
/// grays Stabilization while it's `false`.
fn load_dji_imu_progressive(
    cfg: &DecoderConfig,
    kind: vr180_pipeline::SourceKind,
    control: &std::sync::Arc<DecoderControl>,
    src_w: u32,
    src_h: u32,
) -> Option<vr180_fisheye::DjiOsvImu> {
    use std::sync::atomic::Ordering;
    if !matches!(kind, vr180_pipeline::SourceKind::DjiOsv) {
        control.imu_ready.store(true, Ordering::SeqCst);
        return None;
    }
    // Already have the full IMU cached → use it, stabilization is ready now.
    if let Some(arc) = cached_dji_imu(&cfg.path) {
        if !arc.frame_quats.is_empty() {
            let imu = (*arc).clone();
            seed_detected_calib(control, &imu, src_w, src_h);
            control.imu_ready.store(true, Ordering::SeqCst);
            return Some(imu);
        }
    }
    // Merged recording (parse_multi) has no single-file fast-calib path — load
    // it fully (blocking). Same for a fast-calib miss below.
    // Fast calib from the FIRST segment's head (single file → that IS cfg.path).
    // Works for merged recordings too: the per-lens calib is identical across
    // segments, so we can dewarp immediately and stream ALL segments' quats in
    // the background — same progressive/pausable path as a single file.
    let calib_src = cfg.segments.first().cloned().unwrap_or_else(|| cfg.path.clone());
    let fast = vr180_pipeline::decode::extract_dji_calib_blob(&calib_src).ok()
        .and_then(|b| vr180_fisheye::DjiOsvImu::parse(&b).ok())
        .filter(|imu| imu.lens_a.fx.is_some() || imu.lens_b.fx.is_some());
    match fast {
        Some(partial) => {
            tracing::info!(
                "decoder (fisheye/osv): fast calib loaded (lens_a.fx={:?}) — preview now; \
                 full IMU streaming in background", partial.lens_a.fx);
            seed_detected_calib(control, &partial, src_w, src_h);
            // Publish a CALIB-ONLY copy (quats stripped) so the full-res zoom-still
            // resolves the SAME lens calibration as the live preview *during*
            // loading. Without it the still's cache lookup misses and falls back to
            // the centered preset — the image shifts when you zoom until the full
            // IMU lands. Quats are cleared so (a) the still doesn't stabilize while
            // the live view (gated on imu_ready) doesn't, and (b) the "cache is
            // full" check (`!frame_quats.is_empty()`) still skips it. The
            // background thread overwrites this with the full IMU below.
            {
                let mut calib_only = partial.clone();
                calib_only.frame_quats.clear();
                calib_only.gravity.clear();
                calib_only.high_rate_quats.clear();
                cache_dji_imu(&cfg.path, std::sync::Arc::new(calib_only));
            }
            control.imu_ready.store(false, Ordering::SeqCst);
            // Only read the heavy per-frame quats if stabilization is ON. If it's
            // off, skip the (minutes-long on SMB) read entirely — it starts when the
            // user enables stabilization (the decode loop's live toggle calls
            // spawn_dji_quat_load). The calib above is already cached for dewarp.
            if control.settings.read().stabilize {
                spawn_dji_quat_load(cfg, control);
            }
            Some(partial)
        }
        None => {
            // No fast calib (merged clip, or sample-table miss) → full blocking.
            let imu = load_or_publish_dji_imu(cfg, kind, control, src_w, src_h);
            control.imu_ready.store(true, Ordering::SeqCst);
            imu
        }
    }
}

/// Spawn the background per-frame IMU (quaternion) read for a DJI OSV clip and
/// publish the full IMU to the cache when done. Only called when stabilization
/// is ON (at load, or when the user toggles it on). Resets progress + the cancel
/// flag, marks `stab_loading`, and on completion bumps `settings_generation` so
/// the decode loop (even paused) wakes to swap it in. Canceling stabilization
/// sets `imu_progress.cancel`, which aborts the in-flight read.
fn spawn_dji_quat_load(cfg: &DecoderConfig, control: &std::sync::Arc<DecoderControl>) {
    use std::sync::atomic::Ordering;
    control.imu_progress.done.store(0, Ordering::SeqCst);
    control.imu_progress.total.store(0, Ordering::SeqCst);
    control.imu_progress.pause.store(false, Ordering::SeqCst);
    control.imu_progress.abort.store(false, Ordering::SeqCst);
    control.stab_loading.store(true, Ordering::SeqCst);
    let path = cfg.path.clone();
    // Segments to read: a merged recording reads + concatenates each segment's
    // djmd (parse_multi indexes by ABSOLUTE frame); a single file is one segment.
    // The shared `progress` accumulates `total`/`done` across all segments, so the
    // % spans the whole recording, and pause/abort are honored per-segment read.
    let segments: Vec<std::path::PathBuf> = if cfg.segments.len() > 1 {
        cfg.segments.clone()
    } else {
        vec![path.clone()]
    };
    let progress = control.imu_progress.clone();
    let control_bg = std::sync::Arc::clone(control);
    std::thread::spawn(move || {
        let parsed = (|| -> vr180_pipeline::Result<vr180_fisheye::DjiOsvImu> {
            if segments.len() == 1 {
                let b = vr180_pipeline::decode::extract_dji_meta_stream_with_progress(&segments[0], &progress)?;
                vr180_fisheye::DjiOsvImu::parse(&b)
                    .map_err(|e| vr180_pipeline::Error::Ffmpeg(format!("dji parse: {e}")))
            } else {
                let mut blobs = Vec::with_capacity(segments.len());
                for seg in &segments {
                    blobs.push(vr180_pipeline::decode::extract_dji_meta_stream_with_progress(seg, &progress)?);
                }
                vr180_fisheye::DjiOsvImu::parse_multi(&blobs)
                    .map_err(|e| vr180_pipeline::Error::Ffmpeg(format!("dji parse_multi: {e}")))
            }
        })();
        match parsed {
            Ok(full) => {
                tracing::info!("decoder (fisheye/osv): full IMU ready ({} frame_quats, {} seg) for {}",
                    full.frame_quats.len(), segments.len(), path.display());
                cache_dji_imu(&path, std::sync::Arc::new(full));
                // Wake the decode loop to swap it in (even when paused); the loop
                // clears `stab_loading` once it applies the IMU.
                control_bg.settings_generation.fetch_add(1, Ordering::SeqCst);
            }
            Err(e) => {
                // Failed OR aborted — stop the spinner; quats just aren't loaded.
                tracing::warn!("background full IMU load ended: {e}");
                control_bg.stab_loading.store(false, Ordering::SeqCst);
                control_bg.settings_generation.fetch_add(1, Ordering::SeqCst);
            }
        }
    });
}

/// Decode-loop helper for [`load_dji_imu_progressive`]: while we're on the
/// calib-fast partial and the background thread has since cached the full IMU
/// (MORE frame_quats than we hold now), return it so the loop can swap it in +
/// recompute stab. `None` otherwise (non-OSV, or no newer IMU yet).
fn try_upgrade_dji_imu(
    cfg: &DecoderConfig, current: Option<&vr180_fisheye::DjiOsvImu>,
) -> Option<vr180_fisheye::DjiOsvImu> {
    // Compare COUNTS, not emptiness: the first djmd sample can already carry the
    // first frame's quat, so the calib-fast partial isn't necessarily empty — an
    // `is_empty()` test would wrongly conclude "already full" and never upgrade.
    let cur_n = current?.frame_quats.len(); // None ⇒ non-OSV ⇒ never upgrade
    let arc = cached_dji_imu(&cfg.path)?;
    if arc.frame_quats.len() > cur_n {
        Some((*arc).clone())
    } else {
        None
    }
}

/// One frame ready for display. `texture` is a square BGRA-bytes
/// SBS texture (left eye on the left half, right eye on the right
/// half) sitting on the shared wgpu device — egui can register it
/// without copying.
pub struct DecodedFrame {
    pub texture: Arc<wgpu::Texture>,
    pub width: u32,
    pub height: u32,
    pub frame_idx: u32,
    pub timestamp_s: f64,
}

/// Entry point. Runs to completion (or until a Stop command arrives).
///
/// macOS: uses the zero-copy IOSurface path (`ZeroCopyStreamPairIter`
/// + `nv12_to_eac_cross` shader), so decode → EAC assembly →
/// projection → SBS compose stays on the GPU end-to-end. No host
/// memory transfer per frame. This is what makes the preview hit
/// 30 fps at GoPro Max source resolution.
///
/// Non-macOS: falls back to the CPU-assemble path (decode +
/// `av_hwframe_transfer_data` + swscale + CPU `assemble_lens_*` +
/// GPU project). Real-time on smaller clips; for full-resolution
/// 5952×1920 it'll lean on the frame-skip path. Phase 0.6.8 lands
/// the Windows CUDA↔Vulkan equivalent.
pub fn start_decoder(
    pipeline: Arc<Device>,
    cfg: DecoderConfig,
    control: Arc<DecoderControl>,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    tracing::info!("decoder: starting for {}", cfg.path.display());

    // Detect source kind. GoPro EAC stays on the existing
    // zero-copy / CPU-assemble path; fisheye sources (OSV, SBS,
    // BRAW) take a new path that uses the KB projection shader.
    let kind = vr180_pipeline::source_kind::detect(&cfg.path)
        .map_err(|e| { tracing::error!("decoder: source detect failed: {e}"); e })?;
    tracing::info!("decoder: source kind = {:?} ({})", kind, kind.display());

    let probe = vr180_pipeline::decode::probe_video(&cfg.path)
        .map_err(|e| { tracing::error!("decoder: probe failed: {e}"); e })?;
    let fps = probe.fps.max(1e-3);
    let dt = 1.0 / fps as f64;
    let total_frames = if cfg.segments.len() > 1 {
        segments_total_frames(&cfg.segments)
    } else {
        (probe.duration_sec as f64 * fps as f64).round() as usize
    };
    tracing::info!("decoder: probed {} × {} @ {:.2} fps, {} frames total",
        probe.width, probe.height, fps, total_frames);

    let eye_w = cfg.eye_w.clamp(256, 2048);
    let eye_h = eye_w;

    // Fisheye sources skip the GoPro CORI/IORI per-eye bundle (it's a
    // GoPro-specific format). They get identity rotation for the
    // first cut; BRAW VQF integration comes with Task #8.
    if kind.is_fisheye() {
        tracing::info!("decoder: routing to fisheye path");
        return run_fisheye(pipeline, &cfg, control, kind, fps, dt, eye_w, eye_h, frame_tx, cmd_rx);
    }

    let snapshot = control.settings.read().clone();
    let per_eye = build_per_eye_frames_multi(&cfg.segments, &snapshot, fps, total_frames)
        .map_err(|e| { tracing::error!("decoder: per-eye frames build failed: {e}"); e })?;
    let stab_key = stab_settings_key(&snapshot);
    let cached_gen = control.settings_generation.load(Ordering::SeqCst);
    tracing::info!("decoder: per-eye bundles ready ({} entries) gen={}, eye={}×{}",
        per_eye.len(), cached_gen, eye_w, eye_h);

    // GEOC factory lens calibration (file tail, ~1 MiB read) — powers the
    // ".360 Lens calibration" Override: published as the detected calib
    // (panel display + Override seeding, like OSV) and used per frame to
    // resolve the per-eye re-dewarp warp. Missing GEOC → panel shows the
    // preset fallback and the Override warp stays disabled.
    let geoc = vr180_core::geoc::parse_geoc(
        cfg.segments.first().unwrap_or(&cfg.path)).ok().flatten();
    match &geoc {
        Some(g) => seed_eac_detected_calib(&control, g),
        None => tracing::warn!("decoder (eac): no GEOC found — lens Override unavailable"),
    }

    #[cfg(target_os = "macos")]
    return run_zero_copy(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, stab_key, cached_gen, geoc, frame_tx, cmd_rx);

    // Windows: try the GPU-resident zero-copy EAC preview (NVDEC→D3D11→Vulkan,
    // GPU cross assembly), matching the macOS zero-copy path. Handles merged
    // multi-segment recordings via SegmentedD3d11SharedStreamPairIter (chained
    // segments, globalized pts). Falls back to the CPU-assemble path only when
    // the backend isn't Vulkan or d3d11va can't attach.
    #[cfg(target_os = "windows")]
    {
        let zc = vr180_pipeline::interop_windows::VulkanImportCtx::from_wgpu(
            &pipeline.adapter, &pipeline.device,
        ).and_then(|ctx| {
            match vr180_pipeline::fisheye_decode::SegmentedD3d11SharedStreamPairIter::new(&cfg.segments) {
                Ok(iter) => Some((ctx, iter)),
                Err(e) => { tracing::warn!("decoder: EAC zero-copy iter unavailable ({e})"); None }
            }
        });
        match zc {
            Some((ctx, iter)) => {
                tracing::info!(
                    "decoder: EAC zero-copy (d3d11va→Vulkan, GPU cross) preview ENGAGED ({} segment(s))",
                    cfg.segments.len()
                );
                return run_eac_zerocopy(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, stab_key, cached_gen, geoc, ctx, iter, frame_tx, cmd_rx);
            }
            None => {
                tracing::info!("decoder: EAC CPU-assemble preview (zero-copy unavailable)");
                return run_cpu_assemble(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, stab_key, cached_gen, geoc, frame_tx, cmd_rx);
            }
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    return run_cpu_assemble(pipeline, &cfg, control, fps, dt, eye_w, eye_h, per_eye, stab_key, cached_gen, geoc, frame_tx, cmd_rx);
}

/// Open ONE fisheye segment into a `FisheyePairIter`, at the working
/// resolution the live preview uses. Shared by the single-file worker and
/// the multi-segment [`SegmentedFisheyeIter`] opener so both decode a
/// merged OSV/SBS/BRAW recording identically. Returns `vr180_pipeline`'s
/// `Result` so it can be the segmented iterator's opener directly.
fn open_fisheye_segment(
    path: &std::path::Path,
    kind: vr180_pipeline::SourceKind,
    swap_eyes: bool,
    fps: f32,
) -> vr180_pipeline::Result<Box<dyn vr180_pipeline::fisheye_decode::FisheyePairIter>> {
    use vr180_pipeline::fisheye_decode::{
        SbsFisheyeIter, DualStreamFisheyeIter, BrawFisheyeIter, max_decode_side_for_fps,
    };
    use vr180_pipeline::decode::HwDecode;
    use vr180_pipeline::Error;
    Ok(match kind {
        vr180_pipeline::SourceKind::DjiOsv => {
            let cap = max_decode_side_for_fps(fps);
            // XOR with DJI's "swap by default" (stream 0 = right eye).
            Box::new(DualStreamFisheyeIter::new_with_options(
                path, HwDecode::Auto, 0, !swap_eyes, cap, 8)?)
        }
        vr180_pipeline::SourceKind::SbsFisheye =>
            Box::new(SbsFisheyeIter::new(path, HwDecode::Auto, 0)?),
        vr180_pipeline::SourceKind::BlackmagicRaw => {
            let info = vr180_braw::BrawInfo::probe(path)
                .map_err(|e| Error::Ffmpeg(format!("braw probe: {e}")))?;
            let opts = vr180_braw::decoder::DecodeOptions::default();
            Box::new(BrawFisheyeIter::new(path, &info, &opts, 0)
                .map_err(|e| Error::Ffmpeg(format!("braw start: {e}")))?)
        }
        _ => return Err(Error::Ffmpeg(format!("non-fisheye source: {kind:?}"))),
    })
}

/// Fisheye source decoder loop (DJI OSV, SBS fisheye, Blackmagic BRAW).
///
/// Same wall-clock pacing / pause-resume / seek / trim semantics as the
/// GoPro paths above. Multi-segment (`cfg.segments.len() > 1`) chains the
/// files via `SegmentedFisheyeIter` with aggregated DJI IMU — one
/// continuous timeline for decode, stabilization, and seek.
fn run_fisheye(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    kind: vr180_pipeline::SourceKind,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    use vr180_pipeline::fisheye_decode::FisheyePairIter;
    use vr180_fisheye::presets;

    // ── Windows zero-copy fast path (DJI OSV only) ───────────────────
    // Probe synchronously so the decision is made BEFORE the live
    // channels are consumed: if wgpu is on Vulkan with P010 support AND
    // d3d11va attaches to both OSV streams, run the GPU-resident
    // decode→import→project path (no CPU download / swscale). Any miss
    // falls through to the portable CPU path below with frame_tx/cmd_rx
    // untouched.
    #[cfg(target_os = "windows")]
    {
        let has_p010 = pipeline.device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_P010);
        let on_vulkan = vr180_pipeline::interop_windows::is_vulkan_backend(&pipeline.device);
        if matches!(kind, vr180_pipeline::SourceKind::DjiOsv) && on_vulkan && has_p010 {
            // XOR with DJI's "swap by default" (matches the CPU worker).
            let swap = !control.settings.read().effective_swap_eyes();
            let ctx = vr180_pipeline::interop_windows::VulkanImportCtx::from_wgpu(
                &pipeline.adapter, &pipeline.device,
            );
            // Working res for the D3D11-side P010→RGBA16 downscale: ~1280
            // (matches the proven CPU working res), never below the preview eye.
            // The iter clamps it to native.
            let work = eye_w.max(eye_h).max(1280);
            let iter = vr180_pipeline::fisheye_decode::D3d11SharedDualStreamIter::new(
                &cfg.path, swap, work, work,
            );
            match (ctx, iter) {
                (Some(ctx), Ok(iter)) => {
                    tracing::info!(
                        "decoder (fisheye): ZERO-COPY d3d11va→Vulkan path ENGAGED \
                         (full-res P010, no CPU download/swscale)"
                    );
                    return run_fisheye_zerocopy(
                        pipeline, cfg, control, kind, fps, dt, eye_w, eye_h,
                        ctx, iter, frame_tx, cmd_rx,
                    );
                }
                (c, i) => {
                    tracing::warn!(
                        "decoder (fisheye): zero-copy unavailable (vulkan_ctx={}, \
                         d3d11va_iter_ok={}) — falling back to CPU download path",
                        c.is_some(), i.is_ok()
                    );
                }
            }
        } else {
            tracing::info!(
                "decoder (fisheye): zero-copy preconditions not met \
                 (dji_osv={}, vulkan={}, p010={}) — CPU path",
                matches!(kind, vr180_pipeline::SourceKind::DjiOsv), on_vulkan, has_p010
            );
        }
    }

    // ── macOS zero-copy fast path (DJI OSV, single-segment) ──────────
    // VideoToolbox-decoded P010 IOSurfaces, wrapped zero-copy and resolved →
    // projected on the GPU — eliminates the per-eye HW→host download + libswscale
    // the CPU `DualStreamFisheyeIter` pays. Single-segment only (merged OSV stays
    // on the CPU path); any failure falls through to the CPU path below untouched.
    #[cfg(target_os = "macos")]
    {
        if matches!(kind, vr180_pipeline::SourceKind::DjiOsv) && cfg.segments.len() <= 1 {
            // XOR with DJI's "swap by default" (matches open_fisheye_segment).
            let swap = !control.settings.read().effective_swap_eyes();
            match vr180_pipeline::fisheye_decode::VtSharedDualStreamIter::new(&cfg.path, swap) {
                Ok(iter) => {
                    tracing::info!(
                        "decoder (fisheye): macOS ZERO-COPY VideoToolbox path ENGAGED \
                         (P010 IOSurface → resolve → project, no host download/swscale)"
                    );
                    return run_fisheye_vt_zerocopy(
                        pipeline, cfg, control, kind, fps, dt, eye_w, eye_h,
                        iter, frame_tx, cmd_rx,
                    );
                }
                Err(e) => {
                    tracing::info!(
                        "decoder (fisheye): macOS zero-copy unavailable ({e}) — CPU path");
                }
            }
        }
    }

    // === Pipelined decode ===
    //
    // The iter is created and run inside a worker thread so decode and
    // render happen in parallel. On a 3840² dual-stream HEVC source,
    // per-frame VT decode is ~20 ms — same order as the GPU
    // project+compose budget. Serial execution sums them and blows
    // the per-frame budget at 50 fps. Pipelining drops the wall-clock
    // floor to max(decode, render).
    //
    // Channels:
    //   pair_tx/rx: bounded(2), pre-decoded pairs tagged with a
    //               generation counter so seeks can discard stale
    //               in-flight frames without restarting the thread.
    //   iter_cmd:   main → worker (seek)
    //   dims_tx/rx: one-shot, worker → main with eye_dims after init.
    enum IterCmd { Seek(f64) }
    // Buffer 8 decoded pairs ahead. At 50 fps source that's ~160 ms
    // of jitter absorption — enough to ride out GC/wgpu cleanup pauses
    // and the occasional slow-decode frame (saw 21 ms outliers vs the
    // 20 ms steady state in the 50 fps trace). Latency cost: a fresh
    // seek may take ~160 ms before the first new frame surfaces, which
    // is invisible for normal playback.
    let (pair_tx, pair_rx) =
        crossbeam_channel::bounded::<(u64, vr180_pipeline::fisheye_decode::FisheyePair)>(8);
    let (iter_cmd_tx, iter_cmd_rx) = crossbeam_channel::bounded::<IterCmd>(8);
    let (dims_tx, dims_rx) = crossbeam_channel::bounded::<(u32, u32)>(1);

    let path_for_worker = cfg.path.clone();
    let segments_for_worker = cfg.segments.clone();
    let kind_for_worker = kind;
    let initial_swap_eyes = control.settings.read().effective_swap_eyes();
    let initial_trim_in = control.settings.read().trim_in_s;

    let _decode_handle = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut iter: Box<dyn FisheyePairIter> = if segments_for_worker.len() > 1 {
            // Merged recording (sequential OSV / SBS files): chain the
            // segments into one continuous timeline. Per-segment durations
            // (probed) rebase timestamps and map seeks across boundaries.
            let durations: Vec<f64> = segments_for_worker.iter()
                .map(|p| vr180_pipeline::decode::probe_video(p)
                    .map(|pr| pr.duration_sec).unwrap_or(0.0))
                .collect();
            tracing::info!(
                "decoder (fisheye): {} segments, {:.1}s total — SegmentedFisheyeIter",
                segments_for_worker.len(), durations.iter().sum::<f64>());
            let opener = Box::new(move |p: &std::path::Path|
                open_fisheye_segment(p, kind_for_worker, initial_swap_eyes, fps));
            Box::new(vr180_pipeline::fisheye_decode::SegmentedFisheyeIter::new(
                &segments_for_worker, &durations, opener)?)
        } else {
            open_fisheye_segment(&path_for_worker, kind_for_worker, initial_swap_eyes, fps)?
        };
        if let Some(t_in) = initial_trim_in {
            if t_in > 0.001 { let _ = iter.seek(t_in); }
        }
        let (sw, sh) = iter.eye_dims();
        if dims_tx.send((sw, sh)).is_err() { return Ok(()); }

        let mut gen: u64 = 0;
        loop {
            // Drain pending commands before each decode.
            while let Ok(cmd) = iter_cmd_rx.try_recv() {
                match cmd {
                    IterCmd::Seek(t) => {
                        let _ = iter.seek(t);
                        gen = gen.wrapping_add(1);
                        // `seek` lands on the keyframe AT OR BEFORE t and
                        // does not decode forward, so the next frame would be
                        // that keyframe — up to a whole GOP (~1 s) early.
                        // Decode forward to the frame AT t and yield THAT
                        // first, so the preview lands on the correct frame
                        // (and matches the native detail still, which does
                        // the same). Pre-target frames advance the decoder
                        // but aren't shown. Bounded so a bad pts can't spin.
                        let dt = 1.0 / (fps.max(1e-3) as f64);
                        for _ in 0..1200 {
                            match iter.next_pair() {
                                Ok(Some(p)) => {
                                    if p.pts_s >= t - dt * 0.5 {
                                        if pair_tx.send((gen, p)).is_err() {
                                            return Ok(());
                                        }
                                        break;
                                    }
                                }
                                _ => break, // EOS or error
                            }
                        }
                    }
                }
            }
            match iter.next_pair() {
                Ok(Some(pair)) => {
                    if pair_tx.send((gen, pair)).is_err() { break; }
                }
                Ok(None) => break,        // EOS
                Err(e) => { tracing::warn!("decode-worker: {e}"); break; }
            }
        }
        Ok(())
    });

    let (src_w, src_h) = match dims_rx.recv() {
        Ok(d) => d,
        Err(_) => anyhow::bail!("decode worker failed to start"),
    };
    tracing::info!("decoder (fisheye): eye dims = {}×{} (pipelined)", src_w, src_h);
    // Main thread tracks the latest seek generation. Pairs tagged
    // with an older generation are stale (decoded before the latest
    // seek took effect) and silently discarded.
    let mut expected_gen: u64 = 0;

    // ── Wire swap_eyes from settings if this is a dual-stream source.
    {
        let s = control.settings.read();
        let _ = &s;
        // (Dual-stream iter exposes swap_eyes as a public field, but we
        //  built `decoder` as a Box<dyn> trait object — can't downcast
        //  cleanly. The Python OSV path swaps based on settings; we
        //  defer wiring until we expose swap_eyes on the trait.)
    }

    // ── DJI OSV: load the per-lens factory calib + IMU, publish to the shared
    //    cache (so the zoom-still matches) and seed the Override UI — one call,
    //    so no decode path can forget a step.
    let mut dji_osv_imu = load_dji_imu_progressive(cfg, kind, &control, src_w, src_h);

    // ── Resolve initial calibrations from settings + preset + protobuf
    let mut cached_gen = control.settings_generation.load(Ordering::SeqCst);
    let (mut calib_left, mut calib_right) = resolve_fisheye_calib_pair(
        &control.settings.read(), kind, src_w, src_h, dji_osv_imu.as_ref(),
    );
    tracing::info!(
        "decoder (fisheye): initial calib L fx={:.1}, cx={:.1}, cy={:.1}, k={:?}, k5={:.6} | R fx={:.1}, cx={:.1}, cy={:.1}, k={:?}, k5={:.6}",
        calib_left.fx, calib_left.cx, calib_left.cy, calib_left.k, calib_left.k5,
        calib_right.fx, calib_right.cx, calib_right.cy, calib_right.k, calib_right.k5
    );
    let _ = presets::presets(); // anchor the import for downstream use

    // ── Source-specific stabilization (Tasks #8, #15) ───────────────
    // BRAW: VQF 6D on raw gyro+accel from braw_helper --gyro.
    // DJI OSV: direct integration of camera-supplied quats from the
    //   `djmd` protobuf stream (no VQF — camera already fused).
    // SBS fisheye: identity (Insta360 / Vuze / Canon parsers are
    //   per-format and not yet implemented).
    //
    // Compute is lazy + live: the toggle is checked at each iteration
    // of the main loop; first time stabilize flips on, the gyro is
    // extracted and the per-frame rotations are cached. Toggle off
    // drops the cache. This lets the user enable stab mid-playback
    // without Stop→Play.
    let mut stab_rotations: Option<Vec<EquirectRotation>> = None;
    let mut last_stabilize_state = false;
    let mut compute_stab = |dji_osv_imu: Option<&vr180_fisheye::DjiOsvImu>,
                            control: &DecoderControl,
                            fps: f32| -> Option<Vec<EquirectRotation>> {
        // Merged recording: total frames across ALL segments (the IMU is
        // aggregated the same way, so the per-frame rotations span the whole
        // timeline). For a lone clip `segments` is `[path]`, so this is the
        // single-file count.
        let total_frames = segments_total_frames(&cfg.segments).max(1);
        match kind {
            vr180_pipeline::SourceKind::BlackmagicRaw => {
                match vr180_braw::BrawGyroData::extract(&cfg.path) {
                    Ok(gyro_data) => match vr180_pipeline::braw_imu::compute_braw_stabilization(
                        &gyro_data, fps, total_frames
                    ) {
                        Ok(stab) => {
                            tracing::info!(
                                "decoder (fisheye/braw): stab ready, {} frames, bias={:?}°/s",
                                stab.per_frame.len(), stab.bias_deg_s
                            );
                            Some(stab.per_frame)
                        }
                        Err(e) => { tracing::warn!("braw stab failed: {e}"); None }
                    },
                    Err(e) => { tracing::warn!("braw gyro extract failed: {e}"); None }
                }
            }
            vr180_pipeline::SourceKind::DjiOsv => {
                if let Some(osv) = dji_osv_imu {
                    // Base mode is camera-lock (smooth_ms = 0 →
                    // q_corr = q_actual.conjugate(), no cap → no
                    // clamp). The OSV sliders ONLY activate when the
                    // user moves them off 0 — at slider=0 the values
                    // passed below are bit-identical to the legacy
                    // hardcoded `f32::INFINITY` / `0.0`.
                    let s = control.settings.read();
                    let max_corr_deg = if s.dji_max_corr_deg > 0.0 {
                        s.dji_max_corr_deg
                    } else {
                        f32::INFINITY
                    };
                    let smooth_ms = s.dji_smooth_ms;
                    let responsiveness = s.dji_responsiveness;
                    vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(s.dji_imu_phase_ms);
                    drop(s);
                    match vr180_pipeline::dji_imu::compute_dji_stabilization(
                        osv, total_frames, max_corr_deg, smooth_ms, fps, responsiveness,
                    ) {
                        Ok(stab) => {
                            tracing::info!(
                                "decoder (fisheye/osv): stab ready, {} frames, {} with HR quats",
                                stab.per_frame.len(), stab.frames_with_hr_quat
                            );
                            Some(stab.per_frame)
                        }
                        Err(e) => { tracing::warn!("dji stab failed: {e}"); None }
                    }
                } else { None }
            }
            _ => None,
        }
    };
    // Live-recompute trigger for the MAIN preview: bump-compare the
    // stab-affecting settings hash so slider moves (smooth / max-corr /
    // imu-phase) re-derive stabilization mid-playback, like the toggle.
    let mut last_stab_key = stab_key(&control.settings.read());
    if control.settings.read().stabilize {
        last_stabilize_state = true;
        // Compute now only if the full IMU is already in (cached). For a fresh
        // OSV clip the calib-only partial has no quats yet — the progressive-
        // upgrade block computes stab once the background load finishes.
        if control.imu_ready.load(Ordering::SeqCst) {
            stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control, fps);
        }
    }
    tracing::info!(
        "decoder (fisheye): startup stab_rotations = {} (toggle is now LIVE)",
        if stab_rotations.is_some() { "Some(per-frame matrices)" } else { "None (identity)" }
    );

    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let _ = fps;

    // Initial trim_in is applied by the worker thread BEFORE it
    // starts sending pairs (see worker spawn above). Sync the
    // main-thread `time_offset` to match.
    if let Some(t_in) = initial_trim_in {
        if t_in > 0.001 {
            tracing::info!("decoder (fisheye): initial seek to trim_in = {:.3}s (in worker)", t_in);
            time_offset = t_in;
        }
    }

    // Decimate at >30 fps even with pipelining. Tested: 50 fps native
    // preview of 3840² dual-stream HEVC is borderline on M5 Max — VT
    // decode is ~20 ms per frame (matching the 50 fps budget exactly),
    // so any sustained slowdown drains the channel buffer and we fall
    // behind permanently. Empirically, smooth playback survives ~28 s
    // at 50 fps before slipping; 25 fps preview from a 50 fps source
    // ran 60+ s without a single drop. Trades temporal resolution for
    // stable playback. Export uses full source frames separately.
    let preview_decimation: u32 = if fps > 30.5 { 2 } else { 1 };
    let preview_fps = fps / preview_decimation as f32;
    let preview_dt = 1.0_f64 / preview_fps as f64;
    if preview_decimation > 1 {
        tracing::info!(
            "decoder (fisheye): fps={:.2} → preview decimation {}× → effective {:.2} fps",
            fps, preview_decimation, preview_fps,
        );
    }

    // Time the decode by measuring from the end of the previous iter
    // to the start of this one — `pair_rx.recv()` is what's between them.
    // Because decode runs on a worker thread now, this measures channel
    // *wait* time, not raw decode time. When decode is faster than
    // render we wait ~0 µs (worker keeps the buffer full).
    let mut last_iter_end = std::time::Instant::now();
    let mut decode_us: u128 = 0;
    // Pull a pair from the worker's channel, dropping any tagged
    // with stale generations from before the latest seek.
    let recv_next_pair = |rx: &crossbeam_channel::Receiver<(u64, vr180_pipeline::fisheye_decode::FisheyePair)>,
                          expected: u64|
        -> Option<vr180_pipeline::fisheye_decode::FisheyePair>
    {
        loop {
            match rx.recv() {
                Ok((g, p)) if g == expected => return Some(p),
                Ok(_) => continue,    // stale, drop
                Err(_) => return None,
            }
        }
    };
    // Persist the current pair across iterations so the pause loop
    // can re-render it on slider changes (live preview while paused).
    // While not paused we discard it each iteration and pull fresh.
    let mut maybe_pair: Option<vr180_pipeline::fisheye_decode::FisheyePair> = None;
    'main: loop {
        // Pull a new pair from the channel UNLESS we're paused and
        // already holding a pair we want to re-render with new
        // settings (the pause loop below sets `force_render_next` on
        // a settings_generation bump and breaks out to this point).
        let stay_on_pair = control.paused.load(Ordering::SeqCst)
            && maybe_pair.is_some();
        if !stay_on_pair {
            // Block for one pair to render.
            let mut p = recv_next_pair(&pair_rx, expected_gen);
            // Drop up to (preview_decimation-1) MORE — but only ones
            // already buffered (non-blocking try_recv). Forcing blocking
            // recvs for the dropped frames would double the wait when
            // decode-bound (channel empty), without any benefit: the GPU
            // render is not the bottleneck there, so there is nothing to
            // save by thinning it. Non-blocking drain thins the displayed
            // framerate ONLY when the worker is genuinely ahead (channel
            // has spare pairs); when VT is struggling we render every
            // pair it delivers, which is what keeps throttled playback
            // smooth instead of frozen.
            if p.is_some() {
                for _ in 1..preview_decimation {
                    match pair_rx.try_recv() {
                        Ok((g, np)) if g == expected_gen => p = Some(np),
                        Ok(_) => {}             // stale, drop, keep draining
                        Err(_) => break,        // empty or disconnected
                    }
                }
            }
            match p {
                Some(p) => maybe_pair = Some(p),
                None => break 'main, // EOS or worker disconnected
            }
        }
        let pair = maybe_pair.as_ref().expect("maybe_pair populated above");
        decode_us = last_iter_end.elapsed().as_micros();
        // ── Progressive IMU: swap the calib-only partial for the full IMU once
        //    the background load finishes, then compute stabilization (auto-on
        //    if `stabilize` is set). Cheap no-op once `imu_ready`.
        if !control.imu_ready.load(Ordering::SeqCst) && control.stab_loading.load(Ordering::SeqCst) {
            if let Some(full) = try_upgrade_dji_imu(cfg, dji_osv_imu.as_ref()) {
                dji_osv_imu = Some(full);
                if control.settings.read().stabilize {
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control, fps);
                    last_stabilize_state = true;
                }
                control.imu_ready.store(true, Ordering::SeqCst);
                control.stab_loading.store(false, Ordering::SeqCst);
                force_render_next = true; // re-render the held frame with the new IMU/stab
                tracing::info!("decoder (fisheye): full IMU applied — stabilization enabled");
            }
        }
        // ── 0. Live stabilize toggle. When the user flips the
        //       checkbox during playback, recompute on this iteration.
        //       The compute takes a few hundred ms (a brief stutter)
        //       but the user doesn't have to Stop→Play any more.
        let now_stabilize = control.settings.read().stabilize;
        if now_stabilize != last_stabilize_state {
            if now_stabilize {
                if control.imu_ready.load(Ordering::SeqCst) {
                    tracing::info!("decoder (fisheye): stabilize ON → computing stab");
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control, fps);
                } else if control.stab_loading.load(Ordering::SeqCst) {
                    tracing::info!("decoder (fisheye): stabilize ON → resuming IMU load");
                    control.imu_progress.pause.store(false, Ordering::SeqCst);
                } else {
                    tracing::info!("decoder (fisheye): stabilize ON → starting IMU load");
                    spawn_dji_quat_load(cfg, &control);
                }
            } else {
                tracing::info!("decoder (fisheye): stabilize OFF → pausing IMU load");
                stab_rotations = None;
                control.imu_progress.pause.store(true, Ordering::SeqCst);
            }
            last_stabilize_state = now_stabilize;
            // Keep the live-recompute hash in sync so the settings-change
            // block below doesn't redundantly recompute on this toggle.
            last_stab_key = stab_key(&control.settings.read());
        }

        // ── 1. Drain commands.
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            let _ = iter_cmd_tx.send(IterCmd::Seek(t));
            expected_gen = expected_gen.wrapping_add(1);
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            continue 'main;
        }

        // ── 2. Trim-out loop.
        // `frame_idx` counts loop iterations (= rendered frames after
        // decimation), so pacing uses `preview_dt` not the source `dt`.
        // `frame_t_abs` is still in source-time for trim/seek lookups.
        let frame_t_rel = frame_idx as f64 * preview_dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    let _ = iter_cmd_tx.send(IterCmd::Seek(in_t));
                    expected_gen = expected_gen.wrapping_add(1);
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    continue 'main;
                }
            }
        }

        // ── 3. Pause handling. Sleep until unpaused, a slider change,
        //       or a command. Slider changes (settings_generation bump)
        //       break out with `force_render_next = true` so the existing
        //       render code re-runs against the in-scope `pair` with the
        //       new settings — that's how live preview works while paused.
        //
        //       Skip the pause loop when we JUST pulled a new pair
        //       (`!stay_on_pair`): we want to render it once before
        //       parking. That makes the first frame visible right after
        //       a paused-state spawn — no need to press Play.
        if control.paused.load(Ordering::SeqCst) && !force_render_next && stay_on_pair {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true;
                    break;
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            let _ = iter_cmd_tx.send(IterCmd::Seek(t.max(0.0)));
                            expected_gen = expected_gen.wrapping_add(1);
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            maybe_pair = None; // discard stale pair, pull fresh
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // ── 4. Settings changed → re-resolve per-eye calibs.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snap = control.settings.read();
            let (l, r) = resolve_fisheye_calib_pair(
                &snap, kind, src_w, src_h, dji_osv_imu.as_ref(),
            );
            calib_left = l;
            calib_right = r;
            drop(snap);
            cached_gen = current_gen;
            // Live stab recompute when a stab-affecting slider (smooth /
            // max-corr / imu-phase) changed — keeps the MAIN preview in
            // sync, not just the zoom/DetailCache view. Only fires when the
            // stab hash actually moves (brief stutter on change, then live).
            let nk = stab_key(&control.settings.read());
            if nk != last_stab_key {
                let on = control.settings.read().stabilize;
                stab_rotations = if on {
                    compute_stab(dji_osv_imu.as_ref(), &control, fps)
                } else {
                    None
                };
                last_stab_key = nk;
            }
            tracing::info!(
                "decoder (fisheye): live calib update gen={}, L.fx={:.1} k5={:.6}, R.fx={:.1} k5={:.6}",
                current_gen, calib_left.fx, calib_left.k5, calib_right.fx, calib_right.k5
            );
        }

        // ── 5. Wall-clock pacing.
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();

        // Decode-bound vs render-bound. `decode_us` is how long the
        // recv for THIS iteration's pair(s) blocked on the worker. If
        // it ate a big slice of the frame budget, VT decode — not the
        // GPU render — is the bottleneck (the channel is draining).
        //
        // HEVC can't skip-decode (every frame is a reference), so when
        // VT thermally throttles below the source rate the worker simply
        // can't deliver pairs fast enough. Skipping the *render* here is
        // futile: we'd still block on the next recv, the idealized wall
        // clock keeps outrunning the frame clock, and playback spirals
        // into rendering almost nothing — the "frozen after ~30 s"
        // symptom. Instead, resync the playback clock to now and render
        // every pair the worker manages to deliver. 50p that VT can only
        // sustain at ~18 fps then plays *smoothly* at ~18 fps rather than
        // freezing.
        //
        // The skip path below is kept for the genuinely render-bound
        // case (channel full → decode_us ≈ 0, but the GPU render fell
        // behind real time): there, skipping a frame DOES let us catch
        // up to the audio clock.
        // Decode-bound only when the decode of the whole decimation group overruns
        // its real-time budget (preview_dt). `decode_us` covers all `preview_decimation`
        // frames, so comparing against the FULL budget (not ½) is what keeps the 2×-
        // decimated 50p path classified the same as the un-decimated 30p path: below
        // budget → render-bound → drop frames and stay locked to the clock; only above
        // it (decode truly can't sustain real time) do we rewind into smooth slow-mo.
        // (½-budget here mis-flagged 50p as decode-bound on Macs that decode 2 frames
        // in 20–40 ms → slow motion even though dropping frames would have kept up.)
        let decode_bound = (decode_us as f64) > preview_dt * 1.0e6;
        if decode_bound {
            // Pace to actual decode throughput — drop accumulated debt.
            start_wall = Some(std::time::Instant::now()
                - std::time::Duration::from_secs_f64(frame_t_rel)
                - paused_offset);
        } else if !force_render_next && wall_t > frame_t_rel + preview_dt * 0.5 {
            frame_idx += 1;
            skipped_count += 1;
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel)
                    - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (fisheye): render-bound, behind by {:.1} ms — skipped {} (decode={}µs)",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count, decode_us);
            }
            last_iter_end = std::time::Instant::now();
            continue;
        }

        // ── 6. GPU project each eye + compose SBS.
        // Look up the stabilization rotation by the ACTUAL frame's
        // PTS, not the loop counter — seeks land on keyframes which
        // may be before the requested timestamp, and the iterator
        // can buffer frames internally. PTS is the only thing
        // guaranteed to line up with the protobuf's frame-block
        // index (one frame_block per video frame, by construction).
        let phase_t0 = std::time::Instant::now();
        let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
            (pair.pts_s / dt).round() as usize
        } else {
            // Fallback if the iterator didn't supply PTS (e.g. BRAW —
            // we synthesise PTS from frame_idx there, but be defensive).
            let abs = ((time_offset / dt).round() as i64) + frame_idx as i64;
            abs.max(0) as usize
        };
        let absolute_frame_idx = stab_idx as u32;
        let rot = stab_rotations
            .as_ref()
            .and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);

        // Per-row RS matrices for this frame. Built lazily on CPU; ~46 KB
        // for src_h=1280 so the per-frame `queue.write_buffer` is cheap.
        // When present, fused per-row stab + RS gives us DJI Studio's
        // per-slab quality (matches their `getMatrixForEisAndHorizontal`
        // x slab x scanline approach) without leaving the live pipeline.
        let rs_quats: Option<Vec<vr180_core::gyro::cori_iori::Quat>> =
            if control.settings.read().stabilize {
                dji_osv_imu.as_ref().and_then(|osv| {
                    vr180_pipeline::dji_imu::compute_per_row_quaternions_for_frame(
                        osv,
                        stab_idx,
                        // FPS-aware readout: OSMO 360 sensor mode at 50fps
                        // gives 16.23 ms vs 18.3 ms at 30fps.
                        vr180_pipeline::dji_imu::dji_osmo_readout_ms_for_fps(fps) / 1000.0,
                        src_h,
                        fps,
                    )
                })
            } else {
                None
            };
        let phase_rs_quats = phase_t0.elapsed();
        let phase_t1 = std::time::Instant::now();
        // Per-clip lens_a (factory mount) feeds into the per-row
        // basis change. Same fallback as compute_dji_stabilization.
        let lens_a_for_pack: [f32; 4] = dji_osv_imu
            .as_ref()
            .and_then(|osv| osv.lens_a.mount_quat_xyzw)
            .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
        let rs_rows_f32: Option<Vec<f32>> = rs_quats.as_ref()
            .map(|q| vr180_pipeline::dji_imu::pack_per_row_camera_matrices(q, lens_a_for_pack));
        let phase_pack = phase_t1.elapsed();

        // Diagnostic: per-axis decomposition of the stab rotation so we
        // can see if specific axes (roll, pitch, yaw) are under-
        // corrected. Roll = rotation around camera Z (optical axis);
        // pitch around X; yaw around Y. Decomposition uses Tait-Bryan
        // angles (ZYX intrinsic) computed from the row-major matrix.
        if frame_idx % 10 == 0 {
            if let Some(rq) = rs_quats.as_ref() {
                let to_deg = |q: &vr180_core::gyro::cori_iori::Quat| {
                    2.0 * q.w.abs().min(1.0).acos().to_degrees()
                };
                let intra_d = (to_deg(&rq[0]) - to_deg(&rq[rq.len() - 1])).abs();
                let tr = rot.0[0] + rot.0[4] + rot.0[8];
                let cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
                let stab_mag = cos_theta.acos().to_degrees();
                // YXZ-intrinsic Tait-Bryan for camera convention
                // (X=right, Y=up, Z=forward): yaw is around Y, pitch
                // around X, roll around Z. Indexing the row-major
                // mat3 stored as `[r00 r01 r02; r10 r11 r12; r20 r21 r22]`.
                let r00 = rot.0[0]; let r10 = rot.0[3];
                let r20 = rot.0[6]; let r21 = rot.0[7]; let r22 = rot.0[8];
                let yaw   = (-r20).atan2(r00).to_degrees();
                let pitch = r21.atan2(r22).to_degrees();
                let roll  = (-r10).asin().to_degrees();
                // Note on the previous diagnostic: it mislabeled the
                // axes (the math we feed to the shader was — and is —
                // correct, the log just printed wrong column headers).
                // Pure camera pitch (tilt) used to print as "roll"; pure
                // camera roll printed as "pitch"; pure camera yaw printed
                // negated under "yaw". Confirmed against a +130° tilt
                // motion that came through as `roll=-129.96°` in the
                // old log.
                tracing::info!(
                    "f={:>4} stab={:>6.2}° (yaw={:+6.2} pitch={:+6.2} roll={:+6.2}) intra-Δ={:.3}°",
                    frame_idx, stab_mag, yaw, pitch, roll, intra_d
                );
            }
        }

        let phase_t2 = std::time::Instant::now();
        // Compose the per-frame stab matrix with the per-eye view
        // adjustment (global pano-map + stereo offset). When all six
        // sliders are 0 (default), `is_identity()` short-circuits
        // and `rot` is passed through verbatim → byte-identical to
        // the no-pano-map pipeline.
        let view_adjust = {
            let s = control.settings.read();
            vr180_pipeline::panomap::ViewAdjust {
                pano_yaw_deg: s.pano_yaw_deg,
                pano_pitch_deg: s.pano_pitch_deg,
                pano_roll_deg: s.pano_roll_deg,
                stereo_yaw_deg: s.stereo_yaw_deg,
                stereo_pitch_deg: s.stereo_pitch_deg,
                stereo_roll_deg: s.stereo_roll_deg,
                upside_down: s.camera_upside_down,
            }
        };
        let (rot_left, rot_right) = if view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = view_adjust.per_eye_matrices();
            (
                vr180_pipeline::gpu::EquirectRotation(
                    vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                vr180_pipeline::gpu::EquirectRotation(
                    vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };
        // Decide projection target: half-equirect VR180 (default) or
        // stabilized fisheye output. Both paths apply the per-frame stab
        // matrix composed with the per-eye view-adjust (so panomap +
        // stereo-offset sliders work in either mode).
        let (left_tex, right_tex) = {
            let s = control.settings.read();
            let mode = s.fisheye_output_mode;
            drop(s);
            match mode {
                FisheyeOutputMode::HalfEquirect => {
                    if let Some(rs_buf) = rs_rows_f32.as_deref() {
                        let l = pipeline.project_fisheye_to_equirect_rs_texture(
                            &pair.left, src_w, src_h, eye_w, eye_h, rot_left, calib_left, rs_buf, 0,
                        )?;
                        let r = pipeline.project_fisheye_to_equirect_rs_texture(
                            &pair.right, src_w, src_h, eye_w, eye_h, rot_right, calib_right, rs_buf, 1,
                        )?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_to_equirect_texture(
                            &pair.left, src_w, src_h, eye_w, eye_h, rot_left, calib_left, 0,
                        )?;
                        let r = pipeline.project_fisheye_to_equirect_texture(
                            &pair.right, src_w, src_h, eye_w, eye_h, rot_right, calib_right, 1,
                        )?;
                        (l, r)
                    }
                }
                FisheyeOutputMode::Fisheye => {
                    // Square fisheye output: use min(eye_w, eye_h) for
                    // both axes so the result is a circle in a square
                    // frame regardless of the half-equirect aspect.
                    let side = eye_w.min(eye_h);
                    if let Some(rs_buf) = rs_rows_f32.as_deref() {
                        // Per-row rolling-shutter correction, same as the
                        // half-equirect path — the RS warp operates on the
                        // source-frame direction independent of the output
                        // projection.
                        let l = pipeline.project_fisheye_to_fisheye_rs_texture(
                            &pair.left, src_w, src_h, side, side, rot_left, calib_left, rs_buf,
                        )?;
                        let r = pipeline.project_fisheye_to_fisheye_rs_texture(
                            &pair.right, src_w, src_h, side, side, rot_right, calib_right, rs_buf,
                        )?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_to_fisheye_texture(
                            &pair.left, src_w, src_h, side, side, rot_left, calib_left,
                        )?;
                        let r = pipeline.project_fisheye_to_fisheye_texture(
                            &pair.right, src_w, src_h, side, side, rot_right, calib_right,
                        )?;
                        (l, r)
                    }
                }
            }
        };
        let (preview_eye_w, preview_eye_h) = (left_tex.width(), left_tex.height());
        let phase_project = phase_t2.elapsed();
        let phase_t3 = std::time::Instant::now();
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex,
            preview_eye_w, preview_eye_h,
        )?;
        let phase_compose = phase_t3.elapsed();
        // Per-frame phase timing. First 20 rendered frames log every
        // time so we see warm-cache numbers; after that, every 30.
        // `decode` is the time `decoder.next_pair()` blocked (i.e.,
        // VT decode + sws downscale).
        let should_log = frame_idx < 20 || frame_idx % 30 == 0;
        if should_log {
            tracing::info!(
                "perf f={:>4} decode={:>5}µs rs_quats={:>5}µs pack={:>5}µs project={:>5}µs compose={:>5}µs  render={:>5}µs  budget@{:.0}fps={:.0}µs (dec={}x)",
                frame_idx,
                decode_us,
                phase_rs_quats.as_micros(),
                phase_pack.as_micros(),
                phase_project.as_micros(),
                phase_compose.as_micros(),
                phase_t0.elapsed().as_micros(),
                preview_fps,
                1_000_000.0 / preview_fps,
                preview_decimation,
            );
        }

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w,
            height: out_h,
            frame_idx: absolute_frame_idx,
            // Shown frame's own time (pts via stab_idx), not the pacing clock —
            // keeps the zoom still (DetailCache) on the same frame as the preview
            // even after render-bound skips. See the macOS vt-zc path.
            timestamp_s: absolute_frame_idx as f64 * dt,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;
        let now = start_wall.unwrap().elapsed().as_secs_f64()
            - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        // While paused we re-render the SAME source pair on slider
        // changes — don't advance the timeline cursor.
        if !control.paused.load(Ordering::SeqCst) {
            frame_idx += 1;
        }
        let _ = _check_pair_dims(pair, src_w, src_h);
        last_iter_end = std::time::Instant::now();
    }
    Ok(())
}

/// Holds one source frame's GPU-resident resources alive together: the shared
/// D3D11 pair and the two wgpu P010 textures imported from it. The imports
/// **alias** the pair's D3D11 memory, so the pair must outlive them — bundling
/// them guarantees that. Kept in a short retire queue so a frame is freed only
/// a couple of frames after it stops being projected; by then the GPU has
/// finished sampling it. (We can't block-poll for GPU completion on this
/// thread without risking the cross-thread wgpu deadlock from Lesson #1, so we
/// defer the drop instead of fencing.)
#[cfg(target_os = "windows")]
struct FrameHold {
    /// Held for its `Drop` (closes NT handles / releases D3D11 textures) and
    /// for `pts_s`. Field is read, so no underscore needed.
    pair: vr180_pipeline::fisheye_decode::SharedFisheyePair,
    /// The two eyes' imported RGBA16 textures (D3D11-converted, working-res),
    /// aliasing the D3D11 memory in `pair`.
    l_tex: wgpu::Texture,
    r_tex: wgpu::Texture,
}

/// Push a just-replaced frame onto the retire queue and drop anything older
/// than the last two frames (deferred GPU-completion drop — see [`FrameHold`]).
#[cfg(target_os = "windows")]
fn retire_frame(
    q: &mut std::collections::VecDeque<FrameHold>,
    old: Option<FrameHold>,
) {
    if let Some(fh) = old {
        q.push_back(fh);
        while q.len() > 2 { q.pop_front(); }
    }
}

/// Windows zero-copy fisheye decoder loop (DJI OSV dual-stream).
///
/// Mirrors [`run_fisheye`]'s pacing / pause-resume / seek / trim / live-stab /
/// live-calib semantics, but the pixel source is GPU-resident: a worker
/// sub-thread decodes both streams with `d3d11va` and shares each eye out as an
/// NT-handle P010 texture (no CPU download, no swscale); this thread imports
/// each into wgpu's Vulkan device (zero-copy memory alias) and projects the
/// P010 planes directly via `project_fisheye_p010_planar_to_equirect_texture_16`.
///
/// Honors both output modes: `HalfEquirect` (RS-corrected when OSV stab is on)
/// and `Fisheye` (raw stabilized fisheye SBS), each via the matching RGBA16
/// texture-input projection — so the live preview matches the export 1:1.
/// v1 limitation vs the CPU path: the `Fisheye` projection applies frame-level
/// stabilization only (no per-row RS), matching the GPU-resident export.
#[cfg(target_os = "windows")]
fn run_fisheye_zerocopy(
    pipeline: Arc<vr180_pipeline::gpu::Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    kind: vr180_pipeline::SourceKind,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    ctx: vr180_pipeline::interop_windows::VulkanImportCtx,
    iter: vr180_pipeline::fisheye_decode::D3d11SharedDualStreamIter,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    use vr180_pipeline::fisheye_decode::SharedFisheyePair;

    // ── Worker sub-thread: decode + share (D3D11 only, never touches wgpu) ──
    enum IterCmd { Seek(f64) }
    let (pair_tx, pair_rx) = crossbeam_channel::bounded::<(u64, SharedFisheyePair)>(8);
    let (iter_cmd_tx, iter_cmd_rx) = crossbeam_channel::bounded::<IterCmd>(8);
    let (dims_tx, dims_rx) = crossbeam_channel::bounded::<(u32, u32)>(1);

    let initial_trim_in = control.settings.read().trim_in_s;
    let mut iter = iter;
    let _decode_handle = std::thread::spawn(move || {
        if let Some(t_in) = initial_trim_in {
            if t_in > 0.001 { let _ = iter.seek(t_in); }
        }
        let (sw, sh) = iter.eye_dims();
        if dims_tx.send((sw, sh)).is_err() { return; }
        let mut gen: u64 = 0;
        loop {
            while let Ok(cmd) = iter_cmd_rx.try_recv() {
                match cmd {
                    IterCmd::Seek(t) => {
                        let _ = iter.seek(t);
                        gen = gen.wrapping_add(1);
                        // Decode forward to the frame AT t and yield it first
                        // (seek lands on the keyframe ≤ t; same as the CPU path).
                        let dtl = 1.0 / (fps.max(1e-3) as f64);
                        for _ in 0..1200 {
                            match iter.next_pair() {
                                Ok(Some(p)) => {
                                    if p.pts_s >= t - dtl * 0.5 {
                                        if pair_tx.send((gen, p)).is_err() { return; }
                                        break;
                                    }
                                }
                                _ => break,
                            }
                        }
                    }
                }
            }
            match iter.next_pair() {
                Ok(Some(p)) => { if pair_tx.send((gen, p)).is_err() { break; } }
                Ok(None) => break,
                Err(e) => { tracing::warn!("zero-copy decode worker: {e}"); break; }
            }
        }
    });

    let (src_w, src_h) = match dims_rx.recv() {
        Ok(d) => d,
        Err(_) => anyhow::bail!("zero-copy decode worker failed to start"),
    };
    tracing::info!("decoder (fisheye/zc): native eye dims = {}x{} (full-res zero-copy)", src_w, src_h);

    // ── Per-eye calibration + stabilization from the OSV protobuf ────
    let mut dji_osv_imu = load_dji_imu_progressive(cfg, kind, &control, src_w, src_h);

    // `src_w`/`src_h` ARE the working (downscaled) dims the worker yields — the
    // D3D11 side already converted P010→RGBA16 AND box-downscaled to this res,
    // so the projection minifies only mildly (single tap, no alias) and we
    // import a clean single-plane RGBA16. Calib resolves against THIS res (the
    // projection samples the imported working-res texture).
    let mut cached_gen = control.settings_generation.load(Ordering::SeqCst);
    let (mut calib_left, mut calib_right) = resolve_fisheye_calib_pair(
        &control.settings.read(), kind, src_w, src_h, dji_osv_imu.as_ref(),
    );
    tracing::info!(
        "decoder (fisheye/zc): initial calib L fx={:.1} cx={:.1} cy={:.1} | R fx={:.1} cx={:.1} cy={:.1}",
        calib_left.fx, calib_left.cx, calib_left.cy, calib_right.fx, calib_right.cx, calib_right.cy
    );

    let total_frames = vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)
        .max(1);
    let compute_stab = |osv: Option<&vr180_fisheye::DjiOsvImu>,
                        control: &DecoderControl| -> Option<Vec<EquirectRotation>> {
        let osv = osv?;
        let s = control.settings.read();
        let max_corr_deg = if s.dji_max_corr_deg > 0.0 { s.dji_max_corr_deg } else { f32::INFINITY };
        let smooth_ms = s.dji_smooth_ms;
        let responsiveness = s.dji_responsiveness;
        vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(s.dji_imu_phase_ms);
        drop(s);
        match vr180_pipeline::dji_imu::compute_dji_stabilization(
            osv, total_frames, max_corr_deg, smooth_ms, fps, responsiveness,
        ) {
            Ok(stab) => Some(stab.per_frame),
            Err(e) => { tracing::warn!("zc: dji stab failed: {e}"); None }
        }
    };
    let mut stab_rotations: Option<Vec<EquirectRotation>> = None;
    let mut last_stabilize_state = false;
    if control.settings.read().stabilize {
        last_stabilize_state = true;
        // Compute now only if the full IMU is cached; the progressive-upgrade
        // block computes stab once the background quats finish loading.
        if control.imu_ready.load(Ordering::SeqCst) {
            stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
        }
    }

    // ── Pacing / control state (same model as run_fisheye) ───────────
    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let mut expected_gen: u64 = 0;
    if let Some(t_in) = initial_trim_in {
        if t_in > 0.001 { time_offset = t_in; }
    }

    // Decimate >30 fps to half: the preview render contends with eframe and
    // can't reliably make a 50 fps (20 ms) budget on this heavy 3840² 10-bit
    // dual-stream source, so a clean locked 25 fps beats a render-bound,
    // stuttery ~30. (Matches the CPU/VT path's behaviour.)
    let preview_decimation: u32 = if fps > 30.5 { 2 } else { 1 };
    let preview_fps = fps / preview_decimation as f32;
    let preview_dt = 1.0_f64 / preview_fps as f64;
    if preview_decimation > 1 {
        tracing::info!("decoder (fisheye/zc): fps={:.2} → decimation {}× → {:.2} fps",
            fps, preview_decimation, preview_fps);
    }

    let mut last_iter_end = std::time::Instant::now();
    let mut decode_us: u128;

    let recv_next_pair = |rx: &crossbeam_channel::Receiver<(u64, SharedFisheyePair)>, expected: u64|
        -> Option<SharedFisheyePair>
    {
        loop {
            match rx.recv() {
                Ok((g, p)) if g == expected => return Some(p),
                Ok(_) => continue,
                Err(_) => return None,
            }
        }
    };

    // `current` is the frame being shown; `retire_q` holds recently-replaced
    // frames until the GPU is surely done with them (deferred drop, 2 deep).
    let mut current: Option<FrameHold> = None;
    let mut retire_q: std::collections::VecDeque<FrameHold> = std::collections::VecDeque::new();

    'main: loop {
        let stay_on_pair = control.paused.load(Ordering::SeqCst) && current.is_some();
        if !stay_on_pair {
            let mut p = recv_next_pair(&pair_rx, expected_gen);
            if p.is_some() {
                for _ in 1..preview_decimation {
                    match pair_rx.try_recv() {
                        Ok((g, np)) if g == expected_gen => p = Some(np),
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
            match p {
                Some(sp) => {
                    // Import both eyes → single-plane RGBA16 wgpu textures
                    // aliasing the D3D11-converted memory.
                    let l_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.left) };
                    let r_tex = unsafe { ctx.import_rgba16(&pipeline.device, &sp.right) };
                    let prev = current.take();
                    retire_frame(&mut retire_q, prev);
                    current = Some(FrameHold { pair: sp, l_tex, r_tex });
                }
                None => break 'main,
            }
        }
        let fh = current.as_ref().expect("current populated above");
        decode_us = last_iter_end.elapsed().as_micros();

        // Progressive IMU: swap the calib-only partial for the full IMU once the
        // background load finishes, then compute stabilization (auto-on if
        // `stabilize` is set). Cheap no-op once `imu_ready`.
        if !control.imu_ready.load(Ordering::SeqCst) && control.stab_loading.load(Ordering::SeqCst) {
            if let Some(full) = try_upgrade_dji_imu(cfg, dji_osv_imu.as_ref()) {
                dji_osv_imu = Some(full);
                if control.settings.read().stabilize {
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
                    last_stabilize_state = true;
                }
                control.imu_ready.store(true, Ordering::SeqCst);
                control.stab_loading.store(false, Ordering::SeqCst);
                force_render_next = true; // re-render the held frame with the new IMU/stab
                tracing::info!("decoder (fisheye): full IMU applied — stabilization enabled");
            }
        }

        // Live stabilize toggle.
        let now_stabilize = control.settings.read().stabilize;
        if now_stabilize != last_stabilize_state {
            if now_stabilize {
                if control.imu_ready.load(Ordering::SeqCst) {
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
                } else if control.stab_loading.load(Ordering::SeqCst) {
                    tracing::info!("decoder (fisheye): stabilize ON → resuming IMU load");
                    control.imu_progress.pause.store(false, Ordering::SeqCst);
                } else {
                    tracing::info!("decoder (fisheye): stabilize ON → starting IMU load");
                    spawn_dji_quat_load(cfg, &control);
                }
            } else {
                tracing::info!("decoder (fisheye): stabilize OFF → pausing IMU load");
                stab_rotations = None;
                control.imu_progress.pause.store(true, Ordering::SeqCst);
            }
            last_stabilize_state = now_stabilize;
        }

        // Drain commands (seek/stop).
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            let _ = iter_cmd_tx.send(IterCmd::Seek(t));
            expected_gen = expected_gen.wrapping_add(1);
            time_offset = t; frame_idx = 0; start_wall = None;
            paused_offset = std::time::Duration::ZERO; force_render_next = true;
            let prev = current.take(); retire_frame(&mut retire_q, prev);
            continue 'main;
        }

        // Trim-out loop.
        let frame_t_rel = frame_idx as f64 * preview_dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    let _ = iter_cmd_tx.send(IterCmd::Seek(in_t));
                    expected_gen = expected_gen.wrapping_add(1);
                    time_offset = in_t; frame_idx = 0; start_wall = None;
                    paused_offset = std::time::Duration::ZERO; force_render_next = true;
                    let prev = current.take(); retire_frame(&mut retire_q, prev);
                    continue 'main;
                }
            }
        }

        // Pause handling (re-render current source on slider changes).
        if control.paused.load(Ordering::SeqCst) && !force_render_next && stay_on_pair {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true; break;
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            let _ = iter_cmd_tx.send(IterCmd::Seek(t.max(0.0)));
                            expected_gen = expected_gen.wrapping_add(1);
                            time_offset = t.max(0.0); frame_idx = 0; start_wall = None;
                            paused_offset = std::time::Duration::ZERO; force_render_next = true;
                            let prev = current.take(); retire_frame(&mut retire_q, prev);
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // Settings changed → re-resolve per-eye calib.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snap = control.settings.read();
            let (l, r) = resolve_fisheye_calib_pair(&snap, kind, src_w, src_h, dji_osv_imu.as_ref());
            calib_left = l; calib_right = r; drop(snap); cached_gen = current_gen;
        }

        // Wall-clock pacing.
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        // Decode-bound only when the decode of the whole decimation group overruns
        // its real-time budget (preview_dt). `decode_us` covers all `preview_decimation`
        // frames, so comparing against the FULL budget (not ½) is what keeps the 2×-
        // decimated 50p path classified the same as the un-decimated 30p path: below
        // budget → render-bound → drop frames and stay locked to the clock; only above
        // it (decode truly can't sustain real time) do we rewind into smooth slow-mo.
        // (½-budget here mis-flagged 50p as decode-bound on Macs that decode 2 frames
        // in 20–40 ms → slow motion even though dropping frames would have kept up.)
        let decode_bound = (decode_us as f64) > preview_dt * 1.0e6;
        if decode_bound {
            start_wall = Some(std::time::Instant::now()
                - std::time::Duration::from_secs_f64(frame_t_rel) - paused_offset);
        } else if !force_render_next && wall_t > frame_t_rel + preview_dt * 0.5 {
            frame_idx += 1; skipped_count += 1;
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel) - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (fisheye/zc): render-bound, behind {:.1} ms, skipped {}",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count);
            }
            last_iter_end = std::time::Instant::now();
            continue;
        }

        // ── Project each eye (P010 planar) + compose SBS ─────────────
        let phase_t0 = std::time::Instant::now();
        let stab_idx = if fh.pair.pts_s.is_finite() && fh.pair.pts_s >= 0.0 {
            (fh.pair.pts_s / dt).round() as usize
        } else {
            (((time_offset / dt).round() as i64) + frame_idx as i64).max(0) as usize
        };
        let absolute_frame_idx = stab_idx as u32;
        let rot = stab_rotations.as_ref().and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);

        // Diagnostic: confirm frame-level stab is actually varying (rules out a
        // "does nothing" wiring bug vs. a "wobbly = missing rolling-shutter" issue).
        if frame_idx % 30 == 0 {
            let tr = rot.0[0] + rot.0[4] + rot.0[8];
            let mag = (((tr - 1.0) * 0.5).clamp(-1.0, 1.0)).acos().to_degrees();
            tracing::info!(
                "zc stab: idx={} mag={:.2}° (on={}, n_rots={}, rs_correct={})",
                stab_idx, mag,
                stab_rotations.is_some(),
                stab_rotations.as_ref().map(|v| v.len()).unwrap_or(0),
                control.settings.read().rs_correct,
            );
        }

        // Per-row rolling-shutter correction for THIS frame (DJI OSV). Computed
        // whenever stabilize is on — RS is part of the OSV stab (gated on
        // `stabilize`, not `rs_correct`), matching the CPU/paused-still path.
        // `src_h` is the working-res fisheye height the projection samples; the
        // same per-row quats apply to both eyes (one IMU). Without this the OSV
        // rolling-shutter jello is uncorrected (the "broken" stab).
        let rs_rows_f32: Option<Vec<f32>> = if control.settings.read().stabilize {
            dji_osv_imu.as_ref().and_then(|osv| {
                vr180_pipeline::dji_imu::compute_per_row_quaternions_for_frame(
                    osv,
                    stab_idx,
                    vr180_pipeline::dji_imu::dji_osmo_readout_ms_for_fps(fps) / 1000.0,
                    src_h,
                    fps,
                )
            }).map(|q| {
                let lens_a = dji_osv_imu.as_ref()
                    .and_then(|osv| osv.lens_a.mount_quat_xyzw)
                    .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
                vr180_pipeline::dji_imu::pack_per_row_camera_matrices(&q, lens_a)
            })
        } else {
            None
        };

        let view_adjust = {
            let s = control.settings.read();
            vr180_pipeline::panomap::ViewAdjust {
                pano_yaw_deg: s.pano_yaw_deg, pano_pitch_deg: s.pano_pitch_deg, pano_roll_deg: s.pano_roll_deg,
                stereo_yaw_deg: s.stereo_yaw_deg, stereo_pitch_deg: s.stereo_pitch_deg, stereo_roll_deg: s.stereo_roll_deg,
                upside_down: s.camera_upside_down,
            }
        };
        let (rot_left, rot_right) = if view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = view_adjust.per_eye_matrices();
            (
                EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };

        let (left_tex, right_tex) = {
            let s = control.settings.read();
            let mode = s.fisheye_output_mode;
            drop(s);
            let (ow, oh) = match mode {
                FisheyeOutputMode::HalfEquirect => (eye_w, eye_h),
                // Fisheye output is a CIRCLE in a SQUARE frame (per eye).
                FisheyeOutputMode::Fisheye => { let side = eye_w.min(eye_h); (side, side) }
            };
            // The imported textures are already RGBA16 at the working res (the
            // D3D11 side did P010→RGBA16 + downscale), so project single-tap
            // straight from them — no luma moiré, no chroma colour-fringing.
            match mode {
                // Normalized fisheye SBS — match the GPU-resident export's
                // Fisheye path EXACTLY so the preview previews what the export
                // writes. When OSV stab is on, the same per-row RS correction
                // as half-equirect (parity with the macOS p010 + CPU-worker
                // fisheye paths, which have always applied RS here).
                FisheyeOutputMode::Fisheye => {
                    if let Some(rs) = rs_rows_f32.as_deref() {
                        let l = pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                            &fh.l_tex, src_w, src_h, ow, oh, rot_left, calib_left, rs, 0,
                        )?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                            &fh.r_tex, src_w, src_h, ow, oh, rot_right, calib_right, rs, 1,
                        )?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                            &fh.l_tex, src_w, src_h, ow, oh, rot_left, calib_left, 0,
                        )?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                            &fh.r_tex, src_w, src_h, ow, oh, rot_right, calib_right, 1,
                        )?;
                        (l, r)
                    }
                }
                // Half-equirect VR180: when OSV stab is on use the RS variant so
                // per-row rolling shutter is corrected (kills the jello); same
                // per-row quats for both eyes.
                FisheyeOutputMode::HalfEquirect => {
                    if let Some(rs) = rs_rows_f32.as_deref() {
                        let l = pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                            &fh.l_tex, src_w, src_h, ow, oh, rot_left, calib_left, rs, 0,
                        )?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                            &fh.r_tex, src_w, src_h, ow, oh, rot_right, calib_right, rs, 1,
                        )?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                            &fh.l_tex, src_w, src_h, ow, oh, rot_left, calib_left, 0,
                        )?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                            &fh.r_tex, src_w, src_h, ow, oh, rot_right, calib_right, 1,
                        )?;
                        (l, r)
                    }
                }
            }
        };
        let (pe_w, pe_h) = (left_tex.width(), left_tex.height());
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex, pe_w, pe_h,
        )?;

        let should_log = frame_idx < 10 || frame_idx % 60 == 0;
        if should_log {
            tracing::info!(
                "perf(zc) f={:>4} wait={:>5}µs render={:>5}µs budget@{:.0}fps={:.0}µs (dec={}× native {}×{})",
                frame_idx, decode_us, phase_t0.elapsed().as_micros(),
                preview_fps, 1_000_000.0 / preview_fps, preview_decimation, src_w, src_h,
            );
        }

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w, height: out_h,
            frame_idx: absolute_frame_idx,
            // Shown frame's own time (pts via stab_idx), not the pacing clock —
            // keeps the zoom still (DetailCache) on the same frame as the preview
            // even after render-bound skips. See the macOS vt-zc path.
            timestamp_s: absolute_frame_idx as f64 * dt,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;
        let now = start_wall.unwrap().elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        if !control.paused.load(Ordering::SeqCst) { frame_idx += 1; }
        last_iter_end = std::time::Instant::now();
    }
    Ok(())
}

/// Defer-drop a just-replaced VT zero-copy pair so its IOSurface isn't
/// released until a couple of frames after the GPU last read it (the macOS
/// analogue of the Windows [`retire_frame`]; we can't fence on this thread).
#[cfg(target_os = "macos")]
fn retire_vt_pair(
    q: &mut std::collections::VecDeque<vr180_pipeline::fisheye_decode::VtSharedFisheyePair>,
    old: Option<vr180_pipeline::fisheye_decode::VtSharedFisheyePair>,
) {
    if let Some(p) = old { q.push_back(p); while q.len() > 2 { q.pop_front(); } }
}

/// macOS VideoToolbox zero-copy fisheye decoder loop (DJI OSV dual-stream).
///
/// The macOS counterpart to [`run_fisheye_zerocopy`] (Windows): same pacing /
/// pause-resume / seek / trim / live-stab / live-calib / projection logic, but
/// the pixel source is GPU-resident. VideoToolbox decodes both streams; each
/// eye's P010 IOSurface planes are wrapped as wgpu textures aliasing the
/// surface (no host download, no swscale — the win over the CPU
/// `DualStreamFisheyeIter` path), box-downscaled to the working res with
/// `resolve_p010_planes_to_rgba16`, then projected via the same
/// `project_fisheye_rgba16_texture_to_*` family the Windows path uses. Decode
/// is inline (no worker sub-thread) so the IOSurface textures never cross a
/// thread — matching the proven macOS EAC zero-copy loop [`run_zero_copy`].
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_fisheye_vt_zerocopy(
    pipeline: Arc<vr180_pipeline::gpu::Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    kind: vr180_pipeline::SourceKind,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut iter: vr180_pipeline::fisheye_decode::VtSharedDualStreamIter,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    use vr180_pipeline::fisheye_decode::VtSharedFisheyePair;

    // Native fisheye dims (what the iterator yields) → working preview res (what
    // we resolve to + project from). Calib resolves against the WORKING res,
    // exactly like the Windows zero-copy path (which yields already-downscaled
    // frames). Matches the CPU path's anti-aliased 1280-cap working res.
    let (native_w, native_h) = iter.eye_dims();
    let cap = vr180_pipeline::fisheye_decode::max_decode_side_for_fps(fps);
    let scale = (cap as f32 / native_w.max(native_h) as f32).min(1.0);
    let work_w = (((native_w as f32 * scale) as u32 + 1) & !1).max(2);
    let work_h = (((native_h as f32 * scale) as u32 + 1) & !1).max(2);
    let (src_w, src_h) = (work_w, work_h);
    tracing::info!(
        "decoder (fisheye/vt-zc): native {}x{} → working {}x{} (zero-copy P010 → resolve → project)",
        native_w, native_h, work_w, work_h
    );

    // Initial seek to trim_in.
    let initial_trim_in = control.settings.read().trim_in_s;
    if let Some(t_in) = initial_trim_in {
        if t_in > 0.001 { let _ = iter.seek(t_in); }
    }

    // ── Per-eye calibration + stabilization from the OSV protobuf ────
    let mut dji_osv_imu = load_dji_imu_progressive(cfg, kind, &control, src_w, src_h);

    let mut cached_gen = control.settings_generation.load(Ordering::SeqCst);
    let (mut calib_left, mut calib_right) = resolve_fisheye_calib_pair(
        &control.settings.read(), kind, src_w, src_h, dji_osv_imu.as_ref(),
    );
    tracing::info!(
        "decoder (fisheye/vt-zc): initial calib L fx={:.1} cx={:.1} cy={:.1} | R fx={:.1} cx={:.1} cy={:.1}",
        calib_left.fx, calib_left.cx, calib_left.cy, calib_right.fx, calib_right.cx, calib_right.cy
    );

    let total_frames = vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0).max(1);
    let compute_stab = |osv: Option<&vr180_fisheye::DjiOsvImu>,
                        control: &DecoderControl| -> Option<Vec<EquirectRotation>> {
        let osv = osv?;
        let s = control.settings.read();
        let max_corr_deg = if s.dji_max_corr_deg > 0.0 { s.dji_max_corr_deg } else { f32::INFINITY };
        let smooth_ms = s.dji_smooth_ms;
        let responsiveness = s.dji_responsiveness;
        vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(s.dji_imu_phase_ms);
        drop(s);
        match vr180_pipeline::dji_imu::compute_dji_stabilization(
            osv, total_frames, max_corr_deg, smooth_ms, fps, responsiveness,
        ) {
            Ok(stab) => Some(stab.per_frame),
            Err(e) => { tracing::warn!("vt-zc: dji stab failed: {e}"); None }
        }
    };
    let mut stab_rotations: Option<Vec<EquirectRotation>> = None;
    let mut last_stabilize_state = false;
    if control.settings.read().stabilize {
        last_stabilize_state = true;
        // Compute now only if the full IMU is cached; the progressive-upgrade
        // block computes stab once the background quats finish loading.
        if control.imu_ready.load(Ordering::SeqCst) {
            stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
        }
    }

    // ── Pacing / control state (same model as run_fisheye_zerocopy) ──
    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    if let Some(t_in) = initial_trim_in {
        if t_in > 0.001 { time_offset = t_in; }
    }

    let preview_decimation: u32 = if fps > 30.5 { 2 } else { 1 };
    let preview_fps = fps / preview_decimation as f32;
    let preview_dt = 1.0_f64 / preview_fps as f64;
    if preview_decimation > 1 {
        tracing::info!("decoder (fisheye/vt-zc): fps={:.2} → decimation {}× → {:.2} fps",
            fps, preview_decimation, preview_fps);
    }

    let mut last_iter_end = std::time::Instant::now();
    let mut decode_us: u128;

    // `held` is the frame currently in hand (re-rendered while paused);
    // `retire` defers the drop of replaced frames so the GPU finishes reading
    // their IOSurfaces first.
    let mut held: Option<VtSharedFisheyePair> = None;
    let mut retire: std::collections::VecDeque<VtSharedFisheyePair> = std::collections::VecDeque::new();

    'main: loop {
        let stay_on_pair = control.paused.load(Ordering::SeqCst) && held.is_some();
        if !stay_on_pair {
            // Pull one new pair (drop up to decimation-1 extra to thin >30 fps).
            let mut pulled = match iter.next_pair(&pipeline.device)? {
                Some(p) => Some(p),
                None => break 'main,
            };
            for _ in 1..preview_decimation {
                match iter.next_pair(&pipeline.device) {
                    Ok(Some(np)) => pulled = Some(np),
                    Ok(None) => break,
                    Err(e) => { tracing::warn!("vt-zc decode: {e}"); break; }
                }
            }
            retire_vt_pair(&mut retire, held.take());
            held = pulled;
        }
        let pair = held.as_ref().expect("held populated above");
        decode_us = last_iter_end.elapsed().as_micros();

        // Progressive IMU: swap the calib-only partial for the full IMU once the
        // background load finishes, then compute stabilization (auto-on if
        // `stabilize` is set). Cheap no-op once `imu_ready`.
        if !control.imu_ready.load(Ordering::SeqCst) && control.stab_loading.load(Ordering::SeqCst) {
            if let Some(full) = try_upgrade_dji_imu(cfg, dji_osv_imu.as_ref()) {
                dji_osv_imu = Some(full);
                if control.settings.read().stabilize {
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
                    last_stabilize_state = true;
                }
                control.imu_ready.store(true, Ordering::SeqCst);
                control.stab_loading.store(false, Ordering::SeqCst);
                force_render_next = true; // re-render the held frame with the new IMU/stab
                tracing::info!("decoder (fisheye): full IMU applied — stabilization enabled");
            }
        }

        // Live stabilize toggle.
        let now_stabilize = control.settings.read().stabilize;
        if now_stabilize != last_stabilize_state {
            if now_stabilize {
                if control.imu_ready.load(Ordering::SeqCst) {
                    stab_rotations = compute_stab(dji_osv_imu.as_ref(), &control);
                } else if control.stab_loading.load(Ordering::SeqCst) {
                    tracing::info!("decoder (fisheye): stabilize ON → resuming IMU load");
                    control.imu_progress.pause.store(false, Ordering::SeqCst);
                } else {
                    tracing::info!("decoder (fisheye): stabilize ON → starting IMU load");
                    spawn_dji_quat_load(cfg, &control);
                }
            } else {
                tracing::info!("decoder (fisheye): stabilize OFF → pausing IMU load");
                stab_rotations = None;
                control.imu_progress.pause.store(true, Ordering::SeqCst);
            }
            last_stabilize_state = now_stabilize;
        }

        // Drain commands (seek/stop).
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            iter.seek(t)?;
            time_offset = t; frame_idx = 0; start_wall = None;
            paused_offset = std::time::Duration::ZERO; force_render_next = true;
            retire_vt_pair(&mut retire, held.take());
            continue 'main;
        }

        // Trim-out loop.
        let frame_t_rel = frame_idx as f64 * preview_dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    iter.seek(in_t)?;
                    time_offset = in_t; frame_idx = 0; start_wall = None;
                    paused_offset = std::time::Duration::ZERO; force_render_next = true;
                    retire_vt_pair(&mut retire, held.take());
                    continue 'main;
                }
            }
        }

        // Pause handling (re-render held frame on slider changes).
        if control.paused.load(Ordering::SeqCst) && !force_render_next && stay_on_pair {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true; break;
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            iter.seek(t.max(0.0))?;
                            time_offset = t.max(0.0); frame_idx = 0; start_wall = None;
                            paused_offset = std::time::Duration::ZERO; force_render_next = true;
                            retire_vt_pair(&mut retire, held.take());
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
        }

        // Settings changed → re-resolve per-eye calib.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snap = control.settings.read();
            let (l, r) = resolve_fisheye_calib_pair(&snap, kind, src_w, src_h, dji_osv_imu.as_ref());
            calib_left = l; calib_right = r; drop(snap); cached_gen = current_gen;
        }

        // Wall-clock pacing.
        let wall_t = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        // Decode-bound only when the decode of the whole decimation group overruns
        // its real-time budget (preview_dt). `decode_us` covers all `preview_decimation`
        // frames, so comparing against the FULL budget (not ½) is what keeps the 2×-
        // decimated 50p path classified the same as the un-decimated 30p path: below
        // budget → render-bound → drop frames and stay locked to the clock; only above
        // it (decode truly can't sustain real time) do we rewind into smooth slow-mo.
        // (½-budget here mis-flagged 50p as decode-bound on Macs that decode 2 frames
        // in 20–40 ms → slow motion even though dropping frames would have kept up.)
        let decode_bound = (decode_us as f64) > preview_dt * 1.0e6;
        if decode_bound {
            start_wall = Some(std::time::Instant::now()
                - std::time::Duration::from_secs_f64(frame_t_rel) - paused_offset);
        } else if !force_render_next && wall_t > frame_t_rel + preview_dt * 0.5 {
            frame_idx += 1; skipped_count += 1;
            if wall_t > frame_t_rel + 1.0 {
                start_wall = Some(std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(frame_t_rel) - paused_offset);
            }
            if skipped_count % 30 == 0 {
                tracing::debug!("decoder (fisheye/vt-zc): render-bound, behind {:.1} ms, skipped {}",
                    (wall_t - frame_t_rel) * 1000.0, skipped_count);
            }
            // Drop the held frame so the next iteration pulls a fresh one.
            retire_vt_pair(&mut retire, held.take());
            last_iter_end = std::time::Instant::now();
            continue;
        }

        // ── Resolve P010 planes → working-res RGBA16 (GPU, zero host hop) ──
        let phase_t0 = std::time::Instant::now();
        let l_tex = pipeline.resolve_p010_planes_to_rgba16(
            &pair.left_y.texture, &pair.left_uv.texture, native_w, native_h, work_w, work_h)?;
        let r_tex = pipeline.resolve_p010_planes_to_rgba16(
            &pair.right_y.texture, &pair.right_uv.texture, native_w, native_h, work_w, work_h)?;

        // Frame-level stab rotation.
        let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
            (pair.pts_s / dt).round() as usize
        } else {
            (((time_offset / dt).round() as i64) + frame_idx as i64).max(0) as usize
        };
        let absolute_frame_idx = stab_idx as u32;
        let rot = stab_rotations.as_ref().and_then(|v| v.get(stab_idx).copied())
            .unwrap_or(EquirectRotation::IDENTITY);

        // Per-row rolling-shutter correction (OSV; gated on stabilize, like the
        // CPU/paused-still path). `src_h` is the working-res fisheye height.
        let rs_rows_f32: Option<Vec<f32>> = if control.settings.read().stabilize {
            dji_osv_imu.as_ref().and_then(|osv| {
                vr180_pipeline::dji_imu::compute_per_row_quaternions_for_frame(
                    osv, stab_idx,
                    vr180_pipeline::dji_imu::dji_osmo_readout_ms_for_fps(fps) / 1000.0,
                    src_h, fps,
                )
            }).map(|q| {
                let lens_a = dji_osv_imu.as_ref()
                    .and_then(|osv| osv.lens_a.mount_quat_xyzw)
                    .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
                vr180_pipeline::dji_imu::pack_per_row_camera_matrices(&q, lens_a)
            })
        } else {
            None
        };

        let view_adjust = {
            let s = control.settings.read();
            vr180_pipeline::panomap::ViewAdjust {
                pano_yaw_deg: s.pano_yaw_deg, pano_pitch_deg: s.pano_pitch_deg, pano_roll_deg: s.pano_roll_deg,
                stereo_yaw_deg: s.stereo_yaw_deg, stereo_pitch_deg: s.stereo_pitch_deg, stereo_roll_deg: s.stereo_roll_deg,
                upside_down: s.camera_upside_down,
            }
        };
        let (rot_left, rot_right) = if view_adjust.is_identity() {
            (rot, rot)
        } else {
            let (v_l, v_r) = view_adjust.per_eye_matrices();
            (
                EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_l)),
                EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &v_r)),
            )
        };

        let (left_tex, right_tex) = {
            let mode = control.settings.read().fisheye_output_mode;
            let (ow, oh) = match mode {
                FisheyeOutputMode::HalfEquirect => (eye_w, eye_h),
                FisheyeOutputMode::Fisheye => { let side = eye_w.min(eye_h); (side, side) }
            };
            match mode {
                FisheyeOutputMode::Fisheye => {
                    if let Some(rs) = rs_rows_f32.as_deref() {
                        let l = pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                            &l_tex, src_w, src_h, ow, oh, rot_left, calib_left, rs, 0)?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_fisheye_rs_16(
                            &r_tex, src_w, src_h, ow, oh, rot_right, calib_right, rs, 1)?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                            &l_tex, src_w, src_h, ow, oh, rot_left, calib_left, 0)?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_fisheye_16(
                            &r_tex, src_w, src_h, ow, oh, rot_right, calib_right, 1)?;
                        (l, r)
                    }
                }
                FisheyeOutputMode::HalfEquirect => {
                    if let Some(rs) = rs_rows_f32.as_deref() {
                        let l = pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                            &l_tex, src_w, src_h, ow, oh, rot_left, calib_left, rs, 0)?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_equirect_rs_16(
                            &r_tex, src_w, src_h, ow, oh, rot_right, calib_right, rs, 1)?;
                        (l, r)
                    } else {
                        let l = pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                            &l_tex, src_w, src_h, ow, oh, rot_left, calib_left, 0)?;
                        let r = pipeline.project_fisheye_rgba16_texture_to_equirect_16(
                            &r_tex, src_w, src_h, ow, oh, rot_right, calib_right, 1)?;
                        (l, r)
                    }
                }
            }
        };
        let (pe_w, pe_h) = (left_tex.width(), left_tex.height());
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex, pe_w, pe_h,
        )?;

        if frame_idx < 10 || frame_idx % 60 == 0 {
            tracing::info!(
                "perf(vt-zc) f={:>4} wait={:>5}µs render={:>5}µs budget@{:.0}fps={:.0}µs (dec={}× native {}×{})",
                frame_idx, decode_us, phase_t0.elapsed().as_micros(),
                preview_fps, 1_000_000.0 / preview_fps, preview_decimation, native_w, native_h,
            );
        }

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w, height: out_h,
            frame_idx: absolute_frame_idx,
            // Report the SHOWN frame's own time (its pts, via stab_idx), NOT the
            // pacing clock: render-bound frame-skips drift `frame_t_abs` ahead of
            // the frame actually on screen, and the zoom still (DetailCache) is
            // keyed on this timestamp — a drift makes it decode a DIFFERENT frame.
            timestamp_s: absolute_frame_idx as f64 * dt,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;
        let now = start_wall.unwrap().elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        if !control.paused.load(Ordering::SeqCst) { frame_idx += 1; }
        last_iter_end = std::time::Instant::now();
    }
    Ok(())
}

/// Upload an RGBA8 pixel buffer into a Rgba8Unorm texture with
/// TEXTURE_BINDING + COPY_SRC + STORAGE_BINDING, suitable for the
/// preview-mode composer.
fn upload_rgba8_for_preview(
    pipeline: &vr180_pipeline::gpu::Device,
    rgba: &[u8],
    w: u32, h: u32,
    label: &str,
) -> anyhow::Result<wgpu::Texture> {
    let tex = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    pipeline.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    Ok(tex)
}

fn _check_pair_dims(
    pair: &vr180_pipeline::fisheye_decode::FisheyePair,
    src_w: u32,
    src_h: u32,
) -> bool {
    // Debug-only sanity check — Python would have crashed loudly if
    // the iterator returned the wrong size; Rust just logs it.
    if pair.eye_w != src_w || pair.eye_h != src_h {
        tracing::warn!("decoder (fisheye): pair dims drift ({},{}) vs ({},{})",
            pair.eye_w, pair.eye_h, src_w, src_h);
        false
    } else {
        true
    }
}

/// Build per-eye `FisheyeCalib`s for the current settings + source +
/// working resolution. Called once at decoder start, and again on
/// every `settings_generation` bump.
///
/// Returns `(calib_left, calib_right)`. For non-DJI sources both are
/// identical. For DJI OSV, when a parsed protobuf is supplied, each
/// eye gets its FULL per-lens factory calibration from the protobuf:
/// fx/fy (→FOV), principal point (cx/cy), AND the KB k1–k4 distortion.
/// The file's KB polynomial folds past ~89°, so it is valid only out
/// to the 180°-VR180 edge (max sampled θ = 90°); for full-FOV output
/// past 180° prefer the preset/override. Anything the file omits
/// falls back to the preset.
///
/// Eye→lens mapping: DJI's stream-to-lens layout is stream 0 = Lens A
/// = right eye, stream 1 = Lens B = left eye, AND we swap_eyes at
/// the iter (see DualStreamFisheyeIter::new_with_swap). So after the
/// swap: left output = Lens B, right output = Lens A.
pub(crate) fn resolve_fisheye_calib_pair(
    s: &Settings,
    kind: vr180_pipeline::SourceKind,
    src_w: u32,
    src_h: u32,
    osv: Option<&vr180_fisheye::DjiOsvImu>,
) -> (vr180_pipeline::gpu::FisheyeCalib, vr180_pipeline::gpu::FisheyeCalib) {
    use vr180_fisheye::presets;

    // Pick a preset. Explicit name wins; else fall back to the
    // source-kind default.
    let preset = if !s.fisheye_preset.is_empty() && s.fisheye_preset != "Auto" {
        presets::find(&s.fisheye_preset)
    } else {
        None
    }
    .unwrap_or_else(|| {
        let auto_name = match kind {
            vr180_pipeline::SourceKind::DjiOsv         => "DJI Osmo 360",
            vr180_pipeline::SourceKind::BlackmagicRaw  => "Blackmagic Pyxis 12K",
            _                                          => "Custom",
        };
        presets::find(auto_name).unwrap_or(&presets::presets()[7])
    });

    // Preset-derived auto fx (used when override is off and the
    // protobuf doesn't supply fx, or for non-OSV sources).
    let fx_auto = if preset.calib.fx > 0.0 && preset.calib.calib_w > 0 {
        (preset.calib.fx as f32) * (src_w as f32) / (preset.calib.calib_w as f32)
    } else {
        let half = (preset.default_fov_deg as f32).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };
    let preset_k = [preset.calib.k[0] as f32, preset.calib.k[1] as f32,
                    preset.calib.k[2] as f32, preset.calib.k[3] as f32];
    // Manual fx from a per-eye FOV (degrees). fx = w / (2·half_fov).
    let fx_from_fov = |fov_deg: f32| -> f32 {
        let half = fov_deg.max(1.0).to_radians() * 0.5;
        (src_w as f32) / (2.0 * half)
    };

    // For OSV: per-lens protobuf fx/fy/cx/cy + pure-KB projection. After
    // the DJI iter's default swap: left = Lens B, right = Lens A.
    let (calib_l, calib_r) = match (kind, osv) {
        (vr180_pipeline::SourceKind::DjiOsv, Some(imu)) => {
            let scale_x = imu.lens_b.width.map(|w| (src_w as f32) / w).unwrap_or(1.0);
            let scale_y = imu.lens_b.height.map(|h| (src_h as f32) / h).unwrap_or(1.0);
            let eye = |lens: &vr180_fisheye::DjiLensCalib,
                       ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4], km5: f32,
                       km_p: [f32; 2]|
                -> vr180_pipeline::gpu::FisheyeCalib
            {
                let (fx, fy, cx, cy, k, k5) = if ov {
                    // Manual: fx from FOV, absolute cx/cy from normalized,
                    // manual k1–k4 (preset if all-zero) + manual k5 (km5).
                    let fx = fx_from_fov(fov);
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx, fx, cxn * src_w as f32, cyn * src_h as f32, k, km5)
                } else {
                    // Auto: load the per-lens FACTORY calibration from the
                    // OSV protobuf — principal point (cx/cy), focal length
                    // (fx/fy → FOV) AND the KB k1–k4 distortion. fx/fy/cx/cy
                    // are stored at the calib resolution (lens.width/height),
                    // so scale to the working res. k is dimensionless (no
                    // scale). Anything the file omits falls back to the
                    // preset (fx_auto / preset_k).
                    //
                    // FOV note: the optical FOV is implied by fx (theta_max =
                    // max_r/fx in new_pure_kb), NOT by the file's nominal
                    // "half_fov" field — that reads ~90° and does NOT match
                    // fx·image-circle (~106° half). So loading fx loads FOV.
                    //
                    // k-fold note: this lens's KB polynomial peaks near θ≈89°
                    // then folds, so it is valid only out to ~88°. For 180°
                    // VR180 output (max sampled θ = 90°) we sit right at that
                    // edge, where the factory k is both safe and more accurate
                    // than the smoothed preset. Output past 180° would sample
                    // the folded region — use the preset/override there.
                    let cx = lens.cx.map(|v| v * scale_x).unwrap_or(src_w as f32 * 0.5);
                    let cy = lens.cy.map(|v| v * scale_y).unwrap_or(src_h as f32 * 0.5);
                    let fx = lens.fx.map(|v| v * scale_x).unwrap_or(fx_auto);
                    let fy = lens.fy.map(|v| v * scale_y).unwrap_or(fx);
                    let k = match (lens.k1, lens.k2, lens.k3, lens.k4) {
                        (Some(a), Some(b), Some(c), Some(d)) => [a, b, c, d],
                        _ => preset_k,
                    };
                    // k5 (protobuf field 15) — DJI's 5th radial KB coeff.
                    // Keeps the projection monotonic past ~90° out to the full
                    // lens FOV. 0 when the file omits it (4-coeff fallback).
                    (fx, fy, cx, cy, k, lens.k5.unwrap_or(0.0))
                };
                let mut calib = vr180_pipeline::gpu::FisheyeCalib::new_pure_kb(
                    fx, fy, cx, cy, k, src_w as f32, src_h as f32,
                );
                calib.k5 = k5;
                // Brown-Conrady tangential (field 20). Auto loads it per-lens
                // from the file; override uses the manual [p1, p2] (km_p),
                // seeded from the file so toggling override doesn't drop it.
                calib.p1 = if ov { km_p[0] } else { lens.p1.unwrap_or(0.0) };
                calib.p2 = if ov { km_p[1] } else { lens.p2.unwrap_or(0.0) };
                calib
            };
            // The factory calib must follow the STREAM, not the output
            // slot: with the user swap on, the left output carries Lens A,
            // so it must be dewarped with Lens A's calib (and vice versa).
            let (lens_l, lens_r) = if s.effective_swap_eyes() {
                (&imu.lens_a, &imu.lens_b)
            } else {
                (&imu.lens_b, &imu.lens_a)
            };
            (
                eye(lens_l, s.fisheye_override_left, s.fisheye_fov_deg_left,
                    s.fisheye_cx_norm_left, s.fisheye_cy_norm_left, s.fisheye_k_left,
                    s.fisheye_k5_left, s.fisheye_p_left),
                eye(lens_r, s.fisheye_override_right, s.fisheye_fov_deg_right,
                    s.fisheye_cx_norm_right, s.fisheye_cy_norm_right, s.fisheye_k_right,
                    s.fisheye_k5_right, s.fisheye_p_right),
            )
        }
        _ => {
            // Non-OSV: Hermite-default constructor, per-eye. No protobuf,
            // so "auto" = preset; override = manual fov/cx/cy/k.
            let r_max = (src_w.min(src_h) as f32) * 0.5;
            let eye = |ov: bool, fov: f32, cxn: f32, cyn: f32, km: [f32; 4]|
                -> vr180_pipeline::gpu::FisheyeCalib
            {
                let (fx, cx, cy, k) = if ov {
                    let k = if km.iter().any(|c| c.abs() > 1e-9) { km } else { preset_k };
                    (fx_from_fov(fov), cxn * src_w as f32, cyn * src_h as f32, k)
                } else {
                    (fx_auto, src_w as f32 * 0.5, src_h as f32 * 0.5, preset_k)
                };
                vr180_pipeline::gpu::FisheyeCalib::new(
                    fx, fx, cx, cy, k, src_w as f32, src_h as f32, r_max,
                )
            };
            (
                eye(s.fisheye_override_left, s.fisheye_fov_deg_left,
                    s.fisheye_cx_norm_left, s.fisheye_cy_norm_left, s.fisheye_k_left),
                eye(s.fisheye_override_right, s.fisheye_fov_deg_right,
                    s.fisheye_cx_norm_right, s.fisheye_cy_norm_right, s.fisheye_k_right),
            )
        }
    };

    // Equidistant FISHEYE output target: set the output half-FOV so the
    // fisheye shaders map the inscribed circle to a clean normalized
    // fisheye (matches the export path). Equirect keeps the default 90°.
    if matches!(s.fisheye_output_mode, FisheyeOutputMode::Fisheye) {
        let hfov = (vr180_pipeline::fisheye_export::FISHEYE_OUT_FULL_FOV_DEG * 0.5).to_radians();
        (calib_l.with_output_hfov(hfov), calib_r.with_output_hfov(hfov))
    } else {
        (calib_l, calib_r)
    }
}

/// Per-eye ".360 Lens calibration" Override → shader re-dewarp params.
/// The FACTORY model comes from the file's GEOC atom (left eye = Lens B =
/// BACK, right eye = Lens A = FRNT — same mapping as the RS path); the
/// USER model from the same per-eye `fisheye_*` Override fields the OSV
/// panel uses (independent per source kind via the settings-by-kind map).
/// Override-off eyes / missing GEOC → the disabled sentinel (shader warp
/// skipped entirely). Must mirror `fisheye_export::resolve_eac_lens_pair`
/// so preview == export.
pub(crate) fn resolve_eac_lens_pair(
    s: &Settings,
    geoc: Option<&vr180_core::geoc::Geoc>,
) -> (vr180_pipeline::gpu::EacLensAdjust, vr180_pipeline::gpu::EacLensAdjust) {
    use vr180_pipeline::gpu::EacLensAdjust;
    let Some(g) = geoc else {
        return (EacLensAdjust::DISABLED, EacLensAdjust::DISABLED);
    };
    let cal_dim = g.cal_dim as f32;
    let mk = |cal: Option<&vr180_core::geoc::LensCal>, ov: bool,
              fov: f32, cxn: f32, cyn: f32, k: [f32; 4]| -> EacLensAdjust {
        match (cal, ov) {
            (Some(c), true) => EacLensAdjust::from_geoc_override(
                c.klns, c.ctrx, c.ctry, cal_dim, fov, cxn, cyn, k),
            _ => EacLensAdjust::DISABLED,
        }
    };
    (
        mk(g.back.as_ref(), s.fisheye_override_left, s.fisheye_fov_deg_left,
           s.fisheye_cx_norm_left, s.fisheye_cy_norm_left, s.fisheye_k_left),
        mk(g.front.as_ref(), s.fisheye_override_right, s.fisheye_fov_deg_right,
           s.fisheye_cx_norm_right, s.fisheye_cy_norm_right, s.fisheye_k_right),
    )
}

/// Publish the GEOC factory lens calibration as the `.360` clip's
/// detected calib, in the SAME per-eye seed units the OSV panel uses
/// (fov via the equidistant fov↔fx convention with `cal_dim` as the
/// width — feeding it back through `from_geoc_override` reproduces
/// c0 exactly, i.e. an identity warp; cx/cy normalized to cal_dim;
/// dimensionless KB k). Drives the panel's in-file display + the
/// Override-on seeding + per-clip reseeding, identical to OSV.
fn seed_eac_detected_calib(control: &DecoderControl, geoc: &vr180_core::geoc::Geoc) {
    let cal_dim = geoc.cal_dim as f32;
    if !(cal_dim > 1.0) { return; }
    let seed = |cal: &vr180_core::geoc::LensCal| -> EyeCalibSeed {
        let c0 = (cal.klns[0] as f32).max(1e-3);
        EyeCalibSeed {
            fov_deg: (cal_dim / c0).to_degrees(),
            cx_norm: 0.5 + cal.ctrx / cal_dim,
            cy_norm: 0.5 + cal.ctry / cal_dim,
            k: [
                (cal.klns[1] as f32) / c0,
                (cal.klns[2] as f32) / c0,
                (cal.klns[3] as f32) / c0,
                (cal.klns[4] as f32) / c0,
            ],
            k5: 0.0,
            p: [0.0, 0.0],
        }
    };
    // Left eye = BACK lens, right eye = FRNT (post-yaw-mod mapping).
    if let (Some(back), Some(front)) = (geoc.back.as_ref(), geoc.front.as_ref()) {
        let calib = DetectedLensCalib { left: seed(back), right: seed(front) };
        tracing::info!(
            "decoder (eac): GEOC lens calib published — L: fov={:.2}° cx={:.4} cy={:.4}, R: fov={:.2}° cx={:.4} cy={:.4}",
            calib.left.fov_deg, calib.left.cx_norm, calib.left.cy_norm,
            calib.right.fov_deg, calib.right.cx_norm, calib.right.cy_norm,
        );
        *control.detected_calib.lock() = Some(calib);
    }
}

/// Project an already-assembled native EAC cross into a full-resolution
/// SBS still for the zoom magnifier, using the SAME stab/RS/view-adjust/
/// color/compose path as the live preview — just at `detail_eye` output
/// resolution. Called only from the paused branch of `run_zero_copy`, so
/// it never interacts with playback pacing or frame advance.
#[cfg(target_os = "macos")]
fn render_zoom_detail(
    pipeline: &Device,
    control: &DecoderControl,
    per_eye: &[((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))],
    cross: &(u32, wgpu::Texture, wgpu::Texture),
    abs_frame_idx: u32,
    timestamp_s: f64,
    detail_eye: u32,
    geoc: Option<&vr180_core::geoc::Geoc>,
    tx: &Sender<DecodedFrame>,
) -> anyhow::Result<()> {
    let (_, cross_a, cross_b) = cross;
    let ((rl, sl), (rr, sr)) = per_eye.get(abs_frame_idx as usize).copied()
        .unwrap_or((
            (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
        ));
    let s = control.settings.read();
    let fisheye_out = matches!(s.fisheye_output_mode, FisheyeOutputMode::Fisheye);
    let (rl, rr) = apply_view_adjust(&s, rl, rr);
    let (lens_l, lens_r) = resolve_eac_lens_pair(&s, geoc);
    let (dl, dr) = if fisheye_out {
        (pipeline.project_cross_texture_to_fisheye_texture(cross_b, detail_eye, detail_eye, rl, sl, &lens_l)?,
         pipeline.project_cross_texture_to_fisheye_texture(cross_a, detail_eye, detail_eye, rr, sr, &lens_r)?)
    } else {
        (pipeline.project_cross_texture_to_equirect_texture(cross_b, detail_eye, detail_eye, rl, sl, &lens_l)?,
         pipeline.project_cross_texture_to_equirect_texture(cross_a, detail_eye, detail_eye, rr, sr, &lens_r)?)
    };
    let (dsbs, dw, dh) = compose_with_color_and_mode(
        pipeline, &s, &dl, &dr, detail_eye, detail_eye)?;
    let _ = tx.try_send(DecodedFrame {
        texture: Arc::new(dsbs), width: dw, height: dh,
        frame_idx: abs_frame_idx, timestamp_s,
    });
    Ok(())
}

/// macOS zero-copy path. Decode → IOSurface → wgpu::Texture → GPU
/// EAC assembly → GPU equirect projection → GPU SBS compose. No CPU
/// readback, no swscale, no IPC.
#[cfg(target_os = "macos")]
fn run_zero_copy(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut per_eye: Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>,
    mut stab_key: StabKey,
    mut cached_gen: u64,
    geoc: Option<vr180_core::geoc::Geoc>,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    let mut decoder = vr180_pipeline::decode::SegmentedZeroCopyPairIter::new(&cfg.segments, 0)
        .map_err(|e| { tracing::error!("decoder (zc): iter init failed: {e}"); e })?;
    let dims = decoder.dims();
    tracing::info!("decoder (zc): iter ready, EAC dims tile_w={}, cross_w={}",
        dims.tile_w(), dims.cross_w());
    if !dims.is_valid() {
        anyhow::bail!("invalid EAC layout (stream w={})", dims.stream_w);
    }

    // segment-relative frame index (resets to 0 on Seek / trim loop)
    let mut frame_idx: u32 = 0;
    // clip-absolute time at frame_idx = 0 (updated on Seek / trim loop)
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut rendered_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    // After any seek (Seek command OR trim-out loop), this flag is
    // set so the next iteration bypasses the pause check exactly
    // once and renders a fresh frame at the new position. Otherwise
    // paused + click-on-timeline would seek silently — the displayed
    // frame stays at the pre-seek timestamp until the user unpaused.
    let mut force_render_next = false;
    let _ = fps;

    let total_frames = if cfg.segments.len() > 1 {
        segments_total_frames(&cfg.segments)
    } else {
        (cfg.path.exists()
            .then(|| vr180_pipeline::decode::probe_video(&cfg.path).ok())
            .flatten().map(|p| (p.duration_sec * p.fps as f64).round() as usize)
            .unwrap_or(0)) as usize
    };

    // Initial seek to trim_in, if set.
    {
        let s = control.settings.read();
        if let Some(t_in) = s.trim_in_s {
            if t_in > 0.001 {
                tracing::info!("decoder (zc): initial seek to trim_in = {:.3}s", t_in);
                drop(s);
                decoder.seek(t_in)?;
                time_offset = t_in;
            }
        }
    }

    // Hold-on-pause: the pair we last pulled stays in hand while parked,
    // so a live settings change (View-adjust / stab / color) re-renders
    // the SAME frame instead of advancing. `frame_idx` always indexes
    // the frame currently in `held_pair`. Mirrors the OSV decoder's
    // `stay_on_pair` mechanism.
    let mut held_pair = None;
    // Cached (frame_idx, cross_a, cross_b) so a paused re-render reuses
    // the assembled crosses instead of rebuilding two cross_w² textures.
    let mut cached_cross: Option<(u32, wgpu::Texture, wgpu::Texture)> = None;

    // ── Native-res zoom still (the .360 equivalent of the fisheye
    //    `DetailCache`). When the UI is paused + zoomed it sets
    //    `control.want_detail`; we then ALSO project the already-native
    //    cross at full source resolution and ship it on `detail_tx` for
    //    the magnifier — the live preview keeps its capped working size.
    //    One render per (frame, settings-gen), throttled so a slider drag
    //    while zoomed doesn't pay a native render per pointer event.
    let detail_tx = control.detail_tx.lock().clone();
    // Full-detail per-eye output = the assembled cross resolution, capped
    // so the SBS width (2×) stays within the GPU's max texture dimension.
    let detail_eye = dims.cross_w().min(4096);
    let mut last_detail_key: (u32, u64) = (u32::MAX, u64::MAX);
    let mut last_detail_at: Option<std::time::Instant> = None;
    const DETAIL_THROTTLE: std::time::Duration = std::time::Duration::from_millis(120);

    'main: loop {
        // Pull the next pair ONLY when we don't already hold one. While
        // paused we re-render the held frame (settings update live, frame
        // never advances). Crucially this is also what keeps RESUME aligned:
        // on unpause we still hold the paused frame, so we render IT at its
        // own `frame_idx` and let step 7 advance — instead of immediately
        // pulling the NEXT pair while `frame_idx` still points at the paused
        // frame. The old `if !stay_on_pair` gate did the latter, desyncing
        // the displayed pixels from the per-frame stab/RS by one frame on
        // EVERY pause→play (the offset compounds, so stabilization visibly
        // drifts apart after a few replays until a full decoder restart).
        let stay_on_pair = control.paused.load(Ordering::SeqCst) && held_pair.is_some();
        if held_pair.is_none() {
            match decoder.next_pair(&pipeline.device)? {
                Some(p) => held_pair = Some(p),
                None => break,
            }
        }

        // ── 1. Drain command queue. Seek wins over Stop within
        //       a single batch; subsequent Seeks overwrite earlier ones.
        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            held_pair = None;
            decoder.seek(t)?;
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            tracing::debug!("decoder (zc): seek → {:.3}s", t);
            continue 'main;
        }

        // ── 2. Trim-out loop. If we've played past trim_out, jump
        //       back to trim_in (or 0). Uses absolute clip time, not
        //       segment-relative.
        let frame_t_rel = frame_idx as f64 * dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    held_pair = None;
                    decoder.seek(in_t)?;
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    tracing::debug!("decoder (zc): trim loop → {:.3}s", in_t);
                    continue 'main;
                }
            }
        }

        // ── 3. Settings changed → rebuild per-eye bundle. Done before
        //       render so a live change reflects in this frame.
        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snapshot = control.settings.read().clone();
            let new_key = stab_settings_key(&snapshot);
            if new_key == stab_key {
                // Settings changed but not the stab inputs (view-adjust /
                // color / LUT / preview-mode) — keep the bundle, just
                // re-render. This is what keeps those sliders snappy.
                cached_gen = current_gen;
            } else {
                match build_per_eye_frames_multi(&cfg.segments, &snapshot, fps, total_frames) {
                    Ok(new) => {
                        per_eye = new;
                        stab_key = new_key;
                        cached_gen = current_gen;
                        tracing::debug!("decoder (zc): per-eye bundles rebuilt @ gen {}", current_gen);
                    }
                    Err(e) => tracing::warn!("decoder (zc): per-eye rebuild failed: {e}"),
                }
            }
        }

        // ── 4. Wall-clock pacing + behind-real-time skip (playing only;
        //       a held paused frame and a post-seek frame never skip).
        if !stay_on_pair && !force_render_next {
            let wall_t = start_wall
                .get_or_insert_with(std::time::Instant::now)
                .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
            if wall_t > frame_t_rel + dt * 0.5 {
                frame_idx += 1;
                skipped_count += 1;
                if skipped_count % 30 == 0 {
                    tracing::debug!("decoder (zc): behind by {:.1} ms — skipped {}",
                        (wall_t - frame_t_rel) * 1000.0, skipped_count);
                }
                if wall_t > frame_t_rel + 1.0 {
                    start_wall = Some(std::time::Instant::now()
                        - std::time::Duration::from_secs_f64(frame_t_rel)
                        - paused_offset);
                }
                held_pair = None;
                continue 'main;
            }
        }

        // ── 5. GPU render.
        // We look up per-eye stab/RS by clip-absolute frame index so
        // that the rotation matrices stay tied to source time after
        // a seek (not segment-relative time).
        let pair = held_pair.as_ref().expect("held_pair populated above");
        let absolute_frame_idx = ((time_offset / dt).round() as u32) + frame_idx;

        // Assemble the two EAC crosses — the EAC-specific cost OSV
        // doesn't pay (2 × cross_w² P010→RGB writes). Cache them per
        // SOURCE frame: when re-rendering the SAME frame (paused while
        // the user drags a View-adjust / color / LUT slider), the source
        // pixels are unchanged, so reuse the cached crosses and only
        // re-project + re-compose. This is what makes paused slider drags
        // on `.360` as snappy as OSV (which has no cross to rebuild).
        let need_cross = cached_cross.as_ref()
            .map(|(i, _, _)| *i != absolute_frame_idx).unwrap_or(true);
        if need_cross {
            let a = pipeline.nv12_to_eac_cross(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::A, dims)?;
            let b = pipeline.nv12_to_eac_cross(
                &pair.s0_y.texture, &pair.s0_uv.texture,
                &pair.s4_y.texture, &pair.s4_uv.texture, Lens::B, dims)?;
            cached_cross = Some((absolute_frame_idx, a, b));
        }
        let (_, cross_a, cross_b) = cached_cross.as_ref().unwrap();

        let ((rl, sl), (rr, sr)) = per_eye.get(absolute_frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));
        // Compose the View-adjustment (pano-map + stereo) onto the stab
        // rotations — same as OSV + export_eac, so the sliders work live.
        let fisheye_out = matches!(
            control.settings.read().fisheye_output_mode,
            FisheyeOutputMode::Fisheye);
        let (rl, rr) = apply_view_adjust(&control.settings.read(), rl, rr);
        // Per-eye ".360 Lens calibration" Override (disabled sentinels when
        // off). Resolved from live settings each frame so slider drags
        // re-project immediately (the gen bump re-renders a paused frame).
        let (lens_l, lens_r) = resolve_eac_lens_pair(
            &control.settings.read(), geoc.as_ref());

        let (left_tex, right_tex) = if fisheye_out {
            (pipeline.project_cross_texture_to_fisheye_texture(cross_b, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_texture_to_fisheye_texture(cross_a, eye_w, eye_h, rr, sr, &lens_r)?)
        } else {
            (pipeline.project_cross_texture_to_equirect_texture(cross_b, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_texture_to_equirect_texture(cross_a, eye_w, eye_h, rr, sr, &lens_r)?)
        };
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex, eye_w, eye_h,
        )?;

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w,
            height: out_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {
                rendered_count += 1;
                if rendered_count == 1 {
                    tracing::info!("decoder (zc): first frame rendered + sent");
                } else if rendered_count % 30 == 0 {
                    tracing::debug!("decoder (zc): rendered {} frames", rendered_count);
                }
            }
            Err(TrySendError::Full(_)) => { /* UI behind; drop newest */ }
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        // We rendered + shipped one frame; clear the post-seek override.
        force_render_next = false;

        // ── 6. Pause handling. We've just shown the current frame; now
        //       park if paused. Breaking on a settings-gen change loops
        //       back with `stay_on_pair` true, which re-renders the SAME
        //       held frame with the new settings — sliders update live
        //       while paused and the frame never creeps forward.
        if control.paused.load(Ordering::SeqCst) {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true;
                    break;
                }
                // Native-res zoom still for the magnifier. While paused +
                // zoomed the UI sets `want_detail`; render it HERE, fully
                // inside the pause path, reusing the cross already assembled
                // for this frame. The playback loop (pacing / frame advance /
                // `force_render_next`) is NEVER touched by the zoom feature,
                // so it cannot desync stabilization on resume. One render per
                // (frame, settings-gen), throttled so a zoomed slider-drag
                // stays fluid. Best-effort: a detail-render error is logged,
                // never propagated (it must not kill playback).
                if let Some(tx) = detail_tx.as_ref() {
                    if control.want_detail.load(Ordering::SeqCst) {
                        let abs = ((time_offset / dt).round() as u32) + frame_idx;
                        let throttle_ok = last_detail_at
                            .map(|t| t.elapsed() >= DETAIL_THROTTLE).unwrap_or(true);
                        let have_cross = cached_cross.as_ref()
                            .map(|(i, _, _)| *i == abs).unwrap_or(false);
                        if have_cross && (abs, cached_gen) != last_detail_key && throttle_ok {
                            if let Err(e) = render_zoom_detail(
                                &pipeline, &control, &per_eye,
                                cached_cross.as_ref().unwrap(), abs,
                                time_offset + frame_idx as f64 * dt, detail_eye,
                                geoc.as_ref(), tx)
                            {
                                tracing::debug!("decoder (zc): zoom detail failed: {e}");
                            }
                            last_detail_key = (abs, cached_gen);
                            last_detail_at = Some(std::time::Instant::now());
                        }
                    }
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            held_pair = None;
                            decoder.seek(t.max(0.0))?;
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
            continue 'main;
        }

        // ── 7. Playing: pace to this frame's wall-clock time, then
        //       advance. Dropping `held_pair` makes the next iteration
        //       pull the next pair (frame_idx already bumped).
        let now = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
        held_pair = None;
    }
    tracing::info!("decoder (zc): finished after {} segment frames ({} skipped)",
        frame_idx, skipped_count);
    Ok(())
}

/// Non-macOS fallback — CPU EAC assembly path.
#[cfg(not(target_os = "macos"))]
/// Windows zero-copy EAC preview (GoPro `.360`) — the GPU-resident sibling of
/// [`run_cpu_assemble`], and the Windows counterpart of the macOS
/// [`run_zero_copy`]. Decode + EAC cross assembly stay on the GPU: NVDEC
/// (`d3d11va`) decodes both EAC HEVC streams, each P010 is converted to a
/// single-plane Rgba16Unorm on the D3D11 side and shared into wgpu's Vulkan
/// device (zero-copy alias), then `rgba8_to_eac_cross` assembles each lens's
/// cross on the GPU and `project_cross_texture_to_equirect_texture` projects it
/// — no CPU download, no swscale, no CPU `assemble_lens_*`. Identical conversion
/// math + cross geometry to the macOS zero-copy path. All pacing / pause-resume
/// / seek / trim / live-stab / live-view-adjust semantics mirror
/// `run_cpu_assemble` exactly; only the decode + assemble change.
#[cfg(target_os = "windows")]
#[allow(clippy::too_many_arguments)]
fn run_eac_zerocopy(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut per_eye: Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>,
    mut stab_key: StabKey,
    mut cached_gen: u64,
    geoc: Option<vr180_core::geoc::Geoc>,
    ctx: vr180_pipeline::interop_windows::VulkanImportCtx,
    iter: vr180_pipeline::fisheye_decode::SegmentedD3d11SharedStreamPairIter,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    use vr180_pipeline::fisheye_decode::SharedEacPair;
    use vr180_pipeline::gpu::Lens;

    let mut iter = iter;
    let dims = iter.dims();
    if !dims.is_valid() {
        anyhow::bail!("invalid EAC layout (stream w={})", dims.stream_w);
    }

    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let _ = fps;
    let total_frames = vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0);

    {
        let s = control.settings.read();
        if let Some(t_in) = s.trim_in_s {
            if t_in > 0.001 { drop(s); iter.seek(t_in)?; time_offset = t_in; }
        }
    }

    let mut held_pair: Option<SharedEacPair> = None;

    'main: loop {
        let stay_on_pair = control.paused.load(Ordering::SeqCst) && held_pair.is_some();
        if held_pair.is_none() {
            match iter.next_pair()? {
                Some(p) => held_pair = Some(p),
                None => break,
            }
        }

        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            held_pair = None;
            iter.seek(t)?;
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            continue 'main;
        }

        let frame_t_rel = frame_idx as f64 * dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    held_pair = None;
                    iter.seek(in_t)?;
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    continue 'main;
                }
            }
        }

        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snapshot = control.settings.read().clone();
            let new_key = stab_settings_key(&snapshot);
            if new_key == stab_key {
                cached_gen = current_gen;
            } else {
                match build_per_eye_frames_multi(&cfg.segments, &snapshot, fps, total_frames) {
                    Ok(new) => { per_eye = new; stab_key = new_key; cached_gen = current_gen; }
                    Err(e) => tracing::warn!("decoder (eac/zc): per-eye rebuild failed: {e}"),
                }
            }
        }

        if !stay_on_pair && !force_render_next {
            let wall_t = start_wall
                .get_or_insert_with(std::time::Instant::now)
                .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
            if wall_t > frame_t_rel + dt * 0.5 {
                frame_idx += 1;
                skipped_count += 1;
                if wall_t > frame_t_rel + 1.0 {
                    start_wall = Some(std::time::Instant::now()
                        - std::time::Duration::from_secs_f64(frame_t_rel)
                        - paused_offset);
                }
                if skipped_count % 30 == 0 {
                    tracing::debug!("decoder (eac/zc): behind by {:.1} ms — skipped {}",
                        (wall_t - frame_t_rel) * 1000.0, skipped_count);
                }
                held_pair = None;
                continue 'main;
            }
        }

        let pair = held_pair.as_ref().expect("held_pair populated above");
        let absolute_frame_idx = ((time_offset / dt).round() as u32) + frame_idx;

        // GPU EAC cross assembly from the two zero-copy stream textures.
        let s0 = unsafe { ctx.import_rgba16(&pipeline.device, &pair.s0) };
        let s4 = unsafe { ctx.import_rgba16(&pipeline.device, &pair.s4) };
        let cross_a = pipeline.rgba8_to_eac_cross(&s0, &s4, Lens::A, dims)?;
        let cross_b = pipeline.rgba8_to_eac_cross(&s0, &s4, Lens::B, dims)?;

        let ((rl, sl), (rr, sr)) = per_eye.get(absolute_frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));
        let (rl, rr) = apply_view_adjust(&control.settings.read(), rl, rr);

        // Left eye = Lens B, right eye = Lens A (same as the CPU + export paths).
        // Honor the Fisheye output mode (normalized fisheye SBS) like the macOS
        // run_zero_copy + the export; default is half-equirect (VR180).
        let (lens_l, lens_r) = resolve_eac_lens_pair(
            &control.settings.read(), geoc.as_ref());
        let (left_tex, right_tex) = if matches!(
            control.settings.read().fisheye_output_mode, FisheyeOutputMode::Fisheye
        ) {
            (pipeline.project_cross_texture_to_fisheye_texture(&cross_b, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_texture_to_fisheye_texture(&cross_a, eye_w, eye_h, rr, sr, &lens_r)?)
        } else {
            (pipeline.project_cross_texture_to_equirect_texture(&cross_b, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_texture_to_equirect_texture(&cross_a, eye_w, eye_h, rr, sr, &lens_r)?)
        };
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex, eye_w, eye_h,
        )?;

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w,
            height: out_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;

        if control.paused.load(Ordering::SeqCst) {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true;
                    break;
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            held_pair = None;
                            iter.seek(t.max(0.0))?;
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
            continue 'main;
        }

        let now = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
        held_pair = None;
    }
    Ok(())
}

fn run_cpu_assemble(
    pipeline: Arc<Device>,
    cfg: &DecoderConfig,
    control: Arc<DecoderControl>,
    fps: f32,
    dt: f64,
    eye_w: u32,
    eye_h: u32,
    mut per_eye: Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>,
    mut stab_key: StabKey,
    mut cached_gen: u64,
    geoc: Option<vr180_core::geoc::Geoc>,
    frame_tx: Sender<DecodedFrame>,
    cmd_rx: Receiver<DecoderCommand>,
) -> anyhow::Result<()> {
    let mut decoder = vr180_pipeline::decode::SegmentedStreamPairIter::new(&cfg.segments, HwDecode::Auto, 0)?;
    let dims = decoder.dims();
    if !dims.is_valid() {
        anyhow::bail!("invalid EAC layout (stream w={})", dims.stream_w);
    }
    let cross_w_px = dims.cross_w();
    let cw = cross_w_px as usize;

    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];

    let mut frame_idx: u32 = 0;
    let mut time_offset: f64 = 0.0;
    let mut skipped_count: u32 = 0;
    let mut start_wall: Option<std::time::Instant> = None;
    let mut paused_offset = std::time::Duration::ZERO;
    let mut force_render_next = false;
    let _ = fps;
    let total_frames = (vr180_pipeline::decode::probe_video(&cfg.path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize)
        .unwrap_or(0)) as usize;

    // Initial seek to trim_in, if set.
    {
        let s = control.settings.read();
        if let Some(t_in) = s.trim_in_s {
            if t_in > 0.001 {
                drop(s);
                decoder.seek(t_in)?;
                time_offset = t_in;
            }
        }
    }

    // Hold-on-pause: mirror the zero-copy decoder. `held_pair` keeps the
    // last-pulled frame in hand while parked so a live settings change
    // re-renders the SAME frame instead of advancing. `frame_idx` always
    // indexes the frame currently in `held_pair`.
    let mut held_pair = None;

    'main: loop {
        // Pull only when we don't already hold a pair — see `run_zero_copy`
        // for why: pulling on resume (while still holding the paused frame)
        // desyncs the EAC frame-index-keyed stab/RS by one frame per replay.
        let stay_on_pair = control.paused.load(Ordering::SeqCst) && held_pair.is_some();
        if held_pair.is_none() {
            match decoder.next_pair()? {
                Some(p) => held_pair = Some(p),
                None => break,
            }
        }

        let mut seek_target: Option<f64> = None;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                DecoderCommand::Stop => return Ok(()),
                DecoderCommand::Seek(t) => seek_target = Some(t.max(0.0)),
            }
        }
        if let Some(t) = seek_target {
            held_pair = None;
            decoder.seek(t)?;
            time_offset = t;
            frame_idx = 0;
            start_wall = None;
            paused_offset = std::time::Duration::ZERO;
            force_render_next = true;
            continue 'main;
        }

        let frame_t_rel = frame_idx as f64 * dt;
        let frame_t_abs = time_offset + frame_t_rel;
        {
            let s = control.settings.read();
            if let Some(out_t) = s.trim_out_s {
                if frame_t_abs >= out_t {
                    let in_t = s.trim_in_s.unwrap_or(0.0);
                    drop(s);
                    held_pair = None;
                    decoder.seek(in_t)?;
                    time_offset = in_t;
                    frame_idx = 0;
                    start_wall = None;
                    paused_offset = std::time::Duration::ZERO;
                    force_render_next = true;
                    continue 'main;
                }
            }
        }

        let current_gen = control.settings_generation.load(Ordering::SeqCst);
        if current_gen != cached_gen {
            let snapshot = control.settings.read().clone();
            let new_key = stab_settings_key(&snapshot);
            if new_key == stab_key {
                cached_gen = current_gen; // non-stab change: re-render only
            } else {
                match build_per_eye_frames_multi(&cfg.segments, &snapshot, fps, total_frames) {
                    Ok(new) => { per_eye = new; stab_key = new_key; cached_gen = current_gen; }
                    Err(e) => tracing::warn!("decoder (cpu): per-eye rebuild failed: {e}"),
                }
            }
        }

        if !stay_on_pair && !force_render_next {
            let wall_t = start_wall
                .get_or_insert_with(std::time::Instant::now)
                .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
            if wall_t > frame_t_rel + dt * 0.5 {
                frame_idx += 1;
                skipped_count += 1;
                if wall_t > frame_t_rel + 1.0 {
                    start_wall = Some(std::time::Instant::now()
                        - std::time::Duration::from_secs_f64(frame_t_rel)
                        - paused_offset);
                }
                if skipped_count % 30 == 0 {
                    tracing::debug!("decoder (cpu): behind by {:.1} ms — skipped {}",
                        (wall_t - frame_t_rel) * 1000.0, skipped_count);
                }
                held_pair = None;
                continue 'main;
            }
        }

        let pair = held_pair.as_ref().expect("held_pair populated above");
        let absolute_frame_idx = ((time_offset / dt).round() as u32) + frame_idx;

        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        let ((rl, sl), (rr, sr)) = per_eye.get(absolute_frame_idx as usize).copied()
            .unwrap_or((
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
                (EquirectRotation::IDENTITY, EquirectRsParams::DISABLED),
            ));
        let (rl, rr) = apply_view_adjust(&control.settings.read(), rl, rr);

        // Honor the Fisheye output mode (parity with the zero-copy preview + the
        // export); default is half-equirect (VR180).
        let (lens_l, lens_r) = resolve_eac_lens_pair(
            &control.settings.read(), geoc.as_ref());
        let (left_tex, right_tex) = if matches!(
            control.settings.read().fisheye_output_mode, FisheyeOutputMode::Fisheye
        ) {
            (pipeline.project_cross_to_fisheye_texture(&cross_b, cross_w_px, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_to_fisheye_texture(&cross_a, cross_w_px, eye_w, eye_h, rr, sr, &lens_r)?)
        } else {
            (pipeline.project_cross_to_equirect_texture(&cross_b, cross_w_px, eye_w, eye_h, rl, sl, &lens_l)?,
             pipeline.project_cross_to_equirect_texture(&cross_a, cross_w_px, eye_w, eye_h, rr, sr, &lens_r)?)
        };
        let (sbs_tex, out_w, out_h) = compose_with_color_and_mode(
            &pipeline, &control.settings.read(), &left_tex, &right_tex, eye_w, eye_h,
        )?;

        let out = DecodedFrame {
            texture: Arc::new(sbs_tex),
            width: out_w,
            height: out_h,
            frame_idx: absolute_frame_idx,
            timestamp_s: frame_t_abs,
        };
        match frame_tx.try_send(out) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
        force_render_next = false;

        // Park if paused (after rendering). Breaking on a gen change loops
        // back with `stay_on_pair` true → re-render the held frame live.
        if control.paused.load(Ordering::SeqCst) {
            let pause_start = std::time::Instant::now();
            while control.paused.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(16));
                if control.settings_generation.load(Ordering::SeqCst) != cached_gen {
                    force_render_next = true;
                    break;
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        DecoderCommand::Stop => return Ok(()),
                        DecoderCommand::Seek(t) => {
                            held_pair = None;
                            decoder.seek(t.max(0.0))?;
                            time_offset = t.max(0.0);
                            frame_idx = 0;
                            start_wall = None;
                            paused_offset = std::time::Duration::ZERO;
                            force_render_next = true;
                            continue 'main;
                        }
                    }
                }
            }
            paused_offset += pause_start.elapsed();
            continue 'main;
        }

        let now = start_wall
            .get_or_insert_with(std::time::Instant::now)
            .elapsed().as_secs_f64() - paused_offset.as_secs_f64();
        if now < frame_t_rel {
            std::thread::sleep(std::time::Duration::from_secs_f64(frame_t_rel - now));
        }
        frame_idx += 1;
        held_pair = None;
    }
    Ok(())
}

/// Compose left + right eye textures into the preview output, applying
/// the user's color stack and selected preview mode (SBS / anaglyph /
/// 50% overlay). Returns `(texture, width, height)` of the result.
///
/// Color stack runs only when at least one knob is active — identity
/// (all defaults) takes a zero-cost passthrough.
/// Compose the global pano-map + per-eye stereo view adjustment (the
/// "View adjustment" panel) onto a pair of per-eye rotations. Mirrors the
/// OSV preview + `export_eac`: `R_eye_final = R_eye · R_view_eye`, with
/// the fast identity short-circuit so a default (all-zero) clip is
/// byte-identical to the no-view-adjust path. Used by the GoPro EAC
/// preview so its View-adjustment sliders take effect live.
fn apply_view_adjust(
    s: &Settings,
    rl: EquirectRotation,
    rr: EquirectRotation,
) -> (EquirectRotation, EquirectRotation) {
    let va = vr180_pipeline::panomap::ViewAdjust {
        pano_yaw_deg: s.pano_yaw_deg,
        pano_pitch_deg: s.pano_pitch_deg,
        pano_roll_deg: s.pano_roll_deg,
        stereo_yaw_deg: s.stereo_yaw_deg,
        stereo_pitch_deg: s.stereo_pitch_deg,
        stereo_roll_deg: s.stereo_roll_deg,
        upside_down: s.camera_upside_down,
    };
    if va.is_identity() {
        return (rl, rr);
    }
    let (v_l, v_r) = va.per_eye_matrices();
    (
        EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rl.0, &v_l)),
        EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rr.0, &v_r)),
    )
}

fn compose_with_color_and_mode(
    pipeline: &Device,
    s: &Settings,
    left_tex: &wgpu::Texture,
    right_tex: &wgpu::Texture,
    eye_w: u32,
    eye_h: u32,
) -> anyhow::Result<(wgpu::Texture, u32, u32)> {
    let color_plan = s.build_color_stack();
    let preview_mode = s.preview_mode.to_pipeline();
    // Single-eye: the compose shader (mode 3) outputs the LEFT input. To show
    // the RIGHT eye, swap the bindings so the right eye lands in that slot.
    let (left_tex, right_tex) =
        if preview_mode == vr180_pipeline::gpu::PreviewMode::SingleEye && s.preview_eye_right {
            (right_tex, left_tex)
        } else {
            (left_tex, right_tex)
        };

    // ── No grade: composite the projection outputs directly. ──
    // This is the hot path for ordinary playback — no per-eye color
    // passes, no extra texture allocations. compose_preview only READS
    // the inputs (TEXTURE_BINDING), so we pass them straight through.
    if !color_plan.any_active() {
        if preview_mode == vr180_pipeline::gpu::PreviewMode::Sbs {
            // 8-bit eyes (CPU path): single texture-copy compose —
            // byte-identical to the pre-color-stack pipeline (1 submit, 1
            // alloc). 16-bit eyes (Windows zero-copy P010 path) can't use the
            // copy (it can't convert Rgba16Unorm→Rgba8Unorm), so route them
            // through `compose_preview`, which reads filterable-float and
            // writes the Rgba8Unorm SBS egui displays.
            if left_tex.format() == wgpu::TextureFormat::Rgba8Unorm {
                let sbs = compose_sbs(pipeline, left_tex, right_tex, eye_w, eye_h)?;
                return Ok((sbs, eye_w * 2, eye_h));
            }
            let out = pipeline.compose_preview(left_tex, right_tex, eye_w, eye_h, preview_mode)?;
            let (out_w, out_h) = preview_mode.output_dims(eye_w, eye_h);
            return Ok((out, out_w, out_h));
        }
        // Anaglyph / overlay: one compute dispatch, no extra copies.
        let out = pipeline.compose_preview(
            left_tex, right_tex, eye_w, eye_h, preview_mode,
        )?;
        let (out_w, out_h) = preview_mode.output_dims(eye_w, eye_h);
        return Ok((out, out_w, out_h));
    }

    // ── Grade active: per-eye color stack in 16-BIT, then composite. ──
    // Run the SAME 16-bit color stack the export uses (`apply_color_stack_
    // per_eye_16`) so the preview's CDL / white-balance / LUT / saturation
    // match the export's math — no 8-bit banding introduced by the grade.
    // `compose_preview` reads its inputs as filterable float, so the
    // Rgba16Unorm graded eyes feed it directly and it writes the Rgba8Unorm
    // SBS that egui displays. (The source is still the preview's working-res
    // 8-bit decode, but the grade math now matches the export exactly.)
    let left_g  = pipeline.apply_color_stack_per_eye_16(left_tex,  eye_w, eye_h, &color_plan.for_eye(true))?;
    let right_g = pipeline.apply_color_stack_per_eye_16(right_tex, eye_w, eye_h, &color_plan.for_eye(false))?;
    let left_post  = left_g .as_ref().unwrap_or(left_tex);
    let right_post = right_g.as_ref().unwrap_or(right_tex);
    let sbs_tex = pipeline.compose_preview(
        left_post, right_post, eye_w, eye_h, preview_mode,
    )?;
    let (out_w, out_h) = preview_mode.output_dims(eye_w, eye_h);
    Ok((sbs_tex, out_w, out_h))
}

/// Native-resolution still renderer for the zoom magnifier.
///
/// Split for responsiveness:
///   • A BACKGROUND THREAD owns the native decoder and does the heavy,
///     I/O-bound seek + decode-forward (a whole GOP at 3840² ≈ 1–2 s on the
///     external volume). This is CPU / VideoToolbox ONLY — it never touches
///     the wgpu device, so it can't wedge Metal's drawable queue against the
///     main thread (the deadlock we hit when GPU work ran off-thread). It
///     posts the decoded native pair back and pokes egui to repaint.
///   • The MAIN THREAD does the GPU project + color + compose (~10 ms) once
///     the native pair is ready, keeping GPU work single-party.
///
/// Net: zooming/seeking no longer freezes — the live preview stays
/// interactive while the still streams in, then swaps in when decoded.
/// Stabilization is recomputed only when stab params change.
/// Process-wide cache of parsed DJI OSV IMU, keyed by source path. The
/// DECODER thread (`run_fisheye`) parses it once (off the main thread —
/// extracting the scattered `djmd` samples from a cold file on a network
/// volume is slow) and publishes it here; the main-thread `DetailCache`
/// READS it instead of re-extracting, so opening an OSV never freezes the
/// UI. Bounded to the few most-recent clips so it can't grow unbounded.
fn dji_imu_cache(
) -> &'static parking_lot::Mutex<Vec<(PathBuf, std::sync::Arc<vr180_fisheye::DjiOsvImu>)>> {
    static CACHE: std::sync::OnceLock<
        parking_lot::Mutex<Vec<(PathBuf, std::sync::Arc<vr180_fisheye::DjiOsvImu>)>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| parking_lot::Mutex::new(Vec::new()))
}

pub(crate) fn cache_dji_imu(path: &std::path::Path, imu: std::sync::Arc<vr180_fisheye::DjiOsvImu>) {
    let mut c = dji_imu_cache().lock();
    c.retain(|(p, _)| p != path);
    c.push((path.to_path_buf(), imu));
    let n = c.len();
    if n > 8 { c.drain(0..n - 8); } // keep the 8 most recent
}

pub(crate) fn cached_dji_imu(path: &std::path::Path) -> Option<std::sync::Arc<vr180_fisheye::DjiOsvImu>> {
    dji_imu_cache().lock().iter().find(|(p, _)| p == path).map(|(_, imu)| imu.clone())
}

fn gpmf_cache(
) -> &'static parking_lot::Mutex<Vec<(PathBuf, std::sync::Arc<Vec<u8>>)>> {
    static CACHE: std::sync::OnceLock<
        parking_lot::Mutex<Vec<(PathBuf, std::sync::Arc<Vec<u8>>)>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| parking_lot::Mutex::new(Vec::new()))
}

/// Extract the GoPro GPMF metadata blob, reusing a path-keyed cache so the
/// `load_file` read, the decoder's stab read, and any respawn share ONE disk
/// read of the `gpmd` stream instead of re-extracting it each time (the
/// `.360` analogue of [`cached_dji_imu`]). Preview-only — the export pipeline
/// reads the file fresh and never touches this cache.
pub(crate) fn extract_gpmf_cached(path: &std::path::Path) -> vr180_pipeline::Result<Vec<u8>> {
    if let Some(blob) = gpmf_cache().lock().iter()
        .find(|(p, _)| p == path).map(|(_, b)| b.clone())
    {
        return Ok((*blob).clone());
    }
    let arc = std::sync::Arc::new(vr180_pipeline::decode::extract_gpmf_stream(path)?);
    let mut c = gpmf_cache().lock();
    c.retain(|(p, _)| p != path);
    c.push((path.to_path_buf(), arc.clone()));
    let n = c.len();
    if n > 8 { c.drain(0..n - 8); } // keep the 8 most recent
    Ok((*arc).clone())
}

pub(crate) struct DetailCache {
    path: PathBuf,
    kind: vr180_pipeline::SourceKind,
    fps: f32,
    /// Background decode worker: send a requested timestamp, receive the
    /// decoded native pair `(ts_ms, pair)`.
    req_tx: crossbeam_channel::Sender<f64>,
    res_rx: crossbeam_channel::Receiver<(i64, vr180_pipeline::fisheye_decode::FisheyePair)>,
    _handle: std::thread::JoinHandle<()>,
    /// Latest decoded native pair, held on the main thread.
    cached_ts: i64,
    cached_pair: Option<vr180_pipeline::fisheye_decode::FisheyePair>,
    cached_stab_key: u64,
    cached_stab: Option<Vec<EquirectRotation>>,
}

impl DetailCache {
    pub fn new(
        path: PathBuf,
        kind: vr180_pipeline::SourceKind,
        fps: f32,
        swap_eyes: bool,
        ctx: egui::Context,
    ) -> Self {
        // NOTE: the DJI IMU is NOT extracted here — that read is slow on a
        // cold network volume and `new()` runs on the main thread. The
        // decoder thread publishes it to `dji_imu_cache`; `render()` reads it.
        let (req_tx, req_rx) = crossbeam_channel::bounded::<f64>(1);
        let (res_tx, res_rx) =
            crossbeam_channel::unbounded::<(i64, vr180_pipeline::fisheye_decode::FisheyePair)>();
        let path_w = path.clone();
        let handle = std::thread::spawn(move || {
            detail_decode_worker(path_w, kind, swap_eyes, fps, req_rx, res_tx, ctx);
        });
        Self {
            path, kind, fps,
            req_tx, res_rx, _handle: handle,
            cached_ts: i64::MIN,
            cached_pair: None,
            cached_stab_key: u64::MAX,
            cached_stab: None,
        }
    }

    /// Render the still for `timestamp` under `s`. The native DECODE runs on
    /// the background thread; this only does the (fast) GPU project + color +
    /// compose once the frame is ready. Returns:
    ///   • `Some((tex, w, h))` when the native frame is decoded & projected.
    ///   • `None` when the frame isn't decoded yet — a background decode is
    ///     (re)requested and the caller keeps the live preview up + retries.
    /// `out_cap` bounds the OUTPUT resolution (source is always native);
    /// `0` = native full detail.
    pub fn render(
        &mut self,
        pipeline: &Device,
        timestamp: f64,
        s: &Settings,
        out_cap: u32,
    ) -> Option<(wgpu::Texture, u32, u32)> {
        // Pick up any freshly-decoded native pair from the worker.
        while let Ok((ts_ms, pair)) = self.res_rx.try_recv() {
            self.cached_ts = ts_ms;
            self.cached_pair = Some(pair);
        }
        let ts_ms = (timestamp * 1000.0).round() as i64;
        if self.cached_ts != ts_ms || self.cached_pair.is_none() {
            // Frame not decoded yet → ask the worker. The request channel is
            // bounded(1) + drained-to-latest in the worker, so a fast scrub
            // just moves the target instead of queuing every frame.
            let _ = self.req_tx.try_send(timestamp.max(0.0));
            return None;
        }
        let pair = self.cached_pair.as_ref()?;

        // IMU read from the cache the decoder thread populated (None until
        // it finishes parsing — the still renders un-stabilized until then,
        // which only matters if you zoom during the initial cold load).
        let imu = if matches!(self.kind, vr180_pipeline::SourceKind::DjiOsv) {
            cached_dji_imu(&self.path)
        } else { None };

        // Stabilization — recompute when the stab params change OR when the
        // IMU first becomes available (folded into the key).
        let sk = stab_key(s) ^ imu.as_ref().map_or(0, |i| i.frame_quats.len() as u64);
        if self.cached_stab_key != sk {
            self.cached_stab = compute_stab_for(
                self.kind, &self.path, imu.as_deref(), s, self.fps);
            self.cached_stab_key = sk;
        }

        // Project + color + compose on THIS (main) thread (≈10 ms).
        render_still_from_pair(
            pipeline, self.kind, s, self.fps, pair,
            imu.as_deref(), self.cached_stab.as_deref(), out_cap,
        ).ok().flatten()
    }
}

/// Background decode worker for `DetailCache`. CPU / VideoToolbox only — it
/// never touches the wgpu device, so it can't deadlock the main thread's
/// Metal present. Seeks + decodes-forward to the requested timestamp at
/// native resolution, posts the pair back, and pokes egui to repaint.
fn detail_decode_worker(
    path: PathBuf,
    kind: vr180_pipeline::SourceKind,
    swap_eyes: bool,
    fps: f32,
    req_rx: crossbeam_channel::Receiver<f64>,
    res_tx: crossbeam_channel::Sender<(i64, vr180_pipeline::fisheye_decode::FisheyePair)>,
    ctx: egui::Context,
) {
    let mut iter = match open_native_iter(&path, kind, swap_eyes) {
        Ok(it) => it,
        Err(e) => { tracing::warn!("detail decode: open failed: {e}"); return; }
    };
    let dt = 1.0 / (fps.max(1e-3) as f64);
    loop {
        // Block for one request, then drain to the latest (coalesce scrubs).
        let mut ts = match req_rx.recv() { Ok(t) => t, Err(_) => return };
        while let Ok(t) = req_rx.try_recv() { ts = t; }
        let target = ts.max(0.0);
        if iter.seek(target).is_err() { continue; }
        // `seek` lands on the keyframe at/before the target; decode forward
        // to the exact frame so the still matches the live preview. For the
        // dual-stream OSV path this decodes the intermediate GOP frames on the
        // HW engine but converts (HW→CPU download + swscale) only the target
        // frame — the bulk of the per-seek cost at 8K (≈11 s → ≈2 s).
        let t0 = std::time::Instant::now();
        let latest = iter.decode_forward_to(target, dt).ok().flatten();
        if let Some(p) = latest {
            tracing::info!("detail decode → {:.3}s in {:?}",
                target, t0.elapsed());
            let ts_ms = (target * 1000.0).round() as i64;
            if res_tx.send((ts_ms, p)).is_err() { return; }
            ctx.request_repaint(); // wake the UI to project + show the still
        }
    }
}

fn open_native_iter(
    path: &std::path::Path,
    kind: vr180_pipeline::SourceKind,
    swap_eyes: bool,
) -> anyhow::Result<Box<dyn vr180_pipeline::fisheye_decode::FisheyePairIter>> {
    use vr180_pipeline::fisheye_decode::{SbsFisheyeIter, DualStreamFisheyeIter, BrawFisheyeIter};
    use vr180_pipeline::decode::HwDecode;
    Ok(match kind {
        vr180_pipeline::SourceKind::DjiOsv => {
            let swap = !swap_eyes;
            Box::new(DualStreamFisheyeIter::new_with_options(path, HwDecode::Auto, 0, swap, 0, 8)?)
        }
        vr180_pipeline::SourceKind::SbsFisheye =>
            Box::new(SbsFisheyeIter::new(path, HwDecode::Auto, 0)?),
        vr180_pipeline::SourceKind::BlackmagicRaw => {
            let info = vr180_braw::BrawInfo::probe(path)
                .map_err(|e| anyhow::anyhow!("braw probe: {e}"))?;
            let opts = vr180_braw::decoder::DecodeOptions::default();
            Box::new(BrawFisheyeIter::new(path, &info, &opts, 0)
                .map_err(|e| anyhow::anyhow!("braw start: {e}"))?)
        }
        _ => anyhow::bail!("non-fisheye source"),
    })
}

/// Hash the stab-affecting settings so the worker re-computes per-frame
/// stabilization only when these change (not on every calib/view tweak).
fn stab_key(s: &Settings) -> u64 {
    let mut h: u64 = if s.stabilize { 1 } else { 0 };
    h = h.wrapping_mul(0x100000001B3).wrapping_add(s.dji_smooth_ms.to_bits() as u64);
    h = h.wrapping_mul(0x100000001B3).wrapping_add(s.dji_max_corr_deg.to_bits() as u64);
    h = h.wrapping_mul(0x100000001B3).wrapping_add(s.dji_imu_phase_ms.to_bits() as u64);
    h = h.wrapping_mul(0x100000001B3).wrapping_add(s.dji_responsiveness.to_bits() as u64);
    h
}

fn compute_stab_for(
    kind: vr180_pipeline::SourceKind,
    path: &std::path::Path,
    imu: Option<&vr180_fisheye::DjiOsvImu>,
    s: &Settings,
    fps: f32,
) -> Option<Vec<EquirectRotation>> {
    if !s.stabilize { return None; }
    let total = vr180_pipeline::decode::probe_video(path)
        .map(|p| (p.duration_sec * p.fps as f64).round() as usize).unwrap_or(0).max(1);
    match kind {
        vr180_pipeline::SourceKind::DjiOsv => imu.and_then(|osv| {
            let max_corr = if s.dji_max_corr_deg > 0.0 { s.dji_max_corr_deg } else { f32::INFINITY };
            vr180_pipeline::dji_imu::set_dji_imu_phase_after_start_ms(s.dji_imu_phase_ms);
            vr180_pipeline::dji_imu::compute_dji_stabilization(
                osv, total, max_corr, s.dji_smooth_ms, fps, s.dji_responsiveness)
                .ok().map(|st| st.per_frame)
        }),
        vr180_pipeline::SourceKind::BlackmagicRaw =>
            vr180_braw::BrawGyroData::extract(path).ok()
                .and_then(|g| vr180_pipeline::braw_imu::compute_braw_stabilization(&g, fps, total).ok())
                .map(|st| st.per_frame),
        _ => None,
    }
}

/// Project + compose ONE already-decoded native fisheye pair into an SBS
/// texture using the current settings. Re-runs the same calib / stab /
/// RS / view-adjust / color path as the live preview, at native res.
fn render_still_from_pair(
    pipeline: &Device,
    kind: vr180_pipeline::SourceKind,
    s: &Settings,
    fps: f32,
    pair: &vr180_pipeline::fisheye_decode::FisheyePair,
    imu: Option<&vr180_fisheye::DjiOsvImu>,
    stab_rotations: Option<&[EquirectRotation]>,
    out_cap: u32,
) -> anyhow::Result<Option<(wgpu::Texture, u32, u32)>> {
    let (src_w, src_h) = (pair.eye_w, pair.eye_h);
    let (calib_left, calib_right) =
        resolve_fisheye_calib_pair(s, kind, src_w, src_h, imu);

    // Output-resolution cap. The SOURCE is always sampled at native res
    // (calib is built from src_w/src_h above), so capping only changes how
    // many OUTPUT pixels we project into — the framing is identical. A small
    // cap gives a fast, soft still for live dragging; `0` / large = native,
    // full-detail. Because both come from THIS same projection of the same
    // cached frame, the low-res and full-res stills are pixel-aligned and
    // never jump when one replaces the other.
    let cap = if out_cap == 0 { u32::MAX } else { out_cap };
    let proj_scale = (cap as f32 / src_w.max(src_h).max(1) as f32).min(1.0);
    let oeq_w = ((src_w as f32 * proj_scale).round() as u32).max(16);
    let oeq_h = ((src_h as f32 * proj_scale).round() as u32).max(16);
    let oside = (((src_w.min(src_h)) as f32 * proj_scale).round() as u32).max(16);

    let dt = 1.0 / fps as f64;
    let stab_idx = if pair.pts_s.is_finite() && pair.pts_s >= 0.0 {
        (pair.pts_s / dt).round() as usize
    } else { 0 };
    let rot = stab_rotations
        .and_then(|v| v.get(stab_idx).copied())
        .unwrap_or(EquirectRotation::IDENTITY);

    let view_adjust = vr180_pipeline::panomap::ViewAdjust {
        pano_yaw_deg: s.pano_yaw_deg, pano_pitch_deg: s.pano_pitch_deg, pano_roll_deg: s.pano_roll_deg,
        stereo_yaw_deg: s.stereo_yaw_deg, stereo_pitch_deg: s.stereo_pitch_deg, stereo_roll_deg: s.stereo_roll_deg,
        upside_down: s.camera_upside_down,
    };
    let (rot_left, rot_right) = if view_adjust.is_identity() { (rot, rot) } else {
        let (vl, vr) = view_adjust.per_eye_matrices();
        (EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &vl)),
         EquirectRotation(vr180_pipeline::panomap::mat3_mul_row_major(&rot.0, &vr)))
    };

    let rs_rows: Option<Vec<f32>> = if s.stabilize && stab_rotations.is_some() {
        imu.and_then(|osv| {
            let lens_a = osv.lens_a.mount_quat_xyzw
                .unwrap_or([-0.0060261087, 0.0048986990, -0.7059469223, 0.7082221508]);
            vr180_pipeline::dji_imu::compute_per_row_quaternions_for_frame(
                osv, stab_idx,
                vr180_pipeline::dji_imu::dji_osmo_readout_ms_for_fps(fps) / 1000.0,
                src_h, fps,
            ).map(|q| vr180_pipeline::dji_imu::pack_per_row_camera_matrices(&q, lens_a))
        })
    } else { None };

    let (left_tex, right_tex) = match s.fisheye_output_mode {
        FisheyeOutputMode::HalfEquirect => {
            if let Some(buf) = rs_rows.as_deref() {
                (pipeline.project_fisheye_to_equirect_rs_texture(&pair.left, src_w, src_h, oeq_w, oeq_h, rot_left, calib_left, buf, 20)?,
                 pipeline.project_fisheye_to_equirect_rs_texture(&pair.right, src_w, src_h, oeq_w, oeq_h, rot_right, calib_right, buf, 21)?)
            } else {
                (pipeline.project_fisheye_to_equirect_texture(&pair.left, src_w, src_h, oeq_w, oeq_h, rot_left, calib_left, 20)?,
                 pipeline.project_fisheye_to_equirect_texture(&pair.right, src_w, src_h, oeq_w, oeq_h, rot_right, calib_right, 21)?)
            }
        }
        FisheyeOutputMode::Fisheye => {
            if let Some(buf) = rs_rows.as_deref() {
                (pipeline.project_fisheye_to_fisheye_rs_texture(&pair.left, src_w, src_h, oside, oside, rot_left, calib_left, buf)?,
                 pipeline.project_fisheye_to_fisheye_rs_texture(&pair.right, src_w, src_h, oside, oside, rot_right, calib_right, buf)?)
            } else {
                (pipeline.project_fisheye_to_fisheye_texture(&pair.left, src_w, src_h, oside, oside, rot_left, calib_left)?,
                 pipeline.project_fisheye_to_fisheye_texture(&pair.right, src_w, src_h, oside, oside, rot_right, calib_right)?)
            }
        }
    };
    let (ew, eh) = (left_tex.width(), left_tex.height());
    let (sbs, ow, oh) = compose_with_color_and_mode(pipeline, s, &left_tex, &right_tex, ew, eh)?;
    Ok(Some((sbs, ow, oh)))
}

/// Build a one-shot stitch: left + right equirect → 2×W SBS, in
/// `Rgba8Unorm` format (egui-wgpu expects this format and applies
/// the swapchain's gamma curve correctly for it).
fn compose_sbs(
    pipeline: &Device,
    left: &wgpu::Texture,
    right: &wgpu::Texture,
    eye_w: u32,
    eye_h: u32,
) -> anyhow::Result<wgpu::Texture> {
    let out_w = eye_w * 2;
    let sbs = pipeline.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sbs_preview"),
        size: wgpu::Extent3d { width: out_w, height: eye_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
             | wgpu::TextureUsages::TEXTURE_BINDING
             | wgpu::TextureUsages::COPY_SRC,
        // Allow an sRGB view so egui (which treats sampled textures as
        // LINEAR and re-applies the OETF) decodes our Rec.709-gamma values
        // to linear on sample — otherwise it double-gamma-encodes them and
        // the preview looks washed out / flatter than the export.
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });

    let mut encoder = pipeline.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sbs_compose_enc"),
    });
    // Copy left → (0, 0)
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: left, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &sbs, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
    );
    // Copy right → (eye_w, 0)
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: right, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &sbs, mip_level: 0,
            origin: wgpu::Origin3d { x: eye_w, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: eye_w, height: eye_h, depth_or_array_layers: 1 },
    );
    pipeline.queue.submit(Some(encoder.finish()));
    Ok(sbs)
}

/// Fingerprint of the Settings fields that feed `build_per_eye_frames`.
/// The per-eye stab/RS bundle ONLY depends on these — view-adjust,
/// color, LUT, preview-mode, trim, etc. are applied per frame in the
/// render loop. The decoder uses this to skip the expensive rebuild
/// (GPMF extract + 255k-sample VQF fusion + smoothing, ~0.3-1 s) when
/// a settings change didn't touch stabilization — THE fix for sliders
/// feeling laggy on `.360` vs OSV.
pub(crate) type StabKey = (bool, CoriSource, u32, u32, u32, bool, RsMode, u32);

pub(crate) fn stab_settings_key(s: &Settings) -> StabKey {
    (
        s.stabilize,
        s.cori_source,
        s.smooth_ms.to_bits(),
        s.max_corr_deg.to_bits(),
        s.gyro_responsiveness.to_bits(),
        s.rs_correct,
        s.rs_mode,
        s.rs_readout_ms.to_bits(),
    )
}

/// Build the per-eye rotation + RS params vec (one tuple per video
/// frame). Empty when neither stabilization nor RS is on.
/// `pub(crate)` so the EAC export path (`app.rs`) can compute the same
/// stab the preview uses and hand it to `export_eac`.
/// Merge per-segment RS ω sources into one continuous `GyroAngvel`,
/// shifting each segment's sample times by the cumulative video
/// duration so the per-scanline readout-time lookups stay continuous
/// across segment boundaries. Returns `None` if no segment yielded
/// usable gyro (caller falls back to identity RS).
fn gather_multi_angvel(
    segments: &[std::path::PathBuf],
    fps: f32,
) -> Option<vr180_core::gyro::GyroAngvel> {
    use vr180_pipeline::decode::probe_video;
    let mut times: Vec<f32> = Vec::new();
    let mut raw: Vec<[f32; 3]> = Vec::new();
    let mut smoothed: Vec<[f32; 3]> = Vec::new();
    let mut toff = 0.0_f32;
    for seg in segments {
        let dur = probe_video(seg).map(|p| p.duration_sec as f32).unwrap_or(0.0);
        if let Ok(gpmf) = extract_gpmf_cached(seg) {
            let raw_imu = vr180_core::gyro::parse_raw_imu(&gpmf);
            let (_, stmps) = vr180_core::gyro::parse_cori_with_stmps(&gpmf);
            if let Some(av) = vr180_core::gyro::GyroAngvel::build(
                &raw_imu.gyro, &stmps, fps, dur.max(1e-3),
            ) {
                for &t in &av.times { times.push(t + toff); }
                raw.extend_from_slice(&av.raw);
                smoothed.extend_from_slice(&av.smoothed);
            }
        }
        toff += dur;
    }
    if times.len() < 2 { return None; }
    Some(vr180_core::gyro::GyroAngvel { times, raw, smoothed })
}

/// Single-segment entry — used everywhere a lone `.360` is processed.
pub(crate) fn build_per_eye_frames(
    input: &std::path::Path,
    s: &Settings,
    fps: f32,
    n_frames_video: usize,
) -> anyhow::Result<Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>> {
    build_per_eye_frames_multi(std::slice::from_ref(&input.to_path_buf()), s, fps, n_frames_video)
}

/// Multi-segment per-eye stab/RS bundle. `segments` is the ordered
/// chain (GS01…, GS02…, …); a single-element slice is byte-identical to
/// the single-file path. For multiple segments the gyro is aggregated
/// across the whole recording (VQF integrated ONCE, RS ω merged with
/// cumulative time offsets) so stabilization is continuous across
/// boundaries — mirrors Python's `concatenate_gyro_data`.
pub(crate) fn build_per_eye_frames_multi(
    segments: &[std::path::PathBuf],
    s: &Settings,
    fps: f32,
    n_frames_video: usize,
) -> anyhow::Result<Vec<((EquirectRotation, EquirectRsParams), (EquirectRotation, EquirectRsParams))>> {
    use vr180_core::gyro::{
        parse_cori, parse_iori, parse_raw_imu, Quat,
        bidirectional_smooth, per_eye_rotations,
        gravity_alignment_quat, apply_gravity_alignment_inplace,
        compute_per_frame_omega, SmoothParams, SMOOTH_WINDOW_S,
    };
    if !s.stabilize && !s.rs_correct {
        return Ok(Vec::new());
    }

    let input = segments.first()
        .map(|p| p.as_path())
        .ok_or_else(|| anyhow::anyhow!("build_per_eye_frames: empty segment list"))?;
    let is_multi = segments.len() > 1;

    // Segment 0 carries the grav for gravity-alignment (recording start)
    // and the GEOC calibration; CORI/IORI concatenate across segments.
    let gpmf = extract_gpmf_cached(input)?;
    let mut cori = parse_cori(&gpmf);
    let mut iori = parse_iori(&gpmf);
    let raw_imu = parse_raw_imu(&gpmf);
    if is_multi {
        for seg in &segments[1..] {
            let g = extract_gpmf_cached(seg)?;
            cori.extend(parse_cori(&g));
            iori.extend(parse_iori(&g));
        }
    }

    let mut used_vqf = false;
    if s.stabilize {
        let want_vqf = match s.cori_source {
            CoriSource::Direct => false,
            CoriSource::Vqf    => true,
            // Auto: route to VQF when the firmware CORI is UNCORRECTED gyro
            // integration (firmware ERS off) — exactly Python's detection at
            // `vr180_gui.py:4161-4169`. Two signatures:
            //   1. Zeroed CORI: the first ≤10 quats are all-components < 0.01.
            //   2. Bias-drift: `cori[0]` starts at EXACT identity
            //      (xyz_max < 5e-4). Firmware-ERS-ON clips begin with a REAL
            //      initial orientation (the camera's tilt vs gravity, so
            //      xyz_max > 1e-3); uncorrected integration starts at exact
            //      zero rotation and only drifts from gyro bias later.
            // Check `cori[0]` ONLY: a no-firmware clip's LATER frames
            // accumulate large gyro drift (xyz → ~0.5 on GS010192), so a
            // max-over-the-stream test mis-routes it to Direct. That was a
            // real regression — it sent no-firmware GS010192 to drifty CORI
            // instead of VQF, which (with the MNOR-frame fix) now matches the
            // Python app to ~0.1°.
            CoriSource::Auto => cori_indicates_no_firmware(&cori),
        };
        if want_vqf {
            cori = if is_multi {
                vr180_pipeline::imu::vqf_cori_equivalent_stream_multi(
                    segments, fps, n_frames_video)?
            } else {
                vr180_pipeline::imu::vqf_cori_equivalent_stream(
                    input, fps, n_frames_video)?
            };
            // Re-reference to frame 0 (USER CHOICE, diverges from Python
            // here): camera lock holds the view where the camera pointed
            // at frame 0, instead of Python's world-frame lock (VQF is
            // gravity+north anchored, so Python's lock target is the
            // world axes). For soft-stab this constant right-factor
            // provably cancels in `heading = raw·smooth⁻¹` (final
            // matrices match Python to 0.02° either way).
            if let Some(ref0) = cori.first().copied() {
                let inv = ref0.conjugate();
                for q in cori.iter_mut() { *q = q.mul(inv); }
            }
            used_vqf = true;
        }
    }

    let rotations: Vec<(EquirectRotation, EquirectRotation)> =
        if s.stabilize && !cori.is_empty() {
            if !used_vqf {
                if let Some(g) = raw_imu.grav.first() {
                    let gq = gravity_alignment_quat(&g.samples, g.scal, 10);
                    apply_gravity_alignment_inplace(&mut cori, gq.conjugate());
                }
            }
            // `smooth_ms = 0` → CAMERA LOCK, Python semantics
            // (`vr180_gui.py:4652-4654`): `q_heading = q_raw` — counteract
            // the FULL camera rotation, no clamp. Achieved by an identity
            // smoothed anchor + max_corr = 0. The old code anchored to
            // `cori[0]` AND still applied the 15° elastic clamp, so on a
            // clip that rotates 56° the lock corrected at most ~35° — the
            // "camera lock still moves around" bug.
            //
            // `smooth_ms > 0` → soft-stab (the legacy GoPro path):
            // anchor is a bidirectional-smoothed orientation, so
            // slow camera motion passes through and jitter is killed.
            // The elastic max-corr clamp applies here only (Python keeps
            // its soft-limit inside the smoother; same formula, same
            // (raw, smoothed) operands — final matrices match to 0.02°).
            let camera_lock = s.smooth_ms <= 0.0;
            let smoothed = if camera_lock {
                vec![Quat::IDENTITY; cori.len()]
            } else {
                bidirectional_smooth(&cori, fps, &SmoothParams {
                    smooth_ms: s.smooth_ms,
                    responsiveness: s.gyro_responsiveness.clamp(0.2, 3.0),
                    ..Default::default()
                })
            };
            let eff_max_corr = if camera_lock { 0.0 } else { s.max_corr_deg };
            // VQF mode: Python forces IORI to identity (the VQF result
            // dict carries `iori_quat = (1,0,0,0)`); the file's parsed
            // IORI belongs to the firmware-ERS pipeline.
            (0..cori.len()).map(|i| {
                let iori_i = if used_vqf { Quat::IDENTITY } else {
                    iori.get(i).copied().unwrap_or(Quat::IDENTITY)
                };
                let (l, r) = per_eye_rotations(
                    cori[i], smoothed[i],
                    iori_i,
                    eff_max_corr,
                );
                (EquirectRotation::from_quat(l), EquirectRotation::from_quat(r))
            }).collect()
        } else {
            Vec::new()
        };

    let rs_eyes: Vec<(EquirectRsParams, EquirectRsParams)> = if s.rs_correct {
        use vr180_pipeline::gpu::RS_N_GROUPS;
        let geoc = vr180_core::geoc::parse_geoc(input)?
            .ok_or_else(|| anyhow::anyhow!("GEOC block missing"))?;
        let front = geoc.front.as_ref().ok_or_else(|| anyhow::anyhow!("GEOC FRNT missing"))?;
        let back  = geoc.back.as_ref().ok_or_else(|| anyhow::anyhow!("GEOC BACK missing"))?;
        let srot_s = s.rs_readout_ms / 1000.0;
        let probe = vr180_pipeline::decode::probe_video(input)?;

        // Python-parity ω source: STMP-anchored timestamps + lerp (the
        // old `int(t/dt)` nearest-index sampler drifts ~33 ms over 71 s
        // — the documented Python bug, fixed there the same way).
        // Smoothed (15 ms box) for the single-ω value; RAW for the 32
        // per-scanline row groups. Multi-segment: build each segment's
        // GyroAngvel and merge with cumulative video-duration offsets so
        // the readout-time lookups stay continuous across boundaries.
        let angvel = if is_multi {
            gather_multi_angvel(segments, fps)
        } else {
            let (_, cori_stmps) = vr180_core::gyro::parse_cori_with_stmps(&gpmf);
            vr180_core::gyro::GyroAngvel::build(
                &raw_imu.gyro, &cori_stmps, fps, probe.duration_sec as f32,
            )
        };

        // Per-axis factors in SHADER order [f_ωx(pitch), f_ωy(yaw),
        // f_ωz(roll)] — matches Python's auto-set values exactly:
        //   no-firmware → (1, 1, 1) on BOTH eyes (the old code zeroed
        //   the yaw component — a real divergence);
        //   firmware    → right eye (pitch 2, yaw 0, roll 2) to cancel
        //   the wrong-direction firmware RS, left eye none.
        let (left_f, right_f): ([f32; 3], [f32; 3]) = match s.rs_mode {
            RsMode::NoFirmware => ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            RsMode::Firmware   => ([0.0, 0.0, 0.0], [2.0, 0.0, 2.0]),
        };

        let mk = |single: [f32; 3], groups: &[[f32; 3]], f: [f32; 3],
                  cal: &vr180_core::geoc::LensCal| {
            let active = f[0] != 0.0 || f[1] != 0.0 || f[2] != 0.0;
            let mut omega_groups = [[0.0_f32; 3]; RS_N_GROUPS];
            let ng = groups.len().min(RS_N_GROUPS);
            for g in 0..ng {
                for c in 0..3 {
                    omega_groups[g][c] = groups[g][c] * f[c];
                }
            }
            EquirectRsParams {
                omega: [single[0] * f[0], single[1] * f[1], single[2] * f[2]],
                omega_groups,
                n_groups: ng as u32,
                srot_s: if active { srot_s } else { 0.0 },
                klns: [
                    cal.klns[0] as f32, cal.klns[1] as f32, cal.klns[2] as f32,
                    cal.klns[3] as f32, cal.klns[4] as f32,
                ],
                ctry: cal.ctry,
                cal_dim: geoc.cal_dim as f32,
            }
        };

        match angvel {
            Some(av) => (0..n_frames_video).map(|i| {
                // Frame time base matches Python's export precompute
                // (`t = start + i/fps`, frame start — not readout mid).
                let t = i as f32 / fps.max(1e-6);
                let single = av.single_at(t);
                let groups = av.groups_at(t, srot_s, RS_N_GROUPS);
                (mk(single, &groups, left_f, back),
                 mk(single, &groups, right_f, front))
            }).collect(),
            None => {
                // No usable gyro stream: legacy constant-ω fallback.
                tracing::warn!("RS: GyroAngvel build failed — using legacy sampler");
                let omegas = compute_per_frame_omega(
                    &raw_imu.gyro, n_frames_video, fps,
                    probe.duration_sec as f32, 0.0, SMOOTH_WINDOW_S,
                );
                (0..n_frames_video).map(|i| {
                    (mk(omegas[i], &[], left_f, back),
                     mk(omegas[i], &[], right_f, front))
                }).collect()
            }
        }
    } else {
        Vec::new()
    };

    let n = rotations.len().max(rs_eyes.len());
    tracing::info!(
        "build_per_eye_frames: stabilize={} rs_correct={} cori_samples={} iori_samples={} rotations={} rs_eyes={} → bundle n={}",
        s.stabilize, s.rs_correct, cori.len(), iori.len(),
        rotations.len(), rs_eyes.len(), n
    );
    Ok((0..n).map(|i| {
        let (rl, rr) = rotations.get(i).copied()
            .unwrap_or((EquirectRotation::IDENTITY, EquirectRotation::IDENTITY));
        let (sl, sr) = rs_eyes.get(i).copied()
            .unwrap_or((EquirectRsParams::DISABLED, EquirectRsParams::DISABLED));
        ((rl, sl), (rr, sr))
    }).collect())
}

