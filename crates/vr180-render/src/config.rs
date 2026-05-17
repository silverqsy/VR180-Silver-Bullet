//! JSON config sidecar — Phase 0.9.
//!
//! The Python GUI on `main` writes one of these to disk, then spawns
//! `vr180-render render --config foo.json`. We deserialize, convert
//! to the existing `export()` parameter set, run the same code path
//! as the CLI-flag invocation. Identical behavior; the JSON is just
//! a more durable plumbing surface for an out-of-process caller.
//!
//! Design choices:
//! - **Identity defaults.** Every field has a sensible default that
//!   maps to "this stage off" (CDL identity, no LUT, no APAC, etc.).
//!   So a minimal config can be just `{"input": ..., "output": ...}`.
//! - **`deny_unknown_fields`.** Mistyped field names should error
//!   rather than silently no-op. The cost is forward-compatibility:
//!   a Python GUI from a future Neo version writing a new field will
//!   fail on an older Rust binary. Acceptable — version both sides
//!   in lockstep.
//! - **No serde on `vr180-pipeline` types.** This keeps the pipeline
//!   crate dependency-light. We mirror the field set here and convert
//!   via `From` impls on submission.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use vr180_pipeline::gpu::{
    CdlParams, ColorGradeParams, ColorStackPlan, MidDetailParams, SharpenParams,
};

/// The whole user-facing config. One file → one export.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportConfig {
    // ─── I/O ────────────────────────────────────────────────────────
    /// Path to the source `.360` file (or first segment of a chain).
    pub input: PathBuf,
    /// Path to the output `.mov` / `.mp4` file. Container is picked
    /// by extension; `.mov` is recommended for APAC + APMP.
    pub output: PathBuf,

    // ─── Video output ───────────────────────────────────────────────
    /// Half-equirect width per eye. Final SBS frame is `2 * eye_w × eye_w`.
    /// 2048 → 4K SBS; 4096 → 8K SBS.
    #[serde(default = "default_eye_w")]
    pub eye_w: u32,
    /// Number of frames to export (0 = all). Mostly useful for tests.
    #[serde(default)]
    pub frames: u32,
    /// Output FPS. `null` = match source.
    #[serde(default)]
    pub fps: Option<f32>,
    /// HEVC target bitrate in kbps. Sensible: 12000 at 4K, 40000 at 8K.
    #[serde(default = "default_bitrate")]
    pub bitrate: u32,

    // ─── Decode + encode backend ────────────────────────────────────
    /// Hardware decode: `"auto"` (VT on macOS, sw elsewhere), `"sw"`, `"vt"`.
    #[serde(default)]
    pub hw_accel: HwAccelStr,
    /// HEVC encoder: `"auto"`, `"sw"` (libx265), `"vt"` (hevc_videotoolbox).
    #[serde(default)]
    pub encoder: EncoderStr,
    /// Skip CPU-EAC-assemble step (VT decoder → wgpu IOSurface). macOS-only.
    #[serde(default)]
    pub zero_copy: bool,
    /// Skip GPU→host readback (color stack → IOSurface BGRA → VT encoder).
    /// Requires `zero_copy: true` + `encoder: "vt"`. macOS-only.
    #[serde(default)]
    pub zero_copy_encode: bool,

    // ─── Color tools (all default to identity) ──────────────────────
    /// CDL: lift / gain / shadow / highlight / gamma.
    #[serde(default)]
    pub cdl: CdlConfig,
    /// Color grade: temperature, tint, saturation.
    #[serde(default)]
    pub grade: GradeConfig,
    /// Unsharp-mask sharpen.
    #[serde(default)]
    pub sharpen: SharpenConfig,
    /// Mid-detail clarity.
    #[serde(default)]
    pub mid_detail: MidDetailConfig,
    /// `.cube` LUT path, or `"bundled"` for the bundled GP-Log LUT.
    /// `null` = no LUT.
    #[serde(default)]
    pub lut: Option<String>,
    /// LUT blend factor [0..1]. 0 = original, 1 = full LUT.
    #[serde(default = "default_intensity")]
    pub lut_intensity: f32,

    // ─── Audio + metadata ───────────────────────────────────────────
    /// Embed APAC (Apple Positional Audio Codec) spatial audio.
    /// macOS-only. Requires the source has a 4-channel ambisonic
    /// track (e.g. stream 5 of a GoPro Max `.360`).
    #[serde(default)]
    pub apac_audio: bool,
    /// APAC target bitrate in bits/sec. 384000 is Apple's recommended
    /// target for ambisonic.
    #[serde(default = "default_apac_bitrate")]
    pub apac_bitrate: u32,
    /// Tag the output as APMP VR180 SBS so visionOS / Vision Pro
    /// recognizes it as immersive projected media. Writes the
    /// `vexu/proj/prji=hequ` + `eyes/stri` atoms.
    #[serde(default)]
    pub apmp: bool,

    /// Enable gyro stabilization (Phase A/B/C). Reads CORI per
    /// frame, applies bidirectional SLERP smoothing, GRAV gravity
    /// alignment, soft elastic max-corr cap, and IORI per-eye split.
    #[serde(default)]
    pub stabilize: bool,
    /// Smoothing time constant in ms for calm periods (Phase B).
    /// `0` = camera lock (no smoothing). Python default: 1000.
    #[serde(default = "default_smooth_ms")]
    pub gyro_smooth_ms: f32,
    /// Smoothing curve power exponent (Phase B). `1.0` = linear.
    #[serde(default = "one")]
    pub gyro_responsiveness: f32,
    /// Soft elastic cap on heading correction angle, in degrees.
    /// Python default: 15.0. `0` disables.
    #[serde(default = "default_max_corr_deg")]
    pub max_corr_deg: f32,
    /// Phase C+: horizon lock. Drops the Z-twist (roll) of the
    /// smoothed CORI before computing the per-frame heading, so
    /// the rendered horizon stays level regardless of how the
    /// camera was rolled during a pan. Off by default.
    #[serde(default)]
    pub horizon_lock: bool,

    /// Phase E: source of the camera orientation stream feeding
    /// stabilization. `"direct"` (default) reads GPMF CORI verbatim
    /// — correct for firmware-stabilized clips. `"vqf"` runs the VQF
    /// 9D fusion pipeline (GYRO+GRAV+MNOR) — needed for bias-drifting
    /// CORI. `"auto"` picks based on a CORI[0] xyz-norm probe.
    #[serde(default)]
    pub cori_source: CoriSourceStr,

    // ─── Phase D: rolling-shutter correction ────────────────────────
    /// Enable per-pixel rolling-shutter correction in the equirect
    /// projection shader. Requires the source `.360` to have a GEOC
    /// calibration block. Off by default.
    #[serde(default)]
    pub rs_correct: bool,
    /// RS mode: `"firmware"` (default) → right eye gets +2.5 factors
    /// to cancel firmware's reversed correction. `"no-firmware"` →
    /// both eyes get 1.0 factors (rare; for clips recorded with
    /// firmware RS disabled).
    #[serde(default)]
    pub rs_mode: RsModeStr,
    /// Sensor readout time in milliseconds. Default 15.224 (GoPro MAX).
    #[serde(default = "default_rs_readout_ms")]
    pub rs_readout_ms: f32,
    /// Override the per-axis RS factors (set to `null` to use the
    /// `rs_mode` defaults).
    #[serde(default)]
    pub rs_pitch_factor: Option<f32>,
    #[serde(default)]
    pub rs_roll_factor:  Option<f32>,
    #[serde(default)]
    pub rs_yaw_factor:   Option<f32>,
}

/// RS mode picker. Serialized as lowercase strings matching the CLI
/// `--rs-mode` enum (`"firmware"` / `"no-firmware"`).
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RsModeStr {
    #[default]
    Firmware,
    NoFirmware,
}

/// Phase E: CORI source picker for stabilization. Serialized as
/// lowercase strings matching the CLI `--cori-source` enum
/// (`"direct"` / `"vqf"` / `"auto"`).
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CoriSourceStr {
    #[default]
    Direct,
    Vqf,
    Auto,
}

// ─── Color-tool sub-configs ─────────────────────────────────────────
//
// Each mirrors the corresponding `vr180_pipeline::gpu::*Params` struct
// AND its identity defaults. We don't derive `Default` because the
// derived all-zero default would be wrong (e.g. CDL gamma=0 would be
// invalid; we want gamma=1).

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CdlConfig {
    #[serde(default)]
    pub lift: f32,
    #[serde(default = "one")]
    pub gamma: f32,
    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub shadow: f32,
    #[serde(default)]
    pub highlight: f32,
}

impl Default for CdlConfig {
    fn default() -> Self {
        Self { lift: 0.0, gamma: 1.0, gain: 1.0, shadow: 0.0, highlight: 0.0 }
    }
}

impl From<&CdlConfig> for CdlParams {
    fn from(c: &CdlConfig) -> Self {
        Self {
            lift: c.lift, gamma: c.gamma, gain: c.gain,
            shadow: c.shadow, highlight: c.highlight,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GradeConfig {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub tint: f32,
    #[serde(default = "one")]
    pub saturation: f32,
}

impl Default for GradeConfig {
    fn default() -> Self {
        Self { temperature: 0.0, tint: 0.0, saturation: 1.0 }
    }
}

impl From<&GradeConfig> for ColorGradeParams {
    fn from(c: &GradeConfig) -> Self {
        Self { temperature: c.temperature, tint: c.tint, saturation: c.saturation }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharpenConfig {
    #[serde(default)]
    pub amount: f32,
    #[serde(default = "default_sharpen_sigma")]
    pub sigma: f32,
}

impl Default for SharpenConfig {
    fn default() -> Self { Self { amount: 0.0, sigma: 1.4 } }
}

impl From<&SharpenConfig> for SharpenParams {
    fn from(c: &SharpenConfig) -> Self {
        Self {
            amount: c.amount, sigma: c.sigma,
            apply_lat_weight: true,   // exports are always equirect
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MidDetailConfig {
    #[serde(default)]
    pub amount: f32,
    #[serde(default = "one")]
    pub sigma: f32,
}

impl Default for MidDetailConfig {
    fn default() -> Self { Self { amount: 0.0, sigma: 1.0 } }
}

impl From<&MidDetailConfig> for MidDetailParams {
    fn from(c: &MidDetailConfig) -> Self {
        Self { amount: c.amount, sigma: c.sigma }
    }
}

// ─── Backend string enums ───────────────────────────────────────────
//
// These match the existing CLI `--hw-accel` / `--encoder` value-enum.
// Serialized as lowercase strings ("auto" / "sw" / "vt") because that's
// what the Python side will write.

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HwAccelStr {
    #[default]
    Auto,
    Sw,
    Vt,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderStr {
    #[default]
    Auto,
    Sw,
    Vt,
}

impl HwAccelStr {
    pub fn into_pipeline(self) -> vr180_pipeline::decode::HwDecode {
        use vr180_pipeline::decode::HwDecode;
        match self {
            HwAccelStr::Auto => HwDecode::Auto,
            HwAccelStr::Sw   => HwDecode::Software,
            HwAccelStr::Vt   => HwDecode::VideoToolbox,
        }
    }
}

impl EncoderStr {
    pub fn resolve(self) -> vr180_pipeline::encode::EncoderBackend {
        use vr180_pipeline::encode::EncoderBackend;
        match self {
            EncoderStr::Auto => {
                if cfg!(target_os = "macos") { EncoderBackend::VideoToolbox }
                else                          { EncoderBackend::Libx265 }
            }
            EncoderStr::Sw => EncoderBackend::Libx265,
            EncoderStr::Vt => EncoderBackend::VideoToolbox,
        }
    }
}

// ─── Conversion from ExportConfig → pipeline call args ──────────────

impl ExportConfig {
    /// Build the color-stack plan (CDL + grade + sharpen + mid-detail).
    /// The LUT is left empty here; the caller fills it in after parsing
    /// the .cube file (which happens inside the export functions —
    /// `build_color_plan`).
    pub fn build_color_plan(&self) -> ColorStackPlan {
        ColorStackPlan {
            cdl:         (&self.cdl).into(),
            color_grade: (&self.grade).into(),
            sharpen:     (&self.sharpen).into(),
            mid_detail:  (&self.mid_detail).into(),
            lut:         None,
        }
    }

    /// Read + parse a JSON config file. Surface a useful error for
    /// the common "wrong field name" case (`deny_unknown_fields` is
    /// strict, which is what we want for forward error detection).
    pub fn from_json_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read config {}: {e}", path.display()))?;
        serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse config {}: {e}", path.display()))
    }
}

// ─── Default value helpers (serde `default = "fn"`) ─────────────────

fn default_eye_w() -> u32 { 2048 }
fn default_bitrate() -> u32 { 12_000 }
fn default_intensity() -> f32 { 1.0 }
fn default_apac_bitrate() -> u32 { 384_000 }
fn default_sharpen_sigma() -> f32 { 1.4 }
fn default_smooth_ms() -> f32 { 1000.0 }
fn default_max_corr_deg() -> f32 { 15.0 }
fn default_rs_readout_ms() -> f32 { vr180_core::gyro::SROT_S * 1000.0 }
fn one() -> f32 { 1.0 }

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal config with only `input` + `output` parses correctly,
    /// fills identity defaults everywhere else.
    #[test]
    fn minimal_config_parses_with_identity_defaults() {
        let json = r#"{ "input": "/in.360", "output": "/out.mov" }"#;
        let cfg: ExportConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(cfg.input.to_str(), Some("/in.360"));
        assert_eq!(cfg.output.to_str(), Some("/out.mov"));
        assert_eq!(cfg.eye_w, 2048);
        assert_eq!(cfg.bitrate, 12_000);
        assert!(matches!(cfg.encoder, EncoderStr::Auto));
        assert!(matches!(cfg.hw_accel, HwAccelStr::Auto));
        // Color stack defaults are identity.
        let plan = cfg.build_color_plan();
        assert!(plan.cdl.is_identity());
        assert!(plan.color_grade.is_identity());
        assert!(plan.sharpen.is_identity());
        assert!(plan.mid_detail.is_identity());
        assert!(plan.lut.is_none());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let json = r#"{ "input": "/in.360", "output": "/out.mov", "what": 1 }"#;
        let err = serde_json::from_str::<ExportConfig>(json).unwrap_err();
        assert!(err.to_string().contains("what"),
            "expected error to name the unknown field; got: {err}");
    }

    #[test]
    fn full_config_round_trips() {
        // Build a non-identity config in code, serialize, parse back,
        // verify field-for-field. This is the key "what the Python GUI
        // would write → what we'd run" round-trip.
        let original_json = r#"{
            "input": "/clip.360",
            "output": "/clip_export.mov",
            "eye_w": 4096,
            "frames": 200,
            "fps": 30.0,
            "bitrate": 40000,
            "hw_accel": "vt",
            "encoder": "vt",
            "zero_copy": true,
            "zero_copy_encode": true,
            "stabilize": true,
            "horizon_lock": true,
            "cori_source": "auto",
            "rs_correct": true,
            "rs_mode": "firmware",
            "rs_readout_ms": 15.224,
            "cdl": {
                "lift": 0.0, "gamma": 0.92, "gain": 1.15,
                "shadow": 0.3, "highlight": -0.2
            },
            "grade": { "temperature": 0.4, "tint": -0.1, "saturation": 1.3 },
            "sharpen": { "amount": 0.8, "sigma": 1.4 },
            "mid_detail": { "amount": -0.4, "sigma": 1.0 },
            "lut": "bundled",
            "lut_intensity": 1.0,
            "apac_audio": true,
            "apac_bitrate": 384000,
            "apmp": true
        }"#;
        let cfg: ExportConfig = serde_json::from_str(original_json).expect("parse");
        assert_eq!(cfg.eye_w, 4096);
        assert_eq!(cfg.bitrate, 40_000);
        assert!(cfg.zero_copy && cfg.zero_copy_encode);
        assert!(matches!(cfg.encoder, EncoderStr::Vt));
        assert_eq!(cfg.cdl.gain, 1.15);
        assert_eq!(cfg.grade.temperature, 0.4);
        assert_eq!(cfg.sharpen.amount, 0.8);
        assert_eq!(cfg.mid_detail.amount, -0.4);
        assert_eq!(cfg.lut.as_deref(), Some("bundled"));
        assert!(cfg.apac_audio && cfg.apmp);
        // Phase C+ / D / E fields.
        assert!(cfg.stabilize && cfg.horizon_lock);
        assert!(matches!(cfg.cori_source, CoriSourceStr::Auto));
        assert!(cfg.rs_correct);
        assert!(matches!(cfg.rs_mode, RsModeStr::Firmware));
        assert!((cfg.rs_readout_ms - 15.224).abs() < 1e-4);

        // Round-trip: serialize back, re-parse, all fields still match.
        let serialized = serde_json::to_string(&cfg).unwrap();
        let cfg2: ExportConfig = serde_json::from_str(&serialized).expect("re-parse");
        assert_eq!(cfg.cdl.gain, cfg2.cdl.gain);
        assert_eq!(cfg.apac_bitrate, cfg2.apac_bitrate);
        assert!(matches!(cfg2.cori_source, CoriSourceStr::Auto));
        assert!(cfg2.rs_correct);
    }
}
