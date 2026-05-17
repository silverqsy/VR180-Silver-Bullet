//! `vr180-render` — CLI binary.
//!
//! Phase 0.1 — `--help` placeholder.
//! Phase 0.2 — `probe-gyro <file.360>` reads the file via ffmpeg-next,
//!             extracts the GPMF data stream, parses CORI/IORI, prints
//!             counts and Euler-angle ranges. Validates the headless
//!             gyro pipeline end-to-end against a real file.
//!
//! Phase 0.9 — `export --config <json>` will be the wedge: the existing
//! Python GUI on `main` shells out to this binary for the heavy work,
//! no UI port needed.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod config;
pub use config::ExportConfig;

#[derive(Parser, Debug)]
#[command(name = "vr180-render", version, about)]
struct Cli {
    /// Verbosity (repeat: -v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum HwAccel {
    /// Platform default: VideoToolbox on macOS, software elsewhere
    /// (silently falls back to software if hwaccel init fails).
    Auto,
    /// Force software decode.
    Sw,
    /// Force VideoToolbox (macOS only). Errors out if VT unavailable.
    Vt,
}

impl From<HwAccel> for vr180_pipeline::decode::HwDecode {
    fn from(h: HwAccel) -> Self {
        match h {
            HwAccel::Auto => Self::Auto,
            HwAccel::Sw   => Self::Software,
            HwAccel::Vt   => Self::VideoToolbox,
        }
    }
}

/// Which HEVC encoder to use on the export side.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Encoder {
    /// Platform default: VideoToolbox on macOS, libx265 elsewhere.
    Auto,
    /// libx265 software encode. Cross-platform, slower (~5-10 fps on
    /// 4K@30 single-core), quality-tunable via bitrate.
    Sw,
    /// hevc_videotoolbox hardware encode. macOS only. Several times
    /// faster than libx265, refuses to fall back to software.
    Vt,
}

impl Encoder {
    /// Resolve `Auto` to a concrete backend for the current platform.
    fn resolve(self) -> vr180_pipeline::encode::EncoderBackend {
        use vr180_pipeline::encode::EncoderBackend;
        match self {
            Encoder::Auto => {
                if cfg!(target_os = "macos") { EncoderBackend::VideoToolbox }
                else                          { EncoderBackend::Libx265 }
            }
            Encoder::Sw => EncoderBackend::Libx265,
            Encoder::Vt => EncoderBackend::VideoToolbox,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Probe gyro data from a .360 file: CORI/IORI count + Euler ranges.
    ProbeGyro {
        /// Path to the .360 file (or first segment of a chain).
        path: PathBuf,
        /// Also run the VQF (Versatile Quaternion Filter) 9D fusion
        /// pipeline (GYRO + GRAV×9.81 + MNOR → quat + bias estimate).
        /// Use this to validate the no-firmware-RS path on a clip where
        /// CORI is bias-drifting (xyz_max < ~0.001 at t=0).
        #[arg(long)]
        vqf: bool,
    },

    /// Probe the EAC layout of a .360 file: stream dimensions, derived
    /// tile width, cross size. With `--out <file.png>` also writes the
    /// assembled Lens-A + Lens-B EAC crosses (3936×7872 RGB PNG by
    /// default on a standard Max — smaller on Max 2 variants).
    ProbeEac {
        path: PathBuf,
        /// Output PNG path for the raw assembled cross pair (no projection).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Output PNG path for the GPU-projected half-equirect SBS image
        /// (left eye = Lens A, right eye = Lens B). Triggers the wgpu
        /// compute kernel.
        #[arg(long)]
        equirect: Option<PathBuf>,
        /// Half-equirect output width per eye (default 2048). Final PNG
        /// is `2 * eye_w × eye_w` — a square half-equirect per eye,
        /// stitched side-by-side.
        #[arg(long, default_value_t = 2048)]
        eye_w: u32,
        /// Hardware-accelerated decode: auto (platform default), sw
        /// (force software), or vt (force VideoToolbox, macOS only).
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
        /// Optional .cube 3D LUT to apply after the equirect projection.
        /// On macOS the bundled GoPro GP-Log LUT at
        /// `assets/Recommended Lut GPLOG.cube` is a sensible default;
        /// pass `--lut bundled` to use it.
        #[arg(long)]
        lut: Option<String>,
        /// LUT blend factor [0..1]. 0 = original, 1 = full LUT.
        #[arg(long, default_value_t = 1.0)]
        lut_intensity: f32,
    },

    /// Phase 0.6 decode-throughput benchmark. Decode the first N frames
    /// of one video stream end-to-end (including `av_hwframe_transfer_data`
    /// when VT is in use) and report fps. Compares software vs VideoToolbox
    /// hwaccel under steady-state conditions where the single-frame
    /// cold-start cost has been amortized.
    BenchDecode {
        path: PathBuf,
        /// Number of frames to decode.
        #[arg(short, long, default_value_t = 100)]
        frames: u32,
        /// Hardware decode path: auto, sw, or vt.
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
    },

    /// Phase 0.6.5: decode one VideoToolbox frame, extract its
    /// CVPixelBuffer's IOSurface, wrap the Y and UV planes as
    /// `wgpu::Texture`s via the IOSurface↔Metal↔wgpu-hal bridge, and
    /// read back a few Y-plane bytes to confirm the chain works.
    /// **macOS only.**
    #[cfg(target_os = "macos")]
    ProbeIosurface {
        path: PathBuf,
    },

    /// Phase 0.8 export: decode → assemble → equirect → color grade → H.265 mp4.
    /// Picks `hevc_videotoolbox` on macOS by default (Phase 0.8.5),
    /// `libx265` everywhere else; override with `--encoder`.
    ///
    /// Color pipeline (Phase 0.7.5), in the order applied:
    /// CDL → 3D LUT → sharpen → mid-detail clarity → temp/tint/sat.
    /// All knobs default to identity (no change); pass any flag to opt
    /// into that stage. Identity stages are skipped entirely (no GPU
    /// dispatch).
    Export {
        /// Input `.360` file.
        input: PathBuf,
        /// Output `.mov` / `.mp4` file.
        output: PathBuf,
        /// Half-equirect width per eye. Final SBS frame is `2 * eye_w × eye_w`.
        #[arg(long, default_value_t = 2048)]
        eye_w: u32,
        /// Number of frames to export (0 = all).
        #[arg(short = 'n', long, default_value_t = 0)]
        frames: u32,
        /// Output FPS. Defaults to the source clip's FPS.
        #[arg(long)]
        fps: Option<f32>,
        /// HEVC target bitrate in kbps (libx265 ABR target /
        /// VT `bit_rate` field; both backends honor it).
        #[arg(long, default_value_t = 12_000)]
        bitrate: u32,
        /// Optional .cube 3D LUT. `bundled` = the GP-Log LUT from
        /// `assets/`.
        #[arg(long)]
        lut: Option<String>,
        /// LUT blend factor [0..1].
        #[arg(long, default_value_t = 1.0)]
        lut_intensity: f32,
        /// Hardware-accelerated decode: auto / sw / vt.
        #[arg(long, value_enum, default_value_t = HwAccel::Auto)]
        hw_accel: HwAccel,
        /// HEVC encoder backend: auto (VT on macOS, libx265 elsewhere) /
        /// sw (force libx265) / vt (force VideoToolbox, macOS only).
        #[arg(long, value_enum, default_value_t = Encoder::Auto)]
        encoder: Encoder,
        /// Skip host-memory hop entirely: VideoToolbox decoder's IOSurface
        /// goes straight to wgpu via the Metal HAL escape, EAC assembly
        /// happens on the GPU (`nv12_to_eac_cross.wgsl`). **macOS only.**
        /// Forces `--hw-accel vt`.
        #[arg(long, default_value_t = false)]
        zero_copy: bool,
        /// Also skip the GPU→host readback on the encode side: the
        /// color stack writes directly to an IOSurface-backed BGRA
        /// CVPixelBuffer that `hevc_videotoolbox` reads in place.
        /// **macOS only, requires `--zero-copy --encoder vt`.** Drops
        /// the staging-buffer readback + swscale RGB→YUV from every
        /// frame, ~1.5× faster end-to-end on graded exports.
        #[arg(long, default_value_t = false)]
        zero_copy_encode: bool,
        /// Embed APAC (Apple Positional Audio Codec) spatial audio in
        /// the output. Extracts the 4-channel ambisonic PCM track from
        /// the source `.360`, encodes via the `apac_encode` Swift helper
        /// (Vision Pro spatial audio), and mux-passthroughs the encoded
        /// video bit-exact into a final `.mov`. **macOS only** (helper
        /// needs `kAudioFormatAPAC`, available macOS 14+). If the
        /// source has no ambisonic track, the flag is a no-op + warning.
        #[arg(long, default_value_t = false)]
        apac_audio: bool,
        /// APAC target bitrate in bits/sec. 384 kbps is Apple's
        /// recommended target for ambisonic spatial audio.
        #[arg(long, default_value_t = 384_000)]
        apac_bitrate: u32,
        /// Tag the output video track with APMP (Apple Projected Media
        /// Profile) atoms so visionOS / Vision Pro recognizes it as
        /// VR180 immersive media. Adds `vexu/proj/prji=hequ`,
        /// `vexu/eyes/stri`, `vexu/pack/pkin=side`, `hfov=180°` to
        /// the `hvc1` sample description. Writes Google `sv3d/st3d/SA3D`
        /// alongside for YouTube / Quest compat.
        #[arg(long, default_value_t = false)]
        apmp: bool,
        /// Enable gyro stabilization (Phase A/B/C). Reads CORI per
        /// frame, optionally smoothes via velocity-dampened
        /// bidirectional SLERP, applies GRAV-based gravity alignment,
        /// soft-clamps the heading correction, and splits IORI per eye.
        /// See `--gyro-smooth-ms`, `--gyro-responsiveness`, and
        /// `--max-corr-deg` for tuning. Set `--gyro-smooth-ms 0` for
        /// pure camera-lock mode (Phase A behavior).
        #[arg(long, default_value_t = false)]
        stabilize: bool,
        /// Smoothing time constant in ms for calm periods (Phase B).
        /// Higher = more stable but laggier on intentional pans.
        /// `0` = camera lock (no smoothing). Python default: 1000.
        #[arg(long, default_value_t = 1000.0)]
        gyro_smooth_ms: f32,
        /// Smoothing curve power exponent (Phase B). `1.0` = linear
        /// blend between smooth (slow) and fast (high-vel) τ;
        /// `<1` anticipatory; `>1` laggy. Python default: 1.0.
        #[arg(long, default_value_t = 1.0)]
        gyro_responsiveness: f32,
        /// Soft elastic cap on heading correction angle, in degrees
        /// (Phase C). Beyond this, the smoothed quat is logarithmically
        /// pulled back toward raw to prevent black borders during
        /// extreme motion. Python default: 15.0. `0` disables the cap.
        #[arg(long, default_value_t = 15.0)]
        max_corr_deg: f32,
        /// Phase C+: horizon lock. Decomposes the smoothed CORI via
        /// swing-twist around the camera forward axis (+Z) and drops
        /// the Z-twist (roll) before computing the per-frame heading
        /// correction. Net effect: the rendered horizon stays level
        /// even during slow camera pans that the smoother would
        /// otherwise carry along (e.g. holding the camera with a
        /// constant 10° tilt while panning). Best combined with
        /// `--stabilize` so GRAV alignment anchors world-up to true
        /// gravity at the start of the clip. Rolling around the
        /// view direction is symmetric in a VR180 half-equirect, so
        /// horizon-lock never crops the rendered hemisphere.
        #[arg(long, default_value_t = false)]
        horizon_lock: bool,
        /// Phase E: pick the source of the camera orientation stream
        /// feeding the stabilization smoother. `direct` (default):
        /// GPMF CORI verbatim (right for firmware-stabilized clips).
        /// `vqf`: run the VQF 9D fusion pipeline on raw
        /// GYRO+GRAV+MNOR (right for bias-drifting CORI). `auto`:
        /// probe the first CORI sample and pick `vqf` if its xyz
        /// norm is < 1e-3 (a sign of bias drift), else `direct`.
        #[arg(long, value_enum, default_value_t = CoriSource::Direct)]
        cori_source: CoriSource,

        // --- Phase D: rolling-shutter correction ---
        /// Phase D: enable per-pixel rolling-shutter correction in
        /// the equirect projection shader. Uses the source's GEOC
        /// KLNS fisheye time map + smoothed 800 Hz GYRO to apply a
        /// small-angle per-pixel rotation, compensating for the
        /// sensor's top-to-bottom readout over SROT≈15.2 ms. Required
        /// to clean up the yaw-modded right eye on firmware-RS clips
        /// (where firmware applied the wrong-direction correction).
        /// Requires the source `.360` to have a GEOC block at the
        /// file tail (true for all GoPro Max calibrated `.360`s).
        #[arg(long, default_value_t = false)]
        rs_correct: bool,
        /// RS mode: `firmware` (default, most common) → right eye
        /// gets +2.5 factors to cancel firmware's reversed correction,
        /// left eye gets none. `no-firmware` → both eyes get 1.0
        /// factors. The mode picks default per-axis factors; you can
        /// override any axis with `--rs-pitch-factor` etc.
        #[arg(long, value_enum, default_value_t = RsMode::Firmware)]
        rs_mode: RsMode,
        /// Sensor readout time in milliseconds. Default 15.224 (GoPro
        /// MAX SROT). `0` disables RS regardless of `--rs-correct`.
        #[arg(long, default_value_t = vr180_core::gyro::SROT_S * 1000.0)]
        rs_readout_ms: f32,
        /// Override the pitch RS factor (X axis). Default depends on
        /// `--rs-mode`. Applies to whichever eye(s) the mode enables.
        #[arg(long, allow_hyphen_values = true)]
        rs_pitch_factor: Option<f32>,
        /// Override the roll RS factor (Z axis). Default depends on
        /// `--rs-mode`.
        #[arg(long, allow_hyphen_values = true)]
        rs_roll_factor: Option<f32>,
        /// Override the yaw RS factor (Y axis). Default 0.0 (Python's
        /// `rs_yaw_factor` is also 0 — yaw has negligible RS effect).
        #[arg(long, allow_hyphen_values = true)]
        rs_yaw_factor: Option<f32>,

        // --- CDL (ASC Color Decision List) ---
        /// CDL lift: black-point shift. 0 = no change, +1 pushes blacks
        /// to mid-gray. Range typically [-0.5, +0.5].
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        cdl_lift: f32,
        /// CDL gamma. 1.0 = no change, >1 brightens midtones, <1 darkens.
        #[arg(long, default_value_t = 1.0)]
        cdl_gamma: f32,
        /// CDL gain: white-point multiplier. 1.0 = no change.
        #[arg(long, default_value_t = 1.0)]
        cdl_gain: f32,
        /// CDL shadow adjustment via Hermite-smoothstep mask. 0 = no change;
        /// scaled by 0.6 internally so [-1..+1] maps to ~[-0.6..+0.6].
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        cdl_shadow: f32,
        /// CDL highlight adjustment via Hermite-smoothstep mask.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        cdl_highlight: f32,

        // --- Color grade (temp/tint/sat) ---
        /// Temperature: +1 warms (more red, less blue), -1 cools.
        /// Internal scale is 0.30 — `[-1..+1]` slider maps to ±30% channel shift.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        temperature: f32,
        /// Tint: +1 toward magenta (less green), -1 toward green.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        tint: f32,
        /// Saturation. 0 = grayscale, 1 = unchanged, >1 = boosted (may clip).
        #[arg(long, default_value_t = 1.0)]
        saturation: f32,

        // --- Sharpen (unsharp mask) ---
        /// Sharpen amount. 0 = off; typical use [0.3, 1.5].
        #[arg(long, default_value_t = 0.0)]
        sharpen: f32,
        /// Sharpen Gaussian σ (blur radius for the LP component).
        /// Default 1.4 matches the Python app.
        #[arg(long, default_value_t = 1.4)]
        sharpen_sigma: f32,

        // --- Mid-detail clarity ---
        /// Mid-detail clarity amount. 0 = off; positive = haze removal feel
        /// (low-pass-blend in midtones), negative = local contrast boost.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        mid_detail: f32,
        /// Mid-detail Gaussian σ on the 1/4-res blur. Default 1.0.
        #[arg(long, default_value_t = 1.0)]
        mid_detail_sigma: f32,
    },

    /// Phase 0.9: JSON config sidecar mode. Reads an `ExportConfig`
    /// JSON file with every knob the `export` subcommand's flags
    /// expose, then runs the same export pipeline.
    ///
    /// Intended for the Python GUI on `main`: write the user's
    /// settings to a JSON, spawn `vr180-render render --config foo.json`,
    /// get the Neo-speed export pipeline for free.
    ///
    /// See `crates/vr180-render/src/config.rs` for the schema.
    /// Identity defaults everywhere; a minimal config can be just
    /// `{ "input": "...", "output": "..." }`.
    Render {
        /// Path to a JSON config file (see `ExportConfig`).
        #[arg(long)]
        config: PathBuf,
    },
}

fn init_tracing(verbosity: u8) {
    use tracing_subscriber::EnvFilter;
    let default = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match cli.command {
        None => {
            println!(
                "vr180-render {} — Phase 0.2 (GPMF + CORI/IORI)",
                env!("CARGO_PKG_VERSION")
            );
            println!("Try: vr180-render --help");
            Ok(())
        }
        Some(Cmd::ProbeGyro { path, vqf }) => probe_gyro(&path, vqf),
        Some(Cmd::ProbeEac { path, out, equirect, eye_w, hw_accel, lut, lut_intensity }) =>
            probe_eac(&path, out.as_deref(), equirect.as_deref(), eye_w,
                      hw_accel.into(), lut.as_deref(), lut_intensity),
        Some(Cmd::BenchDecode { path, frames, hw_accel }) =>
            bench_decode(&path, frames, hw_accel.into()),
        #[cfg(target_os = "macos")]
        Some(Cmd::ProbeIosurface { path }) => probe_iosurface(&path),
        Some(Cmd::Render { config }) => run_render_from_config(&config),
        Some(Cmd::Export {
            input, output, eye_w, frames, fps, bitrate,
            lut, lut_intensity, hw_accel, encoder, zero_copy, zero_copy_encode,
            apac_audio, apac_bitrate, apmp, stabilize,
            gyro_smooth_ms, gyro_responsiveness, max_corr_deg, horizon_lock,
            cori_source,
            rs_correct, rs_mode, rs_readout_ms,
            rs_pitch_factor, rs_roll_factor, rs_yaw_factor,
            cdl_lift, cdl_gamma, cdl_gain, cdl_shadow, cdl_highlight,
            temperature, tint, saturation,
            sharpen, sharpen_sigma,
            mid_detail, mid_detail_sigma,
        }) => {
            use vr180_pipeline::gpu::{
                CdlParams, ColorGradeParams, SharpenParams, MidDetailParams,
                ColorStackPlan,
            };
            // Build the color plan from the CLI knobs. The LUT field is
            // filled in inside each export function (after the .cube file
            // is parsed) so the plan can be cloned per-frame cheaply.
            let plan_template = ColorStackPlan {
                cdl: CdlParams {
                    lift: cdl_lift, gamma: cdl_gamma, gain: cdl_gain,
                    shadow: cdl_shadow, highlight: cdl_highlight,
                },
                lut: None,
                color_grade: ColorGradeParams { temperature, tint, saturation },
                sharpen: SharpenParams {
                    amount: sharpen, sigma: sharpen_sigma,
                    apply_lat_weight: true,  // exports are always equirect
                },
                mid_detail: MidDetailParams {
                    amount: mid_detail, sigma: mid_detail_sigma,
                },
            };
            let post = PostProcess { apac_audio, apac_bitrate, apmp };
            let stab = StabilizeParams {
                enabled: stabilize,
                smooth: vr180_core::gyro::SmoothParams {
                    smooth_ms: gyro_smooth_ms,
                    responsiveness: gyro_responsiveness,
                    ..Default::default()
                },
                max_corr_deg,
                horizon_lock,
                cori_source,
            };
            let rs = RsParams::from_mode(
                rs_correct, rs_mode, rs_readout_ms / 1000.0,
                rs_pitch_factor, rs_roll_factor, rs_yaw_factor,
            );
            export(&input, &output, eye_w, frames, fps, bitrate,
                   lut.as_deref(), lut_intensity, hw_accel.into(),
                   encoder.resolve(), zero_copy, zero_copy_encode,
                   plan_template, post, stab, rs)
        }
    }
}

/// Phase 0.9: dispatch from a JSON `ExportConfig` into the same
/// `export()` function the CLI-flag path uses. Just a thin adapter —
/// parse → field-map → call. Every knob exposed in the JSON has an
/// existing equivalent in the export pipeline.
fn run_render_from_config(config_path: &std::path::Path) -> anyhow::Result<()> {
    println!("Loading config: {}", config_path.display());
    let cfg = ExportConfig::from_json_file(config_path)?;
    let plan = cfg.build_color_plan();
    let post = PostProcess {
        apac_audio: cfg.apac_audio,
        apac_bitrate: cfg.apac_bitrate,
        apmp: cfg.apmp,
    };
    let cori_source = match cfg.cori_source {
        config::CoriSourceStr::Direct => CoriSource::Direct,
        config::CoriSourceStr::Vqf    => CoriSource::Vqf,
        config::CoriSourceStr::Auto   => CoriSource::Auto,
    };
    let stab = StabilizeParams {
        enabled: cfg.stabilize,
        smooth: vr180_core::gyro::SmoothParams {
            smooth_ms: cfg.gyro_smooth_ms,
            responsiveness: cfg.gyro_responsiveness,
            ..Default::default()
        },
        max_corr_deg: cfg.max_corr_deg,
        horizon_lock: cfg.horizon_lock,
        cori_source,
    };
    let rs_mode = match cfg.rs_mode {
        config::RsModeStr::Firmware   => RsMode::Firmware,
        config::RsModeStr::NoFirmware => RsMode::NoFirmware,
    };
    let rs = RsParams::from_mode(
        cfg.rs_correct, rs_mode, cfg.rs_readout_ms / 1000.0,
        cfg.rs_pitch_factor, cfg.rs_roll_factor, cfg.rs_yaw_factor,
    );
    export(
        &cfg.input,
        &cfg.output,
        cfg.eye_w,
        cfg.frames,
        cfg.fps,
        cfg.bitrate,
        cfg.lut.as_deref(),
        cfg.lut_intensity,
        cfg.hw_accel.into_pipeline(),
        cfg.encoder.resolve(),
        cfg.zero_copy,
        cfg.zero_copy_encode,
        plan,
        post,
        stab,
        rs,
    )
}

/// Stabilization knobs the export functions consume. Bundles the
/// `--stabilize` enable flag, the smoothing parameters, and the
/// max-correction soft cap into one struct so the export function
/// signatures don't explode every time a knob is added.
#[derive(Debug, Clone, Copy)]
struct StabilizeParams {
    enabled: bool,
    smooth: vr180_core::gyro::SmoothParams,
    max_corr_deg: f32,
    /// Phase C+: drop the Z-twist (roll) component of the smoothed
    /// CORI before computing the per-frame heading. Net effect: the
    /// rendered horizon stays level even when the camera was rolled
    /// during a pan that the smoother would otherwise carry along.
    /// Off by default — GRAV alignment alone is sufficient for clips
    /// where the camera is held roughly level throughout.
    horizon_lock: bool,
    /// Phase E: which quaternion stream to use as the input camera
    /// orientation. `Direct` reads CORI from GPMF (correct for
    /// firmware-stabilized clips). `Vqf` runs the VQF 9D fusion
    /// pipeline on raw GYRO+GRAV+MNOR (needed for bias-drifting
    /// CORI, e.g. no-firmware-RS clips). `Auto` picks `Vqf` when
    /// the first CORI sample has near-zero xyz norm (a signature
    /// of bias drift), else `Direct`.
    cori_source: CoriSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CoriSource {
    /// GPMF CORI stream verbatim (default; correct when firmware
    /// stabilization is on, e.g. all 3 of our reference clips).
    Direct,
    /// VQF 9D fusion (GYRO+GRAV+MNOR → orientation), then Y↔Z swap
    /// at the boundary to match CORI's on-disk component order.
    /// Use this for clips where CORI is bias-drifting (`xyz_norm <
    /// ~0.001` at t=0).
    Vqf,
    /// Probe the first CORI sample; if its xyz norm is below ~1e-3
    /// (indicating bias-drifting firmware), substitute VQF output.
    /// Otherwise use the direct CORI stream.
    Auto,
}

impl Default for StabilizeParams {
    fn default() -> Self {
        Self {
            enabled: false,
            smooth: vr180_core::gyro::SmoothParams::default(),
            max_corr_deg: 15.0,
            horizon_lock: false,
            cori_source: CoriSource::Direct,
        }
    }
}

/// Phase D — rolling-shutter correction parameters.
///
/// Off by default. When `enabled = true`, the equirect projection
/// shader runs the per-pixel KLNS-fisheye time map and applies a
/// small-angle 3D rotation by the per-frame angular velocity scaled
/// by the per-pixel `t_offset`. Per-eye factors decide which eye
/// gets the correction; the no-firmware-RS case uses 1.0 for both,
/// the firmware-RS case uses +2.5 for the right eye only (the
/// yaw-modded lens whose firmware correction is reversed).
#[derive(Debug, Clone, Copy)]
struct RsParams {
    enabled: bool,
    /// Sensor readout time in seconds. Default: 0.015224 (GoPro Max SROT).
    srot_s: f32,
    /// Per-axis RS factors for the left eye (BACK lens).
    /// Firmware mode default: 0.0 (no correction).
    /// No-firmware mode default: 1.0.
    left_factor:  RsFactor,
    /// Per-axis RS factors for the right eye (FRNT lens, yaw-modded).
    /// Firmware mode default: 2.5 (cancels firmware's reversed correction).
    /// No-firmware mode default: 1.0.
    right_factor: RsFactor,
}

#[derive(Debug, Clone, Copy)]
struct RsFactor {
    pitch: f32,
    roll:  f32,
    yaw:   f32,
}

impl RsFactor {
    const ZERO:           Self = Self { pitch: 0.0, roll: 0.0, yaw: 0.0 };
    const NO_FIRMWARE:    Self = Self { pitch: 1.0, roll: 1.0, yaw: 0.0 };
    const FIRMWARE_RIGHT: Self = Self { pitch: 2.5, roll: 2.5, yaw: 0.0 };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RsMode {
    /// Most common case: GoPro firmware already applied its RS
    /// correction. Right eye (yaw-modded FRNT lens) gets +2.5 factors
    /// to cancel the wrong-direction firmware correction; left eye
    /// gets no further correction.
    Firmware,
    /// Footage with firmware RS disabled (rare — usually requires the
    /// "no firmware RS" mod or a custom recording profile). Both eyes
    /// get factors of 1.0.
    NoFirmware,
}

impl RsParams {
    fn from_mode(enabled: bool, mode: RsMode, srot_s: f32,
                 pitch_override: Option<f32>, roll_override: Option<f32>,
                 yaw_override: Option<f32>) -> Self {
        let (left_default, right_default) = match mode {
            RsMode::Firmware   => (RsFactor::ZERO,        RsFactor::FIRMWARE_RIGHT),
            RsMode::NoFirmware => (RsFactor::NO_FIRMWARE, RsFactor::NO_FIRMWARE),
        };
        // Per-axis CLI overrides — apply uniformly to both eyes if set.
        let apply_override = |def: RsFactor| RsFactor {
            pitch: pitch_override.unwrap_or(def.pitch),
            roll:  roll_override.unwrap_or(def.roll),
            yaw:   yaw_override.unwrap_or(def.yaw),
        };
        let mut left  = left_default;
        let mut right = right_default;
        // Overrides only apply to whichever eye(s) are already non-zero
        // so the user's `--rs-pitch-factor 1.5` doesn't accidentally
        // turn on RS for the LEFT eye in firmware mode (where it
        // should stay off).
        if left.pitch != 0.0 || left.roll != 0.0 || left.yaw != 0.0 {
            left = apply_override(left);
        }
        if right.pitch != 0.0 || right.roll != 0.0 || right.yaw != 0.0 {
            right = apply_override(right);
        }
        Self { enabled, srot_s, left_factor: left, right_factor: right }
    }
}

impl Default for RsParams {
    fn default() -> Self {
        Self {
            enabled: false,
            srot_s: vr180_core::gyro::SROT_S,
            left_factor:  RsFactor::ZERO,
            right_factor: RsFactor::ZERO,
        }
    }
}

/// One per-frame per-eye projection bundle (rotation + RS params).
/// Empty Vec → both stabilization and RS are off → callers use
/// `(IDENTITY, DISABLED)` for every frame.
#[derive(Clone, Copy, Debug)]
struct EquirectEyeFrame {
    rotation: vr180_pipeline::gpu::EquirectRotation,
    rs:       vr180_pipeline::gpu::EquirectRsParams,
}

impl EquirectEyeFrame {
    const IDLE: Self = Self {
        rotation: vr180_pipeline::gpu::EquirectRotation::IDENTITY,
        rs:       vr180_pipeline::gpu::EquirectRsParams::DISABLED,
    };
}

type PerFrameEyeFrames = Vec<(EquirectEyeFrame, EquirectEyeFrame)>;

/// Compute per-frame, per-eye projection bundles (rotation + RS) from
/// the source's GPMF.
///
/// Stabilization pipeline (matches the Python `GyroStabilizer.smooth()`):
/// 1. Parse CORI, IORI, GRAV from GPMF.
/// 2. Compute the gravity-alignment quaternion `g` from the first
///    10 GRAV samples (if any). Right-multiply every CORI by `g⁻¹`
///    so the world frame has Y-down = true gravity.
/// 3. Bidirectional velocity-dampened SLERP smoothing on the
///    (aligned) CORI stream. Calm spans get heavy smoothing
///    (`smooth_ms`); fast spans get light (`fast_ms`).
/// 4. Per-frame: heading = raw · smooth⁻¹, soft-clamped against
///    `max_corr_deg`, swing-twist horizon-locked if requested,
///    then split per-eye with IORI.
///
/// RS pipeline (Phase D): if `rs.enabled`:
/// 1. Parse GEOC (KLNS / CTRY / CAL_DIM) from file tail.
/// 2. Smooth 800 Hz GYRO with a 20 ms moving average.
/// 3. Per-frame sample at video time → effective ω in shader frame.
/// 4. Multiply ω by per-axis RS factors per eye.
/// 5. Bundle into `EquirectRsParams` alongside the rotation.
///
/// Returns a `Vec<(left_frame, right_frame)>`. Empty if neither
/// stabilization nor RS is on.
fn compute_per_eye_frames(
    segments: &[std::path::PathBuf],
    stab: &StabilizeParams,
    rs: &RsParams,
    fps: f32,
    n_frames_video: usize,
    total_duration_s: f64,
) -> anyhow::Result<PerFrameEyeFrames> {
    use vr180_core::gyro::{
        parse_cori, parse_iori, parse_raw_imu, Quat,
        bidirectional_smooth, per_eye_rotations_horizon_lock,
        gravity_alignment_quat, apply_gravity_alignment_inplace,
        compute_per_frame_omega, SMOOTH_WINDOW_S,
    };
    use vr180_pipeline::decode::extract_gpmf_stream;
    use vr180_pipeline::gpu::{EquirectRotation, EquirectRsParams};

    if !stab.enabled && !rs.enabled {
        return Ok(Vec::new());
    }
    let primary = segments.first()
        .ok_or_else(|| anyhow::anyhow!("compute_per_eye_frames: no segments"))?;

    // Aggregate GPMF across all segments. CORI / IORI / GYRO blocks
    // from later segments simply append to the parsed lists — naive
    // byte concat works because GPMF records are self-delimiting.
    let mut gpmf: Vec<u8> = Vec::new();
    for seg in segments {
        gpmf.extend_from_slice(&extract_gpmf_stream(seg)?);
    }
    let mut cori = parse_cori(&gpmf);
    let iori   = parse_iori(&gpmf);
    let raw_imu = parse_raw_imu(&gpmf);

    // Phase E: optionally substitute the VQF-derived stream for the
    // direct GPMF CORI. Done BEFORE gravity alignment so the VQF
    // output flows through the same downstream pipeline as direct
    // CORI — modulo the GRAV-alignment skip below.
    let mut used_vqf = false;
    if stab.enabled {
        let want_vqf = match stab.cori_source {
            CoriSource::Direct => false,
            CoriSource::Vqf    => true,
            CoriSource::Auto   => is_cori_bias_drifting(&cori),
        };
        if want_vqf {
            println!("Stabilization: substituting VQF-derived CORI-equivalent stream \
                      (source={:?}{})",
                stab.cori_source,
                if stab.cori_source == CoriSource::Auto {
                    " — detected bias drift in raw CORI"
                } else { "" });
            // VQF only consumes the first segment for now — extending
            // vqf_cori_equivalent_stream to chain segments is tracked
            // separately. For single-segment files this is the full
            // clip; for multi-segment, this is a known limitation
            // (Phase E follow-up).
            cori = vr180_pipeline::imu::vqf_cori_equivalent_stream(
                primary, fps, n_frames_video,
            )?;
            // Reference-correct: VQF emits orientation relative to its
            // own gravity-aligned world frame, so cori[0] is typically
            // 100°+ from identity for a normally-held camera (the
            // camera's body-Z axis is not aligned with VQF's world
            // axis). Direct GPMF CORI starts at ~identity by firmware
            // convention. Right-multiply everything by cori[0]⁻¹ so
            // our VQF output behaves like direct CORI: frame 0 is
            // identity (camera looking forward in the equirect world
            // frame), subsequent frames carry only the rotation that
            // happened since the start of the clip. Required for
            // horizon-lock (which assumes +Z = camera forward) to do
            // the right thing.
            if let Some(ref0) = cori.first().copied() {
                let ref_inv = ref0.conjugate();
                for q in cori.iter_mut() {
                    *q = q.mul(ref_inv);
                }
                println!("Stabilization: VQF reference frame (cori[0]) anchored to identity");
            }
            used_vqf = true;
        }
    }

    // Stabilization branch: build the per-eye rotation list.
    let rotations: Vec<(EquirectRotation, EquirectRotation)> = if stab.enabled && !cori.is_empty() {
        // NOTE: do NOT call `cori_swap_yz` — see Phase A "stabilize"
        // fix commit for the long-form explanation of why removing the
        // swap matches Python's output.

        // Step 1: gravity alignment. Skipped when the CORI source is
        // VQF — VQF's accelerometer-based inclination correction
        // already produces gravity-aligned quaternions. Applying our
        // own GRAV alignment on top double-corrects and produces a
        // garbage heading (matches Python `_q_gravity_align = None`
        // when source contains "vqf").
        if used_vqf {
            println!("Stabilization: GRAV alignment skipped (VQF already gravity-aligned)");
        } else if let Some(first_grav) = raw_imu.grav.first() {
            let g = gravity_alignment_quat(&first_grav.samples, first_grav.scal, 10);
            apply_gravity_alignment_inplace(&mut cori, g.conjugate());
            println!("Stabilization: GRAV alignment {:?}", g);
        } else {
            println!("Stabilization: no GRAV stream — skipping gravity alignment");
        }

        // Step 2: bidirectional SLERP smoothing.
        let smoothed = bidirectional_smooth(&cori, fps, &stab.smooth);
        println!(
            "Stabilization: {} CORI frames, {} IORI frames, smoothed (τ_calm={} ms, τ_fast={} ms, resp={:.2}), horizon_lock={}",
            cori.len(), iori.len(),
            stab.smooth.smooth_ms, stab.smooth.fast_ms, stab.smooth.responsiveness,
            stab.horizon_lock,
        );

        // Step 3: per-frame heading + horizon-lock + IORI per-eye split.
        let n = cori.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let raw = cori[i];
            let smooth = smoothed[i];
            let iori_q = iori.get(i).copied().unwrap_or(Quat::IDENTITY);
            let (q_left, q_right) = per_eye_rotations_horizon_lock(
                raw, smooth, iori_q, stab.max_corr_deg, stab.horizon_lock,
            );
            out.push((
                EquirectRotation::from_quat(q_left),
                EquirectRotation::from_quat(q_right),
            ));
        }
        out
    } else {
        if stab.enabled {
            eprintln!("warning: --stabilize requested but source has no CORI \
                       stream; falling back to identity rotation");
        }
        Vec::new()
    };

    // RS branch: build the per-eye RS params list (one per VIDEO frame).
    let rs_eyes: Vec<(EquirectRsParams, EquirectRsParams)> = if rs.enabled {
        // GEOC: use the LAST segment's calibration. GoPro embeds GEOC
        // in every chapter (same camera = same lens cal), but Python
        // uses the last-segment convention since that's where geometry
        // refinement runs in the firmware.
        let last_seg = segments.last().expect("at least one segment");
        let geoc = vr180_core::geoc::parse_geoc(last_seg)
            .map_err(|e| anyhow::anyhow!("GEOC parse io error: {e}"))?
            .ok_or_else(|| anyhow::anyhow!(
                "--rs-correct requested but the source `.360` has no GEOC \
                 calibration block; cannot build a KLNS time map"
            ))?;
        let front = geoc.front.as_ref().ok_or_else(|| anyhow::anyhow!(
            "GEOC missing FRNT lens calibration; cannot apply RS to right eye"))?;
        let back  = geoc.back.as_ref().ok_or_else(|| anyhow::anyhow!(
            "GEOC missing BACK lens calibration; cannot apply RS to left eye"))?;
        // SROT: auto-detected from GPMF / file tail. Falls back to the
        // user-supplied `rs.srot_s` if not found anywhere.
        let srot_s = vr180_core::geoc::lookup_srot_s(last_seg, Some(&gpmf))
            .map_err(|e| anyhow::anyhow!("SROT lookup io error: {e}"))?
            .unwrap_or(rs.srot_s);
        println!(
            "RS correction: GEOC OK (cal_dim={}, FRNT c0={:.2}, BACK c0={:.2}), \
             SROT={:.3} ms{}, mode={}, factors L=({:.2},{:.2},{:.2}) R=({:.2},{:.2},{:.2})",
            geoc.cal_dim, front.klns[0], back.klns[0],
            srot_s * 1000.0,
            if (srot_s - rs.srot_s).abs() > 1e-5 { " (auto-detected)" } else { "" },
            if rs.right_factor.pitch != 0.0 && rs.left_factor.pitch == 0.0 { "firmware" } else { "no-firmware" },
            rs.left_factor.pitch, rs.left_factor.roll, rs.left_factor.yaw,
            rs.right_factor.pitch, rs.right_factor.roll, rs.right_factor.yaw,
        );

        // Per-frame ω in shader frame (rad/s), sampled at frame
        // center. `total_duration_s` is the sum across all segments,
        // giving the right dt = duration / total_gyro_samples.
        let duration_s = total_duration_s as f32;
        let n = n_frames_video;
        let omegas = compute_per_frame_omega(
            &raw_imu.gyro, n, fps, duration_s,
            srot_s * 0.5,    // sample at center of readout window
            SMOOTH_WINDOW_S,
        );

        // Pre-build per-eye RsParams shells (everything except omega +
        // srot_s is constant across frames). Note: `srot_s` here is
        // the auto-detected value (or the user override fallback).
        let mut out: Vec<(EquirectRsParams, EquirectRsParams)> = Vec::with_capacity(n);
        for i in 0..n {
            let om = omegas[i];
            let left = make_rs_params(om, rs.left_factor,  srot_s, back,  geoc.cal_dim);
            let right = make_rs_params(om, rs.right_factor, srot_s, front, geoc.cal_dim);
            out.push((left, right));
        }
        out
    } else {
        Vec::new()
    };

    // Combine: zip stabilization + RS into per-eye frames. Use the
    // longer of the two (typical: stabilization has CORI-count
    // frames, RS has video-frame count; they should match for a
    // well-formed clip, but be defensive).
    let n_out = rotations.len().max(rs_eyes.len());
    if n_out == 0 {
        return Ok(Vec::new());
    }
    let mut out: PerFrameEyeFrames = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let (r_l, r_r) = rotations.get(i).copied().unwrap_or((
            EquirectRotation::IDENTITY, EquirectRotation::IDENTITY,
        ));
        let (rs_l, rs_r) = rs_eyes.get(i).copied().unwrap_or((
            EquirectRsParams::DISABLED, EquirectRsParams::DISABLED,
        ));
        out.push((
            EquirectEyeFrame { rotation: r_l, rs: rs_l },
            EquirectEyeFrame { rotation: r_r, rs: rs_r },
        ));
    }
    Ok(out)
}

/// Cheap heuristic: is the source's CORI stream bias-drifting?
///
/// On clips where firmware orientation fusion is disabled (some "no
/// firmware RS" GoPro recording profiles), the CORI quaternions are
/// emitted as near-identity at t=0 and drift slowly as the IMU's bias
/// integrates uncorrected. The signature: at sample 0, the imaginary
/// part `(x, y, z)` has near-zero norm (≪ 1e-3) — a genuine camera
/// pose would have a small but non-trivial vector part even at the
/// very start because of mounting tilt.
///
/// We use this as the `CoriSource::Auto` decision: bias-drift → use
/// VQF; otherwise → use direct CORI.
fn is_cori_bias_drifting(cori: &[vr180_core::gyro::Quat]) -> bool {
    let Some(q0) = cori.first() else { return false; };
    let xyz_norm = (q0.x * q0.x + q0.y * q0.y + q0.z * q0.z).sqrt();
    xyz_norm < 1e-3
}

/// Build a single eye's `EquirectRsParams` for one frame from the
/// raw ω, the eye's RS factors, the SROT, and the lens calibration.
/// Factors apply per axis (degrees/sec → effective rad/sec after the
/// `× factor` and the implicit `deg2rad`).
fn make_rs_params(
    omega_rad_s: [f32; 3],
    factor: RsFactor,
    srot_s: f32,
    cal: &vr180_core::geoc::LensCal,
    cal_dim: u32,
) -> vr180_pipeline::gpu::EquirectRsParams {
    // ω input is in rad/s (shader frame). Factors are unitless
    // multipliers applied per axis. Matches Python's
    // `pitch_coeff = pitch_rate * rs_pitch_factor * deg2rad`
    // when pitch_rate is in deg/s — we pre-convert to rad/s upstream
    // so the deg2rad is implicit.
    //
    // Disable this eye entirely if all factors are zero (the
    // `srot_s = 0` sentinel makes the shader short-circuit).
    let all_zero = factor.pitch == 0.0 && factor.roll == 0.0 && factor.yaw == 0.0;
    let effective_srot = if all_zero { 0.0 } else { srot_s };
    let klns: [f32; 5] = [
        cal.klns[0] as f32, cal.klns[1] as f32, cal.klns[2] as f32,
        cal.klns[3] as f32, cal.klns[4] as f32,
    ];
    vr180_pipeline::gpu::EquirectRsParams {
        omega: [
            omega_rad_s[0] * factor.pitch,   // shader X axis = pitch
            omega_rad_s[1] * factor.yaw,     // shader Y axis = yaw
            omega_rad_s[2] * factor.roll,    // shader Z axis = roll
        ],
        srot_s: effective_srot,
        klns,
        ctry: cal.ctry,
        cal_dim: cal_dim as f32,
    }
}

/// Look up the per-eye projection bundle for a given frame index.
/// Returns `(IDLE, IDLE)` if out of range (defensive against CORI /
/// video frame-count mismatch).
fn per_eye_frame_for_frame(
    frames: &PerFrameEyeFrames,
    frame_idx: u32,
) -> (EquirectEyeFrame, EquirectEyeFrame) {
    frames.get(frame_idx as usize).copied().unwrap_or((
        EquirectEyeFrame::IDLE,
        EquirectEyeFrame::IDLE,
    ))
}

/// Post-encode steps the user opted into via CLI flags. Run after the
/// video-only `.mov` is encoded and closed. APAC audio mux happens
/// first (because it produces a fresh `.mov` file with the audio
/// track added), then APMP atom tagging in-place on the result.
#[derive(Debug, Clone, Copy)]
struct PostProcess {
    apac_audio:    bool,
    apac_bitrate:  u32,
    apmp:          bool,
}

impl PostProcess {
    fn any(&self) -> bool { self.apac_audio || self.apmp }
}

fn export(
    input: &std::path::Path,
    output: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: Option<f32>,
    bitrate_kbps: u32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
    hw: vr180_pipeline::decode::HwDecode,
    encoder: vr180_pipeline::encode::EncoderBackend,
    zero_copy: bool,
    zero_copy_encode: bool,
    plan_template: vr180_pipeline::gpu::ColorStackPlan,
    post: PostProcess,
    stab: StabilizeParams,
    rs:   RsParams,
) -> anyhow::Result<()> {
    if zero_copy && !cfg!(target_os = "macos") {
        anyhow::bail!(
            "--zero-copy is macOS-only (requires IOSurface↔Metal interop). \
             Phase 0.6.8 will add the Windows CUDA↔Vulkan equivalent."
        );
    }
    if encoder == vr180_pipeline::encode::EncoderBackend::VideoToolbox
        && !cfg!(target_os = "macos")
    {
        anyhow::bail!(
            "--encoder vt (hevc_videotoolbox) is macOS-only. \
             Use --encoder sw (libx265) on Windows / Linux."
        );
    }
    if zero_copy_encode {
        if !cfg!(target_os = "macos") {
            anyhow::bail!("--zero-copy-encode is macOS-only.");
        }
        if !zero_copy {
            anyhow::bail!(
                "--zero-copy-encode requires --zero-copy \
                 (full GPU pipeline from decode to encode)."
            );
        }
        if encoder != vr180_pipeline::encode::EncoderBackend::VideoToolbox {
            anyhow::bail!(
                "--zero-copy-encode requires --encoder vt (hevc_videotoolbox). \
                 libx265 cannot accept IOSurface-backed input frames."
            );
        }
    }
    if post.apac_audio && !cfg!(target_os = "macos") {
        anyhow::bail!(
            "--apac-audio is macOS-only (requires the apac_encode Swift helper)."
        );
    }

    // If APAC audio is requested, the encoder writes a video-only
    // temp file; apac_encode then muxes audio + video into `output`.
    // Without APAC, the encoder writes directly to `output`.
    let video_only_path = if post.apac_audio {
        tempfile_sibling(output, "video_only")?
    } else {
        output.to_path_buf()
    };

    // Detect chapter chain. `GS010172.360` may auto-find GS02...,
    // GS03..., etc. For single-segment files this returns just
    // `[input]`. All segments share one combined gyro stream and one
    // encoder so the output is one continuous mov/mp4.
    let segments = vr180_core::segments::detect_segments(input);
    if segments.len() > 1 {
        println!("Detected {} chapter segments:", segments.len());
        for s in &segments {
            println!("  - {}", s.display());
        }
    }

    // Probe each segment for duration; sum for total clip length.
    let mut per_segment_frames: Vec<u32> = Vec::with_capacity(segments.len());
    let mut total_duration_s: f64 = 0.0;
    let mut probe_first: Option<vr180_pipeline::decode::VideoProbe> = None;
    for seg in &segments {
        let p = vr180_pipeline::decode::probe_video(seg)?;
        let seg_frames = (p.duration_sec * p.fps as f64).round() as u32;
        total_duration_s += p.duration_sec;
        per_segment_frames.push(seg_frames);
        if probe_first.is_none() { probe_first = Some(p); }
    }
    let probe = probe_first.expect("at least one segment");
    let actual_fps = fps.unwrap_or(probe.fps);
    let total_frames = if n_frames > 0 {
        n_frames as usize
    } else {
        (total_duration_s * actual_fps as f64).round() as usize
    };

    // Precompute per-frame, per-eye stabilization + RS bundles once
    // across all segments. Empty vec → callers use IDLE per frame.
    let frames = compute_per_eye_frames(
        &segments, &stab, &rs, actual_fps, total_frames, total_duration_s,
    )?;

    // Open the encoder once for all segments.
    let out_w = eye_w * 2;
    let out_h = eye_w;
    let mut encoder_owned = if zero_copy_encode {
        #[cfg(target_os = "macos")]
        { vr180_pipeline::encode::H265Encoder::create_zero_copy_vt(
            &video_only_path, out_w, out_h, actual_fps, bitrate_kbps,
        )? }
        #[cfg(not(target_os = "macos"))]
        unreachable!()
    } else {
        vr180_pipeline::encode::H265Encoder::create(
            &video_only_path, out_w, out_h, actual_fps, bitrate_kbps, encoder,
        )?
    };
    if post.apmp { encoder_owned.tag_apmp_vr180_sbs()?; }

    let t_start = std::time::Instant::now();
    let mut frames_remaining: u32 = if n_frames > 0 { n_frames } else { u32::MAX };
    let mut frame_offset_so_far: u32 = 0;
    for (seg_idx, seg_path) in segments.iter().enumerate() {
        let seg_total = per_segment_frames[seg_idx];
        let seg_budget = seg_total.min(frames_remaining);
        if seg_budget == 0 { break; }
        if segments.len() > 1 {
            println!("\n=== Segment {}/{}: {} ({} frames) ===",
                seg_idx + 1, segments.len(),
                seg_path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                seg_budget);
        }
        if zero_copy {
            #[cfg(target_os = "macos")]
            export_zero_copy(seg_path, eye_w, seg_budget, actual_fps,
                             lut_spec, lut_intensity,
                             zero_copy_encode, plan_template.clone(),
                             &frames, frame_offset_so_far,
                             &mut encoder_owned, t_start)?;
            #[cfg(not(target_os = "macos"))]
            unreachable!();
        } else {
            export_cpu_assemble(seg_path, eye_w, seg_budget, actual_fps, bitrate_kbps,
                                lut_spec, lut_intensity, hw, encoder,
                                plan_template.clone(),
                                &frames, frame_offset_so_far,
                                &mut encoder_owned, t_start)?;
        }
        frames_remaining = frames_remaining.saturating_sub(seg_budget);
        frame_offset_so_far += seg_budget;
        if frames_remaining == 0 { break; }
    }
    encoder_owned.finish()?;
    let total = t_start.elapsed();
    let avg_fps = frame_offset_so_far as f32 / total.as_secs_f32().max(1e-6);
    println!("\nDone: {} frames in {:.2?} ({:.2} fps)",
        frame_offset_so_far, total, avg_fps);
    let size = std::fs::metadata(&video_only_path).map(|m| m.len()).unwrap_or(0);
    println!("Output size: {:.1} MB", size as f64 / 1_048_576.0);

    // Post-encode: APAC audio mux. APMP atoms were already set as
    // codec side-data on the encoder, so they're already in
    // video_only_path's stsd — apac_encode's passthrough preserves
    // them bit-exact when it copies the video track.
    if post.apac_audio {
        run_apac_audio(input, &video_only_path, output, post.apac_bitrate)?;
        // Best-effort cleanup of the video-only temp file.
        let _ = std::fs::remove_file(&video_only_path);
    }
    Ok(())
}

/// Build a sibling path for a temp file: `<output>.<suffix>.mov`.
/// Lives next to `output` so it's on the same filesystem (cheap rename
/// fallback if we ever switch to that pattern) and gets cleaned up by
/// the export pipeline on success.
fn tempfile_sibling(output: &std::path::Path, suffix: &str)
    -> anyhow::Result<std::path::PathBuf>
{
    let stem = output.file_stem()
        .ok_or_else(|| anyhow::anyhow!("output path has no file stem"))?
        .to_string_lossy().to_string();
    let ext = output.extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "mov".into());
    let parent = output.parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    Ok(parent.join(format!(".{stem}.{suffix}.{ext}")))
}

/// Extract the source `.360`'s ambisonic audio track to a temp WAV,
/// then spawn `apac_encode` to encode the WAV as APAC and mux it
/// alongside the (passthrough) video track of `video_only` into
/// `final_out`. macOS-only.
#[cfg(target_os = "macos")]
fn run_apac_audio(
    source: &std::path::Path,
    video_only: &std::path::Path,
    final_out: &std::path::Path,
    apac_bitrate: u32,
) -> anyhow::Result<()> {
    use vr180_pipeline::audio::{probe_ambisonic, extract_ambisonic_to_wav};
    use vr180_pipeline::helpers::spawn_apac_encode;

    let amb = probe_ambisonic(source)?;
    let Some(amb) = amb else {
        eprintln!("warning: --apac-audio requested but {} has no ambisonic track; \
                   copying video-only to {}", source.display(), final_out.display());
        std::fs::rename(video_only, final_out)?;
        return Ok(());
    };
    println!("APAC: extracting ambisonic (stream {}, {}ch @ {} Hz) ...",
        amb.stream_index, amb.channels, amb.sample_rate);

    let temp_wav = tempfile_sibling(final_out, "amb")?;
    let temp_wav = temp_wav.with_extension("wav");
    extract_ambisonic_to_wav(source, &temp_wav)?;
    let wav_size = std::fs::metadata(&temp_wav).map(|m| m.len()).unwrap_or(0);
    println!("APAC: extracted {:.1} MB WAV at {}", wav_size as f64 / 1_048_576.0, temp_wav.display());

    println!("APAC: muxing video + APAC audio via apac_encode @ {} kbps ...", apac_bitrate / 1000);
    let t = std::time::Instant::now();
    spawn_apac_encode(&temp_wav, Some(video_only), final_out, apac_bitrate)?;
    println!("APAC: done in {:.2?}", t.elapsed());

    // Cleanup temp WAV.
    let _ = std::fs::remove_file(&temp_wav);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run_apac_audio(
    _source: &std::path::Path,
    _video_only: &std::path::Path,
    _final_out: &std::path::Path,
    _apac_bitrate: u32,
) -> anyhow::Result<()> {
    anyhow::bail!("--apac-audio is macOS-only");
}

/// Phase 0.6 / 0.7 / 0.8 path: hwaccel decode → host-memory NV12 →
/// swscale RGB → CPU EAC assembly → GPU projection + chained color
/// stack → HEVC. As of Phase 0.7.5.5 the equirect output stays on the
/// GPU through the entire color chain (one submit, one readback).
fn export_cpu_assemble(
    input: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: f32,
    bitrate_kbps: u32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
    hw: vr180_pipeline::decode::HwDecode,
    encoder_backend: vr180_pipeline::encode::EncoderBackend,
    plan_template: vr180_pipeline::gpu::ColorStackPlan,
    frames: &PerFrameEyeFrames,
    frame_offset: u32,
    encoder: &mut vr180_pipeline::encode::H265Encoder,
    t_start: std::time::Instant,
) -> anyhow::Result<()> {
    use vr180_core::eac::{assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::{iter_stream_pairs, probe_video};
    use vr180_pipeline::gpu::Device;

    let probe = probe_video(input)?;
    let out_w = eye_w * 2;
    let out_h = eye_w;

    println!("Export (CPU-assemble path) segment: {}", input.display());
    println!("  source : {} × {}  @ {:.3} fps  ({:.1}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("  output : {} × {}  @ {:.3} fps  H.265 ({}) {} kbps",
        out_w, out_h, fps, encoder_label(encoder_backend), bitrate_kbps);

    let plan = build_color_plan(plan_template, lut_spec, lut_intensity)?;
    print_color_stack(&plan);

    let device = Device::new()?;
    let mut decoder = iter_stream_pairs(input, hw, n_frames)?;
    let dims = decoder.dims();
    let cw = dims.cross_w() as usize;
    println!("  decode : {} (probed {}×{}, EAC tile_w={})",
        decoder.decode_path(), dims.stream_w, dims.stream_h, dims.tile_w());

    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];

    let mut frame_idx: u32 = 0;
    let mut last_print = std::time::Instant::now();
    while let Some(pair) = decoder.next_pair()? {
        assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
        assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);

        // Cross → equirect texture → chained color stack → RGB8.
        // The equirect texture stays GPU-resident through every color
        // stage; only the final readback hits host memory.
        //
        // Eye assignment (matches Python convention + project memory):
        //   Lens A (s0/FRNT) → RIGHT eye
        //   Lens B (s4/BACK) → LEFT eye
        // (After the yaw mod, the lens labeled A is physically on
        // the right side of the user's head.)
        let global_idx = frame_offset + frame_idx;
        let (f_left, f_right) = per_eye_frame_for_frame(frames, global_idx);
        let left_tex  = device.project_cross_to_equirect_texture(
            &cross_b, dims.cross_w(), eye_w, out_h, f_left.rotation, f_left.rs)?;
        let right_tex = device.project_cross_to_equirect_texture(
            &cross_a, dims.cross_w(), eye_w, out_h, f_right.rotation, f_right.rs)?;
        let left  = device.apply_color_stack_texture(&left_tex,  eye_w, out_h, &plan)?;
        let right = device.apply_color_stack_texture(&right_tex, eye_w, out_h, &plan)?;
        encoder.encode_frame(&stitch_sbs(&left, &right, eye_w, out_h))?;
        frame_idx += 1;
        progress_tick(global_idx + 1, t_start, &mut last_print);
        if frame_idx >= n_frames { break; }
    }
    Ok(())
}

/// Phase 0.6.6 + 0.7.5.5 + 0.7.5.6 path: IOSurface → wgpu zero-copy →
/// GPU EAC assembly + equirect projection texture + chained color
/// stack → (readback OR IOSurface-encode) → HEVC. macOS only.
///
/// When `zero_copy_encode = true`, the color stack writes directly to
/// an IOSurface-backed BGRA CVPixelBuffer that `hevc_videotoolbox`
/// reads in place — no GPU→host readback, no swscale RGB→YUV.
#[cfg(target_os = "macos")]
fn export_zero_copy(
    input: &std::path::Path,
    eye_w: u32,
    n_frames: u32,
    fps: f32,
    lut_spec: Option<&str>,
    lut_intensity: f32,
    zero_copy_encode: bool,
    plan_template: vr180_pipeline::gpu::ColorStackPlan,
    frames: &PerFrameEyeFrames,
    frame_offset: u32,
    encoder: &mut vr180_pipeline::encode::H265Encoder,
    t_start: std::time::Instant,
) -> anyhow::Result<()> {
    use vr180_pipeline::decode::{ZeroCopyStreamPairIter, probe_video};
    use vr180_pipeline::gpu::{Device, Lens};
    use vr180_pipeline::interop_macos::create_bgra_encode_buffer;

    let probe = probe_video(input)?;
    let out_w = eye_w * 2;
    let out_h = eye_w;

    println!("Export (zero-copy IOSurface path) segment: {}", input.display());
    println!("  source : {} × {}  @ {:.3} fps  ({:.1}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("  output : {} × {}  @ {:.3} fps", out_w, out_h, fps);
    if zero_copy_encode {
        println!("  pipeline: VT → IOSurface → wgpu (nv12_to_eac_cross → equirect_tex → color stack → SBS BGRA IOSurface) → VT encoder (zero-copy)");
    } else {
        println!("  pipeline: VT → IOSurface → wgpu (nv12_to_eac_cross → equirect_tex → color stack) → readback → HEVC");
    }

    let plan = build_color_plan(plan_template, lut_spec, lut_intensity)?;
    print_color_stack(&plan);

    let device = Device::new()?;
    let mut decoder = ZeroCopyStreamPairIter::new(input, n_frames)?;
    let dims = decoder.dims();
    println!("  decode : VideoToolbox (probed {}×{}, EAC tile_w={})",
        dims.stream_w, dims.stream_h, dims.tile_w());

    let mut frame_idx: u32 = 0;
    let mut last_print = std::time::Instant::now();
    while let Some(pair) = decoder.next_pair(&device.device)? {
        // GPU EAC assembly: NV12 plane textures → RGB cross texture.
        let cross_a = device.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::A, dims,
        )?;
        let cross_b = device.nv12_to_eac_cross(
            &pair.s0_y.texture, &pair.s0_uv.texture,
            &pair.s4_y.texture, &pair.s4_uv.texture,
            Lens::B, dims,
        )?;

        // Cross texture → equirect texture → color stack.
        // Eye assignment: Lens A → RIGHT eye, Lens B → LEFT eye
        // (matches Python convention + project memory; after yaw mod
        // the lens labeled A is physically on the right side of the
        // user's head).
        let global_idx = frame_offset + frame_idx;
        let (f_left, f_right) = per_eye_frame_for_frame(frames, global_idx);
        let left_tex  = device.project_cross_texture_to_equirect_texture(
            &cross_b, eye_w, out_h, f_left.rotation, f_left.rs)?;
        let right_tex = device.project_cross_texture_to_equirect_texture(
            &cross_a, eye_w, out_h, f_right.rotation, f_right.rs)?;

        if zero_copy_encode {
            // Allocate a fresh IOSurface-backed BGRA CVPixelBuffer. The
            // color stack writes through to its bytes; VT encoder reads
            // them in place. CVPixelBuffer is retained by encode_pixel_buffer
            // for the encoder's lifetime needs; our local `pb` can drop
            // at end of iteration (the retain on the encoder side keeps
            // the surface alive while VT is using it).
            let pb = create_bgra_encode_buffer(&device.device, out_w, out_h)?;
            device.apply_color_stack_to_sbs_bgra(
                &left_tex, &right_tex, &pb.wgpu_tex, eye_w, out_h, &plan,
            )?;
            encoder.encode_pixel_buffer(&pb)?;
        } else {
            let left  = device.apply_color_stack_texture(&left_tex,  eye_w, out_h, &plan)?;
            let right = device.apply_color_stack_texture(&right_tex, eye_w, out_h, &plan)?;
            encoder.encode_frame(&stitch_sbs(&left, &right, eye_w, out_h))?;
        }

        // Drop the zero-copy pair AFTER the kernel dispatch is done —
        // releases the IOSurface retains so the VT decoder can recycle
        // the underlying CVPixelBuffer for the next frame.
        drop(pair);
        frame_idx += 1;
        progress_tick(global_idx + 1, t_start, &mut last_print);
        if frame_idx >= n_frames { break; }
    }
    Ok(())
}

fn encoder_label(b: vr180_pipeline::encode::EncoderBackend) -> &'static str {
    use vr180_pipeline::encode::EncoderBackend;
    match b {
        EncoderBackend::Libx265      => "libx265 (SW)",
        EncoderBackend::VideoToolbox => "hevc_videotoolbox (HW)",
    }
}

/// Fill the LUT field of a partially-built ColorStackPlan from the
/// CLI `--lut <path>` argument. Resolves `bundled` → assets path.
/// The plan template carries the rest of the user knobs unchanged.
fn build_color_plan(
    mut plan: vr180_pipeline::gpu::ColorStackPlan,
    lut_spec: Option<&str>,
    lut_intensity: f32,
) -> anyhow::Result<vr180_pipeline::gpu::ColorStackPlan> {
    if let Some(spec) = lut_spec {
        let path = match spec {
            "bundled" => resolve_bundled_lut()
                .ok_or_else(|| anyhow::anyhow!("bundled LUT not found"))?,
            s => std::path::PathBuf::from(s),
        };
        let lut = vr180_core::Cube3DLut::from_file(&path)?;
        println!("  LUT    : {}^3 @ intensity {:.2} ({})",
            lut.size, lut_intensity, path.display());
        plan.lut = Some((lut, lut_intensity));
    }
    Ok(plan)
}

/// Print a one-line summary of which color tools will actually run.
/// Helps the user confirm their CLI flags landed without spelunking
/// the param struct in trace logs.
fn print_color_stack(plan: &vr180_pipeline::gpu::ColorStackPlan) {
    if !plan.any_active() {
        println!("  color  : (none — no CDL / LUT / sharpen / mid-detail / grade)");
        return;
    }
    let mut stages: Vec<String> = Vec::new();
    if !plan.cdl.is_identity() {
        let c = plan.cdl;
        stages.push(format!(
            "cdl(lift={:.2}, gamma={:.2}, gain={:.2}, sh={:.2}, hl={:.2})",
            c.lift, c.gamma, c.gain, c.shadow, c.highlight));
    }
    if plan.lut.is_some() { stages.push("lut3d".into()); }
    if !plan.sharpen.is_identity() {
        let s = plan.sharpen;
        stages.push(format!("sharpen(amount={:.2}, σ={:.2})", s.amount, s.sigma));
    }
    if !plan.mid_detail.is_identity() {
        let m = plan.mid_detail;
        stages.push(format!("mid_detail(amount={:.2}, σ={:.2})", m.amount, m.sigma));
    }
    if !plan.color_grade.is_identity() {
        let g = plan.color_grade;
        stages.push(format!(
            "color_grade(temp={:+.2}, tint={:+.2}, sat={:.2})",
            g.temperature, g.tint, g.saturation));
    }
    println!("  color  : {}", stages.join(" → "));
}

fn stitch_sbs(left: &[u8], right: &[u8], eye_w: u32, eye_h: u32) -> Vec<u8> {
    let out_w = eye_w * 2;
    let row_l = (eye_w * 3) as usize;
    let row_sbs = (out_w * 3) as usize;
    let mut sbs = vec![0u8; (out_w * eye_h * 3) as usize];
    for y in 0..eye_h as usize {
        sbs[y * row_sbs..y * row_sbs + row_l]
            .copy_from_slice(&left[y * row_l..y * row_l + row_l]);
        sbs[y * row_sbs + row_l..y * row_sbs + row_l * 2]
            .copy_from_slice(&right[y * row_l..y * row_l + row_l]);
    }
    sbs
}

fn progress_tick(frame_idx: u32, t_start: std::time::Instant, last: &mut std::time::Instant) {
    if last.elapsed().as_secs_f32() > 1.0 {
        let avg_fps = frame_idx as f32 / t_start.elapsed().as_secs_f32();
        print!("\r  frame {frame_idx}  @ {avg_fps:.2} fps  ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        *last = std::time::Instant::now();
    }
}

fn finish_export(output: &std::path::Path, frame_idx: u32, t_start: std::time::Instant)
    -> anyhow::Result<()>
{
    let total = t_start.elapsed();
    let avg_fps = frame_idx as f32 / total.as_secs_f32();
    println!("\nDone: {frame_idx} frames in {total:.2?} ({avg_fps:.2} fps)");
    let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!("Output size: {:.1} MB", size as f64 / 1_048_576.0);
    Ok(())
}

/// Locate `assets/Recommended Lut GPLOG.cube` relative to either the
/// workspace root (dev runs) or the running binary (release builds).
fn resolve_bundled_lut() -> Option<std::path::PathBuf> {
    // Dev: walk up from CWD looking for the assets dir.
    let mut p = std::env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = p.join("assets/Recommended Lut GPLOG.cube");
        if candidate.is_file() {
            return Some(candidate);
        }
        p = p.parent()?.to_path_buf();
    }
    // Release: next to the exe.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("Recommended Lut GPLOG.cube");
            if p.is_file() { return Some(p); }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn probe_iosurface(path: &std::path::Path) -> anyhow::Result<()> {
    use vr180_pipeline::decode::decode_first_vt_frame;
    use vr180_pipeline::interop_macos::{
        extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
        IOSurfaceNv12Descriptor,
    };
    use vr180_pipeline::gpu::Device;
    use std::time::Instant;

    println!("=== Phase 0.6.5: IOSurface ↔ Metal ↔ wgpu zero-copy bridge ===");
    println!("file: {}", path.display());
    println!();

    // 1. Decode one VT frame (no av_hwframe_transfer_data).
    let t = Instant::now();
    let vt_frame = decode_first_vt_frame(path)?;
    println!("[1] decode_first_vt_frame:           {:?}  (format={:?}, {}×{})",
        t.elapsed(), vt_frame.format(), vt_frame.width(), vt_frame.height());

    // 2. Pull the IOSurface backing the CVPixelBuffer.
    let t = Instant::now();
    let surface = extract_iosurface_from_vt_frame(&vt_frame)?;
    println!("[2] CVPixelBufferGetIOSurface:       {:?}  (planes={}, y={}×{}, uv={}×{})",
        t.elapsed(),
        surface.plane_count(),
        surface.plane_width(0), surface.plane_height(0),
        surface.plane_width(1), surface.plane_height(1));

    // 3. Cache geometry (NV12).
    let desc = IOSurfaceNv12Descriptor::new(surface)?;
    println!("[3] IOSurfaceNv12Descriptor:         OK  (y_bpr={}, uv_bpr={})",
        desc.y_bytes_per_row, desc.uv_bytes_per_row);

    // 4. Wrap Y plane as a wgpu::Texture (Metal R8Unorm view).
    let device = Device::new()?;
    let (w, h) = (desc.width, desc.height);
    let t = Instant::now();
    let y_tex = wgpu_texture_from_iosurface_plane(
        &device.device, desc.surface, 0,
        metal::MTLPixelFormat::R8Unorm,
        wgpu::TextureFormat::R8Unorm,
        w, h,
        "iosurface_y",
    )?;
    println!("[4] wgpu_texture_from_iosurface(Y):  {:?}  ({}×{} R8Unorm)",
        t.elapsed(), y_tex.width, y_tex.height);

    // 5. Read back the first row of Y so we can prove the chain is live
    //    (not just a successful-but-blank texture handoff). 256-byte
    //    aligned for the staging buffer requirement.
    let row_bytes_unpadded = w;
    let row_bytes_padded = (row_bytes_unpadded + 255) & !255;
    let buf_size = row_bytes_padded as u64;
    let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("y_readback_enc"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &y_tex.texture, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes_padded),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d { width: w, height: 1, depth_or_array_layers: 1 },
    );
    let t = Instant::now();
    device.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    let mapped = slice.get_mapped_range();
    let first_row = &mapped[..w as usize];
    let min = *first_row.iter().min().unwrap_or(&0);
    let max = *first_row.iter().max().unwrap_or(&0);
    let avg: f32 = first_row.iter().map(|&v| v as f32).sum::<f32>() / w as f32;
    println!("[5] read back Y row 0:               {:?}  (min={min} max={max} avg={avg:.1})",
        t.elapsed());
    drop(mapped);
    staging.unmap();

    println!();
    println!("✓ zero-copy chain works end-to-end.");
    println!("  VT decoder → CVPixelBuffer → IOSurface → MTLTexture → wgpu::Texture");
    println!("  No av_hwframe_transfer_data on this path. The bytes the GPU sees are");
    println!("  the same bytes the VideoToolbox decoder wrote — direct, unified memory.");
    Ok(())
}

fn bench_decode(
    path: &std::path::Path,
    n_frames: u32,
    hw: vr180_pipeline::decode::HwDecode,
) -> anyhow::Result<()> {
    let result = vr180_pipeline::decode::bench_decode_throughput(path, n_frames, hw)?;
    let ms_per_frame = result.total.as_secs_f32() * 1000.0 / result.frames.max(1) as f32;
    println!("file:        {}", path.display());
    println!("frames:      {} (requested {})", result.frames, n_frames);
    println!("decode path: {}", result.decode_path);
    println!("total:       {:.2?}", result.total);
    println!("per frame:   {ms_per_frame:.2} ms");
    println!("throughput:  {:.2} fps", result.fps());
    Ok(())
}

fn probe_eac(
    path: &std::path::Path,
    out: Option<&std::path::Path>,
    equirect: Option<&std::path::Path>,
    eye_w: u32,
    hw: vr180_pipeline::decode::HwDecode,
    lut_spec: Option<&str>,
    lut_intensity: f32,
) -> anyhow::Result<()> {
    use vr180_core::eac::{Dims, assemble_lens_a, assemble_lens_b};
    use vr180_pipeline::decode::{extract_first_stream_pair_with, probe_video};

    let probe = probe_video(path)?;
    let dims = Dims::new(probe.width, probe.height);
    println!("file:        {}", path.display());
    println!("stream:      {} × {}  ({:.3} fps, {:.2}s)",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("EAC tile_w:  {} px  (formula: (w-1920)/4)", dims.tile_w());
    println!("EAC cross:   {0} × {0}  (per lens)", dims.cross_w());
    if !dims.is_valid() {
        anyhow::bail!("stream width {} is not a valid EAC layout (need (w-1920) % 4 == 0)",
            probe.width);
    }

    if out.is_none() && equirect.is_none() {
        println!("(pass --out <png> for the raw cross pair, or --equirect <png> for the GPU projection)");
        return Ok(());
    }

    // Decode + assemble once; reuse for both output paths.
    let t0 = std::time::Instant::now();
    let pair = extract_first_stream_pair_with(path, hw)?;
    let decode_t = t0.elapsed();
    let dims = pair.dims;
    let cw = dims.cross_w() as usize;

    let t1 = std::time::Instant::now();
    let mut cross_a = vec![0u8; cw * cw * 3];
    let mut cross_b = vec![0u8; cw * cw * 3];
    assemble_lens_a(&pair.s0, &pair.s4, dims, &mut cross_a);
    assemble_lens_b(&pair.s0, &pair.s4, dims, &mut cross_b);
    let assemble_t = t1.elapsed();

    println!();
    println!("decoded streams:  {decode_t:.2?}  ({} bytes / stream)  [path: {}]",
        pair.s0.len(), pair.decode_path);
    println!("assembled crosses:{assemble_t:.2?}  (2 × {}×{})", cw, cw);

    if let Some(out) = out {
        let combined_w = cw;
        let combined_h = cw * 2;
        let mut combined = Vec::with_capacity(combined_w * combined_h * 3);
        combined.extend_from_slice(&cross_a);
        combined.extend_from_slice(&cross_b);
        let t = std::time::Instant::now();
        let img = image::RgbImage::from_raw(combined_w as u32, combined_h as u32, combined)
            .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw size mismatch"))?;
        img.save(out)?;
        println!("wrote cross PNG:  {:?}  → {}  ({} × {})",
            t.elapsed(), out.display(), combined_w, combined_h);
    }

    if let Some(eq) = equirect {
        use vr180_pipeline::gpu::Device;
        let eye_h = eye_w; // square half-equirect per eye (±90° × ±90°)
        let t = std::time::Instant::now();
        let device = Device::new()?;
        let device_t = t.elapsed();

        // Resolve LUT path: `bundled` → assets path; anything else → as-is.
        let lut = if let Some(spec) = lut_spec {
            let path = if spec == "bundled" {
                resolve_bundled_lut().ok_or_else(|| anyhow::anyhow!(
                    "--lut bundled but assets/Recommended Lut GPLOG.cube not found"
                ))?
            } else {
                std::path::PathBuf::from(spec)
            };
            let t = std::time::Instant::now();
            let lut = vr180_core::Cube3DLut::from_file(&path)?;
            println!("loaded LUT:       {}^3 from {}  ({:?})",
                lut.size, path.display(), t.elapsed());
            Some(lut)
        } else { None };

        // Eye assignment: Lens A → RIGHT, Lens B → LEFT (yaw-mod convention).
        // probe-eac is a single-frame inspection tool — no stabilization, no RS here.
        let t = std::time::Instant::now();
        let id_rot = vr180_pipeline::gpu::EquirectRotation::IDENTITY;
        let id_rs  = vr180_pipeline::gpu::EquirectRsParams::DISABLED;
        let mut left_eye = device.project_cross_to_equirect(&cross_b, dims.cross_w(), eye_w, eye_h, id_rot, id_rs)?;
        let mut right_eye = device.project_cross_to_equirect(&cross_a, dims.cross_w(), eye_w, eye_h, id_rot, id_rs)?;
        let gpu_t = t.elapsed();

        if let Some(lut) = &lut {
            let t = std::time::Instant::now();
            left_eye = device.apply_lut3d(&left_eye, eye_w, eye_h, lut, lut_intensity)?;
            right_eye = device.apply_lut3d(&right_eye, eye_w, eye_h, lut, lut_intensity)?;
            println!("applied LUT (2x): {:?}", t.elapsed());
        }

        // Stitch L | R side-by-side.
        let sbs_w = eye_w * 2;
        let sbs_h = eye_h;
        let mut sbs = vec![0u8; (sbs_w * sbs_h * 3) as usize];
        let row_l = (eye_w * 3) as usize;
        let row_sbs = (sbs_w * 3) as usize;
        for y in 0..eye_h as usize {
            sbs[y * row_sbs.. y * row_sbs + row_l]
                .copy_from_slice(&left_eye[y * row_l..y * row_l + row_l]);
            sbs[y * row_sbs + row_l..y * row_sbs + row_l * 2]
                .copy_from_slice(&right_eye[y * row_l..y * row_l + row_l]);
        }

        let t = std::time::Instant::now();
        let img = image::RgbImage::from_raw(sbs_w, sbs_h, sbs)
            .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw size mismatch"))?;
        img.save(eq)?;
        let save_t = t.elapsed();

        println!("wgpu device:      {device_t:.2?}");
        println!("gpu project (2x): {gpu_t:.2?}  (per eye: ~{:.2?})", gpu_t / 2);
        println!("wrote SBS PNG:    {save_t:.2?}  → {}  ({} × {})",
            eq.display(), sbs_w, sbs_h);
    }

    Ok(())
}

fn probe_gyro(path: &std::path::Path, do_vqf: bool) -> anyhow::Result<()> {
    use vr180_core::gyro::{parse_cori, parse_iori, quat_to_euler_zyx};
    use vr180_pipeline::decode::{extract_gpmf_stream, probe_video};

    let t0 = std::time::Instant::now();
    let probe = probe_video(path)?;
    let gpmf = extract_gpmf_stream(path)?;
    let cori = parse_cori(&gpmf);
    let iori = parse_iori(&gpmf);
    let elapsed = t0.elapsed();

    println!("file:          {}", path.display());
    println!("video:         {}×{} @ {:.3} fps, {:.2}s",
        probe.width, probe.height, probe.fps, probe.duration_sec);
    println!("GPMF stream:   {} bytes", gpmf.len());
    println!("CORI samples:  {}", cori.len());
    println!("IORI samples:  {}", iori.len());

    if let Some(&q) = cori.first() {
        println!("CORI[0]:       w={:.6} x={:.6} y={:.6} z={:.6}", q.w, q.x, q.y, q.z);
    }
    if let Some(&q) = iori.first() {
        println!("IORI[0]:       w={:.6} x={:.6} y={:.6} z={:.6}", q.w, q.x, q.y, q.z);
    }

    if !cori.is_empty() {
        let (mut rmin, mut rmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut pmin, mut pmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &q in &cori {
            let (r, p, y) = quat_to_euler_zyx(q);
            rmin = rmin.min(r); rmax = rmax.max(r);
            pmin = pmin.min(p); pmax = pmax.max(p);
            ymin = ymin.min(y); ymax = ymax.max(y);
        }
        println!("CORI Euler ranges (deg):");
        println!("  roll : [{rmin:>8.3}, {rmax:>8.3}]");
        println!("  pitch: [{pmin:>8.3}, {pmax:>8.3}]");
        println!("  yaw  : [{ymin:>8.3}, {ymax:>8.3}]");
    }
    println!("elapsed:       {elapsed:.2?}");

    if do_vqf {
        probe_gyro_vqf(path)?;
    }
    Ok(())
}

fn probe_gyro_vqf(path: &std::path::Path) -> anyhow::Result<()> {
    use vr180_core::gyro::vqf;
    use vr180_pipeline::imu::{prepare_for_vqf, MagSource, AccSource};

    let t0 = std::time::Instant::now();
    let prep = prepare_for_vqf(path)?;
    let prep_elapsed = t0.elapsed();

    let acc_label = match prep.acc_source {
        AccSource::Grav => "GRAV×9.81",
        AccSource::Raw  => "raw ACCL",
        AccSource::None => "none",
    };
    let mag_label = match prep.mag_source {
        MagSource::Mnor => "MNOR",
        MagSource::None => "none",
    };
    let mode = if prep.mag_body.is_some() { "9D" } else { "6D" };

    let t1 = std::time::Instant::now();
    let run = vqf::run(
        &prep.gyro_body,
        &prep.acc_body,
        prep.mag_body.as_deref(),
        prep.gyr_ts,
    );
    let vqf_elapsed = t1.elapsed();

    let bias_deg = run.bias_deg_s();
    let qf = run.quats.first().copied().unwrap_or(vr180_core::gyro::Quat::IDENTITY);
    let ql = run.quats.last().copied().unwrap_or(vr180_core::gyro::Quat::IDENTITY);

    println!();
    println!("VQF {mode} ({acc_label}+{mag_label})");
    println!("  gyro samples:  {}  ({:.2} Hz)", prep.gyro_body.len(), 1.0 / prep.gyr_ts);
    println!("  acc input :    {}  → resampled to {}", prep.n_acc_input, prep.acc_body.len());
    if let Some(ref m) = prep.mag_body {
        println!("  mag input :    {}  → resampled to {}", prep.n_mag_input, m.len());
    }
    println!("  bias deg/s:    [{:.3}, {:.3}, {:.3}]  σ={:.4} rad/s",
        bias_deg[0], bias_deg[1], bias_deg[2], run.bias_sigma);
    println!("  quat[ 0]:      w={:.6} x={:.6} y={:.6} z={:.6}", qf.w, qf.x, qf.y, qf.z);
    println!("  quat[-1]:      w={:.6} x={:.6} y={:.6} z={:.6}", ql.w, ql.x, ql.y, ql.z);
    println!("  prep elapsed:  {prep_elapsed:.2?}");
    println!("  vqf  elapsed:  {vqf_elapsed:.2?}");
    Ok(())
}
