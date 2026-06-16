//! Fisheye projection math, camera presets, and Gyroflow JSON loader.
//!
//! This crate is the fisheye-footage counterpart to `vr180-core`'s
//! EAC / GEOC math. The two camera-format families both feed into the
//! same downstream pipeline (decode → IMU → GPU project → encode), but
//! the optics math at the front is fundamentally different:
//!
//! - GoPro Max `.360`: equirectangular cube faces (EAC) packed across
//!   two HEVC streams. Already corrected for fisheye distortion by the
//!   camera firmware.
//! - Everything else (DJI Osmo `.osv`, SBS fisheye `.mp4`, Blackmagic
//!   `.braw`): raw fisheye on the sensor, projected via the
//!   Kannala-Brandt forward model
//!   `r = fx · θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)`.
//!   We undo it by inverting that polynomial (Newton-Raphson + cubic
//!   Hermite extension past the calibrated angle) to get the world ray
//!   for any pixel.
//!
//! ## Modules
//!
//! - [`projection`] — Kannala-Brandt forward and inverse. CPU reference
//!   implementation used by tests, FOV derivation, and the GUI's
//!   live-preview sliders. The hot per-pixel path lives in the WGSL
//!   shader in `vr180-render`; this module is the source of truth that
//!   the shader is validated against.
//! - [`calib`] — `FisheyeCalibration` struct (fx/fy/cx/cy + k1..k4 +
//!   working resolution) and the Gyroflow lens-profile JSON loader.
//! - [`presets`] — Curated camera catalog: DJI Osmo, Insta360,
//!   Blackmagic Pyxis, URSA Cine Immersive, etc. Each preset holds a
//!   default `FisheyeCalibration` + a human-readable name + the native
//!   sensor resolution it was calibrated against.
//!
//! Phase 0.13 — initial scaffold. Real shader port lands in 0.14.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod calib;
pub mod dji_osv;
pub mod presets;
pub mod projection;

pub use calib::{FisheyeCalibration, GyroflowLensProfile};
pub use dji_osv::{DjiLensCalib, DjiOsvImu};
pub use presets::{CameraPreset, presets};

/// Top-level error type. Pipeline layer maps via `From`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid calibration: {0}")]
    InvalidCalibration(String),

    #[error("gyroflow JSON parse error: {0}")]
    GyroflowJson(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
