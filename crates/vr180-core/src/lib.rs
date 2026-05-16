//! VR180 Silver Bullet Neo — headless core.
//!
//! Pure Rust, no I/O, no GPU. All camera-format math, gyro fusion,
//! color math, and atom writers live here so they can be tested
//! without spinning up wgpu or libav.
//!
//! Phase 0.1: skeleton only. See [docs/ROADMAP.md] for the phased
//! plan; each module's stub points to the Python implementation it
//! will eventually replace.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod project;
pub mod gyro;
pub mod eac;
pub mod geoc;
pub mod color;
pub mod atoms;

/// Top-level error type for the core crate. Pipeline + render
/// layers map their own errors into / from this via `From` impls.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid project config: {0}")]
    InvalidProject(String),

    #[error("gpmf parse error: {0}")]
    GpmfParse(String),

    #[error("invalid EAC stream dimensions: {0}")]
    InvalidDims(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
