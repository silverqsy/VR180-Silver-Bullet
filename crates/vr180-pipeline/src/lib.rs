//! VR180 Silver Bullet Neo — I/O + GPU pipeline.
//!
//! All side-effects live here: ffmpeg-next decode/encode, wgpu
//! kernels, Swift helper subprocess spawn. `vr180-core` stays
//! pure-Rust and headless.
//!
//! Phase 0.1: skeleton.

#![warn(missing_debug_implementations)]

pub mod decode;
pub mod encode;
pub mod gpu;
pub mod helpers;
pub mod imu;
pub mod render;

#[cfg(target_os = "macos")]
pub mod interop_macos;

#[cfg(target_os = "windows")]
pub mod interop_windows;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("core: {0}")]
    Core(#[from] vr180_core::Error),

    #[error("ffmpeg: {0}")]
    Ffmpeg(String),

    #[error("wgpu: {0}")]
    Wgpu(String),

    #[error("helper '{name}' failed (exit {code}): {stderr}")]
    Helper { name: String, code: i32, stderr: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
