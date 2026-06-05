//! VR180 Silver Bullet Neo — I/O + GPU pipeline.
//!
//! All side-effects live here: ffmpeg-next decode/encode, wgpu
//! kernels, Swift helper subprocess spawn. `vr180-core` stays
//! pure-Rust and headless.
//!
//! Phase 0.1: skeleton.

#![warn(missing_debug_implementations)]

pub mod audio;
pub mod braw_imu;
pub mod decode;
pub mod dji_imu;
pub mod encode;
pub mod fisheye_decode;
pub mod fisheye_export;
pub mod gpu;
pub mod helpers;
pub mod imu;
pub mod panomap;
pub mod render;
pub mod source_kind;
pub mod spherical_inject;

pub use source_kind::SourceKind;

#[cfg(target_os = "macos")]
pub mod interop_macos;
#[cfg(target_os = "macos")]
pub use interop_macos::{
    RetainedIOSurface, IOSurfaceNv12Descriptor, IOSurfacePlaneTexture,
    extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
};

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
