//! VR180 Silver Bullet — I/O + GPU pipeline.
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
pub mod raw_develop;
pub mod ljpeg;
pub mod dng_write;
pub mod encode;
pub mod fisheye_decode;
pub mod fisheye_export;
pub mod stereo_align;
pub mod gpu;
pub mod helpers;
pub mod imu;
pub mod insv_imu;
pub mod kandao_imu;
pub mod panomap;
pub mod render;
pub mod source_kind;
pub mod spherical_inject;
pub mod still_export;

pub use source_kind::SourceKind;

#[cfg(target_os = "macos")]
pub mod interop_macos;
#[cfg(target_os = "macos")]
pub use interop_macos::{
    RetainedIOSurface, IOSurfaceNv12Descriptor, IOSurfacePlaneTexture,
    extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
};

// In-process VideoToolbox temporal noise reduction (objc2 FFI, no Swift
// helper). macOS-only; the consuming paths are gated the same way.
#[cfg(target_os = "macos")]
pub mod vt_denoise;

#[cfg(target_os = "windows")]
pub mod interop_windows;

// Vulkan Video decode: VkDevices shared by wgpu and FFmpeg's Vulkan decoder
// (the export's and the app's own; the ProRes encoder keeps its own device).
#[cfg(target_os = "windows")]
pub mod vk_shared;

#[cfg(target_os = "windows")]
pub mod nvenc_cuda;

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
