//! VR180 Silver Bullet — I/O + GPU pipeline.
//!
//! All side-effects live here: ffmpeg-next decode/encode, wgpu
//! kernels, Swift helper subprocess spawn. `vr180-core` stays
//! pure-Rust and headless.
//!
//! Phase 0.1: skeleton.

#![warn(missing_debug_implementations)]

pub mod audio;
pub mod bayer;
pub mod braw_imu;
pub mod circles;
pub mod clip_stats;
pub mod decode;
pub mod dji_imu;
pub mod raw_archive;
pub mod raw_chroma;
pub mod raw_color;
pub mod raw_develop;
pub mod raw_export;
pub mod raw_global;
pub mod raw_gpu;
pub mod raw_local;
pub mod raw_sharpen;
pub mod raw_tone;
pub mod tone_gpu;
pub mod tone_linear;
pub mod ljpeg;
pub mod dng_write;
pub mod encode;
pub mod enhanced_dng;
pub mod fisheye_decode;
pub mod fisheye_export;
pub mod frame_timeline;
pub mod stereo_align;
pub mod gpu;
pub mod helpers;
pub mod imu;
pub mod insv_imu;
pub mod join;
pub mod kandao_imu;
pub mod lens_model;
pub mod lens_frame;
pub mod panomap;
pub mod render;
mod shader_compose;
pub mod sensor_dng;
pub mod source_kind;
pub mod spherical_inject;
pub mod still_export;
mod swscale;
pub mod x6_pair;
pub mod x6_raw;

pub use source_kind::SourceKind;

#[cfg(target_os = "macos")]
pub mod interop_macos;
#[cfg(target_os = "macos")]
pub use interop_macos::{
    RetainedIOSurface, IOSurfaceNv12Descriptor, IOSurfacePlaneTexture,
    extract_iosurface_from_vt_frame, wgpu_texture_from_iosurface_plane,
};

// JPEG XL through ImageIO, for a raw converter's enhanced frames
// (`enhanced_dng`); the one caller is gated the same way.
#[cfg(target_os = "macos")]
mod jxl_macos;

// In-process VideoToolbox temporal noise reduction (objc2 FFI, no Swift
// helper). macOS-only; the consuming paths are gated the same way.
#[cfg(target_os = "macos")]
pub mod vt_denoise;

// MV-HEVC spatial video encode through the bundled `mvhevc_encode` helper
// (VideoToolbox). macOS-only; `encode::ExportEncoder` carries the gated arm.
#[cfg(target_os = "macos")]
pub mod mvhevc;

#[cfg(target_os = "windows")]
pub mod interop_windows;

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
