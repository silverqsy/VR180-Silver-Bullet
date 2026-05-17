//! Windows CUDA ↔ Vulkan zero-copy.
//!
//! `cudarc` driver API attaches to FFmpeg's `AVCUDADeviceContext`;
//! the NV12 / P010 CUDA pixel buffers get exported to Vulkan as
//! external images, then handed to wgpu via the wgpu-hal Vulkan escape.
//! Mac equivalent (already shipped in 0.6.5) lives in `interop_macos`.
//!
//! Phase 0.6.8.
