//! Windows CUDA ↔ Vulkan zero-copy.
//!
//! Mirrors `SLRStudioNeo::crates::mosaic-pipeline::interop` (Windows
//! arm). `cudarc` driver API attaches to FFmpeg's
//! `AVCUDADeviceContext`; the NV12 / P010 CUDA pixel buffers get
//! exported to Vulkan as external images, then handed to wgpu via
//! the wgpu-hal Vulkan escape.
//!
//! Phase 0.6.
