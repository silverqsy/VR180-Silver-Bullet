//! wgpu device + queue + WGSL shader orchestration.
//!
//! Single backend, no fallbacks: Metal on macOS, Vulkan on Linux,
//! DX12 on Windows. wgpu picks at runtime.
//!
//! Phase 0.5 — `Device::new`, first kernel (`cross_remap.wgsl`),
//! CPU↔GPU upload/readback.
//!
//! Shaders live in `crates/vr180-core/src/gpu/shaders/` (will
//! be moved here in 0.5 once we know the include strategy).
