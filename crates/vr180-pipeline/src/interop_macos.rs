//! macOS IOSurface ↔ MTLTexture zero-copy.
//!
//! Mirrors `SLRStudioNeo::crates::mosaic-pipeline::interop_macos`:
//! `RetainedIOSurface` RAII, FFI to CoreFoundation / CoreVideo /
//! IOSurface, objc `msg_send!` for
//! `newTextureWithDescriptor:iosurface:plane:`, wgpu-hal Metal
//! escape via `Device::texture_from_raw`.
//!
//! Phase 0.6.
