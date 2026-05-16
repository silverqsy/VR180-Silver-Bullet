//! ffmpeg-next decode wrapper.
//!
//! Phase 0.4 — software decode of the .360 file's two HEVC streams
//! (s0 + s4), producing aligned frames into a host `Rgba16Float`
//! buffer for assembly.
//!
//! Phase 0.6 — hardware decode (VideoToolbox / NVDEC) with
//! IOSurface ↔ Metal or CUDA ↔ Vulkan interop into wgpu.
//!
//! NOTE: this replaces the Python `_decode_360_pyav` + the legacy
//! subprocess ffmpeg fallback path. One implementation, no fallback
//! to a different code path on failure — failures bubble as
//! `pipeline::Error::Ffmpeg(_)`.
