//! Color pipeline math.
//!
//! All ops operate on `Rgba16Float` (kept consistent with the
//! Python app's true-10-bit pipeline; no 8-bit intermediates).
//!
//! Order (from Python `apply_export_post`):
//!   1. tonal zones (smoothstep masks: shadow/highlight, pre-LUT)
//!   2. temperature / tint
//!   3. CDL (lift / gamma / gain / saturation)
//!   4. 3D LUT (trilinear .cube)
//!   5. mid-detail clarity (downsample-blur-upsample)
//!
//! Phase 0.7 — port each op as a WGSL shader + Rust uniform builder.
