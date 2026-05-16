//! Render orchestrator: decode → assemble → kernel → encode loop.
//!
//! This is the rough equivalent of the Python `VideoProcessor.run`
//! method but rewritten as a typed state machine without the
//! threading.Queue gymnastics. wgpu's command encoder gives us
//! natural pipelining between frame N's read and frame N-1's
//! kernel + encode.
//!
//! Phase 0.4 — CPU-baseline single-frame loop.
//! Phase 0.5 — GPU kernel.
//! Phase 0.6 — hardware decode + interop, real fps targets.
