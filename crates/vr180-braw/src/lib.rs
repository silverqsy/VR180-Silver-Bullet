//! Blackmagic RAW (`.braw`) decoder wrapper.
//!
//! `vr180-braw` is a thin Rust shell around the C++ `braw_helper`
//! binary in `helpers/bin/braw_helper`. The helper does the heavy
//! lifting (Blackmagic RAW SDK calls, color processing, multi-track
//! stitching); this crate handles process spawning, JSON parsing, and
//! presents a Rust-friendly API to the rest of the pipeline.
//!
//! Why a subprocess and not bindgen? The BlackmagicRAW SDK is C++ with
//! COM-like reference counting (`AddRef` / `Release`), CFRunLoop
//! callbacks, and a vendor-supplied dispatch table file
//! (`BlackmagicRawAPIDispatch.cpp`) that's `#include`d into the
//! application code. Wrapping it natively would be ~10× the effort vs.
//! the C++ glue already shipping in the Python app. The subprocess
//! pattern is mature — the Python app has used it in production for
//! over a year.
//!
//! ## Helper interface
//!
//! ### `--info <file>`
//! Returns one JSON line on stdout with `width`, `height`,
//! `frame_count`, `frame_rate`, `duration`, `camera_model`,
//! `firmware_version`, `video_track_count`, `gyro_sample_count`,
//! `gyro_sample_rate`, `accel_sample_count`, `accel_sample_rate`,
//! `audio_sample_count`, `audio_sample_rate`, `audio_bit_depth`,
//! `audio_channels`, `readout_ms`. See [`BrawInfo`].
//!
//! ### `--decode <file> [--start N] [--count M] [--track T] [--16bit]`
//! - JSON header on **stderr** (one line, includes `dual_stream`,
//!   `bit_depth`, and per-track sizes if multi-video).
//! - Raw BGRA pixel data on **stdout**. 4 bytes/pixel at 8-bit, 8
//!   bytes/pixel at 16-bit. Multi-track (no `--track`) writes
//!   side-by-side at `max(h0, h1)` height with zero-padding.
//!
//! ### `--gyro <file>`
//! - JSON header on **stderr** with `gyro_sample_count`,
//!   `gyro_sample_rate`, `accel_sample_count`, `accel_sample_rate`,
//!   `sample_count`, `frame_rate`.
//! - Binary float32 samples on **stdout**, interleaved
//!   `[gx, gy, gz, ax, ay, az]` (24 B/sample). Units: rad/s for gyro,
//!   m/s² for accel. Samples are paired by index — gyro/accel come
//!   off the camera at the same rate.
//!
//! ### `--audio <file>`
//! - JSON header on stderr (`sample_rate`, `bit_depth`, `channels`,
//!   `sample_count`, `data_bytes`).
//! - Full RIFF/WAV stream on stdout.

#![warn(missing_debug_implementations)]

pub mod audio;
pub mod decoder;
pub mod gyro;
pub mod info;

pub use audio::{extract_audio_to_tempfile, extract_audio_to_wav, BrawAudioHeader, TempWavPath};
pub use decoder::BrawDecoder;
pub use gyro::{BrawGyroData, BrawGyroHeader};
pub use info::BrawInfo;

/// Crate-level error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("braw_helper binary not found at {expected}")]
    HelperMissing { expected: String },

    #[error("braw_helper exited with code {code}: {stderr}")]
    HelperFailed { code: i32, stderr: String },

    #[error("braw_helper produced malformed JSON: {0}")]
    BadJson(String),

    #[error("braw_helper produced unexpected output: {0}")]
    BadOutput(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Locate the `braw_helper` binary.
///
/// Mirrors the Swift-helper logic in `vr180-pipeline::helpers`:
/// 1. `helpers/bin/braw_helper` relative to the workspace root (dev).
/// 2. Next to the running executable (release bundle).
///
/// Returns `None` if neither exists. Callers should surface a clear
/// "install Blackmagic RAW SDK + run helpers/braw/build_braw.sh"
/// message when they get `None`.
pub fn locate_helper() -> Option<std::path::PathBuf> {
    // Dev path: walk up from CARGO_MANIFEST_DIR to workspace root.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(ws_root) = manifest.parent().and_then(|p| p.parent()) {
        let dev = ws_root.join("helpers").join("bin").join("braw_helper");
        if dev.is_file() {
            return Some(dev);
        }
    }
    // Release: next to current exe.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("braw_helper");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}
