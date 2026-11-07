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
//! `audio_channels`, `readout_ms`, …, and `color` — the SDK's colour
//! processing (current values, ranges, lists; [`BrawColorInfo`]). See
//! [`BrawInfo`].
//!
//! ### `--decode <file> [--start N] [--count M] [--track T] [--16bit]`
//! Colour overrides ride along as flags (`--gamma`, `--wb`, `--tc-contrast`,
//! …; [`BrawColorParams`]); the zero-copy mode also takes them live as
//! `P <flag> <value>` lines on stdin ([`ZcReleaser::set_params`]).
//! - JSON header on **stderr** (one line, includes `dual_stream`,
//!   `bit_depth`, and per-track sizes if multi-video).
//! - Raw RGBA pixel data on **stdout** (BGRA from older helper builds;
//!   the header's `format` tells). 4 bytes/pixel at 8-bit, 8
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
pub mod color;
pub mod decoder;
pub mod gyro;
pub mod info;
pub mod lut;
pub mod shutters;
pub mod zc;

pub use audio::{extract_audio_to_tempfile, extract_audio_to_wav, BrawAudioHeader, TempWavPath};
pub use color::{BrawColorInfo, BrawColorParams, BrawColorValues, GenColor, GenColorView, Ranged, TonePreset, CUSTOM_GAMMA};
pub use decoder::{BrawDecoder, DecodeLook, DecodeScale};
pub use lut::extract_embedded_lut;
pub use shutters::BrawShutters;
pub use zc::{BrawZcDecoder, ZcEvent, ZcFrame, ZcHeader, ZcReleaser};
pub use gyro::{BrawGyroData, BrawGyroHeader};
pub use info::BrawInfo;

/// Crate-level error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("braw_helper binary not found at {expected} — build it with helpers/braw/build_braw.sh (macOS) or helpers/braw/build_braw.ps1 (Windows); both need the Blackmagic RAW SDK installed")]
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

/// The helper's file name on this platform (`braw_helper.exe` on Windows).
pub fn helper_file_name() -> String {
    format!("braw_helper{}", std::env::consts::EXE_SUFFIX)
}

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
        let dev = ws_root.join("helpers").join("bin").join(helper_file_name());
        if dev.is_file() {
            return Some(dev);
        }
    }
    // Release: next to current exe.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(helper_file_name());
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// A `Command` for the helper. On Windows it carries `CREATE_NO_WINDOW`:
/// `braw_helper.exe` is a console-subsystem program while the GUI has no
/// console, so without the flag every spawn (probe, gyro, LUT, decode)
/// opened a terminal window on the desktop. Every helper call in this crate
/// pipes or nulls the helper's stdio, so the missing console is never seen.
pub(crate) fn helper_command(helper: &std::path::Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(helper);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
