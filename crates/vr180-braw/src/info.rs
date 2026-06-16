//! `braw_helper --info <file>` JSON parsing.
//!
//! Schema mirrors `braw_helper.cpp:386-404` `do_info()` output.

use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;
use std::process::{Command, Stdio};

/// Metadata extracted from a `.braw` file by `braw_helper --info`.
#[derive(Debug, Clone, Deserialize)]
pub struct BrawInfo {
    /// Frame width in pixels (track 0 if multi-track).
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Total frame count.
    pub frame_count: u64,
    /// Frame rate (e.g. 24.0, 23.976, 29.97).
    pub frame_rate: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Free-form camera model string, e.g. "Blackmagic URSA Cine
    /// Immersive".
    #[serde(default)]
    pub camera_model: String,
    /// Camera firmware version (used to detect the v7.9 readout-time
    /// bug).
    #[serde(default)]
    pub firmware_version: String,
    /// Number of video tracks in the container. `1` for single-camera
    /// shots, `2` for URSA Cine Immersive / Pyxis stereo.
    #[serde(default = "default_one")]
    pub video_track_count: u32,
    /// Total gyro sample count across the entire clip.
    #[serde(default)]
    pub gyro_sample_count: u64,
    /// Gyro sample rate in Hz (typically 5000 Hz on URSA Cine).
    #[serde(default)]
    pub gyro_sample_rate: f64,
    /// Accelerometer sample count.
    #[serde(default)]
    pub accel_sample_count: u64,
    /// Accelerometer sample rate.
    #[serde(default)]
    pub accel_sample_rate: f64,
    /// Audio sample count (PCM frames, not channel-frames).
    #[serde(default)]
    pub audio_sample_count: u64,
    /// Audio sample rate in Hz.
    #[serde(default)]
    pub audio_sample_rate: u32,
    /// Audio bit depth.
    #[serde(default)]
    pub audio_bit_depth: u32,
    /// Audio channel count.
    #[serde(default)]
    pub audio_channels: u32,
    /// Sensor readout time in milliseconds (used for per-row rolling
    /// shutter correction — currently disabled for BRAW).
    #[serde(default)]
    pub readout_ms: f64,
}

fn default_one() -> u32 { 1 }

impl BrawInfo {
    /// Run `braw_helper --info <path>` and parse the result.
    pub fn probe(path: impl AsRef<Path>) -> Result<Self> {
        let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
            expected: "helpers/bin/braw_helper".into(),
        })?;
        let out = Command::new(&helper)
            .arg("--info")
            .arg(path.as_ref())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        if !out.status.success() {
            return Err(Error::HelperFailed {
                code: out.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let stdout = std::str::from_utf8(&out.stdout)
            .map_err(|e| Error::BadOutput(format!("non-UTF-8 stdout: {e}")))?
            .trim();
        // braw_helper emits one JSON object on stdout. Some versions
        // also emit progress lines on stderr; we only consume stdout.
        let info: BrawInfo = serde_json::from_str(stdout)
            .map_err(|e| Error::BadJson(format!("stdout='{stdout}': {e}")))?;
        Ok(info)
    }

    /// True if this clip has two video tracks (stereo URSA Cine
    /// Immersive, Pyxis 12K dual-body shoot).
    pub fn is_dual_track(&self) -> bool {
        self.video_track_count >= 2
    }
}
