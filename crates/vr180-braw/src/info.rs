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
    /// Sensor readout time in milliseconds: `sensor_area_captured` rows ×
    /// `sensor_line_time` (µs) from the clip metadata (PYXIS 12K: 3 µs ×
    /// 6480 = 19.44 ms at 12288×6480 / 50 fps, 15.34 ms at ×5112 / 60).
    /// 0 when the file states no line time.
    #[serde(default)]
    pub readout_ms: f64,
    /// Sensor scan-line time, µs (`sensor_line_time`); 0 if absent.
    #[serde(default)]
    pub sensor_line_time_us: f64,
    /// `sensor_area_captured` = [w, h] of the sensor window read out (the
    /// frame); [0, 0] if absent.
    #[serde(default)]
    pub sensor_area_captured: [f64; 2],
    /// `crop_origin` = [x, y] of that window; the PYXIS 12K writes [24, 12]
    /// for both the 6480- and the 5112-row windows, so it is NOT the
    /// window's position on the sensor (assume the windows are centred).
    #[serde(default)]
    pub crop_origin: [f64; 2],
    /// `shutter_type`: "Angle" or "Speed".
    #[serde(default)]
    pub shutter_type: String,
    /// The first sampled frame's `shutter_value` text ("7.2°", "1/535").
    #[serde(default)]
    pub shutter_text: String,
    /// Shutter in seconds over five frames sampled across the clip —
    /// min / max; the camera's auto exposure varies it within a take
    /// (0.4–3.3 ms on the sample clip). Per-frame values: [`crate::shutters`].
    #[serde(default)]
    pub shutter_s_min: f64,
    #[serde(default)]
    pub shutter_s_max: f64,
    /// As-shot colour: `viewing_gamma` ("Blackmagic Design Film"),
    /// `viewing_gamut` ("Blackmagic Design"), `viewing_bmdgen` (5).
    #[serde(default)]
    pub viewing_gamma: String,
    #[serde(default)]
    pub viewing_gamut: String,
    #[serde(default)]
    pub viewing_bmdgen: f64,
    /// Title of the clip's embedded post-3D-LUT ("Gen 5 Film to Extended
    /// Video"); empty if none. Dump it with [`crate::lut::extract_embedded_lut`].
    #[serde(default)]
    pub embedded_lut: String,
    /// The SDK pipeline the helper decodes with ("cpu", "metal", …), when
    /// the helper reports it.
    #[serde(default)]
    pub pipeline: String,
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

    /// Readout (rolling-shutter sweep) time in seconds, `None` when the file
    /// states no line time.
    pub fn readout_s(&self) -> Option<f64> {
        (self.readout_ms > 0.0).then_some(self.readout_ms * 1e-3)
    }

    /// Size of ONE eye as RECORDED: a dual-track clip's track (the helper
    /// composes the two side by side), a single-track clip's half — a
    /// VR180 lens on one sensor puts its two circles side by side.
    pub fn raw_eye_dims(&self) -> (u32, u32) {
        if self.is_dual_track() { (self.width, self.height) } else { (self.width / 2, self.height) }
    }

    /// The composed frame the helper emits (both eyes), as recorded.
    pub fn raw_frame_dims(&self) -> (u32, u32) {
        if self.is_dual_track() { (self.width * 2, self.height) } else { (self.width, self.height) }
    }

    /// Size of ONE eye as the app sees it: the recorded eye fitted on a
    /// SQUARE canvas of its own width (the two eyes side by side make a 2:1
    /// frame) — a shorter eye (the 12288×5112 PYXIS window: 2.4:1) is
    /// padded black top and bottom, a taller one (12288×6480, 17:9) is
    /// cropped top and bottom. Lens parameters, the Override UI and the
    /// "fisheyecenter" web tool all speak this canvas; [`BrawFisheyeIter`]
    /// (in the pipeline crate) emits it.
    pub fn eye_dims(&self) -> (u32, u32) {
        let (w, _) = self.raw_eye_dims();
        (w, w)
    }

    /// The 2:1 canvas frame (both eyes).
    pub fn frame_dims(&self) -> (u32, u32) {
        let (w, h) = self.eye_dims();
        (w * 2, h)
    }

    /// Rows added above (and below) the recorded eye to reach the canvas —
    /// negative when the canvas CROPS rows instead.
    pub fn canvas_pad_rows(&self) -> i32 {
        let (_, raw_h) = self.raw_eye_dims();
        let (_, canvas_h) = self.eye_dims();
        (canvas_h as i32 - raw_h as i32) / 2
    }
}
