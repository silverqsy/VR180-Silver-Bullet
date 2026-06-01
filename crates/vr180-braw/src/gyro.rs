//! `braw_helper --gyro <file>` parser.
//!
//! Reads the JSON header on stderr + the float32 interleaved samples
//! on stdout. Output schema mirrors `braw_helper.cpp:670-673`.

use crate::{Error, Result};
use serde::Deserialize;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

/// JSON header emitted on stderr by `braw_helper --gyro`.
#[derive(Debug, Clone, Deserialize)]
pub struct BrawGyroHeader {
    pub gyro_sample_count: u64,
    pub gyro_sample_rate: f64,
    pub accel_sample_count: u64,
    pub accel_sample_rate: f64,
    /// `min(gyro_sample_count, accel_sample_count)` — the actual
    /// number of paired samples on stdout.
    pub sample_count: u64,
    /// Video frame rate of the clip (for time-base conversion).
    pub frame_rate: f64,
}

/// Decoded gyro+accel time series.
#[derive(Debug, Clone)]
pub struct BrawGyroData {
    pub header: BrawGyroHeader,
    /// Gyro in rad/s, shape `(sample_count, 3)` flattened row-major
    /// `[gx_0, gy_0, gz_0, gx_1, …]`. Length is `3 * sample_count`.
    pub gyro: Vec<f32>,
    /// Accelerometer in m/s², same layout as `gyro`.
    pub accel: Vec<f32>,
}

impl BrawGyroData {
    /// Number of paired samples (gyro & accel).
    pub fn len(&self) -> usize {
        self.header.sample_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sampling period between consecutive samples (1 / gyro_rate).
    pub fn sample_period_s(&self) -> f64 {
        if self.header.gyro_sample_rate > 0.0 {
            1.0 / self.header.gyro_sample_rate
        } else {
            0.0
        }
    }

    /// Run `braw_helper --gyro <path>` and parse the result into a
    /// fully-buffered struct. Uses synchronous spawn — gyro data is
    /// small (a few hundred KB at 5 kHz × seconds), no streaming needed.
    pub fn extract(path: impl AsRef<Path>) -> Result<Self> {
        let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
            expected: "helpers/bin/braw_helper".into(),
        })?;
        let mut child = Command::new(&helper)
            .arg("--gyro")
            .arg(path.as_ref())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("stdout piped");
        let mut stderr = child.stderr.take().expect("stderr piped");

        // Read both streams to EOF in parallel. The header is small
        // and arrives early; the binary is the bulk of stdout.
        let stderr_thread = std::thread::spawn(move || -> Result<String> {
            let mut s = String::new();
            stderr.read_to_string(&mut s)?;
            Ok(s)
        });

        let mut stdout_bytes: Vec<u8> = Vec::new();
        stdout.read_to_end(&mut stdout_bytes)?;

        let status = child.wait()?;
        let stderr_text = stderr_thread.join().expect("stderr thread panicked")?;

        if !status.success() {
            return Err(Error::HelperFailed {
                code: status.code().unwrap_or(-1),
                stderr: stderr_text,
            });
        }

        // The JSON header is the FIRST non-empty line of stderr.
        // braw_helper also logs progress on stderr; the JSON should be
        // first.
        let header_line = stderr_text
            .lines()
            .find(|l| !l.trim().is_empty() && l.trim_start().starts_with('{'))
            .ok_or_else(|| {
                Error::BadOutput(format!(
                    "no JSON header on stderr. Full stderr: {stderr_text}"
                ))
            })?;
        let header: BrawGyroHeader = serde_json::from_str(header_line)
            .map_err(|e| Error::BadJson(format!("header='{header_line}': {e}")))?;

        // 6 × float32 per sample = 24 bytes.
        let expected_bytes = (header.sample_count as usize) * 24;
        if stdout_bytes.len() < expected_bytes {
            return Err(Error::BadOutput(format!(
                "stdout shorter than header claims: got {} bytes, want {} ({} samples × 24)",
                stdout_bytes.len(), expected_bytes, header.sample_count
            )));
        }

        // De-interleave into gyro / accel.
        let n = header.sample_count as usize;
        let mut gyro = vec![0f32; n * 3];
        let mut accel = vec![0f32; n * 3];
        for i in 0..n {
            let off = i * 24;
            // Read 6 little-endian f32s.
            let g0 = f32::from_le_bytes(stdout_bytes[off..off + 4].try_into().unwrap());
            let g1 = f32::from_le_bytes(stdout_bytes[off + 4..off + 8].try_into().unwrap());
            let g2 = f32::from_le_bytes(stdout_bytes[off + 8..off + 12].try_into().unwrap());
            let a0 = f32::from_le_bytes(stdout_bytes[off + 12..off + 16].try_into().unwrap());
            let a1 = f32::from_le_bytes(stdout_bytes[off + 16..off + 20].try_into().unwrap());
            let a2 = f32::from_le_bytes(stdout_bytes[off + 20..off + 24].try_into().unwrap());
            gyro[i * 3 + 0] = g0;
            gyro[i * 3 + 1] = g1;
            gyro[i * 3 + 2] = g2;
            accel[i * 3 + 0] = a0;
            accel[i * 3 + 1] = a1;
            accel[i * 3 + 2] = a2;
        }

        Ok(BrawGyroData { header, gyro, accel })
    }
}
