//! `braw_helper --shutters <file> [--step N]`: the per-frame shutter.
//!
//! The camera writes each frame's `shutter_value` as text ("7.2°" at a
//! frame rate, or "1/535"); the helper parses it to seconds and prints one
//! line per sampled frame: `<frame> <seconds> "<text>"`. Reading a frame's
//! metadata means reading the frame, so a step > 1 samples the take and
//! the lookup holds the last sampled value — auto exposure moves the
//! shutter slowly (0.4–3.3 ms over the sample clip), a few frames of hold
//! are harmless.

use crate::{Error, Result};
use std::path::Path;
use std::process::{Command, Stdio};

/// Sampled per-frame shutter times.
#[derive(Debug, Clone, Default)]
pub struct BrawShutters {
    /// `(frame_index, shutter_seconds)`, ascending frame index.
    pub samples: Vec<(u64, f64)>,
}

impl BrawShutters {
    /// Run `braw_helper --shutters <path> --step <step>`.
    pub fn extract(path: impl AsRef<Path>, step: u64) -> Result<Self> {
        let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
            expected: "helpers/bin/braw_helper".into(),
        })?;
        let out = Command::new(&helper)
            .arg("--shutters").arg(path.as_ref())
            .arg("--step").arg(step.max(1).to_string())
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .output()?;
        if !out.status.success() {
            return Err(Error::HelperFailed {
                code: out.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut samples = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if let (Some(f), Some(s)) = (it.next(), it.next()) {
                if let (Ok(f), Ok(s)) = (f.parse::<u64>(), s.parse::<f64>()) {
                    if s > 0.0 { samples.push((f, s)); }
                }
            }
        }
        samples.sort_by_key(|(f, _)| *f);
        Ok(Self { samples })
    }

    /// Shutter of `frame`: the last sample at or before it (the first one
    /// before the first sample); `None` with no samples.
    pub fn at(&self, frame: u64) -> Option<f64> {
        if self.samples.is_empty() { return None; }
        let i = self.samples.partition_point(|(f, _)| *f <= frame);
        Some(self.samples[i.saturating_sub(1)].1)
    }

    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hold_lookup() {
        let s = BrawShutters { samples: vec![(0, 0.001), (30, 0.002), (60, 0.003)] };
        assert_eq!(s.at(0), Some(0.001));
        assert_eq!(s.at(29), Some(0.001));
        assert_eq!(s.at(30), Some(0.002));
        assert_eq!(s.at(1000), Some(0.003));
        assert_eq!(BrawShutters::default().at(5), None);
    }
}
