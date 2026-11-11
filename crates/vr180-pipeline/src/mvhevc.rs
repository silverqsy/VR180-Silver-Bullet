//! MV-HEVC spatial video encode (macOS).
//!
//! Apple's spatial video is one MV-HEVC stream holding both eye views plus
//! the spatial metadata (eye views, hero eye, camera baseline, horizontal
//! field of view, disparity adjustment, rectilinear projection). FFmpeg's
//! VideoToolbox encoder cannot produce it, so the frames go to the bundled
//! `mvhevc_encode` helper, which drives a multi-image VideoToolbox session
//! and writes the movie.
//!
//! The frames are the export's own encode frames — the side-by-side P010
//! surface the GPU composed — written to the helper's stdin plane by plane.
//! The helper splits them into the two views and encodes them as they are:
//! one RGB→YCbCr conversion (the GPU's), one encode, no intermediate file.
//!
//! The pipe is also the flow control: the helper stops reading while its
//! encoder is busy, the write below blocks, and the export runs at the
//! encoder's pace.

use crate::{Error, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

const HELPER: &str = "mvhevc_encode";

/// What the movie is: per-eye frame, rate, and the spatial metadata.
#[derive(Debug, Clone, Copy)]
pub struct MvHevcParams {
    /// One eye's frame; both even (4:2:0, and the seam between the eyes
    /// must fall between chroma samples).
    pub eye_w: u32,
    pub eye_h: u32,
    pub fps: f32,
    /// Average rate of the whole stream (both views).
    pub bitrate_kbps: u32,
    /// Horizontal field of view of one eye's frame, degrees.
    pub hfov_deg: f32,
    /// Distance between the two lenses, millimetres.
    pub baseline_mm: f32,
    /// Horizontal disparity adjustment as a fraction of the frame width
    /// (0.02 = 2 %); positive moves the scene away from the viewer.
    pub disparity: f32,
}

/// A running `mvhevc_encode`. Feed it frames, then [`MvHevcEncoder::finish`].
pub struct MvHevcEncoder {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stderr: Option<std::thread::JoinHandle<String>>,
    eye_w: u32,
    eye_h: u32,
    frames: u64,
    /// Rows repacked without padding, for a plane whose stride is wider
    /// than its row.
    packed: Vec<u8>,
}

fn helper_error(code: i32, stderr: impl Into<String>) -> Error {
    Error::Helper { name: HELPER.into(), code, stderr: stderr.into() }
}

impl MvHevcEncoder {
    /// Start the helper writing a video-only `.mov` at `out`.
    pub fn create(out: &Path, p: &MvHevcParams) -> Result<Self> {
        if p.eye_w == 0 || p.eye_h == 0 || p.eye_w % 2 != 0 || p.eye_h % 2 != 0 {
            return Err(Error::Ffmpeg(format!(
                "spatial video needs an even per-eye frame, got {}×{}", p.eye_w, p.eye_h)));
        }
        let helper = crate::helpers::locate_helper(HELPER).ok_or_else(|| helper_error(
            -1, "binary not found in helpers/bin/ or next to the app. Run ./helpers/build_swift.sh"))?;
        let (fps_num, fps_den) = crate::encode::approx_rational(p.fps);
        let mut cmd = Command::new(&helper);
        cmd.arg("--width").arg(p.eye_w.to_string())
            .arg("--height").arg(p.eye_h.to_string())
            .arg("--fps-num").arg(fps_num.to_string())
            .arg("--fps-den").arg(fps_den.to_string())
            .arg("--bitrate").arg((p.bitrate_kbps.max(1) as u64 * 1000).to_string())
            .arg("--hfov-deg").arg(format!("{:.3}", p.hfov_deg))
            .arg("--baseline-mm").arg(format!("{:.3}", p.baseline_mm))
            .arg("--disparity").arg(format!("{:.4}", p.disparity))
            .arg("--projection").arg("rect")
            .arg("--hero").arg("left")
            .arg("--output").arg(out);
        tracing::info!(
            helper = %helper.display(), out = %out.display(),
            "mvhevc: {}×{} per eye @ {fps_num}/{fps_den} fps, {} kbps, hfov {:.1}°, baseline {:.1} mm, disparity {:+.2} %",
            p.eye_w, p.eye_h, p.bitrate_kbps, p.hfov_deg, p.baseline_mm, p.disparity * 100.0,
        );
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| helper_error(-1, format!("spawn failed: {e}")))?;
        let stdin = child.stdin.take();
        let stderr = child.stderr.take().map(|h| std::thread::spawn(move || {
            // Into the log as it comes, and kept for the error message.
            let mut captured = String::new();
            for line in BufReader::new(h).lines().map_while(|r| r.ok()) {
                if line.starts_with("ERROR") {
                    tracing::error!(target: "mvhevc_encode", "{line}");
                } else {
                    tracing::info!(target: "mvhevc_encode", "{line}");
                }
                captured.push_str(&line);
                captured.push('\n');
            }
            captured
        }));
        Ok(Self {
            child: Some(child), stdin, stderr,
            eye_w: p.eye_w, eye_h: p.eye_h,
            frames: 0, packed: Vec::new(),
        })
    }

    /// Frames handed to the helper so far.
    pub fn frames(&self) -> u64 { self.frames }

    /// Encode one side-by-side P010 frame (`2·eye_w × eye_h`, the left eye
    /// in the left half) straight from its encode surface. Every GPU write
    /// to the surface must have finished — the export's compose step waits
    /// for the device before it returns.
    pub fn encode_pixel_buffer_p010(
        &mut self,
        pb: &crate::interop_macos::EncodePixelBufferP010,
    ) -> Result<()> {
        if pb.width != self.eye_w * 2 || pb.height != self.eye_h {
            return Err(Error::Ffmpeg(format!(
                "mvhevc: frame is {}×{}, the encoder was opened for {}×{}",
                pb.width, pb.height, self.eye_w * 2, self.eye_h)));
        }
        // Both planes have rows of `4·eye_w` bytes: 2·eye_w 16-bit luma
        // samples, or eye_w 16-bit Cb,Cr pairs.
        let row_bytes = self.eye_w as usize * 4;
        for (plane, rows) in [(0usize, self.eye_h as usize), (1, self.eye_h as usize / 2)] {
            let wrote = pb.iosurface.with_plane_bytes(plane, |bytes, stride| {
                if stride < row_bytes || bytes.len() < stride * (rows - 1) + row_bytes {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                        "plane {plane} is {} bytes at stride {stride}, expected rows of {row_bytes}",
                        bytes.len())));
                }
                let Some(stdin) = self.stdin.as_mut() else {
                    return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "encoder closed"));
                };
                if stride == row_bytes {
                    stdin.write_all(&bytes[..row_bytes * rows])
                } else {
                    self.packed.clear();
                    self.packed.reserve(row_bytes * rows);
                    for r in 0..rows {
                        self.packed.extend_from_slice(&bytes[r * stride..r * stride + row_bytes]);
                    }
                    stdin.write_all(&self.packed)
                }
            });
            match wrote {
                Some(Ok(())) => {}
                Some(Err(e)) => return Err(self.abort(e)),
                None => return Err(Error::Ffmpeg(format!(
                    "mvhevc: could not read plane {plane} of the encode surface"))),
            }
        }
        self.frames += 1;
        Ok(())
    }

    /// A failed write means the helper has gone: collect why.
    fn abort(&mut self, cause: std::io::Error) -> Error {
        let (code, stderr) = self.reap();
        let stderr = if stderr.trim().is_empty() { format!("write to the encoder failed: {cause}") } else { stderr };
        helper_error(code, stderr)
    }

    /// Close the pipe, wait for the helper, return its exit code and output.
    fn reap(&mut self) -> (i32, String) {
        self.stdin = None; // EOF: the helper finishes the movie and exits
        let code = match self.child.take() {
            Some(mut c) => c.wait().ok().and_then(|s| s.code()).unwrap_or(-1),
            None => -1,
        };
        let stderr = self.stderr.take().and_then(|t| t.join().ok()).unwrap_or_default();
        (code, stderr)
    }

    /// End the stream: the helper flushes its encoder and closes the movie.
    pub fn finish(mut self) -> Result<()> {
        let frames = self.frames;
        let (code, stderr) = self.reap();
        if code != 0 {
            return Err(helper_error(code, stderr));
        }
        tracing::info!("mvhevc: movie closed, {frames} frames");
        Ok(())
    }
}

impl Drop for MvHevcEncoder {
    fn drop(&mut self) {
        // Dropped without `finish` (an error upstream): still let the helper
        // close what it has and go away, so no process or half-open file
        // is left behind.
        if self.child.is_some() {
            let _ = self.reap();
        }
    }
}
