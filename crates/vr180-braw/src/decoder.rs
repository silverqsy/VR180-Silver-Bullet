//! `braw_helper --decode` streaming wrapper.
//!
//! Spawns `braw_helper` and exposes the BGRA-on-stdout byte stream as
//! a frame iterator. JSON header from stderr is parsed eagerly so the
//! consumer knows the per-frame byte count up front.
//!
//! Multi-track files (URSA Cine Immersive stereo, Pyxis 12K stereo)
//! emit a single side-by-side BGRA buffer per frame when `--track` is
//! omitted. That's the same layout the rest of the pipeline expects
//! from the SBS-fisheye path, so the downstream code stays uniform.

use crate::{Error, Result};
use serde::Deserialize;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// JSON header on stderr from `--decode`.
///
/// Single-track shape (`braw_helper.cpp:588`):
/// ```text
/// {"width":..,"height":..,"frames":..,"dual_stream":false,"bit_depth":8|16}
/// ```
///
/// Dual-track shape (`braw_helper.cpp:525`) also has
/// `track0_width/height` and `track1_width/height`. The composed
/// frame is `(track0_width + track1_width) × max(track0_height, track1_height)`.
#[derive(Debug, Clone, Deserialize)]
pub struct DecodeHeader {
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    #[serde(default)]
    pub dual_stream: bool,
    #[serde(default = "default_bit_depth")]
    pub bit_depth: u32,
    #[serde(default)]
    pub track0_width: Option<u32>,
    #[serde(default)]
    pub track0_height: Option<u32>,
    #[serde(default)]
    pub track1_width: Option<u32>,
    #[serde(default)]
    pub track1_height: Option<u32>,
    /// The pipeline the helper decodes this stream with ("metal" / "cpu").
    #[serde(default)]
    pub pipeline: Option<String>,
    /// The GPU's name when the pipeline is a GPU one.
    #[serde(default)]
    pub gpu_device: Option<String>,
}

fn default_bit_depth() -> u32 { 8 }

/// Decode resolution: the SDK decodes a BRAW frame at a fraction of its
/// size from a fraction of the bit stream (`--scale`). The fraction is the
/// SDK's choice per clip — on the 12K PYXIS "half" is 8192 wide (2/3),
/// "quarter" 4096 (1/3), "eighth" 2048 (1/6) — so consumers take the size
/// from the decode header, never from [`Self::divisor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeScale { #[default] Full, Half, Quarter, Eighth }

impl DecodeScale {
    pub fn arg(self) -> &'static str {
        match self { Self::Full => "full", Self::Half => "half", Self::Quarter => "quarter", Self::Eighth => "eighth" }
    }
    /// NOMINAL divisor on each dimension (the SDK may divide by less).
    pub fn divisor(self) -> u32 {
        match self { Self::Full => 1, Self::Half => 2, Self::Quarter => 4, Self::Eighth => 8 }
    }
    /// The coarsest scale whose eye is still at least `want` pixels wide by
    /// the nominal divisor — conservative when the SDK divides by less.
    pub fn for_eye_width(native_eye_w: u32, want: u32) -> Self {
        for s in [Self::Eighth, Self::Quarter, Self::Half] {
            if native_eye_w / s.divisor() >= want { return s; }
        }
        Self::Full
    }
}

/// The helper's colour presets (`--look`): `Film` = as shot (Blackmagic
/// Design Film Gen 5 log in the BMD gamut), the others ask the SDK for a
/// display-referred render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeLook { #[default] Film, Video, Extended, Rec709 }

impl DecodeLook {
    pub fn arg(self) -> &'static str {
        match self { Self::Film => "film", Self::Video => "video", Self::Extended => "extended", Self::Rec709 => "rec709" }
    }
}

impl DecodeHeader {
    /// Bytes per pixel (4 for 8-bit BGRA, 8 for 16-bit).
    pub fn bytes_per_pixel(&self) -> usize {
        match self.bit_depth {
            16 => 8,
            _ => 4,
        }
    }

    /// Bytes per frame.
    pub fn bytes_per_frame(&self) -> usize {
        (self.width as usize) * (self.height as usize) * self.bytes_per_pixel()
    }
}

/// Color-processing knobs forwarded to `braw_helper`. All optional.
///
/// Field names match the CLI flags 1:1, see `braw_helper.cpp:833-858`.
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// Output 16-bit BGRA instead of 8-bit (for 10-bit MV-HEVC export).
    pub bit_16: bool,
    /// Restrict to one video track (0 or 1). When `None`, multi-track
    /// is composed side-by-side.
    pub track: Option<u32>,
    /// Starting frame (0-based).
    pub start_frame: Option<u64>,
    /// Number of frames to decode. `None` = to end of clip.
    pub count: Option<u64>,
    /// `--gamma <BMD_GAMMA_*>` enum name.
    pub gamma: Option<String>,
    /// `--gamut <BMD_GAMUT_*>` enum name.
    pub gamut: Option<String>,
    /// `--wb <kelvin>` white balance temperature.
    pub wb: Option<i32>,
    /// `--tint <int>`.
    pub tint: Option<i32>,
    /// `--exposure <ev>` stops above/below.
    pub exposure: Option<f32>,
    /// `--iso <int>`.
    pub iso: Option<i32>,
    /// `--scale`: decode at a fraction of the recorded size.
    pub scale: DecodeScale,
    /// `--look`: colour preset.
    pub look: DecodeLook,
    /// `--pipeline auto|metal|cpu`: where the SDK decodes. `None` leaves
    /// the helper's default (`auto`: the first Metal device, else the CPU).
    pub pipeline: Option<String>,
}

impl DecodeOptions {
    fn extend_argv(&self, cmd: &mut Command) {
        if self.bit_16 {
            cmd.arg("--16bit");
        }
        if let Some(p) = &self.pipeline {
            cmd.arg("--pipeline").arg(p);
        }
        if let Some(t) = self.track {
            cmd.arg("--track").arg(t.to_string());
        }
        if let Some(s) = self.start_frame {
            cmd.arg("--start").arg(s.to_string());
        }
        if let Some(c) = self.count {
            cmd.arg("--count").arg(c.to_string());
        }
        if let Some(g) = &self.gamma {
            cmd.arg("--gamma").arg(g);
        }
        if let Some(g) = &self.gamut {
            cmd.arg("--gamut").arg(g);
        }
        if let Some(v) = self.wb {
            cmd.arg("--wb").arg(v.to_string());
        }
        if let Some(v) = self.tint {
            cmd.arg("--tint").arg(v.to_string());
        }
        if let Some(v) = self.exposure {
            cmd.arg("--exposure").arg(format!("{v}"));
        }
        if let Some(v) = self.iso {
            cmd.arg("--iso").arg(v.to_string());
        }
        if self.scale != DecodeScale::Full {
            cmd.arg("--scale").arg(self.scale.arg());
        }
        if self.look != DecodeLook::Film {
            cmd.arg("--look").arg(self.look.arg());
        }
    }
}

/// Streaming BRAW decoder.
///
/// Lifecycle:
/// 1. `BrawDecoder::start(path, opts)` spawns the helper and reads
///    the stderr JSON header. The header tells us frame dimensions
///    and bit depth.
/// 2. `next_frame(&mut buf)` reads exactly `bytes_per_frame()` bytes
///    from stdout into the caller's buffer. Returns `false` at EOF.
/// 3. Drop closes stdin/stdout/stderr; the helper sees EOF and exits.
pub struct BrawDecoder {
    helper_path: PathBuf,
    child: Child,
    stdout: BufReader<std::process::ChildStdout>,
    header: DecodeHeader,
    /// Thread reading stderr to keep the helper unblocked. Joined on
    /// drop.
    stderr_log: Option<std::thread::JoinHandle<String>>,
    /// Frames handed out so far — an EOF before the header's `frames`
    /// count (or with a non-zero exit) is a helper failure, not the end.
    delivered: u64,
}

impl std::fmt::Debug for BrawDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrawDecoder")
            .field("helper_path", &self.helper_path)
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

impl BrawDecoder {
    /// Spawn `braw_helper --decode <path> [opts…]` and parse the
    /// JSON header from stderr. The first non-empty `{`-starting line
    /// on stderr is the header; subsequent stderr lines are forwarded
    /// to a background thread (tracing::debug) so the helper never
    /// blocks on a full stderr pipe.
    pub fn start(path: impl AsRef<Path>, opts: &DecodeOptions) -> Result<Self> {
        let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
            expected: "helpers/bin/braw_helper".into(),
        })?;
        let mut cmd = Command::new(&helper);
        cmd.arg("--decode").arg(path.as_ref());
        opts.extend_argv(&mut cmd);

        tracing::info!(
            helper = %helper.display(),
            file = %path.as_ref().display(),
            "spawning braw_helper --decode"
        );

        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        // Read stderr until we see the JSON header. Subsequent lines
        // (progress, warnings) go to tracing::debug.
        let (header_tx, header_rx) = std::sync::mpsc::channel::<Result<DecodeHeader>>();
        let stderr_log = std::thread::spawn(move || {
            let mut header_sent = false;
            let reader = std::io::BufReader::new(stderr);
            let mut captured = String::new();
            for line in std::io::BufRead::lines(reader).map_while(|r| r.ok()) {
                if !header_sent && line.trim_start().starts_with('{') {
                    match serde_json::from_str::<DecodeHeader>(&line) {
                        Ok(h) => {
                            let _ = header_tx.send(Ok(h));
                        }
                        Err(e) => {
                            let _ = header_tx.send(Err(Error::BadJson(format!(
                                "stderr header='{line}': {e}"
                            ))));
                        }
                    }
                    header_sent = true;
                    continue;
                }
                tracing::debug!(target: "braw_helper", "{line}");
                captured.push_str(&line);
                captured.push('\n');
            }
            if !header_sent {
                let _ = header_tx.send(Err(Error::BadOutput(
                    "braw_helper closed stderr before emitting JSON header".into(),
                )));
            }
            captured
        });

        let header = match header_rx.recv() {
            Ok(Ok(h)) => h,
            Ok(Err(e)) => {
                // No header: the helper failed to open the clip — its only
                // diagnostic is the `ERROR:` line it printed; surface it.
                let status = child.wait().ok();
                let captured = stderr_log.join().unwrap_or_default();
                return Err(Error::HelperFailed {
                    code: status.and_then(|s| s.code()).unwrap_or(-1),
                    stderr: if captured.trim().is_empty() { e.to_string() } else { captured.trim().to_string() },
                });
            }
            Err(_) => return Err(Error::BadOutput("stderr thread died before header".into())),
        };
        tracing::info!(
            "braw_helper: {}x{} {}-bit, {} frames, pipeline {}{}",
            header.width, header.height, header.bit_depth, header.frames,
            header.pipeline.as_deref().unwrap_or("?"),
            header.gpu_device.as_deref().map(|d| format!(" ({d})")).unwrap_or_default(),
        );

        Ok(BrawDecoder {
            helper_path: helper,
            child,
            stdout: BufReader::with_capacity(1 << 20, stdout), // 1 MiB
            header,
            stderr_log: Some(stderr_log),
            delivered: 0,
        })
    }

    /// Decoded header (dimensions, bit depth, multi-track info).
    pub fn header(&self) -> &DecodeHeader {
        &self.header
    }

    /// Read exactly one frame into the caller's buffer. Resizes `buf`
    /// to `bytes_per_frame()` if needed. Returns `Ok(true)` on a
    /// successful read, `Ok(false)` on clean EOF, `Err` on helper
    /// failure mid-stream.
    pub fn next_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        let need = self.header.bytes_per_frame();
        if buf.len() < need {
            buf.resize(need, 0);
        }
        let mut total = 0usize;
        while total < need {
            match self.stdout.read(&mut buf[total..need])? {
                0 => {
                    if total == 0 {
                        // EOF on a frame boundary: clean only when the helper
                        // exited cleanly after the announced count — a decode
                        // error mid-clip prints an `ERROR:` line and exits 1,
                        // which must not pass for the end of the clip.
                        let status = self.child.wait()?;
                        let short = self.header.frames > 0 && self.delivered < self.header.frames;
                        if !status.success() || short {
                            let captured = self.stderr_log.take().and_then(|h| h.join().ok()).unwrap_or_default();
                            return Err(Error::HelperFailed {
                                code: status.code().unwrap_or(-1),
                                stderr: format!("braw_helper ended after {} of {} frames (exit {:?}): {}",
                                    self.delivered, self.header.frames, status.code(), captured.trim()),
                            });
                        }
                        return Ok(false);
                    } else {
                        return Err(Error::BadOutput(format!(
                            "short frame: got {} of {} bytes",
                            total, need
                        )));
                    }
                }
                n => total += n,
            }
        }
        self.delivered += 1;
        Ok(true)
    }
}

impl Drop for BrawDecoder {
    fn drop(&mut self) {
        // Kill the helper if it's still alive; collect stderr.
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stderr_log.take() {
            if let Ok(captured) = handle.join() {
                if !captured.is_empty() {
                    tracing::debug!(target: "braw_helper", "exit stderr: {captured}");
                }
            }
        }
    }
}
