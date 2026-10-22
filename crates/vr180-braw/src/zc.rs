//! Zero-copy frame hand-off from `braw_helper --zerocopy` (macOS).
//!
//! The helper owns a pool of global IOSurfaces — one or two per slot, each
//! an eye on the 2:1 canvas (RGBA8) — and fills a slot per decoded frame on
//! the GPU. Only `F <frame> <slot>` lines cross the pipe; this client maps
//! the surfaces (by ID, once) and returns slots with `R <slot>` on the
//! helper's stdin once its GPU work on them is done. The helper never
//! submits a decode without a free slot, so a slot is only rewritten after
//! its release.

use crate::decoder::DecodeOptions;
use crate::{Error, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

/// The zero-copy decode header (`--zerocopy`): the pipe header's fields
/// plus the pool.
#[derive(Debug, Clone, Deserialize)]
pub struct ZcHeader {
    /// The processed frame (both eyes of a single-track clip).
    pub width: u32,
    pub height: u32,
    /// Frames the helper will hand over (after `--step`).
    pub frames: u64,
    #[serde(default)]
    pub dual_stream: bool,
    #[serde(default)]
    pub step: Option<u32>,
    #[serde(default)]
    pub pipeline: Option<String>,
    #[serde(default)]
    pub gpu_device: Option<String>,
    /// `"iosurface"`.
    #[serde(default)]
    pub transport: String,
    pub slots: u32,
    /// One eye's surface: the square canvas for a pair, the frame for a
    /// mono clip.
    pub eye_width: u32,
    pub eye_height: u32,
    /// Rows the canvas adds above the recorded eye (negative: cropped).
    #[serde(default)]
    pub pad_rows: i32,
    #[serde(default)]
    pub mono: bool,
    /// Per slot `[left, right]` global IOSurface IDs (equal for mono).
    pub surfaces: Vec<[u32; 2]>,
}

/// A frame the helper placed in a slot.
#[derive(Debug, Clone, Copy)]
pub struct ZcFrame {
    /// Absolute frame index in the clip.
    pub index: u64,
    pub slot: u32,
}

/// What the helper says next.
#[derive(Debug, Clone, Copy)]
pub enum ZcEvent {
    Frame(ZcFrame),
    /// The seek with this generation took effect: everything before it was
    /// discarded, the frames that follow start at its target.
    SeekAck(u32),
    /// The end of the clip; the helper stays alive for a later seek.
    End,
}

/// Returns slots to the helper. Shared by the frames handed out, so a
/// frame can release its slot from wherever it is dropped.
pub struct ZcReleaser {
    stdin: Mutex<Option<std::process::ChildStdin>>,
}

impl ZcReleaser {
    pub fn release(&self, slot: u32) {
        if let Some(w) = self.stdin.lock().unwrap().as_mut() {
            // A dead helper just means the slot is gone with it.
            let _ = writeln!(w, "R {slot}").and_then(|_| w.flush());
        }
    }
    /// Ask the helper to continue from `frame`; it answers `SeekAck(gen)`
    /// once the frames it was decoding are dropped. `false` when the
    /// helper's stdin is gone (restart instead).
    pub fn seek(&self, frame: u64, gen: u32) -> bool {
        match self.stdin.lock().unwrap().as_mut() {
            Some(w) => writeln!(w, "S {frame} {gen}").and_then(|_| w.flush()).is_ok(),
            None => false,
        }
    }
    fn close(&self) {
        *self.stdin.lock().unwrap() = None;
    }
}

/// One helper run in zero-copy mode.
pub struct BrawZcDecoder {
    child: Child,
    stdout: BufReader<std::process::ChildStdout>,
    header: ZcHeader,
    releaser: Arc<ZcReleaser>,
    stderr_log: Option<std::thread::JoinHandle<String>>,
    delivered: u64,
}

impl BrawZcDecoder {
    /// Spawn `braw_helper --decode <path> --zerocopy <slots> …` and read
    /// its header. `mono`: one fisheye per frame (one surface per slot).
    pub fn start(path: impl AsRef<Path>, opts: &DecodeOptions, slots: u32, mono: bool) -> Result<Self> {
        let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
            expected: "helpers/bin/braw_helper".into(),
        })?;
        let mut cmd = Command::new(&helper);
        cmd.arg("--decode").arg(path.as_ref());
        opts.extend_argv(&mut cmd);
        cmd.arg("--zerocopy").arg(slots.max(2).to_string());
        if mono { cmd.arg("--mono"); }
        tracing::info!(helper = %helper.display(), file = %path.as_ref().display(), slots, "spawning braw_helper --decode --zerocopy");
        let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let (header_tx, header_rx) = std::sync::mpsc::channel::<Result<ZcHeader>>();
        let stderr_log = std::thread::spawn(move || {
            let mut captured = String::new();
            let mut header_sent = false;
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if !header_sent && line.trim_start().starts_with('{') {
                    let _ = header_tx.send(serde_json::from_str::<ZcHeader>(&line)
                        .map_err(|e| Error::BadJson(format!("zero-copy header='{line}': {e}"))));
                    header_sent = true;
                    continue;
                }
                tracing::debug!(target: "braw_helper", "{line}");
                if line.starts_with("ERROR") || line.starts_with("note") { captured.push_str(&line); captured.push('\n'); }
            }
            if !header_sent {
                let _ = header_tx.send(Err(Error::BadOutput("braw_helper closed stderr before its zero-copy header".into())));
            }
            captured
        });
        let header = match header_rx.recv() {
            Ok(Ok(h)) => h,
            Ok(Err(e)) => {
                let status = child.wait().ok();
                let captured = stderr_log.join().unwrap_or_default();
                return Err(Error::HelperFailed {
                    code: status.and_then(|s| s.code()).unwrap_or(-1),
                    stderr: if captured.trim().is_empty() { e.to_string() } else { captured.trim().to_string() },
                });
            }
            Err(_) => return Err(Error::BadOutput("stderr thread died before the zero-copy header".into())),
        };
        if header.transport != "iosurface" {
            return Err(Error::BadOutput(format!("helper transport '{}' is not iosurface", header.transport)));
        }
        if header.surfaces.len() != header.slots as usize {
            return Err(Error::BadOutput(format!("{} surface pairs for {} slots", header.surfaces.len(), header.slots)));
        }
        tracing::info!(
            "braw_helper zero-copy: {}x{} frame, {} eyes of {}x{} (pad {}), {} slots, {} frames, pipeline {}{}",
            header.width, header.height, if header.mono { "mono" } else { "two" }, header.eye_width, header.eye_height,
            header.pad_rows, header.slots, header.frames,
            header.pipeline.as_deref().unwrap_or("?"),
            header.gpu_device.as_deref().map(|d| format!(" ({d})")).unwrap_or_default(),
        );
        Ok(Self {
            child,
            stdout: BufReader::new(stdout),
            header,
            releaser: Arc::new(ZcReleaser { stdin: Mutex::new(Some(stdin)) }),
            stderr_log: Some(stderr_log),
            delivered: 0,
        })
    }

    pub fn header(&self) -> &ZcHeader { &self.header }
    pub fn releaser(&self) -> Arc<ZcReleaser> { self.releaser.clone() }

    /// The helper's next event. The helper does not exit at the end of the
    /// clip (it waits for a seek), so a closed pipe is a failure.
    pub fn next_event(&mut self) -> Result<ZcEvent> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                let status = self.child.wait()?;
                let captured = self.stderr_log.take().and_then(|h| h.join().ok()).unwrap_or_default();
                return Err(Error::HelperFailed {
                    code: status.code().unwrap_or(-1),
                    stderr: format!("braw_helper ended after {} zero-copy frames (exit {:?}): {}",
                        self.delivered, status.code(), captured.trim()),
                });
            }
            let mut it = line.split_whitespace();
            match it.next() {
                Some("F") => {
                    let (Some(idx), Some(slot)) = (it.next().and_then(|s| s.parse::<u64>().ok()), it.next().and_then(|s| s.parse::<u32>().ok())) else {
                        return Err(Error::BadOutput(format!("bad zero-copy line '{}'", line.trim())));
                    };
                    if slot >= self.header.slots {
                        return Err(Error::BadOutput(format!("slot {slot} out of {}", self.header.slots)));
                    }
                    self.delivered += 1;
                    return Ok(ZcEvent::Frame(ZcFrame { index: idx, slot }));
                }
                Some("K") => {
                    let gen = it.next().and_then(|s| s.parse::<u32>().ok())
                        .ok_or_else(|| Error::BadOutput(format!("bad seek ack '{}'", line.trim())))?;
                    return Ok(ZcEvent::SeekAck(gen));
                }
                Some("E") => return Ok(ZcEvent::End),
                _ => continue,
            }
        }
    }
}

impl Drop for BrawZcDecoder {
    fn drop(&mut self) {
        // Closing stdin ends the helper's release reader; kill for good
        // measure (a helper mid-decode). Surfaces the app still maps stay
        // alive through its own references.
        self.releaser.close();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(h) = self.stderr_log.take() {
            if let Ok(captured) = h.join() {
                if !captured.trim().is_empty() { tracing::debug!(target: "braw_helper", "zero-copy exit stderr: {}", captured.trim()); }
            }
        }
    }
}

impl std::fmt::Debug for BrawZcDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrawZcDecoder").field("header", &self.header).field("delivered", &self.delivered).finish()
    }
}

/// Keep the paths in scope for callers that log them.
pub fn _path(p: &Path) -> PathBuf { p.to_path_buf() }
