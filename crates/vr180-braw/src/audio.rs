//! `braw_helper --audio <file>` extraction.
//!
//! BRAW containers don't expose audio through ffmpeg — the audio data
//! is wrapped inside the proprietary BRAW format. The `braw_helper`
//! C++ binary uses the BlackmagicRAW SDK to pull the PCM samples out
//! and writes a standard RIFF/WAV stream to stdout, along with a JSON
//! header on stderr.
//!
//! Usage at export time: spawn the helper, redirect stdout to a temp
//! WAV file, then pass that WAV path to ffmpeg as a separate `-i`
//! input to mux against the encoded video.

use crate::{Error, Result};
use serde::Deserialize;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// JSON header on stderr.
///
/// Mirrors `braw_helper.cpp:do_audio()` output (around line 739-814):
/// ```text
/// {"sample_rate":48000,"bit_depth":24,"channels":2,
///  "sample_count":<N>,"data_bytes":<bytes>}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct BrawAudioHeader {
    pub sample_rate: u32,
    pub bit_depth: u32,
    #[serde(alias = "channel_count")]
    pub channels: u32,
    pub sample_count: u64,
    pub data_bytes: u64,
}

/// Extract audio from a `.braw` file into a freshly-created WAV file
/// at `dst`. The destination directory must exist; the file is
/// over-written if present.
///
/// Returns the parsed header (for the caller's metadata display +
/// validation that the WAV write completed successfully — bytes
/// landed on disk equals `header.data_bytes` plus the 44-byte RIFF
/// preamble).
pub fn extract_audio_to_wav(
    braw_path: &Path,
    dst: &Path,
) -> Result<BrawAudioHeader> {
    let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
        expected: "helpers/bin/braw_helper".into(),
    })?;

    tracing::info!(
        "braw audio: spawning {} --audio {} → {}",
        helper.display(), braw_path.display(), dst.display()
    );

    let mut child = Command::new(&helper)
        .arg("--audio")
        .arg(braw_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");

    // Read stderr (header) into memory in a side thread. Audio is the
    // bulk on stdout, so we keep that on the foreground thread.
    let stderr_thread = std::thread::spawn(move || -> Result<String> {
        let mut s = String::new();
        stderr.read_to_string(&mut s)?;
        Ok(s)
    });

    // Pipe stdout straight to the destination file.
    let mut dst_file = std::fs::File::create(dst)?;
    let mut buf = vec![0u8; 1 << 16]; // 64 KiB
    let mut written: u64 = 0;
    loop {
        let n = stdout.read(&mut buf)?;
        if n == 0 { break; }
        dst_file.write_all(&buf[..n])?;
        written += n as u64;
    }
    dst_file.flush()?;
    drop(dst_file);

    let status = child.wait()?;
    let stderr_text = stderr_thread.join().expect("stderr thread panicked")?;

    if !status.success() {
        // Clean up partial output.
        let _ = std::fs::remove_file(dst);
        return Err(Error::HelperFailed {
            code: status.code().unwrap_or(-1),
            stderr: stderr_text,
        });
    }

    // Parse the JSON header from stderr.
    let header_line = stderr_text
        .lines()
        .find(|l| !l.trim().is_empty() && l.trim_start().starts_with('{'))
        .ok_or_else(|| Error::BadOutput(format!(
            "no audio header on stderr. Full stderr: {stderr_text}"
        )))?;
    let header: BrawAudioHeader = serde_json::from_str(header_line)
        .map_err(|e| Error::BadJson(format!("audio header='{header_line}': {e}")))?;

    tracing::info!(
        "braw audio: wrote {} bytes → {} ({} ch, {} Hz, {}-bit, {} samples)",
        written, dst.display(),
        header.channels, header.sample_rate,
        header.bit_depth, header.sample_count
    );

    Ok(header)
}

/// Convenience: extract audio to a unique temp WAV file under the
/// system temp dir and return its path. Caller is responsible for
/// deleting the file after muxing.
///
/// Useful at export time: `let wav = extract_audio_to_tempfile(&braw)?;`
/// then pass `wav.path` to ffmpeg as `-i <wav.path>` alongside the
/// encoded video. After ffmpeg finishes, `std::fs::remove_file(wav.path)`.
pub fn extract_audio_to_tempfile(braw_path: &Path) -> Result<TempWavPath> {
    let stem = braw_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("braw_audio");
    let dst = std::env::temp_dir().join(format!(
        "vr180_{}_{}.wav",
        stem,
        std::process::id()
    ));
    let header = extract_audio_to_wav(braw_path, &dst)?;
    Ok(TempWavPath { path: dst, header })
}

/// RAII wrapper around a temp WAV file. Deletes the file on drop —
/// caller can `.path()` to feed it to ffmpeg, and `mem::forget` the
/// wrapper if they want to keep the file around.
#[derive(Debug)]
pub struct TempWavPath {
    pub path: PathBuf,
    pub header: BrawAudioHeader,
}

impl TempWavPath {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWavPath {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(&self.path) {
            tracing::warn!(
                "TempWavPath: failed to delete {}: {e}", self.path.display()
            );
        }
    }
}
