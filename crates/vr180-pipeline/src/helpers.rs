//! Mac-native helper subprocess glue.
//!
//! Spawn pattern:
//! - resolve helper binary path (next to `vr180-render` in release,
//!   `helpers/bin/` in dev)
//! - spawn with stdin/stdout/stderr piped
//! - stream stderr lines into our `tracing` log
//! - map exit code → `pipeline::Error::Helper`
//!
//! Three helpers (all macOS-only):
//! - `mvhevc_encode`   — MV-HEVC spatial video encode via VideoToolbox
//! - `apac_encode`     — Apple Positional Audio Codec (Vision Pro spatial audio)
//! - `vt_denoise`      — VTTemporalNoiseFilter (true 10-bit)
//!
//! Phase 0.8.6 wires `apac_encode` end-to-end. The others stay as
//! built binaries in `helpers/bin/` for future phases.

use crate::{Error, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Locate a Swift helper binary. Checks dev tree first, then the
/// directory next to the running `vr180-render` exe (release).
pub(crate) fn locate_helper(name: &str) -> Option<PathBuf> {
    // Dev: helpers/bin/<name>
    let dev = workspace_root().join("helpers").join("bin").join(name);
    if dev.is_file() {
        return Some(dev);
    }
    // Release: next to current exe
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn workspace_root() -> &'static Path {
    // CARGO_MANIFEST_DIR is set at compile time; walk up to workspace root.
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let pkg = Path::new(env!("CARGO_MANIFEST_DIR"));
        // crates/vr180-pipeline → workspace root
        pkg.parent().and_then(|p| p.parent()).unwrap_or(pkg).to_path_buf()
    })
}

/// Spawn `apac_encode` to mux Apple Positional Audio Codec audio
/// into a video file. Three modes:
///
/// - `video_input=None`: standalone audio-only `.mp4` output.
///   Reads `pcm_wav` (4ch ambisonic) → writes `output` as APAC mp4.
/// - `video_input=Some(v)`: full mux. Copies the video track from
///   `v` bit-exact (sample data + format description preserved) AND
///   adds an APAC audio track from `pcm_wav` to `output`.
///
/// The Swift helper handles the APAC encode (kAudioFormatAPAC) and
/// AVAssetWriter setup that ffmpeg's mov muxer can't produce (it
/// drops the `dapa` config atom on `-c:a copy`).
///
/// Stderr is streamed into `tracing::warn` (the helper logs progress
/// + warnings there); stdout is ignored.
///
/// macOS only. Returns `Error::Helper` with the helper's exit code +
/// captured stderr on non-zero exit.
#[cfg(target_os = "macos")]
pub fn spawn_apac_encode(
    pcm_wav: &Path,
    video_input: Option<&Path>,
    output: &Path,
    bitrate_bps: u32,
) -> Result<()> {
    let helper = locate_helper("apac_encode").ok_or_else(|| Error::Helper {
        name: "apac_encode".into(),
        code: -1,
        stderr: "binary not found in helpers/bin/ or next to vr180-render. \
                 Run ./helpers/build_swift.sh".into(),
    })?;

    let mut cmd = Command::new(&helper);
    cmd.arg("--input").arg(pcm_wav)
       .arg("--output").arg(output)
       .arg("--bitrate").arg(bitrate_bps.to_string());
    if let Some(v) = video_input {
        cmd.arg("--video-input").arg(v);
    }
    tracing::info!(
        helper = %helper.display(),
        pcm = %pcm_wav.display(),
        video = ?video_input,
        output = %output.display(),
        bitrate = bitrate_bps,
        "spawning apac_encode"
    );

    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Helper {
            name: "apac_encode".into(),
            code: -1,
            stderr: format!("spawn failed: {e}"),
        })?;

    // Stream stderr into tracing AND collect into a buffer for the
    // error-on-failure case. Two birds, one thread.
    let stderr_handle = child.stderr.take()
        .ok_or_else(|| Error::Helper {
            name: "apac_encode".into(), code: -1,
            stderr: "stderr handle vanished".into(),
        })?;
    let stderr_thread = std::thread::spawn(move || {
        let mut captured = String::new();
        let reader = BufReader::new(stderr_handle);
        for line in reader.lines().map_while(|r| r.ok()) {
            tracing::warn!(target: "apac_encode", "{line}");
            captured.push_str(&line);
            captured.push('\n');
        }
        captured
    });

    let status = child.wait().map_err(|e| Error::Helper {
        name: "apac_encode".into(), code: -1,
        stderr: format!("wait failed: {e}"),
    })?;
    let stderr_text = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        return Err(Error::Helper {
            name: "apac_encode".into(),
            code: status.code().unwrap_or(-1),
            stderr: stderr_text,
        });
    }
    Ok(())
}
