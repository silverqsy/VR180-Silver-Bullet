//! Mac-native helper subprocess glue.
//!
//! Spawn pattern:
//! - resolve helper binary path (next to `vr180-render` in release,
//!   `helpers/bin/` in dev)
//! - spawn with stdin/stdout/stderr piped
//! - stream stderr lines into our `tracing` log
//! - map exit code → `pipeline::Error::Helper`
//!
//! Four helpers (all macOS-only):
//! - `mvhevc_encode`   — MV-HEVC spatial video encode via VideoToolbox
//! - `apac_encode`     — Apple Positional Audio Codec (Vision Pro spatial audio)
//! - `vt_denoise`      — VTTemporalNoiseFilter (true 10-bit)
//! - `aivu_wrap`       — a joined file of Lens Space clips as an Apple
//!   Immersive Video file, a lens for every clip (macOS 26)
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

/// Whether this system can write an Apple Immersive Video file: the
/// `aivu_wrap` helper is there and starts. (It is built for macOS 26, the
/// first system with the framework it uses, and does not start on an
/// earlier one: asking it for its version is the test.) Asked once; always
/// `false` off macOS.
pub fn aivu_wrap_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            locate_helper("aivu_wrap").is_some_and(|helper| {
                Command::new(helper).arg("--version")
                    .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
                    .status().is_ok_and(|s| s.success())
            })
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// One line of `aivu_wrap`'s report: `progress <written> <in all>`.
fn aivu_progress(line: &str) -> Option<(u64, u64)> {
    let mut words = line.split_whitespace();
    (words.next() == Some("progress")).then_some(())?;
    Some((words.next()?.parse().ok()?, words.next()?.parse().ok()?))
}

/// Run `aivu_wrap`: write `joined` (the parts' video packets one after the
/// other, and the sound) again as the Apple Immersive Video file `output`.
/// `parts`: each part's own file, which states its lens, and the number of
/// video frames it gave to `joined`, in order. `progress(written, in all)`
/// counts frames; `cancel` stops the helper and leaves no file.
///
/// macOS only. `Error::Helper` carries what the helper said when it fails:
/// a part without a lens, a file Apple's validator refuses.
#[cfg(target_os = "macos")]
pub fn spawn_aivu_wrap(
    joined: &Path,
    output: &Path,
    parts: &[(PathBuf, u64)],
    mut progress: impl FnMut(u64, u64),
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let failed = |code: i32, stderr: String| Error::Helper { name: "aivu_wrap".into(), code, stderr };
    let helper = locate_helper("aivu_wrap")
        .ok_or_else(|| failed(-1, "binary not found in helpers/bin/ or next to the app. Run ./helpers/build_swift.sh".into()))?;
    let mut cmd = Command::new(&helper);
    cmd.arg("--joined").arg(joined).arg("--output").arg(output);
    for (part, frames) in parts {
        cmd.arg("--part").arg(part).arg("--frames").arg(frames.to_string());
    }
    tracing::info!(joined = %joined.display(), output = %output.display(), parts = parts.len(), "spawning aivu_wrap");
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn()
        .map_err(|e| failed(-1, format!("spawn failed: {e}")))?;
    let stderr = child.stderr.take().ok_or_else(|| failed(-1, "stderr handle vanished".into()))?;
    // Its lines come over a channel, so that a cancel is seen between them.
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(|r| r.ok()) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut said = String::new();
    let mut cancelled = false;
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(line) => match aivu_progress(&line) {
                Some((done, total)) => progress(done, total),
                None => {
                    tracing::info!(target: "aivu_wrap", "{line}");
                    said.push_str(&line);
                    said.push('\n');
                }
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if cancel.load(std::sync::atomic::Ordering::SeqCst) && !cancelled {
            cancelled = true;
            let _ = child.kill();
        }
    }
    let status = child.wait().map_err(|e| failed(-1, format!("wait failed: {e}")))?;
    let _ = reader.join();
    if cancelled {
        let _ = std::fs::remove_file(output);
        return Err(failed(-1, "cancelled".into()));
    }
    if !status.success() {
        let _ = std::fs::remove_file(output);
        // What it gave as the reason, when it gave one.
        let why = said.lines().rev().find_map(|l| l.strip_prefix("ERROR: ")).map(str::to_string).unwrap_or(said);
        return Err(failed(status.code().unwrap_or(-1), why));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_progress_line_is_two_counts() {
        assert_eq!(aivu_progress("progress 120 451"), Some((120, 451)));
        assert_eq!(aivu_progress("aivu_wrap: clip-1: 4096 x 4096 an eye"), None);
        assert_eq!(aivu_progress("progress 12"), None);
        assert_eq!(aivu_progress("ERROR: progress 1 2"), None);
    }
}
