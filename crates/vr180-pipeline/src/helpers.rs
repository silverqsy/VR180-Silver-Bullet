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
//! Phase 0.8.

use std::path::{Path, PathBuf};

/// Locate a Swift helper binary. Checks dev tree first, then the
/// directory next to the running `vr180-render` exe (release).
#[allow(dead_code)]
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
