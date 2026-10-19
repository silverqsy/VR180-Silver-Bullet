//! `braw_helper --lut <file> <out.cube>`: the clip's embedded post-3D-LUT
//! ("Gen 5 Film to Extended Video" on the PYXIS 12K — Blackmagic's own
//! Film-log → display transform, carried in every clip) written out as a
//! standard `.cube`, so the app can apply it like the other cameras'
//! log LUTs.

use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Dump the embedded LUT of `path` to a `.cube` in the system temp dir,
/// named after the LUT's `title` (the same title → the same file, reused
/// across clips of a camera: the file name is what the LUT field shows).
/// `Err` when the clip carries none or the helper is missing.
pub fn extract_embedded_lut(path: impl AsRef<Path>, title: &str) -> Result<PathBuf> {
    let helper = crate::locate_helper().ok_or_else(|| Error::HelperMissing {
        expected: "helpers/bin/braw_helper".into(),
    })?;
    let src = path.as_ref();
    let safe: String = title.chars().map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' { c } else { '_' }).collect();
    let safe = safe.trim();
    let out = std::env::temp_dir().join(format!("{} (embedded).cube", if safe.is_empty() { "BRAW LUT" } else { safe }));
    if cube_is_complete(&out) {
        return Ok(out);
    }
    // Write to a per-process partial and rename, so a crash or a second
    // process never leaves a half-written cube under the shared name.
    let partial = out.with_extension(format!("{}.partial", std::process::id()));
    let res = Command::new(&helper)
        .arg("--lut").arg(src).arg(&partial)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .output()?;
    if !res.status.success() {
        std::fs::remove_file(&partial).ok();
        return Err(Error::HelperFailed {
            code: res.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&res.stderr).into_owned(),
        });
    }
    if !cube_is_complete(&partial) {
        std::fs::remove_file(&partial).ok();
        return Err(Error::BadOutput("braw_helper --lut wrote an incomplete cube".into()));
    }
    std::fs::rename(&partial, &out)?;
    Ok(out)
}

/// A `.cube` whose `LUT_3D_SIZE n` is followed by n³ data rows.
fn cube_is_complete(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else { return false };
    let mut size = 0usize;
    let mut rows = 0usize;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        if let Some(n) = l.strip_prefix("LUT_3D_SIZE") { size = n.trim().parse().unwrap_or(0); continue; }
        if l.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '.') { rows += 1; }
    }
    size >= 2 && rows == size * size * size
}
