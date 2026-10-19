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
    if out.is_file() && std::fs::metadata(&out).map(|m| m.len() > 1000).unwrap_or(false) {
        return Ok(out);
    }
    let res = Command::new(&helper)
        .arg("--lut").arg(src).arg(&out)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .output()?;
    if !res.status.success() {
        return Err(Error::HelperFailed {
            code: res.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&res.stderr).into_owned(),
        });
    }
    if !out.is_file() {
        return Err(Error::BadOutput("braw_helper --lut wrote no file".into()));
    }
    Ok(out)
}
