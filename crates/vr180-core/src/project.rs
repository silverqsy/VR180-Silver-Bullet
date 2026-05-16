//! Project configuration model.
//!
//! Mirrors the Python `ProcessingConfig` dataclass in
//! `vr180_gui.py:5260+`. JSON shape is intentionally identical so
//! the Python GUI can write configs that this crate reads — that's
//! the wedge for Phase 0.9 (GUI shells out to `vr180-render`
//! without a UI port).
//!
//! Phase 0.1: skeleton; fields land in 0.2 alongside GPMF parsing.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Eventual mirror of Python `ProcessingConfig`. Populated module-
/// by-module as each pipeline stage lands.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessingConfig {
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    // TODO 0.2: trim, gyro window, RS factors
    // TODO 0.7: color (lift/gamma/gain/sat, shadow/highlight, temp/tint, mid_detail, lut)
    // TODO 0.8: vision_pro_mode, audio_ambisonics, audio_apac
}
