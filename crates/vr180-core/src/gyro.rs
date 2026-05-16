//! Gyro / orientation pipeline.
//!
//! Will port `parse_gyro_raw.py` (GPMF stream walking, CORI/IORI/
//! GRAV/MNOR parsing with DEVC/STRM/STMP nesting) and the VQF 9D
//! fusion path that handles bias-drifting CORI (no-firmware-RS
//! mode noted in project memory).
//!
//! Phase 0.2 — GPMF parse + CORI/IORI extraction.
//! Phase 0.3 — VQF 9D fallback.
