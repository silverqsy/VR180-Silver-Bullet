//! GEOC lens calibration parsing (KLNS Kannala-Brandt coefficients,
//! CTRX/CTRY principal point offsets, CALW/CALH sensor dimensions).
//!
//! GoPro stores GEOC in the file tail as a separate atom block.
//! The Python implementation in `vr180_gui.py::parse_geoc` reads
//! the last 1 MiB of the file and walks GPMF-style records.
//!
//! Confirmed stream→lens mapping (from project memory):
//!   - s0 → FRNT lens → use `FRNT.KLNS` (right eye after yaw mod)
//!   - s4 → BACK lens → use `BACK.KLNS` (left eye after yaw mod)
//!
//! Phase 0.4 — `parse_geoc(path) -> Geoc`.
