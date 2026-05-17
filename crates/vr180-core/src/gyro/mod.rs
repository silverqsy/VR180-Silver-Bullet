//! Gyro / orientation pipeline.
//!
//! Ports `parse_gyro_raw.py` and `parse_gopro_gyro_data` from the Python
//! app, broken into focused modules:
//!
//! - [`gpmf`] — low-level GPMF byte-stream walker (DEVC/STRM container
//!              traversal, fourcc / type / struct_size / repeat decoding).
//! - [`cori_iori`] — CORI / IORI quaternion extraction (Q15 fixed-point
//!              W,X,Y,Z 4-shorts-per-sample).
//!
//! Higher-level fusion (VQF 9D for bias-drifting CORI, GRAV roll
//! complementary filter, MNOR heading) lands in Phase 0.3.

pub mod gpmf;
pub mod cori_iori;
pub mod raw;
pub mod resample;
pub mod vqf;

pub use cori_iori::{Quat, parse_cori, parse_iori, quat_to_euler_zyx};
pub use gpmf::{GpmfEntry, GpmfWalker, FourCC};
pub use raw::{ImuBlock, RawImu, parse_raw_imu};
pub use resample::{resample_quats_to_frames, cori_swap_yz, SROT_S};
