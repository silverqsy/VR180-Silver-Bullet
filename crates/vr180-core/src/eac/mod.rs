//! EAC (Equi-Angular Cubemap) cross assembly.
//!
//! Two-level module:
//! - [`dims`] — the [`Dims`] struct and the layout math (`tile_w`,
//!   `cross_w`, `is_valid`).
//! - [`assemble`] — the actual byte-shuffling: take the two HEVC
//!   streams (`s0` and `s4`) and produce a 3936×3936 (or smaller
//!   on Max 2) RGB cross per lens, with corners edge-replicated to
//!   avoid bilinear bleed in the projection step.
//!
//! All ops work on packed RGB8 (`&[u8]`, 3 bytes per pixel,
//! stride = `w * 3`). 10-bit upgrade lands in Phase 0.5 alongside
//! the wgpu compute pipeline.

pub mod assemble;
pub mod dims;

pub use assemble::{assemble_lens_a, assemble_lens_b, rotate_90_cw, rotate_90_ccw};
pub use dims::{Dims, CENTER_W};
