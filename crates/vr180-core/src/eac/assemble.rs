//! EAC cross assembly from the two HEVC streams (s0 + s4).
//!
//! Each stream is `stream_w × stream_h × RGB8` (stride = `stream_w * 3`).
//! Output is `cross_w × cross_w × RGB8` per lens. All slice offsets are
//! derived from [`Dims::tile_w`] — no hardcoded `1008` or `5952`.
//!
//! Tile layout (Lens A example, all dims scale with `tile_w` and `cross_w`):
//!
//! ```text
//!   ┌────────────────────────────────────────────┐
//!   │   ↻ corner  │   TOP face   │   ↻ corner   │  rows 0           .. tile_w
//!   │ (replicated)│ s4[-tile_w..] rot 90 CW     │  cols 0   .. cross_w
//!   ├─────────────┼──────────────┼──────────────┤
//!   │  LEFT face  │  CENTER face │  RIGHT face  │  rows tile_w      .. tile_w+1920
//!   │ s0[tw..2tw] │ s0[2tw..2tw+1920] │ s0[2tw+1920..3tw+1920]    │
//!   ├─────────────┼──────────────┼──────────────┤
//!   │  corner     │  BOTTOM face │   corner     │  rows tile_w+1920 .. cross_w
//!   │ (replicated)│ s4[0..tile_w] rot 90 CW     │
//!   └─────────────┴──────────────┴──────────────┘
//! ```
//!
//! Mirrors `FrameExtractor._assemble_lensA` / `_assemble_lensB` /
//! `_fill_cross_corners` in `vr180_gui.py`.

use super::dims::{Dims, CENTER_W};

/// Assemble the Lens-A EAC cross from the two streams. `out` must be a
/// `cross_w × cross_w × 3` byte buffer (uninitialized memory is OK
/// because every output pixel is written exactly once).
///
/// Source notation: `s0` / `s4` are the two HEVC streams, each laid out
/// as `stream_h × stream_w × RGB8` (packed, row-major). The 5 tiles
/// per stream are at columns `[0, tw)`, `[tw, 2tw)`, `[2tw, 2tw+cn)`
/// (center), `[2tw+cn, 3tw+cn)`, `[3tw+cn, 4tw+cn)`.
pub fn assemble_lens_a(s0: &[u8], s4: &[u8], dims: Dims, out: &mut [u8]) {
    let tw = dims.tile_w() as usize;
    let cw = dims.cross_w() as usize;
    let sw = dims.stream_w as usize;
    let sh = dims.stream_h as usize;
    let cn = CENTER_W as usize;
    debug_assert_eq!(s0.len(), sw * sh * 3);
    debug_assert_eq!(s4.len(), sw * sh * 3);
    debug_assert_eq!(out.len(), cw * cw * 3);

    // Top face: s4 last-tile (h=sh, w=tw) rotated 90 CW → (h'=tw, w'=sh=cn).
    let top = rotate_90_cw(slice_cols(s4, sw, sh, sw - tw, sw), tw, sh);
    blit_rect(out, cw, &top, /*src_w=*/ cn, /*src_h=*/ tw, /*dst_x=*/ tw, /*dst_y=*/ 0);

    // Left face: s0 cols [tw, 2*tw)  →  shape (sh, tw)
    let left = slice_cols(s0, sw, sh, tw, 2 * tw);
    blit_rect(out, cw, &left, tw, sh, 0, tw);

    // Center face: s0 cols [2*tw, 2*tw+cn)  →  shape (sh, cn)
    let center = slice_cols(s0, sw, sh, 2 * tw, 2 * tw + cn);
    blit_rect(out, cw, &center, cn, sh, tw, tw);

    // Right face: s0 cols [2*tw+cn, 3*tw+cn)  →  shape (sh, tw)
    let right = slice_cols(s0, sw, sh, 2 * tw + cn, 3 * tw + cn);
    blit_rect(out, cw, &right, tw, sh, tw + cn, tw);

    // Bottom face: s4 first-tile (h=sh, w=tw) rotated 90 CW.
    let bottom = rotate_90_cw(slice_cols(s4, sw, sh, 0, tw), tw, sh);
    blit_rect(out, cw, &bottom, cn, tw, tw, tw + sh);

    fill_cross_corners(out, dims);
}

/// Assemble the Lens-B EAC cross from the two streams. Layout differs:
/// the center face comes from `s4` rotated 90 CCW (Python: `s4_rot`),
/// and the left/right faces come from `s0`'s outermost tiles.
pub fn assemble_lens_b(s0: &[u8], s4: &[u8], dims: Dims, out: &mut [u8]) {
    let tw = dims.tile_w() as usize;
    let cw = dims.cross_w() as usize;
    let sw = dims.stream_w as usize;
    let sh = dims.stream_h as usize;
    let cn = CENTER_W as usize;
    debug_assert_eq!(s0.len(), sw * sh * 3);
    debug_assert_eq!(s4.len(), sw * sh * 3);
    debug_assert_eq!(out.len(), cw * cw * 3);

    // s4_rot = s4[:, tw .. sw-tw] rotated 90 CCW.
    // Input width = sw - 2*tw = 2*tw + cn = cross_w. Input shape (sh, cw).
    // After 90 CCW: shape (cw, sh).
    let s4_rot_src = slice_cols(s4, sw, sh, tw, sw - tw);
    let s4_rot = rotate_90_ccw(s4_rot_src, cw, sh);

    // Top face: s4_rot[0..tw, :]  →  shape (tw, sh)
    let top = rows(s4_rot.as_slice(), sh, 0, tw);
    blit_rect(out, cw, &top, sh, tw, tw, 0);

    // Left face: s0 cols [sw-tw, sw)  →  shape (sh, tw)
    let left = slice_cols(s0, sw, sh, sw - tw, sw);
    blit_rect(out, cw, &left, tw, sh, 0, tw);

    // Center face: s4_rot[tw .. tw+sh, :]  →  shape (sh, sh==cn)
    let center = rows(s4_rot.as_slice(), sh, tw, tw + sh);
    blit_rect(out, cw, &center, sh, sh, tw, tw);

    // Right face: s0 cols [0, tw)  →  shape (sh, tw)
    let right = slice_cols(s0, sw, sh, 0, tw);
    blit_rect(out, cw, &right, tw, sh, tw + cn, tw);

    // Bottom face: s4_rot[tw+sh .. tw+sh+tw, :]  →  shape (tw, sh)
    let bottom = rows(s4_rot.as_slice(), sh, tw + sh, tw + sh + tw);
    blit_rect(out, cw, &bottom, sh, tw, tw, tw + sh);

    fill_cross_corners(out, dims);
}

// ─── Geometry helpers ──────────────────────────────────────────────────

/// Take rows `[r0, r1)` from a packed RGB8 image of width `w`. Returns a
/// new Vec; cheap because the inner loop is contiguous.
fn rows(src: &[u8], w: usize, r0: usize, r1: usize) -> Vec<u8> {
    let stride = w * 3;
    src[r0 * stride..r1 * stride].to_vec()
}

/// Take a column slice `[c0, c1)` from a packed RGB8 image of width `w`
/// and height `h`. Returns a new contiguous Vec of size `h * (c1-c0) * 3`.
fn slice_cols(src: &[u8], w: usize, h: usize, c0: usize, c1: usize) -> Vec<u8> {
    let new_w = c1 - c0;
    let mut out = Vec::with_capacity(h * new_w * 3);
    let in_stride = w * 3;
    let copy_start = c0 * 3;
    let copy_end = c1 * 3;
    for y in 0..h {
        let row = &src[y * in_stride..y * in_stride + in_stride];
        out.extend_from_slice(&row[copy_start..copy_end]);
    }
    out
}

/// Rotate an RGB8 image 90° clockwise. Input shape `(w × h × 3)`,
/// output shape `(h × w × 3)`.
///
/// Mapping: `dst[x, h-1-y] = src[y, x]`.
pub fn rotate_90_cw(src: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), w * h * 3);
    let mut dst = vec![0u8; w * h * 3];
    let dst_stride = h * 3;
    for y in 0..h {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        let dst_x = h - 1 - y;
        for x in 0..w {
            let src_off = x * 3;
            let dst_off = x * dst_stride + dst_x * 3;
            dst[dst_off..dst_off + 3].copy_from_slice(&src_row[src_off..src_off + 3]);
        }
    }
    dst
}

/// Rotate an RGB8 image 90° counter-clockwise. Input shape `(w × h × 3)`,
/// output shape `(h × w × 3)`.
///
/// Mapping: `dst[w-1-x, y] = src[y, x]`.
pub fn rotate_90_ccw(src: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), w * h * 3);
    let mut dst = vec![0u8; w * h * 3];
    let dst_stride = h * 3;
    for y in 0..h {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        for x in 0..w {
            let src_off = x * 3;
            let dst_off = (w - 1 - x) * dst_stride + y * 3;
            dst[dst_off..dst_off + 3].copy_from_slice(&src_row[src_off..src_off + 3]);
        }
    }
    dst
}

/// Copy a packed RGB8 source image (shape `src_h × src_w × 3`) into the
/// rectangle `(dst_x..dst_x+src_w, dst_y..dst_y+src_h)` of a packed RGB8
/// destination image with width `dst_w`. Copies the FULL source — no
/// separate `copy_w` parameter — so call-site mistakes that swap
/// `dst_x`/`copy_w` are impossible.
fn blit_rect(
    dst: &mut [u8], dst_w: usize,
    src: &[u8], src_w: usize, src_h: usize,
    dst_x: usize, dst_y: usize,
) {
    debug_assert_eq!(src.len(), src_w * src_h * 3);
    let dst_stride = dst_w * 3;
    let src_stride = src_w * 3;
    for y in 0..src_h {
        let src_off = y * src_stride;
        let dst_off = (dst_y + y) * dst_stride + dst_x * 3;
        dst[dst_off..dst_off + src_stride]
            .copy_from_slice(&src[src_off..src_off + src_stride]);
    }
}

// ─── Corner / seam padding ────────────────────────────────────────────

/// Fill the four corners of an assembled cross with edge-replicated
/// pixels from the adjacent side faces. Mirrors
/// `FrameExtractor._fill_cross_corners` in `vr180_gui.py` — same logic,
/// same justification (kills bilinear bleed at face-corner boundaries
/// in the equirect projection step).
pub fn fill_cross_corners(cross: &mut [u8], dims: Dims) {
    let tw = dims.tile_w() as usize;
    let cw = dims.cross_w() as usize;
    let cn = CENTER_W as usize;
    let center_top = tw;             // first row of the center band
    let center_bot = tw + cn;        // first row AFTER the center band

    // Each corner is a tw×tw rectangle.
    // - TL corner: rows [0, tw),         cols [0, tw)         ← replicate row `tw`     cols [0, tw)
    // - TR corner: rows [0, tw),         cols [tw+cn, cw)     ← replicate row `tw`     cols [tw+cn, cw)
    // - BL corner: rows [tw+cn, cw),     cols [0, tw)         ← replicate row `tw+cn-1`cols [0, tw)
    // - BR corner: rows [tw+cn, cw),     cols [tw+cn, cw)     ← replicate row `tw+cn-1`cols [tw+cn, cw)
    replicate_band(cross, cw, center_top,     0,         tw, 0,           tw); // TL
    replicate_band(cross, cw, center_top,     tw + cn,   tw, 0,           tw); // TR
    replicate_band(cross, cw, center_bot - 1, 0,         tw, center_bot,  cw); // BL
    replicate_band(cross, cw, center_bot - 1, tw + cn,   tw, center_bot,  cw); // BR

    // 1-pixel seam fix: replicate the inner-face edge pixel one column
    // outward at the top/bottom-face↔corner boundary. Mirrors the four
    // 1-px column-copy lines in the Python `_fill_cross_corners`.
    //
    // - top    band rows [0, tw),         col (tw - 1)      ← col tw     (top↔TL)
    // - top    band rows [0, tw),         col (tw + cn)     ← col tw+cn-1(top↔TR)
    // - bottom band rows [tw+cn, cw),     col (tw - 1)      ← col tw     (bottom↔BL)
    // - bottom band rows [tw+cn, cw),     col (tw + cn)     ← col tw+cn-1(bottom↔BR)
    copy_column(cross, cw, tw,        tw - 1,  0,          tw);
    copy_column(cross, cw, tw + cn - 1, tw + cn, 0,          tw);
    copy_column(cross, cw, tw,        tw - 1,  center_bot, cw);
    copy_column(cross, cw, tw + cn - 1, tw + cn, center_bot, cw);
}

/// Replicate one source row's `[c0, c0+band_w)` column band into rows
/// `[dst_r0, dst_r1)` of the same image.
fn replicate_band(
    cross: &mut [u8],
    img_w: usize,
    src_row: usize,
    c0: usize,
    band_w: usize,
    dst_r0: usize,
    dst_r1: usize,
) {
    let stride = img_w * 3;
    let copy_bytes = band_w * 3;
    let src_off = src_row * stride + c0 * 3;
    let src_slice: Vec<u8> = cross[src_off..src_off + copy_bytes].to_vec();
    for r in dst_r0..dst_r1 {
        let off = r * stride + c0 * 3;
        cross[off..off + copy_bytes].copy_from_slice(&src_slice);
    }
}

/// Copy a single column from `src_col` to `dst_col` for rows `[r0, r1)`.
fn copy_column(
    cross: &mut [u8],
    img_w: usize,
    src_col: usize,
    dst_col: usize,
    r0: usize,
    r1: usize,
) {
    let stride = img_w * 3;
    for r in r0..r1 {
        let src_off = r * stride + src_col * 3;
        let dst_off = r * stride + dst_col * 3;
        // SAFETY-free: split into disjoint slices via copy_within when
        // possible. The two offsets are always different (Δ ≥ 1 col),
        // so this is well-defined.
        if src_off < dst_off {
            let (lo, hi) = cross.split_at_mut(dst_off);
            hi[..3].copy_from_slice(&lo[src_off..src_off + 3]);
        } else {
            let (lo, hi) = cross.split_at_mut(src_off);
            lo[dst_off..dst_off + 3].copy_from_slice(&hi[..3]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eac::Dims;

    fn make_solid_stream(dims: Dims, color: [u8; 3]) -> Vec<u8> {
        let n = (dims.stream_w * dims.stream_h * 3) as usize;
        let mut v = vec![0u8; n];
        for px in v.chunks_exact_mut(3) {
            px.copy_from_slice(&color);
        }
        v
    }

    #[test]
    fn assemble_lens_a_buffer_size_matches_dims() {
        let dims = Dims::new(5952, 1920);
        let s0 = make_solid_stream(dims, [255, 0, 0]);
        let s4 = make_solid_stream(dims, [0, 255, 0]);
        let cw = dims.cross_w() as usize;
        let mut out = vec![0u8; cw * cw * 3];
        assemble_lens_a(&s0, &s4, dims, &mut out);
        // Sanity: center pixel comes from s0 (red).
        let c = cw / 2;
        let off = (c * cw + c) * 3;
        assert_eq!(&out[off..off + 3], &[255, 0, 0], "center should be from s0 (red)");
    }

    #[test]
    fn rotate_90_cw_then_ccw_is_identity() {
        let w = 4usize; let h = 3usize;
        let mut src = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            src.push(i as u8); src.push(i as u8); src.push(i as u8);
        }
        let rotated = rotate_90_cw(src.clone(), w, h);
        let back = rotate_90_ccw(rotated, h, w);
        assert_eq!(back, src);
    }

    #[test]
    fn assemble_works_for_max2_dims() {
        // Smoke test: the assembly doesn't panic / out-of-bounds on a
        // non-standard stream width. (Would have caught the
        // hardcoded-5952 family of bugs.)
        let dims = Dims::new(5888, 1920);
        let s0 = make_solid_stream(dims, [123, 45, 67]);
        let s4 = make_solid_stream(dims, [10, 20, 30]);
        let cw = dims.cross_w() as usize;
        let mut a = vec![0u8; cw * cw * 3];
        let mut b = vec![0u8; cw * cw * 3];
        assemble_lens_a(&s0, &s4, dims, &mut a);
        assemble_lens_b(&s0, &s4, dims, &mut b);
        // Center of cross_w (3904) is at (1952, 1952). Lens A center
        // comes from s0 → (123, 45, 67).
        let c = cw / 2;
        let off = (c * cw + c) * 3;
        assert_eq!(&a[off..off + 3], &[123, 45, 67]);
    }
}
