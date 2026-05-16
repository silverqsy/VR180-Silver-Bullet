//! EAC cross assembly geometry.
//!
//! Dimensions are derived from the actual decoded stream — NEVER
//! hardcoded. The Python app's bug class around `5952×1920`
//! constants is impossible by construction here.
//!
//! See [eac_cross_assembly.md] in the Python project memory for the
//! tile layout and per-eye assembly rules.
//!
//! Phase 0.4 — `Dims::from_probe`, `assemble_lens_a`, `assemble_lens_b`.

/// Runtime-detected EAC stream dimensions. Constructed from the
/// libav format probe; no `Default`, no hardcoded constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    /// Width of a single HEVC stream (s0 or s4). Typically 5952 on
    /// the original Max, may differ on Max 2 (~5888 reported).
    pub stream_w: u32,
    /// Height of a single HEVC stream. Typically 1920.
    pub stream_h: u32,
}

impl Dims {
    /// Per-eye EAC cross side-tile width.
    ///
    /// EAC layout: each row of a stream is `[side | center | side]`
    /// where center is 1920 px (the center face), and the two sides
    /// pack the corner + edge tiles. `side_w = (stream_w - 1920) / 2`.
    pub fn side_w(self) -> u32 {
        (self.stream_w.saturating_sub(1920)) / 2
    }

    /// EAC cross size (each lens assembles into a `cross_w × cross_w`
    /// square once its tiles are laid out).
    pub fn cross_w(self) -> u32 {
        2 * self.side_w() + 1920
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_max_dims() {
        let d = Dims { stream_w: 5952, stream_h: 1920 };
        assert_eq!(d.side_w(), 2016);
        assert_eq!(d.cross_w(), 5952);
    }

    #[test]
    fn max2_dims_reported_by_user() {
        // birgernaert's footage that broke the Python hardcoded path.
        let d = Dims { stream_w: 5888, stream_h: 1920 };
        assert_eq!(d.side_w(), 1984);
        assert_eq!(d.cross_w(), 5888);
    }
}
