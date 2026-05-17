//! EAC (Equi-Angular Cubemap) cross assembly geometry.
//!
//! GoPro Max .360 files store each lens as a 5-tile row in one HEVC
//! stream. Layout per stream (height = `stream_h`, width = `stream_w`):
//!
//! ```text
//!  [ tile | tile | CENTER (1920 wide) | tile | tile ]
//!    ↑      ↑                          ↑      ↑
//!    edge  side                        side   edge
//! ```
//!
//! - Each `tile` has width `tile_w = (stream_w - 1920) / 4`.
//! - The center face is fixed at **1920 px**.
//! - Total: `4 * tile_w + 1920 = stream_w`.
//!
//! For the original GoPro Max: `stream_w = 5952`, `tile_w = 1008`.
//! For the Max 2 variants seen in the wild: `stream_w = 5888 → tile_w = 992`,
//! or `stream_w = 5696 → tile_w = 944` (the value that broke the Python
//! pipeline's hardcoded `1008` slice for one user this session).
//!
//! Dimensions are **always** derived from the actual probed stream — the
//! Python app's hardcoded `5952` / `1008` constants are impossible to
//! introduce here by construction.

/// Runtime-detected EAC stream dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    /// Width of a single HEVC stream (s0 or s4).
    pub stream_w: u32,
    /// Height of a single HEVC stream.
    pub stream_h: u32,
}

/// Fixed center-face width in the GoPro EAC layout (all camera variants).
pub const CENTER_W: u32 = 1920;

impl Dims {
    pub const fn new(stream_w: u32, stream_h: u32) -> Self {
        Self { stream_w, stream_h }
    }

    /// Width of one side / corner / edge tile in the per-stream 5-tile row.
    ///
    /// `tile_w = (stream_w - 1920) / 4`.
    pub fn tile_w(self) -> u32 {
        (self.stream_w.saturating_sub(CENTER_W)) / 4
    }

    /// Side of the per-lens EAC cross. `cross_w = 2 * tile_w + 1920`.
    /// Equal to `stream_w / 1.512…` for normal aspect; on Max it's 3936.
    pub fn cross_w(self) -> u32 {
        2 * self.tile_w() + CENTER_W
    }

    /// True if the stream width is consistent with the EAC layout
    /// (i.e. `stream_w > 1920` and `(stream_w - 1920) % 4 == 0`).
    /// Useful for surfacing a clear error on unknown camera variants.
    pub fn is_valid(self) -> bool {
        self.stream_w > CENTER_W && (self.stream_w - CENTER_W) % 4 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_max_dims() {
        let d = Dims::new(5952, 1920);
        assert_eq!(d.tile_w(), 1008);
        assert_eq!(d.cross_w(), 3936);
        assert!(d.is_valid());
    }

    #[test]
    fn max2_5888_dims() {
        let d = Dims::new(5888, 1920);
        assert_eq!(d.tile_w(), 992);
        assert_eq!(d.cross_w(), 3904);
        assert!(d.is_valid());
    }

    #[test]
    fn max2_5696_dims_matches_user_bug_width() {
        // The Python crash was:
        //   could not broadcast input array from shape (1920,944,3)
        //                              into shape (1920,1008,3)
        // 944 = (5696 - 1920) / 4 — matches a stream width Python's
        // hardcoded `[:, 4944:5952]` slice silently truncated to 944 px.
        let d = Dims::new(5696, 1920);
        assert_eq!(d.tile_w(), 944);
        assert_eq!(d.cross_w(), 3808);
        assert!(d.is_valid());
    }

    #[test]
    fn invalid_dims_rejected() {
        // Not divisible by 4 after removing center → flag as invalid.
        let d = Dims::new(5950, 1920);
        assert!(!d.is_valid());
    }
}
