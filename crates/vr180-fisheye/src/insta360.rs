//! Insta360 `.insv` trailer parser.
//!
//! Insta360 X-series cameras write a standard MP4 (`ftyp` / `mdat` /
//! `moov`) followed by a proprietary trailer that ends with a 32-byte
//! ASCII magic. The trailer carries the camera identity, the factory
//! per-lens calibration string, the raw IMU stream (~1 kHz gyro + accel)
//! and the per-frame exposure timestamps — everything the stabilizer
//! needs (the MP4 tracks themselves hold no motion data).
//!
//! Layout, read from the end of the file backwards:
//!
//! ```text
//!   … records … | directory (10-byte entries) | footer (78 bytes)
//!
//!   footer    = [u16 0][u32 directory_len][30 × 0]
//!               [u32 trailer_len][u32 version][32-byte magic]
//!   dir entry = [u16 record_id][u32 size][u32 offset]   (zero slots = padding)
//! ```
//!
//! Record offsets are relative to `file_len − trailer_len − 10`; a
//! record's payload starts 10 bytes past its offset (a small per-record
//! header) and runs for `size` bytes.
//!
//! Records consumed here (X6, firmware v1.0.2xx):
//!
//! | id     | content |
//! |--------|---------|
//! | 0x0002 | cover thumbnail: `[u32 1][u32 size][u32 1][u32 1][u32 w][u32 h][16×0]` + YUV420, a 360° equirect the camera stitched from frame 0 (1280×640) — the calibration ground truth |
//! | 0x0101 | identity: serial, model ("Insta360 X6"), firmware, file path, calibration string |
//! | 0x0003 | IMU: 20-byte samples `[u64 t_µs][ax ay az][gx gy gz]`, u16 offset-binary |
//! | 0x0004 | per-frame `[u64 t_µs][f64 exposure_s]` — `t` is the END of the exposure |
//!
//! The IMU and frame timestamps share one boot-time clock, so frame↔gyro
//! sync is direct (no drift correction needed).
//!
//! Stills (`.insp`) use the older layout instead of a directory: records are
//! chained backwards, each `[data][u16 id (byte-swapped)][u32 size]`, and the
//! footer's first 6 bytes are the last record's entry. Not parsed here (the
//! app never opens stills), but their 2560×1280 stitched preview (record
//! 0x0200) is what the lens model was decoded against.

use crate::{Error, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// ASCII magic that terminates every Insta360 trailer.
pub const INSV_TRAILER_MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";
const FOOTER_LEN: u64 = 78;
const RECORD_HEADER_LEN: u64 = 10;
const REC_IDENTITY: u16 = 0x0101;
const REC_IMU: u16 = 0x0003;
const REC_FRAME_STAMPS: u16 = 0x0004;
/// Same shape as 0x0004 for the second sensor (calibration entry 2 —
/// stream 0 on the X6). Its exposure runs independently (auto-exposure per
/// sensor), so its mid-exposure time can differ by several ms.
const REC_FRAME_STAMPS_B: u16 = 0x000c;

/// Accelerometer scale: 1024 LSB per g (±32 g full scale).
pub const INSV_ACCEL_LSB_PER_G: f32 = 1024.0;
/// Gyro scale: ±2000 °/s full scale over ±32768 → 16.384 LSB per °/s.
pub const INSV_GYRO_LSB_PER_DPS: f32 = 16.384;

/// One lens' factory calibration entry from the identity record —
/// **decoded 2026-09-08** against the camera's own stitched previews
/// (feature matches between the raw fisheye and the equirect the camera
/// wrote into record 0x0002 / 0x0200): a **Unified Camera Model** (Mei /
/// Geyer, parameter `ξ`) followed by an even radial polynomial on the
/// normalised UCM plane. Parameter-free residuals vs the camera's stitch:
/// ~4 px median in the 7744² native frame, ~3 px in the 3840² video
/// stream — the ground truth's own noise floor.
///
/// Projection of a ray at angle `θ` from the optical axis:
/// ```text
///   m = sin θ / (ξ + cos θ)                    (normalised UCM radius)
///   D = 1 + c₁m² + c₂m⁴ + c₃m⁶ + c₄m⁸          (`radial`)
///   r = fx · m · D                              (pixels, native frame)
/// ```
/// so the paraxial (centre) scale is `fx / (1 + ξ)` ≈ 2088 px/rad — the
/// raw `fx` (≈7217) is NOT an equidistant focal.
///
/// Coordinate frame: the native **15488 × 7744** dual-sensor image, two
/// 7744² halves side by side (entry 2's `cx` sits in the right half). The
/// recorded frame is the sensor window from the file's `window_crop_info`
/// ([`InsvWindowCrop`]; X6 video: 7744² → 7680² centred, i.e. a 32 px
/// border dropped) scaled to the stream — 3840² is the **2 × 2-binned**
/// sensor, so `fx` halves and the principal point is `(c − 32)/2`
/// (entry 2 lands ~24 px left of the frame centre; stills pad to 7760²) —
/// see `vr180-pipeline::insv_imu::stream_lens_calib`. Track order is
/// **reversed** on the X6 (`track_order_reversed`): stream 0 = back lens =
/// entry 2, stream 1 = screen-side lens = entry 1 — verified by matching
/// features across the two streams' overlap through the factory
/// extrinsics. The nine `tail` terms are `[k5, A, B, C, E, s1, s3, s2, s4]`:
/// the 5th radial coefficient, four tangential terms and four thin-prism
/// terms of the projection in [`InsvLensCalib::project`] (decoded by
/// matching Insta360 Studio's output to f32 precision).
#[derive(Debug, Clone, PartialEq)]
pub struct InsvLensCalib {
    /// Unified-camera-model `ξ` (`2.455430` on the X6, identical per lens).
    pub xi: f32,
    /// UCM focal in native pixels (paraxial scale = `fx / (1 + ξ)`).
    pub fx: f32,
    pub fy: f32,
    /// Principal point in the native 15488 × 7744 frame.
    pub cx: f32,
    pub cy: f32,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    /// ≈ ±89.5° — the sensor is mounted rotated; streams are already upright.
    pub roll_deg: f32,
    /// Lens position (translation, m) relative to lens 1 — `[0,0,0]` for
    /// entry 1, `≈ [0, 0, −0.033]` for the back-to-back entry 2.
    pub extra: [f32; 3],
    /// Even radial polynomial `c₁..c₄` on the normalised UCM plane.
    pub radial: [f32; 4],
    /// `[k5, A, B, C, E, s1, s3, s2, s4]` — see [`InsvLensCalib::project`].
    pub tail: [f32; 9],
    pub sensor_w: u32,
    pub sensor_h: u32,
    /// Trailing tag (`193` on the X6).
    pub tag: u32,
}

impl InsvLensCalib {
    /// Paraxial (centre) scale in native px/rad: `fx / (1 + ξ)`.
    pub fn paraxial_focal(&self) -> f64 {
        self.fx as f64 / (1.0 + self.xi as f64)
    }

    /// Normalised UCM radius `m(θ)`.
    pub fn ucm_m(&self, theta: f64) -> f64 {
        theta.sin() / (self.xi as f64 + theta.cos())
    }

    /// Native-frame image radius (px) of a ray at `theta` radians from the
    /// optical axis: `fx · m · D(m)` (radial part only).
    pub fn project_radius(&self, theta: f64) -> f64 {
        let m = self.ucm_m(theta);
        let m2 = m * m;
        let k = self.radial5();
        let d = 1.0 + m2 * (k[0] as f64 + m2 * (k[1] as f64 + m2 * (k[2] as f64
            + m2 * (k[3] as f64 + m2 * k[4] as f64))));
        self.fx as f64 * m * d
    }

    /// X origin of this lens' sensor half in the native dual-sensor frame:
    /// the halves are `sensor_h` wide (square) and side by side, so entry 2's
    /// `cx` (≈11569 on the X6) lives in the half starting at 7744. Subtract
    /// this to get per-lens coordinates (what the recorded frame uses).
    pub fn half_origin_x(&self) -> f32 {
        let half = self.sensor_h as f32;
        if half > 0.0 && self.cx >= half { (self.cx / half).floor() * half } else { 0.0 }
    }

    /// Radial polynomial `[k1, k2, k3, k4, k5]` (`k5` is the first tail term).
    pub fn radial5(&self) -> [f32; 5] {
        [self.radial[0], self.radial[1], self.radial[2], self.radial[3], self.tail[0]]
    }

    /// Tangential terms `[A, B, C, E]`.
    pub fn tangential(&self) -> [f32; 4] {
        [self.tail[1], self.tail[2], self.tail[3], self.tail[4]]
    }

    /// Thin-prism terms `[s1, s2, s3, s4]` (x: `s1 r² + s2 r⁴`, y: `s3 r² + s4 r⁴`).
    /// The string stores them as `s1, s3, s2, s4`.
    pub fn prism(&self) -> [f32; 4] {
        [self.tail[5], self.tail[7], self.tail[6], self.tail[8]]
    }

    /// Exact factory projection of a camera-frame ray `(X, Y, Z)` (x right,
    /// y DOWN, z along the optical axis; any length) to native-frame pixels
    /// `(u, v)`, matching Insta360 Studio's dewarp to f32 precision:
    /// `(x, y) = (X, Y)/(ξ·|d| + Z)`, `D = 1 + k1 r² + … + k5 r¹⁰`,
    /// `x' = x·D + (r²+2x²)(A + C r²) + 2xy(B + E r²) + s1 r² + s2 r⁴`,
    /// `y' = y·D + (r²+2y²)(B + E r²) + 2xy(A + C r²) + s3 r² + s4 r⁴`,
    /// `u = cx + fx·x'`, `v = cy + fy·y'`. Returns `None` beyond the fold of
    /// the UCM radius (`ξ·cosθ + 1 ≤ 0`, ≈114° off-axis on the X6), where
    /// the projection stops being injective.
    pub fn project(&self, dir: [f64; 3]) -> Option<(f64, f64)> {
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if !(norm > 1e-12) || self.xi as f64 * dir[2] / norm + 1.0 <= 0.0 {
            return None;
        }
        let den = self.xi as f64 * norm + dir[2];
        let (x, y) = (dir[0] / den, dir[1] / den);
        let r2 = x * x + y * y;
        let k = self.radial5();
        let d = 1.0 + r2 * (k[0] as f64 + r2 * (k[1] as f64 + r2 * (k[2] as f64
            + r2 * (k[3] as f64 + r2 * k[4] as f64))));
        let [a, b, c, e] = self.tangential().map(|v| v as f64);
        let [s1, s2, s3, s4] = self.prism().map(|v| v as f64);
        let xy2 = 2.0 * x * y;
        let xd = x * d + (r2 + 2.0 * x * x) * (a + c * r2) + xy2 * (b + e * r2) + s1 * r2 + s2 * r2 * r2;
        let yd = y * d + (r2 + 2.0 * y * y) * (b + e * r2) + xy2 * (a + c * r2) + s3 * r2 + s4 * r2 * r2;
        Some((self.cx as f64 + self.fx as f64 * xd, self.cy as f64 + self.fy as f64 * yd))
    }
}

/// The sensor window the camera stored in the file (`window_crop_info`):
/// the recorded frame is the `src` sensor area windowed to `dst` — centred
/// when `offset` is 0, else shifted by `offset` — and then scaled to the
/// stream size. On the X6 videos: `7744² → 7680²`, offset 0 (so a 32 px
/// border is dropped before the 2×2 binning to 3840²); stills pad to 7760².
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InsvWindowCrop {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub offset_x: i32,
    pub offset_y: i32,
}

/// The X6 video window when the file omits `window_crop_info`: the 7744²
/// sensor half minus a 32 px border.
pub const X6_DEFAULT_WINDOW: InsvWindowCrop = InsvWindowCrop {
    src_w: 7744, src_h: 7744, dst_w: 7680, dst_h: 7680, offset_x: 0, offset_y: 0,
};
/// The 10K mode's window (`7912²`, shifted 4 px in x) when a 4864² clip
/// does not carry it.
pub const X6_10K_WINDOW: InsvWindowCrop = InsvWindowCrop {
    src_w: 7744, src_h: 7744, dst_w: 7912, dst_h: 7912, offset_x: 4, offset_y: 0,
};
/// Rows the sensor reads in the 4912 × 3968 native-pixel mode behind the
/// 7.4K Crop (and the older 7.6K Crop); the encoded frame is the centred
/// part of them.
pub const X6_CROP_WINDOW_ROWS: u32 = 3968;

/// The user's answer for a clip whose view the file does not state — a
/// 3712² clip without a usable `window_crop_info` can be the wide 7.4K or
/// the 7.4K Crop, and nothing else in the file tells them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum X6ViewChoice {
    /// Follow the file; an unconfirmed clip renders as the wide mode.
    #[default]
    Auto,
    Wide,
    Crop,
}

/// Recording mode of an X6 dual-track clip, told from each eye's encoded
/// size and the trailer's `window_crop_info` (its `dst` is the mode's
/// signature — it is NOT a decode size or a crop rectangle).
///
/// | mode      | eye   | window `dst` | what the frame is |
/// |-----------|-------|--------------|-------------------|
/// | stock     | any   | 7680²        | the window scaled to the stream (8K: the 2×2 binning) |
/// | 7.4K      | 3712² | 7680²        | the same binned window, scaled to 3712² |
/// | 7.4K Crop | 3712² | 4912 × 3968  | native pixels: the centred 3712² of the sensor half |
/// | 10K       | 4864² | 7912²        | the wide readout scaled to 4864² |
/// | 10K Crop  | 4912² | 4912²        | native pixels: the centred 4912² |
/// | 7.6K Crop | 3808² | (older)      | native pixels: the centred 3808² |
///
/// Scale, timing and row span of every mode were checked on footage by
/// predicting feature motion from the gyro through the factory lens model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum X6Mode {
    /// A factory mode: the 7680² window scaled to the stream.
    #[default]
    Stock,
    Wide7K4,
    Crop7K4,
    Wide10K,
    Crop10K,
    /// The older 3808² native-pixel crop.
    Crop7K6,
    /// 3712² with a missing or contradictory window: wide or crop is the
    /// user's call ([`X6ViewChoice`]); rendered as wide until they say.
    Unconfirmed7K4,
    /// A window this table does not know. Rendered with the generic
    /// scaled-window rule so both tracks can be inspected; the geometry is
    /// not verified.
    Unknown,
}

impl X6Mode {
    /// Identify the mode. `window` is the file's `window_crop_info`.
    pub fn classify(window: Option<&InsvWindowCrop>, stream_w: u32, stream_h: u32) -> X6Mode {
        let dst = window.map(|w| (w.dst_w, w.dst_h));
        match (stream_w, stream_h) {
            (3712, 3712) => match dst {
                Some((7680, 7680)) => X6Mode::Wide7K4,
                Some((4912, 3968)) => X6Mode::Crop7K4,
                _ => X6Mode::Unconfirmed7K4,
            },
            // The older "9.7K" wrote the same 4864² frame.
            (4864, 4864) => X6Mode::Wide10K,
            (4912, 4912) => X6Mode::Crop10K,
            (3808, 3808) => X6Mode::Crop7K6,
            // Early 4928² test footage: not the 10K Crop, whatever it carries.
            (4928, 4928) => X6Mode::Unknown,
            _ => match dst {
                None | Some((7680, 7680)) => X6Mode::Stock,
                _ => X6Mode::Unknown,
            },
        }
    }

    /// Menu name of the mode; `None` for the factory modes.
    pub fn label(self) -> Option<&'static str> {
        match self {
            X6Mode::Stock => None,
            X6Mode::Wide7K4 => Some("7.4K"),
            X6Mode::Crop7K4 => Some("7.4K Crop"),
            X6Mode::Wide10K => Some("10K"),
            X6Mode::Crop10K => Some("10K Crop"),
            X6Mode::Crop7K6 => Some("7.6K Crop"),
            X6Mode::Unconfirmed7K4 => Some("7.4K60"),
            X6Mode::Unknown => Some("X6"),
        }
    }

    /// The frame holds a native-pixel crop of the image circle.
    pub fn is_crop(self) -> bool {
        matches!(self, X6Mode::Crop7K4 | X6Mode::Crop10K | X6Mode::Crop7K6)
    }
}

/// How a stream's pixels sit in the factory calibration's native frame
/// (one 7744² sensor half per lens): `stream = (native − origin) · scale`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsvStreamGeometry {
    /// The mode this geometry belongs to — after the user's choice, so
    /// never [`X6Mode::Unconfirmed7K4`].
    pub mode: X6Mode,
    /// What the file alone says (may be `Unconfirmed7K4`).
    pub detected: X6Mode,
    /// Stream pixels per native pixel.
    pub scale_x: f64,
    pub scale_y: f64,
    /// Native coordinates, within the lens' half, of the stream's origin.
    pub origin_x: f64,
    pub origin_y: f64,
    /// Share of the file's `rolling_shutter_time` the encoded rows span:
    /// 1 when the frame holds every row the sensor read, less when it is
    /// the centred part of them (7.4K Crop: 3712 of 3968).
    pub readout_fraction: f64,
    /// The frame is a crop of the image circle: a ray that leaves the
    /// frame has no data (the wide modes replicate the rim instead).
    pub cropped: bool,
}

impl InsvStreamGeometry {
    /// The generic rule (Insta360 Studio's): the `window` — centred in its
    /// `src`, shifted by its offset — scaled to the stream.
    pub fn scaled_window(window: &InsvWindowCrop, stream_w: u32, stream_h: u32) -> Self {
        Self {
            mode: X6Mode::Stock,
            detected: X6Mode::Stock,
            scale_x: stream_w as f64 / window.dst_w.max(1) as f64,
            scale_y: stream_h as f64 / window.dst_h.max(1) as f64,
            origin_x: (window.src_w as f64 - window.dst_w as f64) * 0.5 + window.offset_x as f64,
            origin_y: (window.src_h as f64 - window.dst_h as f64) * 0.5 + window.offset_y as f64,
            readout_fraction: 1.0,
            cropped: false,
        }
    }

    /// Native pixels, the stream centred in the sensor half (`src`) and
    /// shifted by the window's offset; `window_rows` rows were read out.
    fn native_crop(src_w: u32, src_h: u32, offset: (i32, i32), window_rows: u32, stream_w: u32, stream_h: u32) -> Self {
        Self {
            mode: X6Mode::Stock,
            detected: X6Mode::Stock,
            scale_x: 1.0,
            scale_y: 1.0,
            origin_x: (src_w as f64 - stream_w as f64) * 0.5 + offset.0 as f64,
            origin_y: (src_h as f64 - stream_h as f64) * 0.5 + offset.1 as f64,
            readout_fraction: if window_rows >= stream_h && window_rows > 0 {
                stream_h as f64 / window_rows as f64
            } else {
                1.0
            },
            cropped: true,
        }
    }

    /// Geometry of an X6 video stream. `window` is the file's
    /// `window_crop_info`; `choice` settles a clip the file leaves
    /// unconfirmed (and is ignored otherwise).
    pub fn resolve(window: Option<&InsvWindowCrop>, stream_w: u32, stream_h: u32, choice: X6ViewChoice) -> Self {
        let detected = X6Mode::classify(window, stream_w, stream_h);
        let mode = match (detected, choice) {
            (X6Mode::Unconfirmed7K4, X6ViewChoice::Crop) => X6Mode::Crop7K4,
            (X6Mode::Unconfirmed7K4, _) => X6Mode::Wide7K4,
            (m, _) => m,
        };
        // The sensor half the calibration is expressed in.
        let (src_w, src_h) = window.map(|w| (w.src_w, w.src_h)).filter(|s| s.0 > 0 && s.1 > 0)
            .unwrap_or((X6_DEFAULT_WINDOW.src_w, X6_DEFAULT_WINDOW.src_h));
        // The record contributes to a mode only when it carries that mode's
        // own signature: a window that contradicts the mode (or an
        // unconfirmed clip's) lends it neither its rows nor its offset.
        let signed = |dst: (u32, u32)| window.copied().filter(|w| (w.dst_w, w.dst_h) == dst);
        let native = |w: Option<InsvWindowCrop>, rows_default: u32| {
            Self::native_crop(
                src_w, src_h,
                w.map(|w| (w.offset_x, w.offset_y)).unwrap_or((0, 0)),
                w.map(|w| w.dst_h).unwrap_or(rows_default),
                stream_w, stream_h,
            )
        };
        let mut g = match mode {
            X6Mode::Crop7K4 | X6Mode::Crop7K6 => native(signed((4912, X6_CROP_WINDOW_ROWS)), X6_CROP_WINDOW_ROWS),
            X6Mode::Crop10K => native(signed((stream_w, stream_h)), stream_h),
            X6Mode::Wide10K => Self::scaled_window(
                &signed((X6_10K_WINDOW.dst_w, X6_10K_WINDOW.dst_h)).unwrap_or(X6_10K_WINDOW), stream_w, stream_h),
            X6Mode::Wide7K4 => Self::scaled_window(
                &signed((X6_DEFAULT_WINDOW.dst_w, X6_DEFAULT_WINDOW.dst_h)).unwrap_or(X6_DEFAULT_WINDOW), stream_w, stream_h),
            X6Mode::Stock | X6Mode::Unknown | X6Mode::Unconfirmed7K4 => Self::scaled_window(
                &window.copied().unwrap_or(X6_DEFAULT_WINDOW), stream_w, stream_h),
        };
        g.mode = mode;
        g.detected = detected;
        g
    }

    /// Principal point of a factory entry in stream coordinates.
    pub fn principal_point(&self, entry: &InsvLensCalib) -> (f64, f64) {
        (
            (entry.cx as f64 - entry.half_origin_x() as f64 - self.origin_x) * self.scale_x,
            (entry.cy as f64 - self.origin_y) * self.scale_y,
        )
    }
}

/// One raw IMU sample, converted to physical units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsvImuSample {
    /// Boot-time clock, microseconds (same clock as [`InsvFrameStamp::t_us`]).
    pub t_us: u64,
    /// Accelerometer, g.
    pub acc_g: [f32; 3],
    /// Gyro, degrees per second.
    pub gyr_dps: [f32; 3],
}

/// Per-video-frame exposure stamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsvFrameStamp {
    /// End of the frame's exposure, boot-time clock, microseconds.
    pub t_us: u64,
    /// Exposure duration in seconds.
    pub exposure_s: f64,
}

/// Everything we read out of an `.insv` trailer.
#[derive(Debug, Clone, Default)]
pub struct Insta360Meta {
    pub serial: String,
    /// e.g. `"Insta360 X6"`.
    pub model: String,
    pub firmware: String,
    /// The raw calibration string (`2_…`) as found in the file.
    pub calib_string: String,
    /// Parsed per-lens entries, in file order (lens 1 = left half of the
    /// native dual-sensor frame, lens 2 = right half).
    pub lenses: Vec<InsvLensCalib>,
    pub imu: Vec<InsvImuSample>,
    /// Per-frame exposure stamps of the first sensor (calibration entry 1;
    /// record 0x0004).
    pub frames: Vec<InsvFrameStamp>,
    /// Per-frame exposure stamps of the second sensor (entry 2; record
    /// 0x000c). Empty when the file has none; then `frames` serves both.
    pub frames_b: Vec<InsvFrameStamp>,
    /// Every record id present in the directory (diagnostics).
    pub record_ids: Vec<u16>,
    pub trailer_version: u32,
    /// Sensor readout (rolling-shutter) time in ms (`rolling_shutter_time`;
    /// 15.36 ms at 50 fps / 14.56 ms at 30 fps on the X6 3840² modes).
    pub readout_ms: Option<f64>,
    /// Gyro-to-frame time offset in ms (`gyro_timestamp`; 1.1 on the X6):
    /// added to the centre-row mid-exposure time to sample the gyro.
    pub gyro_offset_ms: Option<f64>,
    /// Sensor window → recorded frame (`window_crop_info`).
    pub window_crop: Option<InsvWindowCrop>,
    /// Recorded frame size (`dimension`), e.g. `(3840, 3840)`.
    pub dimension: Option<(u32, u32)>,
    /// Nominal frame rate (`frame_rate`), e.g. 30 / 50 / 60.
    pub frame_rate: Option<u32>,
    /// Picture profile (`gamma_mode`), e.g. `"I_Log"`. The log LUT follows
    /// this, never the frame size.
    pub gamma_mode: Option<String>,
    /// `pano_record_type` (2 = both lenses on their own tracks).
    pub pano_record_type: Option<u32>,
    /// `Some(true)` when the file says its two video tracks are in reverse
    /// lens order (track 0 = calibration entry 2). X6 dual-stream clips set
    /// this; `None` when the file doesn't say.
    pub track_order_reversed: Option<bool>,
    /// IMU full-scale ranges from `gyro_cfg_info` (°/s, g).
    pub gyro_range_dps: Option<u32>,
    pub acc_range_g: Option<u32>,
}

impl Insta360Meta {
    /// Mean IMU sample rate in Hz (0 when fewer than two samples).
    pub fn imu_rate_hz(&self) -> f64 {
        if self.imu.len() < 2 {
            return 0.0;
        }
        let span = (self.imu[self.imu.len() - 1].t_us - self.imu[0].t_us) as f64 * 1e-6;
        if span <= 0.0 { 0.0 } else { (self.imu.len() - 1) as f64 / span }
    }

    /// Mean IMU sample period in seconds (0 when unknown).
    pub fn imu_period_s(&self) -> f64 {
        let hz = self.imu_rate_hz();
        if hz > 0.0 { 1.0 / hz } else { 0.0 }
    }

    /// True for the Insta360 X6 (the only model calibrated so far).
    pub fn is_x6(&self) -> bool {
        self.model.to_ascii_lowercase().contains("x6")
    }

    /// Geometry of this clip's video streams (see [`InsvStreamGeometry`]).
    pub fn stream_geometry(&self, stream_w: u32, stream_h: u32, choice: X6ViewChoice) -> InsvStreamGeometry {
        InsvStreamGeometry::resolve(self.window_crop.as_ref(), stream_w, stream_h, choice)
    }

    /// The clip was recorded in I-Log (`gamma_mode`). `None` when the file
    /// does not say.
    pub fn is_ilog(&self) -> Option<bool> {
        self.gamma_mode.as_deref().map(|g| g.eq_ignore_ascii_case("I_Log") || g.eq_ignore_ascii_case("ILog"))
    }
}

/// Cheap check: does the file end with the Insta360 trailer magic?
/// Used to classify `.mp4`-renamed `.insv` files.
pub fn has_insv_trailer(path: &Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let Ok(len) = f.metadata().map(|m| m.len()) else { return false };
    if len < FOOTER_LEN { return false; }
    if f.seek(SeekFrom::Start(len - 32)).is_err() { return false; }
    let mut tail = [0u8; 32];
    f.read_exact(&mut tail).is_ok() && &tail == INSV_TRAILER_MAGIC
}

/// Directory entry: where a record's payload lives in the file.
#[derive(Debug, Clone, Copy)]
struct RecordLoc {
    id: u16,
    /// Absolute file offset of the payload (past the 10-byte header).
    payload_start: u64,
    size: u64,
}

fn err(msg: impl Into<String>) -> Error {
    Error::Insv(msg.into())
}

/// Locate the trailer directory. Returns `(entries, trailer_version)`.
fn read_directory(f: &mut std::fs::File, file_len: u64) -> Result<(Vec<RecordLoc>, u32)> {
    if file_len < FOOTER_LEN + RECORD_HEADER_LEN {
        return Err(err("file too small for an Insta360 trailer"));
    }
    f.seek(SeekFrom::Start(file_len - FOOTER_LEN))?;
    let mut foot = [0u8; FOOTER_LEN as usize];
    f.read_exact(&mut foot)?;
    if &foot[46..78] != INSV_TRAILER_MAGIC {
        return Err(err("Insta360 trailer magic not found"));
    }
    let dir_len = u32::from_le_bytes(foot[2..6].try_into().unwrap()) as u64;
    let trailer_len = u32::from_le_bytes(foot[38..42].try_into().unwrap()) as u64;
    let version = u32::from_le_bytes(foot[42..46].try_into().unwrap());
    if trailer_len + RECORD_HEADER_LEN > file_len {
        return Err(err(format!("bad trailer_len {trailer_len} for file of {file_len} bytes")));
    }
    let base = file_len - trailer_len - RECORD_HEADER_LEN;
    let dir_end = file_len - FOOTER_LEN;
    if dir_len == 0 || dir_len % 10 != 0 || dir_len > (1 << 20) || dir_end < dir_len || dir_end - dir_len < base {
        return Err(err(format!("bad directory length {dir_len}")));
    }
    let dir_start = dir_end - dir_len;
    f.seek(SeekFrom::Start(dir_start))?;
    let mut dir = vec![0u8; dir_len as usize];
    f.read_exact(&mut dir)?;

    let mut entries: Vec<RecordLoc> = Vec::new();
    for chunk in dir.chunks_exact(10) {
        let id = u16::from_le_bytes([chunk[0], chunk[1]]);
        let size = u32::from_le_bytes(chunk[2..6].try_into().unwrap()) as u64;
        let off = u32::from_le_bytes(chunk[6..10].try_into().unwrap()) as u64;
        if id == 0 && size == 0 && off == 0 {
            continue; // padding slot
        }
        let payload_start = base + off + RECORD_HEADER_LEN;
        if payload_start + size > dir_start {
            // Points past the record area — a stale / duplicate slot.
            continue;
        }
        // Duplicate ids (the identity record can be listed twice, once with
        // offset 0): keep the larger, real one.
        if let Some(existing) = entries.iter_mut().find(|e| e.id == id) {
            if size > existing.size {
                *existing = RecordLoc { id, payload_start, size };
            }
            continue;
        }
        entries.push(RecordLoc { id, payload_start, size });
    }
    if entries.is_empty() {
        return Err(err("empty Insta360 trailer directory"));
    }
    Ok((entries, version))
}

fn read_record(f: &mut std::fs::File, loc: RecordLoc) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(loc.payload_start))?;
    let mut buf = vec![0u8; loc.size as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read the identity record only (serial / model / firmware / lens
/// calibration). Cheap — a couple of KB from the end of the file.
pub fn read_insta360_identity(path: &Path) -> Result<Insta360Meta> {
    read_meta(path, false)
}

/// Read identity + IMU + frame stamps (everything the stabilizer needs).
/// Still cheap: ~1 MB per minute of footage, all at the end of the file.
pub fn read_insta360_meta(path: &Path) -> Result<Insta360Meta> {
    read_meta(path, true)
}

fn read_meta(path: &Path, with_motion: bool) -> Result<Insta360Meta> {
    let mut f = std::fs::File::open(path)?;
    let file_len = f.metadata()?.len();
    let (entries, version) = read_directory(&mut f, file_len)?;
    let mut meta = Insta360Meta {
        record_ids: entries.iter().map(|e| e.id).collect(),
        trailer_version: version,
        ..Default::default()
    };
    let find = |id: u16| entries.iter().copied().find(|e| e.id == id);

    if let Some(loc) = find(REC_IDENTITY) {
        let bytes = read_record(&mut f, loc)?;
        parse_identity(&bytes, &mut meta);
    }
    if with_motion {
        if let Some(loc) = find(REC_IMU) {
            let bytes = read_record(&mut f, loc)?;
            // 16-bit offset-binary samples: LSB/unit = 32768 / full-scale
            // range, taken from the file's `gyro_cfg_info` when present
            // (X6: ±32 g, ±2000 °/s — the constants).
            let acc_lsb = meta.acc_range_g.map(|r| 32768.0 / r as f32).unwrap_or(INSV_ACCEL_LSB_PER_G);
            let gyr_lsb = meta.gyro_range_dps.map(|r| 32768.0 / r as f32).unwrap_or(INSV_GYRO_LSB_PER_DPS);
            meta.imu = parse_imu(&bytes, acc_lsb, gyr_lsb);
        }
        if let Some(loc) = find(REC_FRAME_STAMPS) {
            let bytes = read_record(&mut f, loc)?;
            meta.frames = parse_frame_stamps(&bytes);
        }
        if let Some(loc) = find(REC_FRAME_STAMPS_B) {
            let bytes = read_record(&mut f, loc)?;
            meta.frames_b = parse_frame_stamps(&bytes);
        }
    }
    if meta.model.is_empty() && meta.lenses.is_empty() && meta.imu.is_empty() {
        return Err(err("Insta360 trailer has none of the expected records"));
    }
    Ok(meta)
}

// ── Identity record ────────────────────────────────────────────────

/// The identity record is Insta360's `ExtraMetadata` protobuf. Fields used:
/// 1 serial, 2 model, 3 firmware (strings); 19 `dimension {w, h}`;
/// 20 `frame_rate`; 22 `gamma_mode` (string); 79 `pano_record_type`;
/// 25 `rolling_shutter_time` (f64 ms); 27 `window_crop_info {src_w, src_h,
/// dst_w, dst_h, crop_offset_x, crop_offset_y}`; 28 `gyro_timestamp` (f64
/// ms); 65 `gyro_cfg_info {acc_range, gyro_range}`; 80
/// `pano_record_multi_track_order` (1 = track 0 carries lens 2); 131
/// `stream_type` (4 = dual-stream tracks in reverse order). The calibration
/// string (fields 5/17/111/112, all identical) is located by scanning for
/// the printable `2_…` run so any copy works.
fn parse_identity(bytes: &[u8], meta: &mut Insta360Meta) {
    let mut track_order = None;
    let mut stream_type = None;
    for (field_num, wire) in protobuf_fields(bytes) {
        match (field_num, wire) {
            (1 | 2 | 3, Wire::Len(payload)) => {
                if let Ok(s) = std::str::from_utf8(payload) {
                    if !s.is_empty() && s.chars().all(|c| !c.is_control()) {
                        match field_num {
                            1 => meta.serial = s.to_string(),
                            2 => meta.model = s.to_string(),
                            _ => meta.firmware = s.to_string(),
                        }
                    }
                }
            }
            (19, Wire::Len(payload)) => {
                let v = varint_fields(payload);
                if let (Some(w), Some(h)) = (v.get(&1), v.get(&2)) {
                    meta.dimension = Some((*w as u32, *h as u32));
                }
            }
            (20, Wire::Varint(v)) => meta.frame_rate = Some(v as u32).filter(|v| *v > 0),
            (22, Wire::Len(payload)) => {
                if let Ok(g) = std::str::from_utf8(payload) {
                    if !g.is_empty() && g.chars().all(|c| !c.is_control()) {
                        meta.gamma_mode = Some(g.to_string());
                    }
                }
            }
            (79, Wire::Varint(v)) => meta.pano_record_type = Some(v as u32),
            (27, Wire::Len(payload)) => {
                let v = varint_fields(payload);
                let g = |k: u64| v.get(&k).copied().unwrap_or(0);
                if g(1) > 0 && g(3) > 0 {
                    meta.window_crop = Some(InsvWindowCrop {
                        src_w: g(1) as u32,
                        src_h: g(2) as u32,
                        dst_w: g(3) as u32,
                        dst_h: g(4) as u32,
                        offset_x: g(5) as i64 as i32,
                        offset_y: g(6) as i64 as i32,
                    });
                }
            }
            (65, Wire::Len(payload)) => {
                let v = varint_fields(payload);
                meta.acc_range_g = v.get(&1).map(|&x| x as u32).filter(|&x| x > 0);
                meta.gyro_range_dps = v.get(&2).map(|&x| x as u32).filter(|&x| x > 0);
            }
            (25, Wire::F64(v)) => meta.readout_ms = Some(v).filter(|v| v.is_finite() && *v > 0.0),
            (28, Wire::F64(v)) => meta.gyro_offset_ms = Some(v).filter(|v| v.is_finite()),
            (80, Wire::Varint(v)) => track_order = Some(v),
            (131, Wire::Varint(v)) => stream_type = Some(v),
            _ => {}
        }
    }
    meta.track_order_reversed = match (track_order, stream_type) {
        (Some(1), _) => Some(true),
        (Some(2), _) => Some(false),
        (_, Some(4)) => Some(true),
        (_, Some(3)) => Some(false),
        _ => None,
    };
    if let Some((s, lenses)) = find_calib_string(bytes) {
        meta.calib_string = s;
        meta.lenses = lenses;
    }
}

enum Wire<'a> {
    Varint(u64),
    F64(f64),
    F32(f32),
    Len(&'a [u8]),
}

/// Walk a protobuf message's top-level fields. Stops at the first
/// malformed tag; unknown wire types end the walk.
fn protobuf_fields(bytes: &[u8]) -> Vec<(u32, Wire<'_>)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let Some((tag, next)) = read_varint(bytes, cursor) else { break };
        cursor = next;
        let field_num = (tag >> 3) as u32;
        match tag & 7 {
            0 => {
                let Some((v, next)) = read_varint(bytes, cursor) else { break };
                cursor = next;
                out.push((field_num, Wire::Varint(v)));
            }
            1 => {
                let Some(b) = bytes.get(cursor..cursor + 8) else { break };
                cursor += 8;
                out.push((field_num, Wire::F64(f64::from_le_bytes(b.try_into().unwrap()))));
            }
            2 => {
                let Some((len, next)) = read_varint(bytes, cursor) else { break };
                cursor = next;
                let end = cursor.saturating_add(len as usize);
                if end > bytes.len() { break; }
                out.push((field_num, Wire::Len(&bytes[cursor..end])));
                cursor = end;
            }
            5 => {
                let Some(b) = bytes.get(cursor..cursor + 4) else { break };
                cursor += 4;
                out.push((field_num, Wire::F32(f32::from_le_bytes(b.try_into().unwrap()))));
            }
            _ => break,
        }
    }
    out
}

/// Varint-valued fields of a small sub-message, keyed by field number.
fn varint_fields(bytes: &[u8]) -> std::collections::HashMap<u64, u64> {
    protobuf_fields(bytes)
        .into_iter()
        .filter_map(|(f, w)| match w { Wire::Varint(v) => Some((f as u64, v)), _ => None })
        .collect()
}

fn read_varint(buf: &[u8], mut cursor: usize) -> Option<(u64, usize)> {
    let mut shift = 0u32;
    let mut value = 0u64;
    loop {
        let b = *buf.get(cursor)?;
        cursor += 1;
        value |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((value, cursor));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_digit() || b == b'_' || b == b'.' || b == b'-' || b == b'e' || b == b'E'
}

/// Scan for the `2_<27 values per lens>…` calibration run.
fn find_calib_string(bytes: &[u8]) -> Option<(String, Vec<InsvLensCalib>)> {
    let n = bytes.len();
    let mut i = 0usize;
    while i + 2 < n {
        let starts_here = bytes[i] == b'2' && bytes[i + 1] == b'_'
            && (i == 0 || !is_token_byte(bytes[i - 1]));
        if !starts_here {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && is_token_byte(bytes[j]) {
            j += 1;
        }
        let run = std::str::from_utf8(&bytes[i..j]).unwrap_or("");
        if let Some(lenses) = parse_calib_string(run) {
            return Some((run.to_string(), lenses));
        }
        i = j.max(i + 1);
    }
    None
}

/// `2_` + 27 underscore-separated values per lens (+ a trailing tag).
pub fn parse_calib_string(s: &str) -> Option<Vec<InsvLensCalib>> {
    const PER_LENS: usize = 27;
    let tokens: Vec<&str> = s.split('_').collect();
    if tokens.len() < 1 + PER_LENS || tokens[0] != "2" {
        return None;
    }
    let mut lenses = Vec::new();
    let mut idx = 1usize;
    while idx + PER_LENS <= tokens.len() {
        let t = &tokens[idx..idx + PER_LENS];
        let f = |k: usize| t[k].parse::<f32>().ok();
        let u = |k: usize| t[k].parse::<f32>().ok().map(|v| v.round() as u32);
        let lens = InsvLensCalib {
            xi: f(0)?,
            fx: f(1)?, fy: f(2)?, cx: f(3)?, cy: f(4)?,
            yaw_deg: f(5)?, pitch_deg: f(6)?, roll_deg: f(7)?,
            extra: [f(8)?, f(9)?, f(10)?],
            radial: [f(11)?, f(12)?, f(13)?, f(14)?],
            tail: [f(15)?, f(16)?, f(17)?, f(18)?, f(19)?, f(20)?, f(21)?, f(22)?, f(23)?],
            sensor_w: u(24)?, sensor_h: u(25)?, tag: u(26)?,
        };
        if lens.fx <= 0.0 || lens.xi <= 0.0 || lens.sensor_w == 0 || lens.sensor_h == 0 {
            return None;
        }
        lenses.push(lens);
        idx += PER_LENS;
    }
    if lenses.is_empty() { None } else { Some(lenses) }
}

// ── IMU + frame stamps ─────────────────────────────────────────────

fn parse_imu(bytes: &[u8], acc_lsb_per_g: f32, gyr_lsb_per_dps: f32) -> Vec<InsvImuSample> {
    let acc_scale = 1.0 / acc_lsb_per_g;
    let gyr_scale = 1.0 / gyr_lsb_per_dps;
    let mut out = Vec::with_capacity(bytes.len() / 20);
    let mut last_t = 0u64;
    for rec in bytes.chunks_exact(20) {
        let t_us = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        // Zero / non-monotonic stamps mark unused tail padding.
        if t_us == 0 || (last_t != 0 && t_us <= last_t) {
            break;
        }
        last_t = t_us;
        let v = |k: usize| -> f32 {
            let raw = u16::from_le_bytes([rec[8 + 2 * k], rec[9 + 2 * k]]) as i32 - 32768;
            raw as f32
        };
        out.push(InsvImuSample {
            t_us,
            acc_g: [v(0) * acc_scale, v(1) * acc_scale, v(2) * acc_scale],
            gyr_dps: [v(3) * gyr_scale, v(4) * gyr_scale, v(5) * gyr_scale],
        });
    }
    out
}

fn parse_frame_stamps(bytes: &[u8]) -> Vec<InsvFrameStamp> {
    let mut out = Vec::with_capacity(bytes.len() / 16);
    for rec in bytes.chunks_exact(16) {
        let t_us = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        let exposure_s = f64::from_le_bytes(rec[8..16].try_into().unwrap());
        if t_us == 0 {
            break;
        }
        out.push(InsvFrameStamp { t_us, exposure_s });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const X6_STRING: &str = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193_2.455430_7227.070_7227.070_11569.100_3877.970_-0.132_0.081_89.544_-0.000064_-0.000049_-0.032576_1.31835759_-1.57224715_5.29361629_3.47188210_0.00000000_0.00016541_-0.00072159_-0.00095306_0.01344961_-0.00061933_-0.00056505_0.01970853_-0.02243911_15488_7744_193_394240";

    #[test]
    fn parses_x6_calib_string() {
        let lenses = parse_calib_string(X6_STRING).expect("parse");
        assert_eq!(lenses.len(), 2);
        assert!((lenses[0].fx - 7216.62).abs() < 1e-2);
        assert!((lenses[1].cx - 11569.1).abs() < 1e-2);
        assert_eq!(lenses[0].sensor_w, 15488);
        assert!((lenses[1].radial[3] - 3.47188210).abs() < 1e-6);
        assert!((lenses[0].roll_deg - 89.578).abs() < 1e-3);
        // Decoded projection model: ground-truth spot checks (native px).
        let l = &lenses[0];
        assert!((l.paraxial_focal() - 2088.5).abs() < 0.5, "{}", l.paraxial_focal());
        assert!((l.project_radius(45f64.to_radians()) - 1715.5).abs() < 1.0);
        assert!((l.project_radius(70f64.to_radians()) - 2760.1).abs() < 1.0);
    }

    #[test]
    fn finds_calib_string_inside_record() {
        let mut blob = vec![0x0a, 0x03, b'A', b'B', b'C', 0x12, 0x0b];
        blob.extend_from_slice(b"Insta360 X6");
        blob.extend_from_slice(&[0x1a, 0x08]);
        blob.extend_from_slice(b"v1.0.208");
        blob.extend_from_slice(&[0x22, 0xff, 0x01]);
        blob.extend_from_slice(X6_STRING.as_bytes());
        blob.push(0);
        let mut meta = Insta360Meta::default();
        parse_identity(&blob, &mut meta);
        assert_eq!(meta.serial, "ABC");
        assert_eq!(meta.model, "Insta360 X6");
        assert_eq!(meta.firmware, "v1.0.208");
        assert_eq!(meta.lenses.len(), 2);
        assert!(meta.is_x6());
    }

    #[test]
    fn imu_offset_binary_decode() {
        let mut rec = Vec::new();
        rec.extend_from_slice(&1_000_000u64.to_le_bytes());
        for v in [32768u16, 32768 + 1024, 32768, 32768 + 16, 32768 - 16, 32768] {
            rec.extend_from_slice(&v.to_le_bytes());
        }
        let s = parse_imu(&rec, INSV_ACCEL_LSB_PER_G, INSV_GYRO_LSB_PER_DPS);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].acc_g, [0.0, 1.0, 0.0]);
        assert!((s[0].gyr_dps[0] - 16.0 / 16.384).abs() < 1e-6);
        assert!((s[0].gyr_dps[1] + 16.0 / 16.384).abs() < 1e-6);
    }

    /// Real-file smoke test — skipped when the fixture isn't mounted.
    #[test]
    fn real_x6_clip() {
        let p = Path::new("/Volumes/Database/x6/concert/VID_20260815_193658_00_030.insv");
        if !p.exists() { return; }
        assert!(has_insv_trailer(p));
        let m = read_insta360_meta(p).expect("read");
        assert_eq!(m.model, "Insta360 X6");
        assert_eq!(m.lenses.len(), 2);
        assert_eq!(m.frames.len(), 2161);
        assert_eq!(m.frames_b.len(), 2161);
        // Second sensor, frame 0: same end stamp (+31 µs) but a 19.7 ms exposure
        // vs 10 ms → its mid-exposure sits 4.84 ms earlier.
        assert_eq!(m.frames_b[0].t_us, 2694093379);
        assert!((m.frames_b[0].exposure_s - 0.019744).abs() < 1e-6 && (m.frames[0].exposure_s - 0.01).abs() < 1e-9);
        assert!(m.imu.len() > 43_000);
        assert!((m.imu_rate_hz() - 995.7).abs() < 2.0, "rate {}", m.imu_rate_hz());
        assert!((m.readout_ms.unwrap() - 15.36).abs() < 1e-9);
        assert!((m.gyro_offset_ms.unwrap() - 1.1).abs() < 1e-9);
        assert_eq!(m.window_crop, Some(InsvWindowCrop { src_w: 7744, src_h: 7744, dst_w: 7680, dst_h: 7680, offset_x: 0, offset_y: 0 }));
        assert_eq!(m.dimension, Some((3840, 3840)));
        assert_eq!(m.track_order_reversed, Some(true));
        assert_eq!((m.acc_range_g, m.gyro_range_dps), (Some(32), Some(2000)));
    }

    /// The exact projection against reference values computed
    /// independently from the same factory string (entry 1 of clip 030).
    #[test]
    fn exact_projection_matches_reference() {
        let s = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193";
        let e = &parse_calib_string(s).unwrap()[0];
        assert_eq!(e.tangential(), [-0.00032840, 0.00113587, 0.00412989, 0.00653467]);
        assert_eq!(e.prism(), [-0.00106616, 0.02084749, -0.00235222, -0.01150168]);
        assert_eq!(e.half_origin_x(), 0.0);
        let e2 = &parse_calib_string("2_2.455430_7227.070_7227.070_11569.100_3877.970_-0.132_0.081_89.544_-0.000064_-0.000049_-0.032576_1.31835759_-1.57224715_5.29361629_3.47188210_0.00000000_0.00016541_-0.00072159_-0.00095306_0.01344961_-0.00061933_-0.00056505_0.01970853_-0.02243911_15488_7744_193").unwrap()[0];
        assert_eq!(e2.half_origin_x(), 7744.0);
        for (dir, want) in [
            ([0.3, -0.2, 0.9], (4525.5626, 3437.6202)),
            ([0.95, 0.4, -0.1], (7290.6577, 5326.5786)),
            ([0.0, 0.0, 1.0], (3855.46, 3884.53)),
            ([-0.6, 0.7, 0.2], (1851.4448, 6222.9365)),
        ] {
            let (u, v) = e.project(dir).unwrap();
            assert!((u - want.0).abs() < 2e-3 && (v - want.1).abs() < 2e-3, "{dir:?} → ({u}, {v}) want {want:?}");
        }
        assert!(e.project([0.0, 0.0, -1.0]).is_none());
    }

    fn win(dst_w: u32, dst_h: u32, offset_x: i32, offset_y: i32) -> InsvWindowCrop {
        InsvWindowCrop { src_w: 7744, src_h: 7744, dst_w, dst_h, offset_x, offset_y }
    }

    #[test]
    fn x6_modes_are_told_apart_by_size_and_window() {
        let c = |w: Option<InsvWindowCrop>, side: u32| X6Mode::classify(w.as_ref(), side, side);
        assert_eq!(c(Some(win(7680, 7680, 0, 0)), 3840), X6Mode::Stock);
        assert_eq!(c(None, 3840), X6Mode::Stock);
        assert_eq!(c(Some(win(7680, 7680, 0, 0)), 3712), X6Mode::Wide7K4);
        assert_eq!(c(Some(win(4912, 3968, 0, 0)), 3712), X6Mode::Crop7K4);
        // The two 3712² modes share size and rate: without the window (or
        // with one that fits neither) the file cannot say which it is.
        assert_eq!(c(None, 3712), X6Mode::Unconfirmed7K4);
        assert_eq!(c(Some(win(4912, 4912, 0, 0)), 3712), X6Mode::Unconfirmed7K4);
        assert_eq!(c(Some(win(7912, 7912, 4, 0)), 4864), X6Mode::Wide10K);
        assert_eq!(c(None, 4864), X6Mode::Wide10K);               // the older 9.7K
        assert_eq!(c(Some(win(4912, 4912, 0, 0)), 4912), X6Mode::Crop10K);
        assert_eq!(c(None, 3808), X6Mode::Crop7K6);
        // The 4928² test footage is not the 10K Crop — with or without a window.
        assert_eq!(c(Some(win(4928, 4928, 0, 0)), 4928), X6Mode::Unknown);
        assert_eq!(c(None, 4928), X6Mode::Unknown);
        // Any other window the table does not know.
        assert_eq!(c(Some(win(6000, 6000, 0, 0)), 3000), X6Mode::Unknown);
        assert!(X6Mode::Crop7K4.is_crop() && X6Mode::Crop10K.is_crop() && !X6Mode::Wide10K.is_crop());
        assert_eq!(X6Mode::Stock.label(), None);
        assert_eq!(X6Mode::Crop10K.label(), Some("10K Crop"));
    }

    #[test]
    fn x6_mode_geometry() {
        let r = |w: Option<InsvWindowCrop>, side: u32, ch| InsvStreamGeometry::resolve(w.as_ref(), side, side, ch);
        // Stock 8K: the 7680² window binned 2×2.
        let g = r(Some(win(7680, 7680, 0, 0)), 3840, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.scale_x, g.scale_y, g.origin_x, g.origin_y), (X6Mode::Stock, 0.5, 0.5, 32.0, 32.0));
        assert!(!g.cropped && g.readout_fraction == 1.0);
        // 7.4K: the same window scaled to 3712².
        let g = r(Some(win(7680, 7680, 0, 0)), 3712, X6ViewChoice::Auto);
        assert_eq!(g.mode, X6Mode::Wide7K4);
        assert!((g.scale_x - 3712.0 / 7680.0).abs() < 1e-12 && g.origin_x == 32.0 && !g.cropped);
        // 7.4K Crop: (128,128,3712,3712) of the central 3968² of the
        // 4912 × 3968 readout = the centred 3712² of the sensor half, 1:1,
        // 3712 of the 3968 rows that were read.
        let g = r(Some(win(4912, 3968, 0, 0)), 3712, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.scale_x, g.origin_x, g.origin_y), (X6Mode::Crop7K4, 1.0, 2016.0, 2016.0));
        assert_eq!((7744 - 4912) / 2 + (4912 - 3968) / 2 + 128, 2016);
        assert_eq!((7744 - 3968) / 2 + 128, 2016);
        assert!(g.cropped && (g.readout_fraction - 3712.0 / 3968.0).abs() < 1e-12);
        // 10K: the 7912² window (wider than the calibration frame, 4 px
        // right of centre) scaled to 4864².
        let g = r(Some(win(7912, 7912, 4, 0)), 4864, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.origin_x, g.origin_y), (X6Mode::Wide10K, -80.0, -84.0));
        assert!((g.scale_x - 4864.0 / 7912.0).abs() < 1e-12 && !g.cropped && g.readout_fraction == 1.0);
        // … and the same without the record (an older 9.7K file).
        assert_eq!(r(None, 4864, X6ViewChoice::Auto), g);
        // 10K Crop: the centred 4912², every row.
        let g = r(Some(win(4912, 4912, 0, 0)), 4912, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.scale_x, g.origin_x, g.origin_y, g.readout_fraction), (X6Mode::Crop10K, 1.0, 1416.0, 1416.0, 1.0));
        assert!(g.cropped);
        // … and a window that is not its own lends it nothing.
        assert_eq!(r(Some(win(7680, 7680, 9, 9)), 4912, X6ViewChoice::Auto), g);
        assert_eq!(r(None, 4912, X6ViewChoice::Auto), g);
        // Older 7.6K Crop: offset 80 inside the 3968² square.
        let g = r(None, 3808, X6ViewChoice::Auto);
        assert_eq!((g.mode, g.origin_x, g.origin_y), (X6Mode::Crop7K6, 1968.0, 1968.0));
        assert_eq!((7744 - 3968) / 2 + 80, 1968);
        assert!((g.readout_fraction - 3808.0 / 3968.0).abs() < 1e-12);
    }

    #[test]
    fn unconfirmed_7k4_follows_the_users_choice() {
        let r = |w: Option<InsvWindowCrop>, ch| InsvStreamGeometry::resolve(w.as_ref(), 3712, 3712, ch);
        let wide = r(Some(win(7680, 7680, 0, 0)), X6ViewChoice::Auto);
        let crop = r(Some(win(4912, 3968, 0, 0)), X6ViewChoice::Auto);
        for missing in [None, Some(win(4912, 4912, 9, 9))] {
            let auto = r(missing, X6ViewChoice::Auto);
            assert_eq!((auto.detected, auto.mode), (X6Mode::Unconfirmed7K4, X6Mode::Wide7K4));
            assert_eq!((auto.scale_x, auto.origin_x, auto.cropped), (wide.scale_x, wide.origin_x, false));
            let c = r(missing, X6ViewChoice::Crop);
            assert_eq!((c.detected, c.mode), (X6Mode::Unconfirmed7K4, X6Mode::Crop7K4));
            // A contradictory window contributes nothing — not its offset either.
            assert_eq!((c.scale_x, c.origin_x, c.origin_y, c.readout_fraction), (1.0, crop.origin_x, crop.origin_y, crop.readout_fraction));
            assert_eq!(r(missing, X6ViewChoice::Wide).mode, X6Mode::Wide7K4);
        }
        // A clip the file does identify ignores the choice.
        assert_eq!(r(Some(win(7680, 7680, 0, 0)), X6ViewChoice::Crop), wide);
        assert_eq!(r(Some(win(4912, 3968, 0, 0)), X6ViewChoice::Wide), crop);
    }

    #[test]
    fn principal_point_in_stream_coordinates() {
        let lenses = parse_calib_string(X6_STRING).unwrap();
        // Stock: (c − 32)/2 per lens, entry 2 relative to its own half.
        let g = InsvStreamGeometry::resolve(Some(&win(7680, 7680, 0, 0)), 3840, 3840, X6ViewChoice::Auto);
        let (x, y) = g.principal_point(&lenses[0]);
        assert!((x - (3855.46 - 32.0) / 2.0).abs() < 1e-3 && (y - (3884.53 - 32.0) / 2.0).abs() < 1e-3, "({x}, {y})");
        let (x2, _) = g.principal_point(&lenses[1]);
        assert!((x2 - (11569.1 - 7744.0 - 32.0) / 2.0).abs() < 1e-3, "{x2}");
        // 10K Crop: native pixels, 1416 in from the half's corner.
        let g = InsvStreamGeometry::resolve(Some(&win(4912, 4912, 0, 0)), 4912, 4912, X6ViewChoice::Auto);
        let (x, y) = g.principal_point(&lenses[0]);
        assert!((x - (3855.46 - 1416.0)).abs() < 1e-3 && (y - (3884.53 - 1416.0)).abs() < 1e-3);
    }

    /// The four new modes on the real samples — skipped when not mounted.
    #[test]
    fn real_x6_mode_samples() {
        let dir = Path::new("/Volumes/Database/x6/1110");
        let cases: [(&str, u32, X6Mode, f64, u32); 4] = [
            ("VID_20261110_144736_00_003.insv", 3712, X6Mode::Wide7K4, 14.56, 60),
            ("drop frame/VID_20261110_144829_00_004.insv", 3712, X6Mode::Crop7K4, 16.037333, 60),
            ("VID_20261110_145518_00_008.insv", 4864, X6Mode::Wide10K, 31.648, 30),
            ("VID_20261110_145135_00_006.insv", 4912, X6Mode::Crop10K, 21.285333, 30),
        ];
        for (name, side, mode, readout, fps) in cases {
            let p = dir.join(name);
            if !p.exists() { continue; }
            let m = read_insta360_identity(&p).expect("read");
            assert_eq!(m.dimension, Some((side, side)), "{name}");
            let g = m.stream_geometry(side, side, X6ViewChoice::Auto);
            assert_eq!((g.mode, g.detected), (mode, mode), "{name}");
            assert!((m.readout_ms.unwrap() - readout).abs() < 1e-5, "{name}");
            assert_eq!(m.frame_rate, Some(fps), "{name}");
            assert_eq!(m.gamma_mode.as_deref(), Some("I_Log"), "{name}");
            assert_eq!(m.is_ilog(), Some(true));
            assert_eq!((m.pano_record_type, m.track_order_reversed), (Some(2), Some(true)), "{name}");
            // The same file without its window record: the 3712² clip can
            // no longer be told apart, the others still can.
            let stripped = InsvStreamGeometry::resolve(None, side, side, X6ViewChoice::Auto);
            if side == 3712 {
                assert_eq!(stripped.detected, X6Mode::Unconfirmed7K4, "{name}");
            } else {
                assert_eq!((stripped.mode, stripped.scale_x, stripped.origin_x), (g.mode, g.scale_x, g.origin_x), "{name}");
            }
        }
    }
}
