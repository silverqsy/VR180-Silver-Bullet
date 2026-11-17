//! Insta360 X6 raw video: a take folder holding a paired Bayer capture.
//!
//! One recording is a FOLDER (`RAWVIDEO-YYYYMMDD-HHMMSS`), shown as one
//! clip. Both sensors record a 3840×3840 14-bit Bayer frame per time step:
//!
//! - `clip.json` — `layout` (`paired-bayer-v1`), `size`, `fps_setting`
//!   (30 / 50 / 60), the clock anchor of the start, the audio file names.
//! - `seal.json` — written when the camera has finished the take: the byte
//!   count of the capture, the number of pairs saved and the inclusive
//!   request window they come from. A folder without it is not finished.
//! - `capture.x6b` — the pairs (below).
//! - `calibration.json` — the factory lens string (the `2_…` string an
//!   `.insv` carries, see [`crate::insta360`]).
//! - `gyro.csv` (`counter_ticks,x,y,z`) + `gyro-info.json` — the gyro on a
//!   hardware counter (`counter_hz`, 19.2 MHz on the samples).
//! - `audio.aac` / `audio.wav` + `audio-info.json` — the sound and its
//!   timing on the same counter.
//!
//! ## `capture.x6b`
//!
//! Little-endian records, one per pair, each starting on a 4096-byte
//! boundary. A record is a 4096-byte header, then the first channel's 24
//! tiles and the second channel's 24 (3840×160 samples each), every tile
//! padded to 4096 bytes. The header (`<8s6I2Q48I`): an 8-byte signature,
//! the header length, the request number, width, height, tiles per eye, a
//! threshold field, the two exposure-start stamps in nanoseconds and the 48
//! payload lengths. Records may land out of request order (the camera
//! compresses in parallel), so pairs are ordered by request number, never by
//! file position; a request number may appear once.
//!
//! A tile is `u32` predictor byte count, the predictor stream and a tail
//! stream. The predictor stream codes SYMBOLS: per row, per 64 samples, one
//! header byte holds two bit widths (one per 32-sample half); a half is
//! `4 × width` bytes in eight interleaved lanes of four values. A value is a
//! zigzag difference from `(left₂ + up₂ + 1) >> 1`, the neighbours of the
//! same Bayer site (zero outside the tile). A symbol indexes the profile's
//! table of 14-bit levels; the tail stream adds the low four bits where the
//! table steps by 16 (a bitmap marks the groups of 128 samples that carry
//! them).
//!
//! The signature names the table, and must agree with `fps_setting`: 30 fps
//! keeps every sensor value, 50 and 60 fps store coarser steps below 13472
//! (the levels are the centres of those steps) and every value above.
//!
//! The samples are GBRG (`[1, 2, 0, 1]`), black level 64, white level 15408.
//!
//! ## A take of one lens
//!
//! A take may hold ONE sensor's frames instead (`layout`
//! `single-bayer-v1`): 7680×7680 samples of 10 bits at 30 fps, every sensor
//! value kept. `clip.json` then names the sensor (`eye`: `CAM0` / `CAM1`,
//! the channel a paired take stores it as) and states the sample format
//! (`cfa_pattern`, `black_level`, `white_level`); `seal.json` counts
//! `frames` inside a window of the sensor's own frame numbers
//! (`hardware_window`); `frames.csv` lists each record's request, sensor
//! frame number, stamp and integration time, and `exposure.csv` has the
//! integration time in the recorded sensor's column only.
//!
//! Its records: a 4096-byte header (signature `X6S8B01\0`, request number,
//! width, rows, tile count as `u32`; the sensor's frame number, the
//! exposure-start stamp and the integration time as `u64` nanoseconds; then
//! 24 payload lengths and 24 checksums), then 24 tiles of 7680×320 samples,
//! each padded to 4096 bytes. A tile is the predictor stream of the paired
//! format with no tail stream: a symbol is the 10-bit sample.

use crate::{Error, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Frame side of one eye of a paired take.
pub const EYE_SIDE: usize = 3840;
/// Rows of one of its tiles.
pub const TILE_ROWS: usize = 160;
/// Tiles of one eye (either layout).
pub const TILES_PER_EYE: usize = 24;
/// Samples of one tile of a paired take.
pub const TILE_SAMPLES: usize = EYE_SIDE * TILE_ROWS;
/// Samples of one eye of a paired take.
pub const EYE_SAMPLES: usize = EYE_SIDE * EYE_SIDE;
/// Frame side of a take of one lens.
pub const SINGLE_SIDE: usize = 7680;
/// Header length, and the boundary every record and tile is padded to.
pub const BLOCK: u64 = 4096;
/// The header's threshold field.
const THRESHOLD: u32 = 13472;
/// Bayer pattern of the samples, row-major 2×2 (0 = R, 1 = G, 2 = B).
pub const CFA: [u8; 4] = [1, 2, 0, 1];
/// Sensor black level of a paired take.
pub const BLACK_LEVEL: u16 = 64;
/// Sensor white level of a paired take (saturated samples read exactly this).
pub const WHITE_LEVEL: u16 = 15408;
/// Black and white level of a take of one lens that states none.
const SINGLE_BLACK_LEVEL: u16 = 64;
const SINGLE_WHITE_LEVEL: u16 = 1023;

pub const CLIP_FILE: &str = "clip.json";
pub const SEAL_FILE: &str = "seal.json";
pub const CAPTURE_FILE: &str = "capture.x6b";
pub const CALIBRATION_FILE: &str = "calibration.json";
pub const GYRO_FILE: &str = "gyro.csv";
pub const GYRO_INFO_FILE: &str = "gyro-info.json";
pub const ACCEL_FILE: &str = "accel.csv";
pub const ACCEL_INFO_FILE: &str = "accel-info.json";
pub const EXPOSURE_FILE: &str = "exposure.csv";
/// The index of a folder that holds a take's frames as DNG files (one line
/// per pair) in place of the capture.
pub const FRAMES_FILE: &str = "frames.jsonl";

/// How a capture holds its frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Both sensors: a 3840×3840 frame of each per record, 14 bits.
    Paired,
    /// One sensor (`channel`: the channel a paired take stores it as): a
    /// 7680×7680 frame per record, 10 bits.
    Single { channel: usize },
}

impl Layout {
    /// The `layout` of `clip.json`.
    fn name(self) -> &'static str {
        match self {
            Self::Paired => "paired-bayer-v1",
            Self::Single { .. } => "single-bayer-v1",
        }
    }

    /// The layout a `clip.json` states (`eye` names a single layout's sensor).
    fn of_clip(clip: &serde_json::Value) -> Option<Self> {
        match clip.get("layout").and_then(|l| l.as_str())? {
            "paired-bayer-v1" => Some(Self::Paired),
            "single-bayer-v1" => match clip.get("eye").and_then(|e| e.as_str())? {
                "CAM0" => Some(Self::Single { channel: 0 }),
                "CAM1" => Some(Self::Single { channel: 1 }),
                _ => None,
            },
            _ => None,
        }
    }

    /// Channels a record holds.
    pub fn channels(self) -> usize {
        match self {
            Self::Paired => 2,
            Self::Single { .. } => 1,
        }
    }

    /// The one sensor a take of one lens holds, as a paired take's channel.
    pub fn single_channel(self) -> Option<usize> {
        match self {
            Self::Paired => None,
            Self::Single { channel } => Some(channel),
        }
    }

    /// Frame side of one channel.
    pub fn side(self) -> usize {
        match self {
            Self::Paired => EYE_SIDE,
            Self::Single { .. } => SINGLE_SIDE,
        }
    }

    /// Rows of one tile.
    pub fn tile_rows(self) -> usize {
        self.side() / TILES_PER_EYE
    }

    /// Samples of one tile.
    pub fn tile_samples(self) -> usize {
        self.side() * self.tile_rows()
    }

    /// Samples of one channel.
    pub fn eye_samples(self) -> usize {
        self.side() * self.side()
    }

    /// Samples of one record: its channels one after the other.
    pub fn samples(self) -> usize {
        self.channels() * self.eye_samples()
    }

    /// Tiles of one record.
    fn tiles(self) -> usize {
        self.channels() * TILES_PER_EYE
    }
}

/// What a take's records must be: its layout and frame rate setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub layout: Layout,
    pub fps: u32,
}

impl Format {
    /// A paired take at `fps`.
    pub fn paired(fps: u32) -> Self {
        Self { layout: Layout::Paired, fps }
    }
}

fn err(msg: impl Into<String>) -> Error {
    Error::Insv(format!("raw take: {}", msg.into()))
}

/// How a record's symbols map to sensor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// 30 fps: every sensor value kept.
    Exact30,
    /// 50 fps, the earlier table.
    Mid50,
    /// 50 fps, the current table (finer between 1024 and 2047).
    Raised50,
    /// 60 fps, the earlier table.
    Coarse60,
    /// 60 fps, the current table (finer between 1024 and 2047).
    Raised60,
    /// One lens at 30 fps: every 10-bit sensor value kept.
    Single10,
}

impl Profile {
    pub const ALL: [Profile; 6] = [Self::Exact30, Self::Mid50, Self::Raised50, Self::Coarse60, Self::Raised60, Self::Single10];

    /// The record signature.
    pub fn signature(self) -> &'static [u8; 8] {
        match self {
            Self::Exact30 => b"L1024___",
            Self::Mid50 => b"X6M50ABC",
            Self::Raised50 => b"X6M50AB2",
            Self::Coarse60 => b"X6QBLK05",
            Self::Raised60 => b"X6QBLK0B",
            Self::Single10 => b"X6S8B01\0",
        }
    }

    /// The frame rate setting this signature belongs to.
    pub fn fps(self) -> u32 {
        match self {
            Self::Exact30 | Self::Single10 => 30,
            Self::Mid50 | Self::Raised50 => 50,
            Self::Coarse60 | Self::Raised60 => 60,
        }
    }

    /// The profile of a signature recorded at `fps`; `None` for an unknown
    /// signature or one that belongs to another frame rate.
    pub fn from_signature(signature: &[u8], fps: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.signature()[..] == *signature && p.fps() == fps)
    }

    /// The layout whose records carry this signature.
    pub fn layout_is(self, layout: Layout) -> bool {
        (self == Self::Single10) == matches!(layout, Layout::Single { .. })
    }

    /// Widest bit width a block may use.
    fn max_width(self) -> usize {
        if self == Self::Exact30 { 15 } else { 11 }
    }

    /// `(first value, step, count)` runs of the level table.
    fn runs(self) -> &'static [(u16, u16, u16)] {
        match self {
            Self::Exact30 => &[(0, 1, 1024), (1024, 16, 960)],
            Self::Mid50 => &[(0, 4, 64), (256, 16, 48), (1024, 64, 48), (4096, 128, 73), (13440, 32, 1), (13472, 16, 182)],
            Self::Raised50 => &[(0, 4, 64), (256, 16, 48), (1024, 32, 32), (2048, 64, 32), (4096, 128, 73), (13440, 32, 1), (13472, 16, 182)],
            Self::Coarse60 => &[(0, 8, 32), (256, 32, 24), (1024, 128, 24), (4096, 256, 16), (8192, 512, 10), (13312, 160, 1), (13472, 16, 182)],
            Self::Raised60 => &[(0, 8, 32), (256, 32, 24), (1024, 64, 16), (2048, 128, 16), (4096, 256, 16), (8192, 512, 10), (13312, 160, 1), (13472, 16, 182)],
            Self::Single10 => &[(0, 1, 1024)],
        }
    }

    /// Symbol → sensor level, in symbol order.
    pub fn levels(self) -> &'static [u16] {
        static TABLES: OnceLock<[Vec<u16>; 6]> = OnceLock::new();
        let tables = TABLES.get_or_init(|| {
            Profile::ALL.map(|p| {
                p.runs().iter()
                    .flat_map(|&(first, step, count)| (0..count).map(move |i| first + i * step))
                    .collect()
            })
        });
        &tables[Self::ALL.iter().position(|p| *p == self).unwrap_or(0)]
    }

    /// The values of this profile are every sensor value (no coarser steps).
    pub fn is_exact(self) -> bool {
        matches!(self, Self::Exact30 | Self::Single10)
    }
}

/// The clock anchor a take writes at its start and end: one reading of the
/// hardware counter between two readings of the system clock.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClockAnchor {
    pub counter_ticks: u64,
    pub counter_hz: u64,
    pub mono_before_ns: u64,
    pub mono_after_ns: u64,
    pub boottime_ns: u64,
}

impl ClockAnchor {
    fn from_json(v: &serde_json::Value) -> Option<Self> {
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64());
        Some(Self {
            counter_ticks: u("counter_ticks")?,
            counter_hz: u("counter_hz")?,
            mono_before_ns: u("mono_before_ns").unwrap_or(0),
            mono_after_ns: u("mono_after_ns").unwrap_or(0),
            boottime_ns: u("boottime_ns").unwrap_or(0),
        })
    }
}

/// `seal.json`: the camera's closing record of a finished take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seal {
    pub fps_setting: u32,
    /// First and last request number the take may hold (both included). A
    /// take of one lens states the sensor's own frame numbers here, which
    /// its records do not go by: see [`Take::window_numbers_records`].
    pub request_window: (u32, u32),
    /// Pairs the camera received (may exceed `pairs` when the card fell behind).
    pub received_pairs: u64,
    /// Complete pairs in the capture (records, for a take of one lens).
    pub pairs: u64,
    /// Length of the capture in bytes.
    pub bytes: u64,
}

/// Why a folder that looks like a take cannot be opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TakeProblem {
    /// No `seal.json`: the camera has not finished (or never finished) it.
    NotSealed,
    /// The capture is not the length the seal states.
    LengthMismatch { sealed: u64, actual: u64 },
    /// The seal states no pairs.
    Empty,
    /// No `capture.x6b`.
    NoCapture,
    /// No `capture.x6b` but an index of DNG frames: the folder a take's
    /// frames were written into, not the take.
    FramesOnly,
}

impl std::fmt::Display for TakeProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSealed => write!(f, "the take was never closed by the camera (no {SEAL_FILE})"),
            Self::LengthMismatch { sealed, actual } =>
                write!(f, "the capture is {actual} bytes but the camera closed it at {sealed} (an incomplete copy?)"),
            Self::Empty => write!(f, "the take holds no frames"),
            Self::NoCapture => write!(f, "the take has no {CAPTURE_FILE}"),
            Self::FramesOnly => write!(f, "the folder holds a take's frames as DNG files, not the capture ({CAPTURE_FILE}): open the take they were written from"),
        }
    }
}

/// A take folder's own description (the two small JSON files and the size
/// of the capture; the capture itself is not read).
#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    pub dir: PathBuf,
    pub layout: Layout,
    pub fps_setting: u32,
    /// One channel's frame.
    pub width: u32,
    pub height: u32,
    /// Bayer pattern of the samples, row-major 2×2 (0 = R, 1 = G, 2 = B).
    pub cfa: [u8; 4],
    pub black_level: u16,
    /// Saturated samples read this.
    pub white_level: u16,
    pub clock_start: Option<ClockAnchor>,
    /// Sound file inside the folder (`clip.json` → `audio_file`).
    pub audio_file: Option<String>,
    /// Its timing record (`clip.json` → `audio_timing`).
    pub audio_timing: Option<String>,
    /// The sensor record (`clip.json` → `sensor_metadata`): the readout
    /// timing and the motion streams. Takes of an earlier camera build have
    /// none.
    pub sensor_metadata: Option<String>,
    /// The list of each pair's integration times (`clip.json` →
    /// `exposure_file`); a take that does not name one may still have
    /// [`EXPOSURE_FILE`] beside the capture.
    pub exposure_file: Option<String>,
    pub seal: Seal,
    pub capture_bytes: u64,
}

impl Take {
    pub fn capture_path(&self) -> PathBuf {
        self.dir.join(CAPTURE_FILE)
    }

    /// What the take's records must be.
    pub fn format(&self) -> Format {
        Format { layout: self.layout, fps: self.fps_setting }
    }

    /// The largest value a sample of the take can have (the top of its
    /// profiles' level tables, low bits included).
    pub fn sample_max(&self) -> u16 {
        match self.layout {
            Layout::Paired => 16383,
            Layout::Single { .. } => *Profile::Single10.levels().last().unwrap_or(&1023),
        }
    }

    /// The seal's window is of the numbers the records carry (a paired
    /// take's request numbers), so every record lies inside it and a number
    /// of it without a record is a frame that was never saved.
    pub fn window_numbers_records(&self) -> bool {
        self.layout == Layout::Paired
    }

    /// Frame rate of a constant-rate file made from the take.
    pub fn nominal_fps(&self) -> f64 {
        match self.fps_setting {
            30 => 30000.0 / 1001.0,
            60 => 60000.0 / 1001.0,
            n => n as f64,
        }
    }

    /// Pairs the camera saved.
    pub fn pairs(&self) -> u64 {
        self.seal.pairs
    }

    /// Requests of the window that were never saved.
    pub fn missing_pairs(&self) -> u64 {
        let (lo, hi) = self.seal.request_window;
        (hi as u64 + 1).saturating_sub(lo as u64).saturating_sub(self.seal.pairs)
    }
}

/// The take folder a path stands for: the folder itself, or one of the
/// take's own files inside it (`capture.x6b`, `clip.json`, …). `None` when
/// the folder has no `clip.json` with a layout this module reads.
pub fn take_dir(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        let name = path.file_name()?.to_str()?;
        let own = [CLIP_FILE, SEAL_FILE, CAPTURE_FILE, CALIBRATION_FILE, GYRO_FILE, GYRO_INFO_FILE];
        if !own.iter().any(|f| f.eq_ignore_ascii_case(name)) {
            return None;
        }
        path.parent()?.to_path_buf()
    };
    let clip = std::fs::read(dir.join(CLIP_FILE)).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&clip).ok()?;
    Layout::of_clip(&v).map(|_| dir)
}

/// The layout of the take `path` names, from its `clip.json` alone.
pub fn take_layout(path: &Path) -> Option<Layout> {
    let dir = take_dir(path)?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(CLIP_FILE)).ok()?).ok()?;
    Layout::of_clip(&v)
}

/// Read a take's description. `Err` for a folder that is no take at all;
/// `Ok(Err(problem))` for a take that cannot be opened as it is.
pub fn read_take(dir: &Path) -> Result<std::result::Result<Take, TakeProblem>> {
    let clip: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(CLIP_FILE))?)?;
    let Some(layout) = Layout::of_clip(&clip) else {
        return Err(err(format!("{}: not a raw take this build reads", dir.display())));
    };
    let fps_setting = clip.get("fps_setting").and_then(|v| v.as_u64()).ok_or_else(|| err("clip.json has no fps_setting"))? as u32;
    let size: Vec<u64> = clip.get("size").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
    let side = layout.side();
    if size != [side as u64, side as u64] {
        return Err(err(format!("unsupported frame size {size:?}")));
    }
    if !Profile::ALL.iter().any(|p| p.fps() == fps_setting && p.layout_is(layout)) {
        return Err(err(format!("unsupported frame rate setting {fps_setting}")));
    }
    // The sample format: a paired take's is fixed, a take of one lens states its own.
    let (cfa, black_level, white_level) = match layout {
        Layout::Paired => (CFA, BLACK_LEVEL, WHITE_LEVEL),
        Layout::Single { .. } => {
            let pattern: Vec<u64> = clip.get("cfa_pattern").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
            let cfa = match pattern.as_slice() {
                [] => CFA,
                &[a, b, c, d] if [a, b, c, d].iter().all(|v| *v <= 2) => [a as u8, b as u8, c as u8, d as u8],
                _ => return Err(err(format!("unsupported Bayer pattern {pattern:?}"))),
            };
            // Two greens, one red, one blue, the greens on a diagonal.
            let count = |colour: u8| cfa.iter().filter(|c| **c == colour).count();
            if (count(0), count(1), count(2)) != (1, 2, 1) || (cfa[0] == 1) != (cfa[3] == 1) {
                return Err(err(format!("unsupported Bayer pattern {cfa:?}")));
            }
            let level = |k: &str, or: u16| clip.get(k).and_then(|v| v.as_u64()).map_or(Ok(or), |v| u16::try_from(v).map_err(|_| err(format!("{k} {v} out of range"))));
            let (black, white) = (level("black_level", SINGLE_BLACK_LEVEL)?, level("white_level", SINGLE_WHITE_LEVEL)?);
            let top = *Profile::Single10.levels().last().unwrap_or(&0);
            if black >= white || white > top {
                return Err(err(format!("unsupported levels: black {black}, white {white}")));
            }
            (cfa, black, white)
        }
    };
    let text = |k: &str| clip.get(k).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(str::to_owned);
    let capture = dir.join(CAPTURE_FILE);
    let Ok(meta) = std::fs::metadata(&capture) else {
        return Ok(Err(if dir.join(FRAMES_FILE).is_file() { TakeProblem::FramesOnly } else { TakeProblem::NoCapture }));
    };
    let Ok(seal_bytes) = std::fs::read(dir.join(SEAL_FILE)) else { return Ok(Err(TakeProblem::NotSealed)) };
    let seal: serde_json::Value = serde_json::from_slice(&seal_bytes)?;
    let u = |k: &str| seal.get(k).and_then(|v| v.as_u64());
    let (window_key, count_key) = match layout {
        Layout::Paired => ("request_window", "pairs"),
        Layout::Single { .. } => ("hardware_window", "frames"),
    };
    let window: Vec<u64> = seal.get(window_key).and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
    let (Some(seal_fps), Some(pairs), Some(bytes), &[lo, hi]) = (u("fps_setting"), u(count_key), u("bytes"), window.as_slice()) else {
        return Err(err("seal.json is incomplete"));
    };
    let received = u("received_pairs").unwrap_or(pairs);
    if seal_fps as u32 != fps_setting {
        return Err(err(format!("clip.json says {fps_setting} fps, seal.json {seal_fps}")));
    }
    // A take of one lens is closed for the sensor its clip names.
    if matches!(layout, Layout::Single { .. }) && seal.get("eye") != clip.get("eye") {
        return Err(err("clip.json and seal.json name different sensors"));
    }
    if hi < lo || pairs > hi - lo + 1 || received < pairs || received > hi - lo + 1 || hi > u32::MAX as u64 {
        return Err(err("seal.json is inconsistent"));
    }
    if pairs == 0 {
        return Ok(Err(TakeProblem::Empty));
    }
    if meta.len() != bytes {
        return Ok(Err(TakeProblem::LengthMismatch { sealed: bytes, actual: meta.len() }));
    }
    Ok(Ok(Take {
        dir: dir.to_path_buf(),
        layout,
        fps_setting,
        width: side as u32,
        height: side as u32,
        cfa,
        black_level,
        white_level,
        clock_start: clip.get("clock_start").and_then(ClockAnchor::from_json),
        audio_file: text("audio_file"),
        audio_timing: text("audio_timing"),
        sensor_metadata: text("sensor_metadata"),
        exposure_file: text("exposure_file"),
        seal: Seal { fps_setting, request_window: (lo as u32, hi as u32), received_pairs: received, pairs, bytes },
        capture_bytes: meta.len(),
    }))
}

/// One pair of the capture (one record: a take of one lens has one frame
/// in it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairEntry {
    /// The camera's request number (the pair's identity and order).
    pub request: u32,
    /// Byte offset of the record's header.
    pub offset: u64,
    /// Length of the whole record, header included.
    pub len: u64,
    pub profile: Profile,
    /// Exposure start of each channel, nanoseconds on the take's counter
    /// (a take of one lens: its one stamp, twice).
    pub exposure_ns: [u64; 2],
}

impl PairEntry {
    /// The pair's time: the middle of the two exposure starts, nanoseconds.
    pub fn time_ns(&self) -> u64 {
        self.exposure_ns[0] / 2 + self.exposure_ns[1] / 2 + (self.exposure_ns[0] % 2 + self.exposure_ns[1] % 2) / 2
    }
}

/// A record header, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Header {
    profile: Profile,
    request: u32,
    exposure_ns: [u64; 2],
    /// The payload lengths of the record's tiles (`tiles` of them).
    tile_len: [u32; 2 * TILES_PER_EYE],
    tiles: usize,
}

impl Header {
    fn parse(h: &[u8], format: Format, at: u64) -> Result<Self> {
        if h.len() < BLOCK as usize {
            return Err(err(format!("truncated header at byte {at}")));
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let u64_at = |o: usize| u64::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3], h[o + 4], h[o + 5], h[o + 6], h[o + 7]]);
        let layout = format.layout;
        let profile = Profile::from_signature(&h[..8], format.fps).filter(|p| p.layout_is(layout)).ok_or_else(|| {
            let shown: String = h[..8].iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect();
            err(format!("unsupported signature \"{shown}\" for {} fps at byte {at}", format.fps))
        })?;
        let side = layout.side() as u32;
        // Where the fields sit: a paired record's header, or one lens's.
        let (fields_ok, request, exposure_ns, lens_at) = match layout {
            Layout::Paired => (
                (u32_at(8), u32_at(16), u32_at(20), u32_at(24), u32_at(28)) == (BLOCK as u32, side, side, TILES_PER_EYE as u32, THRESHOLD),
                u32_at(12), [u64_at(32), u64_at(40)], 48,
            ),
            Layout::Single { .. } => (
                (u32_at(12), u32_at(16), u32_at(20)) == (side, side, TILES_PER_EYE as u32),
                u32_at(8), [u64_at(32); 2], 48,
            ),
        };
        if !fields_ok {
            return Err(err(format!("invalid header at byte {at}")));
        }
        let tiles = layout.tiles();
        let mut tile_len = [0u32; 2 * TILES_PER_EYE];
        for (i, len) in tile_len.iter_mut().take(tiles).enumerate() {
            *len = u32_at(lens_at + 4 * i);
            if *len <= 4 || *len as usize > layout.tile_samples() * 4 {
                return Err(err(format!("invalid tile size at byte {at}")));
            }
        }
        Ok(Self { profile, request, exposure_ns, tile_len, tiles })
    }

    /// Length of the record, header and padding included.
    fn record_len(&self) -> u64 {
        BLOCK + self.tile_len[..self.tiles].iter().map(|&n| pad(n as u64)).sum::<u64>()
    }
}

fn pad(n: u64) -> u64 {
    n.div_ceil(BLOCK) * BLOCK
}

/// Walk the capture and list its pairs in request order. Reads one header
/// per pair; `progress(bytes walked, bytes total)` returning `false` stops
/// the walk (`Ok(None)`).
///
/// Checks what the camera states against what the file holds: every
/// signature against the layout and frame rate, the header fields, every
/// record inside the file, request numbers used once (and inside the
/// window, where the window is of request numbers), positive stamps, and
/// the number of pairs against the seal.
pub fn scan_pairs(take: &Take, mut progress: impl FnMut(u64, u64) -> bool) -> Result<Option<Vec<PairEntry>>> {
    let mut file = File::open(take.capture_path())?;
    let size = take.capture_bytes;
    let (lo, hi) = take.seal.request_window;
    let mut pairs: Vec<PairEntry> = Vec::with_capacity(take.seal.pairs as usize);
    let mut seen = std::collections::HashSet::with_capacity(take.seal.pairs as usize);
    let mut head = vec![0u8; BLOCK as usize];
    let mut at = 0u64;
    while at < size {
        if !progress(at, size) {
            return Ok(None);
        }
        file.seek(SeekFrom::Start(at))?;
        file.read_exact(&mut head).map_err(|_| err(format!("truncated header at byte {at}")))?;
        let h = Header::parse(&head, take.format(), at)?;
        if take.window_numbers_records() && (h.request < lo || h.request > hi) {
            return Err(err(format!("request {} at byte {at} is outside the take's window {lo}..={hi}", h.request)));
        }
        if !seen.insert(h.request) {
            return Err(err(format!("duplicate request {} at byte {at}", h.request)));
        }
        if h.exposure_ns[0] == 0 || h.exposure_ns[1] == 0 {
            return Err(err(format!("request {} at byte {at} has no exposure time", h.request)));
        }
        let len = h.record_len();
        if at + len > size {
            return Err(err(format!("truncated pair at byte {at}")));
        }
        pairs.push(PairEntry { request: h.request, offset: at, len, profile: h.profile, exposure_ns: h.exposure_ns });
        at += len;
    }
    if pairs.len() as u64 != take.seal.pairs {
        return Err(err(format!("the capture holds {} pairs, the camera closed it with {}", pairs.len(), take.seal.pairs)));
    }
    pairs.sort_by_key(|p| p.request);
    progress(size, size);
    Ok(Some(pairs))
}

/// Request numbers of the window the capture does not hold, in order. For a
/// take whose window is not of request numbers: those between its first and
/// last record that no record has.
pub fn missing_requests(take: &Take, pairs: &[PairEntry]) -> Vec<u32> {
    let (lo, hi) = if take.window_numbers_records() {
        take.seal.request_window
    } else {
        match (pairs.first(), pairs.last()) {
            (Some(first), Some(last)) => (first.request, last.request),
            _ => return Vec::new(),
        }
    };
    let mut have = pairs.iter().map(|p| p.request).peekable();
    let mut out = Vec::new();
    for r in lo..=hi {
        if have.peek() == Some(&r) { have.next(); } else { out.push(r); }
    }
    out
}

/// Which channels of a pair to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Both,
    /// Only channel 0 or 1; the other half of the output is left untouched.
    /// (A take of one lens has its one channel whatever is asked for.)
    One(usize),
}

/// Read one record whole (`entry.len` bytes) into `buf`.
pub fn read_record(file: &mut File, entry: &PairEntry, buf: &mut Vec<u8>) -> Result<()> {
    buf.resize(entry.len as usize, 0);
    file.seek(SeekFrom::Start(entry.offset))?;
    file.read_exact(buf).map_err(|_| err(format!("truncated pair at byte {}", entry.offset)))?;
    Ok(())
}

/// Decode a record read by [`read_record`] into `out`, `format.layout`'s
/// samples: a paired take's channel 0 (3840×3840, 14-bit), then its channel
/// 1; a take of one lens's one channel (7680×7680, 10-bit). Tiles decode on
/// `threads` threads (0 = one per core, at most one per tile).
pub fn decode_record(record: &[u8], format: Format, channels: Channels, out: &mut [u16], threads: usize) -> Result<PairEntry> {
    let layout = format.layout;
    if out.len() != layout.samples() {
        return Err(err("output buffer is not the record's size"));
    }
    let h = Header::parse(record, format, 0)?;
    if (record.len() as u64) < h.record_len() {
        return Err(err("truncated pair"));
    }
    // Tile payloads, in file order.
    let mut tiles: Vec<(usize, &[u8])> = Vec::with_capacity(h.tiles);
    let mut at = BLOCK as usize;
    for (i, &n) in h.tile_len[..h.tiles].iter().enumerate() {
        tiles.push((i, &record[at..at + n as usize]));
        at += pad(n as u64) as usize;
    }
    let wanted = |tile: usize| match channels {
        Channels::One(c) if layout.channels() > 1 => tile / TILES_PER_EYE == c,
        _ => true,
    };
    let jobs: Vec<(&[u8], &mut [u16])> = tiles.iter()
        .zip(out.chunks_mut(layout.tile_samples()))
        .filter(|((i, _), _)| wanted(*i))
        .map(|((_, bytes), dst)| (*bytes, dst))
        .collect();
    let profile = h.profile;
    let workers = match threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        n => n,
    }.clamp(1, jobs.len().max(1));
    if workers == 1 {
        for (bytes, dst) in jobs {
            decode_tile(bytes, profile, layout, dst)?;
        }
    } else {
        let queue = std::sync::Mutex::new(jobs.into_iter());
        let failed: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let job = queue.lock().ok().and_then(|mut q| q.next());
                    let Some((bytes, dst)) = job else { break };
                    if let Err(e) = decode_tile(bytes, profile, layout, dst) {
                        if let Ok(mut slot) = failed.lock() { slot.get_or_insert(e); }
                        break;
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().ok().flatten() {
            return Err(e);
        }
    }
    Ok(PairEntry { request: h.request, offset: 0, len: h.record_len(), profile, exposure_ns: h.exposure_ns })
}

/// Decode one tile (`bytes` = its payload without padding) into its
/// samples (`layout`'s tile: 3840×160 or 7680×320).
fn decode_tile(bytes: &[u8], profile: Profile, layout: Layout, dst: &mut [u16]) -> Result<()> {
    debug_assert_eq!(dst.len(), layout.tile_samples());
    let main = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if main > bytes.len() - 4 {
        return Err(err("invalid predictor length"));
    }
    let levels = profile.levels();
    unpack_symbols(&bytes[4..4 + main], dst, layout.side(), profile.max_width(), levels.len() as i32)?;
    let tail = &bytes[4 + main..];
    // One lens's symbols are its samples: nothing follows the predictor stream.
    if profile == Profile::Single10 {
        return if tail.is_empty() { Ok(()) } else { Err(err("tile has bytes left over")) };
    }
    apply_levels(tail, levels, dst)
}

/// The predictor stream of one tile (rows of `width` samples) → symbols in
/// `dst`.
fn unpack_symbols(stream: &[u8], dst: &mut [u16], width: usize, max_width: usize, count: i32) -> Result<()> {
    let w = width;
    let rows = dst.len() / w;
    let mut p = 0usize;
    for y in 0..rows {
        // The row above the one being written, two rows up (same Bayer site).
        let (done, rest) = dst.split_at_mut(y * w);
        let row = &mut rest[..w];
        let up: Option<&[u16]> = (y >= 2).then(|| &done[(y - 2) * w..(y - 1) * w]);
        for x in (0..w).step_by(64) {
            let Some(&widths) = stream.get(p) else { return Err(err("predictor stream ends early")) };
            p += 1;
            for half in 0..2 {
                let b = ((widths >> (4 * half)) & 15) as usize;
                if b > max_width || stream.len() - p < 4 * b {
                    return Err(err("invalid block length"));
                }
                let block = &stream[p..p + 4 * b];
                p += 4 * b;
                let mut lanes = [0u64; 8];
                let whole = b / 2;
                for (lane, word) in lanes.iter_mut().enumerate() {
                    let mut bits = 0u64;
                    for k in 0..whole {
                        bits |= (block[k * 8 + lane] as u64) << (8 * k);
                    }
                    if b & 1 == 1 {
                        bits |= (((block[whole * 8 + lane / 2] >> (4 * (lane & 1))) & 15) as u64) << (8 * whole);
                    }
                    *word = bits;
                }
                let mask = (1u64 << b) - 1;
                let x0 = x + half * 32;
                for j in 0..32 {
                    let v = ((lanes[j % 8] >> (b * (j / 8))) & mask) as i32;
                    let col = x0 + j;
                    let left = if col >= 2 { row[col - 2] as i32 } else { 0 };
                    let above = up.map_or(0, |u| u[col] as i32);
                    let delta = if v & 1 == 1 { -((v + 1) >> 1) } else { v >> 1 };
                    let symbol = ((left + above + 1) >> 1) + delta;
                    if symbol < 0 || symbol >= count {
                        return Err(err("invalid symbol"));
                    }
                    row[col] = symbol as u16;
                }
            }
        }
    }
    if p != stream.len() {
        return Err(err("predictor stream has bytes left over"));
    }
    Ok(())
}

/// Symbols → sensor values, in place: the level table plus the tail
/// stream's low four bits. An empty tail stream means none.
fn apply_levels(tail: &[u8], levels: &[u16], dst: &mut [u16]) -> Result<()> {
    const GROUP: usize = 128;
    let groups = dst.len() / GROUP;
    let bitmap_len = groups.div_ceil(8);
    if tail.is_empty() {
        for v in dst.iter_mut() {
            *v = levels[*v as usize];
        }
        return Ok(());
    }
    if tail.len() < bitmap_len {
        return Err(err("tail stream ends early"));
    }
    let mut at = bitmap_len;
    for (g, samples) in dst.chunks_mut(GROUP).enumerate() {
        if tail[g >> 3] & (1 << (g & 7)) != 0 {
            let Some(nibbles) = tail.get(at..at + GROUP / 2) else { return Err(err("tail stream ends early")) };
            at += GROUP / 2;
            for (k, v) in samples.iter_mut().enumerate() {
                *v = levels[*v as usize] + ((nibbles[k / 2] >> (4 * (k & 1))) & 15) as u16;
            }
        } else {
            for v in samples.iter_mut() {
                *v = levels[*v as usize];
            }
        }
    }
    if at != tail.len() {
        return Err(err("tail stream has bytes left over"));
    }
    Ok(())
}

/// Read and decode the pair at `entry`. `record` is scratch space reused
/// across calls.
pub fn decode_pair(file: &mut File, entry: &PairEntry, format: Format, channels: Channels, record: &mut Vec<u8>, out: &mut [u16], threads: usize) -> Result<()> {
    read_record(file, entry, record)?;
    let got = decode_record(record, format, channels, out, threads)?;
    if got.request != entry.request {
        return Err(err(format!("byte {} holds request {}, {} expected", entry.offset, got.request, entry.request)));
    }
    Ok(())
}

// ── The take's other files ─────────────────────────────────────────────

/// The factory lens string of `calibration.json` (`eis_calib.offset_v6`,
/// or the first non-empty string of that record that parses).
pub fn read_calibration_string(dir: &Path) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(CALIBRATION_FILE)).ok()?).ok()?;
    let eis = v.get("eis_calib")?.as_object()?;
    let parses = |s: &str| crate::insta360::parse_calib_string(s).is_some_and(|l| l.len() >= 2);
    if let Some(s) = eis.get("offset_v6").and_then(|x| x.as_str()).filter(|s| parses(s)) {
        return Some(s.to_owned());
    }
    eis.values().filter_map(|x| x.as_str()).find(|s| parses(s)).map(str::to_owned)
}

/// One gyro sample of `gyro.csv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroSample {
    /// Nanoseconds on the take's counter (the clock of the exposure stamps).
    pub t_ns: u64,
    /// The three columns as written.
    pub xyz: [f32; 3],
}

/// Read `gyro.csv` (`counter_ticks,x,y,z`), ticks converted with
/// `counter_hz`. Lines that do not parse are skipped.
pub fn read_gyro(dir: &Path, counter_hz: u64) -> Result<Vec<GyroSample>> {
    read_motion(&dir.join(GYRO_FILE), counter_hz)
}

/// Read `accel.csv`, the accelerometer in m/s² with gravity in it, on the
/// gyro's axes and clock (the same columns as `gyro.csv`). Empty for a take
/// that has none: no file (an earlier camera build), a file with no
/// samples, or a record that says the stream did not run.
pub fn read_accel(dir: &Path, counter_hz: u64) -> Vec<GyroSample> {
    let ran = std::fs::read(dir.join(ACCEL_INFO_FILE)).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .map_or(true, |v| v.get("status").and_then(|x| x.as_str()).map_or(true, |x| x.eq_ignore_ascii_case("ok")));
    if !ran {
        return Vec::new();
    }
    read_motion(&dir.join(ACCEL_FILE), counter_hz).unwrap_or_default()
}

fn read_motion(file: &Path, counter_hz: u64) -> Result<Vec<GyroSample>> {
    let text = std::fs::read_to_string(file)?;
    let hz = counter_hz.max(1) as u128;
    let mut out = Vec::with_capacity(text.len() / 40);
    for line in text.lines() {
        let mut cols = line.split(',');
        let (Some(t), Some(x), Some(y), Some(z)) = (cols.next(), cols.next(), cols.next(), cols.next()) else { continue };
        let (Ok(t), Ok(x), Ok(y), Ok(z)) = (t.trim().parse::<u64>(), x.trim().parse::<f32>(), y.trim().parse::<f32>(), z.trim().parse::<f32>()) else { continue };
        out.push(GyroSample { t_ns: (t as u128 * 1_000_000_000 / hz) as u64, xyz: [x, y, z] });
    }
    Ok(out)
}

/// The integration time of every written pair's two eyes, nanoseconds, by
/// request number: the take's exposure list (the file `clip.json` names,
/// else `exposure.csv`; `request,cam0_exposure_ns,cam1_exposure_ns`). An
/// eye whose value is empty or zero is not known (`None`). Empty for a take
/// without the file: takes of camera builds that did not record it.
pub fn read_exposures(take: &Take) -> std::collections::HashMap<u32, [Option<u64>; 2]> {
    let name = take.exposure_file.as_deref().unwrap_or(EXPOSURE_FILE);
    let Ok(text) = std::fs::read_to_string(take.dir.join(name)) else { return Default::default() };
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let mut cols = line.split(',');
        let Some(Ok(request)) = cols.next().map(|v| v.trim().parse::<u32>()) else { continue };
        // A second is far beyond any frame's exposure: such a value is no reading.
        let mut eye = || cols.next().and_then(|v| v.trim().parse::<u64>().ok()).filter(|ns| *ns > 0 && *ns < 1_000_000_000);
        let (cam0, cam1) = (eye(), eye());
        out.insert(request, [cam0, cam1]);
    }
    out
}

/// One eye's row timing, as the take's sensor record states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeReadout {
    /// From one row's start to the next, ns.
    pub line_period_ns: f64,
    /// From the frame's first row to its last, ns.
    pub first_to_last_row_ns: f64,
}

/// The row timing of the take's two eyes (channel 0, then 1) from its
/// sensor record: a frame's stamp is the start of its first row's exposure
/// and every row starts a line period after the one above it. `None` when
/// the take has no such record, or the record is of another frame size. A
/// take of one lens has its one sensor's timing in both places.
pub fn read_sensor_readout(take: &Take) -> Option<[EyeReadout; 2]> {
    let name = take.sensor_metadata.as_deref()?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(name)).ok()?).ok()?;
    let eyes = v.get("eyes")?;
    let eye = |key: &str| -> Option<EyeReadout> {
        let e = eyes.get(key)?;
        let size: Vec<u64> = e.get("bayer_output_wh")?.as_array()?.iter().filter_map(|x| x.as_u64()).collect();
        if size != [take.width as u64, take.height as u64] {
            return None;
        }
        let line = e.get("line_period_ns")?.as_f64()?;
        let rows = line * (take.height as f64 - 1.0);
        // The record's own span where it agrees with its line period.
        let span = e.get("rolling_shutter_first_to_last_row_ns").and_then(|x| x.as_f64())
            .filter(|s| (s - rows).abs() <= rows * 0.01).unwrap_or(rows);
        (line.is_finite() && line > 0.0 && span < 1e9).then_some(EyeReadout { line_period_ns: line, first_to_last_row_ns: span })
    };
    match take.layout {
        Layout::Paired => Some([eye("CAM0")?, eye("CAM1")?]),
        Layout::Single { channel } => eye(["CAM0", "CAM1"][channel.min(1)]).map(|e| [e, e]),
    }
}

/// `audio-info.json`: where the sound sits on the take's clock.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTiming {
    pub file: String,
    pub channels: u32,
    pub sample_rate_hz: u32,
    /// Sample frames of sound, after the codec delay and before the padding.
    pub sample_frames: u64,
    /// Decoded samples to drop at the start (an AAC stream's priming).
    pub codec_delay_samples: u64,
    pub trailing_padding_samples: u64,
    /// Time of the first kept sample, nanoseconds on the take's counter.
    pub first_sample_ns: u64,
    /// `(sample index, counter ns)` once a second, for drift checks.
    pub anchors: Vec<(u64, u64)>,
    /// The four channels are first-order ambisonics (the camera's 360° mode).
    pub ambisonic: bool,
}

/// Read the take's audio timing record; `None` when the take has no sound.
pub fn read_audio_timing(take: &Take) -> Option<AudioTiming> {
    let name = take.audio_timing.as_deref()?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(take.dir.join(name)).ok()?).ok()?;
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64());
    let file = v.get("file").and_then(|x| x.as_str()).map(str::to_owned).or_else(|| take.audio_file.clone())?;
    let anchors = v.get("anchors").and_then(|a| a.as_array()).map(|a| {
        a.iter().filter_map(|e| Some((e.get("sample_index")?.as_u64()?, e.get("counter_ns")?.as_u64()?))).collect()
    }).unwrap_or_default();
    let text = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("");
    Some(AudioTiming {
        file,
        channels: u("channels")? as u32,
        sample_rate_hz: u("sample_rate_hz")? as u32,
        sample_frames: u("sample_frames")?,
        codec_delay_samples: u("codec_delay_samples").unwrap_or(0),
        trailing_padding_samples: u("trailing_padding_samples").unwrap_or(0),
        first_sample_ns: u("first_sample_counter_ns")?,
        anchors,
        ambisonic: text("source").eq_ignore_ascii_case("ambisonic") || text("camera_audio_mode").to_ascii_lowercase().contains("ambisonic"),
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    //! A writer for the capture format, for tests: the inverse of the
    //! decoder above.
    use super::*;

    /// Symbol and low bits of a sensor value under `profile` (the value
    /// rounded to the profile's nearest level where the steps are coarse).
    pub fn symbol_of(profile: Profile, value: u16) -> (u16, u8) {
        let levels = profile.levels();
        let i = levels.partition_point(|&l| l <= value).saturating_sub(1);
        let step16 = levels.get(i + 1).map_or(true, |&n| n - levels[i] == 16);
        if step16 && value - levels[i] < 16 && (profile.is_exact() || levels[i] >= THRESHOLD as u16) {
            return (i as u16, (value - levels[i]) as u8);
        }
        let near = if i + 1 < levels.len() && levels[i + 1] - value < value - levels[i] { i + 1 } else { i };
        (near as u16, 0)
    }

    /// The value [`symbol_of`] stands for.
    pub fn stored_value(profile: Profile, value: u16) -> u16 {
        let (s, low) = symbol_of(profile, value);
        profile.levels()[s as usize] + low as u16
    }

    /// One tile: rows of `w` samples.
    fn encode_tile(profile: Profile, samples: &[u16], w: usize) -> Vec<u8> {
        #[allow(non_snake_case)]
        let W = w;
        let coded: Vec<(u16, u8)> = samples.iter().map(|&v| symbol_of(profile, v)).collect();
        let mut main = Vec::new();
        for y in 0..samples.len() / W {
            for x in (0..W).step_by(64) {
                let header_at = main.len();
                main.push(0u8);
                for half in 0..2 {
                    let mut vals = [0u32; 32];
                    for (j, v) in vals.iter_mut().enumerate() {
                        let col = x + half * 32 + j;
                        let i = y * W + col;
                        let left = if col >= 2 { coded[i - 2].0 as i32 } else { 0 };
                        let up = if y >= 2 { coded[i - 2 * W].0 as i32 } else { 0 };
                        let d = coded[i].0 as i32 - ((left + up + 1) >> 1);
                        *v = if d < 0 { (-d * 2 - 1) as u32 } else { (d * 2) as u32 };
                    }
                    let b = (32 - vals.iter().fold(0, |a, v| a | v).leading_zeros()) as usize;
                    assert!(b <= profile.max_width());
                    main[header_at] |= (b as u8) << (4 * half);
                    let mut block = vec![0u8; 4 * b];
                    for lane in 0..8 {
                        let mut w = 0u64;
                        for q in 0..4 {
                            w |= (vals[lane + 8 * q] as u64) << (b * q);
                        }
                        for k in 0..b / 2 {
                            block[k * 8 + lane] = (w >> (8 * k)) as u8;
                        }
                        if b & 1 == 1 {
                            block[(b / 2) * 8 + lane / 2] |= (((w >> (8 * (b / 2))) & 15) as u8) << (4 * (lane & 1));
                        }
                    }
                    main.extend_from_slice(&block);
                }
            }
        }
        let groups = samples.len() / 128;
        let mut bitmap = vec![0u8; groups.div_ceil(8)];
        let mut nibbles = Vec::new();
        for (g, chunk) in coded.chunks(128).enumerate() {
            if chunk.iter().any(|c| c.1 != 0) {
                bitmap[g >> 3] |= 1 << (g & 7);
                for pair in chunk.chunks(2) {
                    nibbles.push(pair[0].1 | (pair[1].1 << 4));
                }
            }
        }
        let mut out = (main.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&main);
        if !nibbles.is_empty() {
            out.extend_from_slice(&bitmap);
            out.extend_from_slice(&nibbles);
        }
        out
    }

    /// One record holding `samples` (channel 0 then channel 1).
    pub fn encode_record(profile: Profile, request: u32, exposure_ns: [u64; 2], samples: &[u16]) -> Vec<u8> {
        assert_eq!(samples.len(), 2 * EYE_SAMPLES);
        let tiles: Vec<Vec<u8>> = samples.chunks(TILE_SAMPLES).map(|t| encode_tile(profile, t, EYE_SIDE)).collect();
        let mut out = vec![0u8; BLOCK as usize];
        out[..8].copy_from_slice(profile.signature());
        for (o, v) in [(8, BLOCK as u32), (12, request), (16, EYE_SIDE as u32), (20, EYE_SIDE as u32), (24, TILES_PER_EYE as u32), (28, THRESHOLD)] {
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        out[32..40].copy_from_slice(&exposure_ns[0].to_le_bytes());
        out[40..48].copy_from_slice(&exposure_ns[1].to_le_bytes());
        for (i, t) in tiles.iter().enumerate() {
            out[48 + 4 * i..52 + 4 * i].copy_from_slice(&(t.len() as u32).to_le_bytes());
        }
        for t in &tiles {
            out.extend_from_slice(t);
            out.resize(pad(out.len() as u64) as usize, 0);
        }
        out
    }

    /// One record of a take of one lens holding `samples` (7680×7680, below
    /// 1024), as the camera lays it out: the sensor's frame number, the
    /// stamp and the integration time after the four `u32` fields, the
    /// payload lengths, then the checksums (left at zero: the first record
    /// alone carries any).
    pub fn encode_single_record(request: u32, hardware: u64, exposure_ns: u64, duration_ns: u64, samples: &[u16]) -> Vec<u8> {
        let layout = Layout::Single { channel: 0 };
        assert_eq!(samples.len(), layout.samples());
        let tiles: Vec<Vec<u8>> = samples.chunks(layout.tile_samples()).map(|t| encode_tile(Profile::Single10, t, SINGLE_SIDE)).collect();
        let mut out = vec![0u8; BLOCK as usize];
        out[..8].copy_from_slice(Profile::Single10.signature());
        for (o, v) in [(8, request), (12, SINGLE_SIDE as u32), (16, SINGLE_SIDE as u32), (20, TILES_PER_EYE as u32)] {
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (o, v) in [(24, hardware), (32, exposure_ns), (40, duration_ns)] {
            out[o..o + 8].copy_from_slice(&v.to_le_bytes());
        }
        for (i, t) in tiles.iter().enumerate() {
            out[48 + 4 * i..52 + 4 * i].copy_from_slice(&(t.len() as u32).to_le_bytes());
        }
        for t in &tiles {
            out.extend_from_slice(t);
            out.resize(pad(out.len() as u64) as usize, 0);
        }
        out
    }

    /// A deterministic 10-bit test picture of one lens: a ramp with noise.
    pub fn single_test_samples(seed: u32) -> Vec<u16> {
        let mut s = seed.wrapping_mul(2654435761).max(1);
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        (0..SINGLE_SIDE * SINGLE_SIDE).map(|i| {
            let (x, y) = (i % SINGLE_SIDE, i / SINGLE_SIDE);
            let ramp = ((x + y * 3) % 1000) as u32;
            (ramp + next() % 24).min(1023) as u16
        }).collect()
    }

    /// Write a finished take of one lens (`channel`) holding `records` (in
    /// the order given) into `dir`; `window` is the seal's window of sensor
    /// frame numbers.
    pub fn write_single_take(dir: &Path, channel: usize, window: (u64, u64), records: &[Vec<u8>]) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes: Vec<u8> = records.concat();
        std::fs::write(dir.join(CAPTURE_FILE), &bytes).unwrap();
        let eye = ["CAM0", "CAM1"][channel];
        std::fs::write(dir.join(CLIP_FILE), format!(
            r#"{{"layout":"single-bayer-v1","fps_setting":30,"size":[7680,7680],"eye":"{eye}","cfa_pattern":[1,2,0,1],"black_level":64,"white_level":1023,"sensor_bits":10,"codec":"X6S8B01","frames_file":"frames.csv","exposure_file":"exposure.csv","sensor_metadata":"sensor-metadata.json","clock_start":{{"counter_ticks":1920000,"counter_hz":19200000,"mono_before_ns":1,"mono_after_ns":2,"boottime_ns":1}}}}"#)).unwrap();
        std::fs::write(dir.join(SEAL_FILE), format!(
            r#"{{"fps_setting":30,"layout":"single-bayer-v1","eye":"{eye}","hardware_window":[{},{}],"frames":{n},"queue_peak":1,"gaps":{},"bytes":{},"exposure_status":"ok"}}"#,
            window.0, window.1, (window.1 + 1 - window.0) as usize - records.len(), bytes.len(), n = records.len())).unwrap();
    }

    /// A deterministic test picture: a smooth ramp with noise, both channels.
    pub fn test_samples(seed: u32) -> Vec<u16> {
        let mut s = seed.wrapping_mul(2654435761).max(1);
        let mut next = move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        (0..2 * EYE_SAMPLES).map(|i| {
            let (x, y) = (i % EYE_SIDE, (i / EYE_SIDE) % EYE_SIDE);
            let ramp = ((x * 4 + y * 3) % 16384) as u32;
            let noise = next() % 96;
            (ramp + noise).min(16383) as u16
        }).collect()
    }

    /// Write a finished take of the given pairs (in the order given) into
    /// `dir`; `window` is the seal's request window.
    pub fn write_take(dir: &Path, fps: u32, window: (u32, u32), records: &[Vec<u8>]) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes: Vec<u8> = records.concat();
        std::fs::write(dir.join(CAPTURE_FILE), &bytes).unwrap();
        std::fs::write(dir.join(CLIP_FILE), format!(
            r#"{{"layout":"paired-bayer-v1","fps_setting":{fps},"size":[3840,3840],"clock_start":{{"counter_ticks":1920000,"counter_hz":19200000,"mono_before_ns":1,"mono_after_ns":2,"boottime_ns":1}}}}"#)).unwrap();
        std::fs::write(dir.join(SEAL_FILE), format!(
            r#"{{"fps_setting":{fps},"request_window":[{},{}],"received_pairs":{n},"pairs":{n},"bytes":{}}}"#,
            window.0, window.1, bytes.len(), n = records.len())).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn fnv(values: &[u16]) -> u64 {
        values.iter().flat_map(|v| v.to_le_bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("vr180-x6b-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn level_tables_are_the_cameras() {
        // Length and an FNV-1a of each table as the camera side publishes them.
        let want: [(usize, u64); 5] = [
            (1984, 0xdb55487329ac38a5),
            (416, 0x9986b593b4e9da05),
            (432, 0xd5ad6ef2f31739a5),
            (289, 0xb0a8b5deb01a5cbb),
            (297, 0x02449f72339e1fab),
        ];
        for (p, (len, hash)) in Profile::ALL.iter().zip(want) {
            let t = p.levels();
            assert_eq!(t.len(), len, "{p:?}");
            assert_eq!(fnv(t), hash, "{p:?}");
            assert!(t.windows(2).all(|w| w[0] < w[1]), "{p:?} not rising");
            assert_eq!(*t.last().unwrap(), 16368, "{p:?}");
            assert_eq!(t[0], 0);
        }
        // One lens: a symbol is the 10-bit sample.
        let t = Profile::Single10.levels();
        assert!(t.len() == 1024 && t.iter().enumerate().all(|(i, v)| *v as usize == i));
    }

    #[test]
    fn a_signature_belongs_to_one_frame_rate() {
        assert_eq!(Profile::from_signature(b"L1024___", 30), Some(Profile::Exact30));
        assert_eq!(Profile::from_signature(b"X6QBLK0B", 60), Some(Profile::Raised60));
        assert_eq!(Profile::from_signature(b"X6QBLK0B", 50), None);
        assert_eq!(Profile::from_signature(b"X6M50AB2", 60), None);
        assert_eq!(Profile::from_signature(b"UNKNOWN!", 30), None);
        // Two signatures at 30 fps, one for each layout.
        assert_eq!(Profile::from_signature(b"X6S8B01\0", 30), Some(Profile::Single10));
        assert_eq!(Profile::from_signature(b"X6S8B01\0", 60), None);
        assert!(Profile::Single10.layout_is(Layout::Single { channel: 1 }) && !Profile::Single10.layout_is(Layout::Paired));
        assert!(Profile::Exact30.layout_is(Layout::Paired) && !Profile::Exact30.layout_is(Layout::Single { channel: 0 }));
    }

    #[test]
    fn every_profile_decodes_what_was_stored() {
        for (n, profile) in Profile::ALL.into_iter().filter(|p| p.layout_is(Layout::Paired)).enumerate() {
            let samples = test_samples(7 + n as u32);
            let record = encode_record(profile, 100 + n as u32, [1_000_000, 999_971], &samples);
            let mut out = vec![0u16; 2 * EYE_SAMPLES];
            let got = decode_record(&record, Format::paired(profile.fps()), Channels::Both, &mut out, 0).unwrap();
            assert_eq!((got.request, got.exposure_ns, got.len as usize), (100 + n as u32, [1_000_000, 999_971], record.len()));
            let want: Vec<u16> = samples.iter().map(|&v| stored_value(profile, v)).collect();
            assert_eq!(fnv(&out), fnv(&want), "{profile:?}");
            if profile.is_exact() {
                assert_eq!(fnv(&out), fnv(&samples), "30 fps keeps every value");
            }
            // One thread, and one channel only, give the same samples.
            let mut one = vec![0u16; 2 * EYE_SAMPLES];
            decode_record(&record, Format::paired(profile.fps()), Channels::Both, &mut one, 1).unwrap();
            assert_eq!(fnv(&one), fnv(&out));
            let mut right = vec![7u16; 2 * EYE_SAMPLES];
            decode_record(&record, Format::paired(profile.fps()), Channels::One(1), &mut right, 0).unwrap();
            assert!(right[..EYE_SAMPLES].iter().all(|&v| v == 7));
            assert_eq!(fnv(&right[EYE_SAMPLES..]), fnv(&out[EYE_SAMPLES..]));
        }
    }

    #[test]
    fn a_damaged_record_is_refused() {
        let profile = Profile::Coarse60;
        let record = encode_record(profile, 5, [10, 20], &test_samples(1));
        let mut out = vec![0u16; 2 * EYE_SAMPLES];
        let mut decode = |r: &[u8], fps: u32| decode_record(r, Format::paired(fps), Channels::Both, &mut out, 0);
        assert!(decode(&record, 60).is_ok());
        assert!(decode(&record, 50).is_err(), "a 60 fps signature in a 50 fps take");
        let mut unknown = record.clone();
        unknown[..8].copy_from_slice(b"UNKNOWN!");
        assert!(decode(&unknown, 60).is_err());
        assert!(decode(&record[..record.len() - 1], 60).is_err(), "cut short");
        let mut size = record.clone();
        size[48..52].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&size, 60).is_err(), "tile length out of range");
        let mut stream = record.clone();
        let first = BLOCK as usize + 4;
        stream[first] = 0xff;
        assert!(decode(&stream, 60).is_err(), "bit width beyond the profile's");
    }

    #[test]
    fn pairs_come_back_in_request_order_with_their_gaps() {
        let dir = temp_dir("order");
        let samples = test_samples(3);
        let a = encode_record(Profile::Raised60, 2484, [5_000_000_000, 4_999_971_000], &samples);
        let b = encode_record(Profile::Raised60, 2486, [5_033_366_667, 5_033_337_667], &samples);
        // Later request first on disk; 2485 was never saved.
        write_take(&dir, 60, (2484, 2486), &[b.clone(), a.clone()]);
        assert_eq!(take_dir(&dir).as_deref(), Some(dir.as_path()));
        assert_eq!(take_dir(&dir.join(CAPTURE_FILE)).as_deref(), Some(dir.as_path()));
        assert_eq!(take_dir(&dir.join("audio.aac")), None);
        let take = read_take(&dir).unwrap().unwrap();
        assert_eq!((take.fps_setting, take.pairs(), take.missing_pairs()), (60, 2, 1));
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(pairs.iter().map(|p| (p.request, p.offset)).collect::<Vec<_>>(), vec![(2484, b.len() as u64), (2486, 0)]);
        assert_eq!(missing_requests(&take, &pairs), vec![2485]);
        assert_eq!(pairs[0].time_ns(), 4_999_985_500);
        // The pair decodes from its place in the file.
        let mut file = File::open(take.capture_path()).unwrap();
        let (mut record, mut out) = (Vec::new(), vec![0u16; 2 * EYE_SAMPLES]);
        decode_pair(&mut file, &pairs[0], take.format(), Channels::Both, &mut record, &mut out, 0).unwrap();
        assert_eq!(fnv(&out), fnv(&samples.iter().map(|&v| stored_value(Profile::Raised60, v)).collect::<Vec<_>>()));
        // A stopped walk says so.
        assert!(scan_pairs(&take, |_, _| false).unwrap().is_none());

        // The same request twice is refused.
        write_take(&dir, 60, (2484, 2486), &[a.clone(), a.clone()]);
        let take = read_take(&dir).unwrap().unwrap();
        assert!(scan_pairs(&take, |_, _| true).unwrap_err().to_string().contains("duplicate"));
        // A pair count that is not the seal's is refused.
        write_take(&dir, 60, (2484, 2486), &[a.clone(), b.clone()]);
        std::fs::write(dir.join(SEAL_FILE), format!(
            r#"{{"fps_setting":60,"request_window":[2484,2486],"received_pairs":3,"pairs":3,"bytes":{}}}"#, a.len() + b.len())).unwrap();
        let take = read_take(&dir).unwrap().unwrap();
        assert!(scan_pairs(&take, |_, _| true).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unfinished_take_says_why() {
        let dir = temp_dir("unfinished");
        let a = encode_record(Profile::Exact30, 1, [10, 20], &test_samples(2));
        write_take(&dir, 30, (1, 1), &[a.clone()]);
        assert!(read_take(&dir).unwrap().is_ok());
        // A copy that stopped early.
        std::fs::write(dir.join(CAPTURE_FILE), &a[..a.len() - 4096]).unwrap();
        assert_eq!(read_take(&dir).unwrap(), Err(TakeProblem::LengthMismatch { sealed: a.len() as u64, actual: a.len() as u64 - 4096 }));
        std::fs::remove_file(dir.join(SEAL_FILE)).unwrap();
        assert_eq!(read_take(&dir).unwrap(), Err(TakeProblem::NotSealed));
        std::fs::remove_file(dir.join(CAPTURE_FILE)).unwrap();
        assert_eq!(read_take(&dir).unwrap(), Err(TakeProblem::NoCapture));
        // The folder a take's frames were written into is named as such.
        std::fs::write(dir.join(FRAMES_FILE), "{}\n").unwrap();
        assert_eq!(read_take(&dir).unwrap(), Err(TakeProblem::FramesOnly));
        assert!(TakeProblem::FramesOnly.to_string().contains("DNG"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_takes_other_files_read() {
        let dir = temp_dir("files");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(GYRO_FILE), "counter_ticks,x,y,z\n19200000,0.5,-0.25,1\n19219200,0.1,0.2,0.3\nbroken\n").unwrap();
        let g = read_gyro(&dir, 19_200_000).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].t_ns, g[0].xyz), (1_000_000_000, [0.5, -0.25, 1.0]));
        assert_eq!(g[1].t_ns, 1_001_000_000);
        let lens = "2_2.455430_7216.620_7216.620_3855.460_3884.530_0.276_-0.042_89.578_0.000000_0.000000_0.000000_1.32059765_-1.38414502_4.43894196_5.25202703_0.00000000_-0.00032840_0.00113587_0.00412989_0.00653467_-0.00106616_-0.00235222_0.02084749_-0.01150168_15488_7744_193_2.455430_7227.070_7227.070_11569.100_3877.970_-0.132_0.081_89.544_-0.000064_-0.000049_-0.032576_1.31835759_-1.57224715_5.29361629_3.47188210_0.00000000_0.00016541_-0.00072159_-0.00095306_0.01344961_-0.00061933_-0.00056505_0.01970853_-0.02243911_15488_7744_193_394240";
        std::fs::write(dir.join(CALIBRATION_FILE), format!(r#"{{"eis_calib":{{"flag":0,"offset_v1":"","offset_v6":"{lens}"}},"offset_calib":{{}}}}"#)).unwrap();
        assert_eq!(read_calibration_string(&dir).as_deref(), Some(lens));
        let a = encode_record(Profile::Exact30, 1, [10, 20], &test_samples(2));
        write_take(&dir, 30, (1, 1), &[a]);
        let mut take = read_take(&dir).unwrap().unwrap();
        assert!(read_audio_timing(&take).is_none());
        // The accelerometer: the gyro's columns; none without the file, with
        // an empty one, or when its record says it did not run.
        assert!(read_accel(&dir, 19_200_000).is_empty());
        std::fs::write(dir.join(ACCEL_FILE), "counter_ticks,x,y,z\n").unwrap();
        assert!(read_accel(&dir, 19_200_000).is_empty());
        std::fs::write(dir.join(ACCEL_FILE), "counter_ticks,x,y,z\n19200000,-9.8,-0.5,0.125\n19219200,-9.75,-0.5,0.25\n").unwrap();
        let acc = read_accel(&dir, 19_200_000);
        assert_eq!((acc.len(), acc[0].t_ns, acc[1].xyz), (2, 1_000_000_000, [-9.75, -0.5, 0.25]));
        std::fs::write(dir.join(ACCEL_INFO_FILE), r#"{"unit":"m/s^2","status":"unavailable","samples":0}"#).unwrap();
        assert!(read_accel(&dir, 19_200_000).is_empty());
        std::fs::write(dir.join(ACCEL_INFO_FILE), r#"{"unit":"m/s^2","status":"ok","samples":2}"#).unwrap();
        assert_eq!(read_accel(&dir, 19_200_000).len(), 2);
        // The sensor record: each eye's row timing. No record, none.
        assert!(take.sensor_metadata.is_none() && read_sensor_readout(&take).is_none());
        take.sensor_metadata = Some("sensor-metadata.json".into());
        assert!(read_sensor_readout(&take).is_none());
        let eye = |line: f64, span: u64, w: u32| format!(r#"{{"bayer_output_wh":[{w},3840],"line_period_ns":{line},"rolling_shutter_first_to_last_row_ns":{span}}}"#);
        let record = |a: String, b: String| format!(r#"{{"schema":"raw-sensor-metadata-v1","eyes":{{"CAM0":{a},"CAM1":{b}}}}}"#);
        std::fs::write(dir.join("sensor-metadata.json"), record(eye(3791.6666628749995, 14556208, 3840), eye(3791.6666973791675, 14556208, 3840))).unwrap();
        let r = read_sensor_readout(&take).unwrap();
        assert!(r[0].first_to_last_row_ns == 14556208.0 && (r[1].line_period_ns - 3791.6666973791675).abs() < 1e-6, "{r:?}");
        // A span that is not the rows times the line period gives way to that.
        std::fs::write(dir.join("sensor-metadata.json"), record(eye(4000.0, 99, 3840), eye(4000.0, 15356000, 3840))).unwrap();
        let r = read_sensor_readout(&take).unwrap();
        assert_eq!((r[0].first_to_last_row_ns, r[1].first_to_last_row_ns), (15356000.0, 15356000.0));
        // A record of another frame size says nothing of this one.
        std::fs::write(dir.join("sensor-metadata.json"), record(eye(4000.0, 15356000, 4096), eye(4000.0, 15356000, 3840))).unwrap();
        assert!(read_sensor_readout(&take).is_none());
        take.sensor_metadata = None;
        // The integration times: by request, an eye left empty or at zero is unknown; no file, nothing.
        assert!(take.exposure_file.is_none() && read_exposures(&take).is_empty());
        std::fs::write(dir.join(EXPOSURE_FILE), "request,cam0_exposure_ns,cam1_exposure_ns\n191,8660735,8660734\n192,12596735,\n193,0,9000000\nbroken,1,2\n194,5000000000,7\n").unwrap();
        // A take that names another file has that one read, not the usual name.
        take.exposure_file = Some("shutter.csv".into());
        assert!(read_exposures(&take).is_empty());
        take.exposure_file = None;
        let e = read_exposures(&take);
        assert_eq!(e.len(), 4);
        assert_eq!((e[&191], e[&192], e[&193], e[&194]), ([Some(8_660_735), Some(8_660_734)], [Some(12_596_735), None], [None, Some(9_000_000)], [None, Some(7)]));
        take.audio_timing = Some("audio-info.json".into());
        std::fs::write(dir.join("audio-info.json"), r#"{"version":2,"file":"audio.aac","source":"ambisonic","channels":4,"codec_delay_samples":2048,"trailing_padding_samples":384,"sample_rate_hz":48000,"sample_frames":1142400,"first_sample_counter_ns":372569353830,"anchors":[{"sample_index":960,"counter_ns":372589353830}]}"#).unwrap();
        let t = read_audio_timing(&take).unwrap();
        assert_eq!((t.file.as_str(), t.channels, t.sample_rate_hz, t.sample_frames, t.codec_delay_samples), ("audio.aac", 4, 48000, 1142400, 2048));
        assert_eq!((t.first_sample_ns, t.anchors.len(), t.ambisonic), (372569353830, 1, true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_take_of_one_lens_reads_and_decodes() {
        let dir = temp_dir("single");
        let samples = single_test_samples(11);
        let single = Format { layout: Layout::Single { channel: 1 }, fps: 30 };
        // Requests 41 and 43 (42 was never written), sensor frames 900 and 902.
        let a = encode_single_record(41, 900, 7_000_000_000, 8_000_000, &samples);
        let b = encode_single_record(43, 902, 7_066_733_333, 4_000_000, &samples);
        write_single_take(&dir, 1, (900, 902), &[a.clone(), b.clone()]);
        assert_eq!(take_dir(&dir).as_deref(), Some(dir.as_path()));
        assert_eq!(take_layout(&dir.join(CAPTURE_FILE)), Some(Layout::Single { channel: 1 }));
        let take = read_take(&dir).unwrap().unwrap();
        assert_eq!((take.layout, take.format(), take.width, take.height), (Layout::Single { channel: 1 }, single, 7680, 7680));
        assert_eq!((take.cfa, take.black_level, take.white_level, take.fps_setting), ([1, 2, 0, 1], 64, 1023, 30));
        // The seal counts sensor frames: two written of three, none of the
        // window's numbers a request.
        assert_eq!((take.seal.request_window, take.pairs(), take.missing_pairs(), take.window_numbers_records()), ((900, 902), 2, 1, false));
        let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
        assert_eq!(pairs.iter().map(|p| (p.request, p.offset, p.profile)).collect::<Vec<_>>(),
            vec![(41, 0, Profile::Single10), (43, a.len() as u64, Profile::Single10)]);
        assert_eq!((pairs[0].exposure_ns, pairs[0].time_ns()), ([7_000_000_000; 2], 7_000_000_000));
        assert_eq!(missing_requests(&take, &pairs), vec![42]);
        // Every sample comes back, on one thread as on many, whichever channel is asked for.
        let mut file = File::open(take.capture_path()).unwrap();
        let (mut record, mut out) = (Vec::new(), vec![0u16; SINGLE_SIDE * SINGLE_SIDE]);
        decode_pair(&mut file, &pairs[1], take.format(), Channels::Both, &mut record, &mut out, 0).unwrap();
        assert_eq!(fnv(&out), fnv(&samples));
        let mut one = vec![0u16; SINGLE_SIDE * SINGLE_SIDE];
        let got = decode_record(&a, single, Channels::One(1), &mut one, 1).unwrap();
        assert_eq!((fnv(&one), got.request, got.len as usize), (fnv(&samples), 41, a.len()));
        // A buffer of a pair's size is not this record's.
        assert!(decode_record(&a, single, Channels::Both, &mut vec![0u16; 2 * EYE_SAMPLES], 0).is_err());
        // Neither layout takes the other's records.
        assert!(decode_record(&a, Format::paired(30), Channels::Both, &mut vec![0u16; 2 * EYE_SAMPLES], 0).is_err());
        let paired = encode_record(Profile::Exact30, 1, [10, 20], &test_samples(2));
        assert!(decode_record(&paired, single, Channels::Both, &mut out, 0).is_err());
        // A damaged record is refused: a wrong size field, bytes after a tile's stream.
        let mut wide = a.clone();
        wide[12..16].copy_from_slice(&3840u32.to_le_bytes());
        assert!(decode_record(&wide, single, Channels::Both, &mut out, 0).is_err());
        let mut longer = a.clone();
        let first = u32::from_le_bytes(a[48..52].try_into().unwrap());
        longer[48..52].copy_from_slice(&(first + 1).to_le_bytes());
        assert!(decode_record(&longer, single, Channels::Both, &mut out, 0).is_err());

        // The sensor record of one lens: its timing stands for both places.
        std::fs::write(dir.join("sensor-metadata.json"),
            r#"{"schema":"raw-sensor-metadata-v1","eyes":{"CAM1":{"bayer_output_wh":[7680,7680],"line_period_ns":4000.000009472,"rolling_shutter_first_to_last_row_ns":30716000}}}"#).unwrap();
        let r = read_sensor_readout(&take).unwrap();
        assert!(r[0] == r[1] && r[0].first_to_last_row_ns == 30_716_000.0, "{r:?}");
        // The integration times: the recorded sensor's column only.
        std::fs::write(dir.join(EXPOSURE_FILE), "request,cam0_exposure_ns,cam1_exposure_ns\n41,,8000000\n43,,4000000\n").unwrap();
        let e = read_exposures(&take);
        assert_eq!((e[&41], e[&43]), ([None, Some(8_000_000)], [None, Some(4_000_000)]));

        // A take closed for another sensor than its clip names, a frame size
        // that is not the layout's and levels that cannot be are refused.
        let clip = std::fs::read_to_string(dir.join(CLIP_FILE)).unwrap();
        std::fs::write(dir.join(CLIP_FILE), clip.replace("CAM1", "CAM0")).unwrap();
        assert!(read_take(&dir).is_err());
        std::fs::write(dir.join(CLIP_FILE), clip.replace("[7680,7680]", "[3840,3840]")).unwrap();
        assert!(read_take(&dir).is_err());
        std::fs::write(dir.join(CLIP_FILE), clip.replace("\"white_level\":1023", "\"white_level\":4095")).unwrap();
        assert!(read_take(&dir).is_err());
        for not_bayer in ["[1,1,0,2]", "[0,1,2,1]", "[1,0,1,2]", "[0,1,1,1]"] {
            std::fs::write(dir.join(CLIP_FILE), clip.replace("[1,2,0,1]", not_bayer)).unwrap();
            assert!(read_take(&dir).is_err(), "{not_bayer}");
        }
        // The other three Bayer patterns are read as they are stated.
        for (text, cfa) in [("[0,1,1,2]", [0u8, 1, 1, 2]), ("[2,1,1,0]", [2, 1, 1, 0]), ("[1,0,2,1]", [1, 0, 2, 1])] {
            std::fs::write(dir.join(CLIP_FILE), clip.replace("[1,2,0,1]", text)).unwrap();
            assert_eq!(read_take(&dir).unwrap().unwrap().cfa, cfa);
        }
        // An unknown sensor name is no take at all.
        std::fs::write(dir.join(CLIP_FILE), clip.replace("CAM1", "CAM7")).unwrap();
        assert_eq!(take_dir(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The camera side's reference digests on the sample takes (skipped when
    /// the volume is not mounted): FNV-1a of a pair's two eyes.
    #[test]
    fn sample_takes_decode_to_the_reference_pixels() {
        let root = Path::new("/Volumes/Database/x6/1115 raw video");
        let cases: [(&str, u32, u64); 3] = [
            ("RAWVIDEO-20261116-011723", 697, 0xd6caf546b27ba370),
            ("RAWVIDEO-20261116-011809", 519, 0x05f4d4ae0cab5e68),
            ("RAWVIDEO-20261116-012111", 2484, 0x40139dc23ade7ea6),
        ];
        for (take_name, request, want) in cases {
            let dir = root.join(take_name);
            if !dir.exists() { continue; }
            let take = read_take(&dir).unwrap().unwrap();
            let pairs = scan_pairs(&take, |_, _| true).unwrap().unwrap();
            let entry = pairs.iter().find(|p| p.request == request).unwrap();
            let mut file = File::open(take.capture_path()).unwrap();
            let (mut record, mut out) = (Vec::new(), vec![0u16; 2 * EYE_SAMPLES]);
            decode_pair(&mut file, entry, take.format(), Channels::Both, &mut record, &mut out, 0).unwrap();
            assert_eq!(fnv(&out), want, "{take_name} request {request}");
        }
    }
}
